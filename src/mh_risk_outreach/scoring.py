"""Score members, explain model predictions, and order a comparison worklist.

Run it:
    uv run python -m mh_risk_outreach.scoring
    uv run python -m mh_risk_outreach.scoring --model random_forest
    uv run python -m mh_risk_outreach.scoring --model all --scope all
    uv run python -m mh_risk_outreach.scoring --operational
"""

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from mh_risk_outreach.bands import BAND_LABELS, assign_bands
from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    DEFAULT_TABLE,
    MEMBER_ID,
    RANK_SCORE_DECIMALS,
    RANK_SORT_KEYS,
)
from mh_risk_outreach.config import RANK_ORDER as RANK_ORDER  # noqa: PLC0414
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.repository import load_from_db
from mh_risk_outreach.db.scores import save_member_scores
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    ScoringArtifact,
    load_scoring_artifact,
)
from mh_risk_outreach.models.base import predict_risk_scores
from mh_risk_outreach.reasons import build_reasons

# Members explained per SHAP call when reporting progress.
REASON_CHUNK_SIZE = 100
CRITICAL_INDICATORS = ("Self_Harm_Thoughts", "Suicidal_Thoughts")


def safety_review_flags(records: pd.DataFrame) -> np.ndarray:
    """Mark members who reported self-harm or suicidal thoughts."""
    return records[list(CRITICAL_INDICATORS)].eq(1).any(axis=1).to_numpy(dtype=bool)


def rank_members(worklist: pd.DataFrame) -> pd.DataFrame:
    """Rank safety reviews first, then rounded scores descending and IDs ascending."""
    ranked = worklist.sort_values(
        by=[column for column, _ in RANK_SORT_KEYS],
        ascending=[ascending for _, ascending in RANK_SORT_KEYS],
        key=lambda column: (
            column.astype("float64").round(RANK_SCORE_DECIMALS)
            if column.name == "risk_score"
            else column
        ),
    ).reset_index(drop=True)
    ranked["rank_priority"] = np.arange(1, len(ranked) + 1)
    return ranked


def score_members(
    records: pd.DataFrame,
    artifact: ScoringArtifact,
    *,
    background: pd.DataFrame,
    progress: Callable[[float], None] | None = None,
    chunk_size: int = REASON_CHUNK_SIZE,
) -> pd.DataFrame:
    """Score ``records`` and return the worklist DataFrame for one model.

    ``background`` is the training rows the pipeline was fitted on; the SHAP
    reasons measure each member against them, so a member's reasons do not depend
    on which other members are scored alongside them. That also makes it safe to
    explain members in chunks of ``chunk_size``: ``progress``, if given, is called
    with the fraction done (0-1] after each chunk.

    Columns: ``Member_ID``, ``model_version``, ``risk_score`` (0-100),
    ``band_level_priority``, ``safety_review_required``,
    ``rank_priority`` (1 = contact first), ``reasoning``.
    Rows are returned already sorted by ``rank_priority``.
    """
    risk_score = predict_risk_scores(artifact.pipeline, records)
    bands = assign_bands(risk_score, artifact.cut_points)
    reasons = build_reasons(
        artifact.pipeline, records, background, chunk_size=chunk_size, progress=progress
    )

    worklist = pd.DataFrame(
        {
            MEMBER_ID: records[MEMBER_ID].to_numpy(),
            "model_version": artifact.model_version,
            "risk_score": risk_score,
            "band_level_priority": bands,
            "safety_review_required": safety_review_flags(records),
            "reasoning": reasons,
        }
    )
    return rank_members(worklist)


def _summary(model_name: str, worklist: pd.DataFrame) -> str:
    counts = worklist["band_level_priority"].value_counts()
    bands = "  ".join(
        f"{band}: {int(counts.get(band, 0))}" for band in reversed(BAND_LABELS)
    )
    return (
        f"[{model_name}] scored {len(worklist):,} members\n"
        f"  bands   {bands}\n"
        f"  top of worklist: Member {worklist.iloc[0][MEMBER_ID]} "
        f"(score {worklist.iloc[0]['risk_score']:.1f}, {worklist.iloc[0]['band_level_priority']})"
    )


def run(
    model_name: str = DEFAULT_MODEL,
    *,
    db_path: str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    scope: str = "evaluation",
) -> pd.DataFrame:
    """Score one model's eligible members and write its ``<model>_scores`` table.

    ``scope="evaluation"`` scores the held-out 2,000 rows the model never saw (an
    honest demonstration); ``scope="all"`` scores every stored member.
    """
    records = load_from_db(db_path, table)
    split = split_dataset(records)
    target = split.evaluation if scope == "evaluation" else records

    artifact = load_scoring_artifact(model_name, artifacts_dir)
    worklist = score_members(target, artifact, background=split.train)
    save_member_scores(worklist, model_name, db_path=db_path)

    print(_summary(model_name, worklist))
    print(f"  wrote {len(worklist):,} rows to table {model_name}_scores in {db_path}\n")
    return worklist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score members and build the worklist."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--model",
        choices=[*sorted(REGISTRY), "all"],
        help="comparison model to score with, or 'all'",
    )
    mode.add_argument(
        "--operational",
        action="store_true",
        help="score the full operational population",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path")
    parser.add_argument("--table", help="source records table for comparison scoring")
    parser.add_argument(
        "--artifacts", default=DEFAULT_ARTIFACTS_DIR, help="artifacts dir"
    )
    parser.add_argument(
        "--scope",
        choices=["evaluation", "all"],
        help="score the held-out evaluation rows (default) or every member",
    )
    args = parser.parse_args()
    if args.operational:
        if args.scope is not None or args.table is not None:
            parser.error("--operational always scores the full patients table")
        from mh_risk_outreach.operational import (
            OPERATIONAL_SCOPE,
            refresh_operational_worklist,
        )

        try:
            worklist = refresh_operational_worklist(
                db_path=args.db, artifacts_dir=args.artifacts
            )
        except ValueError as exc:
            parser.error(str(exc))
        model = str(worklist["operational_model_name"].iat[0])
        version = worklist["model_version"].iat[0]
        print(
            f"Operational model: {model}, version: {version}, scope: {OPERATIONAL_SCOPE}"
        )
        print(_summary(model, worklist))
        print(f"  wrote {len(worklist):,} rows to operational_scores in {args.db}")
        return

    models = sorted(REGISTRY) if args.model == "all" else [args.model or DEFAULT_MODEL]
    for model_name in models:
        run(
            model_name,
            db_path=args.db,
            table=args.table or DEFAULT_TABLE,
            artifacts_dir=args.artifacts,
            scope=args.scope or "evaluation",
        )


if __name__ == "__main__":
    main()
