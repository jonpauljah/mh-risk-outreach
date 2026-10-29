"""Score members, explain the score, and order the outreach worklist (issue #16).

This is the scoring stage from docs/ml-spec.md, implemented per the decisions in
docs/risk-scoring.md. For a trained model it produces, per member:

* ``risk_score``          – the model's probability of an issue, as 0-100.
* ``band_level_priority`` – a risk band (Very High … Very Low), found by K-means
  on the scores so the boundaries are learned from the data, not fixed in advance.
* ``rank_priority``       – a strict contact order, 1 = contact first.
* ``reasoning``           – a short SHAP-based description of the top drivers.

A **critical-indicator override** (``Self_Harm_Thoughts`` or
``Suicidal_Thoughts`` = 1) forces a member to the Very High band and to the top
of the rank, regardless of score; ``risk_score`` is left raw so the override is
visible and the reason names it.

Run it:
    uv run python -m mh_risk_outreach.scoring                 # logistic_regression
    uv run python -m mh_risk_outreach.scoring --model random_forest
    uv run python -m mh_risk_outreach.scoring --model all --scope all
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    DEFAULT_TABLE,
    MEMBER_ID,
    RANDOM_STATE,
)
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.repository import load_from_db
from mh_risk_outreach.db.scores import save_member_scores
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.reasons import build_reasons

DEFAULT_ARTIFACTS_DIR = "artifacts"

# Lowest band first; index doubles as the within-band ordering weight.
BAND_LABELS: tuple[str, ...] = ("Very Low", "Low", "Medium", "High", "Very High")
TOP_BAND = BAND_LABELS[-1]

# Members with either of these set to 1 are the critical-indicator override.
CRITICAL_INDICATORS = ("Self_Harm_Thoughts", "Suicidal_Thoughts")
CRITICAL_REASON = "critical indicator (self-harm / suicidal thoughts), safety override"


def load_scoring_pipeline(
    model_name: str, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
) -> tuple[Pipeline, str]:
    """Load a saved model pipeline and its ``model_version`` from the artifact.

    Reloading (rather than re-fitting) means every stored score traces back to a
    saved model version. Missing artifacts stop the run with how to create them.
    """
    folder = Path(artifacts_dir) / model_name
    model_path = folder / "model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"no saved model at {model_path}; train it first with "
            f"`uv run python -m mh_risk_outreach.models.train --model {model_name}`"
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return pipeline, str(metadata.get("model_version", model_name))


def assign_bands(scores: np.ndarray) -> np.ndarray:
    """Map scores to the five risk bands with 1-D K-means (learned boundaries).

    Clusters are ordered by centre and spread across ``BAND_LABELS`` so the
    lowest-scoring cluster is "Very Low" and the highest "Very High". With fewer
    distinct scores than labels (only in tiny inputs), the available clusters are
    spread evenly across the five names. ``RANDOM_STATE`` makes it reproducible.
    """
    scores = np.asarray(scores, dtype=float)
    distinct = np.unique(scores)
    k = min(len(BAND_LABELS), len(distinct))
    if k <= 1:  # every score identical: one band, the top
        return np.full(len(scores), TOP_BAND, dtype=object)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(scores.reshape(-1, 1))
    centres = kmeans.cluster_centers_.ravel()
    # rank 0 = lowest-centre cluster ... rank k-1 = highest.
    rank_of_cluster = {
        cluster: rank for rank, cluster in enumerate(np.argsort(centres))
    }
    label_for_rank = [
        BAND_LABELS[round(rank / (k - 1) * (len(BAND_LABELS) - 1))] for rank in range(k)
    ]
    return np.array(
        [label_for_rank[rank_of_cluster[c]] for c in clusters], dtype=object
    )


def score_members(
    records: pd.DataFrame,
    pipeline: Pipeline,
    *,
    model_version: str,
    background: pd.DataFrame,
) -> pd.DataFrame:
    """Score ``records`` and return the worklist DataFrame for one model.

    ``background`` is the training rows the pipeline was fitted on; the SHAP
    reasons measure each member against them, so a member's reasons do not depend
    on which other members are scored alongside them. Columns: ``Member_ID``, ``model_version``, ``risk_score`` (0-100),
    ``band_level_priority``, ``rank_priority`` (1 = contact first), ``reasoning``.
    Rows are returned already sorted by ``rank_priority``.
    """
    risk_score = pipeline.predict_proba(records)[:, 1] * 100.0
    bands = assign_bands(risk_score)
    reasons = build_reasons(pipeline, records, background)

    critical = np.zeros(len(records), dtype=bool)
    for column in CRITICAL_INDICATORS:
        critical |= records[column].to_numpy() == 1

    bands = np.where(
        critical, TOP_BAND, bands
    )  # override pins criticals to the top band
    reasons = [
        f"{CRITICAL_REASON}; {reason}" if flag else reason
        for reason, flag in zip(reasons, critical, strict=True)
    ]

    band_order = {label: index for index, label in enumerate(BAND_LABELS)}
    worklist = pd.DataFrame(
        {
            MEMBER_ID: records[MEMBER_ID].to_numpy(),
            "model_version": model_version,
            "risk_score": risk_score,
            "band_level_priority": bands,
            "reasoning": reasons,
            "_critical": critical,
            "_band_rank": [band_order[band] for band in bands],
        }
    )
    # Criticals first, then higher band, then higher score; ties by Member_ID.
    worklist = worklist.sort_values(
        by=["_critical", "_band_rank", "risk_score", MEMBER_ID],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    worklist["rank_priority"] = np.arange(1, len(worklist) + 1)
    return worklist.drop(columns=["_critical", "_band_rank"])[
        [
            MEMBER_ID,
            "model_version",
            "risk_score",
            "band_level_priority",
            "rank_priority",
            "reasoning",
        ]
    ]


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

    pipeline, model_version = load_scoring_pipeline(model_name, artifacts_dir)
    worklist = score_members(
        target, pipeline, model_version=model_version, background=split.train
    )
    save_member_scores(worklist, model_name, db_path=db_path)

    print(_summary(model_name, worklist))
    print(f"  wrote {len(worklist):,} rows to table {model_name}_scores in {db_path}\n")
    return worklist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score members and build the worklist."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[*sorted(REGISTRY), "all"],
        help="model to score with, or 'all'",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="source records table")
    parser.add_argument(
        "--artifacts", default=DEFAULT_ARTIFACTS_DIR, help="artifacts dir"
    )
    parser.add_argument(
        "--scope",
        default="evaluation",
        choices=["evaluation", "all"],
        help="score the held-out evaluation rows (default) or every member",
    )
    args = parser.parse_args()
    models = sorted(REGISTRY) if args.model == "all" else [args.model]
    for model_name in models:
        run(
            model_name,
            db_path=args.db,
            table=args.table,
            artifacts_dir=args.artifacts,
            scope=args.scope,
        )


if __name__ == "__main__":
    main()
