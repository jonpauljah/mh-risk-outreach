"""Streamlit page: run the trained models' risk scoring and compare the results.

A click-and-run demo of the scoring stage (issue #16) for issue #28: pick the
models, see that they are trained and how many members each will score, press
Run, watch a progress bar per model, then compare one worklist table per model
(risk score, band, priority, and SHAP-based reason). Reset clears the run.

Scoring reuses ``scoring.score_members`` on the held-out evaluation rows of the
shared split, with the training rows as the SHAP background, exactly like
``python -m mh_risk_outreach.scoring``. Each run replaces the model's scores
table and displays the rows read back from SQLite. A "Train" button is offered
for any model without a saved artifact.

Streamlit lists this page automatically beside ``ui/app.py``.
"""

import time

import pandas as pd
import streamlit as st

from mh_risk_outreach.bands import BAND_LABELS
from mh_risk_outreach.config import DEFAULT_DB_PATH, DEFAULT_TABLE, MEMBER_ID
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db import scores as scores_db
from mh_risk_outreach.db.repository import database_status, load_from_db
from mh_risk_outreach.models import REGISTRY
from mh_risk_outreach.models import train as model_training
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    artifact_model_version,
    load_scoring_artifact,
)
from mh_risk_outreach.scoring import score_members

IMPORT_COMMAND = "uv run python -m mh_risk_outreach.etl.pipeline"

# Session-state key holding the finished run: {model name: run details}.
RESULTS_KEY = "risk_scoring_results"

DISPLAY_COLUMNS = {
    "rank_priority": "Contact rank",
    MEMBER_ID: "Member ID",
    "risk_score": "Risk score",
    "band_level_priority": "Risk band",
    "safety_review_required": "Safety review required",
    "reasoning": "Reason",
}


@st.cache_data
def _load(db_path: str, table: str) -> pd.DataFrame:
    return load_from_db(db_path, table)


def _model_status(
    names: list[str], n_members: int, artifacts_dir: str, db_path: str, locked: bool
) -> bool:
    """Show one status card per selected model; return True if all are trained."""
    all_ready = True
    for column, name in zip(st.columns(len(names)), names, strict=True):
        try:
            version = artifact_model_version(name, artifacts_dir)
        except (TypeError, ValueError):
            version = None
            stale = True
        else:
            stale = False
        with column.container(border=True):
            st.markdown(f"**{name}**")
            if version:
                st.markdown(f"✅ Ready · {n_members:,} members prepared")
                st.caption(f"Model version `{version}`")
                continue
            all_ready = False
            st.markdown("⚠️ Needs retraining" if stale else "⚠️ Not trained yet")
            if not locked and st.button("Train", key=f"train_{name}"):
                with st.spinner(f"Training {name}..."):
                    model_training.run(
                        name, db_path=db_path, artifacts_dir=artifacts_dir
                    )
                st.rerun()
    return all_ready


def _run_scoring(
    names: list[str],
    members: pd.DataFrame,
    background: pd.DataFrame,
    artifacts_dir: str,
    db_path: str,
) -> dict[str, dict]:
    """Score, persist, and reload ``members`` for each model's worklist."""
    bars = {name: st.progress(0.0, text=f"{name}: waiting") for name in names}
    results: dict[str, dict] = {}
    for name in names:
        artifact = load_scoring_artifact(name, artifacts_dir)
        bar = bars[name]

        def update(fraction: float, name: str = name, bar=bar) -> None:
            done = round(fraction * len(members))
            bar.progress(
                fraction, text=f"{name}: explained {done:,} of {len(members):,}"
            )

        bar.progress(0.0, text=f"{name}: scoring {len(members):,} members")
        start = time.perf_counter()
        computed = score_members(
            members,
            artifact,
            background=background,
            progress=update,
        )
        scores_db.save_member_scores(computed, name, db_path=db_path)
        worklist = scores_db.load_member_scores(name, db_path=db_path)
        results[name] = {
            "worklist": worklist,
            "model_version": str(worklist["model_version"].iat[0]),
            "seconds": time.perf_counter() - start,
        }
        bar.progress(1.0, text=f"{name}: done")
    return results


def _show_results(results: dict[str, dict]) -> None:
    """One worklist table per model, in contact order."""
    for name, run in results.items():
        worklist = run["worklist"]
        st.subheader(name)
        counts = worklist["band_level_priority"].value_counts()
        bands = " · ".join(
            f"{band} {int(counts.get(band, 0)):,}" for band in reversed(BAND_LABELS)
        )
        st.caption(
            f"Model version `{run['model_version']}` · scored {len(worklist):,} members "
            f"in {run['seconds']:.1f}s · {bands}"
        )
        table = worklist[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Risk score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0, max_value=100, format="%.1f"
                ),
                "Safety review required": st.column_config.CheckboxColumn(
                    "Safety review required",
                    help="Reported self-harm or suicidal thoughts",
                ),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
            },
        )


def main() -> None:
    st.set_page_config(page_title="Risk scoring", layout="wide")
    st.title("Risk scoring")
    st.markdown(
        "Run the trained models on members they have never seen and compare the "
        "outreach worklists they produce. Each member gets a **risk score** (the "
        "model's estimated chance of a mental health issue, 0-100), a **risk band**, "
        "and a short **reason** naming the answers that pushed their score up. "
        "**Contact rank** is relative to the members in this run, with 1 first. "
        "Members requiring safety review are ranked first, then members are "
        "ordered by model score and ties by member ID."
    )

    exists, row_count = database_status(DEFAULT_DB_PATH, DEFAULT_TABLE)
    if not exists or row_count == 0:
        st.warning(
            f"No records found in `{DEFAULT_DB_PATH}` (table `{DEFAULT_TABLE}`). "
            "Run the import to populate the database, then reload this page."
        )
        st.code(IMPORT_COMMAND, language="bash")
        st.stop()

    records = _load(DEFAULT_DB_PATH, DEFAULT_TABLE)
    split = split_dataset(records)
    members = split.evaluation
    results = st.session_state.get(RESULTS_KEY)
    locked = results is not None

    names = st.multiselect(
        "Models to run",
        list(REGISTRY),
        default=list(REGISTRY),
        disabled=locked,
        key="risk_scoring_models",
    )
    if not names:
        st.info("Pick at least one model.")
        st.stop()

    all_ready = _model_status(
        names, len(members), DEFAULT_ARTIFACTS_DIR, DEFAULT_DB_PATH, locked
    )

    st.markdown(
        f"**Output:** for each of the {len(members):,} held-out members — risk score, "
        "SHAP-based reason, risk band, safety review flag, and contact rank, "
        "in one table per model."
    )
    if "random_forest" in names and not locked:
        st.caption("Random Forest explanations are slow: expect several minutes.")

    action = st.empty()
    if results is not None:
        if action.button("Reset", type="secondary"):
            del st.session_state[RESULTS_KEY]
            st.rerun()
        _show_results(results)
        return

    if not all_ready:
        action.info("Train the models marked above before running.")
        return
    if action.button("Run", type="primary"):
        action.empty()
        try:
            st.session_state[RESULTS_KEY] = _run_scoring(
                names, members, split.train, DEFAULT_ARTIFACTS_DIR, DEFAULT_DB_PATH
            )
        except SystemExit as exc:
            action.error(str(exc))
            return
        st.rerun()


main()
