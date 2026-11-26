"""Store the replaceable full-population worklist independently of history."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from mh_risk_outreach.config import DEFAULT_DB_PATH, OPERATIONAL_SCOPE, RANK_ORDER
from mh_risk_outreach.db.assessments import _finalize_scoring, _validate_scoring
from mh_risk_outreach.db.repository import init_schema


class AssessmentInputs(NamedTuple):
    member_id: int
    current_id: int
    current_inputs: str
    status: str
    scored_id: int | None
    scored_inputs: str | None


def _assessment_inputs(conn: sqlite3.Connection) -> list[AssessmentInputs]:
    rows = conn.execute(
        """SELECT state.Member_ID, current.assessment_id, current.inputs_json,
                  current.scoring_status, scored.assessment_id, scored.inputs_json
           FROM member_assessment_state AS state
           JOIN risk_assessments AS current ON current.assessment_id = state.assessment_id
           LEFT JOIN risk_assessments AS scored ON scored.assessment_id = (
               SELECT a.assessment_id FROM risk_assessments AS a
               WHERE a.Member_ID = state.Member_ID AND a.scoring_status = 'Scored'
               ORDER BY a.assessment_id DESC LIMIT 1
           )
           ORDER BY state.Member_ID"""
    )
    return [AssessmentInputs(*row) for row in rows]


class PublishedCohort(NamedTuple):
    cohort_id: str
    model_name: str
    model_version: str


def _published_cohort(conn: sqlite3.Connection) -> PublishedCohort | None:
    row = conn.execute(
        "SELECT cohort_id, operational_model_name, model_version "
        "FROM operational_scores LIMIT 1"
    ).fetchone()
    return PublishedCohort(*row) if row is not None else None


def assessment_score_payload(
    row: sqlite3.Row | pd.Series[Any],
    *,
    cohort_id: str,
    cohort_calculated_at: str,
    cohort_size: int,
    calculated_at: str | None = None,
) -> dict[str, object]:
    """Build the history payload for a published row or a candidate rank."""
    return {
        "risk_score": float(row["risk_score"]),
        "band_level_priority": str(row["band_level_priority"]),
        "rank_priority": int(row["rank_priority"]),
        "explanation": str(row["reasoning"]),
        "operational_model_name": str(row["operational_model_name"]),
        "model_version": str(row["model_version"]),
        "rank_context": {
            "cohort_id": cohort_id,
            "cohort_calculated_at": cohort_calculated_at,
            "calculated_at": calculated_at or cohort_calculated_at,
            "scope": OPERATIONAL_SCOPE,
            "cohort_size": cohort_size,
            "safety_review_required": bool(row["safety_review_required"]),
            "ordering": list(RANK_ORDER),
        },
    }


def load_operational_inputs(
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, list[AssessmentInputs], PublishedCohort | None]:
    """Read imported inputs, assessment provenance, and the current cohort together."""
    if not Path(db_path).is_file():
        raise ValueError(
            "import the patient dataset before running operational scoring"
        )
    with closing(sqlite3.connect(db_path)) as conn, conn:
        init_schema(conn)
        conn.execute("BEGIN")
        patients = pd.read_sql_query("SELECT * FROM patients ORDER BY Member_ID", conn)
        return patients, _assessment_inputs(conn), _published_cohort(conn)


def save_operational_scores(
    scores: pd.DataFrame,
    *,
    expected_inputs: list[AssessmentInputs],
    expected_cohort_id: str | None,
    reassessment: tuple[int, dict[str, object]] | None = None,
    validate_model: Callable[[], None] | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Publish the worklist and optionally finalize its pending assessment."""
    if reassessment is not None:
        _validate_scoring(reassessment[1])
    columns = (
        "Member_ID",
        "operational_model_name",
        "model_version",
        "risk_score",
        "band_level_priority",
        "rank_priority",
        "reasoning",
        "safety_review_required",
        "score_assessment_id",
        "safety_assessment_id",
        "inputs_pending_scoring",
        "cohort_id",
        "calculated_at",
    )
    rows = scores[list(columns)].astype(object)
    rows = rows.where(rows.notna(), None)
    placeholders = ", ".join("?" for _ in columns)
    statement = (
        f"INSERT INTO operational_scores ({', '.join(columns)}) VALUES ({placeholders})"
    )

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if _assessment_inputs(conn) != expected_inputs:
            raise ValueError(
                "assessment inputs changed during scoring; run scoring again"
            )
        cohort = _published_cohort(conn)
        if (cohort.cohort_id if cohort else None) != expected_cohort_id:
            raise ValueError(
                "operational worklist changed during scoring; run scoring again"
            )
        if reassessment is not None:
            _finalize_scoring(conn, *reassessment)
        conn.execute("DELETE FROM operational_scores")
        conn.executemany(statement, rows.itertuples(index=False, name=None))
        if validate_model is not None:
            validate_model()


def load_operational_scores(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return the published worklist in contact order."""
    if Path(db_path).is_file():
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("BEGIN")
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='operational_scores'"
            ).fetchone()
            if table_exists:
                scores = pd.read_sql_query(
                    "SELECT * FROM operational_scores ORDER BY rank_priority", conn
                )
                if not scores.empty:
                    return scores
    raise ValueError("no operational worklist; run scoring with --operational first")
