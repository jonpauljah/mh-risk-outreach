"""Store the replaceable full-population worklist independently of history."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from mh_risk_outreach.config import DEFAULT_DB_PATH
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


def load_operational_inputs(
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, list[AssessmentInputs]]:
    """Read imported inputs and assessment provenance from one SQLite snapshot."""
    if not Path(db_path).is_file():
        raise ValueError(
            "import the patient dataset before running operational scoring"
        )
    with closing(sqlite3.connect(db_path)) as conn, conn:
        init_schema(conn)
        conn.execute("BEGIN")
        patients = pd.read_sql_query("SELECT * FROM patients ORDER BY Member_ID", conn)
        return patients, _assessment_inputs(conn)


def save_operational_scores(
    scores: pd.DataFrame,
    *,
    expected_inputs: list[AssessmentInputs],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Publish the full worklist together, provided its assessment inputs are current."""
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
        conn.execute("DELETE FROM operational_scores")
        conn.executemany(statement, rows.itertuples(index=False, name=None))


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
