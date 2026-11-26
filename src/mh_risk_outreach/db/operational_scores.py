"""Store the replaceable held-out worklist independently of history."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    MEMBER_ID,
    OPERATIONAL_SCOPE,
    RANK_ORDER,
)
from mh_risk_outreach.datasets import DatasetSplit
from mh_risk_outreach.db.assessments import _finalize_scoring, _validate_scoring
from mh_risk_outreach.db.operational_population import (
    load_operational_member_ids,
    load_operational_split,
    require_operational_member,
)
from mh_risk_outreach.db.repository import init_schema
from mh_risk_outreach.recommendations import Recommendation

NO_WORKLIST = "no operational worklist; run scoring with --operational first"
WRONG_POPULATION = (
    "published operational worklist differs from the held-out population; "
    "refresh scoring with --operational"
)


class AssessmentInputs(NamedTuple):
    member_id: int
    current_id: int
    current_inputs: str
    status: str
    scored_id: int | None
    scored_inputs: str | None


def _assessment_inputs(
    conn: sqlite3.Connection, member_ids: set[int]
) -> list[AssessmentInputs]:
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
    return [AssessmentInputs(*row) for row in rows if row[0] in member_ids]


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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _published_member_ids(conn: sqlite3.Connection) -> set[int]:
    if not _table_exists(conn, "operational_scores"):
        return set()
    return {row[0] for row in conn.execute("SELECT Member_ID FROM operational_scores")}


def worklist_requires_refresh(conn: sqlite3.Connection, member_ids: set[int]) -> bool:
    """Detect any published cohort with missing or out-of-scope members."""
    published = _published_member_ids(conn)
    return bool(published and published != member_ids)


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
    db_path: str = DEFAULT_DB_PATH, *, require_current_population: bool = False
) -> tuple[DatasetSplit, list[AssessmentInputs], PublishedCohort | None]:
    """Read the original split, in-scope assessment inputs, and cohort together."""
    if not Path(db_path).is_file():
        raise ValueError(
            "import the patient dataset before running operational scoring"
        )
    with closing(sqlite3.connect(db_path)) as conn, conn:
        init_schema(conn)
        conn.execute("BEGIN")
        split = load_operational_split(conn)
        member_ids = set(split.evaluation[MEMBER_ID])
        if require_current_population and worklist_requires_refresh(conn, member_ids):
            raise ValueError(WRONG_POPULATION)
        return split, _assessment_inputs(conn, member_ids), _published_cohort(conn)


def save_operational_scores(
    scores: pd.DataFrame,
    *,
    expected_inputs: list[AssessmentInputs],
    expected_cohort_id: str | None,
    baseline_recommendations: dict[int, Recommendation] | None = None,
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
        member_ids = load_operational_member_ids(conn)
        if len(rows) != len(member_ids) or set(rows[MEMBER_ID]) != member_ids:
            raise ValueError("operational scores must cover the held-out population")
        if _assessment_inputs(conn, member_ids) != expected_inputs:
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
        if baseline_recommendations:
            _save_baseline_recommendations(
                conn, str(rows["cohort_id"].iat[0]), baseline_recommendations
            )
        if validate_model is not None:
            validate_model()


def _save_baseline_recommendations(
    conn: sqlite3.Connection, cohort_id: str, recommendations: dict[int, Recommendation]
) -> None:
    conn.executemany(
        """INSERT INTO operational_recommendations
           (Member_ID, cohort_id, recommendation_action,
            recommendation_reason, recommendation_rule_version)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (Member_ID, cohort_id) DO NOTHING""",
        (
            (
                member_id,
                cohort_id,
                recommendation["recommendation_action"],
                recommendation["recommendation_reason"],
                recommendation["recommendation_rule_version"],
            )
            for member_id, recommendation in recommendations.items()
        ),
    )


def load_operational_scores(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return the published worklist in contact order."""
    if Path(db_path).is_file():
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("BEGIN")
            if _table_exists(conn, "patients"):
                return _load_operational_scores(conn, load_operational_member_ids(conn))
    raise ValueError(NO_WORKLIST)


def load_candidate_scoring_inputs(
    member_id: int, db_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the ranking cohort and original SHAP background from one snapshot."""
    if Path(db_path).is_file():
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("BEGIN")
            if _table_exists(conn, "patients"):
                split = load_operational_split(conn)
                member_ids = set(split.evaluation[MEMBER_ID])
                require_operational_member(member_id, member_ids)
                return _load_operational_scores(conn, member_ids), split.train
    raise ValueError(NO_WORKLIST)


def _load_operational_scores(
    conn: sqlite3.Connection, member_ids: set[int]
) -> pd.DataFrame:
    published = _published_member_ids(conn)
    if not published:
        raise ValueError(NO_WORKLIST)
    if published != member_ids:
        raise ValueError(WRONG_POPULATION)
    return pd.read_sql_query(
        "SELECT * FROM operational_scores ORDER BY rank_priority", conn
    )


class OperationalPopulationAudit(NamedTuple):
    training_count: int
    held_out_count: int
    published_count: int
    requires_refresh: bool
    excluded_assessments: list[tuple[int, int, str]]
    excluded_cases: list[tuple[int, int, str]]


def audit_operational_population(
    db_path: str = DEFAULT_DB_PATH,
) -> OperationalPopulationAudit:
    """Read the worklist scope and any out-of-scope history without writing to SQLite."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn, conn:
        conn.execute("BEGIN")
        if not _table_exists(conn, "patients"):
            raise ValueError(
                "import the patient dataset before auditing operational scoring"
            )
        split = load_operational_split(conn, ids_only=True)
        member_ids = set(split.evaluation[MEMBER_ID])
        published = _published_member_ids(conn)
        assessments = (
            [
                (member_id, assessment_id, status)
                for member_id, assessment_id, status in conn.execute(
                    "SELECT Member_ID, assessment_id, scoring_status FROM risk_assessments "
                    "ORDER BY Member_ID, assessment_id"
                )
                if member_id not in member_ids
            ]
            if _table_exists(conn, "risk_assessments")
            else []
        )
        cases = (
            [
                (member_id, case_id, status)
                for member_id, case_id, status in conn.execute(
                    "SELECT Member_ID, case_id, status FROM outreach_cases "
                    "ORDER BY Member_ID, case_id"
                )
                if member_id not in member_ids
            ]
            if _table_exists(conn, "outreach_cases")
            else []
        )
        return OperationalPopulationAudit(
            len(split.train),
            len(member_ids),
            len(published),
            bool(published and published != member_ids),
            assessments,
            cases,
        )
