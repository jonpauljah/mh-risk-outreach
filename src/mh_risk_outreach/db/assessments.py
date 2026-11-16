"""Store full survey snapshots separately from replaceable model-score rows.

Reassessments are saved immediately; scoring is supplied by a later explicit run.
"""

import json
import math
import sqlite3
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from mh_risk_outreach.config import (
    BINARY_FLAGS,
    DEFAULT_DB_PATH,
    FEATURES,
    INTEGER_SCORES,
    REAL_HOURS,
    TEXT_CATEGORIES,
)
from mh_risk_outreach.db.repository import init_schema


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # SQLite foreign keys are disabled by default on each new connection.
    conn.execute("PRAGMA foreign_keys = ON")
    # The tracked schema adds missing tables without replacing existing history.
    init_schema(conn)
    return conn


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    # Persist the complete source-input contract, even when the form edits only
    # a few fields. Member_ID and the source target are stored elsewhere.
    if set(inputs) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(inputs))
        extra = sorted(set(inputs) - set(FEATURES))
        raise ValueError(
            f"inputs must contain exactly the 50 survey fields (missing={missing}, extra={extra})"
        )
    validated: dict[str, Any] = {}
    # Normalize numeric values so JSON snapshots have stable Python types.
    for name in INTEGER_SCORES:
        value = inputs[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a non-null integer")
        validated[name] = int(value)
    for name in REAL_HOURS:
        value = inputs[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{name} must be a finite, non-null number")
        validated[name] = float(value)
    for name in TEXT_CATEGORIES:
        value = inputs[name]
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a non-null string")
        validated[name] = value
    for name in BINARY_FLAGS:
        # These flags use the same 0/1 contract as the imported patient row.
        value = inputs[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value not in (0, 1)
        ):
            raise ValueError(f"{name} must be integer 0 or 1")
        validated[name] = int(value)
    return {name: validated[name] for name in FEATURES}


def _insert_assessment(
    conn: sqlite3.Connection,
    member_id: int,
    inputs: dict[str, Any],
    source: str,
    scoring: dict[str, Any] | None = None,
) -> int:
    snapshot = _validate_inputs(inputs)
    scored = scoring is not None
    values = scoring or {}
    # A fully scored baseline needs a real cohort context. json.dumps(None)
    # would otherwise store the JSON literal "null", which is SQL NOT NULL.
    if scored and not isinstance(values.get("rank_context"), dict):
        raise TypeError("rank_context must be a non-null dictionary")
    cursor = conn.execute(
        """INSERT INTO risk_assessments
        (Member_ID, source, created_at, inputs_json, scoring_status, risk_score,
         band_level_priority, rank_priority, rank_context_json, explanation,
         recommendation_action, recommendation_reason, recommendation_rule_version,
         operational_model_name, model_version, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            member_id,
            source,
            _now(),
            json.dumps(snapshot, sort_keys=True),
            "Scored" if scored else "PendingScoring",
            values.get("risk_score"),
            values.get("band_level_priority"),
            values.get("rank_priority"),
            json.dumps(values["rank_context"], sort_keys=True) if scored else None,
            values.get("explanation"),
            values.get("recommendation_action", values.get("recommendation")),
            values.get("recommendation_reason"),
            values.get("recommendation_rule_version"),
            values.get("operational_model_name"),
            values.get("model_version"),
            _now() if scored else None,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("assessment insert did not return an ID")
    return cursor.lastrowid


def submit_reassessment(
    member_id: int,
    edits: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Append a pending full snapshot, merging edits over the current snapshot."""
    unknown = set(edits) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown survey fields: {sorted(unknown)}")
    conn = _connect(db_path)
    try:
        # The pointer read and both writes share a write lock. This also keeps
        # concurrent submissions from merging against an outdated snapshot.
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            """SELECT a.inputs_json FROM member_assessment_state c
               JOIN risk_assessments a USING (assessment_id)
               WHERE c.Member_ID = ?""",
            (member_id,),
        ).fetchone()
        if current is None:
            exists = conn.execute(
                "SELECT 1 FROM patients WHERE Member_ID = ?", (member_id,)
            ).fetchone()
            if exists is None:
                raise ValueError(f"unknown member {member_id}")
            raise ValueError(
                "member has no assessment history; first case creates the baseline"
            )
        merged = json.loads(current["inputs_json"])
        merged.update(edits)
        # Every explicit submission gets its own history row, including an
        # unchanged submission. No outreach-case column is updated here.
        assessment_id = _insert_assessment(conn, member_id, merged, "reassessment")
        conn.execute(
            """INSERT INTO member_assessment_state(Member_ID, assessment_id, updated_at)
               VALUES (?, ?, ?) ON CONFLICT(Member_ID) DO UPDATE
               SET assessment_id=excluded.assessment_id, updated_at=excluded.updated_at""",
            (member_id, assessment_id, _now()),
        )
        conn.commit()
        return assessment_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finalize_scoring(
    assessment_id: int,
    result: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Finalize the current pending assessment once with a supplied score result."""
    required = {
        "risk_score",
        "band_level_priority",
        "rank_priority",
        "rank_context",
        "explanation",
        "model_version",
    }
    optional = {
        "recommendation",
        "recommendation_action",
        "recommendation_reason",
        "recommendation_rule_version",
        "operational_model_name",
    }
    # Apply the same context shape rule as fully scored baseline creation.
    # This also catches a missing key before the broader payload check.
    if not isinstance(result.get("rank_context"), dict):
        raise TypeError("rank_context must be a non-null dictionary")
    if not required.issubset(result) or not set(result).issubset(required | optional):
        raise ValueError(
            f"scoring result must contain {sorted(required)} and only supported optional recommendation metadata"
        )
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Only the latest pending assessment may transition to Scored. A trigger
        # prevents subsequent edits to any finalized scoring snapshot.
        cur = conn.execute(
            """UPDATE risk_assessments SET scoring_status='Scored', risk_score=?,
               band_level_priority=?, rank_priority=?, rank_context_json=?, explanation=?,
               recommendation_action=?, recommendation_reason=?, recommendation_rule_version=?,
               operational_model_name=?, model_version=?, scored_at=?
               WHERE assessment_id=? AND scoring_status='PendingScoring'
               AND assessment_id=(SELECT assessment_id FROM member_assessment_state WHERE Member_ID=risk_assessments.Member_ID)""",
            (
                result["risk_score"],
                result["band_level_priority"],
                result["rank_priority"],
                json.dumps(result["rank_context"], sort_keys=True),
                result["explanation"],
                result.get("recommendation_action", result.get("recommendation")),
                result.get("recommendation_reason"),
                result.get("recommendation_rule_version"),
                result.get("operational_model_name"),
                result["model_version"],
                _now(),
                assessment_id,
            ),
        )
        if cur.rowcount != 1:
            raise ValueError("assessment is absent, not current, or already finalized")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_assessment(
    assessment_id: int, db_path: str | Path = DEFAULT_DB_PATH
) -> dict[str, Any] | None:
    """Load one assessment with decoded JSON snapshots."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM risk_assessments WHERE assessment_id=?", (assessment_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["inputs"] = json.loads(result.pop("inputs_json"))
        rank_context = result.pop("rank_context_json")
        # PendingScoring is a valid readable state with no rank context yet.
        result["rank_context"] = json.loads(rank_context) if rank_context else None
        return result
    finally:
        conn.close()


def get_current_assessment(
    member_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Return a member's current-state pointer and its full assessment."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT assessment_id, updated_at FROM member_assessment_state WHERE Member_ID=?",
            (member_id,),
        ).fetchone()
        if row is None:
            return None
        state = dict(row)
    finally:
        conn.close()
    # Current means latest submitted; it may still be awaiting scoring.
    state["Member_ID"] = member_id
    state["assessment"] = get_assessment(state["assessment_id"], db_path)
    return state
