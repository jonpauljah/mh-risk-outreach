"""Atomic case creation and read access for outreach history."""

import json
from pathlib import Path
from typing import Any

from mh_risk_outreach.config import DEFAULT_DB_PATH, FEATURES
from mh_risk_outreach.db.assessments import _connect, _insert_assessment, _now


def create_case(
    member_id: int,
    recommendation: str,
    scoring: dict[str, Any] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Create a Pending case using current scored history or an on-demand baseline."""
    if not recommendation:
        raise ValueError("recommendation is required")
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        member = conn.execute(
            "SELECT * FROM patients WHERE Member_ID=?", (member_id,)
        ).fetchone()
        if member is None:
            raise ValueError(f"unknown member {member_id}")
        if conn.execute(
            "SELECT 1 FROM outreach_cases WHERE Member_ID=? AND status IN ('Pending','Accepted')",
            (member_id,),
        ).fetchone():
            raise ValueError("member already has an active case")
        current = conn.execute(
            """SELECT a.* FROM member_assessment_state c JOIN risk_assessments a
               ON a.assessment_id=c.assessment_id WHERE c.Member_ID=?""",
            (member_id,),
        ).fetchone()
        if current is None:
            if scoring is None:
                raise ValueError(
                    "a supplied scoring result is required for the first-case baseline"
                )
            inputs = {name: member[name] for name in FEATURES}
            baseline_result = dict(scoring)
            baseline_result["recommendation_action"] = recommendation
            assessment_id = _insert_assessment(
                conn, member_id, inputs, "baseline", baseline_result
            )
            conn.execute(
                "INSERT INTO member_assessment_state(Member_ID, assessment_id, updated_at) VALUES (?,?,?)",
                (member_id, assessment_id, _now()),
            )
            current = conn.execute(
                "SELECT * FROM risk_assessments WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
        else:
            if current["scoring_status"] != "Scored":
                raise ValueError("current assessment is pending scoring")
            assessment_id = int(current["assessment_id"])
        stored_action = current["recommendation_action"]
        if stored_action is None:
            raise ValueError("current assessment has no stored recommendation action")
        if recommendation != stored_action:
            raise ValueError(
                "case action conflicts with current assessment recommendation"
            )
        case_action = stored_action
        if conn.execute(
            "SELECT 1 FROM outreach_cases WHERE assessment_id=?", (assessment_id,)
        ).fetchone():
            raise ValueError("current assessment has already been used for a case")
        cursor = conn.execute(
            """INSERT INTO outreach_cases
               (Member_ID, assessment_id, status, outcome, recommended_action,
                recommendation_reason, operational_model_name, model_version,
                recommendation_rule_version, created_at)
               VALUES (?, ?, 'Pending', NULL, ?, ?, ?, ?, ?, ?)""",
            (
                member_id,
                assessment_id,
                case_action,
                current["recommendation_reason"],
                current["operational_model_name"],
                current["model_version"],
                current["recommendation_rule_version"],
                _now(),
            ),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("case insert did not return an ID")
        return cursor.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_case(
    case_id: int, db_path: str | Path = DEFAULT_DB_PATH
) -> dict[str, Any] | None:
    """Load one outreach case and its original assessment."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT c.*, a.source, a.created_at AS assessment_created_at,
                      a.scoring_status, a.risk_score, a.band_level_priority,
                      a.rank_priority, a.rank_context_json, a.explanation, a.model_version
               FROM outreach_cases c JOIN risk_assessments a USING (assessment_id)
               WHERE c.case_id=?""",
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["rank_context"] = json.loads(result.pop("rank_context_json"))
        return result
    finally:
        conn.close()


def update_case_decision(
    case_id: int,
    updates: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Persist manager decision fields without implementing transition policy."""
    allowed = {
        "status",
        "outcome",
        "selected_action",
        "accepted_action",
        "override_justification",
        "accepted_at",
        "outcome_recorded_at",
        "completed_at",
    }
    if not updates or not set(updates).issubset(allowed):
        raise ValueError(f"updates must use one or more of {sorted(allowed)}")
    assignments = ", ".join(f"{column}=?" for column in updates)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            f"UPDATE outreach_cases SET {assignments} WHERE case_id=?",
            (*updates.values(), case_id),
        )
        if cur.rowcount != 1:
            raise ValueError(f"unknown case {case_id}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_member_history(
    member_id: int, db_path: str | Path = DEFAULT_DB_PATH
) -> dict[str, list[dict[str, Any]]]:
    """Return assessment and case history in stable insertion order."""
    conn = _connect(db_path)
    try:
        assessments = conn.execute(
            "SELECT assessment_id FROM risk_assessments WHERE Member_ID=? ORDER BY assessment_id",
            (member_id,),
        ).fetchall()
        cases = conn.execute(
            "SELECT case_id FROM outreach_cases WHERE Member_ID=? ORDER BY case_id",
            (member_id,),
        ).fetchall()
    finally:
        conn.close()
    from mh_risk_outreach.db.assessments import get_assessment

    assessment_history = [
        assessment
        for row in assessments
        if (assessment := get_assessment(row[0], db_path)) is not None
    ]
    case_history = [
        case for row in cases if (case := get_case(row[0], db_path)) is not None
    ]

    return {
        "assessments": assessment_history,
        "cases": case_history,
    }
