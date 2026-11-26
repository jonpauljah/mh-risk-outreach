"""Persist outreach cases while retaining their original assessment link.

The manager-facing policy lives in mh_risk_outreach.outreach_policy.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from mh_risk_outreach.config import DEFAULT_DB_PATH, FEATURES
from mh_risk_outreach.db.assessments import _connect, _insert_assessment, _now
from mh_risk_outreach.db.operational_population import (
    load_operational_member_ids,
    require_operational_member,
)
from mh_risk_outreach.db.operational_scores import (
    WRONG_POPULATION,
    assessment_score_payload,
    worklist_requires_refresh,
)


def create_case(
    member_id: int,
    recommendation: str,
    scoring: dict[str, Any] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    recommendation_reason: str | None = None,
    recommendation_rule_version: str | None = None,
) -> int:
    """Persist a Pending case. Manager selections use OutreachPolicy.create_case."""
    if not recommendation:
        raise ValueError("recommendation is required")
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        case_id = _create_case(
            conn,
            member_id,
            recommendation,
            scoring,
            recommendation_reason=recommendation_reason,
            recommendation_rule_version=recommendation_rule_version,
        )
        conn.commit()
        return case_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_case(
    conn: sqlite3.Connection,
    member_id: int,
    recommendation: str,
    scoring: dict[str, Any] | None = None,
    *,
    recommendation_reason: str | None = None,
    recommendation_rule_version: str | None = None,
) -> int:
    """Write a case and any new baseline inside the caller's write transaction."""
    member = conn.execute(
        "SELECT * FROM patients WHERE Member_ID=?", (member_id,)
    ).fetchone()
    if member is None:
        raise ValueError(f"unknown member {member_id}")
    member_ids = load_operational_member_ids(conn)
    require_operational_member(member_id, member_ids)
    if worklist_requires_refresh(conn, member_ids):
        raise ValueError(WRONG_POPULATION)
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
        # The first case creates its baseline, current pointer, and case
        # together. Import and population scoring never create history.
        if scoring is None:
            published = conn.execute(
                """SELECT s.*, (SELECT COUNT(*) FROM operational_scores) AS cohort_size
                   FROM operational_scores AS s WHERE s.Member_ID=?""",
                (member_id,),
            ).fetchone()
            if published is None:
                raise ValueError(
                    f"member {member_id} has no published operational worklist score; "
                    "run scoring with --operational first"
                )
            baseline_result = assessment_score_payload(
                published,
                cohort_id=published["cohort_id"],
                cohort_calculated_at=published["calculated_at"],
                cohort_size=published["cohort_size"],
            )
        else:
            baseline_result = dict(scoring)
        inputs = {name: member[name] for name in FEATURES}
        baseline_result["recommendation_action"] = recommendation
        if recommendation_reason is not None:
            baseline_result["recommendation_reason"] = recommendation_reason
        if recommendation_rule_version is not None:
            baseline_result["recommendation_rule_version"] = recommendation_rule_version
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
        # Reuse the current snapshot only after its separate scoring run
        # succeeded; an older score must not stand in for pending inputs.
        if current["scoring_status"] != "Scored":
            raise ValueError("current assessment is pending scoring")
        assessment_id = int(current["assessment_id"])
    stored_action = current["recommendation_action"]
    # Copy the assessment's proposal into the case without allowing the
    # caller to silently change the historical recommendation.
    if stored_action is None:
        raise ValueError("current assessment has no stored recommendation action")
    if recommendation != stored_action:
        raise ValueError("case action conflicts with current assessment recommendation")
    case_action = stored_action
    # A completed case frees the active slot, but its assessment remains
    # used; a later case needs a new, finalized assessment.
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
    if cursor.lastrowid is None:
        raise RuntimeError("case insert did not return an ID")
    return cursor.lastrowid


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
        # The schema permits pending assessments with SQL NULL scoring fields.
        # Preserve that valid state when reading case history.
        rank_context = result.pop("rank_context_json")
        result["rank_context"] = (
            json.loads(rank_context) if rank_context is not None else None
        )
        return result
    finally:
        conn.close()


def update_case_decision(
    case_id: int,
    updates: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Persist manager decision fields without implementing transition policy."""
    # Field names are interpolated only after a fixed allowlist check. The
    # values remain bound parameters, and the schema validates stored enums.
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
    # IDs provide a stable order even when timestamps share the same instant.
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
