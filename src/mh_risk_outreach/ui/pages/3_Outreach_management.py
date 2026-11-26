"""Streamlit worklists and manager-selected outreach case creation."""

import logging
import os
import sqlite3
from dataclasses import replace
from hashlib import sha256
from typing import Any

import pandas as pd
import streamlit as st

from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    DEFAULT_OUTREACH_CAPACITY,
    DEFAULT_TABLE,
    OUTREACH_CAPACITY_ENV,
)
from mh_risk_outreach.db.repository import database_status
from mh_risk_outreach.models.artifacts import DEFAULT_ARTIFACTS_DIR
from mh_risk_outreach.outreach_policy import OutreachCandidate, OutreachPolicy

SCORE_COMMAND = "uv run python -m mh_risk_outreach.scoring --operational"
SAFETY_LABELS = {
    "Self_Harm_Thoughts": "Reported self-harm thoughts",
    "Suicidal_Thoughts": "Reported suicidal thoughts",
}
CREATION_RESULT = "outreach_creation_result"
UPDATE_CASE_ID = "outreach_update_case_id"
logger = logging.getLogger(__name__)


def _selection_status(candidate: OutreachCandidate) -> str:
    if candidate.selectable:
        return (
            "Eligible"
            if candidate.recommendation_action is not None
            else "Recommendation unavailable"
        )
    reasons = []
    if candidate.has_active_case:
        reasons.append("Member already has an active case")
    if candidate.pending_scoring:
        reasons.append("Current assessment is awaiting scoring")
    elif candidate.cohort_id is None:
        reasons.append("Current operational score unavailable")
    return ". ".join(reasons) or "Current assessment is ineligible for case creation"


def _candidate_row(candidate: OutreachCandidate) -> dict[str, object]:
    row: dict[str, object] = {"Member ID": candidate.member_id}
    if candidate.safety_flags:
        row["Safety concerns"] = ". ".join(
            SAFETY_LABELS[flag] for flag in candidate.safety_flags
        )
    row.update(
        {
            "Risk score": candidate.risk_score,
            "Risk band": candidate.band,
            "Contact rank": candidate.rank,
            "Recommended action": candidate.recommendation_action,
            "Recommendation explanation": candidate.recommendation_reason,
            "Model explanation": candidate.explanation,
            "Case creation": _selection_status(candidate),
        }
    )
    return row


def _create_cases(
    policy: OutreachPolicy,
    candidates: tuple[OutreachCandidate, ...],
    selection_key: str,
) -> None:
    # The callback receives the candidates from the screen the manager submitted.
    # It runs before the new worklists are loaded for the rerun.
    member_ids = set(st.session_state[selection_key])
    selections = [
        candidate for candidate in candidates if candidate.member_id in member_ids
    ]
    if not selections:
        st.session_state[CREATION_RESULT] = (
            False,
            "Select at least one eligible member.",
        )
        return

    try:
        with st.spinner("Creating outreach cases…"):
            case_ids = policy.create_cases(selections)
    except sqlite3.IntegrityError as exc:
        message = (
            "A selected member or assessment already has a case. Review the refreshed worklist."
            if "UNIQUE constraint failed" in str(exc)
            else "A database conflict prevented the batch from being saved. Please try again."
        )
    except sqlite3.Error:
        message = "The database could not save the batch. Please try again."
    except (SystemExit, TypeError, ValueError) as exc:
        message = str(exc).rstrip(".") + "."
        if "selection is no longer eligible" in message:
            message = (
                "A selected member is no longer eligible or already has an active case. "
                "Review the refreshed worklist and select the members again."
            )
        elif "standard outreach is at capacity" in message:
            message = "The selection exceeds the current standard outreach capacity."
    else:
        count = len(case_ids)
        message = (
            "Created one Pending case."
            if count == 1
            else f"Created {count} Pending cases."
        )
        st.session_state[CREATION_RESULT] = (True, message)
        return
    st.session_state[CREATION_RESULT] = (False, f"No cases were created. {message}")


def _worklist(
    policy: OutreachPolicy,
    candidates: tuple[OutreachCandidate, ...],
    *,
    queue: str,
    empty_message: str,
) -> None:
    if not candidates:
        st.info(empty_message)
        return

    if any(c.selectable and c.recommendation_action is None for c in candidates):
        try:
            with st.spinner("Preparing recommendations…"):
                previews = policy.prepare_recommendations(candidates)
        except (SystemExit, Exception) as exc:
            logger.exception("Could not prepare outreach recommendations")
            st.warning(f"Some recommendations could not be prepared. {exc}")
        else:
            candidates = tuple(
                replace(c, **previews[c.member_id]) if c.member_id in previews else c
                for c in candidates
            )

    st.dataframe(
        pd.DataFrame([_candidate_row(c) for c in candidates]),
        hide_index=True,
        column_config={
            "Risk score": st.column_config.NumberColumn(
                "Risk score (0–100)", format="%.1f"
            ),
        },
    )
    eligible = tuple(
        c for c in candidates if c.selectable and c.recommendation_action is not None
    )
    if not eligible:
        return

    # Clear a draft when an intervening rerun changes the offered snapshots.
    snapshot = sha256(repr(eligible).encode()).hexdigest()
    selection_key = f"outreach_{queue}_members_{snapshot}"
    with st.form(f"outreach_{queue}", clear_on_submit=True, enter_to_submit=False):
        st.multiselect(
            "Members to create cases for",
            [c.member_id for c in eligible],
            key=selection_key,
        )
        st.form_submit_button(
            "Create Outreach Cases",
            type="primary",
            on_click=_create_cases,
            args=(policy, eligible, selection_key),
        )


def _active_cases(cases: list[dict[str, Any]]) -> None:
    by_id = {case["case_id"]: case for case in cases}
    open_id = st.session_state.get(UPDATE_CASE_ID)
    if open_id in by_id:
        case = by_id[open_id]
        if st.button("Back to Active Cases"):
            del st.session_state[UPDATE_CASE_ID]
            st.rerun()
        st.subheader(f"Case {open_id} · Member {case['Member_ID']}")
        st.write(f"**{case['status']}** · Assessment {case['assessment_id']}")
        st.write(f"Original recommendation · {case['recommended_action']}")
        if case["recommendation_reason"]:
            st.write(case["recommendation_reason"])
        return
    if open_id is not None:
        st.session_state.pop(UPDATE_CASE_ID)
        st.info(f"Case {open_id} is no longer active.")
    if not cases:
        st.info("There are no active outreach cases.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Case ID": case["case_id"],
                    "Member ID": case["Member_ID"],
                    "Status": case["status"],
                    "Recommended action": case["recommended_action"],
                    "Accepted action": case["accepted_action"],
                }
                for case in cases
            ]
        ),
        hide_index=True,
    )
    case_id = st.selectbox(
        "Case to update",
        list(by_id),
        index=None,
        format_func=lambda item: f"Case {item} · Member {by_id[item]['Member_ID']}",
    )
    if st.button("Update Case", disabled=case_id is None):
        st.session_state[UPDATE_CASE_ID] = case_id
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Outreach Management", layout="wide")
    st.title("Outreach Management")
    st.caption(
        "Model scores estimate risk. Review the recommendation and member context before creating a case."
    )
    st.button("Refresh outreach")

    result = st.session_state.pop(CREATION_RESULT, None)
    if result:
        success, message = result
        (st.success if success else st.error)(message)

    try:
        capacity = int(
            os.environ.get(OUTREACH_CAPACITY_ENV, str(DEFAULT_OUTREACH_CAPACITY))
        )
        policy = OutreachPolicy(
            capacity, db_path=DEFAULT_DB_PATH, artifacts_dir=DEFAULT_ARTIFACTS_DIR
        )
    except ValueError:
        st.error(f"Set {OUTREACH_CAPACITY_ENV} to a nonnegative integer.")
        return

    try:
        _, rows = database_status(policy.db_path, DEFAULT_TABLE)
        worklists, cases = policy.dashboard() if rows else (None, [])
    except (sqlite3.Error, ValueError):
        st.error(
            "The outreach database could not be read. Refresh outreach to try again."
        )
        return

    if worklists:
        for column, (label, value) in zip(
            st.columns(3),
            (
                ("Shared capacity (K)", policy.capacity),
                ("Active cases", worklists.active_cases),
                ("Remaining standard slots", worklists.standard_slots),
            ),
            strict=True,
        ):
            column.metric(label, value)
        if worklists.scoring_issue:
            st.warning(
                f"{worklists.scoring_issue} Run `{SCORE_COMMAND}` when the model is ready."
            )
    else:
        st.warning(
            "No member records found. Run `uv run python -m mh_risk_outreach.etl.pipeline` to import them."
        )

    safety, standard, active = st.tabs(
        ["Safety Review", "Standard Outreach", "Active Cases"]
    )
    with safety:
        _worklist(
            policy,
            worklists.safety_review if worklists else (),
            queue="safety",
            empty_message="There are no unreviewed safety concerns.",
        )
    with standard:
        _worklist(
            policy,
            worklists.standard_outreach if worklists else (),
            queue="standard",
            empty_message=(
                "Standard outreach is at capacity."
                if worklists and worklists.standard_slots == 0
                else "There are no eligible standard outreach candidates."
            ),
        )
    with active:
        _active_cases(cases)


main()
