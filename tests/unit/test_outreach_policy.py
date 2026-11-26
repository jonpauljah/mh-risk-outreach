"""Exercise outreach selections and write-lock rechecks against SQLite."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier

import pandas as pd
import pytest

from mh_risk_outreach import outreach_policy
from mh_risk_outreach.config import (
    DEFAULT_CSV_PATH,
    MEMBER_ID,
    OPERATIONAL_MODEL_ENV,
    TARGET,
)
from mh_risk_outreach.db.assessments import finalize_scoring, submit_reassessment
from mh_risk_outreach.db.outreach import (
    accept_case,
    complete_case,
    get_case,
    list_active_cases,
    list_member_history,
    record_case_outcome,
)
from mh_risk_outreach.db.repository import save_records
from mh_risk_outreach.outreach_policy import OutreachCandidate, OutreachPolicy
from mh_risk_outreach.recommendations import SAFETY_REVIEW_ACTION, recommend_action


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.delenv(OPERATIONAL_MODEL_ENV, raising=False)
    path = str(tmp_path / "outreach.sqlite")
    records = pd.read_csv(DEFAULT_CSV_PATH, nrows=25)
    records.insert(0, MEMBER_ID, range(1, 26))
    records[TARGET] = records.index % 2
    # Member 7 is held out and member 1 is in training. Both are flagged.
    records["Self_Harm_Thoughts"] = records[MEMBER_ID].isin([1, 7]).astype(int)
    records["Suicidal_Thoughts"] = 0
    save_records(records, path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executemany(
            """INSERT INTO operational_scores
               VALUES (?, 'logistic_regression', 'v1', ?, ?, ?, 'Score evidence',
                       ?, NULL, NULL, 0, 'cohort-1', '2026-11-16T11:00:00+00:00')""",
            [
                (7, 25, "Low", 1, 1),
                (13, 98, "Very High", 2, 0),
                (16, 86, "High", 3, 0),
                (21, 83, "High", 4, 0),
                (24, 54, "Medium", 5, 0),
            ],
        )

    monkeypatch.setattr(outreach_policy, "artifact_model_version", lambda *args: "v1")

    def score_member(member_id, inputs, **kwargs):
        return {**recommend_action(inputs), "rank_context": {"cohort_id": "cohort-1"}}

    monkeypatch.setattr(outreach_policy, "score_operational_member", score_member)
    return OutreachPolicy(2, db_path=path)


def complete(case_id, db_path):
    accept_case(case_id, db_path)
    record_case_outcome(case_id, "Reached — engaged", db_path)
    complete_case(case_id, db_path)


def test_bulk_selection_creates_only_selected_cases_and_honors_safety_exception(policy):
    worklists = policy.worklists()
    first, second = worklists.standard_outreach
    (safety,) = worklists.safety_review
    assert policy.create_cases([]) == []
    with pytest.raises(ValueError, match="only once"):
        policy.create_cases([first, first])
    assert list_active_cases(policy.db_path) == []

    selected = (safety, second, first)
    case_ids = policy.create_cases(iter(selected))
    cases = list_active_cases(policy.db_path)
    assert [case["case_id"] for case in cases] == case_ids
    assert [case["Member_ID"] for case in cases] == [c.member_id for c in selected]
    assert all(
        case["status"] == "Pending" and case["outcome"] is None for case in cases
    )
    assert list_member_history(21, policy.db_path) == {"cases": [], "assessments": []}
    assert policy.worklists().active_cases == 3
    assert policy.worklists().standard_slots == 0

    wider = replace(policy, capacity=10)
    with pytest.raises(ValueError, match="no longer eligible"):
        wider.create_cases([first, wider.worklists().standard_outreach[0]])
    assert list_active_cases(policy.db_path) == cases


def test_bulk_selection_rolls_back_earlier_cases_and_baselines_on_insert_failure(
    policy,
):
    first, second = policy.worklists().standard_outreach
    with closing(sqlite3.connect(policy.db_path)) as conn, conn:
        conn.execute(
            f"""CREATE TRIGGER fail_second_case BEFORE INSERT ON outreach_cases
                WHEN NEW.Member_ID = {second.member_id}
                BEGIN SELECT RAISE(ABORT, 'second case failed'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="second case failed"):
        policy.create_cases([first, second])

    with closing(sqlite3.connect(policy.db_path)) as conn:
        for table in ("outreach_cases", "risk_assessments", "member_assessment_state"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert policy.worklists().standard_outreach == (first, second)


def test_bulk_selection_rechecks_every_candidate_after_recommendation(
    policy, monkeypatch
):
    first, second = policy.worklists().standard_outreach
    original = outreach_policy.score_operational_member

    def preview(member_id, inputs, **kwargs):
        if member_id == second.member_id:
            with closing(sqlite3.connect(policy.db_path)) as conn, conn:
                conn.execute(
                    "UPDATE operational_scores SET band_level_priority='Medium' WHERE Member_ID=?",
                    (member_id,),
                )
        return original(member_id, inputs, **kwargs)

    monkeypatch.setattr(outreach_policy, "score_operational_member", preview)
    with pytest.raises(ValueError, match="no longer eligible"):
        policy.create_cases([first, second])
    assert list_active_cases(policy.db_path) == []
    assert list_member_history(first.member_id, policy.db_path) == {
        "cases": [],
        "assessments": [],
    }


def test_bulk_selection_rechecks_shared_capacity_after_recommendation(
    policy, monkeypatch
):
    worklists = policy.worklists()
    first, second = worklists.standard_outreach
    (safety,) = worklists.safety_review
    original = outreach_policy.score_operational_member

    def preview(member_id, inputs, **kwargs):
        if member_id == second.member_id:
            policy.create_case(safety)
        return original(member_id, inputs, **kwargs)

    monkeypatch.setattr(outreach_policy, "score_operational_member", preview)
    with pytest.raises(ValueError, match="at capacity"):
        policy.create_cases([first, second])
    assert [case["Member_ID"] for case in list_active_cases(policy.db_path)] == [
        safety.member_id
    ]
    assert policy.worklists().standard_slots == 1


def test_worklists_and_safety_exception_use_shared_capacity(policy):
    first = policy.worklists()
    assert first.active_cases == 0
    assert first.standard_slots == 2
    assert [c.member_id for c in first.standard_outreach] == [13, 16]
    wider = replace(policy, capacity=10).worklists()
    assert [c.member_id for c in wider.standard_outreach] == [13, 16, 21]
    (safety,) = first.safety_review
    assert safety.band == "Low"
    assert safety.safety_flags == ("Self_Harm_Thoughts",)

    standard_id = policy.create_case(first.standard_outreach[0])
    assert [c.member_id for c in policy.worklists().standard_outreach] == [16]
    accept_case(standard_id, policy.db_path)
    other_id = policy.create_case(first.standard_outreach[1])
    with pytest.raises(ValueError, match="at capacity"):
        policy.create_case(first.standard_outreach[1])

    full = policy.worklists()
    assert full.active_cases == 2
    assert full.standard_slots == 0
    assert full.standard_outreach == ()
    assert full.safety_review == (safety,)
    safety_id = policy.create_case(safety)
    case = get_case(safety_id, policy.db_path)
    assert case["recommended_action"] == SAFETY_REVIEW_ACTION
    assert case["band_level_priority"] == "Low"
    assert case["risk_score"] == 25
    assert policy.worklists().active_cases == 3
    assert policy.worklists().safety_review == ()

    complete(other_id, policy.db_path)
    assert policy.worklists().standard_slots == 0
    complete(safety_id, policy.db_path)
    freed = policy.worklists()
    assert freed.standard_slots == 1
    assert [c.member_id for c in freed.standard_outreach] == [21]
    assert freed.safety_review == ()


def test_latest_flags_visible_before_scoring_even_with_active_case(policy):
    first = policy.worklists().standard_outreach[0]
    case_id = policy.create_case(first)
    pending_id = submit_reassessment(13, {"Suicidal_Thoughts": 1}, policy.db_path)
    full = replace(policy, capacity=1)
    entries = full.worklists()
    assert entries.standard_slots == 0
    pending = next(c for c in entries.safety_review if c.member_id == 13)
    assert pending.assessment_id == pending_id
    assert pending.safety_flags == ("Suicidal_Thoughts",)
    assert pending.pending_scoring
    assert pending.band is None
    assert not pending.selectable
    with pytest.raises(ValueError, match="no longer eligible"):
        full.create_case(pending)

    complete(case_id, policy.db_path)
    submit_reassessment(13, {"Suicidal_Thoughts": 0}, policy.db_path)
    assert [c.member_id for c in full.worklists().safety_review] == [7]
    assert [c.member_id for c in full.worklists().standard_outreach] == [16]


@pytest.mark.parametrize("superseded", [False, True])
def test_repeat_outreach_requires_new_finalized_operational_assessment(
    policy, superseded
):
    selection = policy.worklists().standard_outreach[0]
    case_id = policy.create_case(selection)
    complete(case_id, policy.db_path)
    assert [c.member_id for c in policy.worklists().standard_outreach] == [16, 21]
    with pytest.raises(ValueError, match="no longer eligible"):
        policy.create_case(selection)

    reassessment_id = submit_reassessment(13, {"Feeling_Sad_Down": 9}, policy.db_path)
    result = {
        "risk_score": 96,
        "band_level_priority": "Very High",
        "rank_priority": 2,
        "explanation": "Updated score evidence",
        "operational_model_name": "logistic_regression",
        "model_version": "v1",
        "rank_context": {"cohort_id": "cohort-2"},
        "recommendation_action": "Behavioral-health resource review",
        "recommendation_reason": "Reported feeling sad or down was 9.",
        "recommendation_rule_version": "outreach-rules-v1",
    }
    assert [c.member_id for c in policy.worklists().standard_outreach] == [16, 21]
    finalize_scoring(reassessment_id, result, policy.db_path)
    assert [c.member_id for c in policy.worklists().standard_outreach] == [16, 21]
    with closing(sqlite3.connect(policy.db_path)) as conn, conn:
        conn.execute("UPDATE operational_scores SET cohort_id = 'cohort-2'")
        conn.execute(
            """UPDATE operational_scores SET score_assessment_id = ?, safety_assessment_id = ?
               WHERE Member_ID = 13""",
            (reassessment_id, reassessment_id),
        )

    current = policy.worklists().standard_outreach[0]
    assert current.member_id == 13
    assert current.assessment_id == reassessment_id
    if superseded:
        submit_reassessment(13, {}, policy.db_path)
        with pytest.raises(ValueError, match="no longer eligible"):
            policy.create_case(current)
        assert len(list_member_history(13, policy.db_path)["cases"]) == 1
    else:
        repeat = get_case(policy.create_case(current), policy.db_path)
        assert repeat["assessment_id"] == reassessment_id
        assert repeat["recommended_action"] == result["recommendation_action"]
        assert repeat["recommendation_reason"] == result["recommendation_reason"]


@pytest.mark.parametrize("same_member", [False, True])
def test_concurrent_selections_cannot_exceed_capacity_or_duplicate(
    policy, monkeypatch, same_member
):
    first, second = policy.worklists().standard_outreach
    if same_member:
        second = first
    ready = Barrier(3, timeout=10)
    release = Barrier(3, timeout=10)
    original = outreach_policy.score_operational_member

    def preview(member_id, inputs, **kwargs):
        if member_id != 7:
            ready.wait()
            release.wait()
        return original(member_id, inputs, **kwargs)

    monkeypatch.setattr(outreach_policy, "score_operational_member", preview)

    def create(selection):
        try:
            return policy.create_case(selection)
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(create, selection) for selection in (first, second)]
        ready.wait()
        if not same_member:
            # A safety case takes one slot after both standard requests have
            # read the worklist and before either acquires the write lock.
            policy.create_case(policy.worklists().safety_review[0])
        release.wait()
        results = [future.result() for future in pending]
    assert sum(isinstance(result, int) for result in results) == 1
    allowed_errors = {"selection is no longer eligible; refresh the outreach worklists"}
    if not same_member:
        allowed_errors.add("standard outreach is at capacity")
    assert (
        next(result for result in results if isinstance(result, str)) in allowed_errors
    )
    assert policy.worklists().active_cases == (1 if same_member else 2)
    with closing(sqlite3.connect(policy.db_path)) as conn:
        baselines = conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0]
    assert baselines == (1 if same_member else 2)


def test_cohort_refresh_during_recommendation_rejects_stale_selection(
    policy, monkeypatch
):
    selection = policy.worklists().standard_outreach[0]
    original = outreach_policy.score_operational_member

    def preview(member_id, inputs, **kwargs):
        result = original(member_id, inputs, **kwargs)
        with closing(sqlite3.connect(policy.db_path)) as conn, conn:
            conn.execute("UPDATE operational_scores SET cohort_id = 'cohort-2'")
        return result

    monkeypatch.setattr(outreach_policy, "score_operational_member", preview)
    with pytest.raises(ValueError, match="no longer eligible"):
        policy.create_case(selection)
    assert list_member_history(13, policy.db_path) == {"cases": [], "assessments": []}


def test_safety_visible_without_operational_scores(policy, monkeypatch):
    monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "random_forest")
    worklists = policy.worklists()
    assert worklists.standard_outreach == ()
    (concern,) = worklists.safety_review
    assert concern.member_id == 7
    assert concern.cohort_id is None
    assert not concern.selectable


@pytest.mark.parametrize("queue", ["safety_review", "standard_outreach"])
def test_training_member_cannot_be_selected(policy, queue):
    selection = getattr(policy.worklists(), queue)[0]
    with pytest.raises(ValueError, match="no longer eligible"):
        policy.create_case(replace(selection, member_id=1))
    assert list_member_history(1, policy.db_path) == {"cases": [], "assessments": []}


def test_wrong_member_ids_invalidate_same_size_worklist(policy):
    old_selection = policy.worklists().standard_outreach[0]
    with closing(sqlite3.connect(policy.db_path)) as conn, conn:
        conn.execute("UPDATE operational_scores SET Member_ID = 1 WHERE Member_ID = 24")

    worklists = policy.worklists()
    assert worklists.requires_score_refresh
    assert worklists.standard_outreach == ()
    (concern,) = worklists.safety_review
    assert concern == OutreachCandidate(
        member_id=7,
        assessment_id=None,
        cohort_id=None,
        band=None,
        rank=None,
        safety_flags=("Self_Harm_Thoughts",),
        pending_scoring=False,
        selectable=False,
    )
    with pytest.raises(ValueError, match="refresh scoring"):
        policy.create_case(old_selection)


@pytest.mark.parametrize("capacity", [-1, True, 1.5])
def test_capacity_must_be_a_nonnegative_integer(capacity):
    with pytest.raises(ValueError, match="nonnegative integer"):
        OutreachPolicy(capacity)
