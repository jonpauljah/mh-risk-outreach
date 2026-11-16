"""SQLite integration coverage for durable assessment and case history."""

import json
import sqlite3

import pandas as pd
import pytest

from mh_risk_outreach.config import FEATURES
from mh_risk_outreach.db.assessments import (
    finalize_scoring,
    get_assessment,
    get_current_assessment,
    submit_reassessment,
)
from mh_risk_outreach.db.outreach import (
    create_case,
    get_case,
    list_member_history,
    update_case_decision,
)
from mh_risk_outreach.db.repository import init_schema, save_records
from mh_risk_outreach.db.scores import save_member_scores


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "history.sqlite"
    conn = sqlite3.connect(path)
    init_schema(conn)
    inputs = {
        name: (
            1
            if name.endswith(
                (
                    "Thoughts",
                    "History",
                    "Diagnosed",
                    "Treatment",
                    "Therapy_Now",
                    "Medication",
                )
            )
            else "sample"
            if name
            in {
                "Gender",
                "Country",
                "Education",
                "Marital_Status",
                "Income_Level",
                "Employment_Status",
                "Remote_Work",
                "Company_Mental_Health_Support",
                "Exercise_Per_Week",
                "Alcohol_Frequency",
                "Smoking",
                "Diet_Quality",
                "Discuss_Mental_Health",
            }
            else 1
        )
        for name in FEATURES
    }
    columns = ",".join(
        f'"{name}"' for name in ["Member_ID", *FEATURES, "Has_Mental_Health_Issue"]
    )
    conn.execute(
        f"INSERT INTO patients ({columns}) VALUES ({','.join('?' for _ in range(len(FEATURES) + 2))})",
        [7, *inputs.values(), 0],
    )
    conn.commit()
    conn.close()
    return path, inputs


def score(**overrides):
    result = {
        "risk_score": 72.0,
        "band_level_priority": "High",
        "rank_priority": 1,
        "rank_context": {"population": 20},
        "explanation": "Supplied result",
        "recommendation": "Contact",
        "model_version": "model-v1",
    }
    result.update(overrides)
    return result


def test_on_demand_baseline_and_reassessments_preserve_case(database):
    path, inputs = database
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM outreach_cases").fetchone()[0] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM member_assessment_state").fetchone()[0] == 0
    )
    conn.close()
    case_id = create_case(
        7,
        "Contact",
        score(
            operational_model_name="random_forest",
            recommendation_reason="Stored outreach policy",
            recommendation_rule_version="rules-v2",
        ),
        path,
    )
    case_before = get_case(case_id, path)
    baseline_id = case_before["assessment_id"]
    assert case_before["source"] == "baseline"
    assert case_before["recommendation_reason"] == "Stored outreach policy"
    assert case_before["recommendation_rule_version"] == "rules-v2"
    assert case_before["operational_model_name"] == "random_forest"
    assert len(get_assessment(baseline_id, path)["inputs"]) == 50

    pending_id = submit_reassessment(7, {"Age": 42}, path)
    assert get_assessment(pending_id, path)["scoring_status"] == "PendingScoring"
    assert get_assessment(pending_id, path)["inputs"]["Gender"] == inputs["Gender"]
    current = get_current_assessment(7, path)
    assert current["assessment_id"] == pending_id
    assert current["assessment"]["scoring_status"] == "PendingScoring"
    assert current["updated_at"]
    assert get_case(case_id, path)["assessment_id"] == baseline_id
    assert get_case(case_id, path)["status"] == "Pending"
    with pytest.raises(ValueError, match="active case"):
        create_case(7, "Again", score(), path)

    finalize_scoring(pending_id, score(), path)
    with pytest.raises(ValueError, match="already finalized"):
        finalize_scoring(pending_id, score(), path)
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE outreach_cases SET status='Accepted', accepted_at='2026-11-16T11:00:00+00:00' WHERE case_id=?",
        (case_id,),
    )
    conn.commit()
    conn.close()
    accepted_reassessment = submit_reassessment(7, {}, path)
    assert get_case(case_id, path)["status"] == "Accepted"
    assert get_case(case_id, path)["assessment_id"] == baseline_id
    finalize_scoring(accepted_reassessment, score(), path)

    update_case_decision(
        case_id,
        {
            "status": "Completed",
            "outcome": "Reached — engaged",
            "outcome_recorded_at": "2026-11-16T11:30:00+00:00",
            "completed_at": "2026-11-16T12:00:00+00:00",
        },
        path,
    )
    conn = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError, match="scored assessment is immutable"):
        conn.execute(
            "UPDATE risk_assessments SET risk_score=0.1 WHERE assessment_id=?",
            (pending_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="case origin"):
        conn.execute(
            "UPDATE outreach_cases SET assessment_id=? WHERE case_id=?",
            (pending_id, case_id),
        )
    conn.rollback()
    conn.close()
    second_id = submit_reassessment(
        7, {}, path
    )  # unchanged submission is still distinct
    with pytest.raises(ValueError, match="pending scoring"):
        create_case(7, "Again", score(), path)
    finalize_scoring(second_id, score(), path)
    assert second_id != pending_id
    assert get_case(case_id, path)["status"] == "Completed"
    with pytest.raises(ValueError, match="conflicts"):
        create_case(7, "Wrong action", db_path=path)
    third_case = create_case(7, "Contact", db_path=path)
    assert get_case(third_case, path)["assessment_id"] == second_id
    history = list_member_history(7, path)
    assert len(history["assessments"]) == 4
    assert len(history["cases"]) == 2


def test_reassessment_requires_first_case_history(database):
    path, _ = database
    with pytest.raises(ValueError, match="first case creates the baseline"):
        submit_reassessment(7, {}, path)
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM member_assessment_state").fetchone()[0] == 0
    )
    conn.close()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("Age", "42"),
        ("Sleep_Hours_Night", "eight"),
        ("Gender", None),
        ("Self_Harm_Thoughts", 2),
    ],
)
def test_invalid_partial_reassessment_preserves_history_and_current(
    database, field, invalid_value
):
    path, _ = database
    create_case(7, "Contact", score(), path)
    before = get_current_assessment(7, path)
    with pytest.raises((TypeError, ValueError), match=field):
        submit_reassessment(7, {field: invalid_value}, path)
    after = get_current_assessment(7, path)
    assert after["assessment_id"] == before["assessment_id"]
    assert after["updated_at"] == before["updated_at"]
    assert len(list_member_history(7, path)["assessments"]) == 1


def test_reimport_and_score_replacement_preserve_history(database):
    path, inputs = database
    case_id = create_case(7, "Contact", score(), path)
    pending = submit_reassessment(7, {"Age": 47}, path)
    finalize_scoring(pending, score(), path)
    records = pd.DataFrame([{"Member_ID": 7, **inputs, "Has_Mental_Health_Issue": 0}])
    assert save_records(records, db_path=str(path)) is False

    columns = {
        "Member_ID": [7],
        "model_version": ["v1"],
        "risk_score": [25.0],
        "band_level_priority": ["Low"],
        "rank_priority": [1],
        "reasoning": ["old worklist"],
    }
    save_member_scores(pd.DataFrame(columns), "random_forest", str(path))
    columns.update(
        model_version=["v2"], risk_score=[75.0], reasoning=["replacement worklist"]
    )
    save_member_scores(pd.DataFrame(columns), "random_forest", str(path))

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0] == 2
    assert conn.execute(
        "SELECT model_version, risk_score, reasoning FROM random_forest_scores"
    ).fetchone() == ("v2", 75.0, "replacement worklist")
    conn.close()
    current = get_current_assessment(7, path)
    assert current["assessment_id"] == pending
    assert current["assessment"]["inputs"]["Age"] == 47
    assert get_case(case_id, path)["assessment_id"] != pending
    assert len(list_member_history(7, path)["cases"]) == 1


def test_completion_without_reassessment_and_constraints(database):
    path, _ = database
    case_id = create_case(7, "Contact", score(), path)
    update_case_decision(
        case_id,
        {"status": "Accepted", "accepted_at": "2026-11-16T10:00:00+00:00"},
        path,
    )
    update_case_decision(
        case_id,
        {
            "status": "Completed",
            "outcome": "Not reached",
            "outcome_recorded_at": "2026-11-16T11:00:00+00:00",
            "completed_at": "2026-11-16T12:00:00+00:00",
        },
        path,
    )
    assert get_case(case_id, path)["status"] == "Completed"
    assert len(list_member_history(7, path)["assessments"]) == 1

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE outreach_cases SET status='bogus' WHERE case_id=?", (case_id,)
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE outreach_cases SET outcome='bogus' WHERE case_id=?", (case_id,)
        )
    conn.execute(
        "INSERT INTO patients SELECT 8, "
        + ",".join('"' + field + '"' for field in FEATURES)
        + ", 0 FROM patients WHERE Member_ID=7"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO member_assessment_state(Member_ID, assessment_id, updated_at) VALUES (8, 1, 'now')"
        )
    conn.rollback()
    conn.close()

    with pytest.raises(ValueError, match="already been used"):
        create_case(7, "Contact", score(), path)


def test_case_requires_scored_snapshot_recommendation(database):
    path, _ = database
    case_id = create_case(7, "Contact", score(), path)
    update_case_decision(
        case_id,
        {"status": "Accepted", "accepted_at": "2026-11-16T10:00:00+00:00"},
        path,
    )
    update_case_decision(
        case_id,
        {
            "status": "Completed",
            "outcome": "Not reached",
            "outcome_recorded_at": "2026-11-16T11:00:00+00:00",
            "completed_at": "2026-11-16T12:00:00+00:00",
        },
        path,
    )
    pending = submit_reassessment(7, {}, path)
    score_without_recommendation = score()
    score_without_recommendation.pop("recommendation")
    finalize_scoring(pending, score_without_recommendation, path)
    with pytest.raises(ValueError, match="no stored recommendation"):
        create_case(7, "Caller supplied action", db_path=path)


def test_case_reads_nullable_rank_context_for_pending_assessment(database):
    path, inputs = database
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    # The schema permits this stored relationship even though create_case()
    # ordinarily requires a scored assessment. Read APIs must handle it.
    assessment_id = conn.execute(
        """INSERT INTO risk_assessments
           (Member_ID, source, created_at, inputs_json, scoring_status)
           VALUES (?, 'reassessment', ?, ?, 'PendingScoring')""",
        (7, "2026-11-16T12:00:00+00:00", json.dumps(inputs)),
    ).lastrowid
    case_id = conn.execute(
        """INSERT INTO outreach_cases
           (Member_ID, assessment_id, status, recommended_action, created_at)
           VALUES (?, ?, 'Pending', 'Contact', ?)""",
        (7, assessment_id, "2026-11-16T12:01:00+00:00"),
    ).lastrowid
    conn.commit()
    conn.close()

    assert get_case(case_id, path)["rank_context"] is None
    history = list_member_history(7, path)
    assert history["cases"][0]["rank_context"] is None
    assert history["assessments"][0]["scoring_status"] == "PendingScoring"
    assert history["assessments"][0]["rank_context"] is None


@pytest.mark.parametrize("bad_context", ["missing", None, [], "cohort"])
def test_first_case_rejects_missing_null_or_non_dict_rank_context(
    database, bad_context
):
    path, _ = database
    scoring = score()
    if bad_context == "missing":
        scoring.pop("rank_context")
    else:
        scoring["rank_context"] = bad_context

    with pytest.raises(TypeError, match="rank_context"):
        create_case(7, "Contact", scoring, path)

    conn = sqlite3.connect(path)
    for table in ("risk_assessments", "member_assessment_state", "outreach_cases"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize("bad_context", ["missing", None, [], "cohort"])
def test_finalization_rejects_missing_null_or_non_dict_rank_context(
    database, bad_context
):
    path, _ = database
    case_id = create_case(7, "Contact", score(), path)
    pending_id = submit_reassessment(7, {"Age": 42}, path)
    scoring = score()
    if bad_context == "missing":
        scoring.pop("rank_context")
    else:
        scoring["rank_context"] = bad_context

    with pytest.raises(TypeError, match="rank_context"):
        finalize_scoring(pending_id, scoring, path)

    current = get_current_assessment(7, path)
    assert current["assessment_id"] == pending_id
    assert current["assessment"]["scoring_status"] == "PendingScoring"
    assert current["assessment"]["risk_score"] is None
    assert current["assessment"]["rank_context"] is None
    assert get_case(case_id, path)["status"] == "Pending"
    assert len(list_member_history(7, path)["assessments"]) == 2


def test_failed_case_insert_rolls_back_on_durable_file(database):
    path, _ = database
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TRIGGER fail_case BEFORE INSERT ON outreach_cases BEGIN SELECT RAISE(ABORT, 'forced'); END"
    )
    conn.commit()
    conn.close()
    with pytest.raises(sqlite3.IntegrityError, match="forced"):
        create_case(7, "Contact", score(), path)
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM member_assessment_state").fetchone()[0] == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM outreach_cases").fetchone()[0] == 0
    conn.close()


def test_failed_pointer_update_rolls_back_reassessment(database):
    path, _ = database
    create_case(7, "Contact", score(), path)
    conn = sqlite3.connect(path)
    old_id = conn.execute(
        "SELECT assessment_id FROM member_assessment_state WHERE Member_ID=7"
    ).fetchone()[0]
    conn.execute(
        "CREATE TRIGGER fail_pointer BEFORE UPDATE ON member_assessment_state BEGIN SELECT RAISE(ABORT, 'forced'); END"
    )
    conn.commit()
    conn.close()
    with pytest.raises(sqlite3.IntegrityError, match="forced"):
        submit_reassessment(7, {"Age": 40}, path)
    conn = sqlite3.connect(path)
    assert (
        conn.execute(
            "SELECT assessment_id FROM member_assessment_state WHERE Member_ID=7"
        ).fetchone()[0]
        == old_id
    )
    assert conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0] == 1
    conn.close()


def test_failed_scoring_finalization_rolls_back(database):
    path, _ = database
    create_case(7, "Contact", score(), path)
    pending = submit_reassessment(7, {}, path)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TRIGGER fail_score BEFORE UPDATE OF scoring_status ON risk_assessments BEGIN SELECT RAISE(ABORT, 'forced'); END"
    )
    conn.commit()
    conn.close()
    with pytest.raises(sqlite3.IntegrityError, match="forced"):
        finalize_scoring(pending, score(), path)
    assert get_assessment(pending, path)["scoring_status"] == "PendingScoring"


def test_invalid_inputs_and_foreign_keys_are_rejected(database):
    path, _ = database
    with pytest.raises(ValueError, match="unknown survey fields"):
        submit_reassessment(7, {"nonsense": 1}, path)
    with pytest.raises(ValueError, match="unknown member"):
        submit_reassessment(404, {}, path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO member_assessment_state VALUES (7, 999, 'now')")
    conn.close()
