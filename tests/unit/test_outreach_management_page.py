"""Exercise the Outreach Management page against SQLite with Streamlit AppTest."""

import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mh_risk_outreach import config, operational, outreach_policy
from mh_risk_outreach.db.assessments import finalize_scoring, submit_reassessment
from mh_risk_outreach.db.outreach import (
    accept_case,
    complete_case,
    list_active_cases,
    list_member_history,
    record_case_outcome,
)
from mh_risk_outreach.db.repository import save_records
from mh_risk_outreach.recommendations import recommend_action

UI = Path(__file__).resolve().parents[2] / "src" / "mh_risk_outreach" / "ui"
PAGE = UI / "pages" / "3_Outreach_management.py"


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> str:
    path = str(tmp_path / "outreach.sqlite")
    monkeypatch.setattr(config, "DEFAULT_DB_PATH", path)
    monkeypatch.setenv(config.OUTREACH_CAPACITY_ENV, "2")
    monkeypatch.delenv(config.OPERATIONAL_MODEL_ENV, raising=False)
    monkeypatch.setattr(outreach_policy, "artifact_model_version", lambda *args: "v1")
    return path


@pytest.fixture
def scored_db(db_path, monkeypatch) -> str:
    records = pd.read_csv(config.DEFAULT_CSV_PATH, nrows=25)
    records.insert(0, config.MEMBER_ID, range(1, 26))
    records[config.TARGET] = records.index % 2
    records["Self_Harm_Thoughts"] = records[config.MEMBER_ID].isin([1, 7]).astype(int)
    records["Suicidal_Thoughts"] = 0
    save_records(records, db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executemany(
            """INSERT INTO operational_scores
               VALUES (?, 'logistic_regression', 'v1', ?, ?, ?, 'Model evidence',
                       ?, NULL, NULL, 0, 'cohort', '2026-11-16T11:00:00+00:00')""",
            [
                (7, 25, "Low", 1, 1),
                (13, 98, "Very High", 2, 0),
                (16, 86, "High", 3, 0),
                (21, 83, "High", 4, 0),
                (24, 54, "Medium", 5, 0),
            ],
        )

    monkeypatch.setattr(operational, "artifact_model_version", lambda *args: "v1")
    monkeypatch.setattr(
        operational,
        "load_scoring_artifact",
        lambda name, *args, version: SimpleNamespace(
            model_name=name, model_version=version
        ),
    )

    def compute(members, background, artifact):
        return {
            row[config.MEMBER_ID]: recommend_action(row)
            for row in members.to_dict("records")
        }

    monkeypatch.setattr(operational, "_compute_baseline_recommendations", compute)
    return db_path


def _page() -> AppTest:
    return AppTest.from_file(PAGE)


def _button(container, label):
    return next(button for button in container.button if button.label == label)


def _metrics(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


def test_page_is_in_navigation_and_shows_empty_states(db_path):
    at = AppTest.from_file(UI / "app.py").run()
    at.switch_page("pages/3_Outreach_management.py").run()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == [
        "Safety Review",
        "Standard Outreach",
        "Active Cases",
    ]
    assert "No member records found" in at.warning[0].value
    assert [item.value for item in at.info] == [
        "There are no unreviewed safety concerns.",
        "There are no eligible standard outreach candidates.",
        "There are no active outreach cases.",
    ]
    assert not at.multiselect
    assert not Path(db_path).exists()


def test_create_selected_cases_and_refresh_through_lifecycle(scored_db, monkeypatch):
    monkeypatch.setenv(config.OUTREACH_CAPACITY_ENV, "3")
    at = _page().run()
    safety, standard, _ = at.tabs
    assert not at.exception
    assert _metrics(at) == {
        "Shared capacity (K)": "3",
        "Active cases": "0",
        "Remaining standard slots": "3",
    }
    safety_row = safety.dataframe[0].value.iloc[0]
    assert safety_row["Member ID"] == 7
    assert safety_row["Risk band"] == "Low"
    assert safety_row["Safety concerns"] == "Reported self-harm thoughts"
    offered = standard.dataframe[0].value.set_index("Member ID")
    assert offered.index.tolist() == [13, 16, 21]
    assert offered.loc[13, "Risk score"] == 98
    assert offered.loc[13, "Contact rank"] == 2
    assert offered.loc[13, "Model explanation"] == "Model evidence"

    standard.multiselect[0].set_value([13, 16])
    _button(standard, "Create Outreach Cases").click().run()
    assert not at.exception
    assert at.success[0].value == "Created 2 Pending cases."
    cases = list_active_cases(scored_db)
    assert [(case["Member_ID"], case["status"]) for case in cases] == [
        (13, "Pending"),
        (16, "Pending"),
    ]
    assert all(
        case["recommended_action"]
        == offered.loc[case["Member_ID"], "Recommended action"]
        and case["recommendation_reason"]
        == offered.loc[case["Member_ID"], "Recommendation explanation"]
        for case in cases
    )
    assert list_member_history(21, scored_db) == {"cases": [], "assessments": []}
    assert _metrics(at)["Remaining standard slots"] == "1"
    assert at.tabs[1].multiselect[0].options == ["21"]

    first = cases[0]
    case_id = first["case_id"]
    accept_case(case_id, scored_db, expected_action=first["recommended_action"])
    _button(at, "Refresh outreach").click().run()
    active = at.tabs[2]
    assert active.dataframe[0].value["Status"].tolist() == ["Accepted", "Pending"]
    active.selectbox[0].select(case_id).run()
    _button(at.tabs[2], "Update Case").click().run()
    assert at.tabs[2].subheader[0].value == f"Case {case_id} · Member 13"

    record_case_outcome(case_id, "Reached — engaged", scored_db)
    complete_case(case_id, scored_db, expected_outcome="Reached — engaged")
    _button(at, "Refresh outreach").click().run()
    assert not at.exception
    assert _metrics(at)["Remaining standard slots"] == "2"
    assert at.tabs[2].dataframe[0].value["Member ID"].tolist() == [16]
    assert list_member_history(13, scored_db)["cases"][0]["status"] == "Completed"


def test_safety_can_be_selected_at_full_capacity_and_pending_flags_stay_visible(
    scored_db,
):
    policy = outreach_policy.OutreachPolicy(2, db_path=scored_db)
    policy.create_cases(policy.worklists().standard_outreach)
    submit_reassessment(13, {"Suicidal_Thoughts": 1}, scored_db)
    at = _page().run()
    safety, standard, _ = at.tabs
    assert _metrics(at)["Remaining standard slots"] == "0"
    assert standard.info[0].value == "Standard outreach is at capacity."
    assert not standard.multiselect
    assert safety.multiselect[0].options == ["7"]
    pending = safety.dataframe[0].value.set_index("Member ID").loc[13]
    assert pending["Safety concerns"] == "Reported suicidal thoughts"
    assert "active case" in pending["Case creation"]
    assert "awaiting scoring" in pending["Case creation"]
    assert pd.isna(pending["Risk score"])

    safety.multiselect[0].set_value([7])
    _button(safety, "Create Outreach Cases").click().run()
    assert not at.exception
    assert at.success[0].value == "Created one Pending case."
    assert _metrics(at)["Active cases"] == "3"
    assert _metrics(at)["Remaining standard slots"] == "0"
    assert at.tabs[0].dataframe[0].value["Member ID"].tolist() == [13]
    assert not at.tabs[0].multiselect


@pytest.mark.parametrize(
    ("conflict", "message"),
    [
        ("stale", "no longer eligible"),
        ("duplicate", "already has an active case"),
        ("capacity", "current standard outreach capacity"),
        ("transaction", "database conflict"),
    ],
)
def test_failed_submission_creates_no_partial_cases(
    scored_db, monkeypatch, conflict, message
):
    capacity = 3 if conflict == "duplicate" else 2
    monkeypatch.setenv(config.OUTREACH_CAPACITY_ENV, str(capacity))
    policy = outreach_policy.OutreachPolicy(capacity, db_path=scored_db)
    at = _page().run()
    standard = at.tabs[1]
    standard.multiselect[0].set_value([13, 16])
    if conflict == "stale":
        with closing(sqlite3.connect(scored_db)) as conn, conn:
            conn.execute("UPDATE operational_scores SET cohort_id = 'refreshed-cohort'")
    elif conflict == "duplicate":
        policy.create_case(policy.worklists().standard_outreach[0])
    elif conflict == "capacity":
        policy.create_case(policy.worklists().safety_review[0])
    else:
        with closing(sqlite3.connect(scored_db)) as conn, conn:
            conn.execute(
                """CREATE TRIGGER fail_second_case BEFORE INSERT ON outreach_cases
                   WHEN NEW.Member_ID = 16
                   BEGIN SELECT RAISE(ABORT, 'second case failed'); END"""
            )

    previous = list_active_cases(scored_db)
    _button(standard, "Create Outreach Cases").click().run()
    assert not at.exception
    assert "No cases were created" in at.error[0].value
    assert message in at.error[0].value
    assert list_active_cases(scored_db) == previous
    assert list_member_history(16, scored_db) == {"cases": [], "assessments": []}


def test_missing_current_scores_keep_safety_concerns_visible(scored_db):
    with closing(sqlite3.connect(scored_db)) as conn, conn:
        conn.execute("DELETE FROM operational_scores")
    at = _page().run()

    assert not at.exception
    assert "--operational" in at.warning[0].value
    assert at.tabs[0].dataframe[0].value["Member ID"].tolist() == [7]
    assert not at.multiselect


def test_failed_preview_leaves_finalized_recommendation_selectable(
    scored_db, monkeypatch
):
    policy = outreach_policy.OutreachPolicy(2, db_path=scored_db)
    first = policy.worklists().standard_outreach[0]
    case_id = policy.create_case(first)
    case = list_active_cases(scored_db)[0]
    accept_case(case_id, scored_db, expected_action=case["recommended_action"])
    record_case_outcome(case_id, "Reached — engaged", scored_db)
    complete_case(case_id, scored_db, expected_outcome="Reached — engaged")
    assessment_id = submit_reassessment(13, {"Feeling_Sad_Down": 9}, scored_db)
    finalize_scoring(
        assessment_id,
        {
            "risk_score": 98,
            "band_level_priority": "Very High",
            "rank_priority": 2,
            "explanation": "Updated evidence",
            "operational_model_name": "logistic_regression",
            "model_version": "v1",
            "rank_context": {"cohort_id": "cohort"},
            "recommendation_action": "Behavioral-health resource review",
            "recommendation_reason": "Reported feeling sad or down was 9.",
            "recommendation_rule_version": "outreach-rules-v1",
        },
        scored_db,
    )
    with closing(sqlite3.connect(scored_db)) as conn, conn:
        conn.execute(
            """UPDATE operational_scores SET score_assessment_id = ?, safety_assessment_id = ?
               WHERE Member_ID = 13""",
            (assessment_id, assessment_id),
        )

    def fail_preview(members, background, artifact):
        assert members[config.MEMBER_ID].tolist() == [16]
        raise RuntimeError("Preview failed for this member.")

    monkeypatch.setattr(operational, "_compute_baseline_recommendations", fail_preview)
    at = _page().run()
    standard = at.tabs[1]
    status = standard.dataframe[0].value.set_index("Member ID")["Case creation"]
    assert status.to_dict() == {13: "Eligible", 16: "Recommendation unavailable"}
    assert standard.multiselect[0].options == ["13"]
    standard.multiselect[0].set_value([13])
    _button(standard, "Create Outreach Cases").click().run()
    assert not at.exception
    assert at.success[0].value == "Created one Pending case."
    assert [case["assessment_id"] for case in list_active_cases(scored_db)] == [
        assessment_id
    ]


def test_legacy_previews_only_score_new_members_and_submission_reuses_them(
    scored_db, monkeypatch
):
    computed = []
    original = operational._compute_baseline_recommendations

    def track(members, background, artifact):
        computed.append(sorted(members[config.MEMBER_ID]))
        return original(members, background, artifact)

    monkeypatch.setattr(operational, "_compute_baseline_recommendations", track)
    at = _page().run()
    assert computed == [[13, 16]]

    with closing(sqlite3.connect(scored_db)) as conn, conn:
        conn.execute(
            "UPDATE operational_scores SET band_level_priority = 'Medium' WHERE Member_ID = 13"
        )
    _button(at, "Refresh outreach").click().run()
    standard = at.tabs[1]
    assert computed == [[13, 16], [21]]
    assert standard.multiselect[0].options == ["16", "21"]
    standard.multiselect[0].set_value([16, 21])
    _button(standard, "Create Outreach Cases").click().run()

    assert not at.exception
    assert at.success[0].value == "Created 2 Pending cases."
    assert computed == [[13, 16], [21]]


def test_dashboard_reads_capacity_and_active_cases_in_one_snapshot(
    scored_db, monkeypatch
):
    with closing(sqlite3.connect(scored_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    policy = outreach_policy.OutreachPolicy(2, db_path=scored_db)
    candidate = policy.worklists().standard_outreach[0]
    original = outreach_policy.OutreachPolicy._worklists
    wrote_case = False

    def interleave_write(self, conn):
        nonlocal wrote_case
        snapshot = original(self, conn)
        if not wrote_case:
            wrote_case = True
            policy.create_case(candidate)
        return snapshot

    monkeypatch.setattr(outreach_policy.OutreachPolicy, "_worklists", interleave_write)
    at = _page().run()

    assert not at.exception
    assert len(list_active_cases(scored_db)) == 1
    assert _metrics(at)["Active cases"] == "0"
    assert _metrics(at)["Remaining standard slots"] == "2"
    assert at.tabs[1].multiselect[0].options == ["13", "16"]
    assert at.tabs[2].info[0].value == "There are no active outreach cases."

    _button(at, "Refresh outreach").click().run()
    assert _metrics(at)["Active cases"] == "1"
    assert _metrics(at)["Remaining standard slots"] == "1"
    assert at.tabs[1].multiselect[0].options == ["16"]
    assert at.tabs[2].dataframe[0].value["Member ID"].tolist() == [13]
