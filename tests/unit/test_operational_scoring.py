"""Exercise operational scoring with the SQLite assessment contract."""

import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mh_risk_outreach import operational
from mh_risk_outreach.config import (
    DEFAULT_CSV_PATH,
    FEATURES,
    MEMBER_ID,
    OPERATIONAL_MODEL_ENV,
    TARGET,
)
from mh_risk_outreach.db.assessments import (
    get_assessment,
    get_current_assessment,
    submit_reassessment,
)
from mh_risk_outreach.db.operational_scores import load_operational_scores
from mh_risk_outreach.db.outreach import (
    create_case,
    get_case,
    list_member_history,
    update_case_decision,
)
from mh_risk_outreach.db.repository import save_records
from mh_risk_outreach.models.train import run as train
from mh_risk_outreach.operational import (
    OPERATIONAL_SCOPE,
    finalize_operational_reassessment,
    operational_model_name,
    refresh_operational_worklist,
    score_operational_member,
)
from mh_risk_outreach.scoring import run as score_comparison


@pytest.fixture
def setup_scoring(tmp_path, monkeypatch):
    monkeypatch.delenv(OPERATIONAL_MODEL_ENV, raising=False)
    records = pd.read_csv(DEFAULT_CSV_PATH)
    records.insert(0, MEMBER_ID, range(1, len(records) + 1))
    records = (
        records.groupby(TARGET, group_keys=False)
        .sample(frac=0.04, random_state=42)
        .sort_values(MEMBER_ID)
    )
    paths = {
        "db_path": str(tmp_path / "patients.db"),
        "artifacts_dir": str(tmp_path / "artifacts"),
    }
    save_records(records, paths["db_path"])
    train(**paths)
    return paths, records


def test_configuration(monkeypatch):
    monkeypatch.delenv(OPERATIONAL_MODEL_ENV, raising=False)
    assert operational_model_name() == "logistic_regression"
    monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "random_forest")
    assert operational_model_name() == "random_forest"
    monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "all")
    with pytest.raises(ValueError, match=OPERATIONAL_MODEL_ENV):
        operational_model_name()


def test_candidate_is_read_only_and_snapshots_survive_refresh(
    setup_scoring, monkeypatch
):
    paths, records = setup_scoring
    db = paths["db_path"]
    published = refresh_operational_worklist(**paths)
    assert len(published) == len(records)
    assert published["operational_model_name"].eq("logistic_regression").all()
    assert published["rank_priority"].tolist() == list(range(1, len(records) + 1))
    with closing(sqlite3.connect(db)) as conn:
        for table in ("risk_assessments", "outreach_cases", "member_assessment_state"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    train("xgboost", **paths)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("scoring must use stored cut points")

    monkeypatch.setattr(KMeans, "fit", unexpected_fit)
    original = load_operational_scores(db)
    baseline_row = published.loc[~published["safety_review_required"]].iloc[-1]
    member = int(baseline_row[MEMBER_ID])
    inputs = records.loc[records[MEMBER_ID].eq(member), FEATURES].iloc[0].to_dict()
    baseline_score = score_operational_member(member, inputs, **paths)
    assert baseline_score["risk_score"] == pytest.approx(baseline_row["risk_score"])
    assert baseline_score["band_level_priority"] == baseline_row["band_level_priority"]
    assert baseline_score["rank_priority"] == baseline_row["rank_priority"]
    assert baseline_score["rank_context"]["scope"] == OPERATIONAL_SCOPE
    assert baseline_score["rank_context"]["cohort_size"] == len(records)
    assert baseline_score["rank_context"]["cohort_id"] == baseline_row["cohort_id"]
    pd.testing.assert_frame_equal(load_operational_scores(db), original)

    case_id = create_case(
        member,
        "Contact",
        db_path=db,
        recommendation_reason="Baseline reason",
        recommendation_rule_version="rules-1",
    )
    baseline_id = get_case(case_id, db)["assessment_id"]
    baseline = get_assessment(baseline_id, db)
    assert baseline["source"] == "baseline"
    assert baseline["inputs"] == inputs
    assert get_current_assessment(member, db)["assessment"] == baseline
    assert_snapshot_matches_worklist(baseline, baseline_row, len(records))
    assert baseline["rank_context"]["calculated_at"] == baseline_row["calculated_at"]
    assert (
        baseline["rank_context"]["ordering"]
        == baseline_score["rank_context"]["ordering"]
    )
    assert datetime.fromisoformat(baseline["scored_at"]) >= datetime.fromisoformat(
        baseline["rank_context"]["calculated_at"]
    )
    assert baseline["recommendation_action"] == "Contact"
    assert baseline["recommendation_reason"] == "Baseline reason"
    assert baseline["recommendation_rule_version"] == "rules-1"
    completed_at = get_case(case_id, db)["created_at"]
    update_case_decision(
        case_id,
        {
            "status": "Completed",
            "outcome": "Reached — engaged",
            "outcome_recorded_at": completed_at,
            "accepted_at": completed_at,
            "completed_at": completed_at,
        },
        db,
    )
    original_case = get_case(case_id, db)
    with pytest.raises(ValueError, match="already been used"):
        create_case(member, "Contact", db_path=db)
    assert original_case["rank_context"] == baseline["rank_context"]
    assert original_case["recommendation_reason"] == "Baseline reason"
    assert original_case["recommendation_rule_version"] == "rules-1"

    pending_id = submit_reassessment(
        member, {"Self_Harm_Thoughts": 1, "Sleep_Trouble": 1}, db
    )
    pending = get_assessment(pending_id, db)
    with pytest.raises(ValueError, match="pending scoring"):
        create_case(member, "Contact", db_path=db)
    candidate = score_operational_member(member, pending["inputs"], **paths)
    assert candidate["rank_context"]["safety_review_required"] is True
    assert candidate["rank_priority"] < baseline_score["rank_priority"]
    assert get_assessment(pending_id, db) == pending
    pd.testing.assert_frame_equal(load_operational_scores(db), original)

    pending_worklist = refresh_operational_worklist(**paths).set_index(MEMBER_ID)
    pending_row = pending_worklist.loc[member]
    assert pending_row["inputs_pending_scoring"]
    assert pending_row["score_assessment_id"] == baseline_id
    assert pending_row["safety_assessment_id"] == pending_id
    assert pending_row["safety_review_required"]
    assert pending_row["band_level_priority"] == baseline_score["band_level_priority"]
    assert pending_row["risk_score"] == pytest.approx(baseline_score["risk_score"])

    peer = int(published.loc[~published["safety_review_required"], MEMBER_ID].iat[0])
    peer_case_id = create_case(peer, "Contact", db_path=db)
    peer_baseline_id = get_case(peer_case_id, db)["assessment_id"]
    peer_pending_id = submit_reassessment(peer, {"Self_Harm_Thoughts": 1}, db)
    peer_pending = get_assessment(peer_pending_id, db)

    finalize_operational_reassessment(
        pending_id,
        recommendation_action="Contact",
        recommendation_reason="Reassessment reason",
        recommendation_rule_version="rules-2",
        **paths,
    )
    finalized = get_assessment(pending_id, db)
    refreshed_worklist = load_operational_scores(db).set_index(MEMBER_ID)
    refreshed = refreshed_worklist.loc[member]
    peer_row = refreshed_worklist.loc[peer]
    assert peer_row["inputs_pending_scoring"]
    assert peer_row["score_assessment_id"] == peer_baseline_id
    assert peer_row["safety_assessment_id"] == peer_pending_id
    assert peer_row["safety_review_required"]
    assert get_assessment(peer_pending_id, db) == peer_pending
    assert finalized["scoring_status"] == "Scored"
    assert finalized["created_at"] == pending["created_at"]
    assert finalized["inputs"] == pending["inputs"]
    assert get_current_assessment(member, db)["assessment"] == finalized
    assert not refreshed["inputs_pending_scoring"]
    assert refreshed["score_assessment_id"] == pending_id
    assert refreshed["safety_assessment_id"] == pending_id
    assert refreshed["cohort_id"] != pending_row["cohort_id"]
    assert finalized["rank_priority"] < baseline["rank_priority"]
    assert (
        refreshed_worklist["rank_priority"].drop(member).sort_index()
        != original.set_index(MEMBER_ID)["rank_priority"].drop(member).sort_index()
    ).any()
    assert_snapshot_matches_worklist(finalized, refreshed, len(records))
    assert finalized["rank_context"]["calculated_at"] == refreshed["calculated_at"]
    assert finalized["recommendation_action"] == "Contact"
    assert finalized["recommendation_reason"] == "Reassessment reason"
    assert finalized["recommendation_rule_version"] == "rules-2"
    reused_case_id = create_case(member, "Contact", db_path=db)
    reused_case = get_case(reused_case_id, db)
    assert reused_case["assessment_id"] == pending_id
    assert reused_case["recommendation_reason"] == "Reassessment reason"
    assert reused_case["recommendation_rule_version"] == "rules-2"
    expected_history = {
        "assessments": [baseline, finalized],
        "cases": [original_case, reused_case],
    }
    assert list_member_history(member, db) == expected_history
    published_after_finalization = load_operational_scores(db)
    with pytest.raises(ValueError, match="current pending reassessment"):
        finalize_operational_reassessment(pending_id, **paths)
    pd.testing.assert_frame_equal(
        load_operational_scores(db), published_after_finalization
    )

    refresh_operational_worklist(**paths)
    before_comparison = load_operational_scores(db)
    score_comparison(**paths)
    pd.testing.assert_frame_equal(load_operational_scores(db), before_comparison)
    monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "xgboost")
    switched = refresh_operational_worklist(**paths)
    assert switched["operational_model_name"].eq("xgboost").all()
    assert list_member_history(member, db) == expected_history
    with closing(sqlite3.connect(db)) as conn:
        imported = pd.read_sql_query("SELECT * FROM patients", conn)
    assert imported.set_index(MEMBER_ID).loc[member, FEATURES].to_dict() == inputs


def assert_snapshot_matches_worklist(snapshot, row, cohort_size):
    for field in (
        "risk_score",
        "band_level_priority",
        "rank_priority",
        "operational_model_name",
        "model_version",
    ):
        assert snapshot[field] == row[field]
    assert snapshot["explanation"] == row["reasoning"]
    context = snapshot["rank_context"]
    assert context["cohort_id"] == row["cohort_id"]
    assert context["cohort_calculated_at"] == row["calculated_at"]
    assert context["scope"] == OPERATIONAL_SCOPE
    assert context["cohort_size"] == cohort_size
    assert context["safety_review_required"] == bool(row["safety_review_required"])


def test_first_case_requires_published_member_and_rolls_back_on_insert_failure(
    setup_scoring,
):
    paths, records = setup_scoring
    db = paths["db_path"]
    member = int(records[MEMBER_ID].iat[0])
    with pytest.raises(ValueError, match="no published operational worklist score"):
        create_case(member, "Contact", db_path=db)
    assert get_current_assessment(member, db) is None
    assert list_member_history(member, db) == {"assessments": [], "cases": []}

    refresh_operational_worklist(**paths)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            """CREATE TRIGGER fail_case_insert BEFORE INSERT ON outreach_cases
               BEGIN SELECT RAISE(ABORT, 'injected failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        create_case(member, "Contact", db_path=db)
    assert get_current_assessment(member, db) is None
    assert list_member_history(member, db) == {"assessments": [], "cases": []}


def test_operational_finalization_rejects_superseded_and_rolls_back(
    setup_scoring, monkeypatch
):
    paths, records = setup_scoring
    db = paths["db_path"]
    member = int(records[MEMBER_ID].iat[0])
    refresh_operational_worklist(**paths)
    create_case(member, "Contact", db_path=db)
    superseded_id = submit_reassessment(member, {"Sleep_Trouble": 1}, db)
    superseded = get_assessment(superseded_id, db)
    pending_id = submit_reassessment(member, {}, db)
    pending_state = get_current_assessment(member, db)
    assert pending_id != superseded_id
    assert pending_state["assessment"]["inputs"] == superseded["inputs"]
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("DELETE FROM operational_scores")
    with pytest.raises(ValueError, match="no operational worklist"):
        finalize_operational_reassessment(pending_id, **paths)
    assert get_current_assessment(member, db) == pending_state

    refresh_operational_worklist(**paths)
    original = load_operational_scores(db)
    with pytest.raises(ValueError, match="current pending reassessment"):
        finalize_operational_reassessment(superseded_id, **paths)
    monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "random_forest")
    with pytest.raises(ValueError, match="refresh scoring"):
        finalize_operational_reassessment(pending_id, **paths)
    monkeypatch.delenv(OPERATIONAL_MODEL_ENV)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("UPDATE operational_scores SET model_version='outdated'")
    outdated = load_operational_scores(db)
    with pytest.raises(ValueError, match="refresh scoring"):
        finalize_operational_reassessment(pending_id, **paths)
    pd.testing.assert_frame_equal(load_operational_scores(db), outdated)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "UPDATE operational_scores SET model_version=?",
            (original["model_version"].iat[0],),
        )
        conn.execute(
            """CREATE TRIGGER fail_finalized_worklist_insert
               BEFORE INSERT ON operational_scores WHEN NEW.rank_priority = 2
               BEGIN SELECT RAISE(ABORT, 'injected failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        finalize_operational_reassessment(pending_id, **paths)
    assert get_current_assessment(member, db) == pending_state
    assert get_assessment(superseded_id, db) == superseded
    pd.testing.assert_frame_equal(load_operational_scores(db), original)

    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("DROP TRIGGER fail_finalized_worklist_insert")
    finalize_operational_reassessment(pending_id, **paths)
    finalized = get_assessment(pending_id, db)
    assert finalized["recommendation_action"] is None
    assert finalized["recommendation_reason"] is None
    assert finalized["recommendation_rule_version"] is None


@pytest.mark.parametrize(
    "change", ["assessment inputs", "operational worklist", "operational model"]
)
def test_operational_finalization_rejects_concurrent_changes(
    setup_scoring, monkeypatch, change
):
    paths, records = setup_scoring
    db = paths["db_path"]
    member = int(records[MEMBER_ID].iat[0])
    refresh_operational_worklist(**paths)
    create_case(member, "Contact", db_path=db)
    pending_id = submit_reassessment(member, {"Sleep_Trouble": 1}, db)
    pending = get_assessment(pending_id, db)
    score_members = operational.score_members
    latest_worklist = None
    latest_state = None

    def score_with_concurrent_change(*args, **kwargs):
        nonlocal latest_worklist, latest_state
        scored = score_members(*args, **kwargs)
        monkeypatch.setattr(operational, "score_members", score_members)
        if change == "assessment inputs":
            submit_reassessment(member, {"Self_Harm_Thoughts": 1}, db)
        elif change == "operational worklist":
            refresh_operational_worklist(**paths)
        else:
            monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "xgboost")
        latest_worklist = load_operational_scores(db)
        latest_state = get_current_assessment(member, db)
        return scored

    monkeypatch.setattr(operational, "score_members", score_with_concurrent_change)
    with pytest.raises(ValueError, match=f"{change} changed during scoring"):
        finalize_operational_reassessment(pending_id, **paths)
    assert get_assessment(pending_id, db) == pending
    assert get_current_assessment(member, db) == latest_state
    pd.testing.assert_frame_equal(load_operational_scores(db), latest_worklist)


def test_finalization_checks_artifact_version_under_publication_lock(
    setup_scoring, monkeypatch
):
    paths, records = setup_scoring
    db = paths["db_path"]
    member = int(records[MEMBER_ID].iat[0])
    refresh_operational_worklist(**paths)
    original = load_operational_scores(db)
    create_case(member, "Contact", db_path=db)
    pending_id = submit_reassessment(member, {"Sleep_Trouble": 1}, db)
    pending_state = get_current_assessment(member, db)
    artifact_model_version = operational.artifact_model_version

    def version_with_concurrent_retraining(*args):
        with closing(sqlite3.connect(db, timeout=0)) as probe:
            try:
                probe.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                assert exc.sqlite_errorcode == sqlite3.SQLITE_BUSY
                return "new"
        return artifact_model_version(*args)

    monkeypatch.setattr(
        operational, "artifact_model_version", version_with_concurrent_retraining
    )
    with pytest.raises(ValueError, match="operational model changed during scoring"):
        finalize_operational_reassessment(pending_id, **paths)
    assert get_current_assessment(member, db) == pending_state
    pd.testing.assert_frame_equal(load_operational_scores(db), original)


def test_model_switch_requires_refresh_and_failed_refresh_preserves_worklist(
    setup_scoring, monkeypatch
):
    paths, records = setup_scoring
    db = paths["db_path"]
    refresh_operational_worklist(**paths)
    original = load_operational_scores(db)
    row = records.iloc[0]
    member = int(row[MEMBER_ID])
    inputs = row[FEATURES].to_dict()

    train("xgboost", **paths)
    monkeypatch.setenv(OPERATIONAL_MODEL_ENV, "xgboost")
    with pytest.raises(ValueError, match="refresh scoring"):
        score_operational_member(member, inputs, **paths)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            """CREATE TRIGGER fail_worklist_insert BEFORE INSERT ON operational_scores
               WHEN NEW.rank_priority = 2
               BEGIN SELECT RAISE(ABORT, 'injected failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        refresh_operational_worklist(**paths)
    pd.testing.assert_frame_equal(load_operational_scores(db), original)

    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("DROP TRIGGER fail_worklist_insert")
    updated = refresh_operational_worklist(**paths)
    assert updated["operational_model_name"].eq("xgboost").all()
    candidate = score_operational_member(member, inputs, **paths)
    assert candidate["model_version"] == updated["model_version"].iat[0]
    assert candidate["operational_model_name"] == "xgboost"


def test_assessment_change_during_refresh_preserves_worklist(
    setup_scoring, monkeypatch
):
    paths, records = setup_scoring
    db = paths["db_path"]
    refresh_operational_worklist(**paths)
    original = load_operational_scores(db)
    row = records.iloc[0]
    member = int(row[MEMBER_ID])
    result = score_operational_member(member, row[FEATURES].to_dict(), **paths)
    create_case(member, "Contact", result, db)
    score_members = operational.score_members

    def score_with_concurrent_change(*args, **kwargs):
        worklist = score_members(*args, **kwargs)
        submit_reassessment(member, {"Self_Harm_Thoughts": 1}, db)
        return worklist

    monkeypatch.setattr(operational, "score_members", score_with_concurrent_change)
    with pytest.raises(ValueError, match="assessment inputs changed during scoring"):
        refresh_operational_worklist(**paths)
    pd.testing.assert_frame_equal(load_operational_scores(db), original)


@pytest.mark.parametrize("has_existing_worklist", [False, True])
def test_overlapping_refresh_preserves_newer_worklist(
    setup_scoring, monkeypatch, has_existing_worklist
):
    paths, _ = setup_scoring
    db = paths["db_path"]
    if has_existing_worklist:
        refresh_operational_worklist(**paths)
    save_scores = operational.save_operational_scores
    newer = None

    def save_after_newer_refresh(*args, **kwargs):
        nonlocal newer
        monkeypatch.setattr(operational, "save_operational_scores", save_scores)
        refresh_operational_worklist(**paths)
        newer = load_operational_scores(db)
        save_scores(*args, **kwargs)

    monkeypatch.setattr(
        operational, "save_operational_scores", save_after_newer_refresh
    )
    with pytest.raises(ValueError, match="operational worklist changed during scoring"):
        refresh_operational_worklist(**paths)
    pd.testing.assert_frame_equal(load_operational_scores(db), newer)
