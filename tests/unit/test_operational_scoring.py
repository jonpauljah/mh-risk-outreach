"""Exercise operational scoring with the SQLite assessment contract."""

import sqlite3
from contextlib import closing

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
    finalize_scoring,
    get_assessment,
    submit_reassessment,
)
from mh_risk_outreach.db.operational_scores import load_operational_scores
from mh_risk_outreach.db.outreach import create_case, get_case
from mh_risk_outreach.db.repository import save_records
from mh_risk_outreach.models.train import run as train
from mh_risk_outreach.operational import (
    OPERATIONAL_SCOPE,
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

    case_id = create_case(member, "Contact", baseline_score, db)
    baseline_id = get_case(case_id, db)["assessment_id"]
    baseline = get_assessment(baseline_id, db)
    assert baseline["rank_context"] == baseline_score["rank_context"]

    pending_id = submit_reassessment(
        member, {"Self_Harm_Thoughts": 1, "Sleep_Trouble": 1}, db
    )
    pending = get_assessment(pending_id, db)
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

    finalize_scoring(pending_id, candidate, db)
    finalized = get_assessment(pending_id, db)
    assert finalized["rank_context"] == candidate["rank_context"]
    refreshed = refresh_operational_worklist(**paths).set_index(MEMBER_ID).loc[member]
    assert not refreshed["inputs_pending_scoring"]
    assert refreshed["score_assessment_id"] == pending_id
    assert refreshed["rank_priority"] == candidate["rank_priority"]
    assert refreshed["risk_score"] == pytest.approx(candidate["risk_score"])

    before_comparison = load_operational_scores(db)
    score_comparison(**paths)
    pd.testing.assert_frame_equal(load_operational_scores(db), before_comparison)
    assert get_assessment(baseline_id, db) == baseline
    assert get_assessment(pending_id, db) == finalized


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
