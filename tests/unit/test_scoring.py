"""Scoring, SHAP reasons, risk bands, and priority order (issue #16).

These run against a stratified subsample of the survey (as the DB stores it, with
a ``Member_ID``) and a freshly trained pipeline, so they need no pre-built
artifact. They assert the behaviours docs/ml-spec.md lists for the scoring stage:
valid scores, safety review, monotonic reasons, reproducible
bands, and a stable worklist order.
"""

import json
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mh_risk_outreach import reasons, scoring
from mh_risk_outreach.bands import BAND_LABELS, assign_bands
from mh_risk_outreach.config import DEFAULT_CSV_PATH, MEMBER_ID
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.scores import (
    load_member_scores,
    save_member_scores,
    scores_status,
)
from mh_risk_outreach.models.artifacts import (
    ScoringArtifact,
    artifact_model_version,
    load_scoring_artifact,
)
from mh_risk_outreach.models.logistic_regression import LogisticRegressionRiskModel
from mh_risk_outreach.models.random_forest import RandomForestRiskModel
from mh_risk_outreach.models.train import train_and_evaluate
from mh_risk_outreach.scoring import (
    CRITICAL_INDICATORS,
    SHAP_CONTRIBUTIONS,
    rank_members,
    score_members,
)

CUT_POINTS = (20.0, 40.0, 60.0, 80.0)

SYMPTOM_COLUMNS = (
    "Feeling_Sad_Down",
    "Loss_Of_Interest",
    "Sleep_Trouble",
    "Fatigue",
    "Poor_Appetite_Or_Overeating",
    "Feeling_Worthless",
    "Concentration_Difficulty",
    "Anxious_Nervous",
    "Mood_Swings",
    "Irritability",
    "Obsessive_Thoughts",
    "Compulsive_Behavior",
    "Panic_Attacks",
)


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    frame = pd.read_csv(DEFAULT_CSV_PATH)
    frame.insert(0, MEMBER_ID, range(1, len(frame) + 1))
    sample = frame.groupby("Has_Mental_Health_Issue", group_keys=False).sample(
        frac=0.15, random_state=42
    )
    return sample.reset_index(drop=True)


@pytest.fixture(scope="module")
def background(records: pd.DataFrame) -> pd.DataFrame:
    """The training rows train_and_evaluate fits on: the SHAP reference set."""
    return split_dataset(records).train


@pytest.fixture(scope="module")
def lr_artifact(records: pd.DataFrame) -> ScoringArtifact:
    model = LogisticRegressionRiskModel()
    pipeline = train_and_evaluate(model, split_dataset(records)).pipeline
    return ScoringArtifact(model.name, "test_v1", pipeline, CUT_POINTS)


@pytest.fixture(scope="module")
def worklist(records: pd.DataFrame, lr_artifact, background) -> pd.DataFrame:
    return score_members(
        records,
        lr_artifact,
        background=background,
    )


def test_scores_are_0_to_100_one_row_per_member(records, worklist) -> None:
    assert len(worklist) == len(records)
    assert set(worklist[MEMBER_ID]) == set(records[MEMBER_ID])
    assert worklist["risk_score"].between(0, 100).all()


def test_rank_is_a_dense_1_to_n_order(worklist) -> None:
    ranks = sorted(worklist["rank_priority"])
    assert ranks == list(range(1, len(worklist) + 1))


def test_critical_indicators_take_the_top_priority(records, worklist) -> None:
    """Every self-harm / suicidal member is ranked above the rest."""
    critical = np.zeros(len(records), dtype=bool)
    for column in CRITICAL_INDICATORS:
        critical |= records[column].to_numpy() == 1
    critical_ids = set(records.loc[critical, MEMBER_ID])
    if not critical_ids:  # subsample happened to have none
        pytest.skip("no critical members in the subsample")

    flagged = worklist[worklist[MEMBER_ID].isin(critical_ids)]
    assert flagged["safety_review_required"].all()
    assert not worklist.loc[
        ~worklist[MEMBER_ID].isin(critical_ids), "safety_review_required"
    ].any()
    # They occupy the first len(critical_ids) ranks, above every other member.
    assert flagged["rank_priority"].max() == len(critical_ids)


def test_a_low_scoring_critical_member_is_still_top_priority(
    records, lr_artifact, background
) -> None:
    """Safety review determines contact order even when the model score is low."""
    calm = records.sort_values("Has_Mental_Health_Issue").iloc[0].copy()
    for column in SYMPTOM_COLUMNS:
        calm[column] = 0
    calm["Self_Harm_Thoughts"] = 1
    calm["Suicidal_Thoughts"] = 0
    frame = pd.concat([records, pd.DataFrame([calm])], ignore_index=True)
    frame[MEMBER_ID] = range(1, len(frame) + 1)

    result = score_members(
        frame,
        lr_artifact,
        background=background,
    )
    flagged = result[result[MEMBER_ID] == frame[MEMBER_ID].iloc[-1]].iloc[0]
    assert flagged["band_level_priority"] == "Low"
    assert flagged["safety_review_required"]
    assert flagged["rank_priority"] <= result["safety_review_required"].sum()
    assert (
        result.loc[~result["safety_review_required"], "risk_score"].max()
        > flagged["risk_score"]
    )


@pytest.mark.parametrize("flag", CRITICAL_INDICATORS)
def test_safety_flag_changes_rank_without_changing_model_outputs(
    records, lr_artifact, background, flag
) -> None:
    base = records.loc[
        (records["Self_Harm_Thoughts"] == 0) & (records["Suicidal_Thoughts"] == 0)
    ].iloc[0]
    unflagged = base.copy()
    flagged = base.copy()
    flagged[flag] = 1
    frame = pd.DataFrame([unflagged, flagged]).reset_index(drop=True)
    frame[MEMBER_ID] = [100001, 100002]

    scored = score_members(
        frame,
        lr_artifact,
        background=background,
    ).set_index(MEMBER_ID)

    assert scored.loc[100001, "risk_score"] == pytest.approx(
        scored.loc[100002, "risk_score"]
    )
    assert scored.loc[100002, "rank_priority"] == 1
    assert scored.loc[100002, "safety_review_required"]
    assert not scored.loc[100001, "safety_review_required"]
    assert (
        scored.loc[100002, "band_level_priority"]
        == scored.loc[100001, "band_level_priority"]
    )
    assert scored.loc[100002, "reasoning"] == scored.loc[100001, "reasoning"]


def test_old_artifact_requires_retraining_before_unpickling(tmp_path) -> None:
    folder = tmp_path / "logistic_regression"
    folder.mkdir()
    (folder / "model.joblib").write_bytes(b"not a pickle")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression",
                "model_version": "old_version",
                "model_features": ["Self_Harm_Thoughts", "Suicidal_Thoughts"],
                "prepared_column_count": 44,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outdated feature contract"):
        artifact_model_version("logistic_regression", str(tmp_path))
    with pytest.raises(SystemExit, match="retrain"):
        load_scoring_artifact("logistic_regression", str(tmp_path))


def test_maxed_symptoms_score_higher_than_zeroed(
    records, lr_artifact, background
) -> None:
    base = records.iloc[0].copy()
    high, low = base.copy(), base.copy()
    for column in SYMPTOM_COLUMNS:
        high[column] = 4 if column == "Panic_Attacks" else 10
        low[column] = 0
    frame = pd.DataFrame([high, low])
    frame[MEMBER_ID] = [1, 2]

    scored = score_members(
        frame,
        lr_artifact,
        background=background,
    ).set_index(MEMBER_ID)
    assert scored.loc[1, "risk_score"] > scored.loc[2, "risk_score"]


def test_reasoning_is_present_and_readable(worklist) -> None:
    assert (worklist["reasoning"].str.len() > 0).all()
    assert SHAP_CONTRIBUTIONS not in worklist


def test_structured_contributions_stay_with_requested_member_after_sorting(
    records, lr_artifact, background, monkeypatch
) -> None:
    frame = records.iloc[:2].copy().set_axis([30, 10])
    frame[MEMBER_ID] = [101, 102]
    frame["Self_Harm_Thoughts"] = [0, 1]
    frame["Suicidal_Thoughts"] = 0
    calls = []

    def contributions(_, prepared, *args, **kwargs):
        calls.append(prepared)
        signed = pd.DataFrame(0.0, index=prepared.index, columns=prepared.columns)
        signed["Age"] = [-0.5, 0.75]
        signed["Social_Support"] = [1.25, -1.5]
        return signed.to_numpy()

    monkeypatch.setattr(reasons, "shap_contributions", contributions)
    sorted_rows = score_members(
        frame, lr_artifact, background=background, contributions_for_member=101
    )

    assert len(calls) == 1
    assert sorted_rows[MEMBER_ID].tolist() == [102, 101]
    assert sorted_rows[SHAP_CONTRIBUTIONS].iat[0] is None
    captured = sorted_rows[SHAP_CONTRIBUTIONS].iat[1]
    assert set(captured) == set(calls[0].columns)
    assert captured["Age"] == -0.5
    assert captured["Social_Support"] == 1.25
    assert (
        reasons.build_reasons(lr_artifact.pipeline, frame, background)
        == sorted_rows["reasoning"].iloc[::-1].tolist()
    )


def test_assign_bands_is_reproducible_and_ordered() -> None:
    scores = np.linspace(40, 100, 500)
    first = assign_bands(scores, CUT_POINTS)
    second = assign_bands(scores, CUT_POINTS)
    assert list(first) == list(second)
    assert set(first) <= set(BAND_LABELS)
    # Higher scores never land in a lower band than lower scores.
    order = {label: index for index, label in enumerate(BAND_LABELS)}
    band_rank = np.array([order[band] for band in first])
    assert np.all(np.diff(band_rank) >= 0)


def test_assign_bands_treats_prediction_roundoff_as_cut_point_equality() -> None:
    cut = CUT_POINTS[0]
    scores = np.array([cut - 1e-11, np.nextafter(cut, -np.inf), cut])
    assert assign_bands(scores, CUT_POINTS).tolist() == ["Very Low", "Low", "Low"]


def test_scoring_is_deterministic(records, lr_artifact, background) -> None:
    a = score_members(
        records,
        lr_artifact,
        background=background,
    )
    b = score_members(
        records,
        lr_artifact,
        background=background,
    )
    pd.testing.assert_frame_equal(a, b)


def test_model_outputs_do_not_depend_on_the_batch(
    records, lr_artifact, background, worklist
) -> None:
    """A member scored alone gets the same model outputs as in the batch."""
    for member in records[MEMBER_ID].iloc[:5]:
        alone = score_members(
            records[records[MEMBER_ID] == member],
            lr_artifact,
            background=background,
        )
        in_batch = worklist[worklist[MEMBER_ID] == member]
        assert alone["risk_score"].iloc[0] == pytest.approx(
            in_batch["risk_score"].iloc[0]
        )
        assert (
            alone["band_level_priority"].iloc[0]
            == in_batch["band_level_priority"].iloc[0]
        )
        assert alone["reasoning"].iloc[0] == in_batch["reasoning"].iloc[0]
        assert alone["reasoning"].iloc[0] != "no strong risk factors identified"


def test_chunked_scoring_reports_progress_and_matches(
    records, lr_artifact, background, worklist
) -> None:
    """Explaining in chunks reports rising progress to 1.0 and changes nothing."""
    seen: list[float] = []
    chunked = score_members(
        records,
        lr_artifact,
        background=background,
        progress=seen.append,
        chunk_size=7,
    )
    pd.testing.assert_frame_equal(chunked, worklist)
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert len(seen) == -(-len(records) // 7)  # one call per chunk


def test_random_forest_also_scores_and_explains(records, background) -> None:
    """A tree model drops into the same scoring path (TreeExplainer reasons)."""
    pipeline = train_and_evaluate(
        RandomForestRiskModel(), split_dataset(records)
    ).pipeline
    result = score_members(
        records,
        ScoringArtifact("random_forest", "rf_test", pipeline, CUT_POINTS),
        background=background,
    )
    assert result["risk_score"].between(0, 100).all()
    assert (result["reasoning"].str.len() > 0).all()


def test_scores_table_round_trips(worklist, tmp_path) -> None:
    db = str(tmp_path / "scores.db")
    save_member_scores(worklist, "logistic_regression", db_path=db)
    stored = load_member_scores("logistic_regression", db_path=db)

    assert len(stored) == len(worklist)
    assert list(stored["rank_priority"]) == list(range(1, len(stored) + 1))
    assert set(stored.columns) >= {
        MEMBER_ID,
        "model_version",
        "risk_score",
        "band_level_priority",
        "rank_priority",
        "reasoning",
        "safety_review_required",
    }
    assert stored["safety_review_required"].equals(worklist["safety_review_required"])


def test_saving_scores_rejects_a_missing_column(worklist, tmp_path) -> None:
    db = str(tmp_path / "scores.db")
    with pytest.raises(KeyError):
        save_member_scores(
            worklist.drop(columns=["reasoning"]), "logistic_regression", db_path=db
        )


def test_failed_score_write_preserves_previous_table(worklist, tmp_path) -> None:
    db = str(tmp_path / "scores.db")
    save_member_scores(worklist, "logistic_regression", db_path=db)
    before = load_member_scores("logistic_regression", db_path=db)
    invalid = worklist.copy()
    invalid.loc[invalid.index[0], "reasoning"] = None

    with pytest.raises(pd.errors.DatabaseError) as error:
        save_member_scores(invalid, "logistic_regression", db_path=db)

    assert isinstance(error.value.__cause__, sqlite3.IntegrityError)
    pd.testing.assert_frame_equal(
        before, load_member_scores("logistic_regression", db_path=db)
    )


def test_scores_status_reports_missing_then_present(worklist, tmp_path) -> None:
    """The next-stage UI uses this to show 'run scoring first' without crashing."""
    db = str(tmp_path / "scores.db")
    assert scores_status("logistic_regression", db_path=db) == (False, 0)

    save_member_scores(worklist, "logistic_regression", db_path=db)
    exists, count = scores_status("logistic_regression", db_path=db)
    assert exists and count == len(worklist)
    # A model that was never scored still reports missing, not an error.
    assert scores_status("random_forest", db_path=db) == (False, 0)


def test_rank_members_uses_rounded_scores_and_member_id_for_ties() -> None:
    worklist = pd.DataFrame(
        {
            MEMBER_ID: [4, 3, 2, 1, 5],
            "risk_score": [85.0100000001, 85.010000000001, 85.01, 20.0, 19.0],
            "safety_review_required": [False, False, False, True, True],
            "reasoning": ["four", "three", "two", "one", "five"],
        }
    )
    ranked = rank_members(worklist)
    assert ranked[MEMBER_ID].tolist() == [1, 5, 4, 2, 3]
    assert ranked["rank_priority"].tolist() == [1, 2, 3, 4, 5]
    assert ranked["risk_score"].tolist() == [
        20.0,
        19.0,
        85.0100000001,
        85.01,
        85.010000000001,
    ]
    assert ranked["reasoning"].tolist() == ["one", "five", "four", "two", "three"]
    assert scoring.RANK_ORDER == (
        "safety_review_required desc",
        "risk_score rounded to 10 decimals desc",
        "Member_ID asc",
    )


def test_populated_legacy_table_refreshes_atomically(worklist, tmp_path) -> None:
    db = str(tmp_path / "scores.db")
    save_member_scores(worklist, "logistic_regression", db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "ALTER TABLE logistic_regression_scores DROP COLUMN safety_review_required"
        )
        before = pd.read_sql("SELECT * FROM logistic_regression_scores", conn)
    assert len(before) == len(worklist)
    with pytest.raises(ValueError, match="run scoring to refresh"):
        load_member_scores("logistic_regression", db_path=db)

    invalid = worklist.copy()
    invalid.loc[invalid.index[-1], "reasoning"] = None
    with pytest.raises(pd.errors.DatabaseError):
        save_member_scores(invalid, "logistic_regression", db_path=db)
    with sqlite3.connect(db) as conn:
        after = pd.read_sql("SELECT * FROM logistic_regression_scores", conn)
    pd.testing.assert_frame_equal(before, after)

    save_member_scores(worklist, "logistic_regression", db_path=db)
    refreshed = load_member_scores("logistic_regression", db_path=db)
    pd.testing.assert_frame_equal(worklist[refreshed.columns], refreshed)


def test_comparison_cli_uses_each_models_own_cut_points(
    records, tmp_path, monkeypatch
) -> None:
    db = str(tmp_path / "scores.db")
    models = {
        "logistic_regression": ((10.0, 20.0, 30.0, 40.0), "Very High"),
        "random_forest": ((30.0, 40.0, 60.0, 70.0), "Medium"),
        "xgboost": ((60.0, 70.0, 80.0, 90.0), "Very Low"),
    }
    members = records.iloc[:4]
    evaluation = members.iloc[:2]

    def load_artifact(model_name, artifacts_dir):
        assert artifacts_dir == str(tmp_path)
        pipeline = SimpleNamespace(
            predict_proba=lambda frame: np.full((len(frame), 2), 0.5)
        )
        return ScoringArtifact(
            model_name, f"{model_name}_test", pipeline, models[model_name][0]
        )

    monkeypatch.setattr(scoring, "load_from_db", lambda *_: members)
    monkeypatch.setattr(
        scoring,
        "split_dataset",
        lambda _: SimpleNamespace(evaluation=evaluation, train=members.iloc[2:]),
    )
    monkeypatch.setattr(scoring, "load_scoring_artifact", load_artifact)
    monkeypatch.setattr(
        scoring,
        "build_reasons_and_contributions",
        lambda _, frame, *args, **kwargs: (["reason"] * len(frame), {}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scoring", "--model", "all", "--db", db, "--artifacts", str(tmp_path)],
    )

    scoring.main()

    for model_name, (_, band) in models.items():
        worklist = load_member_scores(model_name, db_path=db)
        assert set(worklist[MEMBER_ID]) == set(evaluation[MEMBER_ID])
        assert worklist["model_version"].eq(f"{model_name}_test").all()
        assert worklist["band_level_priority"].eq(band).all()
