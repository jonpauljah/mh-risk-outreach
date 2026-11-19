import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from mh_risk_outreach.bands import BAND_LABELS, assign_bands, fit_cut_points
from mh_risk_outreach.config import DEFAULT_CSV_PATH, MEMBER_ID, RANDOM_STATE, TARGET
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.results import (
    RESULTS_COLUMNS,
    load_model_results,
    save_model_results,
)
from mh_risk_outreach.evaluation import majority_baseline
from mh_risk_outreach.model_inputs import MODEL_FEATURES
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY, artifacts
from mh_risk_outreach.models.artifacts import (
    artifact_model_version,
    load_scoring_artifact,
)
from mh_risk_outreach.models.base import RiskModel
from mh_risk_outreach.models.logistic_regression import LogisticRegressionRiskModel
from mh_risk_outreach.models.random_forest import RandomForestRiskModel
from mh_risk_outreach.models.train import save_artifact, train_and_evaluate
from mh_risk_outreach.models.xgboost_model import XGBoostRiskModel
from mh_risk_outreach.prepare import PREPARED_COLUMN_COUNT


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    """A stratified subsample of the survey with a member ID, as the DB stores it.

    Subsampled so the forest trains in a second or two; kept stratified so both
    label classes survive the split.
    """
    frame = pd.read_csv(DEFAULT_CSV_PATH)
    frame.insert(0, MEMBER_ID, range(1, len(frame) + 1))
    sample = frame.groupby(TARGET, group_keys=False).sample(frac=0.15, random_state=42)
    return sample.reset_index(drop=True)


@pytest.fixture(scope="module")
def rf_result(records: pd.DataFrame):
    return train_and_evaluate(RandomForestRiskModel(), records)


def test_all_models_are_registered() -> None:
    assert REGISTRY["logistic_regression"] is LogisticRegressionRiskModel
    assert REGISTRY["random_forest"] is RandomForestRiskModel
    assert REGISTRY["xgboost"] is XGBoostRiskModel


def test_logistic_regression_is_the_default_model() -> None:
    """The MVP path trains logistic regression unless another model is asked for."""
    assert DEFAULT_MODEL == "logistic_regression"


def test_logistic_regression_trains_and_beats_the_baseline(records) -> None:
    """Logistic regression clears the PR-AUC gate with valid probabilities."""
    result = train_and_evaluate(LogisticRegressionRiskModel(), records)

    probs = result.pipeline.predict_proba(records)[:, 1]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    baseline = majority_baseline(records[TARGET])
    assert result.results_row["pr_auc"] >= baseline["pr_auc"]


def test_logistic_regression_has_one_weight_per_prepared_column(records) -> None:
    """The per-member top reasons depend on one signed weight per column."""
    result = train_and_evaluate(LogisticRegressionRiskModel(), records)

    coefficients = result.pipeline.named_steps["model"].coef_
    assert coefficients.shape == (1, result.metadata["prepared_column_count"])


def test_xgboost_trains_and_beats_the_baseline(records) -> None:
    """XGBoost drops into the same framework and clears the PR-AUC gate."""
    result = train_and_evaluate(XGBoostRiskModel(), records)

    probs = result.pipeline.predict_proba(records)[:, 1]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    baseline = majority_baseline(records[TARGET])
    assert result.results_row["pr_auc"] >= baseline["pr_auc"]


def test_scores_are_probabilities_one_per_eval_row(records, rf_result) -> None:
    row = rf_result.results_row
    pipeline = rf_result.pipeline
    # Re-score the pipeline's view of the data to check the probability range.
    probs = pipeline.predict_proba(records)[:, 1]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert row["n_eval"] > 0
    assert row["n_train"] > row["n_eval"]


def test_random_forest_beats_the_majority_baseline(records, rf_result) -> None:
    """The gate: PR-AUC must clear always-predict-the-common-label."""
    # Baseline PR-AUC on the same eval share equals the positive prevalence.
    baseline = majority_baseline(records[TARGET])
    assert rf_result.results_row["pr_auc"] >= baseline["pr_auc"]


def test_saved_and_reloaded_model_scores_identically(
    records, rf_result, tmp_path
) -> None:
    save_artifact(
        rf_result,
        "random_forest",
        str(tmp_path),
        training_rows=split_dataset(records).train,
    )
    reloaded = load_scoring_artifact("random_forest", str(tmp_path))

    before = rf_result.pipeline.predict_proba(records)[:, 1]
    after = reloaded.pipeline.predict_proba(records)[:, 1]
    assert np.allclose(before, after)
    assert reloaded.model_version == rf_result.metadata["model_version"]
    assert rf_result.metadata["model_features"] == list(MODEL_FEATURES)
    assert rf_result.metadata["prepared_column_count"] == PREPARED_COLUMN_COUNT


def test_frozen_cuts_are_midpoints_and_equality_enters_higher_band() -> None:
    scores = np.array([0.1, 10.2, 50.3, 80.4, 99.5] * 2)
    cuts = fit_cut_points(scores)
    assert cuts == pytest.approx((5.15, 30.25, 65.35, 89.95))
    assert assign_bands(np.array([0.0, *cuts, 100.0]), cuts).tolist() == [
        *BAND_LABELS,
        BAND_LABELS[-1],
    ]
    with pytest.raises(ValueError, match="at least five distinct"):
        fit_cut_points(np.array([0.0, 5.0, 5.0, 100.0]))


def test_saved_bands_are_learned_from_the_training_split(
    records, rf_result, tmp_path
) -> None:
    train = split_dataset(records).train
    scores = rf_result.pipeline.predict_proba(train)[:, 1] * 100.0
    out = save_artifact(rf_result, "random_forest", str(tmp_path), training_rows=train)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["risk_bands"] == {
        "labels": list(BAND_LABELS),
        "cut_points": list(fit_cut_points(scores)),
        "reference_scope": "train",
        "reference_row_count": len(train),
        "algorithm": "kmeans",
        "random_state": RANDOM_STATE,
        "n_init": 10,
    }


def test_constant_scores_allow_evaluation_but_prevent_artifact_publication(
    records, tmp_path
) -> None:
    constant_inputs = records.iloc[[0] * 20].assign(
        **{MEMBER_ID: range(1, 21), TARGET: [0, 1] * 10}
    )
    model = LogisticRegressionRiskModel()
    result = train_and_evaluate(model, constant_inputs)

    assert result.results_row["roc_auc"] == pytest.approx(0.5)
    assert "risk_bands" not in result.metadata
    with pytest.raises(ValueError, match="at least five distinct"):
        save_artifact(
            result,
            model.name,
            str(tmp_path),
            training_rows=split_dataset(constant_inputs).train,
        )
    assert not list(tmp_path.iterdir())


def test_retraining_keeps_the_older_version_addressable(records, tmp_path) -> None:
    model = LogisticRegressionRiskModel()
    subsample = records.iloc[::2]
    first = train_and_evaluate(model, records)
    second = train_and_evaluate(model, subsample)
    first_train = split_dataset(records).train
    first_path = save_artifact(
        first, model.name, str(tmp_path), training_rows=first_train
    )
    latest_path = save_artifact(
        second, model.name, str(tmp_path), training_rows=split_dataset(subsample).train
    )
    first_metadata = json.loads((first_path / "metadata.json").read_text())
    latest_metadata = json.loads((latest_path / "metadata.json").read_text())
    older_version = str(first.metadata["model_version"])
    latest = load_scoring_artifact(model.name, str(tmp_path))
    older = load_scoring_artifact(model.name, str(tmp_path), version=older_version)

    assert latest.model_version == latest_path.name == second.metadata["model_version"]
    assert older.model_version == older_version != latest.model_version
    assert older.cut_points == tuple(first_metadata["risk_bands"]["cut_points"])
    assert latest.cut_points == tuple(latest_metadata["risk_bands"]["cut_points"])
    assert older.cut_points != latest.cut_points
    assert np.allclose(
        older.pipeline.predict_proba(records), first.pipeline.predict_proba(records)
    )
    with pytest.raises(OSError):
        save_artifact(first, model.name, str(tmp_path), training_rows=first_train)
    assert artifact_model_version(model.name, str(tmp_path)) == latest.model_version


def test_failed_save_does_not_advance_latest(
    records, rf_result, tmp_path, monkeypatch
) -> None:
    model = RandomForestRiskModel()
    train = split_dataset(records).train
    out = save_artifact(rf_result, model.name, str(tmp_path), training_rows=train)
    second = train_and_evaluate(model, records)

    def fail_dump(*args, **kwargs):
        raise OSError("failed write")

    monkeypatch.setattr(artifacts.joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="failed write"):
        save_artifact(second, model.name, str(tmp_path), training_rows=train)
    assert list(out.parent.iterdir()) == [out]
    assert artifact_model_version(model.name, str(tmp_path)) == out.name


@pytest.mark.parametrize("cuts", [[10, 10, 20, 30], [10, 20, 30, 100], [10, 20, 30]])
def test_loader_rejects_invalid_saved_cut_points(
    cuts, records, rf_result, tmp_path
) -> None:
    out = save_artifact(
        rf_result,
        "random_forest",
        str(tmp_path),
        training_rows=split_dataset(records).train,
    )
    path = out / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["risk_bands"]["cut_points"] = cuts
    path.write_text(json.dumps(metadata))

    with pytest.raises(SystemExit, match="cut points.*retrain"):
        load_scoring_artifact("random_forest", str(tmp_path))


def test_legacy_artifact_with_current_features_requires_retraining(
    rf_result, tmp_path
) -> None:
    folder = tmp_path / "random_forest"
    folder.mkdir()
    (folder / "model.joblib").write_bytes(b"legacy")
    (folder / "metadata.json").write_text(json.dumps(rf_result.metadata))

    with pytest.raises(SystemExit, match="predates frozen risk bands.*retrain"):
        load_scoring_artifact("random_forest", str(tmp_path))


def test_results_row_round_trips_through_its_table(rf_result, tmp_path) -> None:
    db = str(tmp_path / "results.db")
    save_model_results(rf_result.results_row, table="random_forest", db_path=db)

    stored = load_model_results("random_forest", db_path=db)
    assert len(stored) == 1
    assert set(RESULTS_COLUMNS).issubset(stored.columns)
    assert stored.loc[0, "pr_auc"] == pytest.approx(rf_result.results_row["pr_auc"])


def test_a_new_model_extends_the_framework_without_modifying_it(
    records, tmp_path
) -> None:
    """Open/closed: a new RiskModel trains and writes to its own table, untouched base."""

    class DummyRiskModel(RiskModel):
        name = "dummy_model"

        def build_estimator(self) -> BaseEstimator:
            return LogisticRegression(max_iter=1000)

    result = train_and_evaluate(DummyRiskModel(), records)

    db = str(tmp_path / "results.db")
    save_model_results(result.results_row, table="dummy_model", db_path=db)
    stored = load_model_results("dummy_model", db_path=db)
    assert len(stored) == 1
    assert stored.loc[0, "model_version"].startswith("dummy_model_")


def test_a_model_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match="non-empty 'name'"):

        class Nameless(RiskModel):  # no name set
            def build_estimator(self) -> BaseEstimator:
                return LogisticRegression()
