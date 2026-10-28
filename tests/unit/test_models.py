import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from mh_risk_outreach.config import DEFAULT_CSV_PATH, MEMBER_ID, TARGET
from mh_risk_outreach.db.results import (
    RESULTS_COLUMNS,
    load_model_results,
    save_model_results,
)
from mh_risk_outreach.evaluation import majority_baseline
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.base import RiskModel
from mh_risk_outreach.models.logistic_regression import LogisticRegressionRiskModel
from mh_risk_outreach.models.random_forest import RandomForestRiskModel
from mh_risk_outreach.models.train import save_artifact, train_and_evaluate
from mh_risk_outreach.models.xgboost_model import XGBoostRiskModel


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


def test_saved_and_reloaded_model_scores_identically(records, rf_result, tmp_path) -> None:
    out = save_artifact(rf_result, "random_forest", artifacts_dir=str(tmp_path))
    reloaded = joblib.load(out / "model.joblib")

    before = rf_result.pipeline.predict_proba(records)[:, 1]
    after = reloaded.predict_proba(records)[:, 1]
    assert np.allclose(before, after)


def test_results_row_round_trips_through_its_table(rf_result, tmp_path) -> None:
    db = str(tmp_path / "results.db")
    save_model_results(rf_result.results_row, table="random_forest", db_path=db)

    stored = load_model_results("random_forest", db_path=db)
    assert len(stored) == 1
    assert set(RESULTS_COLUMNS).issubset(stored.columns)
    assert stored.loc[0, "pr_auc"] == pytest.approx(rf_result.results_row["pr_auc"])


def test_a_new_model_extends_the_framework_without_modifying_it(records, tmp_path) -> None:
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
