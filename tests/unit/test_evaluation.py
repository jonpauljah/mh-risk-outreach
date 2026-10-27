import numpy as np
import pytest

from mh_risk_outreach.evaluation import evaluate_scores, majority_baseline


def test_majority_baseline_on_an_imbalanced_label() -> None:
    """Always predicting the common label: accuracy and PR-AUC are the prevalence."""
    y_true = [1, 1, 1, 0]  # 75% positive

    baseline = majority_baseline(y_true)

    assert baseline["majority_label"] == 1
    assert baseline["positive_rate"] == pytest.approx(0.75)
    assert baseline["accuracy"] == pytest.approx(0.75)
    assert baseline["pr_auc"] == pytest.approx(0.75)


def test_perfect_probabilities_score_perfectly() -> None:
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.1, 0.2]  # ranks and thresholds every row correctly

    metrics = evaluate_scores(y_true, y_prob)

    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["negative_recall"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.05


def test_every_metric_is_present_and_in_range() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(size=200)

    metrics = evaluate_scores(y_true, y_prob)

    expected = {
        "recall", "precision", "pr_auc", "roc_auc",
        "accuracy", "negative_recall", "brier",
    }
    assert set(metrics) == expected
    for name, value in metrics.items():
        assert np.isfinite(value), name
        assert 0.0 <= value <= 1.0, name
