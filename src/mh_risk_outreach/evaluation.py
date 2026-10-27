"""Measure how useful a model's risk scores are, next to a simple baseline.

The functions here learn nothing and fit nothing; they only compare predicted
probabilities against the true labels. ``evaluate_scores`` answers the four
questions issue #15 asks — how well the model finds positive cases (recall), how
reliable its positive predictions are (precision), how well it ranks members
(PR-AUC), and how closely its probabilities match reality (calibration) — plus a
few supporting numbers. ``majority_baseline`` gives the same headline numbers for
the "always predict the most common label" approach, so every result is reported
against the bar it has to beat. See docs/ml-spec.md.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Score at/above this probability is predicted positive for the threshold
# metrics (recall, precision, accuracy). Ranking metrics (PR-AUC, ROC-AUC) and
# calibration (Brier) use the probabilities directly and ignore it.
DECISION_THRESHOLD = 0.5


def evaluate_scores(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    y_prob: Sequence[float] | pd.Series | np.ndarray,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Score predicted probabilities against the true 0/1 labels.

    ``y_prob`` is the probability of the positive class (an issue). Returns one
    flat dict so it can be written straight into a results-table row:

    * ``recall`` / ``precision`` – of the members with an issue, the share we
      flag; of those we flag, the share who truly had one (threshold applied).
    * ``pr_auc`` – ranking quality across all thresholds (the primary gate).
    * ``roc_auc`` / ``accuracy`` – reported for context, not used as gates.
    * ``negative_recall`` – share of the "no issue" minority caught; the harder,
      more telling signal given the ~92% positive rate.
    * ``brier`` – calibration error (mean squared gap between probability and
      outcome); lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "negative_recall": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def majority_baseline(
    y_true: Sequence[int] | pd.Series | np.ndarray,
) -> dict[str, float]:
    """Headline numbers for always predicting the most common label.

    This is the bar a real model has to clear. With a ~92% positive rate,
    ``accuracy`` lands near 0.92 by itself, which is why PR-AUC is the gate: for a
    constant score the average-precision equals the positive prevalence, so
    ``pr_auc`` here is that prevalence.
    """
    y_true = np.asarray(y_true, dtype=int)
    positive_rate = float(y_true.mean())
    majority = 1 if positive_rate >= 0.5 else 0
    accuracy = positive_rate if majority == 1 else 1.0 - positive_rate
    return {
        "majority_label": float(majority),
        "positive_rate": positive_rate,
        "accuracy": accuracy,
        "pr_auc": positive_rate,
    }


def format_report(
    model_name: str,
    metrics: dict[str, float],
    baseline: dict[str, float],
    *,
    n_train: int,
    n_eval: int,
) -> str:
    """Build a plain-language evaluation report comparing model to baseline."""
    beats = "ABOVE" if metrics["pr_auc"] > baseline["pr_auc"] else "NOT ABOVE"
    return "\n".join(
        [
            f"[{model_name}] trained on {n_train:,} rows, evaluated on {n_eval:,}.",
            "",
            "How well it ranks members (the gate):",
            (
                f"  PR-AUC          {metrics['pr_auc']:.3f}   "
                f"baseline {baseline['pr_auc']:.3f}  -> {beats} baseline"
            ),
            f"  ROC-AUC         {metrics['roc_auc']:.3f}",
            "",
            "At the 0.5 threshold:",
            (
                f"  Recall          {metrics['recall']:.3f}   "
                "(of members with an issue, the share flagged)"
            ),
            (
                f"  Precision       {metrics['precision']:.3f}   "
                "(of those flagged, the share who had one)"
            ),
            (
                f"  Neg. recall     {metrics['negative_recall']:.3f}   "
                "(of members with no issue, the share correctly cleared)"
            ),
            (
                f"  Accuracy        {metrics['accuracy']:.3f}   "
                f"baseline {baseline['accuracy']:.3f}  (not a gate; ~0.92 by itself)"
            ),
            "",
            "Calibration:",
            (
                f"  Brier           {metrics['brier']:.3f}   "
                "(mean squared gap between probability and outcome; lower is better)"
            ),
        ]
    )
