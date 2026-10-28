"""Logistic Regression — the first, interpretable risk model (issue #15).

The MVP model and the baseline that Random Forest and XGBoost are compared
against. It learns one weight per prepared column, so a member's top reasons come
straight from the fitted coefficients (see docs/ml-spec.md, Interpreting
Outputs). The preparation pipeline (issue #12) already scales the numeric inputs,
which logistic regression needs.

No class weighting: re-weighting the ~92% positive label barely changes the
ranking (PR-AUC) but badly distorts the probabilities (Brier 0.070 -> 0.231), and
members are prioritized by rank, not a 0.5 cutoff.
"""

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from mh_risk_outreach.config import RANDOM_STATE
from mh_risk_outreach.models.base import RiskModel


class LogisticRegressionRiskModel(RiskModel):
    """Plain L2-regularized logistic regression on the prepared inputs."""

    name = "logistic_regression"

    def build_estimator(self) -> BaseEstimator:
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
