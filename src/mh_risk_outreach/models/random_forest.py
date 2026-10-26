"""Random Forest — the first risk model (issue #15).

Chosen over logistic regression as the first implemented model. It is supervised,
gives usable probabilities to rank members by, and handles the ~92% positive rate
with ``class_weight="balanced"``. The whole model is the two lines below; the
shared framework (``base.RiskModel``, ``train``) does the rest.
"""

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from mh_risk_outreach.config import RANDOM_STATE
from mh_risk_outreach.models.base import RiskModel


class RandomForestRiskModel(RiskModel):
    """Random forest classifier, balanced for the imbalanced label."""

    name = "random_forest"

    def build_estimator(self) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
