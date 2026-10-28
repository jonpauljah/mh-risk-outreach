"""The risk-model framework and the registry of available models.

Adding a model (issue #24 XGBoost, K-means, ...) is two edits and no more:
write a ``RiskModel`` subclass in its own module, then add one line to
``REGISTRY`` below. Training, evaluation, persistence, and the per-model results
table all work from the registry, so none of them change.
"""

from mh_risk_outreach.models.base import RiskModel
from mh_risk_outreach.models.logistic_regression import LogisticRegressionRiskModel
from mh_risk_outreach.models.random_forest import RandomForestRiskModel
from mh_risk_outreach.models.xgboost_model import XGBoostRiskModel

# name -> model class. The name is also the results-table and artifact-folder name.
REGISTRY: dict[str, type[RiskModel]] = {
    LogisticRegressionRiskModel.name: LogisticRegressionRiskModel,
    RandomForestRiskModel.name: RandomForestRiskModel,
    XGBoostRiskModel.name: XGBoostRiskModel,
}

DEFAULT_MODEL = LogisticRegressionRiskModel.name

__all__ = [
    "DEFAULT_MODEL",
    "REGISTRY",
    "LogisticRegressionRiskModel",
    "RandomForestRiskModel",
    "RiskModel",
    "XGBoostRiskModel",
]
