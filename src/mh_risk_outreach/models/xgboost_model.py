"""XGBoost — the gradient-boosted tree model (issue #24).

The second model, added to compare against Random Forest and the majority
baseline on the same split (#14), inputs (#13), and metrics. Like Random Forest
it is a supervised classifier with ``predict_proba``, so it drops into the shared
framework unchanged: the whole model is a name and an estimator.
"""

from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

from mh_risk_outreach.config import RANDOM_STATE
from mh_risk_outreach.models.base import RiskModel

# ~92% of rows are positive, so positives are the majority. scale_pos_weight is
# negatives / positives, which down-weights the majority class the way
# RandomForest's class_weight="balanced" does. 784 / 9216 ≈ 0.085 on the Kaggle
# snapshot; revisit if the label balance changes with new data.
_SCALE_POS_WEIGHT = 0.085


class XGBoostRiskModel(RiskModel):
    """Gradient-boosted trees, weighted for the imbalanced label."""

    name = "xgboost"

    def build_estimator(self) -> BaseEstimator:
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=_SCALE_POS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )
