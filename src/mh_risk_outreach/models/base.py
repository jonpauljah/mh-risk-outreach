"""The one thing every risk model shares: a name and an unfitted estimator.

A new model (XGBoost, K-means, ...) is added by subclassing ``RiskModel``,
setting ``name``, and returning its estimator from ``build_estimator``. Nothing
else changes: ``build_pipeline`` wires the shared preparation pipeline (issue
#12) in front of the estimator, and ``mh_risk_outreach.models.train`` fits,
evaluates, saves, and records results for any ``RiskModel``. That is the
open/closed boundary — extension by subclass, no modification of the workflow.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from joblib import parallel_config
from scipy.special import expit  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mh_risk_outreach.prepare import build_preparation_pipeline


def predict_risk_scores(pipeline: Pipeline, records: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores on the 0–100 scale with stable row arithmetic."""
    steps = getattr(pipeline, "named_steps", {})
    model = steps.get("model")
    if isinstance(model, LogisticRegression) and len(model.classes_) == 2:
        prepared = steps["prepare"].transform(records)
        values = np.ascontiguousarray(prepared.to_numpy(dtype=float))
        # A fixed reduction avoids different BLAS paths for a row and a batch.
        logits = (
            np.einsum("ij,j->i", values, model.coef_[0], optimize=False)
            + model.intercept_[0]
        )
        return expit(logits) * 100.0
    if isinstance(model, RandomForestClassifier):
        # Sum the tree probabilities in estimator order for each prediction.
        with parallel_config(backend="sequential"):
            return pipeline.predict_proba(records)[:, 1] * 100.0
    return pipeline.predict_proba(records)[:, 1] * 100.0


class RiskModel(ABC):
    """A named model plus the preparation pipeline that feeds it.

    Subclasses set the class attribute ``name`` (used for the results table and
    the saved-artifact folder, so it must be a plain identifier) and implement
    ``build_estimator``. Everything a training run needs beyond that is here.
    """

    name: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(f"{cls.__name__} must set a non-empty 'name'")

    @abstractmethod
    def build_estimator(self) -> BaseEstimator:
        """Return the unfitted classifier for this model (the only new code)."""

    def build_pipeline(self) -> Pipeline:
        """Clean + preprocess (issue #12) then the estimator, as one pipeline.

        Fitting this on the training split fits the preprocessing on those rows
        only, so no evaluation-row values leak in. Saving the fitted pipeline
        bundles preprocessing with the model, so scoring prepares rows identically.
        """
        return Pipeline(
            [
                ("prepare", build_preparation_pipeline()),
                ("model", self.build_estimator()),
            ]
        )
