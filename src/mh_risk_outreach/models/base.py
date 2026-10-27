"""The one thing every risk model shares: a name and an unfitted estimator.

A new model (XGBoost, K-means, ...) is added by subclassing ``RiskModel``,
setting ``name``, and returning its estimator from ``build_estimator``. Nothing
else changes: ``build_pipeline`` wires the shared preparation pipeline (issue
#12) in front of the estimator, and ``mh_risk_outreach.models.train`` fits,
evaluates, saves, and records results for any ``RiskModel``. That is the
open/closed boundary — extension by subclass, no modification of the workflow.
"""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from mh_risk_outreach.prepare import build_preparation_pipeline


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
