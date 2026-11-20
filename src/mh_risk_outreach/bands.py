"""Learn five risk bands during training and apply their frozen cut points."""

from numbers import Real

import numpy as np
from sklearn.cluster import KMeans

from mh_risk_outreach.config import RANDOM_STATE

BAND_LABELS = ("Very Low", "Low", "Medium", "High", "Very High")
BAND_N_INIT = 10
BAND_BOUNDARY_TOLERANCE = 1e-12


def validate_cut_points(cut_points: object) -> tuple[float, ...]:
    """Require four finite, strictly increasing numeric points inside 0–100."""
    if (
        not isinstance(cut_points, (list, tuple))
        or len(cut_points) != len(BAND_LABELS) - 1
    ):
        raise ValueError("risk bands require four cut points")
    if any(
        isinstance(point, bool) or not isinstance(point, Real) for point in cut_points
    ):
        raise ValueError("risk band cut points must be numeric")
    points = tuple(float(point) for point in cut_points)
    if not all(a < b for a, b in zip((0.0, *points), (*points, 100.0), strict=True)):
        raise ValueError("risk band cut points must increase strictly inside 0–100")
    return points


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all((values >= 0.0) & (values <= 100.0)):
        raise ValueError("risk scores must be one-dimensional and within 0–100")
    return values


def fit_cut_points(scores: np.ndarray) -> tuple[float, ...]:
    """Fit seeded 1-D K-means once and return midpoints between sorted centers."""
    values = _validate_scores(scores)
    if np.unique(values).size < len(BAND_LABELS):
        raise ValueError("risk band training requires at least five distinct scores")
    kmeans = KMeans(
        n_clusters=len(BAND_LABELS), random_state=RANDOM_STATE, n_init=BAND_N_INIT
    ).fit(values.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.ravel())
    if np.unique(centers).size != len(BAND_LABELS):
        raise ValueError("risk band training requires five distinct cluster centers")
    return validate_cut_points((centers[:-1] + np.diff(centers) / 2.0).tolist())


def assign_bands(scores: np.ndarray, cut_points: object) -> np.ndarray:
    """Assign saved bands, treating prediction noise below a cut as equality."""
    values = _validate_scores(scores)
    boundaries = np.array([*validate_cut_points(cut_points), np.inf])
    indices = np.searchsorted(boundaries, values, side="left")
    at_boundary = (values > 0.0) & (
        boundaries[indices] - values < BAND_BOUNDARY_TOLERANCE
    )
    return np.asarray(BAND_LABELS, dtype=object)[indices + at_boundary]
