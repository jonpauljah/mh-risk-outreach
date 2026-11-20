"""Train and evaluate one risk model, save it, and record its results.

This is the shared workflow for every ``RiskModel``: load the stored survey,
reuse the issue #14 split, fit the model's pipeline on the training rows only,
score the held-out rows, measure the result against the majority baseline, save
the fitted pipeline with reproducibility metadata, and write one row to the
model's own results table. Adding a model changes nothing here.

Run it:
    uv run python -m mh_risk_outreach.models.train               # logistic_regression
    uv run python -m mh_risk_outreach.models.train --model random_forest
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from mh_risk_outreach.bands import BAND_LABELS, BAND_N_INIT, fit_cut_points
from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    DEFAULT_TABLE,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from mh_risk_outreach.datasets import DatasetSplit, split_dataset
from mh_risk_outreach.db.repository import load_from_db
from mh_risk_outreach.db.results import save_model_results
from mh_risk_outreach.evaluation import (
    evaluate_scores,
    format_report,
    majority_baseline,
)
from mh_risk_outreach.model_inputs import MODEL_FEATURES
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    save_versioned_artifact,
)
from mh_risk_outreach.models.base import RiskModel, predict_risk_scores
from mh_risk_outreach.prepare import PREPARED_COLUMN_COUNT


class TrainingResult(NamedTuple):
    """What one training run produced, in memory."""

    pipeline: Pipeline
    results_row: dict[str, object]
    metadata: dict[str, object]
    report: str


def train_and_evaluate(model: RiskModel, split: DatasetSplit) -> TrainingResult:
    """Fit on the training split and measure performance on the held-out rows.

    Returns the fitted pipeline, the flat results-table row, run metadata,
    and a plain-language report.
    """
    pipeline = model.build_pipeline()
    pipeline.fit(split.train, split.train[TARGET])

    y_true = split.evaluation[TARGET]
    y_prob = pipeline.predict_proba(split.evaluation)[:, 1]
    metrics = evaluate_scores(y_true, y_prob)
    baseline = majority_baseline(y_true)

    now = datetime.now(UTC)
    run_at = now.isoformat(timespec="seconds")
    version = f"{model.name}_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex}"
    n_train, n_eval = len(split.train), len(split.evaluation)

    results_row: dict[str, object] = {
        "run_at": run_at,
        "model_version": version,
        **{
            k: metrics[k]
            for k in (
                "recall",
                "precision",
                "pr_auc",
                "roc_auc",
                "accuracy",
                "negative_recall",
                "brier",
            )
        },
        "baseline_accuracy": baseline["accuracy"],
        "baseline_pr_auc": baseline["pr_auc"],
        "n_train": n_train,
        "n_eval": n_eval,
        "random_state": RANDOM_STATE,
    }
    metadata: dict[str, object] = {
        **results_row,
        "model_name": model.name,
        "estimator": repr(model.build_estimator()),
        "sklearn_version": sklearn.__version__,
        "test_size": TEST_SIZE,
        "model_features": list(MODEL_FEATURES),
        "prepared_column_count": PREPARED_COLUMN_COUNT,
        "baseline_positive_rate": baseline["positive_rate"],
    }
    report = format_report(
        model.name, metrics, baseline, n_train=n_train, n_eval=n_eval
    )
    return TrainingResult(pipeline, results_row, metadata, report)


def save_artifact(
    result: TrainingResult,
    model_name: str,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    *,
    training_rows: pd.DataFrame,
) -> Path:
    """Learn bands from the model's training rows and save the complete bundle."""
    cut_points = fit_cut_points(predict_risk_scores(result.pipeline, training_rows))
    metadata = {
        **result.metadata,
        "risk_bands": {
            "labels": list(BAND_LABELS),
            "cut_points": list(cut_points),
            "reference_scope": "train",
            "reference_row_count": len(training_rows),
            "algorithm": "kmeans",
            "random_state": RANDOM_STATE,
            "n_init": BAND_N_INIT,
        },
    }
    return save_versioned_artifact(result.pipeline, metadata, model_name, artifacts_dir)


def run(
    model_name: str = DEFAULT_MODEL,
    *,
    db_path: str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> TrainingResult:
    """Train one registered model end to end and persist everything."""
    try:
        model = REGISTRY[model_name]()
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise SystemExit(f"unknown model {model_name!r}; choose from: {known}")

    records = load_from_db(db_path, table)
    split = split_dataset(records)
    result = train_and_evaluate(model, split)
    out = save_artifact(result, model.name, artifacts_dir, training_rows=split.train)
    save_model_results(result.results_row, table=model.name, db_path=db_path)

    print(result.report)
    print(
        f"\nSaved model to {out}\nWrote results row to table {model.name!r} in {db_path}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a risk model.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(REGISTRY),
        help="model to train",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="source records table")
    parser.add_argument(
        "--artifacts", default=DEFAULT_ARTIFACTS_DIR, help="artifacts dir"
    )
    args = parser.parse_args()
    run(args.model, db_path=args.db, table=args.table, artifacts_dir=args.artifacts)


if __name__ == "__main__":
    main()
