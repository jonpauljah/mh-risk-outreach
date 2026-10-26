"""Persist one evaluation row per training run, in a table named per model.

Every model writes its results through this one writer, so adding a model (issue
#24 XGBoost, K-means, ...) needs no change here: it just passes its own ``table``
name and gets its own table with the same columns. That is the closed-for-
modification half of the model framework; the open-for-extension half is a new
``RiskModel`` subclass (see ``mh_risk_outreach.models``).

The table name is a model name, so it is interpolated into the SQL rather than
bound as a parameter (SQLite cannot bind identifiers). ``_safe_table`` rejects
anything that is not a plain identifier to keep that interpolation safe.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

from mh_risk_outreach.config import DEFAULT_DB_PATH

# One row per run. Keys are column names; values are their SQLite types. The
# metric columns mirror evaluation.evaluate_scores plus the baseline it is judged
# against, with enough run metadata to reproduce the number later.
RESULTS_COLUMNS: dict[str, str] = {
    "run_at": "TEXT NOT NULL",
    "model_version": "TEXT NOT NULL",
    "recall": "REAL NOT NULL",
    "precision": "REAL NOT NULL",
    "pr_auc": "REAL NOT NULL",
    "roc_auc": "REAL NOT NULL",
    "accuracy": "REAL NOT NULL",
    "negative_recall": "REAL NOT NULL",
    "brier": "REAL NOT NULL",
    "baseline_accuracy": "REAL NOT NULL",
    "baseline_pr_auc": "REAL NOT NULL",
    "n_train": "INTEGER NOT NULL",
    "n_eval": "INTEGER NOT NULL",
    "random_state": "INTEGER NOT NULL",
}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _safe_table(table: str) -> str:
    """Return ``table`` if it is a plain SQL identifier, else raise ``ValueError``."""
    if not _IDENTIFIER.match(table):
        raise ValueError(
            f"{table!r} is not a valid table name; use letters, digits, and "
            "underscores, starting with a letter or underscore"
        )
    return table


def _create_sql(table: str) -> str:
    columns = ",\n    ".join(f"{name} {sql_type}" for name, sql_type in RESULTS_COLUMNS.items())
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {columns}\n)"


def save_model_results(
    metrics: dict[str, object], table: str, db_path: str = DEFAULT_DB_PATH
) -> None:
    """Create the per-model table if needed and append one results row.

    ``metrics`` must supply every key in ``RESULTS_COLUMNS``; extra keys are
    ignored, so callers can pass a richer dict. The create and insert run in one
    transaction.
    """
    table = _safe_table(table)
    missing = sorted(set(RESULTS_COLUMNS) - set(metrics))
    if missing:
        raise KeyError(f"results for {table!r} are missing columns: {missing}")

    names = list(RESULTS_COLUMNS)
    placeholders = ", ".join("?" for _ in names)
    row = [metrics[name] for name in names]

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with conn:  # commits on success, rolls back on any exception
            conn.execute(_create_sql(table))
            conn.execute(
                f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})",
                row,
            )
    finally:
        conn.close()


def load_model_results(table: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Read a model's results table back, newest run first."""
    table = _safe_table(table)
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY run_at DESC", conn)
    finally:
        conn.close()
