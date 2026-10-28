"""Persist member risk scores in a table named per model (issue #16).

Each model gets its own ``<model>_scores`` table — ``logistic_regression_scores``,
``random_forest_scores``, ``xgboost_scores`` — the same per-model pattern as
``db/results.py``. A scoring run **replaces** that table's contents in one
transaction (like ``repository.save_records``), so each table always holds the
current worklist for its model, with ``model_version`` recording which trained
model produced it.

The table name is built from a model name, so it is interpolated into the SQL
rather than bound (SQLite cannot bind identifiers). ``_safe_table`` rejects
anything that is not a plain identifier to keep that interpolation safe.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

from mh_risk_outreach.config import DEFAULT_DB_PATH

# One row per scored member. Keys are column names; values are their SQLite
# types. Mirrors the DataFrame that scoring.score_members produces.
SCORES_COLUMNS: dict[str, str] = {
    "Member_ID": "INTEGER NOT NULL",
    "model_version": "TEXT NOT NULL",
    "risk_score": "REAL NOT NULL",
    "band_level_priority": "TEXT NOT NULL",
    "rank_priority": "INTEGER NOT NULL",
    "reasoning": "TEXT NOT NULL",
}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def scores_table_name(model_name: str) -> str:
    """Return the scores-table name for ``model_name`` (``<model>_scores``)."""
    return _safe_table(f"{model_name}_scores")


def _safe_table(table: str) -> str:
    """Return ``table`` if it is a plain SQL identifier, else raise ``ValueError``."""
    if not _IDENTIFIER.match(table):
        raise ValueError(
            f"{table!r} is not a valid table name; use letters, digits, and "
            "underscores, starting with a letter or underscore"
        )
    return table


def _create_sql(table: str) -> str:
    columns = ",\n    ".join(
        f"{name} {sql_type}" for name, sql_type in SCORES_COLUMNS.items()
    )
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {columns}\n)"


def save_member_scores(
    scores: pd.DataFrame, model_name: str, db_path: str = DEFAULT_DB_PATH
) -> None:
    """Replace the model's scores table with ``scores`` in one transaction.

    ``scores`` must contain every column in ``SCORES_COLUMNS``; extra columns are
    ignored. The table is created from the explicit schema first (never a
    pandas-inferred one), then its rows are deleted and ``scores`` appended, so a
    failure rolls the whole write back and leaves the previous worklist intact.
    """
    table = scores_table_name(model_name)
    missing = sorted(set(SCORES_COLUMNS) - set(scores.columns))
    if missing:
        raise KeyError(f"scores for {table!r} are missing columns: {missing}")

    rows = scores[list(SCORES_COLUMNS)]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with conn:  # commits on success, rolls back on any exception
            conn.execute(_create_sql(table))
            conn.execute(f"DELETE FROM {table}")
            rows.to_sql(table, conn, if_exists="append", index=False)
    finally:
        conn.close()


def load_member_scores(model_name: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Read a model's scores table back, in priority order (contact first)."""
    table = scores_table_name(model_name)
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY rank_priority", conn)
    finally:
        conn.close()


def scores_status(model_name: str, db_path: str = DEFAULT_DB_PATH) -> tuple[bool, int]:
    """Report whether a model's scores table exists and how many rows it holds.

    Returns ``(exists, row_count)`` and never raises for a missing file or table,
    so a consumer (e.g. the Streamlit worklist page) can show a "run scoring
    first" message instead of an error. ``(False, 0)`` means the database file or
    the ``<model>_scores`` table is not there yet. Mirrors
    ``repository.database_status`` for the source records.
    """
    table = scores_table_name(model_name)
    if not Path(db_path).exists():
        return (False, 0)
    conn = sqlite3.connect(db_path)
    try:
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not found:
            return (False, 0)
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return (True, int(count))
    finally:
        conn.close()
