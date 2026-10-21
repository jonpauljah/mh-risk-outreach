"""SQLite access for the prepared member records.

All database access lives here, so moving from SQLite to another store (e.g.
Postgres) later touches only this module. The table is defined by the tracked
``schema.sql``, not inferred by pandas: ``init_schema`` creates it, and
``save_records`` appends rows into that existing table.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from mh_risk_outreach.config import DEFAULT_DB_PATH, DEFAULT_TABLE

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the tracked tables from ``schema.sql`` if they do not exist."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def save_records(
    df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE
) -> None:
    """Replace the table contents with ``df`` in a single transaction.

    The schema is created from ``schema.sql`` first, so pandas appends into the
    tracked table (never ``if_exists="replace"``, which would discard it). On a
    repeat import the existing rows are deleted and the fresh load appended
    inside one transaction; any failure rolls the whole operation back, leaving
    the previously stored data intact.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        with conn:  # commits on success, rolls back on any exception
            conn.execute(f"DELETE FROM {table}")
            df.to_sql(table, conn, if_exists="append", index=False)
    finally:
        conn.close()


def load_from_db(
    db_path: str = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE
) -> pd.DataFrame:
    """Read the clean table back out of SQLite (what later segments call)."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


def database_status(
    db_path: str = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE
) -> tuple[bool, int]:
    """Report whether the table exists and how many rows it holds.

    Returns ``(exists, row_count)`` and never raises for a missing file or
    table, so the UI can show a populate-the-database message instead of an
    error. ``(False, 0)`` means the database file or table is not there yet.
    """
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
