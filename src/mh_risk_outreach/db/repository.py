"""SQLite access for the prepared member records.

Saves the transformed records into a single table and reads them back for later
segments. All database access lives here, so moving from SQLite to another store
(e.g. Postgres) later touches only this module.
"""

import sqlite3

import pandas as pd

from mh_risk_outreach.config import DEFAULT_DB_PATH, DEFAULT_TABLE


def save_records(
    df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE
) -> None:
    """Write the prepared records into SQLite, replacing any prior load."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()


def load_from_db(
    db_path: str = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE
) -> pd.DataFrame:
    """Read the clean table back out of SQLite (what later segments call)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df
