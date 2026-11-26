"""Identify the fixed operational cohort from the original imported survey."""

import sqlite3

import pandas as pd

from mh_risk_outreach.config import MEMBER_ID, TARGET
from mh_risk_outreach.datasets import DatasetSplit, split_dataset


def load_operational_split(
    conn: sqlite3.Connection, *, ids_only: bool = False
) -> DatasetSplit:
    """Reproduce the original split using the imported rows in member ID order."""
    columns = f"{MEMBER_ID}, {TARGET}" if ids_only else "*"
    records = pd.read_sql_query(
        f"SELECT {columns} FROM patients ORDER BY {MEMBER_ID}", conn
    )
    if records.empty:
        return DatasetSplit(train=records, evaluation=records)
    return split_dataset(records)


def load_operational_member_ids(conn: sqlite3.Connection) -> set[int]:
    """Return the held-out IDs independently of scores and assessments."""
    return set(load_operational_split(conn, ids_only=True).evaluation[MEMBER_ID])


def require_operational_member(member_id: int, member_ids: set[int]) -> None:
    if member_id not in member_ids:
        raise ValueError(f"member {member_id} is outside the operational population")
