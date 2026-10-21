"""Run the full data import in order and report its result.

Calls extract -> transform -> load, then prints a one-line summary. This is the
orchestrator for Segment 1; it holds the sequencing while each step lives in its
own module.
"""

import pandas as pd

from mh_risk_outreach.config import (
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE,
    FEATURES,
)
from mh_risk_outreach.db.repository import save_records
from mh_risk_outreach.etl.extract import extract
from mh_risk_outreach.etl.transform import transform


def run_etl(
    csv_path=DEFAULT_CSV_PATH,
    db_path: str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    """Extract CSV -> keep chosen columns + target (+ Member_ID) -> load into SQLite."""
    df = extract(csv_path)
    df = transform(df)
    save_records(df, db_path, table)

    print(
        f"[Segment 1] Loaded {len(df)} rows, {len(FEATURES)} features -> {db_path} (table '{table}')"
    )
    return df


if __name__ == "__main__":
    run_etl()
