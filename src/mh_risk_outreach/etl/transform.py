"""Select the chosen fields and put the records into the shape we store.

Keeps the agreed feature columns plus the target and adds a stable ``Member_ID``.
The transform is intentionally minimal for the POC: no imputation, scaling, or
category encoding (those belong to later segments). Values are kept as read.
"""

import pandas as pd

from mh_risk_outreach.config import FEATURES, MEMBER_ID, TARGET


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Keep FEATURES + TARGET and add a sequential Member_ID as the first column."""
    df = df[FEATURES + [TARGET]].copy()
    df.insert(0, MEMBER_ID, range(1, len(df) + 1))  # stable surrogate key
    return df
