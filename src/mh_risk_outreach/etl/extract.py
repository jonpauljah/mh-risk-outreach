"""Read member data from the selected healthcare CSV.

Opens the CSV and returns the raw records as a DataFrame for the transform step.
Column selection and cleaning happen later in `transform.py`.
"""

import pandas as pd


def extract(csv_path) -> pd.DataFrame:
    """Read the CSV into a DataFrame."""
    return pd.read_csv(csv_path)
