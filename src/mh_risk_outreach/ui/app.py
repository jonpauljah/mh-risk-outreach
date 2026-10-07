"""Streamlit display for the imported member records (Milestone 1).

Reads the stored records from SQLite through the repository functions (never the
CSV directly) and shows them in a table. If the database has not been populated
yet, it explains how to run the import instead of erroring.

Launch:
    uv run streamlit run src/mh_risk_outreach/ui/app.py
"""

import streamlit as st

from mh_risk_outreach.config import DEFAULT_DB_PATH, DEFAULT_TABLE
from mh_risk_outreach.db.repository import database_status, load_from_db

IMPORT_COMMAND = "uv run python -m mh_risk_outreach.etl.pipeline"


@st.cache_data
def _load(db_path: str, table: str):
    """Load the records once per session; cleared when the inputs change."""
    return load_from_db(db_path, table)


def main() -> None:
    st.set_page_config(page_title="Member records", layout="wide")
    st.title("Member records")

    exists, row_count = database_status(DEFAULT_DB_PATH, DEFAULT_TABLE)
    if not exists or row_count == 0:
        st.warning(
            f"No records found in `{DEFAULT_DB_PATH}` (table `{DEFAULT_TABLE}`). "
            "Run the import to populate the database, then reload this page."
        )
        st.code(IMPORT_COMMAND, language="bash")
        st.stop()

    df = _load(DEFAULT_DB_PATH, DEFAULT_TABLE)
    st.caption(
        f"{len(df):,} records · {len(df.columns)} columns · "
        f"source `{DEFAULT_DB_PATH}` (table `{DEFAULT_TABLE}`)"
    )
    st.dataframe(df, width="stretch", hide_index=True)


main()
