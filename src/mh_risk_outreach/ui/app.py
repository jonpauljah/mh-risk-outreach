"""Display imported member records in a lightweight Streamlit interface.

This file will read member information through the database functions and show
the resulting SQL records in a table. The interface will let users inspect the
records and relevant features loaded for milestone 1.

Launch with: uv run streamlit run src/mh_risk_outreach/ui/app.py

Currently this shows a title only; database queries are not wired up yet.
"""

import streamlit as st

APP_TITLE = "MH Health Risk Identification System - WIP"


def main() -> None:
    """Render the first screen of the app."""
    st.set_page_config(page_title=APP_TITLE)
    st.title(APP_TITLE)


if __name__ == "__main__":
    main()
