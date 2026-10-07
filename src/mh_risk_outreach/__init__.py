"""Package entry point: launch the Streamlit member-records display.

``main`` is the console-script target (``mh-risk-outreach``), so
``uv run mh-risk-outreach`` opens the GUI. It hands off to Streamlit's CLI to
serve ``ui/app.py``. The ETL import stays a separate command:
``uv run python -m mh_risk_outreach.etl.pipeline``.
"""

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit display for the stored member records."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    stcli.main()
