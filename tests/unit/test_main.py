import sys
from pathlib import Path

import pytest

from mh_risk_outreach import main


def test_main_launches_streamlit_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() hands the app path to Streamlit's CLI instead of printing."""
    import streamlit.web.cli as stcli

    captured: dict[str, list[str]] = {}

    def fake_cli_main() -> None:
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(stcli, "main", fake_cli_main)

    main()

    argv = captured["argv"]
    assert argv[:2] == ["streamlit", "run"]
    assert argv[2].endswith(str(Path("ui") / "app.py"))
