import pytest

from mh_risk_outreach import main


def test_main_output(capsys: pytest.CaptureFixture[str]) -> None:
    main()

    assert capsys.readouterr().out == "Hello from mh-risk-outreach!\n"
