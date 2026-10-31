"""The Risk scoring Streamlit page (issue #28), driven headlessly with AppTest.

The page reads its database and artifact paths from module constants, so each
test points them at a temporary folder; the real ``patients.db`` and
``artifacts/`` are never touched.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mh_risk_outreach import config, scoring
from mh_risk_outreach.etl.pipeline import run_etl

PAGE = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "mh_risk_outreach"
    / "ui"
    / "pages"
    / "2_Risk_scoring.py"
)


@pytest.fixture
def paths(tmp_path, monkeypatch) -> tuple[str, str]:
    db = str(tmp_path / "patients.db")
    artifacts = str(tmp_path / "artifacts")
    monkeypatch.setattr(config, "DEFAULT_DB_PATH", db)
    monkeypatch.setattr(scoring, "DEFAULT_ARTIFACTS_DIR", artifacts)
    return db, artifacts


def _page() -> AppTest:
    return AppTest.from_file(PAGE, default_timeout=120)


def test_empty_database_explains_how_to_import(paths) -> None:
    at = _page().run()

    assert not at.exception
    assert "No records found" in at.warning[0].value
    assert not [b for b in at.button if b.label == "Run"]


def test_train_run_and_reset(paths) -> None:
    """Untrained -> Train -> Run shows a worklist table -> Reset allows another run."""
    db, _ = paths
    run_etl(db_path=db)
    at = _page().run()
    at.multiselect(key="risk_scoring_models").set_value(["logistic_regression"]).run()

    assert "Not trained" in " ".join(m.value for m in at.markdown)
    assert not [b for b in at.button if b.label == "Run"]

    at.button(key="train_logistic_regression").click().run()
    assert "Ready" in " ".join(m.value for m in at.markdown)

    next(b for b in at.button if b.label == "Run").click().run()
    assert not at.exception
    assert [h.value for h in at.subheader] == ["logistic_regression"]
    table = at.dataframe[0].value
    assert len(table) == 2_000
    assert list(table["Priority"]) == list(range(1, 2_001))
    assert table["Safety flag"].iloc[0]  # the override puts flagged members first
    assert not table["Reason"].str.startswith("critical indicator").any()
    assert not [b for b in at.button if b.label == "Run"]
    assert at.multiselect(key="risk_scoring_models").disabled

    next(b for b in at.button if b.label == "Reset").click().run()
    assert not at.dataframe
    assert [b for b in at.button if b.label == "Run"]
