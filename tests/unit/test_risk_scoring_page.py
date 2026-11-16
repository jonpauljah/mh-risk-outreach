"""The Risk scoring Streamlit page (issue #28), driven headlessly with AppTest.

The page reads its database and artifact paths from module constants, so each
test points them at a temporary folder; the real ``patients.db`` and
``artifacts/`` are never touched.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mh_risk_outreach import config, scoring
from mh_risk_outreach.db import scores as scores_db
from mh_risk_outreach.etl.pipeline import run_etl
from mh_risk_outreach.scoring import CRITICAL_REASON

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
    db, artifacts = paths
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
    stored = scores_db.load_member_scores("logistic_regression", db_path=db)
    stored = stored.sort_values("rank_priority").reset_index(drop=True)
    assert list(table["Member ID"]) == list(stored["Member_ID"])
    assert list(table["Priority"]) == list(stored["rank_priority"])
    assert list(table["Risk score"]) == pytest.approx(list(stored["risk_score"]))
    assert list(table["Risk band"]) == list(stored["band_level_priority"])
    assert (
        list(table["Safety flag"])
        == stored["reasoning"].str.startswith(f"{CRITICAL_REASON}; ").tolist()
    )
    assert (
        list(table["Reason"])
        == stored["reasoning"].str.removeprefix(f"{CRITICAL_REASON}; ").tolist()
    )
    metadata = Path(artifacts) / "logistic_regression" / "metadata.json"
    model_version = json.loads(metadata.read_text())["model_version"]
    assert stored["model_version"].eq(model_version).all()
    assert f"Model version `{model_version}`" in " ".join(
        caption.value for caption in at.caption
    )
    assert not [b for b in at.button if b.label == "Run"]
    assert at.multiselect(key="risk_scoring_models").disabled

    next(b for b in at.button if b.label == "Reset").click().run()
    assert not at.dataframe
    after_reset = scores_db.load_member_scores("logistic_regression", db_path=db)
    pd.testing.assert_frame_equal(stored, after_reset)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE logistic_regression_scores SET reasoning = ? WHERE rank_priority = 1",
            ("edited after reset",),
        )
    next(b for b in at.button if b.label == "Run").click().run()
    assert not at.exception
    pd.testing.assert_frame_equal(
        stored, scores_db.load_member_scores("logistic_regression", db_path=db)
    )


def test_page_displays_reason_read_back_from_database(paths, monkeypatch) -> None:
    db, _ = paths
    run_etl(db_path=db)
    at = _page().run()
    at.multiselect(key="risk_scoring_models").set_value(["logistic_regression"]).run()
    at.button(key="train_logistic_regression").click().run()
    original_save = scores_db.save_member_scores

    def save_then_change_reason(worklist, model_name, db_path):
        original_save(worklist, model_name, db_path=db_path)
        persisted = scores_db.load_member_scores(model_name, db_path=db_path)
        persisted.loc[0, "reasoning"] = "read back from SQLite"
        original_save(persisted, model_name, db_path=db_path)

    monkeypatch.setattr(scores_db, "save_member_scores", save_then_change_reason)
    next(b for b in at.button if b.label == "Run").click().run()
    assert not at.exception
    assert at.dataframe[0].value["Reason"].iloc[0] == "read back from SQLite"
    persisted = scores_db.load_member_scores("logistic_regression", db_path=db)
    assert persisted["reasoning"].iloc[0] == "read back from SQLite"
