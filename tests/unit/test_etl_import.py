import sqlite3

import pandas as pd
import pytest

from mh_risk_outreach.config import DEFAULT_CSV_PATH
from mh_risk_outreach.db.repository import load_from_db, save_records
from mh_risk_outreach.etl import pipeline
from mh_risk_outreach.etl.pipeline import run_etl
from mh_risk_outreach.etl.transform import transform


def test_initial_import_assigns_sequential_member_ids(tmp_path) -> None:
    db_path = str(tmp_path / "patients.db")

    loaded = run_etl(db_path=db_path)

    assert loaded is not None
    stored = load_from_db(db_path)
    assert len(stored) == len(loaded)
    assert stored["Member_ID"].tolist() == list(range(1, len(stored) + 1))


@pytest.mark.parametrize("csv_path", ["missing.csv", "invalid.csv"])
def test_repeat_import_skips_before_reading_csv(tmp_path, csv_path, capsys) -> None:
    db_path = str(tmp_path / "patients.db")
    if csv_path == "invalid.csv":
        (tmp_path / csv_path).write_text('"unterminated', encoding="utf-8")
    seed = transform(pd.read_csv(DEFAULT_CSV_PATH).head(2))
    assert save_records(seed, db_path)

    assert run_etl(str(tmp_path / csv_path), db_path) is None

    assert "Skipped import" in capsys.readouterr().out
    pd.testing.assert_frame_equal(load_from_db(db_path), seed)


def test_pipeline_honors_repository_skip_after_concurrent_import(
    tmp_path, monkeypatch, capsys
) -> None:
    db_path = str(tmp_path / "patients.db")
    winner = transform(pd.read_csv(DEFAULT_CSV_PATH).head(2))
    winner.loc[0, "Age"] = 99

    def extract_after_concurrent_import(_csv_path):
        assert save_records(winner, db_path)
        return pd.read_csv(DEFAULT_CSV_PATH).head(2)

    monkeypatch.setattr(pipeline, "extract", extract_after_concurrent_import)

    assert run_etl(db_path=db_path) is None

    assert "Skipped import" in capsys.readouterr().out
    pd.testing.assert_frame_equal(load_from_db(db_path), winner)


def test_direct_repository_skip_preserves_patient_edits_and_history(tmp_path) -> None:
    db_path = str(tmp_path / "patients.db")
    seed = transform(pd.read_csv(DEFAULT_CSV_PATH).head(2))
    assert save_records(seed, db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE patients SET Age = 99 WHERE Member_ID = 1")
        conn.execute("CREATE TABLE outreach_history (Member_ID INTEGER, note TEXT)")
        conn.execute("INSERT INTO outreach_history VALUES (1, 'contacted')")
        conn.execute("CREATE TABLE member_scores (Member_ID INTEGER, score REAL)")
        conn.execute("INSERT INTO member_scores VALUES (1, 0.75)")

    assert not save_records(seed, db_path)

    assert load_from_db(db_path).loc[0, "Age"] == 99
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT note FROM outreach_history").fetchone() == (
            "contacted",
        )
        assert conn.execute("SELECT score FROM member_scores").fetchone() == (0.75,)


def test_failed_initial_load_rolls_back_all_rows(tmp_path) -> None:
    db_path = str(tmp_path / "patients.db")
    records = transform(pd.read_csv(DEFAULT_CSV_PATH).head(3))
    records.loc[1, "Self_Harm_Thoughts"] = 2

    with pytest.raises(sqlite3.IntegrityError):
        save_records(records, db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM patients").fetchone() == (0,)


def test_pipeline_failed_initial_load_reports_no_success(tmp_path, capsys) -> None:
    db_path = str(tmp_path / "patients.db")
    source = pd.read_csv(DEFAULT_CSV_PATH).head(3)
    source.loc[1, "Self_Harm_Thoughts"] = 2
    csv_path = tmp_path / "invalid-flag.csv"
    source.to_csv(csv_path, index=False)

    with pytest.raises(sqlite3.IntegrityError):
        run_etl(str(csv_path), db_path)

    assert "Loaded" not in capsys.readouterr().out
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM patients").fetchone() == (0,)
