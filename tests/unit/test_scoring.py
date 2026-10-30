"""Scoring, SHAP reasons, risk bands, and priority order (issue #16).

These run against a stratified subsample of the survey (as the DB stores it, with
a ``Member_ID``) and a freshly trained pipeline, so they need no pre-built
artifact. They assert the behaviours docs/ml-spec.md lists for the scoring stage:
valid scores, the critical-indicator override, monotonic reasons, reproducible
bands, and a stable worklist order.
"""

import numpy as np
import pandas as pd
import pytest

from mh_risk_outreach.config import DEFAULT_CSV_PATH, MEMBER_ID
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.scores import (
    load_member_scores,
    save_member_scores,
    scores_status,
)
from mh_risk_outreach.models.logistic_regression import LogisticRegressionRiskModel
from mh_risk_outreach.models.random_forest import RandomForestRiskModel
from mh_risk_outreach.models.train import train_and_evaluate
from mh_risk_outreach.scoring import (
    BAND_LABELS,
    CRITICAL_INDICATORS,
    TOP_BAND,
    assign_bands,
    score_members,
)

SYMPTOM_COLUMNS = (
    "Feeling_Sad_Down",
    "Loss_Of_Interest",
    "Sleep_Trouble",
    "Fatigue",
    "Poor_Appetite_Or_Overeating",
    "Feeling_Worthless",
    "Concentration_Difficulty",
    "Anxious_Nervous",
    "Mood_Swings",
    "Irritability",
    "Obsessive_Thoughts",
    "Compulsive_Behavior",
    "Panic_Attacks",
)


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    frame = pd.read_csv(DEFAULT_CSV_PATH)
    frame.insert(0, MEMBER_ID, range(1, len(frame) + 1))
    sample = frame.groupby("Has_Mental_Health_Issue", group_keys=False).sample(
        frac=0.15, random_state=42
    )
    return sample.reset_index(drop=True)


@pytest.fixture(scope="module")
def background(records: pd.DataFrame) -> pd.DataFrame:
    """The training rows train_and_evaluate fits on: the SHAP reference set."""
    return split_dataset(records).train


@pytest.fixture(scope="module")
def lr_pipeline(records: pd.DataFrame):
    return train_and_evaluate(LogisticRegressionRiskModel(), records).pipeline


@pytest.fixture(scope="module")
def worklist(records: pd.DataFrame, lr_pipeline, background) -> pd.DataFrame:
    return score_members(
        records, lr_pipeline, model_version="test_v1", background=background
    )


def test_scores_are_0_to_100_one_row_per_member(records, worklist) -> None:
    assert len(worklist) == len(records)
    assert set(worklist[MEMBER_ID]) == set(records[MEMBER_ID])
    assert worklist["risk_score"].between(0, 100).all()


def test_rank_is_a_dense_1_to_n_order(worklist) -> None:
    ranks = sorted(worklist["rank_priority"])
    assert ranks == list(range(1, len(worklist) + 1))


def test_critical_indicators_take_the_top_priority(records, worklist) -> None:
    """Every self-harm / suicidal member is Very High and ranked above the rest."""
    critical = np.zeros(len(records), dtype=bool)
    for column in CRITICAL_INDICATORS:
        critical |= records[column].to_numpy() == 1
    critical_ids = set(records.loc[critical, MEMBER_ID])
    if not critical_ids:  # subsample happened to have none
        pytest.skip("no critical members in the subsample")

    flagged = worklist[worklist[MEMBER_ID].isin(critical_ids)]
    assert (flagged["band_level_priority"] == TOP_BAND).all()
    # They occupy the first len(critical_ids) ranks, above every other member.
    assert flagged["rank_priority"].max() == len(critical_ids)


def test_a_low_scoring_critical_member_is_still_top_priority(
    records, lr_pipeline, background
) -> None:
    """The override wins even when the model score is low."""
    calm = records.sort_values("Has_Mental_Health_Issue").iloc[0].copy()
    for column in SYMPTOM_COLUMNS:
        calm[column] = 0
    calm["Self_Harm_Thoughts"] = 1
    calm["Suicidal_Thoughts"] = 0
    frame = pd.concat([records, pd.DataFrame([calm])], ignore_index=True)
    frame[MEMBER_ID] = range(1, len(frame) + 1)

    result = score_members(
        frame, lr_pipeline, model_version="test_v1", background=background
    )
    flagged = result[result[MEMBER_ID] == frame[MEMBER_ID].iloc[-1]].iloc[0]
    assert flagged["band_level_priority"] == TOP_BAND
    assert flagged["rank_priority"] <= (result["band_level_priority"] == TOP_BAND).sum()
    assert "critical indicator" in flagged["reasoning"]


def test_maxed_symptoms_score_higher_than_zeroed(
    records, lr_pipeline, background
) -> None:
    base = records.iloc[0].copy()
    high, low = base.copy(), base.copy()
    for column in SYMPTOM_COLUMNS:
        high[column] = 4 if column == "Panic_Attacks" else 10
        low[column] = 0
    frame = pd.DataFrame([high, low])
    frame[MEMBER_ID] = [1, 2]

    scored = score_members(
        frame, lr_pipeline, model_version="test_v1", background=background
    ).set_index(MEMBER_ID)
    assert scored.loc[1, "risk_score"] > scored.loc[2, "risk_score"]


def test_reasoning_is_present_and_readable(worklist) -> None:
    assert (worklist["reasoning"].str.len() > 0).all()
    assert worklist["reasoning"].str.contains("critical indicator").any()


def test_assign_bands_is_reproducible_and_ordered() -> None:
    scores = np.linspace(40, 100, 500)
    first = assign_bands(scores)
    second = assign_bands(scores)
    assert list(first) == list(second)
    assert set(first) <= set(BAND_LABELS)
    # Higher scores never land in a lower band than lower scores.
    order = {label: index for index, label in enumerate(BAND_LABELS)}
    band_rank = np.array([order[band] for band in first])
    assert np.all(np.diff(band_rank) >= 0)


def test_scoring_is_deterministic(records, lr_pipeline, background) -> None:
    a = score_members(
        records, lr_pipeline, model_version="test_v1", background=background
    )
    b = score_members(
        records, lr_pipeline, model_version="test_v1", background=background
    )
    pd.testing.assert_frame_equal(a, b)


def test_reasons_do_not_depend_on_the_batch(
    records, lr_pipeline, background, worklist
) -> None:
    """A member scored alone gets the same reason as when scored with everyone."""
    for member in records[MEMBER_ID].iloc[:5]:
        alone = score_members(
            records[records[MEMBER_ID] == member],
            lr_pipeline,
            model_version="test_v1",
            background=background,
        )
        in_batch = worklist[worklist[MEMBER_ID] == member]
        assert alone["reasoning"].iloc[0] == in_batch["reasoning"].iloc[0]
        assert alone["reasoning"].iloc[0] != "no strong risk factors identified"


def test_chunked_scoring_reports_progress_and_matches(
    records, lr_pipeline, background, worklist
) -> None:
    """Explaining in chunks reports rising progress to 1.0 and changes nothing."""
    seen: list[float] = []
    chunked = score_members(
        records,
        lr_pipeline,
        model_version="test_v1",
        background=background,
        progress=seen.append,
        chunk_size=7,
    )
    pd.testing.assert_frame_equal(chunked, worklist)
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert len(seen) == -(-len(records) // 7)  # one call per chunk


def test_random_forest_also_scores_and_explains(records, background) -> None:
    """A tree model drops into the same scoring path (TreeExplainer reasons)."""
    pipeline = train_and_evaluate(RandomForestRiskModel(), records).pipeline
    result = score_members(
        records, pipeline, model_version="rf_test", background=background
    )
    assert result["risk_score"].between(0, 100).all()
    assert (result["reasoning"].str.len() > 0).all()


def test_scores_table_round_trips(worklist, tmp_path) -> None:
    db = str(tmp_path / "scores.db")
    save_member_scores(worklist, "logistic_regression", db_path=db)
    stored = load_member_scores("logistic_regression", db_path=db)

    assert len(stored) == len(worklist)
    assert list(stored["rank_priority"]) == list(range(1, len(stored) + 1))
    assert set(stored.columns) >= {
        MEMBER_ID,
        "model_version",
        "risk_score",
        "band_level_priority",
        "rank_priority",
        "reasoning",
    }


def test_saving_scores_rejects_a_missing_column(worklist, tmp_path) -> None:
    db = str(tmp_path / "scores.db")
    with pytest.raises(KeyError):
        save_member_scores(
            worklist.drop(columns=["reasoning"]), "logistic_regression", db_path=db
        )


def test_scores_status_reports_missing_then_present(worklist, tmp_path) -> None:
    """The next-stage UI uses this to show 'run scoring first' without crashing."""
    db = str(tmp_path / "scores.db")
    assert scores_status("logistic_regression", db_path=db) == (False, 0)

    save_member_scores(worklist, "logistic_regression", db_path=db)
    exists, count = scores_status("logistic_regression", db_path=db)
    assert exists and count == len(worklist)
    # A model that was never scored still reports missing, not an error.
    assert scores_status("random_forest", db_path=db) == (False, 0)
