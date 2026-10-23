import pickle

import numpy as np
import pandas as pd
import pytest

from mh_risk_outreach.config import DEFAULT_CSV_PATH, MEMBER_ID, TARGET
from mh_risk_outreach.model_inputs import MODEL_FEATURES
from mh_risk_outreach.prepare import (
    BINARY_COLUMNS,
    ONE_HOT_CATEGORIES,
    ORDINAL_CATEGORIES,
    PREPARED_COLUMN_COUNT,
    SCALED_RANGES,
    InvalidInputError,
    build_preparation_pipeline,
    clean_inputs,
)


@pytest.fixture(scope="module")
def survey() -> pd.DataFrame:
    """The committed Kaggle CSV with a member ID, as the database stores it."""
    frame = pd.read_csv(DEFAULT_CSV_PATH)
    frame.insert(0, MEMBER_ID, range(1, len(frame) + 1))
    return frame


@pytest.fixture(scope="module")
def train(survey: pd.DataFrame) -> pd.DataFrame:
    return survey.iloc[:8000]


@pytest.fixture(scope="module")
def fitted(train: pd.DataFrame):
    return build_preparation_pipeline().fit(train)


def test_column_groups_cover_every_model_input_once() -> None:
    groups = [
        *SCALED_RANGES,
        *BINARY_COLUMNS,
        *ORDINAL_CATEGORIES,
        *ONE_HOT_CATEGORIES,
    ]

    assert len(groups) == len(set(groups))
    assert set(groups) == set(MODEL_FEATURES)
    assert PREPARED_COLUMN_COUNT == 44


def test_real_dataset_prepares_to_44_numeric_columns(
    fitted, survey: pd.DataFrame
) -> None:
    prepared = fitted.transform(survey)

    assert prepared.shape == (len(survey), 44)
    assert prepared.notna().all().all()
    assert all(np.issubdtype(dtype, np.number) for dtype in prepared.dtypes)
    assert {TARGET, MEMBER_ID}.isdisjoint(prepared.columns)
    assert prepared.index.equals(survey.index)


def test_output_columns_are_fixed_for_any_batch(fitted, survey: pd.DataFrame) -> None:
    expected = list(fitted.get_feature_names_out())
    no_widowed = survey[survey["Marital_Status"] != "Widowed"].head(50)

    for batch in (survey.iloc[[9000]], no_widowed):
        prepared = fitted.transform(batch)
        assert list(prepared.columns) == expected
        assert prepared.shape[1] == 44

    assert (fitted.transform(no_widowed)["Marital_Status_Widowed"] == 0).all()


def test_text_mappings(fitted, survey: pd.DataFrame) -> None:
    prepared = fitted.transform(survey)

    for column, categories in ORDINAL_CATEGORIES.items():
        for code, category in enumerate(categories):
            rows = survey[column] == category
            assert (prepared.loc[rows, column] == code).all(), (column, category)

    smokers = survey["Smoking"] == "Current"
    assert (prepared.loc[smokers, "Smoking_Current"] == 1).all()
    assert (prepared.loc[smokers, ["Smoking_Never", "Smoking_Former"]] == 0).all().all()


def test_scaler_learns_from_training_rows_only(
    train: pd.DataFrame, survey: pd.DataFrame
) -> None:
    pipeline = build_preparation_pipeline().fit(train)
    scaler = pipeline.named_steps["preprocess"].named_transformers_["scaled"][-1]

    ages = list(SCALED_RANGES).index("Age")
    assert scaler.mean_[ages] == pytest.approx(train["Age"].mean())
    assert scaler.mean_[ages] != pytest.approx(survey["Age"].mean(), abs=1e-9)

    # Training rows are centred; held-out rows are not forced to be.
    assert pipeline.transform(train)["Age"].mean() == pytest.approx(0, abs=1e-9)


def test_cleaning_normalizes_text_and_keeps_input_unchanged(
    survey: pd.DataFrame,
) -> None:
    messy = survey.head(3).copy()
    messy["Discuss_Mental_Health"] = messy["Discuss_Mental_Health"].astype(object)
    messy["Marital_Status"] = messy["Marital_Status"].astype(object)
    messy["Age"] = messy["Age"].astype(object)
    messy.loc[0, "Discuss_Mental_Health"] = "  yes EASILY "
    messy.loc[1, "Marital_Status"] = "MARRIED"
    messy.loc[2, "Age"] = " 40 "
    original = messy.copy(deep=True)

    cleaned = clean_inputs(messy)

    assert cleaned.loc[0, "Discuss_Mental_Health"] == "Yes easily"
    assert cleaned.loc[1, "Marital_Status"] == "Married"
    assert cleaned.loc[2, "Age"] == 40.0
    assert tuple(cleaned.columns) == MODEL_FEATURES
    pd.testing.assert_frame_equal(messy, original)


def test_blanks_are_filled_from_training_values(
    fitted, train: pd.DataFrame, survey: pd.DataFrame
) -> None:
    blank = survey.head(4).copy()
    for column in ("Sleep_Hours_Night", "Trauma_History", "Income_Level", "Smoking"):
        blank[column] = blank[column].astype(object)
    blank.loc[0, "Sleep_Hours_Night"] = np.nan
    blank.loc[1, "Trauma_History"] = ""
    blank.loc[2, "Income_Level"] = "   "
    blank.loc[3, "Smoking"] = None

    prepared = fitted.transform(blank)
    median_row = train.head(1).copy()
    median_row["Sleep_Hours_Night"] = train["Sleep_Hours_Night"].median()

    assert prepared.notna().all().all()
    assert prepared.loc[0, "Sleep_Hours_Night"] == pytest.approx(
        fitted.transform(median_row)["Sleep_Hours_Night"].iloc[0]
    )
    assert prepared.loc[1, "Trauma_History"] == train["Trauma_History"].mode()[0]
    income_code = ORDINAL_CATEGORIES["Income_Level"].index(
        train["Income_Level"].mode()[0]
    )
    assert prepared.loc[2, "Income_Level"] == income_code
    assert prepared.loc[3, f"Smoking_{train['Smoking'].mode()[0]}"] == 1


def test_unknown_unordered_category_warns_and_encodes_as_zeros(
    fitted, survey: pd.DataFrame
) -> None:
    batch = survey.head(2).copy()
    batch["Employment_Status"] = batch["Employment_Status"].astype(object)
    batch.loc[0, "Employment_Status"] = "Retired"

    with pytest.warns(UserWarning, match="Retired"):
        prepared = fitted.transform(batch)

    employment = [c for c in prepared.columns if c.startswith("Employment_Status_")]
    assert (prepared.loc[0, employment] == 0).all()
    assert prepared.loc[1, employment].sum() == 1


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("Age", 12, "below 18"),
        ("Feeling_Sad_Down", 11, "above 10"),
        ("Panic_Attacks", 7, "above 4"),
        ("Sleep_Hours_Night", "eight", "not a number"),
        ("Suicidal_Thoughts", 2, "not 0 or 1"),
        ("Income_Level", "Very High", "unknown category"),
    ],
)
def test_invalid_values_are_rejected_with_row_and_reason(
    survey: pd.DataFrame, column: str, value: object, reason: str
) -> None:
    batch = survey.head(3).copy()
    batch[column] = batch[column].astype(object)
    batch.loc[1, column] = value

    with pytest.raises(InvalidInputError, match=reason) as error:
        clean_inputs(batch)

    assert error.value.problems == [(1, column, value, reason)]


def test_every_invalid_value_is_reported(survey: pd.DataFrame) -> None:
    batch = survey.head(20).copy()
    batch["Age"] = 5

    with pytest.raises(InvalidInputError, match="20 invalid.*and 10 more") as error:
        clean_inputs(batch)

    assert len(error.value.problems) == 20


def test_missing_input_column_raises(survey: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Age"):
        clean_inputs(survey.drop(columns="Age"))


def test_saved_pipeline_scores_identically(fitted, survey: pd.DataFrame) -> None:
    held_out = survey.iloc[8000:]
    restored = pickle.loads(pickle.dumps(fitted))

    pd.testing.assert_frame_equal(
        restored.transform(held_out), fitted.transform(held_out)
    )
