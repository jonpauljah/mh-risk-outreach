import pandas as pd
import pytest

from mh_risk_outreach.config import FEATURES, MEMBER_ID, TARGET
from mh_risk_outreach.model_inputs import MODEL_FEATURES, select_model_inputs


def test_model_feature_policy() -> None:
    """The documented 15 exclusions leave 35 unique source inputs."""
    excluded = {
        "Previously_Diagnosed",
        "Ever_Sought_Treatment",
        "On_Therapy_Now",
        "On_Medication",
        "Gender",
        "Country",
        "Education",
        "Work_Hours_Per_Week",
        "Remote_Work",
        "Company_Mental_Health_Support",
        "Caffeine_Drinks_Day",
        "Screen_Time_Hours_Day",
        "Social_Media_Hours_Day",
        "Hobby_Time_Hours_Week",
        "Diet_Quality",
    }

    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)) == 35
    assert set(MODEL_FEATURES) == set(FEATURES) - excluded
    assert {TARGET, MEMBER_ID}.isdisjoint(MODEL_FEATURES)


@pytest.fixture
def records() -> pd.DataFrame:
    """Use shuffled source columns and a non-default index."""
    frame = pd.DataFrame({name: [1, 2] for name in reversed(FEATURES)}, index=[7, 3])
    frame["Age"] = [25, 42]
    frame["Smoking"] = ["Never", "Former"]
    frame[TARGET] = [1, 0]
    frame[MEMBER_ID] = [101, 102]
    return frame


@pytest.mark.parametrize("include_target", [True, False])
def test_select_inputs_for_training_and_scoring(
    records: pd.DataFrame, include_target: bool
) -> None:
    source = records if include_target else records.drop(columns=TARGET)
    original = source.copy(deep=True)

    selected = select_model_inputs(source)

    assert tuple(selected.columns) == MODEL_FEATURES
    assert selected.index.tolist() == [7, 3]
    assert selected["Age"].tolist() == [25, 42]
    assert selected["Smoking"].tolist() == ["Never", "Former"]
    pd.testing.assert_frame_equal(source, original)

    selected.loc[7, "Age"] = 99
    assert source.loc[7, "Age"] == 25


def test_missing_model_input_raises(records: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Age"):
        select_model_inputs(records.drop(columns="Age"))
