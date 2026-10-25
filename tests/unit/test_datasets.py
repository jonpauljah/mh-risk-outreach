import pandas as pd
import pytest

from mh_risk_outreach.config import (
    DEFAULT_CSV_PATH,
    MEMBER_ID,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from mh_risk_outreach.datasets import (
    DatasetSplit,
    assert_disjoint,
    label_balance,
    split_dataset,
    split_xy,
)
from mh_risk_outreach.model_inputs import MODEL_FEATURES


@pytest.fixture(scope="module")
def survey() -> pd.DataFrame:
    """The committed Kaggle CSV with a member ID, as the database stores it."""
    frame = pd.read_csv(DEFAULT_CSV_PATH)
    frame.insert(0, MEMBER_ID, range(1, len(frame) + 1))
    return frame


def test_split_is_reproducible(survey: pd.DataFrame) -> None:
    """The same seed selects the same members for each part on every run."""
    first = split_dataset(survey)
    second = split_dataset(survey)

    assert set(first.train[MEMBER_ID]) == set(second.train[MEMBER_ID])
    assert set(first.evaluation[MEMBER_ID]) == set(second.evaluation[MEMBER_ID])


def test_a_different_seed_changes_membership(survey: pd.DataFrame) -> None:
    default = split_dataset(survey)
    other = split_dataset(survey, random_state=RANDOM_STATE + 1)

    assert set(default.evaluation[MEMBER_ID]) != set(other.evaluation[MEMBER_ID])


def test_parts_are_disjoint_and_complete(survey: pd.DataFrame) -> None:
    """Every member lands in exactly one part, and none is lost or duplicated."""
    split = split_dataset(survey)

    train_ids = set(split.train[MEMBER_ID])
    eval_ids = set(split.evaluation[MEMBER_ID])
    assert train_ids.isdisjoint(eval_ids)
    assert train_ids | eval_ids == set(survey[MEMBER_ID])
    assert len(split.train) + len(split.evaluation) == len(survey)
    assert_disjoint(split)  # the shipped check agrees


def test_evaluation_share_matches_test_size(survey: pd.DataFrame) -> None:
    split = split_dataset(survey)

    assert len(split.evaluation) == round(len(survey) * TEST_SIZE)


def test_stratification_preserves_label_balance(survey: pd.DataFrame) -> None:
    """Both parts keep the whole dataset's positive rate within rounding."""
    split = split_dataset(survey)
    overall = label_balance(survey)["positive_rate"]

    assert label_balance(split.train)["positive_rate"] == pytest.approx(
        overall, abs=0.005
    )
    assert label_balance(split.evaluation)["positive_rate"] == pytest.approx(
        overall, abs=0.005
    )


def test_split_xy_separates_inputs_from_target(survey: pd.DataFrame) -> None:
    x, y = split_xy(survey)

    assert tuple(x.columns) == MODEL_FEATURES
    assert {TARGET, MEMBER_ID}.isdisjoint(x.columns)
    assert y.name == TARGET
    assert x.index.equals(y.index)
    assert len(x) == len(survey)


def test_assert_disjoint_rejects_an_overlapping_split(survey: pd.DataFrame) -> None:
    overlapping = DatasetSplit(train=survey, evaluation=survey.iloc[:5])

    with pytest.raises(ValueError, match="in both"):
        assert_disjoint(overlapping)


def test_assert_disjoint_rejects_a_duplicated_row(survey: pd.DataFrame) -> None:
    """A member repeated within a part makes the counts disagree and is rejected."""
    split = split_dataset(survey)
    duplicated = DatasetSplit(
        train=split.train,
        evaluation=pd.concat([split.evaluation, split.evaluation.iloc[:1]]),
    )

    with pytest.raises(ValueError, match="dropped or duplicated"):
        assert_disjoint(duplicated)
