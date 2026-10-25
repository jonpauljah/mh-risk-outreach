"""Carve the stored survey into repeatable training and evaluation sets.

The evaluation set is held back so a later stage can check the model on rows it
never learned from. Two rules make that check honest:

* **Split before fitting.** This module only partitions rows; it never fits a
  scaler, imputer, or model. ``prepare.build_preparation_pipeline`` and the
  model (issue #15) are fitted on the training rows only, so no value learned
  from the evaluation rows (a median, an average, a weight) leaks into training.
* **Stratify on the target.** The label is ~92% positive, so a plain random
  split could leave the evaluation set with very few negatives. Stratifying
  keeps each split's positive rate close to the whole dataset's.

The split is repeatable: the same ``random_state`` always produces the same
rows in each part, and ``assert_disjoint`` verifies the two parts share no
member. See docs/ml-spec.md, "Training and evaluation datasets (issue #14)".
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    DEFAULT_TABLE,
    MEMBER_ID,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from mh_risk_outreach.db.repository import load_from_db
from mh_risk_outreach.model_inputs import select_model_inputs


class DatasetSplit(NamedTuple):
    """The two disjoint parts of the survey.

    Each frame holds whole rows (``Member_ID`` + inputs + target), so the
    partition can be verified by member and the target stays available for
    training. Separate the inputs from the target with ``split_xy``.
    """

    train: pd.DataFrame
    evaluation: pd.DataFrame


def split_dataset(
    records: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Split ``records`` into training and evaluation rows, stratified by target.

    ``test_size`` is the share held back for evaluation. The split is stratified
    on ``TARGET`` so both parts keep the dataset's positive rate, and fixed by
    ``random_state`` so repeat runs produce the same rows. Row values and the
    original index are preserved; ``records`` is not modified.
    """
    train, evaluation = train_test_split(
        records,
        test_size=test_size,
        stratify=records[TARGET],
        random_state=random_state,
    )
    return DatasetSplit(train=train, evaluation=evaluation)


def split_xy(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 35 model inputs (``X``) from the target (``y``).

    ``X`` is ``select_model_inputs(records)`` (issue #13); ``y`` is a copy of the
    ``TARGET`` column with the same index. The target is never passed into input
    selection, so it cannot leak in as a predictor.
    """
    x = select_model_inputs(records)
    y = records[TARGET].copy()
    return x, y


def label_balance(frame: pd.DataFrame) -> pd.Series:
    """Count each target value and the positive rate for one frame.

    Returns a Series indexed by ``negatives``, ``positives``, ``total``, and
    ``positive_rate`` so the counts can be compared across splits.
    """
    positives = int(frame[TARGET].sum())
    total = len(frame)
    negatives = total - positives
    return pd.Series(
        {
            "negatives": negatives,
            "positives": positives,
            "total": total,
            "positive_rate": positives / total if total else float("nan"),
        }
    )


def assert_disjoint(split: DatasetSplit) -> None:
    """Raise ``ValueError`` unless the two parts share no member and lose none.

    Members are compared by ``Member_ID``. The check fails if any member appears
    in both parts, or if the parts together do not add up to their inputs (a
    dropped or duplicated row), which would make the evaluation dishonest.
    """
    train_ids = set(split.train[MEMBER_ID])
    eval_ids = set(split.evaluation[MEMBER_ID])
    overlap = train_ids & eval_ids
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(
            f"{len(overlap)} member(s) are in both the training and evaluation "
            f"sets, e.g. {sample}"
        )
    combined = len(split.train) + len(split.evaluation)
    distinct = len(train_ids | eval_ids)
    if combined != distinct:
        raise ValueError(
            f"the split has {combined} rows but {distinct} distinct members; "
            "a row was dropped or duplicated"
        )


def describe_split(split: DatasetSplit) -> str:
    """Build a readable report of counts, positive rate, and member overlap."""
    train_ids = set(split.train[MEMBER_ID])
    eval_ids = set(split.evaluation[MEMBER_ID])
    overlap = len(train_ids & eval_ids)
    total = len(split.train) + len(split.evaluation)

    def line(name: str, frame: pd.DataFrame) -> str:
        balance = label_balance(frame)
        share = len(frame) / total if total else float("nan")
        return (
            f"  {name:<11} {len(frame):>6,} rows ({share:5.1%})  "
            f"positives {int(balance['positives']):>6,} "
            f"({balance['positive_rate']:.1%})  "
            f"negatives {int(balance['negatives']):>5,}"
        )

    return "\n".join(
        [
            f"[Segment 2] Train/evaluation split of {total:,} rows:",
            line("train", split.train),
            line("evaluation", split.evaluation),
            f"  member overlap: {overlap}",
        ]
    )


def write_splits(split: DatasetSplit, out_dir: Path | str) -> tuple[Path, Path]:
    """Write the two parts to ``train.parquet`` and ``evaluation.parquet``.

    Off by default (see ``main``); intended for local inspection, not for
    committing. Returns the two written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.parquet"
    eval_path = out / "evaluation.parquet"
    split.train.to_parquet(train_path, index=False)
    split.evaluation.to_parquet(eval_path, index=False)
    return train_path, eval_path


def main(
    db_path: str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    write: bool = False,
    out_dir: Path | str = "data",
) -> DatasetSplit:
    """Load the stored records, split them, verify separation, and report.

    Reads from SQLite (the ETL output), splits with the fixed seed, fails if the
    parts overlap, and prints the counts and positive rates. Writing the parts
    to parquet is opt-in because the files are large and not meant to be
    committed.
    """
    records = load_from_db(db_path, table)
    split = split_dataset(records)
    assert_disjoint(split)
    print(describe_split(split))
    if write:
        train_path, eval_path = write_splits(split, out_dir)
        print(f"  wrote {train_path} and {eval_path}")
    return split


if __name__ == "__main__":
    main()
