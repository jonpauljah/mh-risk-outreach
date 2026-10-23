"""Clean the 35 model inputs and turn them into the 44 numbers the model reads.

Two steps, used together for training and for scoring:

* ``clean_inputs`` applies fixed rules that learn nothing from the data: select
  the inputs, tidy text, convert numbers, and reject values outside the
  documented ones with ``InvalidInputError``.
* ``build_preprocessor`` returns an unfitted scikit-learn step that fills
  blanks, scales numbers, and encodes text. It learns fill values, averages,
  and spreads when fitted, so it must be fitted on the training rows only.

``build_preparation_pipeline`` chains both. Training code fits it (or a
pipeline that ends in the model) on the training split; scoring reuses the
fitted object unchanged, so both always produce the same 44 columns in order.
Rules and their reasons are documented in docs/ml-spec.md.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from mh_risk_outreach.model_inputs import MODEL_FEATURES, select_model_inputs

# ---- Number scales (23): blank -> training median, then StandardScaler ----
# Inclusive (low, high) bounds; None means no bound. Score ranges follow the
# survey scales seen in the Kaggle snapshot and need confirming for new sources.
SCALED_RANGES: dict[str, tuple[float | None, float | None]] = {
    "Feeling_Sad_Down": (0, 10),
    "Loss_Of_Interest": (0, 10),
    "Sleep_Trouble": (0, 10),
    "Fatigue": (0, 10),
    "Poor_Appetite_Or_Overeating": (0, 10),
    "Feeling_Worthless": (0, 10),
    "Concentration_Difficulty": (0, 10),
    "Anxious_Nervous": (0, 10),
    "Mood_Swings": (0, 10),
    "Irritability": (0, 10),
    "Obsessive_Thoughts": (0, 10),
    "Compulsive_Behavior": (0, 10),
    "Panic_Attacks": (0, 4),
    "Social_Support": (1, 10),
    "Feel_Understood": (1, 10),
    "Loneliness": (1, 10),
    "Close_Friends_Count": (0, None),
    "Financial_Stress": (1, 10),
    "Work_Stress_Level": (1, 10),
    "Job_Satisfaction": (1, 10),
    "Work_Life_Balance": (1, 10),
    "Sleep_Hours_Night": (0, 24),
    "Age": (18, 120),  # the model is for adult members
}

# ---- Already 0/1 (5): blank -> most common training value, no scaling ----
BINARY_COLUMNS = (
    "Self_Harm_Thoughts",
    "Suicidal_Thoughts",
    "Trauma_History",
    "Family_History_Mental_Illness",
    "Ever_Bullied_At_Work",
)

# ---- Ordered text (4): mapped to 0, 1, 2, ... in the listed order ----
# Unknown values are rejected because they have no place in the order.
ORDINAL_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Income_Level": ("Low", "Middle", "High"),
    "Exercise_Per_Week": ("Never", "1-2 times", "3-4 times", "5+ times"),
    "Alcohol_Frequency": ("Never", "Rarely", "Weekly", "Daily"),
    "Discuss_Mental_Health": ("Never", "Rarely", "Sometimes", "Yes easily"),
}

# ---- Unordered text (3): one 0/1 column per listed category ----
# Unknown values are kept (with a warning) and encode as all zeros.
ONE_HOT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Employment_Status": (
        "Full-time",
        "Part-time",
        "Self-employed",
        "Student",
        "Unemployed",
    ),
    "Marital_Status": ("Single", "Married", "Divorced", "Widowed"),
    "Smoking": ("Never", "Former", "Current"),
}

PREPARED_COLUMN_COUNT = (
    len(SCALED_RANGES)
    + len(BINARY_COLUMNS)
    + len(ORDINAL_CATEGORIES)
    + sum(len(categories) for categories in ONE_HOT_CATEGORIES.values())
)  # 44

# Errors listed in the exception message; the full list stays on ``.problems``.
_MAX_REPORTED_PROBLEMS = 10


class InvalidInputError(ValueError):
    """Raised when model inputs hold values outside the documented ones.

    ``problems`` lists every ``(row_index, column, value, reason)`` found, so
    callers can report the affected rows without printing whole records.
    """

    def __init__(self, problems: list[tuple[object, str, object, str]]) -> None:
        self.problems = problems
        shown = "; ".join(
            f"row {row} {column}={value!r} ({reason})"
            for row, column, value, reason in problems[:_MAX_REPORTED_PROBLEMS]
        )
        more = len(problems) - _MAX_REPORTED_PROBLEMS
        suffix = f"; and {more} more" if more > 0 else ""
        super().__init__(
            f"{len(problems)} invalid model input value(s): {shown}{suffix}"
        )


def _is_blank(series: pd.Series) -> pd.Series:
    """True for missing cells and for strings that are empty after trimming."""
    text = series.astype("object")
    return text.isna() | text.map(
        lambda value: isinstance(value, str) and not value.strip()
    )


def _clean_numbers(
    series: pd.Series,
    low: float | None,
    high: float | None,
    problems: list[tuple[object, str, object, str]],
) -> pd.Series:
    """Convert to float, keep blanks as NaN, and record non-numbers and out-of-range values."""
    blank = _is_blank(series)
    numbers = pd.to_numeric(series.where(~blank), errors="coerce").astype(float)
    for row in series.index[numbers.isna() & ~blank]:
        problems.append((row, str(series.name), series[row], "not a number"))
    if low is not None:
        for row in series.index[numbers < low]:
            problems.append((row, str(series.name), series[row], f"below {low}"))
    if high is not None:
        for row in series.index[numbers > high]:
            problems.append((row, str(series.name), series[row], f"above {high}"))
    return numbers


def _clean_text(series: pd.Series, categories: tuple[str, ...]) -> pd.Series:
    """Trim text and match known categories regardless of letter case.

    Blanks become NaN. Values that match no category are returned trimmed but
    otherwise unchanged, for the caller to reject or allow.
    """
    canonical = {category.casefold(): category for category in categories}

    def normalize(value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        return canonical.get(trimmed.casefold(), trimmed)

    blank = _is_blank(series)
    return series.astype("object").map(normalize).where(~blank, np.nan)


def clean_inputs(records: pd.DataFrame) -> pd.DataFrame:
    """Select the 35 model inputs and apply the fixed cleaning rules.

    * Blank cells (missing, empty, or only spaces) become NaN for the
      preprocessor to fill.
    * Text is trimmed and matched to its category regardless of case, so
      ``" yes EASILY "`` becomes ``"Yes easily"``.
    * Numbers must parse and fall inside ``SCALED_RANGES``; 0/1 columns must be
      0 or 1; ordered text must be a listed category. Any failure raises
      ``InvalidInputError`` naming every bad row, column, and value.
    * Unordered text outside its listed categories is kept and reported with a
      ``UserWarning``; the encoder turns it into all-zero columns.

    Returns a new frame with the original row index; ``records`` is unchanged.
    The target, member ID, and other columns are ignored.
    """
    inputs = select_model_inputs(records)
    problems: list[tuple[object, str, object, str]] = []
    cleaned: dict[str, pd.Series] = {}

    for column, (low, high) in SCALED_RANGES.items():
        cleaned[column] = _clean_numbers(inputs[column], low, high, problems)

    for column in BINARY_COLUMNS:
        numbers = _clean_numbers(inputs[column], None, None, problems)
        for row in inputs.index[numbers.notna() & ~numbers.isin([0, 1])]:
            problems.append((row, column, inputs.at[row, column], "not 0 or 1"))
        cleaned[column] = numbers

    for column, categories in ORDINAL_CATEGORIES.items():
        text = _clean_text(inputs[column], categories)
        for row in inputs.index[text.notna() & ~text.isin(categories)]:
            problems.append((row, column, inputs.at[row, column], "unknown category"))
        cleaned[column] = text

    for column, categories in ONE_HOT_CATEGORIES.items():
        text = _clean_text(inputs[column], categories)
        unknown = sorted(set(text[text.notna() & ~text.isin(categories)]))
        if unknown:
            warnings.warn(
                f"{column} has categories the model was not built for: {unknown}; "
                "they will be encoded as all zeros",
                UserWarning,
                stacklevel=2,
            )
        cleaned[column] = text

    if problems:
        raise InvalidInputError(problems)
    return pd.DataFrame(cleaned, index=inputs.index)[list(MODEL_FEATURES)]


def build_preprocessor() -> ColumnTransformer:
    """Return the unfitted step that fills blanks, scales, and encodes.

    Fit it on training rows only. Output is a DataFrame with 44 named columns in
    a fixed order: 23 scaled numbers, 5 flags, 4 ordered text columns, then one
    0/1 column per listed unordered category. Blanks are filled with the
    training median (numbers) or the most common training value (flags, text).
    """
    scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    binary = SimpleImputer(strategy="most_frequent")
    ordinal = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(categories=[list(c) for c in ORDINAL_CATEGORIES.values()]),
    )
    one_hot = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(
            categories=[list(c) for c in ONE_HOT_CATEGORIES.values()],
            handle_unknown="ignore",
            sparse_output=False,
        ),
    )
    preprocessor = ColumnTransformer(
        [
            ("scaled", scaled, list(SCALED_RANGES)),
            ("binary", binary, list(BINARY_COLUMNS)),
            ("ordinal", ordinal, list(ORDINAL_CATEGORIES)),
            ("one_hot", one_hot, list(ONE_HOT_CATEGORIES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def _model_feature_names(transformer: object, input_features: object) -> list[str]:
    """Name the cleaning step's output columns (module-level so it pickles)."""
    return list(MODEL_FEATURES)


def build_preparation_pipeline() -> Pipeline:
    """Return the unfitted clean-then-preprocess pipeline.

    Training code fits this, or appends the model to it, on the training split
    and saves the fitted result; scoring loads and reuses it unchanged.
    """
    clean = FunctionTransformer(clean_inputs, feature_names_out=_model_feature_names)
    return Pipeline([("clean", clean), ("preprocess", build_preprocessor())])
