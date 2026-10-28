"""Turn a model's SHAP contributions into short, readable reasons (issue #16).

SHAP tells us, for one member, how much each prepared column pushed their risk
score up or down. This module takes the columns that pushed the score **up** the
most and turns them into a short phrase a case manager can read, such as
"high work stress, low social support, reported trauma history".

One engine covers all three models (see docs/risk-scoring.md):

* ``logistic_regression`` -> ``shap.LinearExplainer`` (its values equal the
  signed-weight "top reasons" already described in docs/ml-spec.md).
* ``random_forest`` / ``xgboost`` -> ``shap.TreeExplainer``.

Everything here reads a fitted pipeline and member rows; it fits nothing.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from mh_risk_outreach.prepare import ONE_HOT_CATEGORIES, ORDINAL_CATEGORIES

# How many drivers to name in one reason string.
DEFAULT_TOP_N = 3

# Human labels for the scaled numbers and ordered-text columns. The reason
# prefixes these with "high"/"low" depending on the member's value.
NUMERIC_LABELS: dict[str, str] = {
    "Feeling_Sad_Down": "sadness",
    "Loss_Of_Interest": "loss of interest",
    "Sleep_Trouble": "sleep trouble",
    "Fatigue": "fatigue",
    "Poor_Appetite_Or_Overeating": "appetite disturbance",
    "Feeling_Worthless": "feelings of worthlessness",
    "Concentration_Difficulty": "concentration difficulty",
    "Anxious_Nervous": "anxiety",
    "Mood_Swings": "mood swings",
    "Irritability": "irritability",
    "Obsessive_Thoughts": "obsessive thoughts",
    "Compulsive_Behavior": "compulsive behavior",
    "Panic_Attacks": "panic attacks",
    "Social_Support": "social support",
    "Feel_Understood": "feeling understood",
    "Loneliness": "loneliness",
    "Close_Friends_Count": "close friends",
    "Financial_Stress": "financial stress",
    "Work_Stress_Level": "work stress",
    "Job_Satisfaction": "job satisfaction",
    "Work_Life_Balance": "work-life balance",
    "Sleep_Hours_Night": "nightly sleep",
    "Age": "age",
    "Income_Level": "income level",
    "Exercise_Per_Week": "exercise",
    "Alcohol_Frequency": "alcohol use",
    "Discuss_Mental_Health": "openness to discuss mental health",
}

# 0/1 flags: named when present (value 1), as "no ..." when absent.
BINARY_LABELS: dict[str, str] = {
    "Self_Harm_Thoughts": "self-harm thoughts",
    "Suicidal_Thoughts": "suicidal thoughts",
    "Trauma_History": "trauma history",
    "Family_History_Mental_Illness": "family history of mental illness",
    "Ever_Bullied_At_Work": "workplace bullying",
}

# One-hot columns are "<source>_<category>"; a phrase is built from the category.
ONE_HOT_PHRASES: dict[str, Callable[[str], str]] = {
    "Employment_Status": lambda category: category.lower(),
    "Marital_Status": lambda category: category.lower(),
    "Smoking": lambda category: f"{category.lower()} smoker",
}

_NO_DRIVERS = "no strong risk factors identified"


def _prepared_frame(pipeline: Pipeline, records: pd.DataFrame) -> pd.DataFrame:
    """Run member rows through the pipeline's preparation step to the 44 columns."""
    return pipeline.named_steps["prepare"].transform(records)


def shap_contributions(model: object, prepared: pd.DataFrame) -> np.ndarray:
    """Return signed contributions toward the positive class, shape (n_rows, 44).

    A positive value means the column pushed that member's risk score up. The
    explainer is chosen from the model: ``LinearExplainer`` for a linear model
    (it exposes ``coef_``), ``TreeExplainer`` for the forest and XGBoost. Tree
    explainers return a per-class array; we keep the positive class.
    """
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, prepared)
    else:
        explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer(prepared).values)
    if values.ndim == 3:  # (rows, columns, classes) -> positive class
        values = values[..., 1]
    return values


def _describe(column: str, value: float) -> str | None:
    """Phrase one prepared column for a member, given their prepared value."""
    if column in NUMERIC_LABELS:
        # Scaled columns are centered on 0; ordered-text columns run 0..n. In
        # both, a higher value than the column's centre is "high".
        level = "high" if value >= 0 else "low"
        if column in ORDINAL_CATEGORIES:  # not scaled: midpoint of 0..n-1
            midpoint = (len(ORDINAL_CATEGORIES[column]) - 1) / 2
            level = "high" if value >= midpoint else "low"
        return f"{level} {NUMERIC_LABELS[column]}"
    if column in BINARY_LABELS:
        return BINARY_LABELS[column] if value >= 0.5 else f"no {BINARY_LABELS[column]}"
    for source, categories in ONE_HOT_CATEGORIES.items():
        prefix = f"{source}_"
        if column.startswith(prefix):
            category = column[len(prefix) :]
            if category in categories:
                return ONE_HOT_PHRASES[source](category)
    return None


def build_reasons(
    pipeline: Pipeline, records: pd.DataFrame, *, top_n: int = DEFAULT_TOP_N
) -> list[str]:
    """Return one readable reason string per row in ``records``.

    For each member the ``top_n`` columns with the largest positive SHAP
    contribution are turned into phrases and joined. If nothing pushed the score
    up, the reason is "no strong risk factors identified".
    """
    prepared = _prepared_frame(pipeline, records)
    contributions = shap_contributions(pipeline.named_steps["model"], prepared)
    columns = list(prepared.columns)
    values = prepared.to_numpy(dtype=float)

    reasons: list[str] = []
    for row in range(len(prepared)):
        contribution = contributions[row]
        # Largest positive contributions first; stop at zero/negative ones.
        order = np.argsort(contribution)[::-1]
        phrases: list[str] = []
        for index in order:
            if contribution[index] <= 0 or len(phrases) >= top_n:
                break
            phrase = _describe(columns[index], values[row, index])
            if phrase and phrase not in phrases:
                phrases.append(phrase)
        reasons.append(", ".join(phrases) if phrases else _NO_DRIVERS)
    return reasons
