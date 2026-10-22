"""Select model inputs without cleaning, encoding, or reading the target."""

import pandas as pd

# Source columns selected in docs/ml-spec.md, in a fixed model-input order.
MODEL_FEATURES = (
    # Symptoms (13).
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
    # Safety flags (2); also available separately for priority rules.
    "Self_Harm_Thoughts",
    "Suicidal_Thoughts",
    # History (2).
    "Trauma_History",
    "Family_History_Mental_Illness",
    # Social (5).
    "Social_Support",
    "Feel_Understood",
    "Loneliness",
    "Close_Friends_Count",
    "Discuss_Mental_Health",
    # Stress and work (5).
    "Financial_Stress",
    "Work_Stress_Level",
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Ever_Bullied_At_Work",
    # Socioeconomic (3).
    "Income_Level",
    "Employment_Status",
    "Marital_Status",
    # Substance and habits (4).
    "Alcohol_Frequency",
    "Smoking",
    "Sleep_Hours_Night",
    "Exercise_Per_Week",
    # Age (1).
    "Age",
)


def select_model_inputs(records: pd.DataFrame) -> pd.DataFrame:
    """Copy the 35 inputs in order, preserving values and the row index.

    The target and member identifier are neither required nor included.
    Missing input columns raise KeyError; extra columns are ignored.
    """
    return records.loc[:, list(MODEL_FEATURES)].copy()
