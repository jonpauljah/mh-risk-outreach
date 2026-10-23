"""Shared configuration: the column selection and the dataset/database paths.

These constants define storage columns and dataset/database paths. The ETL
stores the full survey; model_inputs.py defines the smaller model-input list.

The ETL ingests the whole survey: all 50 source columns plus the target. Columns
are grouped by SQL storage type so ``schema.sql`` and the transform agree:

* ``INTEGER_SCORES`` – integer scales and counts (stored INTEGER)
* ``REAL_HOURS``     – fractional hour fields (stored REAL)
* ``TEXT_CATEGORIES``– readable text categories (stored TEXT)
* ``BINARY_FLAGS``   – 0/1 flags (stored INTEGER, CHECK IN (0, 1))

``TARGET`` is also a 0/1 flag but is kept separate because it is the answer
column, never a predictor.
"""

from pathlib import Path

# ---- Integer scales and counts (stored INTEGER) ----
INTEGER_SCORES = [
    "Age",
    "Work_Hours_Per_Week",
    "Job_Satisfaction",
    "Work_Stress_Level",
    "Work_Life_Balance",
    "Caffeine_Drinks_Day",
    "Hobby_Time_Hours_Week",
    "Financial_Stress",
    "Feeling_Sad_Down",
    "Loss_Of_Interest",
    "Sleep_Trouble",
    "Fatigue",
    "Poor_Appetite_Or_Overeating",
    "Feeling_Worthless",
    "Concentration_Difficulty",
    "Anxious_Nervous",
    "Panic_Attacks",
    "Mood_Swings",
    "Irritability",
    "Obsessive_Thoughts",
    "Compulsive_Behavior",
    "Social_Support",
    "Close_Friends_Count",
    "Feel_Understood",
    "Loneliness",
]
# ---- Fractional hour fields (stored REAL) ----
REAL_HOURS = [
    "Sleep_Hours_Night",
    "Screen_Time_Hours_Day",
    "Social_Media_Hours_Day",
]
# ---- Readable text categories (stored TEXT) ----
TEXT_CATEGORIES = [
    "Gender",
    "Country",
    "Education",
    "Marital_Status",
    "Income_Level",
    "Employment_Status",
    "Remote_Work",
    "Company_Mental_Health_Support",
    "Exercise_Per_Week",
    "Alcohol_Frequency",
    "Smoking",
    "Diet_Quality",
    "Discuss_Mental_Health",
]
# ---- Binary 0/1 flags (stored INTEGER, CHECK IN (0, 1)) ----
BINARY_FLAGS = [
    "Ever_Bullied_At_Work",
    "Self_Harm_Thoughts",
    "Suicidal_Thoughts",
    "Family_History_Mental_Illness",
    "Previously_Diagnosed",
    "Ever_Sought_Treatment",
    "On_Therapy_Now",
    "On_Medication",
    "Trauma_History",
]

NUMERIC = INTEGER_SCORES + REAL_HOURS  # continuous/ordinal numerics (28)
CATEGORICAL = TEXT_CATEGORIES + BINARY_FLAGS  # text + 0/1 flags (22)
FEATURES = NUMERIC + CATEGORICAL  # all 50 stored source features
TARGET = "Has_Mental_Health_Issue"  # the Y answer key (never a feature)
MEMBER_ID = "Member_ID"  # surrogate key added during transform

# ---- Default paths and table name ----
DEFAULT_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "mental_health.csv"
DEFAULT_DB_PATH = "patients.db"
DEFAULT_TABLE = "patients"
