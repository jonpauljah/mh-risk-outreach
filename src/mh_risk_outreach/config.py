"""Shared configuration: the column selection and the dataset/database paths.

These constants are the single source of truth for the project. The ETL modules
and later segments import them from here instead of redefining them, so the chosen
columns stay consistent everywhere.
"""

from pathlib import Path

# ---- Columns chosen in planning (the single source of truth) ----
NUMERIC = [
    "Age",
    "Work_Hours_Per_Week",
    "Job_Satisfaction",
    "Work_Stress_Level",
    "Work_Life_Balance",
    "Sleep_Hours_Night",
    "Caffeine_Drinks_Day",
    "Screen_Time_Hours_Day",
    "Social_Media_Hours_Day",
    "Hobby_Time_Hours_Week",
    "Financial_Stress",
    "Close_Friends_Count",
    "Feel_Understood",
    "Loneliness",
]
CATEGORICAL = [
    "Gender",
    "Country",
    "Education",
    "Marital_Status",
    "Income_Level",
    "Employment_Status",
    "Remote_Work",
    "Ever_Bullied_At_Work",
    "Company_Mental_Health_Support",
    "Exercise_Per_Week",
    "Alcohol_Frequency",
    "Smoking",
    "Diet_Quality",
    "Discuss_Mental_Health",
    "Family_History_Mental_Illness",
    "Trauma_History",
]
FEATURES = NUMERIC + CATEGORICAL  # the X inputs (30)
TARGET = "Has_Mental_Health_Issue"  # the Y answer key (never a feature)
MEMBER_ID = "Member_ID"  # surrogate key added during transform

# ---- Default paths and table name ----
DEFAULT_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "mental_health.csv"
DEFAULT_DB_PATH = "patients.db"
DEFAULT_TABLE = "patients"
