-- Explicit table definition for the prepared member records (Milestone 1).
--
-- The repository initializes the database from this file BEFORE loading, so the
-- tracked schema (not a pandas-inferred one) owns the column names, types, and
-- constraints. The ETL load appends rows into this table.
--
-- Columns: Member_ID (surrogate key) + 30 features + the source target label.
-- Types follow the CSV snapshot: integer scales and counts as INTEGER, hour
-- fields as REAL, readable categories as TEXT. Binary 0/1 fields carry a
-- CHECK (documented as binary); score fields do NOT, because the dataset guide
-- warns against inferring valid ranges from the observed min/max.
-- All columns are NOT NULL: the snapshot has no missing values. Per-field
-- nullability for future, messier sources is revisited in later work.

CREATE TABLE IF NOT EXISTS patients (
    -- Surrogate key. Not a verified patient identity; a survey row is a
    -- prototype record. Assigned during transform, unique per load.
    Member_ID               INTEGER PRIMARY KEY,

    -- ---- Numeric features (14) ----
    Age                     INTEGER NOT NULL,
    Work_Hours_Per_Week     INTEGER NOT NULL,
    Job_Satisfaction        INTEGER NOT NULL,
    Work_Stress_Level       INTEGER NOT NULL,
    Work_Life_Balance       INTEGER NOT NULL,
    Sleep_Hours_Night       REAL    NOT NULL,
    Caffeine_Drinks_Day     INTEGER NOT NULL,
    Screen_Time_Hours_Day   REAL    NOT NULL,
    Social_Media_Hours_Day  REAL    NOT NULL,
    Hobby_Time_Hours_Week   INTEGER NOT NULL,
    Financial_Stress        INTEGER NOT NULL,
    Close_Friends_Count     INTEGER NOT NULL,
    Feel_Understood         INTEGER NOT NULL,
    Loneliness              INTEGER NOT NULL,

    -- ---- Categorical features (16): 12 readable text + 4 binary 0/1 ----
    Gender                         TEXT    NOT NULL,
    Country                        TEXT    NOT NULL,
    Education                      TEXT    NOT NULL,
    Marital_Status                 TEXT    NOT NULL,
    Income_Level                   TEXT    NOT NULL,
    Employment_Status              TEXT    NOT NULL,
    Remote_Work                    TEXT    NOT NULL,
    Ever_Bullied_At_Work           INTEGER NOT NULL CHECK (Ever_Bullied_At_Work IN (0, 1)),
    Company_Mental_Health_Support  TEXT    NOT NULL,
    Exercise_Per_Week              TEXT    NOT NULL,
    Alcohol_Frequency              TEXT    NOT NULL,
    Smoking                        TEXT    NOT NULL,
    Diet_Quality                   TEXT    NOT NULL,
    Discuss_Mental_Health          TEXT    NOT NULL,
    Family_History_Mental_Illness  INTEGER NOT NULL CHECK (Family_History_Mental_Illness IN (0, 1)),
    Trauma_History                 INTEGER NOT NULL CHECK (Trauma_History IN (0, 1)),

    -- ---- Source target label (never a predictor) ----
    -- Binary label from the source survey. Not a diagnosis or outreach score.
    Has_Mental_Health_Issue INTEGER NOT NULL CHECK (Has_Mental_Health_Issue IN (0, 1))
);
