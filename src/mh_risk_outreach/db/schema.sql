-- Explicit table definition for the imported member records (Milestone 1).
--
-- The repository initializes the database from this file BEFORE loading, so the
-- tracked schema (not a pandas-inferred one) owns the column names, types, and
-- constraints. The ETL load appends rows into this table.
--
-- The table ingests the whole survey: Member_ID (surrogate key) + all 50 source
-- columns + the target label (52 columns). Types follow the CSV snapshot:
-- integer scales and counts as INTEGER, fractional hour fields as REAL, readable
-- categories as TEXT. Binary 0/1 fields carry a CHECK (documented as binary);
-- score fields do NOT, because the dataset guide warns against inferring valid
-- ranges from the observed min/max. All columns are NOT NULL: the snapshot has
-- no missing values. Per-field nullability for future, messier sources is
-- revisited in later work. Column grouping mirrors config.py.

CREATE TABLE IF NOT EXISTS patients (
    -- Surrogate key. Not a verified patient identity; a survey row is a
    -- prototype record. Assigned during transform, unique per load.
    Member_ID                      INTEGER PRIMARY KEY,

    -- ---- Integer scales and counts (25) ----
    Age                            INTEGER NOT NULL,
    Work_Hours_Per_Week            INTEGER NOT NULL,
    Job_Satisfaction               INTEGER NOT NULL,
    Work_Stress_Level              INTEGER NOT NULL,
    Work_Life_Balance              INTEGER NOT NULL,
    Caffeine_Drinks_Day            INTEGER NOT NULL,
    Hobby_Time_Hours_Week          INTEGER NOT NULL,
    Financial_Stress               INTEGER NOT NULL,
    Feeling_Sad_Down               INTEGER NOT NULL,
    Loss_Of_Interest               INTEGER NOT NULL,
    Sleep_Trouble                  INTEGER NOT NULL,
    Fatigue                        INTEGER NOT NULL,
    Poor_Appetite_Or_Overeating    INTEGER NOT NULL,
    Feeling_Worthless              INTEGER NOT NULL,
    Concentration_Difficulty       INTEGER NOT NULL,
    Anxious_Nervous                INTEGER NOT NULL,
    Panic_Attacks                  INTEGER NOT NULL,
    Mood_Swings                    INTEGER NOT NULL,
    Irritability                   INTEGER NOT NULL,
    Obsessive_Thoughts             INTEGER NOT NULL,
    Compulsive_Behavior            INTEGER NOT NULL,
    Social_Support                 INTEGER NOT NULL,
    Close_Friends_Count            INTEGER NOT NULL,
    Feel_Understood                INTEGER NOT NULL,
    Loneliness                     INTEGER NOT NULL,

    -- ---- Fractional hour fields (3) ----
    Sleep_Hours_Night              REAL    NOT NULL,
    Screen_Time_Hours_Day          REAL    NOT NULL,
    Social_Media_Hours_Day         REAL    NOT NULL,

    -- ---- Readable text categories (13) ----
    Gender                         TEXT    NOT NULL,
    Country                        TEXT    NOT NULL,
    Education                      TEXT    NOT NULL,
    Marital_Status                 TEXT    NOT NULL,
    Income_Level                   TEXT    NOT NULL,
    Employment_Status              TEXT    NOT NULL,
    Remote_Work                    TEXT    NOT NULL,
    Company_Mental_Health_Support  TEXT    NOT NULL,
    Exercise_Per_Week              TEXT    NOT NULL,
    Alcohol_Frequency              TEXT    NOT NULL,
    Smoking                        TEXT    NOT NULL,
    Diet_Quality                   TEXT    NOT NULL,
    Discuss_Mental_Health          TEXT    NOT NULL,

    -- ---- Binary 0/1 flags (9) ----
    Ever_Bullied_At_Work           INTEGER NOT NULL CHECK (Ever_Bullied_At_Work IN (0, 1)),
    Self_Harm_Thoughts             INTEGER NOT NULL CHECK (Self_Harm_Thoughts IN (0, 1)),
    Suicidal_Thoughts              INTEGER NOT NULL CHECK (Suicidal_Thoughts IN (0, 1)),
    Family_History_Mental_Illness  INTEGER NOT NULL CHECK (Family_History_Mental_Illness IN (0, 1)),
    Previously_Diagnosed           INTEGER NOT NULL CHECK (Previously_Diagnosed IN (0, 1)),
    Ever_Sought_Treatment          INTEGER NOT NULL CHECK (Ever_Sought_Treatment IN (0, 1)),
    On_Therapy_Now                 INTEGER NOT NULL CHECK (On_Therapy_Now IN (0, 1)),
    On_Medication                  INTEGER NOT NULL CHECK (On_Medication IN (0, 1)),
    Trauma_History                 INTEGER NOT NULL CHECK (Trauma_History IN (0, 1)),

    -- ---- Source target label (never a predictor) ----
    -- Binary label from the source survey. Not a diagnosis or outreach score.
    Has_Mental_Health_Issue        INTEGER NOT NULL CHECK (Has_Mental_Health_Issue IN (0, 1))
);

-- Historical assessment snapshots are created only by explicit case or
-- reassessment operations. Inputs contain exactly the 50 FEATURES as JSON.
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id INTEGER PRIMARY KEY,
    Member_ID INTEGER NOT NULL REFERENCES patients(Member_ID),
    source TEXT NOT NULL CHECK (source IN ('baseline', 'reassessment')),
    created_at TEXT NOT NULL,
    inputs_json TEXT NOT NULL CHECK (json_valid(inputs_json)),
    scoring_status TEXT NOT NULL CHECK (scoring_status IN ('PendingScoring', 'Scored')),
    risk_score REAL CHECK (risk_score IS NULL OR risk_score BETWEEN 0 AND 100),
    band_level_priority TEXT,
    rank_priority INTEGER,
    rank_context_json TEXT CHECK (rank_context_json IS NULL OR json_valid(rank_context_json)),
    explanation TEXT,
    recommendation_action TEXT,
    recommendation_reason TEXT,
    recommendation_rule_version TEXT,
    operational_model_name TEXT,
    model_version TEXT,
    scored_at TEXT,
    -- Pending rows deliberately carry no scoring data. A successful explicit
    -- scoring run fills the required score fields in one guarded update.
    CHECK (
      (scoring_status = 'PendingScoring' AND risk_score IS NULL AND band_level_priority IS NULL
       AND rank_priority IS NULL AND rank_context_json IS NULL AND explanation IS NULL
       AND recommendation_action IS NULL AND recommendation_reason IS NULL
       AND recommendation_rule_version IS NULL AND operational_model_name IS NULL
       AND model_version IS NULL AND scored_at IS NULL)
      OR
      (scoring_status = 'Scored' AND risk_score IS NOT NULL AND band_level_priority IS NOT NULL
       AND rank_priority IS NOT NULL AND rank_context_json IS NOT NULL AND explanation IS NOT NULL
       AND model_version IS NOT NULL AND scored_at IS NOT NULL)
    ),
    UNIQUE (assessment_id, Member_ID)
);
CREATE INDEX IF NOT EXISTS idx_assessments_member_created
  ON risk_assessments(Member_ID, created_at, assessment_id);
CREATE TRIGGER IF NOT EXISTS trg_assessment_inputs_immutable
BEFORE UPDATE OF Member_ID, source, created_at, inputs_json ON risk_assessments
BEGIN
    SELECT RAISE(ABORT, 'assessment identity and inputs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_scored_assessment_immutable
BEFORE UPDATE ON risk_assessments
WHEN OLD.scoring_status = 'Scored'
BEGIN
    SELECT RAISE(ABORT, 'scored assessment is immutable');
END;

-- Cases retain the proposal and assessment used at creation. Reassessment
-- changes current state only; it does not rewrite this historical origin.
CREATE TABLE IF NOT EXISTS outreach_cases (
    case_id INTEGER PRIMARY KEY,
    Member_ID INTEGER NOT NULL REFERENCES patients(Member_ID),
    assessment_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('Pending', 'Accepted', 'Completed')),
    outcome TEXT CHECK (outcome IS NULL OR outcome IN
      ('Not reached', 'Reached — declined', 'Reached — engaged', 'Referral completed')),
    recommended_action TEXT NOT NULL,
    recommendation_reason TEXT,
    operational_model_name TEXT,
    model_version TEXT,
    recommendation_rule_version TEXT,
    selected_action TEXT,
    accepted_action TEXT,
    override_justification TEXT,
    created_at TEXT NOT NULL,
    accepted_at TEXT,
    outcome_recorded_at TEXT,
    completed_at TEXT,
    -- The composite key prevents attaching another member's assessment.
    FOREIGN KEY (assessment_id, Member_ID)
      REFERENCES risk_assessments(assessment_id, Member_ID),
    CHECK ((status = 'Completed' AND outcome IS NOT NULL AND outcome_recorded_at IS NOT NULL AND completed_at IS NOT NULL)
        OR status <> 'Completed'),
    CHECK ((status = 'Pending' AND accepted_at IS NULL)
        OR (status IN ('Accepted', 'Completed') AND accepted_at IS NOT NULL))
);
-- These enforce separate rules: one case per snapshot for all time, and only
-- one open Pending/Accepted case per member at a time.
CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_one_assessment
  ON outreach_cases(assessment_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_one_active_member
  ON outreach_cases(Member_ID) WHERE status IN ('Pending', 'Accepted');
CREATE INDEX IF NOT EXISTS idx_cases_member_created
  ON outreach_cases(Member_ID, created_at, case_id);
CREATE TRIGGER IF NOT EXISTS trg_case_origin_immutable
BEFORE UPDATE OF Member_ID, assessment_id, recommended_action, recommendation_reason,
    operational_model_name, model_version, recommendation_rule_version, created_at
ON outreach_cases
BEGIN
    SELECT RAISE(ABORT, 'case origin and recommendation are immutable');
END;

-- This pointer follows the latest submitted assessment, including one that
-- still awaits scoring. Existing cases keep their original assessment_id.
CREATE TABLE IF NOT EXISTS member_assessment_state (
    Member_ID INTEGER PRIMARY KEY REFERENCES patients(Member_ID),
    assessment_id INTEGER NOT NULL UNIQUE,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (assessment_id, Member_ID)
      REFERENCES risk_assessments(assessment_id, Member_ID)
);

-- Replaceable operational projection. Case and assessment history lives above.
CREATE TABLE IF NOT EXISTS operational_scores (
    Member_ID INTEGER PRIMARY KEY REFERENCES patients(Member_ID),
    operational_model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    risk_score REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    band_level_priority TEXT NOT NULL,
    rank_priority INTEGER NOT NULL UNIQUE CHECK (rank_priority > 0),
    reasoning TEXT NOT NULL,
    safety_review_required INTEGER NOT NULL CHECK (safety_review_required IN (0, 1)),
    score_assessment_id INTEGER,
    safety_assessment_id INTEGER,
    inputs_pending_scoring INTEGER NOT NULL CHECK (inputs_pending_scoring IN (0, 1)),
    cohort_id TEXT NOT NULL,
    calculated_at TEXT NOT NULL,
    FOREIGN KEY (score_assessment_id, Member_ID)
      REFERENCES risk_assessments(assessment_id, Member_ID),
    FOREIGN KEY (safety_assessment_id, Member_ID)
      REFERENCES risk_assessments(assessment_id, Member_ID)
);
