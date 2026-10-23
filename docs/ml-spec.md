
# Machine Learning To Identify Risks

This document walks through the design decisions for our ML approach: the algorithm, the training process, and the implementation. This file is subject to change throughout milestone 1.

We will use a machine learning model to score each member's risk for mental health challenges. Instead of us writing every scoring rule by hand, the model learns how much each member attribute matters from labeled examples. Rules remain alongside the model for two cases: the safety override (any member reporting self-harm or suicidal thoughts is always high priority) and risk factors our dataset does not cover, such as missed appointments.

This doc covers how we use ML to identify members at risk, how we train the model, how we observe and interpret its outputs, and how we improve it over time.

## Summary

We will use logistic regression as the first model for the MVP/POC. It is the simplest of the candidate algorithms and the easiest to explain: it learns one weight per column. Random Forest and XGBoost will be trained later on the same data for comparison.

The training data is a Kaggle survey dataset of 10,000 rows. The column `Has_Mental_Health_Issue` (0 or 1) is the answer column the model learns to predict. The input columns are a mix of yes/no flags, number scales, and text categories.

The sponsor approved `Has_Mental_Health_Issue` as the prediction target. The
prototype will use the model's estimated probability of this label to rank
records for outreach priority.

We hold back 20% of the real rows as a test set. The model never sees these during training, so they give an honest check of its predictions. Synthetic data is used only to test that the code handles unusual inputs. It cannot measure prediction quality, because made-up rows have no true answer.

Model quality is measured with recall, precision, PR-AUC, and calibration. PR-AUC is one number for the whole model that summarizes how well it ranks members. We use it to compare models and pick the best one.

One known limitation: 92% of rows in the dataset are marked "yes". A model that always answers "yes" is therefore 92% accurate, so accuracy alone is not a meaningful result. Every metric is reported next to that baseline.

Unit tests cover each function in the ML implementation, so that bugs in data preparation or scoring are caught before they affect the model.

# Training & Data

Training will take place early to allow ample time for testing predictions, retraining, and adjusting the training dataset if needed.

## Setup and Pre-training

**Input and Data**

Our data is a Kaggle dataset with 10,000 rows and 51 columns. It includes
columns that are useful for predicting mental health risk. The column
`Has_Mental_Health_Issue` is the ground truth the model trains against.

Not every column should go into the model. The first table shows which
columns we keep and which we drop, with the reason. The second shows how
each kept column is transformed before training.

> **Note — storage and model input are intentionally different.** The
> Milestone 1 ETL stores the *whole* survey: all 50 feature columns plus the
> target (52 columns with `Member_ID`) in the `patients` table. The "drop"
> decisions below are **model-input** choices applied at the modeling stage by
> `prepare()`, not at storage. Nothing is dropped from the database. Keeping
> every column stored lets us revisit these choices, run the fairness checks,
> and feed the recommendation stage without re-importing the CSV.


| Group | Columns | Decision |
| --- | --- | --- |
| Symptoms (13) | `Feeling_Sad_Down`, `Loss_Of_Interest`, `Sleep_Trouble`, `Fatigue`, `Poor_Appetite_Or_Overeating`, `Feeling_Worthless`, `Concentration_Difficulty`, `Anxious_Nervous`, `Mood_Swings`, `Irritability`, `Obsessive_Thoughts`, `Compulsive_Behavior`, `Panic_Attacks` | Keep |
| Safety flags (2) | `Self_Harm_Thoughts`, `Suicidal_Thoughts` | Keep, and also use in the override rule |
| History (2) | `Trauma_History`, `Family_History_Mental_Illness` | Keep |
| Social (5) | `Social_Support`, `Feel_Understood`, `Loneliness`, `Close_Friends_Count`, `Discuss_Mental_Health` | Keep |
| Stress and work (5) | `Financial_Stress`, `Work_Stress_Level`, `Job_Satisfaction`, `Work_Life_Balance`, `Ever_Bullied_At_Work` | Keep |
| Socioeconomic (3) | `Income_Level`, `Employment_Status`, `Marital_Status` | Keep |
| Substance and habits (4) | `Alcohol_Frequency`, `Smoking`, `Sleep_Hours_Night`, `Exercise_Per_Week` | Keep |
| Age (1) | `Age` | Keep |
| Treatment status (4) | `Previously_Diagnosed`, `Ever_Sought_Treatment`, `On_Therapy_Now`, `On_Medication` | Drop from the model; they describe someone already identified. Keep in the database for the recommendation stage. |
| Fairness check (2) | `Gender`, `Country` | Drop from the model; keep to compare results across groups |
| Low value (9) | `Education`, `Work_Hours_Per_Week`, `Remote_Work`, `Company_Mental_Health_Support`, `Caffeine_Drinks_Day`, `Screen_Time_Hours_Day`, `Social_Media_Hours_Day`, `Hobby_Time_Hours_Week`, `Diet_Quality` | Drop; a care organization is unlikely to hold these |

| Type | Columns | Transformation |
| --- | --- | --- |
| Already 0/1 (5) | `Self_Harm_Thoughts`, `Suicidal_Thoughts`, `Trauma_History`, `Family_History_Mental_Illness`, `Ever_Bullied_At_Work` | None |
| Number scales (23) | The 13 symptom columns, `Social_Support`, `Feel_Understood`, `Loneliness`, `Close_Friends_Count`, `Financial_Stress`, `Work_Stress_Level`, `Job_Satisfaction`, `Work_Life_Balance`, `Sleep_Hours_Night`, `Age` | Leave as numbers; put on a common scale with `StandardScaler` for logistic regression |
| Ordered text (4) | `Income_Level`, `Exercise_Per_Week`, `Alcohol_Frequency`, `Discuss_Mental_Health` | Map to ordered numbers, e.g. Low 0, Middle 1, High 2 |
| Unordered text (3) | `Employment_Status`, `Marital_Status`, `Smoking` | One yes/no column per category (turns 35 columns into 44) |


Of the 51 columns, one is the answer column and 15 are dropped, leaving 35
input columns. Splitting the three unordered text columns into yes/no
columns turns those 35 into the 44 numeric columns the model trains on.

We drop columns for one of three reasons: they describe a member who has
already been identified (treatment status), they should not influence the
score (gender and country, which we keep only to check fairness), or a care
organization is unlikely to have them.

The transformations put every column into a form the model can use:

- Turn text categories into numbers.
- Put all columns on the same scale, which logistic regression needs.
- Handle missing values. The Kaggle data has none, but future data will, so
  the preparation code must define what happens to a blank.

### Model input selection (issue #13)

`mh_risk_outreach.model_inputs.MODEL_FEATURES` defines the 35 selected source
columns in a fixed order. `select_model_inputs(records)` returns a copy of
those columns, preserving values and the row index. It excludes `Member_ID`,
`Has_Mental_Health_Issue`, and the 15 source features marked for exclusion above.
The full survey remains stored by the ETL.

Training code keeps the inputs (`X`) and target (`y`) separate:

```python
from mh_risk_outreach.config import TARGET
from mh_risk_outreach.model_inputs import select_model_inputs

X = select_model_inputs(records)
y = records[TARGET].copy()  # Training only; never passed into input selection.
```

Scoring uses the same selector without requiring a target column. Missing model
input columns raise `KeyError`; extra columns are ignored. Selection does not
clean values, validate their contents, encode categories, scale numbers, or fit
a model. Those steps remain future work, outside issue #13.

## Training

Training is a build step. The input is our prepared table of member rows.
The output is a saved model file that turns a member's row into a risk score.

### Tools

We use **scikit-learn**, a Python library for classic machine learning. It
includes logistic regression, the train/test split, scaling, and every metric
we report. We do not use PyTorch; it is built for neural networks, which we
do not need for a 10,000-row table.

Supporting libraries: `pandas` to load data, `joblib` to save the model, and
`pytest` for tests.

### Steps

1. **Prepare.** Run the rows through `prepare()`, which keeps our chosen
   columns and applies the transformations listed above.
2. **Split.** Set aside 20% of the rows as the test set. The model never
   sees these during training.
3. **Fit.** Train logistic regression on the other 80%. This finds a weight
   for each column that best matches the known answers. It takes seconds.
4. **Evaluate.** Score the test rows and compare against their real answers.
5. **Save.** Write the model to a versioned file with a record of how it was
   built.

### Code outline

```python
def prepare(df):
    df = df[KEEP_COLUMNS]
    df = map_ordered_text(df)      # Income_Level: Low 0, Middle 1, High 2
    df = one_hot(df, ["Employment_Status", "Marital_Status", "Smoking"])
    return df                      # always 44 numeric columns, same order

def train(df):
    X, y = prepare(df), df["Has_Mental_Health_Issue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000))])
    model.fit(X_train, y_train)
    return model, X_test, y_test

def score(model, new_rows):
    return model.predict_proba(prepare(new_rows))[:, 1] * 100
```

### Things to watch for

Each item states the problem, why it matters, and what the implementation
must do.

#### 1. The answer column is 92% "yes"

In our dataset, 9,216 of the 10,000 rows have `Has_Mental_Health_Issue = 1`.

A "model" that ignores its input and always answers "yes" is therefore right
92% of the time. If our trained model reports 92% accuracy, that alone does
not show it learned anything. In a first baseline run, logistic regression
scored 92% accuracy, the same as always answering "yes".

**Requirement:** report every metric next to the same metric for the
always-"yes" baseline. The model is only useful if it beats that baseline.

#### 2. Split before scaling

To check the model we need rows it has never seen, so we hold back 20% as
the test set. Testing on rows the model trained on is like giving students
the exam questions in advance.

The split must come before scaling. Our columns use different ranges (`Age`
is 18 to 75, `Trauma_History` is 0 or 1), so a scaler rescales each column to
a similar range. To do that it measures each column's average and spread. If
it measures them on all 10,000 rows, information from the test rows leaks
into training.

The required order:

1. Split into 80% train and 20% test.
2. The scaler measures averages on the training rows only.
3. Those same averages are used to rescale the test rows.

**Requirement:** the scaler and the model live in one scikit-learn
`Pipeline`, which does this in the right order. No scaling happens outside
the pipeline.

#### 3. Training and scoring must prepare data identically

The model does not know column names. It only sees positions: the 1st
number, the 2nd, and so on up to the 44th. Training teaches it what each
position means.

Two things can break this:

- **Separate preparation code.** If training and scoring prepare data
  differently, a position can end up holding a different value. The model
  does not crash. It silently returns wrong scores.
- **Missing categories.** `Marital_Status` becomes four yes/no columns:
  Single, Married, Divorced, Widowed. If a later batch has no widowed
  members, a naive conversion makes three columns, and the model gets 43
  numbers when it expects 44.

**Requirement:** one `prepare()` function is used for both training and
scoring. It always outputs the same 44 columns in the same order.

#### 4. Blank or unexpected values

The Kaggle dataset has no missing values. Real data will: a blank
`Sleep_Hours_Night`, or an `Employment_Status` of "Retired" that the model
never saw. Logistic regression cannot compute on a blank and will fail.

For each column we either reject the row and report an error, or fill in a
default such as the column's average.

**Requirement:** `prepare()` defines one of these behaviors for every column.

**Decision needed:** which behavior each column gets.

#### 5. Reproducibility

The 80/20 split is random. Without a fixed seed, two people running the same
code get different test sets and slightly different numbers, and nobody can
tell whether a result changed because of the code or because of the shuffle.

**Requirement:** set `random_state=42` wherever randomness is used. The
number has no meaning; it only has to stay the same.

#### 6. Priority bands

Our architecture defines HIGH as a score of 80 to 100. A score is the model's
estimate of how likely "yes" is, and 92% of rows are "yes", so nearly every
member scores high. In the baseline run about 97% of members scored 80 or
above. A worklist where 97% of members are HIGH tells a case manager nothing.

**Requirement:** prioritize by rank, not by fixed cutoffs. Sort by score and
take the top K members, or define bands by position (for example, top 10% is
HIGH).

**Decision needed:** the value of K or the band percentages.

#### 7. Versioning

We will retrain more than once, and each retrain produces different scores.
If a member scores 87 one week and 74 the next, we need to know which model
produced each score.

**Requirement:** save each model under a version name (`model_v1`,
`model_v2`) with a small file recording the date, the columns used, and the
test results. Store the version name next to every score in the database.

# Scoring

Scoring is how the trained model is used in the pipeline.

1. Load the current model file.
2. Read member rows from the database and run them through `prepare()`.
3. The model returns a score from 0 to 100 for each member.
4. Apply the safety override: any member with `Self_Harm_Thoughts = 1` or
   `Suicidal_Thoughts = 1` is placed at the top priority regardless of score.
5. Sort members by score and assign priority by rank.
6. Write the score, priority, top reasons, and model version back to the
   database for the Prioritization stage.

# Testing

Testing has two parts. Unit tests check that the code is correct. Model
evaluation checks that the predictions are useful.

## Unit tests

Written with `pytest` and run against a 200-row sample file so they finish
in seconds.

**Data preparation**

- Output always has the same 44 columns in the same order, even for one row.
- Each text mapping gives the expected number (`"Low"` becomes 0).
- Dropped columns and the answer column never appear in the output.
- A blank or unknown value triggers the behavior we chose, not a crash.

**Training and scoring**

- Train and test sets share no rows.
- The same seed produces identical scores on two runs.
- Every score is between 0 and 100, with one score per input row.
- A saved and reloaded model gives the same scores as the original.

**Behavior**

- A row with every symptom at maximum scores higher than the same row with
  every symptom at zero.
- A member with a safety flag always lands at top priority.
- A fixed set of about 20 rows keeps the same ranking after any code change.

## Model evaluation

Run on the 20% test set after every training run.

| Measure | Question it answers |
| --- | --- |
| Recall | Of the members who had an issue, how many did we flag? |
| Precision | Of the members we flagged, how many had an issue? |
| PR-AUC | How well does the model rank members overall? |
| Calibration | When the model says 80%, is it right about 8 times in 10? |
| Accuracy | What share of all predictions were correct? |

Each result is reported next to the always-"yes" baseline. Results from the
initial baseline run:

| Measure | Logistic regression | Always "yes" |
| --- | --- | --- |
| Accuracy | 92% | 92% |
| PR-AUC | 0.95 | 0.92 |

The model ranks slightly better than the baseline. Among the tenth of
members it scored lowest, 80% had an issue; among the tenth it scored
highest, 95% did. The signal is real but weak.

We do not use a test such as "accuracy must exceed 90%", because the
baseline already passes it. The quality gate is: PR-AUC must beat the
baseline.

# Interpreting Outputs

Logistic regression learns one weight per column. A positive weight means
the column pushes the score up. A negative weight pushes it down. A larger
weight means a stronger push.

For one member, we multiply each weight by that member's value in the
column. The three largest results are the top reasons for the score. These
are stored with the score and shown to the case manager, for example "low
social support" or "high work stress".

This is why logistic regression is the first model: the reasons come
directly from the model with no extra library.

# Observability

Each scoring run writes one record to a log table:

- Model version and time of the run
- Number of rows scored and how long it took
- Number of rows rejected by `prepare()`
- Average score and how many members fell in each priority group
- Average of each input column

A page in the UI charts these across runs. A sudden change in the average
score or in an input column's average means the incoming data no longer
looks like the training data, and the scores should not be trusted until we
investigate.

# Improving the Model

The model only changes when we retrain it. We retrain when:

- **We get new answers.** The Engagement Tracking stage records what
  happened after outreach. Those outcomes become answer data for the next
  training run.
- **We add a column.** The column is added to `prepare()` and the model is
  retrained from scratch. This only helps if the training rows have real
  values for that column.
- **The data shifts.** The observability page shows incoming data drifting
  away from the training data.

Each retrain produces a new model version. It replaces the current one only
if it scores better on the test set.

Once logistic regression is working end to end, we train Random Forest and
XGBoost on the same split and compare all three on PR-AUC.

# Open Decisions

## For the team

- What `prepare()` does with a blank or unknown value in each column:
  reject the row or fill a default.
- How priority is assigned: top K members, or bands by percentage.
- Where model files and the log table are stored.
- Who owns each part: data preparation, training, scoring, tests, and the
  observability page.
- Whether to generate synthetic data for missed appointments and missed
  medication, or leave those as rules only.

## For Dr. Shalmali

- **The dataset is 92% "yes" and no column predicts the answer well.**
  Should we keep it and document the limitation, or find a dataset with a
  more balanced answer column? Does she have one to recommend?
- **The 90% accuracy target.** Always answering "yes" already reaches 92%.
  What target should we use instead for recall, precision, or PR-AUC?
- **Resolved — prediction target.** The sponsor approved
  `Has_Mental_Health_Issue` for the prototype's risk ranking. The source
  label's definition remains a documented dataset limitation.
- **Outreach capacity.** How many members can a care team contact per
  cycle? This sets the top K.
- **Excluded columns.** We left gender and country out of the model and
  use them only to check fairness. We also left out treatment status. Does
  she agree with both choices?

# Glossary

| Term | Meaning |
| --- | --- |
| Model | A function that takes a member's row and returns a risk score |
| Training | Building the model from rows with known answers |
| Answer column | The column the model learns to predict (also called the label) |
| Test set | Rows held back from training and used to check the model |
| Baseline | The simplest possible approach, used as the bar to beat |
| Recall | Of the members who had an issue, the share we flagged |
| Precision | Of the members we flagged, the share who had an issue |
| PR-AUC | One number for how well the model ranks members |
| Calibration | How closely the scores match real-world frequency |
| Drift | Incoming data no longer looks like the training data |