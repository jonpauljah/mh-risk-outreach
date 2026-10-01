# Healthcare dataset

The selected input is the Global Mental Health & Lifestyle Survey Dataset.
The CSV has been downloaded and inspected. Milestone 1 will extract, clean, and load the selected fields into an explicit
SQL schema. The initial selection includes all 51 columns; the field mapping
and schema must be finalized and implemented during this milestone.

## Source

- **Dataset:** Global Mental Health & Lifestyle Survey Dataset.
- **Source:** [Kaggle dataset page](https://www.kaggle.com/datasets/dhrubangtalukdar/global-mental-health-and-lifestyle-survey-dataset),
  published under the account `dhrubangtalukdar`.

According to the dataset overview provided by the project owner, this is a
synthetic survey dataset designed for machine learning, data analysis, and
research. It contains 10,000 records, more than 50 features, and a binary target
label indicating the presence of a mental health issue. It covers demographics,
work conditions, lifestyle habits, social support, and psychological symptoms.

## Repository dataset

- **Filename:** `mental_health.csv`.
- **Downloaded:** October 1, 2026.
- **Records:** 10,000 data rows.
- **Columns:** 51 total, including `Has_Mental_Health_Issue`.
- **Size:** 1,774,498 bytes (about 1.69 MiB).
- **SHA-256:** `5890f6d073b9313d20ef0b43c231085462bb565dfa44f6307bb549eeb5fcd0ed`.

These counts and the checksum were verified against the downloaded CSV.

The CSV will be committed at [data/mental_health.csv](../../data/mental_health.csv)
so contributors get the same input when cloning the repository. No separate
download is needed.

It exceeds the default 1,024 KiB file-size limit, so the hook has an exception
for exactly `data/mental_health.csv`. Other files remain subject to that limit.

**Pending:** confirm the Kaggle version and redistribution license.

## Selection rationale

The dataset focuses on mental health and lifestyle and provides a broad range
of characteristics relevant to the project. Its demographics, work conditions,
habits, social support, and symptoms offer useful fields to explore when
preparing member records and relevant features.

Its stated machine-learning purpose and binary target also make it a candidate
for future mental health model experiments. Milestone 1 focuses on importing,
cleaning, storing, and displaying the data; model training is outside this
milestone. The initial selection keeps all columns. Field mapping and schema decisions
belong to Milestone 1; model input selection and training belong to later work.

## Relevant columns

Milestone 1 initially selects all 51 source columns, including
`Has_Mental_Health_Issue`. Numeric values and 0/1 fields are preserved, and
categories such as `Gender` and `Employment_Status` remain text. This CSV has
no `Occupation` column. See [ETL rules](etl-rules.md) for mapping and loading requirements.

**Pending for Milestone 1:** document field meanings, SQL destinations, types,
units where known, allowed values, and cleaning rules. Identify the target,
candidate predictors, and metadata separately. Record unknown definitions
explicitly; final model input selection belongs to later ML work.

## Known limitations

The following observations were checked against the downloaded CSV on October
1, 2026. Counts refer to this snapshot, not future versions of the Kaggle dataset.

### Synthetic survey data

The source overview describes synthetic survey records rather than observed
patient records. The file is useful for demonstrating ETL, SQL, and a GUI, but
does not establish how a model or outreach process would perform with real
healthcare members. Many symptom scores are distributed nearly evenly across
0–10, suggesting generated distributions; the CSV alone does not establish
how the values or target were generated.

### No member identity or follow-up history

There is no member identifier, survey date, contact information, insurance
membership, healthcare market identifier, or outreach outcome field. A row
therefore cannot be linked reliably to a person across sources or repeated
surveys using the supplied fields. Any prototype record identifier must be
defined separately, and it must not imply a verified patient identity.
The absence of dates also prevents checking trends or the order of symptoms,
treatment, and outcomes.

### Strong target imbalance

`Has_Mental_Health_Issue` contains 9,216 values of 1 (92.16%) and 784 values of
0 (7.84%). Predicting 1 for every row would achieve 92.16% accuracy without
using any features. Future model evaluation should account for that baseline
and report performance for both classes rather than accuracy alone.

The CSV does not explain how this binary label was assigned. It should not be
described as a confirmed diagnosis or an outreach-priority score. Individual
numeric-feature Pearson correlations with the target are small in this snapshot
(all absolute values below 0.056); this does not rule out nonlinear or combined
relationships or establish model quality.

### Field combinations that need review

| Observation | Records | Why review is needed |
| --- | --- | --- |
| Employment is `Unemployed`, but weekly work hours are positive | 1,232 | Every unemployed record has positive hours; the meaning or time window of work hours needs clarification |
| Social media hours exceed screen-time hours per day | 1,479 | Unexpected if social media is included in total screen time |
| `On_Therapy_Now` is 1, but `Ever_Sought_Treatment` is 0 | 1,203 | Treatment definitions or response consistency need clarification |
| `On_Medication` is 1, but `Ever_Sought_Treatment` is 0 | 1,181 | Medication scope and treatment definitions need clarification |

These groups may overlap. They are review flags, not agreed rejection rules.
Do not change or discard these records until the team defines the relevant
[ETL rules](etl-rules.md).

### Limited population and market detail

Ages range from 18 to 75, so this snapshot contains no children or adults older
than 75. Countries are limited to USA, India, UK, Brazil, Germany, and `Other`;
1,945 records (19.45%) are grouped into `Other`. Country alone does not identify
a healthcare market. Income uses `Low`, `Middle`, and `High` without currency or
threshold fields. These observations limit direct comparisons across populations
and markets; the CSV does not establish representativeness.

### Field definitions are incomplete

The CSV contains values but no data dictionary defining score meanings, survey
time windows, label construction, or how to interpret `Panic_Attacks` values
of 0–4. Some fields use 0/1, others use scores, and others use text categories.
`Not sure` and `Prefer not to say` are explicit responses, not blank cells;
their handling requires a documented decision.

### Limited coverage of messy input

Pandas detected no missing values across the 51 columns and no exact duplicate
rows. Text columns contained no blank strings, surrounding whitespace, or
categories differing only by letter case. This does not prove every value is
correct, and exact-row checks cannot detect duplicate people without identifiers.
Tests for missing values, malformed fields, inconsistent formatting, and
duplicates will need synthetic fixtures because this snapshot does not exercise
those cases.

**Pending:** confirm field definitions and generation details from the source,
finalize the Milestone 1 field mapping and schema, and document how each
limitation affects ETL. Later modeling must account for the retained limitations.
Implement the agreed cleaning rules during Milestone 1.
