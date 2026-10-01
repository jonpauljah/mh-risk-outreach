# ETL rules

Milestone 1 implements extraction, transformation, and loading of
[data/mental_health.csv](../../data/mental_health.csv) into SQLite. The pipeline
must load an explicit schema defined in `src/mh_risk_outreach/db/schema.sql` and
produce records that Streamlit and later ML code can query consistently.
The implementation is currently pending; this document specifies its scope and
the decisions that must be completed during this milestone.

## Extraction

Use `pandas.read_csv()` in `etl/extract.py` to read the selected CSV. Validate
its header and required columns before transformation. Record the input path,
snapshot/checksum, and number of source rows in the processing summary.

The initial field selection includes all 51 source columns, including
`Has_Mental_Health_Issue`, so the survey remains available for inspection.
Document any change to that selection and its rationale before implementation.
The CSV has no member identifier; a survey row represents a prototype record,
not a verified patient identity.

**Pending for Milestone 1:** define accepted input encoding, required columns,
handling of unexpected columns, and errors for unreadable files or malformed CSV.

## Source-to-database mapping

Document every selected source field before loading it. The mapping must agree
with the SQL schema and the transformation code.

| Mapping detail | Required documentation |
| --- | --- |
| Source field | Exact CSV name and known meaning or definition limitation |
| SQL destination | Table and column name; document any renaming |
| Type | SQLite type and Python conversion |
| Missing values | Whether SQL NULL is allowed and how blank values are handled |
| Validation | Allowed values/ranges and action on failure |
| Transformation | Cleaning or normalization applied, including units |
| Consumer role | Display field, candidate ML feature, target, or metadata |

**Pending for Milestone 1:** complete the field mapping and choose the table and
column names. Numeric fields remain numeric, binary fields remain integers 0/1,
and categorical responses remain readable text. Do not add a pandas index as an
unintended database column; use `index=False` if loading with `to_sql()`.

## Cleaning and normalization

Implement the agreed rules in `etl/transform.py`. Specify handling of whitespace,
blank values, category spelling/case, numeric conversion, and invalid values.
Keep already valid values unchanged. The supplied snapshot has no detected
missing values or exact duplicate rows, so fixtures must exercise these cases.

`Not sure` and `Prefer not to say` are explicit responses. Do not silently turn
them into missing values. Do not infer score meanings, units, or valid ranges
solely from the observed minimum and maximum.

**Pending for Milestone 1:** define per-field nullability, category normalization,
conversion rules, and actions for missing or invalid values. State whether each
rule rejects a row, stops the import, retains a flagged row, or changes a value.
Avoid undocumented imputation or silent data loss.

Model-specific category encoding and feature scaling belong to later ML work.
Store `Has_Mental_Health_Issue` as the source target label and document its role
separately from candidate predictors; it is not a diagnosis or prediction.

## Review flags and invalid records

The [dataset guide](dataset.md) lists combinations needing review, such as
therapy without prior treatment and social media time exceeding screen time.
These observations are not established rejection rules. Preserve them unless
a documented rule justifies a change, rejection, or flag.

**Pending for Milestone 1:** define validation outcomes, row references, and
reason codes. Summaries should identify affected rows and reasons without
printing entire records unnecessarily.

## Record identifiers, duplicates, and repeat imports

Define a prototype record key because the source contains no member identifier.
Document whether keys remain stable across repeat imports and how source rows
can be traced. Exact duplicate rows do not establish duplicate people.

**Pending for Milestone 1:** choose duplicate detection and handling, and choose
repeat-import behavior (for example, replace the dataset contents transactionally
or skip an already imported snapshot). Repeating an import must have a documented,
testable outcome and must not silently accumulate duplicate records.

## Schema creation and loading

`db/repository.py` must initialize tables from `db/schema.sql` before loading
transformed records. That SQL file must contain executable definitions with
column types, keys, nullability, and agreed constraints during Milestone 1.

Use parameterized inserts or pandas `to_sql()` against the existing tables.
Do not use `if_exists="replace"` to recreate tables with inferred definitions,
because that would discard the tracked schema and constraints. Keep sequencing
in `etl/pipeline.py` and database access in `db/repository.py`.

**Pending for Milestone 1:** select the database path, transaction boundaries,
and rollback behavior. A failed load must have a documented database state;
report success only after the intended writes commit.

## Processing log or summary

Report the input identity, start/end time or duration, success/failure status,
source rows, accepted rows, rejected rows, duplicates skipped, rows loaded, and
failure/validation reasons. Explain how counts reconcile; flags can overlap and
must not be counted as separate discarded rows. Distinguish rows loaded by this
run from the total already stored in the database.

**Pending for Milestone 1:** choose console/file output and its location, define
the exact count categories, and include example successful and failed summaries.

Link the implemented rules to meaningful tests and [verification evidence](verification.md).
See [database architecture](database.md) for schema documentation.
