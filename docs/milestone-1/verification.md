# Milestone 1 verification

Milestone 1 is complete when the implemented ETL pipeline creates the explicit
SQL schema, loads cleaned survey records, reports the outcome, and displays
those stored records in Streamlit. This guide records the evidence required.
The current scaffold does not yet implement that workflow.

## Current checks

Follow [getting started](../../getting-started.md) to install dependencies.
From the project root, `uv run mh-risk-outreach` currently prints
`Hello from mh-risk-outreach!`, and `uv run pytest` checks that greeting.
These baseline checks do not verify ETL, SQL, or Streamlit.
Coverage remains a [manual check](../dependencies/dev-dependencies.md#pytest-cov).

## Dataset and extraction

Verify the input against the [dataset snapshot](dataset.md):
`data/mental_health.csv`, 10,000 data rows, and 51 source columns.
Record the checksum and confirm required-column validation runs before loading.
If selection or rejection changes the stored row/column counts, explain the
result using the implemented [ETL rules](etl-rules.md).

**Pending for Milestone 1:** add the implemented command with input and database
paths, expected successful output, and an example unreadable or invalid input.
A fresh clone must have sufficient instructions to reproduce the import.

## Schema and SQL evidence

Confirm `src/mh_risk_outreach/db/schema.sql` contains executable definitions,
and demonstrate that importing into a fresh database uses those definitions.
Inspect the resulting tables, column types, keys, and constraints; they must
match the [database documentation](database.md) and field mapping.

**Pending for Milestone 1:** add runnable SQL queries and sample results for:

- Stored record count and representative populated records.
- Relevant numeric, binary, and categorical features with the source target.
- Required-field and agreed value/constraint checks.
- Record identifiers and source-row traceability.
- Data available to later ML code, identifying target and metadata separately
  from candidate predictors.

Reconcile the stored count with accepted, rejected, and skipped source records.
Do not assume every source row is loaded if the documented rules reject rows.

## Repeat imports and failures

Run the same CSV twice and verify the chosen repeat-import policy. Check that
records do not accumulate unexpectedly and identifiers behave as documented.
Exercise a load failure and confirm the database state and summary match the
transaction and rollback policy. Successful status must follow a committed load.

## Streamlit display

**Pending for Milestone 1:** record the implemented launch command and shared
database path. Verify the GUI shows records queried from SQLite, with readable
column names and values, rather than reading the CSV directly. Compare sample
rows with SQL output. Confirm the UI explains a missing or empty database and
how to populate it. CSV upload is not a required feature.

## Processing log or summary

**Pending for Milestone 1:** document the output location and include successful
and failed run examples. Verify input identity, status, timing, source/accepted/
rejected/skipped/loaded counts, and failure reasons. Counts must reconcile with
the import rules and committed SQL records. Overlapping review flags must not
inflate discarded-row counts.

## Implementation tests

Add meaningful unit and integration checks for the documented behavior:

- Extraction accepts the selected CSV and detects missing required columns.
- Transformation applies type and normalization rules to valid and malformed
  values, missing fields, and explicit category responses.
- Database initialization enforces the tracked schema and accepts prepared rows.
- Duplicates, repeat imports, and failed loads follow the chosen policy.
- An end-to-end fixture produces the expected SQL records and summary counts.

Use synthetic fixtures for cases absent from the selected CSV. Record the test
commands and results here once implemented, and capture the Streamlit check
separately. ML training and evaluation are outside Milestone 1 acceptance.
