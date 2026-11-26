# Database architecture

SQLite is the local database. `src/mh_risk_outreach/db/schema.sql` defines the
imported survey table and the additive assessment and outreach history tables.
The ETL imports source rows into `patients`; history is created only by the
explicit assessment and case operations described below.

## Tables and fields

Milestone 3 adds `risk_assessments`, `outreach_cases`, and
`member_assessment_state` to the existing `patients` table. These tables are
created additively by `init_schema()`; initialization and population scoring do
not create assessment history. A baseline snapshot is created only as part of
the first case transaction. Every explicit reassessment appends a distinct
snapshot, even when no values changed. `inputs_json` contains exactly the 50
survey feature fields (not `Member_ID` or the source target label). A member
must already have assessment history to submit a reassessment; the first
assessment is always created atomically with their first case.

Assessment score fields are either all null with
`scoring_status='PendingScoring'`, or supplied together with
`scoring_status='Scored'`. The current pointer tracks the latest submitted
assessment, including one still pending. Scoring finalization accepts
caller-supplied score results and can change only that current pending row
once; recommendation metadata can remain absent until its engine is available.
`risk_score` is constrained to 0–100. No ML is invoked by the database layer.

Assessment snapshots retain the operational model name and version,
recommendation action and reason, and recommendation rule version when
available. Cases copy the original recommended action, reason, and applicable
versions. Selected and accepted actions, override justification, status,
outcome, and acceptance, outcome, and completion timestamps are separate case
fields. `select_case_action()`, `accept_case()`, `record_case_outcome()`, and
`complete_case()` enforce the [manager lifecycle](../milestone-3/architecture-and-workflows.md#case-lifecycle)
using SQLite write transactions.

Case creation reuses the current scored, unused assessment or creates the
on-demand baseline inside the first-case transaction. For a reused assessment,
the case action must match the stored recommendation action; a snapshot with no
recommendation action cannot open a case. `member_assessment_state` records the
current assessment ID and the UTC time its pointer was last advanced.

Each case references the assessment and member that opened it. A unique index
prevents an assessment from being used twice, and a partial unique index allows
only one Pending or Accepted case per member. Completed cases retain their
outcome and completion timestamp. SQLite triggers protect completed cases
from updates and enforce the permitted status transitions. Schema initialization
adds the triggers to existing databases without rewriting case history.
All new persistence connections enable SQLite foreign keys. SQLite checks JSON
syntax; the assessment writer validates the 50 field names and value types.
Member/current/case references are constrained by foreign keys.

`CREATE TABLE IF NOT EXISTS` is additive, but does not migrate an incompatible
preexisting definition of one of these new tables. A database with an
incompatible schema requires a deliberate migration.

`patients.Member_ID` identifies a prototype survey record, not a verified
person. The 50 input field names and their types are defined in `config.py` and
the SQL table; `Has_Mental_Health_Issue` is the separate source target, not an
input feature.

## Creation and access

1. `config.py` supplies the shared database path for ETL and Streamlit.
2. `db/repository.py` ensures the parent directory exists, opens SQLite, and
   initializes tables using the executable definitions in `db/schema.sql`.
3. `etl/pipeline.py` supplies transformed and validated records for loading.
4. Repository functions use parameterized writes or `to_sql()` into the existing
   tables, preserving schema constraints. Commit successful loads and roll back
   failed transactions according to the documented policy.
5. Streamlit queries the stored data through repository functions. Later ML
   code can query typed features and the target from the same database.

The existing import and score repositories preserve their current behavior.
The focused history access modules are `db/assessments.py` and
`db/outreach.py`. Each connection opened by this persistence layer enables
SQLite foreign-key enforcement. New database files are created locally and
ignored by Git; `schema.sql` is tracked.

## Import lifecycle and consumer queries

The default database path is `config.DEFAULT_DB_PATH` (`patients.db`). The
repository creates missing schema tables from tracked `schema.sql`. Patient
imports run only when the `patients` table is empty; a populated table causes
the pipeline to skip before reading the CSV. The repository repeats this guard
under an immediate write transaction, and a failed initial insert rolls back
all inserted rows. Imports never replace the patient table. Duplicate matching
and handling are not implemented.

`CREATE TABLE IF NOT EXISTS` adds missing tables but does not migrate an
incompatible preexisting definition. Such a database needs a deliberate
migration; these operations do not silently replace user data.

Provide example queries for record counts, displayed records, and features plus
the target for later ML consumption. Identify excluded metadata and the target
so consumers do not accidentally treat them as predictors. No model training
or model-specific encoding is required here.

See [data flow](data-flow.md), [ETL rules](etl-rules.md), and
[verification](verification.md) for implementation and evidence requirements.
