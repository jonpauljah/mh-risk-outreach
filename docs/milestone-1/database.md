# Database architecture

SQLite is the chosen database for Milestone 1. The milestone must define its
schema in `src/mh_risk_outreach/db/schema.sql`, initialize a local database from
that file, and populate it through Python ETL. The current SQL file contains
comments only and the repository module is a placeholder; implementing both is
part of Milestone 1.

## Tables and fields

**Pending for Milestone 1:** document the chosen tables and every stored column,
including SQLite type, nullability, defaults, allowed values, and purpose.
Complete the [source-field mapping](etl-rules.md#source-to-database-mapping)
so the SQL definitions and transformed records agree. Store typed numeric and
binary values, readable categories, and the source target label separately in
purpose from predictors and metadata.

The dataset supplies survey rows without member identities. Define a prototype
record primary key and source-row traceability without implying a verified
patient identifier. Document uniqueness and any relationships between tables.
Add a schema diagram if multiple tables or relationships need explanation.

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

Keep connections, SQL, and record queries in `db/repository.py`. The UI must
explain an absent or unpopulated database and direct users to run the import.
SQLite files are generated locally and ignored by Git; `schema.sql` is tracked.

## Import lifecycle and consumer queries

The default database path is `config.DEFAULT_DB_PATH` (`patients.db`). The
repository creates missing schema tables from tracked `schema.sql`. Patient
imports run only when the `patients` table is empty; a populated table causes
the pipeline to skip before reading the CSV. The repository repeats this guard
under an immediate write transaction, and a failed initial insert rolls back
all inserted rows. Imports never replace the patient table. Duplicate matching
and handling are not implemented.

**Pending:** document how future schema changes require a rebuild or migration
rather than silently using an incompatible database. Do not replace explicitly
defined tables with pandas-inferred tables.

Provide example queries for record counts, displayed records, and features plus
the target for later ML consumption. Identify excluded metadata and the target
so consumers do not accidentally treat them as predictors. No model training
or model-specific encoding is required here.

See [data flow](data-flow.md), [ETL rules](etl-rules.md), and
[verification](verification.md) for implementation and evidence requirements.
