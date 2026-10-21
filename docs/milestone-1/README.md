# Milestone 1

Milestone 1 implements the complete local ETL workflow: extract the selected
healthcare CSV, clean and normalize its relevant fields, define and populate a
SQLite database, and display the stored records in a lightweight Streamlit GUI.
The SQL data must be understandable to users and available for later ML work.
Schema design and ETL implementation are deliverables of this milestone.

## Deliverables

- Selected healthcare CSV with source, snapshot details, and limitations.
- Python extraction, transformation, and loading code with documented field mappings.
- Executable table definitions and constraints in `src/mh_risk_outreach/db/schema.sql`.
- Populated SQLite records and example SQL output showing relevant features.
- Lightweight Streamlit display reading those records from SQLite.
- Basic ETL log or summary with processing counts and failure reasons.
- Reproducible run instructions and checks demonstrating the workflow.

| Document | What it records |
| --- | --- |
| [Dataset](dataset.md) | Input snapshot, field meanings, selection rationale, and limitations |
| [ETL rules](etl-rules.md) | Field mapping, cleaning, validation, loading, and summary behavior |
| [Architecture](../architecture/README.md) | Modules, data flow, and database design |
| [Verification](verification.md) | Commands, SQL evidence, GUI checks, and acceptance criteria |

Architecture drafts stay in this milestone until the final deliverable, then
move into their appropriate documentation folders.

## Completion boundary

A successful run must create the database from the tracked SQL schema, process
the CSV using the documented rules, load the accepted records, report the
outcome, and make the resulting data visible in Streamlit. Designing the schema
or copying the CSV into an automatically inferred table alone is insufficient.

ML training, feature scaling, model-specific encoding, and evaluation belong to
later milestones. Milestone 1 stores typed features and the source target label
so later ML code can query them with an explicit separation between predictors
and target. It does not produce predictions or outreach scores.

The application is currently a scaffold. **Pending** items in these documents
are decisions or implementation evidence required before Milestone 1 is complete;
they are not deferred deliverables.

Return to the [documentation index](../README.md).
