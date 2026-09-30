# Project structure

Milestone 1 will ingest a healthcare CSV through a Python ETL pipeline, store
member records and relevant features in SQLite, and display the SQL records in
a lightweight Streamlit interface. The current scaffold reserves modules for
that work; functional implementation belongs in a separate issue/branch.

| Path | Intended responsibility |
| --- | --- |
| `src/mh_risk_outreach/__init__.py` | Current `main()` and CLI greeting |
| `src/mh_risk_outreach/cli.py` | Accept file paths and start the data import from the command line |
| `src/mh_risk_outreach/config.py` | Dataset paths, database paths, and configuration |
| `src/mh_risk_outreach/etl/extract.py` | CSV reading and required-column validation |
| `src/mh_risk_outreach/etl/transform.py` | Member field selection, cleaning, and normalization |
| `src/mh_risk_outreach/etl/pipeline.py` | Run the import steps in order and report the results |
| `src/mh_risk_outreach/db/schema.sql` | SQLite schema and constraints; currently comments only |
| `src/mh_risk_outreach/db/repository.py` | SQLite initialization, record loading, and queries |
| `src/mh_risk_outreach/ui/app.py` | Milestone 1 Streamlit display of SQL records |
| `tests/unit/test_main.py` | One baseline test asserting the current greeting |
| `tests/integration/` | Reserved for future pipeline and database tests |
| `tests/fixtures/` | Reserved for future synthetic test datasets |
| `data/` | Shared public healthcare datasets; see its README |

The CLI supports initial testing with `uv run mh-risk-outreach`. It currently
prints `Hello from mh-risk-outreach!`. Streamlit is the planned milestone 1 GUI;
its placeholder has no functionality yet. All placeholder Python modules contain
only documentation and can be imported without starting processing or opening
a database.

SQLite requires no database server. No schema, database files, CSV processing,
or ETL summary is created by this scaffold. Public datasets in `data/` can be
committed to Git. SQLite files and generated coverage reports are ignored.

Run the baseline test with `uv run pytest`; the pre-push hook runs this same
command. Collect coverage manually when needed:

```bash
uv run pytest --cov=mh_risk_outreach --cov-report=term-missing --cov-report=html
```

The test calls the existing `main()` and asserts its exact stdout, including the
trailing newline. No coverage threshold is configured. See
[development dependency commands](dependencies/dev-dependencies.md#local-commands)
for linting, formatting, and type checking.
