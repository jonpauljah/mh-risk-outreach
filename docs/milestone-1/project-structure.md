# Project structure

Milestone 1 will ingest a healthcare CSV through a Python ETL pipeline, store
member records and relevant features in SQLite, and display the SQL records in
a lightweight Streamlit interface. The current scaffold reserves modules for
that work. Implementing these modules and the SQL schema is part of Milestone 1.

| Path | Intended responsibility |
| --- | --- |
| `src/mh_risk_outreach/__init__.py` | `main()` entry point; launches the Streamlit display |
| `src/mh_risk_outreach/cli.py` | Accept file paths and start the data import from the command line |
| `src/mh_risk_outreach/config.py` | Dataset paths, database paths, and configuration |
| `src/mh_risk_outreach/etl/extract.py` | CSV reading and required-column validation |
| `src/mh_risk_outreach/etl/transform.py` | Member field selection, cleaning, and normalization |
| `src/mh_risk_outreach/etl/pipeline.py` | Run the import steps in order and report the results |
| `src/mh_risk_outreach/db/schema.sql` | Executable SQLite tables and constraints to implement in Milestone 1; currently comments only |
| `src/mh_risk_outreach/db/repository.py` | SQLite initialization, record loading, and queries |
| `src/mh_risk_outreach/ui/app.py` | Milestone 1 Streamlit display of SQL records |
| `tests/unit/test_main.py` | Test asserting `main()` launches the Streamlit display |
| `tests/integration/` | Reserved for future pipeline and database tests |
| `tests/fixtures/` | Reserved for future synthetic test datasets |
| `data/` | Shared public healthcare datasets; see its README |

`uv run mh-risk-outreach` launches the Streamlit display (equivalent to
`uv run streamlit run src/mh_risk_outreach/ui/app.py`). The data import is the
separate command `uv run python -m mh_risk_outreach.etl.pipeline`.

SQLite requires no database server. No schema, database files, CSV processing,
or ETL summary is created by this scaffold. Public datasets in `data/` can be
committed to Git. SQLite files and generated coverage reports are ignored.

Run the baseline test with `uv run pytest`; the pre-push hook runs this same
command. Collect coverage manually when needed:

```bash
uv run pytest --cov=mh_risk_outreach --cov-report=term-missing --cov-report=html
```

The test calls `main()` with Streamlit's CLI stubbed and asserts it is invoked
to run `ui/app.py`, without starting a server. No coverage threshold is
configured. See
[development dependency commands](../dependencies/dev-dependencies.md#local-commands)
for linting, formatting, and type checking.
