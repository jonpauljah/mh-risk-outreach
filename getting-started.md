# Getting started

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Restart your terminal so the `uv` command is available.

2. Open the project folder:

   ```bash
   cd /path/to/mh-risk-outreach
   ```

   Replace the path with your local project's location.

3. Install the project and its dependencies from `pyproject.toml`:

   ```bash
   uv sync
   ```

   uv automatically creates `.venv` and uses `uv.lock` to select dependency versions. Project and development dependencies share this environment, and development dependencies are included by default.

   If you need to install Python, follow the [Python installation guide](https://www.python.org/about/gettingstarted/) and install Python 3.12, the version configured for this project.

4. Run the project:

   Run with project dependencies, excluding the development dependency group:

   ```bash
   uv run --no-dev mh-risk-outreach
   ```

   Run with both project and development dependencies available:

   ```bash
   uv run mh-risk-outreach
   ```

   You do not need to activate the virtual environment. `uv run mh-risk-outreach`
   launches the Streamlit display.

   First import the dataset into SQLite, then open the display:

   ```bash
   uv run python -m mh_risk_outreach.etl.pipeline
   uv run mh-risk-outreach
   ```

   The import initializes an empty `patients` table. If it already contains
   rows, the import reports a skip before reading the CSV and preserves the
   stored patients and related history.

   The second command is equivalent to
   `uv run streamlit run src/mh_risk_outreach/ui/app.py`. The display reads from
   `patients.db`; if it is empty, the page explains how to populate it.

5. Train and evaluate the risk models:

   With the database populated (step 4), train a model. It fits on the training
   split, evaluates on the held-out split, prints a report comparing it to the
   majority baseline, saves the fitted model under
   `artifacts/<name>/versions/<version>/`, and writes the metrics to a results
   table named after the model. The commands below train each model in turn.

   ```bash
   uv run python -m mh_risk_outreach.models.train --model logistic_regression
   uv run python -m mh_risk_outreach.models.train --model random_forest
   uv run python -m mh_risk_outreach.models.train --model xgboost
   ```

   Omit `--model` to train the default (logistic regression). See the
   [ML spec](docs/ml-spec.md) for the model framework, the metric definitions,
   and the measured results. Retrain each model you intend to score if its
   saved artifact predates the versioned band metadata. The Risk scoring page
   offers **Train** for outdated models. The current models use 33 source
   features and 42 prepared columns, with safety flags reviewed separately.
   The Model evaluation page compares all three on the held-out set and its
   unflagged subset, including positive labels among the top 100 unflagged
   members for each model.

6. Score members and build the outreach worklist:

   SQLite retains all 10,000 imported members. The 8,000 training members
   supply model fitting, risk bands, and SHAP background. The original 2,000
   held-out members supply model evaluation and make up the fixed operational
   population. Operational scoring refreshes those 2,000 members and outreach
   uses a separately configured shared capacity K.

   ```bash
   uv run python -m mh_risk_outreach.scoring --operational
   ```

   This replaces the SQLite `operational_scores` worklist. Logistic regression
   is the default. Set `MH_RISK_OUTREACH_OPERATIONAL_MODEL` before running
   to select a different trained model, for example `random_forest`. Each row
   includes the model version, score, model-derived band and SHAP explanation,
   safety-review status, and contact rank. Members requiring safety review
   come first, followed by descending scores compared at 10 decimal places
   and ascending IDs to resolve ties.

   For an existing database, run a read-only scope and history audit.

   ```bash
   uv run python -m mh_risk_outreach.scoring --audit-operational
   ```

   If the audit reports an old or mismatched worklist, rerun `--operational`.
   Review any out-of-scope history reported by the audit. See the
   [recovery procedure](docs/risk-scoring.md#recovering-an-existing-database).

   Use `--model` to score comparison models separately.

   ```bash
   uv run python -m mh_risk_outreach.scoring --model logistic_regression
   uv run python -m mh_risk_outreach.scoring --model all
   ```

   These commands default to the held-out members and write the corresponding
   `<model>_scores` tables. Add `--scope all` to compare over every stored
   member. On the Risk scoring page, **Run** replaces the selected models'
   held-out scores and displays the saved rows. Rerun scoring to refresh
   comparison tables written by earlier versions. **Reset** clears the page
   display and leaves the SQLite tables available. See
   [Risk scoring](docs/risk-scoring.md).

   An import cannot replace or reset an existing dataset. Reimporting the same
   data is safe and skips; the prototype has no dataset reset or replacement
   workflow.

7. Run the baseline test:

   ```bash
   uv run pytest
   ```

   The pre-push hook runs the test suite without coverage to keep routine checks
   lightweight. Generate coverage reports manually when needed:

   ```bash
   uv run pytest --cov=mh_risk_outreach --cov-report=term-missing --cov-report=html
   ```

   The suite includes a test that `main()` launches the Streamlit display.
   See [project structure](docs/milestone-1/project-structure.md) for module responsibilities.

   See [development dependency commands](docs/dependencies/dev-dependencies.md#local-commands) for linting, formatting, type checking, and testing.

8. Initialize the pre-push hooks:

   ```bash
   uv run prek install --hook-type pre-push
   ```

   Each contributor must run this once in their checkout, including after a
   fresh clone. The hooks run the configured checks before `git push`.
   Verify the setup by running:

   ```bash
   uv run prek run --all-files --hook-stage pre-push
   ```

   See the [hook setup guide](docs/hooks/README.md) for details and troubleshooting.
