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

   The second command is equivalent to
   `uv run streamlit run src/mh_risk_outreach/ui/app.py`. The display reads from
   `patients.db`; if it is empty, the page explains how to populate it.

5. Train and evaluate the risk models:

   With the database populated (step 4), train a model. It fits on the training
   split, evaluates on the held-out split, prints a report comparing it to the
   majority baseline, saves the fitted model under `artifacts/<name>/`, and
   writes the metrics to a results table named after the model:

   ```bash
   uv run python -m mh_risk_outreach.models.train --model logistic_regression
   uv run python -m mh_risk_outreach.models.train --model random_forest
   uv run python -m mh_risk_outreach.models.train --model xgboost
   ```

   Omit `--model` to train the default (logistic regression). See the
   [ML spec](docs/ml-spec.md) for the model framework, the metric definitions,
   and the measured results.

6. Score members and build the outreach worklist:

   With a model trained (step 5), score members and write the worklist to a
   `<model>_scores` table. Each row gets a `risk_score`, a risk band
   (`band_level_priority`), a strict `rank_priority`, and a short SHAP-based
   `reasoning`, with self-harm / suicidal members forced to the top:

   ```bash
   uv run python -m mh_risk_outreach.scoring --model logistic_regression
   uv run python -m mh_risk_outreach.scoring --model all
   ```

   By default this scores the held-out evaluation rows; add `--scope all` to
   score every stored member. See [Risk scoring](docs/risk-scoring.md) for the
   band, override, and reason rules.

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
