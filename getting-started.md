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

   You do not need to activate the virtual environment. The current application prints `Hello from mh-risk-outreach!`.

5. Run the baseline test:

   ```bash
   uv run pytest
   ```

   The pre-push hook runs the test suite without coverage to keep routine checks
   lightweight. Generate coverage reports manually when needed:

   ```bash
   uv run pytest --cov=mh_risk_outreach --cov-report=term-missing --cov-report=html
   ```

   The scaffold contains one test for the current CLI greeting. SQLite and the
   milestone 1 Streamlit interface are placeholders awaiting implementation.
   See [project structure](docs/project-structure.md) for module responsibilities.

   See [development dependency commands](docs/dependencies/dev-dependencies.md#local-commands) for linting, formatting, type checking, and testing.
