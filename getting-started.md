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

   uv automatically creates `.venv` and uses `uv.lock` to select dependency versions.

   If you need to install Python, follow the [Python installation guide](https://www.python.org/about/gettingstarted/) and install Python 3.12, the version configured for this project.

4. Run the project:

   ```bash
   uv run mh-risk-outreach
   ```

   You do not need to activate the virtual environment. The current application prints `Hello from mh-risk-outreach!`.
