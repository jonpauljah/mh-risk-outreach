# Development dependencies

## Local commands

Run these commands from the project root. `uv run` installs the default development dependencies as needed; virtual environment activation is optional.

### Ruff

**[Ruff](https://docs.astral.sh/ruff/):** Lints and formats Python code and checks import sorting to maintain consistent code quality and style.

Check lint rules and import sorting:

```bash
uv run ruff check .
```

Apply available safe lint fixes, including import sorting:

```bash
uv run ruff check --fix .
```

Check formatting without changing files:

```bash
uv run ruff format --check .
```

Format Python code:

```bash
uv run ruff format .
```

### mypy

**[mypy](https://mypy.readthedocs.io/en/stable/):** Checks Python type annotations to catch type mismatches before execution.

Check application type annotations:

```bash
uv run mypy src/
```

### pytest

**[pytest](https://docs.pytest.org/en/stable/):** Runs automated tests for ETL transformations, database operations, and application behavior.

Run the test suite:

```bash
uv run pytest
```

### pytest-cov

**[pytest-cov](https://pytest-cov.readthedocs.io/en/stable/):** Adds line and branch coverage reporting to pytest to identify code paths that tests do not exercise.

Run tests with line and branch coverage, showing uncovered lines:

```bash
uv run pytest --cov=mh_risk_outreach --cov-branch --cov-report=term-missing
```

The pytest commands require tests to be added; pytest reports no tests collected until then.
