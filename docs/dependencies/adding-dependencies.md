# Adding dependencies

Run these commands from the project root, replacing `<package>` with the package name.

Add a project dependency required to run the application:

```bash
uv add <package>
```

Add a development dependency used for tasks such as linting, formatting, type checking, or testing:

```bash
uv add --dev <package>
```

Both commands update `pyproject.toml` and `uv.lock` and install the dependency into the project environment. Commit both files so other contributors use the same dependency versions.

For syncing the environment and running the application, see [Getting started](../../getting-started.md).
