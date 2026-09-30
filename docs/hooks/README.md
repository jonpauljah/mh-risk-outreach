# Pre-push hooks with prek

The repository's [prek.toml](../../prek.toml) defines built-in checks and
local hooks for Ruff, mypy, and pytest with coverage. Once installed in your checkout, these hooks
run before `git push`. A failed hook or a hook that modifies files stops the
push. Local commits do not run these hooks.

## Install prek and initialize hooks

Prek is a development dependency recorded in `pyproject.toml` and `uv.lock`.
From the repository root, with uv installed, install the development tools,
verify prek, and register the Git pre-push hook:

```bash
uv sync --group dev
uv run prek --version
uv run prek install --hook-type pre-push
```

After running `uv run prek install --hook-type pre-push`, you should see:

```text
Installed Git hook at `.git/hooks/pre-push`
```

Then verify the hooks by running:

```bash
uv run prek run --all-files --hook-stage pre-push
```

Use `uv run prek` to access the project's installed version. Each contributor
must register the Git pre-push hook in their own checkout, including after
a fresh clone. Once registered, that hook reads the current `prek.toml` on
each push. Pulling changes to hook definitions updates what runs without
requiring another installation. Adding a new Git hook stage requires
installing that stage separately.

If you previously installed prek's pre-commit hook for this project, remove
that registration so only pushes trigger checks:

```bash
uv run prek uninstall --hook-type pre-commit
```

## Available hooks

See the [hook manifest](hook-manifest.md) for all hook IDs, commands, file
selection, and coverage behavior.

Both uv and the project's environment must be available when pushing, including
from an editor or Git client. After recreating the environment, rerun
`uv run prek install --hook-type pre-push`.

## Passing the checks

Run the hooks before pushing to catch problems early:

```bash
uv run prek run --all-files --hook-stage pre-push
```

If a hook fails, follow the [instructions for fixing failures](hook-manifest.md#fixing-failures),
then rerun the checks before pushing.

To rerun an individual hook, use its ID:

```bash
uv run prek run mypy --all-files --hook-stage pre-push
uv run prek run pytest --all-files --hook-stage pre-push
```

Use `uv run prek list` to inspect configured hooks. For more diagnostic output:

```bash
uv run prek run --all-files --hook-stage pre-push -vvv
```

## Bypassing checks

To skip a specific hook for one push in Bash or a similar shell:

```bash
PREK_SKIP=pytest git push
```

Use comma-separated IDs to skip multiple hooks:

```bash
PREK_SKIP=mypy,pytest git push
```

To bypass the entire Git pre-push hook for one push:

```bash
git push --no-verify
```

A bypass does not fix the failure. Local hooks can be skipped or left
uninstalled; enforcing checks on GitHub requires CI and branch protection.

## Maintenance

To update the project's locked prek version, run `uv lock --upgrade-package prek`,
then `uv sync --group dev`. Commit the lockfile change when updating shared
tooling. After other dependency changes, sync the environment again. To remove
the local pre-push registration:

```bash
uv run prek uninstall --hook-type pre-push
```

See the official [local hook documentation](https://prek.j178.dev/local-hooks/)
and [running hooks guide](https://prek.j178.dev/running-hooks/) for more detail.

## Optional standalone installation

For macOS or Linux, the [official installation guide](https://prek.j178.dev/installation/)
also provides a curl installer:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/download/v0.5.4/prek-installer.sh | sh
```

Follow the installer's PATH instructions and verify with `prek --version`.
For a standalone installation, replace `uv run prek` in the commands above
with `prek`. You still need `uv sync --group dev` for the Python hook tools.
The standalone version is managed separately from `uv.lock`; update it with
`prek self update`. Prefer the project dependency setup for consistent tool
versions across contributors.
