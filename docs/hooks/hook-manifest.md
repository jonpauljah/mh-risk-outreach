# Hook manifest

This document describes the hooks defined in [prek.toml](../../prek.toml).
For installation and usage instructions, see the [setup guide](README.md).

## Defined hooks

The built-in hooks run directly in prek without extra Python dependencies:

| Hook ID | Purpose |
| --- | --- |
| `check-toml` | Validate TOML syntax in project and hook configuration. |
| `check-merge-conflict` | Detect unresolved merge conflict markers, including outside an active merge or rebase (`--assume-in-merge`). |
| `detect-private-key` | Detect private keys; this is not a general secret scanner. |
| `check-added-large-files` | Reject selected files larger than 1,024 KiB. `--enforce-all` includes files already committed before pushing. |
| `check-case-conflict` | Detect paths that clash on case-insensitive filesystems. |
| `check-illegal-windows-names` | Detect filenames that Windows cannot use. |
| `trailing-whitespace` | Check trailing whitespace only in `.md` files, preserving Markdown hard line breaks. |
| `end-of-file-fixer` | Check that nonempty text files end with exactly one newline. |

The whitespace and final-newline hooks use `--check`, so they report problems
without editing files. Their behavior is described in the official
[built-in hook reference](https://prek.j178.dev/reference/built-in-hooks/).

The local hooks use the project's uv environment:

| Hook ID | Command | Purpose |
| --- | --- | --- |
| `ruff-fix` | `uv run ruff check --fix` | Fix supported lint issues, including import ordering. |
| `ruff-format` | `uv run ruff format` | Format Python code. |
| `ruff-check` | `uv run ruff check` | Report remaining lint errors. |
| `ruff-format-check` | `uv run ruff format --check` | Verify Python formatting. |
| `mypy` | `uv run mypy src` | Check types throughout the application source. |
| `pytest` | `uv run pytest --cov=mh_risk_outreach --cov-report=term-missing --cov-report=html` | Run the full test suite and measure application coverage. |

All hooks use the `pre-push` stage. Built-in hooks use `repo = "builtin"`;
local hooks use `repo = "local"` and `language = "system"`.
Git invokes prek, which reads the configuration and runs the commands
from the repository root. A failed hook or a hook that modifies files blocks the push. Local commits
do not trigger these hooks.

Ruff receives matching Python filenames selected by prek for the push.
Running with `--all-files` checks all tracked Python files instead. The Ruff
fixing hooks appear before the checking hooks. Mypy and pytest use
`pass_filenames = false` and `always_run = true`: they check the source and
run the full suite even when no Python files changed.

These hooks execute against your local checkout. Keep unrelated working-tree
changes out of the way when checking code you intend to push.

## Viewing coverage

The pytest hook prints coverage percentages and uncovered line numbers in
the terminal. Open `htmlcov/index.html` in a browser to explore coverage by
file. The `.coverage` data file and `htmlcov/` report directory are already
ignored by Git.

Coverage measures which application lines ran during tests; it does not
prove that assertions are sufficient. No minimum coverage percentage is
configured, so low coverage alone does not block a push.

## Fixing failures

### Final newline examples

This is a formatting consistency check. A final newline helps text tools
handle files consistently, keeps concatenated output from running together,
and avoids Git's "No newline at end of file" message. Removing extra trailing
blank lines keeps diffs tidy. A failure usually indicates a minor formatting
issue rather than an application bug.

The `end-of-file-fixer` hook uses `--check`, so it reports failures without
editing files. In these examples, `\n` represents an actual newline character:

| File contents | Result | Reason |
| --- | --- | --- |
| `hello\n` | Pass | Exactly one final newline. |
| `first\nsecond\n` | Pass | Multiple lines are fine; the file ends with one newline. |
| Empty file | Pass | Empty files are allowed. |
| `hello` | Fail | Missing final newline. |
| `hello\n\n` | Fail | Extra blank line at the end. |

Add a missing final newline or remove extra trailing blank lines, then save
and commit the file. Blank lines within the file are allowed.

### Resolving hook failures

Read the complete output and resolve each failure:

- **Built-in checks:** correct invalid configuration, resolve merge markers,
  remove private keys from tracked files, or rename conflicting or unsupported
  paths. Keep files over the size limit outside Git or use Git LFS where
  appropriate. Remove trailing whitespace and ensure a single final newline,
  then commit the corrections.
- **Ruff changed files:** inspect `git diff`, stage the intended fixes, commit
  them, and rerun the hooks before retrying the push.
- **Ruff lint errors remain:** correct the reported code manually; Ruff cannot
  automatically fix every lint rule.
- **Mypy errors:** fix the reported types or implementation. Mypy has no
  automatic fix mode.
- **Pytest failures:** fix the failing behavior or tests and rerun the suite.
  Pytest also fails when it collects no tests; this project needs a test suite
  before the hook can pass.
