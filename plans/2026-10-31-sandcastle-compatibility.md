# Plan: keep the sandcastle image working with the solution

Request (2026-10-31): verify that the sandcastle image still works with the solution; if not,
adjust it (for example the Docker image). `.sandcastle/` may be changed for this task.

## 1. Baseline (before changes)

`npm run sandbox:smoke` on the current `main` (2026-10-31, image built 2026-10-29):

| check | result |
| --- | --- |
| container, worktree on `ai/smoke-*`, toolchain versions | ok (Claude Code 2.1.312, gh 2.103.3, uv 0.13.6, pre-commit 4.7.0, node 22) |
| `uv sync --frozen` hook, `uv run pytest` | ok, 96 passed in 21 s |
| `uv run ruff check . && uv run ruff format --check .` | ok |
| `gh auth status`, `gh issue list` | ok |
| `git push --dry-run` | **403**: the fine-grained token in `.sandcastle/.env` lacks *Contents: Read and write* (or expired). Token setup, not the image; see the README troubleshooting table. |

Not covered by the smoke test, and known to be broken for an agent run:

1. **Commits.** Sandcastle worktrees share `.git/hooks` with the host checkout, where
   `pre-commit install` wrote a CRLF hook with a Windows interpreter path. Git in the container
   cannot run it (`cannot run .git/hooks/pre-commit: No such file or directory`), so an agent's
   `git commit` fails. The `hooks.preCommit` sandbox hook only installs a hook if none exists.
   Also, betterleaks runs git without the global config and needs `safe.directory` for the bind
   mount.
2. **Page verification.** The report and GUI check (`docker/check-pages.py`) ran in a separate
   Playwright image through `docker compose --profile verify`; there is no Docker inside the
   sandbox, so a sandcastle agent changing anything a browser renders could not verify it, and the
   `html-report` skill told it to run a command that cannot work there.

## 2. Changes

| where | change |
| --- | --- |
| `.sandcastle/config.mts` | `sandbox()` passes `GIT_CONFIG_COUNT=2` with `safe.directory=*` and `core.hooksPath=docker/git-hooks` into the container (process environment only, nothing written to the shared `.git/config`), so commits use the LF hook of the repository and betterleaks trusts the mount. The `preCommit` hook is dropped (`pre-commit install` refuses to run with `core.hooksPath` set). |
| `.sandcastle/Dockerfile` | Chromium's system libraries (`npx playwright install-deps chromium`, root) and the Chromium build matching the locked `playwright` package (`playwright install chromium`, agent). |
| `pyproject.toml`, `uv.lock` | `playwright` in the `dev` dependency group, so `uv run python docker/check-pages.py` works in every container built from the image. |
| `docker/check-pages.py` | `--serve`: builds the wheel into `web/dist`, serves `web/` on a free local port and checks the GUI against it, so no second container is needed. |
| `compose.yaml` | The `verify` service uses the sandcastle image (`uv run python docker/check-pages.py`), `docker/verify.Dockerfile` removed: one image to maintain. |
| `.sandcastle/smoke.mts` | New checks: a temporary commit with the hooks (then reset), the report generation and the page check with `--serve`. |
| `.sandcastle/prompt.md`, skills, README | The verify step for browser-rendered changes inside the sandbox: `uv run python -m imigraph report && uv run python docker/check-pages.py --serve`. |

## 3. Verification

1. `npm run sandbox:build` rebuilds `sandcastle:imigraph` (also used by `compose.yaml`).
2. `npm run sandbox:smoke`: all checks including the new commit and page checks.
3. `docker compose --profile verify run --rm verify` still works from the host.
4. Optional, needs a working token: `npm run sandbox:smoke -- --claude` and one real issue with
   `npm run sandbox:run -- --issue <n>`.

## 4. Findings during verification

- First smoke run after the rebuild: the page check inside the sandbox works (report, gas page,
  GUI playground and comparison run through the in-container Chromium, GUI served by
  `--serve`). The commit check failed with `empty ident name (for <moln.tamas@gmail.com>)`:
  sandcastle copies `user.name` and `user.email` from the host into the container's global git
  config, but the name (which contains a space) arrived empty, while the email came from the
  repository's own `.git/config`. Fix: `config.mts` reads both on the host and forwards them as
  `GIT_AUTHOR_*` / `GIT_COMMITTER_*` environment variables, which need no quoting.
- `git push --dry-run` still answers 403 (`Permission to molntamas/imigraph.git denied`): the
  fine-grained token in `.sandcastle/.env` lacks *Contents: Read and write* or has expired.
  Nothing in the repository can fix that; regenerate the token (README, section 1 of the
  sandcastle guide) and rerun `npm run sandbox:smoke`.

## Status

- [x] baseline measured
- [x] changes made (commit `1ff9a19` plus the identity fix)
- [x] image rebuilt (4.0 GB with Chromium), smoke test: every check green except the push
      dry-run (token permissions, see above)
- [x] merged to main (2026-10-31); `docker compose --profile verify` re-verified with the rebuilt image
