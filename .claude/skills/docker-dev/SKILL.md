---
name: docker-dev
description: Run tests, lint, scripts, the HTML report, the browser GUI and git commits for imigraph inside the Docker container (nothing is installed on the host). Use whenever a task needs to execute anything in this repository.
---

# Working in the container

Nothing is installed on the host except Docker and git. Every command runs in the image that
sandcastle agents use too (`sandcastle:imigraph`, built from `.sandcastle/Dockerfile`), through
`compose.yaml`.

```bash
docker compose build dev                                       # once, or after Dockerfile / uv.lock changes
docker compose run --rm -T dev uv run pytest                   # tests
docker compose run --rm -T dev uv run ruff check .             # lint
docker compose run --rm -T dev uv run ruff format .            # format
docker compose run --rm -T dev pre-commit run --all-files      # what the commit hook runs
docker compose run --rm -T dev uv run python -m imigraph report   # new output/brownian_vs_graph_<time>.html + index.html
docker compose run --rm -T dev uv run python -m imigraph compare  # print the comparison report
docker compose run --rm -T dev bash -c 'cmd1 && cmd2'          # several commands in one container
docker compose up web                                          # GUI at http://localhost:8000
docker compose --profile verify run --rm verify                # headless check of report + GUI
```

- `-T` disables the pseudo-terminal; use it from scripts and agents, drop it for an interactive
  shell (`docker compose run --rm dev`).
- The repository is bind-mounted at `/home/agent/workspace`; the Python environment lives in the
  container (`/home/agent/venv`). Generated files (`output/`, `web/dist/`) appear on the host.
- `docker/entrypoint.sh` runs `uv sync --frozen` on every start, so dependency changes in
  `uv.lock` are picked up; rebuild the image only for larger changes.
- Ad-hoc Python: `docker compose run --rm -T dev uv run python - <<'EOF' ... EOF` (heredoc on
  stdin). Do not write scratch scripts into the repository.

## If Docker stops answering

The Rancher Desktop VM has about 5 GB of RAM and 2 CPUs. A runaway allocation inside a
container freezes the whole VM within seconds: `docker` commands then fail with "timed out
dialing Hyper-V socket" and every container command hangs. Recovery: `rdctl shutdown`, then
`rdctl start` (both in `%LOCALAPPDATA%\Programs\Rancher Desktop\resources\resources\win32\bin`;
`rdctl start` blocks the shell, launch it detached), then poll `docker info`. Prevention: run
new numerical code under a memory cap first (`resource.setrlimit(resource.RLIMIT_AS, ...)` plus
`faulthandler.dump_traceback_later`), log long runs to a file under `output/` instead of piping
through `tail`, and wrap docker calls in `timeout`.

## Committing

Commit from inside the container so the pre-commit hooks (betterleaks secret scan, ruff,
whitespace) run with the container's environments:

```bash
docker compose run --rm -T dev bash -c 'git add -A && git commit -m "feat(x): ... (#<issue>)"'
```

- The container uses `docker/git-hooks/pre-commit` (`core.hooksPath`), because the hook that
  `pre-commit install` writes on a Windows host has CRLF line endings and cannot run on Linux.
- Git identity is copied from the host's `~/.gitconfig` (user.name, user.email only).
- `GIT_CONFIG_*` environment variables in `compose.yaml` mark the bind mount as a safe directory
  even for tools that run git without the global config (betterleaks does).
- `git push` works from the host (it uses the host's credentials); inside the container it
  needs `GH_TOKEN` as sandcastle sets it.

## Verifying pages

After changing `imigraph.report`, `web/` or `viz.js`: regenerate the report, then run the
`verify` profile. It opens both pages in headless Chromium (separate Playwright image, not part
of the sandcastle image), fails on JavaScript errors or blank canvases, runs the GUI once and
writes screenshots to `output/screenshots/`. Look at the screenshots. The `web` service keeps
running after a `verify` run; `docker compose --profile verify down` stops it (needed to pick
up changed static assets).
