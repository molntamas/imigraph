# imigraph - project guide for coding agents

Imitating systems with graphs: computational physics and graph-based models in Python, plus a
browser GUI that runs the same Python code in WebAssembly (Pyodide/PyScript). Work happens locally,
in sandboxed interactive Claude Code sessions, and through GitHub issues that sandcastle agents
turn into pull requests (see `README.md`).

## Stack

- Python 3.14, managed with [uv](https://docs.astral.sh/uv/). Core libraries: numpy, scipy, networkx.
- Tests: pytest. Lint/format: ruff. Secret scanning: betterleaks via pre-commit.
- Source in `src/imigraph/`, tests in `tests/`.
- Browser GUI: PyScript/Pyodide, static files under `web/` (not created yet).

## Commands

```bash
uv sync                                               # install deps (dev group included)
uv run pytest                                         # tests
uv run ruff check . && uv run ruff format --check .   # lint + format check
uv add <package>                                      # add a dependency (pyproject.toml + uv.lock)
pre-commit run --all-files                            # what the commit hook runs
```

## Numerical code

- Vectorise with numpy; use `scipy.sparse` for graph operators (Laplacians, adjacency matrices).
- Test against analytic results or invariants (conservation, symmetry, known spectra) with
  explicit tolerances (`np.testing.assert_allclose(..., rtol=..., atol=...)`).
- Seed randomness explicitly: `rng = np.random.default_rng(seed)`, pass `rng` down.
- Keep numerical cores free of GUI and I/O code, so they run natively under pytest and in the browser.

## Code that runs in the browser (Pyodide)

Pyodide 314.x is Python 3.14 compiled to WebAssembly, with its own builds of numpy 2.4, scipy 1.18,
networkx 3.6, matplotlib, sympy and more
([package list](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)). Modules the GUI imports:

- may only depend on packages that Pyodide ships or on pure-Python wheels,
- must not use threads, subprocesses or sockets; file access only to Pyodide's in-memory filesystem,
- must not rely on APIs newer than Pyodide's numpy/scipy/networkx (`uv.lock` may pin newer versions).

## Rules

- The repository is public. Never commit secrets, tokens or `.env` files. Pre-commit must pass;
  never use `--no-verify`.
- One issue per branch (`ai/issue-<n>`). Reference the issue (`#<n>`) in commit messages.
- Do not change `.sandcastle/`, `.github/` or `.pre-commit-config.yaml` unless the task asks for it.
