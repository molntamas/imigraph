# Plan: issue #1 "Initial vibe coded setup"

Issue: https://github.com/molntamas/imigraph/issues/1
Branch: `ai/issue-1`. Status of each phase is kept up to date at the bottom of this file.

## 1. What the issue asks for (first task)

1. Save requirements, documentation and steps into the repository.
2. Create skills that support the agent workflow.
3. A 2D Brownian-motion-in-a-box simulation as the *reference* system.
4. A graph-based simulation that imitates it, with
   - an initial condition and explicit evolution rules,
   - two graphical representations: the graph structure step by step, and a projection to 2D
     that should match the Brownian motion.
5. A reliable comparison framework between the graph system and the reference. This is the
   crucial part.
6. Guardrail tests so the simulations and their verification do not get slow.
7. Unit tests, software engineering best practices.
8. HTML pages the user can simply open to look at the results.

Working constraints given by the user (outside the issue):

- Nothing is installed on the Windows host (no npm packages, no uv packages, no browsers).
  Everything runs inside a Docker container; tools are installed through the Docker image.
- Sandcastle is not required for this round, but the result must keep working with it later
  (same image, same conventions, same `ai/issue-<n>` branch flow).

## 2. Decisions and assumptions

The issue says "start simple, iterate together" and "ask clarifying questions if needed". These
are the calls made for the first iteration; each is cheap to change later.

| Topic | Decision | Why |
| --- | --- | --- |
| Container | Reuse the existing `sandcastle:imigraph` image (built from `.sandcastle/Dockerfile`, already present locally) through a root `compose.yaml` with a `dev` service (tests, lint, scripts, git commits) and a `web` service (serves the browser GUI). | One image for humans, agents and sandcastle; nothing to install on the host; `.sandcastle/` stays untouched. |
| Reference system | Overdamped Brownian motion of `n` independent particles in a square box with reflecting walls, Euler–Maruyama: `x += sqrt(2 D dt) * xi`, `xi ~ N(0, 1)`. | The standard textbook model; analytic MSD `4 D t` before the walls matter, uniform stationary density. |
| Graph system | Space is a square lattice graph (`L x L` nodes, spacing `a`). Boundary nodes get self-loops so every node has degree 4. Each particle is a *walker node* attached by one edge to a space node. Rule, applied to all walkers synchronously each step: `{w, s}, {s, s'} -> {w, s'}, {s, s'}`, choosing `{s, s'}` uniformly among the 4 edges at `s`. | A genuine edge-rewriting rule in the Wolfram spirit (the graph changes every step), yet it is a lattice random walk, whose diffusion coefficient `D = a^2 / (4 tau)` is known exactly. Self-loops make the stationary distribution uniform, matching reflecting Brownian motion. |
| Projection | Walker position = coordinates of its anchored space node; box edges at `-a/2` and `(L-1) a + a/2`, so every node owns a full cell of width `a`. | Coarse-grained densities of both systems match exactly in the uniform state. |
| Time matching | The graph step `tau` and spacing `a` fix `D` for the reference. Comparison happens at observation times that span many graph steps (default: every 20 steps), because single lattice steps are not Gaussian; the central limit theorem makes them Gaussian after a few tens of steps. | Honest comparison: the discrete nature of the graph is real and documented; the imitation is a claim about coarse-grained behaviour. |
| Comparison framework | Both systems produce the same `Trajectory` (times, positions `(T, n, 2)`, box). Checks: (1) mean squared displacement curve, relative error + comparison with analytic `4 D t` in the free regime; (2) two-sample Kolmogorov–Smirnov tests on `x` and `y` marginals at the observation times (Bonferroni-corrected); (3) chi-square homogeneity test on 2D occupancy histograms at the final time. A `ComparisonReport` lists each check with value, threshold and pass/fail. | Mixes a physical observable (MSD) with distributional tests, each with a known null distribution. |
| Reliability of the framework | The framework is itself tested: reference vs reference with other seeds must pass (false positives); reference vs reference with `2 D`, with drift, or with a smaller box must fail (sensitivity); graph vs reference with matched `D` must pass; graph with mismatched `D` must fail. | "Reliable" means known false-positive and false-negative behaviour, not just "it ran". |
| Performance guardrails | `tests/test_performance.py` with wall-clock budgets (generous, roughly 3x the measured time on the dev container) for the reference run, the graph run, the comparison and the HTML report. Whole test suite target: under 30 s in the container. | The issue asks for guardrails; budgets catch accidental O(n^2) regressions without being flaky. |
| HTML output | A self-contained HTML page (inline data + vanilla JS canvas animations + SVG plots generated in Python, no CDN) written by `python -m imigraph report` into `output/` (gitignored). Opens via file://. Sections: reference animation, graph-structure animation, projection next to the reference, comparison plots and the check table. | "Open simply and check out the results" without a server. |
| Browser GUI (Pyodide) | `web/index.html` loads Pyodide 314.0.7 from jsDelivr, installs the project wheel (built into `web/dist/` by the `web` service) and runs the same Python in the browser with user-set parameters. Served by `docker compose up web` at http://localhost:8000. Kept minimal: run, animate, show the comparison table. | This is the project's stated GUI direction (CLAUDE.md); starting it now keeps the numerical core honest about Pyodide constraints. If it costs too much, it is the first thing to trim: the static report already covers the issue's HTML requirement. |
| Skills | Project skills in `.claude/skills/`: `docker-dev` (how to run everything in the container), `simulation-verification` (how to add reference systems, checks and tolerances), `graph-rules` (how to add spaces, rules, initial conditions), `html-report` (how to extend the report and the Pyodide page), `new-skill` (how to write a skill here). The Matt Pocock skills (`grill-me`, `wayfinder`) are vendored with the `skills` CLI run *inside the container* if it works cleanly; otherwise this is recorded as a follow-up. | The issue lists these areas explicitly; project skills are plain markdown, nothing to install. |
| Documentation | `docs/requirements.md` (the issue's brainstorm, cleaned up and kept as the living requirements document), `docs/architecture.md`, `docs/comparison-method.md`, this plan. README gets a "Development in Docker" section. | Issue item 1. |
| Git | Commits on `ai/issue-1` referencing `#1`, made from inside the container so pre-commit runs with the container's hook environments. No push, no pull request: that is the user's call (and later sandcastle's job). | CLAUDE.md conventions; pushing is outward-facing. |

Open questions for the user (answered by defaults above, easy to revisit):

1. Is a lattice random walk with an edge-rewriting rule an acceptable first graph model, or do you
   want the *space itself* to be grown by rules from the start? (Default: lattice first, growth
   later; the `Space` abstraction is the seam for it.)
2. Should the comparison be strict statistical tests (as planned) or tolerance-only? (Default: both,
   each check reports its own verdict.)
3. Should the Pyodide page be part of this round at all? (Default: yes, minimal.)

## 3. Architecture

```
src/imigraph/
  trajectory.py          Trajectory dataclass: times, positions (T, n, 2), box; msd(); to/from JSON
  reference/brownian.py  BrownianBox parameters + simulate(): the reference system
  graph/space.py         Space protocol + SquareLattice (neighbour table, coordinates, networkx view)
  graph/state.py         WalkerGraph: space + walker anchors; to_networkx(); edge list
  graph/rules.py         Rule protocol + RandomHop (the edge-rewriting rule)
  graph/simulation.py    GraphSimulation: initial condition, run(steps) -> GraphHistory; project() -> Trajectory
  compare/checks.py      MsdCheck, MarginalKsCheck, Occupancy2dCheck (+ Check protocol)
  compare/report.py      compare(reference, candidate, checks) -> ComparisonReport (dict / markdown)
  experiments/brownian_vs_graph.py  ExperimentConfig + run(): the end-to-end experiment
  report/svg.py          tiny SVG line/histogram plots (pure Python)
  report/html.py         self-contained HTML page writer
  report/assets/report.js, report.css   inlined into the page
  __main__.py / cli.py   python -m imigraph report [--out output] [--seed ...]
web/
  index.html, app.js, style.css   Pyodide GUI; dist/ holds the wheel (gitignored)
docs/      requirements.md, architecture.md, comparison-method.md
plans/     this file
.claude/skills/<name>/SKILL.md
compose.yaml, docker/entrypoint.sh
```

Numerical cores import only numpy and scipy (networkx only inside `to_networkx`), no I/O, no GUI,
so they run under pytest and in Pyodide unchanged.

## 4. Phases

Each phase ends with `pytest`, `ruff check`, `ruff format --check` passing in the container and a
commit referencing `#1`.

### Phase 0: Docker workflow
- `compose.yaml` (`dev`, `web`), `docker/entrypoint.sh` (`uv sync --frozen`, then exec).
- Verify: `docker compose run --rm dev uv run pytest` passes on the existing test; `pre-commit
  run --all-files` works in the container; a commit from the container runs the hooks.
- `.gitignore`: `output/`, `web/dist/`.

### Phase 1: Reference system
- `Trajectory` + `BrownianBox`.
- Tests: shape and determinism with a seed; particles stay inside the box; MSD matches `4 D t` within
  statistical tolerance in the free regime; long-time density is uniform (chi-square against uniform).

### Phase 2: Graph system
- `SquareLattice`, `WalkerGraph`, `RandomHop`, `GraphSimulation`.
- Tests: lattice degrees are all 4 (self-loops counted), coordinates and neighbour table agree with
  networkx `grid_2d_graph`; the rule moves every walker along an existing edge each step; walkers
  never leave the lattice; projected MSD matches `a^2 t / tau = 4 D t`; `to_networkx()` yields the
  expected node and edge counts.

### Phase 3: Comparison framework
- Checks, report, markdown/dict output.
- Tests: the reliability matrix from section 2 (pass/fail cases) with fixed seeds.

### Phase 4: Experiment, performance guardrails
- `experiments/brownian_vs_graph.py` with a default config that passes.
- `tests/test_performance.py` budgets.

### Phase 5: HTML report
- SVG plots, page writer, CLI. Generated into `output/brownian_vs_graph.html`.
- Tests: page is written, contains the data blob and the section ids, size is bounded;
  `node --check` on the JS asset (run manually in the container, Node is in the image).

### Phase 6: Pyodide GUI (minimal)
- Wheel build + static files, `web` compose service. Verified by loading the page in a browser if
  the Chrome automation skill is available; otherwise verified by running the same Python entry
  points natively and syntax-checking the JS.

### Phase 7: Docs, skills, README
- `docs/*.md`, `.claude/skills/*`, README section, vendored Matt Pocock skills if feasible.

### Phase 8: Wrap-up
- Final full verification in the container, commit, summary of what to look at and what is next.

## 5. Out of scope (this round)

- Hypergraphs, multiway systems, rule search / learning (sections 3 and 9 of the issue).
- Imitating kinematics or functions (section 4): the `Space`/`Rule` seams are built for it.
- Import/export beyond the JSON `Trajectory` format.
- Changes to `.sandcastle/`, `.github/` and `.pre-commit-config.yaml`.

## 6. Follow-ups suggested after this round

- Run the same issue through `npm run sandbox:run` once to confirm the sandcastle flow (the user
  asked for the whole setup to work with sandcastle eventually).
- Grown spaces: a rule that adds nodes/edges and a projection that is not given a priori
  (spectral embedding of the graph Laplacian is the natural first candidate).
- A second reference system (ballistic kinematics) to exercise the comparison framework with a
  non-diffusive target.

## 7. Status

- [ ] Phase 0 Docker workflow
- [ ] Phase 1 Reference system
- [ ] Phase 2 Graph system
- [ ] Phase 3 Comparison framework
- [ ] Phase 4 Experiment + performance guardrails
- [ ] Phase 5 HTML report
- [ ] Phase 6 Pyodide GUI
- [ ] Phase 7 Docs, skills, README
- [ ] Phase 8 Wrap-up
