---
name: html-report
description: How imigraph produces its HTML outputs - the self-contained report (python -m imigraph report), the SVG plot helpers, the shared canvas script viz.js, and the Pyodide browser GUI under web/ including its constraints and how to verify pages headlessly. Use when changing anything a browser renders.
---

# HTML report and browser GUI

Two outputs share the same data and drawing code:

- **Report** (`imigraph.report.html.build_report`): one self-contained HTML file written to
  `output/` by `python -m imigraph report --label <change>`. **Every run creates a new
  timestamped file** (`brownian_vs_graph_<UTC time>_<label>.html`) and refreshes
  `output/index.html`; never overwrite or delete earlier pages, the maintainer wants the
  history of results. Generate a page for bigger changes (new system, rule, check, metric,
  visualisation), not for every commit. Tell the user the full path of the new file.
  Template `report/assets/report.html` uses
  `string.Template` placeholders (`$name`), so braces in CSS/JS need no escaping. Data goes
  into `<script id="data" type="application/json">`, `viz.js` and `report.css` are inlined.
  No network resources: the page must open from `file://`.
- **GUI** (`web/index.html`, `web/app.js`): loads Pyodide from jsDelivr, installs the project
  wheel from `web/dist/` (built by `docker/serve-web.sh`), calls
  `imigraph.report.live.run_json(config_json)` and renders the returned `data` and `fragments`.
  Served by `docker compose up web` at http://localhost:8000. The wheel is unpacked into
  site-packages with `pyodide.unpackArchive` (no micropip).
- **Self-contained GUI page** (`imigraph.report.bundle`, `python -m imigraph gui-page --label
  <change>`): `web/index.html` with `report.css`, `viz.js`, `app.js` and the wheel (base64 in
  `<script id="wheel">`) inlined, written as `output/rule_editor_<time>_<label>.html` and listed
  in the index. Opens from `file://`; only Pyodide, numpy and scipy come from the CDN. This is the
  file published to imigraph-reports so the maintainer can edit and test rules without a server.
  `--wheel` embeds an existing wheel, otherwise `uv build` runs. The verify run checks the newest
  one with the same GUI script (screenshots in `output/screenshots/rule-editor/`).

`imigraph.report.html.page_data()` and `fragments()` are the shared producers; `viz.js`
(`Viewport`, `drawParticles`, `drawSpace`, `drawWalkers`, `drawArrows`, `drawCollisionMarks`,
`drawRuleStep(vp, space, path, target, phase)`, `Player` with `setSpeed`, `speedControl`) is the
shared consumer.

- **Gas page** (`imigraph.report.gas`, template `assets/gas.html`, `python -m imigraph
  gas-report`): hard-disk gas and lattice gas side by side, the lattice-gas rule demo
  (`default_gas_demo`, arrows + collision rings), both check tables (`_checks_table(report,
  informative)` lists non-counting checks as `info: would pass/fail`), the cross-comparison and
  overlaid charts.
- **Graph-model Q&A** (`assets/graph-model.html`) is inserted into both pages as `$graph_model`;
  update it when the answers change (graph type, node types, how energy/momentum/entropy are
  computed, what collisions do).
- **Playground**: `imigraph.report.live.playground(request)` runs a small world from a
  starting universe (`size`, `walls`, `walkers`, `placement`, `sites`), a walker model
  (`walker_model` edge / hyperedge, `collisions` none / hpp, `heading` of hyperedge walkers,
  `overlaps` parallel / ordered / random / oldest for overlapping matches; see `graph/world.py`) and a rule set (`RuleSpec` JSON and built-in entries, a list) and
  returns the demo (matched chains, and `directions` + `collisions` per frame for hyperedge
  walkers), the rule descriptions (Collide first when on, each with an SVG `pattern` and the
  author's `description`), the `model`, the `docs` (the HTML of `assets/collisions.html` and `assets/overlaps.html`, the
  explanation of walker models, collisions and overlapping matches shown under the rules table
  on the GUI and the saved page), the resolved `universe`, measured vs predicted drift/diffusion (plus blocked
  hops and collision counts for hyperedge walkers), or `{"error": message}` for an
  unsupported rule or universe. `viz.js` draws hyperedge walkers with `drawArrows` and the
  collision sites with `drawCollisionMarks`. Hyperedge runs also return `captions` (one per
  frame from `world.explain_frames`: what Collide, Reflect and Stream did, the total momentum)
  shown under the step-by-step animation, and `headings` per walker may be given explicitly
  for clicked sites. Every view draws agents (walkers, balls, disks) with `viz.drawAgents`: one colour and
  number per agent, an arrow for a heading or a velocity, a ring when marked; the playground
  shows the graph and a 2D projection, the ball section the graph, the projection and the
  reference. The worked collision example is the head-on preset of the ball section. `web/app.js` keeps the rule list
  (`library`, persisted in `localStorage`), the graphical editor (`pattern`, `renderPattern`,
  `clickNode`; same layout as `svg.pattern_layout`), the text editor (`editorChanged`), the
  universe canvas (`drawUniverse`), export / import (rule, rule set, session) and *Save result
  page* (`live.playground_page_json`, a Blob download). "Use the rule set in the comparison
  run" writes the JSON list into the hidden `rule` field of the run form.
- **Balls bouncing in a room** (`live.balls(request)`, GUI section of that name): the ball
  gas (`experiments/balls.py`) next to the hard-disk reference from the same start; returns
  `data` (`report.balls.ball_page_data`: the graph run with sites per ball, centres,
  velocities, events, captions; the reference positions), `fragments`
  (`report.balls.ball_fragments`: checks tables, comparison table, charts), `docs`
  (`assets/balls.html`) and a `summary`. `web/app.js` draws the graph, the projection and the
  reference in `renderBalls`; the form `#balls` reruns on change. `live.newton` (one ball,
  `assets/newton.html`) stays as the single-ball core without a GUI section.
- **Playground page** (`imigraph.report.playground`, template `assets/playground.html`,
  `python -m imigraph playground-report --request-file session.json`): rules table with
  patterns, universe table and canvas, the two animations, measured vs predicted, the session
  JSON. Written as `output/playground_<time>_<label>.html`, listed in the index like the others.

## Plots

`imigraph.report.svg` renders line charts (with optional error bands), step histograms and
heatmaps as SVG strings, pure Python, so plots also work inside Pyodide. Add a chart type there
rather than pulling in a plotting library; keep the chart data in `CheckResult.data`.

## Pyodide constraints (web/)

- Pyodide 314.x = Python 3.14 with numpy 2.4, scipy 1.18, networkx 3.6. Code the GUI imports
  may use only packages Pyodide ships or pure-Python wheels, no threads, subprocesses, sockets
  or host files.
- The wheel is installed with `deps: false`; `numpy` and `scipy` are loaded explicitly in
  `app.js`. Import networkx lazily (as `to_networkx()` does) so it is not needed in the browser.
- Python runs on the main thread; keep default GUI parameters small (the default run takes a
  few seconds in Chromium). Use `setTimeout` before a long call so the status repaints.
- Pin the Pyodide version in `web/index.html` and update it deliberately (check the package
  versions in `https://cdn.jsdelivr.net/pyodide/v<version>/full/pyodide-lock.json`).

## Verify

```bash
docker compose run --rm -T dev uv run python -m imigraph report        # regenerate output/
docker compose run --rm -T dev node --check src/imigraph/report/assets/viz.js
docker compose run --rm -T dev node --check web/app.js
docker compose --profile verify down && docker compose --profile verify run --rm verify
# or, inside any container of the image (sandcastle sessions have no Docker):
uv run python docker/check-pages.py --serve
```

The `verify` run (`docker/check-pages.py`, headless Chromium) fails on JavaScript errors or
blank canvases and leaves screenshots in `output/screenshots/`; look at them. Tests in
`tests/test_report.py` and `tests/test_live.py` cover the Python side (page structure, embedded
JSON, CLI, the live entry point) without a browser.
