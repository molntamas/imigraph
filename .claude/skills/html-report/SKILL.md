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
  Served by `docker compose up web` at http://localhost:8000.

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
- **Playground** (GUI only): `imigraph.report.live.playground(request)` runs a small world
  with an editable `RuleSpec` and returns the demo (with matched chains), the rule description,
  measured vs predicted drift/diffusion, or `{"error": message}` for an unsupported rule.
  `web/app.js` keeps the presets, the editor (`editorSpec` / `fillEditor`), export (Blob
  download `<name>.rule.json`) and import; "use in comparison run" writes the JSON into the
  hidden `rule` field of the run form.

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
```

The `verify` run (`docker/check-pages.py`, headless Chromium) fails on JavaScript errors or
blank canvases and leaves screenshots in `output/screenshots/`; look at them. Tests in
`tests/test_report.py` and `tests/test_live.py` cover the Python side (page structure, embedded
JSON, CLI, the live entry point) without a browser.
