#!/usr/bin/env bash
# Builds the project wheel for Pyodide, copies the shared assets and serves the browser GUI
# (web/) on port 8000. Everything generated lands in web/dist (gitignored).
set -euo pipefail
rm -rf web/dist
uv build --wheel --out-dir web/dist --quiet
cp src/imigraph/report/assets/viz.js src/imigraph/report/assets/report.css web/dist/
basename "$(ls web/dist/imigraph-*.whl)" > web/dist/wheel.txt
echo "imigraph GUI: http://localhost:8000  (wheel: $(cat web/dist/wheel.txt))"
exec uv run python -m http.server 8000 --directory web --bind 0.0.0.0
