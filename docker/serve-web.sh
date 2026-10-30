#!/usr/bin/env bash
# Builds the project wheel for Pyodide and serves the browser GUI (web/) on port 8000.
set -euo pipefail
rm -rf web/dist
uv build --wheel --out-dir web/dist
echo "imigraph GUI: http://localhost:8000"
exec uv run python -m http.server 8000 --directory web --bind 0.0.0.0
