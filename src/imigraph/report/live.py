"""Entry point of the browser GUI: one call that runs the experiment and returns page content.

Runs unchanged under CPython (tests) and under Pyodide (``web/app.js`` calls :func:`run_json`).
"""

from __future__ import annotations

import json
from typing import Any

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.report.html import default_demo, fragments, page_data


def run(config: dict[str, Any] | None = None, max_particles: int = 300) -> dict[str, Any]:
    """Run the experiment for ``config`` (keyword arguments of :class:`ExperimentConfig`)."""
    result = run_experiment(ExperimentConfig(**(config or {})))
    demo = default_demo(seed=result.config.seed + 1)
    return {
        "passed": result.passed,
        "markdown": result.report.to_markdown(),
        "data": page_data(result, demo, max_particles),
        "fragments": fragments(result),
    }


def run_json(config_json: str = "{}", max_particles: int = 300) -> str:
    """:func:`run` with JSON in and JSON out, the shape the JavaScript side uses."""
    return json.dumps(run(json.loads(config_json), max_particles), separators=(",", ":"))
