"""Entry points of the browser GUI: the comparison run and the playground, each one call that
returns page content as JSON.

Runs unchanged under CPython (tests) and under Pyodide (``web/app.js`` calls the ``*_json``
functions).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.graph import GraphSimulation, SquareLattice, walkers_at_node
from imigraph.graph.rules import RANDOM_HOP, RuleSpec, rule_from_spec
from imigraph.metrics import StepObserver, occupied_cells
from imigraph.report.html import (
    _demo_blob,
    default_demo,
    default_rule_demo,
    fragments,
    page_data,
)

PLAYGROUND_LIMITS = {"size": (3, 25), "walkers": (1, 50), "steps": (1, 300)}
PLAYGROUND_DEFAULTS = {
    "size": 7,
    "walkers": 4,
    "steps": 40,
    "seed": 1,
    "spacing": 1.0,
    "step_time": 1.0,
    "rule": None,
}


def run(config: dict[str, Any] | None = None, max_particles: int = 300) -> dict[str, Any]:
    """Run the experiment for ``config`` (keyword arguments of :class:`ExperimentConfig`)."""
    result = run_experiment(ExperimentConfig(**(config or {})))
    demo = default_demo(seed=result.config.seed + 1)
    rule_demo_rule = None if result.config.rule is None else rule_from_spec(result.config.rule)
    rule_demo = default_rule_demo(rule=rule_demo_rule)
    return {
        "passed": result.passed,
        "markdown": result.report.to_markdown(),
        "data": page_data(result, demo, max_particles, rule_demo=rule_demo),
        "fragments": fragments(result),
    }


def run_json(config_json: str = "{}", max_particles: int = 300) -> str:
    """:func:`run` with JSON in and JSON out, the shape the JavaScript side uses."""
    return json.dumps(run(json.loads(config_json), max_particles), separators=(",", ":"))


def playground(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """A small world with an editable rule: every state, every matched chain, and what the rule
    produces (measured against the spec's prediction). Fast enough to re-run on every edit.

    ``request`` keys: ``size``, ``walkers``, ``steps``, ``seed``, ``spacing``, ``step_time`` and
    ``rule`` (a :class:`RuleSpec` as JSON, or ``None`` for RandomHop). An unsupported rule gives
    ``{"error": message}`` instead of raising, so the editor can show it.
    """
    options = {**PLAYGROUND_DEFAULTS, **(request or {})}
    for key, (low, high) in PLAYGROUND_LIMITS.items():
        options[key] = int(min(max(int(options[key]), low), high))
    try:
        spec = RANDOM_HOP if options["rule"] is None else RuleSpec.from_dict(options["rule"])
        rule = rule_from_spec(spec)
    except (ValueError, KeyError, TypeError) as error:
        return {"error": str(error)}
    spacing, step_time = float(options["spacing"]), float(options["step_time"])
    lattice = SquareLattice(options["size"], spacing)
    simulation = GraphSimulation(lattice, rule, step_time)
    initial = walkers_at_node(lattice, lattice.center_node, options["walkers"])
    observer = StepObserver()
    rng = np.random.default_rng(int(options["seed"]))
    history = simulation.run(initial, options["steps"], rng, 1, observer, record_paths=True)
    trajectory = history.project()
    total_time = float(trajectory.times[-1])
    displacement = trajectory.positions[-1] - trajectory.positions[0]
    occupied = occupied_cells(trajectory, lattice.size)
    return {
        "error": None,
        "rule": rule.describe(spacing, step_time),
        "spec": spec.to_dict(),
        "demo": _demo_blob(history),
        "measured": {
            "mean_squared_step": float(np.mean(observer.mean_squared)),
            "diffusion": float(trajectory.msd()[-1] / (4 * total_time)),
            "drift": (displacement.mean(axis=0) / total_time).tolist(),
            "wall_rate": float(np.mean(observer.wall_fraction)),
            "occupied_final": int(occupied[-1]),
        },
        "predicted": {
            "diffusion": spec.diffusion(spacing, step_time).tolist(),
            "drift": spec.drift(spacing, step_time).tolist(),
            "hops": spec.hops,
        },
    }


def playground_json(request_json: str = "{}") -> str:
    return json.dumps(playground(json.loads(request_json)), separators=(",", ":"))
