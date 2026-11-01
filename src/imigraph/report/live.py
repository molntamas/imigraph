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
from imigraph.graph import SquareLattice, WalkerGraph, walkers_at_node
from imigraph.graph.rules import DIRECTION_NAMES, rule_from_spec
from imigraph.graph.world import (
    WalkerSystem,
    explain_frames,
    momentum,
    rules_from_data,
)
from imigraph.metrics import StepObserver, occupied_cells
from imigraph.report.html import (
    _asset,
    _demo_blob,
    default_demo,
    default_rule_demo,
    fragments,
    page_data,
)

PLAYGROUND_LIMITS = {"size": (3, 25), "walkers": (1, 50), "steps": (1, 300)}
PLACEMENTS = ("center", "corner", "random", "left", "sites")
WALLS = ("reflecting", "periodic")
PLAYGROUND_DEFAULTS: dict[str, Any] = {
    "size": 7,
    "walkers": 4,
    "steps": 40,
    "seed": 1,
    "spacing": 1.0,
    "step_time": 1.0,
    "walls": "reflecting",
    "placement": "center",
    "sites": [],
    "rule": None,
    "walker_model": "edge",
    "collisions": "none",
    "heading": "random",
    "headings": None,
}

# A world made to show a collision: two hyperedge walkers on a 4x4 world head straight at each
# other, two sites apart, with the HPP rules. Step 1 streams them onto the same site, step 2
# Collide turns them onto the perpendicular pair and Stream moves them apart; the walls then
# reflect them back and a second collision follows.
COLLISION_DEMO: dict[str, Any] = {
    "rule": [
        {
            "name": "Reflect",
            "builtin": "reflect",
            "description": "a walker heading into a wall turns round",
        },
        {
            "name": "Stream",
            "builtin": "stream",
            "description": "every walker moves one edge along its heading",
        },
    ],
    "walker_model": "hyperedge",
    "collisions": "hpp",
    "size": 4,
    "walls": "reflecting",
    "placement": "sites",
    "sites": [1, 9],
    "headings": ["+x", "-x"],
    "walkers": 2,
    "steps": 6,
    "seed": 1,
}


def run(
    config: dict[str, Any] | None = None, max_particles: int = 300
) -> dict[str, Any]:
    """Run the experiment for ``config`` (keyword arguments of :class:`ExperimentConfig`)."""
    result = run_experiment(ExperimentConfig(**(config or {})))
    demo = default_demo(seed=result.config.seed + 1)
    rule_demo_rule = (
        None if result.config.rule is None else rule_from_spec(result.config.rule)
    )
    rule_demo = default_rule_demo(rule=rule_demo_rule)
    return {
        "passed": result.passed,
        "markdown": result.report.to_markdown(),
        "data": page_data(result, demo, max_particles, rule_demo=rule_demo),
        "fragments": fragments(result),
    }


def run_json(config_json: str = "{}", max_particles: int = 300) -> str:
    """:func:`run` with JSON in and JSON out, the shape the JavaScript side uses."""
    return json.dumps(
        run(json.loads(config_json), max_particles), separators=(",", ":")
    )


def initial_universe(
    lattice: SquareLattice,
    placement: str,
    walkers: int,
    sites: list[int],
    rng: np.random.Generator,
) -> WalkerGraph:
    """The starting universe: where the walkers are anchored before the first step.

    ``center``: all walkers on the central site; ``corner``: all on site 0; ``random``: each
    on a uniformly random site; ``left``: each on a random site of the left half (a free
    expansion); ``sites``: exactly the given site indices, one walker per entry (what the GUI
    sends when sites were clicked).
    """
    if placement == "sites":
        chosen = [int(s) for s in sites if 0 <= int(s) < lattice.num_nodes]
        if not chosen:
            raise ValueError("place at least one walker on a site of the world")
        return WalkerGraph(lattice, np.asarray(chosen, dtype=np.int64))
    if placement == "center":
        return walkers_at_node(lattice, lattice.center_node, walkers)
    if placement == "corner":
        return walkers_at_node(lattice, 0, walkers)
    if placement == "random":
        return WalkerGraph(lattice, rng.integers(0, lattice.num_nodes, size=walkers))
    if placement == "left":
        columns = max(1, lattice.size // 2)
        i = rng.integers(0, columns, size=walkers)
        j = rng.integers(0, lattice.size, size=walkers)
        return WalkerGraph(lattice, i * lattice.size + j)
    raise ValueError(
        f"placement must be one of {', '.join(PLACEMENTS)}, not {placement!r}"
    )


def playground(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """A small world with an editable rule set: every state, every matched chain, and what the
    rules produce (measured against the specs' prediction). Fast enough to re-run on every edit.

    ``request`` keys: the universe (``size``, ``walls`` reflecting or periodic, ``walkers``,
    ``placement`` and ``sites``, see :func:`initial_universe`), the walker model
    (``walker_model`` edge or hyperedge, ``heading`` of hyperedge walkers at the start,
    ``collisions`` none or hpp, see :mod:`imigraph.graph.world`), the run (``steps``,
    ``seed``, ``spacing``, ``step_time``) and ``rule``: a :class:`RuleSpec` as JSON, a list of
    them and built-in entries (applied in order in every step) or ``None`` for RandomHop. An
    unsupported rule or universe gives ``{"error": message}`` instead of raising, so the
    editor can show it.
    """
    options: dict[str, Any] = {**PLAYGROUND_DEFAULTS, **(request or {})}
    for key, (low, high) in PLAYGROUND_LIMITS.items():
        options[key] = int(min(max(int(options[key]), low), high))
    sites = list(options.get("sites") or [])[: PLAYGROUND_LIMITS["walkers"][1]]
    spacing, step_time = float(options["spacing"]), float(options["step_time"])
    rng = np.random.default_rng(int(options["seed"]))
    try:
        if options["walls"] not in WALLS:
            raise ValueError(f"walls must be one of {', '.join(WALLS)}")
        lattice = SquareLattice(
            options["size"], spacing, periodic=options["walls"] == "periodic"
        )
        system = WalkerSystem(
            lattice,
            rules_from_data(options["rule"]),
            str(options["walker_model"]),
            str(options["collisions"]),
            step_time,
        )
        anchors = initial_universe(
            lattice, str(options["placement"]), options["walkers"], sites, rng
        ).anchors
        directions = None
        headings = options.get("headings")
        if headings and system.walker_model == "hyperedge":
            if len(headings) != anchors.shape[0]:
                raise ValueError(
                    f"{len(headings)} headings for {anchors.shape[0]} walkers: give one "
                    "heading per walker, or none"
                )
            unknown = [h for h in headings if h not in DIRECTION_NAMES]
            if unknown:
                raise ValueError(
                    f"headings must be +x, -x, +y or -y, not {unknown[0]!r}"
                )
            directions = np.array([DIRECTION_NAMES.index(h) for h in headings])
        initial = system.initial_state(
            anchors, str(options["heading"]), rng, directions
        )
    except (ValueError, KeyError, TypeError) as error:
        return {"error": str(error)}
    observer = StepObserver()
    history, blocked = system.run(initial, options["steps"], rng, observer)
    trajectory = history.project()
    total_time = float(trajectory.times[-1])
    displacement = trajectory.positions[-1] - trajectory.positions[0]
    occupied = occupied_cells(trajectory, lattice.size)
    descriptions = system.descriptions(spacing, step_time)
    hyper = system.walker_model == "hyperedge"
    total = momentum(history)
    demo = _demo_blob(history)
    demo["momentum"] = None if total is None else np.round(total, 6).tolist()
    return {
        "captions": explain_frames(history),
        "error": None,
        "rule": descriptions[-1],
        "rules": descriptions,
        "specs": system.rules_as_data(),
        "model": {
            "walker_model": system.walker_model,
            "collisions": system.collisions,
            "heading": str(options["heading"]) if hyper else None,
        },
        "docs": _asset("collisions.html"),
        "universe": {
            "size": lattice.size,
            "walls": options["walls"],
            "placement": options["placement"],
            "sites": anchors.tolist(),
            "directions": history.directions[0].tolist()
            if history.directions is not None
            else None,
            "walkers": int(anchors.shape[0]),
            "walker_model": system.walker_model,
            "collisions": system.collisions,
            "heading": str(options["heading"]) if hyper else None,
            "headings": heading_names(history) if hyper else None,
            "steps": options["steps"],
            "seed": int(options["seed"]),
            "spacing": spacing,
            "step_time": step_time,
        },
        "demo": demo,
        "measured": {
            "mean_squared_step": float(np.mean(observer.mean_squared)),
            "diffusion": float(trajectory.msd()[-1] / (4 * total_time)),
            "drift": (displacement.mean(axis=0) / total_time).tolist(),
            "wall_rate": float(np.mean(observer.wall_fraction)),
            "occupied_final": int(occupied[-1]),
            "collisions": int(sum(len(s) for s in history.collisions))
            if history.collisions is not None
            else None,
            "blocked_rate": blocked / (options["steps"] * max(1, int(anchors.shape[0])))
            if hyper
            else None,
            "momentum_start": None if total is None else total[0].tolist(),
            "momentum_end": None if total is None else total[-1].tolist(),
            "momentum_changes": None
            if total is None
            else int(np.sum(np.any(np.diff(total, axis=0) != 0, axis=1))),
        },
        "predicted": {
            "diffusion": system.diffusion(spacing, step_time).tolist(),
            "drift": system.drift(spacing, step_time).tolist(),
            "hops": system.hops,
        },
    }


def heading_names(history: Any) -> list[str] | None:
    if history.directions is None:
        return None
    return [DIRECTION_NAMES[int(d)] for d in history.directions[0]]


def collision_demo() -> dict[str, Any]:
    """The predefined collision world (:data:`COLLISION_DEMO`): its request, the playground
    result with the step captions, and the explanation page fragment."""
    return {
        "request": dict(COLLISION_DEMO),
        "result": playground(COLLISION_DEMO),
        "docs": _asset("collision-demo.html"),
    }


def collision_demo_json() -> str:
    return json.dumps(collision_demo(), separators=(",", ":"))


def playground_json(request_json: str = "{}") -> str:
    return json.dumps(playground(json.loads(request_json)), separators=(",", ":"))


def playground_page_json(request_json: str = "{}") -> str:
    """The self-contained result page of a playground run (what the GUI's *Save result page*
    downloads and ``python -m imigraph playground-report`` writes); ``""`` for an invalid
    request, whose message :func:`playground_json` reports."""
    from imigraph.report.playground import build_playground_page

    request = json.loads(request_json)
    result = playground(request)
    return "" if result["error"] else build_playground_page(result, request)
