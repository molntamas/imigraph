"""Entry points of the browser GUI: the comparison run, the playground and the ball gas against
the hard-disk gas, each one call that returns page content as JSON.

Runs unchanged under CPython (tests) and under Pyodide (``web/app.js`` calls the ``*_json``
functions).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.experiments.balls import PLACEMENTS as BALL_PLACEMENTS
from imigraph.experiments.balls import BallGasConfig, run_ball_comparison
from imigraph.graph import SquareLattice, WalkerGraph
from imigraph.graph.rules import rule_from_spec
from imigraph.graph.velocity import DIRECTIONS
from imigraph.graph.world import WalkerSystem, explain_frames, rules_from_data
from imigraph.metrics import StepObserver, occupied_cells
from imigraph.report.html import (
    _asset,
    _demo_blob,
    default_demo,
    default_rule_demo,
    fragments,
    page_data,
)

PLAYGROUND_LIMITS = {
    "size": (3, 25),
    "walkers": (1, 50),
    "steps": (1, 300),
    "max_speed": (0, 4),
    "radius": (0, 2),
}
PLACEMENTS = ("center", "corner", "random", "left", "sites", "head-on", "diagonal")
WALLS = ("reflecting", "periodic")
VELOCITIES = ("rest", "unit", "random")
VELOCITY_TEXT = {
    "rest": "rest: every ball starts with velocity (0, 0), a random walker",
    "unit": "unit: every ball starts with a random unit velocity (the lattice gas)",
    "random": "random: integer velocities with components up to the max speed",
}
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
    "velocity": "rest",
    "max_speed": 2,
    "radius": 0,
    "velocities": None,
    "overlaps": "parallel",
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
    radius: int = 0,
) -> WalkerGraph:
    """The starting universe: where the balls sit before the first step.

    ``center``: all on the central site; ``corner``: all on site 0; ``random``: each on a
    random site; ``left``: each on a random site of the left half (a free expansion);
    ``sites``: exactly the given site indices, one ball per entry (what the GUI sends when
    sites were clicked); ``head-on``: two balls on the middle row at the left and right
    walls; ``diagonal``: two balls at the left and bottom walls that meet at right angles.
    With a ``radius`` the random placements keep the bodies apart.
    """
    size = lattice.size
    if placement == "sites":
        chosen = [int(s) for s in sites if 0 <= int(s) < lattice.num_nodes]
        if not chosen:
            raise ValueError("place at least one walker on a site of the world")
        return WalkerGraph(lattice, np.asarray(chosen, dtype=np.int64), None, radius)
    if placement == "center":
        return WalkerGraph(lattice, np.full(walkers, lattice.center_node), None, radius)
    if placement == "corner":
        return WalkerGraph(
            lattice, np.full(walkers, radius * size + radius), None, radius
        )
    if placement in ("head-on", "diagonal"):
        mid = size // 2
        if placement == "head-on":
            cells = [(radius, mid), (size - 1 - radius, mid)]
        else:
            cells = [(radius, mid), (mid, radius)]
        return WalkerGraph(
            lattice, np.array([i * size + j for i, j in cells]), None, radius
        )
    if placement in ("random", "left"):
        columns = max(1, size // 2) if placement == "left" else size
        i = np.arange(radius, columns - radius)
        j = np.arange(radius, size - radius)
        if radius == 0:
            anchors = rng.integers(0, columns, size=walkers) * size + rng.integers(
                0, size, size=walkers
            )
            return WalkerGraph(lattice, anchors)
        cells = np.array([(a, b) for a in i for b in j])
        cells = cells[
            (cells[:, 0] % (2 * radius + 1) == radius % (2 * radius + 1))
            & (cells[:, 1] % (2 * radius + 1) == radius % (2 * radius + 1))
        ]
        if cells.shape[0] < walkers:
            raise ValueError(
                f"{walkers} balls of radius {radius} do not fit into the placement region "
                f"({cells.shape[0]} places)"
            )
        chosen = cells[rng.choice(cells.shape[0], size=walkers, replace=False)]
        return WalkerGraph(lattice, chosen[:, 0] * size + chosen[:, 1], None, radius)
    raise ValueError(
        f"placement must be one of {', '.join(PLACEMENTS)}, not {placement!r}"
    )


def initial_velocities(
    count: int,
    velocity: str,
    max_speed: int,
    placement: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """The starting velocities of ``count`` balls, ``None`` for balls at rest."""
    if velocity not in VELOCITIES:
        raise ValueError(
            f"velocity must be one of {', '.join(VELOCITIES)}, not {velocity!r}"
        )
    if placement == "head-on":
        return np.array([[max(1, max_speed), 0], [-1, 0]] * (count // 2))[:count]
    if placement == "diagonal":
        return np.array([[1, 0], [0, 1]] * (count // 2))[:count]
    if velocity == "rest":
        return None
    if velocity == "unit":
        return DIRECTIONS[rng.integers(0, 4, size=count)]
    velocities = rng.integers(-max_speed, max_speed + 1, size=(count, 2))
    still = ~np.any(velocities != 0, axis=1)
    velocities[still, 0] = 1 if max_speed > 0 else 0
    return velocities


def playground(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """A small world with an editable rule set: every state, every visited site, and what the
    rules produce (measured against the specs' prediction). Fast enough to re-run on every
    edit.

    ``request`` keys: the universe (``size``, ``walls`` reflecting or periodic, ``walkers``,
    ``placement`` and ``sites``, see :func:`initial_universe`), the balls (``velocity`` rest,
    unit or random, ``max_speed``, ``radius``, or explicit ``velocities`` one per ball), the
    run (``steps``, ``seed``, ``spacing``, ``step_time``, ``overlaps``) and ``rule``: a
    :class:`RuleSpec` as JSON or a list of them (applied in order in every step) or ``None``
    for RandomHop. An unsupported rule or universe gives ``{"error": message}`` instead of
    raising, so the editor can show it.
    """
    options: dict[str, Any] = {**PLAYGROUND_DEFAULTS, **(request or {})}
    for key, (low, high) in PLAYGROUND_LIMITS.items():
        options[key] = int(min(max(int(options[key]), low), high))
    sites = list(options.get("sites") or [])[: PLAYGROUND_LIMITS["walkers"][1]]
    spacing, step_time = float(options["spacing"]), float(options["step_time"])
    rng = np.random.default_rng(int(options["seed"]))
    placement = str(options["placement"])
    try:
        if options["walls"] not in WALLS:
            raise ValueError(f"walls must be one of {', '.join(WALLS)}")
        if placement in ("head-on", "diagonal"):
            options["walkers"] = 2
        lattice = SquareLattice(
            options["size"], spacing, periodic=options["walls"] == "periodic"
        )
        system = WalkerSystem(
            lattice,
            rules_from_data(options["rule"]),
            step_time,
            str(options["overlaps"]),
            max(1, options["max_speed"]),
        )
        universe = initial_universe(
            lattice, placement, options["walkers"], sites, rng, options["radius"]
        )
        explicit = options.get("velocities")
        if explicit:
            velocities = np.asarray(explicit, dtype=np.int64)
            if velocities.shape != (universe.num_walkers, 2):
                raise ValueError(
                    f"{len(explicit)} velocities for {universe.num_walkers} walkers: give "
                    "one (vx, vy) per walker, or none"
                )
            if np.abs(velocities).max(initial=0) > PLAYGROUND_LIMITS["max_speed"][1]:
                raise ValueError(
                    f"velocity components must be at most {PLAYGROUND_LIMITS['max_speed'][1]}"
                )
        else:
            velocities = initial_velocities(
                universe.num_walkers,
                str(options["velocity"]),
                options["max_speed"],
                placement,
                rng,
            )
        initial = WalkerGraph(lattice, universe.anchors, velocities, universe.radius)
    except (ValueError, KeyError, TypeError) as error:
        return {"error": str(error)}
    observer = StepObserver()
    history, waited = system.run(initial, options["steps"], rng, observer)
    trajectory = history.project()
    total_time = float(trajectory.times[-1])
    displacement = trajectory.positions[-1] - trajectory.positions[0]
    occupied = occupied_cells(trajectory, lattice.size)
    descriptions = system.descriptions(spacing, step_time)
    moving = history.velocities is not None
    total = history.momentum_units() if moving else None
    demo = _demo_blob(history)
    demo["momentum"] = None if total is None else total.tolist()
    n = max(1, int(initial.num_walkers))
    return {
        "captions": explain_frames(history),
        "error": None,
        "rule": descriptions[-1],
        "rules": descriptions,
        "specs": system.rules_as_data(),
        "model": {
            "velocity": "explicit" if explicit else str(options["velocity"]),
            "max_speed": options["max_speed"],
            "radius": options["radius"],
            "overlaps": system.overlaps,
            "moving": moving,
            "interacting": system.interacting,
        },
        "docs": _asset("collisions.html") + _asset("overlaps.html"),
        "universe": {
            "size": lattice.size,
            "walls": options["walls"],
            "placement": placement,
            "sites": initial.anchors.tolist(),
            "velocities": None if velocities is None else initial.velocity.tolist(),
            "walkers": int(initial.num_walkers),
            "velocity": "explicit" if explicit else str(options["velocity"]),
            "max_speed": options["max_speed"],
            "radius": options["radius"],
            "overlaps": system.overlaps,
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
            "collisions": int(history.collisions().sum()),
            "collisions_by_rule": {
                k: int(v.sum()) for k, v in history.collisions_by_rule().items()
            },
            "bounces": int(history.bounces().sum()),
            "waiting_rate": waited / (options["steps"] * n),
            "momentum_start": None if total is None else total[0].tolist(),
            "momentum_end": None if total is None else total[-1].tolist(),
            "momentum_changes": None
            if total is None
            else int(np.sum(np.any(np.diff(total, axis=0) != 0, axis=1))),
            "energy": float(history.energy()[0]) if moving else None,
            "energy_constant": bool(np.all(history.energy() == history.energy()[0]))
            if moving
            else None,
        },
        "predicted": {
            "diffusion": system.diffusion(spacing, step_time).tolist(),
            "drift": system.drift(spacing, step_time).tolist(),
            "hops": system.hops,
        },
    }


BALL_LIMITS = {
    "size": (4, 20),
    "balls": (1, 20),
    "radius": (0, 2),
    "max_speed": (0, 4),
    "steps": (1, 300),
    "entropy_bins": (2, 10),
}
BALL_DEFAULTS: dict[str, Any] = {
    "size": 10,
    "balls": 6,
    "radius": 0,
    "max_speed": 2,
    "steps": 80,
    "seed": 0,
    "entropy_bins": 5,
    "rule": "exchange",
    "rules": None,
    "placement": "left",
}


def balls(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ball gas section of the GUI: several balls with integer velocities in a room under
    a collision rule set (a preset, or ``custom`` with the playground's ``rules``), run next
    to the hard-disk gas from the same start. Returns the page data of both systems
    (:func:`~imigraph.report.balls.ball_page_data`), the SVG fragments, the rules described,
    the explanation and a summary, or ``{"error": message}``."""
    from imigraph.report.balls import ball_fragments, ball_page_data

    options: dict[str, Any] = {**BALL_DEFAULTS, **(request or {})}
    for key, (low, high) in BALL_LIMITS.items():
        options[key] = int(min(max(int(options[key]), low), high))
    try:
        if options["placement"] not in BALL_PLACEMENTS:
            raise ValueError(f"placement must be one of {', '.join(BALL_PLACEMENTS)}")
        if options["placement"] in ("head-on", "diagonal"):
            options["balls"] = 2
        config = BallGasConfig(
            size=options["size"],
            balls=options["balls"],
            radius=options["radius"],
            max_speed=options["max_speed"],
            steps=options["steps"],
            seed=int(options["seed"]),
            entropy_bins=options["entropy_bins"],
            rule=str(options["rule"]),
            rules=tuple(options.get("rules") or ()),
            placement=str(options["placement"]),
        )
        comparison = run_ball_comparison(config)
    except (ValueError, KeyError, TypeError) as error:
        return {"error": str(error)}
    run = comparison.balls.run
    by_rule = {k: int(v.sum()) for k, v in run.collisions_by_rule().items()}
    return {
        "error": None,
        "data": ball_page_data(comparison),
        "fragments": ball_fragments(comparison),
        "rules": list(comparison.balls.rules),
        "docs": _asset("balls.html"),
        "summary": {
            "rule": config.rule,
            "rules": config.rule_names,
            "placement": config.placement,
            "balls": config.balls,
            "walkers_per_ball": run.walkers_per_ball,
            "size": config.size,
            "steps": config.steps,
            "collisions": int(run.collisions().sum()),
            "collisions_by_rule": by_rule,
            "bounces": int(run.bounces().sum()),
            "reference_collisions": int(comparison.gas.run.collisions.sum()),
            "passed": comparison.passed,
            "graph_passed": comparison.balls.passed,
            "reference_passed": comparison.gas.passed,
        },
    }


def balls_json(request_json: str = "{}") -> str:
    return json.dumps(balls(json.loads(request_json)), separators=(",", ":"))


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
