"""Running a graph system: initial condition + rule -> history of states -> projected trajectory.

:class:`GraphHistory` is the record of every graph run, whether the balls are at rest (the
Brownian imitation: anchors only) or moving (velocities, the events of the rules, the exact
invariants of the ball gas). :class:`GraphSimulation` runs a single chain rule vectorised, the
Brownian experiment's loop; :class:`~imigraph.graph.world.WalkerSystem` runs rule sets with
rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imigraph.graph.rules import RandomHop, Rule
from imigraph.graph.space import Space
from imigraph.graph.state import WalkerGraph, lattice_cells
from imigraph.graph.velocity import disc_offsets
from imigraph.metrics import StepObserver, occupancy_entropy
from imigraph.reference.gas import GasRun
from imigraph.trajectory import Box, Trajectory


def walkers_at_node(space: Space, node: int, count: int) -> WalkerGraph:
    """Initial condition: ``count`` walkers all anchored at ``node``."""
    return WalkerGraph(space, np.full(count, node, dtype=np.int64))


def lattice_hop_diffusion(spacing: float, step_time: float) -> float:
    """Diffusion coefficient of :class:`RandomHop` on a square lattice.

    Each step moves one coordinate by ``+-spacing`` (or stays at a self-loop), so after ``k``
    free steps ``<|r|^2> = k spacing^2``; with ``t = k step_time`` and ``<|r|^2> = 4 D t`` in two
    dimensions, ``D = spacing^2 / (4 step_time)``.
    """
    return spacing**2 / (4 * step_time)


@dataclass(frozen=True)
class GraphHistory:
    """Recorded states of a graph run: the walker anchors at every recorded step.

    ``anchors`` has shape ``(T, num_walkers)``; frame 0 is the initial condition. The space is
    static, so the full graph at frame ``k`` is :meth:`state` ``(k)``.
    """

    space: Space
    times: np.ndarray
    anchors: np.ndarray
    paths: np.ndarray | None = None
    """Matched chains per step when recorded: shape ``(T - 1, n, k + 1)``, ``paths[t, w]`` is
    the chain of sites the rules matched for walker ``w`` in the step leading to frame
    ``t + 1`` (the sites visited hop by hop; only for runs recorded at every step)."""
    velocities: np.ndarray | None = None
    """Velocities (cells per step) of every ball at every frame, shape ``(T, n, 2)``; ``None``
    for a run of balls at rest."""
    events: tuple[tuple[dict[str, Any], ...], ...] | None = None
    """Per recording interval, the events of the rules that changed a velocity: collisions
    (two balls) and bounces (one ball), each with the rule's name, the balls, their sites and
    velocities before and after."""
    radius: int = 0
    mass: float = 1.0
    step_time: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        anchors = np.asarray(self.anchors, dtype=np.int64)
        if anchors.ndim != 2 or times.ndim != 1 or anchors.shape[0] != times.shape[0]:
            raise ValueError("anchors must have shape (T, n) matching T times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "anchors", anchors)
        if self.paths is not None:
            paths = np.asarray(self.paths, dtype=np.int64)
            if paths.shape[:2] != (anchors.shape[0] - 1, anchors.shape[1]):
                raise ValueError("paths must have shape (T - 1, n, k + 1)")
            object.__setattr__(self, "paths", paths)
        if self.velocities is not None:
            velocities = np.asarray(self.velocities, dtype=np.int64)
            if velocities.shape != (*anchors.shape, 2):
                raise ValueError("velocities must have shape (T, n, 2)")
            object.__setattr__(self, "velocities", velocities)
        if self.events is not None:
            events = tuple(tuple(step) for step in self.events)
            if len(events) != anchors.shape[0] - 1:
                raise ValueError("events must list every recording interval")
            object.__setattr__(self, "events", events)
        if self.step_time <= 0 or self.mass <= 0:
            raise ValueError("step_time and mass must be > 0")

    @property
    def num_frames(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def num_walkers(self) -> int:
        return int(self.anchors.shape[1])

    @property
    def walkers_per_ball(self) -> int:
        return int(disc_offsets(self.radius).shape[0])

    @property
    def moving(self) -> bool:
        return self.velocities is not None and bool(np.any(self.velocities != 0))

    @property
    def velocity(self) -> np.ndarray:
        """The velocities, zeros for balls at rest; shape ``(T, n, 2)``."""
        if self.velocities is None:
            return np.zeros((*self.anchors.shape, 2), dtype=np.int64)
        return self.velocities

    @property
    def speed_unit(self) -> float:
        """One cell per step in box units per time: ``a / tau``."""
        return float(self.space.cell_size[0] / self.step_time)

    def state(self, frame: int) -> WalkerGraph:
        return WalkerGraph(
            self.space,
            self.anchors[frame],
            None if self.velocities is None else self.velocities[frame],
            self.radius,
        )

    def sites(self) -> np.ndarray:
        """The body sites of every ball at every frame, shape ``(T, n, k)``."""
        if self.radius == 0:
            return self.anchors[:, :, None]
        return np.stack([self.state(f).sites() for f in range(self.num_frames)])

    def centres(self) -> np.ndarray:
        """Centre of every ball in box coordinates at every frame, shape ``(T, n, 2)``."""
        return self.space.coordinates[self.anchors]

    def project(self, cell_rng: np.random.Generator | None = None) -> Trajectory:
        """The walkers' 2D positions over time: the graph system's imitation of the reference.

        By default a walker sits exactly on the coordinates of its node. With ``cell_rng`` each
        walker is instead placed at a fixed random offset inside the cell of its node (uniform in
        ``[-spacing/2, spacing/2]^2``, drawn once per walker): the graph then claims to imitate
        the reference *at the resolution of one cell*, which is the statement the distribution
        tests of the comparison framework check. Displacements, and therefore the mean squared
        displacement, are unaffected by the offset.
        """
        positions = self.space.coordinates[self.anchors]
        if cell_rng is not None:
            cell = self.space.cell_size
            offsets = cell_rng.uniform(-0.5, 0.5, size=(self.num_walkers, 2)) * cell
            positions = positions + offsets[None, :, :]
        return Trajectory(self.times, positions, self.space.box)

    # ---------- the ball gas's bookkeeping ----------

    def energy(self) -> np.ndarray:
        """Total kinetic energy ``k m |v|^2 / 2`` per frame (``k`` nodes per ball)."""
        v2 = np.sum((self.velocity * self.speed_unit) ** 2, axis=2)
        return 0.5 * self.mass * self.walkers_per_ball * v2.sum(axis=1)

    def momentum(self) -> np.ndarray:
        """Total momentum ``k m v`` per frame, shape ``(T, 2)``."""
        return (
            self.mass * self.walkers_per_ball * (self.velocity * self.speed_unit)
        ).sum(axis=1)

    def momentum_units(self) -> np.ndarray:
        """Total momentum in units of ``k m a / tau``: the integer sum of the velocities."""
        return self.velocity.sum(axis=1)

    def to_gas_run(self) -> GasRun:
        """The run in the shape of a hard-disk gas run, so the gas checks apply: the centres
        as positions, the velocities in ``a / tau``, the mass of a whole ball."""
        return GasRun(
            self.project(),
            self.velocity * self.speed_unit,
            self.collisions(),
            self.mass * self.walkers_per_ball,
        )

    def _count(self, kind: str, rule: str | None = None) -> np.ndarray:
        if self.events is None:
            return np.zeros(max(self.num_frames - 1, 0), dtype=np.int64)
        return np.array(
            [
                sum(
                    1
                    for e in step
                    if e["kind"] == kind and (rule is None or e["rule"] == rule)
                )
                for step in self.events
            ],
            dtype=np.int64,
        )

    def collisions(self) -> np.ndarray:
        """Collisions (two-ball rules that fired) per recording interval, shape ``(T - 1,)``."""
        return self._count("collision")

    def bounces(self) -> np.ndarray:
        """Wall bounces (one-ball rules that changed a velocity) per recording interval."""
        return self._count("bounce")

    def collision_rules(self) -> list[str]:
        """The names of the two-ball rules that fired, in order of first appearance."""
        names: dict[str, None] = {}
        for step in self.events or ():
            for e in step:
                if e["kind"] == "collision":
                    names.setdefault(e["rule"], None)
        return list(names)

    def collisions_by_rule(self) -> dict[str, np.ndarray]:
        return {name: self._count("collision", name) for name in self.collision_rules()}

    def _histograms(self, bins: int) -> np.ndarray:
        box: Box = self.space.box
        extent = [[box.lower[0], box.upper[0]], [box.lower[1], box.upper[1]]]
        positions = self.centres()
        counts = np.empty((self.num_frames, bins, bins))
        for frame in range(self.num_frames):
            counts[frame], _, _ = np.histogram2d(
                positions[frame, :, 0], positions[frame, :, 1], bins=bins, range=extent
            )
        return counts

    @staticmethod
    def _entropy(counts: np.ndarray) -> np.ndarray:
        p = counts / counts.sum(axis=(1, 2), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, -p * np.log(p), 0.0)
        return terms.sum(axis=(1, 2))

    def configuration_entropy(self, bins: int = 5) -> np.ndarray:
        """Occupancy entropy of the ball centres at every frame (rises as the balls spread)."""
        return occupancy_entropy(self.project(), bins)

    def visitation_entropy(self, bins: int = 5) -> np.ndarray:
        return self._entropy(np.cumsum(self._histograms(bins), axis=0))

    def coverage(self, bins: int = 5) -> np.ndarray:
        seen = np.cumsum(self._histograms(bins), axis=0) > 0
        return seen.sum(axis=(1, 2)) / (bins * bins)

    def overlaps(self) -> np.ndarray:
        """Number of sites shared by two balls at every frame."""
        flat = self.sites().reshape(self.num_frames, -1)
        return np.array(
            [flat[f].size - np.unique(flat[f]).size for f in range(self.num_frames)]
        )

    def checks(self, bins: int = 5, interacting: bool = True) -> list[dict[str, Any]]:
        """Exact invariants of a ball run, each with a verdict and the evidence.
        ``interacting`` says whether the rule set makes balls collide (then shared sites and
        missing collisions are failures)."""
        energy = self.energy()
        momentum = self.momentum()
        changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
        bounced = self.bounces() > 0
        coverage = self.coverage(bins)
        overlaps = self.overlaps()
        collisions = self.collisions()
        by_rule = self.collisions_by_rule()
        total = int(collisions.sum())
        per_rule = ", ".join(f"{int(c.sum())} by {name}" for name, c in by_rule.items())
        return [
            {
                "name": "energy conserved exactly",
                "passed": bool(np.all(energy == energy[0])),
                "detail": f"E = {energy[0]:g} at every one of {self.num_frames} frames; "
                "every collision and every bounce keeps the speeds (integer arithmetic)",
            },
            {
                "name": "momentum changes only at the walls",
                "passed": bool(np.all(changed <= bounced)),
                "detail": f"{int(changed.sum())} intervals changed the total momentum, "
                f"{int(bounced.sum())} had a wall bounce; collisions exchange momentum "
                "between the balls and leave the total unchanged",
            },
            {
                "name": "collisions happened",
                "passed": total > 0 or self.num_walkers == 1 or not interacting,
                "detail": f"{total} collisions ({per_rule or 'no collision rule fired'}), "
                f"{int(self.bounces().sum())} wall bounces",
            },
            {
                "name": "no two balls on one site",
                "passed": bool(np.all(overlaps == 0)) or not interacting,
                "detail": f"{int(overlaps.max())} shared sites at most"
                + (
                    " (allowed: the balls pass through each other)"
                    if not interacting
                    else ""
                ),
            },
            {
                "name": "room coverage never decreases",
                "passed": bool(np.all(np.diff(coverage) >= 0)),
                "detail": f"{100 * coverage[-1]:.0f}% of the {bins}x{bins} cells visited",
            },
        ]

    def captions(self, detail: int = 8) -> list[str]:
        """One caption per frame: the events of the interval that led to it and the total
        momentum (balls listed one by one up to ``detail`` balls, counted beyond that)."""
        if self.events is None:
            return []
        few = self.num_walkers <= detail
        momentum = self.momentum_units()
        cells = lattice_cells(self.space, self.anchors)
        velocity = self.velocity

        def at(frame: int, b: int) -> str:
            return f"({cells[frame, b, 0]},{cells[frame, b, 1]})"

        def vec(v: Any) -> str:
            return f"({v[0]}, {v[1]})"

        captions = []
        for frame in range(self.num_frames):
            parts: list[str] = []
            if frame == 0:
                if few:
                    parts.append(
                        "; ".join(
                            f"ball {b} at {at(0, b)} with v = {vec(velocity[0, b])}"
                            for b in range(self.num_walkers)
                        )
                    )
                else:
                    parts.append(
                        f"{self.num_walkers} balls placed with their velocities"
                    )
            else:
                events = self.events[frame - 1]
                collisions = [e for e in events if e["kind"] == "collision"]
                bounces = [e for e in events if e["kind"] == "bounce"]
                if few:
                    for e in collisions:
                        a, b = e["balls"]
                        parts.append(
                            f"{e['rule']}: balls {a} and {b} met"
                            + (f" along {e['axis']}" if "axis" in e else "")
                            + f", v{a} {vec(e['before'][0])} to {vec(e['after'][0])}, "
                            f"v{b} {vec(e['before'][1])} to {vec(e['after'][1])}"
                        )
                    for e in bounces:
                        parts.append(
                            f"{e['rule']}: ball {e['balls'][0]} at the wall"
                            + (f" ({e['axis']} flipped)" if "axis" in e else "")
                            + f", v {vec(e['before'][0])} to {vec(e['after'][0])}"
                        )
                else:
                    if collisions:
                        parts.append(f"{len(collisions)} collision(s)")
                    if bounces:
                        parts.append(f"{len(bounces)} wall bounce(s)")
                moved = int(np.sum(self.anchors[frame] != self.anchors[frame - 1]))
                if not parts:
                    parts.append(
                        f"free flight, {moved} ball(s) moved"
                        if self.moving
                        else f"{moved} walker(s) hopped"
                    )
            if self.moving:
                px, py = momentum[frame]
                parts.append(f"total momentum ({px}, {py}) m a/tau")
            captions.append(f"step {frame}: " + ". ".join(parts))
        return captions

    def to_dict(self, bins: int = 5) -> dict:
        """JSON-serialisable description of the space and the recorded walker edges, with the
        ball gas's bookkeeping when the balls move."""
        space = self.space
        blob: dict[str, Any] = {
            "times": self.times.tolist(),
            "anchors": self.anchors.tolist(),
            "paths": None if self.paths is None else self.paths.tolist(),
            "velocities": None if self.velocities is None else self.velocities.tolist(),
            "events": None
            if self.events is None
            else [list(step) for step in self.events],
            "radius": self.radius,
            "space": {
                "coordinates": self.space.coordinates.tolist(),
                "edges": self.space.edges().tolist(),
                "box": self.space.box.to_dict(),
                "spacing": float(space.cell_size[0]),
                "neighbors": space.neighbors.tolist(),
            },
        }
        if self.velocities is not None:
            blob.update(
                {
                    "sites": self.sites().tolist(),
                    "centres": np.round(self.centres(), 5).tolist(),
                    "collisions": self.collisions().tolist(),
                    "collisions_by_rule": {
                        k: v.tolist() for k, v in self.collisions_by_rule().items()
                    },
                    "bounces": self.bounces().tolist(),
                    "energy": self.energy().tolist(),
                    "momentum": self.momentum().tolist(),
                    "configuration_entropy": np.round(
                        self.configuration_entropy(bins), 6
                    ).tolist(),
                    "visitation_entropy": np.round(
                        self.visitation_entropy(bins), 6
                    ).tolist(),
                    "coverage": np.round(self.coverage(bins), 6).tolist(),
                    "captions": self.captions(),
                }
            )
        return blob


@dataclass(frozen=True)
class GraphSimulation:
    """A graph system: a space, a rule and the physical duration of one rule application."""

    space: Space
    rule: Rule = field(default_factory=RandomHop)
    step_time: float = 1.0

    def __post_init__(self) -> None:
        if self.step_time <= 0:
            raise ValueError("step_time must be > 0")

    def run(
        self,
        initial: WalkerGraph,
        steps: int,
        rng: np.random.Generator,
        record_every: int = 1,
        observer: StepObserver | None = None,
        record_paths: bool = False,
    ) -> GraphHistory:
        """Apply the rule ``steps`` times, recording every ``record_every``-th state.

        ``observer(step, previous_positions, positions)`` is called after every step with the
        projected positions (see :mod:`imigraph.metrics`). With ``record_paths`` (runs recorded
        at every step, rules with ``apply_with_paths``) the matched chains are kept for the
        rule diagram.
        """
        if initial.space != self.space:
            raise ValueError("initial state must live on the simulation's space")
        if steps < 0 or record_every < 1:
            raise ValueError("steps must be >= 0 and record_every >= 1")
        with_paths = (
            getattr(self.rule, "apply_with_paths", None) if record_paths else None
        )
        if record_paths and (record_every != 1 or with_paths is None):
            raise ValueError(
                "paths can only be recorded at every step of a rule that reports them"
            )
        frames = [initial.anchors]
        paths = []
        state = initial
        coordinates = self.space.coordinates
        for step in range(1, steps + 1):
            previous = state
            if with_paths is not None:
                state, path = with_paths(state, rng)
                paths.append(path)
            else:
                state = self.rule.apply(state, rng)
            if observer is not None:
                # A hop along a self-loop (walker stays) is the graph's wall interaction.
                stayed = state.anchors == previous.anchors
                observer(
                    step,
                    coordinates[previous.anchors],
                    coordinates[state.anchors],
                    stayed,
                )
            if step % record_every == 0:
                frames.append(state.anchors)
        times = self.step_time * record_every * np.arange(len(frames))
        return GraphHistory(
            self.space,
            times,
            np.stack(frames),
            np.stack(paths) if paths else None,
            step_time=self.step_time,
        )
