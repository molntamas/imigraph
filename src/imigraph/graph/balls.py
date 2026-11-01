"""Several rigid balls moving ballistically on the lattice graph, bouncing off the walls and
colliding with each other: the graph imitation of a hard-disk gas, built from the single ball
of :mod:`imigraph.graph.newton`.

A ball is one node (``radius = 0``) or the lattice points within a radius, each a walker on a
site; all walkers of a ball share its integer velocity ``(vx, vy)`` in cells per step. A step
moves every ball hop by hop (``|vx| + |vy|`` hops in Bresenham order, see
:func:`~imigraph.graph.newton.hop_order`), all balls one hop at a time, and a hop into a wall
self-loop flips that velocity component (an elastic bounce). After every round of hops the
balls are checked for *contact*: two balls sharing a site, or two balls that crossed each other
along an edge. A contact is resolved by the *collision rule* of the gas:

* ``exchange``: the elastic collision of equal masses. The line of centres before the hop is a
  lattice direction ``d`` (an axis or a diagonal). The velocity component along ``d`` is
  exchanged: ``va' = va - delta d``, ``vb' = vb + delta d`` with ``delta = (va - vb) . d / d . d``.
  Along an axis ``delta`` is an integer, so the result is exact. Along a diagonal ``delta`` is a
  half-integer whenever ``(va - vb) . d`` is odd; then the whole velocity vectors are swapped
  instead (the head-on outcome), which still conserves momentum and energy exactly but gives
  the wrong scattering angle. Such *fallback* collisions are counted separately.
* ``swap``: the balls always swap their velocity vectors, whatever the contact geometry.
* ``none``: the balls pass through each other (an ideal gas of points; sites may be shared).

Under ``exchange`` and ``swap`` the hop that produced the contact is undone for both balls, so
no two balls ever share a site, and the balls continue the step with their new velocities (the
remaining hops of the step follow the new velocity, never more hops than the ball had).
Separating pairs (relative velocity along ``d`` not towards each other) do not collide. Momentum
and kinetic energy are integers times ``m a / tau`` and ``m a^2 / tau^2``, conserved exactly by
every collision; the walls exchange momentum and leave the energy unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imigraph.graph.newton import AXES, disc_offsets, hop_order
from imigraph.graph.space import SquareLattice
from imigraph.metrics import occupancy_entropy
from imigraph.reference.gas import GasRun
from imigraph.trajectory import Box, Trajectory

COLLISION_RULES = ("exchange", "swap", "none")
COLLISION_RULE_TEXT = {
    "exchange": "exchange: the velocity component along the line of centres is exchanged "
    "(elastic collision of equal masses); when that is not a lattice velocity (an odd diagonal "
    "hit) the velocity vectors are swapped instead",
    "swap": "swap: the balls always exchange their whole velocity vectors",
    "none": "none: the balls pass through each other (point particles)",
}
AXIS_NAMES = ("x", "y")


@dataclass(frozen=True)
class BallGasState:
    """``n`` balls of ``k`` walkers each: sites ``(n, k)`` and velocities ``(n, 2)``."""

    space: SquareLattice
    sites: np.ndarray
    velocities: np.ndarray
    mass: float = 1.0

    def __post_init__(self) -> None:
        sites = np.asarray(self.sites, dtype=np.int64)
        velocities = np.asarray(self.velocities, dtype=np.int64)
        if sites.ndim != 2 or sites.shape[0] == 0 or sites.shape[1] == 0:
            raise ValueError("sites must have shape (balls, walkers per ball)")
        if velocities.shape != (sites.shape[0], 2):
            raise ValueError("velocities must have shape (balls, 2)")
        if sites.min() < 0 or sites.max() >= self.space.num_nodes:
            raise ValueError("sites must be valid node indices")
        if self.mass <= 0:
            raise ValueError("mass must be > 0")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "velocities", velocities)

    @property
    def num_balls(self) -> int:
        return int(self.sites.shape[0])

    @property
    def walkers_per_ball(self) -> int:
        return int(self.sites.shape[1])

    def centres(self) -> np.ndarray:
        """Centre of every ball in box coordinates, shape ``(n, 2)``."""
        return self.space.coordinates[self.sites].mean(axis=1)


def _cells(space: SquareLattice, sites: np.ndarray) -> np.ndarray:
    """Integer cell coordinates ``(i, j)`` of sites of any shape."""
    return np.stack([sites // space.size, sites % space.size], axis=-1)


@dataclass(frozen=True)
class BallGasRun:
    """Recorded run: sites ``(T, n, k)``, velocities ``(T, n, 2)``, and per recording interval
    the collisions (exact + fallback), the fallback collisions and the wall bounces."""

    space: SquareLattice
    times: np.ndarray
    sites: np.ndarray
    velocities: np.ndarray
    collisions: np.ndarray
    fallbacks: np.ndarray
    bounces: np.ndarray
    mass: float
    step_time: float
    rule: str
    events: tuple[tuple[dict[str, Any], ...], ...] = field(default=(), compare=False)
    """Per step (only when every step is recorded): the collision and bounce events."""

    @property
    def num_frames(self) -> int:
        return int(self.sites.shape[0])

    @property
    def num_balls(self) -> int:
        return int(self.sites.shape[1])

    @property
    def walkers_per_ball(self) -> int:
        return int(self.sites.shape[2])

    @property
    def speed_unit(self) -> float:
        return float(self.space.spacing / self.step_time)

    def centres(self) -> np.ndarray:
        return self.space.coordinates[self.sites].mean(axis=2)

    def trajectory(self) -> Trajectory:
        return Trajectory(self.times, self.centres(), self.space.box)

    def to_gas_run(self) -> GasRun:
        """The run in the shape of a hard-disk gas run, so the gas checks apply: the centres
        as positions, the velocities in ``a / tau``, the mass of a whole ball."""
        return GasRun(
            self.trajectory(),
            self.velocities * self.speed_unit,
            self.collisions,
            self.mass * self.walkers_per_ball,
        )

    def energy(self) -> np.ndarray:
        v2 = np.sum((self.velocities * self.speed_unit) ** 2, axis=2)
        return 0.5 * self.mass * self.walkers_per_ball * v2.sum(axis=1)

    def momentum(self) -> np.ndarray:
        return (
            self.mass * self.walkers_per_ball * (self.velocities * self.speed_unit)
        ).sum(axis=1)

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
        return occupancy_entropy(self.trajectory(), bins)

    def visitation_entropy(self, bins: int = 5) -> np.ndarray:
        return self._entropy(np.cumsum(self._histograms(bins), axis=0))

    def coverage(self, bins: int = 5) -> np.ndarray:
        seen = np.cumsum(self._histograms(bins), axis=0) > 0
        return seen.sum(axis=(1, 2)) / (bins * bins)

    def overlaps(self) -> np.ndarray:
        """Number of sites shared by two balls at every frame."""
        flat = self.sites.reshape(self.num_frames, -1)
        return np.array(
            [flat[f].size - np.unique(flat[f]).size for f in range(self.num_frames)]
        )

    def checks(self, bins: int = 5) -> list[dict[str, Any]]:
        """Exact invariants of the graph system, each with a verdict and the evidence."""
        energy = self.energy()
        momentum = self.momentum()
        changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
        bounced = self.bounces > 0
        coverage = self.coverage(bins)
        overlaps = self.overlaps()
        total = int(self.collisions.sum())
        fallback = int(self.fallbacks.sum())
        rows = [
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
                "passed": total > 0 or self.num_balls == 1,
                "detail": f"{total} collisions ({fallback} with the swap fallback of an odd "
                f"diagonal hit), {int(self.bounces.sum())} wall bounces, rule {self.rule}",
            },
            {
                "name": "no two balls on one site",
                "passed": bool(np.all(overlaps == 0)) or self.rule == "none",
                "detail": f"{int(overlaps.max())} shared sites at most"
                + (
                    " (allowed: the balls pass through each other)"
                    if self.rule == "none"
                    else ""
                ),
            },
            {
                "name": "room coverage never decreases",
                "passed": bool(np.all(np.diff(coverage) >= 0)),
                "detail": f"{100 * coverage[-1]:.0f}% of the {bins}x{bins} cells visited",
            },
        ]
        return rows

    def captions(self, detail: int = 8) -> list[str]:
        """One caption per frame (when every step was recorded): the events of the step and
        the total momentum."""
        if not self.events:
            return []
        few = self.num_balls <= detail
        momentum = self.momentum()
        captions = []
        for frame in range(self.num_frames):
            parts: list[str] = []
            if frame == 0:
                cells = _cells(self.space, self.sites[0, :, 0])
                if few:
                    parts.append(
                        "; ".join(
                            f"ball {b} at ({cells[b, 0]},{cells[b, 1]}) with v = "
                            f"({self.velocities[0, b, 0]}, {self.velocities[0, b, 1]})"
                            for b in range(self.num_balls)
                        )
                    )
                else:
                    parts.append(f"{self.num_balls} balls placed with their velocities")
            else:
                events = self.events[frame - 1]
                collisions = [e for e in events if e["kind"] == "collision"]
                bounces = [e for e in events if e["kind"] == "bounce"]
                for e in collisions if few else []:
                    a, b = e["balls"]
                    parts.append(
                        f"balls {a} and {b} collided along {e['axis']}: v{a} "
                        f"({e['before'][0][0]}, {e['before'][0][1]}) to "
                        f"({e['after'][0][0]}, {e['after'][0][1]}), v{b} "
                        f"({e['before'][1][0]}, {e['before'][1][1]}) to "
                        f"({e['after'][1][0]}, {e['after'][1][1]})"
                        + (
                            " (exact exchange)"
                            if e["exact"]
                            else " (swap fallback: the exact exchange is not a lattice velocity)"
                        )
                    )
                if collisions and not few:
                    parts.append(f"{len(collisions)} collision(s)")
                if bounces:
                    parts.append(
                        ", ".join(
                            f"ball {e['balls'][0]} bounced off the wall ({e['axis']} flipped)"
                            for e in bounces
                        )
                        if few
                        else f"{len(bounces)} wall bounce(s)"
                    )
                if not parts:
                    parts.append("free flight")
            px, py = momentum[frame]
            parts.append(f"total momentum ({px:g}, {py:g})")
            captions.append(f"step {frame}: " + ". ".join(parts))
        return captions

    def to_dict(self, bins: int = 5) -> dict[str, Any]:
        space = self.space
        return {
            "times": self.times.tolist(),
            "sites": self.sites.tolist(),
            "velocities": self.velocities.tolist(),
            "centres": np.round(self.centres(), 5).tolist(),
            "collisions": self.collisions.tolist(),
            "fallbacks": self.fallbacks.tolist(),
            "bounces": self.bounces.tolist(),
            "energy": self.energy().tolist(),
            "momentum": self.momentum().tolist(),
            "configuration_entropy": np.round(
                self.configuration_entropy(bins), 6
            ).tolist(),
            "visitation_entropy": np.round(self.visitation_entropy(bins), 6).tolist(),
            "coverage": np.round(self.coverage(bins), 6).tolist(),
            "captions": self.captions(),
            "events": [list(step) for step in self.events],
            "rule": self.rule,
            "space": {
                "coordinates": np.round(space.coordinates, 5).tolist(),
                "edges": space.edges().tolist(),
                "box": space.box.to_dict(),
                "spacing": float(space.spacing),
                "neighbors": space.neighbors.tolist(),
            },
        }


@dataclass(frozen=True)
class BallGas:
    """The room (a reflecting lattice), the step time and the collision rule."""

    space: SquareLattice
    step_time: float = 1.0
    rule: str = "exchange"

    def __post_init__(self) -> None:
        if self.space.periodic:
            raise ValueError("the ball gas needs walls: use a reflecting lattice")
        if self.step_time <= 0:
            raise ValueError("step_time must be > 0")
        if self.rule not in COLLISION_RULES:
            raise ValueError(
                f"collision rule must be one of {', '.join(COLLISION_RULES)}, "
                f"not {self.rule!r}"
            )

    # ---------- initial conditions ----------

    def place(
        self,
        centres: np.ndarray,
        velocities: np.ndarray,
        radius: int = 0,
        mass: float = 1.0,
    ) -> BallGasState:
        """Balls of ``radius`` at integer ``centres`` ``(n, 2)`` with ``velocities`` ``(n, 2)``."""
        centres = np.asarray(centres, dtype=np.int64).reshape(-1, 2)
        velocities = np.asarray(velocities, dtype=np.int64).reshape(-1, 2)
        if centres.shape[0] != velocities.shape[0]:
            raise ValueError("give one velocity per ball")
        offsets = disc_offsets(radius)
        size = self.space.size
        if 2 * radius + 1 >= size:
            raise ValueError("the balls must be smaller than the room")
        nodes = centres[:, None, :] + offsets[None, :, :]
        if nodes.min() < 0 or nodes.max() >= size:
            raise ValueError(
                f"a ball of radius {radius} does not fit at every centre of the "
                f"{size}x{size} room"
            )
        sites = nodes[:, :, 0] * size + nodes[:, :, 1]
        if self.rule != "none" and np.unique(sites).size != sites.size:
            raise ValueError("the balls overlap at the start: use distinct sites")
        return BallGasState(self.space, sites, velocities, mass)

    # ---------- dynamics ----------

    def _hop(
        self, sites: np.ndarray, velocity: np.ndarray, axis: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """One ball one edge along ``axis``; at a wall the component flips (a bounce)."""
        sign = int(np.sign(velocity[axis]))
        if sign == 0:
            return sites, velocity, False
        plus, minus = AXES[axis]
        label = plus if sign > 0 else minus
        neighbors = self.space.neighbors
        bounced = False
        if np.any(neighbors[sites, label] == sites):
            bounced = True
            label = minus if sign > 0 else plus
            velocity = velocity.copy()
            velocity[axis] = -velocity[axis]
            if np.any(neighbors[sites, label] == sites):
                return sites, velocity, bounced
        return neighbors[sites, label], velocity, bounced

    def _contacts(self, before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
        """Pairs that share a site after the round or swapped sites during it."""
        n = after.shape[0]
        pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                shared = np.intersect1d(after[a], after[b]).size > 0
                crossed = (
                    np.intersect1d(after[a], before[b]).size > 0
                    and np.intersect1d(after[b], before[a]).size > 0
                )
                if shared or crossed:
                    pairs.append((a, b))
        return pairs

    def _collide(
        self, va: np.ndarray, vb: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        """Apply the rule to a contact along ``d``; returns the new velocities, whether a
        collision happened (the pair was approaching) and whether it was exact."""
        dd = int(d @ d)
        relative = int((va - vb) @ d)
        if relative <= 0:
            return va, vb, False, True  # separating or parallel: no collision
        if self.rule == "exchange" and relative % dd == 0:
            delta = relative // dd
            return va - delta * d, vb + delta * d, True, True
        return vb.copy(), va.copy(), True, self.rule == "swap"

    def step(self, state: BallGasState) -> tuple[BallGasState, list[dict[str, Any]]]:
        """One step of every ball: the new state and the events (collisions, bounces)."""
        n = state.num_balls
        sites = state.sites.copy()
        velocities = state.velocities.copy()
        hops = [hop_order(velocities[b]) for b in range(n)]
        done = [0] * n
        events: list[dict[str, Any]] = []
        max_rounds = 2 * max([len(h) for h in hops] + [0]) + 2
        for _ in range(max_rounds):
            if all(done[b] >= len(hops[b]) for b in range(n)):
                break
            before_sites = sites.copy()
            hopped = [False] * n
            for b in range(n):
                if done[b] < len(hops[b]):
                    axis = hops[b][done[b]]
                    sites[b], velocities[b], bounced = self._hop(
                        sites[b], velocities[b], axis
                    )
                    if bounced:
                        events.append(
                            {"kind": "bounce", "balls": [b], "axis": AXIS_NAMES[axis]}
                        )
                    hopped[b] = True
                    done[b] += 1
            if self.rule == "none":
                continue
            for _pass in range(n + 1):
                pairs = self._contacts(before_sites, sites)
                if not pairs:
                    break
                for a, b in pairs:
                    if not self._contacts(before_sites[[a, b]], sites[[a, b]]):
                        continue  # resolved by an earlier pair of this pass
                    centre_a = _cells(self.space, before_sites[a]).mean(axis=0)
                    centre_b = _cells(self.space, before_sites[b]).mean(axis=0)
                    d = np.sign(np.rint(centre_b - centre_a)).astype(np.int64)
                    if not d.any():
                        d = np.sign(velocities[a] - velocities[b]).astype(np.int64)
                    if not d.any():
                        d = np.array([1, 0])
                    va, vb, collided, exact = self._collide(
                        velocities[a], velocities[b], d
                    )
                    # The hop that produced the contact is undone for both balls.
                    for ball in (a, b):
                        if hopped[ball]:
                            sites[ball] = before_sites[ball]
                            done[ball] -= 1
                            hopped[ball] = False
                    if not collided:
                        continue
                    before = [velocities[a].tolist(), velocities[b].tolist()]
                    velocities[a], velocities[b] = va, vb
                    for ball in (a, b):
                        # The rest of the step is taken with the new velocity, but a ball
                        # never gets more hops than it had: the tail of the new hop order.
                        remaining = max(0, len(hops[ball]) - done[ball])
                        new_hops = hop_order(velocities[ball])
                        tail = new_hops[max(0, len(new_hops) - remaining) :]
                        hops[ball] = hops[ball][: done[ball]] + tail
                    events.append(
                        {
                            "kind": "collision",
                            "balls": [a, b],
                            "axis": "diagonal"
                            if d[0] and d[1]
                            else AXIS_NAMES[int(d[1] != 0)],
                            "before": before,
                            "after": [va.tolist(), vb.tolist()],
                            "exact": bool(exact),
                        }
                    )
        return BallGasState(self.space, sites, velocities, state.mass), events

    def run(
        self, initial: BallGasState, steps: int, record_every: int = 1
    ) -> BallGasRun:
        if initial.space != self.space:
            raise ValueError("the balls must live in this room")
        if steps < 0 or record_every < 1 or steps % record_every:
            raise ValueError("steps must be a non-negative multiple of record_every")
        sites, velocities = [initial.sites], [initial.velocities]
        collisions, fallbacks, bounces = [], [], []
        step_events: list[tuple[dict[str, Any], ...]] = []
        interval = [0, 0, 0]
        state = initial
        for step in range(1, steps + 1):
            state, events = self.step(state)
            if record_every == 1:
                step_events.append(tuple(events))
            for e in events:
                if e["kind"] == "collision":
                    interval[0] += 1
                    interval[1] += int(not e["exact"])
                else:
                    interval[2] += 1
            if step % record_every == 0:
                sites.append(state.sites)
                velocities.append(state.velocities)
                collisions.append(interval[0])
                fallbacks.append(interval[1])
                bounces.append(interval[2])
                interval = [0, 0, 0]
        return BallGasRun(
            self.space,
            self.step_time * record_every * np.arange(len(sites)),
            np.stack(sites),
            np.stack(velocities),
            np.asarray(collisions, dtype=np.int64),
            np.asarray(fallbacks, dtype=np.int64),
            np.asarray(bounces, dtype=np.int64),
            initial.mass,
            self.step_time,
            self.rule,
            tuple(step_events),
        )
