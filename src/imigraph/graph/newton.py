"""Newtonian motion without gravity on the lattice graph: a rigid object (a *ball* of several
walker nodes) moves in a straight line and bounces elastically off the walls.

The ball is a set of walkers that share one velocity ``v = (vx, vy)`` in lattice cells per step
(integers). One step moves every walker of the ball ``|vx|`` edges along ``sign(vx)`` and
``|vy|`` edges along ``sign(vy)``, interleaved so the path hugs the straight line (Bresenham
order). A hop in a direction in which any walker of the ball faces a wall self-loop ``{s,s}``
is a *bounce*: the velocity component flips sign and the hop is taken the other way, so the
reflection is specular and the remaining hops of the step continue after it. The ball keeps its
shape, every walker keeps its speed, so the kinetic energy ``k m |v|^2 / 2`` is exactly
conserved, and the momentum ``k m v`` changes only at a wall, where one component flips: the
wall takes ``2 k m |v_axis|``. There is no friction and no gravity.

Entropy is tracked two ways: the *configuration* entropy of the walker positions at a frame,
at the resolution of the lattice cells (``ln k`` for a rigid ball of ``k`` walkers at every
frame: it does not disperse, unlike the Brownian walkers, whose entropy rises), and on a
``bins x bins`` grid of the room the *visitation* entropy of all positions seen so far, which tends towards
``ln(bins^2)`` as the ball sweeps the room (or stays low on a periodic orbit); it is not
monotone, since revisiting a bin makes the visitation distribution more peaked, so the monotone
companion is the *coverage*, the fraction of bins visited so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from imigraph.graph.space import SquareLattice
from imigraph.trajectory import Box, Trajectory

AXES = ((0, 1), (2, 3))  # neighbour-table labels (+, -) of the x and the y axis


def disc_offsets(radius: int) -> np.ndarray:
    """Node offsets of a ball: the lattice points with ``i^2 + j^2 <= r^2``, shape ``(k, 2)``."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    r = np.arange(-radius, radius + 1)
    i, j = np.meshgrid(r, r, indexing="ij")
    inside = i**2 + j**2 <= radius**2
    return np.stack([i[inside], j[inside]], axis=1).astype(np.int64)


def hop_order(velocity: np.ndarray) -> list[int]:
    """The hops of one step as axis indices (0 = x, 1 = y), interleaved in Bresenham order so
    that the lattice path follows the straight line of the velocity."""
    nx, ny = int(abs(velocity[0])), int(abs(velocity[1]))
    order: list[int] = []
    if nx >= ny:
        error = 0
        for _ in range(nx):
            order.append(0)
            error += ny
            if 2 * error >= nx:
                order.append(1)
                error -= nx
        order += [1] * (ny - order.count(1))
    else:
        error = 0
        for _ in range(ny):
            order.append(1)
            error += nx
            if 2 * error >= ny:
                order.append(0)
                error -= ny
        order += [0] * (nx - order.count(0))
    return order


@dataclass(frozen=True)
class BallState:
    """A rigid ball: the sites of its walkers and their common velocity (cells per step)."""

    space: SquareLattice
    sites: np.ndarray
    velocity: np.ndarray
    mass: float = 1.0

    def __post_init__(self) -> None:
        sites = np.asarray(self.sites, dtype=np.int64)
        velocity = np.asarray(self.velocity, dtype=np.int64)
        if sites.ndim != 1 or sites.size == 0:
            raise ValueError("a ball needs at least one walker")
        if sites.min() < 0 or sites.max() >= self.space.num_nodes:
            raise ValueError("sites must be valid node indices")
        if np.unique(sites).size != sites.size:
            raise ValueError("the walkers of a ball sit on distinct sites")
        if velocity.shape != (2,):
            raise ValueError("velocity must be (vx, vy) in cells per step")
        if self.mass <= 0:
            raise ValueError("mass must be > 0")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "velocity", velocity)

    @property
    def num_walkers(self) -> int:
        return int(self.sites.shape[0])


@dataclass(frozen=True)
class BallRun:
    """Every frame of a ball run: sites ``(T, k)``, velocities ``(T, 2)``, bounces per step
    ``(T - 1,)`` (number of velocity flips in that step)."""

    space: SquareLattice
    times: np.ndarray
    sites: np.ndarray
    velocities: np.ndarray
    bounces: np.ndarray
    mass: float
    step_time: float

    @property
    def num_frames(self) -> int:
        return int(self.sites.shape[0])

    @property
    def num_walkers(self) -> int:
        return int(self.sites.shape[1])

    @property
    def speed_unit(self) -> float:
        """One cell per step in physical units: ``a / tau``."""
        return float(self.space.spacing / self.step_time)

    def energy(self) -> np.ndarray:
        """Kinetic energy ``k m |v|^2 / 2`` at every frame."""
        v2 = np.sum((self.velocities * self.speed_unit) ** 2, axis=1)
        return 0.5 * self.mass * self.num_walkers * v2

    def momentum(self) -> np.ndarray:
        """Momentum ``k m v`` at every frame, shape ``(T, 2)``."""
        return self.mass * self.num_walkers * self.velocities * self.speed_unit

    def positions(self) -> np.ndarray:
        return self.space.coordinates[self.sites]

    def trajectory(self) -> Trajectory:
        return Trajectory(self.times, self.positions(), self.space.box)

    def _histograms(self, bins: int) -> np.ndarray:
        box: Box = self.space.box
        extent = [[box.lower[0], box.upper[0]], [box.lower[1], box.upper[1]]]
        positions = self.positions()
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

    def configuration_entropy(self) -> np.ndarray:
        """Entropy of the ball's positions at each frame at the resolution of the lattice
        cells: ``ln k`` at every frame for a rigid ball (it does not disperse)."""
        return self._entropy(self._histograms(self.space.size))

    def visitation_entropy(self, bins: int = 6) -> np.ndarray:
        """Entropy of all positions seen up to each frame: the room the ball has swept."""
        return self._entropy(np.cumsum(self._histograms(bins), axis=0))

    def coverage(self, bins: int = 6) -> np.ndarray:
        """Fraction of the ``bins x bins`` cells visited up to each frame (non-decreasing)."""
        seen = np.cumsum(self._histograms(bins), axis=0) > 0
        return seen.sum(axis=(1, 2)) / (bins * bins)

    def shape_intact(self) -> bool:
        """The offsets between the walkers are the same at every frame."""
        positions = self.positions()
        offsets = positions - positions[:, :1, :]
        return bool(np.all(offsets == offsets[:1]))

    def checks(self, bins: int = 6) -> list[dict[str, Any]]:
        """Conservation and shape checks of the run, each with a verdict and the evidence."""
        energy = self.energy()
        momentum = self.momentum()
        configuration = self.configuration_entropy()
        norm = np.linalg.norm(momentum, axis=1)
        changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
        bounced = self.bounces > 0
        visitation = self.visitation_entropy(bins)
        coverage = self.coverage(bins)
        return [
            {
                "name": "energy conserved",
                "passed": bool(np.all(energy == energy[0])),
                "detail": f"E = {energy[0]:g} at every one of {self.num_frames} frames "
                "(every walker keeps its speed; the walls are elastic)",
            },
            {
                "name": "momentum magnitude conserved",
                "passed": bool(np.allclose(norm, norm[0])),
                "detail": f"|P| = {norm[0]:g} at every frame; the direction changes only "
                f"at the walls ({int(bounced.sum())} steps with a bounce)",
            },
            {
                "name": "momentum changes only at the walls",
                "passed": bool(np.array_equal(changed, bounced)),
                "detail": f"{int(changed.sum())} steps changed the momentum, "
                f"{int(bounced.sum())} steps had a bounce",
            },
            {
                "name": "shape intact",
                "passed": self.shape_intact(),
                "detail": f"the {self.num_walkers} walkers keep their offsets "
                "(a rigid body)",
            },
            {
                "name": "configuration entropy constant",
                "passed": bool(np.allclose(configuration, np.log(self.num_walkers))),
                "detail": f"S = ln {self.num_walkers} = {configuration[0]:.3f} nats at "
                "every frame (cell resolution): a rigid ball does not disperse",
            },
            {
                "name": "room coverage never decreases",
                "passed": bool(np.all(np.diff(coverage) >= 0)),
                "detail": f"{100 * coverage[-1]:.0f}% of the {bins}x{bins} cells visited; "
                f"visitation entropy from {visitation[0]:.3f} to {visitation[-1]:.3f} nats "
                f"(uniform: {np.log(bins * bins):.3f})",
            },
        ]

    def to_dict(self, bins: int = 6) -> dict[str, Any]:
        """JSON-serialisable run: space, per-frame sites and velocities, bounces, series."""
        space = self.space
        return {
            "times": self.times.tolist(),
            "sites": self.sites.tolist(),
            "velocities": self.velocities.tolist(),
            "bounces": self.bounces.tolist(),
            "energy": self.energy().tolist(),
            "momentum": self.momentum().tolist(),
            "configuration_entropy": np.round(self.configuration_entropy(), 6).tolist(),
            "visitation_entropy": np.round(self.visitation_entropy(bins), 6).tolist(),
            "coverage": np.round(self.coverage(bins), 6).tolist(),
            "space": {
                "coordinates": np.round(space.coordinates, 4).tolist(),
                "edges": space.edges().tolist(),
                "box": space.box.to_dict(),
                "spacing": float(space.spacing),
                "neighbors": space.neighbors.tolist(),
            },
        }


@dataclass(frozen=True)
class NewtonWorld:
    """A room (a reflecting square lattice) in which rigid balls move ballistically."""

    space: SquareLattice
    step_time: float = 1.0

    def __post_init__(self) -> None:
        if self.space.periodic:
            raise ValueError("a bouncing ball needs walls: use a reflecting lattice")
        if self.step_time <= 0:
            raise ValueError("step_time must be > 0")

    def place(
        self,
        centre: tuple[int, int],
        velocity: tuple[int, int],
        radius: int = 0,
        mass: float = 1.0,
    ) -> BallState:
        """A disc of ``radius`` around ``centre = (i, j)`` with ``velocity`` in cells/step."""
        offsets = disc_offsets(radius)
        size = self.space.size
        if 2 * radius + 1 >= size:
            raise ValueError("the ball must be smaller than the room")
        nodes = offsets + np.asarray(centre, dtype=np.int64)
        if nodes.min() < 0 or nodes.max() >= size:
            raise ValueError(
                f"a ball of radius {radius} at {tuple(int(c) for c in centre)} does not fit "
                f"into the {size}x{size} room"
            )
        sites = nodes[:, 0] * size + nodes[:, 1]
        return BallState(self.space, sites, np.asarray(velocity), mass)

    def hop(self, state: BallState, axis: int) -> tuple[BallState, bool]:
        """Move the ball one edge along ``axis`` in the direction of its velocity; at a wall
        the velocity component flips and the ball moves the other way (a bounce)."""
        sign = int(np.sign(state.velocity[axis]))
        if sign == 0:
            return state, False
        plus, minus = AXES[axis]
        label = plus if sign > 0 else minus
        neighbors = self.space.neighbors
        bounced = False
        if np.any(neighbors[state.sites, label] == state.sites):
            bounced = True
            label = minus if sign > 0 else plus
            velocity = state.velocity.copy()
            velocity[axis] = -velocity[axis]
            state = BallState(self.space, state.sites, velocity, state.mass)
            if np.any(neighbors[state.sites, label] == state.sites):
                return state, bounced  # the ball spans the room: it cannot move
        return (
            BallState(
                self.space, neighbors[state.sites, label], state.velocity, state.mass
            ),
            bounced,
        )

    def step(self, state: BallState) -> tuple[BallState, int]:
        """One step: ``|vx| + |vy|`` hops in Bresenham order; returns the state and the number
        of bounces in the step."""
        bounces = 0
        for axis in hop_order(state.velocity):
            state, bounced = self.hop(state, axis)
            bounces += int(bounced)
        return state, bounces

    def run(self, initial: BallState, steps: int) -> BallRun:
        if initial.space != self.space:
            raise ValueError("the ball must live in this world")
        if steps < 0:
            raise ValueError("steps must be >= 0")
        sites, velocities, bounces = [initial.sites], [initial.velocity], []
        state = initial
        for _ in range(steps):
            state, bounced = self.step(state)
            sites.append(state.sites)
            velocities.append(state.velocity)
            bounces.append(bounced)
        return BallRun(
            self.space,
            self.step_time * np.arange(steps + 1),
            np.stack(sites),
            np.stack(velocities),
            np.asarray(bounces, dtype=np.int64),
            initial.mass,
            self.step_time,
        )
