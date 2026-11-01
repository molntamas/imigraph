"""A hard-disk gas: elastic collisions in a 2D box, kinetic energy conserved exactly.

``n`` disks of radius ``r`` and mass ``m`` move in straight lines; when two disks overlap while
approaching each other, they are moved back to the moment of contact (the exact time of impact
within the step), the components of their velocities along the line of centres are exchanged
(elastic collision of equal masses, which conserves momentum and kinetic energy exactly), and
they move on with the new velocities for the rest of the step. Positions therefore stay
continuous: no disk is ever pushed. Walls are specular (velocity component flipped, position
mirrored) or periodic. Positions advance with a fixed time step; a collision is detected at the
end of the step in which the disks first overlap, and resolved at its exact time.

Analytic facts used by the tests:

* total kinetic energy ``sum m |v|^2 / 2`` is constant for all time (to rounding);
* with periodic walls the total momentum is constant; with specular walls each reflection flips
  one component, so the energy of every particle is still constant between collisions;
* after many collisions the speeds follow the 2D Maxwell-Boltzmann (Rayleigh) distribution
  ``f(s) = s / sigma^2 exp(-s^2 / (2 sigma^2))`` with ``sigma^2 = kT / m = E / (n m)``;
* started in one half of the box, the gas expands and its occupancy entropy rises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from imigraph.trajectory import Box, Trajectory


@dataclass(frozen=True)
class GasRun:
    """Recorded run: positions as a :class:`Trajectory`, velocities ``(T, n, 2)`` and the number
    of collisions in every recording interval ``(T - 1,)``."""

    trajectory: Trajectory
    velocities: np.ndarray
    collisions: np.ndarray
    mass: float = 1.0

    @property
    def times(self) -> np.ndarray:
        return self.trajectory.times

    def kinetic_energy(self) -> np.ndarray:
        """Total kinetic energy at every frame, shape ``(T,)``."""
        return 0.5 * self.mass * np.sum(self.velocities**2, axis=(1, 2))

    def momentum(self) -> np.ndarray:
        """Total momentum at every frame, shape ``(T, 2)``."""
        return self.mass * self.velocities.sum(axis=1)

    def speeds(self, frame: int) -> np.ndarray:
        return np.linalg.norm(self.velocities[frame], axis=1)

    @property
    def thermal_sigma(self) -> float:
        """``sqrt(kT / m)`` from the (conserved) energy: ``sigma^2 = E / (n m)``."""
        n = self.trajectory.num_particles
        return float(np.sqrt(self.kinetic_energy()[0] / (n * self.mass)))


@dataclass(frozen=True)
class HardDiskGas:
    box: Box
    radius: float = 0.012
    dt: float = 0.004
    boundary: Literal["reflecting", "periodic"] = "reflecting"
    mass: float = 1.0

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.dt <= 0 or self.mass <= 0:
            raise ValueError("radius, dt and mass must be > 0")
        if self.boundary not in ("reflecting", "periodic"):
            raise ValueError("boundary must be 'reflecting' or 'periodic'")
        if np.any(self.box.side <= 4 * self.radius):
            raise ValueError("box must be larger than two disk diameters")

    # ---------- initial conditions ----------

    def grid_in_region(
        self, count: int, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """``count`` non-overlapping positions on a grid inside ``[lower, upper]``."""
        lower = np.asarray(lower, dtype=float) + self.radius
        upper = np.asarray(upper, dtype=float) - self.radius
        side = upper - lower
        columns = max(1, int(np.ceil(np.sqrt(count * side[0] / side[1]))))
        rows = int(np.ceil(count / columns))
        if min(side[0] / columns, side[1] / rows) < 2 * self.radius:
            raise ValueError(
                f"{count} disks of radius {self.radius} do not fit into the region"
            )
        xs = lower[0] + (np.arange(columns) + 0.5) * side[0] / columns
        ys = lower[1] + (np.arange(rows) + 0.5) * side[1] / rows
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:count]

    def initial_half_box(
        self, count: int, speed: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Disks on a grid in the left half of the box, all with the same speed in random
        directions (far from Maxwell-Boltzmann, so the relaxation is visible)."""
        lower = np.asarray(self.box.lower, dtype=float)
        upper = np.asarray(self.box.upper, dtype=float)
        half = upper.copy()
        half[0] = lower[0] + self.box.side[0] / 2
        positions = self.grid_in_region(count, lower, half)
        angles = rng.uniform(0, 2 * np.pi, size=count)
        velocities = speed * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return positions, velocities

    # ---------- dynamics ----------

    def _apply_walls(self, x: np.ndarray, v: np.ndarray) -> None:
        lower = np.asarray(self.box.lower, dtype=float) + self.radius
        upper = np.asarray(self.box.upper, dtype=float) - self.radius
        if self.boundary == "periodic":
            side = self.box.side
            x -= side * np.floor((x - self.box.lower) / side)
            return
        below = x < lower
        x[below] = (2 * lower - x)[below]
        v[below] *= -1
        above = x > upper
        x[above] = (2 * upper - x)[above]
        v[above] *= -1

    def _separation(self, x: np.ndarray) -> np.ndarray:
        d = x[:, None, :] - x[None, :, :]
        if self.boundary == "periodic":
            side = self.box.side
            d -= side * np.round(d / side)
        return d

    def _collide(self, x: np.ndarray, v: np.ndarray) -> int:
        """Resolve every overlapping, approaching pair; returns the number of collisions."""
        n = x.shape[0]
        d = self._separation(x)
        dist2 = np.sum(d**2, axis=2)
        i, j = np.triu_indices(n, k=1)
        close = dist2[i, j] < (2 * self.radius) ** 2
        if not np.any(close):
            return 0
        i, j = i[close], j[close]
        count = 0
        diameter = 2 * self.radius
        for a, b in zip(
            i, j, strict=True
        ):  # sequential: a disk may be in several pairs
            sep = self._separation(x[[a, b]])[
                0, 1
            ]  # an earlier pair may have moved them
            dist = float(np.linalg.norm(sep))
            if dist == 0.0 or dist >= diameter:
                continue
            dv = v[a] - v[b]
            approach = float(dv @ sep)
            if approach >= 0:  # already separating
                continue
            # Time since contact: the positive root of |sep - dv t|^2 = diameter^2.
            aa = float(dv @ dv)
            cc = dist**2 - diameter**2
            t = min((approach + np.sqrt(approach**2 - aa * cc)) / aa, self.dt)
            x[a] -= v[a] * t
            x[b] -= v[b] * t
            normal = sep - dv * t
            normal /= np.linalg.norm(normal)
            relative = float(dv @ normal)
            v[a] -= relative * normal
            v[b] += relative * normal
            x[a] += v[a] * t
            x[b] += v[b] * t
            count += 1
        return count

    def simulate(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        steps: int,
        record_every: int = 1,
    ) -> GasRun:
        """Integrate ``steps`` time steps; frame 0 is the initial condition."""
        x = np.array(positions, dtype=float)
        v = np.array(velocities, dtype=float)
        if x.shape != v.shape or x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("positions and velocities must both have shape (n, 2)")
        if steps < 0 or record_every < 1:
            raise ValueError("steps must be >= 0 and record_every >= 1")
        frames, speeds, collisions = [x.copy()], [v.copy()], []
        interval_collisions = 0
        for step in range(1, steps + 1):
            x += v * self.dt
            self._apply_walls(x, v)
            interval_collisions += self._collide(x, v)
            self._apply_walls(
                x, v
            )  # overlap removal may have pushed a disk into a wall
            if step % record_every == 0:
                frames.append(x.copy())
                speeds.append(v.copy())
                collisions.append(interval_collisions)
                interval_collisions = 0
        times = self.dt * record_every * np.arange(len(frames))
        trajectory = Trajectory(times, np.stack(frames), self.box)
        return GasRun(trajectory, np.stack(speeds), np.asarray(collisions), self.mass)
