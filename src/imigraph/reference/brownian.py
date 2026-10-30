"""Brownian motion of independent particles in a 2D box with reflecting walls.

Overdamped Langevin dynamics without drift, integrated with the Euler-Maruyama scheme::

    x(t + dt) = reflect(x(t) + sqrt(2 D dt) * xi),   xi ~ N(0, I_2)

Analytic facts used by the tests and the comparison framework:

* free diffusion (before the walls matter): ``<|x(t) - x(0)|^2> = 4 D t`` in two dimensions,
  and each coordinate is Gaussian with variance ``2 D t``;
* stationary state: uniform density over the box (reflecting walls conserve particles).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imigraph.metrics import StepObserver
from imigraph.trajectory import Box, Trajectory


def reflect_into_box(points: np.ndarray, box: Box) -> np.ndarray:
    """Fold points outside the box back inside by specular reflection at the walls.

    Works for excursions of any length (several bounces) by folding the coordinate onto the
    period ``2 * side`` of the reflected motion.
    """
    lower = np.asarray(box.lower, dtype=float)
    side = box.side
    period = 2 * side
    folded = np.mod(points - lower, period)
    folded = np.where(folded > side, period - folded, folded)
    return lower + folded


@dataclass(frozen=True)
class BrownianBox:
    """Parameters of the reference system.

    ``diffusion`` is the diffusion coefficient ``D`` (length^2 / time), ``dt`` the time step.
    """

    diffusion: float
    dt: float
    box: Box

    def __post_init__(self) -> None:
        if self.diffusion < 0:
            raise ValueError("diffusion must be >= 0")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")

    @property
    def step_scale(self) -> float:
        """Standard deviation of one coordinate increment per step, ``sqrt(2 D dt)``."""
        return float(np.sqrt(2 * self.diffusion * self.dt))

    def simulate(
        self,
        initial: np.ndarray,
        steps: int,
        rng: np.random.Generator,
        record_every: int = 1,
        observer: StepObserver | None = None,
    ) -> Trajectory:
        """Run ``steps`` time steps from ``initial`` (shape ``(n, 2)``).

        Every ``record_every``-th state is recorded, frame 0 is the initial condition, so the
        trajectory has ``steps // record_every + 1`` frames. ``observer(step, previous, x)`` is
        called after every step (see :mod:`imigraph.metrics`).
        """
        initial = np.asarray(initial, dtype=float)
        if initial.ndim != 2 or initial.shape[1] != 2:
            raise ValueError(f"initial must have shape (n, 2), got {initial.shape}")
        if not np.all(self.box.contains(initial)):
            raise ValueError("initial positions must lie inside the box")
        if steps < 0 or record_every < 1:
            raise ValueError("steps must be >= 0 and record_every >= 1")

        frames = [initial]
        x = initial
        for step in range(1, steps + 1):
            previous = x
            proposed = x + self.step_scale * rng.standard_normal(x.shape)
            x = reflect_into_box(proposed, self.box)
            if observer is not None:
                observer(step, previous, x, wall_hits=~self.box.contains(proposed))
            if step % record_every == 0:
                frames.append(x)
        times = self.dt * record_every * np.arange(len(frames))
        return Trajectory(times, np.stack(frames), self.box)

    def all_at(self, point: np.ndarray | tuple[float, float], count: int) -> np.ndarray:
        """Initial condition with every particle at ``point``, shape ``(count, 2)``."""
        return np.tile(np.asarray(point, dtype=float), (count, 1))
