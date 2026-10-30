"""The common trajectory format shared by every system imigraph simulates or imitates.

A :class:`Trajectory` is what the comparison framework consumes, so a reference simulation (e.g.
Brownian motion) and a graph simulation projected to 2D are compared through the same object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Box:
    """An axis-aligned 2D box ``[lower[0], upper[0]] x [lower[1], upper[1]]``."""

    lower: tuple[float, float]
    upper: tuple[float, float]

    def __post_init__(self) -> None:
        if not all(u > lo for lo, u in zip(self.lower, self.upper, strict=True)):
            raise ValueError(f"box upper bounds must exceed lower bounds: {self}")

    @classmethod
    def square(cls, side: float, lower: float = 0.0) -> Box:
        return cls((lower, lower), (lower + side, lower + side))

    @property
    def side(self) -> np.ndarray:
        """Edge lengths, shape ``(2,)``."""
        return np.asarray(self.upper, dtype=float) - np.asarray(self.lower, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.lower, dtype=float) + np.asarray(self.upper, dtype=float)) / 2

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask over the leading axes of ``points`` (shape ``(..., 2)``)."""
        points = np.asarray(points, dtype=float)
        return np.all((points >= self.lower) & (points <= self.upper), axis=-1)

    def to_dict(self) -> dict[str, Any]:
        return {"lower": list(self.lower), "upper": list(self.upper)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Box:
        return cls(tuple(data["lower"]), tuple(data["upper"]))


@dataclass(frozen=True)
class Trajectory:
    """Positions of ``n`` particles at ``T`` recorded times inside a box.

    ``times`` has shape ``(T,)``, ``positions`` has shape ``(T, n, 2)``. Frame 0 is the initial
    condition.
    """

    times: np.ndarray
    positions: np.ndarray
    box: Box

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        positions = np.asarray(self.positions, dtype=float)
        if times.ndim != 1:
            raise ValueError(f"times must be 1D, got shape {times.shape}")
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError(f"positions must have shape (T, n, 2), got {positions.shape}")
        if positions.shape[0] != times.shape[0]:
            raise ValueError(f"{positions.shape[0]} frames of positions but {times.shape[0]} times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.positions.shape[1])

    def displacements(self) -> np.ndarray:
        """Displacement of every particle from its initial position, shape ``(T, n, 2)``."""
        return self.positions - self.positions[0]

    def msd(self) -> np.ndarray:
        """Mean squared displacement from frame 0, averaged over particles, shape ``(T,)``."""
        return np.mean(np.sum(self.displacements() ** 2, axis=-1), axis=1)

    def msd_standard_error(self) -> np.ndarray:
        """Standard error of :meth:`msd` from the spread over particles, shape ``(T,)``."""
        squared = np.sum(self.displacements() ** 2, axis=-1)
        if self.num_particles < 2:
            return np.zeros(self.num_frames)
        return np.std(squared, axis=1, ddof=1) / np.sqrt(self.num_particles)

    def frame(self, index: int) -> np.ndarray:
        return self.positions[index]

    def subsample(self, every: int) -> Trajectory:
        """Keep every ``every``-th frame (frame 0 always included)."""
        if every < 1:
            raise ValueError("every must be >= 1")
        return Trajectory(self.times[::every], self.positions[::every], self.box)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (used for export and the HTML report)."""
        return {
            "times": self.times.tolist(),
            "positions": self.positions.tolist(),
            "box": self.box.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            np.asarray(data["times"], dtype=float),
            np.asarray(data["positions"], dtype=float),
            Box.from_dict(data["box"]),
        )
