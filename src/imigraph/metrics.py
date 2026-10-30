"""Physical metrics of a run: kinetic energy, momentum and entropy.

Overdamped Brownian particles trade energy and momentum with the bath, so neither is conserved
particle by particle. What is stationary is the ensemble: the mean kinetic energy per particle
(the temperature) and the total momentum (mean zero, fluctuating). Entropy is the Shannon
entropy of a coarse-grained occupancy histogram; it rises from zero (everything at the centre)
towards ``ln(bins)`` (uniform), statistically but not monotonically.

Energy and momentum are measured from the *fine* simulation steps through a
:class:`StepObserver` passed to the simulators, not from the recorded frames: between two
recorded frames a particle may have bounced off a wall, which would shrink its displacement and
look like a loss of energy.

Units: the particle mass ``m`` is 1 unless given; velocity of a step is ``displacement / dt``.
For free Brownian motion ``<|dr|^2> = 4 D dt``, so the mean kinetic energy per particle is
``m |dr|^2 / (2 dt^2) = 2 m D / dt`` and the total momentum of ``n`` particles has zero mean and
standard deviation ``sqrt(2 n m^2 D / dt)`` per component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imigraph.trajectory import Trajectory


class StepObserver:
    """Collects the increment statistics of every simulation step.

    Simulators call ``observer(step, previous, current)`` with the positions before and after a
    step, shape ``(n, 2)``. The observer keeps, per step, the mean squared displacement over
    particles and the summed displacement vector.
    """

    def __init__(self) -> None:
        self.mean_squared: list[float] = []
        self.summed: list[np.ndarray] = []
        self.num_particles: int | None = None

    def __call__(self, step: int, previous: np.ndarray, current: np.ndarray) -> None:
        delta = current - previous
        self.num_particles = int(delta.shape[0])
        self.mean_squared.append(float(np.mean(np.sum(delta**2, axis=1))))
        self.summed.append(delta.sum(axis=0))

    @property
    def steps(self) -> int:
        return len(self.mean_squared)

    def kinetic_energy(self, dt: float, mass: float = 1.0) -> np.ndarray:
        """Mean kinetic energy per particle at every step, shape ``(steps,)``."""
        return mass * np.asarray(self.mean_squared) / (2 * dt**2)

    def momentum(self, dt: float, mass: float = 1.0) -> np.ndarray:
        """Total momentum vector at every step, shape ``(steps, 2)``."""
        if not self.summed:
            return np.zeros((0, 2))
        return mass * np.stack(self.summed) / dt


def occupancy_entropy(trajectory: Trajectory, bins: int = 8) -> np.ndarray:
    """Shannon entropy (nats) of the ``bins x bins`` occupancy histogram at every frame.

    The plug-in estimator is biased low by about ``(bins^2 - 1) / (2 n)`` for ``n`` particles;
    its maximum is ``ln(bins^2)`` for a uniform occupancy.
    """
    box = trajectory.box
    extent = [[box.lower[0], box.upper[0]], [box.lower[1], box.upper[1]]]
    entropy = np.empty(trajectory.num_frames)
    for frame in range(trajectory.num_frames):
        counts, _, _ = np.histogram2d(
            trajectory.positions[frame, :, 0],
            trajectory.positions[frame, :, 1],
            bins=bins,
            range=extent,
        )
        probabilities = counts[counts > 0] / counts.sum()
        entropy[frame] = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy


def aggregate_per_interval(values: np.ndarray, record_every: int) -> np.ndarray:
    """Mean of per-step values over each recording interval; leading axis ``steps`` becomes
    ``steps // record_every``."""
    values = np.asarray(values)
    frames = values.shape[0] // record_every
    trimmed = values[: frames * record_every]
    return trimmed.reshape(frames, record_every, *values.shape[1:]).mean(axis=1)


@dataclass(frozen=True)
class Metrics:
    """Metric series of one run, aligned with the recorded frames of its trajectory.

    ``energy`` and ``momentum`` are interval means of the per-step values, given at the end
    time of each interval (``times[1:]``); ``entropy`` is given at every frame (``times``).
    """

    times: np.ndarray
    energy: np.ndarray
    momentum: np.ndarray
    entropy: np.ndarray
    expected_energy: float
    momentum_std: float
    max_entropy: float
    extra: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def interval_times(self) -> np.ndarray:
        return self.times[1:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "energy": self.energy.tolist(),
            "momentum": self.momentum.tolist(),
            "entropy": self.entropy.tolist(),
            "expected_energy": self.expected_energy,
            "momentum_std": self.momentum_std,
            "max_entropy": self.max_entropy,
        }


def compute_metrics(
    trajectory: Trajectory,
    observer: StepObserver,
    dt: float,
    record_every: int,
    diffusion: float,
    bins: int = 8,
    mass: float = 1.0,
) -> Metrics:
    """Combine a recorded trajectory and the step observer of the same run into :class:`Metrics`.

    ``expected_energy`` and ``momentum_std`` are the free-diffusion values for the given
    ``diffusion``; ``momentum_std`` refers to the interval mean of the momentum, which averages
    ``record_every`` independent steps.
    """
    if observer.steps != (trajectory.num_frames - 1) * record_every:
        raise ValueError(
            f"observer saw {observer.steps} steps but the trajectory has "
            f"{trajectory.num_frames} frames recorded every {record_every} steps"
        )
    n = trajectory.num_particles
    return Metrics(
        times=trajectory.times,
        energy=aggregate_per_interval(observer.kinetic_energy(dt, mass), record_every),
        momentum=aggregate_per_interval(observer.momentum(dt, mass), record_every),
        entropy=occupancy_entropy(trajectory, bins),
        expected_energy=2 * mass * diffusion / dt,
        momentum_std=float(np.sqrt(2 * n * mass**2 * diffusion / dt) / np.sqrt(record_every)),
        max_entropy=float(np.log(bins * bins)),
    )
