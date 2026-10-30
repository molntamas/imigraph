"""Free expansion of a hard-disk gas: energy conserved, momentum tracked, entropy rising,
speeds relaxing to Maxwell-Boltzmann. A reference system on its own (no graph imitation yet)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from imigraph.compare import CheckResult, ComparisonReport
from imigraph.metrics import occupancy_entropy
from imigraph.reference.gas import GasRun, HardDiskGas
from imigraph.trajectory import Box


@dataclass(frozen=True)
class GasConfig:
    particles: int = 120
    radius: float = 0.012
    speed: float = 1.0
    dt: float = 0.004
    steps: int = 3000
    record_every: int = 25
    boundary: Literal["reflecting", "periodic"] = "reflecting"
    seed: int = 0
    entropy_bins: int = 8

    def __post_init__(self) -> None:
        if self.steps % self.record_every:
            raise ValueError("steps must be a multiple of record_every")

    @property
    def box(self) -> Box:
        return Box.square(1.0)

    @property
    def system(self) -> HardDiskGas:
        return HardDiskGas(self.box, self.radius, self.dt, self.boundary)

    @property
    def total_time(self) -> float:
        return self.steps * self.dt

    @property
    def packing_fraction(self) -> float:
        return float(self.particles * np.pi * self.radius**2 / np.prod(self.box.side))

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "total_time": self.total_time,
            "packing_fraction": self.packing_fraction,
        }


@dataclass(frozen=True)
class GasResult:
    config: GasConfig
    run: GasRun
    entropy: np.ndarray
    report: ComparisonReport

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def max_entropy(self) -> float:
        return float(np.log(self.config.entropy_bins**2))


def energy_check(run: GasRun, rtol: float = 1e-9) -> CheckResult:
    energy = run.kinetic_energy()
    drift = float(np.max(np.abs(energy / energy[0] - 1)))
    return CheckResult(
        name="energy",
        passed=drift <= rtol,
        value=drift,
        threshold=rtol,
        larger_is_better=False,
        detail=f"largest relative deviation of the total kinetic energy from its initial value "
        f"{energy[0]:.6g} is {drift:.2e}",
        data={"times": run.times, "energy": energy},
    )


def momentum_check(run: GasRun, boundary: str, rtol: float = 1e-9) -> CheckResult:
    momentum = run.momentum()
    scale = float(np.sqrt(2 * run.mass * run.kinetic_energy()[0] * run.trajectory.num_particles))
    if boundary == "periodic":
        drift = float(np.max(np.abs(momentum - momentum[0])) / scale)
        detail = f"largest change of the total momentum, relative to sqrt(2 m E n): {drift:.2e}"
        return CheckResult(
            "momentum",
            drift <= rtol,
            drift,
            rtol,
            False,
            detail,
            {"times": run.times, "momentum": momentum, "scale": scale},
        )
    # Specular walls exchange momentum with the box: the total must stay a random-walk-like
    # fluctuation of the order of one particle's momentum times sqrt(n), not grow.
    bound = 3.0
    value = float(np.max(np.abs(momentum)) / scale)
    detail = (
        f"largest |total momentum| relative to sqrt(2 m E n) is {value:.3f}; walls exchange "
        f"momentum, so it fluctuates around zero (|p| of every particle is conserved between "
        f"collisions)"
    )
    return CheckResult(
        "momentum",
        value <= bound,
        value,
        bound,
        False,
        detail,
        {"times": run.times, "momentum": momentum, "scale": scale},
    )


def maxwell_check(run: GasRun, alpha: float = 1e-3) -> CheckResult:
    from scipy import stats

    speeds = run.speeds(-1)
    result = stats.kstest(speeds, stats.rayleigh(scale=run.thermal_sigma).cdf)
    return CheckResult(
        name="maxwell",
        passed=bool(result.pvalue > alpha),
        value=float(result.pvalue),
        threshold=alpha,
        larger_is_better=True,
        detail=f"KS test of the final speeds against the 2D Maxwell-Boltzmann (Rayleigh) law "
        f"with sigma = sqrt(E / (n m)) = {run.thermal_sigma:.4g}: p = {result.pvalue:.3g}, "
        f"statistic {result.statistic:.3f}",
        data={"sigma": run.thermal_sigma},
    )


def entropy_check(entropy: np.ndarray, bins: int, particles: int) -> CheckResult:
    """The final occupancy entropy must reach the uniform value up to the finite-sample bias
    ``(bins^2 - 1) / (2 n)`` of the plug-in estimator, with 0.2 nats of slack."""
    max_entropy = float(np.log(bins**2))
    expected = max_entropy - (bins**2 - 1) / (2 * particles)
    threshold = expected - 0.2
    return CheckResult(
        name="entropy",
        passed=bool(entropy[-1] > entropy[0] and entropy[-1] > threshold),
        value=float(entropy[-1]),
        threshold=threshold,
        larger_is_better=True,
        detail=f"occupancy entropy rises from {entropy[0]:.3f} to {entropy[-1]:.3f} nats; "
        f"uniform occupancy gives {max_entropy:.3f}, {expected:.3f} after the finite-sample bias "
        f"of {particles} particles on {bins}x{bins} bins",
        data={"entropy": entropy},
    )


def run_gas(config: GasConfig | None = None) -> GasResult:
    config = GasConfig() if config is None else config
    system = config.system
    rng = np.random.default_rng(config.seed)
    positions, velocities = system.initial_half_box(config.particles, config.speed, rng)
    run = system.simulate(positions, velocities, config.steps, config.record_every)
    entropy = occupancy_entropy(run.trajectory, config.entropy_bins)
    checks = (
        energy_check(run),
        momentum_check(run, config.boundary),
        maxwell_check(run),
        entropy_check(entropy, config.entropy_bins, config.particles),
    )
    report = ComparisonReport("kinetic theory", "hard-disk gas", checks)
    return GasResult(config, run, entropy, report)
