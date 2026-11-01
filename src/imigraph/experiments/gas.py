"""Free expansion of a gas that conserves energy: the hard-disk gas (reference) and the lattice gas
on the walker graph (candidate), each checked against kinetic theory and compared with each other.

Both start in the left half of the unit box with speed 1 and are recorded at the same times, so the
distribution checks of :mod:`imigraph.compare` can be run on the two projected trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np

from imigraph.compare import CheckResult, ComparisonReport, compare
from imigraph.graph.gas import LatticeGas
from imigraph.graph.space import SquareLattice
from imigraph.metrics import occupancy_entropy
from imigraph.reference.gas import GasRun, HardDiskGas
from imigraph.trajectory import Box

Boundary = Literal["reflecting", "periodic"]


@dataclass(frozen=True)
class GasConfig:
    particles: int = 120
    radius: float = 0.012
    speed: float = 1.0
    dt: float = 0.004
    steps: int = 3000
    record_every: int = 25
    boundary: Boundary = "reflecting"
    seed: int = 0
    entropy_bins: int = 8
    box: Box = field(default_factory=lambda: Box.square(1.0))

    def __post_init__(self) -> None:
        if self.steps % self.record_every:
            raise ValueError("steps must be a multiple of record_every")

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
            "box": self.box.to_dict(),
            "total_time": self.total_time,
            "packing_fraction": self.packing_fraction,
        }


@dataclass(frozen=True)
class LatticeGasConfig:
    """The lattice gas on a ``size x size`` lattice filling the unit box: spacing ``1 / size``,
    step time ``spacing / speed`` so every walker moves at ``speed``. The defaults record every
    0.1 time units for 12 time units, like :class:`GasConfig`."""

    size: int = 40
    particles: int = 120
    speed: float = 1.0
    steps: int = 480
    record_every: int = 4
    boundary: Boundary = "reflecting"
    seed: int = 0
    entropy_bins: int = 8

    def __post_init__(self) -> None:
        if self.steps % self.record_every:
            raise ValueError("steps must be a multiple of record_every")
        if self.size < 3 or self.speed <= 0:
            raise ValueError("size must be >= 3 and speed > 0")

    @property
    def spacing(self) -> float:
        return 1.0 / self.size

    @property
    def step_time(self) -> float:
        return self.spacing / self.speed

    @property
    def lattice(self) -> SquareLattice:
        return SquareLattice(
            self.size, self.spacing, periodic=self.boundary == "periodic"
        )

    @property
    def system(self) -> LatticeGas:
        return LatticeGas(self.lattice, self.step_time)

    @property
    def box(self) -> Box:
        return self.lattice.box

    @property
    def total_time(self) -> float:
        return self.steps * self.step_time

    @property
    def channel_fraction(self) -> float:
        """Occupied fraction of the ``size^2`` sites (the lattice gas's density)."""
        return self.particles / self.size**2

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "spacing": self.spacing,
            "step_time": self.step_time,
            "total_time": self.total_time,
            "channel_fraction": self.channel_fraction,
        }


@dataclass(frozen=True)
class GasResult[Config: GasConfig | LatticeGasConfig]:
    """One system's run with its checks. ``informative`` holds checks that are reported but do
    not count towards the verdict (the Maxwell check of the lattice gas, which has one speed)."""

    config: Config
    run: GasRun
    entropy: np.ndarray
    report: ComparisonReport
    informative: tuple[CheckResult, ...] = ()
    rules: tuple[dict[str, str], ...] = ()

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def max_entropy(self) -> float:
        return float(np.log(self.config.entropy_bins**2))


@dataclass(frozen=True)
class GasComparison:
    """Both gases from the same kind of initial condition, plus the distribution checks of the
    lattice gas against the hard-disk gas."""

    gas: GasResult[GasConfig]
    lattice: GasResult[LatticeGasConfig]
    report: ComparisonReport

    @property
    def passed(self) -> bool:
        """Verdict of the conservation checks of both systems (the cross-comparison is shown but
        not required: a single-speed lattice gas is not expected to match a hard-disk gas in
        every distribution test)."""
        return self.gas.passed and self.lattice.passed


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
    scale = float(
        np.sqrt(2 * run.mass * run.kinetic_energy()[0] * run.trajectory.num_particles)
    )
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


def collisions_check(run: GasRun) -> CheckResult:
    """Particles must actually have interacted: without collisions a lattice gas is a set of
    independent walkers and the entropy rise says nothing about the collision rule."""
    total = int(run.collisions.sum())
    return CheckResult(
        name="collisions",
        passed=total > 0,
        value=float(total),
        threshold=0.0,
        larger_is_better=True,
        detail=f"{total} collisions in the run ({total / run.trajectory.num_particles:.1f} per "
        "particle)",
        data={"collisions": run.collisions},
    )


def run_gas(config: GasConfig | None = None) -> GasResult[GasConfig]:
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
    return GasResult(
        config,
        run,
        entropy,
        ComparisonReport("kinetic theory", "hard-disk gas", checks),
    )


def run_lattice_gas(
    config: LatticeGasConfig | None = None,
) -> GasResult[LatticeGasConfig]:
    """The lattice gas with the same checks; the Maxwell check is informative only, because every
    walker keeps the speed ``c``: HPP conserves energy but has no thermal speed distribution."""
    config = LatticeGasConfig() if config is None else config
    system = config.system
    rng = np.random.default_rng(config.seed)
    initial = system.initial_half_box(config.particles, rng)
    run = system.simulate(initial, config.steps, rng, config.record_every)
    entropy = occupancy_entropy(run.trajectory, config.entropy_bins)
    checks = (
        energy_check(run),
        momentum_check(run, config.boundary),
        collisions_check(run),
        entropy_check(entropy, config.entropy_bins, config.particles),
    )
    return GasResult(
        config,
        run,
        entropy,
        ComparisonReport("kinetic theory", "lattice gas on the graph", checks),
        informative=(maxwell_check(run),),
        rules=tuple(system.describe_rules()),
    )


def run_gas_comparison(
    gas_config: GasConfig | None = None, lattice_config: LatticeGasConfig | None = None
) -> GasComparison:
    """Run both gases in the lattice's box at the same recording times and compare them."""
    lattice_config = LatticeGasConfig() if lattice_config is None else lattice_config
    gas_config = GasConfig() if gas_config is None else gas_config
    gas_config = replace(gas_config, box=lattice_config.box)
    gas = run_gas(gas_config)
    lattice = run_lattice_gas(lattice_config)
    report = compare(
        gas.run.trajectory,
        lattice.run.trajectory,
        reference_label="hard-disk gas",
        candidate_label="lattice gas on the graph",
    )
    return GasComparison(gas, lattice, report)
