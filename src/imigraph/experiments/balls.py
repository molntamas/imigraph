"""The ball gas on the graph against the hard-disk gas: the same balls, the same start, the same
recording times, so the 2D projection of the graph system can be held against the traditional
simulation (the reference that reflects what we take to be the physics)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from imigraph.compare import CheckResult, ComparisonReport, compare
from imigraph.experiments.gas import (
    GasConfig,
    GasResult,
    collisions_check,
    energy_check,
    entropy_check,
    maxwell_check,
    momentum_check,
)
from imigraph.graph.balls import COLLISION_RULES, collision_rules, place
from imigraph.graph.rules import RuleSpec, rule_item, rule_specs
from imigraph.graph.simulation import GraphHistory
from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph
from imigraph.graph.world import WalkerSystem
from imigraph.metrics import occupancy_entropy
from imigraph.reference.gas import HardDiskGas
from imigraph.trajectory import Box

PLACEMENTS = ("left", "random", "head-on", "diagonal")


@dataclass(frozen=True)
class BallGasConfig:
    """``balls`` balls of ``radius`` on a ``size x size`` room filling the unit box (spacing
    ``1 / size``, step time ``1 / size`` so one cell per step is speed 1). Velocities are
    integers with components up to ``max_speed``. ``placement``: ``left`` (random sites of the
    left half, a free expansion), ``random``, ``head-on`` (two balls on one row flying at each
    other), ``diagonal`` (two balls meeting at right angles, the fallback case). ``rule`` names
    a collision rule set of :mod:`imigraph.graph.balls`, or is ``custom`` with the rule list
    ``rules`` (the playground's rule set, as JSON or :class:`RuleSpec`)."""

    size: int = 10
    balls: int = 6
    radius: int = 0
    max_speed: int = 2
    steps: int = 80
    record_every: int = 1
    seed: int = 0
    entropy_bins: int = 5
    rule: str = "exchange"
    rules: tuple[Any, ...] = field(default=(), compare=False)
    placement: str = "left"
    substeps: int = 8
    """Time steps of the hard-disk reference per lattice step."""

    def __post_init__(self) -> None:
        if self.steps % self.record_every:
            raise ValueError("steps must be a multiple of record_every")
        if self.size < 4:
            raise ValueError("size must be >= 4")
        if self.balls < 1 or self.max_speed < 0 or self.radius < 0:
            raise ValueError("balls must be >= 1, max_speed and radius >= 0")
        if self.rule != "custom" and self.rule not in COLLISION_RULES:
            raise ValueError(
                f"collision rule must be one of {', '.join(COLLISION_RULES)} or custom"
            )
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {', '.join(PLACEMENTS)}")
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1")
        if not self.specs:
            raise ValueError("a rule set needs at least one rule")

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        if self.rule == "custom":
            if not self.rules:
                raise ValueError("a custom rule set needs at least one rule")
            return rule_specs(list(self.rules))
        return collision_rules(self.rule)

    @property
    def rule_names(self) -> str:
        return " + ".join(spec.name for spec in self.specs)

    @property
    def spacing(self) -> float:
        return 1.0 / self.size

    @property
    def step_time(self) -> float:
        return 1.0 / self.size

    @property
    def lattice(self) -> SquareLattice:
        return SquareLattice(self.size, self.spacing)

    @property
    def system(self) -> WalkerSystem:
        return WalkerSystem(
            self.lattice,
            tuple(rule_item(spec) for spec in self.specs),
            self.step_time,
            max_speed=max(1, self.max_speed),
        )

    @property
    def box(self) -> Box:
        return self.lattice.box

    @property
    def disk_radius(self) -> float:
        """The hard disk that stands for a ball: it covers the ball's cells."""
        return (self.radius + 0.5) * self.spacing

    @property
    def total_time(self) -> float:
        return self.steps * self.step_time

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rules"] = self.rule_names
        return {
            **data,
            "spacing": self.spacing,
            "step_time": self.step_time,
            "disk_radius": self.disk_radius,
            "total_time": self.total_time,
        }


def initial_condition(
    config: BallGasConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Integer centres ``(n, 2)`` and velocities ``(n, 2)`` of the balls."""
    size, r, n = config.size, config.radius, config.balls
    if config.placement in ("head-on", "diagonal"):
        mid = size // 2
        if config.placement == "head-on":
            centres = np.array([[r, mid], [size - 1 - r, mid]])
            velocities = np.array([[max(1, config.max_speed), 0], [-1, 0]])
        else:
            centres = np.array([[r, mid], [mid, r]])
            velocities = np.array([[1, 0], [0, 1]])
        if n != 2:
            raise ValueError(f"placement {config.placement} is for exactly 2 balls")
        return centres, velocities
    columns = max(1, size // 2) if config.placement == "left" else size
    i = np.arange(r, columns - r)
    j = np.arange(r, size - r)
    cells = np.array([(a, b) for a in i for b in j])
    # Balls of radius r need 2r + 1 cells of separation to avoid overlapping.
    cells = cells[
        (cells[:, 0] % (2 * r + 1) == r % (2 * r + 1))
        & (cells[:, 1] % (2 * r + 1) == r % (2 * r + 1))
    ]
    if cells.shape[0] < n:
        raise ValueError(
            f"{n} balls of radius {r} do not fit into the placement region "
            f"({cells.shape[0]} places)"
        )
    chosen = rng.choice(cells.shape[0], size=n, replace=False)
    centres = cells[chosen]
    velocities = rng.integers(-config.max_speed, config.max_speed + 1, size=(n, 2))
    still = ~np.any(velocities != 0, axis=1)
    velocities[still, 0] = 1 if config.max_speed > 0 else 0
    return centres, velocities


@dataclass(frozen=True)
class BallResult:
    """The graph system's run with the gas checks (energy, momentum, collisions, entropy; the
    Maxwell check is informative with few balls) and its exact lattice invariants."""

    config: BallGasConfig
    run: GraphHistory
    entropy: np.ndarray
    report: ComparisonReport
    informative: tuple[CheckResult, ...] = ()
    invariants: tuple[dict[str, Any], ...] = ()
    rules: tuple[dict[str, str], ...] = ()

    @property
    def passed(self) -> bool:
        return self.report.passed and all(row["passed"] for row in self.invariants)

    @property
    def max_entropy(self) -> float:
        return float(np.log(self.config.entropy_bins**2))


@dataclass(frozen=True)
class BallComparison:
    """The reference hard-disk gas and the graph ball gas from the same initial condition, plus
    the distribution checks of the graph projection against the reference."""

    gas: GasResult[GasConfig]
    balls: BallResult
    report: ComparisonReport

    @property
    def passed(self) -> bool:
        return self.gas.passed and self.balls.passed


def initial_state(config: BallGasConfig) -> WalkerGraph:
    centres, velocities = initial_condition(config, np.random.default_rng(config.seed))
    return place(config.lattice, centres, velocities, config.radius)


def run_ball_gas(
    config: BallGasConfig | None = None, initial: WalkerGraph | None = None
) -> BallResult:
    config = BallGasConfig() if config is None else config
    system = config.system
    if initial is None:
        initial = initial_state(config)
    run, _ = system.run(
        initial,
        config.steps,
        np.random.default_rng(config.seed + 1),
        record_every=config.record_every,
    )
    gas_run = run.to_gas_run()
    entropy = occupancy_entropy(gas_run.trajectory, config.entropy_bins)
    interacting = system.interacting
    checks = (
        energy_check(gas_run),
        momentum_check(gas_run, "reflecting"),
        collisions_check(gas_run)
        if config.balls > 1 and interacting
        else energy_check(gas_run),
    )
    informative: tuple[CheckResult, ...] = (
        entropy_check(entropy, config.entropy_bins, config.balls),
    )
    if config.balls >= 2:
        informative += (maxwell_check(gas_run),)
    return BallResult(
        config,
        run,
        entropy,
        ComparisonReport("kinetic theory", "ball gas on the graph", checks),
        informative,
        tuple(run.checks(config.entropy_bins, interacting)),
        tuple(system.descriptions(config.spacing, config.step_time)),
    )


def run_reference(config: BallGasConfig, initial: WalkerGraph) -> GasResult[GasConfig]:
    """The hard-disk gas started exactly where the balls start, with their velocities, disks
    covering the balls' cells, recorded at the lattice steps."""
    speed_unit = config.spacing / config.step_time
    velocities = initial.velocity * speed_unit
    gas_config = GasConfig(
        particles=config.balls,
        radius=config.disk_radius,
        speed=float(np.mean(np.linalg.norm(velocities, axis=1))),
        dt=config.step_time / config.substeps,
        steps=config.steps * config.substeps,
        record_every=config.record_every * config.substeps,
        boundary="reflecting",
        seed=config.seed,
        entropy_bins=config.entropy_bins,
        box=config.box,
    )
    system = HardDiskGas(config.box, gas_config.radius, gas_config.dt, "reflecting")
    run = system.simulate(
        initial.centres(), velocities, gas_config.steps, gas_config.record_every
    )
    entropy = occupancy_entropy(run.trajectory, config.entropy_bins)
    checks = (
        energy_check(run),
        momentum_check(run, "reflecting"),
        collisions_check(run) if config.balls > 1 else energy_check(run),
    )
    informative: tuple[CheckResult, ...] = (
        entropy_check(entropy, config.entropy_bins, config.balls),
    )
    if config.balls >= 2:
        informative += (maxwell_check(run),)
    return GasResult(
        gas_config,
        run,
        entropy,
        ComparisonReport("kinetic theory", "hard-disk gas", checks),
        informative,
    )


def run_ball_comparison(config: BallGasConfig | None = None) -> BallComparison:
    """Both systems from the same initial condition, compared at the same recording times."""
    config = BallGasConfig() if config is None else config
    initial = initial_state(config)
    balls = run_ball_gas(config, initial)
    gas = run_reference(config, initial)
    report = compare(
        gas.run.trajectory,
        balls.run.project(),
        reference_label="hard-disk gas",
        candidate_label="ball gas on the graph",
    )
    return BallComparison(gas, balls, report)
