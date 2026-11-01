"""Brownian motion in a box, imitated by random hops of walker nodes on a lattice graph.

The lattice spacing ``a`` and the duration ``tau`` of one rule application fix the diffusion
coefficient ``D = a^2 / (4 tau)`` of the graph system; the reference is run with that ``D`` in the
box of the lattice, from the same initial condition (everything at the centre), and recorded at
the same times. The comparison framework then decides whether the two agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from imigraph.compare import ComparisonReport, compare
from imigraph.graph import GraphHistory, GraphSimulation, SquareLattice, walkers_at_node
from imigraph.graph.rules import rule_from_spec, rule_specs
from imigraph.graph.simulation import lattice_hop_diffusion
from imigraph.metrics import Metrics, StepObserver, compute_metrics
from imigraph.reference import BrownianBox
from imigraph.trajectory import Trajectory


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters of the experiment; the defaults run in about a second."""

    lattice_size: int = 41
    spacing: float = 0.2
    step_time: float = 0.01
    particles: int = 2000
    steps: int = 3000
    record_every: int = 50
    seed: int = 0
    entropy_bins: int = 8
    rule: dict[str, Any] | list[dict[str, Any]] | None = None
    """A :class:`~imigraph.graph.rules.RuleSpec` as JSON, or a list of them (a
    :class:`~imigraph.graph.rules.RuleSequence`, applied in order in every step); ``None``
    runs :class:`RandomHop`. The reference keeps ``D = a^2 / (4 tau)``, so a rule with another
    diffusion or a drift is *meant* to fail the comparison: that is the experiment."""

    def __post_init__(self) -> None:
        if self.steps % self.record_every:
            raise ValueError("steps must be a multiple of record_every")
        if self.particles < 2:
            raise ValueError("at least two particles are needed for the statistics")
        if self.entropy_bins < 1:
            raise ValueError("entropy_bins must be >= 1")
        rule_specs(self.rule)  # validate early, with the editor's error message

    @property
    def diffusion(self) -> float:
        """Diffusion coefficient shared by both systems."""
        return lattice_hop_diffusion(self.spacing, self.step_time)

    @property
    def lattice(self) -> SquareLattice:
        return SquareLattice(self.lattice_size, self.spacing)

    @property
    def reference_system(self) -> BrownianBox:
        return BrownianBox(
            diffusion=self.diffusion, dt=self.step_time, box=self.lattice.box
        )

    @property
    def graph_system(self) -> GraphSimulation:
        return GraphSimulation(
            self.lattice, rule_from_spec(self.rule), step_time=self.step_time
        )

    @property
    def rule_name(self) -> str:
        return " + ".join(spec.name for spec in rule_specs(self.rule))

    @property
    def total_time(self) -> float:
        return self.steps * self.step_time

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "diffusion": self.diffusion,
            "total_time": self.total_time,
        }


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    reference: Trajectory
    graph_history: GraphHistory
    graph_projection: Trajectory
    report: ComparisonReport
    reference_metrics: Metrics
    graph_metrics: Metrics

    @property
    def passed(self) -> bool:
        return self.report.passed


def run_reference(
    config: ExperimentConfig,
    rng: np.random.Generator,
    observer: StepObserver | None = None,
) -> Trajectory:
    system = config.reference_system
    initial = system.all_at(config.lattice.box.center, config.particles)
    return system.simulate(initial, config.steps, rng, config.record_every, observer)


def run_graph(
    config: ExperimentConfig,
    rng: np.random.Generator,
    observer: StepObserver | None = None,
) -> GraphHistory:
    lattice = config.lattice
    initial = walkers_at_node(lattice, lattice.center_node, config.particles)
    return config.graph_system.run(
        initial, config.steps, rng, config.record_every, observer
    )


def metrics_of(
    config: ExperimentConfig, trajectory: Trajectory, observer: StepObserver
) -> Metrics:
    return compute_metrics(
        trajectory,
        observer,
        dt=config.step_time,
        record_every=config.record_every,
        diffusion=config.diffusion,
        bins=config.entropy_bins,
        cells=config.lattice_size,
    )


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    """Run both systems from ``config`` (default :class:`ExperimentConfig`) and compare them.

    The reference and the graph system use independent random streams derived from the seed, so
    changing one system does not change the other's run.
    """
    config = ExperimentConfig() if config is None else config
    reference_rng, graph_rng, cell_rng = np.random.default_rng(config.seed).spawn(3)
    reference_observer, graph_observer = StepObserver(), StepObserver()
    reference = run_reference(config, reference_rng, reference_observer)
    history = run_graph(config, graph_rng, graph_observer)
    projection = history.project(cell_rng=cell_rng)
    report = compare(
        reference,
        projection,
        reference_label="Brownian motion in a box",
        candidate_label=(
            "random hops on a lattice graph"
            if config.rule is None
            else f"rule {config.rule_name} on a lattice graph"
        ),
    )
    return ExperimentResult(
        config,
        reference,
        history,
        projection,
        report,
        reference_metrics=metrics_of(config, reference, reference_observer),
        graph_metrics=metrics_of(config, history.project(), graph_observer),
    )
