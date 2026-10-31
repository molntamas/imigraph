"""End-to-end experiments: a reference system, a graph system imitating it, and the comparison."""

from imigraph.experiments.brownian_vs_graph import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    run_graph,
    run_reference,
)

__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment", "run_graph", "run_reference"]
