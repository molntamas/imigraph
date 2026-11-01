"""Guardrails against the simulations and their verification getting slow.

Budgets are wall-clock seconds, about four times the time measured in the development container
(reference 0.42 s, graph 0.05 s, comparison 0.03 s, full experiment 0.52 s on 2026-10-30), so
they catch an accidental order-of-magnitude regression without being flaky on a loaded machine.
The best of two runs is taken to reduce noise.
"""

import time
from collections.abc import Callable

import numpy as np
import pytest

from imigraph.compare import compare
from imigraph.experiments import (
    ExperimentConfig,
    run_experiment,
    run_graph,
    run_reference,
)

CONFIG = ExperimentConfig()


def best_of(runs: int, function: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def assert_within_budget(
    label: str, seconds: float, function: Callable[[], object]
) -> None:
    elapsed = best_of(2, function)
    assert elapsed < seconds, f"{label} took {elapsed:.2f} s, budget {seconds} s"


def test_reference_simulation_budget():
    assert_within_budget(
        "reference (2000 particles, 3000 steps)",
        2.0,
        lambda: run_reference(CONFIG, np.random.default_rng(0)),
    )


def test_graph_simulation_budget():
    assert_within_budget(
        "graph (2000 walkers, 3000 steps)",
        0.5,
        lambda: run_graph(CONFIG, np.random.default_rng(0)),
    )


def test_comparison_budget():
    reference = run_reference(CONFIG, np.random.default_rng(0))
    candidate = run_graph(CONFIG, np.random.default_rng(1)).project(
        np.random.default_rng(2)
    )
    assert_within_budget("comparison", 0.5, lambda: compare(reference, candidate))


def test_full_experiment_budget():
    assert_within_budget("experiment", 3.0, lambda: run_experiment(CONFIG))


@pytest.mark.parametrize("particles", [200, 2000])
def test_graph_step_cost_scales_linearly_with_walkers(particles):
    # One rule application is O(walkers): 3000 steps must stay far below the budget even at
    # ten times the walkers of the small case (guards against per-walker Python loops).
    config = ExperimentConfig(particles=particles)
    elapsed = best_of(2, lambda: run_graph(config, np.random.default_rng(0)))
    assert elapsed < 0.5
