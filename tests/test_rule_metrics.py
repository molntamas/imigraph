"""Rule-level metrics: wall interactions and spreading over cells, for both systems."""

import numpy as np
import pytest

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.graph import RandomHop
from imigraph.metrics import StepObserver, expected_occupied_cells, occupied_cells
from imigraph.reference import BrownianBox
from imigraph.trajectory import Box, Trajectory


@pytest.fixture(scope="module")
def result():
    return run_experiment(ExperimentConfig())


def test_no_wall_hits_far_from_the_walls():
    system = BrownianBox(diffusion=1.0, dt=0.01, box=Box.square(1000.0))
    observer = StepObserver()
    system.simulate(system.all_at((500.0, 500.0), 100), 20, np.random.default_rng(0), 1, observer)
    assert observer.wall_fraction == [0.0] * 20


def test_occupied_cells_counts_distinct_cells():
    box = Box.square(4.0)
    positions = np.array(
        [[[0.5, 0.5], [0.6, 0.6], [3.5, 3.5]], [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]]
    )
    trajectory = Trajectory([0.0, 1.0], positions, box)
    np.testing.assert_array_equal(occupied_cells(trajectory, cells=4), [2, 3])
    assert expected_occupied_cells(particles=16, cells=4) == pytest.approx(16 * (1 - np.exp(-1)))


def test_wall_rate_starts_at_zero_and_settles_in_both_systems(result):
    for metrics in (result.reference_metrics, result.graph_metrics):
        rate = metrics.wall_rate
        assert rate[0] < 1e-3  # nobody has reached a wall in the first interval
        late = rate[len(rate) // 2 :]
        assert 0.01 < late.mean() < 0.05
        assert abs(late[: len(late) // 2].mean() / late[len(late) // 2 :].mean() - 1) < 0.15
    ratio = (
        result.graph_metrics.wall_rate[-10:].mean()
        / result.reference_metrics.wall_rate[-10:].mean()
    )
    assert 0.5 < ratio < 2.0  # same physics, different discretisation of the wall


def test_occupied_cells_spread_to_the_uniform_value_in_both_systems(result):
    for metrics in (result.reference_metrics, result.graph_metrics):
        assert metrics.occupied[0] == 1
        assert np.all(np.diff(metrics.occupied[:5]) > 0)
        assert abs(metrics.occupied[-5:].mean() / metrics.expected_occupied - 1) < 0.05
    ratio = (
        result.graph_metrics.occupied[-5:].mean() / result.reference_metrics.occupied[-5:].mean()
    )
    assert abs(ratio - 1) < 0.05


def test_rule_describes_itself():
    description = RandomHop().describe()
    assert description["rewrite"] == "{w,s}, {s,s'} -> {w,s'}, {s,s'}"
    assert set(description) == {"name", "rewrite", "matches", "choice", "invariants", "physics"}
