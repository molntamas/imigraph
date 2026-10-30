"""Energy and momentum are stationary in both systems, entropy increases statistically."""

import numpy as np
import pytest
from scipy import stats

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.metrics import (
    StepObserver,
    aggregate_per_interval,
    compute_metrics,
    occupancy_entropy,
)
from imigraph.reference import BrownianBox
from imigraph.trajectory import Box, Trajectory


@pytest.fixture(scope="module")
def result():
    return run_experiment(ExperimentConfig())


def test_observer_collects_every_step_and_the_free_energy_is_2d_over_dt():
    diffusion, dt, n = 0.5, 0.05, 4000
    system = BrownianBox(diffusion=diffusion, dt=dt, box=Box.square(1000.0))
    observer = StepObserver()
    trajectory = system.simulate(
        system.all_at((500.0, 500.0), n), 100, np.random.default_rng(3), 10, observer
    )
    assert observer.steps == 100 and observer.num_particles == n
    energy = observer.kinetic_energy(dt)
    expected = 2 * diffusion / dt
    # |dr|^2 of one particle has relative std 1, so the per-step mean over n has 1/sqrt(n) = 1.6%.
    np.testing.assert_allclose(energy.mean(), expected, rtol=0.03)
    assert np.all(np.abs(energy / expected - 1) < 0.1)
    momentum = observer.momentum(dt)
    std = np.sqrt(2 * n * diffusion / dt)
    assert np.all(np.abs(momentum) < 5 * std)
    np.testing.assert_allclose(momentum.std(axis=0), std, rtol=0.25)
    metrics = compute_metrics(trajectory, observer, dt, 10, diffusion, bins=4)
    assert metrics.energy.shape == (10,) and metrics.momentum.shape == (10, 2)
    assert metrics.entropy.shape == (11,)
    assert metrics.expected_energy == pytest.approx(expected)


def test_aggregate_per_interval_means_blocks():
    values = np.arange(12.0)
    np.testing.assert_allclose(aggregate_per_interval(values, 4), [1.5, 5.5, 9.5])
    vectors = np.stack([values, -values], axis=1)
    np.testing.assert_allclose(aggregate_per_interval(vectors, 6)[:, 1], [-2.5, -8.5])


def test_entropy_of_point_and_uniform_occupancies():
    box = Box.square(1.0)
    point = np.full((1, 1000, 2), 0.5)
    assert occupancy_entropy(Trajectory([0.0], point, box), bins=4)[0] == 0.0
    rng = np.random.default_rng(0)
    uniform = rng.random((1, 100_000, 2))
    entropy = occupancy_entropy(Trajectory([0.0], uniform, box), bins=4)[0]
    assert entropy == pytest.approx(np.log(16), abs=0.01)


def test_energy_is_stationary_and_equal_in_both_systems(result):
    for metrics in (result.reference_metrics, result.graph_metrics):
        energy = metrics.energy
        half = len(energy) // 2
        # No trend beyond the small loss at the walls: halves agree within 5%.
        assert abs(energy[half:].mean() / energy[:half].mean() - 1) < 0.05
        assert abs(energy.mean() / metrics.expected_energy - 1) < 0.06
    ratio = result.graph_metrics.energy.mean() / result.reference_metrics.energy.mean()
    assert abs(ratio - 1) < 0.05


def test_momentum_stays_zero_within_its_fluctuations(result):
    for metrics in (result.reference_metrics, result.graph_metrics):
        z = metrics.momentum / metrics.momentum_std
        assert np.all(np.abs(z) < 5)
        # The time average over 60 intervals has standard error std / sqrt(60).
        assert np.all(np.abs(z.mean(axis=0)) < 4 / np.sqrt(len(z)))
        assert 0.6 < z.std() < 1.5


def test_entropy_increases_statistically_but_not_monotonically(result):
    for metrics in (result.reference_metrics, result.graph_metrics):
        entropy = metrics.entropy
        assert entropy[0] == 0.0
        assert entropy[-1] > 0.9 * metrics.max_entropy
        # Rising phase (until 95% of the maximum) is monotone: the big early jumps dwarf noise.
        reached = int(np.argmax(entropy >= 0.95 * metrics.max_entropy))
        assert 2 <= reached < len(entropy) // 2
        assert np.all(np.diff(entropy[: reached + 1]) > 0)
        # Plateau: still rising on average (rank correlation > 0), but with dips.
        plateau = entropy[reached:]
        assert stats.spearmanr(np.arange(len(plateau)), plateau).statistic > 0
        assert np.any(np.diff(plateau) < 0), "a finite ensemble must fluctuate"
    difference = result.graph_metrics.entropy - result.reference_metrics.entropy
    assert np.max(np.abs(difference)) < 0.15


def test_metrics_serialise(result):
    data = result.reference_metrics.to_dict()
    assert set(data) >= {"times", "energy", "momentum", "entropy", "max_entropy"}
    assert len(data["momentum"]) == len(data["times"]) - 1
