"""The Brownian reference reproduces the analytic properties of diffusion in a box."""

import numpy as np
import pytest
from scipy import stats

from imigraph.reference import BrownianBox, reflect_into_box
from imigraph.trajectory import Box


def test_reflection_folds_any_excursion_back_into_the_box():
    box = Box((0.0, -1.0), (2.0, 1.0))
    points = np.array([[-0.5, 0.0], [2.5, 0.0], [0.5, 1.5], [0.5, -1.5], [4.5, 0.0], [1.0, 0.0]])
    expected = np.array([[0.5, 0.0], [1.5, 0.0], [0.5, 0.5], [0.5, -0.5], [0.5, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(reflect_into_box(points, box), expected)


def test_simulation_is_deterministic_for_a_seed_and_has_the_right_shape():
    system = BrownianBox(diffusion=0.5, dt=0.01, box=Box.square(10.0))
    x0 = system.all_at((5.0, 5.0), count=7)
    first = system.simulate(x0, steps=20, rng=np.random.default_rng(3), record_every=5)
    second = system.simulate(x0, steps=20, rng=np.random.default_rng(3), record_every=5)
    assert first.positions.shape == (5, 7, 2)
    np.testing.assert_allclose(first.times, [0.0, 0.05, 0.1, 0.15, 0.2])
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.positions[0], x0)


def test_particles_never_leave_the_box():
    box = Box.square(1.0)
    system = BrownianBox(diffusion=1.0, dt=0.1, box=box)  # steps comparable to the box size
    trajectory = system.simulate(system.all_at((0.5, 0.5), 200), 50, np.random.default_rng(1))
    assert box.contains(trajectory.positions).all()


def test_free_diffusion_msd_is_4_d_t():
    diffusion, dt = 0.25, 0.01
    system = BrownianBox(diffusion=diffusion, dt=dt, box=Box.square(100.0))
    n = 4000
    trajectory = system.simulate(system.all_at((50.0, 50.0), n), 200, np.random.default_rng(7))
    # Relative standard error of the MSD estimate is 1/sqrt(n) = 1.6%; 8% is five sigma.
    expected = 4 * diffusion * trajectory.times[1:]
    np.testing.assert_allclose(trajectory.msd()[1:], expected, rtol=0.08)
    # Each coordinate is Gaussian with variance 2 D t: a KS test should not reject it.
    final_x = trajectory.positions[-1, :, 0] - 50.0
    sigma = np.sqrt(2 * diffusion * trajectory.times[-1])
    assert stats.kstest(final_x, stats.norm(loc=0.0, scale=sigma).cdf).pvalue > 1e-3


def test_long_time_density_is_uniform():
    box = Box.square(1.0)
    system = BrownianBox(diffusion=1.0, dt=0.01, box=box)
    # Relaxation time of the box is L^2 / D = 1; run 5 of them.
    trajectory = system.simulate(system.all_at((0.5, 0.5), 4000), 500, np.random.default_rng(11))
    counts, _, _ = np.histogram2d(
        trajectory.positions[-1, :, 0], trajectory.positions[-1, :, 1], bins=4, range=[[0, 1]] * 2
    )
    assert stats.chisquare(counts.ravel()).pvalue > 1e-3


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        BrownianBox(diffusion=-1.0, dt=0.1, box=Box.square(1.0))
    system = BrownianBox(diffusion=1.0, dt=0.1, box=Box.square(1.0))
    with pytest.raises(ValueError):
        system.simulate(np.array([[2.0, 0.5]]), 1, np.random.default_rng(0))
