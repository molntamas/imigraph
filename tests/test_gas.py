"""The hard-disk gas conserves energy exactly and relaxes to kinetic theory."""

import numpy as np
import pytest
from scipy import stats

from imigraph.experiments.gas import GasConfig, run_gas
from imigraph.reference import HardDiskGas
from imigraph.trajectory import Box


@pytest.fixture(scope="module")
def result():
    return run_gas(GasConfig())


def test_head_on_collision_exchanges_velocities():
    gas = HardDiskGas(Box.square(1.0), radius=0.05, dt=0.01)
    x = np.array([[0.4, 0.5], [0.6, 0.5]])
    v = np.array([[1.0, 0.0], [-1.0, 0.0]])
    run = gas.simulate(x, v, steps=20)
    # They meet at x = 0.5 after 0.05 time units, swap velocities and move apart again.
    np.testing.assert_allclose(run.velocities[-1], [[-1.0, 0.0], [1.0, 0.0]])
    assert run.collisions.sum() == 1
    assert run.trajectory.positions[-1, 0, 0] < run.trajectory.positions[-1, 1, 0]
    np.testing.assert_allclose(run.kinetic_energy(), run.kinetic_energy()[0])
    np.testing.assert_allclose(run.momentum(), 0.0, atol=1e-15)


def test_oblique_collision_conserves_energy_and_momentum():
    gas = HardDiskGas(Box.square(2.0), radius=0.05, dt=0.005)
    x = np.array([[0.8, 1.0], [1.2, 1.03]])
    v = np.array([[1.0, 0.2], [-0.5, 0.1]])
    run = gas.simulate(x, v, steps=100)
    assert run.collisions.sum() == 1
    np.testing.assert_allclose(
        run.kinetic_energy(), run.kinetic_energy()[0], rtol=1e-12
    )
    np.testing.assert_allclose(run.momentum() - run.momentum()[0], 0.0, atol=1e-12)
    assert not np.allclose(
        run.velocities[-1], v
    )  # the collision did change the velocities


def test_wall_reflection_keeps_speed_and_flips_one_component():
    gas = HardDiskGas(Box.square(1.0), radius=0.05, dt=0.01)
    run = gas.simulate([[0.9, 0.5]], [[2.0, 0.3]], steps=10)
    np.testing.assert_allclose(run.velocities[-1], [[-2.0, 0.3]])
    assert run.trajectory.positions[-1, 0, 0] <= 0.95
    assert Box.square(1.0).contains(run.trajectory.positions).all()


def test_periodic_walls_conserve_momentum_exactly():
    config = GasConfig(
        particles=40, steps=400, record_every=20, boundary="periodic", seed=4
    )
    result = run_gas(config)
    momentum = result.run.momentum()
    np.testing.assert_allclose(momentum - momentum[0], 0.0, atol=1e-12)
    assert result.report["momentum"].passed
    assert result.run.collisions.sum() > 0


def test_energy_is_conserved_to_rounding(result):
    energy = result.run.kinetic_energy()
    assert energy[0] == pytest.approx(
        0.5 * result.config.particles * result.config.speed**2
    )
    np.testing.assert_allclose(energy, energy[0], rtol=1e-12)
    assert result.report["energy"].passed


def test_disks_stay_inside_the_box_and_collide(result):
    assert result.config.box.contains(result.run.trajectory.positions).all()
    assert result.run.collisions.sum() > 10 * result.config.particles


def test_speeds_relax_to_maxwell_boltzmann(result):
    run = result.run
    rayleigh = stats.rayleigh(scale=run.thermal_sigma)
    assert (
        stats.kstest(run.speeds(0), rayleigh.cdf).pvalue < 1e-6
    )  # all equal at the start
    assert stats.kstest(run.speeds(-1), rayleigh.cdf).pvalue > 1e-3
    assert result.report["maxwell"].passed


def test_free_expansion_raises_the_entropy(result):
    entropy = result.entropy
    bins = result.config.entropy_bins
    # One half of the box is empty: at most ln(bins^2 / 2) at the start.
    assert entropy[0] <= np.log(bins**2 / 2) + 1e-9
    # Spread over the whole box at the end, up to the finite-sample bias (bins^2 - 1) / (2 n).
    bias = (bins**2 - 1) / (2 * result.config.particles)
    assert entropy[-1] > result.max_entropy - bias - 0.2
    assert result.report["entropy"].passed
    assert result.passed, result.report.to_markdown()


def test_default_run_stays_within_its_time_budget():
    import time

    start = time.perf_counter()
    run_gas(GasConfig())
    assert time.perf_counter() - start < 8.0  # measured 1.8 s in the dev container


def test_invalid_setups_are_rejected():
    with pytest.raises(ValueError):
        HardDiskGas(Box.square(0.01), radius=0.05)
    gas = HardDiskGas(Box.square(1.0), radius=0.1)
    with pytest.raises(ValueError):
        gas.grid_in_region(1000, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
