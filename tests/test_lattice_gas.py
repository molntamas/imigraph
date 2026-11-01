"""The lattice gas conserves what it claims to conserve and its rules preserve the exclusion."""

import time

import numpy as np
import pytest

from imigraph.graph import SquareLattice
from imigraph.graph.gas import DIRECTIONS, OPPOSITE, LatticeGas, LatticeGasState
from imigraph.metrics import occupancy_entropy


def _gas(size=12, periodic=False, step_time=0.5):
    lattice = SquareLattice(size, spacing=0.5, periodic=periodic)
    return LatticeGas(lattice, step_time=step_time)


def test_periodic_lattice_wraps_and_has_no_self_loops():
    lattice = SquareLattice(5, periodic=True)
    nodes = np.arange(lattice.num_nodes)
    assert not np.any(lattice.neighbors == nodes[:, None])
    assert lattice.neighbors[lattice.node_index(4, 2), 0] == lattice.node_index(0, 2)
    assert lattice.neighbors[lattice.node_index(1, 0), 3] == lattice.node_index(1, 4)
    assert lattice.edges().shape == (2 * 25, 2)  # every node has degree 4 without loops
    with pytest.raises(ValueError):
        SquareLattice(2, periodic=True)


def test_state_enforces_the_exclusion_principle_and_knows_its_heading():
    lattice = SquareLattice(4)
    state = LatticeGasState(lattice, np.array([5, 5]), np.array([0, 2]))
    np.testing.assert_array_equal(
        state.headings(), [lattice.neighbors[5, 0], lattice.neighbors[5, 2]]
    )
    np.testing.assert_allclose(state.velocities(2.0), [[2, 0], [0, 2]])
    assert state.as_walker_graph().num_walkers == 2
    with pytest.raises(ValueError):
        LatticeGasState(lattice, np.array([5, 5]), np.array([0, 0]))
    with pytest.raises(ValueError):
        LatticeGasState(lattice, np.array([5]), np.array([4]))


def test_collision_rotates_a_head_on_pair_and_conserves_momentum_at_the_site():
    gas = _gas(size=5)
    centre = gas.space.center_node
    state = LatticeGasState(gas.space, np.array([centre, centre]), np.array([0, 1]))
    after, collided = gas.collide(state, np.random.default_rng(0))
    assert collided.size == 1
    assert set(after.directions.tolist()) == {2, 3}
    np.testing.assert_allclose(DIRECTIONS[after.directions].sum(axis=0), [0, 0])
    # A blocked perpendicular channel, or a wall channel, prevents the collision.
    blocked = LatticeGasState(gas.space, np.array([centre] * 3), np.array([0, 1, 2]))
    assert gas.collide(blocked, np.random.default_rng(0))[1].size == 0
    edge = gas.space.node_index(2, 4)  # +y neighbour is a self-loop
    at_wall = LatticeGasState(gas.space, np.array([edge, edge]), np.array([0, 1]))
    assert gas.collide(at_wall, np.random.default_rng(0))[1].size == 0


def test_reflect_and_stream_follow_labelled_edges():
    gas = _gas(size=4)
    corner = gas.space.node_index(3, 3)
    state = LatticeGasState(gas.space, np.array([corner, 0]), np.array([0, 2]))
    reflected = gas.reflect(state)
    np.testing.assert_array_equal(reflected.directions, [OPPOSITE[0], 2])
    streamed = gas.stream(reflected)
    np.testing.assert_array_equal(
        streamed.sites, [gas.space.node_index(2, 3), gas.space.node_index(0, 1)]
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_run_conserves_energy_exactly_and_momentum_as_promised(periodic):
    gas = _gas(size=16, periodic=periodic)
    rng = np.random.default_rng(3)
    initial = gas.initial_half_box(100, rng)
    run = gas.simulate(initial, steps=200, rng=rng, record_every=10)
    energy = run.kinetic_energy()
    np.testing.assert_array_equal(energy, energy[0])
    assert energy[0] == pytest.approx(0.5 * 100 * gas.speed**2)
    momentum = run.momentum()
    if periodic:
        np.testing.assert_array_equal(
            momentum, np.broadcast_to(momentum[0], momentum.shape)
        )
    else:
        bound = gas.mass * gas.speed * 100
        assert np.all(np.abs(momentum) <= bound)
    assert run.collisions.sum() > 0
    assert run.trajectory.positions.shape == (21, 100, 2)
    assert run.trajectory.box.contains(run.trajectory.positions.reshape(-1, 2)).all()


def test_every_step_keeps_the_exclusion_and_moves_along_edges():
    gas = _gas(size=10)
    rng = np.random.default_rng(1)
    state = gas.initial_half_box(150, rng)
    for _ in range(50):
        after, _ = gas.step(state, rng)
        channels = after.sites * 4 + after.directions
        assert np.unique(channels).size == channels.size
        moved = after.sites != state.sites
        heads = gas.space.neighbors[state.sites]
        assert np.all(np.any(heads == after.sites[:, None], axis=1) | ~moved)
        state = after


def test_free_expansion_raises_the_entropy_towards_uniform():
    gas = _gas(size=24)
    rng = np.random.default_rng(0)
    run = gas.simulate(
        gas.initial_half_box(200, rng), steps=600, rng=rng, record_every=50
    )
    entropy = occupancy_entropy(run.trajectory, bins=6)
    assert entropy[-1] > entropy[0] + 0.3
    assert entropy[-1] > np.log(36) - 0.3


def test_lattice_gas_budget():
    gas = LatticeGas(SquareLattice(40, spacing=0.025), step_time=0.025)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    gas.simulate(gas.initial_half_box(120, rng), steps=480, rng=rng, record_every=4)
    assert time.perf_counter() - start < 2.0
