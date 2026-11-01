"""The lattice gas (Reflect, Turn, Stream on unit velocities) conserves what it claims."""

import time

import numpy as np
import pytest

from imigraph.graph import LatticeGas, SquareLattice, WalkerGraph
from imigraph.graph.velocity import DIRECTIONS
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


def test_state_holds_unit_velocities_and_the_rules_are_the_hpp_set():
    gas = _gas(size=4)
    state = gas.state(np.array([5, 6]), np.array([0, 2]))
    np.testing.assert_array_equal(state.velocities, [[1, 0], [0, 1]])
    assert state.moving and state.num_walkers == 2
    assert [r["name"] for r in gas.describe_rules()] == [
        "Reflect",
        "Turn",
        "Swap",
        "Stream",
    ]
    assert gas.speed == pytest.approx(1.0)
    with pytest.raises(ValueError):
        WalkerGraph(gas.space, np.array([5]), np.array([[1, 0, 0]]))


def test_a_head_on_pair_turns_at_the_contact_and_conserves_momentum():
    gas = _gas(size=6)
    state = gas.state(
        np.array([gas.space.node_index(1, 3), gas.space.node_index(4, 3)]),
        np.array([0, 1]),
    )
    history, _ = gas.system.run(state, 4, np.random.default_rng(0))
    assert history.collisions().tolist() == [0, 1, 0, 0]
    assert history.collision_rules() == ["Turn"]
    after = history.velocities[2]
    assert sorted(after.tolist()) == [[0, -1], [0, 1]]
    np.testing.assert_array_equal(history.momentum_units()[:3], 0)
    assert np.all(history.overlaps() == 0)
    # A pair meeting at right angles swaps velocities at the contact: each ball continues
    # the other's motion from its own site (a deflection; HPP's points would pass through).
    corner = gas.state(
        np.array([gas.space.node_index(1, 3), gas.space.node_index(3, 1)]),
        np.array([0, 2]),
    )
    history, _ = gas.system.run(corner, 3, np.random.default_rng(0))
    assert history.collision_rules() == ["Swap"]
    assert history.anchors[2].tolist() == [
        gas.space.node_index(2, 4),
        gas.space.node_index(4, 2),
    ]
    assert history.velocities[2].tolist() == [[0, 1], [1, 0]]
    assert np.all(history.overlaps() == 0)


def test_reflect_and_stream_follow_labelled_edges():
    gas = _gas(size=4)
    corner = gas.space.node_index(3, 3)
    state = gas.state(np.array([corner, 0]), np.array([0, 2]))
    after, path, events, _ = gas.system.step(state, np.random.default_rng(0))
    np.testing.assert_array_equal(after.velocities, [DIRECTIONS[1], DIRECTIONS[2]])
    np.testing.assert_array_equal(
        after.anchors, [gas.space.node_index(2, 3), gas.space.node_index(0, 1)]
    )
    assert [e["rule"] for e in events] == ["Reflect"] and path.shape == (2, 2)


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


def test_every_step_moves_along_edges_and_keeps_the_balls_apart():
    gas = _gas(size=10)
    rng = np.random.default_rng(1)
    state = gas.initial_half_box(40, rng)
    for _ in range(50):
        after, _path, _events, _ = gas.system.step(state, rng)
        assert np.unique(after.anchors).size == after.num_walkers
        moved = after.anchors != state.anchors
        heads = gas.space.neighbors[state.anchors]
        assert np.all(np.any(heads == after.anchors[:, None], axis=1) | ~moved)
        assert np.all(np.abs(after.velocities).sum(axis=1) == 1)
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
    assert time.perf_counter() - start < 3.0  # measured 1.0 s in the dev container
