"""The lattice graph system: structure, rule semantics and the diffusion it produces."""

import networkx as nx
import numpy as np
import pytest

from imigraph.graph import (
    GraphSimulation,
    RandomHop,
    Rule,
    Space,
    SquareLattice,
    WalkerGraph,
    walkers_at_node,
)
from imigraph.graph.simulation import lattice_hop_diffusion


def test_lattice_matches_networkx_grid_graph():
    lattice = SquareLattice(size=4, spacing=0.5)
    assert isinstance(lattice, Space)
    expected = nx.grid_2d_graph(4, 4)
    actual = nx.Graph(map(tuple, lattice.edges()))
    relabel = {(i, j): lattice.node_index(i, j) for i, j in expected.nodes}
    assert nx.utils.graphs_equal(nx.relabel_nodes(expected, relabel), actual)
    np.testing.assert_allclose(lattice.coordinates[lattice.node_index(2, 3)], [1.0, 1.5])


def test_every_node_has_four_hop_options_with_self_loops_at_the_boundary():
    lattice = SquareLattice(size=5)
    graph = lattice.to_networkx()
    boundary = {node for node in graph if graph.degree(node) < 4 or graph.has_edge(node, node)}
    assert boundary == {n for n in range(25) if 0 in divmod(n, 5) or 4 in divmod(n, 5)}
    assert all(graph.has_edge(node, node) for node in boundary)
    assert lattice.neighbors.shape == (25, 4)
    corner, interior = lattice.node_index(0, 0), lattice.node_index(2, 2)
    assert np.sum(lattice.neighbors[corner] == corner) == 2
    assert np.sum(lattice.neighbors[interior] == interior) == 0
    assert set(lattice.neighbors[interior]) == {
        lattice.node_index(3, 2),
        lattice.node_index(1, 2),
        lattice.node_index(2, 3),
        lattice.node_index(2, 1),
    }


def test_box_gives_every_node_a_full_cell():
    lattice = SquareLattice(size=3, spacing=2.0)
    np.testing.assert_allclose(lattice.box.lower, [-1.0, -1.0])
    np.testing.assert_allclose(lattice.box.upper, [5.0, 5.0])
    assert lattice.box.contains(lattice.coordinates).all()
    np.testing.assert_allclose(lattice.coordinates[lattice.center_node], [2.0, 2.0])


def test_walker_graph_combines_space_and_walkers():
    lattice = SquareLattice(size=3)
    state = walkers_at_node(lattice, lattice.center_node, count=2)
    graph = state.to_networkx()
    assert graph.number_of_nodes() == 9 + 2
    kinds = nx.get_node_attributes(graph, "kind")
    assert sum(kind == "walker" for kind in kinds.values()) == 2
    assert graph.has_edge(state.walker_node(0), lattice.center_node)
    np.testing.assert_array_equal(state.occupancy()[lattice.center_node], 2)
    with pytest.raises(ValueError):
        WalkerGraph(lattice, np.array([9]))


def test_random_hop_moves_every_walker_along_an_existing_edge():
    lattice = SquareLattice(size=6)
    rule = RandomHop()
    assert isinstance(rule, Rule)
    rng = np.random.default_rng(5)
    state = WalkerGraph(lattice, rng.integers(0, lattice.num_nodes, size=500))
    graph = lattice.to_networkx()
    for _ in range(10):
        after = rule.apply(state, rng)
        assert after.space is lattice
        for before_node, after_node in zip(state.anchors, after.anchors, strict=True):
            assert graph.has_edge(int(before_node), int(after_node))
        state = after


def test_run_records_frames_and_is_reproducible():
    lattice = SquareLattice(size=7)
    simulation = GraphSimulation(lattice, step_time=0.1)
    initial = walkers_at_node(lattice, lattice.center_node, 3)
    first = simulation.run(initial, steps=10, rng=np.random.default_rng(1), record_every=5)
    second = simulation.run(initial, steps=10, rng=np.random.default_rng(1), record_every=5)
    assert first.anchors.shape == (3, 3)
    np.testing.assert_allclose(first.times, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(first.anchors, second.anchors)
    np.testing.assert_array_equal(first.anchors[0], initial.anchors)
    trajectory = first.project()
    assert trajectory.positions.shape == (3, 3, 2)
    assert trajectory.box == lattice.box
    assert first.state(1).num_walkers == 3


def test_free_hopping_has_msd_a_squared_per_step():
    spacing, step_time = 0.5, 0.2
    lattice = SquareLattice(size=101, spacing=spacing)
    simulation = GraphSimulation(lattice, step_time=step_time)
    initial = walkers_at_node(lattice, lattice.center_node, 4000)
    history = simulation.run(initial, steps=40, rng=np.random.default_rng(2))
    trajectory = history.project()
    diffusion = lattice_hop_diffusion(spacing, step_time)
    assert diffusion == pytest.approx(spacing**2 / (4 * step_time))
    # Far from the walls every step adds exactly spacing^2 to the mean squared displacement.
    np.testing.assert_allclose(
        trajectory.msd()[1:], 4 * diffusion * trajectory.times[1:], rtol=0.08
    )


def test_history_to_dict_is_json_like():
    lattice = SquareLattice(size=2)
    history = GraphSimulation(lattice).run(
        walkers_at_node(lattice, 0, 1), 2, np.random.default_rng(0)
    )
    data = history.to_dict()
    assert len(data["anchors"]) == 3 and len(data["space"]["coordinates"]) == 4
    assert sorted(map(tuple, data["space"]["edges"])) == [(0, 1), (0, 2), (1, 3), (2, 3)]
