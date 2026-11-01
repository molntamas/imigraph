"""The unified walker system: both walker models, the exclusion update and HPP collisions."""

import numpy as np
import pytest

from imigraph.graph import SquareLattice, WalkerGraph
from imigraph.graph.gas import LatticeGas, LatticeGasState
from imigraph.graph.rules import RANDOM_HOP, ChainRule, RuleSequence, RuleSpec
from imigraph.graph.world import (
    BuiltinRule,
    WalkerSystem,
    initial_headings,
    resolve_exclusion,
    rules_from_data,
)
from imigraph.report.live import playground

STREAM = {"builtin": "stream", "description": "free flight"}
REFLECT = {"builtin": "reflect"}


def _exclusion_holds(state: LatticeGasState) -> bool:
    channels = state.sites * 4 + state.directions
    return np.unique(channels).size == channels.size


def test_rule_list_accepts_specs_and_builtins_with_descriptions():
    rules = rules_from_data(
        [
            {"name": "Hop", "notation": RANDOM_HOP.notation, "description": "random"},
            STREAM,
        ]
    )
    assert isinstance(rules[0], ChainRule) and rules[0].spec.description == "random"
    assert isinstance(rules[1], BuiltinRule) and rules[1].name == "Stream"
    assert rules[1].to_dict()["description"] == "free flight"
    assert RuleSpec.from_dict(rules[0].spec.to_dict()).description == "random"
    with pytest.raises(ValueError, match="enable at least one rule"):
        rules_from_data([])
    with pytest.raises(ValueError, match="builtin must be one of"):
        rules_from_data([{"builtin": "explode"}])


def test_edge_walkers_reproduce_the_rule_sequence():
    lattice = SquareLattice(7)
    specs = (RANDOM_HOP, RuleSpec.parse("{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}"))
    system = WalkerSystem(lattice, tuple(ChainRule(s) for s in specs))
    anchors = np.full(5, lattice.center_node)
    history, blocked = system.run(
        system.initial_state(anchors, "random", np.random.default_rng(3)),
        30,
        np.random.default_rng(1),
    )
    reference = RuleSequence(tuple(ChainRule(s) for s in specs))
    state = WalkerGraph(lattice, anchors)
    rng = np.random.default_rng(1)
    assert history.paths is not None
    for step in range(1, 31):
        state, path = reference.apply_with_paths(state, rng)
        np.testing.assert_array_equal(history.anchors[step], state.anchors)
        np.testing.assert_array_equal(history.paths[step - 1], path)
    assert blocked == 0 and history.directions is None and history.collisions is None


def test_edge_walkers_refuse_collisions_and_builtin_rules():
    lattice = SquareLattice(5)
    with pytest.raises(ValueError, match="hyperedge walkers"):
        WalkerSystem(lattice, (ChainRule(RANDOM_HOP),), collisions="hpp")
    with pytest.raises(ValueError, match="Stream works on hyperedge walkers"):
        WalkerSystem(lattice, (ChainRule(RANDOM_HOP), BuiltinRule("stream")))
    with pytest.raises(ValueError, match="walker model"):
        WalkerSystem(lattice, (ChainRule(RANDOM_HOP),), walker_model="string")


def test_initial_headings_respect_the_exclusion():
    rng = np.random.default_rng(0)
    sites = np.array([3, 3, 3, 3, 7])
    directions = initial_headings(sites, "random", rng)
    assert sorted(directions[:4]) == [0, 1, 2, 3]
    assert initial_headings(np.array([1, 2]), "+y", rng).tolist() == [2, 2]
    with pytest.raises(ValueError, match="at most 4 per site"):
        initial_headings(np.zeros(5, dtype=int), "random", rng)
    with pytest.raises(ValueError, match="same heading"):
        initial_headings(np.array([1, 1]), "-x", rng)
    with pytest.raises(ValueError, match="heading must be one of"):
        initial_headings(sites, "north", rng)


def test_exclusion_update_blocks_the_loser_and_keeps_one_walker_per_channel():
    lattice = SquareLattice(5)
    state = LatticeGasState(lattice, np.array([0, 1, 2]), np.array([0, 1, 1]))
    # Walkers 0 and 1 both want channel (5, 0); walker 2 wants (1, 1), taken by walker 1.
    new, blocked = resolve_exclusion(
        state, np.array([5, 5, 1]), np.array([0, 0, 1]), np.random.default_rng(0)
    )
    assert blocked.tolist() == [False, True, True] or blocked.tolist() == [
        True,
        False,
        True,
    ]
    assert _exclusion_holds(new)
    assert new.sites[~blocked].tolist() == [5] and new.sites[2] == 2
    # A train: walker 1 moves into the channel walker 0 leaves, in the same step.
    train = LatticeGasState(lattice, np.array([0, 5]), np.array([0, 0]))
    moved, blocked = resolve_exclusion(
        train, np.array([5, 10]), np.array([0, 0]), np.random.default_rng(0)
    )
    assert not blocked.any() and moved.sites.tolist() == [5, 10]
    # A jam: walker 2 stays, walker 1 wants its channel, walker 0 wants walker 1's.
    jam = LatticeGasState(lattice, np.array([0, 5, 10]), np.array([0, 0, 0]))
    kept, blocked = resolve_exclusion(
        jam, np.array([5, 10, 10]), np.array([0, 0, 0]), np.random.default_rng(0)
    )
    assert blocked.tolist() == [True, True, False]
    assert kept.sites.tolist() == [0, 5, 10]
    # A walker may always keep its channel; no clash means no randomness used.
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    _same, blocked = resolve_exclusion(state, state.sites, state.directions, rng)
    assert not blocked.any() and rng.bit_generator.state == before


@pytest.mark.parametrize("collisions", ["none", "hpp"])
def test_hyperedge_chain_rules_keep_the_exclusion_at_high_density(collisions):
    lattice = SquareLattice(4)
    system = WalkerSystem(lattice, (ChainRule(RANDOM_HOP),), "hyperedge", collisions)
    rng = np.random.default_rng(5)
    anchors = np.repeat(np.arange(16), 3)  # 48 walkers on 16 sites: crowded
    initial = system.initial_state(anchors, "random", rng)
    assert isinstance(initial, LatticeGasState)
    history, blocked = system.run(initial, 40, rng)
    assert blocked > 0
    assert history.directions is not None and history.directions.shape == (41, 48)
    for frame in range(41):
        state = LatticeGasState(
            lattice, history.anchors[frame], history.directions[frame]
        )
        assert _exclusion_holds(state)
    assert history.collisions is not None and len(history.collisions) == 40
    if collisions == "hpp":
        assert sum(len(s) for s in history.collisions) > 0


def test_the_heading_follows_the_last_hop():
    lattice = SquareLattice(9, periodic=True)
    right = RuleSpec.parse(RANDOM_HOP.notation, name="Right", weights=(1, 0, 0, 0))
    system = WalkerSystem(lattice, (ChainRule(right),), "hyperedge")
    state = system.initial_state(
        np.array([lattice.center_node]), "-y", np.random.default_rng(0)
    )
    assert isinstance(state, LatticeGasState) and state.directions.tolist() == [3]
    new, _path, _collided, blocked = system.step(state, np.random.default_rng(0))
    assert isinstance(new, LatticeGasState)
    assert new.directions.tolist() == [0] and blocked == 0
    assert new.sites[0] == lattice.neighbors[lattice.center_node, 0]


def test_stream_reflect_with_hpp_collisions_is_the_lattice_gas():
    lattice = SquareLattice(8, spacing=0.5)
    gas = LatticeGas(lattice, step_time=0.5)
    initial = gas.initial_half_box(20, np.random.default_rng(2))
    system = WalkerSystem(
        lattice,
        (BuiltinRule("reflect"), BuiltinRule("stream")),
        "hyperedge",
        "hpp",
        0.5,
    )
    history, blocked = system.run(initial, 60, np.random.default_rng(7))
    assert blocked == 0
    assert history.directions is not None and history.collisions is not None
    state = initial
    rng = np.random.default_rng(7)
    for step in range(1, 61):
        state, collided = gas.step(state, rng)
        np.testing.assert_array_equal(history.anchors[step], state.sites)
        np.testing.assert_array_equal(history.directions[step], state.directions)
        np.testing.assert_array_equal(
            np.sort(history.collisions[step - 1]), np.sort(collided)
        )
    assert sum(len(s) for s in history.collisions) > 0
    described = system.descriptions(0.5, 0.5)
    assert [d["name"] for d in described] == [
        "Collide",
        "Reflect",
        "Stream",
        "Reflect + Stream",
    ]
    assert all(d["pattern"].startswith("<svg") for d in described[:-1])


def test_playground_runs_every_model_and_documents_collisions():
    edge = playground({"steps": 10, "walkers": 3})
    assert edge["error"] is None
    assert edge["model"] == {
        "walker_model": "edge",
        "collisions": "none",
        "heading": None,
    }
    assert edge["demo"]["directions"] is None and edge["measured"]["collisions"] is None
    assert "exclusion principle" in edge["docs"] and "<h3>" in edge["docs"]

    gas = playground(
        {
            "steps": 15,
            "walkers": 12,
            "placement": "left",
            "size": 6,
            "walker_model": "hyperedge",
            "collisions": "hpp",
            "rule": [REFLECT, STREAM],
        }
    )
    assert gas["error"] is None, gas["error"]
    assert (
        gas["rules"][0]["name"] == "Collide"
        and gas["rules"][-1]["name"] == "Reflect + Stream"
    )
    assert gas["rules"][2]["description"] == "free flight"
    assert len(gas["demo"]["directions"]) == 16 and len(gas["demo"]["collisions"]) == 15
    assert gas["universe"]["directions"] == gas["demo"]["directions"][0]
    assert gas["measured"]["blocked_rate"] >= 0 and gas["predicted"]["hops"] == 0
    assert gas["specs"][1] == {
        "name": "Stream",
        "builtin": "stream",
        "notation": "{w,s,s'}_k, {s',s''}_k -> {w,s',s''}_k",
        "description": "free flight",
    }

    bad = playground({"collisions": "hpp"})
    assert "hyperedge" in bad["error"]
    crowded = playground(
        {"walker_model": "hyperedge", "walkers": 5, "placement": "center"}
    )
    assert "at most 4 per site" in crowded["error"]
    mixed = playground(
        {
            "walker_model": "hyperedge",
            "walkers": 6,
            "placement": "random",
            "steps": 20,
            "rule": [{"notation": RANDOM_HOP.notation, "name": "Hop"}, STREAM],
        }
    )
    assert mixed["error"] is None and mixed["predicted"]["hops"] == 1
    assert mixed["measured"]["blocked_rate"] >= 0
