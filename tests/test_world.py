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
    history, blocked, waited = system.run(
        system.initial_state(anchors, "random", np.random.default_rng(3)),
        30,
        np.random.default_rng(1),
    )
    assert waited == 0
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
    history, blocked, _waited = system.run(initial, 40, rng)
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
    new, _path, _collided, blocked, _waited = system.step(
        state, np.random.default_rng(0)
    )
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
    history, blocked, _waited = system.run(initial, 60, np.random.default_rng(7))
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
        "overlaps": "parallel",
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


def test_momentum_and_energy_are_conserved_by_hpp_collisions():
    from imigraph.graph.world import momentum

    lattice = SquareLattice(8, periodic=True)
    system = WalkerSystem(lattice, (BuiltinRule("stream"),), "hyperedge", "hpp")
    rng = np.random.default_rng(11)
    anchors = rng.choice(lattice.num_nodes, size=30, replace=False)
    history, blocked, _waited = system.run(
        system.initial_state(anchors, "random", rng), 80, rng
    )
    assert blocked == 0 and history.collisions is not None
    assert sum(len(c) for c in history.collisions) > 0
    total = momentum(history)
    assert total is not None
    np.testing.assert_array_equal(total, np.repeat(total[:1], 81, axis=0))
    # Energy: every walker keeps its speed, so the kinetic energy is n m c^2 / 2 at all times.
    speeds = np.linalg.norm(
        np.diff(lattice.coordinates[history.anchors], axis=0), axis=2
    )
    wrapped = speeds > 1.5  # periodic wrap-around jumps
    assert np.all(np.isclose(speeds[~wrapped], 1.0))


COLLISION_DEMO = {
    "rule": [REFLECT, STREAM],
    "walker_model": "hyperedge",
    "collisions": "hpp",
    "size": 4,
    "walls": "reflecting",
    "placement": "sites",
    "sites": [1, 9],
    "headings": ["+x", "-x"],
    "walkers": 2,
    "steps": 6,
    "seed": 1,
}


def test_two_hyperedge_walkers_collide_in_step_two_and_the_captions_explain_it():
    result = playground(COLLISION_DEMO)
    assert result["error"] is None, result["error"]
    assert result["universe"]["headings"] == ["+x", "-x"]
    assert result["demo"]["anchors"][1] == [5, 5]  # step 1: both at (1,1)
    assert result["demo"]["collisions"][1] == [5]  # step 2: Collide at (1,1)
    assert sorted(result["demo"]["anchors"][2]) == [4, 6]
    captions = result["captions"]
    assert len(captions) == 7  # steps + 1
    assert captions[0].startswith("step 0: walker 0 at (0,1) heading +x")
    assert "Collide at (1,1)" in captions[2] and "perpendicular" in captions[2]
    assert "Reflect" in captions[3] and "Stream" in captions[1]
    assert all("total momentum" in c for c in captions)
    assert result["measured"]["momentum_start"] == [0.0, 0.0]
    assert result["demo"]["momentum"][2] == [0.0, 0.0]
    # Momentum changes only at wall reflections (steps 3 and 4 here), never at collisions.
    changes = [
        f
        for f in range(1, 7)
        if result["demo"]["momentum"][f] != result["demo"]["momentum"][f - 1]
    ]
    assert changes == [3, 4, 6], changes
    for f in range(1, 7):
        assert ("Reflect" in captions[f]) == (f in changes), captions[f]
        assert ("Collide" in captions[f]) == (f in (2, 5)), captions[f]
    assert result["measured"]["momentum_changes"] == 3


def test_explicit_headings_are_validated():
    base = {
        "walker_model": "hyperedge",
        "placement": "sites",
        "sites": [1, 9],
        "steps": 3,
    }
    assert "one heading per walker" in playground({**base, "headings": ["+x"]})["error"]
    assert "+x, -x, +y or -y" in playground({**base, "headings": ["+x", "up"]})["error"]
    assert (
        "exclusion"
        in playground({**base, "sites": [1, 1], "headings": ["+x", "+x"]})["error"]
    )
    ok = playground({**base, "headings": ["+y", "-y"]})
    assert ok["error"] is None and ok["universe"]["headings"] == ["+y", "-y"]


RIGHT = RuleSpec.parse(RANDOM_HOP.notation, name="Right", weights=(1, 0, 0, 0))


def _two_walkers_one_edge(strategy, steps=6):
    """Two walkers on one site of a periodic row, both always drawing the +x edge: their
    matches overlap in every step."""
    lattice = SquareLattice(5, periodic=True)
    system = WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps=strategy)
    start = np.array([lattice.center_node, lattice.center_node])
    rng = np.random.default_rng(0)
    history, _blocked, waited = system.run(
        system.initial_state(start, "random", rng), steps, rng
    )
    return history, waited


def test_parallel_applies_overlapping_matches_and_ordered_skips_them():
    history, waited = _two_walkers_one_edge("parallel")
    assert waited == 0
    assert np.all(
        history.anchors[1:, 0] == history.anchors[1:, 1]
    )  # both moved together
    history, waited = _two_walkers_one_edge("ordered")
    # Walker 0 always goes first; walker 1 waits until the shared edge is free again, i.e.
    # once walker 0 has moved on it is a different edge and both move: 1 wait in total.
    assert waited == 1
    assert history.anchors[1, 0] != history.anchors[0, 0]
    assert history.anchors[1, 1] == history.anchors[0, 1]
    assert history.paths is not None
    assert (
        history.paths[0, 1].tolist() == [history.anchors[0, 1]] * 2
    )  # a waiting match


def test_oldest_alternates_and_random_picks_one_per_shared_edge():
    lattice = SquareLattice(3)
    # Reflecting 3x3 world: two walkers that always draw +x pile up at the right wall, where
    # the shared self-loop edge is the overlap in every step.
    system = WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps="oldest")
    start = np.array([lattice.node_index(2, 1), lattice.node_index(2, 1)])
    rng = np.random.default_rng(0)
    _history, _blocked, waited = system.run(
        system.initial_state(start, "random", rng), 4, rng
    )
    assert waited == 4  # one of the two waits in every step
    # Step by step: the walker that waited last time goes first, so they alternate.
    state = system.initial_state(start, "random", rng)
    ages = np.zeros(2, dtype=np.int64)
    waiting = []
    for _ in range(4):
        state, _path, _collided, _blocked, waited_mask = system.step(state, rng, ages)
        waiting.append(int(np.flatnonzero(waited_mask)[0]))
        ages = np.where(waited_mask, ages + 1, 0)
    assert waiting == [1, 0, 1, 0]
    system = WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps="random")
    _history, _blocked, waited = system.run(
        system.initial_state(start, "random", rng), 20, rng
    )
    assert waited == 20
    with pytest.raises(ValueError, match="overlaps must be one of"):
        WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps="first")


def test_overlap_strategies_keep_the_exclusion_on_hyperedge_walkers():
    lattice = SquareLattice(4)
    for strategy in ("ordered", "random", "oldest"):
        system = WalkerSystem(
            lattice, (ChainRule(RANDOM_HOP),), "hyperedge", "hpp", overlaps=strategy
        )
        rng = np.random.default_rng(2)
        anchors = np.repeat(np.arange(16), 2)
        history, _blocked, waited = system.run(
            system.initial_state(anchors, "random", rng), 30, rng
        )
        assert waited > 0 and history.directions is not None
        for frame in range(31):
            assert _exclusion_holds(
                LatticeGasState(
                    lattice, history.anchors[frame], history.directions[frame]
                )
            )


def test_playground_reports_the_overlap_strategy():
    result = playground(
        {"overlaps": "ordered", "walkers": 8, "placement": "center", "steps": 20}
    )
    assert result["error"] is None
    assert result["model"]["overlaps"] == "ordered"
    assert result["universe"]["overlaps"] == "ordered"
    assert result["measured"]["waiting_rate"] > 0
    assert "ordered" in result["rule"]["description"]
    assert "Overlapping matches" in result["docs"]
    assert "overlaps must be one of" in playground({"overlaps": "last"})["error"]
    assert playground({})["measured"]["waiting_rate"] == 0
