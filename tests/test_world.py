"""The walker system: one kind of walker, rounds, rule sets of chain and ball rules."""

import numpy as np
import pytest

from imigraph.graph import SquareLattice, WalkerGraph, WalkerSystem
from imigraph.graph.balls import place
from imigraph.graph.rules import (
    EXCHANGE,
    KICK,
    RANDOM_HOP,
    REFLECT,
    STREAM,
    SWAP,
    TURN,
    BallRule,
    ChainRule,
    RuleSequence,
    RuleSpec,
    rule_item,
)
from imigraph.graph.world import rules_from_data
from imigraph.report.live import playground

STREAM_JSON = {
    "name": "Stream",
    "notation": STREAM.notation,
    "description": "free flight",
}
REFLECT_JSON = {"name": "Reflect", "notation": REFLECT.notation}


def _system(size, *specs, **options):
    return WalkerSystem(
        SquareLattice(size), tuple(rule_item(s) for s in specs), **options
    )


def test_rule_list_accepts_specs_with_descriptions_and_sorts_them_by_family():
    rules = rules_from_data(
        [
            {"name": "Hop", "notation": RANDOM_HOP.notation, "description": "random"},
            STREAM_JSON,
        ]
    )
    assert isinstance(rules[0], ChainRule) and rules[0].spec.description == "random"
    assert isinstance(rules[1], BallRule) and rules[1].name == "Stream"
    assert rules[1].spec.to_dict()["description"] == "free flight"
    assert rules[1].spec.to_dict()["family"] == "ball"
    assert RuleSpec.from_dict(rules[0].spec.to_dict()).description == "random"
    with pytest.raises(ValueError, match="enable at least one rule"):
        rules_from_data([])
    with pytest.raises(ValueError, match="built-in rules are gone"):
        rules_from_data([{"builtin": "stream"}])
    system = WalkerSystem(SquareLattice(5), rules)
    assert [r.name for r in system.round_rules] == ["Stream"]
    assert [r.name for r in system.step_rules] == ["Hop"]
    assert not system.interacting


def test_resting_balls_under_chain_rules_reproduce_the_rule_sequence():
    lattice = SquareLattice(7)
    specs = (RANDOM_HOP, RuleSpec.parse("{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}"))
    system = WalkerSystem(lattice, tuple(ChainRule(s) for s in specs))
    anchors = np.full(5, lattice.center_node)
    history, waited = system.run(
        system.initial_state(anchors), 30, np.random.default_rng(1)
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
    assert history.velocities is None and not history.moving
    assert history.collisions().sum() == 0 and history.captions()[1].endswith(
        "walker(s) hopped"
    )


def test_a_step_takes_one_hop_per_round_and_the_budget_is_the_velocity():
    system = _system(9, STREAM)
    state = place(system.space, [[1, 1], [4, 4]], [[2, 1], [0, 0]])
    after, path, events, waited = system.step(state, np.random.default_rng(0))
    size = 9
    assert (after.anchors[0] // size, after.anchors[0] % size) == (3, 2)
    assert after.anchors[1] == state.anchors[1]  # at rest: no hops
    assert path.shape == (2, 4)  # three rounds for the fast ball
    assert path[1].tolist() == [state.anchors[1]] * 4
    assert events == [] and not waited.any()
    # Without Reflect a ball runs into the wall and stays there, hop by hop.
    stuck = place(system.space, [[8, 4]], [[3, 0]])
    after, path, _, _ = system.step(stuck, np.random.default_rng(0))
    assert after.anchors[0] == stuck.anchors[0] and after.velocities.tolist() == [
        [3, 0]
    ]


def test_a_ball_redirected_by_one_collision_meets_a_third_in_the_same_round():
    # Balls 0 and 1 fly at each other; Exchange turns ball 1 round, whose hop then meets
    # ball 2 (at rest) in the same round: a second pass of Exchange hands the momentum on.
    system = _system(9, REFLECT, EXCHANGE, SWAP, STREAM)
    state = place(system.space, [[1, 4], [3, 4], [4, 4]], [[1, 0], [-1, 0], [0, 0]])
    history, _ = system.run(state, 1, np.random.default_rng(0))
    assert history.collisions().tolist() == [2]
    assert history.velocities[1].tolist() == [[-1, 0], [0, 0], [1, 0]]
    assert np.all(history.overlaps() == 0)
    # A moving ball that hits a resting one hands over its velocity; the resting ball has no
    # hops left in this step and moves on in the next.
    state = place(system.space, [[1, 4], [3, 4], [4, 4]], [[1, 0], [0, 0], [0, 0]])
    history, _ = system.run(state, 3, np.random.default_rng(0))
    assert history.collisions().tolist() == [0, 1, 1]
    assert history.velocities[-1].tolist() == [[0, 0], [0, 0], [1, 0]]
    assert np.all(history.overlaps() == 0)


def test_balls_that_share_a_site_ignore_each_other_until_they_separate():
    system = _system(7, REFLECT, EXCHANGE, SWAP, STREAM)
    centre = system.space.center_node
    state = WalkerGraph(
        system.space, np.array([centre, centre]), np.array([[1, 0], [-1, 0]])
    )
    history, _ = system.run(state, 3, np.random.default_rng(0))
    assert history.collisions().sum() == 0
    assert history.anchors[1].tolist() == [centre + 7, centre - 7]


def test_kick_sets_the_velocity_to_the_drawn_hop():
    system = _system(7, KICK)
    state = WalkerGraph(system.space, np.array([24, 24]), np.zeros((2, 2), dtype=int))
    history, _ = system.run(state, 10, np.random.default_rng(2))
    assert history.velocities is not None
    for step in range(1, 11):
        for b in range(2):
            v = history.velocities[step, b]
            assert np.abs(v).sum() == 1
            moved = history.anchors[step, b] != history.anchors[step - 1, b]
            label = int(np.argmax(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) @ v))
            target = system.space.neighbors[history.anchors[step - 1, b], label]
            assert history.anchors[step, b] == target or not moved
    # A kicked ball under Stream as well flies for a step and is kicked again: 2 hops/step.
    both = _system(9, STREAM, KICK)
    history, _ = both.run(
        WalkerGraph(both.space, np.array([40]), np.array([[1, 0]])),
        5,
        np.random.default_rng(3),
    )
    assert history.paths is not None and history.paths.shape[2] == 3


def test_descriptions_list_every_rule_with_a_pattern_and_the_set():
    system = _system(8, REFLECT, TURN, STREAM, step_time=0.5)
    described = system.descriptions(0.5, 0.5)
    assert [d["name"] for d in described] == [
        "Reflect",
        "Turn",
        "Stream",
        "Reflect + Turn + Stream",
    ]
    assert all(d["pattern"].startswith("<svg") for d in described[:-1])
    assert "HPP" in described[1]["physics"]
    assert "round rules (Reflect, Turn, Stream)" in described[-1]["matches"]
    assert system.interacting


def test_playground_runs_resting_moving_and_colliding_balls_and_documents_the_rules():
    rest = playground({"steps": 10, "walkers": 3})
    assert rest["error"] is None
    assert rest["model"] == {
        "velocity": "rest",
        "max_speed": 2,
        "radius": 0,
        "overlaps": "parallel",
        "moving": False,
        "interacting": False,
    }
    assert rest["demo"]["velocities"] is None and rest["measured"]["collisions"] == 0
    assert "velocity lattice" in rest["docs"] and "<h3>" in rest["docs"]

    gas = playground(
        {
            "steps": 15,
            "walkers": 12,
            "placement": "left",
            "size": 6,
            "velocity": "unit",
            "rule": [
                REFLECT_JSON,
                {"name": "Turn", "notation": TURN.notation},
                STREAM_JSON,
            ],
        }
    )
    assert gas["error"] is None, gas["error"]
    assert [r["name"] for r in gas["rules"]] == [
        "Reflect",
        "Turn",
        "Stream",
        "Reflect + Turn + Stream",
    ]
    assert gas["rules"][2]["description"] == "free flight"
    assert len(gas["demo"]["velocities"]) == 16 and len(gas["demo"]["events"]) == 15
    assert gas["universe"]["velocities"] == gas["demo"]["velocities"][0]
    assert all(abs(v[0]) + abs(v[1]) == 1 for v in gas["universe"]["velocities"])
    assert gas["measured"]["energy_constant"] and gas["predicted"]["hops"] == 0
    assert gas["specs"][2]["name"] == "Stream" and gas["specs"][2]["family"] == "ball"
    assert gas["model"]["moving"] and gas["model"]["interacting"]

    mixed = playground(
        {
            "velocity": "random",
            "max_speed": 2,
            "walkers": 6,
            "placement": "random",
            "steps": 20,
            "rule": [{"notation": RANDOM_HOP.notation, "name": "Hop"}, STREAM_JSON],
        }
    )
    assert mixed["error"] is None and mixed["predicted"]["hops"] == 1
    assert mixed["measured"]["collisions"] == 0
    bad = playground({"velocity": "warp"})
    assert "velocity must be one of" in bad["error"]
    assert (
        "built-in rules are gone"
        in playground({"rule": [{"builtin": "stream"}]})["error"]
    )


HEAD_ON = {
    "rule": [
        REFLECT_JSON,
        {"name": "Exchange", "notation": EXCHANGE.notation},
        STREAM_JSON,
    ],
    "size": 5,
    "walls": "reflecting",
    "placement": "sites",
    "sites": [5, 15],
    "velocities": [[1, 0], [-1, 0]],
    "walkers": 2,
    "steps": 6,
    "seed": 1,
}


def test_two_balls_collide_in_step_two_and_the_captions_explain_it():
    result = playground(HEAD_ON)
    assert result["error"] is None, result["error"]
    assert result["universe"]["velocities"] == [[1, 0], [-1, 0]]
    assert result["model"]["velocity"] == "explicit"
    anchors = result["demo"]["anchors"]
    assert anchors[1] == [10, 10] or anchors[2] != anchors[1]
    captions = result["captions"]
    assert len(captions) == 7  # steps + 1
    assert captions[0].startswith("step 0: ball 0 at (1,0) with v = (1, 0)")
    collided = [f for f in range(1, 7) if "Exchange" in captions[f]]
    assert len(collided) >= 1, captions
    first = collided[0]
    assert "met along x" in captions[first]
    assert result["demo"]["velocities"][first] == [[-1, 0], [1, 0]]
    assert all("total momentum" in c for c in captions)
    assert result["measured"]["momentum_start"] == [0, 0]
    assert result["measured"]["collisions_by_rule"]["Exchange"] >= 1
    # Momentum changes only at wall reflections, never at collisions (two reflections in
    # one step may cancel).
    momenta = result["demo"]["momentum"]
    changes = [f for f in range(1, 7) if momenta[f] != momenta[f - 1]]
    assert all("Reflect" in captions[f] for f in changes)
    assert any("Reflect" in c for c in captions) and momenta == [[0, 0]] * 7
    assert result["measured"]["momentum_changes"] == len(changes)
    assert result["measured"]["energy_constant"]


def test_explicit_velocities_and_placements_are_validated():
    base = {"placement": "sites", "sites": [1, 9], "steps": 3}
    assert (
        "one (vx, vy) per walker"
        in playground({**base, "velocities": [[1, 0]]})["error"]
    )
    assert "at most 4" in playground({**base, "velocities": [[9, 0], [0, 0]]})["error"]
    ok = playground({**base, "velocities": [[0, 1], [0, -1]]})
    assert ok["error"] is None and ok["universe"]["velocities"] == [[0, 1], [0, -1]]
    head_on = playground(
        {"placement": "head-on", "size": 8, "max_speed": 3, "steps": 4}
    )
    assert head_on["error"] is None and head_on["universe"]["walkers"] == 2
    assert head_on["universe"]["velocities"] == [[3, 0], [-1, 0]]
    diagonal = playground({"placement": "diagonal", "size": 8, "steps": 4, "radius": 1})
    assert diagonal["error"] is None and diagonal["universe"]["radius"] == 1
    assert diagonal["universe"]["velocities"] == [[1, 0], [0, 1]]
    crowded = playground({"placement": "random", "radius": 2, "walkers": 50, "size": 6})
    assert "do not fit" in crowded["error"]


RIGHT = RuleSpec.parse(RANDOM_HOP.notation, name="Right", weights=(1, 0, 0, 0))


def _two_walkers_one_edge(strategy, steps=6):
    """Two walkers on one site of a periodic row, both always drawing the +x edge: their
    matches overlap in every step."""
    lattice = SquareLattice(5, periodic=True)
    system = WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps=strategy)
    start = np.array([lattice.center_node, lattice.center_node])
    rng = np.random.default_rng(0)
    history, waited = system.run(system.initial_state(start), steps, rng)
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
    _history, waited = system.run(system.initial_state(start), 4, rng)
    assert waited == 4  # one of the two waits in every step
    # Step by step: the walker that waited last time goes first, so they alternate.
    state = system.initial_state(start)
    ages = np.zeros(2, dtype=np.int64)
    waiting = []
    for _ in range(4):
        state, _path, _events, waited_mask = system.step(state, rng, ages)
        waiting.append(int(np.flatnonzero(waited_mask)[0]))
        ages = np.where(waited_mask, ages + 1, 0)
    assert waiting == [1, 0, 1, 0]
    system = WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps="random")
    _history, waited = system.run(system.initial_state(start), 20, rng)
    assert waited == 20
    with pytest.raises(ValueError, match="overlaps must be one of"):
        WalkerSystem(lattice, (ChainRule(RIGHT),), overlaps="first")


def test_contact_pairs_are_disjoint_under_every_strategy():
    lattice = SquareLattice(6)
    for strategy in ("parallel", "ordered", "random", "oldest"):
        system = WalkerSystem(
            lattice,
            tuple(rule_item(s) for s in (REFLECT, EXCHANGE, SWAP, STREAM)),
            overlaps=strategy,
        )
        rng = np.random.default_rng(2)
        anchors = rng.choice(lattice.num_nodes, size=12, replace=False)
        velocities = rng.integers(-2, 3, size=(12, 2))
        history, _ = system.run(WalkerGraph(lattice, anchors, velocities), 30, rng)
        energy = history.energy()
        assert np.all(energy == energy[0])
        assert np.all(history.overlaps() == 0)
        assert history.collisions().sum() > 0


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
