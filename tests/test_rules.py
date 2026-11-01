"""Rules as data: parsing, validation messages, the physics the chain rule family produces,
and the ball rules of the library as graph rewrites."""

import json

import numpy as np
import pytest

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.graph import GraphSimulation, RandomHop, SquareLattice, walkers_at_node
from imigraph.graph.rules import (
    EXCHANGE,
    LIBRARY,
    RANDOM_HOP,
    REFLECT,
    STREAM,
    SWAP,
    TURN,
    BallRule,
    ChainRule,
    Rule,
    RuleSpec,
    rule_from_spec,
    rule_item,
)
from imigraph.graph.simulation import lattice_hop_diffusion
from imigraph.graph.state import WalkerGraph


def test_notation_round_trip_and_json():
    spec = RuleSpec.parse(
        "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}", name="DoubleHop"
    )
    assert spec.notation == "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}"
    assert spec.chain() == ("w", ["s", "t", "u"], 2) and spec.hops == 2
    assert spec.family == "chain"
    data = json.loads(json.dumps(spec.to_dict()))
    assert RuleSpec.from_dict(data) == spec
    assert data["types"] == {"w": "walker", "s": "site", "t": "site", "u": "site"}
    assert data["family"] == "chain"
    assert RuleSpec.from_dict({"name": "x", "notation": spec.notation}).hops == 2
    assert RandomHop().spec == RANDOM_HOP


def test_annotated_notation_round_trips_and_classifies_as_a_ball_rule():
    for spec in LIBRARY:
        again = RuleSpec.parse(spec.notation, name=spec.name)
        assert again.notation == spec.notation and again.family == "ball"
        assert RuleSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
        assert spec.hops == 0 and not spec.drift(1, 1).any()
    plan = EXCHANGE.plan()
    assert plan.is_pair and plan.is_round and plan.unknowns == ("v1'", "v2'")
    assert SWAP.plan().unknowns == () and SWAP.plan().is_pair
    assert plan.hop_targets == {"w1": "x", "w2": "x"}
    assert [b.walker for b in plan.balls] == ["w1", "w2"]
    assert not STREAM.plan().is_pair and STREAM.plan().is_round
    assert not RuleSpec.parse("{w,s,v}, {s,t}_k, {o,v'}_k -> {w,t,v'}").plan().is_round
    assert isinstance(rule_item(STREAM), BallRule)
    assert isinstance(rule_item(RANDOM_HOP), ChainRule)
    assert RuleSpec.parse("{w,s,v}_k, {s,t}_k -> {w,t,v}").type_of("v") == "velocity"


@pytest.mark.parametrize(
    ("notation", "message"),
    [
        ("{w,s}, {s,t}", "'->'"),
        ("{s,t} -> {s,t}", "exactly one walker edge"),
        ("{w,s}, {t,u} -> {w,u}, {t,u}", "must form a chain"),
        ("{w,s}, {s,t} -> {w,s}", "keep the site edges"),
        ("{w,s}, {s,t} -> {w,q}, {s,t}", "end on a site of the chain"),
        ("{w,s}, {w2,s} -> {w,s}, {w2,s}", "names both velocities"),
        ("{w,s,v}, {w2,t,v2} -> {w,s,v}, {w2,t,v2}", "needs a contact"),
        ("{w,s,v}_k, {s,t}_k -> {w,t,q}", "must be bound or solved"),
        ("{w,s,v}_k, {s,t}_k, {v,v'}|j -> {w,t,v'}", "is not bound"),
        ("{w,s,v}, {v,v'}=d -> {w,s,v'}", "direction d"),
        (
            "{w,s,v}_k, {s,t}_k, {v,v'}|k, {v,v''}^, {v,v'''}^ -> {w,t,v'}",
            "at most two",
        ),
        ("{w,s,v}_k -> {w,t,v}", "bound on the left side"),
        ("{w,s,v}_k, {s,t}_k -> {w,t,v}, {t,s}_k", "space is static"),
        ("{w,s,v}_k, {s,t}_k -> {w,t,v}_k", "on the right side a ball"),
        ("{w,s,v}_k, {s,t}_k, {w,t,v}_k -> {w,t,v}", "appears twice"),
        ("{w,s,v}_k, {v,t}_k -> {w,s,v}", "mixes a velocity with a site"),
        ("{w,s,v}_k, {s,t}_-k -> {w,t,v}", "plain variable or"),
    ],
)
def test_unsupported_patterns_are_rejected_with_a_reason(notation, message):
    with pytest.raises(ValueError, match=message):
        RuleSpec.parse(notation)
    with pytest.raises(ValueError):
        RuleSpec.parse("{w,s}, {s,t} -> {w,t}, {s,t}", weights=(1, 1, 1))


def test_chain_rule_with_the_random_hop_spec_reproduces_random_hop_diffusion():
    spacing, step_time = 0.5, 0.2
    lattice = SquareLattice(101, spacing=spacing)
    rule = rule_from_spec(RANDOM_HOP.to_dict())
    assert isinstance(rule, ChainRule) and isinstance(rule, Rule)
    assert rule_from_spec(None).name == "RandomHop"
    simulation = GraphSimulation(lattice, rule, step_time=step_time)
    initial = walkers_at_node(lattice, lattice.center_node, 4000)
    history = simulation.run(initial, 40, np.random.default_rng(2), record_paths=True)
    trajectory = history.project()
    diffusion = lattice_hop_diffusion(spacing, step_time)
    np.testing.assert_allclose(
        trajectory.msd()[1:], 4 * diffusion * trajectory.times[1:], rtol=0.08
    )
    np.testing.assert_allclose(RANDOM_HOP.diffusion(spacing, step_time), diffusion)
    paths = history.paths
    assert paths is not None and paths.shape == (40, 4000, 2)
    np.testing.assert_array_equal(paths[:, :, 1], history.anchors[1:])
    graph = lattice.to_networkx()
    for before, after in zip(paths[0, :50, 0], paths[0, :50, 1], strict=True):
        assert graph.has_edge(int(before), int(after))


def test_biased_double_hop_has_the_predicted_drift_and_diffusion():
    spec = RuleSpec.parse(
        "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}",
        name="Biased",
        weights=(0.5, 0.1, 0.2, 0.2),
    )
    spacing, step_time = 1.0, 1.0
    lattice = SquareLattice(201, spacing=spacing)
    simulation = GraphSimulation(lattice, ChainRule(spec), step_time=step_time)
    initial = walkers_at_node(lattice, lattice.center_node, 5000)
    history = simulation.run(initial, 30, np.random.default_rng(4))
    positions = history.project().positions
    displacement = positions[-1] - positions[0]
    t = history.times[-1]
    np.testing.assert_allclose(
        displacement.mean(axis=0) / t, spec.drift(spacing, step_time), atol=0.03
    )
    variance = displacement.var(axis=0) / (2 * t)
    np.testing.assert_allclose(variance, spec.diffusion(spacing, step_time), rtol=0.1)
    description = spec.describe(spacing, step_time)
    assert (
        "drift v = (0.8" in description["physics"] and description["name"] == "Biased"
    )


def test_forbidding_self_loops_keeps_boundary_walkers_moving():
    spec = RuleSpec.parse(
        "{w,s}, {s,t} -> {w,t}, {s,t}", name="NoStay", self_loops="forbid"
    )
    lattice = SquareLattice(4)
    corner = walkers_at_node(lattice, 0, 200)
    after = ChainRule(spec).apply(corner, np.random.default_rng(0))
    assert not np.any(after.anchors == 0)
    assert set(after.anchors.tolist()) <= {
        lattice.node_index(1, 0),
        lattice.node_index(0, 1),
    }


def test_experiment_runs_an_edited_rule_and_a_drift_fails_the_comparison():
    biased = RuleSpec.parse(
        "{w,s}, {s,t} -> {w,t}, {s,t}", name="Drift", weights=(0.5, 0.1, 0.2, 0.2)
    ).to_dict()
    config = ExperimentConfig(
        particles=300, steps=300, record_every=50, lattice_size=21, rule=biased
    )
    result = run_experiment(config)
    assert config.rule_name == "Drift" and "Drift" in result.report.candidate_label
    assert not result.passed
    with pytest.raises(ValueError, match="walker edge"):
        ExperimentConfig(rule={"notation": "{s,t} -> {s,t}"})
    with pytest.raises(ValueError, match="ball rule"):
        ExperimentConfig(rule=STREAM.to_dict())


def _lattice_and_rule(spec, size=7):
    return SquareLattice(size), BallRule(spec)


def test_stream_moves_along_the_velocity_and_keeps_it():
    lattice, stream = _lattice_and_rule(STREAM)
    state = WalkerGraph(
        lattice, np.array([24, 24, 10]), np.array([[1, 0], [0, -2], [0, 0]])
    )
    after = stream.apply(state, np.random.default_rng(0))
    assert after.anchors.tolist() == [24 + 7, 24 - 1, 10]  # one hop per application
    np.testing.assert_array_equal(after.velocities, state.velocities)
    text = stream.describe()
    assert text["name"] == "Stream" and "free flight" in text["physics"]
    assert text["rewrite"] == STREAM.notation


def test_reflect_mirrors_the_hit_component_only_at_the_wall():
    lattice, reflect = _lattice_and_rule(REFLECT)
    right_wall = lattice.node_index(6, 3)
    state = WalkerGraph(lattice, np.array([right_wall, 24]), np.array([[2, 1], [2, 1]]))
    after = reflect.apply(state, np.random.default_rng(0))
    assert after.velocities.tolist() == [[-2, 1], [2, 1]]
    np.testing.assert_array_equal(after.anchors, state.anchors)
    # On a periodic lattice there is no wall: the rule never matches.
    torus = SquareLattice(7, periodic=True)
    state = WalkerGraph(torus, np.array([torus.node_index(6, 3)]), np.array([[2, 1]]))
    assert reflect.apply(state, np.random.default_rng(0)).velocities.tolist() == [
        [2, 1]
    ]


def test_exchange_is_the_rectangle_in_the_velocity_lattice_and_swap_the_diagonal():
    lattice, exchange = _lattice_and_rule(EXCHANGE)
    # Balls at (2,3) and (4,3) whose first hops meet at (3,3), line of centres +x;
    # velocities (2,1) and (-2,2): the x components are exchanged, the y components kept.
    state = WalkerGraph(
        lattice,
        np.array([lattice.node_index(2, 3), lattice.node_index(4, 3)]),
        np.array([[2, 1], [-2, 2]]),
    )
    after = exchange.apply(state, np.random.default_rng(0))
    assert after.velocities.tolist() == [[-2, 1], [2, 2]]
    np.testing.assert_array_equal(after.anchors, state.anchors)
    # A separating pair (no approach along the line of centres) is left alone.
    apart = WalkerGraph(
        lattice,
        np.array([lattice.node_index(2, 3), lattice.node_index(3, 3)]),
        np.array([[-1, 0], [1, 0]]),
    )
    assert exchange.apply(apart, np.random.default_rng(0)).velocities.tolist() == [
        [-1, 0],
        [1, 0],
    ]
    swap = BallRule(SWAP)
    after = swap.apply(state, np.random.default_rng(0))
    assert after.velocity.tolist() == [[-2, 2], [2, 1]]


def test_turn_rotates_a_head_on_pair_either_way():
    lattice, turn = _lattice_and_rule(TURN)
    state = WalkerGraph(
        lattice,
        np.array([lattice.node_index(2, 3), lattice.node_index(4, 3)]),
        np.array([[1, 0], [-1, 0]]),
    )
    senses = set()
    for seed in range(12):
        after = turn.apply(state, np.random.default_rng(seed))
        v = after.velocities
        assert v[0].tolist() in ([0, 1], [0, -1]) and v[1].tolist() == (-v[0]).tolist()
        senses.add(tuple(v[0].tolist()))
    assert senses == {(0, 1), (0, -1)}
    oblique = WalkerGraph(
        lattice,
        np.array([lattice.node_index(2, 3), lattice.node_index(4, 3)]),
        np.array([[1, 0], [-1, 1]]),
    )
    assert turn.apply(oblique, np.random.default_rng(0)).velocities.tolist() == [
        [1, 0],
        [-1, 1],
    ]
