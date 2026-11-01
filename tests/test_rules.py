"""Rules as data: parsing, validation messages, and the physics the chain rule family produces."""

import json

import numpy as np
import pytest

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.graph import GraphSimulation, RandomHop, SquareLattice, walkers_at_node
from imigraph.graph.rules import RANDOM_HOP, ChainRule, Rule, RuleSpec, rule_from_spec
from imigraph.graph.simulation import lattice_hop_diffusion


def test_notation_round_trip_and_json():
    spec = RuleSpec.parse(
        "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}", name="DoubleHop"
    )
    assert spec.notation == "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}"
    assert spec.chain() == ("w", ["s", "t", "u"], 2) and spec.hops == 2
    data = json.loads(json.dumps(spec.to_dict()))
    assert RuleSpec.from_dict(data) == spec
    assert data["types"] == {"w": "walker", "s": "site", "t": "site", "u": "site"}
    assert RuleSpec.from_dict({"name": "x", "notation": spec.notation}).hops == 2
    assert RandomHop().spec == RANDOM_HOP


@pytest.mark.parametrize(
    ("notation", "message"),
    [
        ("{w,s}, {s,t}", "'->'"),
        ("{s,t} -> {s,t}", "exactly one walker edge"),
        ("{w,s}, {w2,s} -> {w,s}, {w2,s}", "exactly one walker edge"),
        ("{w,s}, {t,u} -> {w,u}, {t,u}", "must form a chain"),
        ("{w,s}, {s,t} -> {w,s}", "keep the site edges"),
        ("{w,s}, {s,t} -> {w,q}, {s,t}", "end on a site of the chain"),
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
