"""The playground: rule sets, starting universes and the saved result page."""

import json
import re

import numpy as np
import pytest

from imigraph.__main__ import main
from imigraph.graph import SquareLattice
from imigraph.graph.rules import RuleSequence
from imigraph.report import INDEX_FILENAME
from imigraph.report.live import (
    PLACEMENTS,
    initial_universe,
    playground,
    playground_page_json,
)
from imigraph.report.playground import (
    PLAYGROUND_STEM,
    build_playground_page,
    playground_filename,
    write_playground_page,
)

HOP = {"name": "Hop", "notation": "{w,s}, {s,t} -> {w,t}, {s,t}"}
BIASED = {
    "name": "Push",
    "notation": "{w,s}, {s,t} -> {w,t}, {s,t}",
    "weights": [0.7, 0.1, 0.1, 0.1],
}
SESSION = {
    "size": 9,
    "walkers": 5,
    "steps": 12,
    "seed": 3,
    "walls": "periodic",
    "placement": "left",
    "rule": [HOP, BIASED],
}


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_every_placement_puts_the_walkers_on_the_world(placement):
    lattice = SquareLattice(7)
    rng = np.random.default_rng(0)
    initial = initial_universe(lattice, placement, 10, [3, 48, 48, 99], rng)
    assert np.all(initial.anchors >= 0) and np.all(initial.anchors < 49)
    if placement == "sites":
        assert initial.anchors.tolist() == [3, 48, 48]  # 99 is not on the world
    else:
        assert initial.num_walkers == 10
    if placement == "left":
        assert np.all(initial.anchors // 7 < 3)
    if placement == "corner":
        assert np.all(initial.anchors == 0)
    with pytest.raises(ValueError, match="place at least one walker"):
        initial_universe(lattice, "sites", 1, [], rng)
    with pytest.raises(ValueError, match="placement must be one of"):
        initial_universe(lattice, "moon", 1, [], rng)


def test_playground_runs_a_rule_set_on_an_edited_universe():
    result = playground(SESSION)
    assert result["error"] is None
    assert result["rule"]["name"] == "Hop + Push"
    assert [r["name"] for r in result["rules"]] == ["Hop", "Push", "Hop + Push"]
    assert all(r["pattern"].startswith("<svg") for r in result["rules"][:2])
    assert [s["name"] for s in result["specs"]] == ["Hop", "Push"]
    universe = result["universe"]
    assert universe["walls"] == "periodic" and universe["walkers"] == 5
    assert all(site // 9 < 4 for site in universe["sites"])
    assert result["demo"]["anchors"][0] == universe["sites"]
    assert len(result["demo"]["paths"][0][0]) == 3  # two chains of one edge, joined
    np.testing.assert_allclose(result["predicted"]["drift"], [0.6, 0.0])
    assert result["predicted"]["hops"] == 2
    # No self-loops on a periodic world (a two-hop step back to the start would count as
    # "stayed", so the check uses a single hop).
    periodic = playground({"walls": "periodic", "rule": HOP, "size": 5, "steps": 30})
    assert periodic["measured"]["wall_rate"] == 0.0
    clicked = playground({"placement": "sites", "sites": [0, 0, 24], "size": 5})
    assert clicked["universe"]["sites"] == [0, 0, 24]
    assert clicked["universe"]["walkers"] == 3
    assert "enable at least one rule" in playground({"rule": []})["error"]
    assert "walls must be" in playground({"walls": "soft"})["error"]
    assert "place at least one" in playground({"placement": "sites"})["error"]


def test_rule_sequence_paths_and_physics_add_up():
    from imigraph.graph import GraphSimulation, walkers_at_node
    from imigraph.graph.rules import ChainRule, RuleSpec, rule_from_spec

    look = RuleSpec.parse("{w,s}, {s,t}, {t,u} -> {w,t}, {s,t}, {t,u}", name="Look")
    push = RuleSpec.from_dict(BIASED)
    rule = rule_from_spec([look.to_dict(), push.to_dict()])
    assert isinstance(rule, RuleSequence) and rule.name == "Look + Push"
    assert isinstance(rule_from_spec([HOP]), ChainRule)
    lattice = SquareLattice(61)
    simulation = GraphSimulation(lattice, rule)
    history = simulation.run(
        walkers_at_node(lattice, lattice.center_node, 3000),
        20,
        np.random.default_rng(5),
        record_paths=True,
    )
    paths = history.paths
    assert paths is not None and paths.shape == (20, 3000, 5)
    # Look ends on the middle site of its chain [s0, s1, s2], so Push starts at s1 again (the
    # joint is only dropped when a rule ends on the last site of its chain).
    np.testing.assert_array_equal(paths[:, :, 3], paths[:, :, 1])
    np.testing.assert_array_equal(paths[:, :, 4], history.anchors[1:])
    graph = lattice.to_networkx()
    for a, b in paths[0, :40, 3:]:
        assert graph.has_edge(int(a), int(b))
    positions = history.project().positions
    displacement = positions[-1] - positions[0]
    t = history.times[-1]
    np.testing.assert_allclose(
        displacement.mean(axis=0) / t, rule.drift(1, 1), atol=0.03
    )
    # Variances of independent increments add: D = D_look + D_push (standard error ~2%).
    np.testing.assert_allclose(
        displacement.var(axis=0) / (2 * t), rule.diffusion(1, 1), rtol=0.1
    )
    described = rule.descriptions()
    assert (
        len(described) == 3 and "2 rules applied in order" in described[-1]["matches"]
    )
    with pytest.raises(ValueError, match="at least one rule"):
        RuleSequence(())


def test_experiment_and_cli_accept_a_rule_list():
    from imigraph.experiments import ExperimentConfig, run_experiment
    from imigraph.report.html import rule_descriptions

    config = ExperimentConfig(
        particles=50, steps=40, record_every=20, lattice_size=9, rule=[HOP, HOP]
    )
    assert config.rule_name == "Hop + Hop"
    result = run_experiment(config)
    names = [r["name"] for r in rule_descriptions(result)]
    assert names == ["Hop", "Hop", "Hop + Hop"]
    with pytest.raises(ValueError, match="at least one rule"):
        ExperimentConfig(rule=[])


def test_page_is_self_contained_and_embeds_the_session():
    result = playground(SESSION)
    page = build_playground_page(result, SESSION)
    assert page.startswith("<!DOCTYPE html>")
    assert "<title>imigraph playground: Hop + Push</title>" in page
    for element_id in ("universe-canvas", "rule-canvas", "demo-canvas", "data"):
        assert f'id="{element_id}"' in page
    assert "imigraphViz" in page and page.count("<svg") >= 2
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', page)
    blob = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL
    )
    assert blob is not None
    data = json.loads(blob.group(1))
    assert data["universe"]["walls"] == "periodic"
    assert "&quot;placement&quot;: &quot;left&quot;" in page
    assert "Hop + Push" in playground_page_json(json.dumps(SESSION))
    assert playground_page_json(json.dumps({"rule": []})) == ""
    with pytest.raises(ValueError, match="at least one rule"):
        build_playground_page(playground({"rule": []}))


def test_every_run_writes_a_new_timestamped_file(tmp_path):
    first = write_playground_page(tmp_path, SESSION, label="rule set")
    assert re.fullmatch(
        rf"{PLAYGROUND_STEM}_\d{{4}}-\d\d-\d\d_\d{{6}}_rule-set\.html", first.name
    )
    second = write_playground_page(tmp_path / "custom.html", SESSION)
    assert second.name == "custom.html" and first.exists()
    index = (tmp_path / INDEX_FILENAME).read_text(encoding="utf-8")
    assert f'href="{first.name}"' in index and 'href="custom.html"' in index
    from datetime import UTC, datetime

    when = datetime(2026, 11, 1, 9, 5, 9, tzinfo=UTC)
    assert (
        playground_filename(when, "two rules")
        == "playground_2026-11-01_090509_two-rules.html"
    )


def test_cli_playground_report(tmp_path, capsys):
    session = tmp_path / "session.json"
    session.write_text(json.dumps(SESSION), encoding="utf-8")
    code = main(
        [
            "playground-report",
            "--out",
            str(tmp_path),
            "--request-file",
            str(session),
            "--label",
            "cli",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "rules: Hop + Push" in out and "Wrote " in out
    assert len(list(tmp_path.glob(f"{PLAYGROUND_STEM}_*_cli.html"))) == 1
    bad = main(
        ["playground-report", "--out", str(tmp_path), "--request", '{"rule": []}']
    )
    assert bad == 2 and "enable at least one rule" in capsys.readouterr().err
    assert (
        len(list(tmp_path.glob("*.html"))) == 2
    )  # the page and the index, nothing else
