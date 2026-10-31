"""The browser entry point returns everything the GUI renders, as JSON."""

import json

from imigraph.report.live import run, run_json


def test_run_json_round_trip():
    config = {"particles": 60, "steps": 100, "record_every": 20, "lattice_size": 15, "seed": 2}
    payload = json.loads(run_json(json.dumps(config), max_particles=50))
    assert set(payload) == {"passed", "markdown", "data", "fragments"}
    assert len(payload["data"]["reference"]["times"]) == 6
    assert len(payload["data"]["reference"]["positions"][0]) == 50
    assert payload["data"]["config"]["seed"] == 2
    assert set(payload["fragments"]) >= {
        "checks_table",
        "msd_chart",
        "marginal_early",
        "marginal_final",
        "occupancy_charts",
        "energy_chart",
        "momentum_chart",
        "entropy_chart",
        "wall_rate_chart",
        "occupied_chart",
        "rules_table",
    }
    assert payload["data"]["rules"][0]["name"] == "RandomHop"
    assert "entropy" in payload["data"]["metrics"]["graph"]
    assert payload["fragments"]["msd_chart"].startswith("<svg")
    assert "| msd |" in payload["markdown"]


def test_run_uses_defaults_when_config_is_empty():
    result = run({"particles": 20, "steps": 40, "record_every": 20, "lattice_size": 9})
    assert result["data"]["config"]["spacing"] == 0.2


def test_playground_runs_an_edited_rule_and_reports_errors_instead_of_raising():
    from imigraph.report.live import playground, playground_json

    result = playground({"size": 9, "walkers": 3, "steps": 20, "seed": 2})
    assert result["error"] is None and result["rule"]["name"] == "RandomHop"
    assert len(result["demo"]["anchors"]) == 21 and len(result["demo"]["paths"]) == 20
    assert len(result["demo"]["paths"][0][0]) == 2
    assert result["predicted"]["hops"] == 1 and result["measured"]["occupied_final"] >= 1
    biased = {
        "name": "Biased",
        "notation": "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}",
        "weights": [0.7, 0.1, 0.1, 0.1],
    }
    payload = json.loads(playground_json(json.dumps({"rule": biased, "size": 25, "walkers": 50})))
    assert payload["error"] is None and payload["predicted"]["drift"][0] > 1.0
    assert len(payload["demo"]["paths"][0][0]) == 3
    assert payload["spec"]["name"] == "Biased"
    bad = playground({"rule": {"notation": "{s,t} -> {s,t}"}})
    assert "walker edge" in bad["error"]
    assert playground({"size": 1000, "walkers": 1, "steps": 1})["error"] is None  # clamped


def test_run_with_a_rule_uses_it_for_the_rule_diagram():
    rule = {"name": "Two", "notation": "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}"}
    result = run(
        {"particles": 20, "steps": 40, "record_every": 20, "lattice_size": 9, "rule": rule}
    )
    assert result["data"]["rules"][0]["name"] == "Two"
    assert len(result["data"]["rule"]["paths"][0][0]) == 3
