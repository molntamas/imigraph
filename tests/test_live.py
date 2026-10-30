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
