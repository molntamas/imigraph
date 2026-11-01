import json
import re

import numpy as np

from imigraph.__main__ import main
from imigraph.experiments.gas import GasConfig, LatticeGasConfig, run_gas_comparison
from imigraph.report.gas import build_gas_report, default_gas_demo, write_gas_report
from imigraph.report.html import INDEX_FILENAME

SMALL = GasConfig(particles=30, steps=200, record_every=25, seed=1)
SMALL_LATTICE = LatticeGasConfig(
    size=20, particles=30, steps=16, record_every=2, seed=1
)


def test_gas_page_has_both_systems_checks_and_charts(tmp_path):
    result = run_gas_comparison(SMALL, SMALL_LATTICE)
    page = build_gas_report(result)
    assert 'id="gas-canvas"' in page and 'id="lattice-canvas"' in page
    assert page.count("<svg") >= 7 and "questions and answers" in page
    assert (
        "info: would fail" in page
    )  # the Maxwell check of the lattice gas is informative
    blob = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL
    )
    assert blob is not None
    data = json.loads(blob.group(1))
    assert len(data["gas"]["positions"]) == 9 and len(data["lattice"]["positions"]) == 9
    assert len(data["lattice"]["velocities"][0]) == 30
    assert [c["name"] for c in data["lattice"]["report"]["checks"]] == [
        "energy",
        "momentum",
        "collisions",
        "entropy",
    ]
    assert [c["name"] for c in data["comparison"]["checks"]] == [
        "msd",
        "marginals",
        "occupancy",
    ]
    assert [r["name"] for r in data["lattice"]["rules"]] == [
        "Reflect",
        "Turn",
        "Swap",
        "Stream",
    ]
    written = write_gas_report(tmp_path, result=result, label="test")
    assert re.fullmatch(r"gas_\d{4}-\d\d-\d\d_\d{6}_test\.html", written.name)
    assert written.name in (tmp_path / INDEX_FILENAME).read_text(encoding="utf-8")


def test_demo_shows_a_collision_and_a_reflection_early():
    demo = default_gas_demo()
    assert len(demo["anchors"]) == 37 and len(demo["events"]) == 36
    assert sum(demo["collisions"][:4]) >= 1 and sum(demo["bounces"][:6]) >= 1
    sites = np.array(demo["anchors"])
    assert np.all(sites >= 0) and np.all(sites < 49)


def test_cli_gas_commands(tmp_path, capsys):
    args = ["--particles", "30", "--steps", "100", "--record-every", "25"]
    lattice = [
        "--lattice-size",
        "20",
        "--lattice-particles",
        "30",
        "--lattice-steps",
        "8",
    ]
    lattice += ["--lattice-record-every", "2"]
    assert main(["gas", *args, *lattice]) in (0, 1)
    out = capsys.readouterr().out
    assert out.count("| energy |") == 2 and "| collisions |" in out and "| msd |" in out
    main(["gas-report", "--out", str(tmp_path), *args, *lattice])
    assert len(list(tmp_path.glob("gas_*.html"))) == 1
