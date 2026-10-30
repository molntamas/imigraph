import json
import re

import pytest

from imigraph.__main__ import main
from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.report import DEFAULT_FILENAME, build_report, default_demo, write_report

SMALL = ExperimentConfig(particles=100, steps=200, record_every=20, lattice_size=21, seed=3)


@pytest.fixture(scope="module")
def page() -> str:
    return build_report(run_experiment(SMALL), demo=default_demo(steps=10, walkers=3))


def test_page_is_self_contained_and_has_every_section(page):
    assert page.startswith("<!DOCTYPE html>")
    for element_id in (
        "reference-canvas",
        "demo-canvas",
        "pair-reference-canvas",
        "pair-graph-canvas",
        "data",
    ):
        assert f'id="{element_id}"' in page
    assert "<svg" in page and "imigraphViz" in page
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', page)


def test_embedded_data_matches_the_experiment(page):
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    data = json.loads(blob.group(1))
    assert len(data["reference"]["times"]) == 11
    assert len(data["reference"]["positions"][0]) == 100
    assert len(data["demo"]["anchors"]) == 11 and len(data["demo"]["anchors"][0]) == 3
    assert data["report"]["checks"][0]["name"] == "msd"
    assert data["config"]["seed"] == 3


def test_write_report_into_a_directory_and_a_file(tmp_path):
    result = run_experiment(SMALL)
    written = write_report(tmp_path, result=result)
    assert written == tmp_path / DEFAULT_FILENAME and written.stat().st_size > 10_000
    explicit = write_report(tmp_path / "custom.html", result=result)
    assert explicit.name == "custom.html" and explicit.exists()


def test_cli_report_command(tmp_path, capsys):
    code = main(
        [
            "report",
            "--out",
            str(tmp_path),
            "--particles",
            "100",
            "--steps",
            "200",
            "--record-every",
            "20",
            "--lattice-size",
            "21",
        ]
    )
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "| msd |" in out and (tmp_path / DEFAULT_FILENAME).exists()
