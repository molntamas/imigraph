import json
import re

import pytest

from imigraph.__main__ import main
from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.report import (
    INDEX_FILENAME,
    REPORT_STEM,
    build_report,
    default_demo,
    report_filename,
    write_report,
)

SMALL = ExperimentConfig(
    particles=100, steps=200, record_every=20, lattice_size=21, seed=3
)


@pytest.fixture(scope="module")
def page() -> str:
    return build_report(run_experiment(SMALL), demo=default_demo(steps=10, walkers=3))


def test_page_is_self_contained_and_has_every_section(page):
    assert page.startswith("<!DOCTYPE html>")
    for element_id in (
        "reference-canvas",
        "rule-canvas",
        "demo-canvas",
        "pair-reference-canvas",
        "pair-graph-canvas",
        "data",
    ):
        assert f'id="{element_id}"' in page
    assert "<svg" in page and "imigraphViz" in page
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', page)


def test_embedded_data_matches_the_experiment(page):
    blob = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL
    )
    assert blob is not None
    data = json.loads(blob.group(1))
    assert len(data["reference"]["times"]) == 11
    assert len(data["reference"]["positions"][0]) == 100
    assert len(data["demo"]["anchors"]) == 11 and len(data["demo"]["anchors"][0]) == 3
    assert data["report"]["checks"][0]["name"] == "msd"
    assert data["config"]["seed"] == 3
    assert data["rules"][0]["name"] == "RandomHop"
    assert (
        len(data["rule"]["space"]["neighbors"]) == 25
        and len(data["rule"]["anchors"]) == 25
    )
    assert "The rule, explained" in page
    assert (
        "<th>RandomHop</th><td><code>{w,s}, {s,s&#x27;} -&gt; {w,s&#x27;}, {s,s&#x27;}</code>"
        in page
    )


def test_every_run_writes_a_new_file_and_keeps_the_old_ones(tmp_path):
    result = run_experiment(SMALL)
    first = write_report(tmp_path, result=result)
    assert re.fullmatch(rf"{REPORT_STEM}_\d{{4}}-\d\d-\d\d_\d{{6}}\.html", first.name)
    assert first.stat().st_size > 10_000
    second = write_report(tmp_path / "custom.html", result=result)
    assert second.name == "custom.html" and first.exists()
    index = (tmp_path / INDEX_FILENAME).read_text(encoding="utf-8")
    assert f'href="{first.name}"' in index and 'href="custom.html"' in index
    assert index.count("<tr>") == 3  # header + two reports
    assert sorted(p.name for p in tmp_path.glob("*.html")) == sorted(
        [first.name, "custom.html", INDEX_FILENAME]
    )


def test_report_filename_is_timestamped():
    from datetime import UTC, datetime

    when = datetime(2026, 10, 30, 18, 5, 9, tzinfo=UTC)
    assert report_filename(when) == "brownian_vs_graph_2026-10-30_180509.html"
    assert report_filename(when, "energy & momentum") == (
        "brownian_vs_graph_2026-10-30_180509_energy-momentum.html"
    )


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
    assert "| msd |" in out
    assert len(list(tmp_path.glob(f"{REPORT_STEM}_*.html"))) == 1
    assert (tmp_path / INDEX_FILENAME).exists()
