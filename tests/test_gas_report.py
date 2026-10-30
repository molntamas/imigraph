import json
import re

from imigraph.__main__ import main
from imigraph.experiments.gas import GasConfig, run_gas
from imigraph.report.gas import build_gas_report, write_gas_report
from imigraph.report.html import INDEX_FILENAME

SMALL = GasConfig(particles=30, steps=200, record_every=20, seed=1)


def test_gas_page_has_animation_data_checks_and_charts(tmp_path):
    result = run_gas(SMALL)
    page = build_gas_report(result)
    assert 'id="gas-canvas"' in page and page.count("<svg") >= 5
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    data = json.loads(blob.group(1))
    assert len(data["trajectory"]["positions"]) == 11
    assert len(data["trajectory"]["positions"][0]) == 30
    assert len(data["metrics"]["collisions"]) == 10
    assert [c["name"] for c in data["report"]["checks"]] == [
        "energy",
        "momentum",
        "maxwell",
        "entropy",
    ]
    written = write_gas_report(tmp_path, result=result, label="test")
    assert re.fullmatch(r"gas_\d{4}-\d\d-\d\d_\d{6}_test\.html", written.name)
    assert written.name in (tmp_path / INDEX_FILENAME).read_text(encoding="utf-8")


def test_cli_gas_commands(tmp_path, capsys):
    assert main(["gas", "--particles", "30", "--steps", "100", "--record-every", "20"]) in (0, 1)
    assert "| energy |" in capsys.readouterr().out
    main(
        [
            "gas-report",
            "--out",
            str(tmp_path),
            "--particles",
            "30",
            "--steps",
            "100",
            "--record-every",
            "20",
        ]
    )
    assert len(list(tmp_path.glob("gas_*.html"))) == 1
