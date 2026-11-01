"""The self-contained GUI page (python -m imigraph gui-page)."""

import base64
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from imigraph.__main__ import main
from imigraph.report.bundle import (
    WEB_DIR,
    build_gui_page,
    gui_filename,
    write_gui_page,
)

WHEEL_BYTES = b"PK\x03\x04 not really a wheel, but any bytes will do \x00\xff"


@pytest.fixture
def wheel(tmp_path):
    path = tmp_path / "imigraph-0.1.0-py3-none-any.whl"
    path.write_bytes(WHEEL_BYTES)
    return path


def test_page_inlines_every_local_resource_and_the_wheel(wheel):
    page = build_gui_page(wheel)
    assert 'href="dist/' not in page and 'src="dist/' not in page
    assert 'src="app.js"' not in page
    assert "https://cdn.jsdelivr.net/pyodide/" in page  # only Pyodide stays remote
    assert "window.imigraphViz" in page  # viz.js
    assert "imigraph.playground.v3" in page  # app.js
    assert "--border" in page  # report.css
    embedded = re.search(
        r'<script id="wheel" type="application/octet-stream" data-name="([^"]+)">'
        r"\n(.*?)</script>",
        page,
        re.DOTALL,
    )
    assert embedded is not None
    assert embedded.group(1) == wheel.name
    assert base64.b64decode(embedded.group(2)) == WHEEL_BYTES
    # The embedded wheel comes before the app script that reads it.
    assert page.index('id="wheel"') < page.index("imigraph.playground.v3")


def test_a_script_that_would_close_the_tag_is_refused(tmp_path, wheel):
    web = tmp_path / "web"
    web.mkdir()
    shutil.copy(WEB_DIR / "index.html", web / "index.html")
    closing = "</" + "script>"
    (web / "app.js").write_text(f'var s = "{closing}";\n', encoding="utf-8")
    with pytest.raises(ValueError, match="app.js"):
        build_gui_page(wheel, web)


def test_a_changed_page_without_the_markers_is_refused(tmp_path, wheel):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html></html>", encoding="utf-8")
    (web / "app.js").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        build_gui_page(wheel, web)


def test_gui_filename_is_timestamped():
    when = datetime(2026, 11, 1, 14, 30, 5, tzinfo=UTC)
    assert gui_filename(when) == "rule_editor_2026-11-01_143005.html"
    assert (
        gui_filename(when, "rule editor!")
        == "rule_editor_2026-11-01_143005_rule-editor.html"
    )


def test_every_run_writes_a_new_file_and_the_index(tmp_path, wheel):
    out = tmp_path / "output"
    first = write_gui_page(out, wheel=wheel, label="a")
    second = write_gui_page(out / "rule_editor_2026-11-01_000000_b.html", wheel=wheel)
    assert first.exists() and second.exists() and first != second
    assert first.name.startswith("rule_editor_") and first.name.endswith("_a.html")
    index = (out / "index.html").read_text(encoding="utf-8")
    assert first.name in index and second.name in index
    assert "gui-page" in index


def test_cli_gui_page(tmp_path, wheel, capsys):
    out = tmp_path / "output"
    argv = ["gui-page", "--out", str(out), "--wheel", str(wheel), "--label", "x"]
    assert main(argv) == 0
    written = capsys.readouterr().out
    assert "Wrote " in written
    path = Path(written.split("Wrote ", 1)[1].strip())
    assert path.exists() and path.name.endswith("_x.html")
    assert wheel.name in path.read_text(encoding="utf-8")
