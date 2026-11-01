"""The browser GUI (rule editor, playground, comparison run) as one self-contained HTML file.

``python -m imigraph gui-page --label <change>`` writes ``output/rule_editor_<UTC time>_<label>.html``:
``web/index.html`` with the stylesheet, ``viz.js``, ``app.js`` and the project wheel (base64) inlined,
so the page opens from ``file://`` or from any static host without a web server for the wheel. Only
Pyodide, numpy and scipy are still downloaded from the jsDelivr CDN when the page loads. Like the
other pages it is a new timestamped file per run, listed in ``output/index.html``.
"""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from imigraph.report.html import _asset, write_index

GUI_STEM = "rule_editor"
WEB_DIR = Path(__file__).resolve().parents[3] / "web"

_STYLESHEET = '<link rel="stylesheet" href="dist/report.css">'
_VIZ_SCRIPT = '<script src="dist/viz.js"></script>'
_APP_SCRIPT = '<script src="app.js"></script>'


def _replace_once(page: str, marker: str, replacement: str) -> str:
    if page.count(marker) != 1:
        raise ValueError(f"web/index.html must contain {marker!r} exactly once")
    return page.replace(marker, replacement)


def _inline_script(source: str, name: str) -> str:
    if "</script" in source.lower():
        raise ValueError(f"{name} contains '</script', it cannot be inlined")
    return source


def build_gui_page(wheel: str | Path, web: str | Path = WEB_DIR) -> str:
    """The GUI page of ``web/`` with every local resource and the wheel at ``wheel`` inlined."""
    web = Path(web)
    wheel = Path(wheel)
    page = (web / "index.html").read_text(encoding="utf-8")
    app_js = _inline_script((web / "app.js").read_text(encoding="utf-8"), "web/app.js")
    viz_js = _inline_script(_asset("viz.js"), "viz.js")
    encoded = base64.encodebytes(wheel.read_bytes()).decode("ascii")
    embedded_wheel = (
        f'<script id="wheel" type="application/octet-stream" '
        f'data-name="{escape(wheel.name)}">\n{encoded}</script>'
    )
    page = _replace_once(
        page, _STYLESHEET, f"<style>\n{_asset('report.css')}\n</style>"
    )
    page = _replace_once(page, _VIZ_SCRIPT, f"<script>\n{viz_js}\n</script>")
    page = _replace_once(
        page, _APP_SCRIPT, f"{embedded_wheel}\n<script>\n{app_js}\n</script>"
    )
    return page


def build_wheel(out_dir: str | Path) -> Path:
    """``uv build --wheel`` into ``out_dir``; returns the wheel path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir), "--quiet"], check=True
    )
    wheels = sorted(out_dir.glob("imigraph-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(f"expected one wheel in {out_dir}, found {wheels}")
    return wheels[0]


def gui_filename(when: datetime | None = None, label: str = "") -> str:
    """``rule_editor_<UTC timestamp>[_<label>].html``: a new file per run, old runs kept."""
    when = datetime.now(UTC) if when is None else when
    suffix = "_" + re.sub(r"[^A-Za-z0-9-]+", "-", label).strip("-") if label else ""
    return f"{GUI_STEM}_{when:%Y-%m-%d_%H%M%S}{suffix}.html"


def write_gui_page(
    path: str | Path,
    wheel: str | Path | None = None,
    label: str = "",
    web: str | Path = WEB_DIR,
) -> Path:
    """Write the self-contained GUI page. ``path`` is normally a directory: a new timestamped
    file is created in it and its ``index.html`` is refreshed; a path ending in ``.html``
    writes exactly that file. Without ``wheel`` the project wheel is built first."""
    path = Path(path)
    if path.suffix.lower() != ".html":
        path = path / gui_filename(label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    if wheel is None:
        with tempfile.TemporaryDirectory() as tmp:
            page = build_gui_page(build_wheel(tmp), web)
    else:
        page = build_gui_page(wheel, web)
    path.write_text(page, encoding="utf-8")
    write_index(path.parent)
    return path
