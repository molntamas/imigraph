"""The result page of a playground run: the rule set, the starting universe, the run step by
step and the measured physics, as one self-contained HTML file.

The GUI builds the same page in the browser (*Save result page*, through
:func:`imigraph.report.live.playground_page_json`); ``python -m imigraph playground-report
--request-file session.json`` builds it from the session file the GUI exports. Like the other
pages it is written as a new timestamped file and listed in ``output/index.html``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Any

from imigraph.report.html import _asset, _rules_table, write_index

PLAYGROUND_STEM = "playground"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in rows
    )
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def _measured_table(result: dict[str, Any]) -> str:
    m, p = result["measured"], result["predicted"]
    f = lambda x: f"{float(x):.3f}"
    return _kv_table(
        [
            ("mean |dr|^2 per step (a^2)", f(m["mean_squared_step"])),
            (
                "diffusion D measured / predicted (x, y)",
                f"{f(m['diffusion'])} / {f(p['diffusion'][0])}, {f(p['diffusion'][1])}",
            ),
            (
                "drift v measured / predicted",
                (
                    f"({f(m['drift'][0])}, {f(m['drift'][1])}) / "
                    f"({f(p['drift'][0])}, {f(p['drift'][1])})"
                ),
            ),
            ("hops per step (predicted)", str(p["hops"])),
            ("wall rate (fraction of walkers per step)", f(m["wall_rate"])),
            ("occupied sites at the end", str(m["occupied_final"])),
        ]
    )


def _universe_table(universe: dict[str, Any]) -> str:
    sites = universe["sites"]
    shown = ", ".join(str(s) for s in sites[:12]) + (" ..." if len(sites) > 12 else "")
    return _kv_table(
        [
            ("world", f"{universe['size']} x {universe['size']} square lattice"),
            ("walls", str(universe["walls"])),
            ("walkers", str(universe["walkers"])),
            ("placement", str(universe["placement"])),
            ("starting sites (node indices)", shown),
            ("steps", str(universe["steps"])),
            ("seed", str(universe["seed"])),
            (
                "spacing a / step time tau",
                f"{universe['spacing']:g} / {universe['step_time']:g}",
            ),
        ]
    )


def build_playground_page(
    result: dict[str, Any], request: dict[str, Any] | None = None
) -> str:
    """Render the page for a :func:`imigraph.report.live.playground` result. ``request`` (the
    session as the GUI exports it) is embedded so the page can be regenerated."""
    if result.get("error"):
        raise ValueError(result["error"])
    request = request or {}
    names = [spec["name"] for spec in result["specs"]]
    template = Template(_asset("playground.html"))
    return template.substitute(
        css=_asset("report.css"),
        viz_js=_asset("viz.js"),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        title=escape(" + ".join(names)),
        num_rules=len(names),
        rules_table=_rules_table(result["rules"]),
        universe_table=_universe_table(result["universe"]),
        measured_table=_measured_table(result),
        request_json=escape(json.dumps(request, indent=2)),
        data_json=json.dumps(result, separators=(",", ":")),
    )


def playground_filename(when: datetime | None = None, label: str = "") -> str:
    """``playground_<UTC timestamp>[_<label>].html``: a new file per run, old runs kept."""
    when = datetime.now(UTC) if when is None else when
    suffix = "_" + re.sub(r"[^A-Za-z0-9-]+", "-", label).strip("-") if label else ""
    return f"{PLAYGROUND_STEM}_{when:%Y-%m-%d_%H%M%S}{suffix}.html"


def write_playground_page(
    path: str | Path, request: dict[str, Any] | None = None, label: str = ""
) -> Path:
    """Run the playground for ``request`` and write its page. ``path`` is normally a directory:
    a new timestamped file is created in it and its ``index.html`` is refreshed; a path ending
    in ``.html`` writes exactly that file."""
    from imigraph.report.live import playground

    result = playground(request)
    path = Path(path)
    if path.suffix.lower() != ".html":
        path = path / playground_filename(label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_playground_page(result, request), encoding="utf-8")
    write_index(path.parent)
    return path
