"""Headless browser check of the generated pages and the Pyodide GUI, with the Chromium that the
sandcastle image ships (playwright is a dev dependency).

    uv run python -m imigraph report                       # output/brownian_vs_graph_<time>.html
    uv run python docker/check-pages.py --serve            # inside any container of the image
    docker compose --profile verify run --rm verify        # from the host: the same, GUI served
                                                           # by the `web` service

It opens the newest report, gas and playground pages in Chromium, fails on JavaScript errors,
checks that the animations have frames and draw onto the canvases, exercises the GUI playground
(rule list, graphical and text editor, the clickable universe, the saved result page) and runs
the comparison once with small parameters, and saves screenshots to output/screenshots/. The
newest self-contained GUI page (output/rule_editor_*.html, from ``python -m imigraph gui-page``)
gets the same GUI check, opened from file:// (screenshots in output/screenshots/rule-editor/). With
``--serve`` the script builds the wheel into web/dist and serves web/ itself on a free port
(the GUI still loads Pyodide from the CDN, so it needs network access).
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def serve_gui(web: Path = Path("web")) -> str:
    """Build the wheel and assets into ``web/dist`` (what docker/serve-web.sh does) and serve
    ``web/`` from a daemon thread on a free port; returns the URL."""
    dist = web / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), "--quiet"], check=True
    )
    assets = Path("src/imigraph/report/assets")
    for name in ("viz.js", "report.css"):
        shutil.copy(assets / name, dist / name)
    wheel = next(dist.glob("imigraph-*.whl")).name
    (dist / "wheel.txt").write_text(wheel + "\n", encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = partial(_QuietHandler, directory=str(web))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"serving {web}/ (wheel {wheel}) at http://127.0.0.1:{port}/")
    return f"http://127.0.0.1:{port}/"


NONBLANK_PIXELS_JS = """(id) => {
  const canvas = document.getElementById(id);
  const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  let count = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 0) count++;
  return count;
}"""


def collect_errors(page: Page) -> list[str]:
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
    page.on(
        "console",
        lambda message: (
            errors.append(f"console.error: {message.text}")
            if message.type == "error"
            else None
        ),
    )
    return errors


def goto_with_retries(page: Page, url: str, attempts: int = 30) -> None:
    for attempt in range(attempts):
        try:
            page.goto(url, wait_until="load")
            return
        except Exception as error:  # the web service may still be starting
            if attempt == attempts - 1:
                raise
            print(f"  waiting for {url} ({error.__class__.__name__})")
            time.sleep(2)


def painted(page: Page, canvas: str, minimum: int = 500) -> int:
    count = page.evaluate(NONBLANK_PIXELS_JS, canvas)
    assert count > minimum, f"{canvas} looks blank ({count} painted pixels)"
    return count


def check_report(page: Page, path: Path, out: Path) -> list[str]:
    print(f"report: {path}")
    errors = collect_errors(page)
    page.goto(path.resolve().as_uri(), wait_until="load")
    for slider in ("reference-slider", "rule-slider", "demo-slider", "pair-slider"):
        frames = page.eval_on_selector(f"#{slider}", "e => Number(e.max) + 1")
        assert frames > 1, f"{slider} has {frames} frame(s)"
        print(f"  {slider}: {frames} frames")
    page.click("#pair-play")
    page.wait_for_timeout(600)
    label = page.text_content("#pair-label") or ""
    assert label.startswith("t = "), f"unexpected pair label {label!r}"
    page.click(
        "#rule-slider"
    )  # any frame of the rule diagram must have drawn something
    for canvas in (
        "reference-canvas",
        "rule-canvas",
        "demo-canvas",
        "pair-reference-canvas",
        "pair-graph-canvas",
    ):
        print(f"  {canvas}: {painted(page, canvas)} painted pixels")
    for frame in (
        1,
        2,
        3,
    ):  # match, choose, rewrite phases of the first rule application
        page.evaluate(
            "(f) => { const s = document.getElementById('rule-slider'); s.value = f;"
            " s.dispatchEvent(new Event('input')); }",
            frame,
        )
        page.wait_for_timeout(100)
        page.locator("#rule-canvas").screenshot(
            path=str(out / f"rule-phase-{frame}.png")
        )
    assert page.locator("table tbody tr").count() >= 3, (
        "check table has fewer than 3 rows"
    )
    assert page.locator("svg").count() >= 4, "fewer than 4 SVG plots"
    assert page.locator("svg.pattern").count() >= 1, "no rule pattern drawing"
    page.screenshot(path=str(out / "report.png"), full_page=True)
    check_phone_layout(page, out / "report-phone.png")
    return errors


def check_phone_layout(page: Page, screenshot: Path) -> None:
    """At phone width the page must not scroll sideways; saves a screenshot of the top."""
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    scroll_width, inner_width = page.evaluate(
        "() => [document.documentElement.scrollWidth, window.innerWidth]"
    )
    assert scroll_width <= inner_width + 1, (
        f"page overflows sideways at phone width: {scroll_width}px > {inner_width}px"
    )
    page.screenshot(path=str(screenshot), full_page=False)
    print(f"  phone width ok ({scroll_width}px <= {inner_width}px)")
    page.set_viewport_size({"width": 1200, "height": 900})


def check_gas_report(page: Page, path: Path, out: Path) -> list[str]:
    print(f"gas report: {path}")
    errors = collect_errors(page)
    page.goto(path.resolve().as_uri(), wait_until="load")
    frames = page.eval_on_selector("#gas-slider", "e => Number(e.max) + 1")
    assert frames > 1, f"gas-slider has {frames} frame(s)"
    page.click("#gas-play")
    page.wait_for_timeout(500)
    print(
        f"  gas-canvas: {frames} frames, {painted(page, 'gas-canvas')} painted pixels"
    )
    assert page.locator("table tbody tr").count() >= 4, (
        "check table has fewer than 4 rows"
    )
    assert page.locator("svg").count() >= 5, "fewer than 5 SVG plots"
    page.screenshot(path=str(out / "gas.png"), full_page=True)
    check_phone_layout(page, out / "gas-phone.png")
    return errors


def check_playground_page(page: Page, path: Path, out: Path, tag: str) -> list[str]:
    """A saved playground result page (from the CLI or from the GUI's *Save result page*)."""
    print(f"playground page: {path}")
    errors = collect_errors(page)
    page.goto(path.resolve().as_uri(), wait_until="load")
    for slider in ("rule-slider", "demo-slider"):
        frames = page.eval_on_selector(f"#{slider}", "e => Number(e.max) + 1")
        assert frames > 1, f"{slider} has {frames} frame(s)"
        print(f"  {slider}: {frames} frames")
    page.evaluate(
        "() => { const s = document.getElementById('rule-slider'); s.value = 2;"
        " s.dispatchEvent(new Event('input')); }"
    )
    page.wait_for_timeout(100)
    for canvas in ("universe-canvas", "rule-canvas", "demo-canvas"):
        print(f"  {canvas}: {painted(page, canvas)} painted pixels")
    assert page.locator("svg.pattern").count() >= 1, "no rule pattern drawing"
    assert page.locator("table").count() >= 3, "fewer than 3 tables"
    assert page.locator("pre.request").count() == 1, "no session JSON to reproduce"
    page.screenshot(path=str(out / f"playground-{tag}.png"), full_page=True)
    check_phone_layout(page, out / f"playground-{tag}-phone.png")
    return errors


def wait_playground_done(page: Page, timeout_ms: int) -> None:
    page.wait_for_function(
        "/^(done|not run)/.test(document.getElementById('pg-status').textContent)",
        timeout=timeout_ms,
    )


def wait_error(page: Page, present: bool, timeout_ms: int) -> str:
    page.wait_for_function(
        "(present) => (document.getElementById('pg-error').textContent !== '') === present",
        arg=present,
        timeout=timeout_ms,
    )
    return page.text_content("#pg-error") or ""


def check_gui(page: Page, url: str, out: Path, timeout_ms: int) -> list[str]:
    print(f"gui: {url}")
    errors = collect_errors(page)
    goto_with_retries(page, url)
    page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
    page.reload(wait_until="load")
    page.wait_for_function(
        "document.getElementById('run').disabled === false", timeout=timeout_ms
    )
    print(f"  python loaded: {page.text_content('#status')}")
    # Playground: runs on load with the preset rule set (RandomHop ticked).
    wait_playground_done(page, timeout_ms)
    assert page.text_content("#pg-error") == "", page.text_content("#pg-error")
    assert page.locator("#pg-library tbody tr").count() == 7, "seven preset rules"
    assert page.locator("#pg-lhs circle.node").count() == 3, "w, s, t on the left"

    # Rule list: tick BiasedHop, untick RandomHop -> the rule set is the biased hop.
    page.click("#pg-library tr[data-name=BiasedHop] input.enabled")
    page.click("#pg-library tr[data-name=RandomHop] input.enabled")
    page.wait_for_function(
        "/^BiasedHop/.test(document.getElementById('pg-physics').textContent)",
        timeout=timeout_ms,
    )
    physics = page.text_content("#pg-physics") or ""
    assert "drift v = (0." in physics, physics
    page.wait_for_timeout(300)
    for canvas in ("pg-rule-canvas", "pg-demo-canvas", "pg-universe-canvas"):
        painted(page, canvas)
    assert page.locator("#pg-measured table tbody tr").count() == 7
    assert page.locator("#pg-rules-table svg.pattern").count() == 1

    # Two rules ticked: a sequence, described as one rule plus its parts.
    page.click("#pg-library tr[data-name=DoubleHop] input.enabled")
    page.wait_for_function(
        r"/^DoubleHop \+ BiasedHop/.test(document.getElementById('pg-physics').textContent)",
        timeout=timeout_ms,
    )
    assert page.locator("#pg-rules-table tbody tr").count() == 3
    assert "3 hops per step" in (page.text_content("#pg-physics") or "")
    page.click("#pg-library tr[data-name=DoubleHop] input.enabled")
    wait_playground_done(page, timeout_ms)

    # Graphical editor on the selected (BiasedHop) rule: removing the site edge {s,t} on the
    # left breaks the rule (the right side keeps it), adding it back repairs it.
    page.click("#pg-library tr[data-name=BiasedHop] td.notation")
    assert page.input_value("#pg-name") == "BiasedHop"
    page.click("#pg-lhs circle[data-node=s]")
    page.click("#pg-lhs circle[data-node=t]")
    message = wait_error(page, True, timeout_ms)
    assert "end on a site of the chain" in message, message
    assert page.input_value("#pg-notation") == "{w,s} -> {w,t}, {s,t}"
    page.click("#pg-lhs circle[data-node=s]")
    page.click("#pg-lhs circle[data-node=t]")
    wait_error(page, False, timeout_ms)
    assert page.input_value("#pg-notation") == "{w,s}, {s,t} -> {w,t}, {s,t}"
    # Add a site, connect it on both sides (a chain of two), then remove it again.
    page.click("#pg-add-site")
    page.click("#pg-lhs circle[data-node=t]")
    page.click("#pg-rhs circle[data-node=u]")
    page.click("#pg-rhs circle[data-node=t]")
    page.click("#pg-apply")
    wait_playground_done(page, timeout_ms)
    assert page.text_content("#pg-error") == "", page.text_content("#pg-error")
    assert (
        page.input_value("#pg-notation") == "{w,s}, {s,t}, {u,t} -> {w,t}, {s,t}, {u,t}"
    )
    assert "1 hop per application" in (page.text_content("#pg-physics") or "")
    page.click("#pg-lhs circle[data-node=u]")
    page.click("#pg-remove-node")
    page.click("#pg-apply")
    wait_playground_done(page, timeout_ms)
    assert page.input_value("#pg-notation") == "{w,s}, {s,t} -> {w,t}, {s,t}"
    print("  graphical editor ok")

    # Text editor: a bad rule is rejected with a reason, a good one restores the drawing.
    page.fill("#pg-notation", "{s,t} -> {s,t}")
    page.click("#pg-apply")
    message = wait_error(page, True, timeout_ms)
    print(f"  playground rejects a bad rule: {message}")
    page.fill("#pg-notation", "{w,s}, {s,t} -> {w,t}, {s,t}")
    page.click("#pg-apply")
    wait_error(page, False, timeout_ms)
    assert page.locator("#pg-lhs circle.node").count() == 3

    # New rule, delete it again.
    page.click("#pg-new")
    wait_playground_done(page, timeout_ms)
    assert page.locator("#pg-library tbody tr").count() == 8
    page.fill("#pg-description", "a fresh rule")
    assert "a fresh rule" in (
        page.text_content("#pg-library tr.selected td.description") or ""
    )
    page.click("#pg-delete")
    wait_playground_done(page, timeout_ms)
    assert page.locator("#pg-library tbody tr").count() == 7

    # Walker models and collisions: HPP collisions need hyperedge walkers; with them, Stream
    # and Reflect (built-in, not editable) plus hpp is the lattice gas: arrows, rings, Collide.
    page.select_option("select[name=collisions]", "hpp")
    message = wait_error(page, True, timeout_ms)
    assert "hyperedge" in message, message
    page.select_option("select[name=walker_model]", "hyperedge")
    page.select_option("select[name=placement]", "left")
    page.fill("input[name=walkers]", "10")
    page.click("#pg-library tr[data-name=Stream] input.enabled")
    page.click("#pg-library tr[data-name=Reflect] input.enabled")
    page.click("#pg-library tr[data-name=BiasedHop] input.enabled")
    page.click("#pg-apply")
    wait_error(page, False, timeout_ms)
    wait_playground_done(page, timeout_ms)
    assert (
        page.locator("#pg-rules-table tbody tr").count() == 4
    )  # Collide, Stream, Reflect, set
    first = page.text_content("#pg-rules-table tbody tr:first-child th") or ""
    assert first.startswith("Collide"), first
    assert "exclusion principle" in (page.text_content("#pg-docs") or "")
    assert "blocked hops" in (page.text_content("#pg-measured") or "")
    page.click("#pg-library tr[data-name=Stream] td.notation")
    assert (
        page.is_disabled("#pg-add-site")
        and page.get_attribute("#pg-notation", "readonly") is not None
    )
    page.wait_for_timeout(300)
    painted(page, "pg-demo-canvas")
    page.screenshot(path=str(out / "gui-lattice-gas.png"), full_page=False)
    print("  hyperedge walkers with hpp collisions ok")
    # Back to the Brownian setting for the rest of the check.
    page.click("#pg-library tr[data-name=Stream] input.enabled")
    page.click("#pg-library tr[data-name=Reflect] input.enabled")
    page.click("#pg-library tr[data-name=BiasedHop] input.enabled")
    page.select_option("select[name=collisions]", "none")
    page.select_option("select[name=walker_model]", "edge")
    page.select_option("select[name=placement]", "center")
    page.fill("input[name=walkers]", "4")
    page.click("#pg-library tr[data-name=BiasedHop] td.notation")
    page.click("#pg-apply")
    wait_playground_done(page, timeout_ms)
    assert page.text_content("#pg-error") == "", page.text_content("#pg-error")

    # Universe: periodic walls, then a walker placed by clicking the corner site (0, 0).
    page.select_option("select[name=walls]", "periodic")
    wait_playground_done(page, timeout_ms)
    assert "wall rate" in (page.text_content("#pg-measured") or "")
    canvas = page.locator("#pg-universe-canvas")
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    assert box is not None
    size = int(page.input_value("input[name=size]"))
    scale = (400 - 40) / size  # Viewport(canvas 400px, margin 20) over size cells
    px, py = 20 + 0.5 * scale, 400 - 20 - 0.5 * scale
    page.mouse.click(
        box["x"] + px * box["width"] / 400, box["y"] + py * box["height"] / 400
    )
    wait_playground_done(page, timeout_ms)
    assert page.input_value("select[name=placement]") == "sites"
    label = page.text_content("#pg-universe-label") or ""
    assert label.startswith("5 edge walkers"), label
    print(f"  universe: {label}")

    # Files: the result page, the session and the rule set.
    with page.expect_download() as download:
        page.click("#pg-save")
    saved = download.value
    assert saved.suggested_filename.startswith("playground_"), saved.suggested_filename
    assert saved.suggested_filename.endswith("_BiasedHop.html"), (
        saved.suggested_filename
    )
    saved_path = out / "gui-saved-playground.html"
    saved.save_as(str(saved_path))
    print(f"  saved result page -> {saved.suggested_filename}")
    with page.expect_download() as download:
        page.click("#pg-export-session")
    assert download.value.suggested_filename == "playground-session.json"
    page.click("#pg-library tr[data-name=BiasedHop] td.notation")  # select it again
    with page.expect_download() as download:
        page.click("#pg-export")
    assert download.value.suggested_filename == "BiasedHop.rule.json"
    page.click("#pg-use")
    assert page.input_value("#run-rule-json").startswith("[{")
    page.screenshot(path=str(out / "gui-playground.png"), full_page=False)

    page.fill("input[name=particles]", "400")
    page.fill("#params input[name=steps]", "1000")
    page.fill("input[name=record_every]", "50")
    page.click("#run")
    page.wait_for_function(
        "/^(done|error)/.test(document.getElementById('status').textContent)",
        timeout=timeout_ms,
    )
    status = page.text_content("#status") or ""
    print(f"  run: {status}")
    assert status.startswith("done"), status
    assert page.is_visible("#results"), "results are hidden"
    verdict = page.text_content("#verdict")
    print(f"  verdict: {verdict} (a biased rule is meant to fail)")
    assert page.locator("#checks-table table tbody tr").count() == 3
    page.wait_for_timeout(400)
    painted(page, "pair-graph-canvas")
    page.screenshot(path=str(out / "gui.png"), full_page=True)
    check_phone_layout(page, out / "gui-phone.png")
    errors += check_playground_page(page, saved_path, out, "gui")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="report page to check (default: the newest output/brownian_vs_graph_*.html)",
    )
    parser.add_argument("--gui", default="http://web:8000/")
    parser.add_argument("--out", type=Path, default=Path("output/screenshots"))
    parser.add_argument(
        "--timeout", type=int, default=300_000, help="GUI timeout in ms"
    )
    parser.add_argument("--skip-gui", action="store_true")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="build web/dist and serve web/ from this process instead of using --gui",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.serve and not args.skip_gui:
        args.gui = serve_gui()
    if args.report is None:
        candidates = sorted(Path("output").glob("brownian_vs_graph_*.html"))
        args.report = (
            candidates[-1] if candidates else Path("output/brownian_vs_graph_none.html")
        )

    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            if args.report.exists():
                failures += check_report(page, args.report, args.out)
            else:
                failures.append(
                    f"report {args.report} does not exist (run python -m imigraph report)"
                )
            gas_pages = sorted(Path("output").glob("gas_*.html"))
            if gas_pages:
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                failures += check_gas_report(page, gas_pages[-1], args.out)
            playground_pages = sorted(Path("output").glob("playground_*.html"))
            if playground_pages:
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                failures += check_playground_page(
                    page, playground_pages[-1], args.out, "cli"
                )
            if not args.skip_gui:
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                failures += check_gui(page, args.gui, args.out, args.timeout)
            gui_pages = sorted(Path("output").glob("rule_editor_*.html"))
            if gui_pages and not args.skip_gui:
                # The self-contained GUI page (python -m imigraph gui-page), opened from file://
                out = args.out / "rule-editor"
                out.mkdir(parents=True, exist_ok=True)
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                failures += check_gui(
                    page, gui_pages[-1].resolve().as_uri(), out, args.timeout
                )
        finally:
            browser.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"OK, screenshots in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
