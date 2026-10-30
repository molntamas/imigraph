"""Headless browser check of the generated report and the Pyodide GUI (runs in the Playwright
container, see the `verify` service in compose.yaml).

    docker compose run --rm dev python -m imigraph report   # output/brownian_vs_graph_<time>.html
    docker compose --profile verify run --rm verify         # this script

It opens both pages in Chromium, fails on JavaScript errors, checks that the animations have
frames and draw onto the canvases, runs the GUI once with small parameters, and saves screenshots
to output/screenshots/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

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
            errors.append(f"console.error: {message.text}") if message.type == "error" else None
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
    page.click("#rule-slider")  # any frame of the rule diagram must have drawn something
    for canvas in (
        "reference-canvas",
        "rule-canvas",
        "demo-canvas",
        "pair-reference-canvas",
        "pair-graph-canvas",
    ):
        painted = page.evaluate(NONBLANK_PIXELS_JS, canvas)
        assert painted > 500, f"{canvas} looks blank ({painted} painted pixels)"
        print(f"  {canvas}: {painted} painted pixels")
    for frame in (1, 2, 3):  # match, choose, rewrite phases of the first rule application
        page.evaluate(
            "(f) => { const s = document.getElementById('rule-slider'); s.value = f;"
            " s.dispatchEvent(new Event('input')); }",
            frame,
        )
        page.wait_for_timeout(100)
        page.locator("#rule-canvas").screenshot(path=str(out / f"rule-phase-{frame}.png"))
    assert page.locator("table tbody tr").count() >= 3, "check table has fewer than 3 rows"
    assert page.locator("svg").count() >= 4, "fewer than 4 SVG plots"
    page.screenshot(path=str(out / "report.png"), full_page=True)
    return errors


def check_gui(page: Page, url: str, out: Path, timeout_ms: int) -> list[str]:
    print(f"gui: {url}")
    errors = collect_errors(page)
    goto_with_retries(page, url)
    page.wait_for_function("document.getElementById('run').disabled === false", timeout=timeout_ms)
    print(f"  python loaded: {page.text_content('#status')}")
    page.fill("input[name=particles]", "400")
    page.fill("input[name=steps]", "1000")
    page.fill("input[name=record_every]", "50")
    page.click("#run")
    page.wait_for_function(
        "/^(done|error)/.test(document.getElementById('status').textContent)", timeout=timeout_ms
    )
    status = page.text_content("#status") or ""
    print(f"  run: {status}")
    assert status.startswith("done"), status
    assert page.is_visible("#results"), "results are hidden"
    verdict = page.text_content("#verdict")
    print(f"  verdict: {verdict}")
    assert page.locator("#checks-table table tbody tr").count() == 3
    page.wait_for_timeout(400)
    painted = page.evaluate(NONBLANK_PIXELS_JS, "pair-graph-canvas")
    assert painted > 500, f"GUI canvas looks blank ({painted} painted pixels)"
    page.screenshot(path=str(out / "gui.png"), full_page=True)
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
    parser.add_argument("--timeout", type=int, default=300_000, help="GUI timeout in ms")
    parser.add_argument("--skip-gui", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report is None:
        candidates = sorted(Path("output").glob("brownian_vs_graph_*.html"))
        args.report = candidates[-1] if candidates else Path("output/brownian_vs_graph_none.html")

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
            if not args.skip_gui:
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                failures += check_gui(page, args.gui, args.out, args.timeout)
        finally:
            browser.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"OK, screenshots in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
