"""A self-contained HTML page with animations and plots of one experiment.

The page embeds its data as JSON and the small visualisation script inline, uses no network
resources and opens from the file system.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from imigraph.compare import ComparisonReport
from imigraph.experiments import ExperimentConfig, ExperimentResult, run_experiment
from imigraph.graph import GraphHistory, GraphSimulation, SquareLattice, walkers_at_node
from imigraph.report import svg
from imigraph.trajectory import Trajectory

DEFAULT_FILENAME = "brownian_vs_graph.html"


def _asset(name: str) -> str:
    return resources.files("imigraph.report.assets").joinpath(name).read_text(encoding="utf-8")


def _rounded(array: np.ndarray, decimals: int = 3) -> list:
    return np.round(np.asarray(array, dtype=float), decimals).tolist()


def default_demo(
    spacing: float = 1.0, seed: int = 1, walkers: int = 6, steps: int = 60
) -> GraphHistory:
    """A small lattice run recorded at every step, for the graph-structure animation."""
    lattice = SquareLattice(9, spacing)
    simulation = GraphSimulation(lattice, step_time=1.0)
    initial = walkers_at_node(lattice, lattice.center_node, walkers)
    return simulation.run(initial, steps, np.random.default_rng(seed))


def _trajectory_blob(trajectory: Trajectory, max_particles: int) -> dict[str, Any]:
    return {
        "times": _rounded(trajectory.times, 4),
        "positions": _rounded(trajectory.positions[:, :max_particles]),
        "box": trajectory.box.to_dict(),
    }


def _demo_blob(demo: GraphHistory) -> dict[str, Any]:
    space = demo.space
    return {
        "times": _rounded(demo.times, 4),
        "anchors": demo.anchors.tolist(),
        "space": {
            "coordinates": _rounded(space.coordinates),
            "edges": space.edges().tolist(),
            "box": space.box.to_dict(),
            "spacing": float(space.cell_size[0]),
        },
    }


def _msd_chart(report: ComparisonReport, config: ExperimentConfig) -> str:
    data = report["msd"].data
    times = data["times"]
    theory = 4 * config.diffusion * times
    limit = 1.5 * max(float(np.max(data["reference"])), float(np.max(data["candidate"])))
    visible = theory <= limit
    series = [
        svg.Series(
            report.reference_label,
            times,
            data["reference"],
            svg.REFERENCE_COLOR,
            band=(
                data["reference"] - 2 * data["reference_se"],
                data["reference"] + 2 * data["reference_se"],
            ),
        ),
        svg.Series(report.candidate_label, times, data["candidate"], svg.CANDIDATE_COLOR),
        svg.Series(
            "free diffusion 4 D t", times[visible], theory[visible], svg.THEORY_COLOR, dashed=True
        ),
    ]
    return svg.line_chart(
        series,
        title="Mean squared displacement (band: reference +/- 2 standard errors)",
        xlabel="time",
        ylabel="MSD",
    )


def _marginal_chart(result: ExperimentResult, frame: int) -> str:
    box = result.reference.box
    edges = np.linspace(box.lower[0], box.upper[0], 41)
    reference_counts, _ = np.histogram(result.reference.positions[frame, :, 0], edges)
    candidate_counts, _ = np.histogram(result.graph_projection.positions[frame, :, 0], edges)
    return svg.histogram_chart(
        edges,
        [
            (result.report.reference_label, reference_counts, svg.REFERENCE_COLOR),
            (result.report.candidate_label, candidate_counts, svg.CANDIDATE_COLOR),
        ],
        title=f"x marginal at t = {result.reference.times[frame]:.4g}",
        xlabel="x",
        width=520,
        height=280,
    )


def _occupancy_charts(report: ComparisonReport) -> str:
    data = report["occupancy"].data
    blocks = []
    for label, key, color, css in (
        (report.reference_label, "reference", svg.REFERENCE_COLOR, "legend-ref"),
        (report.candidate_label, "candidate", svg.CANDIDATE_COLOR, "legend-cand"),
    ):
        blocks.append(
            f'<div class="heatmap-block"><p class="muted {css}">{escape(label)}</p>'
            f"{svg.heatmap(data[key], color)}</div>"
        )
    return "\n".join(blocks)


def _checks_table(report: ComparisonReport) -> str:
    rows = []
    for check in report.results:
        verdict = "pass" if check.passed else "FAIL"
        relation = "&gt;" if check.larger_is_better else "&le;"
        rows.append(
            f'<tr><td><code>{escape(check.name)}</code></td><td class="{verdict.lower()}">{verdict}'
            f"</td><td>{check.value:.4g}</td><td>{relation} {check.threshold:.4g}</td>"
            f"<td>{escape(check.detail)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>check</th><th>verdict</th><th>value</th><th>threshold</th>"
        "<th>detail</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _config_table(config: ExperimentConfig) -> str:
    labels = {
        "lattice_size": "lattice nodes per side",
        "spacing": "lattice spacing a",
        "step_time": "rule step time tau (= reference dt)",
        "diffusion": "diffusion coefficient D = a^2 / (4 tau)",
        "particles": "particles / walkers",
        "steps": "steps",
        "record_every": "record every (steps)",
        "total_time": "total time",
        "seed": "seed",
    }
    values = config.to_dict()
    rows = []
    for key, label in labels.items():
        value = values[key]
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(text)}</td></tr>")
    rows = "".join(rows)
    return f"<table><tbody>{rows}</tbody></table>"


def page_data(
    result: ExperimentResult, demo: GraphHistory, max_particles: int = 300
) -> dict[str, Any]:
    """The JSON-serialisable blob the animations are driven by (report page and browser GUI)."""
    return {
        "config": result.config.to_dict(),
        "reference": _trajectory_blob(result.reference, max_particles),
        "graph": _trajectory_blob(result.graph_history.project(), max_particles),
        "demo": _demo_blob(demo),
        "report": result.report.to_dict(),
    }


def fragments(result: ExperimentResult) -> dict[str, str]:
    """HTML fragments of the comparison section: the check table and the SVG plots."""
    report = result.report
    frames = report["marginals"].data["frames"]
    return {
        "checks_table": _checks_table(report),
        "msd_chart": _msd_chart(report, result.config),
        "marginal_early": _marginal_chart(result, int(frames[0])),
        "marginal_final": _marginal_chart(result, int(frames[-1])),
        "occupancy_charts": _occupancy_charts(report),
    }


def build_report(
    result: ExperimentResult, demo: GraphHistory | None = None, max_particles: int = 300
) -> str:
    """Render the HTML page for an experiment result."""
    demo = default_demo() if demo is None else demo
    report = result.report
    verdict = "pass" if report.passed else "fail"
    template = Template(_asset("report.html"))
    return template.substitute(
        css=_asset("report.css"),
        viz_js=_asset("viz.js"),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        reference_label=escape(report.reference_label),
        candidate_label=escape(report.candidate_label),
        verdict=verdict,
        verdict_upper=verdict.upper(),
        config_table=_config_table(result.config),
        particles=result.config.particles,
        lattice_size=result.config.lattice_size,
        shown=min(max_particles, result.config.particles),
        demo_nodes=demo.space.num_nodes,
        demo_walkers=demo.num_walkers,
        num_marginal_times=len(report["marginals"].data["frames"]),
        occupancy_bins=report["occupancy"].data["reference"].shape[0],
        data_json=json.dumps(page_data(result, demo, max_particles), separators=(",", ":")),
        **fragments(result),
    )


def write_report(
    path: str | Path, config: ExperimentConfig | None = None, result: ExperimentResult | None = None
) -> Path:
    """Run the experiment (unless ``result`` is given) and write the page to ``path``.

    ``path`` may be a directory, in which case :data:`DEFAULT_FILENAME` is used.
    """
    result = run_experiment(config) if result is None else result
    path = Path(path)
    if path.suffix.lower() != ".html":
        path = path / DEFAULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(result), encoding="utf-8")
    return path
