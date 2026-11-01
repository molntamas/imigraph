"""A self-contained HTML page with animations and plots of one experiment.

The page embeds its data as JSON and the small visualisation script inline, uses no network
resources and opens from the file system.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from imigraph.compare import CheckResult, ComparisonReport
from imigraph.experiments import ExperimentConfig, ExperimentResult, run_experiment
from imigraph.graph import (
    GraphHistory,
    GraphSimulation,
    RandomHop,
    Rule,
    SquareLattice,
    walkers_at_node,
)
from imigraph.report import svg
from imigraph.trajectory import Trajectory

REPORT_STEM = "brownian_vs_graph"
INDEX_FILENAME = "index.html"


def _asset(name: str) -> str:
    return (
        resources.files("imigraph.report.assets")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


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
        "paths": None if demo.paths is None else demo.paths.tolist(),
        "velocities": None if demo.velocities is None else demo.velocities.tolist(),
        "events": None if demo.events is None else [list(step) for step in demo.events],
        "radius": demo.radius,
        "space": {
            "coordinates": _rounded(space.coordinates),
            "edges": space.edges().tolist(),
            "box": space.box.to_dict(),
            "spacing": float(space.cell_size[0]),
            "neighbors": space.neighbors.tolist(),
        },
    }


def default_rule_demo(
    spacing: float = 1.0,
    seed: int = 2,
    steps: int = 24,
    rule: Rule | None = None,
    size: int = 5,
    walkers: int = 1,
) -> GraphHistory:
    """One walker on a 5x5 lattice, recorded at every step with the matched chains, for the
    rule diagram (``rule``, ``size`` and ``walkers`` are the playground's knobs)."""
    lattice = SquareLattice(size, spacing)
    simulation = GraphSimulation(lattice, rule or RandomHop(), step_time=1.0)
    start = lattice.node_index(3, 2) if size == 5 else lattice.center_node
    initial = walkers_at_node(lattice, start, walkers)
    return simulation.run(
        initial, steps, np.random.default_rng(seed), record_paths=True
    )


def describe_rule(
    rule: Rule, spacing: float = 1.0, step_time: float = 1.0
) -> list[dict[str, str]]:
    """Human-readable descriptions of a rule (see ``RandomHop.describe``): one entry per rule
    of a :class:`~imigraph.graph.rules.RuleSequence` plus the sequence as a whole, each with a
    ``pattern`` drawing (SVG) when the rule is given as a :class:`RuleSpec`."""
    describe_all = getattr(rule, "descriptions", None)
    if describe_all is not None:
        described = describe_all(spacing, step_time)
    elif getattr(rule, "describe", None) is not None:
        described = [rule.describe(spacing, step_time)]
    else:
        keys = ("rewrite", "matches", "choice", "invariants", "physics")
        described = [{"name": type(rule).__name__, **dict.fromkeys(keys, "")}]
    specs = list(getattr(rule, "specs", ()) or [])
    for entry, spec in zip(described, specs, strict=False):
        entry["pattern"] = svg.rule_pattern(
            spec.lhs, spec.rhs, spec.type_of, spec.lhs_ops, spec.rhs_ops
        )
    return described


def rule_descriptions(result: ExperimentResult) -> list[dict[str, str]]:
    """The rules of the experiment, described (see :func:`describe_rule`)."""
    rule = result.config.graph_system.rule
    return describe_rule(rule, result.config.spacing, result.config.step_time)


def _rules_table(rules: list[dict[str, str]]) -> str:
    rows = []
    for rule in rules:
        description = rule.get("description", "")
        name = escape(rule["name"]) + (
            f'<br><span class="muted description">{escape(description)}</span>'
            if description
            else ""
        )
        rows.append(
            f"<tr><th>{name}</th><td><code>{escape(rule['rewrite'])}</code></td>"
            f"<td>{escape(rule['matches'])}</td><td>{escape(rule['choice'])}</td>"
            f"<td>{escape(rule['invariants'])}</td><td>{escape(rule['physics'])}</td>"
            f'<td class="pattern">{rule.get("pattern", "")}</td></tr>'
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>rule</th><th>rewrite</th>'
        "<th>matches</th><th>choice</th><th>invariants</th><th>physics</th><th>pattern</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _msd_chart(report: ComparisonReport, config: ExperimentConfig) -> str:
    data = report["msd"].data
    times = data["times"]
    theory = 4 * config.diffusion * times
    limit = 1.5 * max(
        float(np.max(data["reference"])), float(np.max(data["candidate"]))
    )
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
        svg.Series(
            report.candidate_label, times, data["candidate"], svg.CANDIDATE_COLOR
        ),
        svg.Series(
            "free diffusion 4 D t",
            times[visible],
            theory[visible],
            svg.THEORY_COLOR,
            dashed=True,
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
    candidate_counts, _ = np.histogram(
        result.graph_projection.positions[frame, :, 0], edges
    )
    return svg.histogram_chart(
        edges,
        [
            (result.report.reference_label, reference_counts, svg.REFERENCE_COLOR),
            (result.report.candidate_label, candidate_counts, svg.CANDIDATE_COLOR),
        ],
        title=f"x marginal at t = {result.reference.times[frame]:.4g}",
        xlabel="x",
        width=520,
        height=316,
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


def _metric_charts(result: ExperimentResult) -> dict[str, str]:
    ref, cand = result.reference_metrics, result.graph_metrics
    labels = (result.report.reference_label, result.report.candidate_label)
    t = ref.interval_times
    energy = svg.line_chart(
        [
            svg.Series(labels[0], t, ref.energy, svg.REFERENCE_COLOR),
            svg.Series(labels[1], t, cand.energy, svg.CANDIDATE_COLOR),
            svg.Series(
                "free diffusion 2 D / dt",
                t,
                np.full(t.shape, ref.expected_energy),
                svg.THEORY_COLOR,
                dashed=True,
            ),
        ],
        title="Mean kinetic energy per particle (interval means of the simulation steps)",
        xlabel="time",
        ylabel="energy",
        zero_based=False,
    )
    band = (
        np.full(t.shape, -2 * ref.momentum_std),
        np.full(t.shape, 2 * ref.momentum_std),
    )
    momentum = svg.line_chart(
        [
            svg.Series(
                f"{labels[0]}, x", t, ref.momentum[:, 0], svg.REFERENCE_COLOR, band=band
            ),
            svg.Series(
                f"{labels[0]}, y",
                t,
                ref.momentum[:, 1],
                svg.REFERENCE_COLOR,
                dashed=True,
            ),
            svg.Series(f"{labels[1]}, x", t, cand.momentum[:, 0], svg.CANDIDATE_COLOR),
            svg.Series(
                f"{labels[1]}, y",
                t,
                cand.momentum[:, 1],
                svg.CANDIDATE_COLOR,
                dashed=True,
            ),
        ],
        title="Total momentum (interval means; band: +/- 2 standard deviations expected)",
        xlabel="time",
        ylabel="momentum",
    )
    bins = result.config.entropy_bins
    entropy = svg.line_chart(
        [
            svg.Series(labels[0], ref.times, ref.entropy, svg.REFERENCE_COLOR),
            svg.Series(labels[1], cand.times, cand.entropy, svg.CANDIDATE_COLOR),
            svg.Series(
                f"uniform, ln({bins}x{bins})",
                ref.times,
                np.full(ref.times.shape, ref.max_entropy),
                svg.THEORY_COLOR,
                dashed=True,
            ),
        ],
        title=f"Occupancy entropy on a {bins}x{bins} grid (nats)",
        xlabel="time",
        ylabel="entropy",
    )
    wall = svg.line_chart(
        [
            svg.Series(labels[0], t, ref.wall_rate, svg.REFERENCE_COLOR),
            svg.Series(labels[1], t, cand.wall_rate, svg.CANDIDATE_COLOR),
        ],
        title="Wall interaction rate (fraction of particles per step: reflected / self-loop hop)",
        xlabel="time",
        ylabel="fraction per step",
    )
    size = result.config.lattice_size
    occupied = svg.line_chart(
        [
            svg.Series(labels[0], ref.times, ref.occupied, svg.REFERENCE_COLOR),
            svg.Series(labels[1], cand.times, cand.occupied, svg.CANDIDATE_COLOR),
            svg.Series(
                f"uniform occupancy, {ref.num_cells} (1 - exp(-n/{ref.num_cells}))",
                ref.times,
                np.full(ref.times.shape, ref.expected_occupied),
                svg.THEORY_COLOR,
                dashed=True,
            ),
        ],
        title=f"Occupied cells of the {size}x{size} lattice grid",
        xlabel="time",
        ylabel="cells",
    )
    return {
        "energy_chart": energy,
        "momentum_chart": momentum,
        "entropy_chart": entropy,
        "wall_rate_chart": wall,
        "occupied_chart": occupied,
    }


def _checks_table(
    report: ComparisonReport, informative: Sequence[CheckResult] = ()
) -> str:
    """``informative`` checks are listed after the real ones with the verdict ``info``: they are
    reported for understanding but do not count (a lattice gas has no Maxwell distribution)."""
    rows = []
    for check, counts in [(c, True) for c in report.results] + [
        (c, False) for c in informative
    ]:
        if counts:
            verdict, css = ("pass", "pass") if check.passed else ("FAIL", "fail")
        else:
            verdict, css = (
                ("info: would pass" if check.passed else "info: would fail"),
                "muted",
            )
        relation = "&gt;" if check.larger_is_better else "&le;"
        rows.append(
            f'<tr><td><code>{escape(check.name)}</code></td><td class="{css}">{verdict}'
            f"</td><td>{check.value:.4g}</td><td>{relation} {check.threshold:.4g}</td>"
            f"<td>{escape(check.detail)}</td></tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>check</th><th>verdict</th><th>value'
        "</th><th>threshold</th><th>detail</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
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
    result: ExperimentResult,
    demo: GraphHistory,
    max_particles: int = 300,
    rule_demo: GraphHistory | None = None,
) -> dict[str, Any]:
    """The JSON-serialisable blob the animations are driven by (report page and browser GUI)."""
    rule_demo = default_rule_demo() if rule_demo is None else rule_demo
    return {
        "rule": _demo_blob(rule_demo),
        "rules": rule_descriptions(result),
        "config": result.config.to_dict(),
        "reference": _trajectory_blob(result.reference, max_particles),
        "graph": _trajectory_blob(result.graph_history.project(), max_particles),
        "demo": _demo_blob(demo),
        "report": result.report.to_dict(),
        "metrics": {
            "reference": result.reference_metrics.to_dict(),
            "graph": result.graph_metrics.to_dict(),
        },
    }


def fragments(result: ExperimentResult) -> dict[str, str]:
    """HTML fragments of the comparison section: the check table and the SVG plots."""
    report = result.report
    frames = report["marginals"].data["frames"]
    return {
        "rules_table": _rules_table(rule_descriptions(result)),
        "checks_table": _checks_table(report),
        "msd_chart": _msd_chart(report, result.config),
        "marginal_early": _marginal_chart(result, int(frames[0])),
        "marginal_final": _marginal_chart(result, int(frames[-1])),
        "occupancy_charts": _occupancy_charts(report),
        **_metric_charts(result),
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
        graph_model=_asset("graph-model.html"),
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
        data_json=json.dumps(
            page_data(result, demo, max_particles), separators=(",", ":")
        ),
        **fragments(result),
    )


def report_filename(when: datetime | None = None, label: str = "") -> str:
    """``brownian_vs_graph_<UTC timestamp>[_<label>].html``: a new file per run, old runs kept."""
    when = datetime.now(UTC) if when is None else when
    suffix = "_" + re.sub(r"[^A-Za-z0-9-]+", "-", label).strip("-") if label else ""
    return f"{REPORT_STEM}_{when:%Y-%m-%d_%H%M%S}{suffix}.html"


def write_index(directory: Path) -> Path:
    """``index.html`` listing every report in ``directory``, newest first, with its verdict."""
    rows = []
    for page in sorted(directory.glob("*.html"), reverse=True):
        if page.name == INDEX_FILENAME:
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        verdict = (
            "pass"
            if 'class="badge pass"' in text
            else "fail"
            if "badge fail" in text
            else ""
        )
        modified = datetime.fromtimestamp(page.stat().st_mtime, UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        size = page.stat().st_size / 1e6
        rows.append(
            f'<tr><td><a href="{escape(page.name)}">{escape(page.name)}</a></td>'
            f'<td class="{verdict}">{verdict.upper()}</td><td>{modified}</td>'
            f"<td>{size:.1f} MB</td></tr>"
        )
    body = (
        "<table><thead><tr><th>report</th><th>verdict</th><th>written</th><th>size</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
        if rows
        else "<p>No reports yet.</p>"
    )
    page = Template(_asset("index.html")).substitute(
        css=_asset("report.css"), body=body
    )
    index = directory / INDEX_FILENAME
    index.write_text(page, encoding="utf-8")
    return index


def write_report(
    path: str | Path,
    config: ExperimentConfig | None = None,
    result: ExperimentResult | None = None,
    label: str = "",
) -> Path:
    """Run the experiment (unless ``result`` is given) and write the page.

    ``path`` is normally a directory: a new timestamped file (:func:`report_filename`) is
    created in it (``label`` names the change, e.g. ``"metrics"``) and ``index.html`` there is
    refreshed, so the history of earlier runs is kept. A path ending in ``.html`` writes exactly
    that file (and refreshes the index of its directory).
    """
    result = run_experiment(config) if result is None else result
    path = Path(path)
    if path.suffix.lower() != ".html":
        path = path / report_filename(label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(result), encoding="utf-8")
    write_index(path.parent)
    return path
