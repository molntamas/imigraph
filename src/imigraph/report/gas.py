"""Self-contained HTML page of a hard-disk gas run (energy conservation, Maxwell-Boltzmann)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from imigraph.experiments.gas import GasConfig, GasResult, run_gas
from imigraph.report import svg
from imigraph.report.html import _asset, _checks_table, _rounded, report_filename, write_index

GAS_STEM = "gas"


def gas_page_data(result: GasResult) -> dict[str, Any]:
    run = result.run
    return {
        "config": result.config.to_dict(),
        "radius": result.config.radius,
        "trajectory": {
            "times": _rounded(run.times, 4),
            "positions": _rounded(run.trajectory.positions, 4),
            "box": run.trajectory.box.to_dict(),
        },
        "report": result.report.to_dict(),
        "metrics": {
            "energy": run.kinetic_energy().tolist(),
            "momentum": run.momentum().tolist(),
            "entropy": result.entropy.tolist(),
            "collisions": run.collisions.tolist(),
            "sigma": run.thermal_sigma,
        },
    }


def _config_table(config: GasConfig) -> str:
    labels = {
        "particles": "disks",
        "radius": "disk radius",
        "packing_fraction": "packing fraction",
        "speed": "initial speed of every disk",
        "dt": "time step",
        "steps": "steps",
        "record_every": "record every (steps)",
        "total_time": "total time",
        "boundary": "walls",
        "seed": "seed",
        "entropy_bins": "entropy grid (bins per side)",
    }
    values = config.to_dict()
    rows = []
    for key, label in labels.items():
        value = values[key]
        text = f"{value:.4g}" if isinstance(value, float) else str(value)
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(text)}</td></tr>")
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def gas_fragments(result: GasResult) -> dict[str, str]:
    run = result.run
    t = run.times
    energy = run.kinetic_energy()
    momentum = run.momentum()
    scale = float(np.sqrt(2 * run.mass * energy[0] * run.trajectory.num_particles))
    sigma = run.thermal_sigma
    from scipy import stats

    rayleigh = stats.rayleigh(scale=sigma)
    edges = np.linspace(0.0, 4 * sigma, 33)
    width = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2
    n = run.trajectory.num_particles
    initial, _ = np.histogram(run.speeds(0), edges)
    final, _ = np.histogram(run.speeds(-1), edges)
    return {
        "checks_table": _checks_table(result.report),
        "config_table": _config_table(result.config),
        "energy_chart": svg.line_chart(
            [svg.Series("total kinetic energy", t, energy, svg.REFERENCE_COLOR)],
            title="Total kinetic energy (constant to rounding)",
            xlabel="time",
            ylabel="energy",
            zero_based=False,
        ),
        "momentum_chart": svg.line_chart(
            [
                svg.Series("total momentum, x", t, momentum[:, 0], svg.REFERENCE_COLOR),
                svg.Series("total momentum, y", t, momentum[:, 1], svg.CANDIDATE_COLOR),
                svg.Series(
                    "one particle's share, sqrt(2 m E / n)",
                    t,
                    np.full(t.shape, scale / n),
                    svg.THEORY_COLOR,
                    dashed=True,
                ),
            ],
            title="Total momentum (specular walls exchange momentum with the box)",
            xlabel="time",
            ylabel="momentum",
        ),
        "entropy_chart": svg.line_chart(
            [
                svg.Series("occupancy entropy", t, result.entropy, svg.REFERENCE_COLOR),
                svg.Series(
                    f"uniform, ln({result.config.entropy_bins}x{result.config.entropy_bins})",
                    t,
                    np.full(t.shape, result.max_entropy),
                    svg.THEORY_COLOR,
                    dashed=True,
                ),
            ],
            title="Occupancy entropy during the free expansion (nats)",
            xlabel="time",
            ylabel="entropy",
        ),
        "collisions_chart": svg.line_chart(
            [
                svg.Series(
                    "collisions per recording interval", t[1:], run.collisions, svg.CANDIDATE_COLOR
                )
            ],
            title="Disk-disk collisions per recording interval",
            xlabel="time",
            ylabel="collisions",
        ),
        "speed_chart": svg.histogram_chart(
            edges,
            [
                ("speeds at t = 0", initial, svg.CANDIDATE_COLOR),
                (f"speeds at t = {t[-1]:.4g}", final, svg.REFERENCE_COLOR),
            ],
            title="Speed distribution: from equal speeds to Maxwell-Boltzmann",
            xlabel="speed",
            curves=[
                svg.Series(
                    "Maxwell-Boltzmann (Rayleigh), sigma = sqrt(E / n m)",
                    centres,
                    n * width * rayleigh.pdf(centres),
                    svg.THEORY_COLOR,
                    dashed=True,
                )
            ],
        ),
    }


def build_gas_report(result: GasResult) -> str:
    verdict = "pass" if result.passed else "fail"
    return Template(_asset("gas.html")).substitute(
        css=_asset("report.css"),
        viz_js=_asset("viz.js"),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        verdict_upper=verdict.upper(),
        particles=result.config.particles,
        collisions=int(result.run.collisions.sum()),
        data_json=json.dumps(gas_page_data(result), separators=(",", ":")),
        **gas_fragments(result),
    )


def write_gas_report(
    path: str | Path,
    config: GasConfig | None = None,
    result: GasResult | None = None,
    label: str = "",
) -> Path:
    """Write a new timestamped ``gas_<time>[_<label>].html`` (older pages are kept) and refresh
    the index of the directory; a path ending in ``.html`` writes exactly that file."""
    result = run_gas(config) if result is None else result
    path = Path(path)
    if path.suffix.lower() != ".html":
        path = path / report_filename(label=label).replace("brownian_vs_graph", GAS_STEM, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_gas_report(result), encoding="utf-8")
    write_index(path.parent)
    return path
