"""Self-contained HTML page of the energy-conserving gases: the hard-disk gas next to the lattice
gas on the walker graph, each checked against kinetic theory and compared with each other."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from imigraph.experiments.gas import (
    GasComparison,
    GasConfig,
    GasResult,
    LatticeGasConfig,
    run_gas_comparison,
)
from imigraph.graph.gas import LatticeGas
from imigraph.graph.space import SquareLattice
from imigraph.report import svg
from imigraph.report.html import (
    _asset,
    _checks_table,
    _occupancy_charts,
    _rounded,
    _rules_table,
    report_filename,
    write_index,
)

GAS_STEM = "gas"


def default_gas_demo(steps: int = 36, seed: int = 3) -> dict[str, Any]:
    """A 7x7 lattice gas with walkers placed so that head-on collisions and wall reflections
    happen within the first steps (the rest of the walkers are random)."""
    lattice = SquareLattice(7)
    gas = LatticeGas(lattice, step_time=1.0)
    rng = np.random.default_rng(seed)
    sites = [
        lattice.node_index(1, 3),
        lattice.node_index(5, 3),
        lattice.node_index(3, 0),
        lattice.node_index(3, 6),
        lattice.node_index(0, 5),
        lattice.node_index(6, 1),
    ]
    directions = [0, 1, 2, 3, 0, 1]
    initial = gas.state(np.array(sites), np.array(directions))
    return gas.demo_dict(initial, steps, rng)


def _system_blob(result: GasResult) -> dict[str, Any]:
    run = result.run
    blob: dict[str, Any] = {
        "config": result.config.to_dict(),
        "times": _rounded(run.times, 4),
        "positions": _rounded(run.trajectory.positions, 4),
        "box": run.trajectory.box.to_dict(),
        "report": result.report.to_dict(),
        "informative": [check.to_dict() for check in result.informative],
        "energy": run.kinetic_energy().tolist(),
        "momentum": run.momentum().tolist(),
        "entropy": result.entropy.tolist(),
        "collisions": run.collisions.tolist(),
        "sigma": run.thermal_sigma,
    }
    if isinstance(result.config, LatticeGasConfig):
        blob["velocities"] = _rounded(run.velocities, 4)
        blob["spacing"] = result.config.spacing
        blob["rules"] = list(result.rules)
    else:
        blob["radius"] = result.config.radius
    return blob


def gas_page_data(
    comparison: GasComparison, demo: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "gas": _system_blob(comparison.gas),
        "lattice": _system_blob(comparison.lattice),
        "comparison": comparison.report.to_dict(),
        "demo": default_gas_demo() if demo is None else demo,
    }


def _config_table(config: GasConfig | LatticeGasConfig) -> str:
    if isinstance(config, LatticeGasConfig):
        labels = {
            "size": "lattice sites per side",
            "spacing": "lattice spacing a (= 1 / size)",
            "step_time": "step time tau (= a / speed)",
            "particles": "balls",
            "channel_fraction": "occupied fraction of the size^2 sites",
            "speed": "speed c = a / tau of every walker",
            "steps": "steps",
            "record_every": "record every (steps)",
            "total_time": "total time",
            "boundary": "walls",
            "seed": "seed",
            "entropy_bins": "entropy grid (bins per side)",
        }
    else:
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


def gas_fragments(comparison: GasComparison) -> dict[str, str]:
    gas, lattice = comparison.gas, comparison.lattice
    labels = ("hard-disk gas", "lattice gas on the graph")
    colors = (svg.REFERENCE_COLOR, svg.CANDIDATE_COLOR)
    t = gas.run.times
    energies = (gas.run.kinetic_energy(), lattice.run.kinetic_energy())
    momenta = (gas.run.momentum(), lattice.run.momentum())
    n = gas.run.trajectory.num_particles
    from scipy import stats

    sigma = gas.run.thermal_sigma
    rayleigh = stats.rayleigh(scale=sigma)
    edges = np.linspace(0.0, 4 * sigma, 33)
    width = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2
    gas_final, _ = np.histogram(gas.run.speeds(-1), edges)
    lattice_final, _ = np.histogram(lattice.run.speeds(-1), edges)
    bins = gas.config.entropy_bins
    return {
        "gas_checks_table": _checks_table(gas.report),
        "lattice_checks_table": _checks_table(lattice.report, lattice.informative),
        "comparison_table": _checks_table(comparison.report),
        "occupancy_charts": _occupancy_charts(comparison.report),
        "gas_config_table": _config_table(gas.config),
        "lattice_config_table": _config_table(lattice.config),
        "rules_table": _rules_table(list(lattice.rules)),
        "energy_chart": svg.line_chart(
            [
                svg.Series(label, t, energy, color)
                for label, energy, color in zip(labels, energies, colors, strict=True)
            ],
            title="Total kinetic energy (both constant: to rounding, and exactly)",
            xlabel="time",
            ylabel="energy",
            zero_based=False,
        ),
        "momentum_chart": svg.line_chart(
            [
                svg.Series(f"{labels[0]}, x", t, momenta[0][:, 0], colors[0]),
                svg.Series(
                    f"{labels[0]}, y", t, momenta[0][:, 1], colors[0], dashed=True
                ),
                svg.Series(f"{labels[1]}, x", t, momenta[1][:, 0], colors[1]),
                svg.Series(
                    f"{labels[1]}, y", t, momenta[1][:, 1], colors[1], dashed=True
                ),
            ],
            title="Total momentum (walls exchange momentum; exact with periodic walls)",
            xlabel="time",
            ylabel="momentum",
        ),
        "entropy_chart": svg.line_chart(
            [
                svg.Series(labels[0], t, gas.entropy, colors[0]),
                svg.Series(labels[1], t, lattice.entropy, colors[1]),
                svg.Series(
                    f"uniform, ln({bins}x{bins})",
                    t,
                    np.full(t.shape, gas.max_entropy),
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
                svg.Series(labels[0], t[1:], gas.run.collisions, colors[0]),
                svg.Series(labels[1], t[1:], lattice.run.collisions, colors[1]),
            ],
            title="Collisions per recording interval",
            xlabel="time",
            ylabel="collisions",
        ),
        "speed_chart": svg.histogram_chart(
            edges,
            [
                (f"{labels[0]} at t = {t[-1]:.4g}", gas_final, colors[0]),
                (
                    f"{labels[1]} at t = {t[-1]:.4g} (all at c)",
                    lattice_final,
                    colors[1],
                ),
            ],
            title="Final speed distributions: Maxwell-Boltzmann vs the single speed of HPP",
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


def build_gas_report(comparison: GasComparison) -> str:
    verdict = "pass" if comparison.passed else "fail"
    return Template(_asset("gas.html")).substitute(
        css=_asset("report.css"),
        viz_js=_asset("viz.js"),
        graph_model=_asset("graph-model.html"),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        verdict_upper=verdict.upper(),
        gas_particles=comparison.gas.config.particles,
        lattice_particles=comparison.lattice.config.particles,
        lattice_size=comparison.lattice.config.size,
        gas_collisions=int(comparison.gas.run.collisions.sum()),
        lattice_collisions=int(comparison.lattice.run.collisions.sum()),
        data_json=json.dumps(gas_page_data(comparison), separators=(",", ":")),
        **gas_fragments(comparison),
    )


def write_gas_report(
    path: str | Path,
    config: GasConfig | None = None,
    result: GasComparison | None = None,
    label: str = "",
    lattice_config: LatticeGasConfig | None = None,
) -> Path:
    """Write a new timestamped ``gas_<time>[_<label>].html`` (older pages are kept) and refresh
    the index of the directory; a path ending in ``.html`` writes exactly that file."""
    result = run_gas_comparison(config, lattice_config) if result is None else result
    path = Path(path)
    if path.suffix.lower() != ".html":
        path = path / report_filename(label=label).replace(
            "brownian_vs_graph", GAS_STEM, 1
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_gas_report(result), encoding="utf-8")
    write_index(path.parent)
    return path
