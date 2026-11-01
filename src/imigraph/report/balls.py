"""Page content of the ball gas against the hard-disk gas: the data blob for the canvases and
the SVG fragments (checks tables, charts), in the shape the gas page uses."""

from __future__ import annotations

from typing import Any

import numpy as np

from imigraph.experiments.balls import BallComparison
from imigraph.report import svg
from imigraph.report.gas import _config_table
from imigraph.report.html import (
    _checks_table,
    _occupancy_charts,
    _rounded,
    _rules_table,
)

BALL_CONFIG_LABELS = {
    "size": "room: sites per side",
    "balls": "balls",
    "radius": "ball radius (cells)",
    "max_speed": "largest velocity component (cells per step)",
    "rule": "collision rule set",
    "rules": "rules, in order",
    "placement": "placement",
    "spacing": "lattice spacing a (= 1 / size)",
    "step_time": "step time tau (= a: one cell per step is speed 1)",
    "disk_radius": "disk radius of the reference (covers a ball's cells)",
    "steps": "steps",
    "total_time": "total time",
    "seed": "seed",
    "entropy_bins": "entropy grid (bins per side)",
    "substeps": "reference time steps per lattice step",
}
FALLBACK_COLORS = ("#f59e0b", "#7c3aed", "#0891b2", "#be185d")


def _ball_config_table(config: Any) -> str:
    from html import escape

    values = config.to_dict()
    rows = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(f'{values[key]:.4g}' if isinstance(values[key], float) else str(values[key]))}</td></tr>"
        for key, label in BALL_CONFIG_LABELS.items()
        if key in values
    )
    return f'<table class="kv"><tbody>{rows}</tbody></table>'


def ball_page_data(comparison: BallComparison) -> dict[str, Any]:
    gas, balls = comparison.gas, comparison.balls
    run = balls.run
    return {
        "graph": {
            **run.to_dict(balls.config.entropy_bins),
            "config": balls.config.to_dict(),
            "report": balls.report.to_dict(),
            "informative": [check.to_dict() for check in balls.informative],
            "invariants": list(balls.invariants),
            "entropy": balls.entropy.tolist(),
            "radius_cells": balls.config.radius,
        },
        "reference": {
            "config": gas.config.to_dict(),
            "times": _rounded(gas.run.times, 5),
            "positions": _rounded(gas.run.trajectory.positions, 5),
            "velocities": _rounded(gas.run.velocities, 5),
            "box": gas.run.trajectory.box.to_dict(),
            "radius": gas.config.radius,
            "report": gas.report.to_dict(),
            "informative": [check.to_dict() for check in gas.informative],
            "energy": gas.run.kinetic_energy().tolist(),
            "momentum": gas.run.momentum().tolist(),
            "entropy": gas.entropy.tolist(),
            "collisions": gas.run.collisions.tolist(),
        },
        "comparison": comparison.report.to_dict(),
        "passed": comparison.passed,
    }


def ball_fragments(comparison: BallComparison) -> dict[str, str]:
    gas, balls = comparison.gas, comparison.balls
    labels = ("hard-disk gas (reference)", "ball gas on the graph")
    colors = (svg.REFERENCE_COLOR, svg.CANDIDATE_COLOR)
    t = gas.run.times
    graph_run = balls.run.to_gas_run()
    energies = (gas.run.kinetic_energy(), graph_run.kinetic_energy())
    momenta = (gas.run.momentum(), graph_run.momentum())
    n = gas.run.trajectory.num_particles
    bins = balls.config.entropy_bins
    sigma = max(gas.run.thermal_sigma, 1e-9)
    edges = np.linspace(0.0, 4 * sigma, 17)
    width = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2
    from scipy import stats

    rayleigh = stats.rayleigh(scale=sigma)
    gas_final, _ = np.histogram(gas.run.speeds(-1), edges)
    graph_final, _ = np.histogram(graph_run.speeds(-1), edges)
    invariants = "".join(
        f'<tr><th>{row["name"]}</th><td><span class="badge {"pass" if row["passed"] else "fail"}">'
        f"{'PASS' if row['passed'] else 'FAIL'}</span></td><td>{row['detail']}</td></tr>"
        for row in balls.invariants
    )
    by_rule = balls.run.collisions_by_rule()
    collision_series = [
        svg.Series(labels[0], t[1:], gas.run.collisions, colors[0]),
        svg.Series(labels[1], t[1:], balls.run.collisions(), colors[1]),
    ] + [
        svg.Series(
            f"graph: by {name}",
            t[1:],
            counts,
            FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
            dashed=True,
        )
        for i, (name, counts) in enumerate(by_rule.items())
        if len(by_rule) > 1
    ]
    return {
        "graph_checks_table": _checks_table(balls.report, balls.informative),
        "graph_invariants_table": '<div class="table-scroll"><table><thead><tr><th>lattice '
        "invariant</th><th>verdict</th><th>evidence</th></tr></thead><tbody>"
        + invariants
        + "</tbody></table></div>",
        "reference_checks_table": _checks_table(gas.report, gas.informative),
        "comparison_table": _checks_table(comparison.report),
        "occupancy_charts": _occupancy_charts(comparison.report),
        "graph_config_table": _ball_config_table(balls.config),
        "reference_config_table": _config_table(gas.config),
        "rules_table": _rules_table(list(balls.rules)),
        "energy_chart": svg.line_chart(
            [
                svg.Series(label, t, energy, color)
                for label, energy, color in zip(labels, energies, colors, strict=True)
            ],
            title="Total kinetic energy (reference: to rounding; graph: exactly)",
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
            title="Total momentum (changes only at the walls)",
            xlabel="time",
            ylabel="momentum",
        ),
        "entropy_chart": svg.line_chart(
            [
                svg.Series(labels[0], t, gas.entropy, colors[0]),
                svg.Series(labels[1], t, balls.entropy, colors[1]),
                svg.Series(
                    f"uniform, ln({bins}x{bins})",
                    t,
                    np.full(t.shape, balls.max_entropy),
                    svg.THEORY_COLOR,
                    dashed=True,
                ),
            ],
            title="Occupancy entropy of the ball centres (nats)",
            xlabel="time",
            ylabel="entropy",
        ),
        "collisions_chart": svg.line_chart(
            collision_series,
            title="Collisions per recording interval",
            xlabel="time",
            ylabel="collisions",
        ),
        "speed_chart": svg.histogram_chart(
            edges,
            [
                (f"{labels[0]} at t = {t[-1]:.3g}", gas_final, colors[0]),
                (f"{labels[1]} at t = {t[-1]:.3g}", graph_final, colors[1]),
            ],
            title="Final speed distributions (integer speeds on the graph)",
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
