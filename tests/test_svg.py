"""The SVG helpers terminate and stay small for every input, including constant data."""

import time

import numpy as np

from imigraph.report import svg


def test_ticks_for_values_constant_to_rounding_terminate():
    # Regression: a y-range a few 1e-16 wide gave a tick step below float resolution, so the
    # loop never advanced and ate all memory (froze the Docker VM on 2026-10-30).
    start = time.perf_counter()
    ticks = svg._nice_ticks(15 - 1e-15, 15 + 1e-15)
    assert len(ticks) <= 100 and time.perf_counter() - start < 1.0
    assert svg._nice_ticks(0.0, 1.0) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert svg._nice_ticks(0.0, 0.0) == svg._nice_ticks(
        0.0, 1.0
    )  # empty range widened to 1


def test_line_chart_with_constant_series_fits_the_axis_around_the_value():
    t = np.arange(5.0)
    energy = 15.0 + np.array([0.0, 2e-15, -2e-15, 0.0, 2e-15])
    start = time.perf_counter()
    chart = svg.line_chart(
        [svg.Series("e", t, energy, svg.REFERENCE_COLOR)], zero_based=False
    )
    assert time.perf_counter() - start < 1.0
    assert chart.startswith("<svg") and len(chart) < 20_000
    assert (
        ">13.5<" in chart and ">16.5<" in chart
    )  # 10% margin around the constant value


def test_histogram_chart_with_curve_and_heatmap_render():
    edges = np.linspace(0.0, 1.0, 5)
    counts = np.array([1, 2, 3, 4])
    chart = svg.histogram_chart(
        edges,
        [("data", counts, svg.CANDIDATE_COLOR)],
        curves=[svg.Series("theory", edges, edges * 4, svg.THEORY_COLOR, dashed=True)],
    )
    assert chart.count("<polyline") == 1 and "theory" in chart
    assert 'class="heatmap"' in svg.heatmap(np.zeros((2, 2)))
