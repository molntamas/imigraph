"""Minimal SVG plots as strings, so reports need no plotting library (and work in Pyodide)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from itertools import pairwise

import numpy as np

REFERENCE_COLOR = "#2563eb"
CANDIDATE_COLOR = "#dc2626"
THEORY_COLOR = "#6b7280"
AXIS_COLOR = "#374151"
GRID_COLOR = "#e5e7eb"
LEGEND_ROWS = 2
LEGEND_ROW_HEIGHT = 18


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick positions covering ``[low, high]`` (``count`` is a target, not exact)."""
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if high <= low:
        high = low + 1.0
    raw = (high - low) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    nice = 1 if residual < 1.5 else 2 if residual < 3 else 5 if residual < 7 else 10
    step = nice * magnitude
    # A step below the float resolution of the values would never advance ``value``.
    if step <= 1e-12 * max(abs(low), abs(high), 1.0):
        return [low, high]
    first = math.floor(low / step) * step
    ticks = []
    value = first
    while value <= high + step * 1e-9 and len(ticks) < 100:
        if value >= low - step * 1e-9:
            ticks.append(round(value, 12))
        value += step
    return ticks or [low]


def _fmt(value: float) -> str:
    text = f"{value:.4g}"
    return text.replace("e+0", "e").replace("e-0", "e-")


@dataclass(frozen=True)
class Series:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str
    dashed: bool = False
    band: tuple[np.ndarray, np.ndarray] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Axes:
    """Pixel layout of a chart and the data-to-pixel transform."""

    width: int
    height: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    left: int = 60
    right: int = 16
    top: int = 32
    bottom: int = 44 + 2 * 18  # axis label plus LEGEND_ROWS legend rows

    def px(self, x: float) -> float:
        lo, hi = self.x_range
        return self.left + (x - lo) / (hi - lo) * (self.width - self.left - self.right)

    def py(self, y: float) -> float:
        lo, hi = self.y_range
        return (
            self.height
            - self.bottom
            - (y - lo) / (hi - lo) * (self.height - self.top - self.bottom)
        )

    def frame(self, title: str, xlabel: str, ylabel: str) -> list[str]:
        parts = [
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>',
            (
                f'<text x="{self.left}" y="20" font-size="14" font-weight="600" '
                f'fill="{AXIS_COLOR}">{escape(title)}</text>'
            ),
        ]
        for tick in _nice_ticks(*self.x_range):
            x = self.px(tick)
            parts.append(
                f'<line x1="{x:.1f}" y1="{self.top}" x2="{x:.1f}" '
                f'y2="{self.height - self.bottom}" stroke="{GRID_COLOR}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{self.height - self.bottom + 16}" font-size="11" '
                f'text-anchor="middle" fill="{AXIS_COLOR}">{_fmt(tick)}</text>'
            )
        for tick in _nice_ticks(*self.y_range):
            y = self.py(tick)
            parts.append(
                f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.width - self.right}" '
                f'y2="{y:.1f}" stroke="{GRID_COLOR}"/>'
            )
            parts.append(
                f'<text x="{self.left - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                f'fill="{AXIS_COLOR}">{_fmt(tick)}</text>'
            )
        parts.append(
            f'<rect x="{self.left}" y="{self.top}" width="{self.width - self.left - self.right}" '
            f'height="{self.height - self.top - self.bottom}" fill="none" stroke="{AXIS_COLOR}"/>'
        )
        parts.append(
            f'<text x="{(self.left + self.width - self.right) / 2:.1f}" '
            f'y="{self.height - LEGEND_ROWS * LEGEND_ROW_HEIGHT - 8}" '
            f'font-size="12" text-anchor="middle" fill="{AXIS_COLOR}">{escape(xlabel)}</text>'
        )
        parts.append(
            f'<text transform="translate(14 {(self.top + self.height - self.bottom) / 2:.1f}) '
            f'rotate(-90)" font-size="12" text-anchor="middle" fill="{AXIS_COLOR}">'
            f"{escape(ylabel)}</text>"
        )
        return parts

    def legend(self, entries: list[tuple[str, str, bool]]) -> list[str]:
        """Legend rows below the x-axis label (inside the ``LEGEND_ROWS`` extra bottom margin)."""
        parts = []
        x, y = self.left, self.height - LEGEND_ROWS * LEGEND_ROW_HEIGHT + 8
        for label, color, dashed in entries:
            width = 34 + 7 * len(label)
            if x + width > self.width - self.right and x > self.left:
                x, y = self.left, y + LEGEND_ROW_HEIGHT
            dash = ' stroke-dasharray="6 4"' if dashed else ""
            parts.append(
                f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{color}" '
                f'stroke-width="2"{dash}/>'
            )
            parts.append(
                f'<text x="{x + 28}" y="{y + 4}" font-size="12" fill="{AXIS_COLOR}">'
                f"{escape(label)}</text>"
            )
            x += width
        return parts


def _svg(width: int, height: int, parts: list[str], css_class: str = "chart") -> str:
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" class="{css_class}" role="img">\n{body}\n</svg>'
    )


def line_chart(
    series: list[Series],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    width: int = 640,
    height: int = 396,
    zero_based: bool = True,
) -> str:
    """Line chart; with ``zero_based`` the y-axis includes 0, otherwise it fits the data."""
    xs = np.concatenate([np.asarray(s.x, dtype=float) for s in series])
    ys = np.concatenate(
        [np.asarray(s.y, dtype=float) for s in series]
        + [
            np.asarray(b, dtype=float)
            for s in series
            if s.band is not None
            for b in s.band
        ]
    )
    y_high = max(float(np.nanmax(ys)), 0.0) if ys.size else 1.0
    y_low = min(float(np.nanmin(ys)), 0.0) if ys.size else 0.0
    if y_high == y_low:
        y_high = y_low + 1.0
    margin = 0.05 * (y_high - y_low)
    if zero_based:
        y_range = (y_low - margin if y_low < 0 else 0.0, y_high + margin)
    else:
        y_low = float(np.nanmin(ys))
        span = y_high - y_low
        if span <= 1e-9 * max(abs(y_high), abs(y_low), 1.0):  # constant to rounding
            margin = 0.1 * max(abs(y_high), 1.0)
        else:
            margin = 0.1 * span
        y_range = (y_low - margin, y_high + margin)
    axes = Axes(width, height, (float(xs.min()), float(xs.max())), y_range)
    parts = axes.frame(title, xlabel, ylabel)
    for s in series:
        if s.band is not None:
            low, high = s.band
            upper = " ".join(
                f"{axes.px(x):.1f},{axes.py(y):.1f}"
                for x, y in zip(s.x, high, strict=True)
            )
            lower = " ".join(
                f"{axes.px(x):.1f},{axes.py(y):.1f}"
                for x, y in zip(s.x[::-1], low[::-1], strict=True)
            )
            parts.append(
                f'<polygon points="{upper} {lower}" fill="{s.color}" opacity="0.15"/>'
            )
        points = " ".join(
            f"{axes.px(x):.1f},{axes.py(y):.1f}" for x, y in zip(s.x, s.y, strict=True)
        )
        dash = ' stroke-dasharray="6 4"' if s.dashed else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{s.color}" stroke-width="2"{dash}/>'
        )
    parts += axes.legend([(s.label, s.color, s.dashed) for s in series])
    return _svg(width, height, parts)


def histogram_chart(
    edges: np.ndarray,
    counts: list[tuple[str, np.ndarray, str]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
    width: int = 640,
    height: int = 336,
    curves: list[Series] | None = None,
) -> str:
    """Overlaid step histograms with shared ``edges``; ``counts`` holds (label, counts, color).

    ``curves`` are drawn as lines on the same axes (e.g. a theoretical density scaled to counts).
    """
    edges = np.asarray(edges, dtype=float)
    top = max(float(np.max(c)) for _, c, _ in counts) if counts else 1.0
    for curve in curves or []:
        top = max(top, float(np.nanmax(curve.y)))
    axes = Axes(
        width, height, (float(edges[0]), float(edges[-1])), (0.0, top * 1.1 or 1.0)
    )
    parts = axes.frame(title, xlabel, ylabel)
    for _label, values, color in counts:
        points = [f"{axes.px(edges[0]):.1f},{axes.py(0):.1f}"]
        for left, right, value in zip(edges[:-1], edges[1:], values, strict=True):
            points.append(f"{axes.px(left):.1f},{axes.py(value):.1f}")
            points.append(f"{axes.px(right):.1f},{axes.py(value):.1f}")
        points.append(f"{axes.px(edges[-1]):.1f},{axes.py(0):.1f}")
        parts.append(
            f'<polygon points="{" ".join(points)}" fill="{color}" opacity="0.18" stroke="{color}" '
            f'stroke-width="1.5"/>'
        )
    for curve in curves or []:
        points = " ".join(
            f"{axes.px(x):.1f},{axes.py(y):.1f}"
            for x, y in zip(curve.x, curve.y, strict=True)
        )
        dash = ' stroke-dasharray="6 4"' if curve.dashed else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{curve.color}" '
            f'stroke-width="2"{dash}/>'
        )
    parts += axes.legend(
        [(label, color, False) for label, _, color in counts]
        + [(curve.label, curve.color, curve.dashed) for curve in curves or []]
    )
    return _svg(width, height, parts)


def heatmap(counts: np.ndarray, color: str = REFERENCE_COLOR, size: int = 220) -> str:
    """A grid of cells shaded by ``counts`` (rows are x bins, columns y bins; y grows upward)."""
    counts = np.asarray(counts, dtype=float)
    rows, cols = counts.shape
    top = float(counts.max()) if counts.size and counts.max() > 0 else 1.0
    cell = size / max(rows, cols)
    parts = [f'<rect width="{size}" height="{size}" fill="white"/>']
    for i in range(rows):
        for j in range(cols):
            opacity = 0.08 + 0.92 * counts[i, j] / top
            x, y = i * cell, (cols - 1 - j) * cell
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="{color}" opacity="{opacity:.3f}" stroke="white">'
                f"<title>x bin {i}, y bin {j}: {int(counts[i, j])}</title></rect>"
            )
    return _svg(size, size, parts, css_class="heatmap")


# ---------- rule patterns ----------

PATTERN_SITE_COLOR = "#9ca3af"
PATTERN_WALKER_COLOR = "#dc2626"
PATTERN_WALKER_EDGE_COLOR = "#f87171"


def pattern_layout(
    lhs: tuple[tuple[str, ...], ...],
    rhs: tuple[tuple[str, ...], ...],
    is_walker,
    width: float,
    height: float,
) -> dict[str, tuple[float, float]]:
    """Positions of the variables of a rule pattern inside one panel: sites in a row along the
    lower part, walkers above the first site they touch (the same layout as the GUI editor,
    ``patternLayout`` in ``web/app.js``, so both drawings look alike)."""
    order: dict[str, None] = {}
    for edge in lhs + rhs:
        for v in edge:
            order.setdefault(v, None)
    sites = [v for v in order if not is_walker(v)]
    walkers = [v for v in order if is_walker(v)]
    positions: dict[str, tuple[float, float]] = {}
    pad = 0.14 * width  # sites spread over the panel, walkers above them
    for k, site in enumerate(sites):
        x = (
            width / 2
            if len(sites) == 1
            else pad + k * (width - 2 * pad) / (len(sites) - 1)
        )
        positions[site] = (x, 0.72 * height)
    used: dict[float, int] = {}
    for walker in walkers:
        x = width / 2
        for edge in lhs + rhs:
            if walker in edge:
                others = [v for v in edge if v != walker and v in positions]
                if others:
                    x = positions[others[0]][0]
                    break
        k = used.get(x, 0)
        used[x] = k + 1
        positions[walker] = (x + 14 * k, 0.26 * height)
    return positions


def rule_pattern(
    lhs: tuple[tuple[str, ...], ...],
    rhs: tuple[tuple[str, ...], ...],
    is_walker,
    panel: int = 150,
    height: int = 80,
) -> str:
    """The two sides of a rule as small graphs: variables as nodes (sites grey, walkers red),
    the edges of each side, an arrow between them."""
    gap = 26
    width = 2 * panel + gap
    positions = pattern_layout(lhs, rhs, is_walker, panel, height)
    parts = [f'<rect width="{width}" height="{height}" fill="white" rx="6"/>']
    for side, offset in ((lhs, 0.0), (rhs, panel + gap)):
        for edge in side:
            walker_edge = any(is_walker(v) for v in edge)
            color = PATTERN_WALKER_EDGE_COLOR if walker_edge else PATTERN_SITE_COLOR
            points = [positions[v] for v in edge if v in positions]
            if len(points) == 1 or (
                len(points) == 2 and points[0] == points[1]
            ):  # a loop {s,s}
                x, y = points[0]
                parts.append(
                    f'<circle cx="{offset + x:.1f}" cy="{y - 11:.1f}" r="7" fill="none" '
                    f'stroke="{color}" stroke-width="1.5"/>'
                )
                continue
            for (x0, y0), (x1, y1) in pairwise(points):
                parts.append(
                    f'<line x1="{offset + x0:.1f}" y1="{y0:.1f}" x2="{offset + x1:.1f}" '
                    f'y2="{y1:.1f}" stroke="{color}" stroke-width="{2 if walker_edge else 1.5}"/>'
                )
        for v, (x, y) in positions.items():
            if not any(v in edge for edge in side):
                continue
            color = PATTERN_WALKER_COLOR if is_walker(v) else PATTERN_SITE_COLOR
            parts.append(
                f'<circle cx="{offset + x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>'
                f'<text x="{offset + x:.1f}" y="{y + 3.5:.1f}" font-size="9" fill="white" '
                f'text-anchor="middle" font-family="ui-monospace, monospace">{escape(v)}</text>'
            )
    mid = panel + gap / 2
    parts.append(
        f'<path d="M {mid - 8:.1f} {height / 2:.1f} h 14 m -4 -4 l 4 4 l -4 4" fill="none" '
        f'stroke="{AXIS_COLOR}" stroke-width="1.5"/>'
    )
    return _svg(width, height, parts, css_class="pattern")
