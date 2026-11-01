"""Individual checks of a candidate trajectory against a reference trajectory.

Every check compares the same observable in both systems and decides with an explicit threshold.
Statistical checks use two-sample tests, so the reference is *also* treated as a finite sample:
the verdict is "no evidence that the two systems differ", at the sample size used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from imigraph.trajectory import Trajectory


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check.

    ``value`` is compared with ``threshold`` in the direction given by ``larger_is_better``;
    ``detail`` explains the numbers for humans, ``data`` carries arrays for plots.
    """

    name: str
    passed: bool
    value: float
    threshold: float
    larger_is_better: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "larger_is_better": self.larger_is_better,
            "detail": self.detail,
        }


@runtime_checkable
class Check(Protocol):
    @property
    def name(self) -> str: ...

    def run(self, reference: Trajectory, candidate: Trajectory) -> CheckResult: ...


def _frames_to_test(num_frames: int, count: int) -> np.ndarray:
    """``count`` frame indices spread evenly over ``1 .. num_frames - 1`` (last one included)."""
    if num_frames < 2:
        raise ValueError("trajectories need at least two frames")
    return np.unique(
        np.linspace(1, num_frames - 1, min(count, num_frames - 1)).round().astype(int)
    )


@dataclass(frozen=True)
class MsdCheck:
    """Mean squared displacement curves agree within their statistical uncertainty.

    At every recorded time ``t > 0`` the deviation ``msd_c(t) - msd_r(t)`` is divided by the
    combined standard error of the two estimates (from the spread over particles); ``value`` is
    the largest such z-score and must stay below ``max_sigma``. With ``rtol`` set, deviations
    that are relatively smaller than ``rtol`` are tolerated as well, whatever their significance
    (an allowance for known, physically negligible systematic effects).
    """

    max_sigma: float = 4.0
    rtol: float | None = None
    name: str = "msd"

    def run(self, reference: Trajectory, candidate: Trajectory) -> CheckResult:
        msd_r, msd_c = reference.msd()[1:], candidate.msd()[1:]
        se_r, se_c = (
            reference.msd_standard_error()[1:],
            candidate.msd_standard_error()[1:],
        )
        combined = np.sqrt(se_r**2 + se_c**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(combined > 0, np.abs(msd_c - msd_r) / combined, 0.0)
            relative = np.where(msd_r > 0, np.abs(msd_c - msd_r) / msd_r, 0.0)
        tolerated = sigma <= self.max_sigma
        if self.rtol is not None:
            tolerated |= relative <= self.rtol
        worst = int(np.argmax(sigma))
        value = float(sigma[worst])
        return CheckResult(
            name=self.name,
            passed=bool(np.all(tolerated)),
            value=value,
            threshold=self.max_sigma,
            larger_is_better=False,
            detail=(
                f"largest MSD deviation {value:.2f} standard errors ({relative[worst]:.1%} "
                f"relative) at t = {reference.times[worst + 1]:.4g}; "
                f"max relative deviation {relative.max():.1%}"
                + (f", tolerance {self.rtol:.0%}" if self.rtol is not None else "")
            ),
            data={
                "times": reference.times,
                "reference": reference.msd(),
                "candidate": candidate.msd(),
                "reference_se": reference.msd_standard_error(),
                "candidate_se": candidate.msd_standard_error(),
            },
        )


@dataclass(frozen=True)
class MarginalKsCheck:
    """Two-sample Kolmogorov-Smirnov tests on the x and y marginals at several times.

    ``value`` is the smallest p-value over the tested frames and coordinates; the threshold is
    ``alpha`` Bonferroni-corrected for the number of tests, so a true match fails with
    probability at most ``alpha``.
    """

    alpha: float = 0.01
    num_frames: int = 5
    name: str = "marginals"

    def run(self, reference: Trajectory, candidate: Trajectory) -> CheckResult:
        from scipy import stats

        frames = _frames_to_test(reference.num_frames, self.num_frames)
        pvalues = np.empty((len(frames), 2))
        statistics = np.empty((len(frames), 2))
        for row, frame in enumerate(frames):
            for axis in range(2):
                result = stats.ks_2samp(
                    reference.positions[frame, :, axis],
                    candidate.positions[frame, :, axis],
                )
                pvalues[row, axis] = result.pvalue
                statistics[row, axis] = result.statistic
        threshold = self.alpha / pvalues.size
        row, axis = np.unravel_index(np.argmin(pvalues), pvalues.shape)
        value = float(pvalues[row, axis])
        return CheckResult(
            name=self.name,
            passed=bool(value > threshold),
            value=value,
            threshold=threshold,
            larger_is_better=True,
            detail=(
                f"smallest KS p-value {value:.3g} for {'xy'[axis]} at "
                f"t = {reference.times[frames[row]]:.4g} (statistic "
                f"{statistics[row, axis]:.3f}); {pvalues.size} tests, "
                f"Bonferroni threshold {threshold:.3g}"
            ),
            data={
                "frames": frames,
                "times": reference.times[frames],
                "pvalues": pvalues,
            },
        )


@dataclass(frozen=True)
class OccupancyCheck:
    """Chi-square homogeneity test of the 2D occupancy histograms at the last frame.

    Both samples are binned on the same ``bins x bins`` grid over the reference box; the test
    asks whether the two sets of counts come from the same distribution. Empty bins are dropped
    from the contingency table (they carry no information for the homogeneity test).
    """

    bins: int = 5
    alpha: float = 0.01
    name: str = "occupancy"

    def run(self, reference: Trajectory, candidate: Trajectory) -> CheckResult:
        from scipy import stats

        box = reference.box
        extent = [[box.lower[0], box.upper[0]], [box.lower[1], box.upper[1]]]
        counts = [
            np.histogram2d(
                trajectory.positions[-1, :, 0],
                trajectory.positions[-1, :, 1],
                bins=self.bins,
                range=extent,
            )[0]
            for trajectory in (reference, candidate)
        ]
        table = np.stack([count.ravel() for count in counts])
        table = table[:, table.sum(axis=0) > 0]
        if np.any(table.sum(axis=1) == 0):  # one system has no particle inside the box
            value, statistic, dof = 0.0, float("inf"), 0
        elif table.shape[1] < 2:
            value, statistic, dof = 1.0, 0.0, 0
        else:
            result = stats.chi2_contingency(table)
            value, statistic, dof = (
                float(result.pvalue),
                float(result.statistic),
                int(result.dof),
            )
        return CheckResult(
            name=self.name,
            passed=bool(value > self.alpha),
            value=value,
            threshold=self.alpha,
            larger_is_better=True,
            detail=(
                f"chi-square homogeneity p-value {value:.3g} (statistic {statistic:.1f}, "
                f"{dof} degrees of freedom) on a {self.bins}x{self.bins} grid at "
                f"t = {reference.times[-1]:.4g}"
            ),
            data={"reference": counts[0], "candidate": counts[1], "extent": extent},
        )


def default_checks() -> tuple[Check, ...]:
    return (MsdCheck(), MarginalKsCheck(), OccupancyCheck())
