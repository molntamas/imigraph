"""Running checks and collecting their results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from imigraph.compare.checks import Check, CheckResult, default_checks
from imigraph.trajectory import Trajectory


@dataclass(frozen=True)
class ComparisonReport:
    reference_label: str
    candidate_label: str
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def __getitem__(self, name: str) -> CheckResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference_label,
            "candidate": self.candidate_label,
            "passed": self.passed,
            "checks": [result.to_dict() for result in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            (
                f"Candidate **{self.candidate_label}** vs reference **{self.reference_label}**: "
                f"{'PASS' if self.passed else 'FAIL'}"
            ),
            "",
            "| check | verdict | value | threshold | detail |",
            "| --- | --- | --- | --- | --- |",
        ]
        for result in self.results:
            relation = ">" if result.larger_is_better else "<="
            lines.append(
                f"| {result.name} | {'pass' if result.passed else 'FAIL'} | {result.value:.4g} "
                f"| {relation} {result.threshold:.4g} | {result.detail} |"
            )
        return "\n".join(lines)


def compare(
    reference: Trajectory,
    candidate: Trajectory,
    checks: Sequence[Check] | None = None,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> ComparisonReport:
    """Run every check of ``candidate`` against ``reference``.

    Both trajectories must be recorded at the same times (particle counts may differ).
    """
    if reference.num_frames != candidate.num_frames or not np.allclose(
        reference.times, candidate.times
    ):
        raise ValueError("reference and candidate must be recorded at the same times")
    if checks is None:
        checks = default_checks()
    results = tuple(check.run(reference, candidate) for check in checks)
    return ComparisonReport(reference_label, candidate_label, results)
