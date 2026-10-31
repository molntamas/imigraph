"""Comparison framework: does a candidate system imitate a reference system?

Both systems are reduced to a :class:`~imigraph.trajectory.Trajectory`; a list of *checks* each
produce a :class:`CheckResult` with a value, a threshold and a verdict, collected in a
:class:`ComparisonReport`. See ``docs/comparison-method.md`` for the method and its validation.
"""

from imigraph.compare.checks import (
    Check,
    CheckResult,
    MarginalKsCheck,
    MsdCheck,
    OccupancyCheck,
    default_checks,
)
from imigraph.compare.framework import ComparisonReport, compare

__all__ = [
    "Check",
    "CheckResult",
    "ComparisonReport",
    "MarginalKsCheck",
    "MsdCheck",
    "OccupancyCheck",
    "compare",
    "default_checks",
]
