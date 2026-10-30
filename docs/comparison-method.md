# Comparison method

How imigraph decides whether a candidate system (a graph system) imitates a reference system,
and how that decision procedure is itself validated. Implementation: `src/imigraph/compare/`.

## Setup

Both systems produce a `Trajectory`: the positions of `n` particles at `T` recorded times,
inside a box. The recorded times must be identical (particle counts may differ). The comparison
treats *both* trajectories as finite random samples: it never assumes that the reference is
exact. The verdict is therefore "no evidence, at this sample size, that the two systems differ
in the checked observables", plus an explicit list of which observables were checked.

## Checks

| check | observable | statistic | threshold (default) |
| --- | --- | --- | --- |
| `msd` | mean squared displacement `<|r(t) - r(0)|^2>` at every recorded `t > 0` | z-score of the difference against the combined standard error (estimated from the spread over particles) | largest z <= 4 |
| `marginals` | x and y marginal distributions at 5 recorded times (spread evenly, last one included) | two-sample Kolmogorov-Smirnov test per time and coordinate | smallest p > 0.01 / (number of tests), Bonferroni |
| `occupancy` | 2D occupancy on a 5 x 5 grid at the last time | chi-square homogeneity test of the two count tables (empty bins dropped) | p > 0.01 |

Why these three:

- The MSD is the physical observable of diffusion (`4 D t` while free, saturating in a box).
  It is sensitive to the diffusion coefficient and to confinement, and comes with an honest
  error bar from the particle ensemble. The z-score form keeps the verdict independent of how
  many frames are recorded; `rtol` can add an explicit allowance for known systematic effects.
- The marginal tests catch shape differences the MSD misses: drift, skew, wrong tails, and
  boundary artefacts. Several times are tested because a system may match early (free
  diffusion) but not late (walls), or vice versa.
- The occupancy test checks the joint 2D distribution (correlations between x and y, corners),
  where the marginals can agree while the joint density does not.

Each check returns a `CheckResult` with the value, the threshold, a verdict and a human-readable
detail line, plus the arrays needed for plots. `ComparisonReport` collects them; `passed` is the
conjunction.

## Resolution

A lattice graph is discrete: its projected positions sit on a grid. A raw KS test between grid
positions and continuous positions rejects at early times simply because the grid is visible
(step size `a` against a spread of a few `a`). That is a real difference, and the framework does
not hide it; instead the graph system states the resolution at which it claims to imitate:
`GraphHistory.project(cell_rng=...)` places each walker at a fixed uniformly random point inside
its node's cell. The jittered sample is an exact sample of the cell-uniform density, so the
two-sample tests remain valid, and the claim becomes "the graph imitates Brownian motion at
resolution `a`". Displacements are unaffected, so the MSD check sees the pure lattice dynamics.

## Validation of the framework (`tests/test_compare.py`)

The framework is only as reliable as its known behaviour on cases with a known answer. With
fixed seeds, 2000 particles, 60 recorded frames:

| case | expected | checks that must react |
| --- | --- | --- |
| reference vs reference, other seeds | pass | (all pass) |
| reference vs reference with `2 D` | fail | `msd` (z > 10) |
| reference vs reference with a drift | fail | `marginals`, `occupancy` |
| reference vs reference in a box half the size | fail | `msd` |
| lattice graph with matched `D = a^2 / (4 tau)` | pass | (all pass) |
| lattice graph with half the hops per recorded interval | fail | `msd`, `marginals` |

False-positive rate: by construction each statistical check fails a true match with probability
about 1% (alpha = 0.01, Bonferroni for the marginals) and the MSD check far less often (4 sigma).
Across ten seeds of graph-vs-reference at 2000 particles, one run failed the marginal check
(p = 3e-5 at one time); with 20 000 particles the KS statistics of graph-vs-reference and
reference-vs-reference are indistinguishable (0.005 to 0.015 at all frames), so this is the
expected false-positive rate, not a systematic difference. The tests pin seeds, so they are
deterministic.

Sensitivity: at 2000 particles the MSD check resolves a few percent in `D`; the marginal check
resolves KS distances of about 0.04 (two-sample, equal sizes). More particles buy more
resolution linearly in cost.

## Using it

```python
from imigraph.compare import compare, MsdCheck, MarginalKsCheck, OccupancyCheck

report = compare(reference_trajectory, candidate_trajectory)  # default checks
report = compare(reference, candidate, checks=[MsdCheck(max_sigma=3.0, rtol=0.02)])
report.passed, report["msd"].value, report.to_markdown(), report.to_dict()
```

`python -m imigraph compare` runs the default experiment and prints the report; the exit code is
0 on pass and 1 on fail, so it can gate a CI job.
