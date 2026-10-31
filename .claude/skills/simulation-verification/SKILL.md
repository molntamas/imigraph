---
name: simulation-verification
description: How imigraph verifies simulations - adding a reference system, choosing analytic invariants and tolerances, adding a comparison check and validating it with pass and fail cases, keeping performance guardrails. Use when writing or reviewing tests for numerical or graph code.
---

# Verifying simulations

Read `docs/comparison-method.md` first. The principles:

1. **Test against something independent.** Analytic results (MSD `= 4 D t`, Gaussian marginals
   with variance `2 D t`, uniform stationary density), invariants (particles stay in the box,
   every hop follows an existing edge, node count conserved), or a trusted reference system.
   Never recompute the expected value the way the code does.
2. **State the tolerance and where it comes from.** Statistical estimates have a standard error:
   the relative error of an MSD estimate from `n` particles is about `1 / sqrt(n)`; a tolerance
   of five standard errors is safe against flakiness. Write the arithmetic in a comment.
3. **Seed everything.** `rng = np.random.default_rng(seed)`; pass `rng` down. Tests must be
   deterministic. Use `rng.spawn(k)` for independent streams.
4. **Validate the verdict procedure itself.** A comparison check needs, in
   `tests/test_compare.py`: a case that must pass (reference vs reference with another seed) and
   cases that must fail (wrong diffusion, drift, wrong box, mismatched graph). Without a fail
   case a check proves nothing.
5. **Know the false-positive rate.** Statistical checks fail a true match with probability
   alpha; Bonferroni-correct families of tests; pin seeds in tests. When a seed fails, decide
   with a high-sample run whether it is noise or a systematic effect (see the diagnostic in
   `docs/comparison-method.md`, "Validation"), do not just change the seed.
6. **Mind discreteness.** Lattice positions are discrete; compare distributions at a declared
   resolution (`GraphHistory.project(cell_rng=...)`), never by smoothing the reference.
7. **Guard performance.** `tests/test_performance.py` has wall-clock budgets of about four times
   the measured time in the container. Measure first (best of three), then set the budget, and
   write the measurement into the docstring. One rule application must be O(walkers) numpy work.

## Adding a reference system

- Put it under `src/imigraph/reference/`, return a `Trajectory`, take `rng` and `record_every`.
- Document the analytic facts in the module docstring; test each of them in
  `tests/test_<system>.py` with explicit tolerances.
- Add an experiment under `src/imigraph/experiments/` that derives the shared physical
  parameters once (as `ExperimentConfig.diffusion` does) and compares.

## Adding a check

- Implement `Check.run(reference, candidate) -> CheckResult` in `compare/checks.py`, with
  `value`, `threshold`, `larger_is_better`, a detail string and the arrays for plots in `data`.
- Add it to `default_checks()` only if it is cheap and meaningful for every system.
- Extend the validation matrix in `tests/test_compare.py` and the table in
  `docs/comparison-method.md`.

## Commands

```bash
docker compose run --rm -T dev uv run pytest -q tests/test_compare.py
docker compose run --rm -T dev uv run pytest -q tests/test_performance.py
docker compose run --rm -T dev uv run python -m imigraph compare --particles 20000   # high-sample diagnostic
```
