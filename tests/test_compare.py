"""The comparison framework is validated against cases with a known answer.

Reference vs reference (other seed) must pass: that bounds the false-positive rate. Reference vs
a deliberately wrong system must fail: that shows the checks are sensitive to the diffusion
coefficient, to drift and to confinement. Finally the graph system with the matched diffusion
coefficient passes and a mismatched one fails.
"""

import numpy as np
import pytest

from imigraph.compare import (
    ComparisonReport,
    MarginalKsCheck,
    MsdCheck,
    OccupancyCheck,
    compare,
)
from imigraph.graph import GraphSimulation, SquareLattice, walkers_at_node
from imigraph.graph.simulation import lattice_hop_diffusion
from imigraph.reference import BrownianBox, reflect_into_box
from imigraph.trajectory import Box, Trajectory

SIZE, SPACING, STEP_TIME = 41, 0.2, 0.01
DIFFUSION = lattice_hop_diffusion(SPACING, STEP_TIME)  # = 1.0
LATTICE = SquareLattice(SIZE, SPACING)
STEPS, RECORD_EVERY, PARTICLES = 3000, 50, 2000


def brownian(
    seed: int, diffusion: float = DIFFUSION, drift=(0.0, 0.0), box=None
) -> Trajectory:
    box = LATTICE.box if box is None else box
    system = BrownianBox(diffusion=diffusion, dt=STEP_TIME, box=box)
    rng = np.random.default_rng(seed)
    trajectory = system.simulate(
        system.all_at(box.center, PARTICLES), STEPS, rng, RECORD_EVERY
    )
    if drift != (
        0.0,
        0.0,
    ):  # add a ballistic component, reflected at the walls like the rest
        shift = trajectory.times[:, None, None] * np.asarray(drift)[None, None, :]
        positions = reflect_into_box(trajectory.positions + shift, box)
        trajectory = Trajectory(trajectory.times, positions, box)
    return trajectory


def graph(seed: int, hops_per_frame: int = RECORD_EVERY) -> Trajectory:
    simulation = GraphSimulation(LATTICE, step_time=STEP_TIME)
    rng = np.random.default_rng(seed)
    initial = walkers_at_node(LATTICE, LATTICE.center_node, PARTICLES)
    steps = STEPS * hops_per_frame // RECORD_EVERY
    history = simulation.run(initial, steps, rng, hops_per_frame)
    return history.project(cell_rng=rng)


@pytest.fixture(scope="module")
def reference() -> Trajectory:
    return brownian(seed=100)


def test_reference_against_itself_with_other_seeds_passes(reference):
    for seed in (1, 2, 3):
        report = compare(reference, brownian(seed))
        assert report.passed, report.to_markdown()


def test_doubled_diffusion_fails_the_msd_check(reference):
    report = compare(reference, brownian(4, diffusion=2 * DIFFUSION))
    assert not report["msd"].passed
    assert report["msd"].value > 10


def test_drift_fails_the_distribution_checks(reference):
    report = compare(reference, brownian(5, drift=(0.05, 0.0)))
    assert not report["marginals"].passed
    assert not report["occupancy"].passed


def test_smaller_box_fails(reference):
    small = Box.square(
        LATTICE.box.side[0] / 2, lower=LATTICE.box.lower[0] + LATTICE.box.side[0] / 4
    )
    candidate = brownian(6, box=small)
    # Same times, same centre, but confined earlier: MSD saturates lower.
    report = compare(
        reference, Trajectory(candidate.times, candidate.positions, LATTICE.box)
    )
    assert not report.passed
    assert not report["msd"].passed


def test_graph_with_matched_diffusion_passes(reference):
    report = compare(reference, graph(7), candidate_label="lattice hop")
    assert report.passed, report.to_markdown()


def test_graph_with_mismatched_diffusion_fails(reference):
    # Half the hops per recorded interval is half the diffusion coefficient.
    candidate = graph(8, hops_per_frame=RECORD_EVERY // 2)
    candidate = Trajectory(reference.times, candidate.positions, candidate.box)
    report = compare(reference, candidate)
    assert not report["msd"].passed
    assert not report["marginals"].passed


def test_report_structure_and_markdown(reference):
    checks = (
        MsdCheck(max_sigma=5.0),
        MarginalKsCheck(num_frames=2),
        OccupancyCheck(bins=3),
    )
    report = compare(
        reference, brownian(9), checks=checks, candidate_label="other seed"
    )
    assert isinstance(report, ComparisonReport)
    assert [result.name for result in report.results] == [
        "msd",
        "marginals",
        "occupancy",
    ]
    assert report["marginals"].data["pvalues"].shape == (2, 2)
    data = report.to_dict()
    assert data["passed"] is True and len(data["checks"]) == 3
    markdown = report.to_markdown()
    assert "other seed" in markdown and "| msd | pass |" in markdown


def test_mismatched_times_are_rejected(reference):
    with pytest.raises(ValueError):
        compare(reference, reference.subsample(2))
