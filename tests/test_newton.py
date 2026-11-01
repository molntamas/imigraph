"""The bouncing ball: free Newtonian motion between elastic walls, conserving what it must."""

import numpy as np
import pytest

from imigraph.graph import SquareLattice
from imigraph.graph.newton import (
    BallState,
    NewtonWorld,
    disc_offsets,
    hop_order,
)
from imigraph.report.live import newton


def test_disc_and_hop_order():
    assert disc_offsets(0).tolist() == [[0, 0]]
    assert len(disc_offsets(1)) == 5 and len(disc_offsets(2)) == 13
    assert hop_order(np.array([3, 0])) == [0, 0, 0]
    assert hop_order(np.array([0, -2])) == [1, 1]
    order = hop_order(np.array([2, 1]))
    assert sorted(order) == [0, 0, 1] and order[0] == 0
    assert sorted(hop_order(np.array([-1, 3]))) == [0, 1, 1, 1]
    with pytest.raises(ValueError):
        disc_offsets(-1)


def test_free_motion_is_a_straight_line_and_walls_reflect_specularly():
    world = NewtonWorld(SquareLattice(9))
    ball = world.place((1, 4), (1, 0))
    run = world.run(ball, 16)
    x = run.sites[:, 0] // 9
    # 1 -> 8 in 7 steps, bounce, back to 1 in step 15, bounce again.
    assert x[:8].tolist() == list(range(1, 9))
    assert x[8:16].tolist() == list(range(7, -1, -1))
    assert run.velocities[8].tolist() == [-1, 0] and run.velocities[16].tolist() == [
        1,
        0,
    ]
    assert run.bounces.tolist() == [0] * 7 + [1] + [0] * 7 + [1]
    # A diagonal ball reflects the hit component only.
    run = world.run(world.place((4, 7), (1, 1)), 3)
    assert run.velocities[-1].tolist() == [1, -1]
    assert np.linalg.norm(run.velocities, axis=1).tolist() == [np.sqrt(2)] * 4


def test_energy_momentum_and_shape_are_conserved_over_a_long_run():
    world = NewtonWorld(SquareLattice(12, spacing=0.5), step_time=0.25)
    ball = world.place((5, 6), (2, -1), radius=1, mass=2.0)
    run = world.run(ball, 300)
    assert run.num_walkers == 5
    energy = run.energy()
    assert np.all(energy == energy[0])
    assert energy[0] == pytest.approx(0.5 * 2.0 * 5 * 5 * (0.5 / 0.25) ** 2)
    momentum = run.momentum()
    np.testing.assert_allclose(
        np.linalg.norm(momentum, axis=1), np.linalg.norm(momentum[0])
    )
    changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
    np.testing.assert_array_equal(changed, run.bounces > 0)
    assert run.bounces.sum() > 10
    assert run.shape_intact()
    assert all(check["passed"] for check in run.checks())
    # Never outside the room, never two walkers on one site.
    assert run.sites.min() >= 0 and run.sites.max() < 144
    assert all(np.unique(frame).size == 5 for frame in run.sites)


def test_entropy_tracks_dispersion_and_sweeping():
    world = NewtonWorld(SquareLattice(12))
    run = world.run(world.place((6, 6), (2, 1), radius=1), 200)
    configuration = run.configuration_entropy()
    assert np.allclose(configuration, np.log(5))
    visitation = run.visitation_entropy(6)
    coverage = run.coverage(6)
    assert np.all(np.diff(coverage) >= 0) and coverage[-1] > 0.8
    assert visitation[-1] > visitation[0] + 1.0
    assert visitation[-1] <= np.log(36) + 1e-9
    # A ball shuttling along one row visits one row of bins only.
    row = world.run(world.place((6, 6), (1, 0)), 200)
    assert row.coverage(6)[-1] <= 6 / 36 + 1e-9
    assert row.visitation_entropy(6)[-1] < np.log(6) + 1e-9


def test_invalid_balls_and_rooms_are_refused():
    lattice = SquareLattice(6)
    world = NewtonWorld(lattice)
    with pytest.raises(ValueError, match="does not fit"):
        world.place((0, 0), (1, 1), radius=1)
    with pytest.raises(ValueError, match="smaller than the room"):
        world.place((3, 3), (1, 1), radius=3)
    with pytest.raises(ValueError, match="reflecting"):
        NewtonWorld(SquareLattice(6, periodic=True))
    with pytest.raises(ValueError, match="distinct sites"):
        BallState(lattice, np.array([1, 1]), np.array([1, 0]))
    with pytest.raises(ValueError, match="velocity"):
        BallState(lattice, np.array([1]), np.array([1, 0, 0]))


def test_live_newton_entry_point():
    result = newton({"size": 10, "radius": 1, "vx": 2, "vy": 1, "steps": 40})
    assert result["error"] is None
    assert len(result["run"]["sites"]) == 41 and len(result["run"]["sites"][0]) == 5
    assert len(result["run"]["energy"]) == 41 and len(result["run"]["bounces"]) == 40
    assert all(c["passed"] for c in result["checks"]), result["checks"]
    assert len(result["checks"]) == 6
    assert len(result["run"]["coverage"]) == 41
    assert result["charts"]["energy"].startswith("<svg")
    assert result["charts"]["momentum"].startswith("<svg")
    assert result["charts"]["entropy"].startswith("<svg")
    assert "specular" in result["docs"]
    assert result["summary"]["bounces"] == int(sum(result["run"]["bounces"]))
    assert result["summary"]["energy"] == result["run"]["energy"][0]
    assert "does not fit" in newton({"size": 6, "radius": 2, "cx": 0, "cy": 0})["error"]
    assert newton({"vx": 0, "vy": 0, "steps": 5})["error"] is None  # a ball at rest
    assert "velocity" in newton({"vx": 9})["error"]
