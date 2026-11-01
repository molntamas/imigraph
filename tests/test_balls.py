"""The ball gas: elastic collisions on the lattice, exact invariants, and the hard-disk reference."""

import numpy as np
import pytest

from imigraph.experiments.balls import (
    BallGasConfig,
    initial_condition,
    run_ball_comparison,
    run_ball_gas,
)
from imigraph.graph import SquareLattice
from imigraph.graph.balls import BallGas, BallGasState
from imigraph.reference.gas import HardDiskGas
from imigraph.report.live import balls as live_balls


def _gas(size=9, rule="exchange"):
    return BallGas(SquareLattice(size), rule=rule)


def test_head_on_collision_exchanges_the_velocities_exactly():
    gas = _gas()
    state = gas.place([[1, 4], [7, 4]], [[2, 0], [-1, 0]])
    run = gas.run(state, 6)
    assert run.collisions.sum() == 1 and run.fallbacks.sum() == 0
    step = int(np.flatnonzero(run.collisions)[0]) + 1
    # Before the hit: (2, 0) and (-1, 0); after: the x components are exchanged.
    assert run.velocities[step].tolist() == [[-1, 0], [2, 0]]
    assert np.all(run.overlaps() == 0)
    event = next(e for e in run.events[step - 1] if e["kind"] == "collision")
    assert event["axis"] == "x" and event["exact"]
    assert "exact exchange" in run.captions()[step]


def test_diagonal_hit_is_exact_when_even_and_a_swap_when_odd():
    gas = _gas()
    # Both balls hop into (4,4) in the first round, from the left and from below: the line of
    # centres is the diagonal d = (1, -1). With (1,0) and (0,1): (va - vb) . d = 2, even: the
    # exact exchange delta = 1 gives (0,1) and (1,0).
    even = gas.place([[3, 4], [4, 3]], [[1, 0], [0, 1]])
    run = gas.run(even, 4)
    assert run.collisions.sum() == 1 and run.fallbacks.sum() == 0
    assert run.velocities[1].tolist() == [[0, 1], [1, 0]]
    # With (2,0) and (0,1): (va - vb) . d = 3, odd: the swap fallback.
    odd = gas.place([[3, 4], [4, 3]], [[2, 0], [0, 1]])
    run = gas.run(odd, 4)
    assert run.velocities[1].tolist() == [[0, 1], [2, 0]]
    assert run.collisions.sum() >= 1 and run.fallbacks.sum() >= 1
    assert np.all(run.overlaps() == 0)
    assert "swap fallback" in " ".join(run.captions())


def test_rules_conserve_energy_and_momentum_exactly_and_keep_balls_apart():
    for rule in ("exchange", "swap"):
        config = BallGasConfig(
            size=8, balls=8, max_speed=3, steps=120, rule=rule, seed=4
        )
        result = run_ball_gas(config)
        run = result.run
        energy = run.energy()
        assert np.all(energy == energy[0])
        momentum = run.momentum()
        changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
        assert np.all(changed <= (run.bounces > 0))
        assert run.collisions.sum() > 0
        assert np.all(run.overlaps() == 0)
        assert np.all(run.sites >= 0) and np.all(run.sites < 64)
        assert all(row["passed"] for row in result.invariants), result.invariants
    none = run_ball_gas(
        BallGasConfig(size=8, balls=8, max_speed=3, steps=60, rule="none")
    )
    assert none.run.collisions.sum() == 0
    energy = none.run.energy()
    assert np.all(energy == energy[0])


def test_radius_balls_stay_rigid_and_apart():
    config = BallGasConfig(size=12, balls=3, radius=1, max_speed=2, steps=80, seed=2)
    result = run_ball_gas(config)
    run = result.run
    assert run.walkers_per_ball == 5
    offsets = run.sites - run.sites[:, :, :1]  # a translation keeps the index offsets
    assert np.all(offsets == offsets[:1])
    # A ball never moves farther in a step than its velocity components allow.
    cells = np.stack([run.sites[:, :, 0] // 12, run.sites[:, :, 0] % 12], axis=-1)
    moved = np.abs(np.diff(cells, axis=0))
    assert np.all(moved <= np.abs(run.velocities[:-1])) or np.all(
        moved.sum(axis=2) <= np.abs(run.velocities[:-1]).sum(axis=2)
    )
    assert np.all(run.overlaps() == 0)


def test_invalid_setups_are_refused():
    gas = _gas()
    with pytest.raises(ValueError, match="overlap"):
        gas.place([[1, 1], [1, 1]], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="does not fit"):
        gas.place([[0, 0]], [[1, 0]], radius=1)
    with pytest.raises(ValueError, match="collision rule"):
        BallGas(SquareLattice(6), rule="bounce")
    with pytest.raises(ValueError, match="walls"):
        BallGas(SquareLattice(6, periodic=True))
    with pytest.raises(ValueError, match="velocities"):
        BallGasState(SquareLattice(6), np.array([[1]]), np.array([[1, 0, 0]]))
    with pytest.raises(ValueError, match="exactly 2 balls"):
        initial_condition(
            BallGasConfig(balls=3, placement="head-on"), np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="do not fit"):
        initial_condition(BallGasConfig(size=6, balls=30), np.random.default_rng(0))


def test_reference_starts_where_the_balls_start_and_both_conserve():
    comparison = run_ball_comparison(BallGasConfig(steps=40, balls=5, seed=1))
    gas, balls = comparison.gas, comparison.balls
    np.testing.assert_allclose(
        gas.run.trajectory.positions[0], balls.run.centres()[0], atol=1e-12
    )
    np.testing.assert_allclose(
        gas.run.velocities[0], balls.run.to_gas_run().velocities[0]
    )
    np.testing.assert_allclose(gas.run.times, balls.run.times)
    assert gas.report["energy"].passed and balls.report["energy"].passed
    assert gas.report["momentum"].passed and balls.report["momentum"].passed
    assert gas.config.radius == pytest.approx(0.5 / 10)


def test_live_balls_entry_point():
    result = live_balls({"size": 8, "balls": 4, "steps": 30, "max_speed": 2})
    assert result["error"] is None, result["error"]
    data, fragments = result["data"], result["fragments"]
    assert (
        len(data["graph"]["sites"]) == 31 and len(data["reference"]["positions"]) == 31
    )
    assert len(data["graph"]["captions"]) == 31
    assert all(row["passed"] for row in data["graph"]["invariants"])
    for key in (
        "energy_chart",
        "momentum_chart",
        "entropy_chart",
        "collisions_chart",
        "speed_chart",
    ):
        assert fragments[key].startswith("<svg"), key
    assert (
        "<table" in fragments["graph_checks_table"]
        and "<table" in fragments["comparison_table"]
    )
    assert "collision rule" in result["docs"]
    assert result["summary"]["rule"] == "exchange"
    demo = live_balls({"placement": "head-on", "balls": 2, "steps": 12})
    assert demo["error"] is None and demo["summary"]["collisions"] >= 1
    assert "collision rule must be one of" in live_balls({"rule": "magnet"})["error"]


def fold(x, length):
    """Reflect the unfolded coordinate ``x`` into ``[0, length]`` (a billiard on a segment);
    works for integers and floats."""
    period = 2 * length
    x = x % period
    return x if x <= length else period - x


def billiard_step(x, v, length):
    """Exact bounce of a point with integer velocity on ``[0, length]`` in one step."""
    unfolded = x + v
    return fold(unfolded, length), (v if unfolded % (2 * length) <= length else -v)


@pytest.mark.parametrize("radius", [0, 1])
def test_a_single_ball_follows_the_exact_billiard_in_every_case(radius):
    """Every start cell and every velocity with components up to 3, for several steps: the
    lattice ball must land where the mirror law puts it, with the mirror law's velocity and
    an unchanged speed. Catches a wrong direction after a wall or a corner bounce."""
    size = 7
    gas = _gas(size)
    length = size - 1 - 2 * radius  # the centre moves on [radius, size - 1 - radius]
    speeds = range(-3, 4)
    for ci in range(radius, size - radius):
        for cj in range(radius, size - radius):
            for vx in speeds:
                for vy in speeds:
                    state = gas.place([[ci, cj]], [[vx, vy]], radius=radius)
                    run = gas.run(state, 5)
                    x, y, ux, uy = ci - radius, cj - radius, vx, vy
                    for step in range(1, 6):
                        x, ux = billiard_step(x, ux, length)
                        y, uy = billiard_step(y, uy, length)
                        centre = (
                            run.sites[step, 0, 0] // size,
                            run.sites[step, 0, 0] % size,
                        )
                        # node 0 of a disc is its lowest-left point; the centre is at + radius
                        # in the first coordinate only for radius 0, so compare via the mean.
                        cells = np.stack(
                            [run.sites[step, 0] // size, run.sites[step, 0] % size],
                            axis=1,
                        ).mean(axis=0)
                        assert cells.tolist() == [x + radius, y + radius], (
                            ci,
                            cj,
                            vx,
                            vy,
                            step,
                            cells,
                            (x + radius, y + radius),
                            centre,
                        )
                        assert run.velocities[step, 0].tolist() == [ux, uy], (
                            ci,
                            cj,
                            vx,
                            vy,
                            step,
                        )
                    assert np.all(run.bounces >= 0)


def test_a_single_disk_in_the_reference_follows_the_exact_billiard():
    """The hard-disk reference with one disk: the recorded positions equal the continuous
    mirror law to rounding, the speed never changes, and no frame-to-frame jump exceeds what
    the velocity allows (a disk cannot vanish and reappear elsewhere)."""
    config = BallGasConfig(size=10, balls=1, steps=60)
    space = config.lattice
    gas = HardDiskGas(
        config.box, config.disk_radius, config.step_time / config.substeps
    )
    lower = np.asarray(config.box.lower) + config.disk_radius
    upper = np.asarray(config.box.upper) - config.disk_radius
    length = upper - lower
    for vx, vy in [(2, 0), (0, -3), (1, 1), (-2, 3), (4, -4), (3, 1)]:
        start = space.coordinates[space.node_index(4, 5)]
        velocity = np.array([vx, vy], dtype=float) * config.spacing / config.step_time
        run = gas.simulate(
            start[None, :],
            velocity[None, :],
            config.steps * config.substeps,
            config.substeps,
        )
        positions = run.trajectory.positions[:, 0, :]
        for frame, t in enumerate(run.times):
            expected = [
                lower[k] + fold(start[k] - lower[k] + velocity[k] * t, length[k])
                for k in range(2)
            ]
            np.testing.assert_allclose(
                positions[frame],
                expected,
                atol=1e-9,
                err_msg=f"{(vx, vy)} frame {frame}",
            )
        speeds = np.linalg.norm(run.velocities[:, 0, :], axis=1)
        np.testing.assert_allclose(speeds, speeds[0])
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert np.all(jumps <= speeds[0] * config.step_time + 1e-9)
        assert np.all(positions >= lower - 1e-9) and np.all(positions <= upper + 1e-9)


def test_reference_disks_stay_inside_and_keep_their_speeds_between_collisions():
    """Few disks: inside the box at every substep, speed of every disk unchanged in every
    interval without a collision, no jump larger than the velocity allows."""
    config = BallGasConfig(size=10, balls=4, steps=120, max_speed=3, seed=7)
    comparison = run_ball_comparison(config)
    run = comparison.gas.run
    positions = run.trajectory.positions
    lower = np.asarray(config.box.lower) + config.disk_radius
    upper = np.asarray(config.box.upper) - config.disk_radius
    assert np.all(positions >= lower - 1e-9) and np.all(positions <= upper + 1e-9)
    speeds = np.linalg.norm(run.velocities, axis=2)
    quiet = run.collisions == 0
    np.testing.assert_allclose(speeds[1:][quiet], speeds[:-1][quiet])
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    # No disk can ever be faster than one carrying the whole kinetic energy.
    fastest = np.sqrt(2 * run.kinetic_energy()[0] / run.mass)
    assert np.all(jumps <= fastest * config.step_time + 1e-9), np.argwhere(
        jumps > fastest * config.step_time
    )[:5]
    energy = run.kinetic_energy()
    np.testing.assert_allclose(energy, energy[0])
