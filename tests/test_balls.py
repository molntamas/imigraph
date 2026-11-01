"""The ball gas: elastic collisions as graph rules, exact invariants, the single ball as a
billiard, and the hard-disk reference."""

import numpy as np
import pytest

from imigraph.experiments.balls import (
    BallGasConfig,
    initial_condition,
    run_ball_comparison,
    run_ball_gas,
)
from imigraph.graph import SquareLattice, WalkerGraph, WalkerSystem
from imigraph.graph.balls import COLLISION_RULES, collision_rules, place
from imigraph.graph.rules import REFLECT, STREAM, rule_item
from imigraph.reference.gas import HardDiskGas
from imigraph.report.live import balls as live_balls


def _system(size=9, rule="exchange", max_speed=None):
    lattice = SquareLattice(size)
    return WalkerSystem(
        lattice, tuple(rule_item(s) for s in collision_rules(rule)), max_speed=max_speed
    )


def _run(system, state, steps, seed=0):
    history, _ = system.run(state, steps, np.random.default_rng(seed))
    return history


def test_head_on_collision_exchanges_the_velocities_exactly():
    system = _system()
    state = place(system.space, [[1, 4], [7, 4]], [[2, 0], [-1, 0]])
    run = _run(system, state, 6)
    assert run.collisions().sum() == 1 and run.collision_rules() == ["Exchange"]
    step = int(np.flatnonzero(run.collisions())[0]) + 1
    # Before the hit: (2, 0) and (-1, 0); after: the x components are exchanged.
    assert run.velocities[step].tolist() == [[-1, 0], [2, 0]]
    assert np.all(run.overlaps() == 0)
    event = next(e for e in run.events[step - 1] if e["kind"] == "collision")
    assert event["axis"] == "x" and event["rule"] == "Exchange"
    assert "Exchange: balls 0 and 1 met along x" in run.captions()[step]
    assert run.bounces().sum() >= 1 and "Reflect: ball" in " ".join(run.captions())


def test_diagonal_hit_is_exact_when_even_and_falls_through_to_swap_when_odd():
    system = _system()
    # Both balls hop into (4,4) in the first round, from the left and from below: the line of
    # centres is the diagonal d = (1, -1). With (1,0) and (0,1): (va - vb) . d = 2, even: the
    # rectangle corners are lattice nodes, Exchange gives (0,1) and (1,0).
    even = place(system.space, [[3, 4], [4, 3]], [[1, 0], [0, 1]])
    run = _run(system, even, 4)
    assert run.velocities[1].tolist() == [[0, 1], [1, 0]]
    assert run.collision_rules() == ["Exchange"]
    # With (2,0) and (0,1): (va - vb) . d = 3, odd: no corner on the lattice, Exchange does
    # not match and Swap takes the pair.
    odd = place(system.space, [[3, 4], [4, 3]], [[2, 0], [0, 1]])
    run = _run(system, odd, 4)
    assert run.velocities[1].tolist() == [[0, 1], [2, 0]]
    assert run.collision_rules() == ["Swap"]
    assert np.all(run.overlaps() == 0)
    assert "Swap: balls 0 and 1 met along diagonal" in " ".join(run.captions())


def test_a_moving_ball_hits_a_resting_one_and_hands_over_its_momentum():
    system = _system()
    state = place(system.space, [[1, 4], [5, 4]], [[1, 0], [0, 0]])
    run = _run(system, state, 6)
    step = int(np.flatnonzero(run.collisions())[0]) + 1
    assert run.velocities[step].tolist() == [[0, 0], [1, 0]]
    assert run.anchors[step, 0] == system.space.node_index(4, 4)
    assert np.all(run.overlaps() == 0)


@pytest.mark.parametrize("rule", COLLISION_RULES)
def test_rules_conserve_energy_and_momentum_exactly_and_keep_balls_apart(rule):
    config = BallGasConfig(size=8, balls=8, max_speed=3, steps=120, rule=rule, seed=4)
    result = run_ball_gas(config)
    run = result.run
    energy = run.energy()
    assert np.all(energy == energy[0])
    momentum = run.momentum()
    changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
    assert np.all(changed <= (run.bounces() > 0))
    assert run.collisions().sum() > 0
    assert np.all(run.overlaps() == 0)
    assert np.all(run.anchors >= 0) and np.all(run.anchors < 64)
    assert all(row["passed"] for row in result.invariants), result.invariants
    assert [r["name"] for r in result.rules][:-1] == [
        s.name for s in collision_rules(rule)
    ]


def test_without_a_collision_rule_meeting_balls_wait_instead_of_passing_through():
    rules = tuple(s.to_dict() for s in (REFLECT, STREAM))
    waiting = run_ball_gas(
        BallGasConfig(
            size=8, balls=8, max_speed=3, steps=60, rule="custom", rules=rules
        )
    )
    run = waiting.run
    assert run.collisions().sum() == 0
    energy = run.energy()
    assert np.all(energy == energy[0])
    assert np.all(run.overlaps() == 0)
    assert all(row["passed"] for row in waiting.invariants)
    # Head-on at distance 2 under Reflect + Stream: neither ball moves (both wait).
    system = _system(9, "swap")
    system = WalkerSystem(system.space, (rule_item(REFLECT), rule_item(STREAM)))
    state = place(system.space, [[3, 4], [5, 4]], [[1, 0], [-1, 0]])
    history = _run(system, state, 3)
    np.testing.assert_array_equal(
        history.anchors, np.broadcast_to(state.anchors, (4, 2))
    )
    assert history.velocities[-1].tolist() == [[1, 0], [-1, 0]]


def test_radius_balls_stay_rigid_and_apart():
    config = BallGasConfig(size=12, balls=3, radius=1, max_speed=2, steps=80, seed=2)
    result = run_ball_gas(config)
    run = result.run
    assert run.walkers_per_ball == 5
    sites = run.sites()
    offsets = sites - sites[:, :, :1]  # a translation keeps the index offsets
    assert np.all(offsets == offsets[:1])
    cells = np.stack([run.anchors // 12, run.anchors % 12], axis=-1)
    moved = np.abs(np.diff(cells, axis=0))
    assert np.all(moved.sum(axis=2) <= np.abs(run.velocity[:-1]).sum(axis=2))
    assert np.all(run.overlaps() == 0)
    assert run.collisions().sum() + run.bounces().sum() > 0


def test_invalid_setups_are_refused():
    lattice = SquareLattice(9)
    with pytest.raises(ValueError, match="overlap"):
        place(lattice, [[1, 1], [1, 1]], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="does not fit"):
        place(lattice, [[0, 0]], [[1, 0]], radius=1)
    with pytest.raises(ValueError, match="collision rule"):
        collision_rules("bounce")
    with pytest.raises(ValueError, match="collision rule"):
        BallGasConfig(rule="bounce")
    with pytest.raises(ValueError, match="at least one rule"):
        BallGasConfig(rule="custom")
    with pytest.raises(ValueError, match="velocities"):
        WalkerGraph(lattice, np.array([1]), np.array([[1, 0, 0]]))
    with pytest.raises(ValueError, match="exactly 2 balls"):
        initial_condition(
            BallGasConfig(balls=3, placement="head-on"), np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="do not fit"):
        initial_condition(BallGasConfig(size=6, balls=30), np.random.default_rng(0))
    assert COLLISION_RULES == ("exchange", "swap", "hpp")


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


def test_a_custom_rule_set_runs_against_the_reference():
    rules = [s.to_dict() for s in collision_rules("swap")]
    config = BallGasConfig(steps=30, balls=4, rule="custom", rules=tuple(rules))
    assert config.rule_names == "Reflect + Swap + Stream"
    comparison = run_ball_comparison(config)
    assert comparison.balls.run.collision_rules() in ([], ["Swap"])
    assert comparison.balls.report["energy"].passed


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
        "rules_table",
    ):
        assert fragments[key].startswith("<svg") or "<table" in fragments[key], key
    assert (
        "<table" in fragments["graph_checks_table"]
        and "<table" in fragments["comparison_table"]
    )
    assert "collision rule set" in result["docs"]
    assert result["summary"]["rules"] == "Reflect + Exchange + Swap + Stream"
    assert result["summary"]["collisions"] > 0
    assert [r["name"] for r in result["rules"]][:4] == [
        "Reflect",
        "Exchange",
        "Swap",
        "Stream",
    ]
    assert all(r["pattern"].startswith("<svg") for r in result["rules"][:4])
    demo = live_balls({"placement": "head-on", "balls": 2, "steps": 12})
    assert demo["error"] is None and demo["summary"]["collisions"] >= 1
    assert "collision rule must be one of" in live_balls({"rule": "magnet"})["error"]
    custom = live_balls(
        {
            "rule": "custom",
            "rules": [s.to_dict() for s in collision_rules("hpp")],
            "balls": 6,
            "steps": 20,
        }
    )
    assert custom["error"] is None and custom["summary"]["rules"] == (
        "Reflect + Turn + Swap + Stream"
    )


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
    """Every start cell and every velocity with components up to 3, for several steps, under
    Reflect and Stream: the lattice ball must land where the mirror law puts it, with the
    mirror law's velocity and an unchanged speed. Catches a wrong direction after a wall or a
    corner bounce."""
    size = 7
    system = _system(size, "swap", max_speed=3)
    length = size - 1 - 2 * radius  # the centre moves on [radius, size - 1 - radius]
    speeds = range(-3, 4)
    for ci in range(radius, size - radius):
        for cj in range(radius, size - radius):
            for vx in speeds:
                for vy in speeds:
                    state = place(system.space, [[ci, cj]], [[vx, vy]], radius=radius)
                    run = _run(system, state, 5)
                    x, y, ux, uy = ci - radius, cj - radius, vx, vy
                    for step in range(1, 6):
                        x, ux = billiard_step(x, ux, length)
                        y, uy = billiard_step(y, uy, length)
                        cell = (
                            run.anchors[step, 0] // size,
                            run.anchors[step, 0] % size,
                        )
                        assert cell == (x + radius, y + radius), (
                            ci,
                            cj,
                            vx,
                            vy,
                            step,
                        )
                        assert run.velocities[step, 0].tolist() == [ux, uy], (
                            ci,
                            cj,
                            vx,
                            vy,
                            step,
                        )
                    assert np.all(run.bounces() >= 0)


def test_free_motion_is_a_straight_line_and_the_invariants_hold_over_a_long_run():
    system = WalkerSystem(
        SquareLattice(12, spacing=0.5),
        tuple(rule_item(s) for s in collision_rules("swap")),
        step_time=0.25,
        mass=2.0,
    )
    ball = place(system.space, [[5, 6]], [[2, -1]], radius=1)
    run = _run(system, ball, 300)
    assert run.walkers_per_ball == 5
    energy = run.energy()
    assert np.all(energy == energy[0])
    assert energy[0] == pytest.approx(0.5 * 2.0 * 5 * 5 * (0.5 / 0.25) ** 2)
    momentum = run.momentum()
    np.testing.assert_allclose(
        np.linalg.norm(momentum, axis=1), np.linalg.norm(momentum[0])
    )
    changed = np.any(np.diff(momentum, axis=0) != 0, axis=1)
    np.testing.assert_array_equal(changed, run.bounces() > 0)
    np.testing.assert_allclose(run.configuration_entropy(6), np.log(1.0), atol=1e-12)
    coverage = run.coverage(6)
    assert np.all(np.diff(coverage) >= 0) and coverage[-1] > coverage[0]
    straight = _run(_system(9, "swap"), place(SquareLattice(9), [[1, 4]], [[1, 0]]), 16)
    x = straight.anchors[:, 0] // 9
    assert x[:8].tolist() == list(range(1, 9))
    assert x[8:16].tolist() == list(range(7, -1, -1))
    assert straight.bounces().tolist() == [0] * 7 + [1] + [0] * 7 + [1]


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
    # A disk may collide more than once in an interval; the fastest it can ever move is
    # when it holds all the energy, sqrt(2 E / m).
    fastest = np.sqrt(2 * run.kinetic_energy()[0] / run.mass)
    assert np.all(jumps <= fastest * config.step_time + 1e-9), np.argwhere(
        jumps > fastest * config.step_time
    )[:5]
    energy = run.kinetic_energy()
    np.testing.assert_allclose(energy, energy[0])
