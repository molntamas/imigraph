import numpy as np
import pytest

from imigraph.trajectory import Box, Trajectory


def test_box_geometry():
    box = Box.square(4.0, lower=-1.0)
    np.testing.assert_allclose(box.side, [4.0, 4.0])
    np.testing.assert_allclose(box.center, [1.0, 1.0])
    assert box.contains(np.array([[0.0, 0.0], [3.0, -1.0]])).all()
    assert not box.contains(np.array([3.1, 0.0]))


def test_box_rejects_empty_extent():
    with pytest.raises(ValueError):
        Box((0.0, 0.0), (1.0, 0.0))


def test_trajectory_validates_shapes():
    with pytest.raises(ValueError):
        Trajectory(np.arange(3.0), np.zeros((2, 5, 2)), Box.square(1.0))
    with pytest.raises(ValueError):
        Trajectory(np.arange(2.0), np.zeros((2, 5, 3)), Box.square(1.0))


def test_msd_of_straight_line_motion():
    # Two particles moving with unit speed along x and y: |r(t) - r(0)|^2 = t^2 for both.
    times = np.arange(4.0)
    positions = np.zeros((4, 2, 2))
    positions[:, 0, 0] = times
    positions[:, 1, 1] = times
    trajectory = Trajectory(times, positions, Box.square(10.0))
    np.testing.assert_allclose(trajectory.msd(), times**2)
    np.testing.assert_allclose(trajectory.msd_standard_error(), 0.0)


def test_subsample_keeps_first_frame():
    trajectory = Trajectory(np.arange(7.0), np.zeros((7, 1, 2)), Box.square(1.0))
    sub = trajectory.subsample(3)
    np.testing.assert_array_equal(sub.times, [0.0, 3.0, 6.0])
    assert sub.num_frames == 3


def test_json_round_trip():
    rng = np.random.default_rng(0)
    trajectory = Trajectory(np.arange(3.0), rng.random((3, 4, 2)), Box.square(1.0))
    restored = Trajectory.from_dict(trajectory.to_dict())
    np.testing.assert_array_equal(restored.times, trajectory.times)
    np.testing.assert_array_equal(restored.positions, trajectory.positions)
    assert restored.box == trajectory.box
