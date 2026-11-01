"""The velocity lattice: the second graph of a ball.

A ball's velocity is an integer vector ``(vx, vy)`` in cells per step, a node of the *velocity
lattice*, the square lattice of integer vectors with components up to ``max_speed``. Its edges
carry the same direction labels as the room (``+x, -x, +y, -y``): the edge of label ``k`` leads
from ``v`` to ``v + e_k``. The origin ``o = (0, 0)`` is the velocity of a ball at rest.

Rules relate velocity nodes by the geometry of this lattice (see :mod:`imigraph.graph.rules`):
*lines* along a lattice direction (an axis or a diagonal), *ahead* along a direction, the
*mirror* across an axis (what a wall does) and the *quarter turn* (what a head-on HPP collision
does). This module holds those elementary relations and the *hop order* of a velocity: the
labels of the ``|vx| + |vy|`` hops a ball takes in one step, interleaved so that the lattice
path hugs the straight line (Bresenham order).
"""

from __future__ import annotations

import numpy as np

DIRECTION_NAMES = ("+x", "-x", "+y", "-y")
DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
OPPOSITE = np.array([1, 0, 3, 2])
AXES = ((0, 1), (2, 3))  # neighbour-table labels (+, -) of the x and the y axis
AXIS_NAMES = ("x", "y")
# The eight lattice directions a line of centres can have: the axes and the diagonals.
LATTICE_DIRECTIONS = np.array(
    [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
    dtype=np.int64,
)


def label_of(name: str) -> int:
    """``"+x"`` -> 0, ``"-x"`` -> 1, ``"+y"`` -> 2, ``"-y"`` -> 3."""
    if name not in DIRECTION_NAMES:
        raise ValueError(f"direction must be one of {', '.join(DIRECTION_NAMES)}")
    return DIRECTION_NAMES.index(name)


def axis_of(labels: np.ndarray) -> np.ndarray:
    """The axis (0 = x, 1 = y) of direction labels."""
    return np.asarray(labels) // 2


def direction_of(delta: np.ndarray) -> np.ndarray:
    """The lattice direction of displacement vectors: ``sign(delta)`` component by component,
    so a displacement that is not along an axis or a diagonal still gets the nearest of the
    eight lattice directions (the line of centres of two bodies)."""
    return np.sign(np.asarray(delta, dtype=np.int64)).astype(np.int64)


def perpendicular(direction: np.ndarray) -> np.ndarray:
    """The direction rotated by a quarter turn (``(dx, dy)`` -> ``(-dy, dx)``)."""
    d = np.asarray(direction, dtype=np.int64)
    return np.stack([-d[..., 1], d[..., 0]], axis=-1)


def mirror(velocity: np.ndarray, axis: np.ndarray | int) -> np.ndarray:
    """The velocity with the component of ``axis`` (0 = x, 1 = y) negated: what a wall
    perpendicular to that axis does."""
    v = np.array(velocity, dtype=np.int64, copy=True)
    axis = np.asarray(axis)
    if axis.ndim == 0:
        v[..., int(axis)] *= -1
    else:
        idx = np.arange(v.shape[0])
        v[idx, axis] *= -1
    return v


def turn(velocity: np.ndarray, sense: int = 1) -> np.ndarray:
    """The velocity rotated by a quarter turn, counter-clockwise for ``sense = 1``."""
    v = np.asarray(velocity, dtype=np.int64)
    if sense >= 0:
        return np.stack([-v[..., 1], v[..., 0]], axis=-1)
    return np.stack([v[..., 1], -v[..., 0]], axis=-1)


def velocity_box(max_speed: int) -> np.ndarray:
    """Every node of the velocity lattice: integer vectors with components in
    ``[-max_speed, max_speed]``, shape ``((2 max_speed + 1)^2, 2)``."""
    r = np.arange(-max_speed, max_speed + 1)
    i, j = np.meshgrid(r, r, indexing="ij")
    return np.stack([i.ravel(), j.ravel()], axis=1).astype(np.int64)


def hop_order(velocity: np.ndarray) -> list[int]:
    """The hops of one step as axis indices (0 = x, 1 = y), interleaved in Bresenham order so
    that the lattice path follows the straight line of the velocity."""
    nx, ny = int(abs(velocity[0])), int(abs(velocity[1]))
    order: list[int] = []
    if nx >= ny:
        error = 0
        for _ in range(nx):
            order.append(0)
            error += ny
            if 2 * error >= nx:
                order.append(1)
                error -= nx
        order += [1] * (ny - order.count(1))
    else:
        error = 0
        for _ in range(ny):
            order.append(1)
            error += nx
            if 2 * error >= ny:
                order.append(0)
                error -= ny
        order += [0] * (nx - order.count(0))
    return order


def hop_label(velocity: np.ndarray, axis: int) -> int:
    """The direction label of a hop along ``axis`` for ``velocity`` (its sign on that axis)."""
    plus, minus = AXES[axis]
    return plus if velocity[axis] > 0 else minus


def disc_offsets(radius: int) -> np.ndarray:
    """Node offsets of a ball's body: the lattice points with ``i^2 + j^2 <= r^2``, shape
    ``(k, 2)``; ``radius = 0`` is the single node."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    r = np.arange(-radius, radius + 1)
    i, j = np.meshgrid(r, r, indexing="ij")
    inside = i**2 + j**2 <= radius**2
    return np.stack([i[inside], j[inside]], axis=1).astype(np.int64)
