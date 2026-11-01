"""The evolving graph: space nodes plus one *ball* per particle.

There is one kind of walker, the ball ``{w, s, v}``: a walker node ``w`` attached to the site
``s`` it sits on and to the node ``v`` of the velocity lattice that is its velocity (integer
cells per step, see :mod:`imigraph.graph.velocity`). A ball at rest sits on the origin ``o`` of
the velocity lattice; that is the random walker of the Brownian imitation. A ball with a unit
velocity is the walker of the lattice gas. A ball may have a *body*: the lattice points within
``radius`` of its site, which move rigidly with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imigraph.graph.space import Space
from imigraph.graph.velocity import disc_offsets

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class WalkerGraph:
    """State of a graph system.

    The graph has the space nodes ``0 .. num_nodes - 1`` with the space edges, the velocity
    lattice, and one walker node per ball, ``num_nodes + w``, joined to the space by the
    hyperedge ``{walker w, anchors[w], velocities[w]}``. Rules rewrite these hyperedges; the
    space and the velocity lattice are static. ``velocities`` may be omitted for balls at rest.
    """

    space: Space
    anchors: np.ndarray
    velocities: np.ndarray | None = None
    radius: int = 0

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchors, dtype=np.int64)
        if anchors.ndim != 1:
            raise ValueError("anchors must be a 1D array of node indices")
        if anchors.size and (
            anchors.min() < 0 or anchors.max() >= self.space.num_nodes
        ):
            raise ValueError("anchors must be valid node indices of the space")
        object.__setattr__(self, "anchors", anchors)
        if self.velocities is not None:
            velocities = np.asarray(self.velocities, dtype=np.int64)
            if velocities.shape != (anchors.shape[0], 2):
                raise ValueError("velocities must have shape (walkers, 2)")
            object.__setattr__(self, "velocities", velocities)
        if self.radius < 0:
            raise ValueError("radius must be >= 0")
        if self.radius > 0:
            self.sites()  # the bodies must fit into the space

    @property
    def num_walkers(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def velocity(self) -> np.ndarray:
        """The velocities, zeros for balls at rest; shape ``(walkers, 2)``."""
        if self.velocities is None:
            return np.zeros((self.num_walkers, 2), dtype=np.int64)
        return self.velocities

    @property
    def moving(self) -> bool:
        """Whether any ball has a velocity."""
        return self.velocities is not None and bool(np.any(self.velocities != 0))

    def walker_node(self, walker: int) -> int:
        """Node index of walker ``walker`` in the combined graph."""
        return self.space.num_nodes + walker

    def positions(self) -> np.ndarray:
        """Projection of the walkers to 2D, shape ``(num_walkers, 2)``: the ball centres."""
        return self.space.coordinates[self.anchors]

    centres = positions

    def sites(self) -> np.ndarray:
        """The body of every ball: its anchor plus the sites within ``radius``, shape
        ``(walkers, k)``; ``k = 1`` without a body."""
        if self.radius == 0:
            return self.anchors[:, None]
        return body_sites(self.space, self.anchors, self.radius)

    def walker_edges(self) -> np.ndarray:
        """The walker edges ``(walker node, anchor)``, shape ``(num_walkers, 2)``."""
        walkers = self.space.num_nodes + np.arange(self.num_walkers)
        return np.stack([walkers, self.anchors], axis=1)

    def occupancy(self) -> np.ndarray:
        """Number of walkers anchored at each space node, shape ``(num_nodes,)``."""
        return np.bincount(self.anchors, minlength=self.space.num_nodes)

    def to_networkx(self) -> nx.Graph:
        """Combined graph with node attribute ``kind`` (``"space"`` or ``"walker"``) and the
        velocity of every walker as its attribute ``velocity``."""
        graph = self.space.to_networkx()
        for node in graph.nodes:
            graph.nodes[node]["kind"] = "space"
        velocity = self.velocity
        for walker, anchor in enumerate(self.anchors):
            node = self.walker_node(walker)
            graph.add_node(
                node, kind="walker", walker=walker, velocity=tuple(velocity[walker])
            )
            graph.add_edge(node, int(anchor))
        return graph


def lattice_cells(space: Space, sites: np.ndarray) -> np.ndarray:
    """Integer cell coordinates ``(i, j)`` of sites of a square lattice, any shape."""
    size = getattr(space, "size", None)
    if size is None:
        raise ValueError("bodies need a square lattice")
    sites = np.asarray(sites, dtype=np.int64)
    return np.stack([sites // size, sites % size], axis=-1)


def body_fits(space: Space, anchors: np.ndarray, radius: int) -> np.ndarray:
    """Whether the body of ``radius`` centred at every anchor lies inside the lattice."""
    if radius == 0:
        return np.ones(np.asarray(anchors).shape[0], dtype=bool)
    size = getattr(space, "size", None)
    if size is None:
        raise ValueError("bodies need a square lattice")
    offsets = disc_offsets(radius)
    cells = lattice_cells(space, anchors)[:, None, :] + offsets[None, :, :]
    return np.all((cells >= 0) & (cells < size), axis=(1, 2))


def body_sites(space: Space, anchors: np.ndarray, radius: int) -> np.ndarray:
    """The body sites of balls of ``radius`` centred at ``anchors``, shape ``(n, k)``.
    Raises ``ValueError`` when a body does not fit into the lattice."""
    size = getattr(space, "size", None)
    if size is None:
        raise ValueError("bodies need a square lattice")
    offsets = disc_offsets(radius)
    if 2 * radius + 1 > size:
        raise ValueError("the balls must be smaller than the room")
    cells = lattice_cells(space, anchors)[:, None, :] + offsets[None, :, :]
    if cells.size and (cells.min() < 0 or cells.max() >= size):
        raise ValueError(
            f"a ball of radius {radius} does not fit at every centre of the "
            f"{size}x{size} room"
        )
    return cells[:, :, 0] * size + cells[:, :, 1]
