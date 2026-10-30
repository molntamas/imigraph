"""Space graphs: the static part of a graph system that walkers move on.

A space is a graph whose nodes carry 2D coordinates (the *projection* of the graph system) and
whose edges are stored as a dense neighbour table, so that rules can be applied to all walkers at
once with numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from imigraph.trajectory import Box

if TYPE_CHECKING:
    import networkx as nx


@runtime_checkable
class Space(Protocol):
    """What rules and projections need from a space graph."""

    @property
    def num_nodes(self) -> int: ...

    @property
    def coordinates(self) -> np.ndarray:
        """Projection of every node to 2D, shape ``(num_nodes, 2)``."""
        ...

    @property
    def neighbors(self) -> np.ndarray:
        """Neighbour table, shape ``(num_nodes, degree)``.

        ``neighbors[i, k]`` is the node at the other end of the ``k``-th edge at node ``i``; a
        self-loop is encoded as ``neighbors[i, k] == i``. Every node has the same degree.
        """
        ...

    @property
    def box(self) -> Box:
        """Box that contains the projection of every node (used by the comparison)."""
        ...

    def edges(self) -> np.ndarray:
        """Undirected edge list without self-loops, shape ``(num_edges, 2)``."""
        ...

    def to_networkx(self) -> nx.Graph: ...


@dataclass(frozen=True)
class SquareLattice:
    """``size x size`` square lattice with spacing ``spacing``; node ``(i, j)`` sits at
    ``(i * spacing, j * spacing)``.

    Boundary nodes get self-loops for their missing neighbours, so every node has degree 4 and a
    random hop stays in place with the probability a reflected Brownian step would bounce back.
    This makes the uniform distribution over nodes stationary, like the uniform density of
    Brownian motion in a box.

    The box of the projection extends ``spacing / 2`` beyond the outer nodes, so every node owns a
    cell of area ``spacing^2``.
    """

    size: int
    spacing: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if self.spacing <= 0:
            raise ValueError("spacing must be > 0")

    @property
    def num_nodes(self) -> int:
        return self.size * self.size

    @property
    def degree(self) -> int:
        return 4

    def node_index(self, i: int, j: int) -> int:
        """Node index of lattice site ``(i, j)``."""
        if not (0 <= i < self.size and 0 <= j < self.size):
            raise ValueError(f"site ({i}, {j}) outside a lattice of size {self.size}")
        return i * self.size + j

    @property
    def center_node(self) -> int:
        """The central site (for even sizes the lower-left of the four central sites)."""
        middle = (self.size - 1) // 2
        return self.node_index(middle, middle)

    @cached_property
    def coordinates(self) -> np.ndarray:
        i, j = np.divmod(np.arange(self.num_nodes), self.size)
        return self.spacing * np.stack([i, j], axis=1).astype(float)

    @cached_property
    def neighbors(self) -> np.ndarray:
        i, j = np.divmod(np.arange(self.num_nodes), self.size)
        last = self.size - 1
        # Order: +i, -i, +j, -j. A missing neighbour at the boundary becomes a self-loop.
        table = np.stack(
            [
                np.where(i < last, i + 1, i) * self.size + j,
                np.where(i > 0, i - 1, i) * self.size + j,
                i * self.size + np.where(j < last, j + 1, j),
                i * self.size + np.where(j > 0, j - 1, j),
            ],
            axis=1,
        )
        table.setflags(write=False)
        return table

    @property
    def box(self) -> Box:
        half = self.spacing / 2
        upper = (self.size - 1) * self.spacing + half
        return Box((-half, -half), (upper, upper))

    def edges(self) -> np.ndarray:
        source = np.repeat(np.arange(self.num_nodes), self.degree)
        target = self.neighbors.ravel()
        keep = source < target  # each undirected edge once, self-loops dropped
        return np.stack([source[keep], target[keep]], axis=1)

    def to_networkx(self) -> nx.Graph:
        """The lattice as a networkx graph with ``pos`` node attributes (self-loops included)."""
        import networkx as nx

        graph = nx.Graph()
        for node, (x, y) in enumerate(self.coordinates):
            graph.add_node(node, pos=(float(x), float(y)))
        graph.add_edges_from(map(tuple, self.edges()))
        loops = np.flatnonzero(np.any(self.neighbors == np.arange(self.num_nodes)[:, None], axis=1))
        graph.add_edges_from((int(node), int(node)) for node in loops)
        return graph
