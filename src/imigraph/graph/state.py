"""The evolving graph: space nodes plus walker nodes, each walker attached by one edge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imigraph.graph.space import Space

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class WalkerGraph:
    """State of a graph system.

    The graph has the space nodes ``0 .. num_nodes - 1`` with the space edges, and one *walker*
    node per particle, ``num_nodes + w``, joined to the space by the single edge
    ``{walker w, anchors[w]}``. Rules rewrite these walker edges; the space is static.
    """

    space: Space
    anchors: np.ndarray

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchors, dtype=np.int64)
        if anchors.ndim != 1:
            raise ValueError("anchors must be a 1D array of node indices")
        if anchors.size and (
            anchors.min() < 0 or anchors.max() >= self.space.num_nodes
        ):
            raise ValueError("anchors must be valid node indices of the space")
        object.__setattr__(self, "anchors", anchors)

    @property
    def num_walkers(self) -> int:
        return int(self.anchors.shape[0])

    def walker_node(self, walker: int) -> int:
        """Node index of walker ``walker`` in the combined graph."""
        return self.space.num_nodes + walker

    def positions(self) -> np.ndarray:
        """Projection of the walkers to 2D, shape ``(num_walkers, 2)``."""
        return self.space.coordinates[self.anchors]

    def walker_edges(self) -> np.ndarray:
        """The walker edges ``(walker node, anchor)``, shape ``(num_walkers, 2)``."""
        walkers = self.space.num_nodes + np.arange(self.num_walkers)
        return np.stack([walkers, self.anchors], axis=1)

    def occupancy(self) -> np.ndarray:
        """Number of walkers anchored at each space node, shape ``(num_nodes,)``."""
        return np.bincount(self.anchors, minlength=self.space.num_nodes)

    def to_networkx(self) -> nx.Graph:
        """Combined graph with node attribute ``kind`` (``"space"`` or ``"walker"``)."""
        graph = self.space.to_networkx()
        for node in graph.nodes:
            graph.nodes[node]["kind"] = "space"
        for walker, anchor in enumerate(self.anchors):
            node = self.walker_node(walker)
            graph.add_node(node, kind="walker", walker=walker)
            graph.add_edge(node, int(anchor))
        return graph
