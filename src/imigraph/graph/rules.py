"""Rules: how a :class:`~imigraph.graph.state.WalkerGraph` evolves in one step.

A rule maps a state to the next state. Rules are written in the spirit of Wolfram-model rewriting
rules (a pattern of edges is replaced by another pattern), applied to every walker simultaneously
(synchronous update). All randomness comes from the ``rng`` passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from imigraph.graph.state import WalkerGraph


@runtime_checkable
class Rule(Protocol):
    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        """Return the state after one step."""
        ...


@dataclass(frozen=True)
class RandomHop:
    """The edge-rewriting rule ``{w, s}, {s, s'}  ->  {w, s'}, {s, s'}``.

    For every walker ``w`` anchored at space node ``s``, one of the edges ``{s, s'}`` at ``s`` is
    chosen uniformly at random (self-loops count, so a walker at the boundary may stay) and the
    walker edge is moved along it. On a lattice with spacing ``a`` and step time ``tau`` this is
    a random walk with diffusion coefficient ``D = a^2 / (4 tau)`` (see
    :func:`~imigraph.graph.simulation.lattice_hop_diffusion`).
    """

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        neighbors = state.space.neighbors
        choice = rng.integers(0, neighbors.shape[1], size=state.num_walkers)
        return WalkerGraph(state.space, neighbors[state.anchors, choice])
