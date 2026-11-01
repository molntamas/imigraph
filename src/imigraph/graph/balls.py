"""The ball gas: balls with integer velocities bouncing off the walls and colliding by the
rules of the library (:mod:`imigraph.graph.rules`), the graph imitation of a hard-disk gas.

A *collision rule set* names what happens at a contact (two balls whose next hops meet):

* ``exchange``: Reflect, Exchange, Swap, Stream. The elastic collision of equal masses: the
  velocity components along the line of centres are exchanged (Exchange, exact along an axis
  and for even diagonal hits); when the exact result is not a lattice velocity Exchange does
  not match and Swap takes the pair, which conserves momentum and energy but scatters at the
  wrong angle. Swap's collisions are counted separately.
* ``swap``: Reflect, Swap, Stream. Every hit is treated as head-on: the balls exchange
  their velocities and each continues the other's motion from its own site.
* ``hpp``: Reflect, Turn, Swap, Stream. Head-on pairs turn by a quarter turn (the lattice
  gas); every other contact swaps, where HPP's point particles would pass through each other
  (balls cannot share a site, so they deflect instead).

A contact that no rule resolves makes the balls wait (they do not hop in that round), so no
two balls ever share a site. Momentum and kinetic energy are integers times ``m a / tau`` and
``m a^2 / tau^2``, conserved exactly by every collision rule; the walls exchange momentum and
leave the energy unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from imigraph.graph.rules import EXCHANGE, REFLECT, STREAM, SWAP, TURN, RuleSpec
from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph

COLLISION_RULES = ("exchange", "swap", "hpp")
COLLISION_RULE_SETS: dict[str, tuple[RuleSpec, ...]] = {
    "exchange": (REFLECT, EXCHANGE, SWAP, STREAM),
    "swap": (REFLECT, SWAP, STREAM),
    "hpp": (REFLECT, TURN, SWAP, STREAM),
}
COLLISION_RULE_TEXT = {
    "exchange": "exchange: Reflect, Exchange, Swap, Stream; the velocity component along the "
    "line of centres is exchanged (elastic collision of equal masses), with Swap as the "
    "fallback when that is not a lattice velocity (an odd diagonal hit)",
    "swap": "swap: Reflect, Swap, Stream; the balls always exchange their whole velocities "
    "(every hit treated as head-on)",
    "hpp": "hpp: Reflect, Turn, Swap, Stream; head-on pairs turn by a quarter turn (the HPP "
    "lattice gas), other contacts swap instead of passing through",
}


def collision_rules(name: str) -> tuple[RuleSpec, ...]:
    if name not in COLLISION_RULE_SETS:
        raise ValueError(
            f"collision rule must be one of {', '.join(COLLISION_RULES)}, not {name!r}"
        )
    return COLLISION_RULE_SETS[name]


def place(
    space: SquareLattice,
    centres: ArrayLike,
    velocities: ArrayLike,
    radius: int = 0,
    overlap: bool = False,
) -> WalkerGraph:
    """Balls of ``radius`` at integer ``centres`` ``(n, 2)`` with ``velocities`` ``(n, 2)``;
    refuses bodies that share a site unless ``overlap`` is allowed."""
    centres = np.asarray(centres, dtype=np.int64).reshape(-1, 2)
    velocities = np.asarray(velocities, dtype=np.int64).reshape(-1, 2)
    if centres.shape[0] != velocities.shape[0]:
        raise ValueError("give one velocity per ball")
    size = space.size
    if centres.size and (centres.min() < 0 or centres.max() >= size):
        raise ValueError(f"a centre is outside the {size}x{size} room")
    state = WalkerGraph(space, centres[:, 0] * size + centres[:, 1], velocities, radius)
    sites = state.sites()
    if not overlap and np.unique(sites).size != sites.size:
        raise ValueError("the balls overlap at the start: use distinct sites")
    return state
