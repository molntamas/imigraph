"""The lattice gas: balls with unit velocities under Reflect, Turn and Stream.

The random hop of :mod:`imigraph.graph.rules` has no velocity, so nothing is conserved from one
step to the next and two walkers never interact. Give every ball a unit velocity ``e_k`` (speed
``c = spacing / step_time``, kinetic energy ``m c^2 / 2`` whatever happens) and run the rule set

* **Reflect** ``{w,s,v}_k, {s,s}_k, {v,v'}|k -> {w,s,v'}``: a ball about to hop into a wall
  self-loop turns round; the wall takes the momentum, as a specular wall does in the hard-disk
  gas;
* **Turn** ``{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2, {v1,v2}|o, {v1,v1'}^, {v1',v2'}|o
  -> {w1,s,v1'}, {w2,t,v2'}``: two balls whose hops meet head-on leave at right angles, either
  sense at random. Momentum (zero before and after) and energy are conserved;
* **Swap** ``{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2 -> {w1,s,v2}, {w2,t,v1}``: any
  other pair whose hops meet swaps velocities, each ball continuing the other's motion from
  its own site (HPP's point particles would pass through each other and share the site;
  balls cannot, so they deflect);
* **Stream** ``{w,s,v}_k, {s,t}_k -> {w,t,v}``: every ball moves one edge along its velocity;

and you have the HPP lattice gas (Hardy, Pomeau, de Pazzis 1973) written as graph rewriting,
with one difference to the textbook model: the collision happens at the *contact*, when the
two hops would meet (on one site or crossing one edge), not after both balls have reached one
site, so head-on pairs at odd and even distance collide alike and no two balls share a site.
On a periodic lattice there are no self-loops, Reflect never matches and the total momentum is
conserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imigraph.graph.rules import REFLECT, STREAM, SWAP, TURN, BallRule
from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph
from imigraph.graph.velocity import DIRECTIONS
from imigraph.graph.world import WalkerSystem
from imigraph.reference.gas import GasRun

HPP_RULES = (REFLECT, TURN, SWAP, STREAM)


@dataclass(frozen=True)
class LatticeGas:
    """The lattice gas: a square lattice, the duration of one step and the walker mass, as a
    :class:`~imigraph.graph.world.WalkerSystem` with the HPP rule set."""

    space: SquareLattice
    step_time: float = 1.0
    mass: float = 1.0
    system: WalkerSystem = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.step_time <= 0 or self.mass <= 0:
            raise ValueError("step_time and mass must be > 0")
        object.__setattr__(
            self,
            "system",
            WalkerSystem(
                self.space,
                tuple(BallRule(spec) for spec in HPP_RULES),
                self.step_time,
                max_speed=1,
                mass=self.mass,
            ),
        )

    @property
    def speed(self) -> float:
        """Every walker moves one spacing per step: ``c = a / tau``."""
        return self.space.spacing / self.step_time

    def describe_rules(self) -> list[dict[str, str]]:
        return self.system.descriptions(self.space.spacing, self.step_time)[:-1]

    # ---------- initial conditions ----------

    def initial_half_box(self, count: int, rng: np.random.Generator) -> WalkerGraph:
        """``count`` balls on distinct random sites of the left half of the lattice, each
        with a random unit velocity."""
        size = self.space.size
        i = np.arange(self.space.num_nodes) // size
        left = np.flatnonzero(i < size / 2)
        if count > left.size:
            raise ValueError(f"{count} walkers do not fit into {left.size} sites")
        sites = rng.choice(left, size=count, replace=False)
        directions = rng.integers(0, 4, size=count)
        return WalkerGraph(self.space, sites, DIRECTIONS[directions])

    def state(self, sites: np.ndarray, directions: np.ndarray) -> WalkerGraph:
        """Balls at ``sites`` heading along the direction labels ``directions`` (0..3)."""
        return WalkerGraph(
            self.space, np.asarray(sites), DIRECTIONS[np.asarray(directions)]
        )

    # ---------- runs ----------

    def simulate(
        self,
        initial: WalkerGraph,
        steps: int,
        rng: np.random.Generator,
        record_every: int = 1,
    ) -> GasRun:
        """Run ``steps`` steps; the result has the shape of a hard-disk gas run (positions,
        velocities, collisions per interval), so the same checks apply to both."""
        history, _ = self.system.run(initial, steps, rng, record_every=record_every)
        return history.to_gas_run()

    def demo_dict(
        self, initial: WalkerGraph, steps: int, rng: np.random.Generator
    ) -> dict[str, Any]:
        """JSON-serialisable short run: space, per-frame anchors and velocities, events."""
        history, _ = self.system.run(initial, steps, rng)
        return history.to_dict()
