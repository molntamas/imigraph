"""A lattice gas on the walker graph: rules that conserve mass, momentum and energy exactly.

The random hop of :mod:`imigraph.graph.rules` has no velocity, so nothing is conserved from one
step to the next and two walkers never interact. Here a walker is the hyperedge ``{w, s, s'}``:
it sits on site ``s`` and *heads* for the neighbour ``s'`` along an edge with direction label
``k`` (the column of the neighbour table: ``+x, -x, +y, -y``). Its velocity is ``c e_k`` with
``c = spacing / step_time``, its kinetic energy ``m c^2 / 2`` whatever happens. This is the HPP
lattice gas (Hardy, Pomeau, de Pazzis 1973) written as graph-rewriting rules.

Exclusion principle: at most one walker per channel ``(s, k)``. Every step applies, to all walkers
at once and in this order,

* **Collide** ``{w1,s,a}, {w2,s,b}, {s,c}, {s,d} -> {w1,s,c}, {w2,s,d}, {s,c}, {s,d}``: two
  walkers on one site heading in opposite directions ``a, b`` leave along the perpendicular pair
  ``c, d``, provided those channels are empty and are real edges (not self-loops). Momentum
  ``a + b = 0 = c + d`` and energy are conserved; which walker takes ``c`` is chosen at random.
* **Reflect** ``{w,s,s} -> {w,s,s'}``: a walker heading into a self-loop (the wall) turns round;
  the wall takes the momentum, as a specular wall does in the hard-disk gas.
* **Stream** ``{w,s,s'}_k, {s',s''}_k -> {w,s',s''}_k``: every walker moves one edge and keeps
  its direction label.

On a periodic lattice there are no self-loops, Reflect never matches and the total momentum is
conserved exactly. The rules preserve the exclusion principle, so the synchronous update never
has to resolve conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph
from imigraph.reference.gas import GasRun
from imigraph.trajectory import Trajectory

DIRECTIONS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
OPPOSITE = np.array([1, 0, 3, 2])
# Head-on pairs and the perpendicular pair they turn into.
COLLISION_PAIRS = ((0, 1, 2, 3), (2, 3, 0, 1))

RULES = [
    {
        "name": "Collide",
        "rewrite": "{w1,s,a}, {w2,s,b}, {s,c}, {s,d} -> {w1,s,c}, {w2,s,d}, {s,c}, {s,d}",
        "matches": "a head-on pair at s (b opposite to a) whose perpendicular channels c, d are "
        "empty and real edges",
        "choice": "which walker takes c is chosen uniformly; all pairs rewritten at once",
        "invariants": "walkers, momentum (a + b = 0 = c + d), energy (speed c = a / tau), "
        "exclusion (one walker per channel)",
        "physics": "the HPP collision: the only interaction of the lattice gas",
    },
    {
        "name": "Reflect",
        "rewrite": "{w,s,s} -> {w,s,s'}  (s' the neighbour opposite to the self-loop)",
        "matches": "a walker heading into a self-loop, i.e. into the wall",
        "choice": "deterministic",
        "invariants": "walkers, energy; momentum is exchanged with the wall",
        "physics": "specular wall; never matches on a periodic lattice",
    },
    {
        "name": "Stream",
        "rewrite": "{w,s,s'}_k, {s',s''}_k -> {w,s',s''}_k",
        "matches": "every walker, with the edge of the same direction label k at its target",
        "choice": "deterministic",
        "invariants": "walkers, momentum, energy, exclusion",
        "physics": "free flight by one lattice spacing per step: speed c = a / tau",
    },
]


@dataclass(frozen=True)
class LatticeGasState:
    """Walkers ``{w, s, s'}``: ``sites[w] = s`` and ``s' = neighbors[s, directions[w]]``."""

    space: SquareLattice
    sites: np.ndarray
    directions: np.ndarray

    def __post_init__(self) -> None:
        sites = np.asarray(self.sites, dtype=np.int64)
        directions = np.asarray(self.directions, dtype=np.int64)
        if sites.shape != directions.shape or sites.ndim != 1:
            raise ValueError("sites and directions must be 1D arrays of the same length")
        if sites.size and (sites.min() < 0 or sites.max() >= self.space.num_nodes):
            raise ValueError("sites must be valid node indices")
        if directions.size and (directions.min() < 0 or directions.max() > 3):
            raise ValueError("directions must be in 0..3 (+x, -x, +y, -y)")
        channels = sites * 4 + directions
        if np.unique(channels).size != channels.size:
            raise ValueError("exclusion principle violated: two walkers in one channel")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "directions", directions)

    @property
    def num_walkers(self) -> int:
        return int(self.sites.shape[0])

    def headings(self) -> np.ndarray:
        """The target site ``s'`` of every walker."""
        return self.space.neighbors[self.sites, self.directions]

    def as_walker_graph(self) -> WalkerGraph:
        """The positions only (the heading dropped), for projections and drawing."""
        return WalkerGraph(self.space, self.sites)

    def velocities(self, speed: float) -> np.ndarray:
        return speed * DIRECTIONS[self.directions]


@dataclass(frozen=True)
class LatticeGas:
    """The lattice gas: a square lattice, the duration of one step and the walker mass."""

    space: SquareLattice
    step_time: float = 1.0
    mass: float = 1.0

    def __post_init__(self) -> None:
        if self.step_time <= 0 or self.mass <= 0:
            raise ValueError("step_time and mass must be > 0")

    @property
    def speed(self) -> float:
        """Every walker moves one spacing per step: ``c = a / tau``."""
        return self.space.spacing / self.step_time

    def describe_rules(self) -> list[dict[str, str]]:
        return [dict(rule) for rule in RULES]

    # ---------- initial conditions ----------

    def initial_half_box(self, count: int, rng: np.random.Generator) -> LatticeGasState:
        """``count`` walkers in random channels of the left half of the lattice (channels
        leading away from a wall excluded, see the comment)."""
        size = self.space.size
        i = np.arange(self.space.num_nodes) // size
        left = np.flatnonzero(i < size / 2)
        channels = (left[:, None] * 4 + np.arange(4)[None, :]).ravel()
        # A channel that points away from a wall can only be reached by Reflect, which streams
        # the walker on in the same step; starting a walker there could collide with a
        # reflected one, so such channels are left empty.
        nodes, directions = channels // 4, channels % 4
        from_wall = self.space.neighbors[nodes, OPPOSITE[directions]] == nodes
        channels = channels[~from_wall]
        if count > channels.size:
            raise ValueError(f"{count} walkers do not fit into {channels.size} channels")
        chosen = rng.choice(channels, size=count, replace=False)
        return LatticeGasState(self.space, chosen // 4, chosen % 4)

    # ---------- rules ----------

    def collide(
        self, state: LatticeGasState, rng: np.random.Generator
    ) -> tuple[LatticeGasState, np.ndarray]:
        """Apply Collide to every head-on pair; returns the new state and the sites where a
        pair collided."""
        space = state.space
        occupant = np.full((space.num_nodes, 4), -1, dtype=np.int64)
        occupant[state.sites, state.directions] = np.arange(state.num_walkers)
        occupied = occupant >= 0
        real = space.neighbors != np.arange(space.num_nodes)[:, None]
        directions = state.directions.copy()
        collided: list[np.ndarray] = []
        for a, b, c, d in COLLISION_PAIRS:
            sites = np.flatnonzero(
                occupied[:, a]
                & occupied[:, b]
                & ~occupied[:, c]
                & ~occupied[:, d]
                & real[:, c]
                & real[:, d]
            )
            if sites.size == 0:
                continue
            swap = rng.integers(0, 2, size=sites.size).astype(bool)
            directions[occupant[sites, a]] = np.where(swap, d, c)
            directions[occupant[sites, b]] = np.where(swap, c, d)
            collided.append(sites)
        sites = np.concatenate(collided) if collided else np.zeros(0, dtype=np.int64)
        return replace(state, directions=directions), sites

    def reflect(self, state: LatticeGasState) -> LatticeGasState:
        """Apply Reflect: walkers heading into a self-loop turn round."""
        wall = state.headings() == state.sites
        return replace(
            state, directions=np.where(wall, OPPOSITE[state.directions], state.directions)
        )

    def stream(self, state: LatticeGasState) -> LatticeGasState:
        """Apply Stream: every walker moves to its heading, keeping its direction label."""
        return replace(state, sites=state.headings())

    def step(
        self, state: LatticeGasState, rng: np.random.Generator
    ) -> tuple[LatticeGasState, np.ndarray]:
        """One step: Collide, Reflect, Stream. Returns the new state and the collision sites."""
        state, collided = self.collide(state, rng)
        return self.stream(self.reflect(state)), collided

    # ---------- runs ----------

    def simulate(
        self,
        initial: LatticeGasState,
        steps: int,
        rng: np.random.Generator,
        record_every: int = 1,
    ) -> GasRun:
        """Run ``steps`` steps; the result has the shape of a hard-disk gas run (positions,
        velocities, collisions per interval), so the same checks apply to both."""
        if initial.space != self.space:
            raise ValueError("initial state must live on the gas's lattice")
        if steps < 0 or record_every < 1:
            raise ValueError("steps must be >= 0 and record_every >= 1")
        coordinates = self.space.coordinates
        state = initial
        frames, velocities, collisions = (
            [coordinates[state.sites]],
            [state.velocities(self.speed)],
            [],
        )
        interval = 0
        for step in range(1, steps + 1):
            state, collided = self.step(state, rng)
            interval += int(collided.size)
            if step % record_every == 0:
                frames.append(coordinates[state.sites])
                velocities.append(state.velocities(self.speed))
                collisions.append(interval)
                interval = 0
        times = self.step_time * record_every * np.arange(len(frames))
        trajectory = Trajectory(times, np.stack(frames), self.space.box)
        return GasRun(trajectory, np.stack(velocities), np.asarray(collisions), self.mass)

    def simulate_states(
        self, initial: LatticeGasState, steps: int, rng: np.random.Generator
    ) -> tuple[list[LatticeGasState], list[np.ndarray]]:
        """Every state of a short run and the collision sites of every step (for the
        step-by-step animation)."""
        states, collisions = [initial], []
        for _ in range(steps):
            state, collided = self.step(states[-1], rng)
            states.append(state)
            collisions.append(collided)
        return states, collisions

    def demo_dict(self, initial: LatticeGasState, steps: int, rng: np.random.Generator) -> dict:
        """JSON-serialisable short run: space, per-frame sites and directions, collision sites."""
        states, collisions = self.simulate_states(initial, steps, rng)
        space = self.space
        return {
            "sites": [state.sites.tolist() for state in states],
            "directions": [state.directions.tolist() for state in states],
            "collisions": [sites.tolist() for sites in collisions],
            "space": {
                "coordinates": np.round(space.coordinates, 4).tolist(),
                "edges": space.edges().tolist(),
                "box": space.box.to_dict(),
                "spacing": float(space.spacing),
                "neighbors": space.neighbors.tolist(),
            },
        }
