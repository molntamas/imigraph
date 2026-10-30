"""Running a graph system: initial condition + rule -> history of states -> projected trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from imigraph.graph.rules import RandomHop, Rule
from imigraph.graph.space import Space
from imigraph.graph.state import WalkerGraph
from imigraph.metrics import StepObserver
from imigraph.trajectory import Trajectory


def walkers_at_node(space: Space, node: int, count: int) -> WalkerGraph:
    """Initial condition: ``count`` walkers all anchored at ``node``."""
    return WalkerGraph(space, np.full(count, node, dtype=np.int64))


def lattice_hop_diffusion(spacing: float, step_time: float) -> float:
    """Diffusion coefficient of :class:`RandomHop` on a square lattice.

    Each step moves one coordinate by ``+-spacing`` (or stays at a self-loop), so after ``k``
    free steps ``<|r|^2> = k spacing^2``; with ``t = k step_time`` and ``<|r|^2> = 4 D t`` in two
    dimensions, ``D = spacing^2 / (4 step_time)``.
    """
    return spacing**2 / (4 * step_time)


@dataclass(frozen=True)
class GraphHistory:
    """Recorded states of a graph run: the walker anchors at every recorded step.

    ``anchors`` has shape ``(T, num_walkers)``; frame 0 is the initial condition. The space is
    static, so the full graph at frame ``k`` is :meth:`state` ``(k)``.
    """

    space: Space
    times: np.ndarray
    anchors: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        anchors = np.asarray(self.anchors, dtype=np.int64)
        if anchors.ndim != 2 or times.ndim != 1 or anchors.shape[0] != times.shape[0]:
            raise ValueError("anchors must have shape (T, n) matching T times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "anchors", anchors)

    @property
    def num_frames(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def num_walkers(self) -> int:
        return int(self.anchors.shape[1])

    def state(self, frame: int) -> WalkerGraph:
        return WalkerGraph(self.space, self.anchors[frame])

    def project(self, cell_rng: np.random.Generator | None = None) -> Trajectory:
        """The walkers' 2D positions over time: the graph system's imitation of the reference.

        By default a walker sits exactly on the coordinates of its node. With ``cell_rng`` each
        walker is instead placed at a fixed random offset inside the cell of its node (uniform in
        ``[-spacing/2, spacing/2]^2``, drawn once per walker): the graph then claims to imitate
        the reference *at the resolution of one cell*, which is the statement the distribution
        tests of the comparison framework check. Displacements, and therefore the mean squared
        displacement, are unaffected by the offset.
        """
        positions = self.space.coordinates[self.anchors]
        if cell_rng is not None:
            cell = self.space.cell_size
            offsets = cell_rng.uniform(-0.5, 0.5, size=(self.num_walkers, 2)) * cell
            positions = positions + offsets[None, :, :]
        return Trajectory(self.times, positions, self.space.box)

    def to_dict(self) -> dict:
        """JSON-serialisable description of the space and the recorded walker edges."""
        return {
            "times": self.times.tolist(),
            "anchors": self.anchors.tolist(),
            "space": {
                "coordinates": self.space.coordinates.tolist(),
                "edges": self.space.edges().tolist(),
                "box": self.space.box.to_dict(),
            },
        }


@dataclass(frozen=True)
class GraphSimulation:
    """A graph system: a space, a rule and the physical duration of one rule application."""

    space: Space
    rule: Rule = field(default_factory=RandomHop)
    step_time: float = 1.0

    def __post_init__(self) -> None:
        if self.step_time <= 0:
            raise ValueError("step_time must be > 0")

    def run(
        self,
        initial: WalkerGraph,
        steps: int,
        rng: np.random.Generator,
        record_every: int = 1,
        observer: StepObserver | None = None,
    ) -> GraphHistory:
        """Apply the rule ``steps`` times, recording every ``record_every``-th state.

        ``observer(step, previous_positions, positions)`` is called after every step with the
        projected positions (see :mod:`imigraph.metrics`).
        """
        if initial.space != self.space:
            raise ValueError("initial state must live on the simulation's space")
        if steps < 0 or record_every < 1:
            raise ValueError("steps must be >= 0 and record_every >= 1")
        frames = [initial.anchors]
        state = initial
        coordinates = self.space.coordinates
        for step in range(1, steps + 1):
            previous = state
            state = self.rule.apply(state, rng)
            if observer is not None:
                observer(step, coordinates[previous.anchors], coordinates[state.anchors])
            if step % record_every == 0:
                frames.append(state.anchors)
        times = self.step_time * record_every * np.arange(len(frames))
        return GraphHistory(self.space, times, np.stack(frames))
