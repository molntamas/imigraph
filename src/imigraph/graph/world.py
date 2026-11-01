"""The walker system: one rule set on balls, run in rounds.

Every walker is a ball ``{w,s,v}`` (:mod:`imigraph.graph.state`). A step of the system:

1. Every ball gets a hop budget ``|vx| + |vy|``, the hops of its velocity in Bresenham order.
   The *round rules* (those whose balls carry a hop label, like Stream, Reflect and the
   collision rules) are applied in order in every round, each one to the balls that still
   have hops left, each one seeing the state the previous one produced. A ball that moves to
   its hop target consumes one hop; a ball whose velocity changes mid-step continues with the
   tail of its new velocity's hop order (never more hops than it had). Rounds end when no ball
   has hops left (a cap of ``2 max + 2`` rounds stops rule sets that never move).
2. The *step rules* (chain rules of the Brownian imitation, and ball rules without hop labels
   such as a random kick) run once each, in order.

Balls at rest have no hops, so a rule set of chain rules on resting balls is the Brownian
imitation of :class:`~imigraph.graph.rules.RuleSequence`, step for step. Balls with unit
velocities under Reflect, Turn and Stream are the HPP lattice gas; balls with integer
velocities under Reflect, Exchange, Swap and Stream are the ball gas held against the hard
disks.

*Overlapping matches.* A chain rule matches once per walker: the walker edge ``{w,s0}`` and the
chain of site edges it drew. Two matches overlap when they share a site edge. In a
Wolfram-style rewriting system every match consumes its left-hand side, so overlapping matches
cannot all be applied in one generation; which ones are applied is the *overlap strategy*:
``parallel`` (every match; site edges are read, not consumed), ``ordered`` (walker 0 first, a
match sharing an edge with one already applied waits), ``random`` (a fresh order every step),
``oldest`` (the walker that has waited longest goes first: the event-age ordering of the
Wolfram models). Skipped matches are counted as *waiting* hops. For collision rules the same
strategy orders the contact pairs when a ball is in several (one pair per ball per round).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imigraph.graph.rules import (
    BallRule,
    ChainRule,
    RoundContext,
    RuleSpec,
    approaching,
    meeting_pairs,
    rule_item,
    rule_specs,
)
from imigraph.graph.simulation import GraphHistory
from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph, body_fits
from imigraph.graph.velocity import AXES, hop_order
from imigraph.metrics import StepObserver

OVERLAPS = ("parallel", "ordered", "random", "oldest")
OVERLAP_TEXT = {
    "parallel": "parallel: every match is applied, site edges are shared read-only",
    "ordered": "ordered: matches applied walker by walker in index order, a match sharing "
    "a site edge with one already applied this step waits",
    "random": "random: like ordered, in an order drawn afresh every step",
    "oldest": "oldest: like ordered, the walker that has waited longest goes first",
}

RuleItem = ChainRule | BallRule
RuleListData = RuleSpec | dict[str, Any] | list[Any] | tuple[Any, ...] | None


def rules_from_data(data: RuleListData) -> tuple[RuleItem, ...]:
    """The rule list of a request: specs (JSON or :class:`RuleSpec`), each run vectorised
    (chain family) or by the matcher (ball family)."""
    items = list(data) if isinstance(data, (list, tuple)) else [data]
    rules: list[RuleItem] = []
    for item in items:
        if isinstance(item, dict) and item.get("builtin"):
            raise ValueError(
                "built-in rules are gone: every rule is written out, pick Stream, Reflect, "
                "Exchange, Swap or Turn from the presets"
            )
        rules.extend(rule_item(spec) for spec in rule_specs(item))
    if not rules:
        raise ValueError("enable at least one rule")
    return tuple(rules)


def chain_edge_keys(path: np.ndarray, num_nodes: int) -> np.ndarray:
    """Undirected keys of the chain edges of every matched path, shape ``(n, k)``: the edge
    ``{a,b}`` is ``min * N + max``, so two walkers drawing the same edge get the same key."""
    a, b = path[:, :-1], path[:, 1:]
    return np.minimum(a, b) * num_nodes + np.maximum(a, b)


def select_matches(
    path: np.ndarray,
    num_nodes: int,
    strategy: str,
    rng: np.random.Generator,
    ages: np.ndarray | None = None,
) -> np.ndarray:
    """The mask of matches applied under ``strategy`` (see the module): greedy in the
    strategy's order, a match is skipped when a chain edge of it was consumed by a match
    applied earlier in this step. ``parallel`` applies all of them."""
    n = path.shape[0]
    if strategy == "parallel" or path.shape[1] < 2 or n < 2:
        return np.ones(n, dtype=bool)
    keys = chain_edge_keys(path, num_nodes)
    if strategy == "ordered":
        order = np.arange(n)
    elif strategy == "random":
        order = rng.permutation(n)
    elif strategy == "oldest":
        waited = np.zeros(n, dtype=np.int64) if ages is None else ages
        order = np.argsort(-waited, kind="stable")
    else:
        raise ValueError(
            f"overlaps must be one of {', '.join(OVERLAPS)}, not {strategy!r}"
        )
    if np.unique(keys).size == keys.size:
        return np.ones(n, dtype=bool)  # no shared edge at all
    accepted = np.zeros(n, dtype=bool)
    consumed: set[int] = set()
    for w in order:
        mine = {int(k) for k in keys[w]}
        if mine & consumed:
            continue
        consumed |= mine
        accepted[w] = True
    return accepted


def next_labels(
    velocities: np.ndarray,
    plans: list[list[int]],
    position: list[int],
    budget: np.ndarray,
) -> np.ndarray:
    """The direction label of every ball's next hop, -1 for a ball without hops left."""
    labels = np.full(velocities.shape[0], -1, dtype=np.int64)
    for b in np.flatnonzero(budget > 0):
        axis = plans[b][position[b]]
        plus, minus = AXES[axis]
        labels[b] = plus if velocities[b, axis] > 0 else minus
    return labels


def round_context(
    state: WalkerGraph,
    active: np.ndarray,
    labels: np.ndarray | None,
    max_speed: int | None = None,
    overlaps: str = "parallel",
    ages: np.ndarray | None = None,
) -> RoundContext:
    """The matcher's view of a round: the hop label of every active ball (by default the
    first hop of its velocity), where its body lands by that hop (its own site when the body
    runs into a wall), whether it does run into a wall, whether it is free to take the hop
    and the contacts of the round.

    A ball that cannot move this round (no hops left, blocked by a wall) stays; two balls
    whose hops meet (they land on one site, one lands on a staying ball, or they cross) are
    held back and stay as well, and a ball landing on a held ball is held too, until nothing
    changes. Meeting pairs that approach each other are the *contacts* the collision rules
    match; after a collision the pair's new hops are looked at afresh. A ball still held at
    the end is not free: it waits in this round instead of passing through another ball."""
    n = state.num_walkers
    velocities = state.velocity
    if labels is None:
        plans = [hop_order(v) for v in velocities]
        budget = np.where(active, np.abs(velocities).sum(axis=1), 0)
        labels = next_labels(velocities, plans, [0] * n, budget)
    neighbors = state.space.neighbors
    bodies = state.sites()
    safe = np.maximum(labels, 0)
    hits = np.any(neighbors[bodies, safe[:, None]] == bodies, axis=1)
    wall = (labels >= 0) & hits
    can_move = active & (labels >= 0) & ~wall
    targets = np.where(can_move, neighbors[state.anchors, safe], state.anchors)
    # Balls that stay in this round: those that cannot move, and those held back by a
    # meeting (a ball landing on a staying ball is a meeting too, until nothing changes).
    staying = ~can_move
    meetings = np.zeros((0, 2), dtype=np.int64)
    if n > 1 and can_move.any():
        while True:
            meetings = meeting_pairs(state, targets, staying)
            held = staying.copy()
            if meetings.size:
                held[np.unique(meetings)] = True
            if np.array_equal(held, staying):
                break
            staying = held
    pairs = meetings[approaching(state, meetings)]
    free = ~staying
    if max_speed is None:
        max_speed = max(1, int(np.abs(velocities).max(initial=0)))
    return RoundContext(
        active & (labels >= 0),
        labels,
        targets,
        wall,
        max_speed,
        overlaps,
        ages,
        free,
        pairs,
    )


@dataclass(frozen=True)
class WalkerSystem:
    """A room, a rule set, the step time and the overlap strategy (see the module).
    ``max_speed`` bounds the velocity lattice the matcher solves on (by default the largest
    component of the initial velocities, at least 1); ``mass`` is the mass of one node."""

    space: SquareLattice
    rules: tuple[RuleItem, ...]
    step_time: float = 1.0
    overlaps: str = "parallel"
    max_speed: int | None = None
    mass: float = 1.0
    round_rules: tuple[BallRule, ...] = field(init=False, repr=False, compare=False)
    step_rules: tuple[RuleItem, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("enable at least one rule")
        if self.overlaps not in OVERLAPS:
            raise ValueError(
                f"overlaps must be one of {', '.join(OVERLAPS)}, not {self.overlaps!r}"
            )
        if self.step_time <= 0 or self.mass <= 0:
            raise ValueError("step_time and mass must be > 0")
        if self.max_speed is not None and self.max_speed < 1:
            raise ValueError("max_speed must be >= 1")
        object.__setattr__(
            self,
            "round_rules",
            tuple(r for r in self.rules if isinstance(r, BallRule) and r.plan.is_round),
        )
        object.__setattr__(
            self,
            "step_rules",
            tuple(
                r
                for r in self.rules
                if not (isinstance(r, BallRule) and r.plan.is_round)
            ),
        )

    # ---------- rule set as a whole ----------

    @property
    def name(self) -> str:
        return " + ".join(rule.name for rule in self.rules)

    @property
    def chain_rules(self) -> tuple[ChainRule, ...]:
        return tuple(r for r in self.rules if isinstance(r, ChainRule))

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return tuple(r.spec for r in self.rules)

    @property
    def hops(self) -> int:
        return sum(r.spec.hops for r in self.chain_rules)

    @property
    def interacting(self) -> bool:
        """Whether the rule set has a two-ball (collision) rule."""
        return any(isinstance(r, BallRule) and r.plan.is_pair for r in self.rules)

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        return np.sum(
            [np.zeros(2)] + [r.drift(spacing, step_time) for r in self.rules], axis=0
        )

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        return np.sum(
            [np.zeros(2)] + [r.diffusion(spacing, step_time) for r in self.rules],
            axis=0,
        )

    def rules_as_data(self) -> list[dict[str, Any]]:
        return [r.spec.to_dict() for r in self.rules]

    def descriptions(
        self, spacing: float = 1.0, step_time: float = 1.0
    ) -> list[dict[str, str]]:
        """One entry per rule, each with an SVG ``pattern``, then the set as a whole."""
        from imigraph.report import svg

        described: list[dict[str, str]] = []
        for rule in self.rules:
            entry = rule.describe(spacing, step_time)
            entry["pattern"] = svg.rule_pattern(
                rule.spec.lhs,
                rule.spec.rhs,
                rule.spec.type_of,
                rule.spec.lhs_ops,
                rule.spec.rhs_ops,
            )
            described.append(entry)
        model = f"overlapping matches: {OVERLAP_TEXT[self.overlaps]}"
        if len(described) == 1:
            if self.overlaps != "parallel":
                described[0]["description"] = (
                    f"{described[0]['description']} [{model}]".strip()
                )
            return described
        drift = self.drift(spacing, step_time)
        diffusion = self.diffusion(spacing, step_time)
        rounds = ", ".join(r.name for r in self.round_rules)
        once = ", ".join(r.name for r in self.step_rules)
        prediction = (
            "the chain rules' drifts and diffusions add up: "
            if self.chain_rules
            else "no chain rule, so no drift or diffusion prediction: "
        )
        waiting = (
            f" Overlapping matches ({self.overlaps}): a walker whose chain shares a site "
            "edge with a match applied earlier in the step waits, which lowers the "
            "measured values where walkers crowd."
            if self.overlaps != "parallel"
            else ""
        )
        described.append(
            {
                "name": self.name,
                "description": model,
                "rewrite": " ; ".join(r.spec.notation for r in self.rules),
                "matches": (
                    f"round rules ({rounds}) applied in order in every round of a step, on "
                    "balls with hops left; "
                    if rounds
                    else ""
                )
                + (
                    f"step rules ({once}) applied once per step in order; "
                    if once
                    else ""
                )
                + "each rule matches on the state the previous one produced",
                "choice": "each rule draws its directions and solves its velocities "
                "independently; all balls rewritten at once by every rule",
                "invariants": "number of balls, one hyperedge per ball, the room and the "
                "velocity lattice"
                + (
                    "; momentum and energy by the collision rules, momentum traded with "
                    "the walls"
                    if self.round_rules
                    else ""
                ),
                "physics": f"{self.hops} chain hops per step; {prediction}"
                f"D = ({diffusion[0]:.4g}, {diffusion[1]:.4g}), drift v = ({drift[0]:.4g}, "
                f"{drift[1]:.4g}) for a = {spacing:g}, tau = {step_time:g}.{waiting}",
            }
        )
        return described

    # ---------- states ----------

    def initial_state(
        self,
        anchors: np.ndarray,
        velocities: np.ndarray | None = None,
        radius: int = 0,
    ) -> WalkerGraph:
        """The state at step 0: balls at ``anchors`` with ``velocities`` (at rest when
        omitted) and bodies of ``radius``."""
        return WalkerGraph(
            self.space, np.asarray(anchors, dtype=np.int64), velocities, radius
        )

    def step(
        self,
        state: WalkerGraph,
        rng: np.random.Generator,
        ages: np.ndarray | None = None,
        max_speed: int | None = None,
    ) -> tuple[WalkerGraph, np.ndarray, list[dict[str, Any]], np.ndarray]:
        """One step: the new state, the sites every ball visited ``(n, L)`` (hop by hop, then
        the chain rules' matched chains), the events of the rules that changed a velocity, and
        the mask of walkers that waited because of an overlapping match (``ages`` orders the
        ``oldest`` strategy)."""
        n = state.num_walkers
        anchors, velocities = state.anchors.copy(), state.velocity.copy()
        radius = state.radius
        top = max(1, int(np.abs(velocities).max(initial=0)))
        bound = top if max_speed is None else max(max_speed, top)
        budget = np.abs(velocities).sum(axis=1)
        plans = [hop_order(v) for v in velocities]
        position = [0] * n
        events: list[dict[str, Any]] = []
        pieces: list[np.ndarray] = [anchors[:, None].copy()]
        waited = np.zeros(n, dtype=bool)
        if self.round_rules:
            cap = 2 * int(budget.max(initial=0)) + 2
            for _ in range(cap):
                if not np.any(budget > 0):
                    break
                for rule in self.round_rules:
                    if not np.any(budget > 0):
                        break
                    # A collision rule runs again while it finds new contacts: a ball
                    # redirected by one collision may meet a third ball in the same round.
                    passes = n + 1 if rule.plan.is_pair else 1
                    for _pass in range(passes):
                        current = WalkerGraph(self.space, anchors, velocities, radius)
                        labels = next_labels(velocities, plans, position, budget)
                        context = round_context(
                            current, budget > 0, labels, bound, self.overlaps, ages
                        )
                        outcome = rule.match(current, rng, context)
                        changed = np.any(outcome.velocities != velocities, axis=1)
                        anchors, velocities = outcome.anchors, outcome.velocities
                        for b in np.flatnonzero(outcome.consumed):
                            position[b] += 1
                            budget[b] -= 1
                        for b in np.flatnonzero(changed):
                            remaining = int(budget[b])
                            order = hop_order(velocities[b])
                            tail = (
                                order[max(0, len(order) - remaining) :]
                                if remaining
                                else []
                            )
                            plans[b], position[b], budget[b] = tail, 0, len(tail)
                        events.extend(outcome.events)
                        if not outcome.events:
                            break
                _append_path(pieces, anchors[:, None])
        for rule in self.step_rules:
            current = WalkerGraph(self.space, anchors, velocities, radius)
            if isinstance(rule, ChainRule):
                _, path, _ = rule.apply_with_choices(current, rng)
                applied = select_matches(
                    path, self.space.num_nodes, self.overlaps, rng, ages
                )
                target = rule.spec.chain()[2]
                applied &= body_fits(self.space, path[:, target], radius)
                anchors = np.where(applied, path[:, target], anchors)
                waited |= ~applied
                _append_path(pieces, np.where(applied[:, None], path, path[:, :1]))
            else:
                context = round_context(
                    current, np.ones(n, dtype=bool), None, bound, self.overlaps, ages
                )
                outcome = rule.match(current, rng, context)
                anchors, velocities = outcome.anchors, outcome.velocities
                events.extend(outcome.events)
                _append_path(pieces, anchors[:, None])
        new_state = WalkerGraph(
            self.space,
            anchors,
            velocities if state.velocities is not None else None,
            radius,
        )
        return new_state, np.concatenate(pieces, axis=1), events, waited

    def run(
        self,
        initial: WalkerGraph,
        steps: int,
        rng: np.random.Generator,
        observer: StepObserver | None = None,
        record_every: int = 1,
    ) -> tuple[GraphHistory, int]:
        """Every recorded state of ``steps`` steps with the visited sites, velocities and
        events; returns the history and the total number of waiting hops (overlapping
        matches skipped)."""
        if initial.space != self.space:
            raise ValueError("the balls must live in this room")
        if steps < 0 or record_every < 1 or steps % record_every:
            raise ValueError("steps must be a non-negative multiple of record_every")
        coordinates = self.space.coordinates
        moving = initial.velocities is not None
        top = max(1, int(np.abs(initial.velocity).max(initial=0)))
        bound = top if self.max_speed is None else max(self.max_speed, top)
        frames, velocities, paths = [initial.anchors], [initial.velocity], []
        events: list[tuple[dict[str, Any], ...]] = []
        interval: list[dict[str, Any]] = []
        state = initial
        waited_total = 0
        ages = np.zeros(initial.num_walkers, dtype=np.int64)
        for step in range(1, steps + 1):
            previous = state
            state, path, step_events, waited = self.step(state, rng, ages, bound)
            waited_total += int(waited.sum())
            ages = np.where(waited, ages + 1, 0)
            interval.extend(step_events)
            if record_every == 1:
                paths.append(path)
            if observer is not None:
                bounced = np.zeros(initial.num_walkers, dtype=bool)
                for e in step_events:
                    if e["kind"] == "bounce":
                        bounced[e["balls"][0]] = True
                stayed = np.any(path[:, 1:] == path[:, :-1], axis=1)
                observer(
                    step,
                    coordinates[previous.anchors],
                    coordinates[state.anchors],
                    bounced if moving else stayed,
                )
            if step % record_every == 0:
                frames.append(state.anchors)
                velocities.append(state.velocity)
                events.append(tuple(interval))
                interval = []
        width = max(p.shape[1] for p in paths) if paths else 1
        padded = [
            np.concatenate(
                [p, np.repeat(p[:, -1:], width - p.shape[1], axis=1)], axis=1
            )
            for p in paths
        ]
        history = GraphHistory(
            self.space,
            self.step_time * record_every * np.arange(len(frames)),
            np.stack(frames),
            np.stack(padded) if padded else None,
            np.stack(velocities) if moving else None,
            tuple(events),
            initial.radius,
            self.mass,
            self.step_time,
        )
        return history, waited_total


def explain_frames(history: GraphHistory, detail: int = 8) -> list[str]:
    """One caption per frame: what the rules did in the step that led to it (collisions and
    bounces by name) and the total momentum when the balls move."""
    return history.captions(detail)


def _append_path(pieces: list[np.ndarray], path: np.ndarray) -> None:
    if pieces and np.array_equal(pieces[-1][:, -1], path[:, 0]):
        path = path[:, 1:]
    if path.shape[1]:
        pieces.append(path)
