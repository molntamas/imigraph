"""One graph system for both walker models: edge walkers that never interact and hyperedge
walkers with an exclusion principle and, optionally, the HPP collision.

A *walker model* says what a walker is:

* ``edge``: the walker is the edge ``{w,s}``, it has a position and nothing else. Any number of
  walkers may share a site and two walkers never notice each other. This is the model of the
  Brownian imitation (:class:`~imigraph.graph.rules.ChainRule`).
* ``hyperedge``: the walker is the hyperedge ``{w,s,s'}`` of the lattice gas
  (:mod:`imigraph.graph.gas`): it sits on ``s`` and heads for the neighbour ``s'`` along the
  edge with direction label ``k``. The *exclusion principle* holds: at most one walker per
  channel ``(s, k)``, so at most four walkers per site, never two with the same heading.

A *collision* option says what happens when walkers meet:

* ``none``: nothing. Edge walkers pass through each other; hyperedge walkers only feel the
  exclusion (a hop into an occupied channel is blocked, the walker stays).
* ``hpp``: the Collide rule of the HPP lattice gas runs first in every step: two walkers on
  one site heading in opposite directions leave along the perpendicular pair of channels, if
  those are empty real edges. Momentum and energy are conserved. Needs hyperedge walkers.

The rules of a step run in order on the state the previous one produced. A chain rule (an
editable :class:`RuleSpec`) moves a walker along a chain of labelled edges; on hyperedge
walkers the label of its last hop becomes the new heading. The built-in rules ``Stream``
(move one edge along the heading, keep the label) and ``Reflect`` (a walker heading into a
wall self-loop turns round) need hyperedge walkers; with ``hpp`` collisions they are the HPP
lattice gas. Every rule's outcome on hyperedge walkers is passed through the exclusion update:
a walker whose target channel is taken in the current state, or wanted by another walker too
(ties drawn at random), is *blocked* and keeps its channel. The HPP set never blocks, so it
reproduces :class:`~imigraph.graph.gas.LatticeGas` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from imigraph.graph.gas import OPPOSITE, RULES, LatticeGas, LatticeGasState
from imigraph.graph.rules import DIRECTION_NAMES, ChainRule, RuleSpec, rule_specs
from imigraph.graph.simulation import GraphHistory
from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph
from imigraph.metrics import StepObserver

WALKER_MODELS = ("edge", "hyperedge")
COLLISIONS = ("none", "hpp")
HEADINGS = ("random", *DIRECTION_NAMES)
BUILTIN_RULES = ("stream", "reflect")
GAS_RULES = {rule["name"].lower(): rule for rule in RULES}

WALKER_MODEL_TEXT = {
    "edge": "edge walkers {w,s}: a position only; any number per site, no interaction",
    "hyperedge": "hyperedge walkers {w,s,s'}: a position and a heading; one walker per "
    "channel (s, k), a blocked hop keeps the walker where it is",
}
COLLISION_TEXT = {
    "none": "no collision: walkers pass through each other (edge) or are only blocked by "
    "the exclusion (hyperedge)",
    "hpp": "HPP collision: a head-on pair on one site leaves along the perpendicular "
    "channels, momentum and energy conserved; runs first in every step",
}


@dataclass(frozen=True)
class BuiltinRule:
    """``Stream`` or ``Reflect`` of the lattice gas, as a rule of the editable rule set.
    Fixed rewrites on hyperedge walkers; ``description`` is the author's text."""

    kind: str
    name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in BUILTIN_RULES:
            raise ValueError(
                f"builtin must be one of {', '.join(BUILTIN_RULES)}, not {self.kind!r}"
            )
        if not self.name:
            object.__setattr__(self, "name", self.kind.capitalize())

    @property
    def hops(self) -> int:
        return 0

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        return np.zeros(2)

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        return np.zeros(2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "builtin": self.kind,
            "notation": GAS_RULES[self.kind]["rewrite"],
            "description": self.description,
        }

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        text = dict(GAS_RULES[self.kind])
        text["name"] = self.name
        text["description"] = self.description
        text["pattern"] = _builtin_pattern(self.kind)
        text["physics"] += "; no drift or diffusion prediction (hyperedge walkers only)"
        return text

    def propose(
        self, state: LatticeGasState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Target sites, headings and matched paths (shape ``(n, 2)``) before exclusion."""
        if self.kind == "stream":
            target = state.headings()
            return target, state.directions, np.stack([state.sites, target], axis=1)
        wall = state.headings() == state.sites
        directions = np.where(wall, OPPOSITE[state.directions], state.directions)
        return state.sites, directions, np.stack([state.sites, state.sites], axis=1)


def _builtin_pattern(kind: str) -> str:
    from imigraph.report import svg

    is_walker = lambda v: v.startswith("w")
    if kind == "stream":
        lhs: tuple[tuple[str, ...], ...] = (("w", "s", "t"), ("t", "u"))
        rhs: tuple[tuple[str, ...], ...] = (("w", "t", "u"), ("t", "u"))
    else:
        lhs, rhs = (("w", "s", "s"), ("s", "t")), (("w", "s", "t"), ("s", "t"))
    return svg.rule_pattern(lhs, rhs, lambda v: "walker" if is_walker(v) else "site")


def collide_pattern() -> str:
    from imigraph.report import svg

    lhs = (("w1", "s", "a"), ("w2", "s", "b"), ("s", "c"), ("s", "d"))
    rhs = (("w1", "s", "c"), ("w2", "s", "d"), ("s", "c"), ("s", "d"))
    return svg.rule_pattern(
        lhs, rhs, lambda v: "walker" if v.startswith("w") else "site"
    )


RuleItem = ChainRule | BuiltinRule
RuleListData = RuleSpec | dict[str, Any] | list[Any] | tuple[Any, ...] | None


def rules_from_data(data: RuleListData) -> tuple[RuleItem, ...]:
    """The rule list of a request: chain specs (JSON or :class:`RuleSpec`) and built-in
    entries ``{"builtin": "stream" | "reflect", "name": ..., "description": ...}``."""
    items = list(data) if isinstance(data, (list, tuple)) else [data]
    rules: list[RuleItem] = []
    for item in items:
        if isinstance(item, dict) and item.get("builtin"):
            rules.append(
                BuiltinRule(
                    str(item["builtin"]),
                    str(item.get("name") or ""),
                    str(item.get("description") or ""),
                )
            )
        else:
            rules.extend(ChainRule(spec) for spec in rule_specs(item))
    if not rules:
        raise ValueError("enable at least one rule")
    return tuple(rules)


def resolve_exclusion(
    state: LatticeGasState,
    sites: np.ndarray,
    directions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[LatticeGasState, np.ndarray]:
    """Synchronous exclusion update: every walker proposes a channel ``(site, direction)``.
    A walker that proposes its own channel always keeps it. Of several walkers proposing one
    channel, one wins (a walker that already sits there, else drawn at random); the others are
    rejected and keep their channel. A walker whose target is the channel a rejected walker
    keeps is rejected too (a jam), until nothing changes. Walkers may move into a channel that
    its occupant leaves in the same step, so a train streams together as in the lattice gas.
    Returns the new state and the mask of blocked walkers. Draws from ``rng`` only when
    proposals clash, so conflict-free rule sets consume no randomness here."""
    current = state.sites * 4 + state.directions
    proposed = sites * 4 + directions
    moved = proposed != current
    n = proposed.size
    order = np.arange(n)
    if np.unique(proposed).size < n:
        # Stayers first (they keep their channel), movers in random order.
        order = rng.permutation(n)
        order = order[np.argsort(moved[order], kind="stable")]
    _, first = np.unique(proposed[order], return_index=True)
    accepted = np.zeros(n, dtype=bool)
    accepted[order[first]] = True
    while True:
        jammed = accepted & moved & np.isin(proposed, current[~accepted])
        if not jammed.any():
            break
        accepted[jammed] = False
    blocked = moved & ~accepted
    new_sites = np.where(accepted, sites, state.sites)
    new_directions = np.where(accepted, directions, state.directions)
    return replace(state, sites=new_sites, directions=new_directions), blocked


def initial_headings(
    sites: np.ndarray, heading: str, rng: np.random.Generator
) -> np.ndarray:
    """Headings of the walkers at ``sites`` that respect the exclusion: a fixed heading
    needs one walker per site, ``random`` gives the walkers of a site distinct labels."""
    if heading not in HEADINGS:
        raise ValueError(
            f"heading must be one of {', '.join(HEADINGS)}, not {heading!r}"
        )
    directions = np.zeros(sites.shape[0], dtype=np.int64)
    for site in np.unique(sites):
        members = np.flatnonzero(sites == site)
        if heading == "random":
            if members.size > 4:
                raise ValueError(
                    f"{members.size} walkers on site {int(site)}: hyperedge walkers need one "
                    "channel each, at most 4 per site"
                )
            directions[members] = rng.permutation(4)[: members.size]
        else:
            if members.size > 1:
                raise ValueError(
                    f"{members.size} walkers on site {int(site)} with the same heading "
                    f"{heading}: choose random headings or spread the walkers"
                )
            directions[members] = DIRECTION_NAMES.index(heading)
    return directions


@dataclass(frozen=True)
class WalkerSystem:
    """A space, a rule set, the walker model and the collision option (see the module)."""

    space: SquareLattice
    rules: tuple[RuleItem, ...]
    walker_model: str = "edge"
    collisions: str = "none"
    step_time: float = 1.0
    gas: LatticeGas = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.walker_model not in WALKER_MODELS:
            raise ValueError(
                f"walker model must be one of {', '.join(WALKER_MODELS)}, "
                f"not {self.walker_model!r}"
            )
        if self.collisions not in COLLISIONS:
            raise ValueError(
                f"collisions must be one of {', '.join(COLLISIONS)}, not {self.collisions!r}"
            )
        if not self.rules:
            raise ValueError("enable at least one rule")
        if self.step_time <= 0:
            raise ValueError("step_time must be > 0")
        builtins = [r.name for r in self.rules if isinstance(r, BuiltinRule)]
        if self.walker_model == "edge":
            if self.collisions == "hpp":
                raise ValueError(
                    "HPP collisions need hyperedge walkers (a heading per walker): switch the "
                    "walker model to hyperedge"
                )
            if builtins:
                raise ValueError(
                    f"{', '.join(builtins)} works on hyperedge walkers (it follows the "
                    "heading): switch the walker model to hyperedge or untick the rule"
                )
        object.__setattr__(self, "gas", LatticeGas(self.space, self.step_time))

    # ---------- rule set as a whole ----------

    @property
    def name(self) -> str:
        return " + ".join(rule.name for rule in self.rules)

    @property
    def chain_rules(self) -> tuple[ChainRule, ...]:
        return tuple(r for r in self.rules if isinstance(r, ChainRule))

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return tuple(r.spec for r in self.chain_rules)

    @property
    def hops(self) -> int:
        return sum(r.spec.hops for r in self.chain_rules)

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
        return [
            r.spec.to_dict() if isinstance(r, ChainRule) else r.to_dict()
            for r in self.rules
        ]

    def descriptions(
        self, spacing: float = 1.0, step_time: float = 1.0
    ) -> list[dict[str, str]]:
        """Collide (when enabled), one entry per rule, then the set as a whole."""
        described: list[dict[str, str]] = []
        if self.collisions == "hpp":
            collide = dict(GAS_RULES["collide"])
            collide["description"] = (
                "collisions: hpp (runs before the rules in every step)"
            )
            collide["pattern"] = collide_pattern()
            described.append(collide)
        for rule in self.rules:
            if isinstance(rule, ChainRule):
                entry = rule.spec.describe(spacing, step_time)
                from imigraph.report import svg

                entry["pattern"] = svg.rule_pattern(
                    rule.spec.lhs, rule.spec.rhs, rule.spec.type_of
                )
            else:
                entry = rule.describe(spacing, step_time)
            described.append(entry)
        model = (
            f"{WALKER_MODEL_TEXT[self.walker_model]}; {COLLISION_TEXT[self.collisions]}"
        )
        if len(described) == 1:
            # One rule and no collision: the rule is the whole set, described once.
            if self.walker_model == "hyperedge":
                described[0]["description"] = (
                    f"{described[0]['description']} [{model}]".strip()
                )
            return described
        drift = self.drift(spacing, step_time)
        diffusion = self.diffusion(spacing, step_time)
        prediction = (
            "the chain rules' drifts and diffusions add up: "
            if self.chain_rules
            else "no chain rule, so no drift or diffusion prediction: "
        )
        blocked = (
            " Hyperedge walkers: a hop into a taken channel is blocked, which lowers the "
            "measured values at high density."
            if self.walker_model == "hyperedge"
            else ""
        )
        described.append(
            {
                "name": self.name,
                "description": model,
                "rewrite": " ; ".join(
                    r.spec.notation
                    if isinstance(r, ChainRule)
                    else r.to_dict()["notation"]
                    for r in self.rules
                ),
                "matches": f"{len(self.rules)} rule(s) applied in order in every step; each "
                "one matches on the state the previous one produced",
                "choice": "each rule draws its directions independently; all walkers "
                "rewritten at once by every rule",
                "invariants": "number of walkers, one anchoring edge per walker, the space "
                "graph"
                + (
                    "; the exclusion principle"
                    if self.walker_model == "hyperedge"
                    else ""
                ),
                "physics": f"{self.hops} hops per step in total; {prediction}"
                f"D = ({diffusion[0]:.4g}, {diffusion[1]:.4g}), drift v = ({drift[0]:.4g}, "
                f"{drift[1]:.4g}) for a = {spacing:g}, tau = {step_time:g}.{blocked}",
            }
        )
        return described

    # ---------- states ----------

    def initial_state(
        self, anchors: np.ndarray, heading: str, rng: np.random.Generator
    ) -> WalkerGraph | LatticeGasState:
        anchors = np.asarray(anchors, dtype=np.int64)
        if self.walker_model == "edge":
            return WalkerGraph(self.space, anchors)
        return LatticeGasState(
            self.space, anchors, initial_headings(anchors, heading, rng)
        )

    def step(
        self, state: WalkerGraph | LatticeGasState, rng: np.random.Generator
    ) -> tuple[WalkerGraph | LatticeGasState, np.ndarray, np.ndarray, int]:
        """One step: the new state, the concatenated matched paths ``(n, L)``, the sites where
        Collide fired and the number of blocked hops."""
        pieces: list[np.ndarray] = []
        collided = np.zeros(0, dtype=np.int64)
        blocked_total = 0
        if isinstance(state, WalkerGraph):
            for rule in self.rules:
                assert isinstance(rule, ChainRule)
                state, path, _ = rule.apply_with_choices(state, rng)
                _append_path(pieces, path)
            return state, np.concatenate(pieces, axis=1), collided, 0
        if self.collisions == "hpp":
            state, collided = self.gas.collide(state, rng)
        for rule in self.rules:
            if isinstance(rule, ChainRule):
                _, path, labels = rule.apply_with_choices(
                    WalkerGraph(self.space, state.sites), rng
                )
                target = rule.spec.chain()[2]
                sites = path[:, target]
                directions = labels[:, target - 1] if target > 0 else state.directions
            else:
                sites, directions, path = rule.propose(state)
            state, blocked = resolve_exclusion(state, sites, directions, rng)
            path = (
                np.where(blocked[:, None], path[:, :1], path)
                if np.any(blocked)
                else path
            )
            blocked_total += int(blocked.sum())
            _append_path(pieces, path)
        return state, np.concatenate(pieces, axis=1), collided, blocked_total

    def run(
        self,
        initial: WalkerGraph | LatticeGasState,
        steps: int,
        rng: np.random.Generator,
        observer: StepObserver | None = None,
    ) -> tuple[GraphHistory, int]:
        """Every state of ``steps`` steps with the matched paths (and headings, collision sites
        for hyperedge walkers); returns the history and the total number of blocked hops."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        coordinates = self.space.coordinates
        anchors_of = lambda s: s.anchors if isinstance(s, WalkerGraph) else s.sites
        frames, directions, paths, collisions = [anchors_of(initial)], [], [], []
        if isinstance(initial, LatticeGasState):
            directions.append(initial.directions)
        state = initial
        blocked_total = 0
        for step in range(1, steps + 1):
            previous = anchors_of(state)
            state, path, collided, blocked = self.step(state, rng)
            current = anchors_of(state)
            blocked_total += blocked
            frames.append(current)
            paths.append(path)
            collisions.append(collided)
            if isinstance(state, LatticeGasState):
                directions.append(state.directions)
            if observer is not None:
                wall_hits = np.any(path[:, 1:] == path[:, :-1], axis=1)
                observer(step, coordinates[previous], coordinates[current], wall_hits)
        width = max(p.shape[1] for p in paths) if paths else 1
        padded = [
            np.concatenate(
                [p, np.repeat(p[:, -1:], width - p.shape[1], axis=1)], axis=1
            )
            for p in paths
        ]
        history = GraphHistory(
            self.space,
            self.step_time * np.arange(steps + 1),
            np.stack(frames),
            np.stack(padded) if padded else None,
            np.stack(directions) if directions else None,
            tuple(collisions) if self.walker_model == "hyperedge" else None,
        )
        return history, blocked_total


def _append_path(pieces: list[np.ndarray], path: np.ndarray) -> None:
    if pieces and np.array_equal(pieces[-1][:, -1], path[:, 0]):
        path = path[:, 1:]
    if path.shape[1]:
        pieces.append(path)
