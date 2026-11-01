"""One graph system for both walker models: edge walkers that never interact and hyperedge
walkers with an exclusion principle and, optionally, the HPP collision, and with a configurable
strategy for *overlapping matches* of one rule.

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

*Overlapping matches.* A chain rule matches once per walker: the walker edge ``{w,s0}`` and the
chain of site edges ``{s0,s1}, ...`` it drew. Two matches overlap when they share a site edge
(two walkers on one site drawing the same edge, or chains crossing the same edge). In a
Wolfram-style rewriting system every match consumes its left-hand side, so overlapping matches
cannot all be applied in one generation; which ones are applied is the *overlap strategy*:

* ``parallel``: every match is applied; site edges are read, not consumed, so sharing them is
  harmless (the rule only rewrites the walker edge). All walkers move at once.
* ``ordered``: matches are applied in a fixed order (walker 0 first); a match that shares a
  site edge with one already applied in this step is skipped, its walker waits.
* ``random``: the same, in an order drawn afresh every step.
* ``oldest``: the same, the walker that has waited longest goes first (ties by index), the
  event-age ordering of the Wolfram models.

Skipped matches are counted as *waiting* hops. Built-in rules match every walker on its own
hyperedge, so they never overlap; Collide's pairs are disjoint by the exclusion principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from imigraph.graph.gas import DIRECTIONS, OPPOSITE, RULES, LatticeGas, LatticeGasState
from imigraph.graph.rules import DIRECTION_NAMES, ChainRule, RuleSpec, rule_specs
from imigraph.graph.simulation import GraphHistory
from imigraph.graph.space import SquareLattice
from imigraph.graph.state import WalkerGraph
from imigraph.metrics import StepObserver

WALKER_MODELS = ("edge", "hyperedge")
COLLISIONS = ("none", "hpp")
OVERLAPS = ("parallel", "ordered", "random", "oldest")
OVERLAP_TEXT = {
    "parallel": "parallel: every match is applied, site edges are shared read-only",
    "ordered": "ordered: matches applied walker by walker in index order, a match sharing "
    "a site edge with one already applied this step waits",
    "random": "random: like ordered, in an order drawn afresh every step",
    "oldest": "oldest: like ordered, the walker that has waited longest goes first",
}
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
    overlaps: str = "parallel"
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
        if self.overlaps not in OVERLAPS:
            raise ValueError(
                f"overlaps must be one of {', '.join(OVERLAPS)}, not {self.overlaps!r}"
            )
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
            f"; overlapping matches: {OVERLAP_TEXT[self.overlaps]}"
        )
        if len(described) == 1:
            # One rule and no collision: the rule is the whole set, described once; the
            # model is named when it is not the plain Brownian setting.
            if self.walker_model == "hyperedge" or self.overlaps != "parallel":
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
        if self.overlaps != "parallel":
            blocked += (
                f" Overlapping matches ({self.overlaps}): a walker whose chain shares a site "
                "edge with a match applied earlier in the step waits, which lowers the "
                "measured values where walkers crowd."
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
        self,
        anchors: np.ndarray,
        heading: str,
        rng: np.random.Generator,
        directions: np.ndarray | None = None,
    ) -> WalkerGraph | LatticeGasState:
        """The state at step 0; ``directions`` gives every hyperedge walker's heading label
        explicitly (one per anchor), otherwise ``heading`` chooses them."""
        anchors = np.asarray(anchors, dtype=np.int64)
        if self.walker_model == "edge":
            return WalkerGraph(self.space, anchors)
        if directions is None:
            directions = initial_headings(anchors, heading, rng)
        directions = np.asarray(directions, dtype=np.int64)
        if directions.shape != anchors.shape:
            raise ValueError("give one heading per walker")
        return LatticeGasState(self.space, anchors, directions)

    def step(
        self,
        state: WalkerGraph | LatticeGasState,
        rng: np.random.Generator,
        ages: np.ndarray | None = None,
    ) -> tuple[WalkerGraph | LatticeGasState, np.ndarray, np.ndarray, int, np.ndarray]:
        """One step: the new state, the concatenated matched paths ``(n, L)``, the sites where
        Collide fired, the number of blocked hops and the mask of walkers that waited because
        of an overlapping match (``ages`` orders the ``oldest`` strategy)."""
        pieces: list[np.ndarray] = []
        collided = np.zeros(0, dtype=np.int64)
        blocked_total = 0
        n = state.num_walkers
        waited = np.zeros(n, dtype=bool)
        if isinstance(state, WalkerGraph):
            for rule in self.rules:
                assert isinstance(rule, ChainRule)
                _, path, _ = rule.apply_with_choices(state, rng)
                applied = select_matches(
                    path, self.space.num_nodes, self.overlaps, rng, ages
                )
                target = rule.spec.chain()[2]
                anchors = np.where(applied, path[:, target], state.anchors)
                state = WalkerGraph(self.space, anchors)
                waited |= ~applied
                _append_path(pieces, np.where(applied[:, None], path, path[:, :1]))
            return state, np.concatenate(pieces, axis=1), collided, 0, waited
        if self.collisions == "hpp":
            state, collided = self.gas.collide(state, rng)
        for rule in self.rules:
            if isinstance(rule, ChainRule):
                _, path, labels = rule.apply_with_choices(
                    WalkerGraph(self.space, state.sites), rng
                )
                applied = select_matches(
                    path, self.space.num_nodes, self.overlaps, rng, ages
                )
                target = rule.spec.chain()[2]
                sites = np.where(applied, path[:, target], state.sites)
                directions = (
                    np.where(applied, labels[:, target - 1], state.directions)
                    if target > 0
                    else state.directions
                )
                path = np.where(applied[:, None], path, path[:, :1])
                waited |= ~applied
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
        return state, np.concatenate(pieces, axis=1), collided, blocked_total, waited

    def run(
        self,
        initial: WalkerGraph | LatticeGasState,
        steps: int,
        rng: np.random.Generator,
        observer: StepObserver | None = None,
    ) -> tuple[GraphHistory, int, int]:
        """Every state of ``steps`` steps with the matched paths (and headings, collision sites
        for hyperedge walkers); returns the history, the total number of blocked hops and the
        total number of waiting hops (overlapping matches skipped)."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        coordinates = self.space.coordinates
        anchors_of = lambda s: s.anchors if isinstance(s, WalkerGraph) else s.sites
        frames, directions, paths, collisions = [anchors_of(initial)], [], [], []
        if isinstance(initial, LatticeGasState):
            directions.append(initial.directions)
        state = initial
        blocked_total = 0
        waited_total = 0
        ages = np.zeros(initial.num_walkers, dtype=np.int64)
        for step in range(1, steps + 1):
            previous = anchors_of(state)
            state, path, collided, blocked, waited = self.step(state, rng, ages)
            current = anchors_of(state)
            blocked_total += blocked
            waited_total += int(waited.sum())
            ages = np.where(waited, ages + 1, 0)
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
        return history, blocked_total, waited_total


def momentum(history: GraphHistory) -> np.ndarray | None:
    """Total momentum of hyperedge walkers at every frame in units of ``m a / tau``: the sum
    of the heading unit vectors, shape ``(T, 2)``. ``None`` for edge walkers."""
    if history.directions is None:
        return None
    return DIRECTIONS[history.directions].sum(axis=1)


def heading_labels(directions: np.ndarray) -> np.ndarray:
    return np.asarray(DIRECTION_NAMES)[directions]


def explain_frames(history: GraphHistory, detail: int = 8) -> list[str]:
    """One caption per frame of a hyperedge run: what Collide, Reflect and Stream did in the
    step that led to the frame, and the total momentum. Walkers are named by index and
    listed one by one up to ``detail`` walkers, counted beyond that."""
    if history.directions is None or history.collisions is None:
        return []
    space = history.space
    coords = space.coordinates
    cell = space.cell_size
    lower = space.box.lower

    def at(site: int) -> str:
        i, j = np.rint((coords[site] - lower - cell / 2) / cell).astype(int)
        return f"({i},{j})"

    total = momentum(history)
    assert total is not None
    n = history.num_walkers
    few = n <= detail
    captions = []
    labels = heading_labels(history.directions)
    for frame in range(history.num_frames):
        parts: list[str] = []
        if frame == 0:
            if few:
                parts.append(
                    "; ".join(
                        f"walker {w} at {at(int(history.anchors[0, w]))} heading "
                        f"{labels[0, w]}"
                        for w in range(n)
                    )
                )
            else:
                parts.append(f"{n} walkers placed with their headings")
        else:
            before, after = frame - 1, frame
            sites_b, sites_a = history.anchors[before], history.anchors[after]
            dirs_b, dirs_a = history.directions[before], history.directions[after]
            collided_sites = history.collisions[before]
            collided = np.isin(sites_b, collided_sites) & (dirs_b != dirs_a)
            for site in collided_sites:
                pair = np.flatnonzero((sites_b == site) & collided)
                if few:
                    parts.append(
                        f"Collide at {at(int(site))}: walkers "
                        + " and ".join(str(w) for w in pair)
                        + " met head-on ("
                        + ", ".join(labels[before, w] for w in pair)
                        + ") and left along the perpendicular pair ("
                        + ", ".join(labels[after, w] for w in pair)
                        + "); momentum of the pair 0 before and after"
                    )
            if not few and collided_sites.size:
                parts.append(f"Collide at {collided_sites.size} site(s)")
            turned = (dirs_a == OPPOSITE[dirs_b]) & ~collided
            wall = space.neighbors[sites_b, dirs_b] == sites_b
            reflected = np.flatnonzero(turned & wall)
            if reflected.size:
                parts.append(
                    "Reflect: "
                    + (
                        ", ".join(
                            f"walker {w} turned round at the wall ({labels[before, w]} to "
                            f"{labels[after, w]})"
                            for w in reflected
                        )
                        if few
                        else f"{reflected.size} walker(s) turned round at a wall"
                    )
                    + "; the wall takes the momentum"
                )
            moved = np.flatnonzero(sites_a != sites_b)
            stayed = np.flatnonzero(sites_a == sites_b)
            if moved.size:
                parts.append(
                    "Stream: "
                    + (
                        ", ".join(
                            f"walker {w} {at(int(sites_b[w]))} to {at(int(sites_a[w]))}"
                            for w in moved
                        )
                        if few
                        else f"{moved.size} walker(s) moved one edge"
                    )
                )
            if stayed.size:
                parts.append(
                    (
                        ", ".join(f"walker {w}" for w in stayed)
                        if few
                        else f"{stayed.size} walker(s)"
                    )
                    + " stayed (heading into a wall without Reflect, or blocked)"
                )
        px, py = total[frame]
        parts.append(
            f"total momentum ({px:g}, {py:g}) m a/tau; every walker keeps speed a/tau"
        )
        captions.append(f"step {frame}: " + ". ".join(parts))
    return captions


def _append_path(pieces: list[np.ndarray], path: np.ndarray) -> None:
    if pieces and np.array_equal(pieces[-1][:, -1], path[:, 0]):
        path = path[:, 1:]
    if path.shape[1]:
        pieces.append(path)
