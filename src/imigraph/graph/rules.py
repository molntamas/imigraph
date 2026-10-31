"""Rules: how a :class:`~imigraph.graph.state.WalkerGraph` evolves in one step.

A rule maps a state to the next state. Rules are written in the spirit of Wolfram-model rewriting
rules (a pattern of edges is replaced by another pattern), applied to every walker simultaneously
(synchronous update). All randomness comes from the ``rng`` passed in.

Two kinds live here:

* :class:`RandomHop`, the hand-written reference rule ``{w,s}, {s,s'} -> {w,s'}, {s,s'}``;
* :class:`ChainRule`, built from a :class:`RuleSpec` (plain data, JSON-serialisable, editable in
  the GUI): a walker-anchored *chain* pattern ``{w,s0}, {s0,s1}, ..., {s(k-1),sk}`` whose right
  side re-attaches the walker to one site of the chain. Each chain edge is matched by a random
  labelled edge (``+x, -x, +y, -y``) with the spec's weights, so the rule family covers the
  uniform random hop, biased hops (drift), multi-hop steps and walks that never stay in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from imigraph.graph.state import WalkerGraph

DIRECTION_NAMES = ("+x", "-x", "+y", "-y")
UNIFORM = (0.25, 0.25, 0.25, 0.25)


@runtime_checkable
class Rule(Protocol):
    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        """Return the state after one step."""
        ...


def _choose_directions(
    neighbors: np.ndarray,
    current: np.ndarray,
    weights: np.ndarray,
    allow_self_loops: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """One labelled edge per walker, drawn with ``weights`` (self-loops optionally excluded)."""
    table = np.broadcast_to(weights, (current.shape[0], weights.shape[0])).astype(float)
    if not allow_self_loops:
        real = neighbors[current] != current[:, None]
        table = np.where(real, table, 0.0)
        stuck = table.sum(axis=1) == 0  # a corner whose only weighted edges are loops
        table[stuck] = weights
    cumulative = np.cumsum(table, axis=1)
    cumulative /= cumulative[:, -1:]
    u = rng.random(current.shape[0])
    return np.minimum((u[:, None] >= cumulative).sum(axis=1), weights.shape[0] - 1)


@dataclass(frozen=True)
class RandomHop:
    """The edge-rewriting rule ``{w, s}, {s, s'}  ->  {w, s'}, {s, s'}``.

    For every walker ``w`` anchored at space node ``s``, one of the edges ``{s, s'}`` at ``s`` is
    chosen uniformly at random (self-loops count, so a walker at the boundary may stay) and the
    walker edge is moved along it. On a lattice with spacing ``a`` and step time ``tau`` this is
    a random walk with diffusion coefficient ``D = a^2 / (4 tau)`` (see
    :func:`~imigraph.graph.simulation.lattice_hop_diffusion`).
    """

    name = "RandomHop"
    rewrite = "{w,s}, {s,s'} -> {w,s'}, {s,s'}"
    matches = "one match per edge {s,s'} at the anchor s of walker w; self-loops {s,s} included"
    choice = "one match per walker, uniformly at random; all walkers rewritten at once"
    invariants = "number of walkers, one anchoring edge per walker, the space graph"
    physics = "random walk with step a per application: D = a^2 / (4 tau); self-loops reflect"

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        """Human-readable description of the rule for reports and GUIs (the arguments are
        accepted for symmetry with :class:`ChainRule`; the text is fixed)."""
        return {
            "name": self.name,
            "rewrite": self.rewrite,
            "matches": self.matches,
            "choice": self.choice,
            "invariants": self.invariants,
            "physics": self.physics,
        }

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec.parse(self.rewrite, name=self.name)

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        neighbors = state.space.neighbors
        choice = rng.integers(0, neighbors.shape[1], size=state.num_walkers)
        return WalkerGraph(state.space, neighbors[state.anchors, choice])

    def apply_with_paths(
        self, state: WalkerGraph, rng: np.random.Generator
    ) -> tuple[WalkerGraph, np.ndarray]:
        after = self.apply(state, rng)
        return after, np.stack([state.anchors, after.anchors], axis=1)


_EDGE = re.compile(r"\{([^{}]*)\}")


@dataclass(frozen=True)
class RuleSpec:
    """A rule as data: the pattern (``lhs``) and its replacement (``rhs``) as edge lists over
    variables, direction ``weights`` for matching the chain edges, and whether self-loops (the
    walls) may be matched. Variables whose name starts with ``w`` are walkers, others sites,
    unless ``types`` says otherwise."""

    name: str
    lhs: tuple[tuple[str, ...], ...]
    rhs: tuple[tuple[str, ...], ...]
    weights: tuple[float, float, float, float] = UNIFORM
    self_loops: str = "allow"
    types: dict[str, str] = field(default_factory=dict, compare=False)

    # ---------- text and JSON forms ----------

    @classmethod
    def parse(cls, notation: str, name: str = "custom", **options: Any) -> RuleSpec:
        """``"{w,s}, {s,t} -> {w,t}, {s,t}"`` -> spec. Edge variables are separated by commas."""
        if "->" not in notation:
            raise ValueError("a rule needs a left side, '->' and a right side")
        left, right = notation.split("->", 1)
        sides = []
        for text in (left, right):
            edges = tuple(
                tuple(v.strip() for v in body.split(",") if v.strip())
                for body in _EDGE.findall(text)
            )
            if not edges:
                raise ValueError(f"no edges found in {text.strip()!r}; write edges as {{a,b}}")
            if any(len(edge) < 2 for edge in edges):
                raise ValueError("every edge needs at least two variables, e.g. {w,s}")
            sides.append(edges)
        return cls(name, sides[0], sides[1], **options)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleSpec:
        if "notation" in data and "lhs" not in data:
            options = {k: v for k, v in data.items() if k not in ("notation", "name")}
            spec = cls.parse(data["notation"], name=data.get("name", "custom"))
            return cls(spec.name, spec.lhs, spec.rhs, **cls._options(options))
        return cls(
            str(data.get("name", "custom")),
            tuple(tuple(str(v) for v in edge) for edge in data["lhs"]),
            tuple(tuple(str(v) for v in edge) for edge in data["rhs"]),
            **cls._options(data),
        )

    @staticmethod
    def _options(data: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if "weights" in data:
            options["weights"] = tuple(float(w) for w in data["weights"])
        if "self_loops" in data:
            options["self_loops"] = str(data["self_loops"])
        if "types" in data:
            options["types"] = dict(data["types"])
        return options

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "notation": self.notation,
            "lhs": [list(edge) for edge in self.lhs],
            "rhs": [list(edge) for edge in self.rhs],
            "weights": list(self.weights),
            "self_loops": self.self_loops,
            "types": {v: self.type_of(v) for v in self.variables},
        }

    @property
    def notation(self) -> str:
        edges = lambda side: ", ".join("{" + ",".join(edge) + "}" for edge in side)  # noqa: E731
        return f"{edges(self.lhs)} -> {edges(self.rhs)}"

    # ---------- structure ----------

    @property
    def variables(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for edge in self.lhs + self.rhs:
            for v in edge:
                seen.setdefault(v, None)
        return tuple(seen)

    def type_of(self, variable: str) -> str:
        return self.types.get(variable, "walker" if variable.startswith("w") else "site")

    def __post_init__(self) -> None:
        if len(self.weights) != 4 or any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("weights must be four non-negative numbers (+x, -x, +y, -y)")
        if self.self_loops not in ("allow", "forbid"):
            raise ValueError("self_loops must be 'allow' or 'forbid'")
        self.chain()  # validate the pattern early, with a message for the editor

    def chain(self) -> tuple[str, list[str], int]:
        """Check that the spec is a walker-anchored chain rule and return
        ``(walker, [s0, s1, ..., sk], target index)``. Raises ``ValueError`` otherwise."""
        walker_edges = [e for e in self.lhs if any(self.type_of(v) == "walker" for v in e)]
        site_edges = [e for e in self.lhs if e not in walker_edges]
        if len(walker_edges) != 1:
            raise ValueError(
                "the left side must contain exactly one walker edge {w,s} (one walker per "
                "pattern; interaction rules are not supported by the chain rule family)"
            )
        anchor_edge = walker_edges[0]
        if len(anchor_edge) != 2 or sum(self.type_of(v) == "walker" for v in anchor_edge) != 1:
            raise ValueError("the walker edge must join one walker and one site: {w,s}")
        walker = next(v for v in anchor_edge if self.type_of(v) == "walker")
        start = next(v for v in anchor_edge if v != walker)
        chain = [start]
        remaining = list(site_edges)
        while remaining:
            for edge in remaining:
                if len(edge) != 2:
                    raise ValueError(f"site edges must have two ends, got {{{','.join(edge)}}}")
                a, b = edge
                if a == chain[-1] and b not in chain:
                    chain.append(b)
                elif b == chain[-1] and a not in chain:
                    chain.append(a)
                else:
                    continue
                remaining.remove(edge)
                break
            else:
                raise ValueError(
                    "the site edges must form a chain {s0,s1}, {s1,s2}, ... starting at the "
                    f"walker's site {start}; could not place {{{','.join(remaining[0])}}}"
                )
        rhs_walker = [e for e in self.rhs if walker in e]
        rhs_sites = [e for e in self.rhs if walker not in e]
        if len(rhs_walker) != 1 or len(rhs_walker[0]) != 2:
            raise ValueError("the right side must re-attach the walker by exactly one edge {w,s}")
        target = next(v for v in rhs_walker[0] if v != walker)
        if target not in chain:
            raise ValueError(f"the walker must end on a site of the chain, not on {target!r}")
        if sorted(map(frozenset, rhs_sites)) != sorted(map(frozenset, site_edges)):
            raise ValueError("the space is static: the right side must keep the site edges")
        return walker, chain, chain.index(target)

    # ---------- physics ----------

    @property
    def normalized_weights(self) -> np.ndarray:
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    @property
    def hops(self) -> int:
        """Number of chain edges the walker actually moves along."""
        return self.chain()[2]

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        """Mean velocity ``hops a (w+x - w-x, w+y - w-y) / tau`` (far from the walls)."""
        w = self.normalized_weights
        return self.hops * spacing * np.array([w[0] - w[1], w[2] - w[3]]) / step_time

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        """Per-component diffusion ``hops a^2 (w+ + w- - (w+ - w-)^2) / (2 tau)``: the variance
        of the displacement per step over ``2 tau``. Uniform weights give ``a^2 / (4 tau)`` per
        hop, the value of :func:`~imigraph.graph.simulation.lattice_hop_diffusion`."""
        w = self.normalized_weights
        var = np.array([w[0] + w[1] - (w[0] - w[1]) ** 2, w[2] + w[3] - (w[2] - w[3]) ** 2])
        return self.hops * spacing**2 * var / (2 * step_time)

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        walker, chain, target = self.chain()
        k = len(chain) - 1
        w = self.normalized_weights
        drift = self.drift(spacing, step_time)
        diffusion = self.diffusion(spacing, step_time)
        weights = ", ".join(f"{n} {p:.3g}" for n, p in zip(DIRECTION_NAMES, w, strict=True))
        loops = (
            "self-loops {s,s} included (a walker at the wall may stay)"
            if self.self_loops == "allow"
            else "self-loops excluded (a walker at the wall must move)"
        )
        return {
            "name": self.name,
            "rewrite": self.notation,
            "matches": f"one labelled edge per chain edge ({k} chain edge{'s' if k != 1 else ''} "
            f"from the anchor of {walker}); {loops}",
            "choice": f"directions drawn with weights {weights}; all walkers rewritten at once",
            "invariants": "number of walkers, one anchoring edge per walker, the space graph",
            "physics": f"{target} hop{'s' if target != 1 else ''} per application: "
            f"D = ({diffusion[0]:.4g}, {diffusion[1]:.4g}), drift v = ({drift[0]:.4g}, "
            f"{drift[1]:.4g}) for a = {spacing:g}, tau = {step_time:g}",
        }


RANDOM_HOP = RuleSpec.parse("{w,s}, {s,s'} -> {w,s'}, {s,s'}", name="RandomHop")


@dataclass(frozen=True)
class ChainRule:
    """The rule described by a :class:`RuleSpec` (see the module docstring)."""

    spec: RuleSpec

    @property
    def name(self) -> str:
        return self.spec.name

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        return self.spec.describe(spacing, step_time)

    def apply_with_paths(
        self, state: WalkerGraph, rng: np.random.Generator
    ) -> tuple[WalkerGraph, np.ndarray]:
        """The new state and the matched chain of every walker, shape ``(n, k + 1)``."""
        _, chain, target = self.spec.chain()
        neighbors = state.space.neighbors
        weights = self.spec.normalized_weights
        allow = self.spec.self_loops == "allow"
        path = [state.anchors]
        for _ in range(len(chain) - 1):
            choice = _choose_directions(neighbors, path[-1], weights, allow, rng)
            path.append(neighbors[path[-1], choice])
        return WalkerGraph(state.space, path[target]), np.stack(path, axis=1)

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        return self.apply_with_paths(state, rng)[0]


def rule_from_spec(spec: RuleSpec | dict[str, Any] | None) -> Rule:
    """``None`` -> :class:`RandomHop`; a spec or its JSON form -> :class:`ChainRule`."""
    if spec is None:
        return RandomHop()
    if isinstance(spec, dict):
        spec = RuleSpec.from_dict(spec)
    return ChainRule(spec)
