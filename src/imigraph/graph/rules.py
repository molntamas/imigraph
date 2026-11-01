"""Rules: how a :class:`~imigraph.graph.state.WalkerGraph` evolves.

A rule is a rewrite in the spirit of the Wolfram models: a pattern of (hyper)edges over
variables on the left, its replacement on the right. The only walker is the *ball*
``{w,s,v}`` (walker ``w`` on site ``s`` with velocity node ``v``, see
:mod:`imigraph.graph.state`); rules rewrite ball hyperedges, the room and the velocity lattice
are static and are only read. All randomness comes from the ``rng`` passed in.

**Notation.** Variables starting with ``w`` are walkers, with ``v`` velocities (``o`` is the
origin, the velocity at rest), everything else sites; ``types`` can override. Edges may carry
an *annotation* that says which edge of the two lattices is meant:

======================  =====================================================================
``{w,s}``               a ball on site ``s`` (its velocity is not mentioned and kept)
``{w,s,v}``             a ball on site ``s`` with velocity ``v``
``{w,s,v}_k``           the same, and ``k`` is the direction of its next hop in this step
``{s,t}``               a room edge from ``s``; its label is drawn with the rule's weights
``{s,t}_k``             the room edge of label ``k`` from ``s`` (``k`` bound earlier, or drawn)
``{s,s}_k``             the edge of label ``k`` at ``s`` is a wall self-loop
``{s,x}_k1, {t,x}_k2``  the next hops of two balls *meet* at ``x`` (they land on one site, cross
                        one edge, or already share a site): a contact
``{s,t}=d``             ``d`` is the lattice direction of the line from ``s`` to ``t`` (an axis
                        or a diagonal; the line of centres of two balls)
``{v,v'}_k``            ``v' = v + e_k``, the edge of label ``k`` of the velocity lattice
``{v,v'}=d``            ``v'`` lies on the line through ``v`` along ``d``
``{v,v'}>d``            ``v'`` is ahead of ``v`` along ``d`` (``(v' - v) . d > 0``)
``{v,v'}|k``            ``v'`` is the mirror image of ``v`` across the axis of ``k``
``{v,v'}|o``            ``v' = -v`` (the point mirror)
``{v,v'}^``             ``v'`` is ``v`` turned by a quarter turn (either sense, drawn at random)
``d^``, ``-k``          the perpendicular of ``d``; the opposite of ``k``
======================  =====================================================================

A velocity variable that is not bound by a ball is *solved*: the matcher enumerates the nodes
of the velocity lattice (components up to the system's ``max_speed``) and keeps those that
satisfy every relation; no node means the rule does not match, several mean one is drawn at
random. This is how the collision rules stay pure graph rewrites: the elastic exchange is the
rectangle spanned by the two velocities in the velocity lattice, the wall is a mirror, the HPP
collision a quarter turn.

The right side names, for every ball of the left side, the site and the velocity it ends
with: ``{w,t,v'}`` (or ``{w,t}`` to keep the velocity). A rule whose balls carry a hop label is a
*round rule*: it is matched once per round of a step, only on balls that still have hops to
take, and a ball that moves to its hop target consumes one hop (:mod:`imigraph.graph.world`).
Rules without hop labels run once per step. Chain rules, the family of the Brownian imitation
(``{w,s}, {s,t}, {t,u} -> {w,u}``), are run vectorised by :class:`ChainRule`; everything else
by the generic matcher of :class:`BallRule`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from imigraph.graph.state import WalkerGraph, body_fits, lattice_cells
from imigraph.graph.velocity import (
    AXIS_NAMES,
    DIRECTION_NAMES,
    DIRECTIONS,
    direction_of,
    velocity_box,
)

UNIFORM = (0.25, 0.25, 0.25, 0.25)
OPS = ("", "_", "=", ">", "|", "^")


@runtime_checkable
class Rule(Protocol):
    @property
    def name(self) -> str: ...

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        """Human-readable description for reports and GUIs."""
        ...

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
    physics = (
        "random walk with step a per application: D = a^2 / (4 tau); self-loops reflect"
    )

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

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return (self.spec,)

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        neighbors = state.space.neighbors
        choice = rng.integers(0, neighbors.shape[1], size=state.num_walkers)
        return WalkerGraph(
            state.space,
            neighbors[state.anchors, choice],
            state.velocities,
            state.radius,
        )

    def apply_with_paths(
        self, state: WalkerGraph, rng: np.random.Generator
    ) -> tuple[WalkerGraph, np.ndarray]:
        after = self.apply(state, rng)
        return after, np.stack([state.anchors, after.anchors], axis=1)


# ---------- the notation ----------

_EDGE = re.compile(r"\{([^{}]*)\}\s*(?:([_=>|])\s*([^\s,{}]+)|(\^))?")
_TOKEN = re.compile(r"^(-)?([A-Za-z][A-Za-z0-9']*)(\^)?$")

Edges = tuple[tuple[str, ...], ...]
Ops = tuple[tuple[str, str], ...]


def _parse_side(text: str) -> tuple[Edges, Ops]:
    edges: list[tuple[str, ...]] = []
    ops: list[tuple[str, str]] = []
    for m in _EDGE.finditer(text):
        variables = tuple(v.strip() for v in m.group(1).split(",") if v.strip())
        op = "^" if m.group(4) else (m.group(2) or "")
        edges.append(variables)
        ops.append((op, m.group(3) or ""))
    if not edges:
        raise ValueError(f"no edges found in {text.strip()!r}; write edges as {{a,b}}")
    if any(len(edge) < 2 for edge in edges):
        raise ValueError("every edge needs at least two variables, e.g. {w,s}")
    return tuple(edges), tuple(ops)


def _edge_text(edge: tuple[str, ...], op: tuple[str, str]) -> str:
    return "{" + ",".join(edge) + "}" + op[0] + op[1]


@dataclass(frozen=True)
class RuleSpec:
    """A rule as data: the pattern (``lhs``) and its replacement (``rhs``) as edge lists over
    variables with their annotations (``lhs_ops`` / ``rhs_ops``, one ``(op, argument)`` pair
    per edge, ``("", "")`` for none), direction ``weights`` for drawn chain edges, and whether
    self-loops (the walls) may be drawn. Variables whose name starts with ``w`` are walkers,
    with ``v`` (or ``o``) velocities, others sites, unless ``types`` says otherwise."""

    name: str
    lhs: Edges
    rhs: Edges
    weights: tuple[float, float, float, float] = UNIFORM
    self_loops: str = "allow"
    types: dict[str, str] = field(default_factory=dict, compare=False)
    description: str = field(default="", compare=False)
    """Free text of the author: what the rule is meant to do (shown with the rule)."""
    lhs_ops: Ops = ()
    rhs_ops: Ops = ()

    # ---------- text and JSON forms ----------

    @classmethod
    def parse(cls, notation: str, name: str = "custom", **options: Any) -> RuleSpec:
        """``"{w,s}, {s,t} -> {w,t}, {s,t}"`` -> spec. Edge variables are separated by commas;
        an annotation follows the closing brace (``{s,t}_k``, ``{v,v'}=d``, ``{v,v'}^``)."""
        if "->" not in notation:
            raise ValueError("a rule needs a left side, '->' and a right side")
        left, right = notation.split("->", 1)
        lhs, lhs_ops = _parse_side(left)
        rhs, rhs_ops = _parse_side(right)
        return cls(name, lhs, rhs, lhs_ops=lhs_ops, rhs_ops=rhs_ops, **options)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleSpec:
        if "notation" in data:
            spec = cls.parse(data["notation"], name=str(data.get("name", "custom")))
            return cls(
                spec.name,
                spec.lhs,
                spec.rhs,
                lhs_ops=spec.lhs_ops,
                rhs_ops=spec.rhs_ops,
                **cls._options(data),
            )
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
        if "description" in data:
            options["description"] = str(data["description"] or "")
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
            "description": self.description,
            "family": self.family,
        }

    @property
    def notation(self) -> str:
        left = ", ".join(
            _edge_text(e, o) for e, o in zip(self.lhs, self.lhs_ops, strict=True)
        )
        right = ", ".join(
            _edge_text(e, o) for e, o in zip(self.rhs, self.rhs_ops, strict=True)
        )
        return f"{left} -> {right}"

    # ---------- structure ----------

    @property
    def variables(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for edge in self.lhs + self.rhs:
            for v in edge:
                seen.setdefault(v, None)
        return tuple(seen)

    def type_of(self, variable: str) -> str:
        if variable in self.types:
            return self.types[variable]
        if variable.startswith("w"):
            return "walker"
        if variable == "o" or variable.startswith("v"):
            return "velocity"
        return "site"

    def __post_init__(self) -> None:
        if (
            len(self.weights) != 4
            or any(w < 0 for w in self.weights)
            or sum(self.weights) <= 0
        ):
            raise ValueError(
                "weights must be four non-negative numbers (+x, -x, +y, -y)"
            )
        if self.self_loops not in ("allow", "forbid"):
            raise ValueError("self_loops must be 'allow' or 'forbid'")
        for name, edges, ops in (
            ("lhs", self.lhs, self.lhs_ops),
            ("rhs", self.rhs, self.rhs_ops),
        ):
            if not ops:
                ops = tuple(("", "") for _ in edges)
                object.__setattr__(self, f"{name}_ops", ops)
            if len(ops) != len(edges):
                raise ValueError("one annotation per edge")
            for op, arg in ops:
                if (
                    op not in OPS
                    or (op == "^" and arg)
                    or (op in "_=>|" and op and not arg)
                ):
                    raise ValueError(f"bad annotation {op}{arg} in {name}")
        if not self.lhs or not self.rhs:
            raise ValueError("both sides need at least one edge")
        if self.family == "chain":
            self.chain()
        else:
            self.plan()  # validate the pattern early, with a message for the editor

    @property
    def family(self) -> str:
        """``chain`` for the vectorised family of the Brownian imitation (one ball ``{w,s}``
        without velocity, drawn site edges only), ``ball`` for everything else."""
        if any(op[0] for op in self.lhs_ops + self.rhs_ops):
            return "ball"
        if any(self.type_of(v) == "velocity" for v in self.variables):
            return "ball"
        walker_edges = [
            e for e in self.lhs if any(self.type_of(v) == "walker" for v in e)
        ]
        if len(walker_edges) != 1 or len(walker_edges[0]) != 2:
            return "ball" if len(walker_edges) == 2 else "chain"
        return "chain"

    def chain(self) -> tuple[str, list[str], int]:
        """Check that the spec is a walker-anchored chain rule and return
        ``(walker, [s0, s1, ..., sk], target index)``. Raises ``ValueError`` otherwise."""
        walker_edges = [
            e for e in self.lhs if any(self.type_of(v) == "walker" for v in e)
        ]
        site_edges = [e for e in self.lhs if e not in walker_edges]
        if len(walker_edges) != 1:
            raise ValueError(
                "the left side must contain exactly one walker edge {w,s} (a rule with two "
                "balls needs a contact {s,x}_k1, {t,x}_k2 and the balls' velocities)"
            )
        anchor_edge = walker_edges[0]
        if (
            len(anchor_edge) != 2
            or sum(self.type_of(v) == "walker" for v in anchor_edge) != 1
        ):
            raise ValueError("the walker edge must join one walker and one site: {w,s}")
        walker = next(v for v in anchor_edge if self.type_of(v) == "walker")
        start = next(v for v in anchor_edge if v != walker)
        chain = [start]
        remaining = list(site_edges)
        while remaining:
            for edge in remaining:
                if len(edge) != 2:
                    raise ValueError(
                        f"site edges must have two ends, got {{{','.join(edge)}}}"
                    )
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
            raise ValueError(
                "the right side must re-attach the walker by exactly one edge {w,s}"
            )
        target = next(v for v in rhs_walker[0] if v != walker)
        if target not in chain:
            raise ValueError(
                f"the walker must end on a site of the chain, not on {target!r}"
            )
        if sorted(map(frozenset, rhs_sites)) != sorted(map(frozenset, site_edges)):
            raise ValueError(
                "the space is static: the right side must keep the site edges"
            )
        return walker, chain, chain.index(target)

    def plan(self) -> BallPlan:
        """The validated, compiled form of a ball rule (see :class:`BallPlan`)."""
        return BallPlan.compile(self)

    # ---------- physics ----------

    @property
    def normalized_weights(self) -> np.ndarray:
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    @property
    def hops(self) -> int:
        """Number of chain edges the walker actually moves along (chain rules)."""
        return self.chain()[2] if self.family == "chain" else 0

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        """Mean velocity ``hops a (w+x - w-x, w+y - w-y) / tau`` (far from the walls)."""
        if self.family != "chain":
            return np.zeros(2)
        w = self.normalized_weights
        return self.hops * spacing * np.array([w[0] - w[1], w[2] - w[3]]) / step_time

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        """Per-component diffusion ``hops a^2 (w+ + w- - (w+ - w-)^2) / (2 tau)``: the variance
        of the displacement per step over ``2 tau``. Uniform weights give ``a^2 / (4 tau)`` per
        hop, the value of :func:`~imigraph.graph.simulation.lattice_hop_diffusion`."""
        if self.family != "chain":
            return np.zeros(2)
        w = self.normalized_weights
        var = np.array(
            [w[0] + w[1] - (w[0] - w[1]) ** 2, w[2] + w[3] - (w[2] - w[3]) ** 2]
        )
        return self.hops * spacing**2 * var / (2 * step_time)

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        if self.family != "chain":
            return self.plan().describe(self)
        walker, chain, target = self.chain()
        k = len(chain) - 1
        w = self.normalized_weights
        drift = self.drift(spacing, step_time)
        diffusion = self.diffusion(spacing, step_time)
        weights = ", ".join(
            f"{n} {p:.3g}" for n, p in zip(DIRECTION_NAMES, w, strict=True)
        )
        loops = (
            "self-loops {s,s} included (a walker at the wall may stay)"
            if self.self_loops == "allow"
            else "self-loops excluded (a walker at the wall must move)"
        )
        return {
            "name": self.name,
            "description": self.description,
            "rewrite": self.notation,
            "matches": f"one labelled edge per chain edge ({k} chain edge{'s' if k != 1 else ''} "
            f"from the anchor of {walker}); {loops}; the ball's velocity is kept",
            "choice": f"directions drawn with weights {weights}; all walkers rewritten at once",
            "invariants": "number of walkers, one anchoring edge per walker, the space graph",
            "physics": f"{target} hop{'s' if target != 1 else ''} per application: "
            f"D = ({diffusion[0]:.4g}, {diffusion[1]:.4g}), drift v = ({drift[0]:.4g}, "
            f"{drift[1]:.4g}) for a = {spacing:g}, tau = {step_time:g}",
        }


# ---------- ball rules: the compiled pattern ----------


@dataclass(frozen=True)
class DirRef:
    """A direction argument of an annotation: a fixed direction (``+x``), the origin (``o``,
    only for the point mirror), or a variable, possibly negated (``-k``) or perpendicular
    (``d^``)."""

    name: str
    negate: bool = False
    perp: bool = False
    fixed: tuple[int, int] | None = None
    origin: bool = False

    @classmethod
    def parse(cls, text: str) -> DirRef:
        if text in DIRECTION_NAMES:
            d = DIRECTIONS[DIRECTION_NAMES.index(text)]
            return cls(text, fixed=(int(d[0]), int(d[1])))
        if text == "o":
            return cls("o", origin=True)
        m = _TOKEN.match(text)
        if m is None:
            raise ValueError(
                f"bad direction {text!r}: use a variable like k or d, -k, d^, or +x, -x, +y, -y"
            )
        return cls(
            m.group(2), negate=m.group(1) is not None, perp=m.group(3) is not None
        )


@dataclass(frozen=True)
class BallEdge:
    walker: str
    site: str
    velocity: str | None
    label: str | None  # the direction variable bound to the ball's next hop


@dataclass(frozen=True)
class Step:
    """One left-side edge of a ball rule, as the matcher executes it in order."""

    kind: str  # wall, hop, contact, chain, line, ahead, relation
    a: str
    b: str
    op: str = ""
    direction: DirRef | None = None


@dataclass(frozen=True)
class BallPlan:
    """A ball rule compiled: its balls, the steps of the match in order, the unknown velocity
    variables to solve, and what every ball ends with."""

    balls: tuple[BallEdge, ...]
    steps: tuple[Step, ...]
    unknowns: tuple[str, ...]
    results: tuple[tuple[str, str, str | None], ...]  # (walker, site, velocity or None)
    hop_targets: dict[str, str]  # walker -> site variable of its hop target

    @property
    def is_round(self) -> bool:
        return any(ball.label is not None for ball in self.balls)

    @property
    def is_pair(self) -> bool:
        return len(self.balls) == 2

    @classmethod
    def compile(cls, spec: RuleSpec) -> BallPlan:
        is_walker = lambda v: spec.type_of(v) == "walker"
        is_velocity = lambda v: spec.type_of(v) == "velocity"
        balls: list[BallEdge] = []
        others: list[tuple[tuple[str, ...], tuple[str, str]]] = []
        for edge, op in zip(spec.lhs, spec.lhs_ops, strict=True):
            walkers = [v for v in edge if is_walker(v)]
            if not walkers:
                others.append((edge, op))
                continue
            if len(walkers) != 1 or edge[0] != walkers[0] or len(edge) not in (2, 3):
                raise ValueError(
                    f"a ball is written {{w,s}} or {{w,s,v}}, not {{{','.join(edge)}}}"
                )
            if spec.type_of(edge[1]) != "site":
                raise ValueError(f"{edge[1]} must be a site: a ball is {{w,s,v}}")
            velocity = edge[2] if len(edge) == 3 else None
            if velocity is not None and not is_velocity(velocity):
                raise ValueError(
                    f"{velocity} must be a velocity (a name starting with v): a ball is {{w,s,v}}"
                )
            if op[0] not in ("", "_"):
                raise ValueError("a ball may only carry a hop label: {w,s,v}_k")
            label = None
            if op[0] == "_":
                ref = DirRef.parse(op[1])
                if ref.fixed or ref.negate or ref.perp or ref.origin:
                    raise ValueError(
                        "the hop label of a ball must be a plain variable: _k"
                    )
                label = ref.name
            if any(b.walker == edge[0] for b in balls):
                raise ValueError(f"the ball {edge[0]} appears twice on the left side")
            balls.append(BallEdge(edge[0], edge[1], velocity, label))
        if not balls:
            raise ValueError(
                "the left side must contain a ball {w,s} or {w,s,v} (one walker per pattern "
                "for motion, two balls with a contact for a collision)"
            )
        if len(balls) > 2:
            raise ValueError("at most two balls per rule")
        if len(balls) == 2 and any(b.velocity is None for b in balls):
            raise ValueError(
                "a rule with two balls names both velocities: {w1,s,v1}, {w2,t,v2}"
            )
        if len(balls) == 2 and balls[0].site == balls[1].site:
            raise ValueError("the two balls of a rule need distinct site variables")

        sites: set[str] = {b.site for b in balls}
        velocities: set[str] = {"o"} | {b.velocity for b in balls if b.velocity}
        directions: set[str] = {b.label for b in balls if b.label}
        hop_targets: dict[str, str] = {}
        steps: list[Step] = []
        unknowns: list[str] = []

        def direction(ref: DirRef, where: str) -> DirRef:
            if ref.fixed is None and not ref.origin and ref.name not in directions:
                raise ValueError(
                    f"direction {ref.name} in {where} is not bound: bind it by a ball "
                    "({w,s,v}_k), a room edge ({s,t}_k) or a line of centres ({s,t}=d) first"
                )
            return ref

        for edge, (op, arg) in others:
            if len(edge) != 2:
                raise ValueError(
                    f"site and velocity edges have two ends, got {{{','.join(edge)}}}"
                )
            a, b = edge
            text = _edge_text(edge, (op, arg))
            if is_velocity(a) or is_velocity(b):
                if not (is_velocity(a) and is_velocity(b)):
                    raise ValueError(f"{text} mixes a velocity with a site")
                if op not in ("_", "=", ">", "|", "^"):
                    raise ValueError(
                        f"{text}: a velocity relation needs an annotation (_k, =d, >d, |k, |o, ^)"
                    )
                ref = DirRef.parse(arg) if op != "^" else None
                if ref is not None and op != "|" and ref.origin:
                    raise ValueError(f"{text}: o is only the point mirror |o")
                if ref is not None and op == "_" and ref.perp:
                    raise ValueError(
                        f"{text}: a velocity lattice edge needs an axis label"
                    )
                if ref is not None and not ref.origin:
                    direction(ref, text)
                for v in (a, b):
                    if v not in velocities and v not in unknowns:
                        unknowns.append(v)
                steps.append(Step("relation", a, b, op, ref))
                continue
            if spec.type_of(a) != "site" or spec.type_of(b) != "site":
                raise ValueError(f"{text}: expected two sites")
            if op == "":
                if a not in sites:
                    raise ValueError(
                        f"{text}: {a} is not bound yet; chain edges start at a bound site"
                    )
                if b in sites:
                    raise ValueError(
                        f"{text}: {b} is already bound; a chain edge reaches a new site"
                    )
                sites.add(b)
                steps.append(Step("chain", a, b))
                continue
            if op == "_":
                ref = DirRef.parse(arg)
                if ref.fixed is None and (ref.negate or ref.perp or ref.origin):
                    raise ValueError(
                        f"{text}: a room edge label is a plain variable or +x, -x, +y, -y"
                    )
                if a not in sites:
                    raise ValueError(f"{text}: {a} is not bound yet")
                if a == b:
                    direction(ref, text)
                    steps.append(Step("wall", a, b, op, ref))
                    continue
                owner = next((ball for ball in balls if ball.site == a), None)
                bound_label = ref.fixed is not None or ref.name in directions
                if b not in sites:
                    if not bound_label:
                        directions.add(ref.name)  # drawn with the weights, then bound
                    sites.add(b)
                    if owner is not None and owner.label == ref.name:
                        hop_targets[owner.walker] = b
                    steps.append(Step("hop", a, b, op, ref))
                    continue
                # Both ends bound: the hops of two balls meet at b.
                if owner is None or owner.label != ref.name or len(balls) != 2:
                    raise ValueError(
                        f"{text}: {b} is already bound; two bound sites joined by a labelled "
                        "edge are only the contact of two balls, {s,x}_k1, {t,x}_k2"
                    )
                other = next(ball for ball in balls if ball is not owner)
                if hop_targets.get(other.walker) != b:
                    raise ValueError(
                        f"{text}: a contact joins the hop targets of both balls at one site x: "
                        "{s,x}_k1, {t,x}_k2"
                    )
                hop_targets[owner.walker] = b
                steps.append(Step("contact", a, b, op, ref))
                continue
            if op in ("=", ">"):
                ref = DirRef.parse(arg)
                if a not in sites or b not in sites:
                    raise ValueError(f"{text}: both sites must be bound")
                if (
                    ref.fixed is None
                    and not ref.negate
                    and not ref.perp
                    and ref.name not in directions
                ):
                    if op == ">":
                        raise ValueError(f"{text}: {ref.name} is not bound")
                    directions.add(ref.name)
                else:
                    direction(ref, text)
                steps.append(Step("line" if op == "=" else "ahead", a, b, op, ref))
                continue
            raise ValueError(f"{text}: the annotation {op} is not for site edges")

        if len(balls) == 2 and not any(step.kind == "contact" for step in steps):
            raise ValueError(
                "a rule with two balls needs a contact: {s,x}_k1, {t,x}_k2 (the two balls' "
                "next hops meet at x)"
            )
        if len(unknowns) > 2:
            raise ValueError("at most two velocities can be solved per rule")

        results: list[tuple[str, str, str | None]] = []
        lhs_texts = {
            _edge_text(e, o)
            for e, o in zip(spec.lhs, spec.lhs_ops, strict=True)
            if not any(is_walker(v) for v in e)
        }
        seen_walkers: set[str] = set()
        for edge, op in zip(spec.rhs, spec.rhs_ops, strict=True):
            walkers = [v for v in edge if is_walker(v)]
            if not walkers:
                if _edge_text(edge, op) not in lhs_texts:
                    raise ValueError(
                        "the space is static: the right side may only repeat the left side's "
                        f"site and velocity edges, not {_edge_text(edge, op)}"
                    )
                continue
            if (
                len(walkers) != 1
                or edge[0] != walkers[0]
                or len(edge) not in (2, 3)
                or op[0]
            ):
                raise ValueError(
                    f"on the right side a ball is written {{w,t}} or {{w,t,v'}}, not "
                    f"{_edge_text(edge, op)}"
                )
            w = edge[0]
            if not any(b.walker == w for b in balls):
                raise ValueError(
                    f"{w} on the right side is not a ball of the left side"
                )
            if w in seen_walkers:
                raise ValueError(f"the ball {w} appears twice on the right side")
            seen_walkers.add(w)
            if edge[1] not in sites:
                raise ValueError(
                    f"the ball {w} must end on a site bound on the left side, not on {edge[1]!r}"
                )
            velocity = edge[2] if len(edge) == 3 else None
            if (
                velocity is not None
                and velocity not in velocities
                and velocity not in unknowns
            ):
                raise ValueError(
                    f"the velocity {velocity} of {w} must be bound or solved on the left side"
                )
            results.append((w, edge[1], velocity))
        missing = [b.walker for b in balls if b.walker not in seen_walkers]
        if missing:
            raise ValueError(
                f"the right side must say where {', '.join(missing)} ends: {{w,t}} or {{w,t,v'}}"
            )
        return cls(
            tuple(balls), tuple(steps), tuple(unknowns), tuple(results), hop_targets
        )

    def describe(self, spec: RuleSpec) -> dict[str, str]:
        """Generic description of a ball rule (the library's rules carry their own texts)."""
        conditions = []
        for step in self.steps:
            if step.kind == "wall":
                conditions.append(
                    f"{step.a} faces a wall along {step.direction.name if step.direction else '?'}"
                )
            elif step.kind == "contact":
                conditions.append("the next hops of the two balls meet")
            elif step.kind == "line":
                conditions.append(
                    f"{step.direction.name if step.direction else 'd'} is the lattice direction from {step.a} to {step.b}"
                )
            elif step.kind == "ahead":
                conditions.append(
                    f"{step.b} is ahead of {step.a} along {step.direction.name if step.direction else 'd'}"
                )
            elif step.kind == "relation":
                conditions.append(
                    f"{step.a} and {step.b} are related by {step.op}{step.direction.name if step.direction else ''}"
                )
        balls = ", ".join(
            f"{b.walker} on {b.site}"
            + (f" with velocity {b.velocity}" if b.velocity else "")
            + (f" about to hop along {b.label}" if b.label else "")
            for b in self.balls
        )
        when = (
            "once per round, on balls with hops left"
            if self.is_round
            else "once per step"
        )
        solved = (
            f"the velocit{'y' if len(self.unknowns) == 1 else 'ies'} {', '.join(self.unknowns)} "
            "solved on the velocity lattice (drawn at random when several nodes fit)"
            if self.unknowns
            else "deterministic"
        )
        return {
            "name": spec.name,
            "description": spec.description,
            "rewrite": spec.notation,
            "matches": f"{balls}; {'; '.join(conditions) if conditions else 'no further condition'}; applied {when}",
            "choice": solved,
            "invariants": "number of balls, one hyperedge per ball, the room and the velocity lattice",
            "physics": "a ball rule: no drift or diffusion prediction; momentum and energy are measured",
        }


# ---------- executing ball rules ----------


@dataclass
class RoundContext:
    """What the matcher needs about the balls of the current round: which may match
    (``active``), the label of every ball's next hop (``labels``, -1 for none), the site its
    body reaches by that hop (``targets``; the site itself when it runs into a wall), whether
    that hop is into a wall (``wall``), whether the ball is *free* to take the hop (``free``:
    not jammed behind a ball that cannot move, not in an unresolved contact) and the contact
    pairs of the round (``pairs``, see :func:`contact_pairs`)."""

    active: np.ndarray
    labels: np.ndarray
    targets: np.ndarray
    wall: np.ndarray
    max_speed: int
    overlaps: str = "parallel"
    ages: np.ndarray | None = None
    free: np.ndarray | None = None
    pairs: np.ndarray | None = None


@dataclass(frozen=True)
class Outcome:
    """The result of one application of a ball rule."""

    anchors: np.ndarray
    velocities: np.ndarray
    matched: np.ndarray  # balls rewritten by this application
    consumed: np.ndarray  # balls that moved to their hop target
    events: list[dict[str, Any]]


def _direction_value(
    ref: DirRef, bound: dict[str, np.ndarray], rows: int
) -> np.ndarray:
    """The direction vectors ``(rows, 2)`` of a direction reference."""
    if ref.fixed is not None:
        d = np.broadcast_to(np.asarray(ref.fixed, dtype=np.int64), (rows, 2))
    else:
        d = bound[ref.name]
    if ref.negate:
        d = -d
    if ref.perp:
        d = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return np.asarray(d, dtype=np.int64)


def _label_index(vectors: np.ndarray) -> np.ndarray:
    """Direction labels (0..3) of axis vectors; -1 for anything else."""
    scores = vectors @ DIRECTIONS.T
    labels = np.argmax(scores, axis=1)
    axis = np.sum(np.abs(vectors), axis=1) == 1
    return np.where(axis, labels, -1)


def _pairs_sharing(
    sites_a: np.ndarray, owners_a: np.ndarray, sites_b: np.ndarray, owners_b: np.ndarray
) -> np.ndarray:
    """All ``(owner_a, owner_b)`` with a common site, ``owner_a != owner_b``, shape ``(m, 2)``."""
    if sites_a.size == 0 or sites_b.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.argsort(sites_b, kind="stable")
    sb, ob = sites_b[order], owners_b[order]
    lo = np.searchsorted(sb, sites_a, "left")
    hi = np.searchsorted(sb, sites_a, "right")
    pairs = []
    for i in np.flatnonzero(hi > lo):
        for j in range(lo[i], hi[i]):
            if ob[j] != owners_a[i]:
                pairs.append((int(owners_a[i]), int(ob[j])))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.asarray(pairs, dtype=np.int64), axis=0)


def meeting_pairs(
    state: WalkerGraph, targets: np.ndarray, immobile: np.ndarray
) -> np.ndarray:
    """Pairs ``(a, b)``, ``a < b``, whose next hops meet: their bodies after the hop share a
    site, one lands on the body of a ball that stays in this round (``immobile``: no hops
    left, blocked by a wall, or held back by a meeting), or they cross (each lands on the
    other's current site). ``targets`` is where every ball would go if it moved. Balls
    whose bodies already share a site are left alone until they have separated. Shape
    ``(m, 2)``, sorted."""
    n = state.num_walkers
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    current = state.sites()
    proposed = WalkerGraph(state.space, targets, None, state.radius).sites()
    effective = np.where(immobile[:, None], current, proposed)
    owners = np.repeat(np.arange(n), current.shape[1])
    cur, prop, eff = current.ravel(), proposed.ravel(), effective.ravel()
    shared = _pairs_sharing(prop, owners, prop, owners)
    hits = _pairs_sharing(prop, owners, eff, owners)  # a lands where b ends up
    overlapping = _pairs_sharing(cur, owners, cur, owners)
    lands = _pairs_sharing(prop, owners, cur, owners)  # a lands where b is now
    crossing = np.zeros((0, 2), dtype=np.int64)
    if lands.size:
        keys = {(int(a), int(b)) for a, b in lands}
        crossing = np.asarray(
            [(a, b) for a, b in keys if (b, a) in keys], dtype=np.int64
        ).reshape(-1, 2)
    pairs = np.concatenate([shared, hits, crossing], axis=0)
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    if overlapping.size:
        together = {(int(a), int(b)) for a, b in np.sort(overlapping, axis=1)}
        pairs = np.asarray(
            [p for p in pairs if (int(p[0]), int(p[1])) not in together], dtype=np.int64
        ).reshape(-1, 2)
    return pairs


def approaching(state: WalkerGraph, pairs: np.ndarray) -> np.ndarray:
    """The mask of pairs whose balls approach each other along the lattice direction of
    their line of centres: ``(v_a - v_b) . sign(c_b - c_a) > 0``."""
    if pairs.size == 0:
        return np.zeros(0, dtype=bool)
    cells = lattice_cells(state.space, state.anchors)
    d = direction_of(cells[pairs[:, 1]] - cells[pairs[:, 0]])
    v = state.velocity
    return np.sum((v[pairs[:, 0]] - v[pairs[:, 1]]) * d, axis=1) > 0


def contact_pairs(state: WalkerGraph, context: RoundContext) -> np.ndarray:
    """The contacts of the round, ``(m, 2)``: the meeting pairs (:func:`meeting_pairs`) that
    approach each other along their line of centres (a pair that has already collided and
    separates, or whose remaining hops only graze, is not a contact; the balls then wait
    instead of passing through each other)."""
    if context.pairs is not None:
        return context.pairs
    immobile = ~(context.active & ~context.wall)
    pairs = meeting_pairs(state, context.targets, immobile)
    return pairs[approaching(state, pairs)]


def body_step(state: WalkerGraph, sites: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The site reached from ``sites`` along ``labels`` by a ball of the state's radius: the
    centre's neighbour, or the site itself when the body would run into a wall."""
    neighbors = state.space.neighbors
    target = neighbors[sites, labels]
    if state.radius == 0:
        return target
    bodies = WalkerGraph(state.space, sites, None, state.radius).sites()
    hits = np.any(neighbors[bodies, labels[:, None]] == bodies, axis=1)
    return np.where(hits, sites, target)


def select_pairs(
    pairs: np.ndarray, strategy: str, rng: np.random.Generator, ages: np.ndarray | None
) -> np.ndarray:
    """A set of disjoint pairs, greedy in the strategy's order (``parallel`` and ``ordered``
    by index, ``random`` in a drawn order, ``oldest`` longest-waiting ball first)."""
    m = pairs.shape[0]
    if m <= 1:
        return np.ones(m, dtype=bool)
    if strategy == "random":
        order = rng.permutation(m)
    elif strategy == "oldest" and ages is not None:
        order = np.argsort(
            -np.maximum(ages[pairs[:, 0]], ages[pairs[:, 1]]), kind="stable"
        )
    else:
        order = np.arange(m)
    taken: set[int] = set()
    chosen = np.zeros(m, dtype=bool)
    for i in order:
        a, b = int(pairs[i, 0]), int(pairs[i, 1])
        if a in taken or b in taken:
            continue
        taken.update((a, b))
        chosen[i] = True
    return chosen


@dataclass(frozen=True)
class BallRule:
    """The rule described by a ball-family :class:`RuleSpec`, executed by the generic matcher
    (see the module docstring)."""

    spec: RuleSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def plan(self) -> BallPlan:
        return self.spec.plan()

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return (self.spec,)

    @property
    def hops(self) -> int:
        return 0

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        return np.zeros(2)

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        return np.zeros(2)

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        text = self.spec.describe(spacing, step_time)
        library = LIBRARY_TEXTS.get(self.spec.notation)
        if library is not None:
            text.update(library)
        return text

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        """One application on every ball, as a step rule (every ball active, its next hop the
        first hop of its velocity); the systems in :mod:`imigraph.graph.world` run rounds."""
        from imigraph.graph.world import round_context

        context = round_context(state, np.ones(state.num_walkers, dtype=bool), None)
        outcome = self.match(state, rng, context)
        return WalkerGraph(
            state.space, outcome.anchors, outcome.velocities, state.radius
        )

    # ---------- the matcher ----------

    def match(
        self, state: WalkerGraph, rng: np.random.Generator, context: RoundContext
    ) -> Outcome:
        plan = self.plan
        space = state.space
        neighbors = space.neighbors
        anchors, velocities = state.anchors, state.velocity
        n = state.num_walkers
        no_change = Outcome(
            anchors, velocities, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), []
        )
        # Rows: candidate matches. One ball rule: every active ball; two balls: contact pairs.
        if plan.is_pair:
            pairs = contact_pairs(state, context)
            if pairs.size == 0:
                return no_change
            index = [pairs[:, 0], pairs[:, 1]]
        else:
            if plan.is_round:
                rows = np.flatnonzero(context.active)
            else:
                rows = np.arange(n)
            if rows.size == 0:
                return no_change
            index = [rows]
        bound: dict[str, np.ndarray] = {}
        rows_count = index[0].shape[0]
        keep = np.ones(rows_count, dtype=bool)
        bound["o"] = np.zeros((rows_count, 2), dtype=np.int64)
        for ball, idx in zip(plan.balls, index, strict=True):
            bound[ball.site] = anchors[idx]
            if ball.velocity:
                bound[ball.velocity] = velocities[idx]
            if ball.label:
                labels = context.labels[idx]
                if not plan.is_pair:
                    keep &= labels >= 0
                bound[ball.label] = np.where(
                    (labels >= 0)[:, None], DIRECTIONS[np.maximum(labels, 0)], 0
                )
        ball_of_site = {ball.site: ball for ball in plan.balls}
        ball_index = {
            ball.walker: idx for ball, idx in zip(plan.balls, index, strict=True)
        }
        relations: list[Step] = []
        for step in plan.steps:
            if step.kind == "relation":
                relations.append(step)
                continue
            ref = step.direction
            if step.kind == "chain" or (
                step.kind == "hop"
                and ref is not None
                and ref.fixed is None
                and ref.name not in bound
            ):
                labels = _choose_directions(
                    neighbors,
                    bound[step.a],
                    self.spec.normalized_weights,
                    self.spec.self_loops == "allow",
                    rng,
                )
                bound[step.b] = body_step(state, bound[step.a], labels)
                if ref is not None:
                    bound[ref.name] = DIRECTIONS[labels]
                continue
            assert ref is not None
            binds = (
                step.kind == "line"
                and ref.fixed is None
                and not ref.negate
                and not ref.perp
                and ref.name not in bound
            )
            d = (
                None
                if binds or ref.origin
                else _direction_value(ref, bound, rows_count)
            )
            if step.kind == "hop":
                assert d is not None
                owner = ball_of_site.get(step.a)
                if owner is not None and owner.label == ref.name:
                    idx = ball_index[owner.walker]
                    free = (
                        np.ones(rows_count, dtype=bool)
                        if context.free is None
                        else context.free[idx]
                    )
                    bound[step.b] = np.where(free, context.targets[idx], anchors[idx])
                else:
                    labels = _label_index(d)
                    keep &= labels >= 0
                    bound[step.b] = body_step(
                        state, bound[step.a], np.maximum(labels, 0)
                    )
                continue
            if step.kind == "wall":
                assert d is not None
                owner = ball_of_site.get(step.a)
                if owner is not None and owner.label == ref.name:
                    keep &= context.wall[ball_index[owner.walker]]
                else:
                    labels = _label_index(d)
                    keep &= (labels >= 0) & (
                        neighbors[bound[step.a], np.maximum(labels, 0)] == bound[step.a]
                    )
                continue
            if step.kind == "contact":
                continue  # the pairs were found by their meeting hops
            cells_a = lattice_cells(space, bound[step.a])
            cells_b = lattice_cells(space, bound[step.b])
            delta = cells_b - cells_a
            if step.kind == "line":
                if binds:
                    direction = direction_of(delta)
                    if plan.is_pair:
                        v1 = bound[plan.balls[0].velocity or "o"]
                        v2 = bound[plan.balls[1].velocity or "o"]
                        fallback = direction_of(v1 - v2)
                        direction = np.where(
                            np.any(direction != 0, axis=1, keepdims=True),
                            direction,
                            fallback,
                        )
                    direction = np.where(
                        np.any(direction != 0, axis=1, keepdims=True),
                        direction,
                        np.array([[1, 0]], dtype=np.int64),
                    )
                    bound[ref.name] = direction
                else:
                    assert d is not None
                    keep &= (delta[:, 0] * d[:, 1] - delta[:, 1] * d[:, 0]) == 0
                continue
            if step.kind == "ahead":
                assert d is not None
                keep &= np.sum(delta * d, axis=1) > 0
                continue
        # Velocity relations: enumerate the unknowns on the velocity lattice.
        box = velocity_box(context.max_speed)
        c = box.shape[0]
        m = len(plan.unknowns)
        if m == 0:
            grid = np.zeros((1, 0, 2), dtype=np.int64)
        elif m == 1:
            grid = box[:, None, :]
        else:
            i, j = np.meshgrid(np.arange(c), np.arange(c), indexing="ij")
            grid = np.stack([box[i.ravel()], box[j.ravel()]], axis=1)
        g = grid.shape[0]
        mask = np.ones((rows_count, g), dtype=bool)

        def value(name: str) -> np.ndarray:
            if name in plan.unknowns:
                return np.broadcast_to(
                    grid[None, :, plan.unknowns.index(name), :], (rows_count, g, 2)
                )
            return np.broadcast_to(bound[name][:, None, :], (rows_count, g, 2))

        for step in relations:
            a, b = value(step.a), value(step.b)
            ref = step.direction
            if step.op == "^":
                mask &= np.all(
                    b == np.stack([-a[..., 1], a[..., 0]], axis=-1), axis=-1
                ) | np.all(b == np.stack([a[..., 1], -a[..., 0]], axis=-1), axis=-1)
                continue
            assert ref is not None
            if step.op == "|" and ref.origin:
                mask &= np.all(b == -a, axis=-1)
                continue
            d = _direction_value(ref, bound, rows_count)[:, None, :]
            if step.op == "_":
                mask &= np.all(b == a + d, axis=-1)
            elif step.op == "=":
                delta = b - a
                mask &= (delta[..., 0] * d[..., 1] - delta[..., 1] * d[..., 0]) == 0
            elif step.op == ">":
                mask &= np.sum((b - a) * d, axis=-1) > 0
            elif step.op == "|":
                axis = np.abs(d[..., 0]) > 0  # True: the x axis of the label
                mirrored = np.where(
                    axis[..., None],
                    np.stack([-a[..., 0], a[..., 1]], axis=-1),
                    np.stack([a[..., 0], -a[..., 1]], axis=-1),
                )
                mask &= np.all(b == mirrored, axis=-1)
        mask &= keep[:, None]
        counts = mask.sum(axis=1)
        keep &= counts > 0
        if m:
            u = rng.random(rows_count)
            wanted = np.floor(u * np.maximum(counts, 1)).astype(np.int64) + 1
            cumulative = np.cumsum(mask, axis=1)
            pick = np.argmax(cumulative >= wanted[:, None], axis=1)
            for j, name in enumerate(plan.unknowns):
                bound[name] = grid[pick, j, :]
        selected = keep
        if plan.is_pair:
            matched_pairs = pairs[keep]
            chosen = select_pairs(matched_pairs, context.overlaps, rng, context.ages)
            selected = keep.copy()
            selected[np.flatnonzero(keep)] = chosen
        if not selected.any():
            return no_change
        new_anchors, new_velocities = anchors.copy(), velocities.copy()
        matched = np.zeros(n, dtype=bool)
        consumed = np.zeros(n, dtype=bool)
        rows_selected = np.flatnonzero(selected)
        changed_velocity = np.zeros(n, dtype=bool)
        for walker, site, velocity in plan.results:
            idx = ball_index[walker][rows_selected]
            new_anchors[idx] = bound[site][rows_selected]
            if velocity is not None:
                new_velocities[idx] = bound[velocity][rows_selected]
                changed_velocity[idx] = np.any(
                    new_velocities[idx] != velocities[idx], axis=1
                )
            matched[idx] = True
            if plan.hop_targets.get(walker) == site:
                consumed[idx] = True
        events: list[dict[str, Any]] = []
        balls_of_rows = np.stack([ball_index[b.walker] for b in plan.balls], axis=1)
        lines = np.zeros((rows_count, 2), dtype=np.int64)
        if plan.is_pair:  # the line of centres, for the caption of the event
            centres = [
                lattice_cells(space, anchors[balls_of_rows[:, i]]) for i in (0, 1)
            ]
            lines = direction_of(centres[1] - centres[0])
        for r in rows_selected:
            balls = [int(b) for b in balls_of_rows[r]]
            if not plan.is_pair and not changed_velocity[balls[0]]:
                continue
            event: dict[str, Any] = {
                "kind": "collision" if plan.is_pair else "bounce",
                "rule": self.spec.name,
                "balls": balls,
                "sites": [int(anchors[b]) for b in balls],
                "before": [velocities[b].tolist() for b in balls],
                "after": [new_velocities[b].tolist() for b in balls],
            }
            if plan.is_pair:
                d = lines[r]
                event["axis"] = (
                    "diagonal" if d[0] and d[1] else AXIS_NAMES[int(d[1] != 0)]
                )
            elif plan.balls[0].label in bound:
                d = bound[plan.balls[0].label or ""][r]
                event["axis"] = AXIS_NAMES[int(d[1] != 0)]
            events.append(event)
        return Outcome(new_anchors, new_velocities, matched, consumed, events)


# ---------- the chain rule family (vectorised) ----------

RANDOM_HOP = RuleSpec.parse("{w,s}, {s,s'} -> {w,s'}, {s,s'}", name="RandomHop")


@dataclass(frozen=True)
class ChainRule:
    """The rule described by a chain-family :class:`RuleSpec` (see the module docstring)."""

    spec: RuleSpec

    def __post_init__(self) -> None:
        if self.spec.family != "chain":
            raise ValueError(
                f"{self.spec.name} is a ball rule (velocities or annotations): it runs in a "
                "walker system, not as a chain rule"
            )

    @property
    def name(self) -> str:
        return self.spec.name

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        return self.spec.describe(spacing, step_time)

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return (self.spec,)

    @property
    def hops(self) -> int:
        return self.spec.hops

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        return self.spec.drift(spacing, step_time)

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        return self.spec.diffusion(spacing, step_time)

    def apply_with_choices(
        self, state: WalkerGraph, rng: np.random.Generator
    ) -> tuple[WalkerGraph, np.ndarray, np.ndarray]:
        """The new state, the matched chain of every walker (shape ``(n, k + 1)``) and the
        direction labels chosen for its chain edges (shape ``(n, k)``)."""
        _, chain, target = self.spec.chain()
        neighbors = state.space.neighbors
        weights = self.spec.normalized_weights
        allow = self.spec.self_loops == "allow"
        path = [state.anchors]
        choices = []
        for _ in range(len(chain) - 1):
            choice = _choose_directions(neighbors, path[-1], weights, allow, rng)
            choices.append(choice)
            path.append(neighbors[path[-1], choice])
        labels = (
            np.stack(choices, axis=1)
            if choices
            else np.zeros((state.num_walkers, 0), dtype=np.int64)
        )
        fits = body_fits(state.space, path[target], state.radius)
        anchors = np.where(fits, path[target], state.anchors)
        new_state = WalkerGraph(state.space, anchors, state.velocities, state.radius)
        return new_state, np.stack(path, axis=1), labels

    def apply_with_paths(
        self, state: WalkerGraph, rng: np.random.Generator
    ) -> tuple[WalkerGraph, np.ndarray]:
        """The new state and the matched chain of every walker, shape ``(n, k + 1)``."""
        new_state, path, _ = self.apply_with_choices(state, rng)
        return new_state, path

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        return self.apply_with_paths(state, rng)[0]


@dataclass(frozen=True)
class RuleSequence:
    """Several chain rules applied one after the other in every step, in the given order.

    Every rule sees the state the previous one produced, so one step moves a walker along the
    concatenated chains. Within a step the directions of the parts are drawn independently,
    so far from the walls the drifts add and the per-component diffusions add:
    ``v = sum v_i``, ``D = sum D_i`` (a sum of independent increments). ``hops`` is the total.
    """

    rules: tuple[ChainRule, ...]

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("a rule sequence needs at least one rule")

    @property
    def name(self) -> str:
        return " + ".join(rule.name for rule in self.rules)

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return tuple(rule.spec for rule in self.rules)

    @property
    def hops(self) -> int:
        return sum(spec.hops for spec in self.specs)

    def drift(self, spacing: float, step_time: float) -> np.ndarray:
        return np.sum([s.drift(spacing, step_time) for s in self.specs], axis=0)

    def diffusion(self, spacing: float, step_time: float) -> np.ndarray:
        return np.sum([s.diffusion(spacing, step_time) for s in self.specs], axis=0)

    def describe(self, spacing: float = 1.0, step_time: float = 1.0) -> dict[str, str]:
        """The sequence as one rule: the rewrites joined by ``;``, the summed physics."""
        drift = self.drift(spacing, step_time)
        diffusion = self.diffusion(spacing, step_time)
        names = ", ".join(rule.name for rule in self.rules)
        return {
            "name": self.name,
            "description": "",
            "rewrite": " ; ".join(spec.notation for spec in self.specs),
            "matches": f"{len(self.rules)} rules applied in order in every step ({names}); "
            "each one matches on the state the previous one produced",
            "choice": "each rule draws its directions independently; all walkers rewritten "
            "at once by every rule",
            "invariants": "number of walkers, one anchoring edge per walker, the space graph",
            "physics": f"{self.hops} hops per step in total; drifts and diffusions add up: "
            f"D = ({diffusion[0]:.4g}, {diffusion[1]:.4g}), drift v = ({drift[0]:.4g}, "
            f"{drift[1]:.4g}) for a = {spacing:g}, tau = {step_time:g}",
        }

    def descriptions(
        self, spacing: float = 1.0, step_time: float = 1.0
    ) -> list[dict[str, str]]:
        """One description per rule, then the sequence as a whole."""
        parts = [rule.describe(spacing, step_time) for rule in self.rules]
        return [*parts, self.describe(spacing, step_time)]

    def apply_with_paths(
        self, state: WalkerGraph, rng: np.random.Generator
    ) -> tuple[WalkerGraph, np.ndarray]:
        """The new state and the concatenated matched chains, shape ``(n, 1 + sum k_i)``
        (the joint site is not repeated when a rule ends on the last site of its chain)."""
        pieces: list[np.ndarray] = []
        for rule in self.rules:
            state, path = rule.apply_with_paths(state, rng)
            if pieces and np.array_equal(pieces[-1][:, -1], path[:, 0]):
                path = path[:, 1:]
            pieces.append(path)
        return state, np.concatenate(pieces, axis=1)

    def apply(self, state: WalkerGraph, rng: np.random.Generator) -> WalkerGraph:
        return self.apply_with_paths(state, rng)[0]


# ---------- the library ----------

STREAM = RuleSpec.parse(
    "{w,s,v}_k, {s,t}_k -> {w,t,v}",
    name="Stream",
    description="free flight: the ball takes the next hop of its velocity",
)
REFLECT = RuleSpec.parse(
    "{w,s,v}_k, {s,s}_k, {v,v'}|k -> {w,s,v'}",
    name="Reflect",
    description="a ball about to hop into a wall mirrors that velocity component",
)
EXCHANGE = RuleSpec.parse(
    "{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2, {s,t}=d, "
    "{v1,v1'}=d, {v2,v2'}=d, {v1,v2'}=d^, {v2,v1'}=d^, {v1',v1}>d -> {w1,s,v1'}, {w2,t,v2'}",
    name="Exchange",
    description="two balls whose hops meet exchange the velocity component along the line "
    "of centres d (the elastic collision of equal masses); no match when the exchanged "
    "velocities are not lattice nodes (an odd diagonal hit)",
)
SWAP = RuleSpec.parse(
    "{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2 -> {w1,s,v2}, {w2,t,v1}",
    name="Swap",
    description="two balls whose hops meet swap their whole velocities: each continues the "
    "other's motion from its own site (the head-on outcome for every geometry); the "
    "fallback after Exchange",
)
TURN = RuleSpec.parse(
    "{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2, {v1,v2}|o, {v1,v1'}^, {v1',v2'}|o "
    "-> {w1,s,v1'}, {w2,t,v2'}",
    name="Turn",
    description="a head-on pair whose hops meet turns by a quarter turn, either sense at "
    "random: the HPP collision",
)
KICK = RuleSpec.parse(
    "{w,s,v}, {s,t}_k, {o,v'}_k -> {w,t,v'}",
    name="Kick",
    description="one hop in a drawn direction, and the ball then heads that way with unit "
    "speed",
)
LIBRARY = (STREAM, REFLECT, EXCHANGE, SWAP, TURN, KICK)
LIBRARY_TEXTS: dict[str, dict[str, str]] = {
    STREAM.notation: {
        "matches": "a ball with a hop left, and the room edge of its next hop label k "
        "(a wall self-loop means it stays); once per round",
        "choice": "deterministic",
        "invariants": "balls, velocities, momentum, energy",
        "physics": "free flight: one hop per round, |vx| + |vy| hops per step, so the "
        "velocity v is in cells per step and the speed |v| a / tau",
    },
    REFLECT.notation: {
        "matches": "a ball whose next hop along k runs into a wall self-loop {s,s}_k, with "
        "v' the mirror image of v across the axis of k; once per round",
        "choice": "deterministic (the mirror image is unique)",
        "invariants": "balls, energy (|v'| = |v|); the wall takes the momentum 2 m v_k",
        "physics": "specular reflection; never matches on a periodic lattice",
    },
    EXCHANGE.notation: {
        "matches": "two balls whose next hops meet at x, approaching along the line of "
        "centres d ({v1',v1}>d); v1' and v2' are the other two corners of the rectangle "
        "spanned by v1 and v2 in the velocity lattice (sides along d and d^): the "
        "components along d exchanged, the perpendicular ones kept",
        "choice": "deterministic; no match when a corner is not a lattice node (an odd "
        "diagonal hit), which lets a later rule (Swap) take the pair",
        "invariants": "balls, momentum (v1' + v2' = v1 + v2), energy (|v1'|^2 + |v2'|^2 = "
        "|v1|^2 + |v2|^2)",
        "physics": "the elastic collision of equal masses, exact for every lattice velocity "
        "along an axis and for even diagonal hits",
    },
    SWAP.notation: {
        "matches": "two balls whose next hops meet at x (they would land on one site or "
        "cross one edge)",
        "choice": "deterministic",
        "invariants": "balls, momentum, energy (the pair keeps both velocities)",
        "physics": "every hit treated as head-on: exact for head-on hits, a deflection "
        "of both balls otherwise (at right angles with unit speeds the same as Exchange)",
    },
    TURN.notation: {
        "matches": "two balls whose next hops meet with opposite velocities (v2 = -v1); "
        "v1' is v1 turned by a quarter turn and v2' = -v1'",
        "choice": "the sense of the turn is drawn uniformly (which ball goes which way)",
        "invariants": "balls, momentum (0 before and after), energy (|v'| = |v|)",
        "physics": "the Collide rule of the HPP lattice gas (Hardy, Pomeau, de Pazzis 1973), "
        "the only interaction of that gas; pairs that are not head-on are left to the next "
        "rule (Swap in the hpp set)",
    },
    KICK.notation: {
        "matches": "every ball once per step; a room edge is drawn at its site with the "
        "weights and the ball's new velocity is the unit vector of that label",
        "choice": "the direction is drawn with the rule's weights",
        "invariants": "balls, one hyperedge per ball",
        "physics": "a random walk whose velocity shows the last hop: a thermal kick of "
        "unit speed per step (energy and momentum are not conserved, as for a Brownian "
        "particle)",
    },
}


RuleData = RuleSpec | dict[str, Any] | list[Any] | tuple[Any, ...] | None


def rule_specs(data: RuleData) -> tuple[RuleSpec, ...]:
    """The specs of a rule given as data: ``None`` (the random hop), one spec or its JSON form,
    or a list of them (a sequence). Raises ``ValueError`` with the editor's message."""
    if data is None:
        return (RANDOM_HOP,)
    items = list(data) if isinstance(data, (list, tuple)) else [data]
    if not items:
        raise ValueError("enable at least one rule")
    return tuple(
        item if isinstance(item, RuleSpec) else RuleSpec.from_dict(item)
        for item in items
    )


def rule_from_spec(spec: RuleData) -> Rule:
    """``None`` -> :class:`RandomHop`; a chain spec or its JSON form -> :class:`ChainRule`; a
    list of them -> :class:`RuleSequence` (a single-element list is just its
    :class:`ChainRule`). Ball rules (velocities, annotations) are refused: they need the
    rounds of a :class:`~imigraph.graph.world.WalkerSystem`."""
    if spec is None:
        return RandomHop()
    rules = tuple(ChainRule(s) for s in rule_specs(spec))
    return rules[0] if len(rules) == 1 else RuleSequence(rules)


def rule_item(spec: RuleSpec) -> ChainRule | BallRule:
    """The executable rule of a spec: vectorised for the chain family, the matcher otherwise."""
    return ChainRule(spec) if spec.family == "chain" else BallRule(spec)
