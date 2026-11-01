---
name: graph-rules
description: How graph systems are built in imigraph - spaces (neighbour tables with coordinates), walker graphs, Wolfram-style edge-rewriting rules, initial conditions, projections and their physical parameters. Use when adding or changing a space, a rule, an initial condition or a projection.
---

# Graph systems

Read `docs/architecture.md` ("The first graph system in detail") before changing anything in
`src/imigraph/graph/`.

## Vocabulary

- **Space**: a static graph whose nodes carry 2D coordinates. Stored as a dense neighbour table
  `neighbors[node, k]` (same degree everywhere; missing neighbours are self-loops) plus
  `coordinates (num_nodes, 2)`, a `box` and a `cell_size`. `SquareLattice` is the first one.
- **WalkerGraph**: the evolving graph. Space nodes `0 .. N-1`, walker nodes `N .. N+n-1`, each
  walker attached to the space by exactly one edge `{walker, anchors[walker]}`.
- **Rule**: `apply(state, rng) -> WalkerGraph`. Written as an edge rewrite in the Wolfram style,
  for example `RandomHop`: `{w, s}, {s, s'} -> {w, s'}, {s, s'}`. Rules are applied to all
  walkers synchronously, vectorised with numpy; the only randomness comes from `rng`.
- **GraphSimulation**: space + rule + `step_time` (the physical duration of one rule
  application). `run(initial, steps, rng, record_every)` returns a `GraphHistory`.
- **Projection**: `GraphHistory.project()` maps walkers to their node coordinates;
  `project(cell_rng=rng)` spreads them inside their node's cell (resolution of the imitation).

## Adding a rule

1. Write the rewrite rule in the docstring in Wolfram notation and say what it means physically
   (which quantity it conserves, what diffusion coefficient or velocity it produces).
2. Implement `apply` with numpy indexing over all walkers (`neighbors[state.anchors, choice]`).
   No Python loop over walkers.
3. Derive the physical parameter the experiment needs (like `lattice_hop_diffusion`) and put
   the derivation in the function docstring.
4. Tests (`tests/test_graph.py`): every move follows an existing edge; walkers stay on the
   space; the derived parameter is reproduced by a free run (`MSD = 4 D t`, or the analogue);
   determinism with a seed.
5. Compare against the reference with `compare()`; add the pass/fail cases to
   `tests/test_compare.py` if the rule is meant to imitate something.

## Adding a space

Implement the `Space` protocol. Keep the neighbour table dense: pad with self-loops (which also
encode reflecting boundaries) or give the rule a per-node degree to sample from. Provide
`edges()` (undirected, no self-loops) and `to_networkx()` with `pos` attributes for views. Test
the structure against the matching networkx generator (`nx.grid_2d_graph` for the lattice).

## Rendering

The report draws the space from `GraphHistory.to_dict()` (`coordinates`, `edges`, `anchors`
per frame); walker nodes are drawn next to their anchor with their edge (`viz.js`,
`drawWalkers`). A new space needs nothing more as long as it provides coordinates.

## Rules as data (editable in the GUI)

- `RuleSpec` (`graph/rules.py`): notation `{w,s}, {s,t} -> {w,t}, {s,t}`, direction weights
  `(+x, -x, +y, -y)`, `self_loops` `allow` / `forbid`; JSON via `to_dict` / `from_dict`. The
  supported family is the walker-anchored chain (one walker edge, a chain of site edges, the
  walker re-attached to a chain site, site edges unchanged); `chain()` raises a `ValueError`
  whose message the editor shows. `ChainRule(spec)` runs it; `rule_from_spec(data)` is the
  entry point used by `ExperimentConfig.rule`, the CLI `--rule` and `report.live.playground`;
  `data` may be a list of specs, which gives a `RuleSequence` (the GUI's rule set: the rules
  run in order in every step; drifts and diffusions add, chains are concatenated for the
  diagram). `rule_specs(data)` validates the same input.
- The GUI edits a spec graphically (`web/app.js`, nodes = variables, one edge set per side) and
  as text; both produce the same JSON. Keep validation in `RuleSpec.chain()`: the editor shows
  its message, so a new constraint needs a message that says what to change.
- Prediction before measurement: `spec.drift(a, tau)` and `spec.diffusion(a, tau)`; the
  playground shows measured vs predicted. A new rule family needs the same: derive, then test.
- Rule diagrams need the matched chains: `GraphSimulation.run(record_paths=True)` for rules
  that implement `apply_with_paths`.
- Every `RuleSpec` has a free-text `description` (JSON key `description`, shown in the rule
  tables and the GUI library); keep it when adding fields to the JSON form.

## Walker models and collisions (`graph/world.py`)

`WalkerSystem(space, rules, walker_model, collisions, step_time)` is the playground's system
and unites the two walker models under one rule set:

- `walker_model="edge"`: walkers `{w,s}` (`WalkerGraph`), no interaction, chain rules only
  (equivalent to `RuleSequence`).
- `walker_model="hyperedge"`: walkers `{w,s,s'}` (`LatticeGasState`), exclusion principle.
  A chain rule's last hop sets the heading; the built-in rules `BuiltinRule("stream")` and
  `BuiltinRule("reflect")` (JSON `{"builtin": "stream", "name": ..., "description": ...}`)
  follow the heading. Every rule's proposal goes through `resolve_exclusion`: stayers keep
  their channel, clashes have one winner (random), jams propagate, trains move together.
  `blocked` hops are counted and reported. `momentum(history)` and `explain_frames(history)`
  give the total momentum per frame and a caption per frame (used by the GUI and the pages);
  HPP collisions conserve momentum and energy exactly, walls exchange momentum.
- `collisions="hpp"` runs `LatticeGas.collide` first in every step (hyperedge walkers only);
  Reflect + Stream + hpp reproduces `LatticeGas` step for step (`tests/test_world.py`).
- `rules_from_data(list)` builds the rule tuple from the request's `rule` list;
  `system.descriptions()` lists Collide (if on), every rule and the set, each with a pattern.

A new collision type: add it to `COLLISIONS`, apply it in `WalkerSystem.step` before the
rules, describe it in `COLLISION_TEXT` and `descriptions()`, document it in
`report/assets/collisions.html` (shown in the GUI and on the saved page), and test that the
exclusion still holds every step. A new walker model needs its own state class, an
`initial_state`, and a decision for every existing rule kind.

## Conserving rules: the lattice gas (`graph/gas.py`)

Walkers with a heading are hyperedges `{w, s, s'}` (`LatticeGasState`: `sites`,
`directions`); rules Collide / Reflect / Stream (HPP) conserve momentum and energy and keep one
walker per channel. The same rules are available in the playground through `graph/world.py`. When adding a conserving rule: state the conserved quantity in the
docstring, keep the exclusion invariant (`tests/test_lattice_gas.py` checks it every step),
return a `GasRun` so the gas checks apply, and describe the rules in `RULES` for the page.

## Next steps on the roadmap

Growing spaces (rules that add nodes/edges), projections computed from the graph (spectral
embedding), hypergraphs, FHP (hexagonal lattice) for an isotropic lattice gas, an editor for
multi-walker (interaction) rules. Keep `Space` and `Rule` as the seams; do not special-case the
lattice in the simulation loop.
