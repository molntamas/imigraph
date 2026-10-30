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

## Next steps on the roadmap

Growing spaces (rules that add nodes/edges), projections computed from the graph (spectral
embedding), hypergraphs. Keep `Space` and `Rule` as the seams; do not special-case the lattice
in the simulation loop.
