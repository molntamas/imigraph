---
name: graph-rules
description: How graph systems are built in imigraph - the room (a lattice with a neighbour table and coordinates), the velocity lattice, the one kind of walker (the ball {w,s,v}), the annotated rule notation and its matcher, rounds and contacts, initial conditions, projections and their physical parameters. Use when adding or changing a space, a rule, a relation of the notation, an initial condition or a projection.
---

# Graph systems

Read `docs/architecture.md` (from "The one walker" on) before changing anything in
`src/imigraph/graph/`.

## Vocabulary

- **Space** (the room): a static graph whose nodes carry 2D coordinates. Stored as a dense
  neighbour table `neighbors[node, k]` (same degree everywhere; missing neighbours are
  self-loops, the walls) plus `coordinates (num_nodes, 2)`, a `box` and a `cell_size`.
  `SquareLattice` is the only one so far.
- **Velocity lattice** (`graph/velocity.py`): the integer vectors `(vx, vy)` in cells per step,
  with the same direction labels `+x, -x, +y, -y`; the origin `o` is rest. Never stored as a
  graph: `velocity_box(max_speed)` enumerates its nodes, `mirror`, `turn`, `direction_of`,
  `perpendicular` are its relations, `hop_order` the Bresenham hops of a velocity.
- **Ball**: the one kind of walker, the hyperedge `{w, s, v}` of `WalkerGraph(space, anchors,
  velocities, radius)`. A ball at rest (velocity `o`, or `velocities=None`) is the random
  walker; a unit velocity is the lattice-gas walker; `radius > 0` gives it a body (the lattice
  points within the radius, `state.sites()`), which moves rigidly.
- **Rule**: a rewrite of ball hyperedges in the annotated notation (`RuleSpec`), see the table
  in `docs/architecture.md`. Two families: *chain* rules (`{w,s}, {s,t} -> {w,t}`, vectorised
  by `ChainRule`, with predicted drift and diffusion) and *ball* rules (everything with a
  velocity or an annotation, executed by the generic matcher `BallRule`).
- **WalkerSystem** (`graph/world.py`): a rule set run in rounds; `GraphHistory` the record.
- **Projection**: `GraphHistory.project()` maps balls to their site coordinates;
  `project(cell_rng=rng)` spreads them inside their cell (resolution of the imitation).

## How a step runs (`WalkerSystem.step`)

1. Every ball gets a hop budget `|vx| + |vy|`, its hops in Bresenham order. The *round rules*
   (balls with a hop label `_k`) run in order in every round on the balls with hops left; a
   ball that moves to its hop target consumes a hop; a ball whose velocity changed continues
   with the tail of the new hop order. Rounds end when no ball has hops left.
2. The *step rules* (chain rules, Kick) run once each.

`round_context` decides who moves: a ball that cannot move (no hops, wall) stays; two balls
whose hops meet (one site, a staying ball, a crossing) are held, and a ball landing on a held
ball is held too, until stable. Held pairs that approach each other are the *contacts*; the
collision rules match those pairs and are re-applied while they find new contacts. A ball still
held when Stream runs waits. Consequence: no two balls ever share a site, whatever the rules.

## Adding a rule

1. Write it in the notation; put the library constant in `graph/rules.py` with its texts in
   `LIBRARY_TEXTS` and the preset in `web/app.js` (same notation). Say in the description what
   it does physically and what it conserves.
2. The matcher runs it; nothing else to implement. A rule that needs a relation the notation
   lacks is a change of the language: parse it (`_EDGE`, `DirRef`), validate it in
   `BallPlan.compile` with a message that says what to change, evaluate it in `BallRule.match`
   (vectorised over candidate rows; unknown velocities are enumerated on the velocity box),
   draw it (`svg.rule_pattern`, `renderPattern` in `web/app.js`), document it in
   `report/assets/collisions.html`.
3. Tests (`tests/test_rules.py`, `tests/test_world.py`, `tests/test_balls.py`): what the rule
   does on a hand-placed pair (`graph.balls.place`), what it conserves over a random run
   (energy exact, momentum changes only at bounces, no shared sites), the messages of invalid
   notations. A chain rule also needs its predicted drift and diffusion reproduced by a free
   run (`MSD = 4 D t`).
4. Compare against a reference with `compare()` when the rule is meant to imitate something
   (the ball gas against the hard disks in `experiments/balls.py`).

## Adding a space

Implement the `Space` protocol. Keep the neighbour table dense: pad with self-loops (which also
encode reflecting boundaries) or give the rule a per-node degree to sample from. Provide
`edges()` (undirected, no self-loops) and `to_networkx()` with `pos` attributes for views. Test
the structure against the matching networkx generator (`nx.grid_2d_graph` for the lattice).
Bodies, the line of centres and the captions use `lattice_cells` and need a square lattice.

## Rendering

The pages draw the space from `GraphHistory.to_dict()` (`coordinates`, `edges`, `anchors`,
`velocities`, `events` per frame); balls are drawn by `viz.js` `drawAgents` in their own colour
with their number, an arrow along the velocity and a ring when a rule changed their velocity
in the step. The rule diagram shows the sites ball 0 visited in a step (`paths`).

## Rules as data (editable in the GUI)

- `RuleSpec` (`graph/rules.py`): notation, `weights` `(+x, -x, +y, -y)` for drawn edges,
  `self_loops`, `description`, `family`; JSON via `to_dict` / `from_dict`. Validation stays in
  Python (`chain()` for the chain family, `plan()` for ball rules): the editor shows the message,
  so a new constraint needs a message that says what to change.
- `rule_from_spec(data)` (chain rules only: `ExperimentConfig.rule`, the CLI `--rule`),
  `rules_from_data(list)` (any rules: `report.live.playground`, the ball gas's `custom` set),
  `rule_item(spec)` picks `ChainRule` or `BallRule`.
- The GUI edits a spec graphically (`web/app.js`, nodes = variables with three types, one edge
  set per side, annotations shown on the edges and edited in the text) and as text; both produce
  the same JSON.

## Presets

- `graph/gas.py` `LatticeGas`: Reflect, Turn, Swap, Stream on unit velocities (HPP at the
  contact); `simulate` returns a `GasRun` for the gas checks.
- `graph/balls.py` `COLLISION_RULE_SETS`: `exchange`, `swap`, `hpp`; `place(space, centres,
  velocities, radius)` refuses overlapping starts.
- `report/live.py` `playground`: placements (`center`, `corner`, `random`, `left`, `sites`,
  `head-on`, `diagonal`), `velocity` (`rest`, `unit`, `random`), `max_speed`, `radius`,
  explicit `velocities`.

## Next steps on the roadmap

Growing spaces (rules that add nodes/edges), projections computed from the graph (spectral
embedding), FHP (hexagonal lattice) for an isotropic lattice gas, gravity (a velocity increment
per step, potential energy), particles made of several nodes and edges with their own collision
rules. Keep `Space`, the notation and `WalkerSystem` as the seams; do not special-case a rule
in the simulation loop.
