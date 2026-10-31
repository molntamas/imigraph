# Plan: energy and momentum conservation in the graph simulation, collisions of graph particles

Branch: `ai/issue-1` (follow-up work on the issue #1 setup). Status at the bottom.

Request (2026-10-31): "implement energy and momentum conservation in the graph simulation,
implement collision of graph particles", and document in the report what type of graph this is,
whether particle nodes differ from space nodes, how entropy, energy and momentum are computed in
the graph simulation and what happens when graph particles collide.

## 1. Why the random-hop graph cannot conserve energy

`RandomHop` walkers carry no velocity: every step picks a fresh random direction. Energy and
momentum are *measured* (from the displacement of each step) but nothing in the rule keeps them;
the ensemble values are stationary only because the random choices are. Two walkers on one node
do not interact at all: the rule has a single walker in its pattern, so they pass through each
other. That is the correct imitation of overdamped Brownian motion (independent particles in a
bath), and the wrong one for a gas.

Conservation needs state that persists between steps: a **velocity**. On a graph the natural
velocity is *which edge the walker is about to traverse*.

## 2. The model: a lattice gas (HPP) written as graph-rewriting rules

Decision: implement the HPP lattice gas (Hardy, Pomeau, de Pazzis 1973), the simplest cellular
automaton that conserves mass, momentum and energy exactly, as rules on the existing walker graph.
Its known limitations (single speed, so no Maxwell-Boltzmann distribution; anisotropy of the square
lattice) are documented, not hidden. FHP on a hexagonal lattice fixes the anisotropy and is a
natural later step (`Space` is the seam).

### State

A walker is a hyperedge `{w, s, s'}`: walker `w` sits on site `s` and heads for the neighbour
`s'`. The space is the square lattice, whose edges carry a **direction label** `k` (the column of
the neighbour table: `+x, -x, +y, -y`), i.e. the lattice is used as the Cayley graph of `Z^2`.
Walkers have speed `c = a / tau` (one edge per step), mass `m`, velocity `c e_k`.

Exclusion principle: at most one walker per channel `(s, k)`. Initial conditions respect it and
the rules preserve it, which is what makes the update free of conflicts.

### Rules, applied to every walker synchronously, in this order every step

| rule | rewrite | matches / choice | conserves |
| --- | --- | --- | --- |
| Collide (HPP) | `{w1,s,a}, {w2,s,b}, {s,c}, {s,d} -> {w1,s,c}, {w2,s,d}, {s,c}, {s,d}` where `b` is opposite `a`, `{c,d}` the perpendicular pair, channels `c`, `d` empty and real edges (no self-loop) | one match per head-on pair; which walker takes `c` is chosen uniformly | mass, momentum (`a + b = 0 = c + d`), energy (speed unchanged) |
| Reflect | `{w,s,s} -> {w,s,s'}` where `{s,s}` is the self-loop of direction `k` and `s'` the neighbour in direction `-k` | deterministic | energy; momentum is handed to the wall (specular wall, as in the hard-disk gas) |
| Stream | `{w,s,s'}_k, {s',s''}_k -> {w,s',s''}_k` | deterministic (same label `k`) | everything |

Periodic lattice (`SquareLattice(periodic=True)`): no self-loops, so Reflect never matches and the
total momentum is conserved exactly, like the periodic hard-disk gas.

### Measured quantities on the graph

* Energy: `E = sum m c^2 / 2 = n m c^2 / 2`, constant by construction. Reported per frame.
* Momentum: `P = m c sum_w e_{k(w)}`. Constant with periodic walls; with self-loop walls bounded
  fluctuation (every reflection flips one component of one walker).
* Entropy: Shannon entropy of the occupancy histogram of the projected positions on a `bins x bins`
  grid (same function as the gas and the Brownian experiment, `metrics.occupancy_entropy`).
* Collisions: number of Collide matches per recording interval.
* Speeds: all equal to `c` (histogram is a spike). The Maxwell check is run and reported as the
  expected failure that explains why HPP is not a gas thermometer.

### Comparison with the hard-disk gas

Both systems run a free expansion from the left half of the unit box with speed 1 and the same
recording times (gas: `dt = 0.004`, every 25 steps; lattice: `40 x 40`, `a = tau = 0.025`, every 4
steps; both record every 0.1 time units, 12 time units in total). Checks on both:

| check | hard-disk gas | lattice gas |
| --- | --- | --- |
| energy drift | ≤ 1e-9 | ≤ 1e-9 (exactly 0) |
| momentum | bounded / exact (periodic) | bounded / exact (periodic) |
| entropy rises to the uniform value | yes | yes |
| Maxwell-Boltzmann speeds | yes | fails by design, reported |

The `compare()` framework (MSD, marginals, occupancy) is run on the two projected trajectories as
well: the informative result is that the late-time occupancy agrees (both uniform) while the early
marginals differ (ballistic lattice streaming vs disks on a grid). Thresholds are not tuned to make
it pass; the page shows the checks and says which ones are expected to differ.

## 3. Code

```
src/imigraph/graph/space.py        SquareLattice(periodic=...): wrap-around neighbour table, no self-loops
src/imigraph/graph/gas.py          LatticeGasState (sites, directions), LatticeGas rules + simulate() -> GasRun
                                   OPPOSITE / PERPENDICULAR direction tables, lattice_gas_initial_half_box()
src/imigraph/experiments/gas.py    LatticeGasConfig, run_lattice_gas(); GasComparison = both systems + compare()
src/imigraph/report/gas.py         page: gas and lattice gas side by side, both check tables, overlaid charts,
                                   rules table of the lattice gas, "Graph model" questions answered
src/imigraph/report/assets/gas.html, viz.js (drawDirectedWalkers: arrows for velocities)
src/imigraph/__main__.py           gas / gas-report run both systems
tests/test_lattice_gas.py          exclusion preserved; every step follows a labelled edge; energy exactly
                                   constant; momentum exact on periodic lattice, bounded otherwise; HPP
                                   collision conserves momentum per site; entropy rises; performance budget
```

`LatticeGas.simulate` returns the same `GasRun` as the hard-disk gas (trajectory, velocities,
collisions, mass), so the energy, momentum, Maxwell and entropy checks apply unchanged.

## 4. Documentation in the report (request part 3)

A "Graph model: questions and answers" card on both pages:

1. *What type of graph is this?* A static space graph (square lattice, edge labels = directions,
   self-loops = walls) plus one node per particle. The combined graph is bipartite between walker
   nodes and space nodes: walker nodes only connect to space nodes. Node types are explicit in
   `WalkerGraph.to_networkx()` (`kind="space"` / `kind="walker"`). With velocities the walker edge
   becomes a hyperedge `{w, s, s'}` (ordered: position, heading).
2. *How are entropy, energy and momentum computed?* Random hop: from the displacement of every
   step (`m |dr|^2 / 2 tau^2`, `m dr / tau`), stationary ensemble values, not conserved. Lattice
   gas: from the heading of every walker, conserved exactly. Entropy: coarse-grained occupancy in
   both.
3. *What happens when graph particles collide?* Random hop: nothing, walkers pass through each
   other (no multi-walker pattern). Lattice gas: the HPP rule rotates a head-on pair by 90°; other
   encounters (walkers crossing, same direction, perpendicular) do not interact, as in HPP.

## 5. Order of work

1. Periodic lattice + `graph/gas.py` + tests.
2. Experiment + CLI; check the numbers.
3. Page + arrows in `viz.js`; headless check; screenshots.
4. Docs (`docs/architecture.md`, graph-rules skill), commit, push, publish the page.

## Status

- [x] plan written (2026-10-31)
- [ ] lattice gas implemented and tested
- [ ] experiment, CLI
- [ ] page with the graph-model Q&A
- [ ] published to imigraph-reports
