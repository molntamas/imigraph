# Plan: GUI playground (world size, speed, particles) and a rule editor with export

Branch: `ai/issue-1`. Status at the bottom.

Request (2026-10-31): a UI element to experiment with the size of the world, the speed of the
steps and the number of particles; a GUI part for editing graph rules and exporting them, with a
detailed plan of the backend model and the GUI. Goal: understand the details of how the systems
work and play around with them.

## 1. What exists

`web/index.html` already has a parameter form for the *comparison run* (particles, steps, record
every, lattice size, spacing, step time, seed) but the run takes seconds, animates at a fixed
speed, and the rule is hard-coded (`RandomHop`, a Python class). The small explanatory
animations (rule diagram, graph step by step) are fixed demos.

## 2. Backend model for editable rules

### 2.1 A rule as data: `RuleSpec`

```json
{
  "name": "RandomHop",
  "variables": {"w": "walker", "s": "site", "t": "site"},
  "lhs": [["w", "s"], ["s", "t"]],
  "rhs": [["w", "t"], ["s", "t"]],
  "weights": [0.25, 0.25, 0.25, 0.25],
  "self_loops": "allow"
}
```

* `lhs` / `rhs`: edge lists over variables; the text form `{w,s},{s,t} -> {w,t},{s,t}` is the
  notation used everywhere else in the project and is parsed / printed by `RuleSpec.parse` /
  `RuleSpec.notation`.
* Supported family in this round: **walker-anchored chain rules**. The left side has exactly one
  walker edge `{w, s0}` and a path of site edges `{s0,s1}, {s1,s2}, ...` (each new site variable
  introduced by an edge from the previous one); the right side keeps the site edges and re-attaches
  the walker to one of the path's sites. Anything else is rejected with a message that says which
  constraint failed (the editor shows it).
* `weights`: probability of each labelled direction `(+x, -x, +y, -y)` per chain edge (default
  uniform). `self_loops`: `allow` (a boundary walker may stay, as in `RandomHop`) or `forbid`
  (choose among real edges only).
* Physics derived from the spec (shown in the editor and the rules table): chain length `k`
  gives `D = k a^2 / (4 tau)` for uniform weights; weights give the drift velocity
  `v = (k a / tau) sum_d w_d e_d` and the per-component diffusion
  `D_x = k a^2 (w_+x + w_-x - (w_+x - w_-x)^2) / (2 tau)` (same for `y`).

### 2.2 `ChainRule`

`ChainRule(spec)` implements the `Rule` protocol with numpy only: for each chain edge draw a
direction per walker (`rng.choice(4, p=weights)`, or uniform over real edges if self-loops are
forbidden), gather through the neighbour table, and anchor the walker at the chosen site.
`apply_with_paths` also returns the visited path `(n, k + 1)` for the rule diagram.
`RandomHop` stays as the hand-written reference; a test checks that `ChainRule` with the
RandomHop spec is statistically identical (same MSD, same stationary distribution) and that the
experiment passes with it.

### 2.3 Where rules plug in

* `ExperimentConfig.rule: dict | None` (a `RuleSpec` as JSON) so the comparison run, the CLI
  (`--rule '<json>'`) and the GUI run the edited rule. `None` means `RandomHop`.
* `report.live.playground_json(request)`: runs a *small* world (size 3..15, 1..30 walkers,
  up to 200 steps) with the given rule and returns the demo blobs (space, anchors, paths),
  the rule description, and quick metrics (mean displacement per step, measured `D`, drift).
  Fast (milliseconds), so the GUI can re-run it on every change.

## 3. GUI

New section at the top of `web/index.html`, before the comparison run: **Playground**.

```
+----------------------------------------------------------------------------------+
| Playground                                                                       |
| world size [ 7 ]  walkers [ 4 ]  steps [ 40 ]  seed [ 1 ]  playback [====] 3 fps |
| Rule  preset [RandomHop v]                                                        |
|   {w,s},{s,t} -> {w,t},{s,t}                        [textarea, notation]          |
|   weights +x [0.25] -x [0.25] +y [0.25] -y [0.25]   self-loops [allow v]          |
|   D = ..., drift = ...   (error message if the rule is not supported)            |
|   [Apply]  [Export JSON]  [Import JSON]  [Use in comparison run]                 |
| +------------------------------+  +------------------------------+              |
| | rule diagram: phases         |  | graph step by step            |              |
| | (match / choose / rewrite)   |  | (walkers on the small world)  |              |
| +------------------------------+  +------------------------------+              |
| measured: <|dr|^2>/step, D, drift, wall rate, occupied cells                     |
+----------------------------------------------------------------------------------+
```

* Presets: RandomHop, DoubleHop (`{w,s},{s,t},{t,u} -> {w,u},...`), BiasedHop (`+x` weight 0.55),
  NoStay (`self_loops: forbid`), each a `RuleSpec` JSON embedded in the page.
* Export: a Blob download `<name>.rule.json`; Import: a file input (and the textarea accepts
  pasted JSON too). The exported JSON is exactly what `ExperimentConfig.rule` and the CLI take,
  so an exported rule can be run from the command line and shows up in the static report.
* Playback speed: one range input that sets the interval of every player (`Player.setInterval`).
* "Use in comparison run" copies the rule into the hidden `rule` field of the comparison form; the
  rules table and the physics line of the comparison then describe the edited rule, and the checks
  show whether it still imitates Brownian motion (a biased rule fails the marginal checks, a double
  hop fails the MSD check unless `tau` is doubled: that is the experiment the user can do).

The static report (`python -m imigraph report --rule '<json>'`) uses the same `ChainRule`, so a
rule found in the GUI can be put into a timestamped page.

## 4. Not in this round (recorded for later)

* Rules with several walkers in the pattern (interaction rules) in the editor: the lattice gas
  (see `plans/2026-10-31-conserving-lattice-gas.md`) has them hand-written; a general matcher
  for multi-walker patterns needs per-site grouping and conflict resolution, a separate design.
* Rules that change the space (growth), hypergraph rules, a visual drag-and-drop pattern editor.
* A Web Worker for the comparison run (the UI blocks for a few seconds during a run).

## 5. Order of work

1. `RuleSpec` + `ChainRule` + tests (parse/print round trip, rejection messages, RandomHop
   equivalence, drift and D of a biased rule reproduced by a free run).
2. `ExperimentConfig.rule`, CLI `--rule`, rules table from the spec.
3. `playground_json` + tests.
4. GUI section, player speed, export/import; headless check extended (apply a preset, export
   button exists, playground canvases painted).
5. Docs, skill update, commit, push.

## Status

- [x] plan written (2026-10-31)
- [ ] RuleSpec / ChainRule
- [ ] config, CLI, report
- [ ] playground endpoint
- [ ] GUI
- [ ] verified headlessly, committed
