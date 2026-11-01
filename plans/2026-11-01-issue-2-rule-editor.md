# Plan: rule editor (graphical and text), rule sets, an editable starting universe, saved result pages

Issue: [#2 Playground for trying out new graph rules](https://github.com/molntamas/imigraph/issues/2).
Branch: `ai/issue-2`. Status at the bottom.

## 1. What the issue asks for, and what existed

| acceptance criterion | before (issue #1) | this round |
| --- | --- | --- |
| rule editor via graph GUI | - | drawing of both sides of the selected rule, nodes and edges clickable (`web/app.js`, pattern editor) |
| rule editor via text | notation textarea, weights, self-loops | kept; text and drawing are two views of the same rule |
| see / add / delete / change rules | one rule at a time, five presets in a `<select>` | a rule list (the *rule set*): tick to run, click to edit, new, duplicate, delete, reorder, reset |
| set the initial world size | world size field | kept, plus walls (reflecting / periodic), walker placement, walkers placed by clicking sites |
| start a world with the rules, world and speed | Apply with one rule, playback speed | Apply runs the ticked rules in order in every step (`RuleSequence`), speed slider kept |
| result file saved | - | *Save result page* downloads one self-contained HTML page; `python -m imigraph playground-report` writes the same page from a session file; a page is published in imigraph-reports |

## 2. Backend

* `RuleSequence(rules)` (`graph/rules.py`): several `ChainRule`s applied in order in every step.
  Drifts add, per-component diffusions add (independent increments), `hops` is the total;
  `apply_with_paths` concatenates the matched chains for the rule diagram; `descriptions()`
  gives one description per rule plus the sequence. `rule_specs(data)` / `rule_from_spec(data)`
  accept `None`, one spec (dict) or a list; `ExperimentConfig.rule` and `--rule` accept the list.
* `report.live.playground(request)` takes the starting universe: `walls` (`reflecting` or
  `periodic`), `placement` (`center`, `corner`, `random`, `left`, `sites`) and `sites` (node
  indices clicked in the GUI); returns `rules` (descriptions with an SVG `pattern` each),
  `specs`, `universe` (the resolved starting sites) besides the demo and the measured values.
* `report.svg.rule_pattern(lhs, rhs, is_walker)`: the two sides of a rule as small graphs; the
  rules table of every page has a *pattern* column.
* `report.playground.build_playground_page(result, request)` + `assets/playground.html`: the
  saved page (rules, universe, the run with both animations, measured vs predicted, the session
  JSON to reproduce it). `write_playground_page` writes `output/playground_<time>_<label>.html`
  and refreshes the index; `live.playground_page_json` builds the page inside Pyodide.
* CLI: `python -m imigraph playground-report --request-file session.json --label <name>`
  (`docs/examples/playground-session.json` is the example session).

## 3. GUI (`web/index.html`, `web/app.js`)

```
1a. Rules
  [rule list: run | name | rewrite | weights]      [graphical editor: left side -> right side]
  New Duplicate Delete Move up Move down Reset      + site  + walker  Remove selected node
  name, self-loops, notation (text), weights
  Apply  Export rule JSON  Export rule set JSON  Import JSON  Use the rule set in the comparison run
1b. Starting universe and run
  world size, walls, walkers, placement, steps, seed, playback speed
  [universe canvas: click sites]  [rule set in four phases]  [world step by step + measured table]
  Save result page  Export session JSON
```

* Graphical editor: nodes are the variables (sites grey, walkers red), each side its own edge
  set, the same layout as the Python drawing. Click a node then another on the same side to
  add or remove the edge (a walker keeps one edge: connecting it elsewhere moves it); click an
  edge to remove it; `+ site` / `+ walker` add a variable; double-click renames. Every change
  rewrites the notation; typing a notation that parses redraws. Validation stays in Python: the
  message of `RuleSpec.chain()` is shown under the buttons.
* The rule list, the selection, the universe form and the clicked sites are kept in
  `localStorage` (per browser, wrapped in try/catch); *Reset to presets* clears them.
* Import accepts a rule, a list of rules, or a session (what *Export session JSON* writes,
  including the whole list).

## 4. Verification

* `tests/test_playground_page.py`: placements, periodic walls, rule sets (paths concatenated,
  drift and diffusion add up on a free run of 3000 walkers, within 10%), the page (self-contained,
  session embedded), the timestamped files, the CLI; `tests/test_live.py` updated.
* `docker/check-pages.py` exercises the GUI: tick / untick rules, a two-rule sequence, the
  graphical editor (remove and re-add an edge, add and connect a site, remove it), the text
  editor's error path, new and delete, periodic walls, a click on the universe canvas, the
  downloads (result page, session, rule), the comparison run with the rule set; then opens the
  downloaded result page and checks it like the CLI page.

## 5. Not in this round

Interaction rules (several walkers in one pattern) in the editor, rules that change the space,
drag-and-drop node positions, a Web Worker for the comparison run.

## Status

- [x] plan written (2026-11-01)
- [x] `RuleSequence`, rule lists in config / CLI / playground, `rule_pattern`
- [x] starting universe (walls, placement, clicked sites)
- [x] playground result page (template, writer, CLI, Pyodide entry point)
- [x] GUI: rule list, graphical editor, text editor synced, universe canvas, save / export / import
- [x] tests, headless check, docs and skills
- [x] pages generated and published to imigraph-reports
