# imigraph: requirements and research notes

The living requirements document of the project. It started as the brainstorm in
[issue #1](https://github.com/molntamas/imigraph/issues/1) and is kept up to date as decisions
are made. What has been built against it is tracked in `plans/`.

## 1. Purpose

Simulate reality using graphs as the engine: imitate how things move around, starting from
simple systems and building up. In the long run graphs should build up everything, from
space-time, forces and fields to quantum systems and the macroscopic universe. The name
*imigraph* stands for *imitating systems with graphs*.

The starting idea is the Wolfram physics model: an initial state, rule definitions, and rules
that are executed. Finding interesting rules is itself part of the project (trial and error,
search, and later machine learning).

## 2. Way of working

- Start small, with something simple, and build on it. Iterate together with the maintainer.
- Make the project easy to develop for AI agents and for humans:
  - scientific libraries (numpy, scipy, networkx) integrate directly,
  - the rules of a system are easy to understand for humans and agents,
  - every system runs natively under pytest *and* in the browser (Pyodide).
- Strict and detailed verification of how a system behaves. Keep reference ("ground truth")
  systems made with reliable computational-physics methods and compare against them.
- Develop and keep different implementations and compare how they behave.
- Keep systems small in the beginning so that verification stays fast; optimise later.
  Execution-time guardrails are part of the test suite.
- Nothing is installed on the developer's machine beyond Docker: all tooling lives in the
  container image, which is the same image sandcastle agents use.

## 3. Goals

- A performant core engine for graph simulations.
- Support simulation, imitation and exploration:
  - **Simulation exploration**: explore different graph worlds and judge whether they are useful
    for simulating nature.
  - **Connect to other modelling tools**: compare the results of other simulations and models
    with imigraph, and build imigraphs that imitate them.
- Interfaces for humans and agents: for now a GUI to see what happens and an automatic way to
  compare a simulation with test data; later CLI, import and export.
- A format for importing, exporting and comparing simulations and models (today:
  `imigraph.trajectory.Trajectory`, JSON-serialisable).

## 4. First task (done in issue #1)

1. A simple 2D Brownian-motion-in-a-box simulation as the reference point.
2. A graph simulation of it with two graphical representations: the graph structure step by
   step, and a projection to 2D that should match the Brownian motion. The simulation has an
   initial condition and explicit rules for how the graph evolves.
3. A reliable comparison framework between the graph system and the reference. Crucial.
4. Guardrail tests so the simulations do not get too slow.
5. Unit tests and software-engineering best practices.
6. HTML pages to open and look at the results.
7. Requirements, documentation and steps saved in the repository; skills that support the
   agent workflow.

See `docs/architecture.md` and `docs/comparison-method.md` for what was built. Added on the
maintainer's request afterwards: energy, momentum and entropy metrics; the rule explained in the
report with an animated diagram; a history of result pages; and a second, energy-conserving
reference system (hard-disk gas) with no graph imitation yet.

## 5. Modelling with graphs: starting points

- Initial state + rules + rule execution (Wolfram model).
- How to find interesting rules: search and trial and error; machine learning with gradient
  descent (see section 8).
- How space is built: the space itself could be grown by rules.
- Next systems to imitate (2D): kinematics; the visualisation of functions (the system builds up
  a function like log, linear, cos); a structure that changes from state A to state B.
- Hypergraphs and multiway graphs are optional.

## 6. Technology

- Python logic separated from the GUI; the GUI runs the same Python in WebAssembly (Pyodide,
  Python 3.14).
- Simulation verification: how to verify that a simulation matches (see
  `docs/comparison-method.md`).

## 7. Related repositories

- https://github.com/phcerdan/wolfram_model
- Visualisation: https://github.com/lguibr/wolfram-physics-explorer
- Graph: https://github.com/lantunes/netomaton/blob/master/netomaton/wolfram_physics_model.py
- https://github.com/met4citizen/Hypergraph/tree/main/modules

## 8. Learning discrete rules (research notes)

Discrete rules can be made machine-learnable; the trick is to train with continuous
mathematics and force the rules to crystallise into hard, discrete steps by the end of training.

1. **Gumbel-Softmax (differentiable switch).** The rule outputs a probability that an edge
   exists. During training the switch is soft and gradient descent works; a temperature is
   annealed towards zero so that the probabilities snap to 0/1. The final network is completely
   discrete.
2. **Genetic algorithms and rule-bashing (no gradients).** Define a small discrete rule table
   (e.g. a node counts its active neighbours and a few bits decide: die, live, grow edge). Keep
   a population of rule sets, score them with a fitness (e.g. how long a glider survives),
   keep the best, mutate, breed. Simple to program, rules stay human-readable.
3. **Neural logic networks (differentiable logic).** Treat the rules as a chain of boolean gates
   (AND, OR, NOT) with learnable soft gates that are hardened after training.

Graph neural networks and graph cellular automata are the search terms to follow up.

## 9. Skills wanted for agents

How to create a new skill; WebAssembly / frontend; Python; computational physics; graphs;
Wolfram models; GitHub; planning (e.g. Matt Pocock's `grill-me`, `wayfinder`). The project
skills live in `.claude/skills/`.
