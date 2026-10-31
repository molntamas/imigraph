// Browser GUI: loads Pyodide, installs the imigraph wheel from dist/, runs the playground (small
// world, editable rule) and the comparison experiment, and renders the results with the shared
// visualisation helpers.
(function () {
  "use strict";

  const PYODIDE_PACKAGES = ["numpy", "scipy", "micropip"];
  const byId = function (id) { return document.getElementById(id); };
  const status = function (text) { byId("status").textContent = text; };
  const V = window.imigraphViz;

  // Rule presets: the JSON form of imigraph.graph.rules.RuleSpec (what "Export" writes).
  const PRESETS = {
    RandomHop: { name: "RandomHop", notation: "{w,s}, {s,t} -> {w,t}, {s,t}",
                 weights: [0.25, 0.25, 0.25, 0.25], self_loops: "allow" },
    DoubleHop: { name: "DoubleHop", notation: "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}",
                 weights: [0.25, 0.25, 0.25, 0.25], self_loops: "allow" },
    BiasedHop: { name: "BiasedHop", notation: "{w,s}, {s,t} -> {w,t}, {s,t}",
                 weights: [0.55, 0.15, 0.15, 0.15], self_loops: "allow" },
    NoStay: { name: "NoStay", notation: "{w,s}, {s,t} -> {w,t}, {s,t}",
              weights: [0.25, 0.25, 0.25, 0.25], self_loops: "forbid" },
    LookAhead: { name: "LookAhead", notation: "{w,s}, {s,t}, {t,u} -> {w,t}, {s,t}, {t,u}",
                 weights: [0.25, 0.25, 0.25, 0.25], self_loops: "allow" },
  };

  let players = [];          // comparison-run players
  let playgroundPlayers = [];
  let pyodide = null;
  let runJson = null;
  let playgroundJson = null;

  async function loadPython() {
    status("downloading Pyodide...");
    pyodide = await loadPyodide();
    status("loading numpy and scipy...");
    await pyodide.loadPackage(PYODIDE_PACKAGES);
    status("installing imigraph...");
    const wheel = (await (await fetch("dist/wheel.txt")).text()).trim();
    const micropip = pyodide.pyimport("micropip");
    await micropip.install("dist/" + wheel, { deps: false });
    const live = pyodide.pyimport("imigraph.report.live");
    runJson = live.run_json;
    playgroundJson = live.playground_json;
    status("ready (" + wheel + ")");
    byId("load-status").textContent = "Python is ready.";
    byId("run").textContent = "Run";
    byId("run").disabled = false;
    byId("pg-apply").disabled = false;
    applyPlayground();
  }

  // ---------- rule editor ----------

  function editorSpec() {
    return {
      name: byId("pg-name").value.trim() || "custom",
      notation: byId("pg-notation").value.trim(),
      weights: [0, 1, 2, 3].map(function (k) { return Number(byId("pg-w" + k).value); }),
      self_loops: byId("pg-loops").value,
    };
  }

  function fillEditor(spec) {
    byId("pg-name").value = spec.name || "custom";
    byId("pg-notation").value = spec.notation;
    (spec.weights || [0.25, 0.25, 0.25, 0.25]).forEach(function (w, k) { byId("pg-w" + k).value = w; });
    byId("pg-loops").value = spec.self_loops || "allow";
  }

  function playgroundRequest() {
    const form = new FormData(byId("playground"));
    const request = { rule: editorSpec() };
    for (const [key, value] of form.entries()) request[key] = parseInt(value, 10);
    return request;
  }

  function fmt(x) { return Number(x).toFixed(3); }

  function renderPlayground(result) {
    playgroundPlayers.forEach(function (p) { p.pause(); });
    playgroundPlayers = [];
    byId("pg-error").textContent = result.error || "";
    if (result.error) { byId("pg-physics").textContent = ""; return; }
    const demo = result.demo;
    byId("pg-physics").innerHTML = "<strong>" + result.rule.name + "</strong>: <code>" + result.rule.rewrite +
      "</code><br>" + result.rule.physics;

    const rule = V.Viewport(byId("pg-rule-canvas"), demo.space.box, 30);
    const steps = demo.times.length - 1;
    playgroundPlayers.push(V.Player(steps * 4, function (f) {
      const step = Math.floor(f / 4), phase = f % 4;
      rule.clear();
      const path = demo.paths ? demo.paths[step][0] : [demo.anchors[step][0], demo.anchors[step + 1][0]];
      V.drawRuleStep(rule, demo.space, path, demo.anchors[step + 1][0], phase);
      byId("pg-rule-phase").textContent = V.RULE_PHASES[phase];
    }, { slider: byId("pg-rule-slider"), button: byId("pg-rule-play"), label: byId("pg-rule-label"),
         labelFor: function (f) { return "step " + Math.floor(f / 4) + "." + (f % 4); }, interval: 700 }));

    const world = V.Viewport(byId("pg-demo-canvas"), demo.space.box, 24);
    playgroundPlayers.push(V.Player(demo.times.length, function (f) {
      world.clear();
      V.drawSpace(world, demo.space.coordinates, demo.space.edges, 4);
      V.drawWalkers(world, demo.space.coordinates, demo.anchors[f], demo.space.spacing);
    }, { slider: byId("pg-demo-slider"), button: byId("pg-demo-play"), label: byId("pg-demo-label"),
         labelFor: function (f) { return "step " + f; }, interval: 500 }));

    const m = result.measured, p = result.predicted;
    byId("pg-measured").innerHTML =
      '<table class="kv"><tbody>' +
      "<tr><th>mean |&Delta;r|&sup2; per step (a&sup2;)</th><td>" + fmt(m.mean_squared_step) + "</td></tr>" +
      "<tr><th>diffusion D measured / predicted (x, y)</th><td>" + fmt(m.diffusion) + " / " +
        fmt(p.diffusion[0]) + ", " + fmt(p.diffusion[1]) + "</td></tr>" +
      "<tr><th>drift v measured / predicted</th><td>(" + fmt(m.drift[0]) + ", " + fmt(m.drift[1]) + ") / (" +
        fmt(p.drift[0]) + ", " + fmt(p.drift[1]) + ")</td></tr>" +
      "<tr><th>wall rate (fraction of walkers per step)</th><td>" + fmt(m.wall_rate) + "</td></tr>" +
      "<tr><th>occupied sites at the end</th><td>" + m.occupied_final + "</td></tr>" +
      "</tbody></table>" +
      '<p class="muted">Few walkers and walls make the measured values noisy; raise the world size and the walkers to see them converge.</p>';
    byId("pg-rules-table").innerHTML = rulesTable([result.rule]);
    V.speedControl(byId("pg-speed"), byId("pg-speed-label"), playgroundPlayers);
    playgroundPlayers.forEach(function (p) { p.play(); });
  }

  function rulesTable(rules) {
    const esc = function (s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); };
    return '<div class="table-scroll"><table><thead><tr><th>rule</th><th>rewrite</th><th>matches</th><th>choice</th>' +
      "<th>invariants</th><th>physics</th></tr></thead><tbody>" +
      rules.map(function (r) {
        return "<tr><th>" + esc(r.name) + "</th><td><code>" + esc(r.rewrite) + "</code></td><td>" + esc(r.matches) +
          "</td><td>" + esc(r.choice) + "</td><td>" + esc(r.invariants) + "</td><td>" + esc(r.physics) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function applyPlayground() {
    if (!playgroundJson) return;
    byId("pg-status").textContent = "running...";
    setTimeout(function () {
      try {
        const started = performance.now();
        const result = JSON.parse(playgroundJson(JSON.stringify(playgroundRequest())));
        renderPlayground(result);
        byId("pg-status").textContent = "done in " + ((performance.now() - started) / 1000).toFixed(2) + " s";
      } catch (error) {
        byId("pg-error").textContent = "error: " + error;
        byId("pg-status").textContent = "failed";
        console.error(error);
      }
    }, 20);
  }

  function exportRule() {
    const spec = editorSpec();
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = (spec.name || "rule").replace(/[^A-Za-z0-9_-]+/g, "-") + ".rule.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
    byId("pg-status").textContent = "exported " + link.download + " (run it with: python -m imigraph report --rule \"$(cat " + link.download + ")\")";
  }

  function importRule(file) {
    const reader = new FileReader();
    reader.onload = function () {
      try {
        fillEditor(JSON.parse(reader.result));
        byId("pg-preset").value = "";
        applyPlayground();
      } catch (error) {
        byId("pg-error").textContent = "could not read the rule file: " + error;
      }
    };
    reader.readAsText(file);
  }

  // ---------- comparison run ----------

  function parameters() {
    const form = new FormData(byId("params"));
    const config = {};
    for (const [key, value] of form.entries()) {
      if (key === "rule") { if (value) config.rule = JSON.parse(value); }
      else config[key] = key === "spacing" || key === "step_time" ? Number(value) : parseInt(value, 10);
    }
    return config;
  }

  function render(result) {
    players.forEach(function (p) { p.pause(); });
    players = [];
    const data = result.data;
    const timeLabel = function (f) { return "t = " + data.reference.times[f].toFixed(2); };

    const verdict = byId("verdict");
    verdict.textContent = result.passed ? "PASS" : "FAIL";
    verdict.className = "badge " + (result.passed ? "pass" : "fail");

    const pairReference = V.Viewport(byId("pair-reference-canvas"), data.reference.box);
    const pairGraph = V.Viewport(byId("pair-graph-canvas"), data.graph.box);
    players.push(V.Player(data.reference.times.length, function (f) {
      pairReference.clear();
      V.drawParticles(pairReference, data.reference.positions[f], V.COLORS.reference, 2.5);
      pairGraph.clear();
      V.drawParticles(pairGraph, data.graph.positions[f], V.COLORS.candidate, 2.5);
    }, { slider: byId("pair-slider"), button: byId("pair-play"), label: byId("pair-label"),
         labelFor: timeLabel, interval: 100 }));

    const rule = V.Viewport(byId("rule-canvas"), data.rule.space.box, 40);
    players.push(V.Player((data.rule.times.length - 1) * 4, function (f) {
      const step = Math.floor(f / 4), phase = f % 4;
      rule.clear();
      const path = data.rule.paths ? data.rule.paths[step][0] : [data.rule.anchors[step][0], data.rule.anchors[step + 1][0]];
      V.drawRuleStep(rule, data.rule.space, path, data.rule.anchors[step + 1][0], phase);
      byId("rule-phase").textContent = V.RULE_PHASES[phase];
    }, { slider: byId("rule-slider"), button: byId("rule-play"), label: byId("rule-label"),
         labelFor: function (f) { return "step " + Math.floor(f / 4) + "." + (f % 4); }, interval: 700 }));

    const demo = V.Viewport(byId("demo-canvas"), data.demo.space.box, 24);
    players.push(V.Player(data.demo.times.length, function (f) {
      demo.clear();
      V.drawSpace(demo, data.demo.space.coordinates, data.demo.space.edges, 4);
      V.drawWalkers(demo, data.demo.space.coordinates, data.demo.anchors[f], data.demo.space.spacing);
    }, { slider: byId("demo-slider"), button: byId("demo-play"), label: byId("demo-label"),
         labelFor: function (f) { return "step " + f; }, interval: 500 }));

    byId("checks-table").innerHTML = result.fragments.checks_table;
    byId("msd-chart").innerHTML = result.fragments.msd_chart;
    byId("marginal-early").innerHTML = result.fragments.marginal_early;
    byId("marginal-final").innerHTML = result.fragments.marginal_final;
    byId("occupancy-charts").innerHTML = result.fragments.occupancy_charts;
    byId("energy-chart").innerHTML = result.fragments.energy_chart;
    byId("momentum-chart").innerHTML = result.fragments.momentum_chart;
    byId("entropy-chart").innerHTML = result.fragments.entropy_chart;
    byId("wall-rate-chart").innerHTML = result.fragments.wall_rate_chart;
    byId("occupied-chart").innerHTML = result.fragments.occupied_chart;
    byId("rules-table").innerHTML = result.fragments.rules_table;
    byId("markdown").textContent = result.markdown;
    byId("results").style.display = "block";
    V.speedControl(byId("speed"), byId("speed-label"), players);
    players.forEach(function (p) { p.play(); });
  }

  function run() {
    const button = byId("run");
    button.disabled = true;
    status("running...");
    // Let the status repaint before Python blocks the main thread.
    setTimeout(function () {
      const started = performance.now();
      try {
        const result = JSON.parse(runJson(JSON.stringify(parameters())));
        render(result);
        status("done in " + ((performance.now() - started) / 1000).toFixed(1) + " s");
      } catch (error) {
        status("error: " + error);
        console.error(error);
      } finally {
        button.disabled = false;
      }
    }, 30);
  }

  // ---------- wiring ----------

  byId("pg-preset").addEventListener("change", function () {
    const preset = PRESETS[byId("pg-preset").value];
    if (preset) { fillEditor(preset); applyPlayground(); }
  });
  byId("pg-apply").addEventListener("click", applyPlayground);
  byId("pg-export").addEventListener("click", exportRule);
  byId("pg-import-button").addEventListener("click", function () { byId("pg-import").click(); });
  byId("pg-import").addEventListener("change", function () {
    if (byId("pg-import").files.length) importRule(byId("pg-import").files[0]);
    byId("pg-import").value = "";
  });
  byId("pg-use").addEventListener("click", function () {
    byId("run-rule-json").value = JSON.stringify(editorSpec());
    byId("run-rule").value = "custom";
    status("the comparison run will use rule " + editorSpec().name);
  });
  byId("run-rule").addEventListener("change", function () {
    byId("run-rule-json").value = byId("run-rule").value === "custom" ? JSON.stringify(editorSpec()) : "";
  });
  byId("run").addEventListener("click", run);
  loadPython().catch(function (error) {
    status("failed to load Python: " + error);
    console.error(error);
  });
})();
