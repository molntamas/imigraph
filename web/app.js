// Browser GUI: loads Pyodide, installs the imigraph wheel from dist/, runs the experiment with the
// parameters of the form and renders the result with the shared visualisation helpers.
(function () {
  "use strict";

  const PYODIDE_PACKAGES = ["numpy", "scipy", "micropip"];
  const byId = function (id) { return document.getElementById(id); };
  const status = function (text) { byId("status").textContent = text; };
  const V = window.imigraphViz;

  let players = [];
  let pyodide = null;
  let runJson = null;

  async function loadPython() {
    status("downloading Pyodide...");
    pyodide = await loadPyodide();
    status("loading numpy and scipy...");
    await pyodide.loadPackage(PYODIDE_PACKAGES);
    status("installing imigraph...");
    const wheel = (await (await fetch("dist/wheel.txt")).text()).trim();
    const micropip = pyodide.pyimport("micropip");
    await micropip.install("dist/" + wheel, { deps: false });
    runJson = pyodide.pyimport("imigraph.report.live").run_json;
    status("ready (" + wheel + ")");
    byId("run").textContent = "Run";
    byId("run").disabled = false;
  }

  function parameters() {
    const form = new FormData(byId("params"));
    const config = {};
    for (const [key, value] of form.entries()) {
      config[key] = key === "spacing" || key === "step_time" ? Number(value) : parseInt(value, 10);
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
    byId("markdown").textContent = result.markdown;
    byId("results").style.display = "block";
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

  byId("run").addEventListener("click", run);
  loadPython().catch(function (error) {
    status("failed to load Python: " + error);
    console.error(error);
  });
})();
