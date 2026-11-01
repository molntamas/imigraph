// Browser GUI: loads Pyodide, installs the imigraph wheel (fetched from dist/, or embedded in the
// page by `python -m imigraph gui-page`), runs the playground (a rule set edited graphically or
// as text, a starting universe, a small world) and the comparison experiment, and renders the
// results with the shared visualisation helpers.
(function () {
  "use strict";

  const PYODIDE_PACKAGES = ["numpy", "scipy"];
  const STORAGE_KEY = "imigraph.playground.v2";
  const byId = function (id) { return document.getElementById(id); };
  const status = function (text) { byId("status").textContent = text; };
  const V = window.imigraphViz;
  const UNIFORM = [0.25, 0.25, 0.25, 0.25];

  // Rule presets: the JSON form of imigraph.graph.rules.RuleSpec (what "Export" writes), plus
  // whether the rule is part of the running rule set.
  const PRESETS = [
    { name: "RandomHop", notation: "{w,s}, {s,t} -> {w,t}, {s,t}", weights: UNIFORM, self_loops: "allow", enabled: true,
      description: "one hop per step in a uniformly random direction: the Brownian imitation" },
    { name: "DoubleHop", notation: "{w,s}, {s,t}, {t,u} -> {w,u}, {s,t}, {t,u}", weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "two hops per step: twice the diffusion" },
    { name: "BiasedHop", notation: "{w,s}, {s,t} -> {w,t}, {s,t}", weights: [0.55, 0.15, 0.15, 0.15], self_loops: "allow", enabled: false,
      description: "one hop per step, +x preferred: a drift" },
    { name: "NoStay", notation: "{w,s}, {s,t} -> {w,t}, {s,t}", weights: UNIFORM, self_loops: "forbid", enabled: false,
      description: "like RandomHop, but a walker at the wall must move" },
    { name: "LookAhead", notation: "{w,s}, {s,t}, {t,u} -> {w,t}, {s,t}, {t,u}", weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "matches two chain edges but moves along the first only" },
    { name: "Stream", builtin: "stream", notation: "{w,s,s'}_k, {s',s''}_k -> {w,s',s''}_k", enabled: false,
      description: "hyperedge walkers: move one edge along the heading and keep it (free flight)" },
    { name: "Reflect", builtin: "reflect", notation: "{w,s,s} -> {w,s,s'}", enabled: false,
      description: "hyperedge walkers: a walker heading into a wall turns round" },
  ];
  const STRING_FIELDS = ["walls", "placement", "walker_model", "collisions", "heading", "overlaps"];
  const SITE_NAMES = ["s", "t", "u", "v", "p", "q", "r"];

  let players = [];          // comparison-run players
  let playgroundPlayers = [];
  let pyodide = null;
  let runJson = null;
  let playgroundJson = null;
  let playgroundPageJson = null;

  // ---------- state ----------

  let library = [];          // [{id, name, notation, weights, self_loops, types, enabled}]
  let selectedId = null;
  let nextId = 1;
  let pattern = null;        // graphical editor state of the selected rule: {variables, lhs, rhs}
  let picked = null;         // {side: "lhs"|"rhs", name} of the node picked in the editor
  let customSites = null;    // walker sites chosen by clicking the universe canvas
  let customHeadings = null; // one heading per custom site (from the collision demo), or null
  let demoPlayers = [];
  let collisionDemoJson = null;
  let newtonJson = null;
  let newtonPlayers = [];
  let lastResult = null;
  let applyTimer = null;

  function freshRule(spec) {
    const builtin = spec.builtin || null;
    const preset = builtin ? PRESETS.find(function (p) { return p.builtin === builtin; }) : null;
    return {
      id: nextId++,
      name: spec.name || (preset ? preset.name : "custom"),
      builtin: builtin,
      notation: builtin ? (preset ? preset.notation : spec.notation || "") : (spec.notation || "{w,s}, {s,t} -> {w,t}, {s,t}"),
      weights: (spec.weights || UNIFORM).map(Number),
      self_loops: spec.self_loops || "allow",
      types: spec.types || {},
      description: spec.description || "",
      enabled: spec.enabled !== undefined ? Boolean(spec.enabled) : true,
    };
  }

  function ruleSpec(rule) {
    if (rule.builtin) return { name: rule.name, builtin: rule.builtin, description: rule.description };
    return { name: rule.name, notation: rule.notation, weights: rule.weights, self_loops: rule.self_loops, types: rule.types,
             description: rule.description };
  }

  function isBuiltin(rule) { return Boolean(rule && rule.builtin); }

  function selectedRule() {
    return library.find(function (r) { return r.id === selectedId; }) || null;
  }

  function enabledRules() {
    return library.filter(function (r) { return r.enabled; });
  }

  function resetLibrary() {
    library = PRESETS.map(freshRule);
    selectedId = library[0].id;
  }

  function persist() {
    try {
      const form = new FormData(byId("playground"));
      const values = {};
      for (const [key, value] of form.entries()) values[key] = value;
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        library: library, selectedId: selectedId, form: values, customSites: customSites, customHeadings: customHeadings,
      }));
    } catch (error) { /* private window or storage blocked: the page still works */ }
  }

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!saved || !Array.isArray(saved.library) || !saved.library.length) return false;
      library = saved.library.map(freshRule);
      selectedId = library[Math.max(0, saved.library.findIndex(function (r) { return r.id === saved.selectedId; }))].id;
      const form = byId("playground");
      Object.keys(saved.form || {}).forEach(function (key) {
        if (form.elements[key]) form.elements[key].value = saved.form[key];
      });
      customSites = Array.isArray(saved.customSites) ? saved.customSites.map(Number) : null;
      customHeadings = Array.isArray(saved.customHeadings) ? saved.customHeadings.map(String) : null;
      return true;
    } catch (error) { return false; }
  }

  // ---------- notation <-> pattern ----------

  // "{w,s}, {s,t} -> {w,t}, {s,t}" -> {lhs: [[w,s],[s,t]], rhs: [[w,t],[s,t]]}; null if unreadable.
  function parseNotation(text) {
    if (text.indexOf("->") < 0) return null;
    const halves = text.split("->");
    const sides = halves.map(function (half) {
      const edges = [];
      const re = /\{([^{}]*)\}/g;
      let m;
      while ((m = re.exec(half)) !== null) {
        const vars = m[1].split(",").map(function (v) { return v.trim(); }).filter(Boolean);
        if (vars.length >= 2) edges.push(vars);
      }
      return edges;
    });
    if (!sides[0].length) return null;
    return { lhs: sides[0], rhs: sides[1] };
  }

  function formatSide(edges) {
    return edges.map(function (e) { return "{" + e.join(",") + "}"; }).join(", ");
  }

  function isWalker(name, types) {
    return (types && types[name]) ? types[name] === "walker" : name.charAt(0) === "w";
  }

  function patternFromRule(rule) {
    const parsed = parseNotation(rule.notation) || { lhs: [], rhs: [] };
    const variables = [];
    parsed.lhs.concat(parsed.rhs).forEach(function (edge) {
      edge.forEach(function (v) {
        if (!variables.some(function (x) { return x.name === v; })) {
          variables.push({ name: v, type: isWalker(v, rule.types) ? "walker" : "site" });
        }
      });
    });
    return { variables: variables, lhs: parsed.lhs, rhs: parsed.rhs };
  }

  // Writes the graphical state back into the selected rule (notation and types).
  function patternToRule() {
    const rule = selectedRule();
    if (!rule || !pattern) return;
    rule.notation = formatSide(pattern.lhs) + " -> " + formatSide(pattern.rhs);
    rule.types = {};
    pattern.variables.forEach(function (v) { rule.types[v.name] = v.type; });
    byId("pg-notation").value = rule.notation;
  }

  // Same layout as imigraph.report.svg.pattern_layout: sites in a row, walkers above the first
  // site they touch (unconnected walkers in the middle).
  function patternLayout(width, height) {
    const positions = {};
    const sites = pattern.variables.filter(function (v) { return v.type === "site"; });
    const walkers = pattern.variables.filter(function (v) { return v.type === "walker"; });
    const pad = 0.14 * width;  // sites spread over the panel, walkers above them
    sites.forEach(function (site, k) {
      const x = sites.length === 1 ? width / 2 : pad + k * (width - 2 * pad) / (sites.length - 1);
      positions[site.name] = [x, 0.72 * height];
    });
    const used = {};
    walkers.forEach(function (walker) {
      let x = width / 2;
      const edges = pattern.lhs.concat(pattern.rhs);
      for (let e = 0; e < edges.length; e++) {
        if (edges[e].indexOf(walker.name) < 0) continue;
        const other = edges[e].find(function (v) { return v !== walker.name && positions[v]; });
        if (other) { x = positions[other][0]; break; }
      }
      const k = used[x] || 0;
      used[x] = k + 1;
      positions[walker.name] = [x + 28 * k, 0.26 * height];
    });
    return positions;
  }

  function svgElement(tag, attributes) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attributes).forEach(function (k) { element.setAttribute(k, attributes[k]); });
    return element;
  }

  function renderPattern() {
    if (!pattern) return;
    const positions = patternLayout(240, 150);
    ["lhs", "rhs"].forEach(function (side) {
      const svg = byId("pg-" + side);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const edges = pattern[side];
      edges.forEach(function (edge, index) {
        const walkerEdge = edge.some(function (v) { return isWalkerName(v); });
        const color = walkerEdge ? V.COLORS.walkerEdge : V.COLORS.space;
        const a = positions[edge[0]], b = positions[edge[edge.length - 1]];
        if (!a || !b) return;
        let element;
        if (edge[0] === edge[edge.length - 1]) {
          element = svgElement("circle", { cx: a[0], cy: a[1] - 20, r: 12, fill: "none", stroke: color, "stroke-width": 3, class: "loop" });
        } else {
          element = svgElement("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color, "stroke-width": walkerEdge ? 4 : 3, class: "edge" });
        }
        element.appendChild(svgElement("title", {})).textContent = "remove edge {" + edge.join(",") + "}";
        element.addEventListener("click", function () { removeEdge(side, index); });
        svg.appendChild(element);
      });
      pattern.variables.forEach(function (v) {
        const p = positions[v.name];
        const isPicked = picked && picked.side === side && picked.name === v.name;
        const node = svgElement("circle", {
          cx: p[0], cy: p[1], r: 13, fill: v.type === "walker" ? V.COLORS.walker : V.COLORS.space,
          class: "node" + (isPicked ? " picked" : ""), "data-node": v.name,
        });
        node.appendChild(svgElement("title", {})).textContent =
          v.type + " " + v.name + (isPicked ? " (picked: click another node to connect, or this one to unpick)" : "");
        node.addEventListener("click", function () { clickNode(side, v.name); });
        node.addEventListener("dblclick", function () { renameNode(v.name); });
        svg.appendChild(node);
        const label = svgElement("text", { x: p[0], y: p[1] + 4, "text-anchor": "middle", "font-size": 12, fill: "white" });
        label.textContent = v.name;
        svg.appendChild(label);
      });
    });
    byId("pg-remove-node").disabled = !picked;
    byId("pg-editor-hint").textContent = picked
      ? "picked " + picked.name + " on the " + (picked.side === "lhs" ? "left" : "right") + " side: click another node there"
      : "";
  }

  function isWalkerName(name) {
    const v = pattern.variables.find(function (x) { return x.name === name; });
    return v ? v.type === "walker" : name.charAt(0) === "w";
  }

  function patternChanged() {
    patternToRule();
    renderPattern();
    renderLibrary();
    persist();
    scheduleApply();
  }

  function removeEdge(side, index) {
    if (isBuiltin(selectedRule())) return;
    pattern[side].splice(index, 1);
    picked = null;
    patternChanged();
  }

  function clickNode(side, name) {
    if (isBuiltin(selectedRule())) return;
    if (!picked || picked.side !== side) { picked = { side: side, name: name }; renderPattern(); return; }
    if (picked.name === name) { picked = null; renderPattern(); return; }
    const edges = pattern[side];
    const a = picked.name, b = name;
    const existing = edges.findIndex(function (e) {
      return e.length === 2 && ((e[0] === a && e[1] === b) || (e[0] === b && e[1] === a));
    });
    if (existing >= 0) {
      edges.splice(existing, 1);
    } else {
      // A walker has exactly one edge: connecting it somewhere else moves that edge.
      [a, b].forEach(function (v) {
        if (!isWalkerName(v)) return;
        for (let i = edges.length - 1; i >= 0; i--) if (edges[i].indexOf(v) >= 0) edges.splice(i, 1);
      });
      const walkerFirst = isWalkerName(b) && !isWalkerName(a);
      edges.push(walkerFirst ? [b, a] : [a, b]);
    }
    picked = null;
    patternChanged();
  }

  function freeName(type) {
    const taken = pattern.variables.map(function (v) { return v.name; });
    if (type === "walker") {
      for (let k = 1; k < 100; k++) { const n = k === 1 ? "w" : "w" + k; if (taken.indexOf(n) < 0) return n; }
    }
    for (let k = 0; k < SITE_NAMES.length; k++) if (taken.indexOf(SITE_NAMES[k]) < 0) return SITE_NAMES[k];
    for (let k = 2; k < 100; k++) if (taken.indexOf("s" + k) < 0) return "s" + k;
    return "x" + Date.now();
  }

  function addNode(type) {
    if (!pattern || isBuiltin(selectedRule())) return;
    const name = freeName(type);
    pattern.variables.push({ name: name, type: type });
    picked = { side: "lhs", name: name };
    patternToRule();
    renderPattern();
    byId("pg-editor-hint").textContent = "added " + type + " " + name + " to both sides; click a node on the left to connect it";
  }

  function removePickedNode() {
    if (!picked || isBuiltin(selectedRule())) return;
    const name = picked.name;
    pattern.variables = pattern.variables.filter(function (v) { return v.name !== name; });
    ["lhs", "rhs"].forEach(function (side) {
      pattern[side] = pattern[side].filter(function (e) { return e.indexOf(name) < 0; });
    });
    picked = null;
    patternChanged();
  }

  function renameNode(oldName) {
    if (isBuiltin(selectedRule())) return;
    const newName = (window.prompt("New name of " + oldName + " (same name on both sides):", oldName) || "").trim();
    if (!newName || newName === oldName || /[{},\s]/.test(newName)) return;
    if (pattern.variables.some(function (v) { return v.name === newName; })) return;
    pattern.variables.forEach(function (v) { if (v.name === oldName) v.name = newName; });
    ["lhs", "rhs"].forEach(function (side) {
      pattern[side] = pattern[side].map(function (e) { return e.map(function (v) { return v === oldName ? newName : v; }); });
    });
    picked = null;
    patternChanged();
  }

  // ---------- rule list and text editor ----------

  function renderLibrary() {
    const esc = function (s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); };
    const body = byId("pg-library").querySelector("tbody");
    body.innerHTML = library.map(function (rule) {
      const valid = isBuiltin(rule) || parseNotation(rule.notation) !== null;
      const weights = isBuiltin(rule) ? "" : rule.weights.map(function (w) { return Number(w).toFixed(2); }).join(" ") +
        (rule.self_loops === "forbid" ? ", no stay" : "");
      return '<tr data-id="' + rule.id + '" data-name="' + esc(rule.name) + '" class="' +
        (rule.id === selectedId ? "selected" : "") + (valid ? "" : " invalid") + (isBuiltin(rule) ? " builtin" : "") + '">' +
        '<td><input type="checkbox" class="enabled" ' + (rule.enabled ? "checked" : "") + ' title="run this rule"></td>' +
        "<td>" + esc(rule.name) + '</td><td class="notation">' + esc(rule.notation) + "</td>" +
        '<td class="muted">' + weights + '</td><td class="description">' + esc(rule.description || "") + "</td></tr>";
    }).join("");
    Array.prototype.forEach.call(body.querySelectorAll("tr"), function (row) {
      const id = Number(row.getAttribute("data-id"));
      row.querySelector("input.enabled").addEventListener("change", function (event) {
        const rule = library.find(function (r) { return r.id === id; });
        rule.enabled = event.target.checked;
        persist();
        applyPlayground();
      });
      row.addEventListener("click", function (event) {
        if (event.target.tagName === "INPUT") return;
        selectRule(id);
      });
    });
  }

  function selectRule(id) {
    selectedId = id;
    picked = null;
    const rule = selectedRule();
    if (!rule) return;
    byId("pg-name").value = rule.name;
    byId("pg-description").value = rule.description || "";
    byId("pg-notation").value = rule.notation;
    rule.weights.forEach(function (w, k) { byId("pg-w" + k).value = w; });
    byId("pg-loops").value = rule.self_loops;
    const builtin = isBuiltin(rule);
    byId("pg-notation").readOnly = builtin;
    byId("pg-loops").disabled = builtin;
    [0, 1, 2, 3].forEach(function (k) { byId("pg-w" + k).disabled = builtin; });
    ["pg-add-site", "pg-add-walker"].forEach(function (id) { byId(id).disabled = builtin; });
    document.querySelector(".pattern-editor").classList.toggle("readonly", builtin);
    pattern = patternFromRule(rule);
    renderPattern();
    if (builtin) byId("pg-editor-hint").textContent = "built-in rule: fixed rewrite on hyperedge walkers; name and description are yours";
    renderLibrary();
    persist();
  }

  // The text editor is the other view of the same rule: every keystroke updates the rule and
  // (when the text parses) the drawing.
  function editorChanged() {
    const rule = selectedRule();
    if (!rule) return;
    rule.name = byId("pg-name").value.trim() || "custom";
    rule.description = byId("pg-description").value.trim();
    if (!isBuiltin(rule)) {
      rule.notation = byId("pg-notation").value.trim();
      rule.weights = [0, 1, 2, 3].map(function (k) { return Number(byId("pg-w" + k).value); });
      rule.self_loops = byId("pg-loops").value;
      if (parseNotation(rule.notation)) { pattern = patternFromRule(rule); picked = null; renderPattern(); }
    }
    renderLibrary();
    persist();
  }

  function newRule(spec) {
    const rule = freshRule(spec || { name: "rule " + nextId, enabled: true });
    library.push(rule);
    selectRule(rule.id);
    applyPlayground();
  }

  function deleteSelected() {
    if (library.length <= 1) { byId("pg-status").textContent = "the rule set keeps at least one rule"; return; }
    const index = library.findIndex(function (r) { return r.id === selectedId; });
    library.splice(index, 1);
    selectRule(library[Math.min(index, library.length - 1)].id);
    applyPlayground();
  }

  function moveSelected(delta) {
    const index = library.findIndex(function (r) { return r.id === selectedId; });
    const target = index + delta;
    if (index < 0 || target < 0 || target >= library.length) return;
    const rule = library.splice(index, 1)[0];
    library.splice(target, 0, rule);
    renderLibrary();
    persist();
    applyPlayground();
  }

  // ---------- universe ----------

  function latticeGeometry(size, spacing) {
    const coordinates = [], edges = [];
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) {
      coordinates.push([i * spacing, j * spacing]);
      if (i + 1 < size) edges.push([i * size + j, (i + 1) * size + j]);
      if (j + 1 < size) edges.push([i * size + j, i * size + j + 1]);
    }
    const half = spacing / 2, upper = (size - 1) * spacing + half;
    return { coordinates: coordinates, edges: edges, box: { lower: [-half, -half], upper: [upper, upper] }, spacing: spacing };
  }

  function currentSize() {
    return Math.min(25, Math.max(3, parseInt(byId("playground").elements.size.value, 10) || 7));
  }

  function shownSites() {
    const placement = byId("playground").elements.placement.value;
    if (placement === "sites") return customSites || [];
    if (lastResult && lastResult.universe && lastResult.universe.size === currentSize()) return lastResult.universe.sites;
    return [];
  }

  function drawUniverse() {
    const size = currentSize();
    const geometry = latticeGeometry(size, 1);
    const canvas = byId("pg-universe-canvas");
    const vp = V.Viewport(canvas, geometry.box, 20);
    vp.clear();
    V.drawSpace(vp, geometry.coordinates, geometry.edges, 4);
    const sites = shownSites();
    const form = byId("playground").elements;
    const hyper = form.walker_model.value === "hyperedge";
    const directions = hyper && lastResult && lastResult.universe && lastResult.universe.directions &&
      lastResult.universe.sites.length === sites.length && lastResult.universe.size === size ? lastResult.universe.directions : null;
    if (directions) V.drawArrows(vp, geometry.coordinates, sites, directions, 1);
    else V.drawWalkers(vp, geometry.coordinates, sites, 1);
    byId("pg-universe-label").textContent = sites.length + (hyper ? " hyperedge" : " edge") + " walker" +
      (sites.length === 1 ? "" : "s") + " on a " + size + "×" + size + " world (" + form.walls.value + " walls, collisions: " +
      form.collisions.value + ")";
    form.heading.disabled = !hyper;
    canvas.onclick = function (event) {
      const point = vp.pointOf(event);
      const i = Math.round(point[0]), j = Math.round(point[1]);
      if (i < 0 || j < 0 || i >= size || j >= size) return;
      const site = i * size + j;
      if (customSites === null || byId("playground").elements.placement.value !== "sites") customSites = shownSites().slice();
      const at = customSites.indexOf(site);
      if (at >= 0) customSites.splice(at, 1); else customSites.push(site);
      customHeadings = null;  // headings of clicked sites follow the "initial heading" choice
      byId("playground").elements.placement.value = "sites";
      syncPlacement();
      drawUniverse();
      persist();
      scheduleApply();
    };
  }

  function syncPlacement() {
    const form = byId("playground");
    const custom = form.elements.placement.value === "sites";
    form.elements.walkers.disabled = custom;
    if (custom) form.elements.walkers.value = (customSites || []).length || form.elements.walkers.value;
  }

  // ---------- running the playground ----------

  function playgroundRequest() {
    const form = new FormData(byId("playground"));
    const request = { rule: enabledRules().map(ruleSpec) };
    for (const [key, value] of form.entries()) {
      request[key] = STRING_FIELDS.indexOf(key) >= 0 ? value : parseInt(value, 10);
    }
    if (request.placement === "sites") {
      request.sites = customSites || [];
      if (customHeadings && customHeadings.length === request.sites.length) request.headings = customHeadings;
    }
    return request;
  }

  function sessionRequest() {
    const request = playgroundRequest();
    request.library = library.map(function (r) { return Object.assign(ruleSpec(r), { enabled: r.enabled }); });
    return request;
  }

  function fmt(x) { return Number(x).toFixed(3); }

  function renderPlayground(result) {
    playgroundPlayers.forEach(function (p) { p.pause(); });
    playgroundPlayers = [];
    byId("pg-error").textContent = result.error || "";
    byId("pg-save").disabled = Boolean(result.error);
    if (result.error) { byId("pg-physics").textContent = ""; drawUniverse(); return; }
    lastResult = result;
    const demo = result.demo;
    byId("pg-physics").innerHTML = "<strong>" + escapeHtml(result.rule.name) + "</strong>: <code>" +
      escapeHtml(result.rule.rewrite) + "</code><br>" + escapeHtml(result.rule.physics);

    const rule = V.Viewport(byId("pg-rule-canvas"), demo.space.box, 30);
    const steps = demo.times.length - 1;
    playgroundPlayers.push(V.Player(steps * 4, function (f) {
      const step = Math.floor(f / 4), phase = f % 4;
      rule.clear();
      V.drawRuleStep(rule, demo.space, demo.paths[step][0], demo.anchors[step + 1][0], phase);
      byId("pg-rule-phase").textContent = V.RULE_PHASES[phase];
    }, { slider: byId("pg-rule-slider"), button: byId("pg-rule-play"), label: byId("pg-rule-label"),
         labelFor: function (f) { return "step " + Math.floor(f / 4) + "." + (f % 4); }, interval: 700 }));

    const world = V.Viewport(byId("pg-demo-canvas"), demo.space.box, 24);
    playgroundPlayers.push(V.Player(demo.times.length, function (f) {
      world.clear();
      V.drawSpace(world, demo.space.coordinates, demo.space.edges, 4, 1.5 * demo.space.spacing);
      if (demo.collisions && f > 0) V.drawCollisionMarks(world, demo.space.coordinates, demo.collisions[f - 1], demo.space.spacing);
      if (demo.directions) V.drawArrows(world, demo.space.coordinates, demo.anchors[f], demo.directions[f], demo.space.spacing);
      else V.drawWalkers(world, demo.space.coordinates, demo.anchors[f], demo.space.spacing);
      byId("pg-demo-caption").textContent = result.captions && result.captions.length ? result.captions[f] : "";
    }, { slider: byId("pg-demo-slider"), button: byId("pg-demo-play"), label: byId("pg-demo-label"),
         labelFor: function (f) { return "step " + f; }, interval: 500 }));

    const m = result.measured, p = result.predicted;
    const modelRows = result.model.walker_model === "hyperedge"
      ? "<tr><th>blocked hops (fraction of walkers per step, exclusion)</th><td>" + fmt(m.blocked_rate) + "</td></tr>" +
        "<tr><th>collisions in the run (collisions: " + escapeHtml(result.model.collisions) + ")</th><td>" + m.collisions + "</td></tr>" +
        "<tr><th>total momentum at the start / end (m a/&tau;), steps where it changed (wall reflections)</th><td>(" +
        m.momentum_start.map(fmt).join(", ") + ") / (" + m.momentum_end.map(fmt).join(", ") + "), " + m.momentum_changes + "</td></tr>"
      : "";
    byId("pg-measured").innerHTML =
      '<table class="kv"><tbody>' +
      "<tr><th>walker model / collisions / overlapping matches</th><td>" + escapeHtml(result.model.walker_model) + " / " +
        escapeHtml(result.model.collisions) + " / " + escapeHtml(result.model.overlaps) + "</td></tr>" +
      "<tr><th>waiting hops (fraction of walkers per step, overlapping matches)</th><td>" + fmt(m.waiting_rate) + "</td></tr>" +
      "<tr><th>mean |&Delta;r|&sup2; per step (a&sup2;)</th><td>" + fmt(m.mean_squared_step) + "</td></tr>" +
      "<tr><th>diffusion D measured / predicted (x, y)</th><td>" + fmt(m.diffusion) + " / " +
        fmt(p.diffusion[0]) + ", " + fmt(p.diffusion[1]) + "</td></tr>" +
      "<tr><th>drift v measured / predicted</th><td>(" + fmt(m.drift[0]) + ", " + fmt(m.drift[1]) + ") / (" +
        fmt(p.drift[0]) + ", " + fmt(p.drift[1]) + ")</td></tr>" +
      "<tr><th>hops per step (predicted)</th><td>" + p.hops + "</td></tr>" +
      "<tr><th>wall rate (fraction of walkers per step)</th><td>" + fmt(m.wall_rate) + "</td></tr>" +
      "<tr><th>occupied sites at the end</th><td>" + m.occupied_final + "</td></tr>" + modelRows +
      "</tbody></table>" +
      '<p class="muted">Few walkers and walls make the measured values noisy; raise the world size and the walkers to see them converge.</p>';
    byId("pg-rules-table").innerHTML = rulesTable(result.rules);
    if (result.docs) byId("pg-docs").innerHTML = result.docs;
    V.speedControl(byId("pg-speed"), byId("pg-speed-label"), playgroundPlayers);
    playgroundPlayers.forEach(function (p) { p.play(); });
    drawUniverse();
  }

  function escapeHtml(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }

  function rulesTable(rules) {
    const esc = escapeHtml;
    return '<div class="table-scroll"><table><thead><tr><th>rule</th><th>rewrite</th><th>matches</th><th>choice</th>' +
      "<th>invariants</th><th>physics</th><th>pattern</th></tr></thead><tbody>" +
      rules.map(function (r) {
        const name = esc(r.name) + (r.description ? '<br><span class="muted description">' + esc(r.description) + "</span>" : "");
        return "<tr><th>" + name + "</th><td><code>" + esc(r.rewrite) + "</code></td><td>" + esc(r.matches) +
          "</td><td>" + esc(r.choice) + "</td><td>" + esc(r.invariants) + "</td><td>" + esc(r.physics) + '</td><td class="pattern">' +
          (r.pattern || "") + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function scheduleApply() {
    if (applyTimer) clearTimeout(applyTimer);
    applyTimer = setTimeout(applyPlayground, 350);
  }

  function applyPlayground() {
    if (applyTimer) { clearTimeout(applyTimer); applyTimer = null; }
    if (!playgroundJson) return;
    byId("pg-status").textContent = "running...";
    setTimeout(function () {
      try {
        const started = performance.now();
        const result = JSON.parse(playgroundJson(JSON.stringify(playgroundRequest())));
        renderPlayground(result);
        byId("pg-status").textContent = (result.error ? "not run" : "done") + " in " +
          ((performance.now() - started) / 1000).toFixed(2) + " s";
      } catch (error) {
        byId("pg-error").textContent = "error: " + error;
        byId("pg-status").textContent = "failed";
        console.error(error);
      }
    }, 20);
  }

  // ---------- files ----------

  function download(text, filename, type) {
    const blob = new Blob([text], { type: type || "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
    return filename;
  }

  function safeName(name) { return String(name || "rule").replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "rule"; }

  function timestamp() {
    const d = new Date();
    const pad = function (n) { return String(n).padStart(2, "0"); };
    return d.getUTCFullYear() + "-" + pad(d.getUTCMonth() + 1) + "-" + pad(d.getUTCDate()) + "_" +
      pad(d.getUTCHours()) + pad(d.getUTCMinutes()) + pad(d.getUTCSeconds());
  }

  function exportRule() {
    const rule = selectedRule();
    if (!rule) return;
    const name = download(JSON.stringify(ruleSpec(rule), null, 2), safeName(rule.name) + ".rule.json");
    byId("pg-status").textContent = "exported " + name + " (run it with: python -m imigraph report --rule \"$(cat " + name + ")\")";
  }

  function exportRuleSet() {
    const rules = enabledRules().map(ruleSpec);
    const name = download(JSON.stringify(rules, null, 2), "rule-set.json");
    byId("pg-status").textContent = "exported " + rules.length + " rule(s) to " + name;
  }

  function exportSession() {
    const name = download(JSON.stringify(sessionRequest(), null, 2), "playground-session.json");
    byId("pg-status").textContent = "exported " + name + " (python -m imigraph playground-report --request-file " + name + ")";
  }

  function saveResultPage() {
    if (!playgroundPageJson) return;
    byId("pg-status").textContent = "building the page...";
    setTimeout(function () {
      try {
        const page = playgroundPageJson(JSON.stringify(sessionRequest()));
        if (!page) { applyPlayground(); return; }
        const names = enabledRules().map(function (r) { return safeName(r.name); }).join("-");
        const name = download(page, "playground_" + timestamp() + "_" + names + ".html", "text/html");
        byId("pg-status").textContent = "saved " + name;
      } catch (error) {
        byId("pg-error").textContent = "could not build the page: " + error;
        console.error(error);
      }
    }, 20);
  }

  // ---------- world 2: the bouncing ball ----------

  function newtonRequest() {
    const request = {};
    for (const [key, value] of new FormData(byId("newton")).entries()) request[key] = parseInt(value, 10);
    return request;
  }

  function renderNewton(result) {
    newtonPlayers.forEach(function (p) { p.pause(); });
    newtonPlayers = [];
    byId("nw-error").textContent = result.error || "";
    if (result.error) return;
    const run = result.run, s = result.summary;
    const vp = V.Viewport(byId("nw-canvas"), run.space.box, 24);
    newtonPlayers.push(V.Player(run.times.length, function (f) {
      vp.clear();
      V.drawSpace(vp, run.space.coordinates, run.space.edges, 3, 1.5 * run.space.spacing);
      const sites = run.sites[f];
      const ctx = vp.ctx;
      ctx.fillStyle = V.COLORS.walker;
      let cx = 0, cy = 0;
      for (let w = 0; w < sites.length; w++) {
        const a = run.space.coordinates[sites[w]];
        cx += a[0] / sites.length; cy += a[1] / sites.length;
        ctx.beginPath();
        ctx.arc(vp.x(a[0]), vp.y(a[1]), Math.max(4, 0.42 * run.space.spacing * vp.scale), 0, 2 * Math.PI);
        ctx.fill();
      }
      if (f > 0 && run.bounces[f - 1] > 0) {
        ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(vp.x(cx), vp.y(cy), (s.radius + 1) * run.space.spacing * vp.scale, 0, 2 * Math.PI);
        ctx.stroke();
      }
      const v = run.velocities[f];
      const speed = Math.hypot(v[0], v[1]);
      if (speed > 0) {
        const len = (s.radius + 1.2) * run.space.spacing;
        const x1 = vp.x(cx + len * v[0] / speed), y1 = vp.y(cy + len * v[1] / speed);
        ctx.strokeStyle = "#374151"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(vp.x(cx), vp.y(cy)); ctx.lineTo(x1, y1); ctx.stroke();
        const angle = Math.atan2(y1 - vp.y(cy), x1 - vp.x(cx));
        ctx.fillStyle = "#374151";
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 9 * Math.cos(angle - 0.5), y1 - 9 * Math.sin(angle - 0.5));
        ctx.lineTo(x1 - 9 * Math.cos(angle + 0.5), y1 - 9 * Math.sin(angle + 0.5));
        ctx.closePath(); ctx.fill();
      }
      const p = run.momentum[f];
      byId("nw-caption").textContent = "step " + f + ": v = (" + v[0] + ", " + v[1] + ") cells/step, E = " +
        fmt(run.energy[f]) + ", P = (" + fmt(p[0]) + ", " + fmt(p[1]) + "), S_config = " + fmt(run.configuration_entropy[f]) +
        ", S_visit = " + fmt(run.visitation_entropy[f]) + (f > 0 && run.bounces[f - 1] > 0 ? "; bounce: a velocity component flipped at the wall" : "");
    }, { slider: byId("nw-slider"), button: byId("nw-play"), label: byId("nw-label"),
         labelFor: function (f) { return "step " + f; }, interval: 400 }));
    byId("nw-summary").innerHTML = '<table class="kv"><tbody>' +
      "<tr><th>ball</th><td>" + s.walkers + " walker(s), radius " + s.radius + ", centre (" + s.centre.join(", ") + ")</td></tr>" +
      "<tr><th>velocity (cells/step)</th><td>(" + s.velocity.join(", ") + ")</td></tr>" +
      "<tr><th>room</th><td>" + s.size + "×" + s.size + ", reflecting walls, " + s.steps + " steps</td></tr>" +
      "<tr><th>kinetic energy E</th><td>" + fmt(s.energy) + " (constant)</td></tr>" +
      "<tr><th>|P|</th><td>" + fmt(s.momentum_norm) + " (constant)</td></tr>" +
      "<tr><th>bounces</th><td>" + s.bounces + "</td></tr>" +
      "<tr><th>room covered</th><td>" + Math.round(100 * s.coverage) + "% of the " + s.bins + "×" + s.bins + " entropy grid</td></tr></tbody></table>";
    byId("nw-checks").innerHTML = '<div class="table-scroll"><table><thead><tr><th>check</th><th>verdict</th><th>evidence</th></tr></thead><tbody>' +
      result.checks.map(function (c) {
        return "<tr><th>" + escapeHtml(c.name) + '</th><td><span class="badge ' + (c.passed ? "pass" : "fail") + '">' +
          (c.passed ? "PASS" : "FAIL") + "</span></td><td>" + escapeHtml(c.detail) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
    byId("nw-energy-chart").innerHTML = result.charts.energy;
    byId("nw-momentum-chart").innerHTML = result.charts.momentum;
    byId("nw-entropy-chart").innerHTML = result.charts.entropy;
    byId("nw-docs").innerHTML = result.docs;
    V.speedControl(byId("nw-speed"), byId("nw-speed-label"), newtonPlayers);
    newtonPlayers.forEach(function (p) { p.play(); });
  }

  function runNewton() {
    if (!newtonJson) return;
    byId("nw-status").textContent = "running...";
    setTimeout(function () {
      try {
        const started = performance.now();
        const result = JSON.parse(newtonJson(JSON.stringify(newtonRequest())));
        renderNewton(result);
        byId("nw-status").textContent = (result.error ? "not run" : "done") + " in " + ((performance.now() - started) / 1000).toFixed(2) + " s";
      } catch (error) {
        byId("nw-error").textContent = "error: " + error;
        byId("nw-status").textContent = "failed";
        console.error(error);
      }
    }, 20);
  }

  // ---------- collision demo (section 1c) ----------

  function renderCollisionDemo(demo) {
    demoPlayers.forEach(function (p) { p.pause(); });
    demoPlayers = [];
    const result = demo.result;
    byId("cd-docs").innerHTML = demo.docs;
    if (result.error) { byId("cd-caption").textContent = "error: " + result.error; return; }
    const d = result.demo;
    const world = V.Viewport(byId("cd-canvas"), d.space.box, 30);
    demoPlayers.push(V.Player(d.times.length, function (f) {
      world.clear();
      V.drawSpace(world, d.space.coordinates, d.space.edges, 5, 1.5 * d.space.spacing);
      if (f > 0) V.drawCollisionMarks(world, d.space.coordinates, d.collisions[f - 1], d.space.spacing);
      V.drawArrows(world, d.space.coordinates, d.anchors[f], d.directions[f], d.space.spacing);
      byId("cd-caption").textContent = result.captions[f];
    }, { slider: byId("cd-slider"), button: byId("cd-play"), label: byId("cd-label"),
         labelFor: function (f) { return "step " + f; }, interval: 1500 }));
    byId("cd-load").onclick = function () {
      const request = demo.request;
      library.forEach(function (r) { r.enabled = false; });
      request.rule.forEach(function (spec) {
        const existing = library.find(function (r) { return r.builtin && r.builtin === spec.builtin; });
        if (existing) { existing.enabled = true; } else { library.push(freshRule(Object.assign({ enabled: true }, spec))); }
      });
      const form = byId("playground");
      ["size", "walls", "walkers", "placement", "steps", "seed", "walker_model", "collisions", "overlaps"].forEach(function (key) {
        if (request[key] !== undefined && form.elements[key]) form.elements[key].value = request[key];
      });
      customSites = request.sites.slice();
      customHeadings = request.headings.slice();
      syncPlacement();
      selectRule(library.find(function (r) { return r.enabled; }).id);
      drawUniverse();
      applyPlayground();
      byId("cd-status").textContent = "loaded: the playground now runs this world (1b); change a heading or a site there";
      byId("pg-demo-canvas").scrollIntoView({ behavior: "smooth", block: "center" });
    };
  }

  // Accepts a single rule, a list of rules, or a session (rule list + universe, with the whole
  // library when it was exported from here).
  function importJson(text) {
    const data = JSON.parse(text);
    const form = byId("playground");
    if (Array.isArray(data) || (data && (data.notation || data.lhs) && !data.rule)) {
      // Imported rules become the running rule set; the rules already here stay, unticked.
      library.forEach(function (r) { r.enabled = false; });
      (Array.isArray(data) ? data : [data]).forEach(function (spec) {
        library.push(freshRule(Object.assign({}, spec, { enabled: true })));
      });
      selectRule(library[library.length - 1].id);
    } else if (data && typeof data === "object") {
      if (Array.isArray(data.library) && data.library.length) {
        library = data.library.map(freshRule);
      } else if (Array.isArray(data.rule)) {
        library = data.rule.map(function (spec) { return freshRule(Object.assign({ enabled: true }, spec)); });
      } else if (data.rule) {
        library = [freshRule(Object.assign({ enabled: true }, data.rule))];
      }
      ["size", "walls", "walkers", "placement", "steps", "seed", "walker_model", "collisions", "heading", "overlaps"].forEach(function (key) {
        if (data[key] !== undefined && data[key] !== null && form.elements[key]) form.elements[key].value = data[key];
      });
      customSites = Array.isArray(data.sites) ? data.sites.map(Number) : null;
      customHeadings = Array.isArray(data.headings) ? data.headings.map(String) : null;
      syncPlacement();
      selectRule(library[0].id);
    } else {
      throw new Error("not a rule, a rule list or a session");
    }
    drawUniverse();
    applyPlayground();
  }

  function importFile(file) {
    const reader = new FileReader();
    reader.onload = function () {
      try { importJson(reader.result); }
      catch (error) { byId("pg-error").textContent = "could not read the file: " + error; }
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

  // ---------- Python ----------

  // The wheel is embedded as base64 in <script id="wheel"> by the self-contained page, otherwise
  // fetched from dist/ (served by docker/serve-web.sh). Unpacked straight into site-packages, so
  // no micropip and no network access beyond Pyodide's own packages are needed.
  async function installWheel() {
    const embedded = byId("wheel");
    let name, bytes;
    if (embedded) {
      name = embedded.dataset.name;
      const binary = atob(embedded.textContent.replace(/\s+/g, ""));
      bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    } else {
      name = (await (await fetch("dist/wheel.txt")).text()).trim();
      bytes = new Uint8Array(await (await fetch("dist/" + name)).arrayBuffer());
    }
    const sitePackages = pyodide.runPython("import site; site.getsitepackages()[0]");
    pyodide.unpackArchive(bytes, "wheel", { extractDir: sitePackages });
    return name;
  }

  async function loadPython() {
    status("downloading Pyodide...");
    pyodide = await loadPyodide();
    status("loading numpy and scipy...");
    await pyodide.loadPackage(PYODIDE_PACKAGES);
    status("installing imigraph...");
    const wheel = await installWheel();
    const live = pyodide.pyimport("imigraph.report.live");
    runJson = live.run_json;
    playgroundJson = live.playground_json;
    playgroundPageJson = live.playground_page_json;
    collisionDemoJson = live.collision_demo_json;
    newtonJson = live.newton_json;
    byId("nw-run").disabled = false;
    status("ready (" + wheel + ")");
    byId("load-status").textContent = "Python is ready.";
    byId("run").textContent = "Run";
    byId("run").disabled = false;
    byId("pg-apply").disabled = false;
    applyPlayground();
    setTimeout(function () {
      try { renderCollisionDemo(JSON.parse(collisionDemoJson())); }
      catch (error) { byId("cd-caption").textContent = "could not run the collision demo: " + error; console.error(error); }
      runNewton();
    }, 60);
  }

  // ---------- wiring ----------

  if (!restore()) resetLibrary();
  nextId = library.reduce(function (m, r) { return Math.max(m, r.id); }, 0) + 1;
  selectRule(selectedId);
  syncPlacement();
  drawUniverse();

  ["pg-name", "pg-description", "pg-notation", "pg-loops", "pg-w0", "pg-w1", "pg-w2", "pg-w3"].forEach(function (id) {
    byId(id).addEventListener("input", editorChanged);
    byId(id).addEventListener("change", editorChanged);
  });
  byId("pg-add-site").addEventListener("click", function () { addNode("site"); });
  byId("pg-add-walker").addEventListener("click", function () { addNode("walker"); });
  byId("pg-remove-node").addEventListener("click", removePickedNode);
  byId("pg-new").addEventListener("click", function () { newRule(); });
  byId("pg-duplicate").addEventListener("click", function () {
    const rule = selectedRule();
    if (rule) newRule(Object.assign(ruleSpec(rule), { name: rule.name + " copy", enabled: rule.enabled }));
  });
  byId("pg-delete").addEventListener("click", deleteSelected);
  byId("pg-up").addEventListener("click", function () { moveSelected(-1); });
  byId("pg-down").addEventListener("click", function () { moveSelected(1); });
  byId("pg-reset").addEventListener("click", function () {
    resetLibrary();
    customSites = null;
    customHeadings = null;
    selectRule(selectedId);
    syncPlacement();
    drawUniverse();
    applyPlayground();
  });
  byId("pg-apply").addEventListener("click", applyPlayground);
  byId("pg-export").addEventListener("click", exportRule);
  byId("pg-export-all").addEventListener("click", exportRuleSet);
  byId("pg-export-session").addEventListener("click", exportSession);
  byId("pg-save").addEventListener("click", saveResultPage);
  byId("pg-import-button").addEventListener("click", function () { byId("pg-import").click(); });
  byId("pg-import").addEventListener("change", function () {
    if (byId("pg-import").files.length) importFile(byId("pg-import").files[0]);
    byId("pg-import").value = "";
  });
  // The comparison run imitates Brownian motion with edge walkers: only the chain rules go in.
  function comparisonRules() {
    return enabledRules().filter(function (r) { return !isBuiltin(r); }).map(ruleSpec);
  }
  byId("pg-use").addEventListener("click", function () {
    const rules = comparisonRules();
    if (!rules.length) { status("the comparison run needs at least one chain rule (built-in rules are left out)"); return; }
    byId("run-rule-json").value = JSON.stringify(rules);
    byId("run-rule").value = "custom";
    status("the comparison run will use the rule set " + rules.map(function (r) { return r.name; }).join(" + ") +
      (rules.length < enabledRules().length ? " (built-in rules left out)" : ""));
  });
  byId("run-rule").addEventListener("change", function () {
    byId("run-rule-json").value = byId("run-rule").value === "custom" ? JSON.stringify(comparisonRules()) : "";
  });
  Array.prototype.forEach.call(byId("playground").elements, function (element) {
    if (element.id === "pg-speed") return;
    element.addEventListener("change", function () {
      if (element.name === "placement") {
        if (element.value !== "sites") customSites = null;
        else if (!customSites) customSites = shownSites().slice();
      }
      syncPlacement();
      drawUniverse();
      persist();
      applyPlayground();
    });
  });
  byId("run").addEventListener("click", run);
  byId("nw-run").addEventListener("click", runNewton);
  Array.prototype.forEach.call(byId("newton").elements, function (element) {
    element.addEventListener("change", runNewton);
  });
  loadPython().catch(function (error) {
    status("failed to load Python: " + error);
    console.error(error);
  });
})();
