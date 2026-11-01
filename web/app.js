// Browser GUI: loads Pyodide, installs the imigraph wheel (fetched from dist/, or embedded in the
// page by `python -m imigraph gui-page`), runs the playground (a rule set edited graphically or
// as text, a starting universe, a small world), the comparison experiment and the ball gas, and
// renders the results with the shared visualisation helpers.
(function () {
  "use strict";

  const PYODIDE_PACKAGES = ["numpy", "scipy"];
  const STORAGE_KEY = "imigraph.playground.v3";
  const byId = function (id) { return document.getElementById(id); };
  const status = function (text) { byId("status").textContent = text; };
  const V = window.imigraphViz;
  const UNIFORM = [0.25, 0.25, 0.25, 0.25];

  // Rule presets: the JSON form of imigraph.graph.rules.RuleSpec (what "Export" writes), plus
  // whether the rule is part of the running rule set. The notations of Stream, Reflect, Exchange,
  // Swap, Turn and Kick are those of the library in imigraph.graph.rules.
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
    { name: "Kick", notation: "{w,s,v}, {s,t}_k, {o,v'}_k -> {w,t,v'}", weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "one hop in a drawn direction, and the ball then heads that way with unit speed" },
    { name: "Reflect", notation: "{w,s,v}_k, {s,s}_k, {v,v'}|k -> {w,s,v'}", weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "a ball about to hop into a wall mirrors that velocity component" },
    { name: "Exchange", notation: "{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2, {s,t}=d, {v1,v1'}=d, {v2,v2'}=d, {v1,v2'}=d^, {v2,v1'}=d^, {v1',v1}>d -> {w1,s,v1'}, {w2,t,v2'}",
      weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "two balls whose hops meet exchange the velocity component along the line of centres d (the elastic collision of equal masses); no match when the exchanged velocities are not lattice nodes (an odd diagonal hit)" },
    { name: "Swap", notation: "{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2 -> {w1,s,v2}, {w2,t,v1}", weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "two balls whose hops meet swap their whole velocities: each continues the other's motion from its own site (the head-on outcome for every geometry); the fallback after Exchange" },
    { name: "Turn", notation: "{w1,s,v1}_k1, {w2,t,v2}_k2, {s,x}_k1, {t,x}_k2, {v1,v2}|o, {v1,v1'}^, {v1',v2'}|o -> {w1,s,v1'}, {w2,t,v2'}",
      weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "a head-on pair whose hops meet turns by a quarter turn, either sense at random: the HPP collision" },
    { name: "Stream", notation: "{w,s,v}_k, {s,t}_k -> {w,t,v}", weights: UNIFORM, self_loops: "allow", enabled: false,
      description: "free flight: the ball takes the next hop of its velocity" },
  ];
  const STRING_FIELDS = ["walls", "placement", "velocity", "overlaps"];
  const SITE_NAMES = ["s", "t", "u", "x", "p", "q", "r"];
  const VELOCITY_NAMES = ["v", "v'", "v1", "v2", "v1'", "v2'"];
  const SESSION_FIELDS = ["size", "walls", "walkers", "placement", "steps", "seed", "velocity", "max_speed", "radius", "overlaps"];

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
  let customSites = null;    // ball sites chosen by clicking the universe canvas
  let customVelocities = null; // one (vx, vy) per custom site (from an imported session), or null
  let ballsJson = null;
  let ballPlayers = [];
  let lastResult = null;
  let applyTimer = null;

  function freshRule(spec) {
    return {
      id: nextId++,
      name: spec.name || "custom",
      notation: spec.notation || "{w,s}, {s,t} -> {w,t}, {s,t}",
      weights: (spec.weights || UNIFORM).map(Number),
      self_loops: spec.self_loops || "allow",
      types: spec.types || {},
      description: spec.description || "",
      enabled: spec.enabled !== undefined ? Boolean(spec.enabled) : true,
    };
  }

  function ruleSpec(rule) {
    return { name: rule.name, notation: rule.notation, weights: rule.weights, self_loops: rule.self_loops, types: rule.types,
             description: rule.description };
  }

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
        library: library, selectedId: selectedId, form: values, customSites: customSites, customVelocities: customVelocities,
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
      customVelocities = Array.isArray(saved.customVelocities) ? saved.customVelocities : null;
      return true;
    } catch (error) { return false; }
  }

  // ---------- notation <-> pattern ----------

  // "{w,s,v}_k, {s,t}_k -> {w,t,v}" -> {lhs: [{vars: [w,s,v], op: "_", arg: "k"}, ...], rhs: [...]};
  // null if unreadable. An annotation follows the closing brace: _k, =d, >d, |k, |o, ^.
  function parseNotation(text) {
    if (text.indexOf("->") < 0) return null;
    const halves = text.split("->");
    const sides = halves.map(function (half) {
      const edges = [];
      const re = /\{([^{}]*)\}\s*(?:([_=>|])\s*([^\s,{}]+)|(\^))?/g;
      let m;
      while ((m = re.exec(half)) !== null) {
        const vars = m[1].split(",").map(function (v) { return v.trim(); }).filter(Boolean);
        if (vars.length >= 2) edges.push({ vars: vars, op: m[4] ? "^" : (m[2] || ""), arg: m[3] || "" });
      }
      return edges;
    });
    if (!sides[0].length) return null;
    return { lhs: sides[0], rhs: sides[1] };
  }

  function edgeText(e) { return "{" + e.vars.join(",") + "}" + e.op + e.arg; }

  function formatSide(edges) {
    return edges.map(edgeText).join(", ");
  }

  function typeOf(name, types) {
    if (types && types[name]) return types[name];
    if (name.charAt(0) === "w") return "walker";
    if (name === "o" || name.charAt(0) === "v") return "velocity";
    return "site";
  }

  function patternFromRule(rule) {
    const parsed = parseNotation(rule.notation) || { lhs: [], rhs: [] };
    const variables = [];
    parsed.lhs.concat(parsed.rhs).forEach(function (edge) {
      edge.vars.forEach(function (v) {
        if (!variables.some(function (x) { return x.name === v; })) {
          variables.push({ name: v, type: typeOf(v, rule.types) });
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

  // A chain rule (the family of the comparison run): one ball {w,s} without velocity, drawn
  // site edges only, no annotations.
  function isChainRule(rule) {
    const parsed = parseNotation(rule.notation);
    if (!parsed) return false;
    const edges = parsed.lhs.concat(parsed.rhs);
    if (edges.some(function (e) { return e.op; })) return false;
    const names = {};
    edges.forEach(function (e) { e.vars.forEach(function (v) { names[v] = typeOf(v, rule.types); }); });
    if (Object.keys(names).some(function (v) { return names[v] === "velocity"; })) return false;
    const walkerEdges = parsed.lhs.filter(function (e) { return e.vars.some(function (v) { return names[v] === "walker"; }); });
    return walkerEdges.length === 1 && walkerEdges[0].vars.length === 2;
  }

  // Same layout as imigraph.report.svg.pattern_layout: sites in a row at the bottom, velocities
  // in a row in the middle, walkers above the first site they touch.
  function patternLayout(width, height) {
    const positions = {};
    const sites = pattern.variables.filter(function (v) { return v.type === "site"; });
    const velocities = pattern.variables.filter(function (v) { return v.type === "velocity"; });
    const walkers = pattern.variables.filter(function (v) { return v.type === "walker"; });
    const pad = 0.14 * width;
    const rows = velocities.length ? [0.8, 0.5, 0.2] : [0.72, 0.5, 0.26];
    sites.forEach(function (site, k) {
      const x = sites.length === 1 ? width / 2 : pad + k * (width - 2 * pad) / (sites.length - 1);
      positions[site.name] = [x, rows[0] * height];
    });
    velocities.forEach(function (velocity, k) {
      const x = velocities.length === 1 ? width / 2 : pad + k * (width - 2 * pad) / (velocities.length - 1);
      positions[velocity.name] = [x, rows[1] * height];
    });
    const used = {};
    walkers.forEach(function (walker) {
      let x = width / 2;
      const edges = pattern.lhs.concat(pattern.rhs);
      for (let e = 0; e < edges.length; e++) {
        if (edges[e].vars.indexOf(walker.name) < 0) continue;
        const other = edges[e].vars.find(function (v) { return v !== walker.name && positions[v]; });
        if (other) { x = positions[other][0]; break; }
      }
      const k = used[x] || 0;
      used[x] = k + 1;
      positions[walker.name] = [x + 28 * k, rows[2] * height];
    });
    return positions;
  }

  function svgElement(tag, attributes) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attributes).forEach(function (k) { element.setAttribute(k, attributes[k]); });
    return element;
  }

  function nodeColor(type) {
    return type === "walker" ? V.COLORS.walker : type === "velocity" ? V.COLORS.reference : V.COLORS.space;
  }

  function renderPattern() {
    if (!pattern) return;
    const positions = patternLayout(240, 170);
    ["lhs", "rhs"].forEach(function (side) {
      const svg = byId("pg-" + side);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const edges = pattern[side];
      edges.forEach(function (edge, index) {
        const kinds = edge.vars.map(typeOfName);
        const walkerEdge = kinds.indexOf("walker") >= 0;
        const velocityEdge = !walkerEdge && kinds.indexOf("velocity") >= 0;
        const color = walkerEdge ? V.COLORS.walkerEdge : velocityEdge ? "#93c5fd" : V.COLORS.space;
        const points = edge.vars.map(function (v) { return positions[v]; }).filter(Boolean);
        if (!points.length) return;
        let element;
        const label = edge.op ? edge.op + edge.arg : "";
        let labelAt;
        if (points.length === 1 || (points.length === 2 && points[0][0] === points[1][0] && points[0][1] === points[1][1])) {
          const a = points[0];
          element = svgElement("circle", { cx: a[0], cy: a[1] - 20, r: 12, fill: "none", stroke: color, "stroke-width": 3, class: "loop" });
          labelAt = [a[0] + 14, a[1] - 24];
        } else {
          element = svgElement("polyline", { points: points.map(function (p) { return p[0] + "," + p[1]; }).join(" "), fill: "none",
                                             stroke: color, "stroke-width": walkerEdge ? 4 : 3, class: "edge",
                                             "stroke-dasharray": velocityEdge ? "6 4" : "none" });
          const a = points[points.length - 2], b = points[points.length - 1];
          labelAt = [(a[0] + b[0]) / 2 + 5, (a[1] + b[1]) / 2 - 5];
        }
        element.appendChild(svgElement("title", {})).textContent = "remove edge " + edgeText(edge);
        element.addEventListener("click", function () { removeEdge(side, index); });
        svg.appendChild(element);
        if (label) {
          const text = svgElement("text", { x: labelAt[0], y: labelAt[1], class: "annotation" });
          text.textContent = label;
          svg.appendChild(text);
        }
      });
      pattern.variables.forEach(function (v) {
        const p = positions[v.name];
        const isPicked = picked && picked.side === side && picked.name === v.name;
        const node = svgElement("circle", {
          cx: p[0], cy: p[1], r: 13, fill: nodeColor(v.type), class: "node" + (isPicked ? " picked" : ""), "data-node": v.name,
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

  function typeOfName(name) {
    const v = pattern.variables.find(function (x) { return x.name === name; });
    return v ? v.type : typeOf(name, null);
  }

  function patternChanged() {
    patternToRule();
    renderPattern();
    renderLibrary();
    persist();
    scheduleApply();
  }

  function removeEdge(side, index) {
    pattern[side].splice(index, 1);
    picked = null;
    patternChanged();
  }

  function clickNode(side, name) {
    if (!picked || picked.side !== side) { picked = { side: side, name: name }; renderPattern(); return; }
    if (picked.name === name) { picked = null; renderPattern(); return; }
    const edges = pattern[side];
    const a = picked.name, b = name;
    const ta = typeOfName(a), tb = typeOfName(b);
    const existing = edges.findIndex(function (e) {
      return e.vars.length === 2 && ((e.vars[0] === a && e.vars[1] === b) || (e.vars[0] === b && e.vars[1] === a));
    });
    if (existing >= 0) {
      edges.splice(existing, 1);
    } else if ((ta === "walker") !== (tb === "walker") && (ta === "velocity" || tb === "velocity")) {
      // A walker and a velocity: extend the walker's hyperedge {w,s} to {w,s,v} (or replace v).
      const walker = ta === "walker" ? a : b, velocity = ta === "walker" ? b : a;
      const own = edges.find(function (e) { return e.vars[0] === walker; });
      if (own) { own.vars = [own.vars[0], own.vars[1], velocity]; }
      else { edges.push({ vars: [walker, velocity], op: "", arg: "" }); }
    } else {
      // A walker has exactly one hyperedge: connecting it to a site moves that hyperedge.
      [a, b].forEach(function (v) {
        if (typeOfName(v) !== "walker") return;
        for (let i = edges.length - 1; i >= 0; i--) if (edges[i].vars.indexOf(v) >= 0) edges.splice(i, 1);
      });
      const walkerFirst = tb === "walker" && ta !== "walker";
      edges.push({ vars: walkerFirst ? [b, a] : [a, b], op: "", arg: "" });
    }
    picked = null;
    patternChanged();
  }

  function freeName(type) {
    const taken = pattern.variables.map(function (v) { return v.name; });
    if (type === "walker") {
      for (let k = 1; k < 100; k++) { const n = k === 1 ? "w" : "w" + k; if (taken.indexOf(n) < 0) return n; }
    }
    const names = type === "velocity" ? VELOCITY_NAMES : SITE_NAMES;
    for (let k = 0; k < names.length; k++) if (taken.indexOf(names[k]) < 0) return names[k];
    const prefix = type === "velocity" ? "v" : "s";
    for (let k = 2; k < 100; k++) if (taken.indexOf(prefix + k) < 0) return prefix + k;
    return "x" + Date.now();
  }

  function addNode(type) {
    if (!pattern) return;
    const name = freeName(type);
    pattern.variables.push({ name: name, type: type });
    picked = { side: "lhs", name: name };
    patternToRule();
    renderPattern();
    byId("pg-editor-hint").textContent = "added " + type + " " + name + " to both sides; click a node on the left to connect it";
  }

  function removePickedNode() {
    if (!picked) return;
    const name = picked.name;
    pattern.variables = pattern.variables.filter(function (v) { return v.name !== name; });
    ["lhs", "rhs"].forEach(function (side) {
      pattern[side] = pattern[side].filter(function (e) { return e.vars.indexOf(name) < 0; });
    });
    picked = null;
    patternChanged();
  }

  function renameNode(oldName) {
    const newName = (window.prompt("New name of " + oldName + " (same name on both sides):", oldName) || "").trim();
    if (!newName || newName === oldName || /[{},\s]/.test(newName)) return;
    if (pattern.variables.some(function (v) { return v.name === newName; })) return;
    pattern.variables.forEach(function (v) { if (v.name === oldName) v.name = newName; });
    ["lhs", "rhs"].forEach(function (side) {
      pattern[side].forEach(function (e) { e.vars = e.vars.map(function (v) { return v === oldName ? newName : v; }); });
    });
    picked = null;
    patternChanged();
  }

  // ---------- rule list and text editor ----------

  function renderLibrary() {
    const esc = function (s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); };
    const body = byId("pg-library").querySelector("tbody");
    body.innerHTML = library.map(function (rule) {
      const valid = parseNotation(rule.notation) !== null;
      const weights = rule.weights.map(function (w) { return Number(w).toFixed(2); }).join(" ") +
        (rule.self_loops === "forbid" ? ", no stay" : "");
      return '<tr data-id="' + rule.id + '" data-name="' + esc(rule.name) + '" class="' +
        (rule.id === selectedId ? "selected" : "") + (valid ? "" : " invalid") + '">' +
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
    pattern = patternFromRule(rule);
    renderPattern();
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
    rule.notation = byId("pg-notation").value.trim();
    rule.weights = [0, 1, 2, 3].map(function (k) { return Number(byId("pg-w" + k).value); });
    rule.self_loops = byId("pg-loops").value;
    if (parseNotation(rule.notation)) { pattern = patternFromRule(rule); picked = null; renderPattern(); }
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
    const universe = lastResult && lastResult.universe;
    const velocities = universe && universe.velocities && universe.sites.length === sites.length && universe.size === size
      ? universe.velocities : null;
    const radius = universe && universe.size === size ? universe.radius : parseInt(form.radius.value, 10) || 0;
    V.drawAgents(vp, sites.map(function (s) { return geometry.coordinates[s]; }),
                 { radius: radius + 0.3, fanOut: true, velocities: velocities });
    byId("pg-universe-label").textContent = sites.length + " ball" + (sites.length === 1 ? "" : "s") + " on a " + size + "×" + size +
      " world (" + form.walls.value + " walls, velocities: " + form.velocity.value + ")";
    canvas.onclick = function (event) {
      const point = vp.pointOf(event);
      const i = Math.round(point[0]), j = Math.round(point[1]);
      if (i < 0 || j < 0 || i >= size || j >= size) return;
      const site = i * size + j;
      if (customSites === null || byId("playground").elements.placement.value !== "sites") customSites = shownSites().slice();
      const at = customSites.indexOf(site);
      if (at >= 0) customSites.splice(at, 1); else customSites.push(site);
      customVelocities = null;  // velocities of clicked sites follow the "velocities at the start" choice
      byId("playground").elements.placement.value = "sites";
      syncPlacement();
      drawUniverse();
      persist();
      scheduleApply();
    };
  }

  function syncPlacement() {
    const form = byId("playground");
    const placement = form.elements.placement.value;
    const custom = placement === "sites";
    form.elements.walkers.disabled = custom || placement === "head-on" || placement === "diagonal";
    if (custom) form.elements.walkers.value = (customSites || []).length || form.elements.walkers.value;
    if (placement === "head-on" || placement === "diagonal") form.elements.walkers.value = 2;
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
      if (customVelocities && customVelocities.length === request.sites.length) request.velocities = customVelocities;
    }
    return request;
  }

  function sessionRequest() {
    const request = playgroundRequest();
    request.library = library.map(function (r) { return Object.assign(ruleSpec(r), { enabled: r.enabled }); });
    return request;
  }

  function fmt(x) { return Number(x).toFixed(3); }

  function markedBalls(events, f) {
    const marked = {};
    if (events && f > 0) events[f - 1].forEach(function (e) { e.balls.forEach(function (b) { marked[b] = true; }); });
    return marked;
  }

  function renderPlayground(result) {
    playgroundPlayers.forEach(function (p) { p.pause(); });
    playgroundPlayers = [];
    byId("pg-error").textContent = result.error || "";
    byId("pg-save").disabled = Boolean(result.error);
    if (result.error) { byId("pg-physics").textContent = ""; drawUniverse(); return; }
    lastResult = result;
    const demo = result.demo;
    const radius = (demo.radius + 0.3) * demo.space.spacing;
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
    const projection = V.Viewport(byId("pg-projection-canvas"), demo.space.box, 24);
    playgroundPlayers.push(V.Player(demo.times.length, function (f) {
      const positions = demo.anchors[f].map(function (s) { return demo.space.coordinates[s]; });
      const marked = markedBalls(demo.events, f);
      const velocities = demo.velocities ? demo.velocities[f] : null;
      world.clear();
      V.drawSpace(world, demo.space.coordinates, demo.space.edges, 4, 1.5 * demo.space.spacing);
      V.drawAgents(world, positions, { radius: radius, fanOut: true, marked: marked, velocities: velocities });
      projection.clear();
      V.drawAgents(projection, positions, { radius: (demo.radius + 0.5) * demo.space.spacing, alpha: 0.85, marked: marked,
                                            velocities: velocities });
      byId("pg-demo-caption").textContent = result.captions && result.captions.length ? result.captions[f] : "";
    }, { slider: byId("pg-demo-slider"), button: byId("pg-demo-play"), label: byId("pg-demo-label"),
         labelFor: function (f) { return "step " + f; }, interval: 500 }));

    const m = result.measured, p = result.predicted, model = result.model;
    const byRule = Object.keys(m.collisions_by_rule || {}).map(function (k) { return m.collisions_by_rule[k] + " by " + escapeHtml(k); }).join(", ");
    const modelRows = model.moving
      ? "<tr><th>collisions in the run</th><td>" + m.collisions + (byRule ? " (" + byRule + ")" : "") + "</td></tr>" +
        "<tr><th>wall bounces in the run</th><td>" + m.bounces + "</td></tr>" +
        "<tr><th>total momentum at the start / end (m a/&tau;), steps where it changed (wall bounces)</th><td>(" +
        m.momentum_start.join(", ") + ") / (" + m.momentum_end.join(", ") + "), " + m.momentum_changes + "</td></tr>" +
        "<tr><th>kinetic energy (m a&sup2;/&tau;&sup2;)</th><td>" + m.energy + (m.energy_constant ? ", constant" : ", changed during the run") + "</td></tr>"
      : "";
    byId("pg-measured").innerHTML =
      '<table class="kv"><tbody>' +
      "<tr><th>velocities / max speed / radius / overlapping matches</th><td>" + escapeHtml(model.velocity) + " / " +
        model.max_speed + " / " + model.radius + " / " + escapeHtml(model.overlaps) + "</td></tr>" +
      "<tr><th>waiting hops (fraction of walkers per step, overlapping matches)</th><td>" + fmt(m.waiting_rate) + "</td></tr>" +
      "<tr><th>mean |&Delta;r|&sup2; per step (a&sup2;)</th><td>" + fmt(m.mean_squared_step) + "</td></tr>" +
      "<tr><th>diffusion D measured / predicted (x, y)</th><td>" + fmt(m.diffusion) + " / " +
        fmt(p.diffusion[0]) + ", " + fmt(p.diffusion[1]) + "</td></tr>" +
      "<tr><th>drift v measured / predicted</th><td>(" + fmt(m.drift[0]) + ", " + fmt(m.drift[1]) + ") / (" +
        fmt(p.drift[0]) + ", " + fmt(p.drift[1]) + ")</td></tr>" +
      "<tr><th>chain hops per step (predicted)</th><td>" + p.hops + "</td></tr>" +
      "<tr><th>wall rate (fraction of walkers per step)</th><td>" + fmt(m.wall_rate) + "</td></tr>" +
      "<tr><th>occupied sites at the end</th><td>" + m.occupied_final + "</td></tr>" + modelRows +
      "</tbody></table>" +
      '<p class="muted">Few walkers and walls make the measured values noisy; raise the world size and the walkers to see them converge. ' +
      "Moving balls have no diffusion prediction; their momentum and energy are measured instead.</p>";
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

  // ---------- the ball gas against the hard-disk gas ----------

  function ballsRequest() {
    const request = {};
    for (const [key, value] of new FormData(byId("balls")).entries()) {
      request[key] = (key === "rule" || key === "placement") ? value : parseInt(value, 10);
    }
    if (request.rule === "custom") request.rules = enabledRules().map(ruleSpec);
    return request;
  }

  function renderBalls(result) {
    ballPlayers.forEach(function (p) { p.pause(); });
    ballPlayers = [];
    byId("bg-error").textContent = result.error || "";
    if (result.error) return;
    const g = result.data.graph, r = result.data.reference, s = result.summary;
    const graph = V.Viewport(byId("bg-graph-canvas"), g.space.box, 20);
    const projection = V.Viewport(byId("bg-projection-canvas"), g.space.box, 20);
    const reference = V.Viewport(byId("bg-reference-canvas"), r.box, 20);
    const diskRadius = r.radius;
    ballPlayers.push(V.Player(g.times.length, function (f) {
      graph.clear();
      V.drawSpace(graph, g.space.coordinates, g.space.edges, 3, 1.5 * g.space.spacing);
      const ctx = graph.ctx;
      const collided = {};
      if (f > 0) g.events[f - 1].forEach(function (e) { if (e.kind === "collision") e.balls.forEach(function (b) { collided[b] = true; }); });
      const nodes = [], owners = [];
      g.sites[f].forEach(function (sites, b) { sites.forEach(function (site) { nodes.push(g.space.coordinates[site]); owners.push(b); }); });
      V.drawAgents(graph, nodes, { radius: 0.3 * g.space.spacing, indices: owners, numbers: g.sites[f][0].length === 1 });
      if (g.sites[f][0].length > 1) {
        ctx.fillStyle = "#111827"; ctx.font = "bold 12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        g.centres[f].forEach(function (c, b) { ctx.fillText(String(b), graph.x(c[0]), graph.y(c[1])); });
      }
      V.drawAgents(graph, g.centres[f], { radius: (g.radius_cells + 0.3) * g.space.spacing, alpha: 0, numbers: false,
                                          velocities: g.velocities[f], marked: collided, arrowLength: 0.8 * g.space.spacing });
      projection.clear();
      V.drawAgents(projection, g.centres[f], { radius: diskRadius, velocities: g.velocities[f], marked: collided, arrowLength: 0.5 * g.space.spacing });
      reference.clear();
      V.drawAgents(reference, r.positions[f], { radius: diskRadius, velocities: r.velocities[f], arrowLength: 0.5 * g.space.spacing });
      byId("bg-caption").textContent = g.captions.length ? g.captions[f] : "";
    }, { slider: byId("bg-slider"), button: byId("bg-play"), label: byId("bg-label"),
         labelFor: function (f) { return "step " + f + " (t = " + Number(g.times[f]).toFixed(3) + ")"; }, interval: 350 }));
    const verdict = byId("bg-verdict");
    verdict.textContent = s.passed ? "PASS" : "FAIL";
    verdict.className = "badge " + (s.passed ? "pass" : "fail");
    const byRule = Object.keys(s.collisions_by_rule || {}).map(function (k) { return s.collisions_by_rule[k] + " by " + k; }).join(", ");
    byId("bg-summary").textContent = s.balls + " ball(s) of " + s.walkers_per_ball + " node(s), rules " + s.rules + ": " +
      s.collisions + " collisions on the graph" + (byRule ? " (" + byRule + ")" : "") + ", " + s.bounces + " wall bounces; " +
      s.reference_collisions + " collisions in the reference. Graph checks " + (s.graph_passed ? "pass" : "fail") +
      ", reference checks " + (s.reference_passed ? "pass" : "fail") + ".";
    const fr = result.fragments;
    byId("bg-invariants").innerHTML = fr.graph_invariants_table;
    byId("bg-graph-checks").innerHTML = fr.graph_checks_table;
    byId("bg-reference-checks").innerHTML = fr.reference_checks_table;
    byId("bg-comparison").innerHTML = fr.comparison_table;
    byId("bg-energy-chart").innerHTML = fr.energy_chart;
    byId("bg-momentum-chart").innerHTML = fr.momentum_chart;
    byId("bg-entropy-chart").innerHTML = fr.entropy_chart;
    byId("bg-collisions-chart").innerHTML = fr.collisions_chart;
    byId("bg-speed-chart").innerHTML = fr.speed_chart;
    byId("bg-occupancy").innerHTML = fr.occupancy_charts;
    byId("bg-rules").innerHTML = rulesTable(result.rules);
    byId("bg-docs").innerHTML = result.docs;
    V.speedControl(byId("bg-speed"), byId("bg-speed-label"), ballPlayers);
    ballPlayers.forEach(function (p) { p.play(); });
  }

  function runBalls() {
    if (!ballsJson) return;
    byId("bg-status").textContent = "running the balls and the reference...";
    setTimeout(function () {
      try {
        const started = performance.now();
        const result = JSON.parse(ballsJson(JSON.stringify(ballsRequest())));
        renderBalls(result);
        byId("bg-status").textContent = (result.error ? "not run" : "done") + " in " + ((performance.now() - started) / 1000).toFixed(2) + " s";
      } catch (error) {
        byId("bg-error").textContent = "error: " + error;
        byId("bg-status").textContent = "failed";
        console.error(error);
      }
    }, 20);
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
      SESSION_FIELDS.forEach(function (key) {
        if (data[key] !== undefined && data[key] !== null && form.elements[key]) form.elements[key].value = data[key];
      });
      customSites = Array.isArray(data.sites) ? data.sites.map(Number) : null;
      customVelocities = Array.isArray(data.velocities) ? data.velocities : null;
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
    ballsJson = live.balls_json;
    byId("bg-run").disabled = false;
    status("ready (" + wheel + ")");
    byId("load-status").textContent = "Python is ready.";
    byId("run").textContent = "Run";
    byId("run").disabled = false;
    byId("pg-apply").disabled = false;
    applyPlayground();
    setTimeout(runBalls, 60);
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
  byId("pg-add-velocity").addEventListener("click", function () { addNode("velocity"); });
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
    customVelocities = null;
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
  // The comparison run imitates Brownian motion with balls at rest: only the chain rules go in.
  function comparisonRules() {
    return enabledRules().filter(isChainRule).map(ruleSpec);
  }
  byId("pg-use").addEventListener("click", function () {
    const rules = comparisonRules();
    if (!rules.length) { status("the comparison run needs at least one chain rule (ball rules with velocities are left out)"); return; }
    byId("run-rule-json").value = JSON.stringify(rules);
    byId("run-rule").value = "custom";
    status("the comparison run will use the rule set " + rules.map(function (r) { return r.name; }).join(" + ") +
      (rules.length < enabledRules().length ? " (ball rules left out)" : ""));
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
  byId("bg-run").addEventListener("click", runBalls);
  Array.prototype.forEach.call(byId("balls").elements, function (element) {
    element.addEventListener("change", runBalls);
  });
  loadPython().catch(function (error) {
    status("failed to load Python: " + error);
    console.error(error);
  });
})();
