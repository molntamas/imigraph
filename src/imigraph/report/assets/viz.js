// imigraph visualisation helpers: canvas drawing of trajectories and graph states, plus a
// frame player. Inlined into the generated HTML report and served to the Pyodide GUI.
(function (global) {
  "use strict";

  const COLORS = {
    reference: "#2563eb",
    candidate: "#dc2626",
    space: "#9ca3af",
    edge: "#d1d5db",
    walker: "#dc2626",
    walkerEdge: "#f87171",
    box: "#374151",
  };

  // Maps box coordinates to canvas pixels (y grows upward in the box, downward on canvas).
  function Viewport(canvas, box, margin) {
    margin = margin === undefined ? 10 : margin;
    const ctx = canvas.getContext("2d");
    const span = [box.upper[0] - box.lower[0], box.upper[1] - box.lower[1]];
    const scale = Math.min(
      (canvas.width - 2 * margin) / span[0],
      (canvas.height - 2 * margin) / span[1]
    );
    const offset = [
      (canvas.width - scale * span[0]) / 2,
      (canvas.height - scale * span[1]) / 2,
    ];
    return {
      ctx: ctx,
      scale: scale,
      x: function (v) { return offset[0] + (v - box.lower[0]) * scale; },
      y: function (v) { return canvas.height - offset[1] - (v - box.lower[1]) * scale; },
      // Inverse maps (canvas pixels to box coordinates), for clicks on the canvas.
      invX: function (px) { return box.lower[0] + (px - offset[0]) / scale; },
      invY: function (py) { return box.lower[1] + (canvas.height - offset[1] - py) / scale; },
      // Box coordinates of a mouse event on this canvas (CSS size may differ from the pixel size).
      pointOf: function (event) {
        const rect = canvas.getBoundingClientRect();
        const px = (event.clientX - rect.left) * canvas.width / rect.width;
        const py = (event.clientY - rect.top) * canvas.height / rect.height;
        return [box.lower[0] + (px - offset[0]) / scale, box.lower[1] + (canvas.height - offset[1] - py) / scale];
      },
      clear: function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = COLORS.box;
        ctx.lineWidth = 1;
        ctx.strokeRect(offset[0], offset[1], scale * span[0], scale * span[1]);
      },
    };
  }

  function drawParticles(vp, positions, color, radius) {
    const ctx = vp.ctx;
    ctx.fillStyle = color;
    for (let i = 0; i < positions.length; i++) {
      ctx.beginPath();
      ctx.arc(vp.x(positions[i][0]), vp.y(positions[i][1]), radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // `maxLength` (box units, optional) hides the wrap-around edges of a periodic lattice.
  function drawSpace(vp, coordinates, edges, nodeRadius, maxLength) {
    const ctx = vp.ctx;
    ctx.strokeStyle = COLORS.edge;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let e = 0; e < edges.length; e++) {
      const a = coordinates[edges[e][0]];
      const b = coordinates[edges[e][1]];
      if (maxLength && Math.hypot(a[0] - b[0], a[1] - b[1]) > maxLength) continue;
      ctx.moveTo(vp.x(a[0]), vp.y(a[1]));
      ctx.lineTo(vp.x(b[0]), vp.y(b[1]));
    }
    ctx.stroke();
    drawParticles(vp, coordinates, COLORS.space, nodeRadius);
  }

  // Lattice-gas walkers: an arrow from the site in the walker's direction (its heading edge).
  function drawArrows(vp, coordinates, sites, directions, spacing, color) {
    const ctx = vp.ctx;
    const len = 0.42 * spacing;
    const head = Math.max(4, 0.12 * spacing * vp.scale);
    ctx.strokeStyle = color || COLORS.walker;
    ctx.fillStyle = color || COLORS.walker;
    ctx.lineWidth = 2;
    for (let w = 0; w < sites.length; w++) {
      const a = coordinates[sites[w]];
      const d = LOOP_DIRECTIONS[directions[w]];
      const x0 = vp.x(a[0]), y0 = vp.y(a[1]);
      const x1 = vp.x(a[0] + len * d[0]), y1 = vp.y(a[1] + len * d[1]);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      const angle = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(angle - 0.5), y1 - head * Math.sin(angle - 0.5));
      ctx.lineTo(x1 - head * Math.cos(angle + 0.5), y1 - head * Math.sin(angle + 0.5));
      ctx.closePath();
      ctx.fill();
    }
  }

  // Rings around the sites where a Collide rule fired in this step.
  function drawCollisionMarks(vp, coordinates, sites, spacing) {
    const ctx = vp.ctx;
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 3;
    for (let i = 0; i < sites.length; i++) {
      const a = coordinates[sites[i]];
      ctx.beginPath();
      ctx.arc(vp.x(a[0]), vp.y(a[1]), 0.3 * spacing * vp.scale, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  // Walker nodes are drawn next to their anchor node, joined by their edge; several walkers on
  // one node fan out around it.
  function drawWalkers(vp, coordinates, anchors, spacing) {
    const ctx = vp.ctx;
    const seen = {};
    const r = 0.32 * spacing;
    for (let w = 0; w < anchors.length; w++) {
      const node = anchors[w];
      const k = seen[node] || 0;
      seen[node] = k + 1;
      const angle = Math.PI / 4 + k * (2 * Math.PI / 5);
      const a = coordinates[node];
      const px = vp.x(a[0] + r * Math.cos(angle));
      const py = vp.y(a[1] + r * Math.sin(angle));
      ctx.strokeStyle = COLORS.walkerEdge;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(vp.x(a[0]), vp.y(a[1]));
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.fillStyle = COLORS.walker;
      ctx.beginPath();
      ctx.arc(px, py, Math.max(3, 0.14 * spacing * vp.scale), 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // Neighbour-table order of SquareLattice: +x, -x, +y, -y. A self-loop replaces a missing
  // neighbour and is drawn as a small loop pointing out of the lattice in that direction.
  const LOOP_DIRECTIONS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

  // Agents (walkers, balls, disks) are drawn the same way in every view: a disc in the agent's
  // own colour with its number, an arrow for its heading (a label 0..3) or velocity, a ring when
  // it is marked (a collision in this step). Agents on one spot fan out when asked to.
  const AGENT_COLORS = ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2",
                        "#be185d", "#4d7c0f", "#b45309", "#1e40af", "#0f766e", "#9f1239"];
  function agentColor(i) { return AGENT_COLORS[i % AGENT_COLORS.length]; }

  function drawAgents(vp, positions, options) {
    const o = options || {};
    const ctx = vp.ctx;
    const r = o.radius || 0.3;
    const seen = {};
    for (let i = 0; i < positions.length; i++) {
      let x = positions[i][0], y = positions[i][1];
      const key = x.toFixed(6) + "," + y.toFixed(6);
      const k = seen[key] || 0;
      seen[key] = k + 1;
      if (k > 0 && o.fanOut) {
        const angle = Math.PI / 4 + k * (2 * Math.PI / 5);
        x += 0.8 * r * Math.cos(angle);
        y += 0.8 * r * Math.sin(angle);
      }
      const index = o.indices ? o.indices[i] : i;
      const px = vp.x(x), py = vp.y(y), pr = Math.max(3, r * vp.scale);
      ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha;
      ctx.fillStyle = o.color || agentColor(index);
      ctx.beginPath();
      ctx.arc(px, py, pr, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (o.numbers !== false && positions.length <= 40 && pr >= 7) {
        ctx.fillStyle = "white";
        ctx.font = "bold " + Math.min(14, pr) + "px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(index), px, py);
      }
      let d = null;
      if (o.directions) d = LOOP_DIRECTIONS[o.directions[i]];
      else if (o.velocities) {
        const v = o.velocities[i], s = Math.hypot(v[0], v[1]);
        if (s > 0) d = [v[0] / s, v[1] / s];
      }
      if (d) {
        const len = o.arrowLength || 1.6 * r;
        const x0 = vp.x(x + r * d[0]), y0 = vp.y(y + r * d[1]);
        const x1 = vp.x(x + (r + len) * d[0]), y1 = vp.y(y + (r + len) * d[1]);
        const head = Math.max(4, 0.35 * pr);
        ctx.strokeStyle = "#374151";
        ctx.fillStyle = "#374151";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        const angle = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - head * Math.cos(angle - 0.5), y1 - head * Math.sin(angle - 0.5));
        ctx.lineTo(x1 - head * Math.cos(angle + 0.5), y1 - head * Math.sin(angle + 0.5));
        ctx.closePath(); ctx.fill();
      }
      if (o.marked && o.marked[i]) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(px, py, pr * 1.7, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
  }

  // Draws the edge from site `from` to site `to` (a loop drawn beside the site when equal).
  function strokeEdge(vp, coords, neighbors, spacing, from, to, color, width) {
    const ctx = vp.ctx;
    const s = coords[from];
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    if (from === to) {
      const k = Math.max(0, neighbors[from].indexOf(from));
      const d = LOOP_DIRECTIONS[k];
      ctx.arc(vp.x(s[0] + 0.28 * spacing * d[0]), vp.y(s[1] + 0.28 * spacing * d[1]),
              0.18 * spacing * vp.scale, 0, 2 * Math.PI);
    } else {
      const t = coords[to];
      ctx.moveTo(vp.x(s[0]), vp.y(s[1]));
      ctx.lineTo(vp.x(t[0]), vp.y(t[1]));
    }
    ctx.stroke();
  }

  // One application of a chain rule on a small space, in four phases: 0 state, 1 match (every
  // labelled edge at the anchor is a candidate for the first chain edge, yellow), 2 choose (the
  // matched chain, orange), 3 rewrite (the walker edge now ends at the chain's target site).
  // `path` is the matched chain [s0, s1, ..., sk]; `target` the walker's new site.
  // `space` needs coordinates, edges, neighbors, spacing.
  function drawRuleStep(vp, space, path, target, phase) {
    const coords = space.coordinates;
    const spacing = space.spacing;
    const nbrs = space.neighbors;
    drawSpace(vp, coords, space.edges, Math.max(4, 0.1 * spacing * vp.scale), 1.5 * spacing);
    const anchor = path[0];
    if (phase === 1) {
      for (let k = 0; k < nbrs[anchor].length; k++) {
        strokeEdge(vp, coords, nbrs, spacing, anchor, nbrs[anchor][k], "#fbbf24", 4);
        if (nbrs[anchor][k] === anchor) {  // loops: draw at their own direction
          const d = LOOP_DIRECTIONS[k], s = coords[anchor], ctx = vp.ctx;
          ctx.beginPath();
          ctx.arc(vp.x(s[0] + 0.28 * spacing * d[0]), vp.y(s[1] + 0.28 * spacing * d[1]),
                  0.18 * spacing * vp.scale, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    } else if (phase === 2) {
      for (let i = 1; i < path.length; i++) {
        strokeEdge(vp, coords, nbrs, spacing, path[i - 1], path[i], "#f59e0b", 5);
      }
    }
    drawWalkers(vp, coords, [phase === 3 ? target : anchor], spacing);
  }

  const RULE_PHASES = [
    "state: walker w (red) is anchored at space node s by one edge",
    "match: every labelled edge at s is a candidate for the first chain edge (yellow); a self-loop counts",
    "choose: the chain is drawn with the rule's direction weights (orange)",
    "rewrite: {w,s} is replaced by the walker edge to the chain's target site; the space edges stay",
  ];

  // Plays frames 0..count-1, calling draw(frame) and keeping slider/label in sync.
  function Player(count, draw, controls) {
    let frame = 0;
    let timer = null;
    let interval = controls.interval || 80;
    function show(f) {
      frame = ((f % count) + count) % count;
      if (controls.slider) controls.slider.value = frame;
      if (controls.label) controls.label.textContent = controls.labelFor ? controls.labelFor(frame) : String(frame);
      draw(frame);
    }
    function play() {
      if (timer) return;
      timer = setInterval(function () { show(frame + 1); }, interval);
      if (controls.button) controls.button.textContent = "Pause";
    }
    function pause() {
      if (timer) clearInterval(timer);
      timer = null;
      if (controls.button) controls.button.textContent = "Play";
    }
    // Playback speed: `factor` scales the player's own interval (1 = as authored, 2 = twice as fast).
    function setSpeed(factor) {
      interval = Math.max(10, (controls.interval || 80) / factor);
      if (timer) { pause(); play(); }
    }
    if (controls.slider) {
      controls.slider.min = 0;
      controls.slider.max = count - 1;
      controls.slider.addEventListener("input", function () { pause(); show(Number(controls.slider.value)); });
    }
    if (controls.button) {
      controls.button.addEventListener("click", function () { if (timer) pause(); else play(); });
    }
    show(0);
    return { play: play, pause: pause, show: show, setSpeed: setSpeed };
  }

  // Wires a range input (value = speed factor) to a list of players.
  function speedControl(input, label, players) {
    function apply() {
      const factor = Number(input.value);
      if (label) label.textContent = factor + "x";
      players.forEach(function (p) { p.setSpeed(factor); });
    }
    input.addEventListener("input", apply);
    apply();
  }

  global.imigraphViz = {
    COLORS: COLORS,
    LOOP_DIRECTIONS: LOOP_DIRECTIONS,
    Viewport: Viewport,
    drawParticles: drawParticles,
    drawSpace: drawSpace,
    drawWalkers: drawWalkers,
    drawArrows: drawArrows,
    drawCollisionMarks: drawCollisionMarks,
    drawRuleStep: drawRuleStep,
    RULE_PHASES: RULE_PHASES,
    drawAgents: drawAgents,
    agentColor: agentColor,
    Player: Player,
    speedControl: speedControl,
  };
})(typeof window !== "undefined" ? window : globalThis);
