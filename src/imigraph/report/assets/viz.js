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

  function drawSpace(vp, coordinates, edges, nodeRadius) {
    const ctx = vp.ctx;
    ctx.strokeStyle = COLORS.edge;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let e = 0; e < edges.length; e++) {
      const a = coordinates[edges[e][0]];
      const b = coordinates[edges[e][1]];
      ctx.moveTo(vp.x(a[0]), vp.y(a[1]));
      ctx.lineTo(vp.x(b[0]), vp.y(b[1]));
    }
    ctx.stroke();
    drawParticles(vp, coordinates, COLORS.space, nodeRadius);
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

  // One application of the RandomHop rule on a small space, in four phases:
  // 0 state, 1 match (all edges {s,s'} at the anchor), 2 choose (one of them), 3 rewrite
  // (the walker edge now ends at s'). `space` needs coordinates, edges, neighbors, spacing.
  function drawRuleStep(vp, space, anchor, next, phase) {
    const ctx = vp.ctx;
    const coords = space.coordinates;
    const spacing = space.spacing;
    drawSpace(vp, coords, space.edges, Math.max(4, 0.1 * spacing * vp.scale));
    const s = coords[anchor];
    const nbrs = space.neighbors[anchor];
    if (phase === 1 || phase === 2) {
      for (let k = 0; k < nbrs.length; k++) {
        const chosen = nbrs[k] === next && (phase === 2);
        if (phase === 2 && !chosen) continue;
        ctx.strokeStyle = chosen ? "#f59e0b" : "#fbbf24";
        ctx.lineWidth = chosen ? 5 : 4;
        ctx.beginPath();
        if (nbrs[k] === anchor) {
          const d = LOOP_DIRECTIONS[k];
          const cx = vp.x(s[0] + 0.28 * spacing * d[0]);
          const cy = vp.y(s[1] + 0.28 * spacing * d[1]);
          ctx.arc(cx, cy, 0.18 * spacing * vp.scale, 0, 2 * Math.PI);
        } else {
          const t = coords[nbrs[k]];
          ctx.moveTo(vp.x(s[0]), vp.y(s[1]));
          ctx.lineTo(vp.x(t[0]), vp.y(t[1]));
        }
        ctx.stroke();
        if (phase === 2 && chosen) break;
      }
    }
    drawWalkers(vp, coords, [phase === 3 ? next : anchor], spacing);
  }

  const RULE_PHASES = [
    "state: walker w (red) is anchored at space node s by one edge",
    "match: every edge {s,s'} at s matches the pattern {w,s},{s,s'} (yellow); a self-loop counts",
    "choose: one match is picked uniformly at random (orange)",
    "rewrite: {w,s} is replaced by {w,s'}; the space edge stays",
  ];

  // Plays frames 0..count-1, calling draw(frame) and keeping slider/label in sync.
  function Player(count, draw, controls) {
    let frame = 0;
    let timer = null;
    const interval = controls.interval || 80;
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
    if (controls.slider) {
      controls.slider.min = 0;
      controls.slider.max = count - 1;
      controls.slider.addEventListener("input", function () { pause(); show(Number(controls.slider.value)); });
    }
    if (controls.button) {
      controls.button.addEventListener("click", function () { if (timer) pause(); else play(); });
    }
    show(0);
    return { play: play, pause: pause, show: show };
  }

  global.imigraphViz = {
    COLORS: COLORS,
    Viewport: Viewport,
    drawParticles: drawParticles,
    drawSpace: drawSpace,
    drawWalkers: drawWalkers,
    drawRuleStep: drawRuleStep,
    RULE_PHASES: RULE_PHASES,
    Player: Player,
  };
})(typeof window !== "undefined" ? window : globalThis);
