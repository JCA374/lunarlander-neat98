// The model tab: draws the evolved network (data/network.json, built by lander/netviz.py every publish).
(async function () {
  const $ = (id) => document.getElementById(id);
  const N = await get('network.json');
  const ST = await get('stats.json');
  if (!N || !N.snapshots || !N.snapshots.length) { $('netstatus').textContent = 'No champion network saved yet.'; return; }
  const snaps = N.snapshots, last = snaps[snaps.length - 1];
  $('netstatus').textContent = `Run ${N.run} · ${snaps.length} saved champions · newest: generation ${last.gen} (${last.stage}) · ` +
    `${last.hidden} hidden nodes · ${last.n_conns} connections.`;

  const NS = 'http://www.w3.org/2000/svg';
  const X0 = 160, X1 = 790, Y0 = 40, Y1 = 545;
  const px = (x) => X0 + x * (X1 - X0), py = (y) => Y0 + y * (Y1 - Y0);
  const inKeys = N.inputs.map((_, i) => -(i + 1));
  const name = (k) => k < 0 ? N.inputs[-k - 1] : k < 4 ? N.outputs[k] : `#${k}`;
  const fmt = (v, d = 2) => (v >= 0 ? '+' : '') + v.toFixed(d);
  const mixc = (v, lim = 1) => {           // diverging: blue (+) / orange (−), pale near zero
    const p = Math.min(100, Math.round(100 * Math.min(1, Math.abs(v) / lim)));
    return `color-mix(in srgb, var(${v >= 0 ? '--pos' : '--neg'}) ${p}%, var(--panel))`;
  };
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };

  // ---- one network drawing: built once, then update() recolours it (fast enough to animate) ----
  function drawNet(svg, snap, tipEl) {
    svg.innerHTML = '';
    svg.classList.remove('focus');
    const nodes = new Map();
    inKeys.forEach((k, i) => nodes.set(k, { k, x: 0, y: (i + 0.5) / inKeys.length, kind: 'in', born: 0, bias: 0 }));
    for (const [k, x, y, bias, born] of snap.nodes) nodes.set(k, { k, x, y, bias, born, kind: k < 4 ? 'out' : 'hid' });
    const conns = snap.conns.filter(([a, b]) => nodes.has(a) && nodes.has(b)).map(([a, b, w]) => ({ a, b, w }));
    const gC = el('g', {}, svg), gN = el('g', {}, svg), gL = el('g', {}, svg);
    el('text', { x: px(0), y: 22, class: 'colhead', 'text-anchor': 'middle' }, gL).textContent = 'inputs';
    el('text', { x: px(0.5), y: 22, class: 'colhead', 'text-anchor': 'middle' }, gL).textContent = 'hidden nodes';
    el('text', { x: px(1), y: 22, class: 'colhead', 'text-anchor': 'middle' }, gL).textContent = 'outputs';
    for (const c of conns) {
      const A = nodes.get(c.a), B = nodes.get(c.b);
      const x1 = px(A.x), y1 = py(A.y), x2 = px(B.x), y2 = py(B.y), dx = (x2 - x1) * 0.45;
      c.el = el('path', { d: `M${x1} ${y1} C${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`, class: 'c' }, gC);
    }
    for (const n of nodes.values()) {
      n.el = el('circle', { cx: px(n.x), cy: py(n.y), r: n.kind === 'hid' ? 6.5 : 8,
                            class: 'node' + (n.born > 0 ? ' born' : '') }, gN);
      if (n.kind !== 'hid') {
        n.lab = el('text', { x: n.kind === 'in' ? px(0) - 14 : px(1) + 14, y: py(n.y) + 4.5, class: 'lbl',
                             'text-anchor': n.kind === 'in' ? 'end' : 'start' }, gL);
      }
      const show = (ev) => {
        svg.classList.add('focus');
        for (const c of conns) c.el.classList.toggle('hl', c.a === n.k || c.b === n.k);
        tipEl.innerHTML = tipText(n, conns, state.values);
        tipEl.style.display = 'block';
        const r = svg.getBoundingClientRect(), box = svg.parentElement.getBoundingClientRect();
        const sx = r.width / 960, x = r.left - box.left + px(n.x) * sx, y = r.top - box.top + py(n.y) * sx;
        tipEl.style.left = Math.min(box.width - 310, Math.max(4, x + 12)) + 'px';
        tipEl.style.top = (y + 14) + 'px';
      };
      const hide = () => { svg.classList.remove('focus'); tipEl.style.display = 'none'; };
      n.el.addEventListener('mouseenter', show); n.el.addEventListener('mouseleave', hide);
      n.el.addEventListener('click', (e) => { e.stopPropagation(); svg.classList.contains('focus') ? hide() : show(); });
    }
    const state = { values: null };
    // keep: share of connections shown (static view); values: node values (thinking view); win: the chosen output
    function update({ keep = 1, values = null, win = null } = {}) {
      state.values = values;
      const val = (k) => values ? (values.get(k) ?? 0) : 1;
      const s = conns.map((c) => Math.abs(c.w * val(c.a)));
      const max = Math.max(1e-9, ...s);
      const cut = [...s].sort((x, y) => y - x)[Math.max(0, Math.ceil(keep * s.length) - 1)] ?? 0;
      conns.forEach((c, i) => {
        const f = s[i] / max, on = values ? true : s[i] >= cut;
        c.el.setAttribute('stroke', (c.w * Math.sign(val(c.a) || 1)) >= 0 ? 'var(--pos)' : 'var(--neg)');
        c.el.setAttribute('stroke-width', (0.4 + 3.2 * f).toFixed(2));
        c.el.setAttribute('opacity', on ? (values ? Math.max(0.03, f) : 0.12 + 0.88 * f).toFixed(3) : 0);
      });
      const outs = [...nodes.values()].filter((n) => n.kind === 'out');
      const outMax = Math.max(1e-9, ...outs.map((n) => Math.abs(val(n.k))));
      for (const n of nodes.values()) {
        n.el.style.fill = values ? mixc(val(n.k), n.kind === 'out' ? outMax : n.kind === 'in' ? 2 : 1) : '';
        n.el.classList.toggle('win', win != null && n.k === win);
        if (n.lab) {
          const v = values ? ' ' + fmt(val(n.k), n.kind === 'out' ? 1 : 2) : '';
          n.lab.textContent = (n.k === win ? '▶ ' : '') + name(n.k) + v;
          n.lab.classList.toggle('win', n.k === win);
        }
      }
    }
    return { update, nodes, conns };
  }

  function tipText(n, conns, values) {
    const v = values ? `<br>value now: <b>${fmt(values.get(n.k) ?? 0)}</b>` : '';
    if (n.kind === 'in') return `<b>Input: ${esc(name(n.k))}</b><br>${esc(N.input_help[-n.k - 1])}${v}`;
    const ins = conns.filter((c) => c.b === n.k).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const outs = conns.filter((c) => c.a === n.k);
    const top = ins.slice(0, 3).map((c) => `${esc(name(c.a))} ${fmt(c.w)}`).join(', ');
    if (n.kind === 'out') return `<b>Output: ${esc(name(n.k))}</b><br>the score for this action; the highest score is fired.<br>` +
      `${ins.length} incoming connections; strongest: ${top || '–'}${v}`;
    return `<b>Hidden node ${name(n.k)}</b> · ${n.born > 0 ? 'born at generation ' + n.born + ' (added by a mutation)' : 'from the start (random)'}<br>` +
      `bias ${fmt(n.bias)} · ${ins.length} in · ${outs.length} out<br>strongest inputs: ${top || '–'}${v}`;
  }

  // ---- 1 · the network right now ----
  const net1 = drawNet($('net1'), last, $('tip1'));
  const keep1 = () => { $('keep1v').textContent = $('keep1').value; net1.update({ keep: $('keep1').value / 100 }); };
  $('keep1').addEventListener('input', keep1); keep1();
  $('net1cap').textContent = `Generation ${last.gen}, stage “${last.stage}”: ${last.hidden} hidden nodes, ${last.n_conns} connections ` +
    `(${last.nodes.filter((n) => n[4] > 0 && n[0] >= 4).length} of the hidden nodes were added by NEAT after the start).`;

  // ---- 2 · watch it think ----
  const T = N.think;
  if (T && T.flights && T.flights.length) {
    const tsnap = snaps.find((s) => s.gen === T.gen) || last;
    const net2 = drawNet($('net2'), tsnap, $('tip2'));
    const sky = $('sky'), ctx = sky.getContext('2d');
    let fi = 0, i = 0, timer = null;
    T.flights.forEach((f, j) => {
      const b = document.createElement('button');
      b.textContent = f.name; b.onclick = () => pick(j);
      $('flightpick').appendChild(b);
    });
    function pick(j) {
      fi = j; i = 0;
      [...$('flightpick').children].forEach((b, k) => b.classList.toggle('on', k === j));
      const f = T.flights[j];
      $('step').max = f.frames.length - 1;
      $('flightcap').textContent = `Champion of generation ${T.gen}, flying ${f.caption}. Outcome: ${f.outcome}.`;
      show();
    }
    function show() {
      const f = T.flights[fi], fr = f.frames[i];
      $('step').value = i;
      $('stepv').textContent = `step ${2 * i} of ${2 * (f.frames.length - 1)}`;
      const values = new Map(T.keys.map((k, j) => [k, fr.v[j]]));
      net2.update({ values, win: fr.act });
      paint(f, i);
      $('act').textContent = `▶ ${N.outputs[fr.act]}` + (i === f.frames.length - 1 ? ` · ${f.outcome}` : '');
      why(values, fr.act);
    }
    // why this action: the biggest pushes into the winning output's score (weight x the value of the node it comes from)
    function why(values, win) {
      const into = net2.conns.filter((c) => c.b === win).map((c) => ({ k: c.a, p: c.w * (values.get(c.a) ?? 0) }))
        .sort((a, b) => Math.abs(b.p) - Math.abs(a.p)).slice(0, 4);
      const scores = N.outputs.map((nm, k) => `${nm} <span class="v">${fmt(values.get(k) ?? 0, 1)}</span>`).join(' · ');
      const lbl = (k) => k < 0 ? 'input “' + N.inputs[-k - 1] + '”' : 'hidden node #' + k;
      $('why').innerHTML = `<h4>Why “${esc(N.outputs[win])}”?</h4><div class="muted small">scores: ${scores}</div>
        <p style="margin:8px 0 2px">Biggest pushes into its score this step:</p><ol>` +
        into.map((c) => `<li>${lbl(c.k)}: <span class="v">${fmt(c.p)}</span> ${c.p >= 0 ? '(raises it)' : '(lowers it)'}</li>`).join('') +
        `</ol><p class="muted small" style="margin:6px 0 0">Hover a hidden node in the drawing below to see what it listens to.</p>`;
    }
    function paint(f, i) {
      const Wd = sky.width, Hd = sky.height, s = Wd / f.W, X = (x) => x * s;
      const yy = (y) => Hd - y * (Hd / f.H);
      ctx.fillStyle = '#0d0f1a'; ctx.fillRect(0, 0, Wd, Hd);
      ctx.beginPath(); ctx.moveTo(0, Hd);
      for (const [x, y] of f.terrain) ctx.lineTo(X(x), yy(y));
      ctx.lineTo(Wd, Hd); ctx.closePath(); ctx.fillStyle = '#d9d9de'; ctx.fill();
      const [p1, p2, pyy] = f.pad;
      for (const fx of [p1, p2]) {
        ctx.strokeStyle = '#d9d9de'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(fx), yy(pyy)); ctx.lineTo(X(fx), yy(pyy + 1.6)); ctx.stroke();
        ctx.fillStyle = '#e6c84c'; ctx.beginPath(); ctx.moveTo(X(fx), yy(pyy + 1.6)); ctx.lineTo(X(fx) + 10, yy(pyy + 1.4)); ctx.lineTo(X(fx), yy(pyy + 1.2)); ctx.fill();
      }
      if (f.boulder) {
        const [bx, by, br] = f.boulder;
        ctx.fillStyle = '#8b8b93'; ctx.beginPath(); ctx.arc(X(bx), yy(by), br * s, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.fillStyle = 'rgba(160,170,255,.35)';
      for (let j = 0; j < i; j += 2) { const q = f.frames[j]; ctx.fillRect(X(q.x) - 1, yy(q.y) - 1, 2, 2); }
      const fr = f.frames[i];
      ctx.save(); ctx.translate(X(fr.x), yy(fr.y)); ctx.rotate(-fr.a); const m = s / 30;
      if (fr.act === 2) { ctx.fillStyle = '#ffb347'; ctx.beginPath(); ctx.moveTo(-8 * m, 10 * m); ctx.lineTo(8 * m, 10 * m); ctx.lineTo(0, 30 * m); ctx.fill(); }
      if (fr.act === 1) { ctx.fillStyle = '#ffb347'; ctx.beginPath(); ctx.moveTo(-17 * m, -8 * m); ctx.lineTo(-17 * m, 0); ctx.lineTo(-30 * m, -4 * m); ctx.fill(); }
      if (fr.act === 3) { ctx.fillStyle = '#ffb347'; ctx.beginPath(); ctx.moveTo(17 * m, -8 * m); ctx.lineTo(17 * m, 0); ctx.lineTo(30 * m, -4 * m); ctx.fill(); }
      ctx.strokeStyle = '#b8b8ff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(-10 * m, 8 * m); ctx.lineTo(-20 * m, 26 * m); ctx.moveTo(10 * m, 8 * m); ctx.lineTo(20 * m, 26 * m); ctx.stroke();
      ctx.fillStyle = '#8075e6';
      ctx.beginPath(); [[-14, -17], [-17, 0], [-17, 10], [17, 10], [17, 0], [14, -17]].forEach(([x, y], j) => j ? ctx.lineTo(x * m, y * m) : ctx.moveTo(x * m, y * m));
      ctx.closePath(); ctx.fill(); ctx.restore();
      if (i === f.frames.length - 1) {
        ctx.fillStyle = f.outcome === 'landed' ? '#69db7c' : '#ff8787'; ctx.font = 'bold 18px system-ui'; ctx.textAlign = 'center';
        ctx.fillText(f.outcome.toUpperCase(), Wd / 2, 26);
      }
    }
    $('step').addEventListener('input', () => { i = +$('step').value; stop(); show(); });
    const stop = () => { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; };
    $('play').onclick = () => {
      if (timer) return stop();
      if (i >= T.flights[fi].frames.length - 1) i = 0;
      $('play').textContent = '❚❚ pause';
      timer = setInterval(() => {                       // plays once and stops on the outcome, never loops
        if (i >= T.flights[fi].frames.length - 1) return stop();
        i++; show();
      }, 80);
    };
    pick(0);
  }

  // ---- 3 · how it developed ----
  const net3holder = $('net3');
  $('snap').max = snaps.length - 1;
  $('snap').value = snaps.length - 1;
  const nk = (k) => '#' + k + (k <= 35 ? ' a start node' : '');   // keys 4..35 are the 32 starting hidden nodes
  function showSnap() {
    const s = snaps[+$('snap').value];
    $('snapv').textContent = `${s.gen} · ${s.stage} · ${s.hidden} hidden · ${s.n_conns} connections`;
    drawNet(net3holder, s, $('tip3')).update({ keep: 0.3 });
    const d = s.diff;
    $('diff').innerHTML = !d ? 'Generation 0: the random start. Every genome began like this: 32 hidden nodes, every input connected to every hidden node, every hidden node to every output, random weights.' :
      `Compared with the champion of generation ${d.from}: ` +
      `<b>${d.added_nodes.length ? '+' + d.added_nodes.length + ' node' + (d.added_nodes.length > 1 ? 's' : '') + ' (' + d.added_nodes.map(nk).join(', ') + ')' : 'no new nodes'}</b>, ` +
      `<b>${d.removed_nodes.length ? '−' + d.removed_nodes.length + ' (' + d.removed_nodes.map(nk).join(', ') + ')' : 'none removed'}</b>; ` +
      `+${d.added_conns} / −${d.removed_conns} connections; the shared weights moved by ${d.weight_move.toFixed(2)} on average` +
      (d.weight_move > 0.3 ? ' (a big jump: most likely a champion from another family line)' : '') + '.';
  }
  $('snap').addEventListener('input', showSnap); showSnap();

  const rows = ST && (ST[N.run] || Object.values(ST)[0]) || [];
  const mini = (id, key) => {
    const data = rows.filter((r) => r[key] !== '' && r[key] != null).map((r) => ({ x: +r.gen, y: +r[key] }));
    new Chart($(id), { type: 'line', data: { datasets: [{ data, borderColor: cssv('--accent'), borderWidth: 2, pointRadius: 0, tension: 0 }] },
      options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: false },
        tooltip: { callbacks: { title: (it) => 'generation ' + it[0].parsed.x } } },
        interaction: { mode: 'index', intersect: false },
        scales: { x: { type: 'linear', ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') }, title: { display: true, text: 'generation', color: cssv('--muted') } },
                  y: { ticks: { color: cssv('--muted'), precision: 0 }, grid: { color: cssv('--line') } } } } });
  };
  if (window.Chart) { mini('c_hidden', 'hidden'); mini('c_conns', 'conns'); mini('c_species', 'species'); }

  // input-usage heatmap: rows = inputs, columns = saved champions (one hue, light -> dark)
  (function heat() {
    const svg = $('heat'), n = snaps.length, LW = 130, cw = Math.max(10, Math.min(30, (900 - LW) / n)), ch = 18;
    const W = LW + n * cw + 10, H = 16 * ch + 40;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.innerHTML = '';
    const max = Math.max(...snaps.flatMap((s) => s.importance));
    N.inputs.forEach((nm, r) => { el('text', { x: LW - 8, y: 14 + r * ch + 12, 'text-anchor': 'end' }, svg).textContent = nm; });
    const every = Math.ceil(n / 12);
    snaps.forEach((s, c) => {
      s.importance.forEach((v, r) => {
        const rect = el('rect', { x: LW + c * cw + 1, y: 14 + r * ch + 1, width: cw - 2, height: ch - 2, rx: 2,
                                  fill: `color-mix(in srgb, var(--pos) ${Math.round(100 * v / max)}%, var(--panel))` }, svg);
        rect.addEventListener('mouseenter', (ev) => {
          const tip = $('tip4'); tip.style.display = 'block';
          tip.innerHTML = `<b>${esc(N.inputs[r])}</b> · generation ${s.gen}<br>${(100 * v).toFixed(1)}% of the input weight`;
          const box = svg.parentElement.getBoundingClientRect();
          tip.style.left = Math.min(box.width - 220, ev.clientX - box.left + 10) + 'px'; tip.style.top = (ev.clientY - box.top + 10) + 'px';
        });
        rect.addEventListener('mouseleave', () => { $('tip4').style.display = 'none'; });
      });
      if (c % every === 0 || c === n - 1) el('text', { x: LW + c * cw + cw / 2, y: 16 * ch + 30, 'text-anchor': 'middle', class: 'dim' }, svg).textContent = s.gen;
    });
    el('text', { x: LW, y: 16 * ch + 30 + 0, 'text-anchor': 'end', class: 'dim', dx: -8 }, svg).textContent = 'generation →';
  })();

  // ---- 4 · a real example: the youngest node that NEAT added and that survives in the newest champion ----
  (function example() {
    const born = last.nodes.filter((n) => n[0] >= 4 && n[4] > 0).sort((a, b) => b[4] - a[4]);
    const box = $('example');
    if (!born.length) { box.innerHTML = '<h3>A real example from this run</h3><p>No node added by NEAT has survived in the newest champion yet: so far, evolution has mostly tuned the 32 starting nodes.</p>'; return; }
    const [k, , , bias, g] = born[0];
    const ins = last.conns.filter((c) => c[1] === k).sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
    const outs = last.conns.filter((c) => c[0] === k).sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
    const li = (arr, side) => arr.slice(0, 4).map((c) => `${esc(name(c[side]))} (${fmt(c[2])})`).join(', ') || 'none';
    box.innerHTML = `<h3>A real example from this run</h3><p>In the newest champion (generation ${last.gen}), ${born.length} of the hidden
      nodes were added by NEAT after the start. The youngest is <b>node #${k}</b>, first seen in the champion of generation ${g}.
      It now takes ${ins.length} inputs, the strongest from ${li(ins, 0)}, and feeds ${outs.length} nodes: ${li(outs, 1)}.
      Its bias is ${fmt(bias)}. Find it in the drawings above: the dashed ring. Hover it to see its links.</p>`;
  })();
})();
