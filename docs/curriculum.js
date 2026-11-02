(async function () {
  const [cur, status, log] = await Promise.all([get('curricula.json'), get('status.json'), get('log.json')]);
  const runs = (status && status.runs) || [];
  $('runs').innerHTML = runs.slice().reverse().filter((r) => cur && cur[r.run]).map((r) => {
    const c = cur[r.run];
    const stIdx = c.stages.findIndex((s) => s.name === (r.last || {}).stage);
    return `<h2>Run ${esc(r.run)} <span class="muted small">${r.alive ? 'training now' : 'stopped'}</span></h2>
      <p>${esc(c.note || '')}</p>` + c.stages.map((s, i) => {
      const passed = r.passed.includes(s.name);
      const cls = passed ? 'passed' : (i === stIdx && r.alive) ? 'current' : '';
      const sup = s.support ? `<dt>Training wheels</dt><dd>${esc(kv({ support: s.support }))}, then off; checks run without help</dd>` : '';
      const hp = s.hard_p ?? s.hard_mix;
      const hard = (hp ? `<dt>Hard cases</dt><dd>${pct(hp, 0)} of flights re-fly training seeds the recent champions failed</dd>` : '') + (s.flights ? `<dt>Flights</dt><dd>${s.flights} per genome per generation (all genomes fly the same seeds)</dd>` : '');
      const ch = (log || []).filter((e) => e.kind === 'override' && e.run === r.run && e.stage_index === i).reverse()
        .map((e) => (() => { const w = [e.flights ? e.flights + ' flights per genome (was ' + (s.flights || '?') + ')' : '', e.next_stage ? 'moved to the next stage' : ''].filter(Boolean).join(', '); return `<dt>Changed</dt><dd>gen ${e.gen}: ${w ? '<b>' + esc(w) + '</b>. ' : ''}${esc(e.reason || '')}</dd>`; })()).join('');
      return `<div class="stage ${cls}"><div class="n">${i + 1}</div><div>
        <h3>${esc(s.name)} <span class="muted small">· ${passed ? 'passed' : i === stIdx ? 'running' : i < stIdx ? 'done' : 'waiting'} · cap ${s.cap}</span></h3>
        <dl><dt>World</dt><dd>${esc(worldText(s.world, s.ramp))}</dd>${sup}${hard}
        <dt>Reward</dt><dd>${esc(rewardText(s.reward))}</dd><dt>Pass</dt><dd>${esc(s.final ? 'none: runs until a champion holds ≥ 95% on 1,000 fresh selection flights, or the deadline' : passText(s.pass))}</dd>
        <dt>Expected</dt><dd>${esc(s.expect || '')}</dd>${ch}</dl></div></div>`;
    }).join('');
  }).join('');
})();
