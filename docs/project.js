// Lander-specific formatters for the generic site (ml-project-site skill). NEAT: the x-axis is generations.
window.worldText = function worldText(w = {}, ramp = {}) {
  const p = [];
  const wind = ramp.wind ? `${ramp.wind[0]}→${ramp.wind[1]}` : (w.wind ?? 15);
  const turb = ramp.turbulence ? `${ramp.turbulence[0]}→${ramp.turbulence[1]}` : (w.turbulence ?? 1.5);
  if (w.steady) p.push(`steady push ${wind}`); else p.push(+wind === 0 ? 'calm' : `wind ${wind}`);
  if (!w.steady && +turb !== 0) p.push(`turbulence ${turb}`);
  if (w.boulder) p.push(ramp.boulder_size ? `boulder v3 (size ${ramp.boulder_size[0]}→${ramp.boulder_size[1]})` : 'boulder v3');
  if (ramp.support) p.push(`training wheels ${ramp.support[0]}→${ramp.support[1]}`);
  else if (w.support) p.push(`training wheels ${w.support}`);
  if (w.goal_mix) p.push(`${Math.round(100 * w.goal_mix)}% of flights are the real task (rehearsal)`);
  if (w.hard_mix) p.push(`${Math.round(100 * w.hard_mix)}% re-fly recently failed training seeds (hard cases)`);
  return p.join(' + ');
}
window.rewardText = function rewardText(r = {}) {
  const p = [`${r.score_weight ?? 0.3} × game score`];
  p.push(`+${r.landing_bonus ?? 300} per landing between the flags`);
  p.push(`−${r.timeout_penalty ?? 200} per timeout`);
  p.push(`−${r.boulder_penalty ?? 100} per boulder touch`);
  if (r.offpad_penalty) p.push(`−${r.offpad_penalty} per rest outside the flags`);
  return p.join(', ') + '; averaged over all of the genome\'s flights (every genome flies the same seeds)';
}
window.passText = function passText(p = {}) {
  const q = [];
  if (p.landed != null) q.push(p.landed > 1 ? 'none (runs to its cap, snapshots kept)' : `≥ ${pct(p.landed, 0)} landed on 200 fresh flights, confirmed on another 200`);
  if (p.offpad != null) q.push(`off-pad ≤ ${pct(p.offpad, 0)}`);
  if (p.timeout != null) q.push(`timeouts ≤ ${pct(p.timeout, 0)}`);
  return q.join(', ');
}
window.eventText = (e) => {
  const at = e.gen != null ? ` at generation ${e.gen}` : '';
  if (e.kind === 'goal_best') return [`Best real-task check so far${at}`, `${pct(e.check.landed)} landed of ${e.check.n}; off-pad ${pct(e.check.offpad)}, crash ${pct(e.check.crashed)}, boulder ${pct(e.check.boulder)}, timeout ${pct(e.check.timeout)}\nA best-of-many check flatters (winner's curse); claims are re-measured on more seeds.`];
  if (e.kind === 'override') {
    const what = [e.flights ? `${e.flights} flights per genome` : '', e.next_stage ? 'next stage' : ''].filter(Boolean).join(', ') || 'settings';
    return [`Manual change${at}: ${what}`, e.reason || ''];
  }
  if (e.kind === 'time_limit') return [`Time limit reached (${e.minutes} min)`, ''];
  return null;
};
