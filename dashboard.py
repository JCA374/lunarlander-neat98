"""Local training dashboard: where the run is in its curriculum, how far it is from the goal, and the trends.

    .venv/bin/python dashboard.py                    # then open http://localhost:8050
    .venv/bin/python dashboard.py --runs DIR         # read runs from another folder (e.g. a demo)
Auto-refreshes every 20 s. Not published: the GitHub Pages site is the public view, this one is for the laptop.
No dependencies beyond the stdlib (Chart.js comes from a CDN in the page).

Reads, per run: runs/<run>/stats.csv (one row per generation, written by evo/train.py), events.jsonl,
curriculum.json; plus site.json (goal, deadline, the 'is it training' pattern) and results/goal_*.json.
The server turns these into plain-language text (stage, goal status, trends) so the page only has to show it.
"""

import csv
import re
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, 'runs')
PORT = 8050
SITE = json.load(open(os.path.join(ROOT, 'site.json')))
GOAL = SITE['goal']['target']                # 0.95 since 2026-11-01
TREND_WINDOW = 50                            # generations the trend sentences look back over
RENDER_EVERY = 150                           # s: at most one "watch it now" render this often (nice 15, ~1 core)


# ------------------------------------------------------------------ data
def machine():
    load = open('/proc/loadavg').read().split()[0]
    mem = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {'load': load, 'mem_free_gb': round(mem['MemAvailable'] / 2**20, 1)}


def runs():
    return [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RUNS, '*')), key=os.path.getmtime)
            if os.path.exists(os.path.join(d, 'stats.csv')) and not os.path.basename(d).startswith('test_')]


def read_stats(path):
    """One row per generation. A crash retry writes a generation twice; the last occurrence is the one that survived."""
    by_gen = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_gen[int(r['gen'])] = r
    return [by_gen[g] for g in sorted(by_gen)]


def plain(name):
    return re.sub(r'^\d+\s+', '', name)


def num(r, k, default=None):
    v = r.get(k, '')
    return default if v in ('', None) else float(v)


def pct(x, d=0):
    return f'{100 * x:.{d}f}%'


def stage_bands(stats):
    """Contiguous runs of the same stage name: [{name, from, to}] (generations)."""
    bands = []
    for r in stats:
        g = int(r['gen'])
        if bands and bands[-1]['name'] == r['stage']:
            bands[-1]['to'] = g
        else:
            bands.append({'name': r['stage'], 'label': plain(r['stage']), 'from': g, 'to': g})
    return bands


def world_now(stage, last):
    """The current stage's world in words: each ramped setting 'now X of Y'."""
    w, ramp, parts = stage.get('world', {}), stage.get('ramp', {}), []
    wind, turb, steady = num(last, 'wind', 0), num(last, 'turbulence', 0), num(last, 'steady', 0)
    if 'wind' in ramp:
        parts.append(f"the wind is growing: now {wind:g} of {ramp['wind'][1]:g}" +
                     (f' ({steady:.0%} of it a steady push, the rest gusts)' if steady > 0.005 else ' (all gusts)'))
    else:
        parts.append('calm air (no wind)' if wind == 0 else f'full wind {wind:g}')
    if 'turbulence' in ramp:
        parts.append(f"turbulence now {turb:g} of {ramp['turbulence'][1]:g}")
    elif wind and turb:
        parts.append(f'turbulence {turb:g}')
    if num(last, 'boulder_mix', 0) and not num(last, 'boulder', 0):
        parts.append(f"{num(last, 'boulder_mix'):.0%} of flights carry a small practice boulder "
                     f"({num(last, 'boulder_size', 1):.0%} of full size); they do not count for the pace or the pass check")
    if num(last, 'boulder', 0):
        bs = num(last, 'boulder_size', 1)
        parts.append(f"the boulder is growing: now {bs:.0%} of full size" if 'boulder_size' in ramp else 'the boulder (full size)')
    elif not num(last, 'boulder_mix', 0):
        parts.append('no boulder')
    sup = num(last, 'support', 0)
    if 'support' in stage:
        parts.append(f'training wheels now {sup:.2f} (fading to 0)' if sup > 0 else 'training wheels already off')
    if w.get('goal_mix'):
        parts.append(f"{w['goal_mix']:.0%} of flights are the full real task (rehearsal)")
    if stage.get('hard_mix'):
        parts.append(f"{stage['hard_mix']:.0%} of flights re-fly seeds it failed before (hard cases)")
    return parts


def stage_info(cur, stats, events, check_every=10):
    last = stats[-1]
    names = [s['name'] for s in cur['stages']]
    si = names.index(last['stage']) if last['stage'] in names else 0
    st = cur['stages'][si]
    sg = int(last['stage_gen'])
    ramp_t = num(last, 'ramp_t')                 # paced ramp (from N1 gen 230); older rows: stage_gen / ramp length
    ramped = bool(st.get('ramp') or st.get('support'))
    ramp_len = max(1, int(st.get('ramp_frac', 0.6) * st['cap'])) if ramped else 0
    passes = {e['stage']: e for e in events if e['kind'] == 'stage_pass'}
    strip = []
    for i, s in enumerate(cur['stages']):
        e = passes.get(s['name'])
        state = 'passed' if e else 'current' if i == si else 'done' if i < si else 'todo'
        item = {'n': i + 1, 'name': plain(s['name']), 'state': state}
        if e:
            item['detail'] = f"passed at generation {e['gen']}: {pct(e['check']['landed'], 1)} then {pct(e['confirm']['landed'], 1)}"
        elif state == 'done':
            item['detail'] = 'ended at its cap without passing'
        elif state == 'current':
            item['detail'] = f'generation {sg} of the stage'
        else:
            item['detail'] = 'not started'
        strip.append(item)
    p = st.get('pass', {})
    if st.get('final'):
        rule = ('This is the last stage: it has no pass check and runs until the goal is reached or the deadline. '
                'A snapshot of the champion is kept every 25 generations.')
    else:
        need = [f"≥ {pct(p['landed'])} landed" if 'landed' in p else '', f"timeouts ≤ {pct(p['timeout'])}" if 'timeout' in p else '']
        need = ' and '.join(x for x in need if x)
        rule = (f"Every {check_every} generations the champion flies 200 fresh flights in the stage's FINAL world "
                f"(the ramp at its end, no help, no practice flights). To pass it needs {need}, confirmed on another "
                f"fresh 200; passing ends the stage at once, even mid-ramp. Cap: {st['cap']} generations "
                f"(then it moves on anyway).")
        if ramped:
            rule += (" The ramp is paced: it moves 3 steps per generation while the champion lands ≥ 90% of its own "
                     "flights in the current world, 1 step at ≥ 70%, half a step below that.")
    checks = [r for r in stats if r['stage'] == last['stage'] and r.get('check_landed')]
    last_check = (f"Last pass check: {pct(float(checks[-1]['check_landed']), 1)} at generation {checks[-1]['gen']}."
                  if checks else '')
    return {'index': si, 'n_stages': len(names), 'name': plain(st['name']), 'stage_gen': sg, 'cap': st['cap'],
            'ramp_len': ramp_len,
            'ramp_progress': (ramp_t if ramp_t is not None else min(1.0, sg / ramp_len)) if ramp_len else None,
            'world': world_now(st, last), 'rule': rule, 'last_check': last_check, 'expect': st.get('expect', ''),
            'strip': strip}


def goal_info(stats, events, run):
    real = [(int(r['gen']), float(r['goal_landed'])) for r in stats if r.get('goal_landed')]
    best = max(real, key=lambda t: t[1]) if real else None
    results = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'goal_*.json'))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if run in (r.get('policy') or '') or run in os.path.basename(f):
            results.append({'file': os.path.basename(f), 'landed': r.get('landed'), 'n': r.get('n'), 'ci': r.get('ci')})
    cand = [e for e in events if e['kind'] in ('goal_candidate', 'run_goal')]
    near = [e for e in events if e['kind'] == 'goal_near']
    if len(results) >= 2 and all((r['landed'] or 0) >= GOAL for r in results):
        status, text = 'reached', 'GOAL REACHED: ' + ' and '.join(f"{pct(r['landed'], 1)} on {r['n']:,} test flights" for r in results)
    elif results:
        status, text = 'tested', 'Test blocks flown: ' + ', '.join(f"{pct(r['landed'], 1)} ({r['file']})" for r in results)
    elif cand:
        e = ([c for c in cand if c['kind'] == 'goal_candidate'] or cand)[-1]
        sel = e.get('selection', {})
        status, text = 'candidate', (f"Goal candidate at generation {e['gen']}: {pct(sel.get('landed', 0), 1)} on 1,000 fresh selection "
                                     'flights. Next: the two test blocks of 1,000, flown once.')
    else:
        status, text = 'not_yet', 'Not reached yet.'
        if near:
            e = near[-1]
            text += (f" Closest: generation {e['gen']} hit {pct(e['check']['landed'], 1)} on 200 flights but "
                     f"{pct(e['selection']['landed'], 1)} on the 1,000-flight re-check.")
    dl = datetime.fromisoformat(SITE['deadline'])
    return {'target': GOAL, 'status': status, 'text': text,
            'latest': {'gen': real[-1][0], 'landed': real[-1][1]} if real else None,
            'best': {'gen': best[0], 'landed': best[1]} if best else None,
            'rule': (f'The run stops by itself when the champion lands ≥ {pct(GOAL)} of 200 fresh real-task flights AND '
                     f'then ≥ {pct(GOAL)} of 1,000 more fresh flights. Then the two test blocks (1,000 flights each, '
                     'never used before) are flown once: both must reach the goal.'),
            'deadline_text': SITE.get('deadline_text', ''), 'deadline_ts': dl.timestamp()}


def arrow(d, eps):
    return '(↑ rising)' if d > eps else '(↓ falling)' if d < -eps else '(flat)'


def trends(stats):
    """Plain sentences about the last TREND_WINDOW generations."""
    if len(stats) < 2:
        return []
    g1 = int(stats[-1]['gen'])
    win = [r for r in stats if int(r['gen']) > g1 - TREND_WINDOW]
    out = []
    real = [float(r['goal_landed']) for r in win if r.get('goal_landed')]
    if len(real) >= 2:
        a, b = sum(real[:3]) / len(real[:3]), sum(real[-3:]) / len(real[-3:])
        out.append(f'Real task (200 fresh flights per check): {pct(a)} → {pct(b)} {arrow(b - a, 0.02)} '
                   f'(average of the first and last checks in the last {TREND_WINDOW} generations; {len(real)} checks)')
    mean = [float(r['mean']) for r in win if r.get('mean')]
    if len(mean) >= 10:
        a, b = sum(mean[:5]) / 5, sum(mean[-5:]) / 5
        out.append(f"Population mean fitness: {a:.0f} → {b:.0f} {arrow(b - a, 5)} (the whole population, in its stage's world)")
    cl = [float(r['champ_landed']) for r in win if r.get('champ_landed')]
    if len(cl) >= 10:
        a, b = sum(cl[:5]) / 5, sum(cl[-5:]) / 5
        out.append(f"Champion's own training flights: {pct(a)} → {pct(b)} {arrow(b - a, 0.02)} "
                   "(in the stage's world, which gets harder as it ramps)")
    sp = [int(r['species']) for r in win if r.get('species')]
    if sp:
        out.append(f'Species: {sp[0]} → {sp[-1]} (the trainer steers towards ~12)')
    hid = [int(r['hidden']) for r in win if r.get('hidden')]
    con = [int(r['conns']) for r in win if r.get('conns')]
    if hid and con:
        out.append(f'Champion size: {hid[0]} → {hid[-1]} hidden nodes, {con[0]} → {con[-1]} connections')
    return out


def speed(stats):
    last = stats[-20:]
    secs = sum(float(r['secs']) for r in last if r.get('secs'))
    return {'gens_per_hour': round(3600 * len(last) / secs, 1) if secs else None,
            'secs_per_gen': round(secs / len(last), 1) if last else None}


EVENT_TEXT = {
    'deadline': lambda e: ('⏰ Deadline: training stopped', e.get('note', '')),
    'final_pick': lambda e: (f"Final pick for the test blocks: {os.path.basename(e.get('champion') or '–')}",
                             '; '.join(f"{os.path.basename(o['champion'])} {pct(o['landed'], 1)}" for o in e.get('others', []))
                             + ' (each on 1,000 fresh selection flights)'),
    'test_blocks': lambda e: (('🎯 GOAL MET' if e.get('goal_met') else 'Test blocks flown') +
                              f": block 1 {pct(e['block1'], 1)}, block 2 {pct(e['block2'], 1)}", os.path.basename(e.get('champion', ''))),
    'run_start': lambda e: ('Run started', e.get('note', '')),
    'stage_start': lambda e: (f"Stage started: {e.get('stage')}", 'Expected: ' + e.get('expect', '')),
    'check_pass': lambda e: (f"Pass check passed in {e.get('stage')}: {pct(e['check']['landed'], 1)}; confirming on 200 more", ''),
    'stage_pass': lambda e: (f"✅ Stage passed: {e.get('stage')}",
                             f"{pct(e['check']['landed'], 1)} then {pct(e['confirm']['landed'], 1)} "
                             "on two fresh sets of 200 flights in the stage's world"),
    'goal_best': lambda e: (f"Best real-task check so far: {pct(e['check']['landed'], 1)}",
                            f"boulder {pct(e['check']['boulder'])}, crash {pct(e['check']['crashed'])}, "
                            f"timeout {pct(e['check']['timeout'])} (200 fresh flights)"),
    'goal_near': lambda e: (f"Near the goal: {pct(e['check']['landed'], 1)} on 200, but "
                            f"{pct(e['selection']['landed'], 1)} on 1,000 more", ''),
    'goal_candidate': lambda e: (f"🎯 Goal candidate: {pct(e['selection']['landed'], 1)} on 1,000 fresh flights", ''),
    'run_goal': lambda e: ('🎯 Run stopped at the goal', e.get('note', '')),
    'crash': lambda e: ('⚠️ Training crashed; restarted from the last checkpoint', e.get('note', '')),
    'override': lambda e: ('Manual change: ' + (e.get('what') or ', '.join(
        k for k in e if k not in ('t', 'kind', 'gen', 'stage_index', 'reason'))), e.get('reason', '')),
    'stage_cap': lambda e: (f"Stage {e.get('stage')} hit its cap; moving on without a pass", ''),
}


def event_rows(events):
    out = []
    for e in events:
        f = EVENT_TEXT.get(e['kind'])
        try:
            title, body = f(e) if f else (e['kind'].replace('_', ' '), e.get('note', ''))
        except Exception:
            title, body = e['kind'].replace('_', ' '), ''
        out.append({'t': e['t'], 'gen': e.get('gen'), 'kind': e['kind'], 'title': title, 'body': body})
    return out[::-1]


def run_data(name):
    names = runs()
    if not names:
        return {'error': 'no runs yet (waiting for runs/*/stats.csv)'}
    if name not in names:                       # missing or unknown (e.g. a run remembered by the browser): the newest
        name = names[-1]
    d = os.path.join(RUNS, os.path.basename(name))
    stats = read_stats(os.path.join(d, 'stats.csv'))
    if not stats:
        return {'error': f'run {name} has no generations yet'}
    ev = os.path.join(d, 'events.jsonl')
    events = [json.loads(l) for l in open(ev) if l.strip()] if os.path.exists(ev) else []
    cp = os.path.join(d, 'curriculum.json')
    cur = json.load(open(cp)) if os.path.exists(cp) else {'stages': [{'name': stats[-1]['stage'], 'cap': 0}]}
    pat = SITE['alive_pattern'].replace('{run}', os.path.basename(name))
    alive = subprocess.run(['pgrep', '-f', pat], capture_output=True).returncode == 0
    keep = ['gen', 'stage', 'champ_landed', 'check_landed', 'goal_landed', 'goal_boulder', 'goal_timeout', 'goal_offpad',
            'mean', 'best', 'hidden', 'conns', 'species', 'secs']
    return {'name': name, 'alive': alive, 'machine': machine(), 'now': time.time(),
            'last_gen': int(stats[-1]['gen']), 'last_t': os.path.getmtime(os.path.join(d, 'stats.csv')),
            'stage': stage_info(cur, stats, events), 'goal': goal_info(stats, events, name), 'trends': trends(stats),
            'speed': speed(stats), 'bands': stage_bands(stats), 'events': event_rows(events),
            'stats': [{k: r.get(k, '') for k in keep} for r in stats], 'now_replay': now_replay(d)}


def now_replay(d):
    p = os.path.join(d, 'dash', 'now.json')
    out = json.load(open(p)) if os.path.exists(p) else None
    live = os.path.join(d, 'champion_now.json')
    if out is not None and os.path.exists(live):
        try:
            out['live_gen'] = json.load(open(live))['gen']
        except ValueError:
            pass
    return out


def renderer():
    """Background: re-render the newest champion's flights when a newer champion exists (at most every RENDER_EVERY s)."""
    last = {}
    while True:
        for name in runs():
            d = os.path.join(RUNS, name)
            live = os.path.join(d, 'champion_now.json')
            if not os.path.exists(live) or time.time() - os.path.getmtime(live) > 3600:
                continue
            done = os.path.join(d, 'dash', 'now.json')
            if os.path.exists(done) and os.path.getmtime(done) >= os.path.getmtime(live):
                continue
            if time.time() - last.get(name, 0) < RENDER_EVERY:
                continue
            last[name] = time.time()
            subprocess.run(['nice', '-n', '15', os.path.join(ROOT, '.venv/bin/python'), '-m', 'lander.dashreplay', d],
                           cwd=ROOT, capture_output=True, timeout=600)
        time.sleep(15)


# ------------------------------------------------------------------ page
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEAT lander: training</title>
<style>
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1da;--band1:rgba(42,120,214,.07);--band2:rgba(42,120,214,0);
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#2f9e44;--goodbg:rgba(47,158,68,.12);--warnbg:rgba(237,161,0,.14)}
@media (prefers-color-scheme: dark){:root{--bg:#111215;--panel:#1a1b20;--ink:#ececf1;--muted:#9a9aa3;--line:#2c2d34;
  --band1:rgba(57,135,229,.10);--band2:rgba(57,135,229,0);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good:#69db7c;
  --goodbg:rgba(105,219,124,.12);--warnbg:rgba(201,133,0,.16)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
header b{font-size:16px} .muted{color:var(--muted)} .small{font-size:13px}
main{max-width:1180px;margin:0 auto;padding:14px 16px 40px}
h2{font-size:18px;margin:26px 0 8px} h2 .q{color:var(--muted);font-weight:400;font-size:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;margin-bottom:10px}
.st{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--panel)}
.st .h{font-weight:650} .st .d{font-size:13px;color:var(--muted)}
.st.passed{background:var(--goodbg)} .st.passed .h:before{content:'✓ ';color:var(--good)}
.st.current{border:2px solid var(--s1)} .st.current .h:before{content:'▶ ';color:var(--s1)}
.st.todo{opacity:.6}
.bar{height:10px;border-radius:5px;background:var(--line);overflow:hidden;margin:6px 0} .bar i{display:block;height:100%;background:var(--s1)}
.big{font-size:20px;font-weight:650} ul.w{margin:6px 0;padding-left:20px}
.goal{display:grid;grid-template-columns:1.2fr 1fr;gap:14px} @media(max-width:800px){.goal{grid-template-columns:1fr}}
.status{border-radius:10px;padding:10px 12px;font-weight:650;background:var(--warnbg)} .status.reached{background:var(--goodbg)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-variant-numeric:tabular-nums}
.gbar{position:relative;height:18px;border-radius:9px;background:var(--line);margin:8px 0 2px}
.gbar i{position:absolute;left:0;top:0;bottom:0;border-radius:9px;background:var(--s1)} .gbar b{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink)}
.trend li{margin:3px 0} .chartbox{position:relative;height:340px}
.multi{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;margin-top:12px}
.mini h3{margin:0 0 4px;font-size:14px} .mini .h{position:relative;height:190px}
.flights{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.flights figure{margin:0} .flights img{width:100%;max-width:420px;border-radius:8px;display:block;cursor:pointer;background:#000}
.flights figcaption{font-size:13px;margin-top:4px} .ok{color:var(--good);font-weight:650} .bad{color:var(--s2);font-weight:650}
.ev>div{border-bottom:1px solid var(--line);padding:6px 0} .ev .t{font-size:12px;color:var(--muted)}
select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 8px}
</style></head><body>
<header><b>🧬 NEAT lander · training</b><label>run <select id="run"></select></label>
<span id="alive"></span><span class="muted small" id="upd"></span><span class="muted small" id="mach" style="margin-left:auto"></span></header>
<main>
<h2>1 · Where is it now? <span class="q">the curriculum and the world it trains in</span></h2>
<div class="strip" id="strip"></div>
<div class="panel" id="stage"></div>

<h3 style="margin:16px 0 6px;font-size:15px">Watch the champion now <span class="q muted" style="font-weight:400;font-size:13px">the newest generation's best network, flying fresh flights (never training seeds) in the world it trains in now</span></h3>
<div class="panel" id="now"><span class="muted">waiting for the first render…</span></div>

<h2>2 · How far from the goal? <span class="q">the real task: wind 15, turbulence 1.5, boulder, no help</span></h2>
<div class="panel goal"><div id="goalL"></div><div id="goalR"></div></div>

<h2>3 · What are the trends?</h2>
<div class="panel"><ul class="trend" id="trends"></ul></div>
<div class="panel" style="margin-top:12px"><h3 style="margin:0 0 2px;font-size:15px">Landed %, all of the run</h3>
<p class="muted small" style="margin:0 0 6px">Shaded bands are the curriculum stages. Blue dots: the real task (200 fresh flights per check), with its trend
line (average of the last 5 checks). Orange: the stage's final world (the pass check, every 10 generations since generation 230; before that only after the ramp). Faint aqua: the champion's own training flights. Dashed: the goal.</p>
<div class="chartbox"><canvas id="main"></canvas></div></div>
<div class="multi">
<div class="panel mini"><h3>Real task: why flights fail (share of 200 per check)</h3><div class="h"><canvas id="c_fail"></canvas></div></div>
<div class="panel mini"><h3>Fitness in the stage's world</h3><div class="h"><canvas id="c_fit"></canvas></div></div>
<div class="panel mini"><h3>Species in the population (target ~12)</h3><div class="h"><canvas id="c_species"></canvas></div></div>
<div class="panel mini"><h3>Champion: hidden nodes</h3><div class="h"><canvas id="c_hidden"></canvas></div></div>
<div class="panel mini"><h3>Champion: connections</h3><div class="h"><canvas id="c_conns"></canvas></div></div>
<div class="panel mini"><h3>Seconds per generation</h3><div class="h"><canvas id="c_secs"></canvas></div></div>
</div>

<h2>4 · What happened <span class="q">newest first</span></h2>
<div class="panel ev" id="ev"></div>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const $=id=>document.getElementById(id), charts={}, css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const pct=(x,d=0)=>x==null||x===''?'–':(100*x).toFixed(d)+'%';
const S=(k,d)=>{try{return localStorage.getItem(k)??d}catch(e){return d}}, P=(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}};
let D=null;
// stage bands behind every chart, labelled at the top of the main chart
const bandsPlugin={id:'bands',beforeDatasetsDraw(c){const b=c.options.plugins.bands; if(!b) return; const {ctx,chartArea:a,scales:{x}}=c;
  ctx.save(); b.forEach((s,i)=>{const x0=Math.max(a.left,x.getPixelForValue(s.from-.5)), x1=Math.min(a.right,x.getPixelForValue(s.to+.5)); if(x1<=x0) return;
    ctx.fillStyle=i%2?css('--band2'):css('--band1'); ctx.fillRect(x0,a.top,x1-x0,a.bottom-a.top);
    ctx.strokeStyle=css('--line'); ctx.beginPath(); ctx.moveTo(x0,a.top); ctx.lineTo(x0,a.bottom); ctx.stroke();
    if(c.options.plugins.bandLabels){ctx.fillStyle=css('--muted'); ctx.font='12px system-ui'; const t=s.label||s.name; if(ctx.measureText(t).width<x1-x0-6) ctx.fillText(t,x0+4,a.top+32);}});
  ctx.restore();}};
Chart.register(bandsPlugin);
function chart(id, sets, {y={}, labels=false}={}){
  const grid=css('--line'), tick=css('--muted');
  const ds=sets.map(s=>({label:s.label,data:s.data,borderColor:s.color,backgroundColor:s.color,pointRadius:s.points?3.5:0,pointHoverRadius:6,
    showLine:!s.points,borderWidth:s.width||2,borderDash:s.dash||[],tension:0,order:s.order??1}));
  const opts={animation:false,maintainAspectRatio:false,interaction:{mode:'nearest',axis:'x',intersect:false},
    plugins:{bands:D.bands,bandLabels:labels,legend:{display:sets.filter(s=>!s.label.startsWith('_')).length>1,labels:{color:tick,boxWidth:12,filter:i=>!i.text.startsWith('_')}},
      tooltip:{filter:i=>!i.dataset.label.startsWith('_'),callbacks:{title:it=>'generation '+it[0].parsed.x}}},
    scales:{x:{type:'linear',min:D.stats.length?+D.stats[0].gen:0,max:D.stats.length?+D.stats[D.stats.length-1].gen:1,ticks:{color:tick},grid:{color:grid},title:{display:true,text:'generation',color:tick}},
            y:{ticks:{color:tick},grid:{color:grid},...y}}};
  if(charts[id]){charts[id].data.datasets=ds; charts[id].options=opts; charts[id].update('none'); return}
  charts[id]=new Chart($(id),{type:'line',data:{datasets:ds},options:opts});
}
const col=(rows,k,s=1)=>rows.filter(r=>r[k]!==''&&r[k]!=null).map(r=>({x:+r.gen,y:s*r[k]}));
function rolling(pts,n=5){return pts.map((p,i)=>{const w=pts.slice(Math.max(0,i-n+1),i+1); return {x:p.x,y:w.reduce((a,b)=>a+b.y,0)/w.length}})}
const span=(rows,v)=>rows.length?[{x:+rows[0].gen,y:v},{x:+rows[rows.length-1].gen,y:v}]:[];
function render(){
  const st=D.stage, g=D.goal, rows=D.stats;
  $('alive').innerHTML=D.alive?'<b style="color:var(--good)">● training</b>':'<b style="color:var(--s2)">● stopped</b>';
  $('mach').textContent=`laptop: load ${D.machine.load} · ${D.machine.mem_free_gb} GB RAM free`;
  const ago=Math.round((D.now-D.last_t)/60);
  $('upd').textContent=`generation ${D.last_gen} · last row ${ago<1?'just now':ago+' min ago'} · ${D.speed.secs_per_gen} s per generation (${D.speed.gens_per_hour} per hour)`;
  // 1 · stage
  $('strip').innerHTML=st.strip.map(s=>`<div class="st ${s.state}"><div class="h">${s.n}. ${esc(s.name)}</div><div class="d">${esc(s.detail)}</div></div>`).join('');
  const rp=st.ramp_progress;
  $('stage').innerHTML=`<div class="big">Stage ${st.index+1} of ${st.n_stages}: ${esc(st.name)} · generation ${st.stage_gen} of the stage (cap ${st.cap})</div>
    <p style="margin:6px 0 2px">What it trains in right now:</p><ul class="w">${st.world.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>
    ${rp!=null?`<div class="small muted">ramp: ${Math.round(100*rp)}% done (the settings reach their final values at generation ${st.ramp_len} of the stage)</div><div class="bar"><i style="width:${100*rp}%"></i></div>`:''}
    <p style="margin:6px 0">${esc(st.rule)} ${esc(st.last_check)}</p>
    <p class="muted small" style="margin:0">Expected when the stage was planned: ${esc(st.expect)}</p>`;
  // 1b · watch it now (only swap the image when the file changes, so a GIF is not restarted by the refresh)
  const nr=D.now_replay;
  if(nr&&nr.flights){
    const key=D.name+':'+nr.gen;
    if($('now').dataset.key!==key){
      $('now').dataset.key=key;
      const c=nr.champ||{};
      $('now').innerHTML=`<p style="margin:0 0 8px">Champion of generation <b>${nr.gen}</b> (${esc(String(nr.stage||'').replace(/^\d+ /,''))}).
        On its ${c.n||'–'} training + selection flights this generation it landed <b>${pct(c.landed,0)}</b>
        (crashed ${pct(c.crashed,0)}, boulder ${pct(c.boulder,0)}, timeout ${pct(c.timeout,0)}).
        Click a flight to replay it. <span class="muted small" id="nowage"></span></p>
        <div class="flights">${nr.flights.map(f=>`<figure><img src="/dash/${encodeURIComponent(D.name)}/${f.file}" alt="${esc(f.caption)}" onclick="this.src=this.src.split('?')[0]+'?r='+Date.now()">
        <figcaption><span class="${f.landed?'ok':'bad'}">${f.landed?'✓':'✗'} ${esc(f.outcome)}</span> · ${esc(f.caption)}</figcaption></figure>`).join('')}</div>`;
    }
    const lag=(nr.live_gen??nr.gen)-nr.gen;
    if($('nowage')) $('nowage').textContent=lag>0?`(the trainer is already at generation ${nr.live_gen}; a new render follows within a few minutes)`:'(the newest champion)';
  }
  // 2 · goal
  const L=g.latest, B=g.best;
  $('goalL').innerHTML=`<div class="status ${g.status==='reached'?'reached':''}">${esc(g.text)}</div>
    <div class="kv" style="margin-top:10px"><span>goal</span><b>≥ ${pct(g.target)} landed</b>
    <span>latest check</span><b>${L?pct(L.landed,1)+' (generation '+L.gen+')':'–'}</b>
    <span>best check</span><b>${B?pct(B.landed,1)+' (generation '+B.gen+')':'–'}</b></div>
    <div class="gbar" title="latest real-task check vs the goal"><i style="width:${L?100*L.landed:0}%"></i><b style="left:${100*g.target}%"></b></div>
    <div class="small muted">bar: latest real-task check · black mark: the goal (${pct(g.target)}). A best-of-many check flatters (luck), so the run re-checks on 1,000 flights before it believes one.</div>`;
  $('goalR').innerHTML=`<p style="margin:0 0 6px"><b>When does it stop?</b> ${esc(g.rule)}</p>
    <p style="margin:0"><b>Deadline:</b> ${esc(g.deadline_text)} · <span id="cd"></span></p>
    <p class="small muted" style="margin:4px 0 0">At the current speed that is about <span id="gl"></span> more generations.</p>`;
  tick();
  // 3 · trends
  $('trends').innerHTML=D.trends.map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">not enough generations yet</li>';
  const real=col(rows,'goal_landed',100);
  chart('main',[{label:'real task, 200 fresh flights',data:real,color:css('--s1'),points:true},
    {label:'real task, trend (last 5 checks)',data:rolling(real),color:css('--s1'),width:2.5},
    {label:"the stage's final world (pass check)",data:col(rows,'check_landed',100),color:css('--s2'),points:true},
    {label:"champion's training flights",data:col(rows,'champ_landed',100),color:css('--s3'),width:1.2,order:3},
    {label:'goal '+pct(g.target),data:span(rows,100*g.target),color:css('--ink'),dash:[6,4],width:1.5}],
    {y:{min:0,max:100,title:{display:true,text:'% landed',color:css('--muted')}},labels:true});
  const crash=rows.filter(r=>r.goal_landed!=='').map(r=>({x:+r.gen,y:100*Math.max(0,1-r.goal_landed-r.goal_boulder-r.goal_timeout-r.goal_offpad)}));
  chart('c_fail',[{label:'boulder',data:col(rows,'goal_boulder',100),color:css('--s1')},{label:'crash',data:crash,color:css('--s2')},
    {label:'timeout',data:col(rows,'goal_timeout',100),color:css('--s3')},{label:'off-pad',data:col(rows,'goal_offpad',100),color:css('--s4')}],{y:{min:0,max:100}});
  chart('c_fit',[{label:'population mean',data:col(rows,'mean'),color:css('--s1')},{label:'best genome',data:col(rows,'best'),color:css('--s2'),width:1.2}]);
  chart('c_species',[{label:'species',data:col(rows,'species'),color:css('--s1')},{label:'_target',data:span(rows,12),color:css('--muted'),dash:[5,4],width:1}],{y:{min:0}});
  chart('c_hidden',[{label:'hidden nodes',data:col(rows,'hidden'),color:css('--s1')}]);
  chart('c_conns',[{label:'connections',data:col(rows,'conns'),color:css('--s1')}]);
  chart('c_secs',[{label:'seconds',data:col(rows,'secs'),color:css('--s1')}],{y:{min:0}});
  // 4 · events
  $('ev').innerHTML=D.events.map(e=>`<div><div class="t">${esc(e.t)} · generation ${e.gen??'–'}</div><b>${esc(e.title)}</b>${e.body?'<div class="small">'+esc(e.body)+'</div>':''}</div>`).join('');
}
function tick(){ if(!D||!$('cd')) return; const s=D.goal.deadline_ts-Date.now()/1000;
  if(s<=0){$('cd').textContent='passed'; $('gl').textContent='0'; return}
  const d=Math.floor(s/86400), h=Math.floor(s%86400/3600), m=Math.floor(s%3600/60);
  $('cd').textContent=`${d} d ${h} h ${m} min left`; $('gl').textContent=D.speed.gens_per_hour?Math.round(s/3600*D.speed.gens_per_hour).toLocaleString():'–';}
async function load(){
  try{
    const runs=await fetch('/api/runs').then(r=>r.json()), sel=$('run');
    if(!runs.length){$('upd').textContent='no runs yet';return}
    if([...sel.options].map(o=>o.value).join()!==runs.join()){const keep=sel.value||S('run',''); sel.innerHTML=''; runs.forEach(r=>sel.add(new Option(r,r))); sel.value=runs.includes(keep)?keep:runs[runs.length-1];}
    const d=await fetch('/api/run?name='+encodeURIComponent(sel.value)).then(r=>r.json());
    if(d.error){$('upd').textContent=d.error;return}
    D=d; render();
  }catch(e){$('upd').textContent='update failed: '+e.message}
}
$('run').onchange=()=>{P('run',$('run').value);Object.values(charts).forEach(c=>c.destroy());for(const k in charts)delete charts[k];load()};
load(); setInterval(load,20000); setInterval(tick,30000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == '/api/runs':
                body, ctype = json.dumps(runs()), 'application/json'
            elif u.path.startswith('/dash/'):
                _, _, run, f = u.path.split('/', 3)
                p = os.path.join(RUNS, os.path.basename(run), 'dash', os.path.basename(f))
                if not (f.endswith('.gif') and os.path.exists(p)):
                    self.send_error(404); return
                b = open(p, 'rb').read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/gif')
                self.send_header('Cache-Control', 'max-age=3600')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            elif u.path == '/api/run':
                body, ctype = json.dumps(run_data(parse_qs(u.query).get('name', [''])[0])), 'application/json'
            else:
                body, ctype = PAGE, 'text/html; charset=utf-8'
        except Exception as ex:                  # one broken file must not take the page down
            body, ctype = json.dumps({'error': f'{type(ex).__name__}: {ex}'}), 'application/json'
        b = body.encode()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    if '--runs' in sys.argv:
        RUNS = os.path.abspath(sys.argv[sys.argv.index('--runs') + 1])
    print(f'dashboard on http://localhost:{PORT} (runs from {RUNS})')
    threading.Thread(target=renderer, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
