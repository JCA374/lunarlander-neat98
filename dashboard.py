"""Local training progress page: live charts straight from runs/*/stats.csv and events.jsonl, plus the machine's load.

    .venv/bin/python dashboard.py                    # then open http://localhost:8050
    .venv/bin/python dashboard.py --runs DIR         # read runs from another folder (e.g. a demo)
Auto-refreshes every 20 s; the run and window are kept across refreshes. No dependencies beyond the stdlib.
Not published: the GitHub Pages site is the public view, this one is for the laptop (every 20 s instead of 15 min).

stats.csv columns it reads (one row per generation; empty = not measured that generation):
  gen, stage, steps, secs, steps_per_s       generation, curriculum stage, simulated steps so far, seconds, speed
  best, mean, champ_fit                      fitness: best genome, population mean, the re-flown champion
  champ_landed, champ_boulder, champ_timeout, champ_offpad, champ_touchdown   the champion on its training flights
  check_landed                               stage check (200 fresh check flights in the stage's final world)
  goal_landed                                the real task (200 fresh flights, no help)
  wind, turbulence, support, boulder         the stage's world this generation (support = training wheels)
  hidden, conns, species                     champion's hidden nodes and enabled connections, number of species
"""

import csv
import glob
import json
import subprocess
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, 'runs')
PORT = 8050
SITE_ALIVE = json.load(open(os.path.join(ROOT, 'site.json')))['alive_pattern']   # same 'is it training' rule as the site

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEAT Lander: training progress</title>
<style>
:root{--bg:#111215;--panel:#1a1b20;--ink:#ececf1;--muted:#9a9aa3;--line:#2c2d34}
@media (prefers-color-scheme: light){:root{--bg:#f7f6f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1da}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
main{padding:12px 16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
.card h3{margin:0 0 6px;font-size:13px;color:var(--muted);font-weight:600}
.kpis{display:flex;gap:18px;flex-wrap:wrap}.kpis b{display:block;font-size:20px}
.ev{font-size:12px;max-height:300px;overflow:auto}.ev div{border-bottom:1px solid var(--line);padding:3px 0}
select,button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 8px}
</style></head><body>
<header><b>🧬 NEAT lander · training progress</b>
<label>run <select id="run"></select></label>
<label>window <select id="win"><option value="0">all</option><option value="100">last 100</option><option value="300">last 300</option><option value="1000">last 1000</option></select></label>
<span id="upd" style="color:var(--muted)"></span></header>
<main>
<div class="card" style="grid-column:1/-1"><div class="kpis" id="kpis"></div></div>
<div class="card"><h3>Landed: champion's training flights, fresh stage checks, the real task (200 fresh)</h3><div style="position:relative;height:240px"><canvas id="c1"></canvas></div></div>
<div class="card"><h3>Fitness: best and population mean</h3><div style="position:relative;height:240px"><canvas id="c2"></canvas></div></div>
<div class="card"><h3>Champion's traps: boulder hits, timeouts, off-pad rests, touchdown speed</h3><div style="position:relative;height:240px"><canvas id="c3"></canvas></div></div>
<div class="card"><h3>World: wind, turbulence, support (training wheels)</h3><div style="position:relative;height:240px"><canvas id="c4"></canvas></div></div>
<div class="card"><h3>Network size and species</h3><div style="position:relative;height:240px"><canvas id="c5"></canvas></div></div>
<div class="card"><h3>Speed: seconds per generation and thousand steps per second</h3><div style="position:relative;height:240px"><canvas id="c6"></canvas></div></div>
<div class="card" style="grid-column:1/-1"><h3>Events (newest first)</h3><div class="ev" id="ev"></div></div>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const $=id=>document.getElementById(id); const charts={};
const S=(k,d)=>{try{return localStorage.getItem(k)??d}catch(e){return d}}, P=(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}};
const C=['#8ea6ff','#69db7c','#ffa94d','#ff8787','#da77f2','#3bc9db'];
function chart(id, sets, opts={}){
  const ds=sets.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||C[i],backgroundColor:s.color||C[i],pointRadius:s.points?3:0,showLine:!s.points,borderWidth:1.5,yAxisID:s.y||'y',tension:.15}));
  if(charts[id]){charts[id].data.datasets=ds;charts[id].update('none');return}
  charts[id]=new Chart($(id),{type:'line',data:{datasets:ds},options:{animation:false,maintainAspectRatio:false,interaction:{mode:'nearest',intersect:false},
    scales:{x:{type:'linear',ticks:{color:'#888'},grid:{color:'rgba(128,128,128,.18)'}},y:{ticks:{color:'#888'},grid:{color:'rgba(128,128,128,.18)'},...(opts.y||{})},
    ...(opts.y2?{y2:{position:'right',ticks:{color:'#888'},grid:{display:false}}}:{})},plugins:{legend:{labels:{color:'#aaa',boxWidth:10}}}}});
}
async function load(){
  const runs=await fetch('/api/runs').then(r=>r.json());
  const sel=$('run'); if(!sel.options.length){runs.forEach(r=>sel.add(new Option(r,r))); sel.value=S('run',runs[runs.length-1]); $('win').value=S('win','0');}
  const run=sel.value, win=+$('win').value;
  const d=await fetch('/api/run?name='+run).then(r=>r.json());
  let rows=d.stats; if(win) rows=rows.slice(-win);
  const pt=(k,s=1)=>rows.filter(r=>r[k]!=null&&r[k]!=='').map(r=>({x:+r.gen,y:s*r[k]}));
  chart('c1',[{label:'champion landed %',data:pt('champ_landed',100)},{label:'fresh check landed %',data:pt('check_landed',100),points:true,color:'#69db7c'},
    {label:'REAL task, 200 fresh %',data:pt('goal_landed',100),points:true,color:'#ffa94d'},
    {label:'goal 98%',data:rows.length?[{x:+rows[0].gen,y:98},{x:+rows[rows.length-1].gen,y:98}]:[],color:'#ff8787'}],{y:{max:100}});
  chart('c2',[{label:'best',data:pt('best')},{label:'mean',data:pt('mean')},{label:'champion',data:pt('champ_fit')}]);
  chart('c3',[{label:'boulder hits %',data:pt('champ_boulder',100)},{label:'timeouts %',data:pt('champ_timeout',100)},{label:'off-pad %',data:pt('champ_offpad',100)},
    {label:'touchdown speed',data:pt('champ_touchdown'),y:'y2'}],{y:{beginAtZero:true},y2:true});
  chart('c4',[{label:'wind',data:pt('wind')},{label:'turbulence ×10',data:pt('turbulence',10)},{label:'support ×10',data:pt('support',10)},{label:'boulder on ×10',data:pt('boulder',10)}]);
  chart('c5',[{label:'hidden',data:pt('hidden')},{label:'species ×5',data:pt('species',5)},{label:'connections',data:pt('conns'),y:'y2'}],{y2:true});
  chart('c6',[{label:'s / gen',data:pt('secs')},{label:'k steps / s',data:pt('steps_per_s',0.001),y:'y2'}],{y:{beginAtZero:true},y2:true});
  const last=d.stats[d.stats.length-1]||{}; const checks=d.stats.filter(r=>r.check_landed);
  const lc=checks[checks.length-1]||{};
  $('kpis').innerHTML=[['run',run+(d.alive?' · training':' · stopped')],['generation',last.gen],['stage',last.stage],['champion landed',last.champ_landed?(100*last.champ_landed).toFixed(0)+'%':'–'],
    ['last stage check',lc.check_landed?(100*lc.check_landed).toFixed(0)+'% (gen '+lc.gen+')':'–'],
    ['last real-task check',(g=>g?(100*g.goal_landed).toFixed(1)+'% (gen '+g.gen+')':'–')(d.stats.filter(r=>r.goal_landed).pop())],
    ['steps',last.steps?(last.steps/1e6).toFixed(1)+' M':'–'],['machine',`load ${d.machine.load} · RAM free ${d.machine.mem_free_gb} GB`],['hidden / conns',(last.hidden||'–')+' / '+(last.conns||'–')],['s / gen',last.secs]]
    .map(([k,v])=>`<div><span style="color:var(--muted)">${k}</span><b>${v??'–'}</b></div>`).join('');
  $('ev').innerHTML=d.events.slice().reverse().map(e=>{const c=e.check||{};return `<div>${e.t} · gen ${e.gen??'–'} · <b>${e.kind}</b> ${e.stage||''} ${c.landed!=null?'· check '+(100*c.landed).toFixed(0)+'% landed'+(c.score!=null?', score '+Math.round(c.score):'')+', boulder '+(100*c.boulder).toFixed(0)+'%':''}${e.confirm?' · confirm '+(100*e.confirm.landed).toFixed(0)+'%':''}</div>`}).join('');
  $('upd').textContent='updated '+new Date().toLocaleTimeString();
}
$('run').onchange=()=>{P('run',$('run').value);Object.values(charts).forEach(c=>c.destroy());for(const k in charts)delete charts[k];load()};
$('win').onchange=()=>{P('win',$('win').value);load()};
load(); setInterval(load,20000);
</script></body></html>"""


def machine():
    load = open('/proc/loadavg').read().split()[0]
    mem = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {'load': load, 'mem_free_gb': round(mem['MemAvailable'] / 2**20, 1)}


def runs():
    return [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RUNS, '*')), key=os.path.getmtime)
            if os.path.exists(os.path.join(d, 'stats.csv')) and not os.path.basename(d).startswith('test_')]


def run_data(name):
    d = os.path.join(RUNS, os.path.basename(name))
    with open(os.path.join(d, 'stats.csv')) as f:
        stats = list(csv.DictReader(f))
    ev = os.path.join(d, 'events.jsonl')
    events = [json.loads(l) for l in open(ev) if l.strip()] if os.path.exists(ev) else []
    for e in events:
        e.pop('args', None)
    pat = SITE_ALIVE.replace('{run}', os.path.basename(name))
    alive = subprocess.run(['pgrep', '-f', pat], capture_output=True).returncode == 0
    return {'stats': stats, 'events': events, 'alive': alive, 'machine': machine()}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/api/runs':
            body, ctype = json.dumps(runs()), 'application/json'
        elif u.path == '/api/run':
            body, ctype = json.dumps(run_data(parse_qs(u.query)['name'][0])), 'application/json'
        else:
            body, ctype = PAGE, 'text/html; charset=utf-8'
        b = body.encode()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    if '--runs' in sys.argv:
        RUNS = os.path.abspath(sys.argv[sys.argv.index('--runs') + 1])
    print(f'dashboard on http://localhost:{PORT} (runs from {RUNS})')
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
