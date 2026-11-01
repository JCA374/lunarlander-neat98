"""Data for The model tab: the champion's network over time, and the network 'thinking' during real flights.

    build(data_dir)      writes data_dir/network.json (called by site_hooks.extra every publish)

Never imported by a policy: this only reads saved champions (runs/*/champions/*.json) and flies them for the page.
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = ['x', 'y', 'vx', 'vy', 'angle', 'spin', 'left leg', 'right leg', 'drift', '|x|', 'boulder dx', 'boulder dy',
          'boulder r', 'waypoint dx', 'above boulder', 'wind estimate']
INPUT_HELP = ['position left/right of the pad centre', 'height above the pad', 'sideways speed', 'vertical speed',
              'tilt', 'how fast it turns', 'left leg touches the ground', 'right leg touches the ground',
              'old wind sensor (average change in sideways speed)', 'distance from the centre line',
              'boulder: sideways distance', 'boulder: height difference', 'boulder radius',
              'distance to the point to aim for (beside the boulder, then the pad)',
              'how far above the safe height over the boulder', 'clean wind estimate: change in speed minus its own push']
OUTPUTS = ['do nothing', 'left engine', 'main engine', 'right engine']
N_OUT, START_HIDDEN_LAST = 4, 35          # the start genome's hidden nodes have keys 4..35; higher keys were born later
MAX_SNAPSHOTS = 40


def _snapshot_files(run):
    files = glob.glob(os.path.join(ROOT, 'runs', run, 'champions', '*.json'))
    by_gen = {}
    for f in files:
        m = re.match(r'g(\d+)_s(\d+)_(\w+)\.json', os.path.basename(f))
        if m:
            by_gen.setdefault(int(m.group(1)), []).append((m.group(3), f))
    order = {'passed': 0, 'candidate': 1, 'goal': 2, 'track': 3, 'snap': 4, 'near': 5}
    return [(g, sorted(v, key=lambda t: order.get(t[0], 9))[0]) for g, v in sorted(by_gen.items())]


def _layout(d):
    """x: 0 inputs, 1 outputs, hidden nodes by depth; y: 0..1, inputs and outputs fixed, hidden sorted by key so a
    node keeps its place from snapshot to snapshot."""
    from neat.graphs import feed_forward_layers
    ins, outs = d['inputs'], d['outputs']
    conns = [(a, b) for a, b, _ in d['conns']]
    layers = feed_forward_layers(ins, outs, conns)
    layers = layers[0] if isinstance(layers, tuple) else layers
    depth = {}
    for i, layer in enumerate(layers):
        for n in layer:
            if n not in outs:
                depth[n] = i + 1
    hidden = sorted(int(k) for k in d['nodes'] if int(k) not in outs)
    for k in hidden:
        depth.setdefault(k, 1)                         # a node nothing feeds still sits in the first hidden column
    D = max(depth.values(), default=1)
    pos = {}
    for i, k in enumerate(ins):
        pos[k] = (0.0, (i + 0.5) / len(ins))
    for i, k in enumerate(outs):
        pos[k] = (1.0, 0.3 + 0.4 * (i + 0.5) / len(outs))
    for dd in range(1, D + 1):
        col = [k for k in hidden if depth[k] == dd]
        for i, k in enumerate(col):
            pos[k] = (dd / (D + 1), (i + 0.5) / len(col))
    return pos


def _importance(d):
    """How much the network listens to each input: the summed |weight| leaving it (inputs are pre-scaled to ~1)."""
    imp = np.zeros(len(d['inputs']))
    idx = {k: i for i, k in enumerate(d['inputs'])}
    for a, b, w in d['conns']:
        if a in idx:
            imp[idx[a]] += abs(w)
    return (imp / imp.sum()).round(4).tolist() if imp.sum() else imp.tolist()


def snapshots(run):
    files = _snapshot_files(run)
    if len(files) > MAX_SNAPSHOTS:                     # thin evenly, keep the first and the last
        keep = {round(i * (len(files) - 1) / (MAX_SNAPSHOTS - 1)) for i in range(MAX_SNAPSHOTS)}
        files = [f for i, f in enumerate(files) if i in keep]
    out, born, prev = [], {}, None
    for gen, (label, f) in files:
        d = json.load(open(f))
        pos = _layout(d)
        hidden = sorted(int(k) for k in d['nodes'] if int(k) not in d['outputs'])
        for k in hidden:
            born.setdefault(k, 0 if k <= START_HIDDEN_LAST else gen)
        conns = {(a, b): w for a, b, w in d['conns']}
        snap = {'gen': gen, 'label': label, 'stage': d.get('stage', ''),
                'hidden': len(hidden), 'n_conns': len(conns), 'importance': _importance(d),
                'nodes': [[k, round(pos[k][0], 3), round(pos[k][1], 4), round(d['nodes'][str(k)][0], 3), born[k]]
                          for k in hidden] +
                         [[k, 1.0, round(pos[k][1], 4), round(d['nodes'][str(k)][0], 3), 0] for k in d['outputs']],
                'conns': [[a, b, round(w, 3)] for (a, b), w in conns.items()]}
        if prev:
            pn, pc = set(prev['hid']), prev['conns']
            shared = [k for k in conns if k in pc]
            snap['diff'] = {'from': prev['gen'],
                            'added_nodes': sorted(set(hidden) - pn), 'removed_nodes': sorted(pn - set(hidden)),
                            'added_conns': len(set(conns) - set(pc)), 'removed_conns': len(set(pc) - set(conns)),
                            'weight_move': round(float(np.mean([abs(conns[k] - pc[k]) for k in shared])), 4) if shared else 0}
        prev = {'gen': gen, 'hid': hidden, 'conns': conns}
        out.append(snap)
    return out


def _training_world(run, gen):
    """The world the population trained in at generation `gen` (the ramps' values then, from stats.csv), without
    training wheels or rehearsal mixing."""
    import csv
    from lander.env import World
    rows = {r['gen']: r for r in csv.DictReader(open(os.path.join(ROOT, 'runs', run, 'stats.csv')))}
    r = rows.get(str(gen)) or list(rows.values())[-1]
    w = World(wind=float(r['wind']), turbulence=float(r['turbulence']), steady=float(r['steady']),
              boulder=r['boulder'] == '1', boulder_size=float(r['boulder_size']))
    parts = ['calm air' if w.wind == 0 else f'wind {w.wind:g}' + (f', {w.steady:.0%} steady' if w.steady else '')]
    if w.wind and w.turbulence:
        parts.append(f'turbulence {w.turbulence:g}')
    parts.append(('boulder' + (f' at {w.boulder_size:.0%} size' if w.boulder_size < 1 else '')) if w.boulder else 'no boulder')
    return w, f"the world it trained in at generation {r['gen']} ({', '.join(parts)}; no training wheels)"


def _fly_recorded(net, keys, seed, world):
    """One flight; every other step the lander's pose, the action and every node's value."""
    from lander.env import run_episode, make_env, boulder_spec, W, H
    env = make_env()
    frames = []

    def pol(v, f):
        a = net(v)
        if len(frames) < 600 and f.steps % 2 == 0:
            u = env.unwrapped
            p = u.lander.position
            frames.append({'x': round(p.x, 2), 'y': round(p.y, 2), 'a': round(u.lander.angle, 3), 'act': a,
                           'v': [round(float(net.v[net.idx[k]]), 2) if k in net.idx else 0 for k in keys]})
        return a
    r = run_episode(pol, seed, world, env=env)
    u = env.unwrapped
    terrain = [[round(q[0][0], 2), round(q[0][1], 2)] for q in u.sky_polys] + \
              [[round(u.sky_polys[-1][1][0], 2), round(u.sky_polys[-1][1][1], 2)]]
    out = {'seed': seed, 'frames': frames, 'landed': bool(r['landed']),
           'outcome': 'landed' if r['landed'] else ('rested off the pad' if r['rested'] else r['outcome']),
           'boulder': [round(v, 2) for v in boulder_spec(seed, world.boulder_size)] if world.boulder else None,
           'pad': [round(u.helipad_x1, 2), round(u.helipad_x2, 2), round(u.helipad_y, 2)],
           'terrain': terrain, 'W': W, 'H': H}
    env.close()
    return out


def think(run, path, worlds, tries=5):
    """Fly the champion in each (name, world, caption, prefer_landing) and record it, so the page can animate the
    network as it flies. prefer_landing: show the first of `tries` fixed analysis seeds where it lands (the caption
    says so); otherwise the first seed, whatever happens."""
    from lander.seeds import ANALYSIS
    from lander.neatnet import NeatJsonNet
    d = json.load(open(path))
    net = NeatJsonNet(d)
    keys = list(d['inputs']) + sorted(int(k) for k in d['nodes'] if int(k) not in d['outputs']) + list(d['outputs'])
    flights = []
    for fi, (name, world, caption, prefer) in enumerate(worlds):
        for t in range(tries if prefer else 1):
            f = _fly_recorded(net, keys, ANALYSIS + 777_000 + 100 * fi + t, world)
            if f['landed'] or not prefer:
                break
        if prefer:
            caption += f"; seed {t + 1} of {tries} tried" + (' (the first where it landed)' if f['landed'] else ', none landed')
        flights.append(dict(f, name=name, caption=caption))
    return {'gen': d.get('gen'), 'stage': d.get('stage'), 'keys': keys, 'flights': flights}


def build(data_dir, run=None):
    from lander.env import GOAL_WORLD
    from lander.seeds import ANALYSIS
    runs = sorted((os.path.basename(r) for r in glob.glob(os.path.join(ROOT, 'runs', '*'))
                   if os.path.isdir(os.path.join(r, 'champions')) and not os.path.basename(r).startswith('test_')),
                  key=lambda r: os.path.getmtime(os.path.join(ROOT, 'runs', r)))
    if not runs:
        return
    run = run or runs[-1]
    snaps = snapshots(run)
    if not snaps:
        return
    out = {'run': run, 'inputs': INPUTS, 'input_help': INPUT_HELP, 'outputs': OUTPUTS, 'snapshots': snaps}
    last_gen, (label, path) = _snapshot_files(run)[-1]
    cache = os.path.join(data_dir, 'network_think.json')
    old = json.load(open(cache)) if os.path.exists(cache) else {}
    if old.get('gen') == last_gen and old.get('run') == run:
        out['think'] = old
    else:
        sw, cap = _training_world(run, last_gen)
        worlds = [('its world now', sw, cap, True),
                  ('real task', GOAL_WORLD, 'the real task (wind 15, turbulence 1.5, boulder, no help)', False)]
        th = think(run, path, worlds)
        th['run'] = run
        json.dump(th, open(cache, 'w'))
        out['think'] = th
    json.dump(out, open(os.path.join(data_dir, 'network.json'), 'w'), separators=(',', ':'))
