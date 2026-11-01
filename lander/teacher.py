"""A hand-written lander (the teacher). Its decisions are cloned into a hidden-layer network that seeds NEAT.

Built on gymnasium's heuristic, with what the wind project learned: lean into the drift (drift sensor),
fly pad-first (reach the pad horizontally, then descend), and for the boulder: aim beside it and stay
above its top until clear of it.
"""

import numpy as np

# Tunable gains (see experiments/tune_teacher.py)
P = dict(kx=0.5, kvx=1.0, kdrift=0.35, amax=0.4, hover=0.55, kang=0.5, kvang=1.0,
         ky=0.5, kvy=0.5, deadband=0.05, pad_first=0.6, side_margin=0.9, top_margin=0.7,
         kwind=0.0, edge=1.0, brake=0.0)
# kwind (v2, 2026-10-30): lean on the clean wind sensor (input 16). edge: largest |target x| (keeps detours on screen).
# brake: extra height target while a boulder is close below (start of the flight), so it stops falling before dodging.


def target_x(v, p=P):
    """Where to aim horizontally (observation units, 0 = pad centre). Normally the pad; with a boulder
    in the way, a point beside it until the lander is below its bottom."""
    x, y = v[0], v[1]
    dx, dy, r = v[10], v[11], v[12]
    if r <= 0:
        return 0.0, None
    bx, by = x + dx, y + dy                 # boulder centre in observation units
    ry = r * 1.5                            # radius in y-units (y scale is 1.5x finer)
    below = y < by - ry - 0.15              # already under it: carry on to the pad
    if below:
        return 0.0, None
    # beside it, on the side nearer the pad's line, unless the lander is already well to one side
    side = 1.0 if (x - bx) > 0 else -1.0
    if abs(bx) < r + 0.25 and abs(x - bx) < 0.05:
        side = 1.0 if bx <= 0 else -1.0
    tx = bx + side * (r + p['side_margin'] * 0.12 + 0.06)
    # keep above the boulder top while horizontally overlapping it
    overlap = abs(x - bx) < r + 0.1
    floor = by + ry + p['top_margin'] * 0.12 if overlap else None
    return tx, floor


def act(v, p=P):
    x, y, vx, vy, ang, vang, l, rleg, drift = v[:9]
    wind = v[15] if len(v) > 15 else 0.0
    tx, floor = target_x(v, p)
    tx = float(np.clip(tx, -p['edge'], p['edge']))
    ex = x - tx
    angle_targ = ex * p['kx'] + vx * p['kvx'] + drift * p['kdrift'] + wind * p['kwind']
    angle_targ = np.clip(angle_targ, -p['amax'], p['amax'])
    hover_targ = p['hover'] * abs(ex)
    if abs(ex) > 0.1:                         # pad-first: hold height while far off the target
        hover_targ = max(hover_targ, p['pad_first'] * min(abs(ex), 0.6))
    if floor is not None:
        hover_targ = max(hover_targ, floor)
    if p['brake'] and v[12] > 0 and -0.6 < v[11] < 0 and abs(v[10]) < v[12] + 0.15:
        hover_targ = max(hover_targ, y + p['brake'])     # boulder close below: stop the fall first
    angle_todo = (angle_targ - ang) * p['kang'] - vang * p['kvang']
    hover_todo = (hover_targ - y) * p['ky'] - vy * p['kvy']
    if l or rleg:
        angle_todo = 0.0
        hover_todo = -vy * 0.5
    d = p['deadband']
    if hover_todo > abs(angle_todo) and hover_todo > d:
        return 2
    if angle_todo < -d:
        return 3
    if angle_todo > d:
        return 1
    return 0


class Teacher:
    def __init__(self, p=None):
        self.p = dict(P, **(p or {}))

    def __call__(self, v, flight=None):
        return act(v, self.p)


def load_params(path=None):
    """The tuned gains (experiments/tune_teacher.py writes lander/teacher_params.json), else the defaults."""
    import json
    import os
    path = path or os.path.join(os.path.dirname(__file__), 'teacher_params.json')
    try:
        return dict(P, **json.load(open(path)))
    except FileNotFoundError:
        return dict(P)
