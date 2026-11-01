"""Fly a policy on N seeds of the real task (reference gymnasium env), in parallel. Library + CLI.

    python -m experiments.evaluate teacher:v2 --start 2000050000 --n 1000
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # one BLAS thread per process: 8 workers x 8 threads overloaded the CPU (load 34)
import argparse
import collections
import json
import math
import time
from multiprocessing import Pool

import numpy as np

from lander.env import GOAL_WORLD, World, run_episode, make_env
from lander.policy import load_policy

_W = {}


def _init(spec):
    _W['pol'] = load_policy(spec)
    _W['env'] = make_env()


def _fly(args):
    seed, world = args
    r = run_episode(_W['pol'], seed, world, env=_W['env'])
    return {k: r[k] for k in ('seed', 'reward', 'landed', 'rested', 'outcome', 'touch_speed', 'x_final', 'steps')}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def fly_seeds(spec, seeds, world=GOAL_WORLD, workers=8):
    with Pool(workers, initializer=_init, initargs=(spec,)) as p:
        return p.map(_fly, [(s, world) for s in seeds], chunksize=max(1, len(seeds) // (workers * 8)))


def summary(rs):
    n = len(rs)
    k = sum(r['landed'] for r in rs)
    kinds = collections.Counter('landed' if r['landed'] else ('off-pad' if r['rested'] else r['outcome']) for r in rs)
    ts = [r['touch_speed'] for r in rs if r['touch_speed'] is not None]
    return {'n': n, 'landed': k / n, 'ci': wilson(k, n), 'score': float(np.mean([r['reward'] for r in rs])),
            'outcomes': dict(kinds), 'timeout': kinds['timeout'] / n, 'offpad': kinds['off-pad'] / n,
            'boulder': kinds['boulder'] / n, 'crashed': kinds['crashed'] / n,
            'touch_speed': float(np.median(ts)) if ts else None, 'steps': float(np.mean([r['steps'] for r in rs])),
            'failed_seeds': [r['seed'] for r in rs if not r['landed']][:300]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('policy')
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out')
    a = ap.parse_args()
    t0 = time.time()
    s = summary(fly_seeds(a.policy, list(range(a.start, a.start + a.n)), workers=a.workers))
    s.update(policy=a.policy, start=a.start, secs=round(time.time() - t0, 1), t=time.strftime('%Y-%m-%d %H:%M:%S'))
    print(json.dumps({k: v for k, v in s.items() if k != 'failed_seeds'}))
    if a.out:
        json.dump(s, open(a.out, 'w'), indent=1)
