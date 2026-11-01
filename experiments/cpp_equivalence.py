"""BRIEF section 6 test for the C++ training env (sim/lander_sim.cpp via lander/cppenv.py) against the reference env.

The same network flies the same 1,000 analysis seeds in both. Passes when the outcome (landed between the flags /
off-pad / crashed / timeout / boulder) is the same on >= 98% of the seeds and the C++ landed rate is inside the
reference's 95% confidence interval. Also counts bit-identical flights against a fresh reference env per flight (the
reference itself differs slightly with what its env flew before: Box2D reuses ids of destroyed bodies), and the speed.

    python -m experiments.cpp_equivalence runs/N1/champion_now.json --n 1000
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import time
from multiprocessing import Pool

from experiments.evaluate import wilson
from lander import cppenv
from lander.env import GOAL_WORLD, World, run_episode, make_env
from lander.neatnet import NeatJsonNet
from lander.seeds import ANALYSIS

_W = {}
KEYS = ('reward', 'steps', 'outcome', 'landed', 'x_final')


def _init(d):
    _W['net'], _W['env'] = NeatJsonNet(d), make_env()


def _ref(args):
    seed, world, fresh = args
    r = run_episode(_W['net'], seed, world, env=None if fresh else _W['env'])
    return {k: r[k] for k in KEYS}


def kind(r):
    return 'landed' if r['landed'] else ('off-pad' if r['outcome'] == 'landed' else r['outcome'])


def compare(d, seeds, world, workers):
    net = NeatJsonNet(d)
    t = time.time()
    cpp = [{k: r[k] for k in KEYS} for r in cppenv.fly(net, seeds, world)]
    t_cpp = time.time() - t
    with Pool(workers, initializer=_init, initargs=(d,)) as p:
        t = time.time()
        ref = p.map(_ref, [(s, world, False) for s in seeds], chunksize=25)
        t_ref = (time.time() - t) * workers                    # one-core seconds, to compare with the one-core C++
        fresh = p.map(_ref, [(s, world, True) for s in seeds], chunksize=25)
    n = len(seeds)
    k_ref, k_cpp = sum(r['landed'] for r in ref), sum(r['landed'] for r in cpp)
    ci = wilson(k_ref, n)
    same = sum(kind(a) == kind(b) for a, b in zip(cpp, ref)) / n
    steps = sum(r['steps'] for r in cpp)
    return {'world': world.as_dict(), 'n': n, 'same_outcome': same,
            'same_outcome_ref_vs_fresh_ref': sum(kind(a) == kind(b) for a, b in zip(ref, fresh)) / n,
            'identical_to_fresh_ref': sum(a == b for a, b in zip(cpp, fresh)) / n,
            'landed_ref': k_ref / n, 'ci_ref': ci, 'landed_cpp': k_cpp / n,
            'passed': same >= 0.98 and ci[0] <= k_cpp / n <= ci[1],
            'steps_per_s_cpp_one_core': steps / t_cpp, 'steps_per_s_ref_one_core': steps / t_ref,
            'speedup_one_core': t_ref / t_cpp}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('net')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--start', type=int, default=ANALYSIS + 700_000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='results/cpp_equivalence.json')
    a = ap.parse_args()
    d = json.load(open(a.net))
    seeds = list(range(a.start, a.start + a.n))
    worlds = [GOAL_WORLD, World(wind=14, turbulence=1.4, steady=0.1, boulder_mix=0.25, boulder_size=0.4),
              World(wind=0, turbulence=0, support=0.5)]
    out = {'net': a.net, 'start': a.start, 't': time.strftime('%Y-%m-%d %H:%M:%S'), 'worlds': []}
    for w in worlds:
        r = compare(d, seeds, w, a.workers)
        out['worlds'].append(r)
        print(json.dumps(r), flush=True)
    out['passed'] = all(r['passed'] for r in out['worlds'])
    json.dump(out, open(a.out, 'w'), indent=1)
    print('passed' if out['passed'] else 'FAILED', '->', a.out)
