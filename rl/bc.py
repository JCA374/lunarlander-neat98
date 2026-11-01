"""Behaviour cloning + DAgger of a teacher into the PPO actor's shape (numpy-loadable json).

    python -m rl.bc --teacher teacher:v2 --hidden 128 128 --rounds 6 --out models/bc_v2_128.json

Round 0 records the teacher's own flights; each later round flies the student and records what the teacher would
have done in the states the student's own mistakes lead to. All flights are the real task (seeds in the CLONE range).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # one BLAS thread per process: 8 workers x 8 threads overloaded the CPU (load 34)
import argparse
import json
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

from lander.env import GOAL_WORLD, run_episode, make_env
from lander.policy import load_policy, NumpyMLP
from lander.seeds import CLONE
from rl.ppo import AC

_W = {}


def _init(teacher):
    _W['t'] = load_policy(teacher)
    _W['env'] = make_env()


def _collect(args):
    student, seed = args
    t = _W['t']
    pol = NumpyMLP(student) if student else t
    X, y = [], []

    def rec(v, f=None):
        X.append(v); y.append(t(v))
        return pol(v)
    r = run_episode(rec, seed, GOAL_WORLD, env=_W['env'])
    return np.array(X), np.array(y), r['landed']


def fit(net, X, y, epochs, lr=1e-3):
    opt = torch.optim.Adam(net.pi.parameters(), lr=lr)
    Xt = net.norm(torch.tensor(X, dtype=torch.float32)); yt = torch.tensor(y)
    for _ in range(epochs):
        idx = torch.randperm(len(X))
        for s in range(0, len(X), 1024):
            b = idx[s:s + 1024]
            loss = nn.functional.cross_entropy(net.pi(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((net.pi(Xt).argmax(1) == yt).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='teacher:v2')
    ap.add_argument('--hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--flights', type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    net = AC(tuple(a.hidden))
    with Pool(a.workers, initializer=_init, initargs=(a.teacher,)) as pool:
        out = pool.map(_collect, [(None, CLONE + i) for i in range(a.flights)], chunksize=8)
        X = np.concatenate([o[0] for o in out]); y = np.concatenate([o[1] for o in out])
        print(f'teacher: {len(X)} steps, landed {np.mean([o[2] for o in out]):.1%}', flush=True)
        with torch.no_grad():
            net.mean.copy_(torch.tensor(X.mean(0), dtype=torch.float32))
            net.std.copy_(torch.tensor(X.std(0) + 1e-3, dtype=torch.float32))
        for r in range(a.rounds):
            acc = fit(net, X, y, 15 if r else 30)
            base = CLONE + (r + 1) * 100_000
            out = pool.map(_collect, [(net.export(), base + i) for i in range(a.flights)], chunksize=8)
            print(f'round {r}: acc {acc:.3f}, student landed {np.mean([o[2] for o in out]):.1%}', flush=True)
            X = np.concatenate([X] + [o[0] for o in out]); y = np.concatenate([y] + [o[1] for o in out])
        acc = fit(net, X, y, 30, lr=3e-4)
        print(f'final acc {acc:.3f} on {len(X)} steps', flush=True)
    json.dump(net.export(), open(a.out, 'w'))
    print('saved', a.out)


if __name__ == '__main__':
    main()
