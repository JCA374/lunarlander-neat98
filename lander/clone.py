"""Clone the teacher into a 13 -> H tanh -> 4 network (behavioural cloning + DAgger), in plain numpy.

DAgger: round 0 records the teacher's own flights; each later round flies the *student* (so it visits the
states its own mistakes lead to) while recording what the teacher would have done there.
The result seeds NEAT's first population (lander/neatnet.py).

    python -m lander.clone --hidden 24 --rounds 6 --out models/seed_mlp.json
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

from lander.env import World, run_episode, N_INPUTS
from lander.teacher import Teacher, load_params


class MLP:
    def __init__(self, n_in, H, n_out=4, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (H, n_in))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, 1 / np.sqrt(H), (n_out, H))
        self.b2 = np.zeros(n_out)

    def logits(self, X):
        h = np.tanh(X @ self.W1.T + self.b1)
        return h, h @ self.W2.T + self.b2

    def __call__(self, v, flight=None):
        return int(np.argmax(self.logits(np.asarray(v)[None])[1][0]))

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ('W1', 'b1', 'W2', 'b2')}

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        for k in ('W1', 'b1', 'W2', 'b2'):
            setattr(m, k, np.array(d[k]))
        return m

    def fit(self, X, y, epochs=30, lr=3e-3, batch=512, wd=1e-5, rng=None):
        rng = rng or np.random.default_rng(1)
        ps = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in ps]
        v = [np.zeros_like(p) for p in ps]
        t = 0
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                xb, yb = X[b], y[b]
                h, z = self.logits(xb)
                z = z - z.max(1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(1, keepdims=True)
                p[np.arange(len(b)), yb] -= 1
                dz = p / len(b)
                gW2 = dz.T @ h + wd * self.W2
                gb2 = dz.sum(0)
                dh = (dz @ self.W2) * (1 - h ** 2)
                gW1 = dh.T @ xb + wd * self.W1
                gb1 = dh.sum(0)
                t += 1
                for i, g in enumerate([gW1, gb1, gW2, gb2]):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    ps[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        _, z = self.logits(X)
        return float((z.argmax(1) == y).mean())


class Recorder:
    """Fly `actor`, record (inputs, teacher action) at every step."""

    def __init__(self, actor, teacher):
        self.actor, self.teacher, self.X, self.y = actor, teacher, [], []

    def __call__(self, v, flight=None):
        self.X.append(v)
        self.y.append(self.teacher(v))
        return self.actor(v)


# mostly the real task (the pilot v2 clone, 2026-10-30), plus plain wind and easier boulder worlds
WORLDS = [World(boulder=True), World(boulder=True), World(boulder=True), World(),
          World(wind=0, turbulence=0, boulder=True), World(turbulence=0, boulder=True)]


def _collect(args):
    student, seed, wi = args
    t = Teacher(load_params())
    rec = Recorder(student if student is not None else t, t)
    r = run_episode(rec, seed, WORLDS[wi])
    return np.array(rec.X), np.array(rec.y), r['landed']


def collect(pool, student, seeds):
    out = pool.map(_collect, [(student, s, i % len(WORLDS)) for i, s in enumerate(seeds)], chunksize=4)
    X = np.concatenate([o[0] for o in out])
    y = np.concatenate([o[1] for o in out])
    return X, y, float(np.mean([o[2] for o in out]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=24)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--flights', type=int, default=300)
    ap.add_argument('--out', default='models/seed_mlp.json')
    a = ap.parse_args()
    mlp = MLP(N_INPUTS, a.hidden)
    base = 500_000_000                   # clone seeds: part of the training range, away from NEAT's
    with Pool(8) as pool:
        X, y, t_land = collect(pool, None, range(base, base + a.flights))
        print(f'teacher flights: {len(X)} steps, landed {t_land:.1%}', flush=True)
        for r in range(a.rounds):
            acc = mlp.fit(X, y, epochs=20 if r else 40)
            Xs, ys, s_land = collect(pool, mlp, range(base + (r + 1) * 10_000, base + (r + 1) * 10_000 + a.flights))
            print(f'round {r}: train acc {acc:.3f}, student landed {s_land:.1%} (mixed worlds), +{len(Xs)} steps', flush=True)
            X, y = np.concatenate([X, Xs]), np.concatenate([y, ys])
        acc = mlp.fit(X, y, epochs=40)
        print(f'final train acc {acc:.3f} on {len(X)} steps')
    json.dump(mlp.to_dict(), open(a.out, 'w'))
    print('saved', a.out)


if __name__ == '__main__':
    main()
