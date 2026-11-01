"""Where does a champion fail? Landing rate over a grid of wind strength x boulder size (C++ sim, analysis seeds).

    python -m experiments.wind_boulder_grid runs/N1/champions/g00382_s2_goal.json --n 400
"""
import argparse
import collections
import json
import os

from lander import cppenv
from lander.env import World
from lander.neatnet import NeatJsonNet
from lander.seeds import ANALYSIS

WINDS = [(0, 0.0), (5, 0.5), (10, 1.0), (15, 1.5)]
SIZES = [None, 0.4, 0.7, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('champion')
    ap.add_argument('--n', type=int, default=400)
    a = ap.parse_args()
    net = NeatJsonNet(json.load(open(a.champion)))
    seeds = [ANALYSIS + 800_000 + i for i in range(a.n)]
    cells = []
    for wind, turb in WINDS:
        for b in SIZES:
            w = World(wind=wind, turbulence=turb, boulder=b is not None, boulder_size=b or 1.0)
            c = collections.Counter('offpad' if r['outcome'] == 'landed' and not r['landed'] else r['outcome']
                                    for r in cppenv.fly(net, seeds, w))
            cell = {'wind': wind, 'turbulence': turb, 'boulder_size': b,
                    **{k: c[k] / a.n for k in ('landed', 'boulder', 'crashed', 'offpad', 'timeout')}}
            cells.append(cell)
            print(cell)
    name = os.path.splitext(os.path.basename(a.champion))[0]
    out = f'results/wind_boulder_grid_{name}.json'
    json.dump({'champion': a.champion, 'n': a.n, 'seeds': f'ANALYSIS+800000..+{a.n - 1}', 'cells': cells},
              open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
