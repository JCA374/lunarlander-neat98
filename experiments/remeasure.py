"""Re-fly champions on the same 1,000 analysis seeds (real task, C++ sim: bit-identical to a fresh reference env,
results/cpp_equivalence.json). Checks of 200 flights swing and the best of many is flattered; this is the fair
comparison between old and new champions. Writes results/baseline_<run>_<gen>.json (the Log page's table).

    python -m experiments.remeasure runs/N1/champions/g02462_s3_goal.json runs/N1/champions/g03462_s3_goal.json
"""
import argparse
import collections
import json
import os
import re
import time

from experiments.evaluate import wilson
from lander import cppenv
from lander.env import GOAL_WORLD
from lander.neatnet import NeatJsonNet
from lander.seeds import ANALYSIS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('champions', nargs='+')
    ap.add_argument('--n', type=int, default=1000)
    a = ap.parse_args()
    seeds = [ANALYSIS + 900_000 + i for i in range(a.n)]
    for p in a.champions:
        c = collections.Counter('offpad' if r['outcome'] == 'landed' and not r['landed'] else r['outcome']
                                for r in cppenv.fly(NeatJsonNet(json.load(open(p))), seeds, GOAL_WORLD))
        run = p.split('/')[1]
        gen = int(re.search(r'g(\d+)', os.path.basename(p)).group(1))
        out = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'policy': p, 'n': a.n, 'seeds': f'ANALYSIS+900000..+{a.n - 1}',
               'label': f'NEAT {run} champion, generation {gen}', 'note': 'same 1,000 unseen analysis flights for every champion',
               **{k: c[k] / a.n for k in ('landed', 'offpad', 'crashed', 'boulder', 'timeout')},
               'ci': list(wilson(c['landed'], a.n))}
        json.dump(out, open(f'results/baseline_{run}_g{gen:05d}.json', 'w'), indent=1)
        print(out['label'], {k: out[k] for k in ('landed', 'crashed', 'boulder', 'offpad', 'timeout')}, flush=True)


if __name__ == '__main__':
    main()
