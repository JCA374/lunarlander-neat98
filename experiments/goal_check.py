"""The goal check: a policy on 1,000 unseen test-range flights of the real task in the reference env, no help.

    python -m experiments.goal_check models/best/M2_i08100.json --block 1 --label "..." [--note "..."]
Block 1 = seeds 2,000,000,000+, block 2 = 2,000,010,000+ (lander/seeds.py). Writes results/goal_<name>_b<block>.json
(shown in the site's goal table and log) with a 95% Wilson interval.
"""
import argparse, json, os, time
from experiments.evaluate import fly_seeds, summary
from lander.seeds import GOAL_BLOCK_1, GOAL_BLOCK_2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('policy'); ap.add_argument('--block', type=int, choices=[1, 2], required=True)
    ap.add_argument('--n', type=int, default=1000); ap.add_argument('--label', default=''); ap.add_argument('--note', default='')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    start = GOAL_BLOCK_1 if a.block == 1 else GOAL_BLOCK_2
    s = summary(fly_seeds(a.policy, list(range(start, start + a.n)), workers=a.workers))
    s.update(t=time.strftime('%Y-%m-%d %H:%M:%S'), policy=a.policy, start=start, block=a.block,
             label=(a.label or os.path.basename(a.policy)) + f' · test block {a.block}', note=a.note)
    out = f"results/goal_{os.path.basename(a.policy).replace('.json', '')}_b{a.block}.json"
    json.dump(s, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != 'failed_seeds'}))
