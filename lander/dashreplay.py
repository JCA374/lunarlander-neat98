"""Replays for the local dashboard: the newest generation champion flying in the world it trains in right now.

    .venv/bin/python -m lander.dashreplay runs/N1        # renders runs/N1/dash/now_*.gif and runs/N1/dash/now.json

Reads runs/<run>/champion_now.json (written by evo/train.py every generation: the network + the world it flew). One
flight in that world without the practice mixes, plus, when the stage mixes in extra flights, one such flight (a
practice boulder, or the full real task). Seeds come from the analysis range and change with the generation, so
every render is a new flight, never a training seed. Started by dashboard.py with nice; no policy imports this.
"""
import dataclasses
import json
import os
import sys

from lander.env import World, GOAL_WORLD
from lander.seeds import ANALYSIS


def flights(world, gen):
    base = ANALYSIS + 900_000 + 10 * gen
    plain = dataclasses.replace(world, boulder_mix=0.0, goal_mix=0.0)
    out = [('its world now', plain, base, 'a flight in the world it trains in now' +
            (' (without the practice flights)' if world.boulder_mix or world.goal_mix else ''))]
    if world.boulder_mix and not world.boulder:
        out.append(('practice boulder', dataclasses.replace(plain, boulder=True), base + 1,
                    f'a practice flight with the small boulder ({world.boulder_mix:.0%} of its flights; '
                    f'boulder at {world.boulder_size:.0%} size)'))
    if world.goal_mix:
        out.append(('real task', dataclasses.replace(GOAL_WORLD, support=world.support), base + 2,
                    f'a rehearsal flight in the full real task ({world.goal_mix:.0%} of its flights)'))
    return out


def main(run_dir):
    from site_hooks import replay_gif, outcome
    src = os.path.join(run_dir, 'champion_now.json')
    d = json.load(open(src))
    gen, world = d['gen'], World(**d['world'])
    out_dir = os.path.join(run_dir, 'dash')
    os.makedirs(out_dir, exist_ok=True)
    pol_path = os.path.join(out_dir, 'champion.json')
    json.dump(d, open(pol_path, 'w'))
    items = []
    for i, (name, w, seed, cap) in enumerate(flights(world, gen)):
        f = f'now_{gen}_{i}.gif'
        ok, txt = outcome(replay_gif(pol_path, seed, os.path.join(out_dir, f), world=w))
        items.append({'name': name, 'file': f, 'caption': cap, 'landed': ok, 'outcome': txt, 'seed': seed})
    meta = {'gen': gen, 'stage': d.get('stage'), 'world': d['world'], 'champ': d.get('champ', {}), 'flights': items}
    json.dump(meta, open(os.path.join(out_dir, 'now.json.tmp'), 'w'))
    os.replace(os.path.join(out_dir, 'now.json.tmp'), os.path.join(out_dir, 'now.json'))
    for f in os.listdir(out_dir):                        # keep only this render's GIFs
        if f.startswith('now_') and f.endswith('.gif') and f not in {x['file'] for x in items}:
            os.remove(os.path.join(out_dir, f))


if __name__ == '__main__':
    main(sys.argv[1])
