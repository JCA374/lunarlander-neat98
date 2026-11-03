"""NEAT side of the PPO-vs-NEAT film: one real-task flight of the best measured champion (gen 3832, 93.0% on 1,000
flights), rendered frames and its size. Run from the repo root: .venv/bin/python video/ppo_vs_neat/extract_neat.py"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from lander.env import GOAL_WORLD, Flight, make_env
from lander.viz import Painter
from lander.neatnet import NeatJsonNet
from lander.seeds import ANALYSIS

CH = 'runs/N1/champions/g03832_s3_track.json'
OUT = 'video/ppo_vs_neat/build/neat'
os.makedirs(OUT, exist_ok=True)
d = json.load(open(CH))
net = NeatJsonNet(d)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else ANALYSIS + 555_000
env = make_env('rgb_array')
f = Flight(env, seed, GOAL_WORLD)
painter = Painter(seed)
n, fin = 0, False
while not fin:
    painter.paint(env, f).save(os.path.join(OUT, f'f{n:04d}.png'))
    fin = f.step(net(f.inputs()))
    n += 1
painter.paint(env, f).save(os.path.join(OUT, f'f{n:04d}.png'))
res = f.result()
json.dump({'seed': seed, 'outcome': res['outcome'], 'landed': bool(res['landed']), 'frames': n + 1, 'champion': CH,
           'hidden': net.n_hidden - 4 if net.n_hidden >= 4 else net.n_hidden, 'conns': net.n_conns},
          open(os.path.join(OUT, 'neat_flight.json'), 'w'))
print(seed, res['outcome'], n, 'steps; conns', net.n_conns, 'hidden', net.n_hidden)
env.close()
