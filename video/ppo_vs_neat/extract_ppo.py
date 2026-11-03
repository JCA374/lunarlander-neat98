"""Data for the PPO-vs-NEAT film: one real-task flight of the PPO run's final state (lunarlander-98, run M2,
iteration 8,451: actor AND critic from runs/M2/checkpoint.pt), every step's action probabilities, the critic's value,
the reward, and the GAE advantage exactly as rl/ppo.py computes it (gamma 0.995, lambda 0.95, reward scale 0.01,
M2's terminal bonus). Run from ../lunarlander-98 with its venv (it has torch and the v2 task PPO trained on):

    cd ../lunarlander-98 && .venv/bin/python ../neat-v2/video/ppo_vs_neat/extract_ppo.py
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from lander.env import GOAL_WORLD, Flight, make_env
from lander.viz import Painter
from lander.seeds import ANALYSIS
from rl.ppo import AC, terminal_bonus

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'ppo')
os.makedirs(OUT, exist_ok=True)
ck = torch.load('runs/M2/checkpoint.pt', map_location='cpu', weights_only=False)
net = AC()
net.load_state_dict(ck['net'])
net.eval()
RW = {'landing_bonus': 200, 'offpad_penalty': 100, 'timeout_penalty': 100}   # curricula/M2.json
GAMMA, LAM, SCALE = 0.995, 0.95, 0.01

seed = int(sys.argv[1]) if len(sys.argv) > 1 else ANALYSIS + 555_000
env = make_env('rgb_array')
f = Flight(env, seed, GOAL_WORLD)
painter = Painter(seed)
steps, fin = [], False
while not fin:
    x = torch.tensor(np.array(f.inputs()), dtype=torch.float32)[None]
    with torch.no_grad():
        z = net.pi(net.norm(x))[0]
        p = torch.softmax(z, -1).numpy()
        v = float(net.v(net.norm(x))[0, 0])
    a = int(np.argmax(p))                    # greedy, as every check and the goal test flew it
    painter.paint(env, f).save(os.path.join(OUT, f'f{len(steps):04d}.png'))
    before = f.reward
    fin = f.step(a)
    r = f.reward - before
    if fin:
        res = f.result()
        r += terminal_bonus(res, RW)
    steps.append({'p': [round(float(q), 4) for q in p], 'a': a, 'v': v, 'r': r * SCALE})
painter.paint(env, f).save(os.path.join(OUT, f'f{len(steps):04d}.png'))
# GAE, as rl/ppo.py update(): delta_t = r_t + gamma V(s_t+1) - V(s_t), A_t = delta_t + gamma lambda A_t+1 (0 after the end)
adv, g = [0.0] * len(steps), 0.0
for t in reversed(range(len(steps))):
    nv = steps[t + 1]['v'] if t + 1 < len(steps) else 0.0
    delta = steps[t]['r'] + GAMMA * nv - steps[t]['v']
    g = delta + GAMMA * LAM * g
    adv[t] = g
    steps[t]['delta'] = delta
for s, A in zip(steps, adv):
    s['adv'] = A
res = f.result()
json.dump({'seed': seed, 'iteration': ck['state']['it'], 'train_steps': ck['state']['steps'],
           'outcome': res['outcome'], 'landed': bool(res['landed']), 'steps': steps,
           'params': sum(p.numel() for p in net.pi.parameters()), 'critic_params': sum(p.numel() for p in net.v.parameters())},
          open(os.path.join(OUT, 'ppo_flight.json'), 'w'))
print(seed, res['outcome'], len(steps), 'steps; actor params', sum(p.numel() for p in net.pi.parameters()))
env.close()
