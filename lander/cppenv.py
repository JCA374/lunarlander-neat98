"""Training flights in C++ (sim/lander_sim.cpp): the frozen flight of lander/env.py, a whole batch per call.

Python prepares each flight exactly as env.Flight does (the world after goal_mix, the steady push, the boulder, the
gust phase) and draws every random number with gymnasium's own RNG in reset()'s order: 12 terrain heights, the
initial push, then two engine-dispersion uniforms per step (the reset's step(0) first). C++ rebuilds the Box2D world
in gymnasium's order and flies the NEAT network to the end.

For training only (BRIEF section 6): checks, the selection and the goal test fly the reference env. The equivalence
test is experiments/cpp_equivalence.py (same seeds and network: same outcome on >= 98% of 1,000 seeds, and the same
landed rate within the reference's confidence interval); its result is results/cpp_equivalence.json.
"""

import glob
import os
import sys
from dataclasses import astuple
from functools import lru_cache

import numpy as np
from gymnasium.utils import seeding

from lander.env import World, W, H, PAD_HALF_WIDTH, LANDING_BONUS, WIND_POWER, MAX_STEPS, centre_bonus, \
    boulder_spec, steady_push, _params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
try:
    import lander_sim
except ImportError:                     # not built (sim/build.sh): callers fall back to the Python env
    lander_sim = None
sys.path.pop(0)

OUTCOMES = ('crashed', 'landed', 'timeout', 'boulder')
N_DISP = 2 + 2 * MAX_STEPS
CHUNKS = 11


def available():
    return lander_sim is not None


@lru_cache(maxsize=512)
def _seed_draws(seed):
    """The reset's draws for this seed: smoothed terrain heights, the initial force, the engine dispersion."""
    rng, _ = seeding.np_random(int(seed))
    height = rng.uniform(0, H / 2, size=(CHUNKS + 1,))
    helipad_y = H / 4
    for k in (-2, -1, 0, 1, 2):
        height[CHUNKS // 2 + k] = helipad_y
    smooth_y = np.array([0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)])
    fx = rng.uniform(-1000.0, 1000.0)
    fy = rng.uniform(-1000.0, 1000.0)
    disp = rng.uniform(-1.0, +1.0, N_DISP)
    g = np.random.default_rng([int(seed), 7])            # env.seed_wind
    wind_idx, torque_idx = int(g.integers(-9999, 9999)), int(g.integers(-9999, 9999))
    return smooth_y, float(fx), float(fy), disp, wind_idx, torque_idx


def flight_spec(seed, world):
    """(smooth_y[11], scalars[13], disp[N_DISP]) for one flight, the same set-up as env.Flight.__init__."""
    return _flight_spec(int(seed), astuple(world))


@lru_cache(maxsize=4096)       # every genome of a generation flies the same seeds in the same world
def _flight_spec(seed, world):
    world = World(*world)
    if world.goal_mix and np.random.default_rng([seed, 19]).random() < world.goal_mix:
        world = World(boulder=True, support=world.support)
    smooth_y, fx, fy, disp, wind_idx, torque_idx = _seed_draws(seed)
    push = world.steady * steady_push(seed) * world.wind / WIND_POWER if world.steady else 0.0
    mixed = world.boulder_mix and np.random.default_rng([seed, 17]).random() < world.boulder_mix
    b = boulder_spec(seed, world.boulder_size) if (world.boulder or mixed) else None
    enable = (world.wind > 0 or world.turbulence > 0) and world.steady < 1
    scal = [fx, fy, wind_idx, torque_idx, world.wind * (1 - world.steady), world.turbulence * (1 - world.steady),
            float(enable), push, world.support, float(b is not None)] + (list(b) if b else [0.0, 0.0, 0.0])
    return smooth_y, scal, disp


def fly(net, seeds, world=World()):
    """Fly a NeatJsonNet on the seeds; one dict per flight with run_episode's keys."""
    specs = [flight_spec(s, world) for s in seeds]
    p = _params()
    res = lander_sim.fly(net.flat_i, net.flat_f, np.array([s[0] for s in specs]), np.array([s[1] for s in specs]),
                         np.array([s[2] for s in specs]), p['side_margin'], p['top_margin'])
    out = []
    for s, (reward, steps, outcome, x, touch, clear), spec in zip(seeds, res.tolist(), specs):
        o = OUTCOMES[int(outcome)]
        landed = o == 'landed' and abs(x) <= PAD_HALF_WIDTH
        r = {'seed': int(s), 'reward': reward, 'steps': int(steps), 'outcome': o, 'landed': landed, 'rested': o == 'landed',
             'x_final': x, 'touch_speed': None if touch != touch else touch,
             'clearance': clear if spec[1][9] else None}
        r['fitness'] = reward + (LANDING_BONUS + centre_bonus(x) if landed else 0.0)
        out.append(r)
    return out
