import hashlib, json, os

import numpy as np
import pytest

from lander.env import World, GOAL_WORLD, Flight, run_episode, make_env, boulder_spec, inputs, lander_xy
from lander.policy import load_policy, NumpyMLP
from lander.teacher import Teacher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The task is frozen: copied from ../lunarlander-98 on 2026-11-01 with one approved change, boulder_spec v3 (the
# boulder's bottom >= 3.5 m above the pad). teacher.py is part of the task because env.py's route sensors
# (inputs 14-15) call teacher.target_x with route_params.json.
FROZEN = {
    'lander/env.py': '4378a11bf954182e7a9f783d5997340b5d2c1c14c0d486a88acbb30edcb2755b',
    'lander/route_params.json': 'beb1a0c56d78368e0ac3835dfa144018778e40ea8736f4e3a4c26daa069e31ca',
    'lander/teacher.py': '4a5d61e3ba779a97027a19f71bbe7350461ad1cc72ebe366fe61194b527abfc6',
}


@pytest.mark.parametrize('path', list(FROZEN))
def test_task_is_frozen(path):
    assert hashlib.sha256(open(os.path.join(ROOT, path), 'rb').read()).hexdigest() == FROZEN[path]


@pytest.mark.parametrize('world', [World(), GOAL_WORLD, World(steady=1.0, support=0.5)])
def test_same_seed_same_flight(world):
    env = make_env()
    a = run_episode(Teacher(), 123, world, env=env)
    run_episode(Teacher(), 999, GOAL_WORLD, env=env)              # reuse the env in between
    b = run_episode(Teacher(), 123, world, env=env)
    c = run_episode(Teacher(), 123, world)                         # and a brand-new env
    assert a['reward'] == b['reward'] == c['reward'] and a['steps'] == b['steps'] == c['steps']


def test_wind_depends_on_seed():
    assert run_episode(Teacher(), 1, World())['reward'] != run_episode(Teacher(), 2, World())['reward']


def test_boulder_hit_is_a_crash():
    seed = next(s for s in range(10000) if abs(boulder_spec(s)[0] - 10) < 0.3)
    r = run_episode(lambda v, f: 0, seed, World(wind=0, turbulence=0, boulder=True))
    assert r['outcome'] == 'boulder' and not r['landed']


def test_seed_ranges_do_not_overlap():
    from lander.seeds import SEED_RANGES, GOAL_BLOCK_1, GOAL_BLOCK_2, BASELINE_BLOCK, TEST, ANALYSIS
    spans = sorted(SEED_RANGES.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert GOAL_BLOCK_1 + 1000 <= GOAL_BLOCK_2 and GOAL_BLOCK_2 + 1000 <= BASELINE_BLOCK < ANALYSIS
    assert all(TEST <= s < ANALYSIS for s in (GOAL_BLOCK_1, GOAL_BLOCK_2))


class Spy:
    """A policy that records what it was given, and checks it never needs the flight object."""
    def __init__(self, inner):
        self.inner, self.seen = inner, []

    def __call__(self, v, flight=None):
        self.seen.append((np.array(v), flight))
        return self.inner(v, None)                                 # the real policy gets no flight at all


@pytest.mark.parametrize('spec', ['teacher:v2', 'models/baselines/ppo_M2_i08100.json'])
def test_policy_sees_only_inputs(spec):
    """The policy's inputs equal inputs() computed from the observation history alone, and the policy flies
    the same flight without access to the env (flight=None)."""
    pol = load_policy(os.path.join(ROOT, spec) if spec.endswith('.json') else spec)
    env = make_env()
    spy = Spy(pol)
    ref = run_episode(lambda v, f: pol(v, f), 4242, GOAL_WORLD, env=env)
    got = run_episode(spy, 4242, GOAL_WORLD, env=env)
    assert ref['reward'] == got['reward'] and ref['steps'] == got['steps']
    # recompute every input vector from observations + own actions only
    f = Flight(env, 4242, GOAL_WORLD)
    for v, _ in spy.seen[:-1]:
        assert np.array_equal(v, f.inputs())
        assert len(v) == 16
        if f.step(pol(v, None)):
            break


def test_old_clone_loads_identically():
    from lander.clone import MLP
    d = json.load(open(os.path.join(ROOT, 'models/baselines/old_neat_seed_clone.json')))
    a, b = MLP.from_dict(d), NumpyMLP(d)
    X = np.random.default_rng(1).normal(0, 1, (2000, 16))
    assert all(a(x) == b(x) for x in X)


# ---- boulder layout v3: every layout passable by construction -------------------------------------------------
@pytest.mark.parametrize('name', ['train', 'clone', 'check', 'select', 'test', 'analysis'])
def test_boulder_v3_geometry(name):
    """For 100,000 seeds of every range: top >= 3.5 m below the start, bottom >= 3.5 m above the pad,
    radius 0.8-1.3 m, centre within 2.5 m of the pad centre."""
    from lander.seeds import SEED_RANGES
    from lander.env import W, PAD_Y, START_Y
    lo = SEED_RANGES[name][0]
    eps = 1e-9
    for s in range(lo, lo + 100_000):
        x, y, r = boulder_spec(s)
        assert 0.8 - eps <= r <= 1.3 + eps
        assert abs(x - W / 2) <= 2.5 + eps
        assert START_Y - (y + r) >= 3.5 - eps
        assert (y - r) - PAD_Y >= 3.5 - eps


def test_boulder_v3_keeps_v2_radius_and_x():
    """Only the height changed: same RNG stream, so radius and x equal v2's for every seed."""
    for s in (0, 7, 2_000_000_123, 3_000_000_001):
        rng = np.random.default_rng([s, 13])
        r, x = rng.uniform(0.8, 1.3), 10 + rng.uniform(-2.5, 2.5)
        bx, _, br = boulder_spec(s)
        assert (bx, br) == (x, r)


# ---- NEAT ------------------------------------------------------------------------------------------------------
def _neat_config():
    import neat
    from lander.neatnet import RandomStartGenome
    return neat.Config(RandomStartGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       os.path.join(ROOT, 'neat-config.ini'))


def test_neat_json_net_matches_neat_python():
    """The network every tool flies (NeatJsonNet) computes exactly what neat-python computes, also after mutations
    that add and delete nodes and connections."""
    import neat
    from lander.neatnet import to_json, NeatJsonNet
    cfg = _neat_config()
    pop = neat.Population(cfg)
    gs = list(pop.population.values())[:6]
    for g in gs[3:]:
        for _ in range(30):
            g.mutate(cfg.genome_config)
    X = np.random.default_rng(2).normal(0, 1, (300, 16))
    for g in gs:
        ref = neat.nn.FeedForwardNetwork.create(g, cfg)
        me = NeatJsonNet(to_json(g, cfg))
        assert np.allclose([ref.activate(list(x / me.scale)) for x in X], [me.activate(x) for x in X])


def test_neat_start_is_random_not_copied():
    """Clean start: 32 hidden nodes, 640 connections, and two genomes have different weights."""
    import neat
    from lander.neatnet import size
    pop = neat.Population(_neat_config())
    a, b = list(pop.population.values())[:2]
    assert size(a) == (32, 640)
    assert any(abs(a.connections[k].weight - b.connections[k].weight) > 1e-6 for k in a.connections)


def test_resume_speciation_uses_the_live_threshold(tmp_path, monkeypatch):
    """After a resume, the species set must read the same settings object the trainer tunes (run N1 bug: the unpickled
    species set kept its own copy and speciated with a stale threshold of 46.8 for 140 generations)."""
    import argparse
    from evo import train
    monkeypatch.chdir(tmp_path)
    os.makedirs('runs')
    cur = os.path.join(ROOT, 'curricula', 'N1.json')
    a = argparse.Namespace(run='t', curriculum=cur, config=os.path.join(ROOT, 'neat-config.ini'), seed=1,
                           target_species=12)
    r = train.Run(a)
    r.state['threshold'] = 0.4
    r.config.species_set_config.compatibility_threshold = 0.4
    r._save()
    r2 = train.Run(a)
    assert r2.pop.species.species_set_config is r2.config.species_set_config
    assert r2.pop.species.species_set_config.compatibility_threshold == 0.4
