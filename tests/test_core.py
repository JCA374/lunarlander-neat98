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


def test_node_values_are_the_last_call_s():
    """The Model page's "watch it think" reads every node's value after each action (lander/netviz.py). The fast
    (numba) call keeps them in its own buffer: node_values() must return them, not the unused numpy buffer (which
    stayed all zero and froze the drawing)."""
    import neat
    from lander.neatnet import to_json, NeatJsonNet
    cfg = _neat_config()
    g = list(neat.Population(cfg).population.values())[0]
    for _ in range(30):
        g.mutate(cfg.genome_config)
    me, ref = NeatJsonNet(to_json(g, cfg)), NeatJsonNet(to_json(g, cfg))
    for x in np.random.default_rng(3).normal(0, 1, (20, 16)):
        me(x)
        ref.activate(x)
        assert np.abs(me.node_values()).max() > 0
        assert np.allclose(me.node_values(), ref.v)


def test_injected_champion_flies_like_the_saved_one():
    """evo.train.inject: a saved champion put back into a population flies exactly like its JSON, the population
    keeps its size, and NEAT can breed from it (one generation runs)."""
    import neat
    from lander.neatnet import to_json, NeatJsonNet
    from evo.train import inject
    import random
    cfg = _neat_config()
    pop = neat.Population(cfg)
    src = list(pop.population.values())[0]
    for _ in range(40):
        src.mutate(cfg.genome_config)
    d = to_json(src, cfg)
    pop2 = neat.Population(cfg)
    n0 = len(pop2.population)
    assert inject(pop2, cfg, [d], 4, random.Random(0)) == 4
    assert len(pop2.population) == n0
    X = np.random.default_rng(4).normal(0, 1, (300, 16))
    ref = NeatJsonNet(d)
    exact = [g for g in pop2.population.values() if to_json(g, cfg)['conns'] == d['conns']]
    assert exact and all(NeatJsonNet(to_json(exact[0], cfg))(x) == ref(x) for x in X)

    def fit(genomes, config):
        for _, g in genomes:
            g.fitness = random.random()
    pop2.run(fit, 1)
    assert len(pop2.population) >= n0 - 5


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


def test_ramp_pace_counts_only_the_ramped_skill():
    """Paced ramps (N1 gen 230): 3x at >= 90% landed, 1x at >= 70%, 0.5 below; practice-boulder flights don't count."""
    import numpy as np
    from evo.train import Run
    from lander.env import World
    w = World(wind=10, boulder_mix=0.25, boulder_size=0.4)
    mixed = lambda s: np.random.default_rng([s, 17]).random() < 0.25
    seeds = list(range(200))
    # every practice flight fails, every other lands: still full speed
    rs = [{'seed': s, 'landed': not mixed(s)} for s in seeds]
    assert Run.pace(rs, w) == 3.0
    assert Run.pace([{'seed': s, 'landed': i % 10 < 8} for i, s in enumerate(seeds)], World(wind=10)) == 1.0
    assert Run.pace([{'seed': s, 'landed': False} for s in seeds], World(wind=10)) == 0.5


def _traj(run, pol, seed, world, env):
    rec = []
    r = run(lambda v, f: rec.append((tuple(map(float, v)), pol(v, f))) or rec[-1][1], seed, world, env=env)
    return rec, r


@pytest.mark.parametrize('world', [World(wind=0, turbulence=0, support=1.0), World(wind=8, turbulence=0.5, steady=0.5,
                                   boulder_mix=0.25, boulder_size=0.4), World(), GOAL_WORLD, World(goal_mix=0.5)])
def test_fast_flight_is_the_frozen_flight(world):
    """Training flies lander/fastenv.py and the numba NeatJsonNet.__call__; both must give the frozen env.run_episode
    with numpy's argmax(activate()) step for step: the same 16 inputs, actions, rewards and results."""
    import random
    from lander import fastenv
    from lander.neatnet import to_json, NeatJsonNet
    cfg = _neat_config()
    import neat
    g = next(iter(neat.Population(cfg).population.values()))
    net = NeatJsonNet(to_json(g, cfg))
    acts = [random.Random(5).randrange(4) for _ in range(1000)]
    policies = [(Teacher(), Teacher()), (lambda v, f: int(np.argmax(net.activate(v))), net),
                (lambda v, f: acts[f.steps], lambda v, f: acts[f.steps])]
    env = make_env()
    for slow, fast in policies:
        for seed in (3, 77, 1234):
            a, ra = _traj(run_episode, slow, seed, world, env)
            b, rb = _traj(fastenv.run_episode, fast, seed, world, env)
            assert a == b and ra == rb


@pytest.mark.parametrize('world', [World(wind=0, turbulence=0, support=0.5), World(wind=10, turbulence=1.0, steady=0.3,
                                   boulder_mix=0.25, boulder_size=0.4), GOAL_WORLD, World(goal_mix=0.5)])
def test_cpp_flight_is_the_fresh_reference_flight(world):
    """sim/lander_sim (training only, --sim cpp) gives a fresh reference env's flight bit for bit. (A reused reference
    env can differ in the last bits after touchdown: Box2D reuses the ids of destroyed bodies.)"""
    from lander import cppenv
    from lander.neatnet import to_json, NeatJsonNet
    if not cppenv.available():
        pytest.skip('sim/lander_sim not built (sim/build.sh)')
    import neat
    cfg = _neat_config()
    for g in list(neat.Population(cfg).population.values())[:2]:
        net = NeatJsonNet(to_json(g, cfg))
        seeds = [11, 222, 3333, 44444]
        assert cppenv.fly(net, seeds, world) == [run_episode(net, s, world) for s in seeds]


def test_fast_crossover_and_distance_match_neat_python():
    """RandomStartGenome's faster configure_crossover and distance give neat-python's children and distances exactly."""
    import random
    import neat
    from lander.neatnet import RandomStartGenome
    cfg = _neat_config()
    gc = cfg.genome_config
    gs = list(neat.Population(cfg).population.values())[:8]
    for i, g in enumerate(gs):
        for _ in range(10 * i):
            g.mutate(gc)
        g.fitness = float(i)
    snap = lambda g: (sorted((k, c.weight, c.enabled) for k, c in g.connections.items()),
                      sorted((k, n.bias, n.response) for k, n in g.nodes.items()))
    for a in gs:
        for b in gs:
            assert RandomStartGenome.distance(a, b, gc) == neat.DefaultGenome.distance(a, b, gc)
            kids = []
            for f in (neat.DefaultGenome.configure_crossover, RandomStartGenome.configure_crossover):
                random.seed(3)
                k = RandomStartGenome(0)
                f(k, a, b, gc)
                kids.append(snap(k))
            assert kids[0] == kids[1]
