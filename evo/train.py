"""NEAT from a clean start through a curriculum, until the champion lands >= 95% of fresh real-task flights.

    .venv/bin/python -m evo.train --curriculum curricula/N1.json --run N1 --workers 7     # new run, or resume

Every generation: all genomes fly the same K training seeds (common random numbers: differences are the network's,
not the weather's). The top `reeval_top` genomes re-fly `reeval_flights` selection seeds; the best average is the
generation champion. Every `check_every` generations the champion flies 200 fresh check seeds of the REAL task (no
help) whatever the stage, and, once the stage's ramps are done, 200 in the stage's final world (pass check, confirmed
on another fresh 200). The last stage never ends by passing; snapshots every 25 generations.

Goal (the user's, 2026-11-01): stop when the champion lands >= 95%. A real-task check >= 95% on 200 triggers 1,000
selection flights; if those land >= 95% the run saves `goal_candidate` and stops. The test blocks (2e9+) are flown
afterwards, once, by experiments/goal_check.py.

Hard cases (stage option hard_mix): that share of each generation's training seeds re-fly TRAINING seeds the recent
champions failed on (never check, selection or test seeds).

SIGTERM/SIGINT: finish the generation, checkpoint, exit. Resume: the same command.
override.json in the run folder (read once, then deleted): {"next_stage": true, "flights": 32, "reason": "..."}.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # one BLAS thread per worker: 8 workers x 8 threads once gave a load of 34
import argparse
import csv
import json
import pickle
import random
import signal
import time
from collections import deque
from itertools import count
from multiprocessing import Pool

import neat
import numpy as np

from lander.env import World, GOAL_WORLD, run_episode, make_env
from lander.neatnet import RandomStartGenome, START, to_json, NeatJsonNet, size
from lander.seeds import TRAIN, CHECK, SELECT, CLONE

GOAL = 0.95
TH_MIN, TH_MAX = 0.05, 1.0           # species compatibility threshold range (floor 0.15 -> 0.05 at gen 151 of N1: species fell to 3)
_W = {}


def _init_worker():
    _W['env'] = make_env()


def fly(args):
    """Fly one network (a to_json dict) on the given seeds; small per-flight results."""
    net, seeds, world = args
    pol = NeatJsonNet(net)
    out = []
    for s in seeds:
        r = run_episode(pol, s, world, env=_W['env'])
        out.append({'seed': s, 'reward': r['reward'], 'landed': r['landed'], 'rested': r['rested'],
                    'outcome': r['outcome'], 'touch': r['touch_speed'], 'steps': r['steps']})
    return out


def fly_many(pool, jobs, chunk):
    """[(net, seeds, world), ...] -> one result list per job; jobs are cut into chunks so all workers stay busy."""
    tasks, owner = [], []
    for j, (net, seeds, world) in enumerate(jobs):
        for k in range(0, len(seeds), chunk):
            tasks.append((net, seeds[k:k + chunk], world)); owner.append(j)
    out = [[] for _ in jobs]
    for j, rs in zip(owner, pool.imap(fly, tasks, chunksize=1)):
        out[j] += rs
    return out


def fitness(r, rw):
    """Pay mainly for landing (what worked before): 0.3 x game score + 300 per landing - 200 per timeout
    - 100 per boulder touch, no centre bonus (the game score alone rewards fuel saving: 'points up, landings down')."""
    f = rw.get('score_weight', 0.3) * r['reward']
    if r['landed']:
        f += rw.get('landing_bonus', 300)
    elif r['rested']:
        f -= rw.get('offpad_penalty', 0)
    if r['outcome'] == 'timeout':
        f -= rw.get('timeout_penalty', 200)
    if r['outcome'] == 'boulder':
        f -= rw.get('boulder_penalty', 100)
    return f


def summary(rs, rw):
    n = len(rs)
    ts = [r['touch'] for r in rs if r['touch'] is not None and r['landed']]
    return {'n': n, 'fitness': float(np.mean([fitness(r, rw) for r in rs])),
            'score': float(np.mean([r['reward'] for r in rs])),
            'landed': sum(r['landed'] for r in rs) / n,
            'offpad': sum(r['rested'] and not r['landed'] for r in rs) / n,
            'crashed': sum(r['outcome'] == 'crashed' for r in rs) / n,
            'boulder': sum(r['outcome'] == 'boulder' for r in rs) / n,
            'timeout': sum(r['outcome'] == 'timeout' for r in rs) / n,
            'touchdown': float(np.median(ts)) if ts else None,
            'steps': int(sum(r['steps'] for r in rs))}


def stage_world(stage, t):
    """The stage's world at ramp progress t (0..1). Training wheels fade over the ramp and are off once t >= 1."""
    w = dict(stage.get('world', {}))
    for k, (a, b) in stage.get('ramp', {}).items():
        w[k] = a + (b - a) * min(t, 1.0)
    if 'support' in stage:
        s0, s1 = stage['support']
        w['support'] = 0.0 if t >= 1 else s0 + (s1 - s0) * t
    return World(**w)


def check_world(stage):
    """What a pass check flies: the stage's final world, no help, no rehearsal mix."""
    w = stage_world(stage, 1.0)
    w.goal_mix = w.boulder_mix = 0.0
    w.support = 0.0
    return w


class Run:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join('runs', a.run)
        os.makedirs(os.path.join(self.dir, 'champions'), exist_ok=True)
        self.cur = json.load(open(a.curriculum))
        json.dump(self.cur, open(os.path.join(self.dir, 'curriculum.json'), 'w'), indent=1)
        START['hidden'] = self.cur.get('start_hidden', 32)
        self.config = neat.Config(RandomStartGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation, a.config)
        self.stop = False
        signal.signal(signal.SIGTERM, self._sigterm)
        signal.signal(signal.SIGINT, self._sigterm)
        ck = os.path.join(self.dir, 'checkpoint.pkl')
        if os.path.exists(ck):
            self._restore(ck)
        else:
            random.seed(a.seed); np.random.seed(a.seed)
            self.pop = neat.Population(self.config)
            self.state = {'gen': 0, 'stage': 0, 'stage_gen': 0, 'n_check': 0, 'n_select': 0, 'steps': 0,
                          'best_goal': -1, 'best_check': {}, 'hard': [], 'done': False,
                          'threshold': self.config.species_set_config.compatibility_threshold}
            self._event('run_start', note=self.cur.get('note', ''), args=vars(a))
            self._stage_start()
        if not TH_MIN <= self.state['threshold'] <= TH_MAX:
            self._event('override', what='threshold', old=self.state['threshold'], new=0.5,
                        reason='species threshold had run away; clamped to 0.15-1.0 (journal 2026-11-01)')
            self.state['threshold'] = 0.5
        self.config.species_set_config.compatibility_threshold = self.state['threshold']
        self.hard = deque(self.state['hard'], maxlen=400)

    def _sigterm(self, *_):
        print('signal: checkpoint after this generation', flush=True)
        self.stop = True

    def stage(self):
        return self.cur['stages'][self.state['stage']]

    # --- persistence ---
    def _save(self):
        p = self.pop
        self.state['hard'] = list(self.hard)
        blob = {'state': self.state, 'population': p.population, 'species': p.species, 'generation': p.generation,
                'innovation_tracker': p.reproduction.innovation_tracker,
                'random': random.getstate(), 'np': np.random.get_state()}
        tmp = os.path.join(self.dir, 'checkpoint.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(blob, f)
        os.replace(tmp, os.path.join(self.dir, 'checkpoint.pkl'))

    def _restore(self, path):
        blob = pickle.load(open(path, 'rb'))
        self.pop = neat.Population(self.config, (blob['population'], blob['species'], blob['generation']))
        # keep innovation numbers and node keys counting on from where they were (a reset once crashed a resumed run)
        self.pop.reproduction.innovation_tracker = blob['innovation_tracker']
        self.config.genome_config.innovation_tracker = blob['innovation_tracker']
        top = max(max(g.nodes) for g in blob['population'].values())
        top = max([top] + [max(s.representative.nodes) for s in blob['species'].species.values()])
        self.config.genome_config.node_indexer = count(top + 1)
        random.setstate(blob['random'])
        np.random.set_state(blob['np'])
        self.state = blob['state']
        print(f"resumed at gen {self.state['gen']} stage {self.stage()['name']}", flush=True)

    def _event(self, kind, **kw):
        e = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind, 'gen': self.state['gen'],
             'stage_index': self.state['stage'], **kw}
        with open(os.path.join(self.dir, 'events.jsonl'), 'a') as f:
            f.write(json.dumps(e) + '\n')

    def _stage_start(self):
        s = self.stage()
        self._event('stage_start', stage=s['name'], expect=s.get('expect', ''), world=s.get('world', {}),
                    ramp=s.get('ramp', {}), reward=s.get('reward', {}), cap=s['cap'], **{'pass': s.get('pass', {})})

    def _save_champion(self, genome, label):
        st = self.state
        path = os.path.join(self.dir, 'champions', f"g{st['gen']:05d}_s{st['stage']}_{label}.json")
        d = to_json(genome, self.config)
        d.update(gen=st['gen'], stage=self.stage()['name'], genome_key=genome.key)
        json.dump(d, open(path, 'w'))
        return path

    def _flights(self, seeds, net, world, chunk=10):
        return fly_many(self.pool, [(net, seeds, world)], chunk)[0]

    def check(self, net, world, rw, n=None):
        st = self.state
        n = n or self.a.check_flights
        seeds = [CHECK + st['n_check'] * 1000 + i for i in range(n)]
        st['n_check'] += 1
        rs = self._flights(seeds, net, world, chunk=8)
        st['steps'] += sum(r['steps'] for r in rs)
        return summary(rs, rw)

    def select_flights(self, net, n):
        """Fresh selection seeds (1.5e9+), each block used once."""
        st = self.state
        seeds = [SELECT + 400_000_000 + st['n_select'] * 2000 + i for i in range(n)]
        st['n_select'] += 1
        rs = self._flights(seeds, net, GOAL_WORLD, chunk=20)
        st['steps'] += sum(r['steps'] for r in rs)
        return summary(rs, {})

    # --- one generation ---
    def evaluate(self, genomes, config):
        st, stg, a = self.state, self.stage(), self.a
        t0 = time.time()
        ramp_len = max(1, int(stg.get('ramp_frac', 0.6) * stg['cap']))
        t = st['stage_gen'] / ramp_len if (stg.get('ramp') or stg.get('support')) else 1.0
        world = stage_world(stg, t)
        rw = stg.get('reward', {})
        K = st.get('flights_override') or stg.get('flights', a.flights)
        n_hard = int(round(stg.get('hard_mix', 0.0) * K)) if len(self.hard) >= 20 else 0
        rng = np.random.default_rng([a.seed, st['gen']])
        hard = [int(s) for s in rng.choice(list(self.hard), n_hard, replace=False)] if n_hard else []
        seeds = hard + [TRAIN + st['gen'] * 1000 + i for i in range(K - n_hard)]
        assert all(TRAIN <= s < CLONE for s in seeds)                   # training range only
        nets = {gid: to_json(g, config) for gid, g in genomes}
        res = fly_many(self.pool, [(nets[gid], seeds, world) for gid, _ in genomes], chunk=K)
        steps = sum(r['steps'] for rs in res for r in rs)
        for (gid, g), rs in zip(genomes, res):
            g.fitness = summary(rs, rw)['fitness']
        # re-fly the top genomes on fresh selection seeds (same world); champion = best average over all its flights
        top = sorted(zip(genomes, res), key=lambda z: -z[0][1].fitness)[:a.reeval_top]
        sel = [SELECT + st['gen'] * 1000 + i for i in range(a.reeval_flights)]
        more = fly_many(self.pool, [(nets[gid], sel, world) for (gid, _), _ in top], chunk=10)
        steps += sum(r['steps'] for rs in more for r in rs)
        scored = [(summary(rs + m, rw), g, rs) for ((_, g), rs), m in zip(top, more)]
        champ_s, champ, champ_train = max(scored, key=lambda z: z[0]['fitness'])
        champ_net = nets[champ.key]
        # hard cases: the champion's failed TRAINING seeds, flown in the real task's sense (only when the world is real)
        if world.boulder and world.wind >= 15 and world.support == 0:
            for r in champ_train:
                if not r['landed'] and r['seed'] not in self.hard:
                    self.hard.append(r['seed'])
        fits = [g.fitness for _, g in genomes]
        hid, con = size(champ)
        row = {'gen': st['gen'], 'stage': stg['name'], 'stage_gen': st['stage_gen'], 'steps': 0, 'secs': 0,
               'steps_per_s': 0, 'flights': K, 'hard': n_hard,
               'wind': round(world.wind, 2), 'turbulence': round(world.turbulence, 2), 'steady': round(world.steady, 2),
               'boulder': int(world.boulder), 'boulder_size': round(world.boulder_size, 2),
               'support': round(world.support, 3), 'goal_mix': world.goal_mix,
               'best': round(max(fits), 1), 'mean': round(float(np.mean(fits)), 1),
               'champ_fit': round(champ_s['fitness'], 1), 'champ_landed': round(champ_s['landed'], 3),
               'champ_score': round(champ_s['score'], 1), 'champ_boulder': round(champ_s['boulder'], 3),
               'champ_timeout': round(champ_s['timeout'], 3), 'champ_offpad': round(champ_s['offpad'], 3),
               'champ_touchdown': '' if champ_s['touchdown'] is None else round(champ_s['touchdown'], 3),
               'hidden': hid, 'conns': con, 'species': len(self.pop.species.species),
               'threshold': round(config.species_set_config.compatibility_threshold, 3),
               'check_landed': '', 'goal_landed': '', 'goal_boulder': '', 'goal_timeout': '', 'goal_offpad': ''}
        st['steps'] += steps
        if st['stage_gen'] % a.check_every == 0:
            # the real task, every check whatever the stage (tracked on the chart)
            g = self.check(champ_net, GOAL_WORLD, rw)
            row.update(goal_landed=round(g['landed'], 3), goal_boulder=round(g['boulder'], 3),
                       goal_timeout=round(g['timeout'], 3), goal_offpad=round(g['offpad'], 3))
            if g['landed'] > st['best_goal']:
                st['best_goal'] = g['landed']
                path = self._save_champion(champ, 'goal')
                self._event('goal_best', stage=stg['name'], check=g, champion=path)
            if g['landed'] >= GOAL:
                self._goal_candidate(champ, g)
            # the stage's pass check, once the ramps are done
            if t >= 1 and not stg.get('final'):
                fw = check_world(stg)
                chk = self.check(champ_net, fw, rw)
                row['check_landed'] = round(chk['landed'], 3)
                if self.passes(chk, stg.get('pass', {})):
                    conf = self.check(champ_net, fw, rw)
                    self._event('check_pass', stage=stg['name'], check=chk, confirm=conf)
                    if self.passes(conf, stg['pass']):
                        path = self._save_champion(champ, 'passed')
                        self._event('stage_pass', stage=stg['name'], check=chk, confirm=conf, champion=path)
                        self._next_stage(passed=True)
        if stg.get('final') and st['stage_gen'] % 25 == 0:
            self._save_champion(champ, 'snap')
        dt = time.time() - t0
        row.update(steps=st['steps'], secs=round(dt, 1), steps_per_s=int(steps / dt))
        self._log(row)
        self._tune_species(config)

    def _goal_candidate(self, champ, g):
        """200 flights >= 95%: re-fly 1,000 fresh selection flights before believing it (the winner's curse)."""
        sel = self.select_flights(to_json(champ, self.config), 1000)
        ok = sel['landed'] >= GOAL
        path = self._save_champion(champ, 'candidate' if ok else 'near')
        self._event('goal_candidate' if ok else 'goal_near', check=g, selection=sel, champion=path)
        if ok:
            self.state['done'] = True
            self._event('run_goal', note='>= 95% on 1,000 selection flights; next: the two test blocks, once',
                        champion=path)

    @staticmethod
    def passes(chk, p):
        return (chk['landed'] >= p.get('landed', 2) and chk['boulder'] <= p.get('boulder', 1.0)
                and chk['timeout'] <= p.get('timeout', 1.0) and chk['offpad'] <= p.get('offpad', 1.0))

    def _tune_species(self, config):
        """Keep about `target_species` species: with 640 start connections the stock threshold gives one species."""
        n, target = len(self.pop.species.species), self.a.target_species
        th = config.species_set_config.compatibility_threshold
        th *= 0.95 if n < target else 1.05 if n > target else 1.0
        # clamped: the count only responds slowly (old species persist), and unclamped the threshold ran 0.25 -> 35 in
        # 54 generations of run N1, which would have stopped new species from forming (fixed 2026-11-01)
        config.species_set_config.compatibility_threshold = self.state['threshold'] = min(TH_MAX, max(TH_MIN, th))

    def _next_stage(self, passed):
        st = self.state
        if not passed:
            self._event('stage_cap', stage=self.stage()['name'])
        if st['stage'] + 1 >= len(self.cur['stages']):
            st['stage_gen'] = -1                       # the last stage: start it over rather than leave the CPU idle
            return
        st['stage'] += 1
        st['stage_gen'] = -1                           # incremented to 0 after this generation
        st.pop('flights_override', None)
        self._stage_start()

    def _log(self, row):
        path = os.path.join(self.dir, 'stats.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(' | '.join(f'{k} {v}' for k, v in row.items() if v != ''), flush=True)

    def _overrides(self):
        path = os.path.join(self.dir, 'override.json')
        if not os.path.exists(path):
            return
        o = json.load(open(path))
        os.remove(path)
        self._event('override', **o)
        if o.get('flights'):
            self.state['flights_override'] = int(o['flights'])
        if o.get('next_stage'):
            self._next_stage(passed=False)
            self.state['stage_gen'] += 1

    def train(self):
        with Pool(self.a.workers, initializer=_init_worker) as self.pool:
            while not self.state['done'] and not self.stop:
                self.pop.run(self.evaluate, 1)
                st = self.state
                st['gen'] += 1
                st['stage_gen'] += 1
                if st['stage_gen'] >= self.stage()['cap'] and not self.stage().get('final'):
                    self._next_stage(passed=False)
                    st['stage_gen'] += 1
                self._overrides()
                if st['gen'] % 5 == 0 or self.stop or st['done']:
                    self._save()
        self._save()
        print('goal reached' if self.state['done'] else 'stopped', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curriculum', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--config', default='neat-config.ini')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--flights', type=int, default=24)
    ap.add_argument('--reeval-top', type=int, default=10)
    ap.add_argument('--reeval-flights', type=int, default=40)
    ap.add_argument('--check-every', type=int, default=10)
    ap.add_argument('--check-flights', type=int, default=200)
    ap.add_argument('--target-species', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-gens', type=int, default=0, help='stop after this many generations (tests)')
    a = ap.parse_args()
    r = Run(a)
    if a.max_gens:
        r.state['done'] = False
        orig = r.evaluate

        def ev(genomes, config):
            orig(genomes, config)
            if r.state['gen'] + 1 >= a.max_gens:
                r.stop = True
        r.evaluate = ev
    r.train()


if __name__ == '__main__':
    main()
