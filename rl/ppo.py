"""PPO on the lander, with a curriculum, checkpoints and checks on fresh seeds.

    python -m rl.ppo --curriculum curricula/P1.json --run P1 [--init models/bc_v2_128.json] [--workers 8]

Rollouts: `workers` processes, each flying `envs` flights side by side in the reference gymnasium env (lander/env.py,
unchanged). The policy's forward pass runs in numpy inside the workers; the PPO update runs in torch in the main
process. The policy sees only Flight.inputs() (the 16 inputs); the critic sees the same.

Reward per step = env reward (the game's shaping, fuel costs, +-100 at the end) + a terminal bonus that pays for what
we want: + landing_bonus for coming to rest between the flags, - offpad_penalty for coming to rest outside them,
- timeout_penalty, - boulder_penalty. All scaled by reward_scale.

Every `check_every` iterations the deterministic policy (argmax) flies `check_flights` fresh CHECK seeds of the real
task (no help) and, if the stage's world differs, of the stage's world. Stage pass: the check passes and a second fresh
block confirms it. SIGTERM: finish the iteration, checkpoint, exit. Resume: the same command.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # one BLAS thread per process: 8 workers x 8 threads overloaded the CPU (load 34)
import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import time

import numpy as np
import torch
import torch.nn as nn

from lander.env import World, GOAL_WORLD, Flight, make_env, N_INPUTS
from lander.seeds import TRAIN, CHECK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- the network (torch for training, exported as numpy layers for flying) ----------

def mlp(n_in, hidden, n_out):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class AC(nn.Module):
    def __init__(self, hidden=(128, 128)):
        super().__init__()
        self.pi = mlp(N_INPUTS, hidden, 4)
        self.v = mlp(N_INPUTS, hidden, 1)
        self.register_buffer('mean', torch.zeros(N_INPUTS))
        self.register_buffer('std', torch.ones(N_INPUTS))

    def norm(self, x):
        return torch.clamp((x - self.mean) / self.std, -10, 10)

    def export(self):
        """numpy dict, loadable by lander.policy.NumpyMLP"""
        lin = [m for m in self.pi if isinstance(m, nn.Linear)]
        return {'layers': [[m.weight.detach().double().tolist(), m.bias.detach().double().tolist()] for m in lin],
                'mean': self.mean.double().tolist(), 'std': self.std.double().tolist(), 'clip': 10.0}

    def load_pi(self, d):
        lin = [m for m in self.pi if isinstance(m, nn.Linear)]
        assert len(lin) == len(d['layers'])
        with torch.no_grad():
            for m, (W, b) in zip(lin, d['layers']):
                m.weight.copy_(torch.tensor(W)); m.bias.copy_(torch.tensor(b))
            self.mean.copy_(torch.tensor(d['mean'])); self.std.copy_(torch.tensor(d['std']))


def np_forward(d, X):
    h = np.clip((X - d['mean']) / d['std'], -10, 10)
    for i, (W, b) in enumerate(d['layers']):
        h = h @ W.T + b
        if i < len(d['layers']) - 1:
            h = np.tanh(h)
    return h


def as_np(d):
    return {'layers': [(np.array(W), np.array(b)) for W, b in d['layers']], 'mean': np.array(d['mean']),
            'std': np.array(d['std'])}


# ---------- the world of a stage ----------

def stage_world(stage, t):
    """The stage's world at progress t (0..1 of its ramp). Support (training wheels) fades from s0 to s1 over the ramp
    and is 0 afterwards."""
    w = dict(stage.get('world', {}))
    for k, (a, b) in stage.get('ramp', {}).items():
        w[k] = a + (b - a) * min(t, 1.0)
    s0, s1 = stage.get('support', [0.0, 0.0])
    w['support'] = 0.0 if t >= 1 else s0 + (s1 - s0) * t
    return World(**w)


def check_world(stage):
    w = stage_world(stage, 1.0)
    w.boulder_mix = w.goal_mix = 0.0
    return w


def terminal_bonus(res, rw):
    if res['landed']:
        return rw.get('landing_bonus', 200.0)
    b = 0.0
    if res['rested']:
        b -= rw.get('offpad_penalty', 100.0)
    if res['outcome'] == 'timeout':
        b -= rw.get('timeout_penalty', 100.0)
    if res['outcome'] == 'boulder':
        b -= rw.get('boulder_penalty', 0.0)
    return b


# ---------- rollout workers ----------

def worker(wid, n_envs, conn):
    torch.set_num_threads(1)
    envs = [make_env() for _ in range(n_envs)]
    counters = [0] * n_envs
    flights = [None] * n_envs
    rng = np.random.default_rng(wid + 1000 * os.getpid())
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == 'close':
            return
        if cmd == 'counters':
            counters = list(msg[1]); flights = [None] * n_envs
            conn.send('ok'); continue
        if cmd == 'rollout':
            _, d, T, world, rw, scale, hard = msg
            d = as_np(d)
            obs = np.zeros((T, n_envs, N_INPUTS)); act = np.zeros((T, n_envs), np.int64)
            logp = np.zeros((T, n_envs)); rew = np.zeros((T, n_envs)); done = np.zeros((T, n_envs))
            episodes = []
            for j in range(n_envs):
                if flights[j] is None:
                    flights[j] = _new_flight(envs[j], wid, j, n_envs, counters, world, hard, rng)
            for t in range(T):
                X = np.array([f.inputs() for f in flights])
                z = np_forward(d, X)
                z = z - z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True)
                a = (p.cumsum(1) > rng.random((n_envs, 1))).argmax(1)
                obs[t], act[t], logp[t] = X, a, np.log(p[np.arange(n_envs), a] + 1e-12)
                for j, f in enumerate(flights):
                    before = f.reward
                    fin = f.step(int(a[j]))
                    r = f.reward - before
                    if fin:
                        res = f.result()
                        r += terminal_bonus(res, rw)
                        episodes.append({k: res[k] for k in ('landed', 'rested', 'outcome', 'reward', 'steps', 'seed')}
                                        | {'goal': f.world.boulder and f.world.wind == 15 and f.world.turbulence == 1.5
                                           and not f.world.support and not f.world.steady and f.world.boulder_size == 1})
                        done[t, j] = 1
                        flights[j] = _new_flight(envs[j], wid, j, n_envs, counters, world, hard, rng)
                    rew[t, j] = r * scale
            last = np.array([f.inputs() for f in flights])
            conn.send((obs, act, logp, rew, done, last, episodes, list(counters)))
        if cmd == 'eval':
            _, d, seeds, world = msg
            d = as_np(d)
            out = []
            for s in seeds:
                f = Flight(envs[0], s, world)
                fin = False
                while not fin:
                    fin = f.step(int(np_forward(d, f.inputs()[None])[0].argmax()))
                res = f.result()
                out.append({k: res[k] for k in ('seed', 'landed', 'rested', 'outcome', 'reward', 'steps', 'touch_speed')})
            conn.send(out)


def _new_flight(env, wid, j, n_envs, counters, world, hard, rng):
    """Next training flight for env j of worker wid. Seeds: TRAIN + slot * 1e6 + n, one slot per env.
    hard: optional list of seeds to re-fly (hard-case curriculum) with probability hard['p']."""
    if hard and hard['seeds'] and rng.random() < hard['p']:
        seed = int(rng.choice(hard['seeds']))
        return Flight(env, seed, GOAL_WORLD)
    slot = wid * n_envs + j
    seed = TRAIN + slot * 1_000_000 + counters[j]
    counters[j] += 1
    return Flight(env, seed, world)


# ---------- the run ----------

class Run:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join(ROOT, 'runs', a.run)
        os.makedirs(os.path.join(self.dir, 'champions'), exist_ok=True)
        self.cur = json.load(open(a.curriculum))
        json.dump(self.cur, open(os.path.join(self.dir, 'curriculum.json'), 'w'), indent=1)
        torch.manual_seed(a.seed)
        self.net = AC(tuple(a.hidden))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=a.lr, eps=1e-5)
        self.stop = False
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)
        ck = os.path.join(self.dir, 'checkpoint.pt')
        if os.path.exists(ck):
            blob = torch.load(ck, weights_only=False)
            self.net.load_state_dict(blob['net']); self.opt.load_state_dict(blob['opt'])
            self.state = blob['state']
            print(f"resumed at iteration {self.state['it']}, stage {self.stage()['name']}", flush=True)
        else:
            if a.init:
                self.net.load_pi(json.load(open(a.init)))
            if a.resume_from:          # continue another run's network + optimiser, with a fresh curriculum state
                blob = torch.load(a.resume_from, weights_only=False)
                self.net.load_state_dict(blob['net']); self.opt.load_state_dict(blob['opt'])
            self.state = {'it': 0, 'stage': 0, 'stage_it': 0, 'n_check': 0, 'steps': 0, 'best_goal': -1.0,
                          'norm_done': bool(a.init or a.resume_from), 'counters': None, 'done': False, 'hard': []}
            self._event('run_start', note=self.cur.get('note', ''), args=vars(a))
            self._event('stage_start', stage=self.stage()['name'], expect=self.stage().get('expect', ''))

    def _sig(self, *_):
        print('signal: checkpoint after this iteration', flush=True)
        self.stop = True

    def stage(self):
        return self.cur['stages'][self.state['stage']]

    def _event(self, kind, **kw):
        e = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind, 'it': self.state['it'] if hasattr(self, 'state') else 0,
             'steps': self.state['steps'] if hasattr(self, 'state') else 0,
             'stage_index': self.state['stage'] if hasattr(self, 'state') else 0, **kw}
        with open(os.path.join(self.dir, 'events.jsonl'), 'a') as f:
            f.write(json.dumps(e) + '\n')

    def _save(self):
        tmp = os.path.join(self.dir, 'checkpoint.pt.tmp')
        torch.save({'net': self.net.state_dict(), 'opt': self.opt.state_dict(), 'state': self.state}, tmp)
        os.replace(tmp, os.path.join(self.dir, 'checkpoint.pt'))

    def _champion(self, label):
        path = os.path.join(self.dir, 'champions', f"i{self.state['it']:05d}_s{self.state['stage']}_{label}.json")
        json.dump(self.net.export(), open(path, 'w'))
        return os.path.relpath(path, ROOT)

    def _log(self, row):
        path = os.path.join(self.dir, 'stats.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(' | '.join(f'{k} {v}' for k, v in row.items() if v != ''), flush=True)

    # --- workers ---
    def start_workers(self):
        a = self.a
        self.conns, self.procs = [], []
        for w in range(a.workers):
            p_conn, c_conn = mp.Pipe()
            p = mp.Process(target=worker, args=(w, a.envs, c_conn), daemon=True)
            p.start()
            self.conns.append(p_conn); self.procs.append(p)
        if self.state['counters']:
            for c, cnt in zip(self.conns, self.state['counters']):
                c.send(('counters', cnt)); c.recv()

    def evaluate(self, seeds, world):
        d = self.net.export()
        n = len(self.conns)
        parts = [seeds[i::n] for i in range(n)]
        for c, p in zip(self.conns, parts):
            c.send(('eval', d, p, world))
        out = []
        for c in self.conns:
            out += c.recv()
        return out

    def check(self, world, n=None):
        n = n or self.a.check_flights
        seeds = [CHECK + self.state['n_check'] * 1000 + i for i in range(n)]
        self.state['n_check'] += 1
        rs = self.evaluate(seeds, world)
        return summarise(rs), rs

    # --- one iteration ---
    def rollout(self, world, rw, hard):
        d = self.net.export()
        for c in self.conns:
            c.send(('rollout', d, self.a.T, world, rw, self.a.reward_scale, hard))
        outs = [c.recv() for c in self.conns]
        self.state['counters'] = [o[7] for o in outs]
        cat = lambda i: np.concatenate([o[i] for o in outs], axis=1)
        return cat(0), cat(1), cat(2), cat(3), cat(4), np.concatenate([o[5] for o in outs]), sum((o[6] for o in outs), [])

    def update(self, obs, act, logp, rew, done, last, critic_only):
        a = self.a
        T, N = rew.shape
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32)
            v = self.net.v(self.net.norm(o)).squeeze(-1).numpy()
            vlast = self.net.v(self.net.norm(torch.tensor(last, dtype=torch.float32))).squeeze(-1).numpy()
        adv = np.zeros((T, N)); g = np.zeros(N)
        for t in reversed(range(T)):
            nv = vlast if t == T - 1 else v[t + 1]
            nonterm = 1 - done[t]
            delta = rew[t] + a.gamma * nv * nonterm - v[t]
            g = delta + a.gamma * a.lam * nonterm * g
            adv[t] = g
        ret = adv + v
        O = o.reshape(T * N, -1)
        A = torch.tensor(act.reshape(-1)); LP = torch.tensor(logp.reshape(-1), dtype=torch.float32)
        ADV = torch.tensor(adv.reshape(-1), dtype=torch.float32); RET = torch.tensor(ret.reshape(-1), dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        M = T * N
        kls, clips, ents, vls = [], [], [], []
        for ep in range(a.epochs):
            idx = torch.randperm(M)
            for s in range(0, M, a.minibatch):
                b = idx[s:s + a.minibatch]
                x = self.net.norm(O[b])
                vl = ((self.net.v(x).squeeze(-1) - RET[b]) ** 2).mean()
                if critic_only:
                    loss = a.vf * vl
                    ent = torch.tensor(0.0); kl = torch.tensor(0.0); cf = torch.tensor(0.0)
                else:
                    logits = self.net.pi(x)
                    lp_all = torch.log_softmax(logits, -1)
                    lp = lp_all.gather(1, A[b][:, None]).squeeze(1)
                    ratio = torch.exp(lp - LP[b])
                    pg = -torch.min(ratio * ADV[b], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADV[b]).mean()
                    ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                    loss = pg + a.vf * vl - a.ent * ent
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean()
                        cf = ((ratio - 1).abs() > a.clip).float().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                kls.append(kl.item()); clips.append(cf.item()); ents.append(ent.item()); vls.append(vl.item())
            if not critic_only and a.target_kl and np.mean(kls[-(M // a.minibatch):]) > a.target_kl:
                break
        return {'kl': float(np.mean(kls)), 'clipfrac': float(np.mean(clips)), 'entropy': float(np.mean(ents)),
                'vloss': float(np.mean(vls))}

    def train(self):
        a = self.a
        self.start_workers()
        st = self.state
        recent = []
        try:
            t_start = time.time()
            while not st['done'] and not self.stop:
                if a.max_minutes and time.time() - t_start > a.max_minutes * 60:
                    self._event('time_limit', minutes=a.max_minutes)
                    break
                t0 = time.time()
                stg = self.stage()
                cap = stg['cap']
                ramp_len = max(1, int(stg.get('ramp_frac', 0.6) * cap))
                tt = st['stage_it'] / ramp_len if (stg.get('ramp') or stg.get('support')) else 1.0
                world = stage_world(stg, tt)
                for k in ('lr', 'ent'):            # per-stage learning rate / entropy bonus
                    if k in stg and getattr(a, k) != stg[k]:
                        setattr(a, k, stg[k])
                        if k == 'lr':
                            for g in self.opt.param_groups:
                                g['lr'] = stg[k]
                rw = stg.get('reward', {})
                hard = {'seeds': st['hard'], 'p': stg.get('hard_p', 0.0)} if stg.get('hard_p') else None
                obs, act, logp, rew, done, last, eps = self.rollout(world, rw, hard)
                st['steps'] += rew.size
                if not st['norm_done']:
                    with torch.no_grad():
                        flat = obs.reshape(-1, obs.shape[-1])
                        self.net.mean.copy_(torch.tensor(flat.mean(0), dtype=torch.float32))
                        self.net.std.copy_(torch.tensor(flat.std(0) + 1e-3, dtype=torch.float32))
                    st['norm_done'] = True
                    continue        # the samples were taken before the normalisation existed
                info = self.update(obs, act, logp, rew, done, last, critic_only=st['it'] < a.critic_warmup)
                recent = (recent + eps)[-400:]
                goal_eps = [e for e in recent if e['goal']]
                row = {'it': st['it'], 'steps': st['steps'], 'stage': stg['name'], 'wind': round(world.wind, 2),
                       'turbulence': round(world.turbulence, 2), 'boulder': int(world.boulder),
                       'support': round(world.support, 3), 'episodes': len(eps),
                       'train_landed': round(np.mean([e['landed'] for e in recent]), 3) if recent else '',
                       'train_goal_landed': round(np.mean([e['landed'] for e in goal_eps]), 3) if goal_eps else '',
                       'train_score': round(np.mean([e['reward'] for e in recent]), 1) if recent else '',
                       **{k: round(v, 4) for k, v in info.items()}, 'check_landed': '', 'goal_landed': '',
                       'goal_offpad': '', 'goal_timeout': '', 'goal_boulder': '', 'secs': 0}
                if st['it'] % a.check_every == 0 and st['it'] >= a.critic_warmup:
                    self._checks(row, stg, tt)
                row['secs'] = round(time.time() - t0, 1)
                self._log(row)
                st['it'] += 1
                st['stage_it'] += 1
                if st['stage_it'] >= cap and not st['done']:
                    self._next_stage(passed=False)
                self._overrides()
                if st['it'] % 10 == 0:
                    self._save()
                if a.snapshot_every and st['it'] % a.snapshot_every == 0:
                    self._champion('snap')         # archive for selection on SELECT seeds later
        finally:
            self._save()
            json.dump(self.net.export(), open(os.path.join(self.dir, 'final.json'), 'w'))
            for c in self.conns:
                try:
                    c.send(('close',))
                except Exception:
                    pass
        print('stopped' if self.stop else 'curriculum done', flush=True)

    def _checks(self, row, stg, tt):
        st = self.state
        g, rs = self.check(GOAL_WORLD)
        row.update(goal_landed=round(g['landed'], 3), goal_offpad=round(g['offpad'], 3),
                   goal_timeout=round(g['timeout'], 3), goal_boulder=round(g['boulder'], 3))
        if stg.get('hard_p'):
            st['hard'] = (st['hard'] + [r['seed'] for r in rs if not r['landed']])[-300:]
        if g['landed'] > st['best_goal']:
            st['best_goal'] = g['landed']
            path = self._champion('goal')
            self._event('goal_best', stage=stg['name'], check=g, champion=path)
        if tt < 1:
            return
        final = check_world(stg)
        chk = g if final == GOAL_WORLD else self.check(final)[0]
        row['check_landed'] = round(chk['landed'], 3)
        if passes(chk, stg['pass']):
            conf = self.check(final)[0]
            self._event('check_pass', stage=stg['name'], check=chk, confirm=conf)
            if passes(conf, stg['pass']):
                path = self._champion('passed')
                self._event('stage_pass', stage=stg['name'], check=chk, confirm=conf, champion=path)
                self._next_stage(passed=True)

    def _next_stage(self, passed):
        st = self.state
        if not passed:
            self._event('stage_cap', stage=self.stage()['name'])
        if st['stage'] + 1 >= len(self.cur['stages']):
            st['done'] = True
            self._event('curriculum_done', passed=passed)
            return
        st['stage'] += 1
        st['stage_it'] = -1
        self._event('stage_start', stage=self.stage()['name'], expect=self.stage().get('expect', ''))

    def _overrides(self):
        path = os.path.join(self.dir, 'override.json')
        if os.path.exists(path):
            o = json.load(open(path))
            os.remove(path)
            if o.get('next_stage'):
                self._event('override', what='next_stage', reason=o.get('reason', ''))
                self._next_stage(passed=False)
            if o.get('lr'):
                for g in self.opt.param_groups:
                    g['lr'] = o['lr']
                self.a.lr = o['lr']
                self._event('override', what=f"lr {o['lr']}", reason=o.get('reason', ''))


def summarise(rs):
    n = len(rs)
    f = lambda c: sum(c(r) for r in rs) / n
    return {'n': n, 'landed': f(lambda r: r['landed']), 'offpad': f(lambda r: r['rested'] and not r['landed']),
            'timeout': f(lambda r: r['outcome'] == 'timeout'), 'boulder': f(lambda r: r['outcome'] == 'boulder'),
            'crashed': f(lambda r: r['outcome'] == 'crashed'), 'score': float(np.mean([r['reward'] for r in rs]))}


def passes(chk, p):
    return (chk['landed'] >= p.get('landed', 0) and chk['timeout'] <= p.get('timeout', 1.0)
            and chk['offpad'] <= p.get('offpad', 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curriculum', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--init', help='numpy policy json to start the actor from (warm start)')
    ap.add_argument('--hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--envs', type=int, default=8, help='flights per worker')
    ap.add_argument('--T', type=int, default=256, help='steps per env per iteration')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--vf', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=2048)
    ap.add_argument('--target-kl', type=float, default=0.03)
    ap.add_argument('--reward-scale', type=float, default=0.01)
    ap.add_argument('--critic-warmup', type=int, default=0, help='iterations that train only the critic')
    ap.add_argument('--check-every', type=int, default=10)
    ap.add_argument('--check-flights', type=int, default=200)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume-from', help='checkpoint.pt of another run to continue from')
    ap.add_argument('--snapshot-every', type=int, default=0, help='save the policy every N iterations')
    ap.add_argument('--max-minutes', type=float, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    Run(a).train()


if __name__ == '__main__':
    main()
