"""Build the film "What PPO does that NEAT does not" (docs/video/ppo_vs_neat.mp4, 1280x720, 25 fps).

    .venv/bin/python video/ppo_vs_neat/build_film.py [--only 4,5] [--preview]

Needs: the voiceover exported (voice/film-ppo-vs-neat/audio/voiceover/chNN.mp3, one per ## scene of script.md), and the
flight data from extract_ppo.py (run in ../lunarlander-98: the real PPO actor and critic) and extract_neat.py.
Every number drawn comes from those files, the run's stats and results/baseline_N1_*.json, or the old projects'
READMEs (cited in the scene functions).
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from lander.viz import font

FPS, SW, SH, GAP = 25, 1280, 720, 0.7
BG, PANEL, LINE = (14, 17, 22), (24, 28, 36), (52, 58, 70)
INK, MUTED = (232, 232, 236), (150, 152, 160)
PPO_C, NEAT_C = (120, 160, 255), (255, 169, 77)
GOOD, BAD = (105, 219, 124), (255, 120, 120)
HERE = 'video/ppo_vs_neat'
B = f'{HERE}/build'
VOICE = 'voice/film-ppo-vs-neat/audio/voiceover'
OUT = 'docs/video'
ACTIONS = ['do nothing', 'left engine', 'main engine', 'right engine']

PPO = json.load(open(f'{B}/ppo/ppo_flight.json'))
NEAT = json.load(open(f'{B}/neat/neat_flight.json'))
N_PPO = len(glob.glob(f'{B}/ppo/f*.png'))
N_NEAT = len(glob.glob(f'{B}/neat/f*.png'))


# ---------------- helpers ----------------

def canvas():
    return Image.new('RGB', (SW, SH), BG)


def wrap(d, s, f, width):
    words, lines, cur = s.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def text(d, xy, s, size=24, fill=INK, bold=False, width=None, anchor=None):
    f = font(size, bold)
    x, y = xy
    for line in (wrap(d, s, f, width) if width else [s]):
        d.text((x, y), line, fill=fill, font=f, anchor=anchor)
        y += int(size * 1.35)
    return y


def header(d, title, sub=None):
    text(d, (48, 28), title, 34, bold=True)
    if sub:
        text(d, (48, 74), sub, 20, MUTED)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


_frames = {}


def frame(kind, i, size):
    """Rendered flight frame i (held on the last frame: replays never loop), scaled to size."""
    n = N_PPO if kind == 'ppo' else N_NEAT
    i = min(i, n - 1)
    key = (kind, i, size)
    if key not in _frames:
        if len(_frames) > 400:
            _frames.clear()
        _frames[key] = Image.open(f'{B}/{kind}/f{i:04d}.png').convert('RGB').resize(size, Image.LANCZOS)
    return _frames[key]


def flight_step(t, n, start=0.0, speed=1.0):
    """Which step of a flight to show at time t (s): real time is 50 steps/s; here played at `speed` x 25 steps/s."""
    return int(max(0.0, t - start) * FPS * speed)


def bar(d, x, y, w, h, frac, col, bgcol=LINE):
    d.rectangle([x, y, x + w, y + h], fill=bgcol)
    if frac > 0:
        d.rectangle([x, y, x + int(w * min(1, frac)), y + h], fill=col)


# ---------------- data from the run ----------------

def best_measured():
    rows = []
    for p in glob.glob('results/baseline_N1_g*.json'):
        r = json.load(open(p))
        rows.append((int(p.split('_g')[-1][:5]), r['landed']))
    return sorted(rows)


def flights_timeline():
    out, last = [], None
    for r in csv.DictReader(open('runs/N1/stats.csv')):
        if r['flights'] != last:
            out.append((int(r['gen']), int(r['flights']))); last = r['flights']
    return out


def total_steps():
    last = None
    for r in csv.DictReader(open('runs/N1/stats.csv')):
        last = r
    return int(last['steps'])


# ---------------- scenes ----------------

def s01_two_pilots(n):
    """PPO (lunarlander-98 M2 final state) and NEAT (gen 3832) on the same seed of the real task (v2 vs v3 layout)."""
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'What PPO does that NEAT does not', 'Two pilots, the same kind of storm: wind 15, turbulence 1.5, a boulder')
        for k, (kind, x, col, name, sub) in enumerate([
                ('ppo', 40, PPO_C, 'PPO', 'last project: 98.8% / 98.4% on two test sets'),
                ('neat', 660, NEAT_C, 'NEAT', 'this project: best 93.0% on 1,000 flights')]):
            img.paste(frame(kind, flight_step(t, 0, 1.0, 1.0), (580, 387)), (x, 150))
            d.rectangle([x, 150, x + 580, 537], outline=col, width=3)
            text(d, (x, 552), name, 30, col, bold=True)
            text(d, (x, 592), sub, 20, MUTED)
        yield img


def s02_scoreboard(n):
    """History: lunarlander-neat-wind README (best run D 85%), lunarlander-curriculum (NEAT ~85%),
    lunarlander-98 CONTINUE.md (98.8 / 98.4, 254 M steps, ~4.5 h), this run (best measured, steps from stats.csv)."""
    best = max(l for _, l in best_measured())
    rows = [('NEAT in wind (earlier project)', 0.85, NEAT_C), ('NEAT + curriculum + cloned pilot', 0.85, NEAT_C),
            ('PPO (last project)', 0.986, PPO_C), ('NEAT, this project', best, NEAT_C)]
    steps = [('PPO', 254e6, PPO_C), ('NEAT', total_steps(), NEAT_C)]
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'The scoreboard', 'Landed between the flags, measured on unseen flights')
        for k, (name, v, col) in enumerate(rows):
            y = 140 + k * 62
            a = ease((t - [3, 4, 9, 14][k]) / 1.5)
            text(d, (48, y + 6), name, 22)
            bar(d, 420, y, 560, 34, a * v, col)
            if a > 0.05:
                text(d, (995, y + 4), f'{100 * v * a:.1f}%', 24, col, bold=True)
        if t > 19:
            a = ease((t - 19) / 3)
            text(d, (48, 410), 'Simulated steps used (log scale)', 24, bold=True)
            for k, (name, v, col) in enumerate(steps):
                y = 460 + k * 70
                frac = (math.log10(v) - 7) / (math.log10(5e10) - 7)
                text(d, (48, y + 6), name, 24, col, bold=True)
                bar(d, 160, y, 820, 40, a * frac, col)
                if a > 0.3:
                    lab = f'{v / 1e6:,.0f} million' if v < 1e9 else f'{v / 1e9:,.1f} billion'
                    text(d, (995, y + 6), lab, 24, col, bold=True)
            if t > 31:
                text(d, (48, 620), f'NEAT has used about {total_steps() / 254e6:,.0f}x the experience of PPO, for a worse pilot.', 24, MUTED)
        yield img


def s03_neat_learns(n):
    """300 genomes, one fitness each (same flights), parents, children with random changes."""
    import random
    rnd = random.Random(3)
    fit = [rnd.gauss(0, 1) for _ in range(300)]
    order = sorted(range(300), key=lambda k: -fit[k])
    top = set(order[:30])
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'How NEAT learns', 'A population of 300 networks; each gets ONE number per generation')
        phase = 0 if t < 8.4 else 1 if t < 13.4 else 2
        for k in range(300):
            cx, cy = 80 + (k % 25) * 30, 150 + (k // 25) * 30
            if phase == 0:
                col = (90, 96, 110)
            else:
                v = (fit[k] + 2.5) / 5
                col = tuple(int(c * (0.3 + 0.7 * max(0, min(1, v)))) for c in NEAT_C)
            if phase >= 2 and k not in top:
                col = (50, 54, 62)
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=col)
            if phase == 1 and k % 37 == 0:
                text(d, (cx - 14, cy - 30), f'{200 + 60 * fit[k]:.0f}', 13, INK)
        x = 860
        steps = [('1. Every network flies the same flights.', 3.7), ('2. A whole flight becomes ONE fitness number.', 8.4),
                 ('3. The best become parents.', 13.4), ('4. Children: copies with random changes - a weight nudged, '
                                                     'a connection or neuron added or removed.', 15.1)]
        y = 160
        for s, ts in steps:
            y = text(d, (x, y), s, 22, INK if t >= ts else (70, 74, 84), width=380) + 18
        if t > 23.9:
            text(d, (x, y + 20), 'It never asks WHICH moment of a flight went well.', 22, BAD, bold=True, width=380)
        yield img


def ppo_panel(img, d, t, speed=0.6, x=40, y=130, w=600):
    h = int(w * 2 / 3)
    k = min(flight_step(t, 0, 0.5, speed), N_PPO - 1)
    img.paste(frame('ppo', k, (w, h)), (x, y))
    d.rectangle([x, y, x + w, y + h], outline=PPO_C, width=2)
    return min(k, len(PPO['steps']) - 1)


def s04_every_step(n):
    """The real PPO actor's action probabilities along this flight (extract_ppo.py), greedy flight."""
    S = PPO['steps']
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'PPO: every step counts', f'This flight has {len(S)} steps; PPO records each one')
        k = ppo_panel(img, d, t, speed=0.27)
        s = S[k]
        x0 = 680
        text(d, (x0, 135), f'Step {k + 1} of {len(S)}: what it records', 24, bold=True)
        text(d, (x0, 175), 'what it saw (16 numbers), which engine it fired,', 20, MUTED)
        text(d, (x0, 202), 'how likely that choice was, and the reward after it', 20, MUTED)
        for j, name in enumerate(ACTIONS):
            yy = 250 + j * 52
            text(d, (x0, yy + 4), name, 22, INK if j == s['a'] else MUTED, bold=j == s['a'])
            bar(d, x0 + 170, yy, 330, 30, s['p'][j], PPO_C if j == s['a'] else (80, 100, 150))
            text(d, (x0 + 512, yy + 3), f"{100 * s['p'][j]:.0f}%", 20)
        text(d, (x0, 470), f"reward this step: {100 * s['r']:+.1f} points", 22, GOOD if s['r'] >= 0 else BAD)
        # probability history strip
        d.rectangle([40, 560, 1240, 690], fill=PANEL)
        cols = [(110, 110, 120), (255, 120, 120), PPO_C, GOOD]
        wstep = 1200 / len(S)
        for j in range(k + 1):
            y = 560
            for a in range(4):
                hh = 130 * S[j]['p'][a]
                d.rectangle([40 + j * wstep, y, 40 + (j + 1) * wstep, y + hh], fill=cols[a])
                y += hh
        for a, name in enumerate(ACTIONS):
            d.rectangle([680 + a * 140, 532, 694 + a * 140, 546], fill=cols[a])
            text(d, (700 + a * 140, 528), name, 16, MUTED)
        text(d, (40, 528), 'Probabilities of the 4 actions, step by step', 16, MUTED)
        yield img


def value_plot(d, k, show_adv=False, box=(680, 130, 1240, 520)):
    S = PPO['steps']
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=PANEL)
    vs = [100 * s['v'] for s in S]
    lo, hi = 100, 320
    Y = lambda v: y1 - 20 - (v - lo) / (hi - lo) * (y1 - y0 - 50)
    X = lambda j: x0 + 50 + j * (x1 - x0 - 70) / (len(S) - 1)
    for v in (150, 200, 250, 300):
        d.line([x0 + 50, Y(v), x1 - 20, Y(v)], fill=LINE)
        text(d, (x0 + 8, Y(v) - 10), str(v), 14, MUTED)
    pts = [(X(j), Y(vs[j])) for j in range(k + 1)]
    if len(pts) > 1:
        d.line(pts, fill=PPO_C, width=3)
    text(d, (x0 + 12, y0 + 10), "Critic: points expected from here to the end", 18, INK, bold=True)
    return X


def s05_critic(n):
    """V(s) of the real critic (runs/M2/checkpoint.pt) x 100 (reward scale 0.01), incl. the +200 landing bonus."""
    S = PPO['steps']
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'The critic: a second network', 'It predicts how many points the rest of the flight will bring')
        k = ppo_panel(img, d, t, speed=0.3)
        value_plot(d, k)
        text(d, (680, 540), f"now: {100 * S[k]['v']:.0f} points expected", 26, PPO_C, bold=True)
        text(d, (40, 560), 'Like a coach who knows from experience how promising each moment is.', 22, MUTED, width=600)
        yield img


def s06_advantage(n):
    """GAE advantages of this flight, computed as rl/ppo.py does (gamma 0.995, lambda 0.95)."""
    S = PPO['steps']
    A = [100 * s['adv'] for s in S]
    m = max(abs(a) for a in A)
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'Advantage: better or worse than expected?', 'Generalized advantage estimation, gamma 0.995, lambda 0.95')
        k = ppo_panel(img, d, t, speed=0.18)
        box = (680, 130, 1240, 520)
        d.rectangle(box, fill=PANEL)
        mid = 330
        d.line([700, mid, 1230, mid], fill=LINE)
        wstep = 520 / len(S)
        for j in range(k + 1):
            h = A[j] / m * 170
            d.rectangle([710 + j * wstep, min(mid, mid - h), 710 + (j + 1) * wstep, max(mid, mid - h)],
                        fill=GOOD if A[j] > 0 else BAD)
        text(d, (692, 140), 'better than expected: make this action MORE likely', 18, GOOD)
        text(d, (692, 495), 'worse than expected: make it LESS likely', 18, BAD)
        text(d, (680, 540), f"step {k + 1}: advantage {A[k]:+.1f} points", 26, GOOD if A[k] > 0 else BAD, bold=True)
        text(d, (40, 560), 'advantage = what happened (the rewards, then the critic\'s view a little later) '
                           '- what the critic expected now', 20, MUTED, width=600)
        yield img


def s07_gradients(n):
    """Actor size from extract_ppo.py (19,204 parameters); valley analogy."""
    import random
    rnd = random.Random(7)
    darts = [(rnd.uniform(-1, 1), None) for _ in range(14)]
    f = lambda x: x * x
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'Gradients: every weight gets a direction', f"Backpropagation, for all {PPO['params']:,} weights of the pilot at once")
        for side, (x0, name, col) in enumerate([(40, 'PPO: feels the slope', PPO_C), (660, 'NEAT: random tries, keep the best', NEAT_C)]):
            d.rectangle([x0, 130, x0 + 580, 560], fill=PANEL)
            text(d, (x0 + 20, 145), name, 26, col, bold=True)
            X = lambda x: x0 + 290 + x * 240
            Y = lambda y: 520 - y * 300
            d.line([(X(x / 50), Y(f(x / 50))) for x in range(-50, 51)], fill=MUTED, width=3)
            if side == 0:
                x = -0.95
                for _ in range(int(t * 3)):
                    x -= 0.12 * 2 * x
                d.ellipse([X(x) - 12, Y(f(x)) - 24, X(x) + 12, Y(f(x))], fill=col)
                if t > 1 and abs(x) > 0.02:
                    d.line([X(x), Y(f(x)) - 12, X(x - 0.25 * x / abs(x)), Y(f(x)) - 12], fill=INK, width=3)
            else:
                shown = min(len(darts), int(max(0, t - 17) * 1.5))
                best = None
                for k in range(shown):
                    x = darts[k][0]
                    best = x if best is None or f(x) < f(best) else best
                    d.ellipse([X(x) - 6, Y(f(x)) - 6, X(x) + 6, Y(f(x)) + 6], outline=col, width=2)
                if best is not None:
                    d.ellipse([X(best) - 12, Y(f(best)) - 24, X(best) + 12, Y(f(best))], fill=col)
        text(d, (40, 590), 'PPO moves every weight a tiny bit in its own best direction, using the advantages.', 22, width=1200)
        if t > 17:
            text(d, (40, 630), 'NEAT changes weights at random and keeps whatever happened to score well.', 22, MUTED, width=1200)
        yield img


def s08_clip(n):
    """L^CLIP per step as in Schulman et al. 2017, Fig. 1; epsilon 0.2; batch 8 workers x 8 flights x 256 steps,
    4 epochs, early stop at KL 0.03 (lunarlander-98 rl/ppo.py defaults)."""
    eps = 0.2
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'Small, careful steps: the clip', 'ratio = new probability of the action / old probability')
        for side, (x0, A, title) in enumerate([(40, 1, 'advantage > 0 (good action)'), (660, -1, 'advantage < 0 (bad action)')]):
            d.rectangle([x0, 130, x0 + 580, 470], fill=PANEL)
            text(d, (x0 + 20, 142), title, 22, GOOD if A > 0 else BAD, bold=True)
            X = lambda r: x0 + 60 + r / 2.0 * 480
            Y = lambda v: 300 - v * 68
            lo, hi = X(1 - eps), X(1 + eps)
            d.rectangle([lo, 180, hi, 440], fill=(34, 44, 60))
            d.line([X(0), Y(0), X(2), Y(0)], fill=LINE, width=1)
            for r, lab in [(1 - eps, '0.8'), (1, '1'), (1 + eps, '1.2')]:
                d.line([X(r), 180, X(r), 440], fill=LINE)
                text(d, (X(r) - 12, 445), lab, 16, MUTED)
            pts = []
            for q in range(0, 201):
                r = q / 100
                v = min(r * A, max(min(r, 1 + eps), 1 - eps) * A)
                pts.append((X(r), Y(v)))
            reveal = int(len(pts) * ease(t / 6))
            if reveal > 1:
                d.line(pts[:reveal], fill=PPO_C, width=4)
            text(d, (x0 + 20, 180), 'what PPO gains', 16, MUTED)
        notes = [('Inside the window (here 20% either way) PPO keeps pushing.', 17.4),
                 ('Outside it, there is nothing more to gain: PPO stops pushing.', 22.2),
                 ('So the same batch can be reused: ~16,000 steps, 4 passes, stop early if the pilot changes too much.', 30.5),
                 ('NEAT uses each flight once and throws it away.', 43.5)]
        y = 500
        for s, ts in notes:
            if t >= ts:
                y = text(d, (40, y), s, 22, NEAT_C if 'NEAT' in s else INK, width=1200) + 6
        yield img


def s09_explore(n):
    S = PPO['steps']
    mid = S[len(S) // 3]
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'Exploring', 'PPO explores in what it DOES; NEAT explores in what it IS')
        d.rectangle([40, 130, 620, 600], fill=PANEL)
        text(d, (60, 145), 'PPO: rolls the dice every step', 24, PPO_C, bold=True)
        for j, name in enumerate(ACTIONS):
            yy = 210 + j * 60
            text(d, (60, yy + 4), name, 20)
            bar(d, 220, yy, 300, 30, mid['p'][j], PPO_C)
            text(d, (530, yy + 3), f"{100 * mid['p'][j]:.0f}%", 20)
        roll = int(t * 4) % 4
        if roll is not None:
            import random
            r = random.Random(int(t * 4)).random()
            a = next(j for j in range(4) if r < sum(mid['p'][:j + 1]))
            d.rectangle([56, 206 + a * 60, 600, 244 + a * 60], outline=INK, width=2)
        text(d, (60, 470), 'Entropy bonus: a small reward for not becoming too certain too early.', 20, MUTED, width=540)
        if t < 15.8:
            yield img
            continue
        d.rectangle([660, 130, 1240, 600], fill=PANEL)
        text(d, (680, 145), 'NEAT: every child is a different network', 24, NEAT_C, bold=True)
        import random
        rr = random.Random(5)
        for c in range(6):
            cx, cy = 760 + (c % 3) * 170, 260 + (c // 3) * 170
            pts = [(cx + rr.uniform(-60, 60), cy + rr.uniform(-50, 50)) for _ in range(6)]
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    if rr.random() < 0.35:
                        d.line([pts[a], pts[b]], fill=(150, 110, 60), width=2)
            for p in pts:
                d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=NEAT_C)
        text(d, (680, 540), 'Each flies a whole flight before anything is learned.', 20, MUTED, width=540)
        yield img


def s10_neat_strengths(n):
    """Sizes: PPO actor 19,204 parameters (extract_ppo.py); NEAT gen 3832: 41 hidden, 110 connections (+45 biases).
    Salimans et al. 2017: ES competitive with RL using >1,000 parallel workers."""
    neat_w = 110 + 45
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'What NEAT has that PPO does not')
        a = ease(t / 6)
        d.rectangle([40, 110, 700, 610], fill=PANEL)
        text(d, (60, 120), f"PPO pilot: {PPO['params']:,} weights", 22, PPO_C, bold=True)
        text(d, (60, 470), f"NEAT pilot: 41 hidden neurons, 110 connections ({neat_w} numbers)", 22, NEAT_C, bold=True, width=620)
        cols, c = 200, 3                         # one 2x2 dot per weight, both pilots on the same scale
        nshow = int(PPO['params'] * a)
        for k in range(nshow):
            x, y = 60 + (k % cols) * c, 160 + (k // cols) * c
            d.rectangle([x, y, x + 1, y + 1], fill=PPO_C)
        for k in range(int(neat_w * ease((t - 3) / 2))):
            x, y = 60 + (k % cols) * c, 530 + (k // cols) * c
            d.rectangle([x, y, x + 1, y + 1], fill=NEAT_C)
        items = [('Grows its own structure', 3.9), ('No gradients, no critic, no reward per step: only a final score', 14.1),
                 ('Every network can fly on its own processor', 24.3), ('Species protect new ideas', 29.2),
                 ('With 1,000+ processors, evolution can compete with RL (Salimans et al., 2017)', 32.7),
                 ('On one laptop: the price is one number per flight', 38.7)]
        y = 130
        for s, ts in items:
            if t >= ts:
                y = text(d, (730, y), '- ' + s, 21, BAD if 'price' in s else INK, width=500) + 14
        yield img


def s11_why(n):
    """Flights per genome from stats.csv; best measured champions from results/baseline_N1_g*.json."""
    tl = flights_timeline()
    bm = [(g, l) for g, l in best_measured() if g >= 2000]
    g0, g1 = 2000, max(g for g, _ in bm) + 50
    for i in range(n):
        t = i / FPS
        img = canvas(); d = ImageDraw.Draw(img)
        header(d, 'Why this matters here', 'NEAT this week: more flights per network each time noise hid the progress')
        box = (40, 120, 1240, 470)
        d.rectangle(box, fill=PANEL)
        X = lambda g: 100 + (g - g0) / (g1 - g0) * 1100
        Y = lambda l: 440 - (l - 0.80) / 0.15 * 290
        for l in (0.85, 0.90, 0.95):
            d.line([100, Y(l), 1220, Y(l)], fill=LINE if l < 0.95 else BAD)
            text(d, (50, Y(l) - 10), f'{100 * l:.0f}%', 16, BAD if l == 0.95 else MUTED)
        for k, (g, fl) in enumerate(tl):
            if g >= g0 and fl >= 32:
                d.line([X(g), 130, X(g), 460], fill=(70, 80, 100))
                text(d, (X(g) + 6, 132), f'{fl} flights', 16, NEAT_C)
        reveal = ease(t / 20)
        shown = [(g, l) for g, l in bm if X(g) <= 100 + reveal * 1100]
        for g, l in shown:
            d.ellipse([X(g) - 6, Y(l) - 6, X(g) + 6, Y(l) + 6], fill=NEAT_C)
        text(d, (110, 445), 'NEAT champions measured on the same 1,000 unseen flights (generation 2000 onward)', 15, MUTED)
        lines = [('PPO learns from every moment,', 29), ('knows which moments mattered,', 30.6),
                 ('moves every weight in a useful direction,', 31.9), ('and takes careful steps.', 34.6),
                 ('NEAT learns from whole flights, by trial and selection.', 35.5),
                 ('Both can land in a storm. One needs a lot more storms.', 38.5)]
        y = 500
        for s, ts in lines:
            if t >= ts:
                y = text(d, (40, y), s, 24, NEAT_C if 'NEAT' in s else PPO_C if 'PPO' in s or s[0].islower() or s.startswith('and') else INK,
                         bold='storms' in s) + 2
        yield img


SCENES = [s01_two_pilots, s02_scoreboard, s03_neat_learns, s04_every_step, s05_critic, s06_advantage, s07_gradients,
          s08_clip, s09_explore, s10_neat_strengths, s11_why]


# ---------------- assembly ----------------

def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout
    return float(out)


def render_scene(k, fn, preview=False):
    a = f'{VOICE}/ch{k:02d}.mp3'
    a = a if os.path.exists(a) else None
    dur = (duration(a) if a else 20.0) + GAP
    n = int(dur * FPS)
    out = f'{B}/scene{k:02d}.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SW}x{SH}', '-r', str(FPS), '-i', '-']
    if a:
        cmd += ['-i', a, '-af', f'apad=pad_dur={GAP + 1}', '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '1']
    cmd += ['-t', f'{n / FPS:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-r', str(FPS), out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    count, img = 0, None
    for img in fn(n):
        if count >= n:
            break
        if preview and count % (n // 3 + 1) == 0:
            img.save(f'{B}/preview{k:02d}_{count}.png')
        p.stdin.write(img.tobytes())
        count += 1
    while count < n:
        p.stdin.write(img.tobytes()); count += 1
    p.stdin.close(); p.wait()
    print(f'scene {k}: {fn.__name__} {dur:.1f} s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    only = {int(x) for x in a.only.split(',') if x}
    files = []
    for k, fn in enumerate(SCENES, 1):
        if not only or k in only:
            render_scene(k, fn, a.preview)
        files.append(f'{B}/scene{k:02d}.mp4')
    if not only:
        lst = f'{B}/list.txt'
        open(lst, 'w').write(''.join(f"file '{os.path.abspath(f)}'\n" for f in files))
        out = os.path.join(OUT, 'ppo_vs_neat.mp4')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy',
                        '-movflags', '+faststart', out], check=True)
        print('film:', out, f'{duration(out):.0f} s')


if __name__ == '__main__':
    main()
