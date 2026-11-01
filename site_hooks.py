"""Site hooks for publish.py (ml-project-site skill): replays, the Play tab's models and games, clip rendering."""
import glob
import json
import math
import os
import subprocess

import numpy as np

from lander.seeds import ANALYSIS

ROOT = os.path.dirname(os.path.abspath(__file__))
N_GAMES = 12
GAME_SEED = ANALYSIS + 900_000


def _fly_frames(policy_path, seed, every=2):
    """One real-task flight with the wind drawn (lander/viz.py); returns frames (PIL 600x400) and the result."""
    from lander.env import GOAL_WORLD, run_episode, make_env
    from lander.policy import load_policy
    from lander.viz import Painter
    pol = load_policy(policy_path)
    env = make_env('rgb_array')
    painter, frames = Painter(seed), []

    def rec(f):
        if f.steps % every == 0:
            frames.append(painter.paint(env, f))
    r = run_episode(lambda v, f: pol(v, None), seed, GOAL_WORLD, env=env, record=rec)
    env.close()
    return frames, r


def outcome(r):
    if r['landed']:
        return True, 'landed'
    if r['rested']:
        return False, 'rested off the pad'
    return False, {'crashed': 'crashed', 'boulder': 'hit the boulder', 'timeout': 'timeout'}[r['outcome']]


def _stamp(img, r):
    from PIL import ImageDraw
    from lander.viz import font
    ok, txt = outcome(r)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 330, 600, 372], fill=(0, 0, 0))
    f = font(24, True)
    d.text((300 - d.textlength(txt.upper(), font=f) / 2, 337), txt.upper(), fill=(105, 219, 124) if ok else (255, 120, 120), font=f)
    return img


def replay_gif(policy_path, seed, out):
    """GIF for the log: plays ONCE (no loop argument) and ends on the outcome; the page replays it on click."""
    from PIL import Image
    frames, r = _fly_frames(policy_path, seed)
    frames = [f.resize((420, 280)) for f in frames]
    last = _stamp(frames[-1].resize((600, 400)), r).resize((420, 280))
    frames = [f.convert('P', palette=Image.ADAPTIVE, colors=48) for f in frames + [last] * 20]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, optimize=True)


def models():
    """The Play tab's models: the newest NEAT champions first (added as the run produces them), then the yardsticks.
    Each file exists in the repo."""
    import glob as _g
    m = []
    for p in sorted(_g.glob(os.path.join(ROOT, 'runs/*/champions/*_goal.json')), key=os.path.getmtime, reverse=True)[:3]:
        run, name = p.split(os.sep)[-3], os.path.basename(p)[:-10]
        m.append((f'{run}_{name}'.lower(), os.path.relpath(p, ROOT), f'NEAT {run} {name}', f'A NEAT champion of run {run} ({name}).'))
    m += [('ppo', 'models/baselines/ppo_M2_i08100.json', 'PPO champion (yardstick)',
           'The last project\'s best: PPO from scratch, 98.6% on two test blocks of layout v2. The number NEAT has to match.'),
          ('teacher_v2', 'teacher:v2', 'Hand-written pilot v2',
           'The hand-tuned pilot (CMA-ES-tuned gains), for comparison.'),
          ('old_clone', 'models/baselines/old_neat_seed_clone.json', 'Old NEAT seed clone',
           'The 16→24→4 clone of pilot v2 that the old NEAT run started from (~85%).')]
    return [{'id': i, 'path': p if p.startswith('teacher:') else os.path.join(ROOT, p), 'label': l, 'note': n}
            for i, p, l, n in m if p.startswith('teacher:') or os.path.exists(os.path.join(ROOT, p))]


def _wind(seed, steps=250):
    """The average wind over a typical descent (~250 steps), as a share of wind power (positive = to the right)."""
    rng = np.random.default_rng([int(seed), 7])
    i0 = int(rng.integers(-9999, 9999))
    return float(np.mean([math.tanh(math.sin(0.02 * i) + math.sin(math.pi * 0.01 * i)) for i in range(i0, i0 + steps)]))


def games():
    """12 fixed flights chosen on purpose to cover every kind: the boulder left of / over / right of the pad, low or high
    above it (layout v3: its bottom 3.5-4.9 m above the pad), and a steady push to the left or the right (3 x 2 x 2).
    The first matching analysis seed for each."""
    from lander.env import boulder_spec, W, PAD_Y
    want = [(pos, low, wdir) for pos in ('left', 'over', 'right') for low in (True, False) for wdir in (-1, 1)]
    found = {}
    for s in range(GAME_SEED, GAME_SEED + 5000):
        x, y, r = boulder_spec(s)
        dx, gap, w = x - W / 2, y - r - PAD_Y, _wind(s)
        pos = 'over' if abs(dx) < 0.8 else ('left' if dx < -1.5 else 'right' if dx > 1.5 else None)
        low = True if gap < 3.65 else False if gap > 4.3 else None
        wdir = -1 if w < -0.3 else 1 if w > 0.3 else None
        k = (pos, low, wdir)
        if None not in k and k not in found:
            found[k] = (s, dx, gap, r, w)
        if len(found) == len(want):
            break
    out = []
    for i, k in enumerate(want):
        s, dx, gap, r, w = found[k]
        where = 'over the pad' if k[0] == 'over' else f'{abs(dx):.1f} m {k[0]} of the pad'
        out.append({'id': f'g{i:02d}', 'seed': s,
                    'title': f'{"Low" if k[1] else "High"} boulder {"over the pad" if k[0] == "over" else "on the " + k[0]}, wind {"←" if k[2] < 0 else "→"}',
                    'desc': f'boulder {where}, its bottom {gap:.1f} m above the pad, {2 * r:.1f} m wide; '
                            f'average wind {abs(w):.0%} of full strength to the {"left" if w < 0 else "right"}'})
    return out


def render_play(policy_path, game, out_mp4):
    """One flight as an MP4 that plays once (25 fps = real time), outcome written on the held last frame."""
    from lander.viz import font
    frames, r = _fly_frames(policy_path, game['seed'])
    frames[-1] = _stamp(frames[-1], r)
    _encode(frames, out_mp4)
    ok, txt = outcome(r)
    return {'ok': ok, 'result': txt, 'detail': f"{r['steps']} steps, score {r['reward']:.0f}"}


def _encode(frames, out, fps=25, hold=30):
    w, h = frames[0].size
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                          '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * hold:
        p.stdin.write(f.convert('RGB').tobytes())
    p.stdin.close()
    p.wait()
