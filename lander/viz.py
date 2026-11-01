"""Drawing flights for people (replays, films). NEVER used by a policy: it reads hidden env state (the true wind)
to show the viewer what the pilot cannot see.

draw(env, flight) -> PIL image (600 x 400): the game frame, the boulder, wind streaks drifting with the true wind,
and a wind gauge at the top: the true wind (blue, hidden from the pilot) and the pilot's own estimate (orange,
input 16), both on the same -1..1 scale (1 = the strongest gust, wind power 15).
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lander.env import SCALE

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BLUE, ORANGE = (90, 170, 255), (255, 160, 60)


def font(size, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def true_wind(env):
    """The gust the env will apply on its next step, as a share of wind_power (-1..1), and the turbulence twist
    (-1..1). Same formula as gymnasium's LunarLander.step."""
    u = env.unwrapped
    f = lambda i: math.tanh(math.sin(0.02 * i) + math.sin(math.pi * 0.01 * i))
    return f(u.wind_idx), f(u.torque_idx)


class Streaks:
    """Dust that drifts with the true wind, so the gusts are visible in the picture."""

    def __init__(self, seed, n=55):
        rng = np.random.default_rng([int(seed), 99])
        self.x = rng.uniform(0, 600, n)
        self.y = rng.uniform(40, 330, n)
        self.len = rng.uniform(6, 16, n)

    def step(self, w):
        self.x = (self.x + 9.0 * w) % 600

    def draw(self, d, w):
        for x, y, l in zip(self.x, self.y, self.len):
            dx = l * max(abs(w), 0.08) * (1 if w >= 0 else -1)
            d.line([(x - dx, y), (x, y)], fill=(170, 190, 230), width=1)


def gauge(d, cx, y, w_true, w_est, half=110):
    """Two arrows from the centre of a -1..1 scale: true wind (blue) above, the pilot's estimate (orange) below."""
    d.rectangle([cx - half - 70, y - 16, cx + half + 70, y + 30], fill=(0, 0, 0))
    d.line([(cx - half, y + 7), (cx + half, y + 7)], fill=(90, 90, 90), width=1)
    for t in (-1, -0.5, 0, 0.5, 1):
        x = cx + t * half
        d.line([(x, y + 3), (x, y + 11)], fill=(120, 120, 120))
    f = font(11)
    d.text((cx - half - 66, y - 6), 'wind', fill=BLUE, font=f)
    d.text((cx - half - 66, y + 8), 'pilot\'s', fill=ORANGE, font=f)
    d.text((cx - half - 66, y + 19), 'guess', fill=ORANGE, font=f)
    for val, yy, col in ((w_true, y, BLUE), (w_est, y + 14, ORANGE)):
        val = float(np.clip(val, -1.3, 1.3))
        x1 = cx + val * half
        d.line([(cx, yy), (x1, yy)], fill=col, width=4)
        s = 1 if val >= 0 else -1
        if abs(val) > 0.03:
            d.polygon([(x1 + 6 * s, yy), (x1 - 2 * s, yy - 5), (x1 - 2 * s, yy + 5)], fill=col)
    d.text((cx + half + 8, y - 6), '→ right' if w_true >= 0 else '← left', fill=BLUE, font=f)


class Painter:
    """Call paint(env, flight) every frame you keep; it also advances the dust once per call."""

    def __init__(self, seed, gauge_on=True, streaks_on=True):
        self.streaks = Streaks(seed) if streaks_on else None
        self.gauge_on = gauge_on

    def paint(self, env, flight, frame=None):
        img = Image.fromarray(frame if frame is not None else env.render()).convert('RGB')
        d = ImageDraw.Draw(img)
        w, tq = true_wind(env)
        if not env.unwrapped.enable_wind:
            w = tq = 0.0
        if self.streaks is not None:
            self.streaks.draw(d, w)
            self.streaks.step(w)
        if flight.boulder:
            bx, by, r = flight.boulder
            cx, cy, rr = bx * SCALE, 400 - by * SCALE, r * SCALE
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(130, 115, 100), outline=(200, 190, 170), width=2)
        if self.gauge_on:
            gauge(d, 300, 18, w, flight.wind)
        return img
