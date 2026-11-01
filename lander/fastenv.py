"""The frozen flight (lander/env.py) without most of its Python overhead, for training and checks. Same numbers.

env.py is the task and stays byte-for-byte frozen; this module only re-does its arithmetic faster:
- fast_step: gymnasium LunarLander.step (0.29.1) for a discrete action without rendering, called on the bare env, so
  the three wrapper layers, the action-space check and numpy's scalar maths are skipped. Same operations in the same
  order, so the same floats.
- FastFlight: env.Flight on Python floats instead of float32 numpy scalars, with the engine dispersion drawn in one
  block per flight (gymnasium draws two uniforms per step and nothing else draws from np_random until the next reset).
- run_episode: env.run_episode with FastFlight. It falls back to env.run_episode when the env renders (videos).

tests/test_core.py flies both versions step by step and requires identical inputs, actions and results.
"""

import math
import struct

import gymnasium.envs.box2d.lunar_lander as _ll
import numpy as np

from lander import env as E
from lander.env import World, MAX_STEPS, W, H, DRIFT_RATE, DRIFT_SCALE, SIDE_PUSH, MAIN_PUSH, WIND_POWER, \
    WIND_RATE, WIND_MAIN_WEIGHT, LANDING_BONUS, centre_bonus, waypoint_inputs

_STATE32 = struct.Struct('8f')
_F32 = struct.Struct('f')


def _f32(x):
    """x rounded to float32. The observations are float32 and numpy subtracts two of them in float32 (anything mixed
    with a Python float is float64): rounding the float64 difference of two float32 numbers gives that same number."""
    return _F32.unpack(_F32.pack(x))[0]


def fast_step(u, action, d0, d1):
    """gymnasium's LunarLander.step on the bare env u, discrete action, no rendering. d0, d1: this step's two
    np_random.uniform(-1, 1) draws. Returns (obs as 8 float32-rounded Python floats, reward, terminated); truncation
    at 1000 steps is Flight's MAX_STEPS."""
    lander, legs = u.lander, u.legs
    if u.enable_wind and not (legs[0].ground_contact or legs[1].ground_contact):
        wind_mag = math.tanh(math.sin(0.02 * u.wind_idx) + (math.sin(math.pi * 0.01 * u.wind_idx))) * u.wind_power
        u.wind_idx += 1
        lander.ApplyForceToCenter((wind_mag, 0.0), True)
        torque_mag = math.tanh(math.sin(0.02 * u.torque_idx)
                               + (math.sin(math.pi * 0.01 * u.torque_idx))) * (u.turbulence_power)
        u.torque_idx += 1
        lander.ApplyTorque((torque_mag), True)
    angle = lander.angle
    tip = (math.sin(angle), math.cos(angle))
    side = (-tip[1], tip[0])
    dispersion = (d0 / _ll.SCALE, d1 / _ll.SCALE)
    m_power = 0.0
    if action == 2:
        m_power = 1.0
        ox = tip[0] * (_ll.MAIN_ENGINE_Y_LOCATION / _ll.SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
        oy = -tip[1] * (_ll.MAIN_ENGINE_Y_LOCATION / _ll.SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
        pos = lander.position
        impulse_pos = (pos[0] + ox, pos[1] + oy)
        lander.ApplyLinearImpulse((-ox * _ll.MAIN_ENGINE_POWER * m_power, -oy * _ll.MAIN_ENGINE_POWER * m_power),
                                  impulse_pos, True)
    s_power = 0.0
    if action == 1 or action == 3:
        direction = action - 2
        s_power = 1.0
        ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * _ll.SIDE_ENGINE_AWAY / _ll.SCALE)
        oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * _ll.SIDE_ENGINE_AWAY / _ll.SCALE)
        pos = lander.position
        impulse_pos = (pos[0] + ox - tip[0] * 17 / _ll.SCALE,
                       pos[1] + oy + tip[1] * _ll.SIDE_ENGINE_HEIGHT / _ll.SCALE)
        lander.ApplyLinearImpulse((-ox * _ll.SIDE_ENGINE_POWER * s_power, -oy * _ll.SIDE_ENGINE_POWER * s_power),
                                  impulse_pos, True)
    u.world.Step(1.0 / _ll.FPS, 6 * 30, 2 * 30)
    pos = lander.position
    vel = lander.linearVelocity
    state = (
        (pos.x - _ll.VIEWPORT_W / _ll.SCALE / 2) / (_ll.VIEWPORT_W / _ll.SCALE / 2),
        (pos.y - (u.helipad_y + _ll.LEG_DOWN / _ll.SCALE)) / (_ll.VIEWPORT_H / _ll.SCALE / 2),
        vel.x * (_ll.VIEWPORT_W / _ll.SCALE / 2) / _ll.FPS,
        vel.y * (_ll.VIEWPORT_H / _ll.SCALE / 2) / _ll.FPS,
        lander.angle,
        20.0 * lander.angularVelocity / _ll.FPS,
        1.0 if legs[0].ground_contact else 0.0,
        1.0 if legs[1].ground_contact else 0.0,
    )
    reward = 0
    shaping = (-100 * math.sqrt(state[0] * state[0] + state[1] * state[1])
               - 100 * math.sqrt(state[2] * state[2] + state[3] * state[3])
               - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7])
    if u.prev_shaping is not None:
        reward = shaping - u.prev_shaping
    u.prev_shaping = shaping
    reward -= m_power * 0.30
    reward -= s_power * 0.03
    terminated = False
    if u.game_over or abs(state[0]) >= 1.0:
        terminated = True
        reward = -100
    if not lander.awake:
        terminated = True
        reward = +100
    return list(_STATE32.unpack(_STATE32.pack(*state))), reward, terminated


class DriftSensor(E.DriftSensor):
    def update(self, obs, last_action):
        touching = obs[6] or obs[7]
        if self.prev_vx is not None and last_action in (0, 2) and not touching:
            self.value += DRIFT_RATE * (_f32(obs[2] - self.prev_vx) * DRIFT_SCALE - self.value)
        self.prev_vx = obs[2]
        return self.value


class WindSensor(E.WindSensor):
    def update(self, prev_obs, obs, action):
        air = not (prev_obs[6] or prev_obs[7] or obs[6] or obs[7])
        if air:
            push = MAIN_PUSH * math.sin(prev_obs[4]) if action == 2 else SIDE_PUSH[action]
            m = (_f32(obs[2] - prev_obs[2]) - push) / (0.0008 * WIND_POWER)
            w = WIND_MAIN_WEIGHT if action == 2 else 1.0
            self.value += WIND_RATE * w * (m - self.value)
        return self.value


class FastFlight(E.Flight):
    def __init__(self, env, seed, world=World()):
        super().__init__(env, seed, world)
        self.u = u = env.unwrapped              # each env.unwrapped walks three wrapper layers
        self.legs = u.legs
        self.obs = self.obs.tolist()
        self.sensor, self.wind_sensor = DriftSensor(), WindSensor()
        self.disp = u.np_random.uniform(-1.0, +1.0, 2 * MAX_STEPS).tolist()

    def inputs(self):
        o = self.obs
        v = o + [self.drift, abs(o[0])]
        b = self.boulder
        if b is None:
            v += [0.0, 0.0, 0.0]
        else:
            p = self.u.lander.position
            v += [(b[0] - p.x) / (W / 2), (b[1] - p.y) / (H / 2), b[2] / (W / 2)]
        v += waypoint_inputs(v)
        v.append(self.wind)
        return np.array(v)

    def touching(self):
        legs = self.legs
        return legs[0].ground_contact or legs[1].ground_contact

    def step(self, action):
        u = self.u
        was_touching = self.touching()
        if self.push and not was_touching:
            u.lander.ApplyForceToCenter((self.push, 0.0), True)
        if self.world.support:
            t = -self.world.support * (8.0 * u.lander.angle + 2.0 * u.lander.angularVelocity)
            u.lander.ApplyTorque(t, True)
        vy_before = self.obs[3]
        prev = self.obs
        k = 2 * self.steps
        self.obs, r, terminated = fast_step(u, int(action), self.disp[k], self.disp[k + 1])
        self.action = int(action)
        self.drift = self.sensor.update(self.obs, self.action)
        self.wind = self.wind_sensor.update(prev, self.obs, self.action)
        self.reward += float(r)
        self.steps += 1
        if not was_touching and self.touching() and self.touch_speed is None:
            self.touch_speed = abs(vy_before)
        if self.boulder:
            hit = E._boulder_hit(u)
            c = E._clearance(u, self.boulder)
            self.min_clearance = min(self.min_clearance, c)
            if hit:
                if not terminated:
                    self.reward += -100.0
                self.outcome = 'boulder'
                return True
        if terminated:
            self.outcome = 'landed' if r > 0 else 'crashed'
            return True
        if self.steps >= MAX_STEPS:
            self.outcome = 'timeout'
            return True
        return False


def run_episode(policy, seed, world=World(), env=None, record=None):
    """env.run_episode, flown with FastFlight when nothing is rendered."""
    if env is None or record is not None or env.unwrapped.render_mode is not None or env.unwrapped.continuous:
        return E.run_episode(policy, seed, world, env=env, record=record)
    f = FastFlight(env, seed, world)
    if hasattr(policy, 'reset'):
        policy.reset()
    done = False
    while not done:
        done = f.step(policy(f.inputs(), f))
    out = f.result()
    out['fitness'] = out['reward'] + (LANDING_BONUS + centre_bonus(out['x_final']) if out['landed'] else 0.0)
    return out
