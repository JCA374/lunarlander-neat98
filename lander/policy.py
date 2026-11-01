"""Policies the evaluators can load, all called as policy(inputs, flight) -> action (0-3).

    teacher:v1 / teacher:v2        the hand-written pilot with its tuned gains
    models/x.json                  a numpy MLP: either the old clone format (W1,b1,W2,b2) or
                                   {"layers": [[W, b], ...], "mean": [...], "std": [...], "act": "tanh"}
A policy only ever sees flight.inputs() (lander/env.py inputs()): the 8 observations and sensors computed
from past observations and its own past actions. Never the seed, the wind index or other hidden env state.
"""
import json
import os

import numpy as np

from lander.teacher import Teacher, load_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NumpyMLP:
    def __init__(self, d):
        if 'layers' in d:
            self.layers = [(np.array(W, dtype=np.float64), np.array(b, dtype=np.float64)) for W, b in d['layers']]
        else:                                   # the old clone format
            self.layers = [(np.array(d['W1']), np.array(d['b1'])), (np.array(d['W2']), np.array(d['b2']))]
        n_in = self.layers[0][0].shape[1]
        self.mean = np.array(d.get('mean', np.zeros(n_in)), dtype=np.float64)
        self.std = np.array(d.get('std', np.ones(n_in)), dtype=np.float64)
        self.clip = float(d.get('clip', 1e9))

    def logits(self, X):
        h = np.clip((np.asarray(X, dtype=np.float64) - self.mean) / self.std, -self.clip, self.clip)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h

    def __call__(self, v, flight=None):
        return int(np.argmax(self.logits(np.asarray(v)[None])[0]))


def load_policy(spec):
    if spec.startswith('teacher:'):
        v = spec.split(':', 1)[1]
        name = {'v1': 'teacher_params_v1.json', 'v2': 'teacher_params_v2.json'}.get(v, v)
        return Teacher(load_params(os.path.join(ROOT, 'lander', name)))
    if spec.endswith('.json'):
        return NumpyMLP(json.load(open(spec)))
    raise ValueError(f'unknown policy {spec}')
