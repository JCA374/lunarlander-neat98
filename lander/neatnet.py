"""NEAT networks from a clean start, and the one format every tool flies them in.

- RandomStartGenome: neat-python's DefaultGenome whose first population is NOT minimal: every genome starts with a
  hidden layer of H random tanh nodes (all 16 inputs -> H hidden -> 4 outputs), weights ~ N(0, 1/fan_in). No copied
  pilot. The size test (results/size_*.json, 2026-11-01) showed why: copies of the PPO pilot reach 56% with 8 hidden
  nodes, 79% with 16, 87% with 32, so a small net is not enough. From there NEAT adds AND deletes nodes and
  connections freely (no node is protected).
- to_json(genome): the genome as a plain dict; NeatJsonNet flies it (training, checks, replays, the goal test all
  use the same code). Inputs are scaled by fixed constants (INPUT_SCALE, measured from the hand-written pilot's
  flights, not from any learned model) so all 16 inputs are of order 1.
"""

import json
import os

import neat
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TANH_GAIN = 2.5                       # neat-python's tanh is tanh(2.5 z)
START = {'hidden': 32}                # set by the trainer before the population is created
_SCALE_PATH = os.path.join(ROOT, 'lander', 'input_scale.json')
INPUT_SCALE = json.load(open(_SCALE_PATH)) if os.path.exists(_SCALE_PATH) else None


class RandomStartGenome(neat.DefaultGenome):
    def configure_new(self, config):
        H = START['hidden']
        if not H:
            return super().configure_new(config)
        n_in, n_out = len(config.input_keys), len(config.output_keys)
        self.nodes, self.connections = {}, {}
        hidden = list(range(n_out, n_out + H))
        for o in config.output_keys:
            n = self.create_node(config, o)
            n.bias, n.activation, n.response = 0.0, 'identity', 1.0
            self.nodes[o] = n
        for k in hidden:
            n = self.create_node(config, k)
            n.bias, n.activation, n.response = float(np.random.normal(0, 0.2)), 'tanh', 1.0
            self.nodes[k] = n
            for ik in config.input_keys:
                c = self.create_connection(config, ik, k, _innov(config, ik, k))
                c.weight, c.enabled = float(np.random.normal(0, 1 / np.sqrt(n_in)) / TANH_GAIN), True
                self.connections[(ik, k)] = c
            for o in config.output_keys:
                c = self.create_connection(config, k, o, _innov(config, k, o))
                c.weight, c.enabled = float(np.random.normal(0, 1 / np.sqrt(H))), True
                self.connections[(k, o)] = c


def _innov(config, a, b):
    """neat-python 2.0 gives every connection gene a historical innovation number; the start layer's genes are the
    same in every genome, so they share their numbers (crossover lines them up)."""
    return config.innovation_tracker.get_innovation_number(a, b, 'initial_connection')


def to_json(genome, config):
    """The genome as a plain dict, enough to fly it without neat-python."""
    gc = config.genome_config
    return {'kind': 'neat', 'inputs': list(gc.input_keys), 'outputs': list(gc.output_keys),
            'scale': INPUT_SCALE,
            'nodes': {str(k): [n.bias, n.response, n.activation] for k, n in genome.nodes.items()},
            'conns': [[a, b, c.weight] for (a, b), c in genome.connections.items() if c.enabled]}


class NeatJsonNet:
    """Feed-forward evaluation of a to_json() dict, compiled to one small matrix product per layer (the old project
    measured neat-python's activate() at 70% of all training time; this was ~10x faster). Same maths as
    neat-python: node = act(bias + response * sum(w * x)), tanh with the 2.5 gain."""

    def __init__(self, d):
        from neat.graphs import feed_forward_layers
        self.inputs, self.outputs = list(d['inputs']), list(d['outputs'])
        nodes = {int(k): v for k, v in d['nodes'].items()}
        conns = [((a, b), w) for a, b, w in d['conns']]
        layers = feed_forward_layers(self.inputs, self.outputs, [k for k, _ in conns])
        if isinstance(layers, tuple):                  # neat-python 2.0 returns (layers, required)
            layers = layers[0]
        layers = [set(l) for l in layers]
        order = self.inputs + [n for layer in layers for n in layer]
        idx = {k: i for i, k in enumerate(order)}
        for o in self.outputs:                         # an output nobody feeds still exists (its value: act(bias))
            if o not in idx:
                idx[o] = len(order); order.append(o); layers.append({o})
        self.size, self.idx = len(order), idx
        self.layers = []
        for layer in layers:
            keys = sorted(layer)
            W = np.zeros((len(keys), self.size))
            for (a, b), w in conns:
                if b in layer and a in idx:
                    W[keys.index(b), idx[a]] = w
            bias = np.array([nodes[k][0] for k in keys]); resp = np.array([nodes[k][1] for k in keys])
            acts = [nodes[k][2] for k in keys]
            if not all(a in ('tanh', 'identity') for a in acts):
                raise ValueError('only tanh and identity nodes are supported')
            tanh = np.array([a == 'tanh' for a in acts])
            rows = np.array([idx[k] for k in keys])
            W = W * resp[:, None]
            t, i = tanh, ~tanh
            self.layers.append((rows[t], TANH_GAIN * W[t], TANH_GAIN * bias[t], rows[i], W[i], bias[i]))
        self.out_idx = np.array([idx[o] for o in self.outputs])
        self.n_in = len(self.inputs)
        self.scale = np.array(d['scale']) if d.get('scale') else np.ones(self.n_in)
        self.v = np.zeros(self.size)
        self.n_hidden = sum(1 for k in nodes if k not in self.outputs)
        self.n_conns = len(conns)

    def activate(self, x):
        v = self.v
        v[:self.n_in] = np.asarray(x[:self.n_in]) / self.scale
        for tr, tW, tb, ir, iW, ib in self.layers:
            if len(tr):
                v[tr] = np.tanh(tW @ v + tb)
            if len(ir):
                v[ir] = iW @ v + ib
        return v[self.out_idx]

    def __call__(self, v, flight=None):
        return int(np.argmax(self.activate(v)))


def size(genome, n_out=4):
    return sum(1 for k in genome.nodes if k >= n_out), sum(c.enabled for c in genome.connections.values())
