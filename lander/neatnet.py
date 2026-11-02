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

import math

import neat
import numpy as np

try:
    from numba import njit
except ImportError:                   # no numba: NeatJsonNet falls back to the numpy layers (same actions, slower)
    njit = None

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

    # Speed (2026-11-01): between generations the trainer's main process spent ~6 s while all workers waited, mostly
    # in these two neat-python 2.0 methods. Same logic, same random draws in the same order, same float sums, so the
    # same children and species; only faster (tests/test_core.py compares them with neat-python's own methods).

    def configure_crossover(self, genome1, genome2, config):
        """neat.DefaultGenome.configure_crossover, with the feed-forward cycle check done on a successor map built as
        the child grows (neat-python rescans every connection so far for every gene: O(n^2) per child)."""
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        parent1_innovations = {cg.innovation: cg for cg in parent1.connections.values()}
        parent2_innovations = {cg.innovation: cg for cg in parent2.connections.values()}
        all_innovations = set(parent1_innovations.keys()) | set(parent2_innovations.keys())
        ff = config.feed_forward
        succ = {}
        for a, b in self.connections:
            succ.setdefault(a, set()).add(b)

        def cycle(key):                      # graphs.creates_cycle: is key's input reachable from its output?
            i, o = key
            if i == o:
                return True
            seen, todo = {o}, [o]
            while todo:
                for b in succ.get(todo.pop(), ()):
                    if b == i:
                        return True
                    if b not in seen:
                        seen.add(b)
                        todo.append(b)
            return False

        for innovation_num in all_innovations:
            cg1 = parent1_innovations.get(innovation_num)
            cg2 = parent2_innovations.get(innovation_num)
            if cg1 is not None and cg2 is not None:
                if cg1.key != cg2.key:
                    import warnings
                    warnings.warn(f"Innovation number collision: innovation {innovation_num} assigned to both "
                                  f"{cg1.key} and {cg2.key}. Treating as disjoint genes.", RuntimeWarning)
                    new_gene = cg1.copy()
                else:
                    new_gene = cg1.crossover(cg2)
            elif cg1 is not None:
                new_gene = cg1.copy()
            else:
                continue
            if ff and cycle(new_gene.key):
                continue
            self.connections[new_gene.key] = new_gene
            succ.setdefault(new_gene.key[0], set()).add(new_gene.key[1])
        parent2_set = parent2.nodes
        for key, ng1 in parent1.nodes.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            self.nodes[key] = ng1.copy() if ng2 is None else ng1.crossover(ng2)

    def distance(self, other, config):
        """neat.DefaultGenome.distance with the gene distances inlined (same terms, same order of summation)."""
        cw, cd = config.compatibility_weight_coefficient, config.compatibility_disjoint_coefficient
        node_distance = 0.0
        sn, on = self.nodes, other.nodes
        if sn or on:
            disjoint_nodes = 0
            for k2 in on:
                if k2 not in sn:
                    disjoint_nodes += 1
            for k1, n1 in sn.items():
                n2 = on.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    d = abs(n1.bias - n2.bias) + abs(n1.response - n2.response)
                    d += abs(n1.time_constant - n2.time_constant)
                    if n1.activation != n2.activation:
                        d += 1.0
                    if n1.aggregation != n2.aggregation:
                        d += 1.0
                    node_distance += d * cw
            node_distance = (node_distance + (cd * disjoint_nodes)) / max(len(sn), len(on))
        connection_distance = 0.0
        sc, oc = self.connections, other.connections
        if sc or oc:
            disjoint_connections = 0
            for k2 in oc:
                if k2 not in sc:
                    disjoint_connections += 1
            for k1, c1 in sc.items():
                c2 = oc.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    d = abs(c1.weight - c2.weight)
                    if c1.enabled != c2.enabled:
                        d += 1.0
                    connection_distance += d * cw
            connection_distance = (connection_distance + (cd * disjoint_connections)) / max(len(sc), len(oc))
        return node_distance + connection_distance


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
        # the same layers flattened for _act_kernel: one entry per non-input node in feed-forward order, with its
        # incoming (source, gain * response * weight) pairs and gain * bias (gain 2.5 for tanh, 1 for identity)
        rows, tanh_f, b_f, start, src, wt = [], [], [], [0], [], []
        for layer in layers:
            for k in sorted(layer):
                g = TANH_GAIN if nodes[k][2] == 'tanh' else 1.0
                ins = sorted((idx[a], w) for (a, b), w in conns if b == k and a in idx)
                rows.append(idx[k]); tanh_f.append(nodes[k][2] == 'tanh'); b_f.append(g * nodes[k][0])
                src += [i for i, _ in ins]; wt += [g * nodes[k][1] * w for _, w in ins]
                start.append(len(src))
        # packed into one int and one float array: numba's call overhead grows with the number of array arguments
        m = len(rows)
        self.flat_i = np.array([self.n_in, m, len(self.outputs)] + rows + [int(t) for t in tanh_f] + start + src
                               + self.out_idx.tolist(), np.int64)
        self.flat_f = np.concatenate([self.scale.astype(np.float64), b_f, wt, np.zeros(self.size)])

    def activate(self, x):
        v = self.v
        v[:self.n_in] = np.asarray(x[:self.n_in]) / self.scale
        for tr, tW, tb, ir, iW, ib in self.layers:
            if len(tr):
                v[tr] = np.tanh(tW @ v + tb)
            if len(ir):
                v[ir] = iW @ v + ib
        return v[self.out_idx]

    def node_values(self):
        """Every node's value after the last call (inputs scaled), indexed by self.idx. The numba kernel keeps them
        at the end of flat_f, the numpy path in self.v."""
        return self.v if _act_kernel is None else self.flat_f[len(self.flat_f) - self.size:]

    def __call__(self, v, flight=None):
        if _act_kernel is None:
            return int(np.argmax(self.activate(v)))
        return _act_kernel(np.asarray(v, np.float64), self.flat_i, self.flat_f)


def _act(x, fi, ff):
    """The action of NeatJsonNet (argmax of activate()) without numpy's per-call overhead: ~20x faster on a
    551-connection net. Sums may differ from the matrix product in the last bit, which changes the action only on
    an exact near-tie of two outputs; flights were checked identical to the numpy path (results/speed_check.json)."""
    n_in, m, n_out = fi[0], fi[1], fi[2]
    rows, tanh_f, start = 3, 3 + m, 3 + 2 * m                 # offsets into fi: rows, tanh flags, conn starts,
    src = start + m + 1                                        # conn sources, output indices
    out_idx = src + fi[start + m]
    bias, wt = n_in, n_in + m                                  # offsets into ff: scale, bias, weights, node values
    v = wt + fi[start + m]
    for i in range(n_in):
        ff[v + i] = x[i] / ff[i]
    for j in range(m):
        s = ff[bias + j]
        for c in range(fi[start + j], fi[start + j + 1]):
            s += ff[wt + c] * ff[v + fi[src + c]]
        ff[v + fi[rows + j]] = math.tanh(s) if fi[tanh_f + j] else s
    best = 0
    for o in range(1, n_out):
        if ff[v + fi[out_idx + o]] > ff[v + fi[out_idx + best]]:
            best = o
    return best


_act_kernel = njit(cache=True, nogil=True)(_act) if njit else None


def size(genome, n_out=4):
    return sum(1 for k in genome.nodes if k >= n_out), sum(c.enabled for c in genome.connections.values())
