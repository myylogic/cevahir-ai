"""Round 22: cross-domain mapping and crossing partitions.

Finite, exact, intentionally non-biological. The world generator hides the
latent IDs; learners see only symbols, relations, categories and bridge pairs.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, platform, random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
PROTOCOL = ROOT / 'docs/research/living_learning_crossmap_2026_09_23/PROTOCOL.md'
METHODS = ['separate', 'persistent', 'recompute', 'oracle', 'wrong', 'disabled', 'tree']


def encode(x): return json.dumps(x, sort_keys=True, separators=(',', ':'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def permute(seed):
    return random.Random(440000 + seed).sample(range(6), 6)


def world(seed):
    rng = random.Random(550000 + seed)
    x_of_latent = permute(seed)
    y_of_latent = permute(seed + 1000)
    latent_of_x = {x: i for i, x in enumerate(x_of_latent)}
    latent_of_y = {y: i for i, y in enumerate(y_of_latent)}
    x_next = {x_of_latent[i]: x_of_latent[(i + 1) % 6] for i in range(6)}
    y_next = {y_of_latent[i]: y_of_latent[i ^ 1] for i in range(6)}
    x_group = {x_of_latent[i]: int(i >= 3) for i in range(6)}
    y_group = {y_of_latent[i]: int(i % 2) for i in range(6)}
    true_map = {x_of_latent[i]: y_of_latent[i] for i in range(6)}
    # Two initial bridges leave 24 maps; each later bridge removes ambiguity.
    order = list(range(6)); rng.shuffle(order)
    bridges = [[x_of_latent[i], y_of_latent[i]] for i in order]
    source_x = [[x, x_next[x]] for x in sorted(x_next)]
    source_y = [[y, y_next[y]] for y in sorted(y_next)]
    queries = []
    # Query IDs make block semantics explicit; each block uses every symbol once.
    for phase in range(4):
        if phase == 0:
            for x in range(6): queries.append({'phase': phase, 'kind': 'xy', 'input': x})
        elif phase == 1:
            for y in range(6): queries.append({'phase': phase, 'kind': 'yx', 'input': y})
        elif phase == 2:
            for x in range(6): queries.append({'phase': phase, 'kind': 'cat', 'input': x})
        else:
            for x in range(6): queries.append({'phase': phase, 'kind': 'xy', 'input': x})
    bridge_before = [2, 3, 4, 5]
    return {'seed': seed, 'x_of_latent': x_of_latent, 'y_of_latent': y_of_latent,
            'source_x': source_x, 'source_y': source_y, 'bridges': bridges,
            'true_map': true_map, 'x_next': x_next, 'y_next': y_next,
            'x_group': x_group, 'y_group': y_group, 'queries': queries,
            'bridge_before': bridge_before}


def candidates(bridges):
    result = []
    for values in itertools.permutations(range(6)):
        mapping = {x: values[x] for x in range(6)}
        if all(mapping[x] == y for x, y in bridges):
            result.append(mapping)
    return result


def truth(w, q):
    xmap = w['true_map']
    if q['kind'] == 'xy':
        return ('y', xmap[w['x_next'][q['input']]])
    if q['kind'] == 'yx':
        inv = {y: x for x, y in xmap.items()}
        return ('x', inv[w['y_next'][q['input']]])
    x = q['input']; y = xmap[x]
    return ('cat', w['x_group'][x], w['y_group'][y])


class Learner:
    def __init__(self, method, w):
        self.method = method
        self.x_next = dict(w['x_next'])
        self.y_next = dict(w['y_next'])
        self.x_group = dict(w['x_group'])
        self.y_group = dict(w['y_group'])
        self.bridges = []
        self.candidates = []
        self.cost = {k: 0 for k in ('source_parse','bridge_filter','candidate_scan','query','writes','tree_copy')}
        self.cache = {}
        self.disabled = method == 'disabled'

    def snapshot(self):
        return json.loads(encode(self.__dict__))

    @classmethod
    def restore(cls, state, w):
        obj = cls(state['method'], w); obj.__dict__.update(json.loads(encode(state))); return obj

    def ingest(self, bridges):
        self.bridges = [list(x) for x in bridges]
        if self.method == 'recompute':
            return
        key = encode(self.bridges)
        if self.method in ('persistent', 'wrong', 'disabled'):
            self.cost['bridge_filter'] += 1
            if self.candidates:
                self.candidates = [m for m in self.candidates if m[self.bridges[-1][0]] == self.bridges[-1][1]]
                self.cost['bridge_filter'] += len(self.candidates)
            else:
                self.candidates = candidates(self.bridges)
                self.cost['bridge_filter'] += 720 * len(self.bridges)
        self.cost['writes'] += len(encode(self.candidates).encode())

    def _true_map(self):
        # Oracle stores the true map in a private field injected by run; not a
        # free input for other methods.
        return getattr(self, '_oracle_map', {})

    def _maps(self):
        if self.method == 'recompute':
            key = encode(self.bridges)
            if key not in self.cache:
                self.cache[key] = candidates(self.bridges)
                self.cost['bridge_filter'] += 720 * len(self.bridges)
            return self.cache[key]
        if self.method == 'wrong':
            return self.candidates[:1]
        return self.candidates

    def query(self, q, target):
        before = self.cost.copy()
        maps = self._maps()
        result_set = set()
        if q['kind'] == 'xy' and not self.disabled:
            for m in maps:
                result_set.add(('y', m[self.x_next[q['input']]]))
                self.cost['candidate_scan'] += 1
        elif q['kind'] == 'yx' and not self.disabled:
            for m in maps:
                inv = {y: x for x, y in m.items()}
                result_set.add(('x', inv[self.y_next[q['input']]]))
                self.cost['candidate_scan'] += 1
        elif q['kind'] == 'cat':
            # X grouping is locally known. Y grouping requires the learned map.
            if self.disabled or not maps or self.method in ('tree', 'separate'):
                # A partial category pair is an abstention, not a wrong label.
                result_set = set()
            else:
                for m in maps:
                    result_set.add(('cat', self.x_group[q['input']], self.y_group[m[q['input']]]))
                    self.cost['candidate_scan'] += 1
        else:
            result_set.add(None)
        prediction = next(iter(result_set)) if len(result_set) == 1 else None
        confident = prediction is not None
        correct = confident and prediction == target
        abstained = not confident
        self.cost['query'] += 1 + len(result_set)
        delta = {k: self.cost[k] - before[k] for k in self.cost}
        return {'prediction': prediction, 'target': target, 'confident': confident,
                'correct': correct, 'abstained': abstained, 'candidate_count': len(maps),
                'costs': delta, 'state_bytes': len(encode(self.snapshot()).encode())}


def run(w, method):
    learner = Learner(method, w)
    records = []
    # Source parsing is identical but only learner state receives parsed structures.
    learner.cost['source_parse'] += len(w['source_x']) + len(w['source_y'])
    if method == 'oracle':
        learner.candidates = [dict(w['true_map'])]
    seen_bridge = 0
    for t, q in enumerate(w['queries']):
        phase = q['phase']
        required = w['bridge_before'][phase]
        if required > seen_bridge:
            learner.ingest(w['bridges'][:required]); seen_bridge = required
        row = learner.query(q, truth(w, q))
        records.append({'t': t, 'phase': phase, 'query': q, 'engine': row})
    return {'method': method, 'records': records, 'final': learner.snapshot()}


def experiment():
    return {'round': 22, 'date': '2026-09-23', 'python': platform.python_version(),
            'source_sha256': sha(__file__), 'protocol_sha256': sha(PROTOCOL), 'methods': METHODS,
            'runs': [{'seed': seed, 'world': world(seed),
                      'tracks': {m: run(world(seed), m) for m in METHODS}} for seed in range(24)]}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--write', action='store_true'); p.add_argument('--resume', nargs=2); a = p.parse_args()
    if a.resume:
        source, dest = map(Path, a.resume); data = json.loads(source.read_text())
        l = Learner.restore(data['state'], data['world'])
        rows = []
        for q in data['tail']:
            rows.append(l.query(q, truth(data['world'], q)))
        dest.write_text(encode({'records': rows, 'state': l.snapshot()})); return
    data = experiment(); path = RESULTS / 'crossmap.json'
    if a.write: path.write_text(encode(data) + '\n'); print('wrote', path, path.stat().st_size)
    else: assert data == json.loads(path.read_text()); print('exact_rerun_passed')


if __name__ == '__main__': main()
