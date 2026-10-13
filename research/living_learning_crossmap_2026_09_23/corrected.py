"""Audited round 22. Query targets and latent world never enter Learner."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
DOCS = ROOT / 'docs/research/living_learning_crossmap_2026_09_23'
METHODS = ['separate', 'persistent', 'recompute', 'completion', 'point', 'disabled', 'oracle']
CATEGORIES = ['input_reads', 'source_parse', 'candidate_slots', 'pair_tests', 'query_reads',
              'answer_counting', 'state_writes', 'transient_writes', 'mode_changes']


def encode(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clone(obj): return json.loads(encode(obj))


# Freeze and reuse ONLY the original world generator; none of its learning code
# runs. Keeping the actual worlds unchanged is part of the correction contract.
spec = importlib.util.spec_from_file_location('archived_world', RESULTS / 'invalid_initial_run/crossmap.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
world = old.world


def observation(w):
    records = []
    for domain, transitions, groups in [('x', w['source_x'], w['x_group']), ('y', w['source_y'], w['y_group'])]:
        records += [[domain, 'next', a, b] for a, b in transitions]
        records += [[domain, 'group', a, groups[a]] for a in range(6)]
    return records


def outcome(tables, mapping, q, meter):
    kind, x = q['kind'], q['input']
    if kind == 'xy':
        meter['query_reads'] += 2
        return mapping[tables['x_next'][x]]
    if kind == 'yx':
        y = tables['y_next'][x]
        meter['query_reads'] += 1
        for a, b in enumerate(mapping):
            meter['query_reads'] += 1
            if b == y: return a
        raise AssertionError('not a bijection')
    if kind == 'cat':
        meter['query_reads'] += 3
        return 2 * tables['x_group'][x] + tables['y_group'][mapping[x]]
    raise ValueError(kind)


class Learner:
    def __init__(self, method, source_records, oracle_map=None):
        assert method in METHODS
        assert (oracle_map is not None) == (method == 'oracle')
        self.method = method
        self.source = clone(source_records)
        self.bridges = []
        self.maps = None
        self.known = [-1] * 6
        self.inverse = [-1] * 6
        self.tables = None
        self.position = 0
        self.enabled = method not in ('separate', 'disabled')
        self.cost = {k: 0 for k in CATEGORIES}
        self.write_bytes = 0
        self.transient_write_bytes = 0
        self.peak_candidate_count = 0
        self.oracle_gift_entries = 6 if oracle_map is not None else 0
        self.cost['input_reads'] += sum(len(r) for r in self.source)
        self.write(self.source)
        if method != 'recompute': self.tables = self.parse()
        if oracle_map is not None:
            assert sorted(oracle_map) == list(range(6))
            self.maps = [list(oracle_map)]
            self.peak_candidate_count = 1
            self.write(self.maps)

    def write(self, value, transient=False):
        def scalars(v):
            if isinstance(v, dict): return sum(1 + scalars(x) for x in v.values())
            if isinstance(v, (list, tuple)): return sum(scalars(x) for x in v)
            return 1
        self.cost['transient_writes' if transient else 'state_writes'] += scalars(value)
        size = len(encode(value).encode())
        self.write_bytes += size
        if transient: self.transient_write_bytes += size

    def parse(self):
        tables = {f'{domain}_{kind}': [None] * 6 for domain in ('x', 'y') for kind in ('next', 'group')}
        for domain, kind, x, value in self.source:
            self.cost['source_parse'] += 5
            assert tables[f'{domain}_{kind}'][x] is None
            tables[f'{domain}_{kind}'][x] = value
        assert all(x is not None for table in tables.values() for x in table)
        self.write(tables, transient=self.method == 'recompute')
        return tables

    def enumerate(self):
        maps = []
        for permutation in itertools.permutations(range(6)):
            self.cost['candidate_slots'] += 6
            valid = True
            for x, y in self.bridges:
                self.cost['pair_tests'] += 1
                if permutation[x] != y:
                    valid = False
                    break
            if valid: maps.append(list(permutation))
        self.peak_candidate_count = max(self.peak_candidate_count, len(maps))
        self.write(maps, transient=self.method == 'recompute')
        return maps

    def ingest(self, new_pairs):
        if not new_pairs: return
        for x, y in new_pairs:
            self.cost['input_reads'] += 2
            assert self.known[x] == -1 and self.inverse[y] == -1
            self.known[x], self.inverse[y] = y, x
            self.bridges.append([x, y])
            self.write([x, y, y, x])
            if self.method in ('persistent', 'point', 'disabled') and self.maps is not None:
                before = self.maps
                self.cost['pair_tests'] += len(before)
                self.maps = [m for m in before if m[x] == y]
                assert self.maps
                self.write(self.maps)
        if self.method in ('persistent', 'point', 'disabled') and self.maps is None:
            self.maps = self.enumerate()

    def snapshot(self): return clone(self.__dict__)

    @classmethod
    def restore(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(clone(state))
        return obj

    def analytic_histogram(self, q):
        # Counts are sufficient for these one-map-reference queries. This is NOT
        # a factorized joint posterior for arbitrary multi-reference queries.
        tables, kind, x = self.tables, q['kind'], q['input']
        self.cost['query_reads'] += 1
        if kind == 'xy':
            key = tables['x_next'][x]
            direction, remaining = self.known, self.inverse
        elif kind == 'yx':
            key = tables['y_next'][x]
            direction, remaining = self.inverse, self.known
        else:
            key = x
            direction, remaining = self.known, self.inverse
        self.cost['query_reads'] += 1
        if direction[key] != -1:
            targets = [direction[key]]
        else:
            targets = []
            for v, pair in enumerate(remaining):
                self.cost['query_reads'] += 1
                if pair == -1: targets.append(v)
        counts = [0] * (4 if kind == 'cat' else 6)
        for v in targets:
            if kind == 'cat':
                v = 2 * tables['x_group'][x] + tables['y_group'][v]
                self.cost['query_reads'] += 2
            counts[v] += 1
            self.cost['answer_counting'] += 1
        return counts

    def query(self, q):
        before = self.cost.copy()
        bytes_before = self.write_bytes
        if self.method == 'disabled' and q['phase'] == 3 and not self.enabled:
            self.enabled = True
            self.cost['mode_changes'] += 1
            self.write(True)
        count = None
        candidate_count = None
        if self.method == 'recompute':
            tables = self.parse()
            maps = self.enumerate()
        else:
            tables, maps = self.tables, self.maps
        if maps is not None: candidate_count = len(maps)
        if self.enabled:
            if self.method == 'completion':
                count = self.analytic_histogram(q)
            else:
                selected = maps[:1] if self.method == 'point' else maps
                count = [0] * (4 if q['kind'] == 'cat' else 6)
                for mapping in selected:
                    count[outcome(tables, mapping, q, self.cost)] += 1
                    self.cost['answer_counting'] += 1
        self.cost['answer_counting'] += 1 + (len(count) if count else 0)
        self.position += 1
        prediction = None
        if count and sum(n > 0 for n in count) == 1:
            prediction = count.index(next(n for n in count if n))
        return {'counts': count, 'prediction': prediction, 'candidate_count': candidate_count,
                'service_cost': {k: self.cost[k] - before[k] for k in CATEGORIES},
                'service_write_bytes': self.write_bytes - bytes_before,
                'state_bytes': len(encode(self.snapshot()).encode())}

    def local_probe(self):
        # Evaluator invokes this to verify retained local competence. Diagnostic
        # work is kept outside the main 24-query life, for every method equally.
        tables = {f'{d}_{k}': [None] * 6 for d in ('x', 'y') for k in ('next', 'group')}
        for d, k, x, value in self.source: tables[f'{d}_{k}'][x] = value
        return clone(tables if self.method == 'recompute' else self.tables)


def target(w, q):
    xof, yof = w['x_of_latent'], w['y_of_latent']
    value = q['input']
    if q['kind'] == 'xy': return yof[(xof.index(value) + 1) % 6]
    if q['kind'] == 'yx': return xof[yof.index(value) ^ 1]
    latent = xof.index(value)
    return 2 * int(latent >= 3) + latent % 2


def scores(counts, answer):
    if counts is None: return {'brier': None, 'expected_argmax_accuracy': None}
    total = sum(counts)
    brier = sum((Fraction(n, total) - int(i == answer)) ** 2 for i, n in enumerate(counts))
    top = max(counts)
    acc = Fraction(int(counts[answer] == top), counts.count(top))
    pack = lambda x: [x.numerator, x.denominator]
    return {'brier': pack(brier), 'expected_argmax_accuracy': pack(acc)}


def step(learner, packet):
    before = learner.cost.copy()
    byte_before = learner.write_bytes
    learner.ingest(packet['new_pairs'])
    after_ingest = learner.cost.copy()
    ingest_bytes = learner.write_bytes - byte_before
    response = learner.query(packet['query'])
    return {'acquisition_cost': {k: after_ingest[k] - before[k] for k in CATEGORIES},
            'acquisition_write_bytes': ingest_bytes, 'response': response}


def packets(w):
    seen = 0
    output = []
    for q in w['queries']:
        required = w['bridge_before'][q['phase']]
        output.append({'new_pairs': w['bridges'][seen:required], 'query': q})
        seen = required
    return output


def run(w, method):
    gift = [w['true_map'][i] for i in range(6)] if method == 'oracle' else None
    l = Learner(method, observation(w), gift)
    setup = {'cost': l.cost.copy(), 'write_bytes': l.write_bytes}
    records, snapshots, seen_queries = [], [], set()
    for t, packet in enumerate(packets(w)):
        response = step(l, packet)
        q = packet['query']
        y = target(w, q)
        prediction = response['response']['prediction']
        key = (q['kind'], q['input'])
        if q['kind'] == 'xy': bridge_known = l.known[l.local_probe()['x_next'][q['input']]] != -1
        elif q['kind'] == 'yx': bridge_known = l.inverse[l.local_probe()['y_next'][q['input']]] != -1
        else: bridge_known = l.known[q['input']] != -1
        records.append({'t': t, 'query': q, 'target': y, 'new_query': key not in seen_queries,
                        'queried_pair_observed': bridge_known,
                        'correct': prediction == y, 'abstained': prediction is None,
                        'confident_error': prediction is not None and prediction != y,
                        'scores': scores(response['response']['counts'], y), 'engine': response})
        seen_queries.add(key)
        if (t + 1) % 6 == 0: snapshots.append(l.snapshot())
    return {'setup': setup, 'records': records, 'snapshots': snapshots, 'final': l.snapshot(),
            'final_local_probe': l.local_probe()}


def experiment():
    return {'round': 22, 'status': 'corrected_post_result', 'python': platform.python_version(),
            'source_sha256': sha(__file__), 'amendment_sha256': sha(DOCS / 'AMENDMENT.md'),
            'archived_generator_sha256': sha(RESULTS / 'invalid_initial_run/crossmap.py'),
            'methods': METHODS,
            'runs': [{'seed': s, 'world': world(s), 'tracks': {m: run(world(s), m) for m in METHODS}} for s in range(24)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--resume', nargs=2)
    args = parser.parse_args()
    if args.resume:
        source, dest = map(Path, args.resume)
        request = json.loads(source.read_text(encoding='utf-8'))
        learner = Learner.restore(request['state'])
        records = [step(learner, p) for p in request['tail']]
        dest.write_text(encode({'records': records, 'final': learner.snapshot()}), encoding='utf-8')
        return
    result = experiment()
    dest = RESULTS / 'crossmap_corrected.json'
    if args.write:
        dest.write_text(encode(result) + '\n', encoding='utf-8')
        print('wrote', dest.stat().st_size, 'bytes')
    else:
        # Generated world dictionaries use integer keys; JSON object keys are
        # strings. Compare the canonical JSON value, not pre-serialization types.
        assert clone(result) == json.loads(dest.read_text(encoding='utf-8'))
        print('exact_rerun_passed')


if __name__ == '__main__': main()
