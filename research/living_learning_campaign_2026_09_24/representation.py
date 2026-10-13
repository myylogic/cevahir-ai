"""REP wave 1: error-directed acquisition of a supplied feature language.

Virtual work counts arithmetic/comparison/write operations; not CPU timings.
No teacher coefficients or future labels enter Learner.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
N = 192


def hypotheses():
    return json.loads((HERE / 'catalog_representation.json').read_text(encoding='utf-8'))


def features(x):
    return np.array([1.0, *x, *(x[i] * x[j] for i, j in PAIRS)], dtype=float)


class Learner:
    def __init__(self, mode='adaptive', decay=1.0, replay=True, seed=0):
        self.mode, self.decay, self.replay, self.seed = mode, decay, replay, seed
        self.active = list(range(11 if mode == 'full' else 5))
        self.P = np.eye(len(self.active))
        self.w = np.zeros(len(self.active))
        self.scores = np.zeros(6)
        self.mass = 0.0
        self.history = []
        self.t = 0
        self.work = {'predict': 0, 'update': 0, 'detect': 0, 'rebuild': 0, 'archive': 0}
        # Initial feature indices, covariance, weights and detector state writes.
        self.work['initialization'] = len(self.active) ** 2 + 2 * len(self.active) + 7
        self.acquisitions = []
        self.last_phi = None

    @property
    def adaptive(self):
        return self.mode in ('adaptive', 'random')

    def predict(self, x):
        # Full raw feature evaluation is charged even if some features are unused.
        self.last_phi = features(x)
        k = len(self.active)
        self.work['predict'] += 6 + 11 + 2 * k + 2
        return float(np.clip(self.last_phi[self.active] @ self.w, -6, 6))

    def _update(self, phi, y, bucket):
        v = phi[self.active]
        k = len(v)
        pv = self.P @ v
        den = self.decay + v @ pv
        residual = y - v @ self.w
        self.w += pv * (residual / den)
        self.P = (self.P - np.outer(pv, pv) / den) / self.decay
        # P@v (2k²), two dot products (4k), denominator/residual/division (3),
        # weight update (2k), outer/subtract/divisions (4k²).
        self.work[bucket] += 6 * k * k + 6 * k + 3

    def observe(self, x, y, prediction):
        phi = self.last_phi
        if phi is None:
            raise RuntimeError('predict must precede observe')
        self.t += 1
        self._update(phi, y, 'update')
        if self.adaptive:
            self.history.append([*[float(a) for a in x], float(y)])
            self.work['archive'] += 5
            residual = y - prediction
            self.scores = .98 * self.scores + residual * phi[5:]
            self.mass = .98 * self.mass + 1
            self.work['detect'] += 3 * 6 + 3
            inactive = [j for j in range(5, 11) if j not in self.active]
            self.work['detect'] += 6 * len(self.active)
            if self.t % 24 == 0 and len(self.active) < 8 and inactive:
                strengths = [abs(self.scores[j - 5]) / self.mass for j in inactive]
                self.work['detect'] += 3 * len(inactive)
                best = int(np.argmax(strengths))
                if strengths[best] > .08:
                    if self.mode == 'random':
                        # Seed/t-only deterministic choice; same error-trigger rule.
                        best = (self.seed * 1103515245 + self.t * 12345) % len(inactive)
                    self._acquire(inactive[best])
        self.last_phi = None

    def _acquire(self, j):
        old_k = len(self.active)
        self.active.append(j)
        self.acquisitions.append([self.t, j])
        k = len(self.active)
        if self.replay:
            self.P = np.eye(k)
            self.w = np.zeros(k)
            self.work['rebuild'] += k * k + k
            for row in self.history:
                self.work['rebuild'] += 17
                self._update(features(row[:4]), row[4], 'rebuild')
        else:
            bigger = np.eye(k)
            bigger[:old_k, :old_k] = self.P
            self.P = bigger
            self.w = np.r_[self.w, 0.0]
            self.work['rebuild'] += k * k + old_k * old_k + k
        self.scores[:] = 0
        self.mass = 0.0
        self.work['rebuild'] += 7

    def dump(self):
        return json.dumps({
            'mode': self.mode, 'decay': self.decay, 'replay': self.replay,
            'seed': self.seed, 'active': self.active, 'P': self.P.tolist(),
            'w': self.w.tolist(), 'scores': self.scores.tolist(), 'mass': self.mass,
            'history': self.history, 't': self.t, 'work': self.work,
            'acquisitions': self.acquisitions,
        }, sort_keys=True)

    @classmethod
    def restore(cls, state):
        d = json.loads(state)
        obj = cls(d['mode'], d['decay'], d['replay'], d['seed'])
        for key, value in d.items():
            setattr(obj, key, np.array(value) if key in ('P', 'w', 'scores') else value)
        return obj


def stream(seed, regime):
    # Sampling sources depend only on seed/regime, never learner behavior.
    code = ['interaction', 'additive', 'noisy', 'rare', 'covariate', 'onset', 'switch'].index(regime)
    rng = np.random.default_rng(seed * 1009 + 71 * code + 31)
    mains = rng.choice([-1., 1.], 4) * np.array([.6, .3, 0., 0.])
    pair_order = rng.permutation(6)
    p, q = int(pair_order[0]), int(pair_order[1])
    sign = float(rng.choice([-1., 1.]))
    rows = []
    for t in range(N):
        x = rng.choice([-1., 1.], 4)
        if regime == 'rare':
            x[0] = 1. if rng.random() < .08 else -1.
            x[1] = 1. if rng.random() < .08 else -1.
        if regime == 'covariate' and t < N // 2:
            x[1] = x[0] if rng.random() < .95 else -x[0]
        phi = features(x)
        y = .2 + float(mains @ x)
        if regime in ('interaction', 'rare', 'covariate'):
            y += sign * phi[5 + p]
        elif regime == 'onset' and t >= N // 2:
            y += sign * phi[5 + p]
        elif regime == 'switch':
            y += sign * phi[5 + p] if t < N // 2 else -sign * phi[5 + q]
        noise = 1.2 if regime == 'noisy' else .15
        y = float(np.clip(y + rng.normal(0, noise), -6, 6))
        rows.append((x.tolist(), y))
    return rows


def evaluate(rows, mode, seed, decay=1., replay=True):
    model = Learner(mode, decay, replay, seed)
    errors = []
    for x, y in rows:
        pred = model.predict(x)
        errors.append((pred - y) ** 2)
        model.observe(x, y, pred)
    k = len(model.active)
    memory = k * k + k + (6 + 1 + 5 * len(rows) if model.adaptive else 0)
    return {'mse': float(np.mean(errors)), 'second_half_mse': float(np.mean(errors[N//2:])),
            'cost': int(sum(model.work.values())), 'cost_breakdown': model.work,
            'memory_scalars': memory, 'active_features': model.active,
            'acquisitions': model.acquisitions, 'bounded_errors': bool(all(0 <= e <= 144 for e in errors)),
            'state': model.dump()}


SPEC = [
    ('REP01', 'interaction', ('adaptive', 1., True), ('linear', 1., True), 'mse'),
    ('REP02', 'interaction', ('adaptive', 1., True), ('full', 1., True), 'mse'),
    ('REP03', 'additive', ('adaptive', 1., True), ('full', 1., True), 'mse'),
    ('REP04', 'interaction', ('adaptive', 1., True), ('random', 1., True), 'mse'),
    ('REP05', 'switch', ('adaptive', .97, True), ('adaptive', 1., True), 'mse'),
    ('REP06', 'switch', ('adaptive', .97, True), ('full', .97, True), 'mse'),
    ('REP07', 'noisy', ('adaptive', 1., True), ('full', 1., True), 'mse'),
    ('REP08', 'rare', ('adaptive', 1., True), ('linear', 1., True), 'mse'),
    ('REP09', 'covariate', ('adaptive', 1., True), ('full', 1., True), 'mse'),
    ('REP10', 'onset', ('adaptive', .97, True), ('full', .97, True), 'mse'),
    ('REP11', 'interaction', ('adaptive', 1., True), ('full', 1., True), 'cost'),
    ('REP12', 'interaction', ('adaptive', 1., True), ('adaptive', 1., False), 'mse'),
]


def method_name(spec):
    mode, decay, replay = spec
    return f'{mode}_decay{decay:g}_' + ('replay' if replay else 'zero_history')


def run(seed):
    cache = {}
    streams = {regime: stream(seed, regime) for regime in {r[1] for r in SPEC}}
    output = {}
    for hid, regime, candidate, control, metric in SPEC:
        records = []
        for cfg in (candidate, control):
            key = (regime, cfg)
            if key not in cache:
                cache[key] = evaluate(streams[regime], cfg[0], seed, cfg[1], cfg[2])
            records.append(cache[key])
        ca, co = records
        output[hid] = {'candidate': ca[metric], 'control': co[metric],
                      'costs': {'candidate': ca['cost'], 'control': co['cost']},
                      'diagnostics': {'candidate': {k: v for k, v in ca.items() if k != 'state'},
                                      'control': {k: v for k, v in co.items() if k != 'state'},
                                      'stream_length': N, 'regime': regime}}
    checks = {
        'all_prequential_errors_bounded': all(d['bounded_errors'] for d in cache.values()),
        'nonnegative_charged_costs': all(d['cost'] > 0 and all(v >= 0 for v in d['cost_breakdown'].values()) for d in cache.values()),
        'feature_budget': all(len(d['active_features']) <= (11 if k[1][0] == 'full' else 8) for k, d in cache.items()),
        'wave1_coverage': set(output) == {h['id'] for h in hypotheses() if h['status'] == 'wave1'},
    }
    return {'hypotheses': output, 'checks': checks,
            'meta': {'seed': seed, 'module': 'representation', 'n_per_stream': N,
                     'teacher_access': 'observed label only, after prediction',
                     'feature_language': 'supplied intercept + 4 coordinates + 6 pair products'}}


def selfcheck():
    checks = []
    # Independent batch normal-equation solution checks the sequential RLS recurrence.
    rng = np.random.default_rng(85931)
    rows = [(rng.normal(size=4).tolist(), float(rng.normal())) for _ in range(42)]
    for decay in (1., .97):
        for mode in ('linear', 'full'):
            model = Learner(mode, decay)
            k = len(model.active)
            S, b = np.eye(k), np.zeros(k)
            for x, y in rows:
                phi = features(x)[model.active]
                pred = model.predict(x)
                checks.append(np.allclose(pred, np.clip(phi @ np.linalg.solve(S, b), -6, 6), atol=1e-9))
                model.observe(x, y, pred)
                S, b = decay * S + np.outer(phi, phi), decay * b + phi * y
            checks.append(np.allclose(model.P, np.linalg.inv(S), atol=1e-9))
    # Dynamic installation reconstructs the same fixed-feature posterior from history.
    adaptive = Learner('adaptive')
    for x, y in rows[:15]:
        p = adaptive.predict(x)
        adaptive.observe(x, y, p)
    adaptive._acquire(7)
    X = np.array([features(x)[adaptive.active] for x, _ in rows[:15]])
    yy = np.array([y for _, y in rows[:15]])
    checks.append(np.allclose(adaptive.w, np.linalg.solve(np.eye(6) + X.T @ X, X.T @ yy), atol=1e-9))
    # Persistence: byte-identical continuation, including costs and acquisition decisions.
    restored = Learner.restore(adaptive.dump())
    for x, y in rows[15:]:
        p1, p2 = adaptive.predict(x), restored.predict(x)
        checks.append(p1 == p2)
        adaptive.observe(x, y, p1)
        restored.observe(x, y, p2)
        checks.append(adaptive.dump() == restored.dump())
    # Mechanism ablation: prediction becomes the exact linear comparator if no installation.
    no_acquisition, linear = Learner('adaptive'), Learner('linear')
    for x, y in rows[:23]:  # first acquisition can occur only after label 24
        p1, p2 = no_acquisition.predict(x), linear.predict(x)
        checks.append(p1 == p2)
        no_acquisition.observe(x, y, p1)
        linear.observe(x, y, p2)
    # Independent scalar feature products, all binary inputs.
    import itertools
    for bits in itertools.product((-1., 1.), repeat=4):
        expected = [1., bits[0], bits[1], bits[2], bits[3],
                    bits[0]*bits[1], bits[0]*bits[2], bits[0]*bits[3],
                    bits[1]*bits[2], bits[1]*bits[3], bits[2]*bits[3]]
        checks.append(features(bits).tolist() == expected)
    return {'checks': len(checks), 'passed': bool(all(checks)),
            'details': 'Independent normal equations; fixed-basis reconstruction; JSON continuation; installation ablation; scalar feature map.'}


if __name__ == '__main__':
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))
