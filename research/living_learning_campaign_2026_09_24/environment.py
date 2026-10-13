"""Action-produced experience: declared bandit mechanisms, no novelty claim."""
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
REGIMES = ['stationary_gap', 'stationary_small', 'single_reversal', 'recurring_reversal']
METHODS = ['greedy', 'epsilon', 'ucb', 'window_ucb']
CONTRASTS = [('ucb', 'greedy'), ('epsilon', 'greedy'), ('window_ucb', 'ucb')]


def hypotheses():
    return json.loads((HERE / 'catalog_environment.json').read_text(encoding='utf-8'))


class Policy:
    def __init__(self, method, epsilon=0.10):
        self.method, self.epsilon = method, epsilon
        self.counts, self.sums = [0, 0], [0, 0]
        self.history = []
        self.t = 0
        self.work = 0
        self.max_records = 0

    def choose(self, uniform):
        self.work += 2
        if self.method == 'window_ucb':
            self.counts, self.sums = [0, 0], [0, 0]
            self.work += 4
            for _, action, reward in self.history:
                self.counts[action] += 1
                self.sums[action] += reward
                self.work += 4
        for a in range(2):
            self.work += 1
            if self.counts[a] == 0: return a
        means = [self.sums[a] / self.counts[a] for a in range(2)]
        self.work += 4
        if self.method == 'epsilon':
            self.work += 2
            if uniform < self.epsilon: return int(uniform / self.epsilon >= 0.5)
        if self.method in ('ucb', 'window_ucb'):
            horizon = max(2, sum(self.counts))
            means = [means[a] + 0.5 * math.sqrt(math.log(horizon) / self.counts[a]) for a in range(2)]
            self.work += 10
        self.work += 1
        return int(means[1] > means[0])

    def observe(self, action, reward):
        self.counts[action] += 1
        self.sums[action] += reward
        self.work += 4
        if self.method == 'window_ucb':
            self.history.append([self.t, action, reward])
            self.work += 3
            if len(self.history) > 64:
                self.history.pop(0)
                self.work += 64  # charge shifted references in this simple list
            self.max_records = max(self.max_records, len(self.history))
        self.t += 1
        self.work += 1

    def snapshot(self): return json.loads(json.dumps(self.__dict__))

    @classmethod
    def restore(cls, state):
        obj = cls(state['method'])
        obj.__dict__.update(json.loads(json.dumps(state)))
        return obj


def make_world(seed, regime):
    rng = random.Random(231000 + seed * 19 + REGIMES.index(regime) * 997)
    first_best = rng.randrange(2)
    jitter = rng.uniform(-0.02, 0.02)
    means, reward = [], []
    for t in range(512):
        flip = int(t >= 256) if regime == 'single_reversal' else t // 128 % 2 if regime == 'recurring_reversal' else 0
        best = first_best ^ flip
        high, low = ((0.55, 0.45) if regime == 'stationary_small' else (0.8, 0.2))
        p = [low + jitter, low + jitter]
        p[best] = high + jitter
        means.append(p)
        reward.append([int(rng.random() < v) for v in p])
    uniforms = [rng.random() for _ in range(512)]
    return means, reward, uniforms


def trajectory(seed, regime, method):
    means, rewards, uniforms = make_world(seed, regime)
    learner = Policy(method)
    regret, observed, counts = [], [], [0, 0]
    trace = []
    for t in range(512):
        a = learner.choose(uniforms[t])
        r = rewards[t][a]
        regret.append(max(means[t]) - means[t][a])
        observed.append(r)
        counts[a] += 1
        learner.observe(a, r)
        if seed in (0, 1000): trace.append([t, a, r, regret[-1]])
    return dict(metric=sum(regret) / 512, cost=learner.work,
                first_half_regret=sum(regret[:256]) / 256,
                second_half_regret=sum(regret[256:]) / 256,
                action_counts=counts, obtained_reward=sum(observed),
                max_raw_records=learner.max_records,
                state_json_bytes=len(json.dumps(learner.snapshot()).encode()),
                trace=trace)


def run(seed):
    results, cache = {}, {}
    for h in hypotheses():
        if h['status'] != 'wave1': continue
        pair = []
        for method in (h['contrast']['candidate'], h['contrast']['control']):
            key = (h['regime'], method)
            if key not in cache: cache[key] = trajectory(seed, *key)
            pair.append(cache[key])
        a, b = pair
        results[h['id']] = dict(candidate=a['metric'], control=b['metric'],
            costs=dict(candidate=a['cost'], control=b['cost']),
            diagnostics=dict(candidate=a, control=b, distinct_observed_histories=True))
    return dict(hypotheses=results,
                checks=dict(bounded_regret=all(0 <= x['metric'] <= 1 for x in cache.values()),
                            complete_lifetimes=all(sum(x['action_counts']) == 512 for x in cache.values()),
                            bounded_window=all(x['max_raw_records'] <= 64 for x in cache.values())),
                meta=dict(steps=512, methods=METHODS, regimes=REGIMES,
                          data_scope='Shared potential outcomes; each policy observes only its selected reward.'))


def selfcheck():
    checks = 0
    means, rewards, uniforms = make_world(777, 'single_reversal')
    for method in METHODS:
        policy = Policy(method)
        for t in range(207):
            action = policy.choose(uniforms[t]); policy.observe(action, rewards[t][action])
        restored = Policy.restore(policy.snapshot())
        for t in range(207, 512):
            a = policy.choose(uniforms[t]); b = restored.choose(uniforms[t])
            assert a == b
            policy.observe(a, rewards[t][a]); restored.observe(b, rewards[t][b])
            assert policy.snapshot() == restored.snapshot()
            checks += 1
    greedy, eps = Policy('greedy'), Policy('epsilon', epsilon=0)
    for t in range(512):
        aa, bb = greedy.choose(uniforms[t]), eps.choose(uniforms[t])
        assert aa == bb
        greedy.observe(aa, rewards[t][aa]); eps.observe(bb, rewards[t][bb]); checks += 1
    window = Policy('window_ucb')
    seen = []
    for t in range(150):
        window.choose(0.3)
        assert window.counts == [sum(a == v for a, r in seen[-64:]) for v in range(2)]
        assert window.sums == [sum(r for a, r in seen[-64:] if a == v) for v in range(2)]
        action, reward = t % 2, int(t % 3 == 0)
        window.observe(action, reward); seen.append((action, reward)); checks += 1
    # Direct conditional-expectation identity on two fixed arms.
    for p0, p1 in ((0.2, 0.8), (0.7, 0.3)):
        for a in (0, 1):
            expected_reward = sum(r * ((p0, p1)[a] if r else 1 - (p0, p1)[a]) for r in (0, 1))
            assert abs((max(p0, p1) - expected_reward) - (max(p0, p1) - (p0, p1)[a])) < 1e-12
            checks += 1
    return dict(checks=checks, passed=True, details='1220 restored-state steps, 512 epsilon-off comparisons, 150 window references, 4 regret identities.')
