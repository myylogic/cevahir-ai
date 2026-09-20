"""Prediction preservation is weaker than preservation of future updates.

Small, deterministic, standard-library experiments. This does not propose a new
optimizer: the transported metric is a change-of-coordinates calculation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def mv(a, x):
    return [dot(row, x) for row in a]


def tr(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
A = [[1., 0., 0.], [0., 3., 1.], [0., 0., .25]]
A_INV = [[1., 0., 0.], [0., 1/3, -4/3], [0., 0., 4.]]
METRIC = mm(tr(A_INV), A_INV)


def update(w, x, y, metric=IDENTITY, eta=.04):
    e = dot(w, x) - y
    step = mv(metric, [e * v for v in x])
    return [v - eta * d for v, d in zip(w, step)]


def mse(w, queries, target, transform=IDENTITY):
    return statistics.fmean((dot(w, mv(transform, x))-dot(target, x))**2
                            for x in queries)


def summary(values):
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.
    return {'mean': mean, 'min': min(values), 'max': max(values),
            'paired_seed_mean_normal_ci95': [mean - 1.96*se, mean + 1.96*se]}


def coordinate_run(seed, changed_target):
    rng = random.Random(9000 + seed)
    sample = lambda: [1., rng.uniform(-1, 1), rng.uniform(-1, 1)]
    queries = [sample() for _ in range(256)]
    old_target = [.3, 1.2, -.8]
    target = [-.1, -.4, 1.1] if changed_target else old_target
    old = [0., 0., 0.]
    for _ in range(96):
        x = sample()
        old = update(old, x, dot(old_target, x) + rng.gauss(0, .02))
    weight_only = mv(tr(A_INV), old)
    full = weight_only[:]
    unmapped = old[:]
    before = {
        'reference_mse': mse(old, queries, old_target),
        'weights_transported_mse': mse(weight_only, queries, old_target, A),
        'unmapped_mse': mse(unmapped, queries, old_target, A),
        'max_prediction_difference_after_transport': max(
            abs(dot(old, x) - dot(weight_only, mv(A, x))) for x in queries),
    }
    trajectory_max = {'weights_only': 0., 'weights_and_metric': 0.}
    checkpoint = []
    for step in range(1, 193):
        x = sample()
        y = dot(target, x) + rng.gauss(0, .02)
        old = update(old, x, y)
        weight_only = update(weight_only, mv(A, x), y)
        full = update(full, mv(A, x), y, METRIC)
        for name, w in [('weights_only', weight_only), ('weights_and_metric', full)]:
            err = max(abs(dot(old, q)-dot(w, mv(A, q))) for q in queries)
            trajectory_max[name] = max(trajectory_max[name], err)
        if step in (1, 16, 64, 192):
            checkpoint.append({'step': step, 'reference_mse': mse(old, queries, target),
                               'weights_only_mse': mse(weight_only, queries, target, A),
                               'weights_and_metric_mse': mse(full, queries, target, A)})
    # JSON persistence discards the experience stream; resumes execution AND update.
    persisted = json.loads(json.dumps({'weights': full, 'metric': METRIC, 'features': A}))
    x = sample()
    y = dot(target, x)
    next_original = update(old, x, y)
    next_restored = update(persisted['weights'], mv(persisted['features'], x), y,
                           persisted['metric'])
    resume_err = max(abs(dot(next_original, q)-dot(next_restored, mv(A, q))) for q in queries)
    return {'seed': seed, 'before': before, 'trajectory_max_difference': trajectory_max,
            'checkpoints': checkpoint, 'json_restored_next_update_max_difference': resume_err,
            'persisted_state': persisted}


def stats(rows, interaction=False):
    dim = 4 if interaction else 3
    g = [[0]*dim for _ in range(dim)]
    b = [0]*dim
    for x1, x2, y in rows:
        phi = [1, x1, x2] + ([x1*x2] if interaction else [])
        for i in range(dim):
            b[i] += phi[i]*y
            for j in range(dim):
                g[i][j] += phi[i]*phi[j]
    return {'count': len(rows), 'G': g, 'b': b, 'sum_y_squared': sum(r[2]**2 for r in rows)}


def lossy_migration_witness():
    positive = [(x, y, x*y) for x in (-1, 1) for y in (-1, 1)]
    negative = [(x, y, -x*y) for x in (-1, 1) for y in (-1, 1)]
    old1, old2 = stats(positive), stats(negative)
    new1, new2 = stats(positive, True), stats(negative, True)
    assert old1 == old2 and new1 != new2
    # Even arbitrary identical predictions over the four points cannot fit both worlds:
    # ((v-q)^2 + (v+q)^2)/2 = v^2+1 >= 1, point by point.
    return {'positive_history': positive, 'negative_history': negative,
            'old_summaries_equal': old1 == old2, 'common_old_summary': old1,
            'new_positive_summary': new1, 'new_negative_summary': new2,
            'old_least_squares_weights': [0, 0, 0],
            'new_positive_weights': [0, 0, 0, 1], 'new_negative_weights': [0, 0, 0, -1],
            'old_model_mse_both_histories': 1,
            'raw_replay_expanded_fit_mse_both_histories': 0,
            'retaining_missing_cross_moment_expanded_fit_mse_both_histories': 0,
            'minimum_worst_world_mse_without_extra_information': 1,
            'scope': 'Exact integer witness. The missing interaction was not retained; future new labels can repair it.'}


def process_probe(state_path):
    state = json.loads(Path(state_path).read_text(encoding='utf-8'))
    x = [1., -.25, .75]
    y = .9
    w = update(state['weights'], mv(state['features'], x), y, state['metric'])
    print(json.dumps({'prediction': dot(state['weights'], mv(state['features'], x)),
                      'after_update': w}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).parent/'results'/'state_transport.json'))
    parser.add_argument('--probe')
    args = parser.parse_args()
    if args.probe:
        process_probe(args.probe)
        return
    assert max(abs(mm(A, A_INV)[i][j]-IDENTITY[i][j]) for i in range(3) for j in range(3)) < 1e-14
    regimes = {}
    for name, shift in [('stationary_target', False), ('changed_target', True)]:
        runs = [coordinate_run(seed, shift) for seed in range(32)]
        regimes[name] = {'runs': runs, 'summary': {
            'initial_prediction_difference': summary([r['before']['max_prediction_difference_after_transport'] for r in runs]),
            'weight_only_trajectory_max_difference': summary([r['trajectory_max_difference']['weights_only'] for r in runs]),
            'full_transport_trajectory_max_difference': summary([r['trajectory_max_difference']['weights_and_metric'] for r in runs]),
            'final_mse_reference': summary([r['checkpoints'][-1]['reference_mse'] for r in runs]),
            'final_mse_weights_only': summary([r['checkpoints'][-1]['weights_only_mse'] for r in runs]),
            'final_mse_difference_weights_only_minus_reference': summary([
                r['checkpoints'][-1]['weights_only_mse']-r['checkpoints'][-1]['reference_mse'] for r in runs]),
        }}
    state = regimes['stationary_target']['runs'][0]['persisted_state']
    with tempfile.TemporaryDirectory(prefix='learning-state-') as tmp:
        path = Path(tmp)/'state.json'
        path.write_text(json.dumps(state), encoding='utf-8')
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe', str(path)],
                                   capture_output=True, text=True, check=True)
        fresh = json.loads(completed.stdout)
    expected = {'prediction': dot(state['weights'], mv(A, [1., -.25, .75])),
                'after_update': update(state['weights'], mv(A, [1., -.25, .75]), .9, METRIC)}
    assert fresh == expected
    largest_error = max(r['trajectory_max_difference']['weights_and_metric']
                        for v in regimes.values() for r in v['runs'])
    assert largest_error < 1e-11
    result = {'experiment': 'state_transport_and_future_sufficiency',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol': {'seeds_per_regime': 32, 'pre_migration_experiences': 96,
                           'post_migration_experiences': 192, 'heldout_queries': 256,
                           'label_noise_sigma': .02, 'step_size': .04,
                           'A': A, 'A_inverse': A_INV, 'transported_metric': METRIC,
                           'paired_random_streams': True,
                           'claim': 'Equivalence of future update dynamics under reparameterization, not superiority of any optimizer.',
                           'resources': 'All methods have 3 trainable weights. Full metric stores 9 numbers and uses dense 3x3 matvec; identity baseline does not need metric storage. Transform is supplied.'},
              'regimes': regimes, 'lossy_migration': lossy_migration_witness(),
              'fresh_process_one_seed': {'expected': expected, 'observed': fresh, 'equal': fresh == expected},
              'checks': {'full_transport_largest_error': largest_error, 'passed': True}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'path': str(out), 'checks': result['checks'],
                      'summary': {k: v['summary'] for k, v in regimes.items()}}))


if __name__ == '__main__':
    main()
