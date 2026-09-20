"""Identify a temporal input/output relation without supplied episode IDs.

This is linear system identification, not anonymous-label matching or a new
credit-assignment method. Fixed finite lag vocabulary, exogenous input positive
case, and observed numeric output are supplied assumptions.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

SEEDS = 24
HORIZON = 1800
CHANGE_AT = 900
NOISE_SD = .05
LAGS = list(range(7))
REGIMES = ['iid_stationary', 'iid_coefficient_change', 'alternating_stationary', 'outside_lag_vocabulary']
METHODS = ['frozen_zero', 'current_only', 'all_lags', 'fading_lags', 'known_support_reference']


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


class RLS:
    def __init__(self, lags, forgetting=1.):
        self.lags = list(lags)
        self.forgetting = forgetting
        d = len(lags)+1
        self.w = [0.]*d
        self.p = [[float(i == j) for j in range(d)] for i in range(d)]
        self.cost = Counter()

    def phi(self, history):
        return [1.] + [history[k] for k in self.lags]

    def predict(self, history):
        return dot(self.w, self.phi(history))

    def observe(self, history, y):
        x = self.phi(history)
        d = len(x)
        px = [dot(row, x) for row in self.p]
        denominator = self.forgetting + dot(x, px)
        gain = [v/denominator for v in px]
        err = y-dot(self.w, x)
        self.w = [w + g*err for w, g in zip(self.w, gain)]
        self.p = [[(self.p[i][j]-gain[i]*px[j])/self.forgetting for j in range(d)] for i in range(d)]
        self.cost['updates'] += 1
        self.cost['multiply_add_pairs'] += 2*d*d+4*d
        self.cost['scalar_divisions'] += d+d*d

    def state(self):
        return {'lags': self.lags, 'forgetting': self.forgetting,
                'w': self.w, 'p': self.p, 'cost': dict(self.cost)}

    @classmethod
    def restore(cls, state):
        model = cls(state['lags'], state['forgetting'])
        model.w = list(state['w'])
        model.p = [list(row) for row in state['p']]
        model.cost = Counter(state['cost'])
        return model


def truth(history, regime, step):
    lag2 = 10 if regime == 'outside_lag_vocabulary' else 5
    a, b = (1.1, -.7)
    if regime == 'iid_coefficient_change' and step >= CHANGE_AT:
        a, b = (-.4, 1.2)
    return .2+a*history[2]+b*history[lag2]


def sample_history(rng):
    return [rng.uniform(-1., 1.) for _ in range(11)]


def expanded_weights(model):
    w = [model.w[0]]+[0.]*11
    for lag, value in zip(model.lags, model.w[1:]):
        w[lag+1] = value
    return w


def summary(values):
    m = statistics.fmean(values)
    se = statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.
    return {'mean': m, 'min': min(values), 'max': max(values),
            'mean_normal_ci95': [m-1.96*se, m+1.96*se]}


def run_seed(seed, regime):
    rng = random.Random(180000+seed)
    noise = random.Random(240000+seed)
    eval_rng = random.Random(360000+seed)
    history = sample_history(rng)
    if regime == 'alternating_stationary':
        history = [(-1.)**(-1-k) for k in range(11)]
    support = [2, 10] if regime == 'outside_lag_vocabulary' else [2, 5]
    models = {name: RLS(([0] if name=='current_only' else support if name=='known_support_reference' else LAGS),
                        .995 if name=='fading_lags' else 1.) for name in METHODS}
    losses = {name: [0., 0.] for name in METHODS}
    tail = {name: 0. for name in METHODS}
    probes = [sample_history(eval_rng) for _ in range(512)]
    checkpoint = {}
    restored = {}
    restore_error = {name: 0. for name in METHODS}
    stripped = {}
    stripped_prediction_difference = {name: 0. for name in METHODS}
    restored_history_equal = True
    for t in range(HORIZON):
        new_u = (-1.)**t if regime=='alternating_stationary' else rng.uniform(-1., 1.)
        history = [new_u]+history[:10]
        clean = truth(history, regime, t)
        y = clean+noise.gauss(0., NOISE_SD)
        for name, model in models.items():
            pred = model.predict(history)
            loss = (pred-clean)**2
            losses[name][int(t>=CHANGE_AT)] += loss/CHANGE_AT
            if t >= HORIZON-200:
                tail[name] += loss/200
            if name in restored:
                clone, saved_history = restored[name]
                saved_history[:] = [new_u]+saved_history[:10]
                restored_history_equal &= saved_history == history
                restore_error[name] = max(restore_error[name], abs(clone.predict(saved_history)-pred))
                stripped_prediction_difference[name] = max(stripped_prediction_difference[name],
                                                          abs(stripped[name].predict(history)-pred))
                if name != 'frozen_zero':
                    clone.observe(saved_history, y)
                    stripped[name].observe(history, y)
            if name != 'frozen_zero':
                model.observe(history, y)
            if t == CHANGE_AT-1:
                bundle = json.loads(json.dumps({'model': model.state(), 'history': history}))
                restored[name] = (RLS.restore(bundle['model']), bundle['history'])
                stripped[name] = RLS(model.lags, model.forgetting)
                stripped[name].w = list(model.w)
                assert all(stripped[name].predict(q)==model.predict(q) for q in probes)
                checkpoint[name] = {'prechange_iid_probe_mse': statistics.fmean(
                    (model.predict(q)-truth(q, regime, t))**2 for q in probes)}
    records = {}
    target_weights = [0.2]+[0.]*11
    target_weights[3] = -.4 if regime=='iid_coefficient_change' else 1.1
    target_weights[(10 if regime=='outside_lag_vocabulary' else 5)+1] = 1.2 if regime=='iid_coefficient_change' else -.7
    for name, model in models.items():
        full_state_equal = model.state()==restored[name][0].state()
        assert full_state_equal and restore_error[name]==0
        records[name] = {
            'lifetime_service_mse': statistics.fmean(losses[name]),
            'first_half_service_mse': losses[name][0], 'second_half_service_mse': losses[name][1],
            'last200_service_mse': tail[name],
            'iid_heldout_mse': statistics.fmean((model.predict(q)-truth(q, regime, HORIZON))**2 for q in probes),
            'coefficient_l2_error': math.sqrt(sum((a-b)**2 for a,b in zip(expanded_weights(model), target_weights))),
            'weights_by_lag': expanded_weights(model), 'checkpoint': checkpoint[name],
            'cost': dict(model.cost),
            'parameter_count': len(model.w), 'inverse_gram_scalar_slots': len(model.w)**2,
            'necessary_input_history_slots': max(model.lags)+1,
            'persistence': {'continued_samples': HORIZON-CHANGE_AT,
                            'max_prediction_difference': restore_error[name], 'full_final_state_equal': full_state_equal,
                            'reset_matrix_max_future_prediction_difference': stripped_prediction_difference[name]},
        }
    assert restored_history_equal
    return {'seed': seed, 'methods': records,
            'environment_state_uses_11_inputs': True, 'method_history_rights': 'unprivileged max7, known-support outside-vocabulary reference max11'}


def exact_witnesses():
    alternating = [(-1)**t for t in range(20)]
    # Distinct nonzero lag assignments give exactly the same observed outputs.
    assert all(alternating[t] == -alternating[t-1] for t in range(1,20))
    common_prediction = .5  # midpoint of structurally different query answers0,1.
    return {'alternating_alias': {'theta_A': {'lag0': 1}, 'theta_B': {'lag1': -1},
                                 'observed_predictions_identical': True,
                                 'query_history_lag0_lag1': [1,0], 'query_answers': [1,0]},
            'causal_alias': {'normal_equations_both': 'U exogenous iid bit, A=U, Y=A=U',
                            'world_1_structural_Y': 'Y=U; action A has no effect',
                            'world_2_structural_Y': 'Y=A; action A has effect',
                            'observational_joint_probabilities': {'A0Y0': .5, 'A1Y1': .5},
                            'P_Y1_do_A1_world_1': .5, 'P_Y1_do_A1_world_2': 1.0},
            'omitted_iid_lag_clean_mse_lower_bound': .7**2/3,
            'two_distinct_target_query_squared_loss_midpoint': common_prediction**2,
            'scope': 'Exact finite witnesses plus independent omitted-coordinate variance calculation; no intervention learner implemented.'}


def fresh_process_probe(state):
    x = [.4,-.3,.7,.2,-.1,.8,.9,-.5,.6,.1,-.8]
    model = RLS.restore(state)
    pred = model.predict(x)
    model.observe(x, .45)
    return {'prediction': pred, 'state_after_update': model.state()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=SEEDS)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results'/'temporal_relation.json')
    parser.add_argument('--probe-state', type=Path)
    args = parser.parse_args()
    if args.probe_state:
        print(json.dumps(fresh_process_probe(json.loads(args.probe_state.read_text(encoding='utf-8')))))
        return
    runs = {regime: [run_seed(seed, regime) for seed in range(args.seeds)] for regime in REGIMES}
    sums = {}
    metrics = ['lifetime_service_mse','first_half_service_mse','second_half_service_mse',
               'last200_service_mse','iid_heldout_mse','coefficient_l2_error']
    for regime, data in runs.items():
        sums[regime] = {name: {k: summary([run['methods'][name][k] for run in data]) for k in metrics} for name in METHODS}
        sums[regime]['fading_minus_all_lags'] = {k: summary([run['methods']['fading_lags'][k]-run['methods']['all_lags'][k] for run in data]) for k in metrics}
    # Independent restart of learned temporal prediction+update state, one fixture.
    fixture = RLS(LAGS)
    rng = random.Random(7474)
    for _ in range(96):
        h = sample_history(rng)
        fixture.observe(h, truth(h, 'iid_stationary', 0))
    expected = fresh_process_probe(fixture.state())
    with tempfile.TemporaryDirectory(prefix='temporal-restart-') as temporary:
        path = Path(temporary)/'state.json'
        path.write_text(json.dumps(fixture.state()), encoding='utf-8')
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-state', str(path)],
                              capture_output=True, text=True, check=True)
        observed = json.loads(proc.stdout)
    assert expected==observed
    report = {'experiment':'temporal_relation_without_episode_ids',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol':{'seeds':args.seeds,'horizon':HORIZON,'noise_sd':NOISE_SD,'candidate_lags':LAGS,
                          'initial_rule':'.2+1.1*u[t-2]-.7*u[t-5]',
                          'changed_rule':'.2-.4*u[t-2]+1.2*u[t-5]',
                          'change_at_zero_based':CHANGE_AT,'fading_factor':.995,
                          'regularization_initial':1.0,'outputs_numeric':True,'episode_ids':False,
                          'sample_clock_and_order_supplied':True,'new_primitive_or_lag_bound_discovery':False,
                          'observations':'current input then predict current sensor output then observe numeric output; not delayed labels with unknown permutation',
                          'costs':'explicit dominant multiply-add/division counts; no FLOP/walltime equality claim; diagnostic clones excluded',
                          'protocol_selection':'fixed config and seeds before results; no heldout tuning'},
              'summary':sums,'runs':runs,'exact_witnesses':exact_witnesses(),
              'fresh_process_fixture_equal':expected==observed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'means':{r:{m:{k:sums[r][m][k]['mean'] for k in ('lifetime_service_mse','last200_service_mse','iid_heldout_mse')} for m in METHODS} for r in REGIMES}},indent=2))


if __name__=='__main__':
    main()
