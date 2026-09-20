"""Independent, exact second-moment check of frozen update matrices.

No learner implementation is imported. This conditions on each source-derived
P and integrates over fresh downstream streams, including observation noise.
The derivation assumes iid uniform unit-circle inputs and the stated random
walk model; it does not describe the adaptive meta-learners.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ['initial_fixed', 'learned_frozen', 'diagonal_only',
           'isotropic_same_trace', 'rotated_90',
           'train_selected_scalar', 'train_selected_matrix']


def mm(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def tr(a):
    return a[0][0]+a[1][1]


def packed(p):
    return [[p[0], p[2]], [p[2], p[1]]]


def covariance(angle, fast, slow):
    c, s = math.cos(angle), math.sin(angle)
    return [[c*c*fast+s*s*slow, c*s*(fast-slow)],
            [c*s*(fast-slow), s*s*fast+c*c*slow]]


def post_moment(c, p, noise_variance):
    # E[xx' C xx'] = (tr(C) I + 2C)/8 in dimension two.
    pc, cp = mm(p, c), mm(c, p)
    fourth = [[(tr(c)*int(i == j)+2*c[i][j])/8 for j in range(2)] for i in range(2)]
    pfp, pp = mm(mm(p, fourth), p), mm(p, p)
    return [[c[i][j]-.5*(pc[i][j]+cp[i][j])+pfp[i][j]+noise_variance*.5*pp[i][j]
             for j in range(2)] for i in range(2)]


def quadrature_post(c, p, noise_variance):
    # An independent calculation of E[A C A' + r Pxx'P]. Sixteen
    # equispaced angles integrate these degree-four trigonometric terms.
    result = [[0., 0.], [0., 0.]]
    for k in range(16):
        x = [math.cos(2*math.pi*k/16), math.sin(2*math.pi*k/16)]
        px = [sum(p[i][j]*x[j] for j in range(2)) for i in range(2)]
        a = [[int(i == j)-px[i]*x[j] for j in range(2)] for i in range(2)]
        at = [list(row) for row in zip(*a)]
        aca = mm(mm(a, c), at)
        for i in range(2):
            for j in range(2):
                result[i][j] += (aca[i][j]+noise_variance*px[i]*px[j])/16
    return result


def checks():
    error = 0.
    for angle in (.13, .77, 1.83):
        c = covariance(angle, .8, .04)
        p = covariance(angle+.43, .41, .008)
        direct, independent = post_moment(c, p, .15**2), quadrature_post(c, p, .15**2)
        error = max(error, *(abs(direct[i][j]-independent[i][j]) for i in range(2) for j in range(2)))
    assert error < 1e-12
    # Scalar P and isotropic C reduce to c'=(1-a+3a^2/8+a^2/8)c + r*a^2/2.
    a, v, r = .23, .71, .0225
    post = post_moment([[v, 0.], [0., v]], [[a, 0.], [0., a]], r)
    scalar = (1-a+.5*a*a)*v+.5*r*a*a
    assert abs(post[0][0]-scalar) < 1e-14
    return {'unit_circle_quadrature_max_error': error, 'scalar_reduction_equal': True}


def finite_risk(p, q, initial_variance, noise_variance, length):
    # stream() adds its first random-walk increment before the first label.
    c = [[q[i][j]+initial_variance*int(i == j) for j in range(2)] for i in range(2)]
    total = [0., 0., 0., 0.]
    for t in range(length):
        risk = .5*tr(c)
        total[0] += risk/length
        if t < 64:
            total[1] += risk/64
        elif t < 512:
            total[2] += risk/448
        else:
            total[3] += risk/(length-512)
        post = post_moment(c, p, noise_variance)
        c = [[post[i][j]+q[i][j] for j in range(2)] for i in range(2)]
    return dict(zip(['service_mse', 'first64_mse', 'samples65_512_mse',
                     'samples513_end_mse', 'final_uniform_query_mse'], total+[.5*tr(post)]))


def matrices(source, initial):
    a, d, b = source['learned_p']
    return {'initial_fixed': packed([initial, initial, 0.]),
            'learned_frozen': packed([a, d, b]),
            'diagonal_only': packed([a, d, 0.]),
            'isotropic_same_trace': packed([(a+d)/2, (a+d)/2, 0.]),
            'rotated_90': packed([d, a, -b]),
            'train_selected_scalar': packed(source['train_selected_scalar']['p']),
            'train_selected_matrix': packed(source['train_selected_matrix']['p'])}


def summary(values):
    m = statistics.fmean(values)
    se = statistics.stdev(values)/math.sqrt(len(values))
    return {'mean': m, 'normal_ci95': [m-1.96*se, m+1.96*se]}


def analyze(source_path):
    data = json.loads(source_path.read_text(encoding='utf-8'))
    protocol = data['protocol']
    output = {}
    for regime in data['runs']:
        exact = {}
        for source in data['sources']:
            angle = source['world_fast_direction']+(math.pi/2 if regime == 'reversed_geometry' else 0.)
            q = covariance(angle, protocol['fast_drift_sd']**2,
                           (protocol['fast_drift_sd'] if regime == 'isotropic_change' else protocol['slow_drift_sd'])**2)
            initial_var = protocol['cold_restart_initial_target_sd']**2 if regime == 'cold_restart' else 0.
            exact[str(source['seed'])] = {
                name: finite_risk(p, q, initial_var, protocol['label_noise_sd']**2,
                                  protocol['downstream_samples_per_task'])
                for name, p in matrices(source, protocol['initial_gain']).items()}
        records = {}
        for method in METHODS:
            seed_gap = [data['summary'][regime]['seed_means'][seed][method]['service_mse']-
                        exact[seed][method]['service_mse'] for seed in exact]
            records[method] = {
                'expected_service_mse': statistics.fmean(e[method]['service_mse'] for e in exact.values()),
                'observed_service_mse': data['summary'][regime]['methods'][method]['service_mse']['mean'],
                'observed_minus_conditional_expected_by_source': summary(seed_gap)}
        output[regime] = {'methods': records, 'per_source_conditional_expectations': exact}
    return {'experiment': 'independent_finite_horizon_second_moment_analysis',
            'input_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checks': checks(), 'regimes': output,
            'scope': 'Exact conditional expectations for frozen P only, under specified data-generating distribution. No fit or tuning to downstream scores.',
            'uncertainty': 'Monte Carlo comparison intervals average four tasks within each source. Pointwise descriptive normal intervals; no simultaneous coverage or acceptance test.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=HERE/'results'/'learned_update.json')
    parser.add_argument('--output', type=Path, default=HERE/'results'/'analytic_risk.json')
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks': result['checks'], 'regimes': {
        r: v['methods'] for r, v in result['regimes'].items()}}, indent=2))


if __name__ == '__main__':
    main()
