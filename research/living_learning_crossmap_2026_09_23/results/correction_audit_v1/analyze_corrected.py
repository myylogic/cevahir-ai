import json
from statistics import mean
import corrected as c


def summarize(data):
    groups = []
    for method in c.METHODS:
        rows = [r for run in data['runs'] for r in run['tracks'][method]['records']]
        lives = []
        for run in data['runs']:
            track = run['tracks'][method]
            lives.append(sum(sum(r['engine']['acquisition_cost'].values()) + sum(r['engine']['response']['service_cost'].values())
                             for r in track['records']) + track['setup']['cost'].get('state_writes', 0))
        phases = []
        for phase in range(4):
            ps = [r for r in rows if r['query']['phase'] == phase]
            phases.append({'phase': phase, 'n': len(ps), 'correct': sum(r['correct'] for r in ps),
                           'abstained': sum(r['abstained'] for r in ps), 'confident_errors': sum(r['confident_error'] for r in ps),
                           'candidate_mean': mean(r['engine']['response']['candidate_count'] or 0 for r in ps),
                           'service_work_mean': mean(sum(r['engine']['response']['service_cost'].values()) for r in ps),
                           'brier_mean': None})
        groups.append({'method': method, 'n': len(rows), 'correct': sum(r['correct'] for r in rows),
                       'abstained': sum(r['abstained'] for r in rows), 'confident_errors': sum(r['confident_error'] for r in rows),
                       'life_work_mean': mean(lives), 'life_work_min': min(lives), 'life_work_max': max(lives),
                       'setup_work_mean': mean(sum(run['tracks'][method]['setup']['cost'].values()) for run in data['runs']),
                       'write_bytes_mean': mean(run['tracks'][method]['final']['write_bytes'] for run in data['runs']),
                       'peak_candidates_mean': mean(run['tracks'][method]['final']['peak_candidate_count'] for run in data['runs']),
                       'phases': phases})
    paired = []
    for other in ('separate', 'recompute', 'completion', 'point', 'disabled', 'oracle'):
        differences = []
        for run in data['runs']:
            def work(m):
                tr = run['tracks'][m]
                return tr['setup']['cost']['state_writes'] + sum(sum(r['engine']['acquisition_cost'].values()) + sum(r['engine']['response']['service_cost'].values()) for r in tr['records'])
            differences.append(work('persistent') - work(other))
        paired.append({'left': 'persistent', 'right': other, 'life_work_difference_mean': mean(differences),
                       'left_cheaper_lives': sum(x < 0 for x in differences), 'equal_cost_lives': sum(x == 0 for x in differences)})
    return {'independent_seeds': len(data['runs']), 'queries_per_method': len(data['runs'][0]['tracks']['persistent']['records']),
            'methods': groups, 'paired': paired,
            'candidate_counts_by_phase': {str(p): sorted(set(r['engine']['response']['candidate_count'] or 0 for run in data['runs'] for r in run['tracks']['persistent']['records'] if r['query']['phase'] == p)) for p in range(4)},
            'unit': 'Scalar operation categories are reported separately; life_work_mean sums declared scalar counters and excludes JSON bytes.',
            'scope': 'Corrected post-result audit of finite six-symbol worlds; bridge identity is supplied by observed pairs.'}


if __name__ == '__main__':
    data = json.loads((c.RESULTS / 'crossmap_corrected.json').read_text())
    result = summarize(data)
    (c.RESULTS / 'summary_corrected.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
