"""Descriptive paired analysis; no independent-world inflation or significance claim."""
import json
from statistics import mean
import deduction as d


def summarize(data):
    groups = []
    for method in d.METHODS:
        lives = [run['tracks'][method]['records'] for run in data['runs']]
        points = [p for life in lives for p in life]
        life_work = [sum(sum(p['engine']['costs'].values()) for p in life) for life in lives]
        categories = sorted({k for p in points for k in p['engine']['costs']})
        phases = []
        for phase in range(4):
            ps = [p for p in points if p['phase'] == phase]
            phases.append({'phase': phase, 'correct': sum(p['correct'] for p in ps), 'n': len(ps),
                           'service_mean': mean(p['engine']['service_work'] for p in ps),
                           'whole_work_mean_per_life': sum(sum(p['engine']['costs'].values()) for p in ps) / len(lives)})
        groups.append({'method': method, 'n': len(points), 'correct': sum(p['correct'] for p in points),
                       'novel_n': sum(p['novel_query'] for p in points),
                       'novel_correct': sum(p['novel_query'] and p['correct'] for p in points),
                       'new_lemmas': sum(p['engine'].get('new') is not None for p in points),
                       'service_mean': mean(p['engine']['service_work'] for p in points),
                       'life_work_mean': mean(life_work), 'life_work_min': min(life_work), 'life_work_max': max(life_work),
                       'mean_max_state_bytes': mean(max(p['engine']['state_bytes'] for p in life) for life in lives),
                       'categories_mean_per_life': {k: sum(p['engine']['costs'].get(k, 0) for p in points) / len(lives) for k in categories},
                       'service_thresholds': {str(b): sum(p['correct'] and p['engine']['service_work'] <= b for p in points) for b in (500, 2000, 10000)},
                       'phases': phases})
    paired = []
    for other in ('fresh', 'erase', 'blind', 'replay', 'full_table', 'full_table_cached'):
        differences = []
        service_improved = novel_improved = service_worse = wrong_difference = 0
        for run in data['runs']:
            left = run['tracks']['scoped']['records']
            right = run['tracks'][other]['records']
            differences.append(sum(sum(p['engine']['costs'].values()) for p in left) - sum(sum(p['engine']['costs'].values()) for p in right))
            for a, b in zip(left, right):
                gain = a['engine']['service_work'] < b['engine']['service_work']
                service_improved += gain
                novel_improved += gain and a['novel_query']
                service_worse += a['engine']['service_work'] > b['engine']['service_work']
                wrong_difference += (not a['correct']) - (not b['correct'])
        paired.append({'left': 'scoped', 'right': other, 'life_work_difference_mean': mean(differences),
                       'left_cheaper_lives': sum(x < 0 for x in differences), 'equal_cost_lives': sum(x == 0 for x in differences),
                       'service_cheaper_points': service_improved, 'novel_service_cheaper_points': novel_improved,
                       'service_more_expensive_points': service_worse, 'wrong_difference': wrong_difference})
    return {'independent_seeds': len(data['runs']), 'methods': groups, 'paired': paired,
            'inconsistent_base_versions': sum(not rows for run in data['runs'] for rows in run['truth']),
            'unit': 'Declared virtual operations, including serialized certificate/history byte writes; not CPU time.',
            'budget_scope': 'Retrospective service thresholds exclude acquisition/compilation/replay. They are not complete budgeted-learning success counts.'}


if __name__ == '__main__':
    result = summarize(json.loads((d.RESULTS / 'deduction.json').read_text(encoding='utf-8')))
    (d.RESULTS / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
