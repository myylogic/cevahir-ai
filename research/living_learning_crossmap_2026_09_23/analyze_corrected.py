"""Scalar work, byte materialization and accuracy are distinct measures."""
import json
from fractions import Fraction
from statistics import mean
import corrected as c


def life_work(track):
    totals = dict(track['setup']['cost'])
    nbytes = track['setup']['write_bytes']
    for row in track['records']:
        engine = row['engine']
        for key in c.CATEGORIES:
            totals[key] += engine['acquisition_cost'][key] + engine['response']['service_cost'][key]
        nbytes += engine['acquisition_write_bytes'] + engine['response']['service_write_bytes']
    assert totals == track['final']['cost']
    assert nbytes == track['final']['write_bytes']
    return sum(totals.values())


def metric(rows, name):
    values = [Fraction(*r['scores'][name]) for r in rows if r['scores'][name] is not None]
    return float(sum(values) / len(values)) if values else None


def accuracy(rows):
    return dict(n=len(rows), correct=sum(r['correct'] for r in rows),
                distribution_n=sum(r['scores']['brier'] is not None for r in rows),
                abstained=sum(r['abstained'] for r in rows),
                confident_errors=sum(r['confident_error'] for r in rows),
                brier_mean=metric(rows, 'brier'),
                expected_argmax_accuracy=metric(rows, 'expected_argmax_accuracy'))


def summarize(data):
    groups = []
    for method in c.METHODS:
        tracks = [run['tracks'][method] for run in data['runs']]
        rows = [r for tr in tracks for r in tr['records']]
        lives = [life_work(tr) for tr in tracks]
        phases = []
        for phase in range(4):
            ps = [r for r in rows if r['query']['phase'] == phase]
            candidate = [r['engine']['response']['candidate_count'] for r in ps]
            phases.append(dict(phase=phase, **accuracy(ps),
                               candidate_mean=mean(candidate) if all(x is not None for x in candidate) else None,
                               service_work_mean=mean(sum(r['engine']['response']['service_cost'].values()) for r in ps)))
        groups.append(dict(method=method, **accuracy(rows),
            life_work_mean=mean(lives), life_work_min=min(lives), life_work_max=max(lives),
            setup_work_mean=mean(sum(tr['setup']['cost'].values()) for tr in tracks),
            work_by_category_mean={k: mean(tr['final']['cost'][k] for tr in tracks) for k in c.CATEGORIES},
            service_work_mean=mean(sum(r['engine']['response']['service_cost'].values()) for r in rows),
            materialized_bytes_mean=mean(tr['final']['write_bytes'] for tr in tracks),
            transient_materialized_bytes_mean=mean(tr['final']['transient_write_bytes'] for tr in tracks),
            largest_candidate_list_mean=mean(tr['final']['peak_candidate_count'] for tr in tracks),
            max_persistent_state_bytes_mean=mean(max(r['engine']['response']['state_bytes'] for r in tr['records']) for tr in tracks),
            weighted_work_plus_materialized_bytes={str(weight): mean(life_work(tr) + weight * tr['final']['write_bytes'] for tr in tracks) for weight in (0, 1, 4)},
            new_queries=accuracy([r for r in rows if r['new_query']]),
            queried_pair_unobserved=accuracy([r for r in rows if not r['queried_pair_observed']]), phases=phases))
    paired = []
    for other in ('separate', 'recompute', 'completion', 'point', 'disabled', 'oracle'):
        diff = [life_work(run['tracks']['persistent']) - life_work(run['tracks'][other]) for run in data['runs']]
        eq = 0
        for run in data['runs']:
            for left, right in zip(run['tracks']['persistent']['records'], run['tracks'][other]['records']):
                aa, bb = left['engine']['response']['counts'], right['engine']['response']['counts']
                if aa is not None and bb is not None and [Fraction(v, sum(aa)) for v in aa] == [Fraction(v, sum(bb)) for v in bb]: eq += 1
        paired.append(dict(left='persistent', right=other, life_work_difference_mean=mean(diff),
                           left_cheaper_lives=sum(v < 0 for v in diff), equal_cost_lives=sum(v == 0 for v in diff),
                           equal_probability_queries=eq))
    return dict(independent_seeds=len(data['runs']), queries_per_life=24, queries_per_method=576,
                methods=groups, paired=paired,
                candidate_counts_by_phase={str(p): sorted({r['engine']['response']['candidate_count'] for run in data['runs'] for r in run['tracks']['persistent']['records'] if r['query']['phase'] == p}) for p in range(4)},
                interpretation='Uniform mixture over bridge-consistent bijections, not the full planted-generator posterior.',
                unit='Declared scalar work includes transient materializations; byte sensitivity is separate, not hardware time.')


if __name__ == '__main__':
    result = summarize(json.loads((c.RESULTS / 'crossmap_corrected.json').read_text(encoding='utf-8')))
    (c.RESULTS / 'summary_corrected.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
