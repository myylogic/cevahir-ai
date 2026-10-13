"""Separate uncertainty, migration error, hypothesis error and measured cost."""
import json
import math
import live_revision as c

def stats(values):return dict(mean=math.fsum(values)/len(values),min=min(values),max=max(values))

def point_stats(points):
    return dict(n=len(points),abstained=sum(p['prediction'] is None for p in points),
                correct=sum(p['prediction']==p['truth'] for p in points),
                wrong=sum(p['prediction'] is not None and p['prediction']!=p['truth'] for p in points),
                replay_not_contained=sum(not p['replay_contained'] for p in points),
                confident_replay_disagreement=sum(p['prediction'] is not None and p['prediction']!=p['replay_output'] for p in points),
                model_errors=sum(p['replay_output']!=p['truth'] for p in points),
                wrong_when_replay_right=sum(p['prediction'] is not None and p['prediction']!=p['truth'] and p['replay_output']==p['truth'] for p in points),
                right_when_replay_wrong=sum(p['prediction']==p['truth'] and p['replay_output']!=p['truth'] for p in points),
                singleton_states=sum(p['size']==1 for p in points),
                ambiguous_states_certain_output=sum(p['size']>1 and p['prediction'] is not None for p in points),
                max_states=max([p['size'] for p in points],default=0))

def group(runs):
    rows=[r for run in runs for r in run['records']];points=[p for row in rows for p in row['points']]
    changed=[row for row in rows if row['install']['changed']]
    final=[run['records'][-1] for run in runs]
    return dict(paths=len(runs),points=point_stats(points),actual_revisions=len(changed),
                changed_t0=point_stats([row['points'][0] for row in changed]),
                changed_first5=point_stats([p for row in changed for p in row['points'][:5]]),
                exact_model_points=point_stats([p for row in rows if row['model_exact'] for p in row['points']]),
                t_curve=[dict(t=t,**point_stats([row['points'][t] for row in changed])) for t in range(17)],
                added_work=stats([row['work'] for row in final]),
                migration_work=stats([sum(row['install']['work'] for row in run['records']) for run in runs]),
                final_trace_bits=stats([len(run['final_state']['suffix'])+len(run['final_state']['history']) for run in runs]),
                peak_relation=stats([run['final_state']['peak_relation'] for run in runs]),
                state_json_bytes=stats([row['state_json_bytes'] for row in final]),
                calibration_work=stats([row['calibration']['work'] for row in final]),
                calibration_environment=stats([row['calibration']['environment_units'] for row in final]),
                calibration_json_bytes=stats([row['calibration']['json_bytes'] for row in final]),
                combined_internal_work=stats([row['work']+row['calibration']['work'] for row in final]))

def summarize(data):
    groups=[];paired=[]
    for family in c.prior.FAMILIES:
        for stride in c.STRIDES:
            for method in c.METHODS:
                runs=[r for r in data['runs'] if (r['family'],r['stride'],r['method'])==(family,stride,method)]
                groups.append(dict(family=family,stride=stride,method=method,**group(runs)))
    methods=[dict(method=method,**group([r for r in data['runs'] if r['method']==method])) for method in c.METHODS]
    lookup={(r['seed'],r['family'],r['stride'],r['method']):r for r in data['runs']}
    for left,right in [('model_relation_suffix8','parity_suffix8'),('model_relation','all_states'),('model_relation_suffix8','full_archive')]:
        for stride in c.STRIDES:
            deltas=[]
            for source in data['sources']:
                a=lookup[source['seed'],source['family'],stride,left];b=lookup[source['seed'],source['family'],stride,right]
                pa=[p for row in a['records'] for p in row['points']];pb=[p for row in b['records'] for p in row['points']]
                deltas.append(dict(seed=source['seed'],family=source['family'],work=a['records'][-1]['work']-b['records'][-1]['work'],
                                   abstained=sum(p['prediction'] is None for p in pa)-sum(p['prediction'] is None for p in pb),
                                   identical_sets=all(x['possible']==y['possible'] for x,y in zip(pa,pb)),
                                   identical_predictions=all(x['prediction']==y['prediction'] for x,y in zip(pa,pb))))
            paired.append(dict(left=left,right=right,stride=stride,per_life=deltas,
                               work_difference=stats([d['work'] for d in deltas]),
                               abstention_difference=sum(d['abstained'] for d in deltas),
                               identical_set_lives=sum(d['identical_sets'] for d in deltas),
                               identical_prediction_lives=sum(d['identical_predictions'] for d in deltas),
                               left_cheaper=sum(d['work']<0 for d in deltas)))
    return dict(paths=len(data['runs']),worlds=len(data['sources']),paired_seeds=24,groups=groups,methods=methods,paired=paired)

def main():
    data=json.loads((c.RESULTS/'live_revision.json').read_text(encoding='utf-8'));summary=summarize(data)
    (c.RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for group_ in summary['methods']:print(c.encode({k:group_[k] for k in ('method','points','changed_t0','added_work','migration_work','calibration_work')}))
    for pair in summary['paired']:print(c.encode({k:v for k,v in pair.items() if k!='per_life'}))

if __name__=='__main__':main()
