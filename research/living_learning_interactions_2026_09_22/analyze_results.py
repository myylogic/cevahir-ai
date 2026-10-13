"""Descriptive paired-life summaries; no independent-task significance claims."""
import json
import math
import interaction_learning as c

PHASES={'initial_additive':range(4),'pairs':range(4,8),'return_additive':range(8,12),'triples':range(12,16)}

def stats(values):
    return dict(mean=math.fsum(values)/len(values),min=min(values),max=max(values))

def records_summary(rows):
    return dict(attempts=len(rows),correct=sum(row['full_field_correct'] for row in rows),
                accepted=sum(row['accepted_mask'] is not None for row in rows),
                false_accepts=sum(row['accepted_mask'] is not None and not row['full_field_correct'] for row in rows),
                different_correct_masks=sum(row['full_field_correct'] and not row['planted_mask_equal'] for row in rows),
                budget_stops=sum(row['reason']=='budget' for row in rows),expansions=sum(row['expanded'] for row in rows),
                model_mismatches=sum(row['model_prediction_mismatches'] for row in rows),
                work=stats([row['work'] for row in rows]),
                probes=stats([len(row['probe_masks']) for row in rows]),
                model_masks=stats([row['model_masks'] for row in rows]))

def summarize(data):
    groups=[];paired=[]
    lookup={(run['seed'],run['method']):run for run in data['runs']}
    for method in c.METHODS:
        runs=[lookup[seed,method] for seed in c.SEEDS]
        rows=[row for run in runs for row in run['records']]
        phases={name:records_summary([run['records'][i] for run in runs for i in indices]) for name,indices in PHASES.items()}
        points=[dict(index=i,**records_summary([run['records'][i] for run in runs])) for i in range(16)]
        groups.append(dict(method=method,**records_summary(rows),phases=phases,points=points,
                           life_total_work=stats([run['total_work'] for run in runs]),
                           final_json_bytes=stats([run['final_json_bytes'] for run in runs]),
                           final_orders={str(order):sum(run['final_state']['order']==order for run in runs) for order in (1,2)}))
    for left,right in [('adaptive_persistent','adaptive_reset'),('second_order','exact_subset'),
                       ('adaptive_reset','first_order'),('adaptive_persistent','second_order')]:
        for phase,indices in [('whole_life',range(16)),*PHASES.items()]:
            deltas=[]
            for seed in c.SEEDS:
                a,b=lookup[seed,left],lookup[seed,right]
                delta=dict(seed=seed,
                           correct=sum(a['records'][i]['full_field_correct']-b['records'][i]['full_field_correct'] for i in indices),
                           work=sum(a['records'][i]['work']-b['records'][i]['work'] for i in indices))
                deltas.append(delta)
            paired.append(dict(left=left,right=right,phase=phase,per_life=deltas,
                               correct_difference=stats([d['correct'] for d in deltas]),
                               work_difference=stats([d['work'] for d in deltas]),
                               left_cheaper=sum(d['work']<0 for d in deltas),equal_work=sum(d['work']==0 for d in deltas),
                               right_cheaper=sum(d['work']>0 for d in deltas)))
    all_rows=[row for run in data['runs'] for row in run['records']]
    return dict(paired_lives=24,trajectories=len(data['runs']),source_cost_per_life=6158,
                totals=records_summary(all_rows),groups=groups,paired=paired,
                scope='24 paired lives in F17, virtual work only; phases are not independent replications.')

def main():
    data=json.loads((c.RESULTS/'interaction_learning.json').read_text(encoding='utf-8'))
    summary=summarize(data)
    (c.RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(c.encode(summary['totals']))
    for group in summary['groups']:
        print(c.encode({k:group[k] for k in ('method','correct','false_accepts','budget_stops','expansions','model_mismatches','life_total_work')}))
        print(c.encode({name:{k:value[k] for k in ('correct','work','probes')} for name,value in group['phases'].items()}))
    for pair in summary['paired']:
        if pair['left']=='adaptive_persistent' and pair['right']=='adaptive_reset':
            print(c.encode({k:pair[k] for k in ('left','right','phase','correct_difference','work_difference','left_cheaper','right_cheaper')}))

if __name__=='__main__':main()
