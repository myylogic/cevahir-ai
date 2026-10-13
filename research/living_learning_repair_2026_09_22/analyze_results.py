"""Paired task acquisition, negative transfer and whole-life cost summaries."""
import json
import math
import repair_learning as r


def stats(values):return dict(mean=math.fsum(values)/len(values),min=min(values),max=max(values))
def summarize(data):
    groups=[];pairs=[]
    for regime in r.REGIMES:
        for method in r.METHODS:
            runs=[run for run in data['runs'] if run['regime']==regime and run['method']==method]
            records=[rec for run in runs for rec in run['records']]
            points=[]
            for index in range(8):
                rows=[run['records'][index] for run in runs]
                points.append(dict(task_index=index,correct=sum(row['full_field_correct'] for row in rows),
                    accepted=sum(row['accepted'] is not None for row in rows),
                    false_accepted=sum(row['accepted'] is not None and not row['full_field_correct'] for row in rows),
                    work=stats([row['work'] for row in rows]),cumulative_work=stats([row['cumulative_work'] for row in rows])))
            groups.append(dict(regime=regime,method=method,n_lives=24,attempted=len(records),
                correct=sum(rec['full_field_correct'] for rec in records),
                accepted=sum(rec['accepted'] is not None for rec in records),
                false_accepted=sum(rec['accepted'] is not None and not rec['full_field_correct'] for rec in records),
                budget_stopped=sum(rec['reason']=='budget' for rec in records),
                correct_per_life=stats([sum(rec['full_field_correct'] for rec in run['records']) for run in runs]),
                total_work=stats([run['total_work'] for run in runs]),
                candidates=stats([sum(rec['candidates'] for rec in run['records']) for run in runs]),
                rejected_rule_candidates=sum(rec['rejected_rule_candidates'] for rec in records),
                rule_proposals=sum(rec['rule_proposals'] for rec in records),
                final_json_bytes=stats([run['final_json_bytes'] for run in runs]),points=points))
        lookup={(run['method'],run['seed']):run for run in data['runs'] if run['regime']==regime}
        for left,right in [('learned_rules','rules_disabled_paid'),('learned_rules','bulk_ranked'),('bulk_ranked','local_ranked')]:
            deltas=[]
            for seed in range(24):
                a,b=lookup[left,seed],lookup[right,seed]
                deltas.append(dict(seed=seed,correct=sum(rec['full_field_correct'] for rec in a['records'])-sum(rec['full_field_correct'] for rec in b['records']),
                                   work=a['total_work']-b['total_work']))
            pairs.append(dict(regime=regime,left=left,right=right,per_life=deltas,
                              correct_difference=stats([d['correct'] for d in deltas]),
                              work_difference=stats([d['work'] for d in deltas]),
                              left_more_correct=sum(d['correct']>0 for d in deltas),
                              tied_correct=sum(d['correct']==0 for d in deltas),
                              right_more_correct=sum(d['correct']<0 for d in deltas)))
    acquired=data['acquisitions']
    source=dict(base_cost=stats([a['base_cost'] for a in acquired]),
                rule_acquisition_cost=stats([a['rule_acquisition_cost'] for a in acquired]),
                rule_count=stats([len(a['state']['rules']) for a in acquired]),
                priority_count=stats([len(a['state']['priorities']) for a in acquired]),
                source_accepts=sum(rec['accepted'] is not None for a in acquired for rec in a['source_records']),
                unique_rules=sorted({r.encode(rule['left'])+' -> '+r.encode(rule['right']) for a in acquired for rule in a['state']['rules']}))
    return dict(runs=len(data['runs']),attempts=2880,source=source,groups=groups,paired=pairs,
                interpretation='24 paired lives, finite field semantics, declared virtual work. Descriptive transfer experiment; not new universal learning theory.')


def main():
    data=json.loads((r.RESULTS/'repair_learning.json').read_text(encoding='utf-8'));summary=summarize(data)
    (r.RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(r.encode(summary['source']))
    for group in summary['groups']:
        print(r.encode({key:group[key] for key in ('regime','method','correct','accepted','false_accepted','budget_stopped','total_work','rule_proposals','rejected_rule_candidates')}))
    for pair in summary['paired']:
        print(r.encode({key:pair[key] for key in ('regime','left','right','correct_difference','left_more_correct','tied_correct','right_more_correct','work_difference')}))


if __name__=='__main__':main()
