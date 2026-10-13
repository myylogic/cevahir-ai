"""Paired-world summaries of learned distinctions, evidence and resource costs."""
import json
import math
import distinction_learning as c

def stats(values):return dict(mean=math.fsum(values)/len(values),min=min(values),max=max(values))

def summarize(data):
    groups=[];pairs=[]
    for family in c.FAMILIES:
        for method in c.METHODS:
            runs=[run for run in data['runs'] if (run['family'],run['method'])==(family,method)]
            points=[]
            for block in range(16):
                records=[run['records'][block] for run in runs]
                points.append(dict(block=block,exact=sum(row['evaluation']['exact']['equal'] for row in records),
                                   long_errors=sum(row['evaluation']['long_errors'] for row in records),
                                   work=stats([row['work'] for row in records]),
                                   environment_units=stats([row['environment_units'] for row in records]),
                                   states=stats([row['evaluation']['states'] for row in records])))
            final=[run['records'][-1] for run in runs]
            groups.append(dict(family=family,method=method,lives=24,
                               final_exact=sum(row['evaluation']['exact']['equal'] for row in final),
                               final_long_errors=sum(row['evaluation']['long_errors'] for row in final),
                               prequential_errors=sum(row['prequential_errors'] for run in runs for row in run['records']),
                               final_known_errors=sum(row['evaluation']['known_errors'] for row in final),
                               frozen=sum(run['final_state']['frozen'] for run in runs),
                               frozen_reasons={reason:sum(run['final_state']['reason']==reason for run in runs) for reason in ('internal','environment')},
                               learning_work=stats([row['work'] for row in final]),
                               operating_work=stats([row['operating_work'] for row in final]),
                               environment_units=stats([row['environment_units'] for row in final]),
                               active_queries=stats([row['active_queries'] for row in final]),
                               states=stats([row['evaluation']['states'] for row in final]),
                               tests=stats([row['evaluation']['tests'] for row in final]),
                               max_test_length=stats([row['evaluation']['max_test_length'] for row in final]),
                               split_count=stats([row['evaluation']['splits'] for row in final]),
                               json_bytes=stats([row['state_json_bytes'] for row in final]),points=points))
        lookup={(run['seed'],run['method']):run for run in data['runs'] if run['family']==family}
        for left,right in [('adaptive','rebuild'),('adaptive','exhaustive_tests')]:
            deltas=[]
            for seed in c.SEEDS:
                a=lookup[seed,left]['records'][-1];b=lookup[seed,right]['records'][-1]
                deltas.append(dict(seed=seed,work=a['work']-b['work'],environment=a['environment_units']-b['environment_units'],
                                   exact=int(a['evaluation']['exact']['equal'])-int(b['evaluation']['exact']['equal']),
                                   same_model=a['state_after']['learner']['model']==b['state_after']['learner']['model'],
                                   same_evidence=a['state_after']['environment']==b['state_after']['environment']))
            pairs.append(dict(family=family,left=left,right=right,per_life=deltas,
                               work_difference=stats([row['work'] for row in deltas]),
                               environment_difference=stats([row['environment'] for row in deltas]),
                               exact_difference=sum(row['exact'] for row in deltas),
                               same_model=sum(row['same_model'] for row in deltas),same_evidence=sum(row['same_evidence'] for row in deltas),
                               left_cheaper=sum(row['work']<0 for row in deltas)))
    return dict(paired_seeds=24,worlds=72,paths=288,blocks=4608,passive_predictions=18432,groups=groups,paired=pairs)

def main():
    data=json.loads((c.RESULTS/'distinction_learning.json').read_text(encoding='utf-8'));result=summarize(data)
    (c.RESULTS/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for group in result['groups']:print(c.encode({k:value for k,value in group.items() if k!='points'}))
    for pair in result['paired']:print(c.encode({k:value for k,value in pair.items() if k!='per_life'}))

if __name__=='__main__':main()
