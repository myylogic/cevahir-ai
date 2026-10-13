import json
from statistics import mean
import crossmap as c


def summarize(data):
    groups=[]
    for method in c.METHODS:
        rows=[row for run in data['runs'] for row in run['tracks'][method]['records']]
        life=[]
        for run in data['runs']:
            rec=run['tracks'][method]['records']
            life.append(sum(sum(x['engine']['costs'].values()) for x in rec))
        phases=[]
        for phase in range(4):
            ps=[r for r in rows if r['phase']==phase]
            phases.append({'phase':phase,'n':len(ps),'correct':sum(r['engine']['correct'] for r in ps),
                           'abstained':sum(r['engine']['abstained'] for r in ps),
                           'confident_errors':sum(r['engine']['confident'] and not r['engine']['correct'] for r in ps),
                           'candidate_mean':mean(r['engine']['candidate_count'] for r in ps),
                           'service_mean':mean(sum(r['engine']['costs'].values()) for r in ps)})
        groups.append({'method':method,'n':len(rows),'correct':sum(r['engine']['correct'] for r in rows),
                       'abstained':sum(r['engine']['abstained'] for r in rows),
                       'confident_errors':sum(r['engine']['confident'] and not r['engine']['correct'] for r in rows),
                       'life_work_mean':mean(life),'life_work_min':min(life),'life_work_max':max(life),
                       'max_state_bytes_mean':mean(max(r['engine']['state_bytes'] for r in run['tracks'][method]['records']) for run in data['runs']),
                       'phases':phases})
    paired=[]
    for other in ('separate','recompute','oracle','wrong','disabled','tree'):
        diff=[]
        for run in data['runs']:
            left=run['tracks']['persistent']['records']; right=run['tracks'][other]['records']
            diff.append(sum(sum(r['engine']['costs'].values()) for r in left)-sum(sum(r['engine']['costs'].values()) for r in right))
        paired.append({'left':'persistent','right':other,'life_work_difference_mean':mean(diff),
                       'left_cheaper_lives':sum(x<0 for x in diff),'equal_cost_lives':sum(x==0 for x in diff)})
    return {'independent_seeds':len(data['runs']),'queries_per_method':len(data['runs'][0]['tracks']['persistent']['records']),
            'methods':groups,'paired':paired,
            'candidate_counts_by_phase':{str(p): sorted(set(r['engine']['candidate_count'] for run in data['runs'] for r in run['tracks']['persistent']['records'] if r['phase']==p)) for p in range(4)},
            'unit':'Declared virtual operations: parsing, filtering, candidate scans, query steps, writes and tree-copy work; not CPU time.',
            'scope':'Finite six-symbol relational worlds with hidden symbol permutations and finite bridge evidence.'}

if __name__=='__main__':
    data=json.loads((c.RESULTS/'crossmap.json').read_text())
    result=summarize(data); (c.RESULTS/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
