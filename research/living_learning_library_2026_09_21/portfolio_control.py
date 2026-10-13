"""Post-hoc fixed half-budget portfolio; primary results remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import library_learning as lib

HERE=Path(__file__).resolve().parent


def run():
    primary=json.loads((HERE/'results/library_learning.json').read_text(encoding='utf-8'))
    assert primary['source_sha256']==hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest()
    rows=[]
    for life in primary['lives']:
        world=life['world'];seed=world['seed'];state=life['states']['12']
        tokens=lib.token_policy(state,'learned_library',seed,world['home_motifs'])
        for budget in lib.BUDGETS:
            for task in world['probes']:
                primitive=lib.solve_task(task,lib.PRIMITIVES,budget//2)
                learned=lib.solve_task(task,tokens,budget//2)
                winner=primitive if primitive['status']=='accepted' else learned
                pe=lib.evaluate(primitive,task);le=lib.evaluate(learned,task);we=lib.evaluate(winner,task)
                assert not pe['exact'] or we['exact']
                spent=primitive['charged_vector_steps']+learned['charged_vector_steps']
                assert spent<=budget
                rows.append(dict(seed=seed,budget=budget,task=task['name'],regime=task['regime'],
                                 primitive_half=dict(**primitive,**pe),learned_half=dict(**learned,**le),
                                 selected_branch='primitive' if primitive['status']=='accepted' else 'learned',
                                 selected=dict(**winner,**we),total_charged_vector_steps=spent))
    aggregate=[]
    for budget in lib.BUDGETS:
        for regime in ('related','shifted'):
            subset=[r for r in rows if (r['budget'],r['regime'])==(budget,regime)]
            aggregate.append(dict(budget=budget,regime=regime,conditions=len(subset),
                                  primitive_half_exact=sum(r['primitive_half']['exact'] for r in subset),
                                  learned_half_exact=sum(r['learned_half']['exact'] for r in subset),
                                  portfolio_exact=sum(r['selected']['exact'] for r in subset),
                                  false_accepts=sum(r['selected']['status']=='accepted' and not r['selected']['exact'] for r in subset),
                                  mean_total_charged_vector_steps=statistics.fmean(r['total_charged_vector_steps'] for r in subset)))
    return dict(status='posthoc_same_world_mechanism_control',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                primary_source_sha256=primary['source_sha256'],runs=rows,aggregate=aggregate,
                preserved_primitive_half_successes=True,
                scope='Two fixed half budgets, primitive acceptance priority; no guarantee relative to the full-budget baseline.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    result=run();path=HERE/'results/portfolio_control.json'
    if args.write:path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:assert result==json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps(dict(status='passed',mode='record' if args.write else 'check',conditions=len(result['runs']),aggregate=result['aggregate'])))


if __name__=='__main__':main()
