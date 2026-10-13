"""Independent scalar intervention semantics, interaction witnesses and persistence."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import interaction_learning as c


def preservation():
    baseline=json.loads((c.RESULTS/'preservation_baseline.json').read_text(encoding='utf-8'))
    changed=[p for p,d in baseline['files'].items() if not (c.ROOT/p).exists() or c.sha(c.ROOT/p)!=d]
    assert not changed,changed
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    assert head==baseline['base_head']
    return dict(files_unchanged=len(baseline['files']),head_unchanged=head)


def scalar(node,x,y):
    if isinstance(node,int):return node%17
    if isinstance(node,str):return x if node=='x' else y
    a,b=scalar(node[1],x,y),scalar(node[2],x,y)
    return (a+b if node[0]=='add' else a-b if node[0]=='sub' else a*b)%17


def scalar_changed(node,mask,x,y,counter):
    # Independent matcher for the particular learned three-hole rule.
    if c.r.branch(node) and node[0]=='mul' and c.r.branch(node[1]) and node[1][0]=='add':
        index=counter[0];counter[0]+=1
        a,b,d=scalar(node[1][1],x,y),scalar(node[1][2],x,y),scalar(node[2],x,y)
        return (a*d+b if mask&(1<<index) else (a+b)*d)%17
    if not c.r.branch(node):return scalar(node,x,y)
    a=scalar_changed(node[1],mask,x,y,counter);b=scalar_changed(node[2],mask,x,y,counter)
    return (a+b if node[0]=='add' else a-b if node[0]=='sub' else a*b)%17


def structural():
    sources=c.source_states();life=c.make_life(0);outputs=0;checks=0
    assert all(source['acquisition_cost']==6158 for source in sources)
    for task in (life['tasks'][0],life['tasks'][4],life['tasks'][12]):
        sites=c.locate(task['initial'],sources[0]['rule'],c.Work());assert len(sites)==8
        xy=np.array([[0,0],[1,2],[3,5],[16,16]])
        table=[]
        for mask in range(256):
            program=c.replace_many(task['initial'],{p:v for i,(p,v) in enumerate(sites) if mask&(1<<i)})
            values=c.r.evaluate(program,xy)
            expected=[scalar_changed(task['initial'],mask,int(x),int(y),[0]) for x,y in xy]
            assert values.tolist()==expected
            table.append(values);outputs+=len(xy)
        coefficients=np.array(table)
        for i in range(8):
            for mask in range(256):
                if mask&(1<<i):coefficients[mask]=(coefficients[mask]-coefficients[mask^(1<<i)])%17
        assert all(np.all(coefficients[mask]==0) for mask in range(256) if mask.bit_count()>task['degree'])
        for mask in range(256):
            reconstructed=sum((coefficients[sub] for sub in range(256) if sub&mask==sub),start=np.zeros(4,dtype=np.int64))%17
            assert np.array_equal(reconstructed,table[mask]);checks+=1
        if task['degree']==3:assert any(np.any(coefficients[mask]) for mask in range(256) if mask.bit_count()==3)
    witness=life['tasks'][4]
    for bit in (0,*[1<<i for i in range(8)]):
        assert all(scalar_changed(witness['initial'],bit,x,y,[0])==0 for x,y in c.GRID.tolist())
    assert any(scalar(witness['target'],x,y)!=0 for x,y in c.GRID.tolist())
    work=c.Work();work.used=c.TASK_LIMIT-1
    try:work.charge(2,'test')
    except c.r.Exhausted:pass
    else:raise AssertionError('budget limit')
    assert work.used==c.TASK_LIMIT-1
    return dict(scalar_intervention_outputs=outputs,full_mobius_reconstructions=checks,
                all_8_disjoint_sites=True,single_edits_masked_pair_witness=True,
                nonzero_third_order_witness=True,source_costs_equal=True,atomic_budget_limit=True)


def restore_life(life):
    result=copy.deepcopy(life)
    for task in result['tasks']:
        task['initial']=c.r.frozen(task['initial']);task['target']=c.r.frozen(task['target'])
    return result


def restore_state(state):
    result=copy.deepcopy(state)
    result['rule']['left']=c.r.frozen(result['rule']['left']);result['rule']['right']=c.r.frozen(result['rule']['right'])
    for entry in result['library']:entry['program']=c.r.frozen(entry['program'])
    return result


def continuation(payload):
    return c.run_life(restore_life(payload['life']),{},payload['method'],start=payload['start'],state=restore_state(payload['state']))


def recorded(data):
    assert data['source_sha256']==c.sha(c.HERE/'interaction_learning.py')
    assert data['prior_source_sha256']==c.sha(c.PREVIOUS/'repair_learning.py')
    assert data['source_result_sha256']==c.sha(c.SOURCE_RESULT)
    assert data['protocol_sha256']==c.sha(c.PROTOCOL)
    assert len(data['runs'])==168
    lives={life['seed']:restore_life(life) for life in data['lives']};accepts=0;false=0;ambiguous=0;probes=0
    for run in data['runs']:
        life=lives[run['seed']];last_work=6158;last_order=1
        assert len(run['records'])==16
        for task,record in zip(life['tasks'],run['records']):
            assert record['work']==sum(record['categories'].values())<=c.TASK_LIMIT
            assert record['cumulative_work']==last_work+record['work'];last_work=record['cumulative_work']
            assert record['order_before']==last_order;last_order=record['order_after']
            assert record['site_count']==8
            assert len(record['probe_masks'])==len(set(record['probe_masks']))
            # Recompute every actual probe's pass/fail independently; no hidden full test used for acceptance.
            for mask in record['probe_masks']:
                values=[scalar_changed(task['initial'],mask,x,y,[0]) for x,y in task['xy']]
                if values==task['labels']:assert mask==record['accepted_mask']
                probes+=1
            if record['accepted_mask'] is not None:
                mask=record['accepted_mask'];assert mask==record['probe_masks'][-1]
                assert [scalar_changed(task['initial'],mask,x,y,[0]) for x,y in task['xy']]==task['labels']
                errors=sum(scalar_changed(task['initial'],mask,x,y,[0])!=scalar(task['target'],x,y) for x,y in c.GRID.tolist())
                assert errors==record['grid_errors'] and record['full_field_correct']==(errors==0)
                assert record['planted_mask_equal']==(mask==task['planted_mask'])
                accepts+=1;false+=errors>0;ambiguous+=errors==0 and mask!=task['planted_mask']
            else:assert record['grid_errors'] is None and not record['full_field_correct']
        assert run['total_work']==last_work
    return dict(trajectories=168,target_attempts=2688,independent_probe_checks=probes,
                full_field_accepted_checks=accepts,false_accepts=false,correct_different_masks=ambiguous,
                causal_acceptance_order=True,cost_categories_match=True)


def restarts(data):
    life=next(life for life in data['lives'] if life['seed']==7);checks=[]
    with tempfile.TemporaryDirectory(prefix='cevahir_interactions_') as folder:
        source,target=Path(folder)/'input.json',Path(folder)/'output.json'
        for run in [item for item in data['runs'] if item['seed']==7]:
            for checkpoint in (4,8):
                payload=dict(life=life,state=run['records'][checkpoint]['state_after'],start=checkpoint+1,method=run['method'])
                source.write_text(c.encode(payload),encoding='utf-8')
                subprocess.run([sys.executable,__file__,'--resume',str(source),str(target)],check=True,capture_output=True,text=True)
                actual=json.loads(target.read_text(encoding='utf-8'))
                assert actual['records']==run['records'][checkpoint+1:]
                assert actual['final_state']==run['final_state'] and actual['total_work']==run['total_work']
                checks.append(dict(method=run['method'],checkpoint=checkpoint,exact_equal=True))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--structural-only',action='store_true')
    p.add_argument('--resume',nargs=2);p.add_argument('--rerun',action='store_true');args=p.parse_args()
    if args.resume:
        source,target=map(Path,args.resume)
        target.write_text(c.encode(continuation(json.loads(source.read_text(encoding='utf-8')))),encoding='utf-8');return
    result=dict(structural=structural(),preservation=preservation())
    if not args.structural_only:
        data=json.loads((c.RESULTS/'interaction_learning.json').read_text(encoding='utf-8'))
        result['recorded']=recorded(data);result['fresh_process_resume']=restarts(data)
        if args.rerun:
            run=subprocess.run([sys.executable,str(c.HERE/'interaction_learning.py')],check=True,capture_output=True,text=True)
            result['full_rerun']=dict(returncode=run.returncode,stdout=run.stdout.strip())
        (c.RESULTS/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
