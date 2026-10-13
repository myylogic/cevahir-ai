"""Scalar semantic checks, shared holes, finite-field counterexamples and resume."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import numpy as np
import repair_learning as r


def preservation():
    baseline=json.loads((r.RESULTS/'preservation_baseline.json').read_text(encoding='utf-8'))
    changed=[p for p,d in baseline['files'].items() if not (r.ROOT/p).exists() or r.sha(r.ROOT/p)!=d]
    assert not changed,changed
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip()
    assert head==baseline['base_head']
    return dict(files_unchanged=len(baseline['files']),head_unchanged=head)


def scalar(node,x,y):
    if isinstance(node,int):return node%17
    if isinstance(node,str):return x if node=='x' else y
    a=scalar(node[1],x,y);b=scalar(node[2],x,y)
    return {'add':a+b,'sub':a-b,'mul':a*b}[node[0]]%17


def structural():
    rng=random.Random(1717);checks=0
    for depth in range(4):
        for _ in range(12):
            node=r.expression(rng,depth)
            actual=r.evaluate(node,r.GRID).tolist()
            assert actual==[scalar(node,x,y) for x,y in r.GRID.tolist()];checks+=289
    edits=[dict(before=('mul',('add','x',2),'y'),after=('add',('mul','x','y'),2)),
           dict(before=('mul',('add','y',3),'x'),after=('add',('mul','y','x'),3))]
    rules=r.derive_rules(edits,r.Work());assert len(rules)==1
    rule=rules[0]
    assert r.holes(rule['left'])==r.holes(rule['right'])=={0,1,2}
    actual=('mul',('add',('sub','x',7),('mul','x','y')),('add',5,'y'))
    bindings={};assert r.match(rule['left'],actual,bindings,r.Work())
    expected=('add',('mul',('sub','x',7),('add',5,'y')),('mul','x','y'))
    assert r.instantiate(rule['right'],bindings)==expected
    repeated=('add',('hole',0),('hole',0))
    assert not r.match(repeated,('add','x','y'),{},r.Work())
    assert r.match(repeated,('add','x','x'),{},r.Work())
    bad_edits=[dict(before=('add','x',1),after=('mul','y',2)),dict(before=('add','x',3),after=('mul','x',4))]
    assert not r.derive_rules(bad_edits,r.Work())
    entry=dict(signature=('L','mul','add'),template=('L','add','mul',(0,2,1)))
    bulk_checks=0
    for depth in range(5):
        for _ in range(8):
            tree=r.expression(rng,depth)
            assert r.bulk_rule(tree,rule,r.Work())==r.bulk_template(tree,entry,r.Work());bulk_checks+=1
    # A one-pass x -> x+x pattern terminates without visiting newly created nodes.
    expanding=dict(left='x',right=('add','x','x'))
    assert r.bulk_rule('x',expanding,r.Work())==('add','x','x')
    work=r.Work();work.cap=3
    try:r.evidence_loss(('add','x',1),np.array([[1,2],[2,3]]),np.array([2,3]),work)
    except r.Exhausted:pass
    else:raise AssertionError('budget should prevent the atomic evaluation')
    assert work.used==0
    # All 16 observed x values collide, a missing 17th value separates functions.
    product=1
    for value in range(16):product=('mul',product,('sub','x',value))
    values=[scalar(product,x,0) for x in range(17)]
    assert values[:16]==[0]*16 and values[16]!=0
    assert all((x**17-x)%17==0 for x in range(17)) and 17**17-17!=0
    return dict(scalar_field_outputs=checks,shared_holes=True,repeated_holes_enforced=True,
                free_rhs_holes_rejected=True,rule_template_equivalence_cases=bulk_checks,
                generated_nodes_not_revisited=True,atomic_budget_limit=True,
                finite_sample_counterexample=True,field_not_integer_semantics=True)


def restore_state(state):
    state=copy.deepcopy(state)
    state['library']=[r.frozen(p) for p in state['library']]
    for entry in state['priorities']:
        entry['signature']=r.frozen(entry['signature']);entry['template']=r.frozen(entry['template'])
    for rule in state['rules']:
        rule['left']=r.frozen(rule['left']);rule['right']=r.frozen(rule['right'])
    for edit in state['source_edits']:
        for key in ('signature','template','before','after'):edit[key]=r.frozen(edit[key])
    return state


def restore_life(life):
    life=copy.deepcopy(life)
    for task in life['sources']:
        task['initial']=r.frozen(task['initial']);task['target']=r.frozen(task['target'])
    for task in life['targets']:
        task['initial']=r.frozen(task['initial'])
        task['targets']={key:r.frozen(value) for key,value in task['targets'].items()}
    return life


def continuation(payload):
    life=restore_life(payload['life']);state=restore_state(payload['state'])
    return r.run_targets(life,{},payload['method'],payload['regime'],start_index=payload['start_index'],state=state,used=payload['used'])


def recorded(data):
    assert data['source_sha256']==r.sha(r.HERE/'repair_learning.py')
    assert data['protocol_sha256']==r.sha(r.PROTOCOL)
    assert len(data['runs'])==360
    lives={life['seed']:restore_life(life) for life in data['lives']}
    acquisitions={a['seed']:a for a in data['acquisitions']};accepted_checks=0;source_checks=0;false_accepts=0
    for acquired in data['acquisitions']:
        life=lives[acquired['seed']]
        for source,record in zip(life['sources'],acquired['source_records']):
            if record['accepted'] is not None:
                program=r.frozen(record['accepted'])
                assert [scalar(program,x,y) for x,y in source['xy']]==source['labels']
                source_checks+=1
        for rule in acquired['state']['rules']:
            left,right=r.frozen(rule['left']),r.frozen(rule['right'])
            assert r.holes(right)<=r.holes(left)
            for i,j in rule['provenance']:
                for index in (i,j):
                    edit=acquired['source_edits'][index];bindings={}
                    assert r.match(left,r.frozen(edit['before']),bindings,r.Work())
                    assert r.instantiate(right,bindings)==r.frozen(edit['after'])
    for run in data['runs']:
        life=lives[run['seed']];acquired=acquisitions[run['seed']]
        previous=acquired['base_cost']+(acquired['rule_acquisition_cost'] if run['method'] in ('learned_rules','rules_disabled_paid') else 0)
        assert len(run['records'])==8
        for record,task in zip(run['records'],life['targets']):
            assert record['cumulative_work']==previous+record['work']<=r.LIMIT
            assert record['work']<=r.TASK_LIMIT;previous=record['cumulative_work']
            accepted=record['accepted']
            if accepted is not None:
                accepted=r.frozen(accepted)
                assert [scalar(accepted,x,y) for x,y in task['xy']]==task['labels'][run['regime']]
                errors=sum(scalar(accepted,x,y)!=scalar(task['targets'][run['regime']],x,y) for x,y in r.GRID.tolist())
                assert errors==record['grid_errors'] and record['full_field_correct']==(errors==0)
                accepted_checks+=1;false_accepts+=errors>0
            else:assert not record['full_field_correct'] and record['grid_errors'] is None
        assert run['total_work']==previous
    return dict(trajectories=360,target_attempts=2880,source_evidence_accepts=source_checks,
                full_grid_accepted_program_checks=accepted_checks,false_accepts=false_accepts,
                rule_provenance_checked=True,budget_accounting_checked=True)


def restarts(data):
    checks=[];life=next(l for l in data['lives'] if l['seed']==7)
    with tempfile.TemporaryDirectory(prefix='cevahir_repair_') as folder:
        source,target=Path(folder)/'input.json',Path(folder)/'output.json'
        for run in [run for run in data['runs'] if run['seed']==7]:
            at=run['records'][3]
            payload=dict(life=life,state=at['state_after'],used=at['cumulative_work'],start_index=4,method=run['method'],regime=run['regime'])
            source.write_text(r.encode(payload),encoding='utf-8')
            subprocess.run([sys.executable,__file__,'--resume',str(source),str(target)],check=True,capture_output=True,text=True)
            actual=json.loads(target.read_text(encoding='utf-8'))
            assert actual['records']==run['records'][4:]
            assert actual['final_state']==run['final_state'] and actual['total_work']==run['total_work']
            checks.append(dict(method=run['method'],regime=run['regime'],exact_equal=True))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--structural-only',action='store_true')
    p.add_argument('--resume',nargs=2);p.add_argument('--rerun',action='store_true');args=p.parse_args()
    if args.resume:
        source,target=map(Path,args.resume);payload=json.loads(source.read_text(encoding='utf-8'))
        target.write_text(r.encode(continuation(payload)),encoding='utf-8');return
    result=dict(structural=structural(),preservation=preservation())
    if not args.structural_only:
        data=json.loads((r.RESULTS/'repair_learning.json').read_text(encoding='utf-8'))
        result['recorded']=recorded(data);result['fresh_process_resume']=restarts(data)
        if args.rerun:
            run=subprocess.run([sys.executable,str(r.HERE/'repair_learning.py')],check=True,capture_output=True,text=True)
            result['full_rerun']=dict(returncode=run.returncode,stdout=run.stdout.strip())
        (r.RESULTS/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
