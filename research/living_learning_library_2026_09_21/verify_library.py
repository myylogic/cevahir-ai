"""Independent scalar semantics, bounded search, state interventions and replay.

--rerun recomputes both experiment records. Writes only verification.json.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import library_learning as lib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESULTS=HERE/'results'


def scalar(program,x):
    for instruction in program:
        if instruction==0:x=x+1
        elif instruction==1:x=2*x
        elif instruction==2:x=x*x
        elif instruction==3:x=-x
        else:raise AssertionError(instruction)
        x=x%101
    return x


def independent_program_checks():
    checked=0
    all_small=[p for length in range(4) for p in product(range(4),repeat=length)]
    xs=tuple(range(8));ax=tuple(range(8,16))
    for p in all_small:
        assert tuple(scalar(p,x) for x in range(101))==lib.execute(p,range(101))
        ys=tuple(scalar(p,x) for x in xs);ay=tuple(scalar(p,x) for x in ax)
        # Brute enumeration has no signature pruning or stored prefix outputs.
        expected=next(q for q in all_small if tuple(scalar(q,x) for x in xs)==ys)
        actual=lib.synthesize(xs,ys,ax,ay,lib.PRIMITIVES,32768)
        assert actual['status']=='accepted' and actual['program']==list(expected)
        checked+=1
    macros=((0,0,0),(2,0),(1,2,1))
    for length in range(4):
        for indices in product(range(len(macros)),repeat=length):
            expanded=tuple(op for index in indices for op in macros[index])
            value=tuple(range(101))
            for index in indices:value=lib.execute(macros[index],value)
            assert value==tuple(scalar(expanded,x) for x in range(101))
    return dict(exhaustive_primitive_programs_up_to_length_three=checked,
                brute_search_program_equality=checked,macro_compositions=40)


def read_results():
    primary=json.loads((RESULTS/'library_learning.json').read_text(encoding='utf-8'))
    portfolio=json.loads((RESULTS/'portfolio_control.json').read_text(encoding='utf-8'))
    assert primary['source_sha256']==hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest()
    assert portfolio['source_sha256']==hashlib.sha256((HERE/'portfolio_control.py').read_bytes()).hexdigest()
    assert portfolio['primary_source_sha256']==primary['source_sha256']
    assert len(primary['runs'])==9216 and len(portfolio['runs'])==384
    assert len({(r['seed'],r['history'],r['budget'],r['task'],r['method']) for r in primary['runs']})==9216
    worlds={life['world']['seed']:life for life in primary['lives']}
    for row in primary['runs']:
        assert row['charged_vector_steps']<=row['budget']
        c=row['costs']
        assert row['charged_vector_steps']==c['search_vector_steps']+c['acceptance_vector_steps']
        t=next(t for t in worlds[row['seed']]['world']['probes'] if t['name']==row['task'])
        if row['status']=='accepted':
            target=[scalar(t['teacher'],x) for x in range(101)]
            pred=[scalar(row['program'],x) for x in range(101)]
            assert row['exact']==(pred==target)
            unseen=set(range(101))-set(t['train_x'])-set(t['accept_x'])
            assert row['unseen_correct']==sum(pred[x]==target[x] for x in unseen)
    assert primary['aggregate']==lib.summarize(primary['runs'])
    for row in portfolio['runs']:
        assert row['total_charged_vector_steps']<=row['budget']
        assert not row['primitive_half']['exact'] or row['selected']['exact']
    return primary,portfolio


def counter_and_leakage_checks(primary):
    checks=0
    for life in primary['lives'][:3]:
        w=life['world'];state=life['states']['12']
        for method in lib.METHODS:
            for task in (w['probes'][0],w['probes'][8]):
                tokens=lib.token_policy(state,method,w['seed'],w['home_motifs'])
                before=copy.deepcopy(state)
                expected=lib.solve_task(task,tokens,512,method=='expanded_proposals')
                # Instrument actual primitive calls, independently of charged counters.
                original=lib.step;calls=[]
                def counted(values,op):
                    calls.append(len(values))
                    return original(values,op)
                lib.step=counted
                try:actual=lib.solve_task(task,tokens,512,method=='expanded_proposals')
                finally:lib.step=original
                assert actual==expected and sum(calls)==8*actual['charged_vector_steps']
                altered=copy.deepcopy(task)
                altered.update(teacher=[3]*300,regime='UNTRUSTED',name='CHANGED',unseen_targets=[-999]*101)
                assert lib.solve_task(altered,tokens,512,method=='expanded_proposals')==expected
                assert state==before
                checks+=1
    return dict(instrumented_primitive_accounting_cases=checks,teacher_and_metadata_intervention_cases=checks,
                state_not_mutated_cases=checks)


def causal_state_checks(primary):
    erased=retained=0
    for life in primary['lives']:
        world=life['world'];accepted=[]
        for index,record in enumerate(life['acquisition'],1):
            t=world['initial'][index-1]
            if record['result']['status']=='accepted':
                accepted.append(record['result']['program'])
                assert all(scalar(accepted[-1],x)==scalar(t['teacher'],x) for x in range(101))
            if str(index) in life['states']:
                assert life['states'][str(index)]['programs']==accepted
        state=life['states']['12']
        assert lib.build_state(state['programs'])==state
        destroyed=copy.deepcopy(state);destroyed['library']=[]
        for p,q in zip(state['programs'],destroyed['programs'],strict=True):
            assert [scalar(p,x) for x in range(101)]==[scalar(q,x) for x in range(101)]
            retained+=1
        for t in world['probes']:
            tokens=lib.token_policy(destroyed,'learned_library',world['seed'],world['home_motifs'])
            assert tokens==lib.PRIMITIVES
            actual=lib.solve_task(t,tokens,512)
            expected=next(r for r in primary['runs'] if (r['seed'],r['history'],r['budget'],r['method'],r['task'])==
                          (world['seed'],12,512,'primitive_only',t['name']))
            assert all(actual[k]==expected[k] for k in actual)
            erased+=1
    return dict(old_programs_retained_after_library_erase=retained,erased_search_equals_primitive_cases=erased,
                all_warm_programs_exact=True,states_mined_only_from_acquired_programs=True)


def process_resume(primary):
    records=[]
    life=primary['lives'][2];world=life['world'];state=life['states']['12']
    with tempfile.TemporaryDirectory(prefix='cevahir-library-verify-') as temporary:
        for method in lib.METHODS:
            for task in (world['probes'][0],world['probes'][8]):
                payload=dict(state=state,method=method,seed=world['seed'],oracle=world['home_motifs'],task=task,budget=2048)
                ip=Path(temporary)/'input.json';op=Path(temporary)/'output.json'
                ip.write_text(json.dumps(payload),encoding='utf-8')
                call=subprocess.run([sys.executable,str(HERE/'library_learning.py'),'--resume-input',str(ip),'--resume-output',str(op)],capture_output=True,text=True)
                assert call.returncode==0,call.stderr
                actual=json.loads(op.read_text(encoding='utf-8'))
                tokens=lib.token_policy(state,method,world['seed'],world['home_motifs'])
                expected=lib.solve_task(task,tokens,2048,method=='expanded_proposals')
                assert actual==expected
                records.append(dict(method=method,regime=task['regime'],exact_json_equal=True))
    return records


def preservation():
    baseline=json.loads((RESULTS/'preservation_baseline.json').read_text(encoding='utf-8'))
    changed=[p for p,h in baseline['files'].items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline['base_head']
    return dict(files_byte_exact=len(baseline['files']),changed=changed,head_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rerun',action='store_true');args=p.parse_args()
    primary,portfolio=read_results()
    result=dict(date='2026-09-21',status='passed',independent_semantics=independent_program_checks(),
                result_contracts=dict(primary_conditions=9216,posthoc_conditions=384,budget_violations=0),
                instrumented_and_leakage=counter_and_leakage_checks(primary),
                causal_state=causal_state_checks(primary),fresh_process_resume=process_resume(primary),
                preservation=preservation())
    if args.rerun:
        def rerun(name):
            call=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True)
            assert call.returncode==0,(name,call.stdout,call.stderr)
            return dict(program=name,returncode=call.returncode,stdout=call.stdout.strip())
        with ThreadPoolExecutor(max_workers=2) as pool:
            result['complete_reruns']=list(pool.map(rerun,('library_learning.py','portfolio_control.py')))
    result['scope']='Independent scalar implementation and enumerator plus interventions/fresh processes; same-code full rerun is not independent-world replication or external peer review.'
    (RESULTS/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='passed',preservation=result['preservation'],fresh_process_cases=len(result['fresh_process_resume']),
                         complete_reruns=len(result.get('complete_reruns',[])))))


if __name__=='__main__':main()
