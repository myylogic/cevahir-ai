"""Independent state replay, relation witnesses, invariance and fresh processes."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import live_revision as c

def trace(model,word,start=None):
    result=[model['start'] if start is None else start]
    for symbol in word:result.append(model['transitions'][result[-1]][0 if symbol=='0' else 1])
    return result

def renumber(model,seed):
    rng=random.Random(seed);perm=list(range(len(model['outputs'])));rng.shuffle(perm)
    transitions=[None]*len(perm);outputs=[None]*len(perm)
    for q in range(len(perm)):
        transitions[perm[q]]=[perm[t] for t in model['transitions'][q]];outputs[perm[q]]=model['outputs'][q]
    return dict(start=perm[model['start']],transitions=transitions,outputs=outputs),perm

def structural():
    parity=dict(start=0,transitions=[[0,1],[1,0]],outputs=[0,1])
    # Parity is always separately retained by the actual methods; use a different
    # lost summary for the compositional counterexample: number of ones mod 3.
    counter=dict(start=0,transitions=[[0,1],[1,2],[2,0]],outputs=[0,0,1])
    constant=dict(start=0,transitions=[[0,0]],outputs=[0])
    direct,_=c.relation(counter,counter,c.Meter())
    erased,_=c.relation(constant,counter,c.Meter())
    assert {row[2] for row in direct if row[0]==0 and row[1]==1}=={1}
    assert {row[2] for row in erased if row[0]==0}=={0,1,2}
    # Equal parity and arbitrarily long equal suffix do not identify mod-3 state.
    suffix='00101010'
    assert sum(map(int,suffix))%2==sum(map(int,'11'+suffix))%2
    assert trace(counter,suffix)[-1]!=trace(counter,'11'+suffix)[-1]
    machines=[dict(start=0,transitions=[list(t[:2]),list(t[2:])],outputs=list(y))
              for t in itertools.product(range(2),repeat=4) for y in itertools.product(range(2),repeat=2)]
    words=list(c.prior.words(5));checks=0;suffix_checks=0
    for old in machines:
        for new in machines:
            relation,_=c.relation(old,new,c.Meter())
            shifted,_=c.relation(old,new,c.Meter(),'01')
            for word in words:
                row=(sum(map(int,word))%2,trace(old,word)[-1],trace(new,word)[-1]);assert row in relation;checks+=1
                extended=word+'01'
                row=(sum(map(int,extended))%2,trace(old,extended)[-1],trace(new,extended)[-1]);assert row in shifted;suffix_checks+=1
    # A wrong model may have perfectly sound transfer but wrong world predictions.
    learner=c.Tracker('model_relation',constant)
    for bit in '111':learner.advance(bit)
    assert learner.predict()==0 and parity['outputs'][trace(parity,'111')[-1]]==1
    return dict(short_history_relation_checks=checks,suffix_relation_checks=suffix_checks,
                retained_parity_does_not_recover_mod3=True,intermediate_summary_can_lose_information=True,
                sound_transfer_can_use_wrong_model=True)

def preservation():
    baseline=json.loads((c.RESULTS/'preservation_baseline.json').read_text(encoding='utf-8'))
    changed=[p for p,d in baseline['files'].items() if not (c.ROOT/p).is_file() or c.sha(c.ROOT/p)!=d]
    assert not changed,changed
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip();assert head==baseline['base_head']
    return dict(files_unchanged=len(baseline['files']),head_unchanged=head)

def recorded(data):
    assert data['source_sha256']==c.sha(c.__file__) and data['protocol_sha256']==c.sha(c.PROTOCOL)
    assert data['previous_source_sha256']==c.sha(c.PREVIOUS/'distinction_learning.py')
    assert data['previous_result_sha256']==c.sha(c.SOURCE)
    lookup={(s['seed'],s['family']):s for s in data['sources']};contexts={};points=0;sound=0;parity_checks=0
    for source in data['sources']:
        for stride in c.STRIDES:
            bits=c.stream_for(source['seed'],stride)
            truths=trace(source['target'],bits);parities=[0]
            for bit in bits:parities.append(parities[-1]^int(bit))
            models=[trace(model,bits) for model in source['models']]
            contexts[source['seed'],source['family'],stride]=(truths,parities,models)
    for run in data['runs']:
        source=lookup[run['seed'],run['family']];truths,parities,models=contexts[run['seed'],run['family'],run['stride']]
        previous=source['initial'];last_work=0
        for record in run['records']:
            block=record['block'];model=source['models'][block];state=record['state_after']
            assert record['install']['changed']==(previous!=model);previous=model
            assert state['model']==model and record['work']==sum(state['categories'].values())>=last_work;last_work=record['work']
            assert record['state_json_bytes']==len(c.encode(state).encode())
            assert record['calibration']==source['calibration'][block]
            expected_pos=(block+1)*run['stride']+16
            assert state['position']==expected_pos
            assert state['parity']==parities[expected_pos]
            bits=c.stream_for(run['seed'],run['stride'])
            assert state['history']==(bits[:expected_pos] if run['method']=='full_archive' else '')
            assert state['suffix']==(bits[expected_pos-8:expected_pos] if run['method'] in ('parity_suffix8','model_relation_suffix8') else '')
            for point in record['points']:
                pos=(block+1)*run['stride']+point['t'];replay=models[block][pos]
                assert pos==point['position'] and point['replay_state']==replay
                assert point['replay_output']==model['outputs'][replay]
                assert point['truth']==source['target']['outputs'][truths[pos]]
                assert point['parity']==parities[pos];parity_checks+=1
                assert point['replay_contained']==(replay in point['possible'])
                outputs={model['outputs'][q] for q in point['possible']}
                assert point['prediction']==(next(iter(outputs)) if len(outputs)==1 else None)
                assert point['size']==len(set(point['possible']))>0
                if run['method']!='reset_start':assert point['replay_contained'];sound+=1
                if run['method']=='full_archive':assert point['possible']==[replay]
                points+=1
        assert run['final_state']==run['records'][-1]['state_after']
    runs={(r['seed'],r['family'],r['stride'],r['method']):r for r in data['runs']};subset=0
    for source in data['sources']:
        for stride in c.STRIDES:
            a=runs[source['seed'],source['family'],stride,'model_relation_suffix8']
            b=runs[source['seed'],source['family'],stride,'parity_suffix8']
            for ra,rb in zip(a['records'],b['records']):
                for pa,pb in zip(ra['points'],rb['points']):assert set(pa['possible'])<=set(pb['possible']);subset+=1
    return dict(paths=len(data['runs']),prediction_checks=points,sound_replay_containment_checks=sound,
                parity_checks=parity_checks,suffix_control_subset_checks=subset)

def metamorphic(data):
    results=[]
    for source in [s for s in data['sources'] if s['seed']==5]:
        changed=copy.deepcopy(source);cache={};maps=[]
        for model in [source['initial'],*source['models']]:
            key=c.encode(model)
            if key not in cache:cache[key]=renumber(model,9187+len(cache)*727)
            maps.append(cache[key])
        changed['initial']=maps[0][0];changed['models']=[value[0] for value in maps[1:]]
        for method in c.METHODS:
            actual=c.run_life(changed,32,method)
            expected=next(run for run in data['runs'] if (run['seed'],run['family'],run['stride'],run['method'])==(5,source['family'],32,method))
            for i,(a,b) in enumerate(zip(actual['records'],expected['records'])):
                permutation=maps[i+1][1]
                assert a['work']==b['work']
                for pa,pb in zip(a['points'],b['points']):
                    assert pa['prediction']==pb['prediction'] and pa['truth']==pb['truth']
                    assert pa['possible']==sorted(permutation[q] for q in pb['possible'])
            results.append(dict(family=source['family'],method=method,renaming_invariant=True))
    return results

def restarts(data):
    checks=[]
    with tempfile.TemporaryDirectory(prefix='cevahir_live_revision_') as folder:
        src=Path(folder)/'input.json';dest=Path(folder)/'output.json'
        for run in [r for r in data['runs'] if r['seed']==5]:
            source=next(s for s in data['sources'] if (s['seed'],s['family'])==(run['seed'],run['family']))
            payload=dict(source=source,stride=run['stride'],method=run['method'],state=run['records'][7]['state_after'])
            src.write_text(c.encode(payload),encoding='utf-8')
            subprocess.run([sys.executable,__file__,'--resume',str(src),str(dest)],check=True,capture_output=True)
            actual=json.loads(dest.read_text(encoding='utf-8'))
            assert actual['records']==run['records'][8:] and actual['final_state']==run['final_state']
            checks.append(dict(family=run['family'],stride=run['stride'],method=run['method'],exact_equal=True))
    return checks

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--structural-only',action='store_true')
    parser.add_argument('--resume',nargs=2);parser.add_argument('--rerun',action='store_true');args=parser.parse_args()
    if args.resume:
        src,dest=map(Path,args.resume);payload=json.loads(src.read_text(encoding='utf-8'))
        result=c.run_life(payload['source'],payload['stride'],payload['method'],payload['state'],8)
        dest.write_text(c.encode(result),encoding='utf-8');return
    result=dict(structural=structural(),preservation=preservation())
    if not args.structural_only:
        data=json.loads((c.RESULTS/'live_revision.json').read_text(encoding='utf-8'))
        result['recorded']=recorded(data);result['renaming']=metamorphic(data);result['fresh_process_resume']=restarts(data)
        if args.rerun:
            run=subprocess.run([sys.executable,str(c.HERE/'live_revision.py')],check=True,capture_output=True,text=True)
            result['full_rerun']=dict(returncode=run.returncode,stdout=run.stdout.strip())
        (c.RESULTS/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({key:value for key,value in result.items() if key not in ('renaming','fresh_process_resume','full_rerun')}))
    if 'full_rerun' in result:print('full rerun, 18 renamings and 36 process restarts passed')

if __name__=='__main__':main()
