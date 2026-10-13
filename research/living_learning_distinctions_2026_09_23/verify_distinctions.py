"""Independent machine semantics, refinement witnesses, persistence and replay."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import distinction_learning as c

def scalar(machine,word):
    states=[machine['start']]
    for character in word:states.append(machine['transitions'][states[-1]][0 if character=='0' else 1])
    return machine['outputs'][states[-1]]

def refinement_partition(machine):
    # Independent Moore partition refinement for all states, including unreachable ones.
    groups=list(machine['outputs'])
    while True:
        keys=[(machine['outputs'][s],*(groups[t] for t in machine['transitions'][s])) for s in range(len(groups))]
        catalog={key:i for i,key in enumerate(dict.fromkeys(keys))};updated=[catalog[key] for key in keys]
        if all((groups[i]==groups[j])==(updated[i]==updated[j]) for i in range(len(groups)) for j in range(len(groups))):return updated
        groups=updated

def independent_equivalent(a,b):
    offset=len(a['outputs'])
    combined=dict(outputs=a['outputs']+b['outputs'],transitions=a['transitions']+[[offset+t for t in row] for row in b['transitions']])
    groups=refinement_partition(combined)
    return groups[a['start']]==groups[offset+b['start']]

def structural():
    target=dict(start=0,transitions=[[1,0],[1,2],[2,2]],outputs=[0,0,1])
    work=c.Work();env=c.Environment(target);learner=c.Learner(env.query,work,'adaptive');learner.stabilize()
    assert len(learner.model['outputs'])==1 and not c.equivalent(learner.model,target)['equal']
    learner.observe('01',env.observe('01'));learner.stabilize()
    assert '1' in learner.E and independent_equivalent(learner.model,target)
    assert all(c.output(learner.model,w)==scalar(target,w) for w in c.words(8))
    assert len(learner.splits)>0
    # Exact equivalence checker cross-check over all two-state binary machines.
    machines=[dict(start=0,transitions=[list(bits[:2]),list(bits[2:])],outputs=list(labels))
              for bits in __import__('itertools').product(range(2),repeat=4)
              for labels in __import__('itertools').product(range(2),repeat=2)]
    checks=0
    for a in machines:
        for b in machines:
            result=c.equivalent(a,b)
            assert result['equal']==independent_equivalent(a,b)
            if not result['equal']:assert scalar(a,result['witness'])!=scalar(b,result['witness'])
            checks+=1
    work=c.Work();work.used=c.INTERNAL_LIMIT-1
    try:work.charge(2,'test')
    except c.Budget:pass
    else:raise AssertionError('budget')
    assert work.used==c.INTERNAL_LIMIT-1
    return dict(constructive_three_state_witness=True,all_words_through_length_8=511,
                two_state_pair_equivalence_checks=checks,atomic_budget=True)

def preservation():
    baseline=json.loads((c.RESULTS/'preservation_baseline.json').read_text(encoding='utf-8'))
    changed=[p for p,d in baseline['files'].items() if not (c.ROOT/p).is_file() or c.sha(c.ROOT/p)!=d]
    assert not changed,changed
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    assert head==baseline['base_head']
    return dict(files_unchanged=len(baseline['files']),head_unchanged=head)

def recorded(data):
    assert data['source_sha256']==c.sha(c.__file__) and data['protocol_sha256']==c.sha(c.PROTOCOL)
    worlds={(world['seed'],world['family']):world for world in data['worlds']}
    snapshots=0;labels=0;splits=0;long_outputs=0;minima={}
    for key,world in worlds.items():
        target=world['target'];reached={target['start']}
        while True:
            new=reached|{t for s in reached for t in target['transitions'][s]}
            if new==reached:break
            reached=new
        assert len(reached)==len(target['outputs'])
        groups=refinement_partition(target);minima[str(key)]=len({groups[s] for s in reached})
    for run in data['runs']:
        world=worlds[run['seed'],run['family']];target=world['target'];last_model=None
        for record in run['records']:
            state=record['state_after'];learner=state['learner'];model=learner['model'];evaluation=record['evaluation']
            assert record['work']==sum(state['work']['categories'].values())<=c.INTERNAL_LIMIT
            assert state['operating_work']==sum(len(w)+1 for batch in state['history'] for w,_ in batch)
            assert record['environment_units']==state['environment']['passive_units']+sum(q['cost'] for q in state['environment']['queries'])
            assert evaluation['exact']['equal']==independent_equivalent(model,target)
            if evaluation['exact']['witness'] is not None:
                w=evaluation['exact']['witness'];assert scalar(model,w)!=scalar(target,w)
            errors=sum(scalar(model,w)!=scalar(target,w) for w in world['tests']);long_outputs+=len(world['tests'])
            assert errors==evaluation['long_errors']
            assert record['prequential_errors']==sum(p!=y for p,(_,y) in zip(record['predictions'],record['observations']))
            if last_model is not None:assert record['predictions']==[scalar(last_model,w) for w,_ in record['observations']]
            last_model=model
            assert record['state_json_bytes']==len(c.encode(state).encode())
            if not state['frozen'] and run['method']!='fixed_tests':
                assert evaluation['known_errors']==0
                # Closed and consistent table independently reconstructed.
                E=learner['E'];S=learner['S'];known=learner['known'];rows={s:tuple(known[s+e] for e in E) for s in set(S)|{s+a for s in S for a in '01'}}
                catalog={rows[s] for s in S}
                assert all(row in catalog for row in rows.values())
                grouped={}
                for s in S:
                    if rows[s] in grouped:
                        t=grouped[rows[s]];assert all(rows[s+a]==rows[t+a] for a in '01')
                    else:grouped[rows[s]]=s
            snapshots+=1
        final=run['final_state'];learner=final['learner']
        for word,value in final['environment']['cache'].items():assert scalar(target,word)==value;labels+=1
        for word,value in learner['known'].items():assert scalar(target,word)==value
        for witness in learner['splits']:
            left,right=witness['left'],witness['right'];old=witness['previous_tests'];new=witness['new_test']
            assert [scalar(target,left+e) for e in old]==[scalar(target,right+e) for e in old]==witness['old_row']
            actual=[scalar(target,left+new),scalar(target,right+new)]
            assert actual==witness['outcomes'] and actual[0]!=actual[1];splits+=1
    pairs=[]
    lookup={(run['seed'],run['family'],run['method']):run for run in data['runs']}
    for world in data['worlds']:
        a=lookup[world['seed'],world['family'],'adaptive'];b=lookup[world['seed'],world['family'],'rebuild'];equal=0
        for x,y in zip(a['records'],b['records']):
            if not x['state_after']['frozen'] and not y['state_after']['frozen']:
                assert x['state_after']['learner']==y['state_after']['learner']
                assert x['state_after']['environment']==y['state_after']['environment']
                assert x['predictions']==y['predictions'];equal+=1
        pairs.append(dict(seed=world['seed'],family=world['family'],equal_unfrozen_blocks=equal))
    return dict(snapshots=snapshots,independent_cached_labels=labels,split_witnesses=splits,
                long_output_checks=long_outputs,exact_checks=snapshots,minimal_target_states=minima,paired_rebuild=pairs)

def restarts(data):
    checks=[]
    with tempfile.TemporaryDirectory(prefix='cevahir_distinctions_') as folder:
        source=Path(folder)/'input.json';dest=Path(folder)/'output.json'
        for run in [item for item in data['runs'] if item['seed']==5]:
            world=next(w for w in data['worlds'] if (w['seed'],w['family'])==(run['seed'],run['family']))
            for checkpoint in (3,7):
                source.write_text(c.encode(dict(world=world,method=run['method'],state=run['records'][checkpoint]['state_after'])),encoding='utf-8')
                subprocess.run([sys.executable,__file__,'--resume',str(source),str(dest)],check=True,capture_output=True)
                actual=json.loads(dest.read_text(encoding='utf-8'))
                assert actual['records']==run['records'][checkpoint+1:] and actual['final_state']==run['final_state']
                checks.append(dict(family=run['family'],method=run['method'],checkpoint=checkpoint,exact_equal=True))
    return checks

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--structural-only',action='store_true')
    parser.add_argument('--resume',nargs=2);parser.add_argument('--rerun',action='store_true');args=parser.parse_args()
    if args.resume:
        source,dest=map(Path,args.resume);payload=json.loads(source.read_text(encoding='utf-8'))
        dest.write_text(c.encode(c.run_life(payload['world'],payload['method'],payload['state'])),encoding='utf-8');return
    result=dict(structural=structural(),preservation=preservation())
    if not args.structural_only:
        data=json.loads((c.RESULTS/'distinction_learning.json').read_text(encoding='utf-8'))
        result['recorded']=recorded(data);result['fresh_process_resume']=restarts(data)
        if args.rerun:
            run=subprocess.run([sys.executable,str(c.HERE/'distinction_learning.py')],check=True,capture_output=True,text=True)
            result['full_rerun']=dict(returncode=run.returncode,stdout=run.stdout.strip())
        (c.RESULTS/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({k:value for k,value in result.items() if k!='recorded'}))
    if 'recorded' in result:print(c.encode({k:value for k,value in result['recorded'].items() if k not in ('minimal_target_states','paired_rebuild')}))

if __name__=='__main__':main()
