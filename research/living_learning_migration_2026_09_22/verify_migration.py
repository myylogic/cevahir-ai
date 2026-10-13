"""Independent finite witnesses, causal ordering, preservation and process resume."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import migration_learning as m


def preservation():
    baseline=json.loads((m.RESULTS/'preservation_baseline.json').read_text(encoding='utf-8'))
    changed=[p for p,d in baseline['files'].items() if not (m.ROOT/p).exists() or m.sha(m.ROOT/p)!=d]
    assert not changed,changed
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip()
    assert head==baseline['base_head']
    return dict(files_unchanged=len(baseline['files']),head_unchanged=head)


def scalar_output(program,word):
    bits=list(map(int,word))
    if program['kind']=='delay': return bits[-program['k']]
    if program['kind']=='mod3': return program['readout'][sum(bits)%3]
    state=0
    for bit in bits: state=program['transitions'][state][bit]
    return program['readout'][state]


def structural():
    delay=m.PROGRAMS[4];mod=m.PROGRAMS[8];transitions=0
    for bits in itertools.product((0,1),repeat=5):
        s=set(range(32));q=set(range(3))
        for i,bit in enumerate(bits,1):
            s={m.step(delay,state,bit) for state in s};q={m.step(mod,state,bit) for state in q}
            assert len(s)==2**(5-i) and len(q)==3;transitions+=1
        assert s=={int(''.join(map(str,bits)),2)}
    # These finite constructions extend to longer fixed lengths by zero padding on the left.
    suffix_by_mod6={i:set() for i in range(6)}
    mod3_by_parity_suffix={}
    for bits in itertools.product('01',repeat=13):
        word=''.join(bits);count=word.count('1')
        suffix_by_mod6[count%6].add(word[-5:])
        mod3_by_parity_suffix.setdefault((count%2,word[-8:]),set()).add(count%3)
    assert all(len(v)==32 for v in suffix_by_mod6.values())
    assert len(mod3_by_parity_suffix)==512 and all(v=={0,1,2} for v in mod3_by_parity_suffix.values())
    for policy in m.POLICIES:
        c=m.Controller(m.OLD_PROGRAMS[26],policy)
        for bit in [1,0,1,1]*5: c.feed(bit)
        c.install(mod);before=m.encode(c.state());p=c.predict()
        assert before==m.encode(c.state())
        copied=m.Controller.restore(c.state());copied.new_states.append(77)
        assert before==m.encode(c.state())
        c.observe_new_label(2)
        assert c.new_states==[2]
    brier=lambda p,y: sum((value-int(i==y))**2 for i,value in enumerate(p))
    assert brier([.5,.5,0],0)==.5
    assert abs(brier([1/3]*3,0)-2/3)<1e-15
    assert brier([1,0,0],1)==2
    return dict(synchronization_transition_checks=transitions,exhaustive_histories=8192,
                reserve8_parity_fibers=512,reserve_mod6_fibers=6,
                distinct_output_brier_checked=True,prediction_does_not_mutate=True,isolated_states=True,
                one_modulo_label_selects_single_state=True)


def run_actions(controller,actions):
    records=[]
    for action in actions:
        controller.feed(action['bit']);records.append(controller.predict())
        if action['label'] is not None: controller.observe_new_label(action['label'])
    return dict(predictions=records,final_state=controller.state())


def restarts(data):
    life=next(l for l in data['lives'] if l['seed']==9);stream=life['streams']['256'];checks=[]
    with tempfile.TemporaryDirectory(prefix='cevahir_migration_') as folder:
        source,target=Path(folder)/'input.json',Path(folder)/'output.json'
        for row in [r for r in data['runs'] if r['seed']==9 and r['prefix']==256]:
            for checkpoint in (3,16):
                c=m.Controller.restore(row['installation'])
                for offset in range(1,checkpoint+1):
                    c.feed(int(stream[256+offset-1]))
                    if offset==16: c.observe_new_label(row['rows'][16]['new_truth'])
                actions=[dict(bit=int(stream[256+i-1]),label=row['rows'][i]['new_truth'] if i==16 else None)
                         for i in range(checkpoint+1,65)]
                payload=dict(state=c.state(),actions=actions)
                expected=run_actions(c,actions)
                source.write_text(m.encode(payload),encoding='utf-8')
                subprocess.run([sys.executable,__file__,'--resume',str(source),str(target)],check=True,capture_output=True,text=True)
                actual=json.loads(target.read_text(encoding='utf-8'))
                assert m.encode(actual)==m.encode(expected)
                checks.append(dict(task=row['task'],policy=row['policy'],checkpoint=checkpoint,exact_equal=True))
    return checks


def recorded(data):
    assert data['source_sha256']==m.sha(m.HERE/'migration_learning.py')
    assert data['protocol_sha256']==m.sha(m.PROTOCOL)
    assert len(data['runs'])==576 and len(data['calibrations'])==72
    lives={l['seed']:l for l in data['lives']};calibration_checks=0;compiled_correct=0
    for c in data['calibrations']:
        programs=m.OLD_PROGRAMS if c['task']=='old' else m.PROGRAMS
        indices=list(range(len(programs)))
        for word,row in zip(lives[c['seed']]['calibration'],c['prequential']):
            values={idx:scalar_output(programs[idx],word) for idx in indices}
            p=[sum(y==value for value in values.values())/len(values) for y in range(3)]
            assert p==row['p'] and len(indices)==row['candidates_before']
            indices=[idx for idx in indices if values[idx]==row['label']];calibration_checks+=1
        assert indices==c['consistent'] and c['selected']==indices[0]
        # Teacher family identity is audited only here, never used by the learner.
        expected=26 if c['task']=='old' else 4 if c['task']=='delay_5' else 8+c['seed']%6
        assert c['selected']==expected and len(indices)==1
        compiled_correct+=1
    event_checks=0
    for row in data['runs']:
        c=m.Controller.restore(row['installation']);stream=lives[row['seed']]['streams'][str(row['prefix'])]
        prefix=row['prefix']
        assert len(row['rows'])==65
        for offset,record in enumerate(row['rows']):
            if offset: c.feed(int(stream[prefix+offset-1]))
            prediction=c.predict();word=stream[:prefix+offset]
            assert record['old_truth']==sum(map(int,word))%2
            expected=int(word[-5]) if row['task']=='delay_5' else m.PROGRAMS[8+row['seed']%6]['readout'][sum(map(int,word))%3]
            assert record['new_truth']==expected
            for key,value in prediction.items(): assert record[key]==value
            actual_state=int(word[-5:],2) if row['task']=='delay_5' else word.count('1')%3
            assert record['true_state_in_set']==(actual_state in c.new_states)
            assert record['certain_wrong']==(len(prediction['outputs'])==1 and prediction['outputs'][0]!=expected)
            assert record['old_mse']==int(record['old']!=record['old_truth'])
            assert abs(record['new_brier']-sum((v-int(i==expected))**2 for i,v in enumerate(prediction['p'])))<1e-15
            if offset==16: c.observe_new_label(expected)
            event_checks+=1
        assert m.encode(c.state())==m.encode(row['final_state'])
        assert hashlib.sha256(m.encode(c.state()).encode()).hexdigest()==row['final_state_sha256']
        assert c.labels==1 and not c.archive and not c.recent
        expected_replay=prefix if row['policy']=='archive' else 8 if row['policy']=='reserve_8' and row['task']=='delay_5' else 0
        assert c.install_replay_bits==expected_replay
    return dict(calibration_predictions=calibration_checks,compiled_correct=compiled_correct,
                unique_consistent_all=True,mainstream_events=event_checks,label_after_prediction=True,
                one_live_new_label_per_path=True,final_states_match=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--structural-only',action='store_true')
    p.add_argument('--resume',nargs=2);p.add_argument('--rerun',action='store_true');args=p.parse_args()
    if args.resume:
        source,target=map(Path,args.resume);payload=json.loads(source.read_text(encoding='utf-8'))
        target.write_text(m.encode(run_actions(m.Controller.restore(payload['state']),payload['actions'])),encoding='utf-8');return
    result=dict(structural=structural(),preservation=preservation())
    if not args.structural_only:
        data=json.loads((m.RESULTS/'migration_learning.json').read_text(encoding='utf-8'))
        result['recorded']=recorded(data);result['fresh_process_resume']=restarts(data)
        if args.rerun:
            run=subprocess.run([sys.executable,str(m.HERE/'migration_learning.py')],check=True,capture_output=True,text=True)
            result['full_rerun']=dict(returncode=run.returncode,stdout=run.stdout.strip())
        (m.RESULTS/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
