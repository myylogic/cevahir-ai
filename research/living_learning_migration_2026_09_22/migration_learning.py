"""Acquire supplied programs from labels, then install them in a live stream."""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESULTS=HERE/'results'
PROTOCOL=ROOT/'docs/research/living_learning_migration_2026_09_22/PROTOCOL.md'
POLICIES=('guess_zero','possible_states','reserve_8','reserve_mod6','archive','reset_all')
TASKS=('delay_5','mod3')
PREFIXES=(256,4096)
PROGRAMS=[dict(kind='delay',k=k) for k in range(1,9)]+[
    dict(kind='mod3',readout=list(p)) for p in itertools.permutations(range(3))]
OLD_PROGRAMS=[dict(kind='dfa',transitions=[[(trans>>(2*s+b))&1 for b in (0,1)] for s in (0,1)],
                   readout=[(out>>s)&1 for s in (0,1)]) for trans in range(16) for out in range(4)]


def encode(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def size(program):
    if program['kind']=='delay': return 2**program['k']
    return 3 if program['kind']=='mod3' else 2
def step(program,state,bit):
    if program['kind']=='delay': return (2*state+bit)%size(program)
    if program['kind']=='mod3': return (state+bit)%3
    return program['transitions'][state][bit]
def output(program,state):
    if program['kind']=='delay': return state>>(program['k']-1)
    return program['readout'][state]
def run(program,word):
    state=0
    for bit in word: state=step(program,state,int(bit))
    return state


def make_life(seed):
    calibration_rng=random.Random(771_229+seed*3571)
    words=[''.join(str(calibration_rng.randrange(2)) for _ in range(calibration_rng.randint(8,24))) for _ in range(128)]
    stream_rng=random.Random(918_011+seed*104729)
    return dict(seed=seed,calibration=words,streams={str(n):''.join(str(stream_rng.randrange(2)) for _ in range(n+64)) for n in PREFIXES})


def labels_for(task,word,seed):
    if task=='old': return word.count('1')%2
    if task=='delay_5': return int(word[-5])
    return list(itertools.permutations(range(3)))[seed%6][word.count('1')%3]


def calibrate(programs,words,labels):
    candidates=list(range(len(programs)));records=[];transitions=0
    for word,label in zip(words,labels):
        values={idx:output(programs[idx],run(programs[idx],word)) for idx in candidates}
        transitions+=len(word)*len(candidates)
        p=[sum(value==y for value in values.values())/len(values) for y in range(3)]
        records.append(dict(p=p,label=label,candidates_before=len(candidates)))
        candidates=[idx for idx in candidates if values[idx]==label]
        if not candidates: raise RuntimeError('no supplied program explains the calibration')
    return dict(selected=candidates[0],consistent=candidates,prequential=records,
                candidate_transitions=transitions,program=copy.deepcopy(programs[candidates[0]]))


class Controller:
    def __init__(self,old_program,policy):
        self.old_program=copy.deepcopy(old_program);self.policy=policy;self.old_state=0
        self.new_program=None;self.new_states=[];self.recent=[];self.mod6=0;self.archive=[]
        self.bits_seen=0;self.new_transition_checks=0;self.install_replay_bits=0;self.repairs=0;self.labels=0
        self.preinstall_storage=None

    def feed(self,bit):
        self.bits_seen+=1
        self.old_state=step(self.old_program,self.old_state,bit)
        if self.new_program is None:
            if self.policy=='reserve_8': self.recent=(self.recent+[bit])[-8:]
            if self.policy=='reserve_mod6': self.mod6=(self.mod6+bit)%6
            if self.policy=='archive': self.archive.append(bit)
        else:
            self.new_transition_checks+=len(self.new_states)
            self.new_states=sorted({step(self.new_program,s,bit) for s in self.new_states})

    def install(self,program):
        assert self.new_program is None
        self.preinstall_storage=dict(raw_bits=len(self.archive)+len(self.recent),
                                    auxiliary_counter_states=6 if self.policy=='reserve_mod6' else 1,
                                    auxiliary_counter_min_fixed_bits=3 if self.policy=='reserve_mod6' else 0)
        self.new_program=copy.deepcopy(program)
        states=list(range(size(program)))
        if self.policy in ('guess_zero','reset_all'): states=[0]
        elif self.policy=='reserve_8' and program['kind']=='delay':
            states=[run(program,self.recent)];self.install_replay_bits=len(self.recent)
        elif self.policy=='reserve_mod6' and program['kind']=='mod3': states=[self.mod6%3]
        elif self.policy=='archive':
            states=[run(program,self.archive)];self.install_replay_bits=len(self.archive)
        if self.policy=='reset_all': self.old_state=0
        self.new_states=states
        self.archive=[];self.recent=[];self.mod6=0

    def predict(self):
        values=sorted({output(self.new_program,state) for state in self.new_states})
        return dict(old=output(self.old_program,self.old_state),
                    p=[1/len(values) if y in values else 0. for y in range(3)],
                    state_count=len(self.new_states),outputs=values)

    def observe_new_label(self,label):
        selected=[state for state in self.new_states if output(self.new_program,state)==label]
        if not selected:
            self.repairs+=1
            selected=[state for state in range(size(self.new_program)) if output(self.new_program,state)==label]
        if not selected: raise RuntimeError('label outside installed program output alphabet')
        self.new_states=selected;self.labels+=1

    def state(self): return copy.deepcopy(self.__dict__)
    @classmethod
    def restore(cls,state):
        obj=cls.__new__(cls);obj.__dict__.update(copy.deepcopy(state));return obj


def record(controller,stream,prefix,offset,task,seed):
    # The evaluator owns the actual past; only predictions are requested here.
    before=encode(controller.state());prediction=controller.predict()
    assert before==encode(controller.state())
    word=stream[:prefix+offset]
    old=labels_for('old',word,seed);label=labels_for(task,word,seed)
    true_state=run(controller.new_program,word)
    p=prediction['p']
    return dict(t=offset,**prediction,old_truth=old,new_truth=label,
                old_mse=(prediction['old']-old)**2,
                new_brier=sum((p[y]-int(y==label))**2 for y in range(3)),
                certain_wrong=len(prediction['outputs'])==1 and prediction['outputs'][0]!=label,
                true_state_in_set=true_state in controller.new_states)


def run_one(life,task,prefix,policy,old,new):
    stream=life['streams'][str(prefix)]
    controller=Controller(old['program'],policy)
    for bit in stream[:prefix]: controller.feed(int(bit))
    preinstall_old_state=controller.old_state
    controller.install(new['program']);installation=controller.state();rows=[]
    for offset in range(65):
        if offset: controller.feed(int(stream[prefix+offset-1]))
        rows.append(record(controller,stream,prefix,offset,task,life['seed']))
        if offset==16: controller.observe_new_label(rows[-1]['new_truth'])
    state=controller.state()
    return dict(seed=life['seed'],task=task,prefix=prefix,policy=policy,
                preinstall_old_state=preinstall_old_state,installation=installation,rows=rows,
                final_state=state,final_state_sha256=hashlib.sha256(encode(state).encode()).hexdigest())


def run_all():
    lives=[make_life(seed) for seed in range(24)];calibrations=[];runs=[]
    for life in lives:
        words=life['calibration'];seed=life['seed']
        old=calibrate(OLD_PROGRAMS,words,[labels_for('old',w,seed) for w in words])
        old['task']='old';old['seed']=seed;calibrations.append(old)
        for task in TASKS:
            new=calibrate(PROGRAMS,words,[labels_for(task,w,seed) for w in words])
            new['task']=task;new['seed']=seed;calibrations.append(new)
            for prefix in PREFIXES:
                for policy in POLICIES: runs.append(run_one(life,task,prefix,policy,old,new))
    return dict(date='2026-09-22',source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                lives=lives,calibrations=calibrations,runs=runs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    data=run_all();path=RESULTS/'migration_learning.json'
    if args.write:
        path.write_text(encode(data)+'\n',encoding='utf-8')
        print(json.dumps(dict(status='written',runs=len(data['runs']),bytes=path.stat().st_size)))
    else:
        assert encode(data)==encode(json.loads(path.read_text(encoding='utf-8')))
        print(json.dumps(dict(status='exact_rerun_passed',runs=len(data['runs']))))


if __name__=='__main__': main()
