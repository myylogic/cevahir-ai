"""Replay-sound migration between actually learned finite-state hypotheses."""
from __future__ import annotations
from collections import deque
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RESULTS=HERE/'results'
PREVIOUS=ROOT/'research/living_learning_distinctions_2026_09_23'
PROTOCOL=ROOT/'docs/research/living_learning_live_revision_2026_09_23/PROTOCOL.md'
SOURCE=PREVIOUS/'results/distinction_learning.json'
spec=importlib.util.spec_from_file_location('prior_distinctions',PREVIOUS/'distinction_learning.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
METHODS=('reset_start','all_states','parity_suffix8','model_relation','model_relation_suffix8','full_archive')
STRIDES=(32,256)
def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Meter:
    def __init__(self,state=None):self.categories={} if state is None else dict(state)
    def add(self,n,key):self.categories[key]=self.categories.get(key,0)+n
    @property
    def used(self):return sum(self.categories.values())

def walk(model,word,state=None):
    q=model['start'] if state is None else state
    for bit in word:q=model['transitions'][q][int(bit)]
    return q

def relation(old,new,meter,suffix=''):
    """Reachable synchronized parity/old/new triples, or parity/new pairs."""
    start=(0,old['start'],new['start']) if old is not None else (0,new['start'])
    queue=deque([start]);seen={start};components=3 if old is not None else 2
    while queue:
        item=queue.popleft()
        for bit in (0,1):
            meter.add(components,'relation_transitions')
            nxt=(item[0]^bit,old['transitions'][item[1]][bit],new['transitions'][item[2]][bit]) if old is not None else (item[0]^bit,new['transitions'][item[1]][bit])
            if nxt not in seen:seen.add(nxt);queue.append(nxt)
    size=len(seen)
    for bit in suffix:
        a=int(bit);updated=set()
        for item in seen:
            meter.add(components,'suffix_relation_transitions')
            nxt=(item[0]^a,old['transitions'][item[1]][a],new['transitions'][item[2]][a]) if old is not None else (item[0]^a,new['transitions'][item[1]][a])
            updated.add(nxt)
        seen=updated
    return seen,size

class Tracker:
    def __init__(self,method,model=None,state=None):
        self.method=method
        if state is None:
            self.model=copy.deepcopy(model);self.possible={model['start']};self.parity=0;self.position=0
            self.suffix='';self.history='';self.meter=Meter();self.revisions=0;self.peak_relation=0;self.peak_belief=1
        else:
            self.model=copy.deepcopy(state['model']);self.possible=set(state['possible']);self.parity=state['parity'];self.position=state['position']
            self.suffix=state['suffix'];self.history=state['history'];self.meter=Meter(state['categories'])
            self.revisions=state['revisions'];self.peak_relation=state['peak_relation'];self.peak_belief=state['peak_belief']
    def export(self):
        return dict(model=self.model,possible=sorted(self.possible),parity=self.parity,position=self.position,
                    suffix=self.suffix,history=self.history,categories=self.meter.categories,revisions=self.revisions,
                    peak_relation=self.peak_relation,peak_belief=self.peak_belief)
    def advance(self,bit):
        a=int(bit);self.meter.add(1,'parity_transitions');self.meter.add(len(self.possible),'live_model_transitions')
        self.parity^=a;self.possible={self.model['transitions'][q][a] for q in self.possible};self.position+=1
        if self.method in ('parity_suffix8','model_relation_suffix8'):
            self.suffix=(self.suffix+bit)[-8:];self.meter.add(1,'trace_bit_writes')
        elif self.method=='full_archive':self.history+=bit;self.meter.add(1,'trace_bit_writes')
    def install(self,new):
        before=self.meter.used
        if self.model==new:return dict(changed=False,work=0,relation_size=0)
        relation_size=0
        if self.method=='reset_start':
            possible={new['start']};self.meter.add(1,'initialization')
        elif self.method=='full_archive':
            possible={walk(new,self.history)};self.meter.add(len(self.history)+1,'archive_replay')
        elif self.method=='all_states':
            seen={new['start']};queue=deque(seen)
            while queue:
                q=queue.popleft()
                for a in (0,1):
                    self.meter.add(1,'reachable_transitions');nxt=new['transitions'][q][a]
                    if nxt not in seen:seen.add(nxt);queue.append(nxt)
            possible=seen;relation_size=len(seen)
        else:
            old=None if self.method=='parity_suffix8' else self.model
            suffix=self.suffix if self.method.endswith('suffix8') else ''
            pairs,relation_size=relation(old,new,self.meter,suffix)
            self.meter.add(len(pairs),'relation_filter')
            possible={item[-1] for item in pairs if item[0]==self.parity and (old is None or item[1] in self.possible)}
        assert possible
        self.model=copy.deepcopy(new);self.possible=possible;self.revisions+=1
        self.peak_relation=max(self.peak_relation,relation_size);self.peak_belief=max(self.peak_belief,len(possible))
        return dict(changed=True,work=self.meter.used-before,relation_size=relation_size)
    def predict(self):
        self.meter.add(len(self.possible),'output_reads')
        outputs={self.model['outputs'][q] for q in self.possible}
        return next(iter(outputs)) if len(outputs)==1 else None

def sources():
    data=json.loads(SOURCE.read_text(encoding='utf-8'));worlds={(w['seed'],w['family']):w for w in data['worlds']};result=[]
    for run in data['runs']:
        if run['method']!='adaptive':continue
        world=worlds[run['seed'],run['family']]
        work=prior.Work();environment=prior.Environment(world['target']);learner=prior.Learner(environment.query,work,'adaptive');learner.stabilize()
        result.append(dict(seed=run['seed'],family=run['family'],target=world['target'],initial=learner.model,
                           models=[r['state_after']['learner']['model'] for r in run['records']],
                           exact=[r['evaluation']['exact']['equal'] for r in run['records']],
                           calibration=[dict(work=r['work'],environment_units=r['environment_units'],json_bytes=r['state_json_bytes']) for r in run['records']]))
    return result

def stream_for(seed,stride):
    rng=random.Random(201_011+seed*131071+stride*37)
    return ''.join(rng.choice('01') for _ in range(16*stride+16))

def run_life(source,stride,method,resume=None,start_block=0):
    bits=stream_for(source['seed'],stride);tracker=Tracker(method,source['initial'],resume);records=[]
    for block in range(start_block,16):
        boundary=(block+1)*stride
        for bit in bits[tracker.position:boundary]:tracker.advance(bit)
        install=tracker.install(source['models'][block]);points=[]
        # Evaluator states are independent of the tracker and never passed into it.
        replay=walk(source['models'][block],bits[:boundary]);actual=walk(source['target'],bits[:boundary])
        for t in range(17):
            if t:
                bit=bits[tracker.position];tracker.advance(bit)
                replay=source['models'][block]['transitions'][replay][int(bit)]
                actual=source['target']['transitions'][actual][int(bit)]
            prediction=tracker.predict();truth=source['target']['outputs'][actual];replay_output=source['models'][block]['outputs'][replay]
            points.append(dict(t=t,size=len(tracker.possible),possible=sorted(tracker.possible),prediction=prediction,
                               truth=truth,replay_state=replay,replay_output=replay_output,replay_contained=replay in tracker.possible,
                               parity=tracker.parity,position=tracker.position))
        state=copy.deepcopy(tracker.export())
        records.append(dict(block=block,model_exact=source['exact'][block],install=install,points=points,
                            work=tracker.meter.used,calibration=source['calibration'][block],
                            state_json_bytes=len(encode(state).encode()),state_after=state))
    return dict(seed=source['seed'],family=source['family'],stride=stride,method=method,records=records,
                final_state=tracker.export())

def run_all():
    acquired=sources();runs=[]
    for source in acquired:
        for stride in STRIDES:
            for method in METHODS:runs.append(run_life(source,stride,method))
        print(f"completed {source['seed']} {source['family']}",flush=True)
    return dict(date='2026-09-23',source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                previous_source_sha256=sha(PREVIOUS/'distinction_learning.py'),previous_result_sha256=sha(SOURCE),sources=acquired,runs=runs)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    data=run_all();path=RESULTS/'live_revision.json'
    if args.write:path.write_text(encode(data)+'\n',encoding='utf-8');print(encode(dict(status='written',runs=len(data['runs']),bytes=path.stat().st_size)))
    else:assert encode(data)==encode(json.loads(path.read_text(encoding='utf-8')));print('exact_rerun_passed')

if __name__=='__main__':main()
