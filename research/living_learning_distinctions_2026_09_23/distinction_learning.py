"""Classical observation-table refinement with measured lifetime retention costs."""
from __future__ import annotations
import argparse
import copy
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RESULTS=HERE/'results'
PROTOCOL=ROOT/'docs/research/living_learning_distinctions_2026_09_23/PROTOCOL.md'
METHODS=('adaptive','rebuild','exhaustive_tests','fixed_tests')
FAMILIES=('cycle_reset','random_cycle','delay');SEEDS=range(24)
INTERNAL_LIMIT=20_000_000;ENV_LIMIT=100_000

def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def order(word):return len(word),word
def words(max_depth):
    for length in range(max_depth+1):
        for bits in itertools.product('01',repeat=length):yield ''.join(bits)

class Budget(Exception):pass

class Work:
    def __init__(self,state=None):
        self.used=0 if state is None else state['used'];self.categories={} if state is None else dict(state['categories'])
    def charge(self,n,category):
        if self.used+n>INTERNAL_LIMIT:raise Budget('internal')
        self.used+=n;self.categories[category]=self.categories.get(category,0)+n
    def export(self):return dict(used=self.used,categories=self.categories)

def output(machine,word,work=None):
    if work:work.charge(len(word)+1,'model_execution')
    state=machine['start']
    for bit in word:state=machine['transitions'][state][int(bit)]
    return machine['outputs'][state]

class Environment:
    """Oracle API returns a single output, never target state or structure."""
    def __init__(self,target,state=None):
        self._target=target
        self.cache={} if state is None else dict(state['cache'])
        self.units=0 if state is None else state['units']
        self.queries=[] if state is None else copy.deepcopy(state['queries'])
        self.passive_units=0 if state is None else state['passive_units']
    def query(self,word):
        if word not in self.cache:
            cost=len(word)+1
            if self.units+cost>ENV_LIMIT:raise Budget('environment')
            value=output(self._target,word);self.units+=cost;self.cache[word]=value
            self.queries.append(dict(word=word,value=value,cost=cost))
        return self.cache[word]
    def observe(self,word):
        value=output(self._target,word);cost=len(word)+1
        self.units+=cost;self.passive_units+=cost;self.cache[word]=value
        return value
    def export(self):return dict(cache=self.cache,units=self.units,queries=self.queries,passive_units=self.passive_units)

class Learner:
    def __init__(self,query,work,method,state=None):
        self.query=query;self.work=work;self.method=method
        self.S={''};self.E={''};self.known={};self.splits=[];self.counterexamples=[]
        self.model=dict(start=0,transitions=[[0,0]],outputs=[0])
        self.status='initial'
        if state:
            self.S=set(state['S']);self.E=set(state['E']);self.known=dict(state['known'])
            self.splits=copy.deepcopy(state['splits']);self.counterexamples=list(state['counterexamples'])
            self.model=copy.deepcopy(state['model']);self.status=state['status']
    def export(self):
        return dict(S=sorted(self.S,key=order),E=sorted(self.E,key=order),known=self.known,
                    splits=self.splits,counterexamples=self.counterexamples,model=self.model,status=self.status)
    def value(self,word):
        self.work.charge(1,'label_lookup')
        if word not in self.known:self.known[word]=self.query(word)
        return self.known[word]
    def observe(self,word,value):
        self.work.charge(len(word)+1,'observation_record')
        if word in self.known:assert self.known[word]==value
        self.known[word]=value
    def stabilize(self):
        while True:
            ss=sorted(self.S,key=order);ee=sorted(self.E,key=order)
            boundary=self.S|{s+a for s in self.S for a in '01'}
            rows={s:tuple(self.value(s+e) for e in ee) for s in sorted(boundary,key=order)}
            representatives={}
            for s in ss:
                self.work.charge(len(ee),'row_grouping')
                representatives.setdefault(rows[s],s)
            missing=None
            for s in sorted(boundary,key=order):
                self.work.charge(len(ee),'closure_compare')
                if rows[s] not in representatives:missing=s;break
            if missing is not None:
                self.work.charge(len(missing)+1,'access_storage');self.S.add(missing);continue
            witness=None
            for s in ss:
                representative=representatives[rows[s]]
                for bit in '01':
                    self.work.charge(len(ee),'consistency_compare')
                    if rows[s+bit]!=rows[representative+bit]:
                        index=next(i for i in range(len(ee)) if rows[s+bit][i]!=rows[representative+bit][i])
                        witness=(representative,s,bit,ee[index]);break
                if witness:break
            if witness:
                if self.method=='fixed_tests':self.status='fixed_inconsistent';return
                left,right,bit,suffix=witness;new=bit+suffix
                assert new not in self.E
                additions=[new] if self.method!='exhaustive_tests' else [e for e in words(len(new)) if e not in self.E]
                self.work.charge(sum(len(e)+1 for e in additions),'test_storage')
                self.splits.append(dict(left=left,right=right,action=bit,suffix=suffix,new_test=new,
                                        previous_tests=ee,old_row=list(rows[left]),
                                        outcomes=[self.value(left+new),self.value(right+new)],added=len(additions)))
                self.E.update(additions);continue
            reps=list(representatives.values());ids={rows[s]:i for i,s in enumerate(reps)}
            self.work.charge(len(reps)*(3+len(ee)),'graph_construction')
            model=dict(start=ids[rows['']],transitions=[[ids[rows[s+a]] for a in '01'] for s in reps],
                       outputs=[rows[s][0] for s in reps])
            bad=None
            for word in sorted(self.known,key=order):
                if output(model,word,self.work)!=self.known[word]:bad=word;break
            if bad is not None:
                prefixes={bad[:i] for i in range(len(bad)+1)}-self.S
                assert prefixes,'closed/consistent table failed on an S history'
                self.work.charge(sum(len(s)+1 for s in prefixes),'access_storage')
                self.S.update(prefixes);self.counterexamples.append(bad);continue
            self.model=model;self.status='consistent';return

def shuffled(machine,rng):
    n=len(machine['outputs']);permutation=list(range(n));rng.shuffle(permutation);flip=rng.randrange(2)
    transitions=[None]*n;outputs=[None]*n
    for i in range(n):
        transitions[permutation[i]]=[permutation[machine['transitions'][i][a^flip]] for a in range(2)]
        outputs[permutation[i]]=machine['outputs'][i]
    return dict(start=permutation[machine['start']],transitions=transitions,outputs=outputs)

def make_world(seed,family):
    rng=random.Random(998_513+seed*65537+FAMILIES.index(family)*9187)
    if family=='cycle_reset':
        n=5+seed%4;machine=dict(start=0,transitions=[[(i+1)%n,0] for i in range(n)],outputs=[int(i==n-1) for i in range(n)])
    elif family=='random_cycle':
        n=6+(seed%3)*4;labels=[rng.randrange(2) for _ in range(n)]
        if len(set(labels))==1:labels[-1]^=1
        machine=dict(start=0,transitions=[[(i+1)%n,rng.randrange(n)] for i in range(n)],outputs=labels)
    else:
        depth=3+seed%3;n=1<<depth
        machine=dict(start=0,transitions=[[(2*i+a)%n for a in range(2)] for i in range(n)],outputs=[i>>(depth-1) for i in range(n)])
    target=shuffled(machine,rng)
    stream_rng=random.Random(412_873+seed*131071)
    stream=[''.join(stream_rng.choice('01') for _ in range(stream_rng.randrange(25))) for _ in range(64)]
    tests=[]
    while len(tests)<128:
        word=''.join(rng.choice('01') for _ in range(rng.randrange(32,65)))
        if word not in tests:tests.append(word)
    return dict(seed=seed,family=family,target=target,stream=stream,tests=tests)

def equivalent(a,b):
    start=(a['start'],b['start']);queue=deque([(start,'')]);seen={start}
    while queue:
        (s,t),word=queue.popleft()
        if a['outputs'][s]!=b['outputs'][t]:return dict(equal=False,witness=word,pairs=len(seen))
        for bit in range(2):
            pair=(a['transitions'][s][bit],b['transitions'][t][bit])
            if pair not in seen:seen.add(pair);queue.append((pair,word+str(bit)))
    return dict(equal=True,witness=None,pairs=len(seen))

def evaluate(learner,world):
    model=learner.model
    return dict(exact=equivalent(model,world['target']),
                long_errors=sum(output(model,w)!=output(world['target'],w) for w in world['tests']),
                known_errors=sum(output(model,w)!=value for w,value in learner.known.items()),
                states=len(model['outputs']),tests=len(learner.E),max_test_length=max(map(len,learner.E)),
                accesses=len(learner.S),known=len(learner.known),splits=len(learner.splits),status=learner.status)

def run_life(world,method,resume=None):
    work=Work(None if resume is None else resume['work'])
    env=Environment(world['target'],None if resume is None else resume['environment'])
    learner=Learner(env.query,work,method,None if resume is None else resume['learner'])
    history=[] if resume is None else copy.deepcopy(resume['history'])
    operating_work=0 if resume is None else resume['operating_work']
    frozen=False if resume is None else resume['frozen'];reason=None if resume is None else resume['reason']
    records=[];initial=None
    if resume is None:
        try:learner.stabilize()
        except Budget as error:frozen=True;reason=str(error)
        initial=evaluate(learner,world)
    start=len(history)
    for block in range(start,16):
        observations=[];predictions=[]
        for word in world['stream'][4*block:4*block+4]:
            # Even after learning freezes, operating predictions and passive exposure continue.
            prediction=output(learner.model,word)
            operating_work+=len(word)+1
            value=env.observe(word);observations.append([word,value]);predictions.append(prediction)
        history.append(observations)
        if not frozen:
            old=copy.deepcopy(learner.model)
            try:
                if method=='rebuild':
                    candidate=Learner(env.query,work,method)
                    candidate.stabilize()
                    for batch in history:
                        for word,value in batch:candidate.observe(word,value)
                        candidate.stabilize()
                    learner=candidate
                else:
                    for word,value in observations:learner.observe(word,value)
                    learner.stabilize()
            except Budget as error:
                learner.model=old;frozen=True;reason=str(error)
        state=dict(work=copy.deepcopy(work.export()),environment=copy.deepcopy(env.export()),
                   learner=copy.deepcopy(learner.export()),history=copy.deepcopy(history),frozen=frozen,reason=reason,
                   operating_work=operating_work)
        records.append(dict(block=block,predictions=predictions,observations=observations,
                            prequential_errors=sum(prediction!=item[1] for prediction,item in zip(predictions,observations)),
                            evaluation=evaluate(learner,world),work=work.used,operating_work=operating_work,environment_units=env.units,
                            active_queries=len(env.queries),state_json_bytes=len(encode(state).encode()),state_after=state))
    return dict(seed=world['seed'],family=world['family'],method=method,initial=initial,records=records,
                final_state=state if records else resume)

def run_all():
    worlds=[make_world(seed,family) for seed in SEEDS for family in FAMILIES];runs=[]
    for world in worlds:
        for method in METHODS:runs.append(run_life(world,method))
        print(f"completed {world['seed']} {world['family']}",flush=True)
    return dict(date='2026-09-23',source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),worlds=worlds,runs=runs)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    data=run_all();path=RESULTS/'distinction_learning.json'
    if args.write:path.write_text(encode(data)+'\n',encoding='utf-8');print(encode(dict(status='written',runs=len(data['runs']),bytes=path.stat().st_size)))
    else:assert encode(data)==encode(json.loads(path.read_text(encoding='utf-8')));print('exact_rerun_passed')

if __name__=='__main__':main()
