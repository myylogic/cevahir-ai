"""Acquire edits jointly through measured internal effects and verify each repair."""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RESULTS=HERE/'results'
PREVIOUS=ROOT/'research/living_learning_repair_2026_09_22'
sys.path.insert(0,str(PREVIOUS))
import repair_learning as r
PROTOCOL=ROOT/'docs/research/living_learning_interactions_2026_09_22/PROTOCOL.md'
SOURCE_RESULT=PREVIOUS/'results/repair_learning.json'
METHODS=('bulk_all','greedy_single','first_order','second_order','adaptive_reset','adaptive_persistent','exact_subset')
TASK_LIMIT=1_000_000
MASKS=sorted(range(256),key=lambda mask:(mask.bit_count(),mask))
SEEDS=range(24)
encode=r.encode;sha=r.sha;GRID=r.GRID


class Work:
    def __init__(self):self.used=0;self.categories={}
    def charge(self,amount,category):
        if self.used+amount>TASK_LIMIT:raise r.Exhausted
        self.used+=amount;self.categories[category]=self.categories.get(category,0)+amount


class Solved(Exception):
    def __init__(self,mask):self.mask=mask


def replace_many(node,replacements,path=()):
    if path in replacements:return replacements[path]
    if not r.branch(node):return node
    return (node[0],replace_many(node[1],replacements,path+(1,)),replace_many(node[2],replacements,path+(2,)))


def locate(program,rule,work):
    result=[]
    for path,node in r.paths(program,work):
        bindings={}
        if r.match(rule['left'],node,bindings,work):
            work.charge(r.nodes(node),'replacement_preparation_nodes')
            result.append((path,r.instantiate(rule['right'],bindings)))
    return result


class Probe:
    def __init__(self,program,xy,labels,rule,work):
        self.program=program;self.xy=np.array(xy,dtype=np.int64);self.labels=np.array(labels,dtype=np.int64)
        self.work=work;self.sites=locate(program,rule,work)
        assert len(self.sites)==8
        assert not any(p!=q and q[:len(p)]==p for p,_ in self.sites for q,_ in self.sites)
        self.cache={};self.probe_masks=[];self.model_masks=0;self.prediction_mismatches=0
        self.coefficients={};self.fit_order=0

    def candidate(self,mask):
        if not mask:return self.program
        self.work.charge(r.nodes(self.program),'construction_nodes')
        return replace_many(self.program,{path:value for i,(path,value) in enumerate(self.sites) if mask&(1<<i)})

    def query(self,mask):
        if mask not in self.cache:
            program=self.candidate(mask)
            self.work.charge(r.nodes(program)*len(self.xy),'evaluation_nodes')
            self.cache[mask]=r.evaluate(program,self.xy);self.probe_masks.append(mask)
        output=self.cache[mask]
        if np.array_equal(output,self.labels):raise Solved(mask)
        return output

    def build_effects(self,order):
        base=self.query(0);self.coefficients[0]=base
        if self.fit_order<1:
            for i in range(8):
                values=self.query(1<<i);self.work.charge(len(self.xy),'effect_arithmetic')
                self.coefficients[1<<i]=(values-base)%17
            self.fit_order=1
        if order==2 and self.fit_order<2:
            for i,j in itertools.combinations(range(8),2):
                mask=(1<<i)|(1<<j);values=self.query(mask)
                self.work.charge(3*len(self.xy),'effect_arithmetic')
                self.coefficients[mask]=(values-self.cache[1<<i]-self.cache[1<<j]+base)%17
            self.fit_order=2

    def modeled(self,mask):
        keys=[key for key in self.coefficients if key&mask==key]
        self.work.charge(len(self.xy)*len(keys),'model_vector_terms')
        values=np.zeros(len(self.xy),dtype=np.int64)
        for key in keys:values=(values+self.coefficients[key])%17
        self.model_masks+=1
        return values

    def effect_search(self,order):
        self.build_effects(order)
        for mask in MASKS:
            if np.array_equal(self.modeled(mask),self.labels):
                actual=self.query(mask)
                if not np.array_equal(actual,self.labels):self.prediction_mismatches+=1


def solve(program,xy,labels,rule,method,start_order):
    work=Work();probe=None;accepted=None;reason='no_verified_candidate';expanded=False;final_order=start_order
    try:
        probe=Probe(program,xy,labels,rule,work)
        baseline=probe.query(0)
        if method=='bulk_all':probe.query(255)
        elif method=='exact_subset':
            for mask in MASKS:probe.query(mask)
        elif method=='greedy_single':
            current=0;loss=int(np.count_nonzero(baseline!=probe.labels))
            for _ in range(8):
                best=current;best_loss=loss
                for i in range(8):
                    mask=current^(1<<i);value=int(np.count_nonzero(probe.query(mask)!=probe.labels))
                    if value<best_loss:best=mask;best_loss=value
                if best==current:break
                current,loss=best,best_loss
        else:
            order=2 if method=='second_order' else start_order if method=='adaptive_persistent' else 1
            probe.effect_search(order)
            if method in ('adaptive_reset','adaptive_persistent') and order==1:
                expanded=True
                if method=='adaptive_persistent':final_order=2
                probe.effect_search(2)
    except Solved as success:accepted=success.mask;reason='evidence_match'
    except r.Exhausted:reason='budget'
    return dict(accepted_mask=accepted,reason=reason,work=work.used,categories=work.categories,
                probe_masks=probe.probe_masks if probe else [],model_masks=probe.model_masks if probe else 0,
                model_prediction_mismatches=probe.prediction_mismatches if probe else 0,
                expanded=expanded,order_before=start_order,order_after=final_order,
                site_count=len(probe.sites) if probe else 0)


def combine(parts,degree):
    terms=[]
    for start in range(0,len(parts),degree):
        value=parts[start]
        for part in parts[start+1:start+degree]:value=('mul',value,part)
        terms.append(value)
    value=terms[0]
    for term in terms[1:]:value=('add',value,term)
    return value


def affine(rng):return ('add',('mul',rng.randrange(1,17),rng.choice(('x','y'))),rng.randrange(17))


def make_life(seed):
    rng=random.Random(827_119+seed*196613);tasks=[]
    for index in range(16):
        degree=1 if index<4 or 8<=index<12 else 2 if index<8 else 3
        before=[];after=[]
        for i in range(8):
            u,v,w=affine(rng),affine(rng),affine(rng)
            if index==4:w=0
            before.append(('mul',('add',u,v),w));after.append(('add',('mul',u,w),v))
        chosen=set(range(4)) if index==4 else set(rng.sample(range(8),4))
        initial=combine(before,degree);target=combine([after[i] if i in chosen else before[i] for i in range(8)],degree)
        indices=rng.sample(range(289),16);xy=GRID[indices]
        tasks.append(dict(index=index,degree=degree,masked_pair_witness=index==4,initial=initial,target=target,
                          planted_mask=sum(1<<i for i in chosen),xy=xy.tolist(),labels=r.evaluate(target,xy).tolist()))
    return dict(seed=seed,tasks=tasks)


def source_states():
    data=json.loads(SOURCE_RESULT.read_text(encoding='utf-8'));sources=[]
    for acquired in data['acquisitions']:
        rules=[rule for rule in acquired['state']['rules'] if len(r.holes(r.frozen(rule['left'])))==3]
        assert len(rules)==1
        rule=copy.deepcopy(rules[0]);rule['left']=r.frozen(rule['left']);rule['right']=r.frozen(rule['right'])
        sources.append(dict(seed=acquired['seed'],rule=rule,acquisition_cost=acquired['base_cost']+acquired['rule_acquisition_cost']))
    return sources


def run_life(life,source,method,start=0,state=None):
    if state is None:state=dict(order=1,rule=copy.deepcopy(source['rule']),library=[],total_work=source['acquisition_cost'],expanded_tasks=[])
    else:state=copy.deepcopy(state)
    records=[]
    for task in life['tasks'][start:]:
        # The solver receives no hidden target or mask.
        result=solve(task['initial'],task['xy'],task['labels'],state['rule'],method,state['order'])
        state['order']=result['order_after'];state['total_work']+=result['work']
        if result['expanded']:state['expanded_tasks'].append(task['index'])
        accepted=result['accepted_mask'];full_correct=False;errors=None;planted_equal=False
        if accepted is not None:
            sites=locate(task['initial'],state['rule'],Work())
            program=replace_many(task['initial'],{path:value for i,(path,value) in enumerate(sites) if accepted&(1<<i)})
            errors=int(np.count_nonzero(r.evaluate(program,GRID)!=r.evaluate(task['target'],GRID)))
            full_correct=errors==0;planted_equal=accepted==task['planted_mask']
            state['library'].append(dict(program=program,examples=dict(xy=task['xy'],labels=task['labels'])))
        records.append(dict(index=task['index'],**result,full_field_correct=full_correct,grid_errors=errors,
                            planted_mask_equal=planted_equal,cumulative_work=state['total_work'],state_after=copy.deepcopy(state)))
    return dict(seed=life['seed'],method=method,records=records,total_work=state['total_work'],final_state=state,
                final_json_bytes=len(encode(state).encode()))


def run_all():
    lives=[make_life(seed) for seed in SEEDS];sources=source_states();runs=[]
    for life,source in zip(lives,sources):
        assert life['seed']==source['seed']
        for method in METHODS:runs.append(run_life(life,source,method))
        print(f"completed life {life['seed']+1}/24",flush=True)
    return dict(date='2026-09-22',source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                prior_source_sha256=sha(PREVIOUS/'repair_learning.py'),source_result_sha256=sha(SOURCE_RESULT),
                lives=lives,sources=sources,runs=runs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    data=run_all();path=RESULTS/'interaction_learning.json'
    if args.write:path.write_text(encode(data)+'\n',encoding='utf-8');print(encode(dict(status='written',runs=len(data['runs']),bytes=path.stat().st_size)))
    else:assert encode(data)==encode(json.loads(path.read_text(encoding='utf-8')));print('exact_rerun_passed')


if __name__=='__main__':main()
