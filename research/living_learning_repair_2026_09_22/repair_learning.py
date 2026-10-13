"""Budgeted arithmetic repairs and patterns learned from self-found edit pairs."""
from __future__ import annotations
import argparse
import copy
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RESULTS=HERE/'results'
PROTOCOL=ROOT/'docs/research/living_learning_repair_2026_09_22/PROTOCOL.md'
OPS=('add','sub','mul');METHODS=('local_plain','local_ranked','bulk_ranked','learned_rules','rules_disabled_paid')
REGIMES=('compatible','changed','partial');SEEDS=range(24)
LIMIT=1_000_000;TASK_LIMIT=120_000
TEMPLATES=tuple((shape,op1,op2,perm) for shape in ('L','R') for op1 in OPS for op2 in OPS
                for perm in itertools.permutations(range(3)))
GRID=np.array(list(itertools.product(range(17),repeat=2)),dtype=np.int64)


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frozen(value):
    if isinstance(value,list):return tuple(frozen(x) for x in value)
    if isinstance(value,dict):return {k:frozen(v) for k,v in value.items()}
    return value
def branch(node):return isinstance(node,tuple) and node[0] in OPS
def hole(node):return isinstance(node,tuple) and node[0]=='hole'
@lru_cache(maxsize=100_000)
def nodes(node):return 1+nodes(node[1])+nodes(node[2]) if branch(node) else 1


class Exhausted(Exception):pass
class Work:
    def __init__(self,used=0):self.used=used;self.cap=LIMIT;self.categories={}
    def charge(self,amount,category):
        if self.used+amount>self.cap:raise Exhausted
        self.used+=amount;self.categories[category]=self.categories.get(category,0)+amount


def evaluate(node,xy):
    if node=='x':return xy[:,0]
    if node=='y':return xy[:,1]
    if isinstance(node,int):return np.full(len(xy),node%17,dtype=np.int64)
    a=evaluate(node[1],xy);b=evaluate(node[2],xy)
    if node[0]=='add':return (a+b)%17
    if node[0]=='sub':return (a-b)%17
    if node[0]=='mul':return (a*b)%17
    raise ValueError(node)


def evidence_loss(node,xy,labels,work):
    work.charge(nodes(node)*len(xy),'evaluation_nodes')
    difference=evaluate(node,xy)-labels
    return int(np.dot(difference,difference))


def paths(node,work,path=()):
    work.charge(1,'traversal_nodes');yield path,node
    if branch(node):
        yield from paths(node[1],work,path+(1,));yield from paths(node[2],work,path+(2,))


def boundaries(node):
    if not branch(node):return []
    result=[]
    if branch(node[1]):result.append((('L',node[0],node[1][0]),(node[1][1],node[1][2],node[2])))
    if branch(node[2]):result.append((('R',node[0],node[2][0]),(node[1],node[2][1],node[2][2])))
    return result


def rebuild(template,leaves):
    shape,outer,inner,perm=template;a,b,c=(leaves[i] for i in perm)
    return (outer,(inner,a,b),c) if shape=='L' else (outer,a,(inner,b,c))


def replace(node,path,value):
    if not path:return value
    side=path[0]
    return (node[0],replace(node[1],path[1:],value),node[2]) if side==1 else (node[0],node[1],replace(node[2],path[1:],value))


def local_proposals(program,priorities,ranked,work):
    sites=[]
    for path,node in paths(program,work):
        for signature,leaves in boundaries(node):sites.append((path,node,signature,leaves))
    emitted=set()
    if ranked:
        for entry in priorities:
            signature=frozen(entry['signature']);template=frozen(entry['template'])
            for path,node,sig,leaves in sites:
                if sig!=signature:continue
                key=(path,sig,template)
                if key in emitted:continue
                emitted.add(key);work.charge(nodes(program),'construction_nodes')
                sub=rebuild(template,leaves);candidate=replace(program,path,sub)
                yield candidate,dict(signature=sig,template=template,before=node,after=sub),False
    for path,node,signature,leaves in sites:
        for template in TEMPLATES:
            key=(path,signature,template)
            if key in emitted:continue
            work.charge(nodes(program),'construction_nodes')
            sub=rebuild(template,leaves);candidate=replace(program,path,sub)
            yield candidate,dict(signature=signature,template=template,before=node,after=sub),False


def generalize(first,second,memo,work):
    work.charge(1,'generalization_nodes')
    work.charge(nodes(first)+nodes(second),'generalization_equality_nodes')
    if first==second:
        return first
    if branch(first) and branch(second) and first[0]==second[0]:
        work.charge(1,'generalization_construction_nodes')
        return (first[0],generalize(first[1],second[1],memo,work),generalize(first[2],second[2],memo,work))
    key=(first,second)
    if key not in memo:
        work.charge(1,'generalization_construction_nodes');memo[key]=('hole',len(memo))
    return memo[key]


def holes(node):
    if hole(node):return {node[1]}
    return holes(node[1])|holes(node[2]) if branch(node) else set()


def derive_rules(edits,work):
    found={}
    for (i,a),(j,b) in itertools.combinations(enumerate(edits),2):
        memo={};left=generalize(a['before'],b['before'],memo,work);right=generalize(a['after'],b['after'],memo,work)
        work.charge(2*(nodes(left)+nodes(right)),'rule_validation_nodes')
        if hole(left) or left==right or not holes(right)<=holes(left):continue
        key=(left,right)
        if key not in found:found[key]=dict(left=left,right=right,provenance=[])
        found[key]['provenance'].append([i,j])
    return sorted(found.values(),key=lambda r:(-len(r['provenance']),nodes(r['left'])+nodes(r['right']),encode(r)))


def match(pattern,actual,bindings,work):
    work.charge(1,'matching_nodes')
    if hole(pattern):
        idx=pattern[1]
        if idx not in bindings:bindings[idx]=actual;return True
        work.charge(nodes(actual),'matching_equality_nodes');return bindings[idx]==actual
    if branch(pattern):
        return branch(actual) and pattern[0]==actual[0] and match(pattern[1],actual[1],bindings,work) and match(pattern[2],actual[2],bindings,work)
    return pattern==actual


def instantiate(pattern,bindings):
    if hole(pattern):return bindings[pattern[1]]
    return (pattern[0],instantiate(pattern[1],bindings),instantiate(pattern[2],bindings)) if branch(pattern) else pattern


def instantiated_size(pattern,bindings,work):
    work.charge(1,'instantiation_plan_nodes')
    if hole(pattern):return nodes(bindings[pattern[1]])
    if branch(pattern):return 1+instantiated_size(pattern[1],bindings,work)+instantiated_size(pattern[2],bindings,work)
    return 1


def bulk_rule(node,rule,work):
    work.charge(1,'traversal_nodes')
    if branch(node):
        work.charge(1,'construction_nodes');node=(node[0],bulk_rule(node[1],rule,work),bulk_rule(node[2],rule,work))
    bindings={}
    if match(rule['left'],node,bindings,work):
        amount=instantiated_size(rule['right'],bindings,work);work.charge(amount,'construction_nodes')
        node=instantiate(rule['right'],bindings)
    return node


def bulk_template(node,entry,work):
    work.charge(1,'traversal_nodes')
    if branch(node):
        work.charge(1,'construction_nodes');node=(node[0],bulk_template(node[1],entry,work),bulk_template(node[2],entry,work))
    for signature,leaves in boundaries(node):
        work.charge(1,'matching_nodes')
        if signature==frozen(entry['signature']):
            work.charge(nodes(node),'construction_nodes');node=rebuild(frozen(entry['template']),leaves);break
    return node


def proposals(program,state,method,work):
    if method=='learned_rules':
        for rule in state['rules']:
            yield bulk_rule(program,rule,work),None,True
            for path,node in paths(program,work):
                bindings={}
                if match(rule['left'],node,bindings,work):
                    amount=instantiated_size(rule['right'],bindings,work)
                    work.charge(nodes(program)-nodes(node)+amount,'construction_nodes')
                    candidate=replace(program,path,instantiate(rule['right'],bindings))
                    yield candidate,None,True
    elif method=='bulk_ranked':
        for entry in state['priorities']:yield bulk_template(program,entry,work),None,False
    yield from local_proposals(program,state['priorities'],method!='local_plain',work)


def solve(initial,xy,labels,state,method,work):
    start=work.used;traces=[];candidate_count=0;rule_proposals=0;rejected_rules=0
    current=initial;accepted=None;reason='no_improvement';seen=set()
    try:
        loss=evidence_loss(current,xy,labels,work);candidate_count+=1;seen.add(current)
        if loss==0:accepted=current;reason='initial_correct'
        if accepted is None:
            for library in state['library']:
                work.charge(nodes(library),'library_read_nodes')
                if library in seen:continue
                seen.add(library);value=evidence_loss(library,xy,labels,work);candidate_count+=1
                if value==0:accepted=library;reason='library';break
        for _ in range(8):
            if accepted is not None:break
            best=current;best_loss=loss;best_edit=None
            for candidate,edit,is_rule in proposals(current,state,method,work):
                if is_rule:rule_proposals+=1
                if candidate in seen:continue
                seen.add(candidate);value=evidence_loss(candidate,xy,labels,work);candidate_count+=1
                if is_rule and value:rejected_rules+=1
                if value==0:
                    accepted=candidate;reason='search';
                    if edit is not None:traces.append(edit)
                    break
                if value<best_loss:best=candidate;best_loss=value;best_edit=edit
            if accepted is not None:break
            if best==current:break
            current=best;loss=best_loss
            if best_edit is not None:traces.append(best_edit)
        else:reason='iteration_limit'
    except Exhausted:reason='budget'
    return dict(accepted=accepted,reason=reason,work=work.used-start,candidates=candidate_count,
                rule_proposals=rule_proposals,rejected_rule_candidates=rejected_rules,traces=traces)


def examples(rng,target):
    indices=rng.sample(range(len(GRID)),16);xy=GRID[indices]
    return dict(xy=xy.tolist(),labels=evaluate(target,xy).tolist())


def expression(rng,depth):
    if depth==0:return rng.choice(('x','y',rng.randrange(1,17)))
    return (rng.choice(OPS),expression(rng,depth-1),expression(rng,depth-1))


def joined(nodes_to_join):
    value=nodes_to_join[0]
    for node in nodes_to_join[1:]:value=('add',value,node)
    return value


def make_life(seed):
    rng=random.Random(602_117+seed*65537);sources=[]
    for index,c in enumerate(rng.sample(range(1,17),4)):
        u,w=('x','y') if index%2==0 else ('y','x')
        bad=('mul',('add',u,c),w);target=('add',('mul',u,w),c)
        sources.append(dict(initial=bad,target=target,**examples(rng,target)))
    targets=[]
    for index,(count,depth) in enumerate(zip((1,2,4,1,2,4,2,4),(1,2,3,1,2,3,2,3))):
        initial=[];correct={regime:[] for regime in REGIMES}
        for site in range(count):
            u=expression(rng,depth);v=expression(rng,depth);w=expression(rng,depth)
            bad=('mul',('add',u,v),w);fixed=('add',('mul',u,w),v)
            initial.append(bad);correct['compatible'].append(fixed)
            correct['changed'].append(('add',('add',u,v),w))
            correct['partial'].append(bad if site==count-1 else fixed)
        programs={regime:joined(value) for regime,value in correct.items()}
        indices=rng.sample(range(len(GRID)),16);xy=GRID[indices]
        targets.append(dict(index=index,sites=count,operand_depth=depth,initial=joined(initial),xy=xy.tolist(),
                            targets=programs,labels={regime:evaluate(value,xy).tolist() for regime,value in programs.items()}))
    return dict(seed=seed,sources=sources,targets=targets)


def acquire(life):
    work=Work();state=dict(library=[],priorities=[],rules=[]);edits=[];records=[]
    for source in life['sources']:
        result=solve(source['initial'],np.array(source['xy']),np.array(source['labels']),state,'local_plain',work)
        records.append(result)
        if result['accepted'] is not None:
            state['library'].append(result['accepted']);edits.extend(result['traces'])
    frequencies={}
    for edit in edits:
        key=(edit['signature'],edit['template']);frequencies[key]=frequencies.get(key,0)+1
        work.charge(nodes(edit['before'])+nodes(edit['after']),'trace_read_nodes')
    state['priorities']=[dict(signature=key[0],template=key[1],count=count) for key,count in
                          sorted(frequencies.items(),key=lambda pair:(-pair[1],encode(pair[0])))]
    state['source_edits']=copy.deepcopy(edits)
    state['source_examples']=[dict(xy=s['xy'],labels=s['labels']) for s in life['sources']]
    work.charge(sum(nodes(e['before'])+nodes(e['after']) for e in edits),'trace_storage_nodes')
    raw_cost=work.used;state['rules']=derive_rules(edits,work)
    return dict(state=state,source_records=records,source_edits=edits,base_cost=raw_cost,
                rule_acquisition_cost=work.used-raw_cost,categories=work.categories)


def run_targets(life,acquired,method,regime,start_index=0,state=None,used=None):
    if state is None:state=copy.deepcopy(acquired['state'])
    if used is None:used=acquired['base_cost']+(acquired['rule_acquisition_cost'] if method in ('learned_rules','rules_disabled_paid') else 0)
    if method not in ('learned_rules','rules_disabled_paid'):state['rules']=[]
    work=Work(used);records=[]
    for task in life['targets'][start_index:]:
        work.cap=min(LIMIT,work.used+TASK_LIMIT)
        result=solve(task['initial'],np.array(task['xy']),np.array(task['labels'][regime]),state,method,work)
        accepted=result['accepted'];exact=False;grid_errors=None
        if accepted is not None:
            grid_errors=int(np.count_nonzero(evaluate(accepted,GRID)!=evaluate(task['targets'][regime],GRID)))
            exact=grid_errors==0;state['library'].append(accepted)
        records.append(dict(index=task['index'],**result,full_field_correct=exact,grid_errors=grid_errors,
                            cumulative_work=work.used,state_after=copy.deepcopy(state)))
    return dict(seed=life['seed'],method=method,regime=regime,records=records,total_work=work.used,
                target_categories=work.categories,final_state=state,final_json_bytes=len(encode(state).encode()))


def run_all():
    lives=[make_life(seed) for seed in SEEDS];acquisitions=[];runs=[]
    for life in lives:
        acquired=acquire(life);acquisitions.append(dict(seed=life['seed'],**acquired))
        for regime in REGIMES:
            for method in METHODS:runs.append(run_targets(life,acquired,method,regime))
        print(f"completed life {life['seed']+1}/24",flush=True)
    return dict(date='2026-09-22',source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),lives=lives,acquisitions=acquisitions,runs=runs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    data=run_all();path=RESULTS/'repair_learning.json'
    if args.write:path.write_text(encode(data)+'\n',encoding='utf-8');print(encode(dict(status='written',runs=len(data['runs']),bytes=path.stat().st_size)))
    else:assert encode(data)==encode(json.loads(path.read_text(encoding='utf-8')));print('exact_rerun_passed')


if __name__=='__main__':main()
