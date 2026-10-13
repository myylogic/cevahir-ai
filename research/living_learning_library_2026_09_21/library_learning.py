"""Finite-field program acquisition with experience-conditioned search proposals.

Default: recompute and check the recorded result. --write explicitly records.
Teacher code and whole-domain targets never enter synthesize() or mine_library().
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
FIELD = 101
DOMAIN = tuple(range(FIELD))
PRIMITIVES = ((0,), (1,), (2,), (3,))
METHODS = ("primitive_only", "frequency_prior", "learned_library", "expanded_proposals",
           "whole_program_cache", "static_all_chunks", "random_library", "oracle_motifs")
HISTORIES = (0, 6, 12)
BUDGETS = (512, 2048)
ALL_CHUNKS = tuple(p for n in (2, 3) for p in product(range(4), repeat=n))


def step(values, op):
    if op == 0: return tuple((v+1) % FIELD for v in values)
    if op == 1: return tuple(2*v % FIELD for v in values)
    if op == 2: return tuple(v*v % FIELD for v in values)
    if op == 3: return tuple(-v % FIELD for v in values)
    raise ValueError(op)


def execute(program, inputs):
    values = tuple(inputs)
    for op in program:
        values = step(values, op)
    return values


def synthesize(train_x, train_y, accept_x, accept_y, tokens, budget, expanded=False):
    """BFS of observed-output signatures. Cost includes duplicate proposals.

    A macro is charged for each primitive; cached prefix values are shared by all
    methods. First fitting candidate is checked once on separate acceptance data.
    No full-domain evaluation or teacher representation is accessible here.
    """
    train_x, train_y = tuple(train_x), tuple(train_y)
    accept_x, accept_y = tuple(accept_x), tuple(accept_y)
    tokens = tuple(tuple(t) for t in tokens)
    costs = dict(search_vector_steps=0, acceptance_vector_steps=0,
                 candidates=0, duplicate_signatures=0, unique_signatures=1,
                 max_queue=1)
    queue = deque([(train_x, (), 0)])
    visited = {train_x}

    def finish(status, program=None, depth=None):
        return dict(status=status, program=list(program) if program is not None else None,
                    token_depth=depth, primitive_length=len(program) if program is not None else None,
                    costs=costs.copy(), charged_vector_steps=costs['search_vector_steps']+costs['acceptance_vector_steps'])

    def validate(program, depth):
        if costs['search_vector_steps'] + len(program) > budget:
            return finish('budget_before_acceptance')
        costs['acceptance_vector_steps'] += len(program)
        status = 'accepted' if execute(program, accept_x) == accept_y else 'acceptance_rejected'
        return finish(status, program, depth)

    if train_x == train_y:
        return validate((), 0)
    while queue:
        parent, program, depth = queue.popleft()
        for token in tokens:
            if costs['search_vector_steps'] + len(token) > budget:
                return finish('search_budget_exhausted')
            costs['candidates'] += 1
            costs['search_vector_steps'] += len(token)
            # The expansion control adds no instruction to a runtime dictionary.
            if expanded:
                child = parent
                for primitive in token:
                    child = step(child, primitive)
            else:
                child = execute(token, parent)
            candidate = program + token
            if child == train_y:
                return validate(candidate, depth+1)
            if child in visited:
                costs['duplicate_signatures'] += 1
                continue
            visited.add(child)
            costs['unique_signatures'] += 1
            queue.append((child, candidate, depth+1))
            costs['max_queue'] = max(costs['max_queue'], len(queue))
    return finish('observed_signature_space_exhausted')


def mine_library(programs):
    counts = Counter()
    scanned = 0
    for raw in programs:
        program = tuple(raw)
        found = set()
        for length in (2, 3):
            for start in range(max(0, len(program)-length+1)):
                scanned += 1
                found.add(program[start:start+length])
        counts.update(found)
    eligible = [p for p, count in counts.items() if count >= 2 and count*(len(p)-1)-len(p) > 0]
    eligible.sort(key=lambda p: (-(counts[p]*(len(p)-1)-len(p)), -len(p), p))
    selected = eligible[:4]
    return selected, dict(substring_windows_scanned=scanned,
                          primitive_occurrences_scanned=sum(map(len, programs)),
                          unique_substrings=len(counts),
                          selected=[dict(program=list(p), support=counts[p],
                                         score=counts[p]*(len(p)-1)-len(p)) for p in selected])


def build_state(programs):
    library, mining = mine_library(programs)
    return dict(programs=[list(p) for p in programs], library=[list(p) for p in library], mining=mining)


def token_policy(state, method, seed, oracle):
    library = [tuple(p) for p in state['library']]
    programs = [tuple(p) for p in state['programs']]
    if method == 'primitive_only': return PRIMITIVES
    if method == 'frequency_prior':
        counts = Counter(op for p in programs for op in p)
        return tuple((op,) for op in sorted(range(4), key=lambda op: (-counts[op], op)))
    if method in ('learned_library', 'expanded_proposals'):
        return tuple(library) + PRIMITIVES
    if method == 'whole_program_cache':
        counts = Counter(programs)
        chosen = sorted(counts, key=lambda p: (-counts[p], len(p), p))[:4]
        return tuple(p for p in chosen if len(p) > 1) + PRIMITIVES
    if method == 'static_all_chunks': return ALL_CHUNKS + PRIMITIVES
    if method == 'random_library':
        rng = random.Random(77100 + seed*101 + len(programs))
        chosen = []
        for p in library:
            choices = [q for q in ALL_CHUNKS if len(q) == len(p) and q not in chosen]
            chosen.append(rng.choice(choices))
        return tuple(chosen) + PRIMITIVES
    if method == 'oracle_motifs': return tuple(tuple(p) for p in oracle) + PRIMITIVES
    raise ValueError(method)


def canonical_motifs():
    shorter = {execute(p, DOMAIN) for length in range(3) for p in product(range(4), repeat=length)}
    unique = {}
    for p in product(range(4), repeat=3):
        signature = execute(p, DOMAIN)
        if signature not in shorter: unique.setdefault(signature, p)
    return list(unique.values())


def unique_programs(programs, forbidden=()):
    seen = set(forbidden)
    result = []
    for p in programs:
        signature = execute(p, DOMAIN)
        if signature not in seen:
            seen.add(signature)
            result.append(p)
    return result


def warm_candidates(motifs):
    return unique_programs([*motifs, *(m+(op,) for m in motifs for op in range(4)),
                            *((op,)+m for m in motifs for op in range(4))])


def future_candidates(motifs, forbidden):
    bare = unique_programs([a+b for a in motifs for b in motifs], forbidden)
    seen = set(forbidden) | {execute(p, DOMAIN) for p in bare}
    extra = unique_programs([p for a in motifs for b in motifs for op in range(4)
                             for p in (a+b+(op,), (op,)+a+b)], seen)
    return bare, extra


def task(program, seed, name, regime):
    rng = random.Random(seed)
    samples = rng.sample(range(FIELD), 16)
    tx, ax = samples[:8], samples[8:]
    return dict(name=name, regime=regime, teacher=list(program), train_x=tx,
                train_y=list(execute(program, tx)), accept_x=ax,
                accept_y=list(execute(program, ax)))


def make_world(seed):
    rng = random.Random(1192100+seed)
    motifs = canonical_motifs()
    # Rejection is only a structural uniqueness/size condition, never learner performance.
    for attempt in range(10000):
        home = rng.sample(motifs, 2)
        warm = warm_candidates(home)
        if len(warm) < 12: continue
        rest = warm[2:]; rng.shuffle(rest)
        warm = warm[:2] + rest[:10]
        forbidden = {execute(p, DOMAIN) for p in warm}
        bare, extra = future_candidates(home, forbidden)
        if len(bare) < 4 or len(extra) < 4: continue
        away = rng.sample([m for m in motifs if m not in home], 2)
        related = bare[:4]; rng.shuffle(extra); related += extra[:4]
        forbidden |= {execute(p, DOMAIN) for p in related}
        away_bare, away_extra = future_candidates(away, forbidden)
        if len(away_bare) < 4 or len(away_extra) < 4: continue
        rng.shuffle(away_extra)
        shifted = away_bare[:4] + away_extra[:4]
        break
    else: raise AssertionError('Could not construct structurally distinct tasks')
    initial = [task(p, 100000+seed*100+i, f'warm{i}', 'warm') for i,p in enumerate(warm)]
    probes = [task(p, 200000+seed*100+i, f'{regime}{i%8}', regime)
              for i,(regime,p) in enumerate([*(('related',p) for p in related), *(('shifted',p) for p in shifted)])]
    return dict(seed=seed, home_motifs=[list(p) for p in home], shifted_motifs=[list(p) for p in away],
                structural_attempts=attempt+1, initial=initial, probes=probes)


def solve_task(t, tokens, budget, expanded=False):
    return synthesize(t['train_x'], t['train_y'], t['accept_x'], t['accept_y'], tokens, budget, expanded)


def evaluate(result, t):
    if result['status'] != 'accepted':
        return dict(exact=False, unseen_correct=0, unseen_total=FIELD-16)
    prediction = execute(result['program'], DOMAIN)
    target = execute(t['teacher'], DOMAIN)
    unseen = set(DOMAIN) - set(t['train_x']) - set(t['accept_x'])
    return dict(exact=prediction == target,
                unseen_correct=sum(prediction[x] == target[x] for x in unseen), unseen_total=len(unseen))


def acquire_history(world):
    programs, records = [], []
    states = {0: build_state(programs)}
    for i,t in enumerate(world['initial'], 1):
        result = solve_task(t, PRIMITIVES, 32768)
        if result['status'] == 'accepted': programs.append(result['program'])
        records.append(dict(task=t['name'], result=result, evaluation=evaluate(result,t)))
        if i in HISTORIES: states[i] = build_state(programs)
    return states, records


def summarize(rows):
    groups = []
    for history in HISTORIES:
        for budget in BUDGETS:
            for regime in ('related','shifted'):
                for method in METHODS:
                    subset=[r for r in rows if (r['history'],r['budget'],r['regime'],r['method'])==(history,budget,regime,method)]
                    seed_rates=[sum(r['exact'] for r in subset if r['seed']==s)/8 for s in range(12)]
                    groups.append(dict(history=history,budget=budget,regime=regime,method=method,conditions=len(subset),
                        exact_count=sum(r['exact'] for r in subset),
                        accepted_count=sum(r['status']=='accepted' for r in subset),
                        false_accept_count=sum(r['status']=='accepted' and not r['exact'] for r in subset),
                        exact_rate=statistics.fmean(seed_rates), seed_exact_rates=seed_rates,
                        mean_charged_vector_steps=statistics.fmean(r['charged_vector_steps'] for r in subset),
                        mean_candidates=statistics.fmean(r['costs']['candidates'] for r in subset),
                        max_queue=max(r['costs']['max_queue'] for r in subset),
                        statuses=dict(Counter(r['status'] for r in subset))))
    return groups


def run():
    rows, lives = [], []
    equality_checks = 0
    for seed in range(12):
        world = make_world(seed)
        states, acquisition = acquire_history(world)
        lives.append(dict(world=world, acquisition=acquisition, states=states,
                          state_json_bytes={n:len(json.dumps(s,sort_keys=True,separators=(',',':')).encode()) for n,s in states.items()}))
        for history,state in states.items():
            for budget in BUDGETS:
                for t in world['probes']:
                    comparisons = {}
                    for method in METHODS:
                        # Oracle receives training-world motifs, including on shifted tasks.
                        # It never receives the shifted task's hidden motif pair.
                        tokens = token_policy(state, method, seed, world['home_motifs'])
                        result = solve_task(t,tokens,budget,method=='expanded_proposals')
                        comparisons[method] = result
                        rows.append(dict(seed=seed,history=history,budget=budget,regime=t['regime'],task=t['name'],method=method,
                                         **result, **evaluate(result,t)))
                    assert comparisons['learned_library'] == comparisons['expanded_proposals']
                    equality_checks += 1
                    if history == 0:
                        assert comparisons['learned_library'] == comparisons['primitive_only'] == comparisons['frequency_prior']
        print(f'life {seed}: complete',file=sys.stderr,flush=True)
    return dict(date='2026-09-21',experiment='experience_conditioned_program_library_and_search',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment=dict(python=platform.python_version(),dependencies='standard library only'),
                protocol=dict(lives=12,histories=list(HISTORIES),budgets=list(BUDGETS),methods=list(METHODS),
                              field=FIELD,train_examples=8,acceptance_examples=8,warmup_budget=32768),
                lives=lives,runs=rows,aggregate=summarize(rows),
                matched_expansion_equal_conditions=equality_checks,
                scope='Finite given DSL and task boundaries; same 12 worlds repeated; not new computability, general living-learning, or a DreamCoder/OOPS/Stitch implementation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--resume-input',type=Path)
    parser.add_argument('--resume-output',type=Path)
    args=parser.parse_args()
    if args.resume_input:
        p=json.loads(args.resume_input.read_text(encoding='utf-8'))
        tokens=token_policy(p['state'],p['method'],p['seed'],p['oracle'])
        result=solve_task(p['task'],tokens,p['budget'],p['method']=='expanded_proposals')
        args.resume_output.write_text(json.dumps(result,sort_keys=True)+'\n',encoding='utf-8')
        return
    result=run();path=HERE/'results/library_learning.json'
    # JSON roundtrip normalizes integer dictionary keys for exact comparison.
    normalized=json.loads(json.dumps(result))
    if args.write:path.write_text(json.dumps(normalized,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:assert normalized==json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps(dict(status='passed',mode='record' if args.write else 'check',conditions=len(result['runs']),
                          lives=len(result['lives']),expansion_equal=result['matched_expansion_equal_conditions'])))


if __name__=='__main__':main()
