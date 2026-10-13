"""Round 21: checked deductions, changing premises, exact compilation controls.

No external solver, labels or learned model. Deliberately small, fully auditable.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
PROTOCOL = ROOT / 'docs/research/living_learning_deduction_2026_09_23/PROTOCOL.md'
METHODS = ['fresh', 'scoped', 'blind', 'erase', 'replay', 'full_table']
from check_proof import check_unsat


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cid(clause):
    return 'b:' + hashlib.sha256(encode(clause).encode()).hexdigest()[:24]


def solve(db, assumptions):
    """DPLL, negative branch first; count literal/header and state operations.

    Assumptions contain distinct variables. Proof leaves cite only db clauses.
    SAT witness completes unassigned variables with False.
    """
    work = 0
    initial = {abs(x): x > 0 for x in assumptions}
    assert len(initial) == len(assumptions)
    work += len(assumptions)

    def visit(assignment):
        nonlocal work
        assignment = assignment.copy()
        work += len(assignment) + 1
        units = []
        while True:
            shortest = None
            unit = None
            for ident, clause in db.items():
                work += 1
                pending = []
                satisfied = False
                for literal in clause:
                    work += 1
                    value = assignment.get(abs(literal))
                    if value is None:
                        pending.append(literal)
                    elif value == (literal > 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                if not pending:
                    work += 1
                    return False, {'units': units, 'conflict': ident}
                if len(pending) == 1:
                    unit = (pending[0], ident)
                    break
                if shortest is None or len(pending) < len(shortest):
                    shortest = pending
            if unit is not None:
                literal, ident = unit
                assignment[abs(literal)] = literal > 0
                units.append([literal, ident])
                work += 3
                continue
            if shortest is None:
                work += len(assignment)
                return True, assignment
            var = abs(shortest[0])
            work += 1
            assignment[var] = False
            sat, left = visit(assignment)
            if sat:
                return True, left
            assignment[var] = True
            sat, right = visit(assignment)
            if sat:
                return True, right
            work += 3
            return False, {'units': units, 'branch': var, 'left': left, 'right': right}

    sat, payload = visit(initial)
    if sat:
        variables = sorted({abs(x) for clause in db.values() for x in clause} | set(initial))
        payload = [v if payload.get(v, False) else -v for v in variables]
        work += len(variables)
    return {'sat': sat, 'witness' if sat else 'proof': payload, 'work': work}


def world(seed):
    rng = random.Random(730000 + seed)
    planted = {v: bool(rng.randrange(2)) for v in range(1, 7)}
    pairs = rng.sample(list(itertools.combinations(range(1, 7), 2)), 4)
    guards = []
    for pair in pairs:
        signs = [bool(rng.randrange(2)) for _ in pair]
        if all(sign != planted[v] for v, sign in zip(pair, signs)):
            signs[0] = planted[pair[0]]
        guards.append([v if sign else -v for v, sign in zip(pair, signs)])

    def gadgets(changed):
        result = []
        for g, guard in enumerate(guards):
            guard = [-x for x in guard] if changed and g == 0 else guard
            var = lambda pigeon, hole: 7 + g * 6 + pigeon * 2 + hole
            private = [[var(p, 0), var(p, 1)] for p in range(3)]
            private += [[-var(p, h), -var(q, h)] for h in range(2)
                        for p, q in itertools.combinations(range(3), 2)]
            result.append([sorted(guard + clause, key=lambda x: (abs(x), x)) for clause in private])
        return result

    bases = []
    blocks = []
    for changed in (False, True):
        gs = gadgets(changed)
        clauses = [c for group in gs for c in group]
        # Identical fixed position permutation in both versions; identity is content.
        order = list(range(len(clauses)))
        random.Random(910000 + seed).shuffle(order)
        bases.append(dict(sorted((cid(clauses[i]), clauses[i]) for i in order)))
        blocks.append(gs)
    queries = []
    for t in range(96):
        query = [v if rng.randrange(2) else -v for v in sorted(rng.sample(range(1, 7), 3))]
        queries.append({'base': int(48 <= t < 72), 'assumptions': query})
    return {'seed': seed, 'bases': bases, 'gadgets': blocks, 'queries': queries}


def projected_truth(gadgets):
    """Independent reference: enumerate private assignments, no guard shortcut."""
    truth = []
    for bits in itertools.product((False, True), repeat=6):
        public = dict(zip(range(1, 7), bits))
        good = True
        for group in gadgets:
            private = sorted({abs(x) for c in group for x in c if abs(x) > 6})
            exists = False
            for private_bits in itertools.product((False, True), repeat=len(private)):
                full = public | dict(zip(private, private_bits))
                if all(any(full[abs(x)] == (x > 0) for x in c) for c in group):
                    exists = True
                    break
            if not exists:
                good = False
                break
        if good:
            truth.append([v if public[v] else -v for v in range(1, 7)])
    return truth


def reference(truth, assumptions):
    return any(all(x in row for x in assumptions) for row in truth)


class Learner:
    def __init__(self, method):
        assert method in ('scoped', 'blind', 'erase')
        self.method = method
        self.lemmas = []
        self.previous = None
        self.costs = {x: 0 for x in ('service', 'acquisition', 'checking', 'dependencies', 'writes')}

    def snapshot(self):
        return json.loads(encode(self.__dict__))

    @classmethod
    def restore(cls, state):
        learner = cls(state['method'])
        learner.__dict__.update(json.loads(encode(state)))
        return learner

    def query(self, base, assumptions):
        before = self.costs.copy()
        signature = sorted(base)
        self.costs['dependencies'] += len(base)
        if self.method == 'erase' and self.previous is not None and self.previous != signature:
            self.costs['writes'] += len(self.lemmas)
            self.lemmas = []
        self.previous = signature
        active = []
        for lemma in self.lemmas:
            self.costs['dependencies'] += len(lemma['dependencies'])
            if self.method == 'blind' or all(dep in base for dep in lemma['dependencies']):
                active.append(lemma)
        db = {lemma['id']: lemma['clause'] for lemma in active}
        db.update(base)
        outcome = solve(db, assumptions)
        self.costs['service'] += outcome['work']
        new = None
        if not outcome['sat']:
            negated = {-x for x in assumptions}
            covered = False
            for lemma in active:
                self.costs['acquisition'] += 1 + len(lemma['clause'])
                if set(lemma['clause']) <= negated:
                    covered = True
                    break
            if not covered:
                core = list(assumptions)
                final = outcome
                for literal in list(core):
                    candidate = [x for x in core if x != literal]
                    result = solve(db, candidate)
                    self.costs['acquisition'] += result['work']
                    if not result['sat']:
                        core = candidate
                        final = result
                checked = check_unsat(db, core, final['proof'])
                self.costs['checking'] += checked['work']
                parents = {lemma['id']: lemma for lemma in active}
                deps = set()
                for ident in checked['dependencies']:
                    if ident in base:
                        deps.add(ident)
                    else:
                        deps.update(parents[ident]['dependencies'])
                clause = sorted([-x for x in core], key=lambda x: (abs(x), x))
                ident = 'l:' + hashlib.sha256(encode([clause, sorted(deps)]).encode()).hexdigest()[:24]
                new = {'id': ident, 'clause': clause, 'dependencies': sorted(deps),
                       'assumptions': core, 'proof': final['proof'],
                       'used_clauses': {i: db[i] for i in checked['dependencies']},
                       'parent_dependencies': {i: parents[i]['dependencies'] for i in checked['dependencies'] if i in parents}}
                # Charge serialized proof, provenance, clause and metadata bytes.
                self.costs['writes'] += len(encode(new).encode())
                self.costs['dependencies'] += sum(len(v) for v in new['parent_dependencies'].values())
                self.lemmas.append(new)
        delta = {k: self.costs[k] - before[k] for k in self.costs}
        return {'sat': outcome['sat'], 'service_work': outcome['work'], 'costs': delta,
                'active': [v['id'] for v in active], 'new': new,
                'stored_lemmas': len(self.lemmas),
                'state_bytes': len(encode(self.snapshot()).encode())}


def run_world(w):
    truth = [projected_truth(gs) for gs in w['gadgets']]
    tracks = {}
    for method in METHODS:
        learner = Learner(method) if method in ('scoped', 'blind', 'erase') else None
        records = []
        previous = None
        table = None
        seen = set()
        for t, item in enumerate(w['queries']):
            base = w['bases'][item['base']]
            q = item['assumptions']
            if learner is not None:
                row = learner.query(base, q)
            elif method == 'fresh':
                result = solve(base, q)
                row = {'sat': result['sat'], 'service_work': result['work'],
                       'costs': {'service': result['work']}, 'state_bytes': 0}
            elif method == 'replay':
                rebuilt = Learner('scoped')
                for old in w['queries'][:t + 1]:
                    row = rebuilt.query(w['bases'][old['base']], old['assumptions'])
                assert row == tracks['scoped']['records'][t]['engine']
                row = json.loads(encode(row))
                row['costs'] = rebuilt.costs.copy()
                # Chronological history is required for this control, including copies of changed bases.
                row['state_bytes'] += len(encode({'bases': w['bases'], 'queries': w['queries'][:t + 1]}).encode())
                row['costs']['history_writes'] = len(encode(item).encode())
                if t in (0, 48):
                    row['costs']['history_writes'] += len(encode(base).encode())
            else:
                preparation = 0
                if previous != item['base']:
                    table = []
                    for bits in itertools.product((False, True), repeat=6):
                        full = [v if bit else -v for v, bit in zip(range(1, 7), bits)]
                        result = solve(base, full)
                        preparation += result['work'] + 7
                        if result['sat']:
                            table.append(full)
                    previous = item['base']
                cost = 1
                sat = False
                for full in table:
                    cost += 1
                    compatible = True
                    for literal in q:
                        cost += 1
                        if full[abs(literal) - 1] != literal:
                            compatible = False
                            break
                    if compatible:
                        sat = True
                        break
                row = {'sat': sat, 'service_work': cost,
                       'costs': {'service': cost, 'compilation': preparation},
                       'state_bytes': len(encode(table).encode())}
            key = tuple(q)
            target = reference(truth[item['base']], q)
            records.append({'t': t, 'phase': t // 24, 'base': item['base'],
                            'assumptions': q, 'novel_query': key not in seen,
                            'target': target, 'correct': row['sat'] == target, 'engine': row})
            seen.add(key)
        final = learner.snapshot() if learner is not None else (rebuilt.snapshot() if method == 'replay' else None)
        tracks[method] = {'records': records, 'final': final}
    return {'seed': w['seed'], 'world': w, 'truth': truth, 'tracks': tracks}


def experiment():
    return {'round': 21, 'date': '2026-09-23', 'python': platform.python_version(),
            'source_sha256': sha(__file__), 'checker_sha256': sha(HERE / 'check_proof.py'),
            'protocol_sha256': sha(PROTOCOL), 'methods': METHODS,
            'runs': [run_world(world(seed)) for seed in range(24)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--resume', nargs=2)
    args = parser.parse_args()
    if args.resume:
        source, dest = map(Path, args.resume)
        data = json.loads(source.read_text(encoding='utf-8'))
        learner = Learner.restore(data['state'])
        records = [learner.query(data['bases'][q['base']], q['assumptions']) for q in data['tail']]
        dest.write_text(encode({'records': records, 'state': learner.snapshot()}), encoding='utf-8')
        return
    data = experiment()
    path = RESULTS / 'deduction.json'
    if args.write:
        path.write_text(encode(data) + '\n', encoding='utf-8')
        print('wrote', path, path.stat().st_size, flush=True)
    else:
        assert data == json.loads(path.read_text(encoding='utf-8'))
        print('exact_rerun_passed', flush=True)


if __name__ == '__main__':
    main()
