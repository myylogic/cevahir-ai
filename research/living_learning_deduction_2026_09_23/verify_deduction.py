"""Independent truth enumeration, proof provenance and exact continuations."""
import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import deduction as d
from check_proof import check_unsat, checker_selftest


def satisfies(clauses, values):
    return all(any(values[abs(x)] == (x > 0) for x in clause) for clause in clauses)


def enumerate_cnf(clauses, n, assumptions=()):
    for mask in range(1 << n):
        values = {v: bool(mask & (1 << (v - 1))) for v in range(1, n + 1)}
        if all(values[abs(x)] == (x > 0) for x in assumptions) and satisfies(clauses, values):
            return True
    return False


def independent_projection(groups):
    # Compute each module's public truth set independently before intersecting;
    # private blocks must be disjoint for this factorization to be valid.
    previous_private = set()
    allowed = set(range(64))
    for group in groups:
        private = sorted({abs(x) for clause in group for x in clause if abs(x) > 6})
        assert not (previous_private & set(private))
        previous_private.update(private)
        module = set()
        for public_mask in range(64):
            values = {v: bool(public_mask & (1 << (v - 1))) for v in range(1, 7)}
            for private_mask in range(1 << len(private)):
                values.update({v: bool(private_mask & (1 << i)) for i, v in enumerate(private)})
                if satisfies(group, values):
                    module.add(public_mask)
                    break
        allowed &= module
    return allowed


def clause_true(clause, mask):
    return any(bool(mask & (1 << (abs(x) - 1))) == (x > 0) for x in clause)


def randomized_solver():
    rng = random.Random(210023)
    sat_count = unsat_count = 0
    for _ in range(1000):
        clauses = {}
        for j in range(rng.randrange(20)):
            clauses[str(j)] = [v if rng.randrange(2) else -v for v in rng.sample(range(1, 7), rng.randrange(5))]
        assumptions = [v if rng.randrange(2) else -v for v in rng.sample(range(1, 7), rng.randrange(5))]
        expected = enumerate_cnf(list(clauses.values()), 6, assumptions)
        result = d.solve(clauses, assumptions)
        assert result['sat'] == expected
        if expected:
            values = {abs(x): x > 0 for x in result['witness']}
            assert satisfies(list(clauses.values()), values)
            assert all(values[abs(x)] == (x > 0) for x in assumptions)
            sat_count += 1
        else:
            check_unsat(clauses, assumptions, result['proof'])
            unsat_count += 1
    return {'cases': 1000, 'sat_witnesses': sat_count, 'unsat_proofs': unsat_count}


def verify_recorded(data):
    checks = accepted = active_checked = invalid_blind_uses = invalid_blind_new = 0
    replay_pairs = 0
    for run in data['runs']:
        world = run['world']
        masks = [independent_projection(gs) for gs in world['gadgets']]
        for index, truth in enumerate(run['truth']):
            expected = {sum(1 << (abs(x) - 1) for x in row if x > 0) for row in truth}
            assert masks[index] == expected
            flat = {d.cid(c): c for group in world['gadgets'][index] for c in group}
            assert flat == world['bases'][index]
        for method, track in run['tracks'].items():
            stored = {}
            last_base = None
            cumulative = {k: 0 for k in ('service', 'acquisition', 'checking', 'dependencies', 'writes')}
            for row in track['records']:
                bi = row['base']
                engine = row['engine']
                q = row['assumptions']
                truth = any(all(bool(mask & (1 << (abs(x) - 1))) == (x > 0) for x in q) for mask in masks[bi])
                assert row['target'] == truth and row['correct'] == (engine['sat'] == truth)
                if method != 'blind':
                    assert row['correct']
                checks += 1
                if method not in ('scoped', 'blind', 'erase', 'replay'):
                    continue
                if method == 'erase' and last_base is not None and bi != last_base:
                    stored.clear()
                last_base = bi
                base = world['bases'][bi]
                active_ids = [i for i, l in stored.items() if method == 'blind' or all(p in base for p in l['dependencies'])]
                assert active_ids == engine['active']
                for i in active_ids:
                    sound = all(clause_true(stored[i]['clause'], mask) for mask in masks[bi])
                    active_checked += 1
                    if method != 'blind':
                        assert sound
                    else:
                        invalid_blind_uses += not sound
                new = engine.get('new')
                if new:
                    result = check_unsat(new['used_clauses'], new['assumptions'], new['proof'])
                    assert set(result['dependencies']) == set(new['used_clauses'])
                    assert new['clause'] == sorted([-x for x in new['assumptions']], key=lambda x: (abs(x), x))
                    flattened = set()
                    expected_parents = {}
                    for ident, clause in new['used_clauses'].items():
                        if ident in base:
                            assert base[ident] == clause
                            flattened.add(ident)
                        else:
                            assert ident in active_ids and stored[ident]['clause'] == clause
                            flattened.update(stored[ident]['dependencies'])
                            expected_parents[ident] = stored[ident]['dependencies']
                    assert set(new['dependencies']) == flattened
                    assert new['parent_dependencies'] == expected_parents
                    sound = all(clause_true(new['clause'], mask) for mask in masks[bi])
                    if method != 'blind':
                        assert sound and flattened <= set(base)
                    else:
                        invalid_blind_new += not sound
                    assert new['id'] not in stored
                    stored[new['id']] = new
                    accepted += 1
                assert len(stored) == engine['stored_lemmas']
                if method != 'replay':
                    for key in cumulative:
                        cumulative[key] += engine['costs'][key]
            if method in ('scoped', 'blind', 'erase'):
                assert list(stored.values()) == track['final']['lemmas']
                assert cumulative == track['final']['costs']
        a = run['tracks']['scoped']
        b = run['tracks']['replay']
        assert a['final'] == b['final']
        prefix_cost = {k: 0 for k in a['final']['costs']}
        for left, right in zip(a['records'], b['records']):
            for key in ('sat', 'service_work', 'active', 'new', 'stored_lemmas'):
                assert left['engine'][key] == right['engine'][key]
            for key in prefix_cost:
                prefix_cost[key] += left['engine']['costs'][key]
                assert right['engine']['costs'][key] == prefix_cost[key]
            replay_pairs += 1
    return {'answers_checked': checks, 'accepted_certificates_checked_including_controls': accepted,
            'active_lemma_truth_checks_including_controls': active_checked,
            'blind_invalid_active_lemma_uses': invalid_blind_uses, 'blind_invalid_new_lemmas': invalid_blind_new,
            'exact_replay_pairs': replay_pairs}


def resume_checks():
    records = []
    with tempfile.TemporaryDirectory(prefix='cevahir_round21_') as folder:
        for seed in (0, 7, 23):
            w = d.world(seed)
            for method in ('scoped', 'blind', 'erase'):
                for cut in (49, 73):
                    learner = d.Learner(method)
                    for q in w['queries'][:cut]:
                        learner.query(w['bases'][q['base']], q['assumptions'])
                    source = Path(folder) / 'in.json'
                    dest = Path(folder) / 'out.json'
                    source.write_text(d.encode({'state': learner.snapshot(), 'bases': w['bases'], 'tail': w['queries'][cut:]}), encoding='utf-8')
                    expected_rows = [learner.query(w['bases'][q['base']], q['assumptions']) for q in w['queries'][cut:]]
                    child = subprocess.run([sys.executable, str(d.HERE / 'deduction.py'), '--resume', str(source), str(dest)], capture_output=True, text=True, check=True)
                    actual = json.loads(dest.read_text(encoding='utf-8'))
                    assert actual == {'records': expected_rows, 'state': learner.snapshot()}
                    records.append({'seed': seed, 'method': method, 'cut': cut, 'exact_equal': True})
    return records


def preservation():
    baseline = json.loads((d.RESULTS / 'preservation_baseline.json').read_text(encoding='utf-8'))
    for p, expected in baseline['files'].items():
        assert d.sha(d.ROOT / p) == expected, p
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT).decode().strip()
    assert head == baseline['base_head']
    return {'files': len(baseline['files']), 'all_unchanged': True, 'head': head}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    data = json.loads((d.RESULTS / 'deduction.json').read_text(encoding='utf-8'))
    assert data['source_sha256'] == d.sha(d.__file__)
    assert data['checker_sha256'] == d.sha(d.HERE / 'check_proof.py')
    assert data['protocol_sha256'] == d.sha(d.PROTOCOL)
    out = {'checker_selftest': checker_selftest(), 'randomized_solver': randomized_solver(),
           'recorded': verify_recorded(data), 'fresh_process_resume': resume_checks(),
           'preservation': preservation()}
    if args.rerun:
        child = subprocess.run([sys.executable, str(d.HERE / 'deduction.py')], capture_output=True, text=True)
        out['full_rerun'] = {'returncode': child.returncode, 'stdout': child.stdout.strip(), 'stderr': child.stderr.strip()}
        assert child.returncode == 0, out['full_rerun']
    (d.RESULTS / 'verification.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
