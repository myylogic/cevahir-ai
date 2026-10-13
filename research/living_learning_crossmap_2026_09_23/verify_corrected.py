"""Independent posterior enumerator, history/probability/cost and snapshot audit."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
import argparse
import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import corrected as c
import analyze_corrected as a
import tree_audit


def preserves():
    baseline = json.loads((c.RESULTS / 'preservation_baseline.json').read_text(encoding='utf-8'))
    for name, expected in baseline['files'].items(): assert c.sha(c.ROOT / name) == expected, name
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT).decode().strip()
    assert head == baseline['base_head']
    archive = c.RESULTS / 'invalid_initial_run'
    for name, value in json.loads((archive / 'archive_manifest.json').read_text()).items(): assert c.sha(archive / name) == value
    return dict(files=len(baseline['files']), all_unchanged=True, head=head, invalid_initial_archive_unchanged=True)


def normalize_world(w):
    w = copy.deepcopy(w)
    for name in ('x_next', 'y_next', 'x_group', 'y_group', 'true_map'):
        w[name] = {int(k): int(v) for k, v in w[name].items()}
    return w


def completions(bridges):
    """Backtrack only free coordinates, unlike engine permutations/filter."""
    fixed = dict(bridges)
    assert len(fixed) == len(bridges) and len(set(fixed.values())) == len(fixed)
    result = []
    def walk(prefix):
        i = len(prefix)
        if i == 6:
            result.append(prefix)
            return
        for y in ([fixed[i]] if i in fixed else range(6)):
            if y in prefix: continue
            if i not in fixed and y in fixed.values(): continue
            walk(prefix + [y])
    walk([])
    assert len(result) == math.factorial(6 - len(bridges))
    return result


def predict_from_records(source, mapping, q):
    records = {(d, kind, x): val for d, kind, x, val in source}
    kind, x = q['kind'], q['input']
    if kind == 'xy': return mapping[records['x', 'next', x]]
    if kind == 'yx': return mapping.index(records['y', 'next', x])
    return 2 * records['x', 'group', x] + records['y', 'group', mapping[x]]


def distribution(source, mappings, q):
    hist = Counter(predict_from_records(source, m, q) for m in mappings)
    return tuple(Fraction(hist[i], len(mappings)) for i in range(4 if q['kind'] == 'cat' else 6))


def probabilities(counts):
    return None if counts is None else tuple(Fraction(x, sum(counts)) for x in counts)


def validate_state(state, source, bridges):
    assert state['source'] == source and state['bridges'] == bridges
    known, inverse = [-1] * 6, [-1] * 6
    for x, y in bridges:
        assert known[x] == -1 and inverse[y] == -1
        known[x], inverse[y] = y, x
    assert state['known'] == known and state['inverse'] == inverse
    if state['method'] in ('persistent', 'disabled', 'point'):
        assert state['maps'] == completions(bridges)
    if state['tables'] is not None:
        for d, kind, x, val in source: assert state['tables'][f'{d}_{kind}'][x] == val
    # This validates artifacts; restore() is not a hostile-state security boundary.


def check_scores(row, probs):
    if probs is None:
        assert row['scores'] == {'brier': None, 'expected_argmax_accuracy': None}
        assert row['abstained'] and not row['correct'] and not row['confident_error']
        return
    y = row['target']
    brier = sum((p - int(i == y)) ** 2 for i, p in enumerate(probs))
    top = max(probs)
    acc = Fraction(int(probs[y] == top), probs.count(top))
    assert Fraction(*row['scores']['brier']) == brier
    assert Fraction(*row['scores']['expected_argmax_accuracy']) == acc
    certain = sum(p != 0 for p in probs) == 1
    assert row['abstained'] == (not certain)
    assert row['correct'] == (certain and probs[y] == 1)
    assert row['confident_error'] == (certain and probs[y] == 0)


def inspect(data):
    answers = state_checks = score_checks = distribution_pairs = local_checks = 0
    original = json.loads((c.RESULTS / 'invalid_initial_run/crossmap.json').read_text(encoding='utf-8'))
    for run, prior in zip(data['runs'], original['runs']):
        assert run['world'] == prior['world']
        w = normalize_world(run['world'])
        source = c.observation(w)
        assert len(source) == 24
        xof, yof = w['x_of_latent'], w['y_of_latent']
        actual_map = [yof[xof.index(x)] for x in range(6)]
        assert actual_map == [w['true_map'][i] for i in range(6)]
        assert all(actual_map[x] == y for x, y in w['bridges'])
        for method, track in run['tracks'].items():
            assert len(track['records']) == 24 and len(track['snapshots']) == 4
            learner = c.Learner(method, source, actual_map if method == 'oracle' else None)
            assert track['setup'] == {'cost': learner.cost, 'write_bytes': learner.write_bytes}
            for row, packet in zip(track['records'], c.packets(w)):
                q = packet['query']
                got = c.step(learner, packet)
                assert got == row['engine']
                bridges = w['bridges'][:w['bridge_before'][q['phase']]]
                maps = completions(bridges)
                assert actual_map in maps
                truth = predict_from_records(source, actual_map, q)
                assert row['target'] == truth
                kind, x = q['kind'], q['input']
                if kind == 'xy': observed = w['x_next'][x] in dict(bridges)
                elif kind == 'yx': observed = w['y_next'][x] in dict(bridges).values()
                else: observed = x in dict(bridges)
                assert row['queried_pair_observed'] == observed
                enabled = method != 'separate' and (method != 'disabled' or q['phase'] == 3)
                if not enabled: expected = None
                elif method == 'point': expected = distribution(source, maps[:1], q)
                elif method == 'oracle': expected = distribution(source, [actual_map], q)
                else: expected = distribution(source, maps, q)
                assert probabilities(got['response']['counts']) == expected
                check_scores(row, expected)
                answers += 1
                score_checks += int(expected is not None)
                if (row['t'] + 1) % 6 == 0:
                    snap = track['snapshots'][row['t'] // 6]
                    assert snap == learner.snapshot()
                    validate_state(snap, source, bridges)
                    state_checks += 1
            assert learner.snapshot() == track['final']
            for d, kind, x, value in source:
                assert track['final_local_probe'][f'{d}_{kind}'][x] == value
                local_checks += 1
            a.life_work(track)
        for left, right in [('persistent', 'recompute'), ('persistent', 'completion'), ('recompute', 'completion')]:
            for l, r in zip(run['tracks'][left]['records'], run['tracks'][right]['records']):
                assert probabilities(l['engine']['response']['counts']) == probabilities(r['engine']['response']['counts'])
                distribution_pairs += 1
    return dict(independent_answer_checks=answers, scored_probability_checks=score_checks,
                state_snapshots_checked=state_checks, exact_distribution_pairs=distribution_pairs,
                local_response_checks=local_checks, cost_telescope_lives=24 * len(c.METHODS),
                original_worlds_unchanged=24, same_code_record_replay=True)


def renaming_checks(data):
    output = []
    for index in range(12):
        w = normalize_world(data['runs'][index]['world'])
        rng = random.Random(90220 + index)
        px, py = rng.sample(range(6), 6), rng.sample(range(6), 6)
        source2 = []
        for domain, kind, x, value in c.observation(w):
            p = px if domain == 'x' else py
            source2.append([domain, kind, p[x], p[value] if kind == 'next' else value])
        transformed = [None] * 6
        for x in range(6): transformed[px[x]] = py[w['true_map'][x]]
        for method in ('separate', 'persistent', 'recompute', 'completion', 'disabled', 'oracle'):
            learner = c.Learner(method, source2, transformed if method == 'oracle' else None)
            for row, packet in zip(data['runs'][index]['tracks'][method]['records'], c.packets(w)):
                q = dict(packet['query'])
                q['input'] = (py if q['kind'] == 'yx' else px)[q['input']]
                translated = {'new_pairs': [[px[x], py[y]] for x, y in packet['new_pairs']], 'query': q}
                answer = c.step(learner, translated)['response']
                oldp = probabilities(row['engine']['response']['counts'])
                newp = probabilities(answer['counts'])
                if oldp is None: assert newp is None
                else:
                    outmap = py if q['kind'] == 'xy' else px if q['kind'] == 'yx' else list(range(4))
                    assert all(oldp[i] == newp[outmap[i]] for i in range(len(oldp)))
            output.append(dict(seed=index, method=method, invariant_probabilities=True))
    return output


def tamper_checks(data):
    run = data['runs'][0]
    w = normalize_world(run['world'])
    state = run['tracks']['persistent']['snapshots'][0]
    tests = []
    for label in ('candidate', 'bridge'):
        bad = copy.deepcopy(state)
        if label == 'candidate': bad['maps'][0][0] = (bad['maps'][0][0] + 1) % 6
        else: bad['bridges'][0][1] = (bad['bridges'][0][1] + 1) % 6
        try: validate_state(bad, c.observation(w), w['bridges'][:2])
        except AssertionError: tests.append(label)
        else: raise AssertionError('corruption accepted by artifact verifier')
    return dict(rejected=tests, scope='Verifier checks against recorded evidence; not a hardened runtime restore.')


def structural_checks():
    result = tree_audit.audit()
    independent_best, max_nodes, count = 100, 0, 0
    def mask(t): return (1 << t) if isinstance(t, int) else mask(t[0]) | mask(t[1])
    def cover(t, group):
        m = mask(t)
        if not (m & group): return 0
        if m & group == m: return 1
        assert not isinstance(t, int)
        return cover(t[0], group) + cover(t[1], group)
    for t in tree_audit.trees(tuple(range(6))):
        costs = [cover(t, group) for group in (7, 56, 21, 42)]
        independent_best = min(independent_best, sum(costs))
        max_nodes = max(max_nodes, costs.count(1))
        count += 1
    assert count == 945 and independent_best == 6 and max_nodes == 2
    result['independent_recursive_cover_checks'] = count * 4
    source = c.observation(c.world(0))
    maps = completions([])
    assert distribution(source, maps, {'kind': 'xy', 'input': 0}) == (Fraction(1, 6),) * 6
    result['no_bridge_abstract_family'] = dict(maps=720, uniform_xy_output='1/6',
        scope='Arbitrary maps with fixed local tables, not the planted-generator posterior.')
    return result


def resume_checks(data):
    checks = []
    with tempfile.TemporaryDirectory(prefix='cevahir_crossmap_') as folder:
        folder = Path(folder)
        for run in data['runs'][:3]:
            w = normalize_world(run['world'])
            packets = c.packets(w)
            for method in c.METHODS:
                for cut in (6, 12, 18, 24):
                    learner = c.Learner.restore(run['tracks'][method]['snapshots'][cut // 6 - 1])
                    src, dst = folder / 'in.json', folder / 'out.json'
                    src.write_text(c.encode({'state': learner.snapshot(), 'tail': packets[cut:]}), encoding='utf-8')
                    expected = [c.step(learner, p) for p in packets[cut:]]
                    subprocess.run([sys.executable, str(c.HERE / 'corrected.py'), '--resume', str(src), str(dst)],
                                   capture_output=True, text=True, check=True)
                    assert json.loads(dst.read_text(encoding='utf-8')) == {'records': expected, 'final': learner.snapshot()}
                    assert learner.snapshot() == run['tracks'][method]['final']
                    checks.append(dict(seed=run['seed'], method=method, cut=cut, exact=True, tail_length=24 - cut))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    data = json.loads((c.RESULTS / 'crossmap_corrected.json').read_text(encoding='utf-8'))
    assert data['source_sha256'] == c.sha(c.__file__)
    assert data['amendment_sha256'] == c.sha(c.DOCS / 'AMENDMENT.md')
    assert data['archived_generator_sha256'] == c.sha(c.RESULTS / 'invalid_initial_run/crossmap.py')
    out = dict(recorded=inspect(data), renaming=renaming_checks(data), tampering=tamper_checks(data),
               structural=structural_checks(), fresh_process_resume=resume_checks(data), preservation=preserves())
    if args.rerun:
        proc = subprocess.run([sys.executable, str(c.HERE / 'corrected.py')], capture_output=True, text=True)
        assert proc.returncode == 0 and proc.stdout.strip() == 'exact_rerun_passed', (proc.stdout, proc.stderr)
        out['full_rerun'] = True
    out['source_sha256'] = c.sha(c.__file__)
    out['verifier_sha256'] = c.sha(__file__)
    (c.RESULTS / 'verification_corrected.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(c.encode({k: v for k, v in out.items() if k not in ('fresh_process_resume', 'renaming', 'structural')}))


if __name__ == '__main__': main()
