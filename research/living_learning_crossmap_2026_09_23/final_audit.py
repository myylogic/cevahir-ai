"""Final audit for corrected round 22; invalid runs remain explicitly archived."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
import corrected as c
import analyze_corrected as a
import verify_corrected as v
import tree_audit
import model_scope_audit


def read(name): return json.loads((c.RESULTS / name).read_text(encoding='utf-8'))


def main():
    raw, summary, verify = read('crossmap_corrected.json'), read('summary_corrected.json'), read('verification_corrected.json')
    assert raw['source_sha256'] == verify['source_sha256'] == c.sha(c.__file__)
    assert verify['verifier_sha256'] == c.sha(v.__file__)
    assert raw['amendment_sha256'] == c.sha(c.DOCS / 'AMENDMENT.md')
    assert raw['archived_generator_sha256'] == c.sha(c.RESULTS / 'invalid_initial_run/crossmap.py')
    assert summary == a.summarize(raw)
    assert verify['full_rerun'] is True
    expected = dict(independent_answer_checks=4032, scored_probability_checks=3024,
                    state_snapshots_checked=672, exact_distribution_pairs=1728,
                    local_response_checks=4032, cost_telescope_lives=168,
                    original_worlds_unchanged=24, same_code_record_replay=True)
    assert verify['recorded'] == expected
    assert len(verify['renaming']) == 72 and all(r['invariant_probabilities'] for r in verify['renaming'])
    assert len(verify['fresh_process_resume']) == 84 and all(r['exact'] for r in verify['fresh_process_resume'])
    assert sum(r['tail_length'] > 0 for r in verify['fresh_process_resume']) == 63
    assert verify['tampering']['rejected'] == ['candidate', 'bridge']
    assert read('tree_audit.json') == c.clone(tree_audit.audit())
    assert verify['structural'] == c.clone(v.structural_checks())
    assert read('model_scope_audit.json') == c.clone(model_scope_audit.audit())
    methods = {r['method']: r for r in summary['methods']}
    expected_counts = {'separate': (0, 576, 0, 394), 'persistent': (374, 202, 0, 6612),
                       'recompute': (374, 202, 0, 130170), 'completion': (374, 202, 0, 694),
                       'point': (450, 0, 126, 6009), 'disabled': (144, 432, 0, 5840),
                       'oracle': (576, 0, 0, 625)}
    for method, expected_row in expected_counts.items():
        row = methods[method]
        assert tuple(row[k] for k in ('correct', 'abstained', 'confident_errors', 'life_work_mean')) == expected_row
    assert methods['persistent']['new_queries']['n'] == 432
    assert methods['persistent']['queried_pair_unobserved']['correct'] == 38
    assert methods['persistent']['queried_pair_unobserved']['n'] == 240
    dominated = 0
    for run in raw['runs']:
        base = run['tracks']['persistent']['final']['cost']
        compact = run['tracks']['completion']['final']['cost']
        assert all(compact[k] <= base[k] for k in c.CATEGORIES)
        assert sum(compact.values()) < sum(base.values())
        dominated += 1
    # The second archived intermediate artifact set also remains byte-identical.
    archive = c.RESULTS / 'correction_audit_v1'
    for name, digest in json.loads((archive / 'manifest.json').read_text()).items(): assert c.sha(archive / name) == digest
    # Literal report correction concerning round 21 is backed by its old summary.
    previous = c.ROOT / 'research/living_learning_deduction_2026_09_23/results/summary.json'
    old = json.loads(previous.read_text(encoding='utf-8'))
    assert next(m for m in old['methods'] if m['method'] == 'scoped')['new_lemmas'] == 185
    links = 0
    for p in [c.HERE / 'README.md', *c.DOCS.glob('*.md')]:
        text = re.sub(r'`[^`\n]*`', '', p.read_text(encoding='utf-8'))
        for href in re.findall(r'\[[^\[\]\n]+\]\(([^)]+)\)', text):
            if re.match(r'^[A-Za-z][\w+.-]*:', href) or href.startswith('#'): continue
            target = (p.parent / unquote(href.split('#', 1)[0])).resolve()
            assert target.exists() or target == c.RESULTS / 'final_audit.json', (p, href)
            links += 1
    files = [*c.HERE.glob('*.py'), c.HERE / 'README.md', *c.DOCS.glob('*.md'),
             *(p for p in c.RESULTS.rglob('*') if p.is_file() and p.suffix in ('.json', '.py', '.md')
               and p.name != 'final_audit.json' and '__pycache__' not in p.parts)]
    out = dict(status='passed', date='2026-09-23', evidence_status='post-result correction; not independent replication',
               checks=expected, fresh_process_loads=84, nonempty_continuations=63, renamed_paths=72,
               complete_tree_enumeration=945, independent_tree_category_covers=3780,
               completion_scalar_category_dominance_lives=dominated,
               original_and_intermediate_invalid_artifacts_preserved=True,
               full_exact_rerun=True, prior_preservation=v.preserves(), local_links_checked=links,
               prior_round21_erratum='185 scoped lemmas is total across 24 lives, not per-life mean.',
               source_result_hashes_match=True, summary_and_scope_diagnostics_recomputed=True,
               artifacts_sha256={p.relative_to(c.ROOT).as_posix(): c.sha(p) for p in sorted(set(files))},
               scope='Explicit identity bridges, supplied bijection/inference language, no discovered common interface or new learning operator. No app/book integration or commit/push/publication.')
    (c.RESULTS / 'final_audit.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(c.encode({k: val for k, val in out.items() if k != 'artifacts_sha256'}))


if __name__ == '__main__': main()
