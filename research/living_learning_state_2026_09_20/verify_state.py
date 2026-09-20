"""Reproduction, independent checks, original preservation, and local links."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE/'results'
DOCS = ROOT/'docs'/'research'/'living_learning_state_2026_09_20'
OUT = RESULTS/'verification.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script, *arguments):
    result = subprocess.run([sys.executable, str(HERE/script), *map(str, arguments)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main():
    old = run('preserve_prior.py')
    unit = run('state_merging.py', '--self-test')
    original = json.loads((RESULTS/'recurrent_state.json').read_text(encoding='utf-8'))
    audit = json.loads((RESULTS/'evidence_audit.json').read_text(encoding='utf-8'))
    for name, expected in original['source_sha256'].items():
        assert sha(HERE/name) == expected, name
    assert audit['input_sha256'] == sha(RESULTS/'recurrent_state.json')
    assert audit['source_sha256'] == sha(HERE/'evidence_audit.py')
    with tempfile.TemporaryDirectory(prefix='recurrent-state-reproduction-') as temp:
        temp = Path(temp)
        run('recurrent_state.py', '--output', temp/'recurrent_state.json')
        assert json.loads((temp/'recurrent_state.json').read_text(encoding='utf-8')) == original
        run('evidence_audit.py', '--output', temp/'evidence_audit.json')
        assert json.loads((temp/'evidence_audit.json').read_text(encoding='utf-8')) == audit
    # Exact domain checks: never confuse epsilon with deployed nonempty episodes.
    main_lives = [r for r in audit['lives'] if r['regime'] == 'random_three_state']
    assert len(main_lives) == 16
    assert all(r['nonempty_language_exact'] for r in main_lives)
    assert sum(r['full_language_exact'] for r in main_lives) == 14
    corrupted = [r for r in audit['lives'] if r['regime'] == 'corrupted_unique_labels']
    assert len(corrupted) == 8 and not any(r['nonempty_language_exact'] for r in corrupted)
    assert all(r['trusted_correction_nonempty_exact'] for r in corrupted)
    # Exhaustive pair-label aliases and ordered-summary examples from theory.
    for x in (0, 1):
        target_a, feedback_a = 1-x, 1-x
        target_b, feedback_b = x, 1-x
        assert feedback_a == feedback_b and target_a != target_b
    xa, xb, y = [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]
    assert sorted(zip(xa, y)) == sorted(zip(xb, y))
    ca = sum(xa[t-1]*y[t] for t in range(1, 4))
    cb = sum(xb[t-1]*y[t] for t in range(1, 4))
    assert (ca, cb) == (1, 0)
    old_after = run('preserve_prior.py')
    assert old_after == old
    missing, link_count = [], 0
    for document in DOCS.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            link_count += 1
            target_path = (document.parent/unquote(target.split('#')[0])).resolve()
            if not target_path.exists() and target_path != OUT:
                missing.append({'document': document.name, 'target': target})
    assert not missing, missing
    tracked = subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    result = {'status': 'passed', 'python': sys.version.split()[0],
              'complete_experiment_json_reproduced': True, 'complete_evidence_audit_json_reproduced': True,
              'source_hashes_match': True, 'state_merger_self_tests': unit,
              'independent_candidate_library_check': audit['enumeration'],
              'mathematical_checks': original['mathematical_checks'],
              'theory_counterexamples_checked': ['same feedback different true target', 'same unordered observations different lag statistic'],
              'domain_audit': {'main_nonempty_exact': 16, 'main_including_empty_exact': 14,
                               'corrupted_nonempty_exact': 0, 'trusted_correction_nonempty_exact': 8},
              'persistence': original['persistence'], 'prior_files': old,
              'local_links': {'checked': link_count, 'missing': missing}, 'tracked_diff_files': tracked,
              'result_sha256': {name: sha(RESULTS/name) for name in ('recurrent_state.json', 'evidence_audit.json')},
              'scope': 'Reproducibility and declared implementation checks, not validation of a general living-learning principle.'}
    OUT.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    assert OUT.exists()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
