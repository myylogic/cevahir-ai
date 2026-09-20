"""Reproduce the complete update experiment and independent moment analysis.

Only this round's verification record is written. Older round hashes are checked
against the pre-experiment manifest. No third-party packages are required.
"""
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
DOCS = ROOT/'docs'/'research'/'living_learning_update_2026_09_20'
RESULTS = HERE/'results'
OUT = RESULTS/'verification.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script, *args):
    proc = subprocess.run([sys.executable, str(HERE/script), *map(str, args)],
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def preservation():
    manifest = json.loads((RESULTS/'prior_manifest.json').read_text(encoding='utf-8'))['files']
    changed = [name for name, entry in manifest.items() if
               not (ROOT/name).is_file() or digest(ROOT/name) != entry['sha256'] or
               (ROOT/name).stat().st_size != entry['bytes']]
    assert not changed, changed
    return {'preserved_files': len(manifest), 'changed': changed}


def local_links():
    count = 0
    missing = []
    for document in DOCS.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            path = (document.parent/unquote(target.split('#')[0])).resolve()
            count += 1
            if not path.exists() and path != OUT:
                missing.append({'document': document.name, 'target': target})
    assert not missing, missing
    return {'checked_local_links': count, 'missing': missing,
            'verification_self_link': 'Checked after writing this output.'}


def main():
    preserved = preservation()
    original = json.loads((RESULTS/'learned_update.json').read_text(encoding='utf-8'))
    analysis = json.loads((RESULTS/'analytic_risk.json').read_text(encoding='utf-8'))
    assert original['source_sha256'] == digest(HERE/'learned_update.py')
    assert analysis['source_sha256'] == digest(HERE/'analytic_risk.py')
    assert analysis['input_sha256'] == digest(RESULTS/'learned_update.json')
    with tempfile.TemporaryDirectory(prefix='living-learning-update-reproduction-') as temp:
        temp = Path(temp)
        run('learned_update.py', '--output', temp/'learned_update.json')
        reproduced = json.loads((temp/'learned_update.json').read_text(encoding='utf-8'))
        assert reproduced == original, 'Full experiment JSON differs on rerun.'
        run('analytic_risk.py', '--output', temp/'analytic_risk.json')
        reproduced_analysis = json.loads((temp/'analytic_risk.json').read_text(encoding='utf-8'))
        assert reproduced_analysis == analysis, 'Independent analytical result differs on rerun.'
    # Exact symmetries of the independent expectation, not Monte Carlo equality.
    regimes = analysis['regimes']
    symmetry_error = 0.
    for seed, same in regimes['same_geometry']['per_source_conditional_expectations'].items():
        reverse = regimes['reversed_geometry']['per_source_conditional_expectations'][seed]
        isotropic = regimes['isotropic_change']['per_source_conditional_expectations'][seed]
        for metric in same['learned_frozen']:
            symmetry_error = max(symmetry_error,
                abs(same['learned_frozen'][metric]-reverse['rotated_90'][metric]),
                abs(same['rotated_90'][metric]-reverse['learned_frozen'][metric]),
                abs(isotropic['learned_frozen'][metric]-isotropic['rotated_90'][metric]))
    assert symmetry_error < 1e-12, symmetry_error
    preservation()
    links = local_links()
    diff = subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.splitlines()
    record = {'status': 'passed', 'python': sys.version.split()[0],
              'complete_experiment_json_reproduced': True,
              'complete_independent_analysis_json_reproduced': True,
              'source_hashes_match': True,
              'independent_analytic_symmetry_max_error': symmetry_error,
              'mathematical_checks': original['mathematical_checks'],
              'independent_moment_checks': analysis['checks'],
              'persistence': original['persistence'], 'prior_files': preserved,
              'local_links': links, 'tracked_diff_files': diff,
              'result_sha256': {p.name: digest(p) for p in [RESULTS/'learned_update.json', RESULTS/'analytic_risk.json']},
              'scope': 'Reproducibility, implementation checks and prior-file preservation; not proof of a general living-learning principle.'}
    OUT.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    assert OUT.is_file()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
