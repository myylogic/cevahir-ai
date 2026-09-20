"""Reproduce new results separately and verify preservation of old research."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent
RESULTS = BASE / 'results'
DOCS = ROOT / 'docs/research/living_learning_reassessment_2026_09_20'
IGNORED_ENVIRONMENT_FIELDS = {'elapsed_seconds', 'traced_python_peak_bytes', 'python', 'platform'}


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k not in IGNORED_ENVIRONMENT_FIELDS}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reproduce(name):
    source = BASE / (name + '.py')
    original = json.loads((RESULTS / (name + '.json')).read_text(encoding='utf-8'))
    # The credit experiment already has a fresh-process exact reproduction.
    # Reuse it when its source and scientific payload still match.
    if name == 'credit_assignment':
        record = json.loads((RESULTS/'credit_assignment_verification.json').read_text(encoding='utf-8'))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        scientific_hash = digest({'summaries': original['summaries'], 'runs': original['runs']})
        assert record['exact_full_json_match']
        assert source_hash == record['source_sha256'] == original['provenance']['source_sha256']
        assert scientific_hash == record['scientific_sha256'] == original['provenance']['scientific_sha256']
        return {'experiment': name, 'scientific_results_equal': True,
                'recorded_source_matches': True, 'independent_reproduction_reused': record,
                'original_output_overwritten': False}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='living-learning-reassessment-') as temporary:
        dest = Path(temporary) / (name + '.json')
        run = subprocess.run([sys.executable, str(source), '--output', str(dest)],
                             cwd=ROOT, text=True, capture_output=True, check=True)
        fresh = json.loads(dest.read_text(encoding='utf-8'))
    scientific_equal = canonical(original) == canonical(fresh)
    expected_hash = original.get('source_sha256', original.get('provenance', {}).get('source_sha256'))
    source_equal = expected_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    record = {'experiment': name, 'scientific_results_equal': scientific_equal,
              'recorded_source_matches': source_equal,
              'canonical_sha256': digest(canonical(fresh)),
              'elapsed_seconds': time.perf_counter() - started,
              'excluded_environment_fields': sorted(IGNORED_ENVIRONMENT_FIELDS),
              'original_output_overwritten': False, 'subprocess_exit_code': run.returncode}
    if not scientific_equal or not source_equal:
        raise AssertionError(record)
    return record


def check_local_links():
    missing = []
    checked = 0
    for path in DOCS.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'^(https?://|#|mailto:)', target):
                continue
            target = target.split('#')[0]
            checked += 1
            if not (path.parent / target).exists():
                missing.append({'document': path.name, 'target': target})
    assert not missing, missing
    return {'checked': checked, 'missing': missing}


def main():
    old = subprocess.run([sys.executable, str(BASE/'preserve_prior.py')], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(reproduce, ('representation_growth', 'credit_assignment', 'state_transport')))
    # Exact arithmetic for the separate mean/addition and ID-only retraction witnesses.
    assert Fraction(0+2, 2) == Fraction(1, 1)
    assert Fraction(0+2+4, 3) != Fraction(1+4, 2)
    assert (0+2, 2) == (1+1, 2) and Fraction(2, 1) != Fraction(1, 1)
    tracked = subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    report = {'old_research': json.loads(old.stdout), 'reproductions': records,
              'analytical_mean_witnesses_exact': True, 'local_links': check_local_links(),
              'tracked_changes_relative_to_HEAD': tracked.stdout.splitlines(),
              'scope': 'Preservation, source provenance and reproducibility. Not independent evidence of generality or novelty.',
              'passed': True}
    (RESULTS/'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
