"""Round 21 final audit. It never rewrites the raw run."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path
import deduction as d
import analyze_results as a
import verify_deduction as v


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw = json.loads((d.RESULTS / 'deduction.json').read_text(encoding='utf-8'))
    summary = json.loads((d.RESULTS / 'summary.json').read_text(encoding='utf-8'))
    verification = json.loads((d.RESULTS / 'verification.json').read_text(encoding='utf-8'))
    assert raw['source_sha256'] == sha(d.HERE / 'deduction.py')
    assert raw['checker_sha256'] == sha(d.HERE / 'check_proof.py')
    assert raw['protocol_sha256'] == sha(d.PROTOCOL)
    assert raw['methods'] == d.METHODS and len(raw['runs']) == 24
    assert summary == a.summarize(raw)
    assert verification['recorded']['answers_checked'] == 24 * 7 * 96
    assert verification['recorded']['exact_replay_pairs'] == 24 * 96
    assert verification['randomized_solver']['cases'] == 1000
    assert verification['checker_selftest']['accepted'] == 5 and verification['checker_selftest']['rejected'] == 14
    assert all(row['exact_equal'] for row in verification['fresh_process_resume'])
    assert verification['preservation']['all_unchanged']
    # Report statements are tied to recorded figures, not regenerated prose.
    methods = {row['method']: row for row in summary['methods']}
    assert methods['scoped']['correct'] == 2304 and methods['blind']['correct'] == 2218
    assert methods['full_table_cached']['life_work_mean'] < methods['scoped']['life_work_mean']
    assert methods['scoped']['life_work_mean'] < methods['fresh']['life_work_mean']
    assert verification['recorded']['blind_invalid_active_lemma_uses'] > 0
    docs = [d.PROTOCOL.parent / x for x in ('PROTOCOL.md', 'REPORT_TR.md', 'HYPOTHESES_TR.md', 'MATHEMATICS_AND_LITERATURE_TR.md')]
    docs.append(d.HERE / 'README.md')
    links = 0
    for path in docs:
        text = path.read_text(encoding='utf-8')
        for href in re.findall(r'\[[^\[\]\n]+\]\(([^)]+)\)', re.sub(r'`[^`\n]*`', '', text)):
            if re.match(r'^[A-Za-z][\w+.-]*:', href) or href.startswith('#'):
                continue
            target = (path.parent / href.split('#', 1)[0]).resolve()
            assert target.exists(), (path, href)
            links += 1
    changed = subprocess.check_output(['git', 'status', '--short'], cwd=d.ROOT).decode()
    result = {
        'date': '2026-09-23', 'status': 'passed', 'methods': d.METHODS,
        'source_worlds': len(raw['runs']), 'queries_per_method': 24 * 96,
        'full_amended_rerun': verification.get('full_rerun', {}).get('stdout') == 'exact_rerun_passed',
        'independent_randomized_solver_cases': 1000,
        'recorded_answer_checks': verification['recorded']['answers_checked'],
        'proof_certificates_checked': verification['recorded']['accepted_certificates_checked_including_controls'],
        'active_lemma_truth_checks': verification['recorded']['active_lemma_truth_checks_including_controls'],
        'exact_replay_pairs': verification['recorded']['exact_replay_pairs'],
        'blind_invalid_active_lemma_uses': verification['recorded']['blind_invalid_active_lemma_uses'],
        'blind_invalid_new_lemmas': verification['recorded']['blind_invalid_new_lemmas'],
        'fresh_process_resumes': len(verification['fresh_process_resume']),
        'preserved_prior_files': verification['preservation']['files'],
        'base_head_unchanged': verification['preservation']['head'],
        'markdown_links_checked': links,
        'initial_pre_audit_run_preserved': (d.RESULTS / 'initial_run/deduction.json').exists(),
        'changed_files_not_committed_or_pushed': True,
        'scope': 'Checked internal deductions in small guarded CNFs; no general living-learning principle, AGI result or biological claim.'
    }
    (d.RESULTS / 'final_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
