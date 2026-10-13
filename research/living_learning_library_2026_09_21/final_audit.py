"""Final local package audit. Writes only results/final_audit.json."""
import hashlib
import json
from pathlib import Path
import re
import statistics
from urllib.parse import unquote

import analyze_results
import verify_library

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DOCS=ROOT/'docs/research/living_learning_library_2026_09_21'
RESULTS=HERE/'results'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_links():
    count=0
    for path in [HERE/'README.md',*DOCS.glob('*.md')]:
        for href in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z][\w+.-]*:',href) or href.startswith('#'):continue
            target=(path.parent/unquote(href.split('#',1)[0])).resolve()
            assert target.exists() or target==RESULTS/'final_audit.json',(str(path),href)
            count+=1
    return count


def main():
    primary=json.loads((RESULTS/'library_learning.json').read_text(encoding='utf-8'))
    portfolio=json.loads((RESULTS/'portfolio_control.json').read_text(encoding='utf-8'))
    verification=json.loads((RESULTS/'verification.json').read_text(encoding='utf-8'))
    assert primary['source_sha256']==sha(HERE/'library_learning.py')
    assert portfolio['source_sha256']==sha(HERE/'portfolio_control.py')
    assert portfolio['primary_source_sha256']==primary['source_sha256']
    summary=json.loads(json.dumps(analyze_results.run()))
    assert summary==json.loads((RESULTS/'paired_summary.json').read_text(encoding='utf-8'))
    assert len(verification['complete_reruns'])==2
    assert all(r['returncode']==0 for r in verification['complete_reruns'])
    assert len(verification['fresh_process_resume'])==16
    assert all(r['exact_json_equal'] for r in verification['fresh_process_resume'])
    tasks={(life['world']['seed'],t['name']):t for life in primary['lives'] for t in life['world']['probes']}
    false_accepts=0
    for row in portfolio['runs']:
        assert row['total_charged_vector_steps']==row['primitive_half']['charged_vector_steps']+row['learned_half']['charged_vector_steps']
        assert row['total_charged_vector_steps']<=row['budget']
        selected='primitive' if row['primitive_half']['status']=='accepted' else 'learned'
        assert row['selected_branch']==selected and row['selected']==row[selected+'_half']
        t=tasks[(row['seed'],row['task'])]
        for branch in ('primitive_half','learned_half','selected'):
            result=row[branch]
            exact=result['status']=='accepted' and all(verify_library.scalar(result['program'],x)==verify_library.scalar(t['teacher'],x) for x in range(101))
            assert result['exact']==exact
        false_accepts+=row['selected']['status']=='accepted' and not row['selected']['exact']
        assert not row['primitive_half']['exact'] or row['selected']['exact']
    for group in portfolio['aggregate']:
        rows=[r for r in portfolio['runs'] if (r['budget'],r['regime'])==(group['budget'],group['regime'])]
        assert len(rows)==group['conditions']==96
        for branch,key in (('primitive_half','primitive_half_exact'),('learned_half','learned_half_exact'),('selected','portfolio_exact')):
            assert group[key]==sum(r[branch]['exact'] for r in rows)
        assert group['mean_total_charged_vector_steps']==statistics.fmean(r['total_charged_vector_steps'] for r in rows)
    files=sorted([*HERE.glob('*.py'),HERE/'README.md',*DOCS.glob('*.md'),
                  *(p for p in RESULTS.glob('*.json') if p.name!='final_audit.json')])
    record=dict(date='2026-09-21',status='passed',preservation=verify_library.preservation(),
                source_result_hashes_match=True,paired_summary_exact=True,
                posthoc_scalar_semantic_cases=len(portfolio['runs'])*3,
                posthoc_false_accepts=false_accepts,portfolio_selection_and_budget_checks=384,
                local_document_links_checked=check_links(),
                artifacts_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
                scope='Local code/results/document audit; no external peer review, independent-world replication, or publication.')
    (RESULTS/'final_audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='passed',preservation=record['preservation'],links=record['local_document_links_checked'],
                         independent_posthoc_semantic_cases=record['posthoc_scalar_semantic_cases'])))


if __name__=='__main__':main()
