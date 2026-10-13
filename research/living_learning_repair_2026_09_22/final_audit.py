"""Final preservation, semantic, paired-control and publication-claim audit."""
import json
import platform
import re
from urllib.parse import unquote
import numpy as np
import repair_learning as r
import analyze_results
import verify_repair


def main():
    data=json.loads((r.RESULTS/'repair_learning.json').read_text(encoding='utf-8'))
    summary=json.loads((r.RESULTS/'summary.json').read_text(encoding='utf-8'))
    verification=json.loads((r.RESULTS/'verification.json').read_text(encoding='utf-8'))
    assert data['source_sha256']==r.sha(r.HERE/'repair_learning.py')
    assert data['protocol_sha256']==r.sha(r.PROTOCOL)
    assert summary==analyze_results.summarize(data)
    assert verification['full_rerun']['returncode']==0
    assert len(verification['fresh_process_resume'])==15
    assert all(item['exact_equal'] for item in verification['fresh_process_resume'])
    assert verification['recorded']['target_attempts']==2880
    assert verification['recorded']['full_grid_accepted_program_checks']==1095
    assert verification['recorded']['false_accepts']==0
    source_checks=0
    for life,acquired in zip(data['lives'],data['acquisitions']):
        assert life['seed']==acquired['seed']
        assert acquired['base_cost']==5690 and acquired['rule_acquisition_cost']==468
        assert len(acquired['state']['rules'])==3 and len(acquired['state']['priorities'])==1
        assert sum(acquired['categories'].values())==6158
        for source,record in zip(life['sources'],acquired['source_records']):
            assert record['accepted'] is not None
            target,accepted=r.frozen(source['target']),r.frozen(record['accepted'])
            assert all(verify_repair.scalar(target,x,y)==verify_repair.scalar(accepted,x,y) for x,y in r.GRID.tolist())
            source_checks+=1
    lookup={(run['seed'],run['regime'],run['method']):run for run in data['runs']}
    identical=0
    for seed in range(24):
        for regime in r.REGIMES:
            a,b=lookup[seed,regime,'learned_rules'],lookup[seed,regime,'bulk_ranked']
            assert [rec['accepted'] for rec in a['records']]==[rec['accepted'] for rec in b['records']]
            identical+=1
    counts={regime:[] for regime in r.REGIMES}
    for regime in r.REGIMES:
        for method in r.METHODS:
            group=next(g for g in summary['groups'] if (g['regime'],g['method'])==(regime,method))
            counts[regime].append(group['correct'])
    assert counts==dict(compatible=[48,54,84,84,54],changed=[48,48,48,48,48],partial=[51,120,120,120,120])
    paired=next(p for p in summary['paired'] if (p['regime'],p['left'],p['right'])==('compatible','learned_rules','rules_disabled_paid'))
    assert (paired['left_more_correct'],paired['tied_correct'],paired['right_more_correct'])==(19,5,0)
    for run in data['runs']:
        initial=6158 if run['method'] in ('learned_rules','rules_disabled_paid') else 5690
        assert initial+sum(run['target_categories'].values())==run['total_work']
        assert run['total_work']<r.LIMIT
    assert max(run['total_work'] for run in data['runs'])==726737
    deep=sum(rec['full_field_correct'] for run in data['runs'] if run['method']=='learned_rules' and run['regime']=='compatible'
             for rec in run['records'] if rec['index'] in (2,5,7))
    assert deep==2
    docs=r.PROTOCOL.parent;links=0
    for path in [r.HERE/'README.md',*docs.glob('*.md')]:
        markdown=re.sub(r'`[^`\n]*`','',path.read_text(encoding='utf-8'))
        for href in re.findall(r'\[[^\[\]\n]+\]\(([^)]+)\)',markdown):
            if re.match(r'^[A-Za-z][\w+.-]*:',href) or href.startswith('#'):continue
            target=(path.parent/unquote(href.split('#',1)[0])).resolve()
            assert target.exists() or target==r.RESULTS/'final_audit.json',(str(path),href)
            links+=1
    artifacts=sorted([*r.HERE.glob('*.py'),r.HERE/'README.md',*docs.glob('*.md'),
                      *(p for p in r.RESULTS.iterdir() if p.is_file() and p.name!='final_audit.json')])
    result=dict(date='2026-09-22',status='passed',preservation=verify_repair.preservation(),
                environment=dict(python=platform.python_version(),numpy=np.__version__),
                paired_lives=24,trajectories=360,target_attempts=2880,
                source_full_field_correct=source_checks,accepted_target_full_field_checks=1095,false_accepts=0,
                scalar_structural_outputs=13872,source_scalar_comparisons=source_checks*289,
                fresh_process_resume_checks=15,full_exact_rerun=True,
                identical_learned_and_bulk_program_sequences=identical,
                all_acquisition_and_target_category_totals_match=True,
                no_path_hit_global_budget=True,deep_compatible_correct=deep,
                source_protocol_result_hashes_match=True,summary_recomputed=True,
                report_claims_checked=True,local_links_checked=links,
                artifacts_sha256={p.relative_to(r.ROOT).as_posix():r.sha(p) for p in artifacts},
                scope='Declared virtual work and finite field semantics. Transfer benefit is matched by learned-priority bulk edits; no new fundamental principle or AGI claim. Local only, no commit/push.')
    (r.RESULTS/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'}))


if __name__=='__main__':main()
