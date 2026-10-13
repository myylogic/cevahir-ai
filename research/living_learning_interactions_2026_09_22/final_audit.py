"""Final preservation, immutable experiment and report-claim audit."""
import json
import importlib.util
import platform
import re
from urllib.parse import unquote
import numpy as np
import interaction_learning as c
import verify_interactions as v

# Round 17 exposes a same-named analyzer; load this round's file explicitly.
analyzer_spec=importlib.util.spec_from_file_location('interaction_results',c.HERE/'analyze_results.py')
a=importlib.util.module_from_spec(analyzer_spec)
analyzer_spec.loader.exec_module(a)

def main():
    read=lambda name:json.loads((c.RESULTS/name).read_text(encoding='utf-8'))
    data=read('interaction_learning.json');summary=read('summary.json')
    verification=read('verification.json');diagnosis=read('coverage_diagnostics.json')
    assert data['source_sha256']==c.sha(c.HERE/'interaction_learning.py')
    assert data['protocol_sha256']==c.sha(c.PROTOCOL)
    assert data['prior_source_sha256']==c.sha(c.PREVIOUS/'repair_learning.py')
    assert data['source_result_sha256']==c.sha(c.SOURCE_RESULT)
    assert summary==a.summarize(data)
    assert len(data['lives'])==24 and len(data['runs'])==168
    assert all(source['acquisition_cost']==6158 for source in data['sources'])
    assert verification['full_rerun']['returncode']==0
    assert verification['full_rerun']['stdout'].endswith('exact_rerun_passed')
    assert len(verification['fresh_process_resume'])==14
    assert all(item['exact_equal'] for item in verification['fresh_process_resume'])
    recorded=verification['recorded']
    assert (recorded['independent_probe_checks'],recorded['full_field_accepted_checks'],recorded['false_accepts'])==(94110,1744,0)
    assert (recorded['target_attempts'],recorded['correct_different_masks'])==(2688,4)
    assert summary['totals']['work']['max']==396593
    assert summary['totals']['model_mismatches']==0 and summary['totals']['budget_stops']==0
    counts={group['method']:[group['phases'][name]['correct'] for name in a.PHASES] for group in summary['groups']}
    assert counts==dict(bulk_all=[0,0,0,0],greedy_single=[2,8,3,21],first_order=[96,18,96,0],
                       second_order=[96,96,96,84],adaptive_reset=[96,96,96,84],
                       adaptive_persistent=[96,96,96,84],exact_subset=[96,96,96,96])
    lookup={(run['seed'],run['method']):run for run in data['runs']}
    for seed in c.SEEDS:
        reset=lookup[seed,'adaptive_reset'];persist=lookup[seed,'adaptive_persistent'];second=lookup[seed,'second_order']
        assert [row['accepted_mask'] for row in reset['records']]==[row['accepted_mask'] for row in persist['records']]==[row['accepted_mask'] for row in second['records']]
        assert persist['total_work']>reset['total_work']
        assert persist['final_state']['expanded_tasks']==[4]
    expected_deltas={'whole_life':232747,'initial_additive':0,'pairs':10939,'return_additive':303728,'triples':-81920}
    for pair in summary['paired']:
        if pair['left']=='adaptive_persistent' and pair['right']=='adaptive_reset':
            assert pair['work_difference']['mean']==expected_deltas[pair['phase']]
            assert pair['correct_difference']==dict(mean=0,min=0,max=0)
    assert diagnosis['status']=='passed'
    assert diagnosis['independent_scalar_coefficient_outputs']==16896
    assert (diagnosis['first_order_pair_successes'],diagnosis['second_order_triple_misses'])==(18,12)
    assert diagnosis['same_accepted_sequence_second_reset_persistent']==24
    aliases=[(run['seed'],row['index']) for run in data['runs'] for row in run['records']
             if row['full_field_correct'] and not row['planted_mask_equal']]
    assert aliases==[(14,7)]*4
    for run in data['runs']:
        assert run['total_work']==6158+sum(row['work'] for row in run['records'])
        assert all(row['work']==sum(row['categories'].values()) for row in run['records'])
        assert len(run['final_state']['library'])==sum(row['accepted_mask'] is not None for row in run['records'])
        assert run['final_json_bytes']==len(c.encode(run['final_state']).encode())
    docs=c.PROTOCOL.parent;links=0
    for path in [c.HERE/'README.md',*docs.glob('*.md')]:
        markdown=re.sub(r'`[^`\n]*`','',path.read_text(encoding='utf-8'))
        for href in re.findall(r'\[[^\[\]\n]+\]\(([^)]+)\)',markdown):
            if re.match(r'^[A-Za-z][\w+.-]*:',href) or href.startswith('#'):continue
            target=(path.parent/unquote(href.split('#',1)[0])).resolve()
            assert target.exists() or target==c.RESULTS/'final_audit.json',(str(path),href)
            links+=1
    artifacts=sorted([*c.HERE.glob('*.py'),c.HERE/'README.md',*docs.glob('*.md'),
                      *(path for path in c.RESULTS.iterdir() if path.is_file() and path.name!='final_audit.json')])
    result=dict(date='2026-09-22',status='passed',preservation=v.preservation(),
                environment=dict(python=platform.python_version(),numpy=np.__version__),
                paired_lives=24,trajectories=168,target_attempts=2688,
                independent_probe_checks=94110,accepted_full_field_checks=1744,false_accepts=0,
                correct_different_masks=4,distinct_alias_tasks=1,full_exact_rerun=True,
                fresh_process_resume_checks=14,structural_scalar_outputs=3072,full_mobius_reconstructions=768,
                post_run_diagnostic_scalar_outputs=16896,source_protocol_result_hashes_match=True,
                all_source_and_target_costs_match=True,no_target_hit_budget=True,
                paired_sequence_equal_lives=24,persistent_more_expensive_lives=24,
                report_claims_checked=True,summary_recomputed=True,local_links_checked=links,
                independent_read_only_review='No blocking error or target leakage found; reporting limits incorporated.',
                diagnostic_development_note='First diagnostic-only attempt omitted JSON AST restoration; first final-audit attempt imported the prior same-named analyzer. Both stopped before reporting success. Restored tuple ASTs and selected the local analyzer by path; reruns passed. Primary experiment and protocol unchanged.',
                artifacts_sha256={path.relative_to(c.ROOT).as_posix():c.sha(path) for path in artifacts},
                scope='Finite field, 8 disjoint edit sites, 16 examples, declared virtual work. Persisted preference is one bit. General living-learning problem remains open. Local only, no commit/push/publication.')
    (c.RESULTS/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({key:value for key,value in result.items() if key!='artifacts_sha256'}))

if __name__=='__main__':main()
