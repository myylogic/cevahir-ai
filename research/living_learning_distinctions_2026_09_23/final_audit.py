"""Final immutable-result, report-count, resource and preservation audit."""
import json
import platform
import re
from urllib.parse import unquote
import distinction_learning as c
import analyze_results as a
import verify_distinctions as v

def main():
    read=lambda name:json.loads((c.RESULTS/name).read_text(encoding='utf-8'))
    data=read('distinction_learning.json');summary=read('summary.json')
    verification=read('verification.json');diagnostics=read('diagnostics.json')
    assert data['source_sha256']==c.sha(c.__file__) and data['protocol_sha256']==c.sha(c.PROTOCOL)
    assert summary==a.summarize(data)
    assert len(data['worlds'])==72 and len(data['runs'])==288
    assert (summary['blocks'],summary['passive_predictions'])==(4608,18432)
    assert verification['full_rerun']['returncode']==0 and verification['full_rerun']['stdout'].endswith('exact_rerun_passed')
    assert len(verification['fresh_process_resume'])==24 and all(row['exact_equal'] for row in verification['fresh_process_resume'])
    recorded=verification['recorded']
    assert (recorded['snapshots'],recorded['exact_checks'],recorded['long_output_checks'])==(4608,4608,589824)
    assert (recorded['independent_cached_labels'],recorded['split_witnesses'])==(57799,592)
    assert len(recorded['paired_rebuild'])==72 and sum(row['equal_unfrozen_blocks'] for row in recorded['paired_rebuild'])==1152
    assert verification['structural']['two_state_pair_equivalence_checks']==4096
    expected={'adaptive':[10,23,24],'rebuild':[10,23,24],'exhaustive_tests':[10,24,24],'fixed_tests':[0,0,0]}
    for method,counts in expected.items():
        groups=[next(row for row in summary['groups'] if row['family']==family and row['method']==method) for family in c.FAMILIES]
        assert [row['final_exact'] for row in groups]==counts
        assert all(row['frozen']==0 for row in groups)
    lookup={(run['seed'],run['family'],run['method']):run for run in data['runs']}
    paired_blocks=0;cache_scope_checks=0
    for world in data['worlds']:
        adaptive=lookup[world['seed'],world['family'],'adaptive'];rebuild=lookup[world['seed'],world['family'],'rebuild']
        assert adaptive['records'][-1]['work']<rebuild['records'][-1]['work']
        for x,y in zip(adaptive['records'],rebuild['records']):
            assert x['state_after']['learner']==y['state_after']['learner']
            assert x['state_after']['environment']==y['state_after']['environment']
            assert x['predictions']==y['predictions'];paired_blocks+=1
    for run in data['runs']:
        state=run['final_state'];environment=state['environment'];learner=state['learner']
        assert state['reason'] is None and not state['frozen']
        assert state['work']['used']==sum(state['work']['categories'].values())<c.INTERNAL_LIMIT
        assert environment['units']==environment['passive_units']+sum(query['cost'] for query in environment['queries'])<c.ENV_LIMIT
        assert state['operating_work']==environment['passive_units']
        assert len(environment['queries'])==len({q['word'] for q in environment['queries']})
        assert learner['known']==environment['cache'];cache_scope_checks+=1
    assert diagnostics['status']=='passed' and diagnostics['no_actual_cap_reached']
    assert diagnostics['independent_first_block_predictions']==1152
    assert diagnostics['independently_minimal_exact_final_models']==172
    assert diagnostics['adaptive_lives_with_new_tests']==58
    assert diagnostics['all_zero_evidence_cycle_failures']==14
    assert diagnostics['wrong_adaptive_models_with_zero_long_test_errors']==4
    assert diagnostics['cache_errors_by_method']==dict(adaptive=0,rebuild=0,exhaustive_tests=0,fixed_tests=1677)
    assert diagnostics['maximum_environment_units']==33256 and diagnostics['maximum_learning_work']==1057573
    assert diagnostics['final_long_test_words_unseen']==36864 and diagnostics['maximum_observed_word_length']==28
    assert sum(row['final_long_errors'] for row in summary['groups'] if row['method']=='adaptive')==26
    for pair in summary['paired']:
        if pair['right']=='rebuild':assert (pair['same_model'],pair['same_evidence'],pair['left_cheaper'])==(24,24,24)
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
    result=dict(date='2026-09-23',status='passed',preservation=v.preservation(),python=platform.python_version(),
                worlds=72,paired_seeds=24,paths=288,block_snapshots=4608,passive_predictions=18432,
                independent_equivalence_checks=4608,independent_long_outputs=589824,
                cached_labels_checked=57799,split_witnesses_checked=592,
                equal_adaptive_rebuild_blocks=paired_blocks,matching_physical_logical_caches=cache_scope_checks,
                exact_final_models=172,adaptive_exact_worlds=57,exhaustive_exact_worlds=58,
                adaptive_wrong_models_with_zero_long_test_errors=4,
                fresh_process_checks=24,full_exact_rerun=True,
                primary_source_and_protocol_unchanged=True,summary_recomputed=True,
                resource_categories_match=True,no_actual_resource_limit_reached=True,
                independent_design_and_code_review='Same-evidence isolation and no target-table path confirmed; cap wording ambiguity and audit coverage gaps identified and documented.',
                post_run_diagnostics='Added cache-wide errors, initial prediction checks, minimality and unobserved failure witnesses; no primary experiment edit.',
                protocol_boundary_note='Implemented cap admits cached queries and rejects new uncached queries exceeding the environment limit. This differs from broad automatic-freeze wording. No actual run reached the limit.',
                report_claims_checked=True,local_links_checked=links,
                artifacts_sha256={path.relative_to(c.ROOT).as_posix():c.sha(path) for path in artifacts},
                scope='Known-alphabet, stationary deterministic finite-state worlds with reset and truthful scalar queries. Classical table learning, not a new general principle. Local only; no commit/push/publication.')
    (c.RESULTS/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({key:value for key,value in result.items() if key!='artifacts_sha256'}))

if __name__=='__main__':main()
