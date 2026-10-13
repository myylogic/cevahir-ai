"""Final immutable-source, paired-result, cost and report audit."""
import json
import platform
import re
from urllib.parse import unquote
import live_revision as c
import analyze_results as a
import diagnose_results as d
import verify_live_revision as v

def main():
    read=lambda name:json.loads((c.RESULTS/name).read_text(encoding='utf-8'))
    data=read('live_revision.json');summary=read('summary.json');verification=read('verification.json');diagnostics=read('diagnostics.json')
    assert data['source_sha256']==c.sha(c.__file__) and data['protocol_sha256']==c.sha(c.PROTOCOL)
    assert data['previous_source_sha256']==c.sha(c.PREVIOUS/'distinction_learning.py')
    assert data['previous_result_sha256']==c.sha(c.SOURCE)
    assert data['sources']==c.sources()
    assert summary==a.summarize(data) and diagnostics==d.diagnose(data)
    assert len(data['runs'])==864 and len(data['sources'])==72
    assert verification['full_rerun']['returncode']==0 and verification['full_rerun']['stdout'].endswith('exact_rerun_passed')
    assert len(verification['renaming'])==18 and all(row['renaming_invariant'] for row in verification['renaming'])
    assert len(verification['fresh_process_resume'])==36 and all(row['exact_equal'] for row in verification['fresh_process_resume'])
    record=verification['recorded']
    assert (record['prediction_checks'],record['parity_checks'],record['sound_replay_containment_checks'],record['suffix_control_subset_checks'])==(235008,235008,195840,39168)
    assert verification['structural']['short_history_relation_checks']==258048
    assert verification['structural']['suffix_relation_checks']==258048
    expected={'reset_start':(38010,1158,0,1471),'all_states':(36945,305,1918,0),
              'parity_suffix8':(37403,324,1441,0),'model_relation':(36956,305,1907,0),
              'model_relation_suffix8':(37403,324,1441,0),'full_archive':(38750,418,0,0)}
    expected_exact={'reset_start':(27745,713,0),'all_states':(26877,0,1581),
                    'parity_suffix8':(27270,0,1188),'model_relation':(26886,0,1572),
                    'model_relation_suffix8':(27270,0,1188),'full_archive':(28458,0,0)}
    methods={g['method']:g for g in summary['methods']}
    for method,group in methods.items():
        point=group['points'];exact=group['exact_model_points']
        assert tuple(point[key] for key in ('correct','wrong','abstained','replay_not_contained'))==expected[method]
        assert tuple(exact[key] for key in ('correct','wrong','abstained'))==expected_exact[method]
        assert point['n']==39168 and exact['n']==28458
        assert point['correct']+point['wrong']+point['abstained']==point['n']
        assert group['actual_revisions']==140
        assert group['calibration_work']['mean']==39979.583333333336
        if method!='reset_start':assert point['confident_replay_disagreement']==0
    assert methods['reset_start']['changed_t0']['wrong']==61
    assert methods['full_archive']['changed_t0']['wrong']==7
    assert methods['parity_suffix8']['changed_t0']['abstained']==56
    assert diagnostics['worlds_by_revision_count']=={'0':14,'1':48,'2':8,'3':2}
    assert (diagnostics['source_revisions'],diagnostics['revisions_with_nontrivial_old_model'])==(70,27)
    assert diagnostics['reset_wrong']==diagnostics['archive_wrong']+diagnostics['transfer_damage']-diagnostics['error_cancellation']==1158
    for pair in summary['paired']:
        if (pair['left'],pair['right'])==('model_relation_suffix8','parity_suffix8'):
            assert (pair['identical_set_lives'],pair['identical_prediction_lives'],pair['left_cheaper'],pair['abstention_difference'])==(72,72,0,0)
    for run in data['runs']:
        state=run['final_state'];pos=16*run['stride']+16
        assert state['position']==pos
        assert state['categories']['parity_transitions']==pos
        assert state['categories']['output_reads']==sum(p['size'] for row in run['records'] for p in row['points'])
        assert run['records'][-1]['work']==sum(state['categories'].values())
        if run['method'] in ('full_archive','parity_suffix8','model_relation_suffix8'):assert state['categories']['trace_bit_writes']==pos
        else:assert 'trace_bit_writes' not in state['categories']
        if run['method']=='full_archive':
            expected_replay=sum((row['block']+1)*run['stride']+1 for row in run['records'] if row['install']['changed'])
            assert state['categories'].get('archive_replay',0)==expected_replay
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
    result=dict(date='2026-09-23',status='passed',python=platform.python_version(),preservation=v.preservation(),
                paired_seeds=24,source_worlds=72,paths=864,prediction_checks=235008,
                source_revisions=70,migration_events_per_method=140,
                sound_replay_containment_checks=195840,short_history_relation_checks=258048,suffix_relation_checks=258048,
                state_renaming_paths=18,fresh_process_resumes=36,full_exact_rerun=True,
                all_sources_match_unchanged_previous_results=True,source_and_protocol_hashes_match=True,
                summary_and_diagnostics_recomputed=True,cost_categories_and_archive_replay_checked=True,
                identical_relation_suffix_and_suffix_only_points=39168,
                correct_model_reset_errors=713,all_parity_predictions_correct=True,
                report_claims_checked=True,local_links_checked=links,
                artifacts_sha256={path.relative_to(c.ROOT).as_posix():c.sha(path) for path in artifacts},
                scope='Separate reset-based calibration and never-reset live input. Transfer soundness is relative to each learned program, not true-world adequacy. Strong suffix control matches added relation. No new fundamental principle; local only, no commit/push/publication.')
    (c.RESULTS/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({key:value for key,value in result.items() if key!='artifacts_sha256'}))

if __name__=='__main__':main()
