"""Final integrity checks, cost correction comparison and report invariants."""
import copy
import hashlib
import json
import platform
import re
from urllib.parse import unquote
import migration_learning as m
import analyze_results
import verify_migration


def normalized(row):
    value=copy.deepcopy(row)
    value.pop('final_state_sha256')
    for key in ('installation','final_state'):value[key].pop('install_replay_bits')
    return value


def main():
    data=json.loads((m.RESULTS/'migration_learning.json').read_text(encoding='utf-8'))
    original=json.loads((m.RESULTS/'migration_learning_initial_cost_omission.json').read_text(encoding='utf-8'))
    summary=json.loads((m.RESULTS/'summary.json').read_text(encoding='utf-8'))
    verification=json.loads((m.RESULTS/'verification.json').read_text(encoding='utf-8'))
    assert data['source_sha256']==m.sha(m.HERE/'migration_learning.py')
    assert original['source_sha256']==m.sha(m.RESULTS/'initial_source_cost_omission.txt')
    assert data['protocol_sha256']==original['protocol_sha256']==m.sha(m.PROTOCOL)
    assert data['lives']==original['lives'] and data['calibrations']==original['calibrations']
    assert all(normalized(a)==normalized(b) for a,b in zip(data['runs'],original['runs']))
    assert summary==analyze_results.summarize(data)
    assert verification['full_rerun']['returncode']==0
    assert verification['recorded']['compiled_correct']==72
    assert verification['recorded']['calibration_predictions']==9216
    assert verification['recorded']['mainstream_events']==37440
    assert len(verification['fresh_process_resume'])==24
    assert all(r['exact_equal'] for r in verification['fresh_process_resume'])
    life_truths={(n,seed) for n in m.PREFIXES for seed in range(24)}
    assert len(life_truths)==48
    groups={(g['prefix'],g['task'],g['policy']):g for g in summary['groups']}
    for row in data['runs']:
        points=row['rows'];policy=row['policy'];task=row['task'];prefix=row['prefix']
        assert all(point['new_brier']==0 for point in points[17:])
        if policy!='reset_all': assert all(point['old_mse']==0 for point in points)
        if policy not in ('guess_zero','reset_all'): assert all(point['true_state_in_set'] for point in points)
        if task=='delay_5':
            assert all(point['new_brier']==0 for point in points[5:])
            if policy in ('possible_states','reserve_mod6'):
                assert [point['state_count'] for point in points[:6]]==[32,16,8,4,2,1]
                assert all(point['new_brier']==.5 for point in points[:5])
                assert row['final_state']['new_transition_checks']==121
            if policy in ('reserve_8','archive'): assert all(point['new_brier']==0 for point in points)
        else:
            if policy in ('possible_states','reserve_8'):
                assert all(point['state_count']==3 and abs(point['new_brier']-2/3)<1e-15 for point in points[:17])
                assert row['final_state']['new_transition_checks']==96
            if policy in ('reserve_mod6','archive'): assert all(point['new_brier']==0 for point in points)
        if policy=='archive':
            assert row['installation']['preinstall_storage']['raw_bits']==prefix
            assert row['final_state']['install_replay_bits']==prefix
        if policy=='reserve_8' and task=='delay_5': assert row['final_state']['install_replay_bits']==8
    for prefix,old_preserved,mod_zero_correct in ((256,13,12),(4096,15,13)):
        for task in m.TASKS: assert groups[prefix,task,'reset_all']['old_all_preserved']==old_preserved
        assert groups[prefix,'mod3','guess_zero']['t0_correct_certain']==mod_zero_correct
    docs=m.PROTOCOL.parent;links=0
    for path in [m.HERE/'README.md',*docs.glob('*.md')]:
        for href in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z][\w+.-]*:',href) or href.startswith('#'): continue
            target=(path.parent/unquote(href.split('#',1)[0])).resolve()
            assert target.exists() or target==m.RESULTS/'final_audit.json',(str(path),href)
            links+=1
    artifacts=sorted([*m.HERE.glob('*.py'),m.HERE/'README.md',*docs.glob('*.md'),
                      *(p for p in m.RESULTS.iterdir() if p.is_file() and p.name!='final_audit.json')])
    result=dict(date='2026-09-22',status='passed',preservation=verify_migration.preservation(),
                environment=dict(python=platform.python_version()),runs=576,paired_lives=24,
                calibration_predictions=9216,mainstream_events=37440,capability_outputs=74880,
                unique_correct_compiled_programs=72,fresh_process_resume_checks=24,
                full_exact_rerun=True,source_protocol_result_hashes_match=True,summary_recomputed=True,
                initial_cost_omission_preserved=True,all_predictions_unchanged_by_correction=True,
                only_installation_replay_counter_and_derived_hashes_changed=True,
                report_claims_checked=True,local_links_checked=links,
                artifacts_sha256={p.relative_to(m.ROOT).as_posix():m.sha(p) for p in artifacts},
                scope='Finite live state migration with learned supplied programs and separate reset calibration. No claim of an invented representation language, general continual learner or AGI. No commit/push.')
    (m.RESULTS/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'}))


if __name__=='__main__':main()
