"""Descriptive paired summaries for live program installation."""
import json
import math
import migration_learning as m


def average(values): return math.fsum(values)/len(values)
def summarize(data):
    groups=[]
    for prefix in m.PREFIXES:
        for task in m.TASKS:
            for policy in m.POLICIES:
                runs=[r for r in data['runs'] if (r['prefix'],r['task'],r['policy'])==(prefix,task,policy)]
                assert len(runs)==24
                points=[]
                for offset in (0,1,2,3,4,5,15,16,17,64):
                    rows=[r['rows'][offset] for r in runs]
                    points.append(dict(t=offset,**{key:average([row[key] for row in rows]) for key in
                                                   ('new_brier','old_mse','state_count','certain_wrong','true_state_in_set')}))
                periods=[]
                for low,high in ((0,4),(5,15),(17,64)):
                    periods.append(dict(t_first=low,t_last=high,
                        new_brier=average([average([r['rows'][i]['new_brier'] for i in range(low,high+1)]) for r in runs]),
                        old_mse=average([average([r['rows'][i]['old_mse'] for i in range(low,high+1)]) for r in runs])))
                groups.append(dict(prefix=prefix,task=task,policy=policy,n_lives=24,points=points,periods=periods,
                    t0_correct_certain=sum(r['rows'][0]['state_count']==1 and r['rows'][0]['new_brier']==0 for r in runs),
                    all_new_after17_exact=sum(all(row['new_brier']==0 for row in r['rows'][17:]) for r in runs),
                    old_all_preserved=sum(all(row['old_mse']==0 for row in r['rows']) for r in runs),
                    repairs=average([r['final_state']['repairs'] for r in runs]),
                    preinstall_raw_bits=average([r['installation']['preinstall_storage']['raw_bits'] for r in runs]),
                    auxiliary_counter_min_fixed_bits=average([r['installation']['preinstall_storage']['auxiliary_counter_min_fixed_bits'] for r in runs]),
                    installation_replay_bits=average([r['final_state']['install_replay_bits'] for r in runs]),
                    new_transition_checks=average([r['final_state']['new_transition_checks'] for r in runs]),
                    installation_state_json_bytes=average([len(m.encode(r['installation']).encode()) for r in runs]),
                    final_state_json_bytes=average([len(m.encode(r['final_state']).encode()) for r in runs])))
    calibration=[]
    for task in ('old',*m.TASKS):
        records=[c for c in data['calibrations'] if c['task']==task]
        calibration.append(dict(task=task,count=len(records),unique_consistent=sum(len(c['consistent'])==1 for c in records),
                                selected_programs=[c['selected'] for c in records],
                                candidate_transitions_mean=average([c['candidate_transitions'] for c in records]),
                                prequential_brier_mean=average([sum((p-int(i==row['label']))**2 for i,p in enumerate(row['p']))
                                                               for c in records for row in c['prequential']])))
    return dict(runs=len(data['runs']),paired_lives=24,mainstream_events=sum(len(r['rows']) for r in data['runs']),
                calibration=calibration,groups=groups,
                scope='Finite paired live-migration experiment; source calibration uses separate reset examples and a supplied program family.')


def main():
    data=json.loads((m.RESULTS/'migration_learning.json').read_text(encoding='utf-8'))
    summary=summarize(data)
    (m.RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['calibration']))
    for g in summary['groups']:
        print(json.dumps({key:g[key] for key in ('prefix','task','policy','periods','t0_correct_certain','all_new_after17_exact','old_all_preserved','repairs')}))


if __name__=='__main__':main()
