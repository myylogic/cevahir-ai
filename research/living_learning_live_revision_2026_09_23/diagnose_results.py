"""Post-run revision composition and error cancellation, without learner changes."""
from collections import Counter
import json
import live_revision as c

def diagnose(data):
    changes=[];counts=Counter()
    for source in data['sources']:
        old=source['initial'];n=0
        for block,new in enumerate(source['models']):
            if old!=new:
                changes.append(dict(seed=source['seed'],family=source['family'],block=block,
                                    old_states=len(old['outputs']),new_states=len(new['outputs']),new_exact=source['exact'][block]))
                n+=1
            old=new
        counts[n]+=1
    cross=[];all_runs={(r['seed'],r['family'],r['stride'],r['method']):r for r in data['runs']}
    for source in data['sources']:
        for stride in c.STRIDES:
            key=(source['seed'],source['family'],stride)
            reset=all_runs[*key,'reset_start'];archive=all_runs[*key,'full_archive']
            wrong=base=damage=cancel=0
            for a,b in zip(reset['records'],archive['records']):
                for x,y in zip(a['points'],b['points']):
                    assert y['prediction']==x['replay_output'] and y['truth']==x['truth']
                    actual=x['prediction']!=x['truth'];original=y['prediction']!=y['truth']
                    wrong+=actual;base+=original;damage+=actual and not original;cancel+=original and not actual
            assert wrong==base+damage-cancel
            cross.append(dict(seed=source['seed'],family=source['family'],stride=stride,
                              reset_wrong=wrong,archive_wrong=base,transfer_damage=damage,error_cancellation=cancel))
    result=dict(status='passed',source_worlds=72,source_revisions=len(changes),
                worlds_by_revision_count={str(key):value for key,value in sorted(counts.items())},
                revisions_with_nontrivial_old_model=sum(row['old_states']>1 for row in changes),
                revisions=changes,paired_error_accounting=cross,
                reset_wrong=sum(row['reset_wrong'] for row in cross),archive_wrong=sum(row['archive_wrong'] for row in cross),
                transfer_damage=sum(row['transfer_damage'] for row in cross),error_cancellation=sum(row['error_cancellation'] for row in cross),
                scope='Post-run descriptive diagnosis; no teacher labels or evaluator state supplied to migration algorithms.')
    assert result['worlds_by_revision_count']=={'0':14,'1':48,'2':8,'3':2}
    assert (result['source_revisions'],result['revisions_with_nontrivial_old_model'])==(70,27)
    assert (result['reset_wrong'],result['archive_wrong'],result['transfer_damage'],result['error_cancellation'])==(1158,418,782,42)
    return result

def main():
    data=json.loads((c.RESULTS/'live_revision.json').read_text(encoding='utf-8'));result=diagnose(data)
    (c.RESULTS/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({key:value for key,value in result.items() if key not in ('revisions','paired_error_accounting')}))

if __name__=='__main__':main()
