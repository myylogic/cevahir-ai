"""Post-run explanation checks; not a changed solver or added performance trial."""
import json
import verify_interactions as v
import interaction_learning as c

def submasks(mask):
    return [sub for sub in range(256) if sub&mask==sub]

def main():
    data=json.loads((c.RESULTS/'interaction_learning.json').read_text(encoding='utf-8'))
    lookup={(run['seed'],run['method']):run for run in data['runs']}
    pair_checks=[];triple_checks=[];scalar_outputs=0
    for saved_life in data['lives']:
        life=v.restore_life(saved_life)
        for task in life['tasks'][4:8]:
            mask=task['planted_mask']
            has_complete_pair=any(mask&(3<<(2*i))==(3<<(2*i)) for i in range(4))
            actual=lookup[life['seed'],'first_order']['records'][task['index']]['full_field_correct']
            assert actual==(not has_complete_pair)
            pair_checks.append(dict(seed=life['seed'],index=task['index'],mask=mask,
                                   contains_full_pair=has_complete_pair,first_order_correct=actual))
        for task in life['tasks'][12:]:
            mask=task['planted_mask'];complete_triple=mask&7==7 or mask&56==56
            coefficients={};values={}
            # Independently evaluate the coefficients needed at the planted mask.
            for sub in sorted(submasks(mask),key=lambda value:(value.bit_count(),value)):
                if sub.bit_count()>2:continue
                values[sub]=[v.scalar_changed(task['initial'],sub,x,y,[0]) for x,y in task['xy']]
                scalar_outputs+=len(task['xy'])
                lower=[other for other in coefficients if other&sub==other]
                coefficients[sub]=[(value-sum(coefficients[other][j] for other in lower))%17
                                    for j,value in enumerate(values[sub])]
            predicted=[sum(coefficient[j] for coefficient in coefficients.values())%17 for j in range(16)]
            nonzero_residual=predicted!=task['labels']
            record=lookup[life['seed'],'second_order']['records'][task['index']]
            assert nonzero_residual==complete_triple
            assert record['full_field_correct']==(not complete_triple)
            assert lookup[life['seed'],'exact_subset']['records'][task['index']]['full_field_correct']
            triple_checks.append(dict(seed=life['seed'],index=task['index'],mask=mask,
                                     contains_full_triple=complete_triple,evidence_residual_nonzero=nonzero_residual,
                                     second_order_correct=record['full_field_correct'],exact_correct=True))
    identical=0;switches=[]
    for seed in c.SEEDS:
        sequences=[[row['accepted_mask'] for row in lookup[seed,method]['records']]
                   for method in ('second_order','adaptive_reset','adaptive_persistent')]
        assert sequences[0]==sequences[1]==sequences[2];identical+=1
        run=lookup[seed,'adaptive_persistent'];assert run['final_state']['expanded_tasks']==[4]
        assert all(row['order_after']==(1 if row['index']<4 else 2) for row in run['records'])
        switches.append(dict(seed=seed,expanded_at=4,permanent_order_2=True))
    result=dict(status='passed',pair_targets=pair_checks,triple_targets=triple_checks,
                first_order_pair_successes=sum(row['first_order_correct'] for row in pair_checks),
                second_order_triple_misses=sum(not row['second_order_correct'] for row in triple_checks),
                independent_scalar_coefficient_outputs=scalar_outputs,
                same_accepted_sequence_second_reset_persistent=identical,persistent_switches=switches,
                scope='Post-run explanatory audit only. Target masks are evaluator-only, never supplied to learning.')
    (c.RESULTS/'coverage_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode({k:value for k,value in result.items() if k not in ('pair_targets','triple_targets','persistent_switches')}))

if __name__=='__main__':main()
