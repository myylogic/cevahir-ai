"""Post-run checks for unobserved distinctions and evidence accounting."""
import json
import distinction_learning as c
import verify_distinctions as v

def diagnose(data):
    worlds={(w['seed'],w['family']):w for w in data['worlds']}
    failures=[];cache_checks=0;initial_checks=0;exact_minimal=0;zero_test_wrong=0;split_lives=0
    cache_errors={};observed_max_env=0;observed_max_work=0;unseen_long_tests=0;maximum_query_length=0
    for run in data['runs']:
        world=worlds[run['seed'],run['family']];target=world['target']
        # Recreate only the small initial hypothesis, before any passive observation.
        work=c.Work();environment=c.Environment(target);initial=c.Learner(environment.query,work,run['method'])
        initial.stabilize()
        assert c.evaluate(initial,world)==run['initial']
        assert run['records'][0]['predictions']==[v.scalar(initial.model,w) for w in world['stream'][:4]]
        initial_checks+=4;previous=initial.model
        for record in run['records']:
            state=record['state_after'];model=state['learner']['model'];known=state['learner']['known']
            errors=sum(v.scalar(model,w)!=value for w,value in known.items())
            assert errors==record['evaluation']['known_errors']
            if state['frozen']:assert model==previous
            previous=model
        last=run['records'][-1];state=run['final_state'];model=state['learner']['model'];cache=state['environment']['cache']
        assert not set(world['tests']).intersection(cache)
        unseen_long_tests+=len(world['tests'])
        maximum_query_length=max(maximum_query_length,max(map(len,cache)))
        cached_error=sum(v.scalar(model,w)!=value for w,value in cache.items());cache_checks+=len(cache)
        cache_errors.setdefault(run['method'],0);cache_errors[run['method']]+=cached_error
        assert cached_error==last['evaluation']['known_errors'] # No aborted/frozen path in this actual run.
        observed_max_env=max(observed_max_env,state['environment']['units']);observed_max_work=max(observed_max_work,state['work']['used'])
        if last['evaluation']['exact']['equal']:
            minimal=len(set(v.refinement_partition(target)))
            assert len(model['outputs'])==minimal;exact_minimal+=1
        if run['method']=='adaptive':
            split_lives+=len(state['learner']['splits'])>0
            if not last['evaluation']['exact']['equal']:
                wrong=last['evaluation']['exact']['witness']
                assert wrong not in cache and v.scalar(model,wrong)!=v.scalar(target,wrong)
                positive=sum(cache.values());zero_test_wrong+=last['evaluation']['long_errors']==0
                failures.append(dict(seed=run['seed'],family=run['family'],states=len(model['outputs']),
                                     target_minimal_states=len(set(v.refinement_partition(target))),
                                     positive_cached_labels=positive,cache_size=len(cache),
                                     counterexample=wrong,long_errors=last['evaluation']['long_errors'],
                                     witness_unobserved=True,known_errors=cached_error))
    rare=[row for row in failures if row['family']=='cycle_reset']
    assert len(rare)==14 and all(row['positive_cached_labels']==0 and row['states']==1 for row in rare)
    assert len(failures)==15 and zero_test_wrong==4
    # The actual implementation rejects a new uncached query at the environment cap;
    # cached reads are still available. This is a boundary audit, not a performance trial.
    target=dict(start=0,transitions=[[0,0]],outputs=[0]);environment=c.Environment(target)
    assert environment.query('')==0;environment.units=c.ENV_LIMIT
    assert environment.query('')==0
    try:environment.query('0')
    except c.Budget as error:assert str(error)=='environment'
    else:raise AssertionError('uncached query at cap')
    return dict(status='passed',independent_first_block_predictions=initial_checks,
                independent_final_cache_predictions=cache_checks,cache_errors_by_method=cache_errors,
                independently_minimal_exact_final_models=exact_minimal,
                adaptive_lives_with_new_tests=split_lives,adaptive_failures=failures,
                all_zero_evidence_cycle_failures=len(rare),wrong_adaptive_models_with_zero_long_test_errors=zero_test_wrong,
                maximum_environment_units=observed_max_env,maximum_learning_work=observed_max_work,
                environment_cap_is_uncached_query_admission=True,
                no_actual_cap_reached=observed_max_env<c.ENV_LIMIT and observed_max_work<c.INTERNAL_LIMIT,
                final_long_test_words_unseen=unseen_long_tests,maximum_observed_word_length=maximum_query_length,
                scope='Post-run evaluator-only diagnosis; no labels or counterexamples returned to learner, no primary source or protocol edits.')

def main():
    data=json.loads((c.RESULTS/'distinction_learning.json').read_text(encoding='utf-8'));result=diagnose(data)
    (c.RESULTS/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(c.encode(result))

if __name__=='__main__':main()
