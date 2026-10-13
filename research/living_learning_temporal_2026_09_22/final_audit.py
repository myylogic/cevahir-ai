"""Artifact integrity, independent metric arithmetic and report claim checks."""
from fractions import Fraction
import json
import math
import platform
import re
from urllib.parse import unquote
import numpy as np
import temporal_learning as t
import analyze_results
import verify_learning_state
import verify_temporal


def scalar_metrics(predictions, labels):
    n = len(labels)
    mse = math.fsum((p-y)**2 for p,y in zip(predictions,labels))/n
    accuracy = math.fsum(.5 if abs(p-.5) <= 1e-12 else float((p>.5)==bool(y))
                         for p,y in zip(predictions,labels))/n
    return dict(n=n,mse=mse,accuracy=accuracy)


def main():
    data=json.loads((t.RESULTS/'temporal_learning.json').read_text(encoding='utf-8'))
    summary=json.loads((t.RESULTS/'summary.json').read_text(encoding='utf-8'))
    verification=json.loads((t.RESULTS/'verification.json').read_text(encoding='utf-8'))
    theory=json.loads((t.RESULTS/'learning_state_verification.json').read_text(encoding='utf-8'))
    assert data['source_sha256']==t.sha(t.HERE/'temporal_learning.py')
    assert data['protocol_sha256']==t.sha(t.PROTOCOL)
    assert summary==analyze_results.summarize(data)
    assert theory==verify_learning_state.verify()
    assert verification['full_rerun']['returncode']==0
    assert len(verification['fresh_process_resume'])==21
    assert all(r['exact_equal'] for r in verification['fresh_process_resume'])
    assert verification['recorded']['prequential_records']==516096
    assert verification['structural']['enumerated_tree_prediction_checks']==168
    assert verification['structural']['scalar_automaton_outputs']==32640

    lives={life['seed']:life for life in data['lives']}
    metric_checks=0
    for row in data['runs']:
        # Teacher calculation independently restated for metric verification.
        words=lives[row['seed']]['training']
        labels=[int(word[-1]) if row['task']=='last_bit' else
                int(word[-5]) if row['task']=='delay_5' else word.count('1')%2 for word in words]
        for period,s in (('all_prequential',slice(None)),('first_128',slice(0,128)),('last_128',slice(-128,None))):
            expected=scalar_metrics(row['predictions'][s],labels[s])
            for key,value in expected.items(): assert abs(row[period][key]-value)<1e-14
            metric_checks+=1
    # Exact-rational expected predictions against the actual constant learner.
    learner=t.Learner('kt_constant');counts=[0,0];kt_checks=0
    for label in [0,1,1,0,1,1,1,0]*10:
        expected=Fraction(2*counts[1]+1,2*(sum(counts)+1))
        assert t.query(learner,'10101010')==float(expected)
        learner.observe_label(label);counts[label]+=1;kt_checks+=1

    groups={(g['task'],g['method']):g for g in summary['groups']}
    for task,leaves,depth in [('last_bit',2,1),('delay_5',32,5),('parity',1,0)]:
        structure=groups[task,'map_tree_8']['checkpoints'][-1]['structure']
        assert structure['selected_leaves']['histogram']=={str(leaves):24}
        assert structure['selected_max_depth']['histogram']=={str(depth):24}
    for task in ('last_bit','parity'):
        final=groups[task,'two_state']['checkpoints'][-1]['long_test']
        assert final['lives_accuracy_one']==24 and final['mse']['max']==0
    for task in ('last_bit','delay_5'):
        assert groups[task,'map_tree_8']['checkpoints'][-1]['long_test']['lives_accuracy_one']==24
        pair=next(p for p in summary['paired'] if (p['task'],p['left'],p['right'])==(task,'map_tree_8','suffix_8'))
        assert pair['left_better']==24
    delay=groups['delay_5','two_state']['checkpoints']
    assert delay[0]['long_test']['mse']['mean']==.25
    assert abs(delay[-1]['long_test']['mse']['mean']-.47058360439802027)<1e-14
    for group in summary['groups']:
        resources=group['final_resources']
        if group['method'] in ('map_tree_8','mixture_tree_8'):
            assert resources['counter_updates']['min']==resources['counter_updates']['max']==9216
            assert resources['numeric_array_bytes']['mean']==20951
        if group['method']=='two_state':
            assert resources['candidate_label_checks']['mean']==65536
            assert resources['numeric_array_bytes']['mean']==1040
        assert resources['raw_completed_episodes']['max']==0

    docs=t.PROTOCOL.parent;links=0
    for path in [t.HERE/'README.md',*docs.glob('*.md')]:
        for href in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z][\w+.-]*:',href) or href.startswith('#'): continue
            target=(path.parent/unquote(href.split('#',1)[0])).resolve()
            assert target.exists() or target==t.RESULTS/'final_audit.json',(str(path),href)
            links+=1
    artifacts=sorted([*t.HERE.glob('*.py'),t.HERE/'README.md',*docs.glob('*.md'),
                      *(p for p in t.RESULTS.iterdir() if p.is_file() and p.name!='final_audit.json')])
    result=dict(date='2026-09-22',status='passed',preservation=verify_temporal.preservation(),
                environment=dict(python=platform.python_version(),numpy=np.__version__),
                trajectories=504,paired_lives=24,prequential_predictions=516096,snapshots=2520,
                complete_exact_replays=1,fresh_process_resume_checks=21,
                independent_scalar_metric_checks=metric_checks,exact_fraction_kt_checks=kt_checks,
                enumerated_tree_checks=168,scalar_automaton_outputs=32640,
                auxiliary_memory_witness_pairs=theory['pairwise_distinguished'],
                source_protocol_result_hashes_match=True,summary_recomputed=True,
                report_key_claims_checked=True,local_links_checked=links,
                artifacts_sha256={p.relative_to(t.ROOT).as_posix():t.sha(p) for p in artifacts},
                scope='Local finite experiments and conditional proofs. Supplied memory families; no new fundamental algorithm, biological mechanism or AGI claim. No commit or push.')
    (t.RESULTS/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'}))


if __name__=='__main__': main()
