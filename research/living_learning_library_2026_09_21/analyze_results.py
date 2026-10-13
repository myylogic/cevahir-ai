"""Recompute paired descriptive summaries from immutable experiment records."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def run():
    path=HERE/'results/library_learning.json'
    primary=json.loads(path.read_text(encoding='utf-8'));rows=primary['runs']
    def cells(history,budget,regime,method):
        return {(r['seed'],r['task']):r for r in rows if
                (r['history'],r['budget'],r['regime'],r['method'])==(history,budget,regime,method)}
    paired=[];more=[]
    for history in (6,12):
        for budget in (512,2048):
            for regime in ('related','shifted'):
                a=cells(history,budget,regime,'learned_library');b=cells(history,budget,regime,'primitive_only')
                common=[k for k in a if a[k]['exact'] and b[k]['exact']]
                paired.append(dict(history=history,budget=budget,regime=regime,
                    gained=sum(a[k]['exact'] and not b[k]['exact'] for k in a),
                    lost=sum(b[k]['exact'] and not a[k]['exact'] for k in a),
                    seed_success_count_differences=[sum(int(a[k]['exact'])-int(b[k]['exact']) for k in a if k[0]==s) for s in range(12)],
                    jointly_solved=len(common),
                    common_mean_primitive_steps=statistics.fmean(b[k]['charged_vector_steps'] for k in common) if common else None,
                    common_mean_learned_steps=statistics.fmean(a[k]['charged_vector_steps'] for k in common) if common else None))
    for budget in (512,2048):
        for regime in ('related','shifted'):
            a=cells(6,budget,regime,'learned_library');b=cells(12,budget,regime,'learned_library')
            gained=[k for k in a if not a[k]['exact'] and b[k]['exact']]
            lost=[k for k in a if a[k]['exact'] and not b[k]['exact']]
            more.append(dict(budget=budget,regime=regime,gained=gained,lost=lost,
                             first_loss_records={'history6':a[lost[0]],'history12':b[lost[0]]} if lost else None))
    costs=[sum(x['result']['charged_vector_steps'] for x in v['acquisition']) for v in primary['lives']]
    state_bytes=[v['state_json_bytes']['12'] for v in primary['lives']]
    return dict(primary_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                learned_vs_primitive=paired,history_6_to_12=more,
                warmup_cost=dict(min=min(costs),mean=statistics.fmean(costs),max=max(costs)),
                state_json_bytes_history12=dict(min=min(state_bytes),mean=statistics.fmean(state_bytes),max=max(state_bytes)),
                mean_history12_substring_windows=statistics.fmean(v['states']['12']['mining']['substring_windows_scanned'] for v in primary['lives']),
                false_accepts=sum(r['status']=='accepted' and not r['exact'] for r in rows),
                main_max_queue=max(r['costs']['max_queue'] for r in rows),
                scope='Descriptive paired counts over twelve synthetic lives; no multiple-comparison or population-generalization claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    result=json.loads(json.dumps(run()));path=HERE/'results/paired_summary.json'
    if args.write:path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:assert result==json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps(dict(status='passed',paired_groups=len(result['learned_vs_primitive']),history_comparisons=len(result['history_6_to_12']))))


if __name__=='__main__':main()
