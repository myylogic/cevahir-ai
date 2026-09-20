"""Post-result diagnostic: did the corruption burst scaffold later growth?

The original joint experiment is left byte-for-byte intact. This ablation was
chosen after observing 16/24 activations during the burst/recovery interval.
It removes the structured label burst, retaining targets, inputs and noise.
It is a mechanism check, not new independent confirmatory evidence.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results'/'joint_no_burst.json')
    args = parser.parse_args()
    path = Path(__file__).parent/'joint_stream.py'
    spec = importlib.util.spec_from_file_location('joint_for_ablation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_generator = module.make_stream

    def no_burst(seed):
        changed = []
        for t, (x,z,label,clean,phase) in enumerate(original_generator(seed)):
            if 3*module.PHASE_LENGTH <= t < 3*module.PHASE_LENGTH+32:
                label += clean-module.target(x,z,True)
            changed.append((x,z,label,clean,phase))
        return changed

    module.make_stream = no_burst
    runs = [module.one_seed(seed) for seed in range(module.SEEDS)]
    baseline = json.loads((path.parent/'results'/'joint_stream.json').read_text(encoding='utf-8'))
    changes = []
    for altered, original in zip(runs,baseline['runs']):
        a = altered['methods']['growth_coverage']
        b = original['methods']['growth_coverage']
        steps_a = [e['step'] for e in a['candidate_activation_events'] if e['term']=='max(x,0)*z']
        steps_b = [e['step'] for e in b['candidate_activation_events'] if e['term']=='max(x,0)*z']
        changes.append({'seed':altered['seed'],'hinge_activation_no_burst':steps_a,
                        'hinge_activation_original':steps_b,
                        'local_change_service_no_burst':a['phase_service_clean_mse']['right_local_change'],
                        'local_change_service_original':b['phase_service_clean_mse']['right_local_change'],
                        'terminal_right_no_burst':a['terminal_right_mse'],
                        'terminal_right_original':b['terminal_right_mse']})
    report = {'experiment':'joint_life_without_structured_label_burst',
              'status':'post-result diagnostic, no parameter tuning',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'original_result_sha256':hashlib.sha256((path.parent/'results'/'joint_stream.json').read_bytes()).hexdigest(),
              'changes':'Only corrupt 32-label burst removed; same numeric noise up to roundoff, inputs, target drift, methods and heldouts.',
              'summary':module.aggregate(runs),'checks':module.validate(runs),'runs':runs,
              'paired_activation_and_later_learning':changes}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),
        'growth_hinge_activated_pre_real_change':sum(any(t<=1024 for t in row['hinge_activation_no_burst']) for row in changes),
        'growth_mean_local_change_service_mse':report['summary']['methods']['growth_coverage']['phase_service_clean_mse']['right_local_change'],
        'growth_terminal_right_mse':report['summary']['methods']['growth_coverage']['terminal_right_mse']},indent=2))


if __name__=='__main__':
    main()
