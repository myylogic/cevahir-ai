"""Independent numerical checks, fresh-process continuation and preservation.

--rerun additionally reruns both complete experiments against their stored data.
Only this experiment's verification JSON is written.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import geometry as g
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).parent / "results"


def numerical_checks():
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(993)
    theta = rng.normal(0., .4, g.PARAMS)
    x = rng.normal(size=(7, 2))
    pred, jac = g.values_jacobian(theta, x)
    finite = np.empty_like(jac)
    eps = 1e-6
    for j in range(g.PARAMS):
        delta = np.zeros(g.PARAMS); delta[j] = eps
        finite[:, j] = (g.forward(theta+delta, x)-g.forward(theta-delta, x))/(2*eps)
    t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float64)
    def independent(t):
        return torch.tanh(tx @ t[8:24].reshape(8,2).T+t[24:32]) @ t[:8]+t[-1]
    auto = torch.autograd.functional.jacobian(independent, t).detach().numpy()
    y = rng.normal(size=len(x))
    loss = .5*((independent(t)-torch.tensor(y))**2).mean()
    torch_grad = torch.autograd.grad(loss, t)[0].detach().numpy()
    basis, _ = g.row_basis(jac, 1e-8)
    p = np.eye(g.PARAMS)-basis.T@basis
    # Affine output case: row-null projection preserves outputs at finite steps.
    matrix = rng.normal(size=(3,7)); v=rng.normal(size=7);step=rng.normal(size=7)
    affine_basis, _ = g.row_basis(matrix, 1e-8)
    projected=g.project(step,affine_basis)
    results={"jacobian_finite_difference_max_error":float(np.max(np.abs(finite-jac))),
             "jacobian_autograd_max_error":float(np.max(np.abs(auto-jac))),
             "loss_gradient_autograd_max_error":float(np.max(np.abs(torch_grad-g.gradient(theta,x,y)))),
             "projector_idempotence_max_error":float(np.max(np.abs(p@p-p))),
             "projector_orthogonality_max_error":float(np.max(np.abs(jac@p))),
             "affine_finite_step_output_max_error":float(np.max(np.abs(matrix@(v+projected)-matrix@v))),
             "torch":torch.__version__}
    for key,value in results.items():
        if key.endswith("error"): assert value < (1e-8 if "finite_difference" in key else 1e-10), (key,value)
    return results


def continuation_checks():
    world=g.make_world(999)
    results=[]
    with tempfile.TemporaryDirectory(prefix='cevahir-geometry-verify-') as temp:
        for mode in g.MODES:
            obj=g.Learner(world['theta'],mode,.12,world['old_x'][:16],world['old_y'][:16])
            for x,y in zip(world['new_x'][:23],world['new_y'][:23]):obj.step(x,y)
            payload={'state':obj.snapshot(),'batches':[(x.tolist(),y.tolist()) for x,y in zip(world['new_x'][23:40],world['new_y'][23:40])]}
            expected_pred=[obj.step(np.array(x),np.array(y)).tolist() for x,y in payload['batches']]
            expected={'predictions':expected_pred,'state':obj.snapshot()}
            ip=Path(temp)/(mode+'-input.json');op=Path(temp)/(mode+'-output.json')
            ip.write_text(json.dumps(payload),encoding='utf-8')
            call=subprocess.run([sys.executable,str(Path(g.__file__)),'--resume-input',str(ip),'--resume-output',str(op)],capture_output=True,text=True)
            assert call.returncode==0,(mode,call.stderr)
            actual=json.loads(op.read_text(encoding='utf-8'))
            g.compare(expected,actual)
            results.append({'mode':mode,'continued_batches':17,'exact_json_equality':expected==actual,
                            'prediction_max_error':float(np.max(np.abs(np.array(expected_pred)-actual['predictions'])))})
    return results


def result_checks():
    primary=json.loads((RESULTS/'geometry.json').read_text())
    post=json.loads((RESULTS/'posthoc_controls.json').read_text())
    assert len(primary['runs'])==864 and len(post['runs'])==216
    assert primary['source_sha256']==hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest()
    assert post['primary_source_sha256']==primary['source_sha256']
    for r in primary['runs']:
        if r['mode'].endswith('_guard'):
            assert r['max_anchor_drift_mse']<=g.DRIFT_BUDGET+1e-15
        if r['mode']=='frozen':assert r['anchor_drift_mse']==0.
    for r in post['runs']:
        if r['mode'].endswith('_guard'):
            assert r['max_anchor_label_mse_increase']<=g.DRIFT_BUDGET+1e-15
    assert all(r['initial_max_prediction_difference']==0 for r in primary['continuation_probes'])
    # Warmup, teacher and input layout independently reconstruct one sampled target.
    world=g.make_world(3)
    assert max(g.mse(world['teacher'],x,y) for x,y in world['tests'].values())==0.
    return {'primary_runs':864,'posthoc_runs':216,'continuation_initial_equality':True,
            'output_guard_budget_violations':0,'label_loss_guard_budget_violations':0,
            'teacher_exactly_in_model_class':True}


def preservation():
    baseline=json.loads((RESULTS/'preservation_baseline.json').read_text())
    changed=[path for path,digest in baseline['files'].items() if not (ROOT/path).is_file() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head==baseline['base_head']
    return {'files_byte_exact':len(baseline['files']),'changed':changed,'head_unchanged':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rerun',action='store_true');args=parser.parse_args()
    record={'date':'2026-09-21','numerical':numerical_checks(),'fresh_process_continuation':continuation_checks(),
            'result_contracts':result_checks(),'exact_witnesses':g.exact_witnesses(),'preservation':preservation()}
    if args.rerun:
        def rerun(name):
            call=subprocess.run([sys.executable,str(Path(__file__).parent/name)],capture_output=True,text=True)
            assert call.returncode==0,(name,call.stdout,call.stderr)
            return {'program':name,'returncode':call.returncode,'stdout':call.stdout.strip()}
        with ThreadPoolExecutor(max_workers=2) as pool:record['complete_reruns']=list(pool.map(rerun,('geometry.py','posthoc_controls.py')))
    record['scope']='Independent numerical derivative checks and fresh-process resume, plus optional same-code complete reproduction. This is not external peer review or an independent data replication.'
    (RESULTS/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'passed','preserved_files':record['preservation']['files_byte_exact'],
                      'fresh_process_methods':len(record['fresh_process_continuation']), 'complete_reruns':len(record.get('complete_reruns',[]))}))


if __name__=='__main__':main()
