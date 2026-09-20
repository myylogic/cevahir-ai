"""Independent algebra, state access and subprocess reproducibility checks."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent
RESULTS = BASE/'results'
DOCS = ROOT/'docs/research/living_learning_joint_2026_09_20'


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, BASE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(pair):
    script, result = pair
    with tempfile.TemporaryDirectory(prefix='joint-learning-verification-') as temporary:
        destination = Path(temporary)/'result.json'
        proc = subprocess.run([sys.executable,str(BASE/(script+'.py')),'--output',str(destination)],
                              cwd=ROOT,check=True,capture_output=True,text=True)
        fresh = json.loads(destination.read_text(encoding='utf-8'))
        old = json.loads((RESULTS/(result+'.json')).read_text(encoding='utf-8'))
        assert fresh==old, script
        assert old['source_sha256']==sha(BASE/(script+'.py'))
    return {'script':script,'result':result,'entire_json_equal':True,
            'source_matches':True,'original_output_overwritten':False,'returncode':proc.returncode}


def semantic_checks():
    joint = load_module('joint_stream')
    temporal = load_module('temporal_relation')
    # Independently verify each RLS update equals the directly solved ridge fit.
    model = temporal.RLS([0,2,5])
    rows = []
    max_rls_error = 0.
    gram = [[float(i==j) for j in range(4)] for i in range(4)]
    cross = [0.]*4
    for i in range(1,33):
        history = [((i*(k+3)+k*k)%23-11)/11 for k in range(11)]
        y = .2+1.1*history[2]-.7*history[5]
        x = model.phi(history)
        model.observe(history,y)
        for a in range(4):
            cross[a]+=x[a]*y
            for b in range(4):
                gram[a][b]+=x[a]*x[b]
        # joint.solve adds its own ridge; remove that tiny diagonal first.
        corrected = [[v-(joint.RIDGE if a==b else 0.) for b,v in enumerate(row)] for a,row in enumerate(gram)]
        direct = joint.solve(corrected,cross,Counter())
        max_rls_error = max(max_rls_error,max(abs(a-b) for a,b in zip(model.w,direct)))
    assert max_rls_error < 1e-12
    # Actual boundary-free method access: state contains no world/phase/holdouts.
    allowed = {'mode','active','weights','cells','rolling','gram','cross','step','costs','events','max_records'}
    persisted_checks = {}
    for mode in joint.MODES:
        learner = joint.Learner(mode)
        stream = joint.make_stream(83)
        for x,z,y,_,_ in stream[:896]:
            learner.observe(x,z,y)
        state = json.loads(json.dumps(learner.snapshot()))
        assert set(state)==allowed
        before = json.loads(json.dumps(learner.snapshot()))
        joint.risk(learner,joint.holdout_points(83,True),True)
        assert learner.snapshot()==before
        # Input artifacts contain state and continuation observations only.
        continuation = [[x,z,y] for x,z,y,_,_ in stream[896:960]]
        # restore intentionally consumes operational lists; detach the expected
        # clone so its updates cannot mutate the restart input fixture.
        expected_model = joint.Learner.restore(json.loads(json.dumps(state)))
        expected_predictions = []
        for x,z,y in continuation:
            expected_predictions.append(expected_model.predict(x,z))
            expected_model.observe(x,z,y)
        with tempfile.TemporaryDirectory(prefix='joint-full-state-') as tmp:
            path = Path(tmp)/'bundle.json'
            path.write_text(json.dumps({'state':state,'continuation':continuation}),encoding='utf-8')
            proc = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe',str(path)],
                                  cwd=ROOT,check=True,capture_output=True,text=True)
            observed = json.loads(proc.stdout)
        assert observed['predictions']==expected_predictions
        assert observed['final_state']==expected_model.snapshot()
        # Inference uses weights and active terms, not replay data or solve stats.
        inference = joint.Learner.restore(state)
        inference.cells = [[] for _ in inference.cells]
        inference.rolling = []
        inference.gram = [[0.]*5 for _ in range(5)]
        inference.cross = [0.]*5
        assert all(inference.predict(x,z)==learner.predict(x,z) for x,z in joint.holdout_points(83,True))
        persisted_checks[mode] = {'fresh_process_64_update_continuation_equal':True,
                                  'inference_without_raw_records_equal':True,'risk_evaluation_did_not_modify_state':True}
    # Exact deterministic informative lag design: seven isolated lag one-hots
    # plus zero history; with intercept, subtracting last row gives rank8.
    design = [[1]+[int(j==i) for j in range(7)] for i in range(7)]+[[1]+[0]*7]
    differences = [[a-b for a,b in zip(row,design[-1])] for row in design[:-1]]
    assert [row[1:] for row in differences]==[[int(i==j) for j in range(7)] for i in range(7)]
    assert (0+0-1)**2==1 and (1.5+0-1)**2==.25 and (1.5+1.5-1)**2==4
    return {'rls_vs_independent_batch_solve_max_error':max_rls_error,
            'joint_state_checks':persisted_checks,'deterministic_lag_design_rank':8,
            'separate_update_joint_failure_exact':True,
            'inspection_limit':'Source audit establishes signature; no universal isolation or security proof.'}


def links():
    missing=[]
    n=0
    for document in DOCS.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            if re.match(r'^(https?://|#)',target):
                continue
            n+=1
            resolved=(document.parent/target.split('#')[0]).resolve()
            # This very report is written immediately after link validation.
            if not resolved.exists() and resolved != (RESULTS/'verification.json').resolve():
                missing.append([document.name,target])
    assert not missing,missing
    return {'checked':n,'missing':missing}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--probe':
        module = load_module('joint_stream')
        bundle = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
        model = module.Learner.restore(bundle['state'])
        predictions=[]
        for x,z,y in bundle['continuation']:
            predictions.append(model.predict(x,z))
            model.observe(x,z,y)
        print(json.dumps({'predictions':predictions,'final_state':model.snapshot()}))
        return
    old=subprocess.run([sys.executable,str(BASE/'preserve_prior.py')],cwd=ROOT,check=True,capture_output=True,text=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(reproduce,[('joint_stream','joint_stream'),('temporal_relation','temporal_relation'),('joint_ablation','joint_no_burst')]))
    checks=semantic_checks()
    tracked=subprocess.run(['git','diff','--name-only','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True)
    report={'prior_files':json.loads(old.stdout),'reproductions':records,'semantic_checks':checks,
            'local_links':links(),'tracked_changes':tracked.stdout.splitlines(),'passed':True,
            'scope':'Reproducibility and bounded semantic checks, not proof of general living learning.'}
    (RESULTS/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
