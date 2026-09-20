"""Learn a two-dimensional update geometry from live prediction experience.

Known family of projected symmetric-gain updates, not a new general optimizer.
Exact forward sensitivities within each fixed-gain block; truncated at block
boundaries. Training outcomes alone select meta-parameters, never clean labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

SEEDS=24
TRAIN_STEPS=16384
TEST_STEPS=1536
TASKS_PER_REGIME=4
BLOCK=64
META_RATE=.1
MIN_GAIN=.002
MAX_GAIN=.6
INITIAL_GAIN=.08
NOISE_SD=.15
FAST_SD=.04
SLOW_SD=.001
REGIMES=['same_geometry','reversed_geometry','isotropic_change','cold_restart']
SCALARS=[.004,.008,.016,.032,.064,.128,.256,.512]


def dot(x,y): return x[0]*y[0]+x[1]*y[1]

def mv(p,x): return [p[0]*x[0]+p[2]*x[1],p[2]*x[0]+p[1]*x[1]]

def rotate(x,angle):
    c,s=math.cos(angle),math.sin(angle)
    return [c*x[0]-s*x[1],s*x[0]+c*x[1]]

def gain_from_eigen(high,low,angle):
    c,s=math.cos(angle),math.sin(angle)
    return [high*c*c+low*s*s,high*s*s+low*c*c,(high-low)*s*c]

def rotate_gain(p,angle):
    c,s=math.cos(angle),math.sin(angle)
    a,d,b=p
    return [c*c*a+s*s*d-2*c*s*b,s*s*a+c*c*d+2*c*s*b,c*s*(a-d)+(c*c-s*s)*b]

def project(p):
    a,d,b=p
    mid=(a+d)/2
    spread=math.hypot((a-d)/2,b)
    hi,lo=mid+spread,mid-spread
    clipped=hi>MAX_GAIN or lo<MIN_GAIN
    hi=max(MIN_GAIN,min(MAX_GAIN,hi))
    lo=max(MIN_GAIN,min(MAX_GAIN,lo))
    angle=.5*math.atan2(2*b,a-d)
    return gain_from_eigen(hi,lo,angle),clipped


class Learner:
    def __init__(self,p=None,adaptive=False):
        self.p=list(p if p is not None else [INITIAL_GAIN,INITIAL_GAIN,0.])
        self.w=[0.,0.]
        self.adaptive=adaptive
        self.sens=[[0.,0.] for _ in range(3)]
        self.gradient=[0.,0.,0.]
        self.position=0
        self.updates=0
        self.meta_updates=0
        self.projections_clipped=0

    def predict(self,x): return dot(self.w,x)

    def observe(self,x,y):
        e=y-self.predict(x)
        px=mv(self.p,x)
        if self.adaptive:
            basis_x=[[x[0],0.],[0.,x[1]],[x[1],x[0]]]
            new=[]
            for j,s in enumerate(self.sens):
                xs=dot(x,s)
                self.gradient[j]-=e*xs
                new.append([s[k]-px[k]*xs+basis_x[j][k]*e for k in range(2)])
            self.sens=new
        self.w=[self.w[k]+px[k]*e for k in range(2)]
        self.position+=1
        self.updates+=1
        if self.adaptive and self.position==BLOCK:
            step=META_RATE/BLOCK
            # Independent off-diagonal coordinate represents two entries.
            proposal=[self.p[0]-step*self.gradient[0],self.p[1]-step*self.gradient[1],
                      self.p[2]-.5*step*self.gradient[2]]
            self.p,clipped=project(proposal)
            self.projections_clipped+=int(clipped)
            self.meta_updates+=1
            self.sens=[[0.,0.] for _ in range(3)]
            self.gradient=[0.,0.,0.]
            self.position=0

    def snapshot(self):
        return {k:getattr(self,k) for k in ('p','w','adaptive','sens','gradient','position','updates','meta_updates','projections_clipped')}

    @classmethod
    def restore(cls,state):
        obj=cls()
        for k,v in json.loads(json.dumps(state)).items(): setattr(obj,k,v)
        return obj


def stream(seed,length,angle,isotropic=False,cold=False):
    rng=random.Random(seed)
    target=[rng.gauss(0,.75),rng.gauss(0,.75)] if cold else [0.,0.]
    for t in range(length):
        drift=rotate([rng.gauss(0,FAST_SD),rng.gauss(0,FAST_SD if isotropic else SLOW_SD)],angle)
        target=[target[k]+drift[k] for k in range(2)]
        direction=rng.uniform(-math.pi,math.pi)
        x=[math.cos(direction),math.sin(direction)]
        clean=dot(target,x)
        y=clean+rng.gauss(0,NOISE_SD)
        yield x,y,clean,target[:]


def grid():
    candidates=[{'id':f'scalar_{a}','p':[a,a,0.],'scalar':True} for a in SCALARS]
    for k in range(8):
        for hi,lo in ((.12,.012),(.36,.004),(.36,.04)):
            candidates.append({'id':f'angle_{k}_gains_{hi}_{lo}',
                               'p':gain_from_eigen(hi,lo,k*math.pi/8),'scalar':False})
    return candidates


def source_life(seed):
    angle=random.Random(130000+seed).uniform(0,math.pi)
    learner=Learner(adaptive=True)
    candidates=grid()
    comparator_weights=[[0.,0.] for _ in candidates]
    comparator_losses=[0.]*len(candidates)
    service=0.
    checkpoints=[]
    for t,(x,y,clean,target) in enumerate(stream(230000+seed,TRAIN_STEPS,angle),1):
        service+=(learner.predict(x)-clean)**2/TRAIN_STEPS
        learner.observe(x,y)
        for k,candidate in enumerate(candidates):
            w=comparator_weights[k]
            e=y-dot(w,x)
            comparator_losses[k]+=e*e/TRAIN_STEPS
            px=mv(candidate['p'],x)
            comparator_weights[k]=[w[j]+px[j]*e for j in range(2)]
        if t in (64,512,2048,8192,TRAIN_STEPS):
            checkpoints.append({'sample':t,'p':learner.p[:],
                                'source_clean_service_so_far':service*TRAIN_STEPS/t})
    best_scalar=min(range(len(SCALARS)),key=lambda k:comparator_losses[k])
    best_full=min(range(len(candidates)),key=lambda k:comparator_losses[k])
    fast_axis=rotate([1.,0.],angle)
    slow_axis=rotate([0.,1.],angle)
    return {'seed':seed,'world_fast_direction':angle,'learned_p':learner.p[:],
            'gain_along_actual_fast_axis':dot(fast_axis,mv(learner.p,fast_axis)),
            'gain_along_actual_slow_axis':dot(slow_axis,mv(learner.p,slow_axis)),
            'source_clean_service_mse':service,'checkpoints':checkpoints,
            'train_selected_scalar':{**candidates[best_scalar],'source_observed_mse':comparator_losses[best_scalar]},
            'train_selected_matrix':{**candidates[best_full],'source_observed_mse':comparator_losses[best_full]},
            'candidate_source_observed_mse':[{**c,'observed_mse':loss} for c,loss in zip(candidates,comparator_losses)],
            'cost':{'source_labels':TRAIN_STEPS,'meta_base_updates':learner.updates,
                    'sensitivity_vector_updates':3*learner.updates,'meta_blocks':learner.meta_updates,
                    'clipped_projections':learner.projections_clipped,
                    'parallel_grid_base_updates':len(candidates)*TRAIN_STEPS,
                    'grid_candidates':len(candidates)},
            'transfer_payload':{'p':learner.p[:]}}


def methods_for(source):
    p=source['learned_p']
    isotropic=(p[0]+p[1])/2
    return {
        'initial_fixed':Learner(),
        'learned_frozen':Learner(p),
        'learned_continue':Learner(p,True),
        'fresh_meta':Learner(adaptive=True),
        'diagonal_only':Learner([p[0],p[1],0.]),
        'isotropic_same_trace':Learner([isotropic,isotropic,0.]),
        'rotated_90':Learner(rotate_gain(p,math.pi/2)),
        'train_selected_scalar':Learner(source['train_selected_scalar']['p']),
        'train_selected_matrix':Learner(source['train_selected_matrix']['p']),
    }


def task_run(source,regime,index):
    angle=source['world_fast_direction']+(math.pi/2 if regime=='reversed_geometry' else 0)
    examples=list(stream(330000+source['seed']*37+index,TEST_STEPS,angle,
                         isotropic=regime=='isotropic_change',cold=regime=='cold_restart'))
    models=methods_for(source)
    totals={name:[0.,0.,0.,0.] for name in models}
    for t,(x,y,clean,target) in enumerate(examples):
        if t==0: assert all(model.predict(x)==0. for model in models.values())
        for name,model in models.items():
            loss=(model.predict(x)-clean)**2
            totals[name][0]+=loss/TEST_STEPS
            if t<64: totals[name][1]+=loss/64
            if 64<=t<512: totals[name][2]+=loss/(512-64)
            if t>=512: totals[name][3]+=loss/(TEST_STEPS-512)
            model.observe(x,y)
    records={}
    for name,model in models.items():
        # For uniform unit-circle queries, exact expected final clean risk is
        # 1/2 ||w-w*||². Query labels never enter learning.
        records[name]={'service_mse':totals[name][0],'first64_mse':totals[name][1],
                       'samples65_512_mse':totals[name][2],'samples513_end_mse':totals[name][3],
                       'final_uniform_query_mse':.5*sum((model.w[j]-examples[-1][3][j])**2 for j in range(2)),
                       'final_p':model.p,'base_updates':model.updates,'meta_blocks':model.meta_updates,
                       'sensitivity_vector_updates':3*model.updates if model.adaptive else 0,
                       'p_change_frobenius':math.sqrt(sum((a-b)**2*c for a,b,c in zip(model.p,methods_for(source)[name].p,(1,1,2))))}
    return {'source_seed':source['seed'],'new_task_index':index,'methods':records,
            'first_prediction_identical':True,'transferred_old_predictor_or_examples':False}


def summarize(values):
    mean=statistics.fmean(values)
    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.
    return {'mean':mean,'mean_ci95_normal':[mean-1.96*se,mean+1.96*se],
            'min':min(values),'max':max(values),'independent_source_seed_count':len(values)}


def aggregate(runs,sources):
    methods=list(runs[REGIMES[0]][0]['methods'])
    metrics=['service_mse','first64_mse','samples65_512_mse','samples513_end_mse','final_uniform_query_mse']
    summaries={}
    for regime,tasks in runs.items():
        # The four downstream tasks per source are not independent pretrained
        # states. Average within source before confidence intervals.
        by_seed={s['seed']:[t for t in tasks if t['source_seed']==s['seed']] for s in sources}
        per_seed={seed:{m:{k:statistics.fmean(t['methods'][m][k] for t in group) for k in metrics} for m in methods} for seed,group in by_seed.items()}
        method_summary={m:{k:summarize([p[m][k] for p in per_seed.values()]) for k in metrics} for m in methods}
        paired={}
        for base in ['initial_fixed','fresh_meta','diagonal_only','isotropic_same_trace','rotated_90','train_selected_scalar','train_selected_matrix']:
            paired['learned_frozen_minus_'+base]={k:summarize([p['learned_frozen'][k]-p[base][k] for p in per_seed.values()]) for k in metrics}
        paired['learned_continue_minus_fresh_meta']={k:summarize([p['learned_continue'][k]-p['fresh_meta'][k] for p in per_seed.values()]) for k in metrics}
        paired['learned_continue_minus_learned_frozen']={k:summarize([p['learned_continue'][k]-p['learned_frozen'][k] for p in per_seed.values()]) for k in metrics}
        summaries[regime]={'methods':method_summary,'paired_differences':paired,'seed_means':per_seed}
    return summaries


def block_gradient(p,examples,w0):
    model=Learner(p,True)
    model.w=list(w0)
    loss=0.
    for x,y in examples:
        e=y-model.predict(x)
        loss+=.5*e*e/len(examples)
        model.observe(x,y)
    return loss,[v/len(examples) for v in model.gradient]


def mathematical_checks():
    examples=[(x,y) for x,y,_,_ in stream(4545,23,.73)]
    p=[.18,.09,.035]
    w0=[.2,-.3]
    loss,g=block_gradient(p,examples,w0)
    differences=[]
    for j in range(3):
        upper=p[:];lower=p[:]
        upper[j]+=1e-6;lower[j]-=1e-6
        numeric=(block_gradient(upper,examples,w0)[0]-block_gradient(lower,examples,w0)[0])/2e-6
        differences.append(abs(numeric-g[j]))
    assert max(differences)<1e-7,differences
    # Rotate the entire problem, including P, not merely the updater.
    a=Learner(adaptive=True);b=Learner(adaptive=True)
    angle=.413
    max_error=0.
    for x,y,_,_ in stream(5656,512,.63):
        z=rotate(x,angle)
        max_error=max(max_error,abs(a.predict(x)-b.predict(z)))
        a.observe(x,y);b.observe(z,y)
        expected=rotate_gain(a.p,angle)
        max_error=max(max_error,max(abs(i-j) for i,j in zip(expected,b.p)))
    assert max_error<1e-10,max_error
    return {'block_gradient_finite_difference_max_error':max(differences),
            'full_problem_rotation_max_difference':max_error,
            'rotation_test_steps':512,'finite_difference_block_length':23,
            'claim':'Block-local exact derivative; not differentiation through all past changing gains.'}


def probe(bundle):
    model=Learner.restore(bundle['state'])
    predictions=[]
    for x,y in bundle['continuation']:
        predictions.append(model.predict(x));model.observe(x,y)
    return {'predictions':predictions,'final_state':model.snapshot()}


def persistence_check():
    model=Learner(adaptive=True)
    data=[(x,y) for x,y,_,_ in stream(8989,240,.36)]
    for x,y in data[:91]:model.observe(x,y)
    bundle=json.loads(json.dumps({'state':model.snapshot(),'continuation':data[91:]}))
    expected=probe(bundle)
    with tempfile.TemporaryDirectory(prefix='update-rule-state-') as temporary:
        path=Path(temporary)/'bundle.json'
        path.write_text(json.dumps(bundle),encoding='utf-8')
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe',str(path)],
                            capture_output=True,text=True,check=True)
        actual=json.loads(proc.stdout)
    assert actual==expected
    return {'fresh_process_full_state_equal':True,'checkpoint':91,'continued_steps':149,
            'checkpoint_inside_meta_block':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=SEEDS)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results'/'learned_update.json')
    parser.add_argument('--probe',type=Path)
    parser.add_argument('--math-only',action='store_true')
    args=parser.parse_args()
    if args.probe:
        print(json.dumps(probe(json.loads(args.probe.read_text(encoding='utf-8')))));return
    checks=mathematical_checks()
    if args.math_only:
        print(json.dumps(checks));return
    sources=[source_life(seed) for seed in range(args.seeds)]
    runs={r:[task_run(s,r,i) for s in sources for i in range(TASKS_PER_REGIME)] for r in REGIMES}
    report={'experiment':'experience_learned_update_geometry','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol':{'seeds':args.seeds,'source_samples_per_seed':TRAIN_STEPS,'downstream_tasks_per_regime_source':TASKS_PER_REGIME,
                        'downstream_samples_per_task':TEST_STEPS,'input':'iid uniform unit circle, features supplied',
                        'target':'two-dimensional random walk; no drift direction or clean target passed to learner',
                        'fast_drift_sd':FAST_SD,'slow_drift_sd':SLOW_SD,'label_noise_sd':NOISE_SD,
                        'block':BLOCK,'meta_rate':META_RATE,'initial_gain':INITIAL_GAIN,'gain_interval':[MIN_GAIN,MAX_GAIN],
                        'transfer':'only three independent entries of symmetric P; fresh w=0, sensitivities=0, gradient=0, no past rows',
                        'cold_restart_initial_target_sd':.75,'other_regimes_initial_target':[0.,0.],
                        'gain_grid_candidates':len(grid()),'grid_selection_signal':'mean observed prequential source error, no downstream labels',
                        'hyperparameters':'one specified config; no tuning on downstream results',
                        'confidence_unit':'source seed; four downstream tasks averaged within source',
                        'cost_scope':'base updates, sensitivity-vector updates, eig projections; no FLOP equality with grid search',
                        'not_claimed':['novel primitive','unknown feature language','causal world model','general lifelong guarantee']},
            'sources':sources,'runs':runs,'summary':aggregate(runs,sources),'mathematical_checks':checks,
            'persistence':persistence_check()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'checks':checks,
                     'source_mean_gains':{'fast':statistics.fmean(s['gain_along_actual_fast_axis'] for s in sources),
                                          'slow':statistics.fmean(s['gain_along_actual_slow_axis'] for s in sources)},
                     'mean_service_mse':{r:{m:v['service_mse']['mean'] for m,v in s['methods'].items()} for r,s in report['summary'].items()}},indent=2))


if __name__=='__main__':main()
