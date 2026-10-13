"""Independent tree enumeration, scalar automata, causal ordering and resume."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import temporal_learning as t


def preservation():
    baseline=json.loads((t.RESULTS/"preservation_baseline.json").read_text(encoding="utf-8"))
    changed=[p for p,d in baseline["files"].items() if not (t.ROOT/p).exists() or t.sha(t.ROOT/p)!=d]
    assert not changed,changed
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=t.ROOT,text=True).strip()
    assert head==baseline["base_head"]
    return dict(files_unchanged=len(baseline["files"]),head_unchanged=head)


def scalar_machine(candidate,word):
    transition,readout=divmod(candidate,4)
    state=0
    for symbol in word: state=(transition >> (state*2+int(symbol))) & 1
    return (readout >> state)&1


def trees(depth,path=""):
    if depth==0: return [([path],0.)]
    result=[([path],math.log(.5))]
    for left,lp in trees(depth-1,path+"0"):
        for right,rp in trees(depth-1,path+"1"):
            result.append((left+right,math.log(.5)+lp+rp))
    return result


def kt_log(n0,n1):
    return math.lgamma(n0+.5)+math.lgamma(n1+.5)-math.lgamma(n0+n1+1)-2*math.lgamma(.5)


def enumerate_tree_predictions(rows,word,depth):
    alternatives=[]
    for leaves,log_prior in trees(depth):
        statistics={leaf:[0,0] for leaf in leaves}
        for old,label in rows:
            reverse=old[::-1]
            leaf=next(leaf for leaf in leaves if reverse.startswith(leaf))
            statistics[leaf][label]+=1
        log_weight=log_prior+sum(kt_log(*counts) for counts in statistics.values())
        selected=next(leaf for leaf in leaves if word[::-1].startswith(leaf))
        n0,n1=statistics[selected]
        alternatives.append((log_weight,(n1+.5)/(n0+n1+1),leaves))
    maximum=max(x[0] for x in alternatives)
    weights=[math.exp(x[0]-maximum) for x in alternatives]
    mixture=sum(w*x[1] for w,x in zip(weights,alternatives))/sum(weights)
    best=next(x for x in alternatives if x[0]==maximum)
    return mixture,best[1],maximum,maximum+math.log(sum(weights))


def small_tree(method,depth):
    learner=t.Learner(method);learner.depth=depth
    nodes=2**(depth+1)-1
    learner.counts=np.zeros((nodes,2),dtype=np.int64)
    learner.log_kt=np.zeros(nodes);learner.log_mix=np.zeros(nodes)
    learner.map_score=np.zeros(nodes);learner.split=np.zeros(nodes,dtype=bool)
    for node in reversed(range(nodes)):learner._refresh(node)
    return learner


def structural():
    tree_checks=0
    rows=[("00000000",0),("00000001",1),("00000010",1),("00000011",0)]*3+[("10101011",0)]
    for depth in (1,2,3):
        mixture=small_tree("mixture_tree_8",depth)
        mapped=small_tree("map_tree_8",depth)
        for i in range(len(rows)+1):
            for word in ("00000000","00000001","10101010","11111111"):
                expected_mix,expected_map,expected_max,expected_log_mix=enumerate_tree_predictions(rows[:i],word,depth)
                assert abs(t.query(mixture,word)-expected_mix)<1e-12
                assert abs(t.query(mapped,word)-expected_map)<1e-12
                assert abs(mapped.map_score[0]-expected_max)<1e-12
                assert abs(mixture.log_mix[0]-expected_log_mix)<1e-12
                tree_checks+=1
            if i<len(rows):
                word,label=rows[i]
                for learner in (mixture,mapped):
                    t.query(learner,word);learner.observe_label(label)
    automata_checks=0
    for length in range(1,9):
        for bits in itertools.product("01",repeat=length):
            word="".join(bits)
            learner=t.Learner("two_state")
            t.query(learner,word)
            outputs=t.READOUT[np.arange(64),learner.running_states]
            assert outputs.tolist()==[scalar_machine(i,word) for i in range(64)]
            automata_checks+=64
    delay_possible=set(range(64))
    for bits in itertools.product("01",repeat=8):
        word="".join(bits)
        delay_possible={i for i in delay_possible if scalar_machine(i,word)==int(word[-5])}
    assert not delay_possible
    parity_candidate=next(i for i in range(64) if all(scalar_machine(i,"".join(w))==sum(map(int,w))%2
                                                    for w in itertools.product("01",repeat=8)))
    assert scalar_machine(parity_candidate,"000000000")!=scalar_machine(parity_candidate,"100000000")
    for method in t.METHODS:
        learner=t.Learner(method)
        for word,label in rows:
            t.query(learner,word,True);learner.observe_label(label)
        state=learner.state();encoded=t.encode(state)
        first,second=t.Learner.restore(state),t.Learner.restore(state)
        if method!="two_state":
            assert t.query(first,"000000000")==t.query(second,"100000000")
        else:
            assert len(state["running_states"])==64
        t.query(first,"10101010");first.observe_label(1)
        assert t.encode(state)==encoded
        # Pure evaluation uses a clone and never modifies the actual learner.
        life=t.make_life(0)
        before=t.encode(learner.state())
        t.evaluate(learner,life,"last_bit")
        assert t.encode(learner.state())==before
        learner.begin();learner.feed(1)
        assert learner.counts.tolist()==state["counts"] and learner.errors.tolist()==state["errors"]
    for seed in t.SEEDS:
        life=t.make_life(seed)
        assert len(set(life["training"]+life["short_test"]+life["long_test"]))==1280
    return dict(enumerated_tree_prediction_checks=tree_checks,scalar_automaton_outputs=automata_checks,
                delay5_not_in_two_state_family=True,parity_two_state_witness=parity_candidate,
                fixed_suffix_collision=True,evaluation_does_not_change_learner=True,
                no_parameter_update_before_terminal_label=True,isolated_states=True,
                distinct_train_test_words_all_lives=True)


def continuation(payload):
    learner=t.Learner.restore(payload["state"])
    predictions=[]
    for word,label in payload["remaining"]:
        predictions.append(t.query(learner,word,True));learner.observe_label(label)
    return dict(predictions=predictions,final_state=learner.state())


def restarts():
    checks=[]
    life=t.make_life(9)
    with tempfile.TemporaryDirectory(prefix="cevahir_temporal_resume_") as folder:
        source,output=Path(folder)/"in.json",Path(folder)/"out.json"
        for task in t.TASKS:
            for method in t.METHODS:
                learner=t.Learner(method)
                for word in life["training"][:128]:
                    t.query(learner,word,True);learner.observe_label(t.truth(task,word))
                remaining=[(word,t.truth(task,word)) for word in life["training"][128:144]]
                payload=dict(state=learner.state(),remaining=remaining)
                predictions=[]
                for word,label in remaining:
                    predictions.append(t.query(learner,word,True));learner.observe_label(label)
                expected=dict(predictions=predictions,final_state=learner.state())
                source.write_text(t.encode(payload),encoding="utf-8")
                subprocess.run([sys.executable,__file__,"--resume",str(source),str(output)],check=True,capture_output=True,text=True)
                actual=json.loads(output.read_text(encoding="utf-8"))
                assert t.encode(actual)==t.encode(expected)
                checks.append(dict(task=task,method=method,exact_equal=True,reference="uninterrupted_live_object"))
    return checks


def recorded():
    data=json.loads((t.RESULTS/"temporal_learning.json").read_text(encoding="utf-8"))
    assert data["source_sha256"]==t.sha(t.HERE/"temporal_learning.py")
    assert data["protocol_sha256"]==t.sha(t.PROTOCOL)
    assert len(data["runs"])==504
    lives={l["seed"]:l for l in data["lives"]}
    checks=0
    for row in data["runs"]:
        life=lives[row["seed"]];learner=t.Learner(row["method"])
        assert len(row["predictions"])==1024
        for i,word in enumerate(life["training"]):
            assert t.query(learner,word,True)==row["predictions"][i]
            learner.observe_label(t.truth(row["task"],word));checks+=1
        assert row["final_state_sha256"]==hashlib.sha256(t.encode(learner.state()).encode()).hexdigest()
        assert [s["t"] for s in row["snapshots"]]==list(t.CHECKPOINTS)
        for s in row["snapshots"]:
            for split in ("short_test","long_test"):
                assert 0<=s[split]["mse"]<=1
        assert row["snapshots"][-1]["resources"]["raw_completed_episodes"]==0
    return dict(trajectories=504,prequential_records=checks,snapshots=2520,final_hashes_match=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--structural-only",action="store_true")
    parser.add_argument("--resume",nargs=2);parser.add_argument("--rerun",action="store_true");args=parser.parse_args()
    if args.resume:
        source,target=map(Path,args.resume)
        target.write_text(t.encode(continuation(json.loads(source.read_text(encoding="utf-8")))),encoding="utf-8");return
    result=dict(structural=structural(),preservation=preservation())
    if not args.structural_only:
        result["recorded"]=recorded();result["fresh_process_resume"]=restarts()
        if args.rerun:
            run=subprocess.run([sys.executable,str(t.HERE/"temporal_learning.py")],check=True,capture_output=True,text=True)
            result["full_rerun"]=dict(returncode=run.returncode,stdout=run.stdout.strip())
        (t.RESULTS/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":main()
