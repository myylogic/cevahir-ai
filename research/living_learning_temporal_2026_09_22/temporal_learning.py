"""Round 15: raw sequential bits, learned suffix use and recurrent hypotheses."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
PROTOCOL = ROOT / "docs/research/living_learning_temporal_2026_09_22/PROTOCOL.md"
METHODS = ("kt_constant", "suffix_1", "suffix_5", "suffix_8", "map_tree_8", "mixture_tree_8", "two_state")
TASKS = ("last_bit", "delay_5", "parity")
SEEDS = list(range(24))
CHECKPOINTS = (0, 32, 128, 512, 1024)
TRANSITIONS = np.array([[[((t >> (2*s+b)) & 1) for b in (0,1)] for s in (0,1)]
                        for t in range(16) for r in range(4)], dtype=np.int64)
READOUT = np.array([[(r >> s) & 1 for s in (0,1)] for t in range(16) for r in range(4)], dtype=np.int64)
LOG_HALF = math.log(.5)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(value): return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def truth(task, word):
    if task == "last_bit": return int(word[-1])
    if task == "delay_5": return int(word[-5])
    if task == "parity": return sum(map(int,word)) % 2
    raise ValueError(task)


class Learner:
    def __init__(self, method):
        if method not in METHODS: raise ValueError(method)
        self.method = method
        self.depth = {"kt_constant":0,"suffix_1":1,"suffix_5":5,"suffix_8":8,
                      "map_tree_8":8,"mixture_tree_8":8,"two_state":0}[method]
        self.tree = method in ("map_tree_8","mixture_tree_8")
        self.counts = np.zeros(((2**(self.depth+1)-1) if self.tree else 2**self.depth,2),dtype=np.int64)
        nodes = len(self.counts) if self.tree else 0
        self.log_kt = np.zeros(nodes)
        self.log_mix = np.zeros(nodes)
        self.map_score = np.zeros(nodes)
        self.split = np.zeros(nodes,dtype=bool)
        if self.tree:
            for node in reversed(range(nodes)): self._refresh(node)
        self.errors = np.zeros(64 if method == "two_state" else 0,dtype=np.int64)
        self.labels = 0
        self.input_bits = 0
        self.candidate_state_steps = 0
        self.counter_updates = 0
        self.tree_node_refreshes = 0
        self.candidate_label_checks = 0
        self.begin()

    def begin(self):
        self.buffer = []
        self.running_states = np.zeros(64 if self.method == "two_state" else 0,dtype=np.int64)
        self.current_length = 0

    def feed(self, bit, count_work=True):
        if bit not in (0,1): raise ValueError(bit)
        self.current_length += 1
        if count_work: self.input_bits += 1
        if self.method == "two_state":
            self.running_states = TRANSITIONS[np.arange(64),self.running_states,bit]
            if count_work: self.candidate_state_steps += 64
        elif self.depth:
            self.buffer.append(int(bit))
            if len(self.buffer)>self.depth: del self.buffer[0]

    def path(self):
        if self.current_length < self.depth: raise ValueError("protocol requires full available suffix")
        nodes, node = [0], 0
        for bit in reversed(self.buffer):
            node = 2*node+1+bit
            nodes.append(node)
        return nodes

    def table_key(self):
        value = 0
        for bit in self.buffer: value = 2*value+bit
        return value

    def weights(self):
        logs = self.errors*math.log(.05/.95)
        weights = np.exp(logs-logs.max())
        return weights/weights.sum()

    def _refresh(self,node):
        left = 2*node+1
        if left>=len(self.log_kt):
            self.log_mix[node]=self.log_kt[node]
            self.map_score[node]=self.log_kt[node]
            self.split[node]=False
            return
        stop = LOG_HALF+self.log_kt[node]
        mix_branch = LOG_HALF+self.log_mix[left]+self.log_mix[left+1]
        self.log_mix[node]=float(np.logaddexp(stop,mix_branch))
        map_branch=LOG_HALF+self.map_score[left]+self.map_score[left+1]
        self.split[node]=map_branch>stop
        self.map_score[node]=max(stop,map_branch)

    def terminal_probability(self):
        if self.current_length == 0: raise ValueError("empty episode not in protocol")
        if self.method == "two_state":
            outputs = READOUT[np.arange(64),self.running_states]
            return float(self.weights() @ outputs)
        if not self.tree:
            n0,n1=self.counts[self.table_key()]
            return float((n1+.5)/(n0+n1+1))
        path=self.path()
        if self.method == "map_tree_8":
            node=next((node for node in path if not self.split[node]),path[-1])
            n0,n1=self.counts[node]
            return float((n1+.5)/(n0+n1+1))
        node=path[-1]
        n0,n1=self.counts[node]
        p=float((n1+.5)/(n0+n1+1))
        for node in reversed(path[:-1]):
            n0,n1=self.counts[node]
            local=(n1+.5)/(n0+n1+1)
            stop_weight=math.exp(LOG_HALF+self.log_kt[node]-self.log_mix[node])
            p=stop_weight*local+(1-stop_weight)*p
        return float(p)

    def observe_label(self,label):
        if label not in (0,1): raise ValueError(label)
        if self.method == "two_state":
            self.errors += READOUT[np.arange(64),self.running_states] != label
            self.candidate_label_checks += 64
        elif not self.tree:
            self.counts[self.table_key(),label] += 1
            self.counter_updates += 1
        else:
            path=self.path()
            for node in path:
                counts=self.counts[node]
                self.log_kt[node] += math.log((counts[label]+.5)/(counts.sum()+1))
                self.counts[node,label] += 1
            self.counter_updates += len(path)
            for node in reversed(path): self._refresh(node)
            self.tree_node_refreshes += len(path)
        self.labels += 1
        self.begin()

    def state(self):
        result=copy.deepcopy(self.__dict__)
        for key,value in result.items():
            if isinstance(value,np.ndarray): result[key]=value.tolist()
        return result

    @classmethod
    def restore(cls,state):
        result=cls.__new__(cls)
        result.__dict__.update(copy.deepcopy(state))
        for key in ("counts","errors","running_states"):
            setattr(result,key,np.array(state[key],dtype=np.int64))
        for key in ("log_kt","log_mix","map_score"):
            setattr(result,key,np.array(state[key],dtype=float))
        result.split=np.array(state["split"],dtype=bool)
        return result

    def structure(self):
        result=dict(selected_leaves=None,selected_max_depth=None,root_stop_mass=None,effective_candidates=None)
        if self.tree:
            pending=[(0,0)]; leaves=[]
            while pending:
                node,depth=pending.pop()
                if self.split[node]: pending.extend(((2*node+1,depth+1),(2*node+2,depth+1)))
                else: leaves.append(depth)
            result.update(selected_leaves=len(leaves),selected_max_depth=max(leaves),
                          root_stop_mass=math.exp(LOG_HALF+self.log_kt[0]-self.log_mix[0]))
        if len(self.errors): result["effective_candidates"]=float(1/np.sum(self.weights()**2))
        return result

    def resources(self):
        return dict(input_bits=self.input_bits,candidate_state_steps=self.candidate_state_steps,
                    counter_updates=self.counter_updates,tree_node_refreshes=self.tree_node_refreshes,
                    candidate_label_checks=self.candidate_label_checks,
                    numeric_array_bytes=sum(v.nbytes for v in self.__dict__.values() if isinstance(v,np.ndarray)),
                    state_json_bytes=len(encode(self.state()).encode()),
                    max_episode_buffer_bits=self.depth,raw_completed_episodes=0)


def query(learner,word,count_work=False):
    learner.begin()
    for bit in word: learner.feed(int(bit),count_work=count_work)
    return learner.terminal_probability()


def make_life(seed):
    rng=random.Random(440_019+seed*7607)
    seen=set()
    def words(number,low,high):
        result=[]
        while len(result)<number:
            word="".join(str(rng.randrange(2)) for _ in range(rng.randint(low,high)))
            if word not in seen:
                seen.add(word);result.append(word)
        return result
    return dict(seed=seed,training=words(1024,8,24),short_test=words(128,8,24),long_test=words(128,32,64))


def score(predictions,labels):
    p=np.asarray(predictions);y=np.asarray(labels)
    tie=np.isclose(p,.5,atol=1e-12,rtol=0)
    return dict(n=len(y),mse=float(np.mean((p-y)**2)),accuracy=float(np.mean(np.where(tie,.5,(p>.5)==y))))


def evaluate(learner,life,task):
    probe=Learner.restore(learner.state())
    result=dict(t=learner.labels,structure=learner.structure(),resources=learner.resources())
    for split in ("short_test","long_test"):
        words=life[split]
        result[split]=score([query(probe,word) for word in words],[truth(task,word) for word in words])
    return result


def run_one(life,task,method):
    learner=Learner(method)
    snapshots=[evaluate(learner,life,task)]
    predictions=[]
    for word in life["training"]:
        p=query(learner,word,count_work=True)
        predictions.append(p)
        learner.observe_label(truth(task,word))
        if learner.labels in CHECKPOINTS: snapshots.append(evaluate(learner,life,task))
    labels=[truth(task,w) for w in life["training"]]
    return dict(seed=life["seed"],task=task,method=method,predictions=predictions,
                all_prequential=score(predictions,labels),first_128=score(predictions[:128],labels[:128]),
                last_128=score(predictions[-128:],labels[-128:]),snapshots=snapshots,
                final_state_sha256=hashlib.sha256(encode(learner.state()).encode()).hexdigest())


def run_all():
    lives=[make_life(seed) for seed in SEEDS]
    return dict(date="2026-09-22",source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                lives=lives,runs=[run_one(life,task,method) for life in lives for task in TASKS for method in METHODS])


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--write",action="store_true");args=parser.parse_args()
    data=run_all();path=RESULTS/"temporal_learning.json"
    if args.write:
        path.write_text(encode(data)+"\n",encoding="utf-8")
        print(json.dumps(dict(status="written",trajectories=len(data["runs"]),bytes=path.stat().st_size)))
    else:
        assert encode(data)==encode(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps(dict(status="exact_rerun_passed",trajectories=len(data["runs"]))))


if __name__=="__main__":main()
