"""Online terminal-label experience, passive state merging, and controls.

Models are replaced at observed episode boundaries. No state-count, true graph,
intermediate label, or equivalence-query oracle is given to the merging learner.
The exact-enumeration comparator is explicitly given the <=3-state bound.
"""
import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

import state_merging

HERE = Path(__file__).resolve().parent
TRAIN = 256
CHECKPOINTS = (16, 32, 64, 128, 256)
MAX_TRAIN_LENGTH = 12
WINDOW = 8
METHODS = ('merged', 'enumerated_compiled', 'enumerated_vote', 'suffix8', 'word_table')


def graph(model):
    return {k: model[k] for k in ('transitions', 'outputs')}


def answer(model, word, start=0):
    state = start
    for symbol in word:
        state = model['transitions'][state][symbol]
    return model['outputs'][state]


def witness(first, second):
    """Exact language equivalence from the start states; evaluator only."""
    pending = deque([((0, 0), ())])
    seen = {(0, 0)}
    while pending:
        (a, b), word = pending.popleft()
        if first['outputs'][a] != second['outputs'][b]:
            return list(word)
        for symbol in (0, 1):
            pair = (first['transitions'][a][symbol], second['transitions'][b][symbol])
            if pair not in seen:
                seen.add(pair)
                pending.append((pair, word+(symbol,)))
    return None


def minimal(model):
    partition = model['outputs'][:]
    while True:
        keys = [(partition[s], tuple(partition[t] for t in edges))
                for s, edges in enumerate(model['transitions'])]
        unique = {}
        revised = [unique.setdefault(k, len(unique)) for k in keys]
        if revised == partition:
            return len(set(revised)) == len(partition)
        partition = revised


def candidate_library():
    models = []
    for n in (1, 2, 3):
        for flat in itertools.product(range(n), repeat=2*n):
            transitions = [list(flat[2*s:2*s+2]) for s in range(n)]
            order = [0]
            for state in order:
                for child in transitions[state]:
                    if child not in order:
                        order.append(child)
            # Reachable, canonical BFS labels avoid isomorphic duplicates.
            if order != list(range(n)):
                continue
            for outputs in itertools.product((0, 1), repeat=n):
                model = {'transitions': transitions, 'outputs': list(outputs)}
                if minimal(model):
                    models.append(model)
    return models


class LearningState:
    def __init__(self, library):
        self.library = library
        self.survivors = list(range(len(library)))
        self.samples = []
        self.model = {'transitions': [[0, 0]], 'outputs': [0]}
        self.suffix = {}
        self.table = {}
        self.fit_cost = {}
        self.commits = 0
        self.filter_transitions = 0
        self.first_empty_version_space = None

    def predictions(self, word):
        values = [answer(self.library[i], word) for i in self.survivors]
        counts = self.suffix.get(tuple(word[-WINDOW:]))
        return {'merged': float(answer(self.model, word)),
                'enumerated_compiled': float(values[0]) if values else None,
                'enumerated_vote': statistics.fmean(values) if values else None,
                'suffix8': (counts[0]+1)/(counts[1]+2) if counts else .5,
                'word_table': float(self.table[tuple(word)]) if tuple(word) in self.table else .5}

    def observe(self, word, label):
        word = tuple(word)
        self.filter_transitions += len(word)*len(self.survivors)
        self.survivors = [i for i in self.survivors if answer(self.library[i], word) == label]
        self.samples.append((word, label))
        if not self.survivors and self.first_empty_version_space is None:
            self.first_empty_version_space = len(self.samples)
        key = word[-WINDOW:]
        counts = self.suffix.setdefault(key, [0, 0])
        counts[0] += label
        counts[1] += 1
        self.table[word] = label
        if len(self.samples) in CHECKPOINTS:
            fitted = state_merging.fit(self.samples)
            self.model = graph(fitted)
            self.commits += 1
            for key, value in fitted['fit_cost'].items():
                self.fit_cost[key] = self.fit_cost.get(key, 0)+value
            return fitted['fit_cost']
        return None

    def snapshot(self):
        return {'survivors': self.survivors, 'samples': [[list(w), y] for w, y in self.samples],
                'model': self.model, 'fit_cost': self.fit_cost, 'commits': self.commits,
                'filter_transitions': self.filter_transitions,
                'first_empty_version_space': self.first_empty_version_space}

    @classmethod
    def restore(cls, record, library):
        obj = cls(library)
        state = json.loads(json.dumps(record))
        for name in ('survivors', 'model', 'fit_cost', 'commits', 'filter_transitions', 'first_empty_version_space'):
            setattr(obj, name, state[name])
        obj.samples = [(tuple(w), y) for w, y in state['samples']]
        for w, y in obj.samples:
            obj.table[w] = y
            counts = obj.suffix.setdefault(w[-WINDOW:], [0, 0])
            counts[0] += y
            counts[1] += 1
        return obj


def words(seed, count, excluded=()):
    rng = random.Random(seed)
    seen = set(excluded)
    output = []
    while len(output) < count:
        w = tuple(rng.randrange(2) for _ in range(rng.randint(1, MAX_TRAIN_LENGTH)))
        if w not in seen:
            output.append(w)
            seen.add(w)
    return output


def metrics(predictions, labels):
    covered = [(p, y) for p, y in zip(predictions, labels) if p is not None]
    return {'brier_with_abstention_half': statistics.fmean(((.5 if p is None else p)-y)**2 for p, y in zip(predictions, labels)),
            'classification_error_when_covered': statistics.fmean(int((p > .5) != y) for p, y in covered) if covered else None,
            'coverage': len(covered)/len(labels)}


def evaluate(learner, truth, collections):
    saved = json.dumps(learner.snapshot(), sort_keys=True)
    result = {}
    for name, data in collections.items():
        predictions = [learner.predictions(word) for word in data]
        labels = [answer(truth, word) for word in data]
        result[name] = {m: metrics([p[m] for p in predictions], labels) for m in METHODS}
    assert saved == json.dumps(learner.snapshot(), sort_keys=True)
    return result


def truth_for(regime, seed, library):
    if regime == 'parity':
        return {'transitions': [[0, 1], [1, 0]], 'outputs': [0, 1]}
    if regime == 'outside_three_state_bound':
        return {'transitions': [[s, (s+1) % 4] for s in range(4)], 'outputs': [1, 0, 0, 0]}
    candidates = [m for m in library if len(m['outputs']) == 3]
    return random.Random(71000+seed).choice(candidates)


def life(regime, seed, library):
    truth = truth_for(regime, seed, library)
    training = words(81000+seed, TRAIN)
    clean_labels = [answer(truth, w) for w in training]
    flipped = set(random.Random(91000+seed).sample(range(TRAIN), TRAIN//8)) if regime == 'corrupted_unique_labels' else set()
    labels = [y ^ int(i in flipped) for i, y in enumerate(clean_labels)]
    short = words(101000+seed, 128, training)
    rng = random.Random(111000+seed)
    long = [tuple(rng.randrange(2) for _ in range(length)) for length in (32, 64, 128, 256) for _ in range(16)]
    evaluation = {'unseen_short': short, 'long': long}
    learner = LearningState(library)
    service = {m: [] for m in METHODS}
    service_vote_transitions = 0
    checkpoints = []
    for step, (word, label) in enumerate(zip(training, labels), 1):
        predictions = learner.predictions(word)
        for m in METHODS:
            service[m].append(predictions[m])
        service_vote_transitions += len(word)*len(learner.survivors)
        fit_cost = learner.observe(word, label)
        if fit_cost is not None:
            difference = witness(learner.model, truth)
            compiled = library[learner.survivors[0]] if learner.survivors else None
            checkpoints.append({'labels_seen': step, 'merged_states': len(learner.model['outputs']),
                'version_space_size': len(learner.survivors), 'merged_exact_equivalence': difference is None,
                'merged_counterexample': difference,
                'enumerated_compiled_exact_equivalence': witness(compiled, truth) is None if compiled else None,
                'fit_cost': fit_cost, 'evaluation': evaluate(learner, truth, evaluation)})
    correction = None
    if flipped:
        repaired = state_merging.fit(list(zip(training, clean_labels)))
        correction = {'authorization': 'Evaluator supplies trusted replacements of the corrupted terminal labels; learner did not discover their locations.',
                      'replaced_labels': len(flipped), 'graph': graph(repaired),
                      'exact_equivalence': witness(repaired, truth) is None, 'fit_cost': repaired['fit_cost']}
    return {'regime': regime, 'seed': seed, 'truth_evaluator_only': truth,
            'training_words': [list(w) for w in training], 'observed_labels': labels,
            'clean_labels_evaluator_only': clean_labels, 'corrupted_indices_evaluator_only': sorted(flipped),
            'service_clean': {m: metrics(service[m], clean_labels) for m in METHODS},
            'service_observed': {m: metrics(service[m], labels) for m in METHODS},
            'checkpoints': checkpoints, 'final_graph': learner.model,
            'first_empty_version_space': learner.first_empty_version_space,
            'final_memory': {'retained_words': len(learner.samples), 'retained_symbols': sum(map(len, training)),
                             'graph_states': len(learner.model['outputs']), 'graph_transition_entries': 2*len(learner.model['outputs']),
                             'suffix_keys': len(learner.suffix), 'candidate_ids': len(learner.survivors)},
            'cost': {'merged_cumulative_fit': learner.fit_cost, 'graph_service_transitions': sum(map(len, training)),
                     'enumeration_filter_transitions': learner.filter_transitions,
                     'enumeration_vote_service_transitions': service_vote_transitions,
                     'compiled_service_transitions_upper_bound': sum(map(len, training)),
                     'scope': 'Enumeration compiled and vote share candidate filtering; counters are not matched FLOPs. Held-out evaluation is excluded.'},
            'trusted_correction': correction}


def descriptive(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values)/len(values)**.5 if len(values) > 1 else 0.
    return {'mean': mean, 'normal_ci95': [mean-1.96*se, mean+1.96*se], 'min': min(values), 'max': max(values)}


def aggregate(runs):
    results = {}
    for regime in sorted({r['regime'] for r in runs}):
        group = [r for r in runs if r['regime'] == regime]
        results[regime] = {'lives': len(group),
            'merged_exact_equivalence_count': sum(r['checkpoints'][-1]['merged_exact_equivalence'] for r in group),
            'enumeration_exact_equivalence_count': sum(r['checkpoints'][-1]['enumerated_compiled_exact_equivalence'] is True for r in group),
            'empty_enumeration_count': sum(r['first_empty_version_space'] is not None for r in group),
            'final_merged_states': descriptive([len(r['final_graph']['outputs']) for r in group]),
            'final_long_brier': {m: descriptive([r['checkpoints'][-1]['evaluation']['long'][m]['brier_with_abstention_half'] for r in group]) for m in METHODS},
            'service_clean_brier': {m: descriptive([r['service_clean'][m]['brier_with_abstention_half'] for r in group]) for m in METHODS},
            'paired_long_merged_minus_compiled': descriptive([r['checkpoints'][-1]['evaluation']['long']['merged']['brier_with_abstention_half']-r['checkpoints'][-1]['evaluation']['long']['enumerated_compiled']['brier_with_abstention_half'] for r in group])}
    return results


def mathematical_checks(library):
    # No last-k window (even also receiving total length) separates these words.
    parity = truth_for('parity', 0, library)
    aliases = []
    for k in (1, 8, 32, 128):
        a, b = (0,)*(k+1), (1,)+(0,)*k
        assert len(a) == len(b) and a[-k:] == b[-k:]
        assert answer(parity, a) == 0 and answer(parity, b) == 1
        aliases.append({'window': k, 'same_length': True, 'opposite_labels': True,
                        'balanced_pair_min_error': .5, 'balanced_pair_min_brier': .25})
    try:
        state_merging.fit([((0, 1), 0), ((0, 1), 1)])
    except ValueError:
        contradiction_rejected = True
    else:
        raise AssertionError('Conflicting identical words must not be representable by a deterministic DFA.')
    survivors = [m for m in library if answer(m, (0, 1)) == 0 and answer(m, (0, 1)) == 1]
    assert not survivors
    # Same graph language, different numeric state labels: copying an ID is wrong.
    old = {'transitions': [[0, 1], [1, 2], [2, 0]], 'outputs': [1, 0, 0]}
    new = {'transitions': [[0, 2], [1, 0], [2, 1]], 'outputs': [1, 0, 0]}
    assert witness(old, new) is None
    wrong, mapped = answer(new, (1,), start=1), answer(new, (1,), start=2)
    assert wrong != mapped
    return {'finite_window_aliases': aliases, 'contradiction_rejected': contradiction_rejected,
            'contradiction_empties_any_deterministic_version_space': True,
            'numeric_state_migration_counterexample': {'old_id': 1, 'correct_new_id': 2,
                'wrong_continuation_output': wrong, 'correct_continuation_output': mapped}}


def probe(bundle):
    if bundle['kind'] == 'runtime':
        model, state = bundle['graph'], bundle['state']
        outputs = []
        for symbol in bundle['continuation']:
            state = model['transitions'][state][symbol]
            outputs.append(model['outputs'][state])
        return {'outputs': outputs, 'state': state}
    learner = LearningState.restore(bundle['state'], candidate_library())
    predictions = []
    for word, label in bundle['continuation']:
        predictions.append(learner.predictions(tuple(word)))
        learner.observe(tuple(word), label)
    return {'predictions': predictions, 'state': learner.snapshot()}


def fresh_process(bundle):
    expected = json.loads(json.dumps(probe(bundle)))
    with tempfile.TemporaryDirectory(prefix='recurrent-state-persistence-') as temp:
        path = Path(temp)/'input.json'
        path.write_text(json.dumps(bundle), encoding='utf-8')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe', str(path)],
                                capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual == expected


def persistence(library):
    truth = truth_for('random_three_state', 1, library)
    data = [(w, answer(truth, w)) for w in words(81001, 90)]
    learner = LearningState(library)
    for w, y in data[:37]:
        learner.observe(w, y)
    fresh_process(json.loads(json.dumps({'kind': 'learning', 'state': learner.snapshot(), 'continuation': data[37:]})))
    rng = random.Random(120001)
    word = [rng.randrange(2) for _ in range(512)]
    state = 0
    for symbol in word[:127]:
        state = learner.model['transitions'][state][symbol]
    fresh_process({'kind': 'runtime', 'graph': learner.model, 'state': state, 'continuation': word[127:]})
    return {'fresh_process_learning_continuation_equal': True, 'checkpoint_episodes': 37,
            'continued_episodes': 53, 'continuation_includes_refit_at_64': True,
            'runtime_mid_episode_equal_without_raw_prefix': True, 'prefix_symbols_not_serialized': 127,
            'continued_runtime_symbols': 385,
            'scope': 'Learning resumption keeps retained words and rebuilds lookup indices. Runtime resumption keeps graph and current state only.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'results'/'recurrent_state.json')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe(json.loads(args.probe.read_text(encoding='utf-8'))))); return
    library = candidate_library()
    configs = [('random_three_state', s) for s in range(16)]
    configs += [('parity', s) for s in range(4)]
    configs += [('outside_three_state_bound', s) for s in range(4)]
    configs += [('corrupted_unique_labels', s) for s in range(8)]
    if args.quick:
        configs = [('random_three_state', 0), ('outside_three_state_bound', 0), ('corrupted_unique_labels', 0)]
    runs = [life(regime, seed, library) for regime, seed in configs]
    report = {'experiment': 'experience_formed_recurrent_state',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE/'state_merging.py']},
        'protocol': {'training_episodes': TRAIN, 'unique_training_words': True, 'training_length_range': [1, MAX_TRAIN_LENGTH],
            'refit_checkpoints': CHECKPOINTS, 'short_unseen_queries_per_checkpoint': 128,
            'long_queries_per_checkpoint': 64, 'long_lengths': [32, 64, 128, 256],
            'labels': 'Terminal labels only; observable episode reset and query boundary; no intermediate labels.',
            'candidate_library': 'All reachable minimal binary DFAs with <=3 states, canonical BFS state names.',
            'candidate_count': len(library), 'candidate_counts_by_states': {n: sum(len(m['outputs']) == n for m in library) for n in (1, 2, 3)},
            'enumeration_privilege': 'Supplied correct state-count bound in main family; becomes false in four-state control.',
            'merging_privilege': 'Supplied deterministic finite-state language, binary encoding, terminal alignment, observed reset boundary; no state-count bound.',
            'corruption': '32 distinct sampled terminal labels flipped; no word duplicated, so deterministic consistency remains possible.',
            'evaluation': 'Truth and product-graph witnesses never passed to learners; no learning during held-out evaluation.',
            'configuration': 'Single specified configuration; full experiment not tuned on held-out outcomes.',
            'intervals': 'Descriptive normal intervals over independent sampled lives within each regime; no simultaneous significance claim.',
            'limitations': ['No physical cause identification', 'No newly invented computational primitive', 'No general lifetime guarantee', 'Refits occur only at observed episode boundaries']},
        'runs': runs, 'summary': aggregate(runs), 'mathematical_checks': mathematical_checks(library),
        'persistence': persistence(library)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'candidate_count': len(library),
                     'summary': report['summary'], 'persistence': report['persistence']}, indent=2))


if __name__ == '__main__':
    main()
