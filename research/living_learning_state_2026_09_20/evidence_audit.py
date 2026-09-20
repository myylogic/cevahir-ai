"""Post-result audit: language boundaries, inference rights, and erasure.

Original data and results are never rewritten. Extra labels in a diagnostic
repair are explicitly privileged and are not new independent test evidence.
"""
import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path

import recurrent_state as exp
import state_merging

HERE = Path(__file__).resolve().parent


def nonempty_witness(a, b):
    queue = deque([((a['transitions'][0][x], b['transitions'][0][x]), (x,)) for x in (0, 1)])
    seen = set()
    while queue:
        (s, t), word = queue.popleft()
        if (s, t) in seen:
            continue
        seen.add((s, t))
        if a['outputs'][s] != b['outputs'][t]:
            return list(word)
        for symbol in (0, 1):
            queue.append(((a['transitions'][s][symbol], b['transitions'][t][symbol]), word+(symbol,)))
    return None


def independent_signature(model):
    """Independent table-filling minimization + canonical BFS of reachable quotient."""
    n = len(model['outputs'])
    different = {(a, b) for a in range(n) for b in range(a)
                 if model['outputs'][a] != model['outputs'][b]}
    while True:
        added = set()
        for a in range(n):
            for b in range(a):
                if (a, b) in different:
                    continue
                for x in (0, 1):
                    u, v = model['transitions'][a][x], model['transitions'][b][x]
                    if (max(u, v), min(u, v)) in different:
                        added.add((a, b))
        if not added:
            break
        different |= added
    representatives = [min(t for t in range(n) if t == s or (max(s, t), min(s, t)) not in different) for s in range(n)]
    order = [representatives[0]]
    for state in order:
        for dest in model['transitions'][state]:
            rep = representatives[dest]
            if rep not in order:
                order.append(rep)
    names = {state: i for i, state in enumerate(order)}
    canonical = {'transitions': [[names[representatives[d]] for d in model['transitions'][s]] for s in order],
                 'outputs': [model['outputs'][s] for s in order]}
    return json.dumps(canonical, sort_keys=True)


def enumeration_check(library):
    signatures = set()
    descriptions = 0
    for n in (1, 2, 3):
        for flat in itertools.product(range(n), repeat=2*n):
            for outputs in itertools.product((0, 1), repeat=n):
                signatures.add(independent_signature({'transitions': [list(flat[s*2:s*2+2]) for s in range(n)], 'outputs': list(outputs)}))
                descriptions += 1
    library_signatures = {json.dumps(m, sort_keys=True) for m in library}
    assert signatures == library_signatures
    assert descriptions == 5898 and len(signatures) == 1054
    return {'all_labelled_descriptions': descriptions, 'distinct_languages': len(signatures),
            'independent_table_filling_equals_enumerated_library': True}


def toggle_word(model, exception):
    # Product with singleton recognizer; no need for minimization.
    size = len(exception)+2
    dead = size-1
    pairs = [(0, 0)]
    transitions, outputs = [], []
    for state, match in pairs:
        row = []
        outputs.append(model['outputs'][state] ^ int(match == len(exception)))
        for x in (0, 1):
            next_match = match+1 if match < len(exception) and exception[match] == x else dead
            pair = (model['transitions'][state][x], next_match)
            if pair not in pairs:
                pairs.append(pair)
            row.append(pairs.index(pair))
        transitions.append(row)
    return {'transitions': transitions, 'outputs': outputs}


def erasure(runs):
    learned = next(r['final_graph'] for r in runs if r['regime'] == 'parity')
    pairs = [(0,)*129, (1,)+(0,)*128]
    full = [exp.answer(learned, w) for w in pairs]
    no_graph = [0, 0]
    reset_state = []
    preserved_state = []
    for w in pairs:
        current = learned['transitions'][0][w[0]]
        preserved_state.append(exp.answer(learned, w[1:], start=current))
        reset_state.append(exp.answer(learned, w[1:], start=0))
    assert full == preserved_state == [0, 1] and reset_state == [0, 0]
    return {'learned_graph_predictions': full, 'graph_erased_predictions': no_graph,
            'runtime_state_erased_after_first_symbol': reset_state,
            'graph_and_state_preserved': preserved_state,
            'same_128_symbol_suffix': True,
            'statement': 'Erasing graph loses the learned rule; erasing current state loses this episode distinction. Keeping the graph permits fresh complete episodes without retraining.'}


def analyze():
    original_path = HERE/'results'/'recurrent_state.json'
    data = json.loads(original_path.read_text(encoding='utf-8'))
    library = exp.candidate_library()
    records = []
    for run in data['runs']:
        truth, learned = run['truth_evaluator_only'], run['final_graph']
        training = [(tuple(w), y) for w, y in zip(run['training_words'], run['observed_labels'])]
        nonempty = nonempty_witness(learned, truth)
        survivor_ids = [i for i, model in enumerate(library) if all(exp.answer(model, w) == y for w, y in training)]
        all_same_nonempty = (all(nonempty_witness(library[survivor_ids[0]], library[i]) is None for i in survivor_ids)
                             if survivor_ids else False)
        record = {'regime': run['regime'], 'seed': run['seed'],
                  'full_language_exact': exp.witness(learned, truth) is None,
                  'nonempty_language_exact': nonempty is None, 'nonempty_counterexample': nonempty,
                  'all_remaining_bounded_candidates_agree_on_nonempty_language': all_same_nonempty,
                  'version_space_size': len(survivor_ids)}
        if run['regime'] == 'random_three_state' and not record['full_language_exact']:
            assert exp.witness(learned, truth) == []
            repaired = state_merging.fit(training+[((), exp.answer(truth, ()))])
            record['posthoc_empty_query_diagnostic'] = {
                'supplied_extra_label': exp.answer(truth, ()),
                'exact_after_extra_label': exp.witness(repaired, truth) is None,
                'scope': 'Previously excluded zero-length episode supplied by evaluator; not part of original experiment or a discovery by the learner.'}
        if run['trusted_correction']:
            record['trusted_correction_nonempty_exact'] = nonempty_witness(run['trusted_correction']['graph'], truth) is None
        records.append(record)
    reference = next(r for r in data['runs'] if r['regime'] == 'parity')
    exception = (0,)*13
    alternative = toggle_word(reference['truth_evaluator_only'], exception)
    assert all(exp.answer(alternative, w) == y for w, y in zip(reference['training_words'], reference['observed_labels']))
    assert exp.answer(alternative, exception) != exp.answer(reference['truth_evaluator_only'], exception)
    return {'analysis': 'post_result_evidence_audit', 'input_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'enumeration': enumeration_check(library), 'lives': records, 'erasure': erasure(data['runs']),
            'unseen_regular_alternative': {'exception': list(exception), 'alternative_graph': alternative,
                'all_original_training_labels_equal': True, 'exception_labels_differ': True,
                'scope': 'Finite data cannot certify arbitrary regular targets without further restrictions; this is not an impossibility of useful generalization.'},
            'correction_to_initial_interpretation': 'The two full-language mismatches are exclusively the empty word, excluded from all service/train/test episodes. All 16 main learned graphs are exact on the actual nonempty episode domain. This must not be reported as long-sequence failure.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'results'/'evidence_audit.json')
    args = parser.parse_args()
    result = analyze()
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'enumeration': result['enumeration'],
        'counts': {regime: {'lives': len([r for r in result['lives'] if r['regime'] == regime]),
                           'exact_nonempty': sum(r['nonempty_language_exact'] for r in result['lives'] if r['regime'] == regime)}
                   for regime in sorted({r['regime'] for r in result['lives']})},
        'correction': result['correction_to_initial_interpretation']}, indent=2))


if __name__ == '__main__':
    main()
