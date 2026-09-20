"""Deterministic passive DFA state merging for a bounded research experiment.

This is a conventional RPNI-style red/blue learner, not a new algorithm.
Only supplied complete words receive observed terminal labels. Unlabeled
prefixes remain unknown throughout merging; they are completed with output 0
after learning. Missing transitions then lead to a rejecting sink.

Run ``python state_merging.py --self-test`` for small executable checks.
There are no third-party dependencies and no imposed learned-state bound.
"""

from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
import random


UNKNOWN = -1
MISSING = -1


def _shortlex(word: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return len(word), word


def _validate_samples(samples):
    labels = {}
    input_symbols = 0
    received = 0
    for word, label in samples:
        word = tuple(word)
        if any(type(symbol) is not int or symbol not in (0, 1) for symbol in word):
            raise ValueError("Words must contain only integer binary symbols 0 and 1.")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Terminal labels must be integer 0 or 1.")
        received += 1
        input_symbols += len(word)
        if word in labels and labels[word] != label:
            raise ValueError(f"Contradictory terminal labels for word {word!r}.")
        labels[word] = label
    return labels, input_symbols, received


def _access_words(transitions: dict[int, list[int]]) -> dict[int, tuple[int, ...]]:
    """Return each reachable state's shortest lexicographically first word."""
    access = {0: ()}
    frontier = deque([0])
    while frontier:
        state = frontier.popleft()
        for symbol, child in enumerate(transitions[state]):
            if child != MISSING and child not in access:
                access[child] = access[state] + (symbol,)
                frontier.append(child)
    if len(access) != len(transitions):
        raise AssertionError("A quotient of a reachable prefix tree became unreachable.")
    return access


def _try_merge(transitions, outputs, left, right):
    """Merge two states and their determinism closure, or reject a conflict.

    Returned roots map every old state to its final representative. Using the
    smallest prefix-tree ID as representative preserves the start state's ID
    0, even when closure merges another red state. Incoming edges are resolved
    through disjoint-set roots, so copying the full incoming-edge index is not
    necessary for every attempted merge.
    """
    parent = {state: state for state in transitions}
    trial_t = {state: edges.copy() for state, edges in transitions.items()}
    trial_o = outputs.copy()

    def root(state):
        while parent[state] != state:
            parent[state] = parent[parent[state]]
            state = parent[state]
        return state

    pending = deque([(left, right)])
    visits = 0
    while pending:
        first, second = pending.popleft()
        visits += 1
        first, second = root(first), root(second)
        if first == second:
            continue
        first_label, second_label = trial_o[first], trial_o[second]
        if (first_label != UNKNOWN and second_label != UNKNOWN
                and first_label != second_label):
            return None, visits
        leader, absorbed = sorted((first, second))
        parent[absorbed] = leader
        if trial_o[leader] == UNKNOWN:
            trial_o[leader] = trial_o[absorbed]
        for symbol in (0, 1):
            leader_child = trial_t[leader][symbol]
            absorbed_child = trial_t[absorbed][symbol]
            if leader_child == MISSING:
                trial_t[leader][symbol] = absorbed_child
            elif absorbed_child != MISSING:
                pending.append((leader_child, absorbed_child))
        del trial_t[absorbed]
        del trial_o[absorbed]

    roots = {state: root(state) for state in transitions}
    trial_t = {
        state: [MISSING if child == MISSING else root(child) for child in edges]
        for state, edges in trial_t.items()
    }
    return (trial_t, trial_o, roots), visits


def predict(model: dict, word: tuple[int, ...]) -> int:
    state = 0
    for symbol in word:
        state = model["transitions"][state][symbol]
    return model["outputs"][state]


def fit(samples: list[tuple[tuple[int, ...], int]]) -> dict:
    """Fit a total binary DFA consistent with every noncontradictory sample.

    Cost counters count algorithmic operations, not CPU time or FLOPs.
    ``input_symbols`` includes duplicate supplied words; ``prefix_tree_nodes``
    and ``unique_samples`` count the deduplicated construction. The terminal
    consistency check is charged separately in ``verification_symbols``.
    """
    labels, input_symbols, received = _validate_samples(samples)
    prefixes = {()}
    for word in labels:
        prefixes.update(word[:length] for length in range(1, len(word) + 1))
    ordered = sorted(prefixes, key=_shortlex)
    index = {word: state for state, word in enumerate(ordered)}
    transitions = {
        state: [index.get(word + (symbol,), MISSING) for symbol in (0, 1)]
        for state, word in enumerate(ordered)
    }
    outputs = {state: labels.get(word, UNKNOWN) for state, word in enumerate(ordered)}
    cost = {
        "samples_received": received,
        "unique_samples": len(labels),
        "input_symbols": input_symbols,
        "prefix_tree_nodes": len(ordered),
        "attempted_merges": 0,
        "successful_merges": 0,
        "rejected_merges": 0,
        "closure_pair_visits": 0,
        "promoted_blue_states": 0,
        "copied_states_for_merge_attempts": 0,
        "verification_symbols": 0,
    }
    red = {0}
    while True:
        access = _access_words(transitions)
        blue = {
            child
            for state in red
            for child in transitions[state]
            if child != MISSING and child not in red
        }
        if not blue:
            break
        chosen = min(blue, key=lambda state: _shortlex(access[state]))
        for target in sorted(red, key=lambda state: _shortlex(access[state])):
            cost["attempted_merges"] += 1
            cost["copied_states_for_merge_attempts"] += len(transitions)
            result, visits = _try_merge(transitions, outputs, target, chosen)
            cost["closure_pair_visits"] += visits
            if result is None:
                cost["rejected_merges"] += 1
                continue
            transitions, outputs, roots = result
            red = {roots[state] for state in red}
            cost["successful_merges"] += 1
            break
        else:
            red.add(chosen)
            cost["promoted_blue_states"] += 1

    cost["inferred_states_before_completion"] = len(transitions)
    cost["unknown_outputs_completed_as_zero"] = sum(
        label == UNKNOWN for label in outputs.values()
    )
    outputs = {state: max(label, 0) for state, label in outputs.items()}
    missing_edges = sum(child == MISSING for edges in transitions.values() for child in edges)
    cost["missing_transitions_completed"] = missing_edges
    cost["completion_sink_added"] = 0
    if missing_edges:
        # Reuse a state already rejecting every continuation when one exists.
        sinks = [
            state for state, edges in transitions.items()
            if outputs[state] == 0 and all(child in (MISSING, state) for child in edges)
        ]
        if sinks:
            sink = min(sinks)
        else:
            sink = max(transitions) + 1
            transitions[sink] = [sink, sink]
            outputs[sink] = 0
            cost["completion_sink_added"] = 1
        transitions = {
            state: [sink if child == MISSING else child for child in edges]
            for state, edges in transitions.items()
        }

    access = _access_words(transitions)
    order = sorted(transitions, key=lambda state: _shortlex(access[state]))
    dense = {state: new for new, state in enumerate(order)}
    model = {
        "transitions": [[dense[child] for child in transitions[state]] for state in order],
        "outputs": [outputs[state] for state in order],
        "fit_cost": cost,
    }
    for word, label in labels.items():
        if predict(model, word) != label:
            raise AssertionError(f"Internal error: learned model contradicts sample {word!r}.")
        cost["verification_symbols"] += len(word)
    return model


def _words(maximum_length):
    for length in range(maximum_length + 1):
        yield from itertools.product((0, 1), repeat=length)


def _equivalence_witness(first, second):
    """Exact reachable product check; return a distinguishing word or None."""
    pending = deque([((0, 0), ())])
    seen = {(0, 0)}
    while pending:
        (left, right), word = pending.popleft()
        if first["outputs"][left] != second["outputs"][right]:
            return word
        for symbol in (0, 1):
            pair = (first["transitions"][left][symbol], second["transitions"][right][symbol])
            if pair not in seen:
                seen.add(pair)
                pending.append((pair, word + (symbol,)))
    return None


def self_test():
    known = {
        "odd_number_of_ones": {"transitions": [[0, 1], [1, 0]], "outputs": [0, 1]},
        "ends_in_01": {"transitions": [[1, 0], [1, 2], [1, 0]], "outputs": [0, 0, 1]},
        "ones_modulo_three": {"transitions": [[0, 1], [1, 2], [2, 0]], "outputs": [1, 0, 0]},
        "reject_everything": {"transitions": [[0, 0]], "outputs": [0]},
        "accept_everything": {"transitions": [[0, 0]], "outputs": [1]},
    }
    results = []
    for name, truth in known.items():
        samples = [(word, predict(truth, word)) for word in _words(6)]
        model = fit(samples)
        witness = _equivalence_witness(model, truth)
        assert witness is None, (name, witness)
        assert all(predict(model, word) == predict(truth, word) for word in _words(10))
        assert json.loads(json.dumps(model)) == model
        # Input order and harmless duplicate labels must not affect decisions.
        shuffled = samples.copy()
        random.Random(7).shuffle(shuffled)
        permuted = fit(shuffled)
        assert model == permuted
        repeated = fit(samples + samples[:5])
        assert model["transitions"] == repeated["transitions"]
        assert model["outputs"] == repeated["outputs"]
        results.append({"name": name, "states": len(model["outputs"]), "exact_equivalence": True})

    for seed in range(6):
        rng = random.Random(seed + 500)
        truth = {
            "transitions": [[rng.randrange(6), rng.randrange(6)] for _ in range(6)],
            "outputs": [rng.randrange(2) for _ in range(6)],
        }
        words = [tuple(rng.randrange(2) for _ in range(rng.randrange(13))) for _ in range(400)]
        samples = [(word, predict(truth, word)) for word in words]
        model = fit(samples)
        assert all(predict(model, word) == label for word, label in samples)
    assert fit([])["transitions"] == [[0, 0]]
    assert fit([])["outputs"] == [0]
    assert fit([((), 1)])["outputs"] == [1, 0]
    assert fit([((), 1)])["transitions"] == [[1, 1], [1, 1]]
    for samples in ([((0,), 0), ((0,), 1)], [((), 0), ((), 1)]):
        try:
            fit(samples)
        except ValueError as error:
            assert "Contradictory terminal labels" in str(error)
        else:
            raise AssertionError("Contradictory duplicate labels were accepted.")
    return {"status": "passed", "known_languages": results,
            "random_training_consistency_cases": 6,
            "random_samples_per_case": 400,
            "random_maximum_word_length": 12,
            "contradictory_duplicate_checks": 2,
            "checks": ["exact DFA equivalence", "exhaustive words through length 10",
                       "sample order invariance", "duplicate label invariance",
                       "JSON round trip", "empty-data default", "missing-edge sink"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Import fit(samples) or provide --self-test.")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
