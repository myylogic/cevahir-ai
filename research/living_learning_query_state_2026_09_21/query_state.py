"""Exact finite audit of three future-query contracts; standard library only.

Run from the repository root. --check recomputes without overwriting evidence.
Replay deliberately does not share the signed-count query formula.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = HERE / "results/query_state.json"


def replay(history, hypotheses):
    return tuple(sum(h[x] != y for x, y in history) for h in hypotheses)


def signed(history, m):
    d = [0] * m
    for x, y in history:
        d[x] += 1 - 2 * y
    return len(history), tuple(d)


def query(state, hypotheses):
    n, d = state
    a = (n - sum(d)) // 2
    return tuple(a + sum(v * dx for v, dx in zip(h, d)) for h in hypotheses)


def histogram(history, m):
    return tuple(history.count((x, y)) for x in range(m) for y in (0, 1))


def risk_classes(n, m):
    def shell(s):
        if s == 0:
            return 1
        return sum(math.comb(m, k) * 2**k * math.comb(s-1, k-1)
                   for k in range(1, min(m, s)+1))
    return sum(shell(n-2*j) for j in range(n//2+1))


def bits(k):
    return (k-1).bit_length()


def audit_cell(m, n):
    hypotheses = tuple(itertools.product((0, 1), repeat=m))
    observations = tuple(itertools.product(range(m), (0, 1)))
    risk_signatures, histogram_signatures, id_signatures = set(), set(), set()
    states_to_risk, risk_to_states = {}, {}
    edits = appends = 0
    byte_totals = dict.fromkeys(("all_risks", "signed", "histogram", "id_records"), 0)
    for history in itertools.product(observations, repeat=n):
        current = replay(history, hypotheses)
        state = signed(history, m)
        hist = histogram(history, m)
        assert query(state, hypotheses) == current
        assert states_to_risk.setdefault(state, current) == current
        assert risk_to_states.setdefault(current, state) == state
        risk_signatures.add(current)
        histogram_signatures.add(hist)
        signature = list(current)
        for j, (x, old_y) in enumerate(history):
            for new_y in (0, 1):
                changed = history[:j] + ((x, new_y),) + history[j+1:]
                reference = replay(changed, hypotheses)
                d = list(state[1])
                d[x] += 2 * (old_y - new_y)
                assert query((n, d), hypotheses) == reference
                signature.extend(reference)
                edits += 1
        # Exact integer bytes, not a digest: no collision assumption in counting.
        id_signatures.add(bytes(signature))
        for x, y in observations:
            d = list(state[1])
            d[x] += 1 - 2*y
            assert query((n+1, d), hypotheses) == replay(history+((x,y),), hypotheses)
            appends += 1
        for key, value in zip(byte_totals, (current, state, hist, history)):
            byte_totals[key] += len(json.dumps(value, separators=(",", ":")).encode("utf-8"))
    counts = [len(risk_signatures), len(histogram_signatures), len(id_signatures)]
    expected = [risk_classes(n, m), math.comb(n+2*m-1, 2*m-1), (2*m)**n]
    assert counts == expected, (m, n, counts, expected)
    return {"m": m, "N": n, "histories": (2*m)**n,
            "risk_classes": counts[0], "histogram_classes": counts[1], "id_classes": counts[2],
            "ideal_fixed_bits": dict(zip(("risk", "histogram", "id"), map(bits, counts))),
            "content_edits_checked": edits, "appends_checked": appends,
            "query_mismatches": 0, "formula_count_mismatches": 0,
            "mean_json_payload_bytes": {k: v / ((2*m)**n) for k, v in byte_totals.items()}}


def negative_controls():
    hypotheses = tuple(itertools.product((0, 1), repeat=2))
    a, b = ((0, 0), (0, 1)), ((1, 0), (1, 1))
    assert signed(a, 2) == signed(b, 2)
    assert replay(a, hypotheses) == replay(b, hypotheses)
    assert a.count((0, 0)) != b.count((0, 0))
    # A purported x=0, y=0 record exists in a, but not in b.
    d = list(signed(b, 2)[1]); d[0] -= 2
    counts = query((2, d), hypotheses)
    assert all(0 <= v <= 2 for v in counts)
    c, e = ((0, 0), (1, 0)), ((1, 0), (0, 0))
    assert histogram(c, 2) == histogram(e, 2)
    c_new, e_new = ((0, 1), (1, 0)), ((1, 1), (0, 0))
    assert replay(c_new, hypotheses) != replay(e_new, hypotheses)
    return {"signed_collision": [a, b], "indicator_query_counts": [1, 0],
            "nonexistent_old_content_can_pass_numeric_bounds": True,
            "histogram_id_collision": [c, e],
            "same_id_set_label_produces_distinct_risks": True}


def run():
    cells = [audit_cell(m, n) for m in (1, 2, 3) for n in range(7)]
    return {"experiment": "future_query_state_classes", "date": "2026-09-21",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "contracts": ["all Boolean 0-1 loss counts on a fixed finite X",
                          "all observation indicator counts",
                          "all Boolean loss counts before/after every ID-only set-label command"],
            "cells": cells, "negative_controls": negative_controls(),
            "totals": {key: sum(cell[key] for cell in cells) for key in
                       ("histories", "content_edits_checked", "appends_checked", "query_mismatches", "formula_count_mismatches")},
            "limits": ["Finite exact enumeration is not a learning-quality benchmark",
                       "ID commands include setting the already present label",
                       "Content edits assume truthful old contents supplied externally",
                       "JSON payloads exclude schema, code, hypothesis definitions and allocator overhead",
                       "Class-count bounds do not supply an efficient encoder",
                       "No novelty or universal living-learning solution claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.check:
        assert json.loads(json.dumps(result)) == json.loads(RESULT.read_text(encoding="utf-8"))
    else:
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": "check" if args.check else "record", "totals": result["totals"]}))


if __name__ == "__main__":
    main()
