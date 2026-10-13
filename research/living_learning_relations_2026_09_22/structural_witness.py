"""Post-result diagnostic: equal current predictions, unequal learning response.

No tuned parameter or new benchmark. A finite exact witness using predeclared A/B.
"""
import argparse
from fractions import Fraction
import json

import relation_learning as r


def exact_marginals(functions):
    return [[Fraction(sum(f[x] == y for f in functions), len(functions))
             for y in range(r.N)] for x in range(r.N)]


def run():
    sources = r.make_sources(0)
    a_class = r.observed_class(r.learn_source(sources[0]))
    b_class = r.observed_class(r.learn_source(sources[1]))
    a_before, b_before = exact_marginals(a_class), exact_marginals(b_class)
    assert a_before == b_before
    a_after = [f for f in a_class if f[0] == 1]
    b_after = [f for f in b_class if f[0] == 1]
    pa = exact_marginals(a_after)[1][0]
    pb = exact_marginals(b_after)[1][0]
    assert (len(a_after), len(b_after), pa, pb) == (24, 3, Fraction(0), Fraction(1))
    # Code implementation independently matches rational enumeration.
    a, b = r.Learner("source_a", sources), r.Learner("source_b", sources)
    a.update(0, 0, 1)
    b.update(0, 0, 1)
    assert a.probabilities()[0, 1, 0] == 0 and b.probabilities()[0, 1, 0] == 1
    return dict(date="2026-09-22", status="passed", diagnostic="post_result_not_primary_benchmark",
                source_sha256=r.sha(__file__), primary_source_sha256=r.sha(r.HERE / "relation_learning.py"),
                source_class_sizes=[len(a_class), len(b_class)],
                before_marginals_exact=[[str(value) for value in row] for row in a_before],
                before_predictions_equal=True, new_evidence=dict(action=0, x=0, y=1),
                remaining_candidate_counts=[len(a_after), len(b_after)],
                subsequent_query=dict(action=0, x=1, y=0), after_probability_A=str(pa), after_probability_B=str(pb),
                implication="A deterministic state/update using only the shared current marginal matrix and this same evidence cannot reproduce both correct class-conditioned updates.",
                scope="Not an impossibility theorem for embeddings: a representation may preserve the missing joint constraints or identify the source class.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run()
    path = r.RESULTS / "structural_witness.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        assert result == json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
