"""Post-result witnesses: exact arithmetic and one recorded ambiguous noisy life."""
import argparse
from fractions import Fraction as F
import json

import consolidation as c


def inverse2(a):
    determinant = a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return [[a[1][1]/determinant, -a[0][1]/determinant],
            [-a[1][0]/determinant, a[0][0]/determinant]]


def mv(a, x):
    return [sum(row[j]*x[j] for j in range(2)) for row in a]


def dot(x, y):
    return sum(a*b for a, b in zip(x, y))


def run():
    # Independent exact rational example for the separate fixed-feature theorem.
    rows = [([F(1), F(0)], F(1)), ([F(1), F(1)], F(0)),
            ([F(1), F(-1)], F(2)), ([F(1), F(2)], F(3))]
    regularizer = F(2)
    a = [[regularizer if i == j else F(0) for j in range(2)] for i in range(2)]
    b = [F(0), F(0)]
    p = inverse2(a)
    theta = [F(0), F(0)]
    for x, y in rows:
        px = mv(p, x)
        denominator = 1+dot(x, px)
        k = [value/denominator for value in px]
        residual = y-dot(x, theta)
        theta = [theta[i]+k[i]*residual for i in range(2)]
        p = [[p[i][j]-px[i]*px[j]/denominator for j in range(2)] for i in range(2)]
        a = [[a[i][j]+x[i]*x[j] for j in range(2)] for i in range(2)]
        b = [b[i]+x[i]*y for i in range(2)]
        assert p == inverse2(a)
        assert theta == mv(inverse2(a), b)
    # A concrete ambiguity exposed by the recorded noisy experiment, not a new benchmark.
    data = json.loads((c.RESULTS / "consolidation.json").read_text(encoding="utf-8"))
    world = next(w for w in data["worlds"] if w["seed"] == 22 and w["regime"] == "noisy_day")
    row = next(r for r in data["runs"] if r["seed"] == 22 and r["regime"] == "noisy_day" and r["method"] == "sleep_rich_replay")
    ids = row["final_state"]["core_ids"][1]
    assert ids == [35, 52] and world["new_teacher_id"] in ids
    error_positions = [[i for i, (_, x, y) in enumerate(world["day_rows"]) if int(c.OUTPUTS[h, x]) != y] for h in ids]
    assert all(len(errors) == 4 for errors in error_positions)
    differing_xs = [x for x in range(64) if c.OUTPUTS[ids[0], x] != c.OUTPUTS[ids[1], x]]
    heldout_differences = [x for x in differing_xs if x in world["unseen_xs"]]
    assert len(differing_xs) == 32 and len(heldout_differences) == 24
    return dict(date="2026-09-22", status="passed", phase="post_result_diagnostic_not_primary",
                source_sha256=c.sha(__file__), primary_source_sha256=data["source_sha256"],
                rational_ridge=dict(steps=4, regularizer=str(regularizer), final_theta=list(map(str, theta)),
                                    recursive_equals_batch_every_step=True, scope="separate fixed-feature example, not parity learner"),
                conflicting_labels=dict(single_input_minimum_average_mse="1/4", experiment_full_space_bound="1/8"),
                noisy_ambiguity=dict(seed=22, candidate_ids=ids, true_teacher_id=world["new_teacher_id"],
                                     candidate_programs=[c.PROGRAMS[h] for h in ids],
                                     mismatched_observation_positions=error_positions,
                                     same_observed_data_and_exact_four_corruptions=True,
                                     differing_inputs=32, differing_heldout_inputs=24,
                                     witness_heldout_input=heldout_differences[0],
                                     witness_labels=[int(c.OUTPUTS[h, heldout_differences[0]]) for h in ids],
                                     explanation="Both latent target rules plus four label flips produce the identical available observations. An internal supervisor without additional distinguishing information cannot guarantee which rule is true."),
                source_scope="No new biological, consciousness, or AGI claim; witness selection after primary results is explicit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run()
    path = c.RESULTS / "exact_witnesses.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    else:
        assert c.encode(result) == c.encode(json.loads(path.read_text(encoding="utf-8")))
    print(c.encode(result))


if __name__ == "__main__":
    main()
