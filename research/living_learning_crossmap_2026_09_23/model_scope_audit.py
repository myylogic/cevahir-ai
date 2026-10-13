"""Post-result scope diagnostic, not a fair extra trained competitor."""
import itertools
import json
from collections import Counter
from fractions import Fraction
import corrected as c


def generator_maps(source):
    rows = {(d, k, x): value for d, k, x, value in source}
    xorders = []
    for start in range(6):
        order = [start]
        for _ in range(5): order.append(rows['x', 'next', order[-1]])
        if len(set(order)) == 6 and [rows['x', 'group', x] for x in order] == [0, 0, 0, 1, 1, 1]:
            xorders.append(order)
    assert len(xorders) == 1
    xorder = xorders[0]
    even = [y for y in range(6) if rows['y', 'group', y] == 0]
    assert len(even) == 3
    maps = []
    for evens in itertools.permutations(even):
        yorder = []
        for y in evens:
            other = rows['y', 'next', y]
            assert rows['y', 'next', other] == y and rows['y', 'group', other] == 1
            yorder.extend([y, other])
        mapping = [None] * 6
        for x, y in zip(xorder, yorder): mapping[x] = y
        maps.append(mapping)
    return maps


def audit():
    rows = []
    for seed in range(24):
        w = c.world(seed)
        maps = generator_maps(c.observation(w))
        assert len(maps) == 6 and [w['true_map'][x] for x in range(6)] in maps
        counts = [sum(all(m[x] == y for x, y in w['bridges'][:k]) for m in maps) for k in range(6)]
        assert counts[0] == 6 and counts[1] == 2 and counts[2] in (1, 2) and counts[3:] == [1, 1, 1]
        rows.append({'seed': seed, 'counts_by_bridge_prefix_0_to_5': counts})
    # Two unknown pairs: independent marginals imply a collision with probability
    # 1/2, whereas both jointly legal permutations forbid any collision.
    joint = [[0, 1], [1, 0]]
    assert all(m[0] != m[1] for m in joint)
    return dict(status='post_result_scope_diagnostic', generator_family_rows=rows,
        count_histograms_by_prefix={str(k): dict(Counter(r['counts_by_bridge_prefix_0_to_5'][k] for r in rows)) for k in range(6)},
        extra_information='Knows the planted cross-domain generation law, unlike the main learner family. Not a fair same-prior benchmark.',
        joint_query_counterexample=dict(legal_maps=joint, marginal_probabilities=[[1, 2], [1, 2]],
                                       true_collision_probability=0, independent_marginal_collision_probability=[1, 2]),
        scope='Compact known-pair state can recompute coupled queries; the current single-reference marginal routine cannot multiply marginals independently.')


if __name__ == '__main__':
    result = audit()
    (c.RESULTS / 'model_scope_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(c.encode({k: v for k, v in result.items() if k != 'generator_family_rows'}))
