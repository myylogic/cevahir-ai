"""Complete clade/union audit, not a trained tree model."""
from functools import lru_cache
import itertools
import json
from collections import Counter
from pathlib import Path


@lru_cache(None)
def trees(leaves):
    if len(leaves) == 1: return (leaves[0],)
    result = []
    first, rest = leaves[0], leaves[1:]
    for mask in range(1 << len(rest)):
        left = (first,) + tuple(v for i, v in enumerate(rest) if mask & (1 << i))
        right = tuple(v for i, v in enumerate(rest) if not mask & (1 << i))
        if not right: continue
        for a in trees(left):
            for b in trees(right): result.append((a, b))
    return tuple(result)


def clades(tree):
    if isinstance(tree, int): return {1 << tree}
    left, right = clades(tree[0]), clades(tree[1])
    return left | right | {max(left) | max(right)}


def min_cover(masks, target):
    # Exact subset enumeration, independent of tree recursion used to generate.
    eligible = [m for m in masks if m & target == m]
    for n in range(1, len(eligible) + 1):
        for chosen in itertools.combinations(eligible, n):
            union = 0
            for m in chosen: union |= m
            if union == target and sum(m.bit_count() for m in chosen) == target.bit_count():
                return list(chosen)
    raise AssertionError('leaf cover must exist')


def audit():
    groups = [7, 56, 21, 42]
    counts, best, witness, max_single = Counter(), 99, None, 0
    all_trees = trees(tuple(range(6)))
    for tree in all_trees:
        nodes = clades(tree)
        assert len(nodes) == 11 and max(nodes) == 63
        assert all(a & b in (0, a, b) for a in nodes for b in nodes)
        covers = [min_cover(nodes, g) for g in groups]
        total = sum(map(len, covers))
        counts[total] += 1
        max_single = max(max_single, sum(g in nodes for g in groups))
        if total < best: best, witness = total, {'tree': tree, 'covers': covers}
    assert len(all_trees) == 945 and len(set(all_trees)) == 945
    assert best == 6 and max_single == 2
    return {'trees': len(all_trees), 'category_masks': groups,
            'maximum_categories_represented_by_single_nodes': max_single,
            'all_categories_single_node_possible': False,
            'all_trees_support_exact_union_readout_without_duplicate_leaves': True,
            'minimum_total_union_node_references': best,
            'total_reference_histogram': dict(sorted(counts.items())), 'witness': witness,
            'scope': 'Exact representation audit; no concept-learning or tree-training experiment.'}


if __name__ == '__main__':
    result = audit()
    path = Path(__file__).resolve().parent / 'results/tree_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
