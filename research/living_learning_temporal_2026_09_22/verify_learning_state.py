"""Exact arithmetic and exhaustive witnesses for LEARNING_STATE_TR.md."""
from fractions import Fraction
from itertools import combinations, product
import json
import temporal_learning as t


def verify():
    def kt(n0, n1): return Fraction(2*n1+1, 2*(n0+n1+1))
    assert all(kt(k,k) == Fraction(1,2) for k in range(100))
    assert len({kt(k,k+1) for k in range(100)}) == 100
    assert kt(0,1) == Fraction(3,4) and kt(99,100) == Fraction(201,400)
    histories = [''.join(map(str,bits)) for bits in product((0,1),repeat=6)]
    fibers = {parity: {h[-5:] for h in histories if t.truth('parity',h) == parity} for parity in (0,1)}
    assert all(len(values)==32 for values in fibers.values())
    separated = 0
    for first, second in combinations(histories,2):
        assert any((t.truth('parity',first+z), t.truth('delay_5',first+z)) !=
                   (t.truth('parity',second+z), t.truth('delay_5',second+z))
                   for z in ['0'*n for n in range(5)])
        separated += 1
    result = dict(status='passed', equal_current_prediction_states=100,
                  distinct_after_one_label=100, no_sample_after_one='3/4',
                  balanced_198_after_one='201/400',
                  joint_state_witnesses=64, pairwise_distinguished=separated,
                  max_new_states_per_old_parity_state=32, auxiliary_bits_lower_bound=5,
                  scope='Finite witnesses supplement the proofs; no general learner or biological claim.')
    return result


if __name__ == '__main__':
    result=verify()
    (t.RESULTS/'learning_state_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
