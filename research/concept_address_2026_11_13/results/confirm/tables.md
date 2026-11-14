# KA — confirmatory decisions

Worlds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]

| Hypothesis | family | n | mean | 95% CI | p | Holm | label |
|---|---|---|---|---|---|---|---|
| KA-H1 | primary | — | — | [—, —] | — | — | not in the confirmatory family (G-KA1 discovery screen) |
| KA-H2 | primary | 8 | 0.5823 | [0.5212, 0.6434] | 0.0078 | True | memory better |
| KA-H3 | primary | — | — | [—, —] | 0.0193 | True | supported |
| S1a | secondary | — | — | [—, —] | — | — | not tested |
| S1b | secondary | — | — | [—, —] | — | — | not tested |
| S2 | secondary | — | — | [—, —] | 0.0000 | True | supported |

## Per-world effects

| effect | 1000 | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 |
|---|---|---|---|---|---|---|---|---|
| KA-H1 | — | — | — | — | — | — | — | — |
| KA-H2 | 0.5870 | 0.5618 | 0.7363 | 0.6304 | 0.5054 | 0.5714 | 0.5341 | 0.5319 |
| ECE | 0.0149 | 0.0176 | 0.0157 | 0.0074 | 0.0112 | 0.0150 | 0.0125 | 0.0242 |
| growth | 0.0083 | 0.0139 | 0.0056 | 0.0000 | 0.0056 | 0.0056 | 0.0000 | 0.0083 |
| growth_fixed | 0.0417 | 0.0333 | 0.0139 | 0.0083 | 0.0472 | 0.0417 | 0.0278 | 0.0833 |
| S1a | — | — | — | — | — | — | — | — |
| S1b | — | — | — | — | — | — | — | — |
| S2i | 0.9561 | 0.9681 | 0.9793 | 0.9240 | 0.9167 | 0.9965 | 0.9963 | 0.9996 |
| S2ii | — | — | — | — | — | — | — | — |
| S2iii | — | — | — | — | — | — | — | — |

## Laws (count rules, outside Holm)

- Gaussian false-firing law holds on 8 of 8 worlds: holds.

## Consistency checks (pooled over the worlds, each share must be <= 0.01)

- cached_vs_hooked: 0 of 438 (0.0000)
- hooked_vs_greedy: 0 of 438 (0.0000)
- batched_vs_greedy_part_a: 0 of 0 (not run)
- base model unchanged on every world: True
- result: pass

## Hierarchical bootstrap (descriptive; worlds, then units; 10000 resamples)

- KA-H2: 95% interval [0.5240, 0.6410] over 8 worlds, unit = fact

