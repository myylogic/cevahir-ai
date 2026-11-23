# Kept readings: tried, measured, eliminated (section 30)

An exact cache of the mind's read-only readings across moments, with the dependencies of every reading recorded
and the store stamping what changes. It was built to bring down what the whole mind costs on a mind raised on
running text. It does exactly what the mind does without it (bit for bit), and it is **slower** than not keeping
anything. It is not part of the mind. This folder is the record.

* `onbellek.py`: the module as it stood when it was put away (it was `zihin/onbellek.py`).
* `hooks.diff`: the changes it needed in `zihin/store.py`, `zihin/read.py`, `zihin/aile.py`, `zihin/mind.py`
  against commit f2a4fba. Applying both gives the code that was measured.

## What was measured

| | without | kept readings |
|---|---|---|
| the same lives, outputs and store digests (10 small worlds with unit stamps, 5 with list stamps) | reference | identical |
| small worlds, seconds (chain / kinds / whole) | 4.5 / 30 / 23 | 6.2 / 40 / 37 |
| ten answers on the mind raised on the owner's documents (186,470 moments, 2.3 million ties), seconds each | about 30 | 59 to 115 |
| kept readings used / made / found no longer to hold (those ten answers) | | 144,388 / 314,554 / 208,436 |

## Why it cannot pay

Two thirds of what was kept no longer held when it was wanted. What undid it (counted from the third answer on,
`stale_why`): the list of the units that hold one of the most frequent words (a word held by more than ten
thousand composed units) had changed. That is a real dependency, not a coarse one: F37 weighs the fixed words
of a place's frame by how telling they are, `1 / (1 + the units that hold the word)`, and the share of the most
telling word is one over the sum of all of them. Every sentence makes new contexts that hold "bir", "ve", "."; so
the share of every frame that has such a word among its fixed words is another number after every sentence, in
its last digits. A reading that is exact cannot be used again; one that is used again is not exact.

What was done instead (section 30): the readings are shared within the moment and the run where they are provably
the same, and the lists that only grow are indexed as they grow.
