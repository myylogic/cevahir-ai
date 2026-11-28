# -*- coding: utf-8 -*-
"""Kinds of words from additive vectors: a kill experiment (criteria fixed before it was run, 28 November 2026).

    python kinds_kill.py

The owner's question: can one mind have words as vectors in which like words lie near (as a neural network has),
and still learn from one sentence at once and forget exactly? In a network the two exclude each other. They do not
where a word's vector is a plain sum: every word has a fixed sparse code (from its 64-bit code, nothing learned or
stored), and a word's vector is the sum, over every time it was met, of the code of the word before it and of the
word after it. Meeting a word once adds two codes; forgetting subtracts them; nothing is trained.

What is measured: whether such vectors, made from a life, foresee where a word will stand later. The repository's
English text (en-repo, en-pydoc: no Turkish text of this size is at hand) is cut in two: the first 80 % is lived,
the rest is held out. For every held-out place of a word (lived 20 times or more) between two words where it had
never stood: had one of its five nearest words stood exactly there in the life? Beside it, the same for five
words of like frequency chosen at random.

Killed if the nearest words foresee fewer than twice as many of those places as the random ones, or fewer than
one place in ten. Also checked, and asserted: lived and forgotten, the vectors are equal to never having lived.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_intelligence_2026_11_24"))
import corpora  # noqa: E402
from cevher.units import code_of, mix, tokens  # noqa: E402

D, ONES = 1024, 8


def code_vector(word, role):
    """The fixed code of a word in a role: eight places of 1,024, each plus or minus one, from its 64-bit code."""
    h, at, sign = mix(code_of(word), role), [], []
    for _ in range(ONES):
        h = mix(h, 0x9E3779B97F4A7C15)
        at.append(h % D)
        sign.append(1 if (h >> 40) & 1 else -1)
    return np.array(at), np.array(sign, dtype=np.int32)


class Kinds:
    def __init__(self):
        self.vec, self._codes = {}, {}

    def _code(self, word, role):
        key = (word, role)
        if key not in self._codes:
            self._codes[key] = code_vector(word, role)
        return self._codes[key]

    def live(self, toks, sign=1):
        for i in range(1, len(toks) - 1):
            v = self.vec.get(toks[i])
            if v is None:
                v = self.vec[toks[i]] = np.zeros(D, dtype=np.int32)
            for word, role in ((toks[i - 1], 1), (toks[i + 1], 2)):
                at, s = self._code(word, role)
                np.add.at(v, at, sign * s)


def main():
    text = corpora.stream("en-repo") + "\n" + corpora.stream("en-pydoc")
    toks = [t for t in tokens(text) if t[:1].isalnum()]
    cut = int(len(toks) * 0.8)
    life, held = toks[:cut], toks[cut:]
    k = Kinds()
    k.live(life[:5000])
    before = {w: v.copy() for w, v in k.vec.items()}
    k.live(life[5000:9000])
    k.live(life[5000:9000], sign=-1)                         # forgotten: every vector as it was, and none left over
    assert all((k.vec[w] == before.get(w, 0)).all() for w in k.vec), "forgetting is not exact"
    k.live(life[5000:])
    count = {}
    for t in life:
        count[t] = count.get(t, 0) + 1
    words = sorted((w for w, n in count.items() if n >= 20), key=lambda w: -count[w])
    M = np.stack([k.vec[w] for w in words]).astype(np.float64)
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    near = np.argsort(-(M @ M.T), axis=1)[:, 1:6]
    stood = {}
    for i in range(1, len(life) - 1):
        stood.setdefault(life[i], set()).add((life[i - 1], life[i + 1]))
    rank = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)
    places = by_near = by_chance = 0
    for i in range(1, len(held) - 1):
        w = held[i]
        if w in rank and (held[i - 1], held[i + 1]) not in stood[w]:
            places += 1
            r, place = rank[w], (held[i - 1], held[i + 1])
            by_near += any(place in stood[words[j]] for j in near[r])
            pool = [j for j in range(max(0, r - 25), min(len(words), r + 26)) if j != r]
            by_chance += any(place in stood[words[j]] for j in rng.choice(pool, 5, replace=False))
    shown = {w: [words[j] for j in near[rank[w]]] for w in ("list", "function", "returns", "three", "error", "he") if w in rank}
    out = {"what": "kill experiment: kinds of words from additive vectors", "lived_words": len(life), "held_out": len(held),
           "words_lived_20_times": len(words), "new_places": places, "foreseen_by_nearest_five": by_near,
           "foreseen_by_five_of_like_frequency": by_chance, "share_nearest": round(by_near / max(1, places), 4),
           "share_chance": round(by_chance / max(1, places), 4), "forgetting_exact": True,
           "survives": bool(by_near >= 2 * by_chance and by_near >= 0.1 * places), "nearest": shown}
    with open(os.path.join(HERE, "results", "kinds.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
