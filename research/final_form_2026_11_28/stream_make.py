# -*- coding: utf-8 -*-
"""A generated stream for the memory bound (gate G1): words drawn by Zipf's law from a vocabulary without end.

    python stream_make.py OUT_FILE [--chars 100000000] [--seed 0]

The repository's public corpora hold about 12 million characters, and open Turkish text could not be fetched (the
network policy of the environment refuses the hosts; docs/architecture/FINAL_FORM.md, section 11). To show that
resident memory is bounded for any length of life, this writes as many characters as asked: sentences of 4 to 14
made-up words, each word drawn with probability proportional to rank^-1.1, so that new word types keep arriving for
as long as the stream runs (the hardest case for a store of fixed capacity). The stream is not language; nothing
but memory and speed is measured on it. The file is not kept in the repository.
"""
import argparse

import numpy as np

SYLLABLES = [c + v for c in "bcdfghjklmnprstvyz" for v in "aeiou"] + ["an", "en", "in", "on", "un", "ar", "er"]


def word(rank: int) -> str:
    out = ""
    rank += 1
    while rank:
        rank, r = divmod(rank, len(SYLLABLES))
        out += SYLLABLES[r]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--chars", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    written, types = 0, set()
    with open(a.out, "w", encoding="utf-8") as f:
        while written < a.chars:
            ranks = rng.zipf(1.1, size=200_000)
            ranks = ranks[ranks < 10 ** 12]
            lengths = rng.integers(4, 15, size=len(ranks) // 4)
            names = {int(r): word(int(r)) for r in np.unique(ranks)}
            types.update(names)
            at, parts = 0, []
            for n in lengths.tolist():
                if at + n > len(ranks):
                    break
                parts.append(" ".join(names[int(r)] for r in ranks[at:at + n]) + ".")
                at += n
            block = " ".join(parts) + "\n"
            f.write(block)
            written += len(block)
    print({"chars": written, "word_types": len(types)})


if __name__ == "__main__":
    main()
