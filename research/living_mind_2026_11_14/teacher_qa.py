# -*- coding: utf-8 -*-
"""A teacher asks about what was read: "X nedir ?" answered with a sentence of the documents that starts with X.

    python research/living_mind_2026_11_14/teacher_qa.py DOC_FOLDER --out private_data/ogretmen.tsv --heldout 0.2

Subjects are the first words of sentences, kept only if they are not among the most frequent words of the corpus
(function words are the most frequent: a data-driven rule, no word list) and at least 4 letters long. One question per
subject (its first sentence). A share of the subjects is held out: never asked, written to <out>.heldout so that the
mind can be asked about them later (do the start (F12) and the continuation of its own words (F16) carry over?).
The output derives from the documents and stays wherever the documents are kept (private_data is never committed).
"""
import argparse
import collections
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from zihin.__main__ import read_document, sentences, words  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", type=float, default=0.2)
    ap.add_argument("--frequent", type=int, default=100, help="the most frequent words that cannot be subjects")
    ap.add_argument("--max-words", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = [args.folder] if os.path.isfile(args.folder) else sorted(
        os.path.join(r, n) for r, _, ns in os.walk(args.folder) for n in ns if n.lower().endswith((".docx", ".txt", ".md")))
    sents = [words(s) for p in paths for s in sentences(read_document(p))]
    freq = collections.Counter(w for s in sents for w in s)
    common = {w for w, _ in freq.most_common(args.frequent)}
    first = {}
    for s in sents:
        if 4 <= len(s) <= args.max_words and s[0].isalpha() and len(s[0]) >= 4 and s[0] not in common:
            first.setdefault(s[0], s)
    subjects = sorted(first)
    random.Random(args.seed).shuffle(subjects)
    k = int(len(subjects) * args.heldout)
    held, asked = subjects[:k], subjects[k:]
    with open(args.out, "w", encoding="utf-8") as f:
        for x in asked:
            f.write("%s nedir ?\t%s\n" % (x, " ".join(first[x])))
    with open(args.out + ".heldout", "w", encoding="utf-8") as f:
        for x in held:
            f.write("%s\t%s\n" % (x, " ".join(first[x])))
    print("%d documents, %d sentences, %d subjects: %d asked, %d held out" % (len(paths), len(sents), len(subjects),
                                                                           len(asked), len(held)))


if __name__ == "__main__":
    main()
