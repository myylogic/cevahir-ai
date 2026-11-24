# -*- coding: utf-8 -*-
"""M1: a new channel, a language outside the world's languages, and noise (NOTES.md, section 3).

    python m1_channels.py [--out results/m1_channels.jsonl]

The mind first lives English text on its declared channel "text": 90 % of the Python documentation. It then meets
three streams without being told where they come from, in segments of 256 symbols.

1. **Alien language, unknown script.** A made-up language with a grammar: about 400 words in six classes, a plural
   suffix and a verb ending that agrees with the subject. It is written in 22 symbols of Unicode's Private Use Area,
   which no language uses.
2. **Noise in another unknown script.** The same number of symbols, drawn uniformly and independently.
3. **The alien language in Latin letters.** The same language (the same words and grammar, other sentences), written
   with the letters the text channel already knows.

Measured, in windows of 5,000 symbols:

- which channel each segment went to, and when a channel was born;
- bits per symbol;
- the structure belief;
- how often the mind would have said its guess before the symbol came, and how often that guess was right;
- the belief that a segment is unfamiliar;
- whether living the new streams changed what the text channel predicts for held-out English.
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import os
import random
import sys
import time
import zlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.evidence import SPEAKING_LINE  # noqa: E402
from living.mind import Mind  # noqa: E402

WINDOW = 5000
SEGMENT = 256


def alien_language(n: int, script: str, sentences: int, lexicon: int = 7) -> str:
    """A made-up language with a grammar, ``n`` symbols long, in the private-use script or in Latin letters.

    ``lexicon`` seeds the words (the same language for the same seed), ``sentences`` the sentences said in it."""
    rng = random.Random(lexicon)
    if script == "pua":
        cons = [chr(0xE000 + i) for i in range(14)]
        vows = [chr(0xE00E + i) for i in range(5)]
        sep, end, comma = chr(0xE013), chr(0xE014), chr(0xE015)
    else:
        cons = list("ptkbdgmnfszlrh")
        vows = list("aeiou")
        sep, end, comma = " ", ".", ","

    def syllable() -> str:
        return rng.choice(cons) + rng.choice(vows) + (rng.choice(cons) if rng.random() < 0.3 else "")

    def make(count: int, lo: int, hi: int) -> list[str]:
        out: set[str] = set()
        while len(out) < count:
            out.add("".join(syllable() for _ in range(rng.randint(lo, hi))))
        return sorted(out)

    nouns, verbs, adjs = make(200, 1, 3), make(120, 1, 3), make(50, 1, 2)
    dets, preps, conjs = make(8, 1, 1), make(12, 1, 1), make(4, 1, 1)
    plural, agree_sg, agree_pl = syllable(), syllable(), syllable()
    weights = {id(c): [1.0 / (i + 1) for i in range(len(c))] for c in (nouns, verbs, adjs, dets, preps, conjs)}
    rng = random.Random(sentences)

    def pick(c: list[str]) -> str:
        return rng.choices(c, weights=weights[id(c)])[0]

    def np_(depth: int) -> tuple[list[str], bool]:
        pl = rng.random() < 0.3
        parts = [pick(dets)]
        if rng.random() < 0.4:
            parts.append(pick(adjs))
        parts.append(pick(nouns) + (plural if pl else ""))
        if depth < 2 and rng.random() < 0.2:
            parts += [pick(preps)] + np_(depth + 1)[0]
        return parts, pl

    def sentence() -> list[str]:
        subj, pl = np_(0)
        parts = subj + [pick(verbs) + (agree_pl if pl else agree_sg)]
        if rng.random() < 0.6:
            parts += np_(0)[0]
        if rng.random() < 0.2:
            parts += [pick(preps)] + np_(1)[0]
        return parts

    out: list[str] = []
    total = 0
    while total < n:
        s = sentence()
        if rng.random() < 0.15:
            s[-1] += comma
            s += [pick(conjs)] + sentence()
        text = sep.join(s) + end + sep
        out.append(text)
        total += len(text)
    return "".join(out)[:n]


def noise(n: int, seed: int = 11) -> str:
    rng = random.Random(seed)
    symbols = [chr(0xE100 + i) for i in range(22)]
    return "".join(rng.choice(symbols) for _ in range(n))


def compressors(s: str) -> dict:
    import pyppmd
    b = s.encode("utf-8")
    return {"gzip": round(len(zlib.compress(b, 9)) * 8 / len(s), 4),
            "bz2": round(len(bz2.compress(b, 9)) * 8 / len(s), 4),
            "xz": round(len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)) * 8 / len(s), 4),
            "ppmd8": round(len(pyppmd.compress(b, max_order=8, mem_size=1 << 30)) * 8 / len(s), 4)}


def live(mind: Mind, stream: str, label: str) -> dict:
    """Live ``stream`` in segments without naming its channel; return what happened, window by window."""
    record: list = []
    routes: Counter = Counter()
    born_before = list(mind.born)
    structure_curve = []
    unfamiliar = []
    for i in range(0, len(stream), SEGMENT):
        seg = stream[i:i + SEGMENT]
        guess = mind.which(seg)
        if guess is not None and i % (SEGMENT * 40) == 0:
            unfamiliar.append({"symbol": i, "unfamiliar": round(mind.unfamiliar(guess, seg), 4)})
        name = mind.perceive(seg, record=record)
        routes[name] += len(seg)
        if (i + SEGMENT) % WINDOW < SEGMENT:
            structure_curve.append(round(mind.channels[name].structure(), 4))
    windows = []
    for w in range(0, len(record), WINDOW):
        part = record[w:w + WINDOW]
        said = sum(1 for _, s, _ in part if s)
        right = sum(1 for _, _, r in part if r)
        windows.append({"from": w, "bits_per_symbol": round(sum(b for b, _, _ in part) / len(part), 4),
                        "said": round(said / len(part), 4), "right_when_said": round(right / said, 4) if said else None})
    first_said = next((i for i, (_, s, _) in enumerate(record) if s), None)
    total_said = sum(1 for _, s, _ in record if s)
    total_right = sum(1 for _, _, r in record if r)
    return {"stream": label, "symbols": len(stream), "distinct_symbols": len(set(stream)),
            "routes": dict(routes), "born": [b for b in mind.born if b not in born_before],
            "bits_per_symbol": round(sum(b for b, _, _ in record) / len(record), 4),
            "said": round(total_said / len(record), 4),
            "right_when_said": round(total_right / total_said, 4) if total_said else None,
            "first_said_at": first_said,
            "structure_at_end": structure_curve[-1] if structure_curve else None,
            "structure_curve": structure_curve, "windows": windows, "unfamiliar": unfamiliar,
            "compressors": compressors(stream)}


def held_bits(mind: Mind, held: str) -> float:
    ch = mind.channels["text"]
    b, _ = ch.cortex.stream(held, ch.history, learn=False)
    return b / len(held)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m1_channels.jsonl"))
    a = ap.parse_args()
    t0 = time.time()
    english = corpora.stream("en-pydoc")
    split = int(len(english) * 0.9)
    mind = Mind(depth=8)
    mind.perceive(english[:split], channel="text")
    held = english[split:]
    rows = [{"stage": "english lived", "text_symbols": split, "held_out_bpc": round(held_bits(mind, held), 6),
             "text_structure": round(mind.channels["text"].structure(), 4)}]
    rows.append(live(mind, alien_language(200_000, "pua", sentences=1), "alien language, unknown script"))
    rows.append(live(mind, noise(100_000), "noise, unknown script"))
    rows.append({"stage": "after the two new channels", "held_out_bpc": round(held_bits(mind, held), 6)})
    latin = alien_language(100_000, "latin", sentences=2)
    rows.append(live(mind, latin, "alien language, Latin letters"))
    rows.append({"stage": "after the alien language in Latin letters", "held_out_bpc": round(held_bits(mind, held), 6)})
    # speaking: give the first letter of a word of sentences never seen; the mind continues while its guess holds
    words_said = {}
    examples = []
    for channel, script in (("text", "latin"), ("channel-2", "pua")):
        unseen = alien_language(20_000, script, sentences=3)
        boundary = " " if script == "latin" else chr(0xE013)
        starts = [i for i in range(41, len(unseen) - 41) if unseen[i - 1] == boundary]
        tally = {"asked": 0, "whole_word": 0, "silent_before_the_end": 0, "wrong": 0}
        for i in sorted(random.Random(5).sample(starts, 200)):
            j = i + 1
            while unseen[j] not in (" ", ".", ",", chr(0xE013), chr(0xE014), chr(0xE015)):
                j += 1
            need = unseen[i + 1:j + 1]                       # the rest of the word and the symbol after it
            text, why = mind.say(channel, unseen[i - 40:i + 1], most=40)
            tally["asked"] += 1
            if text[:len(need)] == need:
                tally["whole_word"] += 1
            elif text == unseen[i + 1:i + 1 + len(text)]:
                tally["silent_before_the_end"] += 1
            else:
                tally["wrong"] += 1
            if script == "latin" and len(examples) < 10:
                examples.append({"before": unseen[i - 12:i + 1], "said": text, "was": unseen[i + 1:j + 12],
                                 "stopped": why})
        words_said[channel] = tally
    channels = {name: {"structure": round(ch.structure(), 4)} for name, ch in mind.channels.items()}
    rows.append({"stage": "speaking after living", "channels": channels, "speaking_line": round(SPEAKING_LINE, 4),
                 "word_completions": words_said, "examples": examples,
                 "seconds": round(time.time() - t0, 1)})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k not in ("windows", "structure_curve")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
