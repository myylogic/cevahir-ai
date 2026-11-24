# -*- coding: utf-8 -*-
"""Cevahir M3: Zihin as a voice of the harmony, and the prefrontal decision (living/cevahir.py, living/_core.c)."""
import random
import shutil

import pytest

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")

CFG = dict(depth=8, episode=True, minlen=12, episode_bits=20, mix=True, words=True, two_inputs=True,
           tree_prior=True, recent=6, lines=True)


@pytest.fixture(scope="module")
def FastCortex(tmp_path_factory):
    import os
    os.environ.setdefault("LIVING_CACHE", str(tmp_path_factory.mktemp("living_cache")))
    from living.fast import FastCortex
    return FastCortex


def _words(seed=0, n=3000):
    rng = random.Random(seed)
    vocab = ["".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 7))) for _ in range(60)]
    return [rng.choice(vocab) for _ in range(n)], vocab, rng


def test_a_silent_voice_leaves_the_harmony_bit_identical(FastCortex):
    toks, _, _ = _words()
    text = " ".join(toks)
    plain = FastCortex(**CFG)
    voiced = FastCortex(voice=True, **CFG)
    a, bits_a = plain.stream(text, bits=True)
    total = 0.0
    for i, t in enumerate(toks):                      # chunk by chunk, the voice set and silent each time
        voiced.set_voice([], [], 0.0)
        b, _ = voiced.stream((" " if i else "") + t)
        total += b
    assert total == pytest.approx(a, rel=0, abs=1e-9)


def test_a_voice_that_knows_is_followed_and_one_that_does_not_is_ignored(FastCortex):
    toks, vocab, rng = _words(1, 4000)

    def run(kind):
        m = FastCortex(voice=kind is not None, **CFG)
        total, n = 0.0, 0
        for i, t in enumerate(toks):
            chunk = (" " if i else "") + t
            if kind == "right":
                m.set_voice([chunk], [1.0], 1.0)
            elif kind == "random":
                m.set_voice([(" " if i else "") + rng.choice(vocab)], [1.0], 1.0)
            b, _ = m.stream(chunk)
            total += b
            n += len(chunk)
        return total / n

    alone, right, wrong = run(None), run("right"), run("random")
    assert right < alone / 10
    assert wrong < alone * 1.05


def test_the_voice_follows_what_is_lived(FastCortex):
    m = FastCortex(voice=True, **CFG)
    m.stream("one two three ")
    m.set_voice([" walk", " work", " run"], [1.0, 1.0, 1.0], 1.0)
    assert m.voice_info() == (3, 0, 0)
    m.stream(" w")
    assert m.voice_info() == (2, 0, 2)
    m.stream("alk")
    assert m.voice_info() == (0, 1, 5)


def _world():
    prims = {"walk": "WALK", "run": "RUN", "look": "LOOK", "zap": "ZAP"}
    templates = [("{} twice", "{0} {0}"), ("{} thrice", "{0} {0} {0}"), ("{} left", "LTURN {0}"),
                 ("{} right", "RTURN {0}"), ("{} and {}", "{0} {1}")]
    dialogues = [(p, a) for p, a in prims.items()]
    for tpl, ans in templates:
        for p, a in prims.items():
            if p == "zap":
                continue
            if tpl.count("{}") == 2:
                for q, b in prims.items():
                    if q != "zap":
                        dialogues.append((tpl.format(p, q), ans.format(a, b)))
            else:
                dialogues.append((tpl.format(p), ans.format(a)))
    rng = random.Random(0)
    life = []
    for _ in range(2):
        d = list(dialogues)
        rng.shuffle(d)
        life += d
    tests = [(tpl.format("zap"), ans.format("ZAP")) for tpl, ans in templates if tpl.count("{}") == 1]
    tests += [("zap and walk", "ZAP WALK"), ("walk and zap", "WALK ZAP")]
    return life, tests


def test_cevahir_says_what_zihin_generalised_where_the_harmony_alone_cannot(FastCortex):
    from living.cevahir import Cevahir
    life, tests = _world()
    c = Cevahir(variant="speaking", switches={"wholes": True, "told_apart": True}, window=13)
    for q, a in life:
        c.watch(q, a + " .")
    c.session.close_turn()
    said = c.answers([q for q, _ in tests], way="every", workers=2)
    exact = {way: sum(c.session.body(s[way]) == c.session.words(a) for s, (_, a) in zip(said, tests))
             for way in ("zihin", "harmony", "cevahir")}
    assert exact["zihin"] == len(tests)
    assert exact["cevahir"] == len(tests)
    assert exact["harmony"] == 0
    assert all(s["cevahir"] == a for s, (_, a) in zip(said, tests))      # written as the answers were: "ZAP ZAP"
    assert c.counts["unaligned"] == 0
