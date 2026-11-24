# -*- coding: utf-8 -*-
"""Living Intelligence M1: the rule, the cortex, the lexicon and the channels (research/living_intelligence_2026_11_24)."""
import math
import random

import pytest

from living.cortex import Cortex
from living.evidence import SPEAKING_LINE, belief, novelty
from living.lexicon import Lexicon, words
from living.mind import Mind

TEXT = ("the mind keeps no list of words. the mind keeps contexts, and the contexts keep counts. "
        "evler ev evlerimiz evde. kanıt kanıtı kanıtlar. ") * 3


def test_the_rule_is_the_krichevsky_trofimov_estimator():
    from zihin.evidence import expectation
    for pos, neg in [(0, 0), (1, 0), (0, 1), (3, 2), (10, 0), (7, 30), (0.5, 0.25)]:
        total = pos + neg
        e = belief(pos, total)
        assert e == pytest.approx((pos + 0.5) / (total + 1.0), abs=1e-15)
        assert e == pytest.approx(float(expectation(pos, neg)), abs=1e-15)
    assert novelty(0, 0) == 1.0
    assert novelty(3, 10) == pytest.approx(3.5 / 11)
    assert SPEAKING_LINE == pytest.approx(0.5 + 1.0 / 12)


@pytest.mark.parametrize("mixing", ["exclude", "blend"])
@pytest.mark.parametrize("escape", ["situation", "rule", "ppm-c"])
def test_the_distribution_is_whole_and_agrees_with_prediction(mixing, escape):
    cx = Cortex(depth=5, escape=escape, mixing=mixing)
    cx.stream(TEXT)
    for history in ("", "the m", "kanıt", "evler", "xyz q"):
        dist, unmet = cx.distribution(history)
        assert sum(dist.values()) + unmet == pytest.approx(1.0, abs=1e-12)
        for x, p in dist.items():
            assert cx.predict(history, x) == pytest.approx(p, rel=1e-12)
        y, p = cx.best(history)
        assert p == pytest.approx(max(dist.values()), rel=1e-12)


def test_prequential_bits_are_the_sum_of_predictions_before_learning():
    a = Cortex(depth=4)
    total, _ = a.stream(TEXT)
    b = Cortex(depth=4)
    h = ""
    acc = 0.0
    for x in TEXT:
        acc += -math.log2(b.predict(h, x))
        _, h = b.stream(x, h)
    assert total == pytest.approx(acc, rel=1e-12)
    assert a.digest("all") == b.digest("all")


def test_a_new_symbol_costs_the_code_space_and_joins_the_alphabet():
    cx = Cortex(depth=3)
    bits = []
    cx.stream("ab", bits=bits)
    assert bits[0] == pytest.approx(math.log2(0x110000))
    assert cx.alphabet == {"a": 0, "b": 1}


def test_retraction_leaves_the_substance_of_a_store_that_never_read_the_text():
    docs = [TEXT[:120], TEXT[120:260], TEXT[260:]]
    a = Cortex(depth=6)
    ledgers = []
    for d in docs:
        ledger = {}
        a.stream(d, "", ledger=ledger)
        ledgers.append(ledger)
    a.retract(docs[1], "", ledger=ledgers[1])
    b = Cortex(depth=6)
    for d in (docs[0], docs[2]):
        b.stream(d, "")
    assert a.digest() == b.digest()
    assert a.size()["entries"] == b.size()["entries"]
    a.retract(docs[0], "")
    a.retract(docs[2], "")
    assert a.nodes == {} and a.alphabet == {} and a.entries == 0


def test_deferred_growth_keeps_one_off_contexts_out_and_offers_no_retraction():
    once = Cortex(depth=6, grow=1)
    twice = Cortex(depth=6, grow=2)
    once.stream(TEXT)
    twice.stream(TEXT)
    assert twice.size()["nodes"] < once.size()["nodes"]
    for ctx, (n, counts) in twice.nodes.items():
        assert n >= 2 or ctx == ""
    with pytest.raises(ValueError):
        twice.retract("x")


def test_the_lexicon_numbers_words_as_paths_and_spells_them_back():
    lex = Lexicon()
    numbers = {w: lex.live(w) for w in ["ev", "evler", "evlerimiz", "kanıt", "ev"]}
    for w, n in numbers.items():
        assert lex.spell(n) == w
        assert lex.number(w) == n
    assert lex.number("evl") is None and lex.number("göz") is None
    assert lex.size()["nodes"] == len("evlerimiz") + len("kanıt")      # shared beginnings share nodes
    assert lex.word_belief(numbers["ev"]) == pytest.approx(belief(2, 4))
    assert [s for s in words("evler, 3 kanıt")] == [(0, 5), (7, 8), (9, 14)]


def test_a_new_script_is_a_new_channel_and_noise_is_watched_in_silence():
    mind = Mind(depth=6)
    mind.perceive(TEXT * 4, channel="text")
    assert mind.which("the mind keeps") == "text"
    rng = random.Random(1)
    alien = "".join(chr(0xE000 + rng.randrange(4)) + chr(0xE010 + rng.randrange(3)) for _ in range(30))
    assert mind.which(alien) is None
    name = mind.perceive(alien)
    assert name == "channel-2" and mind.born == ["text", "channel-2"]
    assert mind.which(alien[::-1]) == "channel-2"
    noise = "".join(chr(0xE100 + rng.randrange(20)) for _ in range(6000))
    record = []
    name = mind.perceive(noise, record=record)
    assert name == "channel-3"
    assert not any(said for _, said, _ in record)
    assert mind.channels[name].structure() < SPEAKING_LINE
    assert mind.channels["text"].structure() >= SPEAKING_LINE
