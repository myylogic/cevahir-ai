# -*- coding: utf-8 -*-
"""Living Intelligence M2: the cortex in C and the harmony (living/fast.py, living/_core.c)."""
import math
import shutil

import pytest

from living.cortex import Cortex

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")

TEXT = ("the mind keeps no list of words. the mind keeps contexts, and the contexts keep counts. "
        "evler ev evlerimiz evde. kanıt kanıtı kanıtlar. ") * 6


@pytest.fixture(scope="module")
def FastCortex(tmp_path_factory):
    import os
    os.environ.setdefault("LIVING_CACHE", str(tmp_path_factory.mktemp("living_cache")))
    from living.fast import FastCortex
    return FastCortex


@pytest.mark.parametrize("update", ["all", "found"])
def test_the_c_cortex_gives_the_same_bits_as_the_python_cortex(FastCortex, update):
    split = len(TEXT) * 2 // 3
    py = Cortex(depth=6, update=update)
    fast = FastCortex(depth=6, update=update)
    a, h = py.stream(TEXT[:split])
    b, _ = fast.stream(TEXT[:split])
    assert a == b                                                      # the same floating-point number
    fa, _ = py.stream(TEXT[split:], h, learn=False)
    fb, _ = fast.stream(TEXT[split:], TEXT[:split], learn=False)
    assert fa == fb
    la, _ = py.stream(TEXT[split:], h)
    lb, _ = fast.stream(TEXT[split:], TEXT[:split])
    assert la == lb
    assert fast.size()["nodes"] == py.size()["nodes"]
    assert fast.size()["entries"] == py.size()["entries"]
    assert fast.size()["situations"] == py.size()["situations"]


def test_per_symbol_bits_add_up_and_runs_repeat(FastCortex):
    cfg = dict(depth=6, episode=True, minlen=8, mix=True, recent=4)
    one = FastCortex(**cfg)
    total, bits = one.stream(TEXT, bits=True)
    assert total == pytest.approx(float(bits.sum()), rel=1e-12)
    assert (bits > 0).all()
    two = FastCortex(**cfg)
    again, bits2 = two.stream(TEXT, bits=True)
    assert again == total and (bits == bits2).all()
    assert total / len(TEXT) < 3.0


@pytest.mark.parametrize("cfg", [dict(depth=6), dict(depth=6, episode=True, minlen=8),
                                 dict(depth=6, episode=True, minlen=8, mix=True, recent=4),
                                 dict(depth=6, episode=True, minlen=8, mix=True, recent=4, lines=True,
                                      episode_lengths=(16, 8, 5))])
def test_a_pass_that_does_not_learn_leaves_nothing_behind(FastCortex, cfg):
    split = len(TEXT) * 2 // 3
    a = FastCortex(**cfg)
    a.stream(TEXT[:split])
    a.stream(TEXT[split:], TEXT[:split], learn=False)               # look, without learning
    la, _ = a.stream(TEXT[split:], TEXT[:split])
    b = FastCortex(**cfg)
    b.stream(TEXT[:split])
    lb, _ = b.stream(TEXT[split:], TEXT[:split])
    assert la == lb


def test_the_harmony_learns_a_repeated_text_far_below_the_cortex(FastCortex):
    cortex = FastCortex(depth=6)
    harmony = FastCortex(depth=6, episode=True, minlen=8, mix=True, recent=4)
    c, _ = cortex.stream(TEXT)
    h, _ = harmony.stream(TEXT)
    assert h < c
    first = FastCortex(depth=6, episode=True, minlen=8, mix=True, recent=4)
    once, _ = first.stream(TEXT[:len(TEXT) // 6])
    again, _ = first.stream(TEXT[:len(TEXT) // 6])                   # the same lines a second time
    assert again < once / 4
    assert not math.isnan(h)


def test_recall_locks_onto_the_lived_line_that_began_so(FastCortex):
    lines = ["the river runs to the sea\n", "the river of stars is old\n", "a cat sat on the warm stone\n",
             "kanıt kaydı kesin tutulur\n"]
    mind = FastCortex(depth=6, episode=True, minlen=8, mix=True, recent=4, lines=True)
    mind.stream("\n" + "".join(lines * 2))
    said, locked = mind.recall("\n" + "a cat ")
    assert locked and said == "sat on the warm stone"
    said, locked = mind.recall("\n" + "kanıt ")
    assert locked and said == "kaydı kesin tutulur"
    said, locked = mind.recall("\n" + "the river ")                  # two lines begin so: the latest one
    assert locked and said == "of stars is old"
    _, locked = mind.recall("\n" + "no line began so ")
    assert not locked
    before, _ = mind.stream("a cat sat", "\n", learn=False)
    mind.recall("\n" + "a cat ")
    after, _ = mind.stream("a cat sat", "\n", learn=False)
    assert before == after                                            # recalling leaves the mind as it was


@pytest.mark.parametrize("cfg", [dict(depth=6), dict(depth=6, episode=True, minlen=8, mix=True, recent=4, lines=True,
                                                     episode_lengths=(16, 8, 5)),
                                 dict(depth=6, episode=True, minlen=8, mix=True, recent=4, lines=True, voice=True)])
def test_a_saved_core_goes_on_as_the_one_never_saved(FastCortex, cfg, tmp_path):
    split = len(TEXT) * 2 // 3
    a = FastCortex(**cfg)
    a.stream(TEXT[:split])
    path = str(tmp_path / "core.bin")
    a.save(path)
    b = FastCortex.load(path)
    assert b.size() == a.size()
    la, bits_a = a.stream(TEXT[split:], bits=True)
    lb, bits_b = b.stream(TEXT[split:], bits=True)
    assert la == lb and (bits_a == bits_b).all()
    if cfg.get("mix"):
        assert a.say(TEXT[:split], most=20) == b.say(TEXT[:split], most=20)
    with open(path, "ab") as f:                                      # a file with anything after the core is refused
        f.write(b"x")
    with pytest.raises(IOError):
        FastCortex.load(path)


def test_sleep_forgets_what_did_not_recur_and_keeps_what_did(FastCortex, tmp_path):
    cfg = dict(depth=6, episode=True, minlen=8, mix=True, recent=4, lines=True)
    a = FastCortex(**cfg)
    a.stream(TEXT[:len(TEXT) // 2] + "a line met only once: qzxv wplk mnbv.\n")
    before = a.size()
    gone = a.sleep(2)
    assert gone > 0 and a.size()["nodes"] == before["nodes"] - gone
    assert a.sleep(2) == 0                                             # nothing more to forget
    bits_once, _ = a.stream("qzxv wplk", TEXT[:200], learn=False)     # the one-off line is no longer known
    keep = FastCortex(**cfg)
    keep.stream(TEXT[:len(TEXT) // 2] + "a line met only once: qzxv wplk mnbv.\n")
    bits_kept, _ = keep.stream("qzxv wplk", TEXT[:200], learn=False)
    assert bits_once > bits_kept
    rec_a, _ = a.stream(TEXT[:60], TEXT[:200], learn=False)           # what recurred is predicted much as before
    rec_k, _ = keep.stream(TEXT[:60], TEXT[:200], learn=False)
    assert abs(rec_a - rec_k) < 0.25 * rec_k + 5
    a.stream(TEXT[len(TEXT) // 2:])                                    # it goes on living, and can be kept
    path = str(tmp_path / "slept.bin")
    a.save(path)
    b = FastCortex.load(path)
    assert b.stream("the mind keeps", learn=False) == a.stream("the mind keeps", learn=False)
