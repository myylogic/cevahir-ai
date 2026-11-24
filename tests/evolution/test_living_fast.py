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
                                 dict(depth=6, episode=True, minlen=8, mix=True, recent=4)])
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
