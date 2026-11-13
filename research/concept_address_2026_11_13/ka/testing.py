# -*- coding: utf-8 -*-
"""Tiny random bases for the unit tests of the KA study (no trained weights, no files).

``tiny_base`` builds a small TR-Paradigma world, a byte-BPE tokenizer and a ``CevahirNeuralNetwork`` through
CEV-11's ``build_model`` (``tiny_model_config`` + ``ModelManager.initialize``). The model is random: the tests
check mechanics and exact identities, never a learning claim. ``boost`` multiplies the tied embedding/readout
matrix so that the random readout has margins of realistic size (an analytic value can then reach ``m*``).
"""
from __future__ import annotations

import pathlib
import random

import torch

from . import KA_VERSION  # noqa: F401


def tiny_base(n_names: int = 120, layers: int = 3, d: int = 32, seed: int = 0, world_seed: int = 11,
              boost: float = 1.0):
    """Build the tiny base on one thread and leave the process as it was.

    The thread count is restored before returning: a module-scoped test fixture calls this before any
    function-scoped fixture can record the old value, so a ``torch.set_num_threads(1)`` left behind here would
    make every later test of the process single-threaded."""
    from cev11 import experiment as E
    from cev11.model import build_model, model_sha256
    from cev11.tokenizers import train_tokenizer
    from cev11.world import make_world, sentence
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        world = make_world(world_seed, n_names=n_names)
        rnd = random.Random(0)
        texts = [sentence(n, r, c, world.facts[(n, r)], o, x)
                 for n in world.names[:60] for r in ("A", "B", "C") for c in ("nom", "gen", "abl")
                 for o in ("formal", "informal") for x in ("none", "P1")]
        rnd.shuffle(texts)
        # every Turkish letter frequent enough to become one token (strict character offsets need whole characters)
        letters = " ".join(ch * 3 for ch in "çğıöşüÇĞİÖŞÜâîû") + " "
        tok = train_tokenizer(texts + ["Hava bugün çok güzeldi.", "Kapı yavaşça açıldı.", letters] * 20,
                              vocab_size=900, require_full_vocab=False)
        model, cfg = build_model(tok.vocab_size, layers=layers, d=d, heads=4, kv_heads=4, max_len=64, seed=seed)
        if boost != 1.0:
            with torch.no_grad():
                model.output_layer.weight.mul_(boost)
        model.eval()
        return E.Base(model, cfg, tok, {}, world, texts[:200], texts[200:220], pathlib.Path("/nonexistent-ka-base"),
                      model_sha256(model))
    finally:
        torch.set_num_threads(threads)


__all__ = ["tiny_base"]
