# -*- coding: utf-8 -*-
"""Introspection (LAYERS.md L8): what the mind can read about itself from its own evidence.

The core holds no sentences about the mind. What it is, what it is called and how it says that it does not know are
learned from the people it lives with and from the data it is given, like everything else. This module only reads the
ties of the self unit, which accompanies every moment the mind itself speaks or thinks.
"""
from __future__ import annotations


def self_report(mind, channel: str = "text", top: int = 12) -> str:
    """Introspection (L8), printed as a labelled readout of the evidence (not as sentences in the mind's mouth): how
    much of its life was its own speaking and its own thinking, and the words its self unit is bound to beyond their
    base rate. Read from the ties of the self unit, which accompanies every moment the mind itself acts.

    A word's binding is its lift, the share of the self unit's evidence that goes to it over the share of all use
    that goes to it, weighted by the conviction of that evidence (c = W / (W + 1), the one rule's conviction): a word
    the self is tied to only because every moment holds it has a lift near 1; no word is excluded by its shape or by
    a frequency cut-off."""
    from .events import ACTED, IMAGINED
    from .mind import SELF_CHANNEL, SELF_SYMBOL
    from .store import EXPECT, W_POS
    S = mind.S
    if not mind.ch.has(SELF_CHANNEL):
        return "  öz-birim: henüz yok"
    sid = mind.ch.id(SELF_CHANNEL)
    index = mind.ch.symbol_index(sid, SELF_SYMBOL, create=False)
    me = S.atom_index.get((sid, index), -1) if index >= 0 else -1
    if me < 0:
        return "  öz-birim: kanıt yok (zihin henüz kendi sesiyle hiç konuşmadı, içinden düşünmedi)"
    cid = mind.ch.id(channel)
    words = [u for u in range(S.n_units) if int(S.u_kind[u]) & 1 and int(S.u_channel[u]) == cid and S.u_alive[u]]
    total_use = sum(float(S.u_use[u]) for u in words) or 1.0
    held, spoke, thought = [], 0.0, 0.0
    for tie in S.out.get((me, EXPECT, 0), []):
        dst = int(S.t_dst[tie])
        w = S.w1(W_POS, tie)
        spoke += float(w[ACTED])
        thought += float(w[IMAGINED])
        if int(S.u_channel[dst]) == cid:
            held.append((dst, float((w * mind.trust).sum()), float(w.sum())))
    total_self = sum(m for _, m, _ in held) or 1.0
    bound = []
    for dst, mass, raw in held:
        base = float(S.u_use[dst]) / total_use
        if base > 0.0 and mass > 0.0:
            lift = (mass / total_self) / base
            bound.append((lift * raw / (raw + 1.0), str(mind.ch.symbol(cid, int(S.u_symbol[dst])))))
    bound.sort(key=lambda b: (-b[0], b[1]))
    lines = ["  öz-birim: kendi eylemi %.0f kelime-an, iç ses (hayal) %.0f kelime-an" % (spoke, thought),
             "  öz-birimin taban oranının üstünde bağlı olduğu kelimeler (kaldıraç x kanaat): "
             + ", ".join(w for _, w in bound[:top])]
    return "\n".join(lines)
