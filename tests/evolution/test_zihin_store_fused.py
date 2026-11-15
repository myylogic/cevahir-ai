# -*- coding: utf-8 -*-
"""Zihin store: the fused reads equal the assembled rows, also after a trust vector is changed in place through
numpy (which does not count in a tensor's version), and a moment's touched ties are kept as int64 whatever the
dtype of the ids that were written."""
from __future__ import annotations

import torch

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session


def lived() -> Mind:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.watch("kedi ne yer ?", "kedi balık yer .")
        session.watch("köpek ne yer ?", "köpek kemik yer .")
    return mind


def test_fused_reads_follow_a_trust_changed_through_numpy():
    mind = lived()
    S = mind.S
    ids = torch.arange(S.n_ties, dtype=torch.int64)
    trust = mind.trust.clone()
    S.mass2(ids, trust)                      # fills the lookup for these values
    trust.numpy()[:] *= 0.5                  # in place, without a version bump
    pos, neg = S.w2(ids)
    got = S.mass2(ids, trust)
    assert torch.equal(got[0], (pos * trust).sum(-1)) and torch.equal(got[1], (neg * trust).sum(-1))


def test_touched_ties_are_int64_for_int32_ids():
    mind = lived()
    S = mind.S
    assert S.keep_log
    S.begin(mind.t + 1, mind.ch.sizes())
    ids = torch.tensor([0, 1], dtype=torch.int32)
    S.add(0, 1, ids, torch.tensor([0.5, 0.25], dtype=torch.float64))
    cur = S.end()
    assert cur.last is not None and cur.last[0].dtype == torch.int64
