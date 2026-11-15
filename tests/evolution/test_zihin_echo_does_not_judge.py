# -*- coding: utf-8 -*-
"""Zihin: the echo of the mind's own acts is certain (acted evidence has trust 1) but it is not evidence about the
world, so it does not judge told, imitated, inferred or imagined claims: replying moves no learned trust. Only
sensed evidence judges (events.JUDGES)."""
from __future__ import annotations

import torch

from zihin import Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.events import ACTED, JUDGES, REFERENCE, SENSED


def test_only_sensed_evidence_judges():
    assert JUDGES == (SENSED,) and ACTED in REFERENCE


def test_replying_moves_no_learned_trust():
    mind = Mind(VARIANTS["speaking"]())
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(4):
        session.watch("kedi ne yer ?", "kedi balık yer .")
        session.watch("köpek ne yer ?", "köpek kemik yer .")
    before = mind.trust.clone()
    for question in ("kedi ne yer ?", "köpek ne yer ?", "kedi ne yer ?", "kuş ne yer ?"):
        session.reply(question)
    assert torch.equal(mind.trust, before)
