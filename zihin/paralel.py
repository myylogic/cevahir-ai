# -*- coding: utf-8 -*-
"""At the same time, with nothing lost (paralel).

A mind lives one moment after another, and what a moment writes depends on what the mind expected, which depends on
everything written before. Two stretches of a life therefore cannot be lived at the same time and leave the mind
that living them in turn leaves: that was measured, and the technique that tried it was taken out again
(research/living_mind_2026_11_22/eliminated_siblings). What can be done at the same time *exactly* is what writes
nothing into the mind that asks:

* **Answering many questions, thinking many thoughts.** Each is answered by the whole mind as it is at this instant:
  a forked process is that mind (its memory, not a description of it), it answers one question as the mind would
  answer it now, and ends. The mind that asked is not touched: no moment of these answers is in its life or its
  ledger. The answers are, word for word, what the mind gives when it answers each on a copy of itself, one after
  another (pinned by a test); only the time differs, about by the number of processors.

So this is the mind considering several things at once, each from where it stands now; none of them is lived. To
live one of them (to learn from having thought it, to get feedback on an answer) the mind does it in turn, as before.

``at_once(session, jobs, work, workers)`` is the general form; ``answers`` and ``thoughts`` are the two the program
uses (``!sor_hepsi``, ``!dusun_hepsi``).
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import torch

_SESSION = None       # the session the children are forked with
_WORK = None


def _copy_of(session):
    """In a child: a session of its own over the mind it was forked with, set as the asking session is (who is
    spoken with, word pieces, language), and cut off from everything outside the child (the ledger, the stop
    signal): whatever it lives ends with it."""
    from .__main__ import Session
    mind = session.mind
    mind.S.log = {}                                       # nothing of this reaches the ledger of the mind that asked
    copy = Session(mind, session.pieces)
    copy.partner, copy.dil = session.partner, session.dil
    return copy


def _child(job):
    torch.set_num_threads(1)
    return _WORK(_copy_of(_SESSION), job)


def at_once(session, jobs: List[object], work: Callable[[object, object], object],
            workers: Optional[int] = None) -> List[object]:
    """``work(a copy's session, job)`` for every job at the same time, each in a process that is the mind as it is
    now; the results in the order of the jobs. ``work`` is a function of a module and its results plain data (they
    cross from one process to another). The asking mind is not changed. With one job, or one worker, the same is
    done one after another on copies (``Mind.clone``): the same results."""
    global _SESSION, _WORK
    if not jobs:
        return []
    k = min(workers or os.cpu_count() or 1, len(jobs))
    if k <= 1:
        return [work(_one_after_another(session), job) for job in jobs]
    import multiprocessing
    _SESSION, _WORK = session, work
    try:
        with multiprocessing.get_context("fork").Pool(k, maxtasksperchild=1) as pool:
            return pool.map(_child, jobs, chunksize=1)
    finally:
        _SESSION = _WORK = None


def _one_after_another(session):
    """The same copy made without a process: a clone of the mind (no ledger) with a session set as the asking one."""
    from .__main__ import Session
    copy = Session(session.mind.clone(), session.pieces)
    copy.partner, copy.dil = session.partner, session.dil
    return copy


def _answer(copy, job) -> str:
    question, limit = job
    return copy.reply(question, limit=limit)


def _think(copy, job) -> Tuple[List[str], float]:
    from .dusunce import think
    topic, limit = job
    words, reality = think(copy, topic, limit=limit)
    return list(words), float(reality)


def answers(session, questions: List[str], limit: int = 80, workers: Optional[int] = None) -> List[str]:
    """What the mind answers to each question as it stands now, all at the same time; none of them is lived."""
    return at_once(session, [(q, limit) for q in questions], _answer, workers)


def thoughts(session, topics: List[str], limit: int = 15, workers: Optional[int] = None) -> List[Tuple[List[str], float]]:
    """What the mind thinks about each topic as it stands now (its inner words and their reality score), all at the
    same time; none of them is lived."""
    return at_once(session, [(t, limit) for t in topics], _think, workers)
