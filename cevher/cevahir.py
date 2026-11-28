# -*- coding: utf-8 -*-
"""Cevahir, the final form: one mind over one substance (docs/architecture/FINAL_FORM.md).

The interface is the one ``living.cevahir.Cevahir`` has, so that the same batteries run on both: ``tell``,
``watch``, ``read``, ``converse``, ``feedback``, ``forget``, ``sleep``, ``save``, ``load``, ``answers``.
"""
from __future__ import annotations

import re
from typing import List, Optional

from cevher.mind import Mind
from cevher.store import Exact
from cevher.units import body, tokens

SEPARATOR = re.compile(r"(?<=[.!?…])\s+|\n+")              # Zihin's sentence split (zihin.__main__.sentences)


class Cevahir:
    def __init__(self, ledger: bool = False, store=None):
        self.store = store if store is not None else Exact()
        self.mind = Mind(self.store, ledger=ledger)
        self.entries: Optional[List[dict]] = [] if ledger else None
        self.pending = None                                  # the last turn of a conversation, not yet judged

    # --------------------------------------------------------------------------------------------- the ledger ----
    def _begin(self, kind: str, text: str):
        if self.entries is None:
            return None
        self.mind.entry = []
        return {"kind": kind, "text": text, "forgotten": False}

    def _end(self, entry) -> None:
        if entry is None:
            return
        entry["writes"] = self.mind.entry
        self.mind.entry = None
        self.entries.append(entry)

    # ------------------------------------------------------------------------------------------------- living ----
    def tell(self, sentence: str) -> None:
        """A sentence told."""
        self.flush()
        entry = self._begin("tell", sentence)
        toks, _ = body(tokens(sentence))
        if toks:
            self.mind.tell(toks)
        self._end(entry)

    def watch(self, first: str, answer: str) -> None:
        """A dialogue watched: someone says ``first``, someone answers."""
        self.flush()
        entry = self._begin("watch", first + "\t" + answer)
        q, _ = body(tokens(first))
        a, mark = body(tokens(answer))
        self.mind.watch(q, a, mark)
        self._end(entry)

    def read(self, text: str) -> float:
        """A document read: each of its sentences told."""
        self.flush()
        entry = self._begin("read", text[:60])
        for piece in SEPARATOR.split(text):
            toks, _ = body(tokens(piece))
            if len(toks) >= 2:
                self.mind.tell(toks)
        self._end(entry)
        return 0.0

    # ----------------------------------------------------------------------------------------------- speaking ----
    def reply(self, said: str) -> str:
        """What Cevahir answers, as it stands; nothing is lived."""
        q, _ = body(tokens(said))
        got = self.mind.answer(q) if q else None
        if got is None:
            return ""
        return " ".join(got[0]) + ((" " + got[2]) if got[2] else "")

    def answers(self, questions: List[str], limit: int = 400, workers: int = 0) -> List[str]:
        """What Cevahir answers to each question as it stands now; nothing of the answers is lived."""
        return [self.reply(q) for q in questions]

    def converse(self, said: str, limit: int = 80) -> str:
        """A turn of a conversation: Cevahir hears what is said to it and answers. The turn is lived once it is
        judged (``feedback``) or the next thing happens (``flush``)."""
        self.flush()
        answer = self.reply(said)
        self.pending = (said, answer)
        return answer

    def flush(self) -> None:
        """The last turn, lived as it was: nobody judged it. What Cevahir answered is a dialogue it watched itself
        give; what it could not answer is something it was told."""
        if self.pending is None:
            return
        said, answer = self.pending
        self.pending = None
        if answer:
            self.watch(said, answer)
        else:
            self.tell(said)

    def feedback(self, good: bool) -> None:
        """How Cevahir's last answer was received. A right answer is lived as a dialogue; a wrong one is not lived
        as something said rightly: the question alone is, and the way that spoke is believed less."""
        if self.pending is None:
            return
        said, answer = self.pending
        self.pending = None
        if good and answer:
            self.watch(said, answer)
        else:
            entry = self._begin("judged wrong", said + "\t" + answer)
            q, _ = body(tokens(said))
            if answer:
                self.mind.wrong(q)
            self.mind.sequence(self.mind.codes(q), __import__("cevher.mind", fromlist=["O_ASKED"]).O_ASKED)
            self._end(entry)

    def forget(self, k: int) -> dict:
        """Forget entry ``k`` of the history exactly: the same writes with the opposite sign."""
        if self.entries is None:
            raise ValueError("Cevahir keeps no ledger (Cevahir(ledger=True))")
        self.flush()
        e = self.entries[k]
        if e["forgotten"]:
            return {"forgotten": False}
        self.mind.take_back(e["writes"])
        e["forgotten"] = True
        return {"forgotten": True, "writes": len(e["writes"])}

    def sleep(self) -> dict:
        self.flush()
        return {}

    def size(self) -> dict:
        return dict(self.store.size(), words=len(self.mind.spell), bytes=self.store.nbytes())
