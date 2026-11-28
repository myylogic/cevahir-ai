# -*- coding: utf-8 -*-
"""Cevahir, the final form: one mind over one substance (docs/architecture/FINAL_FORM.md).

The interface is the one ``living.cevahir.Cevahir`` has, so that the same batteries run on both: ``tell``,
``watch``, ``read``, ``converse``, ``feedback``, ``forget``, ``sleep``, ``save``, ``load``, ``answers``.

The substance is three tables of fixed capacity: the ties of what was lived a sentence or a dialogue at a time,
the ties of what long texts wrote (``cevher.store.Slots`` both, read as one by ``cevher.store.Two``), and the
letters' contexts (``cevher.letters.Counts``); and the spellings of the words met. Everything lived is written into
them in one forward pass; nothing is kept per experience except, with a ledger, what is needed to take the
experience back: its text, the writes that were not a function of the text alone, and the ties that gave way to it.

NumPy and the standard library only: no compiled code, no network, no pickle.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zlib
from typing import List, Optional

import numpy as np

from cevher.letters import Counts, Letters
from cevher.mind import O_ASKED, Mind
from cevher.store import Slots, Two
from cevher.units import LINE, TOKEN, Spellings, body, code_of, lower_tr, mix, rule, tokens

SEPARATOR = re.compile(r"(?<=[.!?…])\s+|\n+")              # Zihin's sentence split (zihin.__main__.sentences)
R_FORM, R_GLUE = code_of("\0role written form"), code_of("\0role gap")
GLUED, APART = 1, 2
OVERHEAD_MB = 224                                            # what the budget leaves outside the tables: the
#   interpreter and NumPy (40 MB), the spellings, and the working memory of living (measured: gates.py bound)
PER_CHARACTER = 8                                            # bytes the tables may hold for every character lived
FLOOR_MB = 32                                                # ... and what they may hold however little was lived
BULK = 4096                                                  # a text longer than this is read many ties at once
GROUP = 1 << 15                                              # ... so many of its words at a time, about
VERSION = 4                                                  # of the file a mind is kept in


def _spans(text: str, lines: bool):
    """Where each sentence of a text lies (each line, for source code)."""
    at = 0
    if lines:
        while True:
            j = text.find("\n", at)
            if j < 0:
                break
            yield at, j
            at = j + 1
    else:
        for m in SEPARATOR.finditer(text):
            yield at, m.start()
            at = m.end()
    yield at, len(text)


def _groups(text: str, lines: bool):
    """The sentences of two words or more of a text, in groups of about ``GROUP`` words (a group ends with the
    sentence that fills it), and where each group lies in the text. A text is never held whole as words: the
    working memory of reading is a group's, however long the text."""
    group, held, first, last = [], 0, 0, 0
    for a, b in _spans(text, lines):
        s = body(tokens(text[a:b]))[0]
        if len(s) >= 2:
            if not group:
                first = a
            group.append(s)
            held += len(s)
            last = b
            if held >= GROUP:
                yield group, first, last
                group, held = [], 0
    if group:
        yield group, first, last


class Cevahir:
    """One mind. ``budget_mb`` bounds its resident memory for any length of life: the tables may grow to what is
    left of it after ``OVERHEAD_MB`` and no further: an eighth for the ties of what was lived a sentence or a
    dialogue at a time, five eighths for the ties of what long texts wrote, a quarter for the letters. On the way
    there, what texts write holds at most ``per_character`` bytes for every character lived (and ``FLOOR_MB``
    however little was lived): memory a character is bounded by a constant first, and by the budget in the end.
    ``ledger``: keep what is needed to forget an entry exactly (a file in a temporary folder, or in the folder
    given)."""

    def __init__(self, budget_mb: int = 512, ledger=False, depth: int = 5, per_character: int = PER_CHARACTER):
        rest = max(1, budget_mb - OVERHEAD_MB) << 20
        self.budget_mb, self.rest, self.per_character = budget_mb, rest, per_character
        self.store = Two(Slots(max_bytes=rest // 8), Slots(max_bytes=rest * 5 // 8))
        self.mind = Mind(self.store)
        self.mind.spell.most = max(100000, budget_mb * 2000) # the words it can spell are bounded with the budget
        self.letters = Letters(Counts(max_bytes=rest // 4), depth=depth)
        self.entries: Optional[List[dict]] = None
        self._ledger = None
        self.pending = None                                  # the last turn of a conversation, not yet judged
        self.nights = 0
        if ledger:
            self.keep_ledger(ledger if isinstance(ledger, str) else None)

    # --------------------------------------------------------------------------------------------- the ledger ----
    def keep_ledger(self, folder: Optional[str] = None) -> str:
        """Keep a ledger from now on: one file, an entry appended for every experience lived."""
        if folder is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="cevher-ledger-")
            folder = self._tmp.name
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "ledger.bin")
        self._ledger = open(path, "a+b")
        self.entries = []
        return path

    def _begin(self) -> Optional[dict]:
        if self.entries is None:
            return None
        self.mind.entry = []
        return {"tail": self.letters.tail, "lgone": [], "sgone": [], "sprev": [], "smoment": []}

    def _end(self, open_entry: Optional[dict], kind: str, text: str, stream: str, bulk: int = 0) -> None:
        """An entry of the history is closed and written to the ledger: what was lived (``stream``, as the letters
        lived it), the writes of the word grain that are not a function of the text alone, and what gave way."""
        if open_entry is None:
            return
        log = self.mind.entry or []
        self.mind.entry = None
        arrays = {"stream": np.frombuffer(stream.encode("utf-8", "surrogatepass"), dtype=np.uint8),
                  "tail": np.frombuffer(open_entry["tail"], dtype=np.uint8),
                  "log": np.array([(1,) + tuple(r) + (0,) if len(r) == 4 else (0,) + tuple(r)
                                   for r in log], dtype=np.uint64).reshape(-1, 6),
                  "bulk": np.array([int(bulk)])}            # 0, or a text read many ties at once: 1, by lines 2
        lg = open_entry["lgone"]
        arrays["lg_at"] = np.array([(a, b, len(g[0])) for a, b, g in lg], dtype=np.int64).reshape(-1, 3)
        arrays["lg_k"] = np.concatenate([g[0] for _, _, g in lg]) if lg else np.zeros(0, dtype=np.uint64)
        arrays["lg_c"] = np.concatenate([g[1] for _, _, g in lg]) if lg else np.zeros(0, dtype=np.uint32)
        sg = open_entry["sgone"]
        arrays["sg_n"] = np.array([0 if g is None else len(g[0]) for g in sg], dtype=np.int64)
        for i, name in enumerate(("sg_k", "sg_v", "sg_c", "sg_t")):
            parts = [g[i] for g in sg if g is not None]
            arrays[name] = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
        sp = open_entry["sprev"]                             # the moments a long text's ties had before each group
        arrays["sp_n"] = np.array([len(x) for x in sp], dtype=np.int64)
        arrays["sp"] = np.concatenate(sp) if sp else np.zeros(0, dtype=np.uint32)
        arrays["sm"] = np.array(open_entry["smoment"], dtype=np.int64)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        blob = zlib.compress(buf.getbuffer(), 1)
        self._ledger.seek(0, os.SEEK_END)
        at = self._ledger.tell()
        self._ledger.write(blob)
        self.entries.append({"kind": kind, "text": text[:80], "at": at, "size": len(blob), "forgotten": False})

    def ledger_bytes(self) -> int:
        if self._ledger is None:
            return 0
        self._ledger.seek(0, os.SEEK_END)
        return self._ledger.tell()

    # ------------------------------------------------------------------------------------------------- living ----
    def _letters(self, stream: str, open_entry: Optional[dict]) -> float:
        """The letters live a stream. Before it, the tables that texts write are told how far they may grow: so
        many bytes for every character lived (this stream counted), a floor, and never more than the budget
        leaves. (The table of what is lived a dialogue at a time grows as it fills, to its share of the budget.)"""
        lived = self.letters.lived + len(stream)
        allowed = min(self.rest, max(FLOOR_MB << 20, self.per_character * lived))
        self.store.read.allow(allowed * 3 // 4)
        self.letters.T.allow(allowed // 4)
        return self.letters.live(stream, gone=None if open_entry is None else open_entry["lgone"])

    def tell(self, sentence: str) -> float:
        """A sentence told."""
        self.flush()
        e = self._begin()
        stream = sentence + "\n"
        bits = self._letters(stream, e)
        toks, _ = body(tokens(sentence))
        if toks:
            self.mind.tell(toks)
        self._end(e, "tell", sentence, stream)
        return bits

    def watch(self, first: str, answer: str) -> float:
        """A dialogue watched: someone says ``first``, someone answers."""
        self.flush()
        e = self._begin()
        stream = first + "\t" + answer + "\n"
        bits = self._letters(stream, e)
        q, _ = body(tokens(first))
        a, mark = body(tokens(answer))
        self.mind.watch(q, a, mark)
        self._forms(answer)
        self._end(e, "watch", first + " | " + answer, stream)
        return bits

    def read(self, text: str, code: bool = False) -> float:
        """A document read: the letters live every byte of it; the words live each of its sentences of two words
        or more (each line, for source code). A long text is written many ties at once."""
        self.flush()
        e = self._begin()
        bits = self._letters(text, e)
        bulk = len(text) >= BULK
        for group, _, _ in _groups(text, code):
            if bulk:
                gone = self.mind.sequences(group, before=None if e is None else e["sprev"])
                if e is not None:
                    e["sgone"].append(gone)
                    e["smoment"].append(self.mind.clock)
            else:
                for s in group:
                    self.mind.tell(s)
        self._end(e, "read", text, text, bulk=(2 if code else 1) if bulk else 0)
        return bits

    def _forms(self, answer: str) -> None:
        """How an answer was written: the form of each token where it is not its lower case, and whether it stood
        apart from the token before it. Ties like any others."""
        prev, end = None, 0
        for m in TOKEN.finditer(answer):
            raw = m.group()
            c = self.mind.code(lower_tr(raw))
            if raw != lower_tr(raw):
                form = code_of(raw)
                self.mind.spell.setdefault(form, raw)
                self.mind._tie(mix(R_FORM, c), form)
            if prev is not None:                             # of the pair, and of each of the two whatever the other
                gap = GLUED if m.start() == end else APART
                self.mind._tie(mix(mix(R_GLUE, prev), c), gap)
                self.mind._tie(mix(mix(R_GLUE, prev), 1), gap)
                self.mind._tie(mix(mix(R_GLUE, 2), c), gap)
            prev, end = c, m.end()

    # ----------------------------------------------------------------------------------------------- speaking ----
    def _say(self, words: List[str], mark: str) -> str:
        """An answer as it is written: each token in the form it was written in, apart from the one before it or
        not, as lived (each by the rule, at the speaking line; a token never written so is written as it is)."""
        S, out, prev = self.store, "", None
        for w in words + ([mark] if mark else []):
            c = self.mind.code(w)
            key = mix(R_FORM, c)
            vs = S.vals(key)
            form = w
            if vs and rule(vs[0][1], S.tot(key)) >= LINE:
                form = self.mind.spell.get(vs[0][0], w)
            gap = " "
            if prev is not None:
                pair = self._gap(mix(mix(R_GLUE, prev), c))
                if pair is None:                             # the two never stood together: each as it was written
                    after, before = self._gap(mix(mix(R_GLUE, prev), 1)), self._gap(mix(mix(R_GLUE, 2), c))
                    pair = GLUED if GLUED in (after, before) and APART not in (after, before) else APART
                if pair == GLUED:
                    gap = ""
            out += (gap if out else "") + form
            prev = c
        return out

    def _gap(self, key: int):
        vs = self.store.vals(key)
        return vs[0][0] if vs and rule(vs[0][1], self.store.tot(key)) >= LINE else None

    def reply(self, said: str) -> str:
        """What Cevahir answers, as it stands; nothing is lived."""
        q, _ = body(tokens(said))
        got = self.mind.answer(q) if q else None
        if got is None or None in got[0]:                    # nothing believed, or a word it can no longer spell
            return ""
        return self._say(got[0], got[2])

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
        """The last turn, lived as it was: nobody judged it. What Cevahir answered is lived as something asked and
        something said, and as no evidence that the answer was right. What it could not answer is something it
        was told, unless it reads it as a question (one it knows how to read and has nothing to answer with):
        that is lived as asked."""
        if self.pending is None:
            return
        said, answer = self.pending
        self.pending = None
        q = body(tokens(said))[0]
        if not answer and not self.mind.asks(q):
            self.tell(said)                                  # nothing in it is a question to this mind: it was told
            return
        e = self._begin()
        stream = said + "\t" + answer + "\n"
        self._letters(stream, e)
        self.mind.heard(q, body(tokens(answer))[0])
        self._end(e, "converse", said + " | " + (answer or "(sessiz)"), stream)

    def feedback(self, good: bool) -> None:
        """How Cevahir's last answer was received. A right answer is lived as a dialogue (evidence for the way
        that gave it). A wrong one is not lived as something said rightly: the question alone is, and the way
        that spoke is believed less."""
        if self.pending is None:
            return
        said, answer = self.pending
        self.pending = None
        if good and answer:
            self.watch(said, answer)
            self.entries and self.entries[-1].update(kind="converse, judged right")
            return
        e = self._begin()
        stream = said + "\n"
        self._letters(stream, e)
        q, _ = body(tokens(said))
        if answer:
            self.mind.wrong(q)
        self.mind.sequence(self.mind.codes(q), O_ASKED)
        self._end(e, "converse, judged wrong", said + " | " + answer, stream)

    def forget(self, k: int) -> dict:
        """Forget entry ``k`` of the history: everything it wrote is taken back, the last write first, and every tie
        that gave way to it is put back. Forgetting the latest entry leaves the substance as if the entry had
        never been lived (its digest is equal); so does forgetting an earlier one unless ties gave way since."""
        if self.entries is None:
            raise ValueError("Cevahir keeps no ledger (Cevahir(ledger=True))")
        self.flush()
        e = self.entries[k]
        if e["forgotten"]:
            return {"forgotten": False}
        self._ledger.seek(e["at"])
        with np.load(io.BytesIO(zlib.decompress(self._ledger.read(e["size"])))) as z:
            a = {name: z[name] for name in z.files}
        stream = a["stream"].tobytes().decode("utf-8", "surrogatepass")
        # the word grain: the writes kept, the last first; then what a long text wrote, a function of the text
        log = [tuple(int(x) for x in r[1:5]) if r[0] else tuple(int(x) for x in r[1:]) for r in a["log"]]
        self.mind.take_back(log)
        bulk = int(a["bulk"][0])
        if bulk:                                             # the groups as they were written, the last first
            where = [(x, y) for _, x, y in _groups(stream, bulk == 2)]
            at, sp_at = len(a["sg_k"]), len(a["sp"])
            for (x, y), n, held, moment in reversed(list(zip(where, a["sg_n"].tolist(), a["sp_n"].tolist(),
                                                              a["sm"].tolist()))):
                group = [s for g, _, _ in _groups(stream[x:y], bulk == 2) for s in g]
                self.mind.sequences(group, sign=-1, before=a["sp"][sp_at - held:sp_at], moment=moment)
                sp_at -= held
                if n:
                    self.store.put_back_many(tuple(a[name][at - n:at] for name in ("sg_k", "sg_v", "sg_c", "sg_t")))
                    at -= n
        # the letters
        gone, at = [], 0
        for start, end, n in a["lg_at"].tolist():
            gone.append((start, end, (a["lg_k"][at:at + n], a["lg_c"][at:at + n])))
            at += n
        self.letters.forget(stream, a["tail"].tobytes(), gone)
        e["forgotten"] = True
        return {"forgotten": True, "writes": len(log), "characters": len(stream)}

    def sleep(self, keep_min: int = 2) -> dict:
        """A night: the letters let go of the contexts met fewer than ``keep_min`` times (what recurred stays
        exactly); the words keep everything (among the ties met once is what one reading taught); the spellings
        are packed."""
        self.flush()
        self.nights += 1
        T = self.letters.T
        weak = np.flatnonzero((T.key != 0) & (T.cnt < keep_min))
        T.key[weak], T.cnt[weak] = 0, 0
        T.used -= len(weak)
        self.mind.spell.pack()
        return {"letters_forgotten": int(len(weak)), "letters_kept": int(T.used)}

    # ------------------------------------------------------------------------------------------------ keeping ----
    def digest(self) -> str:
        """A digest of the substance: every tie of both tables with its count, wherever it sits."""
        return "%016x%016x" % (self.store.digest(), self.letters.T.digest())

    def moments(self) -> str:
        """A digest of when each tie of words was last written: what was told last is read from it, so forgetting
        must bring it back too."""
        return "%016x" % self.store.moments()

    def size(self) -> dict:
        return {"ties": self.store.size(), "lived": self.store.lived.size(), "read": self.store.read.size(),
                "letters": self.letters.size(), "words": len(self.mind.spell),
                "table_bytes": self.store.nbytes() + self.letters.T.nbytes(),
                "spelling_bytes": self.mind.spell.nbytes(), "moments": self.mind.clock}

    def save(self, folder: str) -> dict:
        """Keep Cevahir in ``folder``: one file of arrays (``cevher.npz``, compressed) and a manifest with its
        SHA-256. No pickle: loading executes nothing."""
        self.flush()
        os.makedirs(folder, exist_ok=True)
        arrays = {}
        for prefix, st in (("s_", self.store.lived.state()), ("r_", self.store.read.state()),
                           ("l_", self.letters.T.state()), ("w_", self.mind.spell.state())):
            for k, v in st.items():
                arrays[prefix + k] = v
        arrays["zero"], arrays["sit"] = self.letters.zero, self.letters.sit
        arrays["tail"] = np.frombuffer(self.letters.tail, dtype=np.uint8)
        arrays["misc"] = np.array([self.mind.clock, self.letters.lived, self.letters.depth, self.nights,
                                   self.budget_mb], dtype=np.int64)
        path = os.path.join(folder, "cevher.npz")
        np.savez_compressed(path, **arrays)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        manifest = {"what": "Cevahir, the final form (cevher)", "version": VERSION, "sha256": h.hexdigest(),
                    "digest": self.digest(), "size": self.size()}
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        return manifest

    @classmethod
    def load(cls, folder: str, budget_mb: Optional[int] = None, ledger=False) -> "Cevahir":
        """Cevahir read back from ``folder`` after the file is checked against the manifest."""
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        path = os.path.join(folder, "cevher.npz")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != manifest["sha256"]:
            raise IOError("cevher.npz does not match its manifest")
        if manifest.get("version") != VERSION:
            raise IOError("cevher.npz was kept by another version of the package (%r, this is %d): raise the mind "
                          "again" % (manifest.get("version"), VERSION))
        with np.load(path, allow_pickle=False) as z:
            a = {name: z[name] for name in z.files}
        clock, lived, depth, nights, budget = (int(x) for x in a["misc"])
        self = cls(budget_mb=budget_mb or budget, depth=depth)
        self.letters.lived = lived
        allowed = min(self.rest, max(FLOOR_MB << 20, self.per_character * lived))    # the growth law, as in living
        tables = []
        for prefix, most, law in (("s_", self.store.lived.max_buckets, 0),
                                  ("r_", self.store.read.max_buckets, allowed * 3 // 4)):
            st = {k[2:]: v for k, v in a.items() if k.startswith(prefix)}
            st["nb"] = np.array([st["nb"][0], max(int(st["nb"][0]), most), st["nb"][2], st["nb"][3]])
            tables.append(Slots.from_state(st, law))
        self.store = Two(*tables)
        self.mind.S = self.store
        st = {k[2:]: v for k, v in a.items() if k.startswith("l_")}
        st["nb"] = np.array([st["nb"][0], max(int(st["nb"][0]), self.letters.T.max_buckets), st["nb"][2],
                             st["nb"][3]])
        self.letters.T = Counts.from_state(st, allowed // 4)
        self.mind.spell = Spellings.from_state({k[2:]: v for k, v in a.items() if k.startswith("w_")},
                                               most=self.mind.spell.most)
        self.letters.zero, self.letters.sit = a["zero"], a["sit"]
        self.letters.tail, self.letters.lived = a["tail"].tobytes(), lived
        self.mind.clock, self.nights = clock, nights
        if ledger:
            self.keep_ledger(ledger if isinstance(ledger, str) else None)
        return self
