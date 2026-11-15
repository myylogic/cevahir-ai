# -*- coding: utf-8 -*-
"""Locality of a living mind's memory accesses: how much of the mind a moment, an utterance, a dialogue and a topic
really touch, how predictable the next accesses are, and how often old ties are used again. This is the empirical
basis for loading only the region of the mind that a moment needs.

Nothing in ``zihin/`` is changed. The recorder wraps the accessors of the loaded mind's own Substance as instance
attributes; every wrapper calls the original and only records:

* ``w`` (``w2`` and ``w4`` read through it), ``w1``, ``cell``: evidence read; ``column``: a whole-memory scan;
* ``_put`` (the ties ``add`` writes, after its zero amounts are dropped), ``add1`` (when its snapped amount is not 0):
  evidence written; ``use_ties``: the use count written;
* ``tie`` / ``find_tie``: a tie found through the tie index, or born;
* ``out_ids``: the contents of an adjacency list read (the cached id tensors);
* ``S.out`` / ``S.inc`` become dict subclasses: a lookup (get, [], in) records the key, appending a new tie
  (setdefault) records a write, and the lists record when their contents are read (iterated, indexed, sliced or
  converted). A list looked up but never read was only asked for its existence or its length;
* ``dusunce.frequent_words``, the one scan of the whole unit table on the reply path (``Session.think_aloud``): counted
  and timed (unit-level structures are otherwise not measured).

Per moment (one ``Mind.step``, plus whatever the session reads before the next one) the trace keeps the tie ids
touched with flags (read, written, use written, found, born) and the adjacency keys touched with flags (looked up,
contents read, appended to) and the list length at the read, each in the order of first touch. Keys are
``(src, form, lag)`` of ``S.out`` (PART with lag -1) and ``(dst, form)`` of ``S.inc``.

``verify`` runs the same protocol without the recorder and compares, moment by moment, the acts, the focus and the
predictions (one running sha256 over their exact reprs), the answers (hashed) and the final substance digest.

Protocol per mind: watch the first 60 dialogues of the private daily-conversation file, then reply to ten questions
of this script's own about the weather and ten about food (TOPICS). No line of the private file, no answer and no
generated word is printed or written: the results are numbers only.

The page model (section 3 of the task) is a hypothetical CSR layout of the same ties: tie records of TIE_BYTES sorted
by (source unit, form, lag) with creation order inside a key; a reverse index of INC_BYTES per entry sorted by (target
unit, form); and a directory of DIR_BYTES per unit (the offsets of its keys). Every key lookup touches its unit's
directory page; reading a list's contents touches the pages of its first ``len`` records (or reverse entries);
evidence read, written or found touches the page of the tie's record; a born tie also touches its reverse entry.
The layout is that of the ties at the end of the run, so a tie born during the run sits where it will end up. Orders
of the units compared: creation (unit id), first use in this run (an in-sample bound), first use in the first 30
dialogues (applied to the rest), hot units first (u_use), label propagation regions (with and without the 1% units
with most neighbours), breadth-first from the most used units.

    P=private_data; R=research/living_mind_2026_11_15/locality_profile.py; W=<scratch dir>
    python $R selftest
    nice -n 10 python $R run --data $P --mind $P/gonder/cevahir_ilk_nefes.zihin --name ilk_nefes --work $W
    nice -n 10 python $R run --data $P --mind $P/gonder/cevahir_ilk_nefes.zihin --name ilk_nefes --work $W --plain
    python $R verify --name ilk_nefes --work $W
    nice -n 10 python $R analyze --name ilk_nefes --work $W     # rows -> results/locality_profile.jsonl
    (the same for --mind $P/cevahir2.zihin --name cevahir2, and $P/cevahir3b.zihin --name cevahir3b when free -g shows
    at least 7 GB available; analyze --only analyze_need,... runs single parts)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "locality_profile.jsonl")
DIALOGUES = "gunluk-konusma.json"          # found in the data folder by this ending
WATCH = 60
TOPICS = (("hava", ("bugün hava nasıl ?", "dışarıda yağmur yağıyor mu ?", "hava soğuk mu sıcak mı ?",
                    "yarın kar yağacak mı ?", "güneşli havayı sever misin ?", "rüzgar çok sert esiyor mu ?",
                    "şemsiye almam gerekir mi ?", "bu hafta hava nasıl olacak ?", "yağmurlu günlerde ne yaparsın ?",
                    "kışın hava çok soğuk olur mu ?")),
          ("yemek", ("en sevdiğin yemek hangisi ?", "kahvaltıda ne yemeyi seversin ?", "çay mı kahve mi içersin ?",
                     "akşam yemeğinde ne yiyelim ?", "tatlı yemeyi sever misin ?", "hiç pizza yedin mi ?",
                     "meyve sever misin ?", "yemek yapmayı bilir misin ?", "en sevdiğin içecek ne ?",
                     "bugün öğle yemeğinde ne yedin ?")))
PAGE = 4096
TIE_BYTES = 64     # a tie record without its source (implicit in CSR): target, form, lag, class, w+, w-, use, last,
                   # birth (56 B as the store keeps them today), rounded up to a cache line
INC_BYTES = 16     # a reverse entry: tie id and source
DIR_BYTES = 80     # a unit's directory: ten key offsets (EXPECT lag 0/1/2, PART, ABSTRACT, RELY out; four forms in)
FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
PART_FORM = 1

R_EV, W_EV, U_EV, X_EV, B_EV = 1, 2, 4, 8, 16      # tie: evidence read, evidence written, use written, found, born
K_META, K_READ, K_WRITE = 1, 2, 4                  # key: looked up, contents read, appended to
READ_T = R_EV | X_EV
WRITE_T = W_EV | U_EV | B_EV

REC = None          # the active recorder (read by the list and dict subclasses)


# ------------------------------------------------------------------------------------------------------- recording
class RecList(list):
    """An adjacency list that records when its contents are read."""

    __slots__ = ("kid",)

    def __iter__(self):
        r = REC
        if r is not None and r.on:
            r.kev.append((r.seq, self.kid, K_READ, list.__len__(self)))
            r.seq += 1
        return list.__iter__(self)

    def __getitem__(self, i):
        r = REC
        if r is not None and r.on:
            r.kev.append((r.seq, self.kid, K_READ, list.__len__(self)))
            r.seq += 1
        return list.__getitem__(self, i)


class RecDict(dict):
    """``S.out`` / ``S.inc``: a lookup records its key; the list records when its contents are read."""

    __slots__ = ()

    def _meta(self, key) -> None:
        r = REC
        if r is not None and r.on:
            r.kev.append((r.seq, r.key_id(key), K_META, 0))
            r.seq += 1

    def get(self, key, default=None):
        self._meta(key)
        return dict.get(self, key, default)

    def __getitem__(self, key):
        self._meta(key)
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._meta(key)
        return dict.__contains__(self, key)

    def setdefault(self, key, default=None):           # only Substance._index_tie: a new tie is appended
        lst = dict.get(self, key)
        if lst is None:
            lst = RecList()
            lst.kid = REC.key_id(key)
            dict.__setitem__(self, key, lst)
        r = REC
        if r.on:
            r.kev.append((r.seq, lst.kid, K_WRITE, list.__len__(lst)))
            r.seq += 1
        return lst


def first_occurrence(ids, seqs, flags, lens=None):
    """Distinct ids in the order of their first occurrence: (ids, first seq, OR of flags, max of lens)."""
    if ids.size == 0:
        e = np.empty(0, np.int64)
        return e, e, np.empty(0, np.int8), (e if lens is not None else None)
    order = np.lexsort((seqs, ids))
    ids_s = ids[order]
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
    uid, useq = ids_s[starts], seqs[order][starts]
    uflag = np.bitwise_or.reduceat(flags[order], starts)
    ulen = np.maximum.reduceat(lens[order], starts) if lens is not None else None
    o = np.argsort(useq, kind="stable")
    return uid[o], useq[o], uflag[o], (ulen[o] if ulen is not None else None)


class Recorder:
    """Per-moment access sets (``record``) and, in both modes, the running hash of every moment's outputs."""

    def __init__(self, record: bool = True) -> None:
        self.record = record
        self.on = False
        self.seq = 0
        self.tev: list = []          # (seq, flag, tie id or id tensor)
        self.kev: list = []          # (seq, key id, flag, list length)
        self.kid: dict = {}          # key tuple -> id (out keys are 3-tuples, inc keys 2-tuples)
        self.keys: list = []
        self.calls = collections.Counter()
        self.scans = 0
        self.unit_scans, self.unit_scan_ms = 0, 0.0
        self.hash = hashlib.sha256()
        self.parts = collections.defaultdict(list)
        self.meta: list = []         # (phase, dialogue, utterance, ties, units, ms, acts) per moment
        self.phase = self.dialogue = 0
        self.utterance = 0
        self.fresh = True            # the current utterance has no moment yet
        self.speaker = None          # who speaks in the current utterance: "other", "inner" or "self"
        self.text = None             # the text channel

    def key_id(self, key) -> int:
        k = self.kid.get(key)
        if k is None:
            k = self.kid[key] = len(self.keys)
            self.keys.append(key)
        return k

    def new_utterance(self, speaker) -> None:
        if not self.fresh:
            self.utterance += 1
            self.fresh = True
        self.speaker = speaker

    def begin_dialogue(self, phase: int, dialogue: int) -> None:
        self.phase, self.dialogue = phase, dialogue
        self.new_utterance(None)

    def close(self, mind, acts, events, ms: float) -> None:
        """One moment has been lived. Utterances follow who speaks: someone's utterance starts with the session's
        listen, the mind's inner speech with its first imagined word, its own speech with its first word (said as a
        decision of this moment, or said aloud from a thought); silent moments stay in the current utterance."""
        from zihin.events import ACTED, IMAGINED
        self.hash.update(repr(([(e.channel, e.symbol, e.prov) for e in acts], mind.focus,
                               sorted(mind.predicted.items()))).encode())
        provs = {e.prov for e in events if e.channel == self.text}
        if ACTED in provs and self.speaker != "self":
            self.new_utterance("self")
        elif IMAGINED in provs and self.speaker not in ("inner", "self"):
            self.new_utterance("inner")
        self.meta.append((self.phase, self.dialogue, self.utterance, mind.S.n_ties, mind.S.n_units, ms, len(acts)))
        if self.record:
            self._close_access(mind)
        self.fresh = False
        if self.speaker in (None, "other") and any(e.channel == self.text for e in acts):
            self.new_utterance("self")                  # the turn has passed: the mind begins to speak

    def _close_access(self, mind) -> None:
        tev, kev = self.tev, self.kev
        self.tev, self.kev = [], []
        P = self.parts
        arrs, fls, sqs, one_i, one_f, one_s = [], [], [], [], [], []
        for seq, flag, p in tev:
            if type(p) is int:
                one_i.append(p)
                one_f.append(flag)
                one_s.append(seq)
            else:
                a = np.asarray(p.reshape(-1).numpy(), dtype=np.int64)
                if a.size:
                    arrs.append(a)
                    fls.append(np.full(a.size, flag, np.int8))
                    sqs.append(np.full(a.size, seq, np.int64))
        if one_i:
            arrs.append(np.array(one_i, np.int64))
            fls.append(np.array(one_f, np.int8))
            sqs.append(np.array(one_s, np.int64))
        if arrs:
            uid, useq, uflag, _ = first_occurrence(np.concatenate(arrs), np.concatenate(sqs), np.concatenate(fls))
        else:
            uid, useq, uflag, _ = first_occurrence(np.empty(0, np.int64), None, None)
        P["tie"].append(uid.astype(np.int32))
        P["tie_seq"].append(useq.astype(np.int64))
        P["tie_flag"].append(uflag.astype(np.int8))
        if kev:
            a = np.array(kev, dtype=np.int64)
            kid, kseq, kflag, klen = first_occurrence(a[:, 1], a[:, 0], a[:, 2].astype(np.int8), a[:, 3])
        else:
            kid, kseq, kflag, klen = first_occurrence(np.empty(0, np.int64), None, None, np.empty(0, np.int64))
        P["key"].append(kid.astype(np.int32))
        P["key_seq"].append(kseq.astype(np.int64))
        P["key_flag"].append(kflag.astype(np.int8))
        P["key_len"].append(klen.astype(np.int32))
        P["focus"].append(np.array([u for u, _, _ in mind.focus], np.int64))
        pred = mind.predicted
        P["pred"].append(np.fromiter(pred.keys(), np.int64, len(pred)))
        P["pred_e"].append(np.fromiter(pred.values(), np.float64, len(pred)).astype(np.float32))
        F, ep, index = mind.knob["F"], list(mind.episode), mind.S.tuple_index     # a plain dict: reading only
        ctx = []
        for u, e in pred.items():
            if e > 0.5:
                parts = tuple((ep + [u])[-F:])
                c = index.get(parts) if len(parts) > 1 else u
                if c is not None:
                    ctx.append(c)
        P["pctx"].append(np.array(ctx, np.int64))


def install(mind, rec: Recorder, session) -> None:
    """Wrap the mind's step and the session's listen (both modes: the output hash and the utterances) and, when
    recording, the store's accessors."""
    global REC
    from zihin.__main__ import CHANNEL
    REC = rec
    rec.text = CHANNEL
    listen0 = session.listen

    def listen(text, prov, speaker=None):
        rec.new_utterance("other")
        return listen0(text, prov, speaker)

    session.listen = listen
    S = mind.S
    if rec.record:
        for name in ("out", "inc"):
            new = RecDict()
            for key, ties in getattr(S, name).items():
                lst = RecList(ties)
                lst.kid = rec.key_id(key)
                dict.__setitem__(new, key, lst)
            setattr(S, name, new)
        o = {name: getattr(S, name) for name in ("w", "w1", "cell", "column", "_put", "add1", "use_ties", "tie",
                                                  "find_tie", "out_ids")}
        calls = rec.calls

        def w(table, ids, tables=1):
            if rec.on:
                rec.tev.append((rec.seq, R_EV, ids))
                rec.seq += 1
                calls["w"] += 1
            return o["w"](table, ids, tables)

        def w1(table, tie):
            if rec.on:
                rec.tev.append((rec.seq, R_EV, int(tie)))
                rec.seq += 1
                calls["w1"] += 1
            return o["w1"](table, tie)

        def cell(table, tie, slot):
            if rec.on:
                rec.tev.append((rec.seq, R_EV, int(tie)))
                rec.seq += 1
                calls["cell"] += 1
            return o["cell"](table, tie, slot)

        def column(table, slot):
            if rec.on:
                rec.scans += 1
                calls["column"] += 1
            return o["column"](table, slot)

        def _put(table, slot, ids, amounts):
            if rec.on:
                rec.tev.append((rec.seq, W_EV, ids))
                rec.seq += 1
                calls["_put"] += 1
            return o["_put"](table, slot, ids, amounts)

        def add1(table, slot, tie, amount):
            got = o["add1"](table, slot, tie, amount)
            if rec.on:
                calls["add1"] += 1
                if got != 0.0:
                    rec.tev.append((rec.seq, W_EV, int(tie)))
                    rec.seq += 1
            return got

        def use_ties(ids, amount=1.0):
            if rec.on and ids.numel():
                rec.tev.append((rec.seq, U_EV, ids))
                rec.seq += 1
                calls["use_ties"] += 1
            return o["use_ties"](ids, amount)

        def tie(src, dst, form, lag):
            n = S.n_ties
            t = o["tie"](src, dst, form, lag)
            if rec.on:
                rec.kev.append((rec.seq, rec.key_id((src, form, -1 if form == PART_FORM else lag)), K_META, 0))
                rec.tev.append((rec.seq + 1, B_EV if t >= n else X_EV, int(t)))
                rec.seq += 2
                calls["tie"] += 1
            return t

        def find_tie(src, dst, form, lag):
            t = o["find_tie"](src, dst, form, lag)
            if rec.on:
                rec.kev.append((rec.seq, rec.key_id((src, form, -1 if form == PART_FORM else lag)), K_META, 0))
                rec.seq += 1
                if t >= 0:
                    rec.tev.append((rec.seq, X_EV, int(t)))
                    rec.seq += 1
                calls["find_tie"] += 1
            return t

        def out_ids(key):
            if rec.on:
                lst = dict.get(S.out, key)
                rec.kev.append((rec.seq, rec.key_id(key), K_READ, len(lst) if lst is not None else 0))
                rec.seq += 1
                calls["out_ids"] += 1
            return o["out_ids"](key)

        for name, fn in (("w", w), ("w1", w1), ("cell", cell), ("column", column), ("_put", _put), ("add1", add1),
                         ("use_ties", use_ties), ("tie", tie), ("find_tie", find_tie), ("out_ids", out_ids)):
            setattr(S, name, fn)
        import zihin.dusunce as dusunce                # the one whole unit-table scan on the reply path
        frequent0 = dusunce.frequent_words

        def frequent_words(mind_, k=200):
            t = time.perf_counter()
            try:
                return frequent0(mind_, k)
            finally:
                rec.unit_scans += 1
                rec.unit_scan_ms += 1000.0 * (time.perf_counter() - t)

        dusunce.frequent_words = frequent_words
    step0 = mind.step

    def step(events=()):
        events = list(events)
        t = time.perf_counter()
        acts = step0(events)
        rec.close(mind, acts, events, 1000.0 * (time.perf_counter() - t))
        return acts

    mind.step = step


# ------------------------------------------------------------------------------------------------------------ run
def evidence_totals(S, n: int, chunk: int = 1 << 18):
    """Raw W+ and W- (summed over classes) of ties 0..n-1, through the original reader."""
    pos, neg = np.zeros(n), np.zeros(n)
    for a in range(0, n, chunk):
        ids = torch.arange(a, min(n, a + chunk))
        p, q = type(S).w2(S, ids)
        pos[a:a + ids.numel()] = p.sum(-1).numpy()
        neg[a:a + ids.numel()] = q.sum(-1).numpy()
    return pos, neg


def run(args) -> None:
    torch.set_num_threads(1)
    from zihin import persist
    from zihin.__main__ import Session, dialogues_of, pieces_for
    os.makedirs(args.work, exist_ok=True)
    rec = Recorder(record=not args.plain)
    t = time.perf_counter()
    mind = persist.load(args.mind, ledger=False)
    load_s = time.perf_counter() - t
    session = Session(mind, pieces_for(mind))
    S = mind.S
    n0, t0, u0 = S.n_ties, mind.t, S.n_units
    start = {}
    if rec.record:
        wp, wn = evidence_totals(S, n0)
        start = {"w_pos0": wp, "w_neg0": wn, "u_use0": S.u_use[:u0].numpy().copy(),
                 "t_last0": S.t_last[:n0].numpy().copy(), "trust0": mind.trust.numpy().copy()}
    install(mind, rec, session)
    path = os.path.join(args.data, next(n for n in sorted(os.listdir(args.data)) if n.endswith(DIALOGUES)))
    with open(path, encoding="utf-8") as f:
        pairs = dialogues_of(json.load(f))[:WATCH]
    rec.on = rec.record
    times, words_said = {}, []
    t = time.perf_counter()
    for d, (q, a) in enumerate(pairs):
        rec.begin_dialogue(0, d)
        session.watch(q, a)
    times["watch_s"] = time.perf_counter() - t
    for p, (topic, questions) in enumerate(TOPICS, start=1):
        t = time.perf_counter()
        for j, q in enumerate(questions):
            rec.begin_dialogue(p, WATCH + 10 * (p - 1) + j)
            said = session.reply(q)
            words_said.append(0 if said == "(sessiz)" else len(said.split()))
            rec.hash.update(("|" + said).encode())                      # hashed only, never kept
        times[topic + "_s"] = time.perf_counter() - t
    rec.on = False
    if rec.tev or rec.kev:
        rec.close(mind, [], [], 0.0)
    digest = type(S).digest(S)
    meta = np.array(rec.meta, dtype=np.float64)
    summary = {"name": args.name, "plain": bool(args.plain), "load_s": round(load_s, 2),
               "times_s": {k: round(v, 2) for k, v in times.items()}, "moments": len(rec.meta),
               "moments_before": t0, "ties_before": n0, "units_before": u0, "ties_after": S.n_ties,
               "units_after": S.n_units, "hash": rec.hash.hexdigest(), "digest": digest, "words_said": words_said,
               "ms_per_moment_median": round(float(np.median(meta[:, 5])), 3),
               "ms_per_moment_mean": round(float(meta[:, 5].mean()), 3),
               "ms_by_phase_mean": {str(p): round(float(meta[meta[:, 0] == p, 5].mean()), 3) for p in (0, 1, 2)},
               "pieces": session.pieces is not None, "calls": dict(rec.calls), "scans": rec.scans,
               "unit_scans": rec.unit_scans, "unit_scan_ms": round(rec.unit_scan_ms, 1)}
    with open(os.path.join(args.work, args.name + (".plain" if args.plain else ".run") + ".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("words_said",)}), flush=True)
    if not rec.record:
        return
    n = S.n_ties
    wp1, wn1 = evidence_totals(S, n)
    P = rec.parts
    out = {"meta": meta}
    for name, arrs in P.items():
        out[name] = np.concatenate(arrs) if arrs else np.empty(0)
        out[name + "_ptr"] = np.r_[0, np.cumsum([a.size for a in arrs])].astype(np.int64)
    kinds, units, forms, lags, lens = [], [], [], [], []
    for key in rec.keys:
        if len(key) == 3:
            lst = dict.get(S.out, key)
            kinds.append(0)
            lags.append(key[2])
        else:
            lst = dict.get(S.inc, key)
            kinds.append(1)
            lags.append(-9)
        units.append(key[0])
        forms.append(key[1])
        lens.append(len(lst) if lst is not None else 0)
    out.update({"kt_kind": np.array(kinds, np.int8), "kt_unit": np.array(units, np.int64),
                "kt_form": np.array(forms, np.int8), "kt_lag": np.array(lags, np.int16),
                "kt_len": np.array(lens, np.int64),
                "t_src": S.t_src[:n].numpy().copy(), "t_dst": S.t_dst[:n].numpy().copy(),
                "t_form": S.t_form[:n].numpy().copy(), "t_lag": S.t_lag[:n].numpy().copy(),
                "t_birth": S.t_birth[:n].numpy().copy(), "t_last": S.t_last[:n].numpy().copy(),
                "t_alive": S.t_alive[:n].numpy().copy(), "w_pos1": wp1, "w_neg1": wn1,
                "u_kind": S.u_kind[:S.n_units].numpy().copy(), "u_birth": S.u_birth[:S.n_units].numpy().copy(),
                "scalars": np.array([n0, t0, u0, S.n_ties, S.n_units], np.int64)})
    out.update(start)
    np.savez(os.path.join(args.work, args.name + ".trace.npz"), **out)
    print("trace saved: %d moments, %d tie entries, %d key entries, %d keys" % (
        len(rec.meta), out["tie"].size, out["key"].size, len(rec.keys)), flush=True)


def verify(args) -> None:
    with open(os.path.join(args.work, args.name + ".run.json")) as f:
        a = json.load(f)
    with open(os.path.join(args.work, args.name + ".plain.json")) as f:
        b = json.load(f)
    same = {k: a[k] == b[k] for k in ("moments", "hash", "digest", "words_said", "ties_after", "units_after")}
    row = {"kind": "verify", "mind": args.name, **{"same_" + k: v for k, v in same.items()},
           "all_same": all(same.values()), "moments": a["moments"],
           "ms_per_moment_mean": {"plain": b["ms_per_moment_mean"], "recorded": a["ms_per_moment_mean"]},
           "load_s": b["load_s"], "times_s_plain": b["times_s"], "ties_before": a["ties_before"],
           "ties_after": a["ties_after"], "moments_before": a["moments_before"], "units_before": a["units_before"],
           "calls": a["calls"], "scans": a["scans"], "pieces": a["pieces"],
           "unit_scans": a.get("unit_scans"), "unit_scan_ms": a.get("unit_scan_ms"),
           "replies_silent": sum(1 for x in a["words_said"] if x == 0),
           "reply_words_mean": round(sum(a["words_said"]) / len(a["words_said"]), 2)}
    write_rows([row], args.name, ("verify",))


# ------------------------------------------------------------------------------------------------------- analysis
def segsum(x, ptr):
    c = np.zeros(x.size + 1, dtype=np.float64 if x.dtype.kind == "f" else np.int64)
    np.cumsum(x, out=c[1:])
    return c[ptr[1:]] - c[ptr[:-1]]


def stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": round(float(np.median(x)), 1), "p90": round(float(np.percentile(x, 90)), 1),
            "max": round(float(x.max()), 1), "mean": round(float(x.mean()), 1)}


def frac(x, total) -> dict:
    s = stats(x)
    return {k: (round(v / total, 6) if k != "n" else v) for k, v in s.items()}


def ragged(ptr, rows):
    """Indices of the entries of the given rows of a CSR-like array, concatenated."""
    rows = np.asarray(rows, dtype=np.int64)
    a, b = ptr[rows], ptr[rows + 1]
    cnt = b - a
    tot = int(cnt.sum())
    if tot == 0:
        return np.empty(0, np.int64)
    return np.repeat(a - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(tot)


def dominance(p, w):
    """B[k] = sum of w[j] over j < k with p[j] < p[k] (within one moment). Small moments by a triangular matrix;
    larger ones by divide and conquer over positions (at each level the right half of every block counts the left
    half of its block, all blocks of a level in one sorted search), O(K log^2 K) in numpy."""
    K = p.size
    if K <= 256:
        lower = np.tril(np.ones((K, K), dtype=bool), -1)
        return ((p[None, :] < p[:, None]) & lower).astype(np.float64) @ w
    pr = np.unique(p, return_inverse=True)[1].astype(np.int64)     # dense ranks, equal p equal rank
    idx = np.arange(K)
    out = np.zeros(K)
    span = 1
    while span < K:
        block = idx // (2 * span)
        right = (idx // span) % 2 == 1
        left = ~right
        lk = block[left] * (K + 1) + pr[left]
        o = np.argsort(lk, kind="stable")
        lk = lk[o]
        cw = np.r_[0.0, np.cumsum(w[left][o])]
        rb = block[right]
        hi = np.searchsorted(lk, rb * (K + 1) + pr[right], side="left")
        lo = np.searchsorted(lk, rb * (K + 1), side="left")
        out[right] += cw[hi] - cw[lo]
        span *= 2
    return out


def mattson(stream, n_items: int, weights=None):
    """LRU stack distances (Mattson et al. 1970) of a reference stream given moment by moment (each a 1-D array of
    distinct items in the order of first touch). With ``weights`` the distance is the total weight of the distinct
    other items referenced since the item's last reference (variable-size LRU). Returns (distance, weight) for every
    reference in stream order; distance -1 is a first reference."""
    last = np.full(n_items, -1, np.int64)
    mpos = np.empty(0, np.int64)
    mw = np.empty(0, np.float64)
    cw = np.empty(0, np.float64)
    s = 0
    sds, ws = [], []
    for items in stream:
        K = items.size
        if K == 0:
            continue
        w = np.ones(K) if weights is None else weights[items].astype(np.float64)
        p = last[items]
        seen = p >= 0
        sd = np.full(K, -1.0)
        if seen.any():
            idx = np.searchsorted(mpos, p[seen])
            sd[seen] = (cw[-1] - cw[idx]) + dominance(p, w)[seen]
            keep = np.ones(mpos.size, bool)
            keep[idx] = False
            mpos, mw = mpos[keep], mw[keep]
        new = s + np.arange(K)
        mpos = np.concatenate([mpos, new])
        mw = np.concatenate([mw, w])
        cw = np.cumsum(mw)
        last[items] = new
        s += K
        sds.append(sd)
        ws.append(w)
    if not sds:
        return np.empty(0), np.empty(0)
    return np.concatenate(sds), np.concatenate(ws)


def miss_curve(sd, w, caps, weighted: bool, mask=None) -> dict:
    """Miss ratio (and byte-miss ratio when weighted) of an LRU cache of each capacity: a reference hits when it was
    referenced before and its distance plus its own size fits."""
    if mask is not None:
        sd, w = sd[mask], w[mask]
    n = sd.size
    cold = sd < 0
    need = np.sort((sd + (w if weighted else 1.0))[~cold])
    out = {"refs": int(n), "compulsory": round(float(cold.mean()), 4) if n else None}
    hits = np.searchsorted(need, np.asarray(caps, dtype=np.float64), side="right")
    out["miss"] = [round(1.0 - h / n, 4) for h in hits]
    if weighted:
        order = np.argsort((sd + w)[~cold], kind="stable")
        wc = np.r_[0.0, np.cumsum(w[~cold][order])]
        total = float(w.sum())
        out["byte_miss"] = [round(1.0 - wc[h] / total, 4) for h in hits]
    return out


def write_rows(rows, mind: str, kinds) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            old = [json.loads(line) for line in f if line.strip()]
    old = [r for r in old if not (r.get("mind") == mind and r.get("kind") in kinds)]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for r in rows:
        r["when"] = stamp
    with open(OUT, "w", encoding="utf-8") as f:
        for r in old + rows:
            f.write(json.dumps(r) + "\n")


class Trace:
    """The saved trace with the derived tables the analyses share."""

    def __init__(self, path: str) -> None:
        Z = np.load(path)
        self.Z = {k: Z[k] for k in Z.files}
        Z = self.Z
        self.meta = Z["meta"]
        self.M = self.meta.shape[0]
        self.phase = self.meta[:, 0].astype(np.int64)
        self.dlg = self.meta[:, 1].astype(np.int64)
        self.utt = self.meta[:, 2].astype(np.int64)
        self.n0, self.t0, self.u0, self.n1, self.u1 = (int(x) for x in Z["scalars"])
        self.tie, self.tie_ptr, self.tie_flag, self.tie_seq = (Z["tie"].astype(np.int64), Z["tie_ptr"],
                                                               Z["tie_flag"], Z["tie_seq"])
        self.key, self.key_ptr, self.key_flag = Z["key"].astype(np.int64), Z["key_ptr"], Z["key_flag"]
        self.key_len, self.key_seq = Z["key_len"].astype(np.int64), Z["key_seq"]
        self.kt_kind, self.kt_unit, self.kt_form = Z["kt_kind"], Z["kt_unit"], Z["kt_form"]
        self.kt_lag, self.kt_len = Z["kt_lag"], Z["kt_len"]
        self.NK = self.kt_kind.size
        self.alive = Z["t_alive"].astype(bool)
        n = self.n1
        src, dst, form = Z["t_src"], Z["t_dst"], Z["t_form"].astype(np.int64)
        lagk = np.where(form == PART_FORM, -1, Z["t_lag"].astype(np.int64))
        # each tie's out key and in key (ids of the recorder's key table) and its rank inside them (creation order)
        out_code = (self.kt_unit * 4 + self.kt_form) * 8 + (self.kt_lag.astype(np.int64) + 1)
        in_code = self.kt_unit * 4 + self.kt_form
        self.okid = self._match(np.where(self.kt_kind == 0, out_code, -1), (src * 4 + form) * 8 + (lagk + 1))
        self.ikid = self._match(np.where(self.kt_kind == 1, in_code, -1), dst * 4 + form)
        self.orank = self._rank(self.okid)
        self.irank = self._rank(self.ikid)
        self.src, self.dst, self.form, self.lagk = src, dst, form, lagk
        # entries in the moment arrays: their moment
        self.tie_m = np.repeat(np.arange(self.M), np.diff(self.tie_ptr))
        self.key_m = np.repeat(np.arange(self.M), np.diff(self.key_ptr))
        self.ties_total = int(self.alive.sum())
        self.keys_total = int((self.kt_len > 0).sum())
        self.entries_total = int(self.kt_len.sum())

    def _match(self, key_codes, tie_codes):
        valid = key_codes >= 0
        kids = np.flatnonzero(valid)
        order = np.argsort(key_codes[kids])
        sorted_codes = key_codes[kids][order]
        idx = np.clip(np.searchsorted(sorted_codes, tie_codes), 0, max(0, sorted_codes.size - 1))
        found = sorted_codes[idx] == tie_codes
        out = np.where(found, kids[order][idx], -1)
        assert bool(found[self.alive].all()), "a live tie without its key"
        return out

    def _rank(self, kid):
        n = kid.size
        ids = np.arange(n)
        order = np.lexsort((ids, kid))
        k = kid[order]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        first = np.repeat(starts, np.diff(np.r_[starts, n]))
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n) - first
        return rank

    def windows(self):
        """Window labels per moment for each time scale, and the phase group of each window."""
        ten = np.where(self.dlg < WATCH, self.dlg // 10, WATCH // 10 + (self.dlg - WATCH) // 10)
        return {"moment": np.arange(self.M), "utterance": self.utt, "dialogue": self.dlg, "10_dialogues": ten,
                "run": np.zeros(self.M, np.int64)}


def window_sets(T: Trace, labels, mask_t=None, mask_k=None):
    """Per window: distinct ties touched, distinct keys touched (any flag, contents), out-list entries read (the
    union of read prefixes) and ties touched in the union of both (evidence or read entry)."""
    L = int(labels.max()) + 1
    lab_t = labels[T.tie_m]
    t_ok = np.ones(T.tie.size, bool) if mask_t is None else mask_t
    code = np.unique(lab_t[t_ok] * T.n1 + T.tie[t_ok])
    ties = np.bincount(code // T.n1, minlength=L)
    lab_k = labels[T.key_m]
    k_ok = np.ones(T.key.size, bool) if mask_k is None else mask_k
    keys = np.bincount(np.unique(lab_k[k_ok] * T.NK + T.key[k_ok]) // T.NK, minlength=L)
    cont = k_ok & ((T.key_flag & K_READ) != 0)
    keys_read = np.bincount(np.unique(lab_k[cont] * T.NK + T.key[cont]) // T.NK, minlength=L)
    out_read = cont & (T.kt_kind[T.key] == 0)
    kc = lab_k[out_read] * T.NK + T.key[out_read]
    o = np.argsort(kc, kind="stable")
    kc, kl = kc[o], T.key_len[out_read][o]
    if kc.size:
        st = np.flatnonzero(np.r_[True, kc[1:] != kc[:-1]])
        uc, ml = kc[st], np.maximum.reduceat(kl, st)
    else:
        uc, ml = np.empty(0, np.int64), np.empty(0, np.int64)
    entries = np.bincount(uc // T.NK, weights=ml, minlength=L).astype(np.int64)
    tl = code // T.n1
    tt = code % T.n1
    q = tl * T.NK + T.okid[tt]
    idx = np.clip(np.searchsorted(uc, q), 0, max(0, uc.size - 1))
    covered = (uc.size > 0) & (uc[idx] == q) & (ml[idx] > T.orank[tt]) if uc.size else np.zeros(q.size, bool)
    union = entries + np.bincount(tl[~covered], minlength=L)
    return {"ties": ties, "keys": keys, "keys_read": keys_read, "out_entries_read": entries, "ties_union": union}


def analyze_touch(T: Trace) -> list:
    rows = []
    f = T.tie_flag
    per = {"ties": np.diff(T.tie_ptr), "ties_read": segsum(((f & READ_T) != 0).astype(np.int64), T.tie_ptr),
           "ties_written": segsum(((f & WRITE_T) != 0).astype(np.int64), T.tie_ptr),
           "ties_evidence_written": segsum(((f & (W_EV | B_EV)) != 0).astype(np.int64), T.tie_ptr),
           "ties_born": segsum(((f & B_EV) != 0).astype(np.int64), T.tie_ptr),
           "keys": np.diff(T.key_ptr),
           "keys_read": segsum(((T.key_flag & K_READ) != 0).astype(np.int64), T.key_ptr),
           "keys_only_looked_up": segsum((T.key_flag == K_META).astype(np.int64), T.key_ptr),
           "keys_appended": segsum(((T.key_flag & K_WRITE) != 0).astype(np.int64), T.key_ptr)}
    kind = T.kt_kind[T.key]
    rd = (T.key_flag & K_READ) != 0
    per["out_entries_read"] = segsum(np.where(rd & (kind == 0), T.key_len, 0), T.key_ptr)
    per["in_entries_read"] = segsum(np.where(rd & (kind == 1), T.key_len, 0), T.key_ptr)
    groups = {"all": np.ones(T.M, bool), "watch": T.phase == 0, "replies": T.phase > 0}
    row = {"kind": "touched_per_moment", "ties_total": T.ties_total, "keys_total": T.keys_total,
           "moments": T.M, "moments_by_phase": [int((T.phase == p).sum()) for p in (0, 1, 2)]}
    heavy = per["keys"] > 1000
    entries = per["out_entries_read"] + per["in_entries_read"]
    row["heavy_moments"] = {"over_1000_keys": int(heavy.sum()), "in_replies": int((heavy & (T.phase > 0)).sum()),
                            "share_of_key_refs": round(float(per["keys"][heavy].sum()) / max(1, per["keys"].sum()), 4),
                            "share_of_entries_read": round(float(entries[heavy].sum()) / max(1, entries.sum()), 4),
                            "share_of_tie_refs": round(float(per["ties"][heavy].sum()) / max(1, per["ties"].sum()), 4)}
    row["written_per_read_median"] = {g: round(float(np.median(per["ties_written"][m] / np.maximum(1, per["ties_read"][m]))),
                                               4) for g, m in groups.items()}
    for g, m in groups.items():
        row[g] = {k: stats(v[m]) for k, v in per.items()}
    rows.append(row)
    W = T.windows()
    sets = {name: window_sets(T, lab) for name, lab in W.items()}
    ws = {"kind": "working_set", "ties_total": T.ties_total, "keys_total": T.keys_total,
          "entries_total_out": int(T.kt_len[T.kt_kind == 0].sum()), "scales": {}}
    for name, lab in W.items():
        L = int(lab.max()) + 1
        ph = np.zeros(L, np.int64)
        ph[lab] = T.phase
        res = {}
        for g, sel in (("all", np.ones(L, bool)), ("watch", ph == 0), ("hava", ph == 1), ("yemek", ph == 2)):
            if name == "run" and g != "all":
                continue
            if not sel.any():
                continue
            d = {}
            for k, v in sets[name].items():
                d[k] = stats(v[sel])
                tot = T.keys_total if k.startswith("keys") else T.ties_total
                d[k + "_frac"] = frac(v[sel], tot)
            res[g] = d
        ws["scales"][name] = res
    rows.append(ws)
    # topic regions: overlap of the working sets of the two topics and of the watched dialogues
    regions = {}
    for g, m in (("watch", T.phase == 0), ("hava", T.phase == 1), ("yemek", T.phase == 2)):
        mt, mk = m[T.tie_m], m[T.key_m]
        regions[g] = (np.unique(T.tie[mt]), np.unique(T.key[mk]),
                      np.unique(T.key[mk & ((T.key_flag & K_READ) != 0)]))
    ov = {"kind": "topic_overlap"}
    for a, b in (("hava", "yemek"), ("hava", "watch"), ("yemek", "watch")):
        d = {}
        for i, name in enumerate(("ties", "keys", "keys_read")):
            x, y = regions[a][i], regions[b][i]
            inter = np.intersect1d(x, y, assume_unique=True).size
            d[name] = {"a": int(x.size), "b": int(y.size), "both": int(inter),
                       "jaccard": round(inter / max(1, np.union1d(x, y).size), 4),
                       "share_of_a": round(inter / max(1, x.size), 4)}
        ov[a + "_vs_" + b] = d
    rows.append(ov)
    return rows


def analyze_hubs(T: Trace) -> dict:
    row = {"kind": "hubs", "tie_record_bytes": TIE_BYTES, "inc_entry_bytes": INC_BYTES}
    kind = T.kt_kind[T.key]
    rd = (T.key_flag & K_READ) != 0
    ev_out_kid = T.okid[T.tie]
    ev_in_kid = T.ikid[T.tie]
    for side, kk in (("out", 0), ("in", 1)):
        kids = np.flatnonzero(T.kt_kind == kk)
        order = kids[np.argsort(-T.kt_len[kids], kind="stable")]
        lens = T.kt_len[order]
        row[side + "_keys"] = int((lens > 0).sum())
        row[side + "_len_max"] = int(lens[0]) if lens.size else 0
        for top in (100, 1000):
            hub = np.zeros(T.NK, bool)
            hub[order[:top]] = True
            mine = kind == kk
            refs = mine.sum()
            d = {"entries": int(lens[:top].sum()), "share_of_ties": round(float(lens[:top].sum()) / T.ties_total, 4),
                 "mb": round(float(lens[:top].sum()) * (TIE_BYTES if kk == 0 else INC_BYTES) / 2 ** 20, 2),
                 "key_refs_share": round(float((hub[T.key] & mine).sum()) / max(1, refs), 4),
                 "read_refs_share": round(float((hub[T.key] & mine & rd).sum()) / max(1, (mine & rd).sum()), 4),
                 "entries_read_share": round(float(T.key_len[hub[T.key] & mine & rd].sum()) /
                                             max(1, T.key_len[mine & rd].sum()), 4),
                 "smallest_hub_len": int(lens[min(top, lens.size) - 1]) if lens.size else 0}
            hk = ev_out_kid if kk == 0 else ev_in_kid
            d["tie_refs_share"] = round(float(hub[hk[hk >= 0]].sum()) / max(1, T.tie.size), 4)
            row["%s_top%d" % (side, top)] = d
    return row


def unit_keys(T: Trace):
    ex = np.flatnonzero((T.kt_len > 0) & (T.kt_unit >= 0) & (T.kt_unit < T.u1))
    order = ex[np.argsort(T.kt_unit[ex], kind="stable")]
    ptr = np.searchsorted(T.kt_unit[order], np.arange(T.u1 + 1))
    return order, ptr


def analyze_prefetch(T: Trace) -> dict:
    """The mind's own state at moment t as a prefetch signal for the keys of moment t+1 (and t+2): the keys of its
    focus units, of the units it predicts (all, or expected: e > 1/2), and of the next contexts it expects (the
    composed unit of its episode continued by each expected unit, when that unit exists), against recency (the keys
    of moment t). Coverage of all keys touched, of the lists read, weighted by their entries, and of the keys that
    were not touched at t; the size of each prefetch set; its precision (share of it used at t+1)."""
    Z = T.Z
    ukid, uptr = unit_keys(T)
    fptr, fu = Z["focus_ptr"], Z["focus"].astype(np.int64)
    pptr, pu, pe = Z["pred_ptr"], Z["pred"].astype(np.int64), Z["pred_e"]
    cptr, cu = Z["pctx_ptr"], Z["pctx"].astype(np.int64)

    def keys_of(units):
        units = units[(units >= 0) & (units < T.u1)]
        return np.unique(ukid[ragged(uptr, units)]) if units.size else np.empty(0, np.int64)

    sets = []
    for t in range(T.M):
        foc = fu[fptr[t]:fptr[t + 1]]
        pr = pu[pptr[t]:pptr[t + 1]]
        e = pe[pptr[t]:pptr[t + 1]]
        sets.append({"F": keys_of(foc), "Pa": keys_of(pr), "Pe": keys_of(pr[e > 0.5]),
                     "C": keys_of(cu[cptr[t]:cptr[t + 1]])})
    K = [T.key[T.key_ptr[t]:T.key_ptr[t + 1]] for t in range(T.M)]
    KR = [T.key[T.key_ptr[t]:T.key_ptr[t + 1]][(T.key_flag[T.key_ptr[t]:T.key_ptr[t + 1]] & K_READ) != 0]
          for t in range(T.M)]
    acc = collections.defaultdict(float)
    per_moment = collections.defaultdict(list)
    sizes = collections.defaultdict(list)
    names = ("focus", "pred_all", "pred_expected", "pred_contexts", "focus_pred_expected",
             "focus_pred_expected_contexts", "recent", "recent_focus_pred_expected_contexts")
    for t in range(T.M - 1):
        if T.dlg[t + 1] != T.dlg[t]:
            continue
        s = sets[t]
        F, Pa, Pe, C = s["F"], s["Pa"], s["Pe"], s["C"]
        mine = np.union1d(np.union1d(F, Pe), C)
        signals = {"focus": F, "pred_all": Pa, "pred_expected": Pe, "pred_contexts": C,
                   "focus_pred_expected": np.union1d(F, Pe), "focus_pred_expected_contexts": mine,
                   "recent": K[t], "recent_focus_pred_expected_contexts": np.union1d(K[t], mine)}
        nxt, nxt_r = K[t + 1], KR[t + 1]
        new = np.setdiff1d(nxt, K[t], assume_unique=True)
        new_r = np.setdiff1d(nxt_r, K[t], assume_unique=True)
        w_r = T.kt_len[nxt_r]
        acc["n"] += nxt.size
        acc["n_read"] += nxt_r.size
        acc["w_read"] += float(w_r.sum())
        acc["n_new"] += new.size
        acc["n_new_read"] += new_r.size
        for name, P in signals.items():
            hit = np.isin(nxt, P, assume_unique=True)
            hit_r = np.isin(nxt_r, P, assume_unique=True)
            acc[name + ":hit"] += hit.sum()
            acc[name + ":hit_read"] += hit_r.sum()
            acc[name + ":w_hit"] += float(w_r[hit_r].sum())
            acc[name + ":new_hit"] += np.isin(new, P, assume_unique=True).sum()
            acc[name + ":new_read_hit"] += np.isin(new_r, P, assume_unique=True).sum()
            if nxt.size:
                per_moment[name].append(hit.mean())
            sizes[name + ":keys"].append(P.size)
            sizes[name + ":entries"].append(int(T.kt_len[P].sum()))
            if P.size:
                per_moment[name + ":precision"].append(np.isin(P, nxt, assume_unique=True).mean())
        if t + 2 < T.M and T.dlg[t + 2] == T.dlg[t]:
            P2 = np.union1d(mine, np.union1d(np.union1d(sets[t + 1]["F"], sets[t + 1]["Pe"]), sets[t + 1]["C"]))
            n2 = K[t + 2]
            acc["h2:hit"] += np.isin(n2, mine, assume_unique=True).sum()
            acc["h2_union:hit"] += np.isin(n2, P2, assume_unique=True).sum()
            acc["h2_recent:hit"] += np.isin(n2, K[t], assume_unique=True).sum()
            acc["h2:n"] += n2.size
    row = {"kind": "prefetch", "transitions": len(per_moment["focus"]), "signals": {},
           "keys_next_median": stats([k.size for k in K])["median"]}
    for name in names:
        d = {"coverage": round(acc[name + ":hit"] / max(1, acc["n"]), 4),
             "coverage_median": round(float(np.median(per_moment[name])), 4) if per_moment[name] else None,
             "read_coverage": round(acc[name + ":hit_read"] / max(1, acc["n_read"]), 4),
             "entry_coverage": round(acc[name + ":w_hit"] / max(1, acc["w_read"]), 4),
             "new_key_coverage": round(acc[name + ":new_hit"] / max(1, acc["n_new"]), 4),
             "new_read_coverage": round(acc[name + ":new_read_hit"] / max(1, acc["n_new_read"]), 4),
             "size_keys": stats(sizes[name + ":keys"]), "size_entries": stats(sizes[name + ":entries"])}
        if per_moment[name + ":precision"]:
            d["precision_median"] = round(float(np.median(per_moment[name + ":precision"])), 4)
        row["signals"][name] = d
    row["new_keys_share"] = round(acc["n_new"] / max(1, acc["n"]), 4)
    row["new_read_keys_share"] = round(acc["n_new_read"] / max(1, acc["n_read"]), 4)
    row["two_steps"] = {"focus_pred_expected_contexts_at_t": round(acc["h2:hit"] / max(1, acc["h2:n"]), 4),
                        "same_at_t_and_t1": round(acc["h2_union:hit"] / max(1, acc["h2:n"]), 4),
                        "recent_at_t": round(acc["h2_recent:hit"] / max(1, acc["h2:n"]), 4)}
    return row


def fit_power(x, y, n, least: int = 200):
    ok = (n >= least) & (y > 0) & (x > 0)
    if ok.sum() < 3:
        return None
    lx, ly = np.log(x[ok]), np.log(y[ok])
    A = np.vstack([np.ones(lx.size), lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float(((ly - pred) ** 2).sum()) / ss if ss > 0 else None
    return {"d": round(-float(coef[1]), 3), "log_a": round(float(coef[0]), 3),
            "r2": round(r2, 3) if r2 is not None else None, "bins": int(ok.sum()),
            "range": [int(x[ok].min()), int(x[ok].max())]}


def analyze_reuse(T: Trace) -> list:
    Z = T.Z
    n0 = T.n0
    base = np.flatnonzero(T.alive[:n0])
    W = Z["w_pos0"] + Z["w_neg0"]
    touched = np.zeros(T.n1, bool)
    touched[T.tie] = True
    read = np.zeros(T.n1, bool)
    read[T.tie[(T.tie_flag & READ_T) != 0]] = True
    # read as an entry of an out list: the read prefix of its key covers its rank
    maxlen = np.zeros(T.NK, np.int64)
    rd = ((T.key_flag & K_READ) != 0) & (T.kt_kind[T.key] == 0)
    np.maximum.at(maxlen, T.key[rd], T.key_len[rd])
    entry = (T.okid >= 0) & (maxlen[np.maximum(T.okid, 0)] > T.orank)
    edges_w = [0.0, 1e-12, 1.0 + 1e-9, 3.0 + 1e-9, 10.0 + 1e-9, np.inf]
    names_w = ["0", "(0,1]", "(1,3]", "(3,10]", ">10"]
    bw = np.digitize(W[base], edges_w[1:-1])
    by_w = {}
    for i, nm in enumerate(names_w):
        sel = base[bw == i]
        by_w[nm] = {"ties": int(sel.size), "share_of_ties": round(sel.size / max(1, base.size), 5),
                    "share_of_touched": round(float(touched[sel].sum()) / max(1, touched[base].sum()), 5),
                    "share_of_entries_read": round(float(entry[sel].sum()) / max(1, entry[base].sum()), 5),
                    "p_touched": round(float(touched[sel].mean()), 5) if sel.size else None,
                    "p_read": round(float(read[sel].mean()), 5) if sel.size else None,
                    "p_entry_or_touched": round(float((touched[sel] | entry[sel]).mean()), 5) if sel.size else None}
    age = T.t0 - Z["t_birth"][base].astype(np.int64)
    last = Z["t_last0"][base].astype(np.int64)           # the last moment its evidence changed, before the run
    rec_ok = last > 0
    recency = T.t0 - last
    edges = np.r_[0, 2 ** np.arange(0, 20)]
    out = []
    curves = {}
    for nm, val, sel in (("age", age, np.ones(base.size, bool)), ("recency", recency, rec_ok)):
        b = np.digitize(val[sel], edges) - 1
        ids = base[sel]
        n = np.bincount(b, minlength=edges.size)
        k = np.bincount(b, weights=touched[ids], minlength=edges.size)
        kr = np.bincount(b, weights=read[ids], minlength=edges.size)
        lo = edges
        hi = np.r_[edges[1:], edges[-1] * 2]
        mid = np.sqrt(np.maximum(lo, 0.5) * hi)
        p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
        pr = np.where(n > 0, kr / np.maximum(n, 1), 0.0)
        bins = [{"lo": int(lo[i]), "hi": int(hi[i]), "ties": int(n[i]), "p_touched": round(float(p[i]), 5),
                 "p_read": round(float(pr[i]), 5)} for i in range(edges.size) if n[i] > 0]
        curves[nm] = {"bins": bins, "fit_touched": fit_power(mid, p, n), "fit_read": fit_power(mid, pr, n),
                      "fit_touched_from_16": fit_power(np.where(lo >= 16, mid, 0), p, n)}
    out.append({"kind": "reuse", "ties_at_start": int(base.size), "moments_before": T.t0, "moments_run": T.M,
                "touched_in_run": int(touched[base].sum()), "read_in_run": int(read[base].sum()),
                "entry_read_in_run": int(entry[base].sum()),
                "by_evidence": by_w, "by_age": curves["age"], "by_recency": curves["recency"],
                "ties_without_evidence_change_record": int((~rec_ok).sum())})
    # joint table: evidence x age
    joint = {}
    ab = np.digitize(age, [1000, 10000, 50000])
    for i, nm in enumerate(names_w):
        for j, an in enumerate(("<1k", "1k-10k", "10k-50k", ">=50k")):
            sel = base[(bw == i) & (ab == j)]
            if sel.size:
                joint["%s|%s" % (nm, an)] = {"ties": int(sel.size), "p_touched": round(float(touched[sel].mean()), 5)}
    out[0]["by_evidence_and_age"] = joint
    # distribution of W over all live ties at the start, by form
    dist = {"kind": "evidence_distribution", "ties": int(base.size), "bins": {}, "by_form": {}}
    for i, nm in enumerate(names_w):
        dist["bins"][nm] = int((bw == i).sum())
    dist["exactly_one"] = int((W[base] == 1.0).sum())
    dist["one_shot_share"] = round(float((W[base] <= 1.0 + 1e-9).mean()), 4)
    forms = {"EXPECT_lag0": (0, 0), "EXPECT_lag1": (0, 1), "EXPECT_lag2": (0, 2), "PART": (1, -1),
             "ABSTRACT": (2, 0), "RELY": (3, 0)}
    fb, lb = T.form[base], T.lagk[base]
    for nm, (fo, lg) in forms.items():
        sel = (fb == fo) & (lb == lg)
        if sel.any():
            dist["by_form"][nm] = {"ties": int(sel.sum()), "one_shot_share": round(float((W[base][sel] <= 1.0 + 1e-9)
                                                                                            .mean()), 4),
                                   "median_W": round(float(np.median(W[base][sel])), 3),
                                   "p_touched": round(float(touched[base][sel].mean()), 5)}
    out.append(dist)
    return out


def logistic(X, y, iters: int = 50):
    """Newton-Raphson maximum likelihood of P(y = 1) = 1 / (1 + exp(-X b)); X has an intercept column."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -35.0, 35.0)))
        h = (X * (p * (1.0 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(h, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -35.0, 35.0)))
    ll = float((y * np.log(np.maximum(p, 1e-300)) + (1 - y) * np.log(np.maximum(1 - p, 1e-300))).sum())
    return b, ll


def auc(score, y) -> float:
    """Area under the ROC curve (Mann-Whitney, average ranks for equal scores)."""
    order = np.argsort(score, kind="stable")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], s.size]
    avg = (starts + ends + 1) / 2.0                        # 1-based average rank of each group of equal scores
    ranks = np.empty(s.size)
    ranks[order] = np.repeat(avg, ends - starts)
    pos = y.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def capture(score, y, refs, qs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5), seed: int = 0) -> dict:
    """Share of the touched ties, and of their touches (moments), held by the top q of ties by score (equal scores
    in random order)."""
    jitter = np.random.default_rng(seed).random(score.size)
    order = np.lexsort((jitter, -score))
    yy, rr = y[order], refs[order]
    cy, cr = np.cumsum(yy), np.cumsum(rr)
    out = {}
    for q in qs:
        k = max(1, int(round(q * score.size)))
        out[str(q)] = {"ties": round(float(cy[k - 1]) / max(1, cy[-1]), 4),
                       "touches": round(float(cr[k - 1]) / max(1, cr[-1]), 4)}
    return out


def analyze_need(T: Trace) -> dict:
    """Which ties a stretch of life needs, predicted from what the mind knows of each tie before it: its evidence W
    (how much it was lived), its age and the time since its evidence last changed (the rational analysis of memory,
    Anderson and Schooler 1991; ACT-R's base level ln n - d ln t). Logistic fits, AUC, and the share of the touched
    ties a resident set of the top q ties by each score would hold. Targets: touched in the whole run (features at
    its start), and touched in the replies (fresh questions; recency taken at the start of the replies)."""
    Z = T.Z
    base = np.flatnonzero(T.alive[:T.n0])
    W = (Z["w_pos0"] + Z["w_neg0"])[base]
    age = (T.t0 - Z["t_birth"][base]).astype(np.float64)
    last0 = Z["t_last0"][base].astype(np.int64)
    rec = np.where(last0 > 0, T.t0 - last0, age).astype(np.float64)
    # the last moment of the watching in which each tie's evidence changed (t_last as the store would hold it)
    watch_m = np.flatnonzero(T.phase == 0)
    reply0 = T.t0 + int(watch_m.max()) + 1 if watch_m.size else T.t0
    lastw = np.zeros(T.n1, np.int64)
    sel = (T.phase[T.tie_m] == 0) & ((T.tie_flag & (W_EV | B_EV)) != 0)
    np.maximum.at(lastw, T.tie[sel], T.t0 + T.tie_m[sel] + 1)
    last_r = np.maximum(last0, lastw[base])
    rec_r = np.where(last_r > 0, reply0 - last_r, age + (reply0 - T.t0)).astype(np.float64)
    refs_all = np.bincount(T.tie, minlength=T.n1)[base].astype(np.float64)
    rmask = T.phase[T.tie_m] > 0
    refs_rep = np.bincount(T.tie[rmask], minlength=T.n1)[base].astype(np.float64)
    lw, la = np.log(np.maximum(W, 1e-3)), np.log(age + 1.0)
    out = {"kind": "need_model", "ties": int(base.size), "targets": {}}
    for target, y, refs, lr in (("run", (refs_all > 0).astype(np.float64), refs_all, np.log(rec + 1.0)),
                                ("replies", (refs_rep > 0).astype(np.float64), refs_rep, np.log(rec_r + 1.0))):
        one = np.ones(base.size)
        res = {"base_rate": round(float(y.mean()), 5)}
        b0, ll0 = logistic(one[:, None], y)
        for name, cols in (("W", [lw]), ("age", [la]), ("recency", [lr]), ("W_age_recency", [lw, la, lr])):
            X = np.column_stack([one] + cols)
            b, ll = logistic(X, y)
            score = X @ b
            res[name] = {"coef": [round(float(v), 4) for v in b], "auc": round(auc(score, y), 4),
                         "pseudo_r2": round(1.0 - ll / ll0, 4), "capture": capture(score, y, refs)}
        res["features"] = ["intercept", "ln W", "ln(age+1)", "ln(recency+1)"]
        out["targets"][target] = res
    out["reply_start_moment_offset"] = int(reply0 - T.t0)
    edges = np.r_[0, 2 ** np.arange(0, 20)]
    y = refs_rep > 0
    b = np.digitize(rec_r, edges) - 1
    n = np.bincount(b, minlength=edges.size)
    k = np.bincount(b, weights=y, minlength=edges.size)
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    hi = np.r_[edges[1:], edges[-1] * 2]
    mid = np.sqrt(np.maximum(edges, 0.5) * hi)
    out["replies_by_recency"] = {"bins": [{"lo": int(edges[i]), "ties": int(n[i]), "p_touched": round(float(p[i]), 5)}
                                          for i in range(edges.size) if n[i] > 0],
                                 "fit": fit_power(mid, p, n), "fit_from_16": fit_power(np.where(edges >= 16, mid, 0),
                                                                                         p, n)}
    return out


# ----------------------------------------------------------------------------------------------- layouts and pages
def graph(T: Trace):
    """Undirected unit graph of the live ties (no self ties), pair weight = raw evidence at the end of the run,
    normalised by the strengths of both ends (w / sqrt(s_u s_v)) so that a hub's many ties are each weak."""
    Z = T.Z
    a = T.src[T.alive]
    b = T.dst[T.alive]
    w = (Z["w_pos1"] + Z["w_neg1"])[T.alive]
    m = (a != b) & (a >= 0) & (b >= 0)
    a, b, w = a[m], b[m], np.maximum(w[m], 1e-3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n = T.u1
    code, inv = np.unique(lo * n + hi, return_inverse=True)
    pw = np.bincount(inv, weights=w)
    lo, hi = code // n, code % n
    s = np.bincount(lo, weights=pw, minlength=n) + np.bincount(hi, weights=pw, minlength=n)
    norm = pw / np.sqrt(s[lo] * s[hi])
    return lo, hi, norm


def lpa_order(T: Trace, lo, hi, norm, iters: int = 40, seed: int = 0, hubs=None):
    """Label propagation (Raghavan et al. 2007) on the normalised graph, half the units updated per round; regions
    are ordered by their oldest unit, units inside a region by id. ``hubs``: units left out of the propagation (each
    its own region) and placed first."""
    n = T.u1
    if hubs is not None:
        keep = ~(hubs[lo] | hubs[hi])
        lo, hi, norm = lo[keep], hi[keep], norm[keep]
    U, V, Wd = np.r_[lo, hi], np.r_[hi, lo], np.r_[norm, norm]
    labels = np.arange(n)
    rng = np.random.default_rng(seed)
    rounds = 0
    for _ in range(iters):
        rounds += 1
        code = U * n + labels[V]
        o = np.argsort(code, kind="stable")
        c, ws = code[o], Wd[o]
        st = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
        sc = np.add.reduceat(ws, st)
        cu, cl = c[st] // n, c[st] % n
        o2 = np.lexsort((cl, -sc, cu))
        cu, cl = cu[o2], cl[o2]
        first = np.flatnonzero(np.r_[True, cu[1:] != cu[:-1]])
        bu, bl = cu[first], cl[first]
        upd = rng.random(bu.size) < 0.5
        new = labels.copy()
        new[bu[upd]] = bl[upd]
        changed = int((new != labels).sum())
        labels = new
        if changed <= n // 10000:
            break
    oldest = np.full(n, n, np.int64)
    np.minimum.at(oldest, labels, np.arange(n))
    first = np.zeros(n, np.int64) if hubs is None else (~hubs).astype(np.int64)
    order = np.lexsort((np.arange(n), oldest[labels], first))
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    sizes = np.bincount(labels, minlength=n)
    sizes = sizes[sizes > 0]
    internal = float(norm[labels[lo] == labels[hi]].sum() / norm.sum()) if norm.size else 0.0
    if hubs is not None:
        sizes = np.bincount(labels[~hubs], minlength=n)
        sizes = sizes[sizes > 0]
    info = {"rounds": rounds, "hub_units": int(hubs.sum()) if hubs is not None else 0, "regions": int(sizes.size), "largest_share": round(float(sizes.max()) / n, 4),
            "regions_ge_10": int((sizes >= 10).sum()), "internal_weight_share": round(internal, 4),
            "units_in_regions_ge_10": round(float(sizes[sizes >= 10].sum()) / n, 4)}
    return rank, info


def bfs_order(T: Trace, lo, hi, norm, use):
    """Breadth-first order from the most used units, neighbours by descending normalised weight."""
    n = T.u1
    U, V, Wd = np.r_[lo, hi], np.r_[hi, lo], np.r_[norm, norm]
    o = np.lexsort((-Wd, U))
    V = V[o]
    ptr = np.searchsorted(U[o], np.arange(n + 1))
    visited = np.zeros(n, bool)
    order = []
    for s in np.argsort(-use, kind="stable").tolist():
        if visited[s]:
            continue
        visited[s] = True
        queue = collections.deque([s])
        while queue:
            u = queue.popleft()
            order.append(u)
            nb = V[ptr[u]:ptr[u + 1]]
            nb = nb[~visited[nb]]
            if nb.size:
                visited[nb] = True
                queue.extend(nb.tolist())
    rank = np.empty(n, np.int64)
    rank[np.array(order, np.int64)] = np.arange(n)
    return rank


def first_use_rank(T: Trace, moments):
    """Units by the first time in the given moments that one of their keys or tie records was touched; the rest
    after them by id."""
    n = T.u1
    first = np.full(n, np.iinfo(np.int64).max, np.int64)
    mt = moments[T.tie_m]
    mk = moments[T.key_m]
    u_t = T.src[T.tie[mt]]
    np.minimum.at(first, u_t, T.tie_seq[mt])
    u_k = T.kt_unit[T.key[mk]]
    ok = (u_k >= 0) & (u_k < n)
    np.minimum.at(first, u_k[ok], T.key_seq[mk][ok])
    order = np.lexsort((np.arange(n), first))
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    return rank


class Layout:
    """Page numbers of the hypothetical CSR layout for one order of the units."""

    def __init__(self, T: Trace, rank) -> None:
        n = T.n1
        self.upd = PAGE // DIR_BYTES
        self.tpp = PAGE // TIE_BYTES
        self.epp = PAGE // INC_BYTES
        ids = np.flatnonzero(T.alive)
        o = np.lexsort((ids, T.lagk[ids], T.form[ids], rank[T.src[ids]]))
        self.pos_r = np.full(n, -1, np.int64)
        self.pos_r[ids[o]] = np.arange(ids.size)
        o = np.lexsort((ids, T.form[ids], rank[T.dst[ids]]))
        self.pos_c = np.full(n, -1, np.int64)
        self.pos_c[ids[o]] = np.arange(ids.size)
        self.start = np.zeros(T.NK, np.int64)
        big = np.iinfo(np.int64).max
        st = np.full(T.NK, big, np.int64)
        np.minimum.at(st, T.okid[ids], self.pos_r[ids])
        np.minimum.at(st, T.ikid[ids], self.pos_c[ids])
        self.start = np.where(st == big, 0, st)
        self.n_dir = -(-T.u1 // self.upd)
        self.n_rec = -(-ids.size // self.tpp)
        self.n_inc = -(-ids.size // self.epp)
        self.pages = self.n_dir + self.n_rec + self.n_inc
        self.dir_page = rank // self.upd
        self.rec_page = np.where(self.pos_r >= 0, self.n_dir + self.pos_r // self.tpp, -1)
        self.inc_page = np.where(self.pos_c >= 0, self.n_dir + self.n_rec + self.pos_c // self.epp, -1)

    def stream(self, T: Trace):
        """Distinct pages of every moment in the order of first touch."""
        out = []
        for t in range(T.M):
            a, b = T.tie_ptr[t], T.tie_ptr[t + 1]
            ids, seq, fl = T.tie[a:b], T.tie_seq[a:b], T.tie_flag[a:b]
            parts_p, parts_s, parts_o = [self.rec_page[ids]], [seq], [np.zeros(ids.size, np.int64)]
            born = (fl & B_EV) != 0
            if born.any():
                parts_p.append(self.inc_page[ids[born]])
                parts_s.append(seq[born])
                parts_o.append(np.ones(int(born.sum()), np.int64))
            a, b = T.key_ptr[t], T.key_ptr[t + 1]
            kid, kseq, kfl, kln = T.key[a:b], T.key_seq[a:b], T.key_flag[a:b], T.key_len[a:b]
            unit = T.kt_unit[kid]
            ok = (unit >= 0) & (unit < T.u1)
            parts_p.append(self.dir_page[unit[ok]])
            parts_s.append(kseq[ok])
            parts_o.append(np.zeros(int(ok.sum()), np.int64))
            rd = ((kfl & K_READ) != 0) & (kln > 0)
            if rd.any():
                k = kid[rd]
                inc = T.kt_kind[k] == 1
                per = np.where(inc, self.epp, self.tpp)
                base = np.where(inc, self.n_dir + self.n_rec, self.n_dir)
                s0 = self.start[k]
                p0 = base + s0 // per
                p1 = base + (s0 + kln[rd] - 1) // per
                cnt = p1 - p0 + 1
                tot = int(cnt.sum())
                off = np.arange(tot) - np.repeat(np.r_[0, np.cumsum(cnt)[:-1]], cnt)
                parts_p.append(np.repeat(p0, cnt) + off)
                parts_s.append(np.repeat(kseq[rd], cnt))
                parts_o.append(off + 1)
            p = np.concatenate(parts_p)
            s = np.concatenate(parts_s)
            o = np.concatenate(parts_o)
            keep = p >= 0
            p, s, o = p[keep], s[keep], o[keep]
            order = np.lexsort((o, s))
            p = p[order]
            _, first = np.unique(p, return_index=True)
            out.append(p[np.sort(first)])
        return out


def analyze_cache(T: Trace) -> list:
    rows = []
    half = T.dlg >= WATCH // 2                          # second half: dialogues 31-60 and the replies
    # key level: the lists whose contents a moment needs (read or appended to)
    stream = []
    for t in range(T.M):
        a, b = T.key_ptr[t], T.key_ptr[t + 1]
        sel = (T.key_flag[a:b] & (K_READ | K_WRITE)) != 0
        stream.append(T.key[a:b][sel])
    size = np.maximum(T.kt_len, 1).astype(np.float64)
    ref_m = np.concatenate([np.full(s.size, t, np.int64) for t, s in enumerate(stream)])
    sd_c, w_c = mattson(stream, T.NK)
    sd_w, w_w = mattson(stream, T.NK, size)
    caps_k = [max(1.0, f * T.keys_total) for f in FRACTIONS]
    caps_e = [max(1.0, f * T.entries_total) for f in FRACTIONS]
    rows.append({"kind": "mrc_keys", "fractions": list(FRACTIONS), "keys_total": T.keys_total,
                 "entries_total": T.entries_total,
                 "by_count": miss_curve(sd_c, w_c, caps_k, False),
                 "by_entries": miss_curve(sd_w, w_w, caps_e, True),
                 "by_entries_second_half": miss_curve(sd_w, w_w, caps_e, True, half[ref_m]),
                 "stack_distance_entries_reused": stats(sd_w[sd_w >= 0])})
    # the largest lists pinned in memory, the rest in a size-aware LRU over what is left
    allrefs = np.concatenate(stream) if stream else np.empty(0, np.int64)
    ref_bytes = float(size[allrefs].sum())
    for top_out, top_in in ((100, 10), (1000, 100)):
        hub = np.zeros(T.NK, bool)
        for kk, top in ((0, top_out), (1, top_in)):
            kids = np.flatnonzero((T.kt_kind == kk) & (T.kt_len > 0))
            hub[kids[np.argsort(-T.kt_len[kids], kind="stable")[:top]]] = True
        pinned = float(T.kt_len[hub].sum())
        tail = [s_[~hub[s_]] for s_ in stream]
        sd_t, w_t = mattson(tail, T.NK, size)
        rest = T.entries_total - pinned
        caps = [max(1.0, f * rest) for f in FRACTIONS]
        m = miss_curve(sd_t, w_t, caps, True)
        n_t = sd_t.size
        rows.append({"kind": "mrc_keys_pinned", "pinned_out": top_out, "pinned_in": top_in,
                     "pinned_entries": int(pinned), "pinned_share_of_entries": round(pinned / T.entries_total, 4),
                     "pinned_mb": round((float(T.kt_len[hub & (T.kt_kind == 0)].sum()) * TIE_BYTES +
                                         float(T.kt_len[hub & (T.kt_kind == 1)].sum()) * INC_BYTES) / 2 ** 20, 2),
                     "refs_to_pinned": round(1.0 - n_t / max(1, allrefs.size), 4),
                     "ref_bytes_to_pinned": round(1.0 - float(w_t.sum()) / max(1.0, ref_bytes), 4),
                     "fractions_of_rest": list(FRACTIONS), "tail": m,
                     "miss_overall": [round(x * n_t / max(1, allrefs.size), 4) for x in m["miss"]],
                     "byte_miss_overall": [round(x * float(w_t.sum()) / max(1.0, ref_bytes), 4)
                                           for x in m["byte_miss"]]})
    # page level, for several orders of the units
    use = np.zeros(T.u1)
    use[:T.u0] = T.Z["u_use0"]
    lo, hi, norm = graph(T)
    lpa_rank, lpa_info = lpa_order(T, lo, hi, norm)
    degree = np.bincount(lo, minlength=T.u1) + np.bincount(hi, minlength=T.u1)
    hub_units = degree >= np.quantile(degree, 0.99)     # the 1% of units with the most neighbours
    lpa2_rank, lpa2_info = lpa_order(T, lo, hi, norm, hubs=hub_units)
    orders = {"creation": np.arange(T.u1), "first_use_run": first_use_rank(T, np.ones(T.M, bool)),
              "first_use_first_half": first_use_rank(T, ~half), "hot_units": np.argsort(np.lexsort(
                  (np.arange(T.u1), -use))), "regions_lpa": lpa_rank, "regions_lpa_without_hubs": lpa2_rank,
              "bfs_frequent": bfs_order(T, lo, hi, norm, use)}
    infos = {"regions_lpa": lpa_info, "regions_lpa_without_hubs": lpa2_info}
    W = T.windows()
    for name, rank in orders.items():
        L = Layout(T, rank)
        st = L.stream(T)
        ref_m = np.concatenate([np.full(s.size, t, np.int64) for t, s in enumerate(st)])
        sd, w = mattson(st, L.pages)
        caps = [max(1.0, f * L.pages) for f in FRACTIONS]
        row = {"kind": "mrc_pages", "order": name, "pages": L.pages, "mb": round(L.pages * PAGE / 2 ** 20, 1),
               "dir_pages": int(L.n_dir), "record_pages": int(L.n_rec), "inc_pages": int(L.n_inc),
               "fractions": list(FRACTIONS), "all": miss_curve(sd, w, caps, False),
               "second_half": miss_curve(sd, w, caps, False, half[ref_m])}
        allp = np.concatenate(st) if st else np.empty(0, np.int64)
        for scale in ("moment", "utterance", "dialogue", "10_dialogues", "run"):
            lab = W[scale][ref_m]
            cnt = np.bincount(np.unique(lab * L.pages + allp) // L.pages, minlength=int(W[scale].max()) + 1)
            L_ = int(W[scale].max()) + 1
            ph = np.zeros(L_, np.int64)
            ph[W[scale]] = T.phase
            row["pages_" + scale] = {"all": stats(cnt), "watch": stats(cnt[ph == 0]), "replies": stats(cnt[ph > 0])}
            row["pages_" + scale]["share_median"] = round(float(np.median(cnt)) / L.pages, 5)
        if name in infos:
            row["regions"] = infos[name]
        rows.append(row)
    return rows


def analyze(args) -> None:
    t = time.perf_counter()
    T = Trace(os.path.join(args.work, args.name + ".trace.npz"))
    rows = []
    parts = (analyze_touch, analyze_hubs, analyze_prefetch, analyze_reuse, analyze_need, analyze_cache)
    if args.only:
        parts = tuple(p for p in parts if p.__name__ in args.only.split(","))
    for part in parts:
        t1 = time.perf_counter()
        got = part(T)
        rows += got if isinstance(got, list) else [got]
        print("%s: %.1f s" % (part.__name__, time.perf_counter() - t1), flush=True)
    for r in rows:
        r["mind"] = args.name
    write_rows(rows, args.name, tuple({r["kind"] for r in rows}))
    print("analysis written (%d rows, %.1f s)" % (len(rows), time.perf_counter() - t), flush=True)


# ------------------------------------------------------------------------------------------------------- selftest
def selftest(args) -> None:
    rng = np.random.default_rng(1)
    for trial in range(40):                                  # Mattson against a plain LRU stack
        n_items = int(rng.integers(3, 40))
        weights = rng.integers(1, 9, n_items).astype(np.float64)
        stream = []
        for _ in range(int(rng.integers(1, 30))):
            k = int(rng.integers(0, min(n_items, 3000 if trial == 0 else 12) + 1))
            stream.append(rng.permutation(n_items)[:k].astype(np.int64))
        for wts in (None, weights):
            sd, _ = mattson(stream, n_items, wts)
            stack, ref = [], []
            for items in stream:
                for x in items.tolist():
                    if x in stack:
                        i = stack.index(x)
                        ref.append(float(sum(1.0 if wts is None else wts[y] for y in stack[:i])))
                        stack.pop(i)
                    else:
                        ref.append(-1.0)
                    stack.insert(0, x)
            assert np.array_equal(sd, np.array(ref)), "mattson differs from the plain stack"
    p = rng.permutation(3000).astype(np.int64)                # the large-moment path of dominance
    w = rng.integers(1, 5, 3000).astype(np.float64)
    brute = np.array([w[:k][p[:k] < p[k]].sum() for k in range(3000)])
    assert np.allclose(dominance(p, w), brute)
    sd = np.array([-1.0, 0.0, 2.0, 5.0])                     # miss curve
    m = miss_curve(sd, np.ones(4), [1, 3, 10], False)
    assert m["miss"] == [0.75, 0.5, 0.25], m
    # the recorder changes nothing: a fresh speaking mind lives the same toy life with and without it
    from zihin import Config, Mind
    from zihin.__main__ import CHANNEL, Session
    lines = [("merhaba nasılsın ?", "iyiyim teşekkür ederim ."), ("adın ne ?", "benim adım cevahir ."),
             ("ne yapıyorsun ?", "kitap okuyorum ."), ("merhaba nasılsın ?", "iyiyim sen nasılsın ?")]
    results = []
    for record in (False, True):
        mind = Mind(Config.changed(keep_log=False, support="members", bind_repeats=True, next_only_present=True,
                                   neg_floor=1.0 / 14, backoff=True, resemblance=True, race=True, situations=True,
                                   explore=True))
        mind.add_channel(CHANNEL, "act")
        session = Session(mind)
        rec = Recorder(record=record)
        install(mind, rec, session)
        rec.on = record
        for i in range(3):
            for d, (q, a) in enumerate(lines):
                rec.begin_dialogue(0, d)
                session.watch(q, a)
        for j, q in enumerate(("merhaba nasılsın ?", "adın ne ?", "kuantum nedir ?")):
            rec.begin_dialogue(1, 10 + j)
            rec.hash.update(session.reply(q).encode())
        rec.on = False
        results.append((rec.hash.hexdigest(), type(mind.S).digest(mind.S), len(rec.meta)))
        if record:
            assert rec.parts["tie"] and sum(a.size for a in rec.parts["tie"]) > 0
    assert results[0] == results[1], results
    print("selftest ok: mattson, dominance, miss curve, recorder bit-identical (%d moments)" % results[0][2])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--mind", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--plain", action="store_true", help="no recorder: the reference run for verify")
    for name in ("verify", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("--name", required=True)
        s.add_argument("--work", required=True)
        if name == "analyze":
            s.add_argument("--only", default="", help="comma-separated analysis functions (default: all)")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    {"run": run, "verify": verify, "analyze": analyze, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
