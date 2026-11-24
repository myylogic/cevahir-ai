# -*- coding: utf-8 -*-
"""The cortex: a store of symbol contexts that grows with what it lives (MODEL.md, section 3).

Every context the store has met is an origin: the last d symbols (d = 0 .. depth), with counts of the symbols that
followed it. The root (d = 0) has met everything. Nothing has a fixed size. The alphabet is the set of symbols met, each
numbered in the order it was first met. A context becomes a node once it has been met ``grow`` times; until then, its
meetings are held as a pending mark under its parent.

Predicting the next symbol walks from the deepest context met down to the root. At each origin, the rule gives the
belief that what follows is new to it, the *novelty*. The rest, ``1 - novelty``, is shared by the counts, and the
novelty mass passes to the next shorter context.

- With ``mixing="exclude"``, the shorter context shares that mass only among symbols the longer contexts did not have:
  "new to this context" is meant literally.
- With ``mixing="blend"``, it shares the mass among all its symbols.
- At the root, the novelty mass goes to symbols never met, spread evenly over the code space (all Unicode code points
  by default).

Whose record judges novelty (``escape``):

- ``"situation"`` (the default): the record of every origin that was in the same situation, the rule unchanged applied
  to that record. A situation is the context's length, whether longer contexts were excluded, and how much and how
  varied the context has lived (bucketed). This is learned trust per class of origin, as in the evidence core. A
  context's own past novelty rate overstates its future one, because novelty falls as an origin lives; the record of
  its situation learns how fast it falls.
- ``"rule"``: the context's own record, ``(t + 1/2) / (n + 1)`` (living/evidence.py).
- ``"ppm-a"``, ``"ppm-c"``, ``"ppm-d"``: estimators of the compression literature, for comparison.

Learning adds one count to every context on the path (``update="all"``). PPM's update exclusion, where only the
contexts from the longest down to the one that knew the symbol learn it (``update="found"``), is kept for comparison;
it makes the counts depend on the order of life. With ``grow=1`` and ``update="all"``, the contexts and counts (the
substance) are a
function of the multiset of what was lived. Reading a text and then ``retract``-ing it therefore leaves a substance
identical to a store that never read it. The situation records are trust learned from the store's own predictions, so
they depend on the order of life. A ``ledger`` given to ``stream`` keeps a text's situation events, and ``retract``
takes back exactly those events. With ``grow>=2``, a context waits for its second (m-th) meeting, which keeps one-off
contexts out. The order of life then shapes the substance too, and retraction is not offered.
"""
from __future__ import annotations

import hashlib
import math

from .evidence import ESCAPES

UNICODE = 0x110000
_NB = [0, 1, 2, 3, 4, 5, 5, 5] + [6] * 8 + [7] * 16 + [8] * 32       # how much an origin lived: 1,2,3,4,5-7,8-15,...
_TB = [0, 1, 2, 3, 4, 4, 4, 4]                                      # how varied: 1,2,3,4-7, 8+


def situation(d: int, first: int, t: int, n: int) -> int:
    """The situation of an origin: its length, whether longer contexts were excluded, how varied, how much lived."""
    return (((d << 1) | first) * 6 + (_TB[t] if t < 8 else 5)) * 10 + (_NB[n] if n < 64 else 9)


class Cortex:
    """One channel's context store."""

    def __init__(self, depth: int = 8, grow: int = 1, escape: str = "situation", mixing: str = "exclude",
                 code_space: int = UNICODE, update: str = "all"):
        if mixing not in ("exclude", "blend"):
            raise ValueError("mixing is 'exclude' or 'blend'")
        if update not in ("all", "found"):
            raise ValueError("update is 'all' or 'found'")
        if escape != "situation" and escape not in ESCAPES:
            raise ValueError("unknown escape %r" % escape)
        self.depth = int(depth)
        self.grow = int(grow)
        self.escape = escape
        self.mixing = mixing
        self.code_space = int(code_space)
        self.update = update                   # "all": every context on the path learns (exact); "found": only the
        self._found = 0                        # contexts from the longest down to the one that knew the symbol
        self._esc = ESCAPES.get(escape)
        self.nodes: dict[str, list] = {}       # context -> [n, {symbol: count}]
        self.pending: dict[str, list] = {}     # context met fewer than ``grow`` times -> the symbols that followed it
        self.alphabet: dict[str, int] = {}     # symbol -> its number, in the order the store met it
        self.situations: dict[int, list] = {}  # situation -> [times what followed was new, times it was asked]
        self.entries = 0                       # (context, symbol) pairs held: the store's ties
        self.lived = 0                         # symbols learned

    # ----------------------------------------------------------------------------------------------------------------
    def novelty(self, d: int, first: int, t: int, n: int) -> float:
        """Belief that what follows an origin in this situation is new to it."""
        if self._esc is not None:
            return self._esc(t, n)
        rec = self.situations.get(situation(d, first, t, n))
        if rec is None:
            return 0.5                                 # the rule with nothing lived
        return (rec[0] + 0.5) / (rec[1] + 1.0)

    def _chain(self, h: str) -> list:
        """The nodes on the path of history ``h``: the root first, then ever longer contexts, while they exist."""
        nodes = self.nodes
        node = nodes.get("")
        chain = []
        L = len(h)
        top = min(self.depth, L)
        d = 0
        while node is not None:
            chain.append(node)
            d += 1
            if d > top:
                break
            node = nodes.get(h[L - d:])
        return chain

    def _p(self, chain: list, x: str, events: list | None = None) -> float:
        """Probability of symbol ``x`` given the nodes of its path; ``events`` receives (situation, was new)."""
        nov = self.novelty
        record = events is not None and self._esc is None
        if self.mixing == "blend":
            p = 0.0 if x in self.alphabet else 1.0 / (self.code_space - len(self.alphabet))
            for d, (n, counts) in enumerate(chain):
                t = len(counts)
                e = nov(d, 0, t, n)
                cx = counts.get(x, 0)
                if record:
                    events.append((situation(d, 0, t, n), 0 if cx else 1))
                p = (1.0 - e) * cx / n + e * p
            return p
        mass = 1.0
        excluded = None
        for d in range(len(chain) - 1, -1, -1):
            n, counts = chain[d]
            if excluded is None:
                t = len(counts)
                e = nov(d, 1, t, n)
                cx = counts.get(x)
                if record:
                    events.append((situation(d, 1, t, n), 0 if cx else 1))
                if cx is not None:
                    self._found = d
                    return mass * (1.0 - e) * cx / n
                mass *= e
                excluded = set(counts)
                continue
            nn = 0
            tt = 0
            for y, c in counts.items():
                if y not in excluded:
                    nn += c
                    tt += 1
            if tt == 0:
                continue
            e = nov(d, 0, tt, nn)
            cx = counts.get(x)                         # x was in no longer context, so it is not excluded
            if record:
                events.append((situation(d, 0, tt, nn), 0 if cx else 1))
            if cx is not None:
                self._found = d
                return mass * (1.0 - e) * cx / nn
            mass *= e
            excluded.update(counts)
        self._found = 0
        return mass / (self.code_space - len(self.alphabet))

    def _learn(self, chain: list, h: str, x: str) -> None:
        """Add one count of ``x`` to every context on the path; grow the path by the growth rule."""
        if x not in self.alphabet:
            self.alphabet[x] = len(self.alphabet)
        entries = 0
        for node in (chain if self.update == "all" or self.mixing == "blend" else chain[self._found:]):
            node[0] += 1
            counts = node[1]
            v = counts.get(x)
            if v is None:
                counts[x] = 1
                entries += 1
            else:
                counts[x] = v + 1
        L = len(h)
        top = min(self.depth, L)
        d = len(chain)
        if d == 0:                                     # the very first symbol: the root is born
            self.nodes[""] = [1, {x: 1}]
            entries += 1
            d = 1
        if self.grow == 1:
            nodes = self.nodes
            while d <= top:
                nodes[h[L - d:]] = [1, {x: 1}]
                entries += 1
                d += 1
        elif d <= top:
            ctx = h[L - d:]
            seen = self.pending.get(ctx)
            if seen is None:
                self.pending[ctx] = [x]
            elif len(seen) + 1 >= self.grow:
                counts: dict[str, int] = {}
                for y in seen:
                    counts[y] = counts.get(y, 0) + 1
                counts[x] = counts.get(x, 0) + 1
                self.nodes[ctx] = [len(seen) + 1, counts]
                entries += len(counts)
                del self.pending[ctx]
                if d + 1 <= top:
                    self.pending[h[L - d - 1:]] = [x]
            else:
                seen.append(x)
        self.entries += entries
        self.lived += 1

    # ----------------------------------------------------------------------------------------------------------------
    def stream(self, text: str, history: str = "", learn: bool = True, bits: list | None = None,
               ledger: dict | None = None) -> tuple[float, str]:
        """Predict each symbol of ``text`` from what came before, then (if ``learn``) learn it.

        Returns the code length in bits and the history to continue with. ``bits``, if given, receives the bits of
        each symbol; ``ledger``, if given, receives this text's situation events (for ``retract``)."""
        D = self.depth
        h = history[-D:] if D else ""
        total = 0.0
        log2 = math.log2
        chain_of = self._chain
        p_of = self._p
        learn_one = self._learn
        situations = self.situations
        events = [] if (learn and self._esc is None) else None
        for x in text:
            chain = chain_of(h)
            if events is not None:
                events.clear()
                p = p_of(chain, x, events)
                for key, new in events:
                    rec = situations.get(key)
                    if rec is None:
                        situations[key] = [new, 1]
                    else:
                        rec[0] += new
                        rec[1] += 1
                    if ledger is not None:
                        rec = ledger.get(key)
                        if rec is None:
                            ledger[key] = [new, 1]
                        else:
                            rec[0] += new
                            rec[1] += 1
            else:
                p = p_of(chain, x)
            b = -log2(p)
            total += b
            if bits is not None:
                bits.append(b)
            if learn:
                learn_one(chain, h, x)
            if D:
                h = (h + x)[-D:]
        return total, h

    def predict(self, history: str, x: str) -> float:
        """Probability of ``x`` after ``history``, without learning."""
        return self._p(self._chain(history[-self.depth:] if self.depth else ""), x)

    def root_p(self, x: str) -> float:
        """Probability of ``x`` by the root alone, judged by its own record: what a store without context would say.

        The root's novelty is the rule on its own record, ``(t + 1/2) / (n + 1)``. The situation records of a store
        that uses context hold the root's novelty only for the rare times it was the longest context met."""
        root = self.nodes.get("")
        if root is None:
            return 1.0 / self.code_space
        n, counts = root
        e = (len(counts) + 0.5) / (n + 1.0)
        c = counts.get(x)
        return (1.0 - e) * c / n if c else e / (self.code_space - len(self.alphabet))

    def distribution(self, history: str) -> tuple[dict, float]:
        """Every symbol met with its probability after ``history``, and the probability of a symbol never met."""
        chain = self._chain(history[-self.depth:] if self.depth else "")
        nov = self.novelty
        dist: dict[str, float] = {}
        if self.mixing == "blend":
            unmet = 1.0
            for d, (n, counts) in enumerate(chain):
                e = nov(d, 0, len(counts), n)
                for y in dist:
                    dist[y] *= e
                for y, c in counts.items():
                    dist[y] = dist.get(y, 0.0) + (1.0 - e) * c / n
                unmet *= e
            return dist, unmet
        mass = 1.0
        excluded: set = set()
        first = 1
        for d in range(len(chain) - 1, -1, -1):
            n, counts = chain[d]
            items = [(y, c) for y, c in counts.items() if y not in excluded]
            if not items:
                continue
            nn = sum(c for _, c in items)
            e = nov(d, first, len(items), nn)
            first = 0
            for y, c in items:
                dist[y] = mass * (1.0 - e) * c / nn
            mass *= e
            excluded.update(counts)
        return dist, mass

    def best(self, history: str) -> tuple[str | None, float]:
        """The most probable next symbol after ``history`` and its probability (exact)."""
        if self.mixing == "blend":
            dist, _ = self.distribution(history)
            if not dist:
                return None, 0.0
            y = max(dist, key=dist.get)
            return y, dist[y]
        chain = self._chain(history[-self.depth:] if self.depth else "")
        nov = self.novelty
        mass = 1.0
        excluded: set = set()
        best_y, best_p = None, 0.0
        first = 1
        for d in range(len(chain) - 1, -1, -1):
            n, counts = chain[d]
            items = [(y, c) for y, c in counts.items() if y not in excluded]
            if not items:
                continue
            nn = sum(c for _, c in items)
            e = nov(d, first, len(items), nn)
            first = 0
            y, c = max(items, key=lambda yc: yc[1])
            p = mass * (1.0 - e) * c / nn
            if p > best_p:
                best_y, best_p = y, p
            mass *= e
            excluded.update(counts)
            if mass <= best_p:
                break
        return best_y, best_p

    # ----------------------------------------------------------------------------------------------------------------
    def retract(self, text: str, history: str = "", ledger: dict | None = None) -> None:
        """Take back exactly what reading ``text`` after ``history`` added (only for ``grow=1``).

        The substance returns to what a store that never read ``text`` holds. If the text's ``ledger`` is given, its
        situation events are taken back too."""
        if self.grow != 1 or self.update != "all":
            raise ValueError("exact retraction needs grow=1 and update='all': otherwise the order of life matters")
        D = self.depth
        h = history[-D:] if D else ""
        nodes = self.nodes
        for x in text:
            L = len(h)
            for d in range(min(D, L) + 1):
                ctx = h[L - d:] if d else ""
                node = nodes[ctx]
                node[0] -= 1
                counts = node[1]
                v = counts[x] - 1
                if v:
                    counts[x] = v
                else:
                    del counts[x]
                    self.entries -= 1
                if node[0] == 0:
                    del nodes[ctx]
            self.lived -= 1
            if D:
                h = (h + x)[-D:]
        root = nodes.get("")
        for y in list(self.alphabet):
            if root is None or y not in root[1]:
                del self.alphabet[y]
        for key, (new, total) in (ledger or {}).items():
            rec = self.situations[key]
            rec[0] -= new
            rec[1] -= total
            if rec[1] == 0:
                del self.situations[key]

    def digest(self, part: str = "substance") -> str:
        """SHA-256 of the store's substance (contexts, counts, pending marks), or with ``part="all"`` also of its
        situation records. The numbering of symbols is not part of either."""
        out = hashlib.sha256()

        def put(s: str) -> None:
            b = s.encode("utf-8", "surrogatepass")
            out.update(len(b).to_bytes(4, "little"))
            out.update(b)

        for ctx in sorted(self.nodes):
            n, counts = self.nodes[ctx]
            put(ctx)
            out.update(n.to_bytes(8, "little"))
            for y in sorted(counts):
                put(y)
                out.update(counts[y].to_bytes(8, "little"))
        out.update(b"pending")
        for ctx in sorted(self.pending):
            put(ctx)
            put("".join(sorted(self.pending[ctx])))
        if part == "all":
            out.update(b"situations")
            for key in sorted(self.situations):
                new, total = self.situations[key]
                out.update(key.to_bytes(8, "little") + new.to_bytes(8, "little") + total.to_bytes(8, "little"))
        return out.hexdigest()

    def size(self) -> dict:
        """How much the store holds."""
        return {"nodes": len(self.nodes), "pending": len(self.pending), "entries": self.entries,
                "alphabet": len(self.alphabet), "situations": len(self.situations), "lived": self.lived}
