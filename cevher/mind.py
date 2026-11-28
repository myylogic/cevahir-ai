# -*- coding: utf-8 -*-
"""The word grain of the one substance: what living writes and what reading does
(docs/architecture/FINAL_FORM.md, sections 4 to 6).

Everything here reads and writes one table of ties through four calls (``cevher.store``). Nothing is kept per
experience: a sentence told adds counts to what its contexts were followed by; a dialogue adds counts to what the
question's words brought, to the answer as lived, and to the frames that explain the answer from the question; and
every way of reading keeps the record of its own claims in the same table.

No word, language, mark or format is named in this file.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from cevher.units import LINE, code_of, fold, mix, rule

# roles and origins: units like any other, with codes of their own
R_SEQ, R_SKIP, R_REL, R_RELR = code_of("\0role seq"), code_of("\0role skip"), code_of("\0role rel"), code_of("\0role relr")
R_FRAME, R_WAY, R_WAYT = code_of("\0role frame"), code_of("\0role way"), code_of("\0role way total")
R_KNOWN, R_BIRTH = code_of("\0role way known"), code_of("\0role way born of")
R_CONST, R_CONSTF, R_TONE, R_FEAT = (code_of("\0role const"), code_of("\0role const by feature"),
                                     code_of("\0role tone"), code_of("\0role feature"))
O_TOLD, O_ASKED, O_ANSWER = code_of("\0origin told"), code_of("\0origin asked"), code_of("\0origin answer")
BOS, END, HOLE = code_of("\0begin"), code_of("\0end"), code_of("\0place")
R_SEG, F0, F1, F2 = code_of("\0role run"), code_of("\0frame 0"), code_of("\0frame 1"), code_of("\0frame 2")
NGRAM = 3                                                    # the longest context of words (a bound of a search)
PAIRS_UP_TO = 24                                             # questions up to this long are read with two places open
SKELETON = "\0sk "


class Mind:
    """The word grain. ``store`` is the substance; ``spell`` gives a code its spelling back so that it can be said."""

    def __init__(self, store, ledger: bool = False):
        self.S = store
        self.spell: Dict[int, str] = {}
        self._codes: Dict[str, int] = {}
        self._parsed: Dict[int, list] = {}
        self.clock = 0
        self.entry: Optional[list] = None                    # the writes of the experience being lived (a ledger)
        self.keep = ledger
        self.frames_tried = 0
        self.widen = True                                    # a constant the store can explain is opened

    # ------------------------------------------------------------------------------------------------ units -----
    def code(self, token: str) -> int:
        c = self._codes.get(token)
        if c is None:
            c = self._codes[token] = code_of(token)
            self.spell.setdefault(c, token)
        return c

    def codes(self, toks) -> tuple:
        return tuple(self.code(t) for t in toks)

    def words(self, cs) -> list:
        return [self.spell.get(c, "?") for c in cs]

    # ----------------------------------------------------------------------------------------------- writing ----
    def _add(self, key: int, value: int, w: int = 1) -> None:
        if self.entry is not None:
            self.entry.append((key, value, w))
        self.S.add(key, value, w, self.clock)

    def _bump(self, key: int, w: int = 1) -> None:
        if self.entry is not None:
            self.entry.append((key, None, w))
        self.S.bump(key, w)

    def _tie(self, key: int, value: int, w: int = 1) -> None:
        self._add(key, value, w)
        self._bump(key, w)

    def take_back(self, entry: list) -> None:
        """Forget: the same writes with the opposite sign, the last first."""
        for key, value, w in reversed(entry):
            if value is None:
                self.S.bump(key, -w)
            else:
                self.S.add(key, value, -w, 0)

    def sequence(self, cs: tuple, origin: int) -> None:
        """An utterance as a sequence: what each of its contexts was followed by, and what stood two places on."""
        padded = (BOS,) + tuple(cs) + (END,)
        for i in range(1, len(padded)):
            for n in range(1, min(NGRAM, i) + 1):
                self._tie(fold((origin, n) + padded[i - n:i], R_SEQ), padded[i])
            if i >= 2:
                self._tie(mix(mix(R_SKIP, origin), padded[i - 2]), padded[i])

    def tell(self, toks: List[str]) -> None:
        """A sentence told (its body, the closing mark apart)."""
        self.clock += 1
        self.sequence(self.codes(toks), O_TOLD)

    # ----------------------------------------------------------------------------------------- reading a tie ----
    def next(self, ctx: tuple, newest: bool = False, origin: int = O_TOLD) -> Optional[Tuple[int, float]]:
        """What followed a context in what was told: the longest context held speaks. Read in one of two ways:
        what followed it most (believed by the rule on its count), or, with ``newest``, what followed it last
        (believed as one confirmation since, (1 + 1/2) / (1 + 1), or by its count where that is more)."""
        for n in range(min(NGRAM, len(ctx)), 0, -1):
            key = fold((origin, n) + tuple(ctx[-n:]), R_SEQ)
            vs = self.S.vals(key)
            if vs:
                total = self.S.tot(key)
                if newest:
                    last = max(vs, key=lambda x: (x[2], x[1]))
                    return last[0], max(0.75, rule(last[1], total))
                return vs[0][0], rule(vs[0][1], total)
        return None

    def brings(self, x: int) -> Optional[Tuple[int, float]]:
        """What a word of a question brings to the answer: the token that comes with it and not without it. Over
        the dialogues lived, each of the four readings of "the two go together" is believed by the rule: the
        token where the word is, no token where the word is not, the word where the token is, no word where the
        token is not. The word itself before any other token."""
        key = mix(R_REL, x)
        n_x, n_all = self.S.tot(key), self.S.tot(R_REL)
        best = None
        for v, c, _ in self.S.vals(key):
            n_t = self.S.tot(mix(R_RELR, v))
            e = min(rule(c, n_x), rule(c, n_t),
                    rule((n_all - n_x) - (n_t - c), n_all - n_x), rule((n_all - n_t) - (n_x - c), n_all - n_t))
            if e < LINE:
                continue
            rank = (v == x, e, c, -v)
            if best is None or rank > best[0]:
                best = (rank, v)
        return (best[1], best[0][1]) if best else None

    def features(self, x: int) -> list:
        """What a filler is like: what it was followed by, and what stood two places on, in what was told (each
        where believed)."""
        out = []
        for kind, key in ((1, fold((O_TOLD, 1, x), R_SEQ)), (2, mix(mix(R_SKIP, O_TOLD), x))):
            vs = self.S.vals(key)
            if vs and vs[0][0] != END and rule(vs[0][1], self.S.tot(key)) >= LINE:
                out.append(fold((kind, vs[0][0]), R_FEAT))
        return out

    # ------------------------------------------------------------------------------------------------ frames ----
    @staticmethod
    def _runs(qc: tuple) -> list:
        """The code of every run of the question's tokens, run[a][b] for qc[a:b]."""
        n = len(qc)
        run = [[0] * (n + 1) for _ in range(n + 1)]
        for a in range(n + 1):
            h = R_SEG
            run[a][a] = h
            for b in range(a + 1, n + 1):
                h = mix(h, qc[b - 1])
                run[a][b] = h
        return run

    @staticmethod
    def frame_code(run: list, n: int, holes: tuple) -> int:
        """The code of a frame: the question with the given places open (the runs between them, in order)."""
        if not holes:
            return mix(F0, run[0][n])
        if len(holes) == 1:
            return mix(mix(F1, run[0][holes[0][0]]), run[holes[0][1]][n])
        (i, j), (k, m) = holes
        return mix(mix(mix(F2, run[0][i]), run[j][k]), run[m][n])

    @staticmethod
    def elements_of(qc: tuple, holes: tuple) -> tuple:
        out, at = (), 0
        for i, j in holes:
            out += qc[at:i] + (HOLE,)
            at = j
        return out + qc[at:]

    def _frames(self, qc: tuple, memo: dict):
        """The question with no place open, with one, and with two: (the frame's code, the open places as spans)."""
        n = len(qc)
        run = memo.get(("runs", qc))
        if run is None:
            run = memo[("runs", qc)] = self._runs(qc)
        yield mix(F0, run[0][n]), ()
        tail = [run[j][n] for j in range(n + 1)]
        for i in range(n):
            a1 = mix(F1, run[0][i])
            for j in range(i + 1, n + 1):
                yield mix(a1, tail[j]), ((i, j),)
        if n <= PAIRS_UP_TO:
            for i in range(n):
                a1 = mix(F2, run[0][i])
                for j in range(i + 1, n):
                    rj = run[j]
                    for k in range(j, n):
                        a2 = mix(a1, rj[k])
                        for m in range(k + 1, n + 1):
                            yield mix(a2, tail[m]), ((i, j), (k, m))

    def _items(self, sk: int) -> list:
        items = self._parsed.get(sk)
        if items is None:
            items = self._parsed[sk] = self.spell[sk][len(SKELETON):].split(" ")
        return items

    @staticmethod
    def _ref(ref: str, elements: tuple, holes: tuple, qc: tuple) -> tuple:
        if ref[0] == "h":
            i, j = holes[int(ref[1:])]
            return qc[i:j]
        return (elements[int(ref[1:])],)

    def _whole(self, refs: str, elements: tuple, holes: tuple, qc: tuple, memo: dict, depth: int) -> tuple:
        """The part of the question an item asks for, put together from its pieces. Where it is put together from
        several pieces, each open place among them holds one word or a whole: something that is answered on its
        own. (A frame is not read with an arbitrary stretch of words in a place.) Empty where that does not hold."""
        pieces = [self._ref(r, elements, holes, qc) for r in refs.split(",")]
        if len(pieces) > 1:
            for r, piece in zip(refs.split(","), pieces):
                if r[0] == "h" and len(piece) > 1 and self._answer(piece, memo, depth + 1) is None:
                    return ()
        sub: tuple = ()
        for piece in pieces:
            sub += piece
        return sub

    def _const(self, fcode: int, sk: int, p: int, holes: tuple, qc: tuple) -> Optional[Tuple[int, float]]:
        """The token of a skeleton's own at place p. Where it depends on the filler: what was told of the filler
        itself, then what its kind brought here, then what stood here whatever the filler (own, kind, all)."""
        key = fold((fcode, sk, p), R_CONST)
        vs = self.S.vals(key)
        if not vs:
            return None
        single = [(h, qc[i]) for h, (i, j) in enumerate(holes) if j - i == 1]
        if single and len(vs) > 1:
            lived = {v for v, _, _ in vs}
            for h, x in single:                              # what was told of the filler itself
                own = [(v, c) for v, c, _ in self.S.vals(fold((O_TOLD, 1, x), R_SEQ)) if v in lived]
                if own:
                    e = rule(own[0][1], sum(c for _, c in own))
                    if e >= LINE:
                        return own[0][0], e
            best = None
            for h, x in single:                              # what its kind brought here
                for f in self.features(x):
                    k = fold((fcode, sk, p, h, f), R_CONSTF)
                    v2 = self.S.vals(k)
                    if v2:
                        e = rule(v2[0][1], self.S.tot(k))
                        if e >= LINE and (best is None or e > best[1]):
                            best = (v2[0][0], e)
            if best is not None:
                return best
        total = self.S.tot(key)
        if len(vs) == 1:
            e = rule(vs[0][1], total)
        else:
            # the place holds different tokens for different fillers, and nothing lived says which this filler
            # brings: what stood here most is believed only as far as "what stands here is one already met" is
            # (the novelty event, MODEL.md 2.2: nu = (t + 1/2) / (n + 1), the rest shared by the counts)
            e = (1.0 - rule(len(vs), total)) * vs[0][1] / total
        return (vs[0][0], e) if e >= LINE else None

    def _fill(self, elements: tuple, holes: tuple, fcode: int, sk: int, qc: tuple, memo: dict,
              depth: int) -> Optional[Tuple[tuple, float]]:
        """What a way of reading (a frame with a skeleton) claims for the asked question; None where a step is not
        believed at the speaking line."""
        out: list = []
        belief = 1.0
        for p, item in enumerate(self._items(sk)):
            kind = item[0]
            e = 1.0
            if kind == "*":
                got = self._const(fcode, sk, p, holes, qc)
                if got is None:
                    return None
                out.append(got[0])
                e = got[1]
            elif kind == "c":
                out.extend(self._ref(item[2:], elements, holes, qc))
            elif kind == "r":
                toks = self._ref(item[2:], elements, holes, qc)
                got = self.brings(toks[0]) if len(toks) == 1 else None
                if got is None:
                    return None
                out.append(got[0])
                e = got[1]
            elif kind == "a":
                sub = self._whole(item[2:], elements, holes, qc, memo, depth)
                got = self._answer(sub, memo, depth + 1) if sub else None
                if got is None:
                    return None
                out.extend(got[0])
                e = got[1]
            elif kind == "x":
                refs, variant = item[2:].split(":")
                ctx: tuple = ()
                for ref in refs.split("+"):
                    ctx += self._ref(ref, elements, holes, qc)
                got = self.next(ctx, variant == "n")
                if got is None or got[0] == END:
                    return None
                out.append(got[0])
                e = got[1]
            elif kind == "n":
                got = self.next(tuple(out[-NGRAM:]), item[2:] == "n") if out else None
                if got is None or got[0] == END:
                    return None
                out.append(got[0])
                e = got[1]
            elif kind == "e":
                for _ in range(64):
                    got = self.next(tuple(out[-NGRAM:]), item[2:] == "n") if out else None
                    if got is None or got[1] < LINE:
                        return None
                    if got[0] == END:
                        break
                    out.append(got[0])
                    e = min(e, got[1])
                else:
                    return None
            belief = min(belief, e)
            if belief < LINE:
                return None
        return tuple(out), belief

    def _fit(self, elements: tuple, holes: tuple, sk: int, qc: tuple, ac: tuple, memo: dict) -> Optional[list]:
        """Whether a skeleton's structure gives the answer lived, its own tokens left free; returns those tokens
        with their places."""
        pos, consts, n = 0, [], len(ac)
        for p, item in enumerate(self._items(sk)):
            kind = item[0]
            if kind == "*":
                if pos >= n:
                    return None
                consts.append((p, ac[pos]))
                pos += 1
                continue
            if kind == "e":
                while True:
                    got = self.next(ac[max(0, pos - NGRAM):pos], item[2:] == "n") if pos else None
                    if got is None:
                        return None
                    if got[0] == END:
                        break
                    if pos >= n or got[0] != ac[pos]:
                        return None
                    pos += 1
                continue
            if kind == "c":
                run = self._ref(item[2:], elements, holes, qc)
            elif kind == "r":
                toks = self._ref(item[2:], elements, holes, qc)
                got = self.brings(toks[0]) if len(toks) == 1 else None
                run = (got[0],) if got else None
            elif kind == "a":
                sub = self._whole(item[2:], elements, holes, qc, memo, 0)
                got = self._answer(sub, memo, 1) if sub else None
                run = got[0] if got else None
            elif kind == "x":
                refs, variant = item[2:].split(":")
                ctx: tuple = ()
                for ref in refs.split("+"):
                    ctx += self._ref(ref, elements, holes, qc)
                got = self.next(ctx, variant == "n")
                run = (got[0],) if got and got[0] != END else None
            else:                                            # "n"
                got = self.next(ac[max(0, pos - NGRAM):pos], item[2:] == "n") if pos else None
                run = (got[0],) if got and got[0] != END else None
            if run is None or ac[pos:pos + len(run)] != tuple(run):
                return None
            pos += len(run)
        return consts if pos == n else None

    def claims(self, qc: tuple, memo: dict, depth: int = 0, silent: bool = False) -> list:
        """Every claim the store makes for a question: (places kept, the way's belief, the claim's belief, the
        answer, frame, skeleton, elements, open places). With ``silent``, also the claims of ways not believed at
        the line (to keep their record)."""
        out = []
        for fcode, holes in self._frames(qc, memo):
            ways = self.S.vals(mix(R_KNOWN, fcode))
            self.frames_tried += 1
            if not ways:
                continue
            elements = self.elements_of(qc, holes)
            spec = len(qc) - sum(j - i for i, j in holes)
            for sk, _, _ in ways:
                e_way = rule(self.S.get(mix(R_WAY, fcode), sk), self.S.get(mix(R_WAYT, fcode), sk))
                if e_way < LINE and not silent:
                    continue
                got = self._fill(elements, holes, fcode, sk, qc, memo, depth)
                if got is not None:
                    out.append((spec, e_way, min(e_way, got[1]), got[0], fcode, sk, elements, holes))
        return out

    @staticmethod
    def _places(c: tuple) -> frozenset:
        return frozenset(i for a, b in c[7] for i in range(a, b))

    def chosen(self, claims: list):
        """The claim that is said, of those believed at the speaking line. A claim gives way to one that says
        otherwise and reads the question more closely (whose open places lie within its own: the question as lived
        before a frame of it, a frame before a wider one). Of the claims left: the one whose way has the better
        record; then the one whose weakest step is believed most; then the one that keeps more of the question."""
        live = [c for c in claims if c[2] >= LINE]
        if len(live) > 1:
            places = [self._places(c) for c in live]
            keep = [c for x, c in enumerate(live)
                    if not any(places[y] < places[x] and live[y][3] != c[3] for y in range(len(live)))]
            live = keep or live
        return max(live, key=lambda c: (c[1], c[2], c[0]), default=None)

    def _answer(self, qc: tuple, memo: dict, depth: int = 0) -> Optional[tuple]:
        """The claim that is said: the one that keeps the most of the question as it was asked, then the one
        believed most; (the answer, its belief, frame, skeleton) or None below the speaking line."""
        if qc in memo:
            return memo[qc]
        memo[qc] = None                                      # a question is not answered through itself
        if depth > 12 or not qc:
            return None
        best = self.chosen(self.claims(qc, memo, depth))
        if best is not None:
            memo[qc] = (best[3], best[2], best[4], best[5])
        return memo[qc]

    def wrong(self, q: List[str]) -> None:
        """What the mind said to a question was judged wrong: the way that spoke is believed less."""
        self.clock += 1
        qc = self.codes(q)
        said = self.chosen(self.claims(qc, {}))
        if said is not None:
            self._add(mix(R_WAYT, said[4]), said[5])

    def tone(self, fcode: int, sk: int) -> int:
        vs = self.S.vals(fold((fcode, sk), R_TONE))
        return vs[0][0] if vs else 0

    def answer(self, toks: List[str]) -> Optional[Tuple[List[str], float, str]]:
        """What the mind says to a question (its body): (the answer's words, its belief, the closing mark)."""
        got = self._answer(self.codes(toks), {})
        if got is None:
            return None
        mark = self.tone(got[2], got[3])
        return self.words(got[0]), got[1], (self.spell.get(mark, "") if mark else "")

    # ------------------------------------------------------------------------------------- living a dialogue ----
    def _skeleton(self, items: list) -> int:
        text = SKELETON + " ".join(items)
        sk = code_of(text)
        self.spell.setdefault(sk, text)
        return sk

    def _write_consts(self, fcode: int, sk: int, consts: list, holes: tuple, qc: tuple, mark: int) -> None:
        single = [(h, qc[i]) for h, (i, j) in enumerate(holes) if j - i == 1]
        feats = [(h, f) for h, x in single for f in self.features(x)]
        for p, tok in consts:
            self._tie(fold((fcode, sk, p), R_CONST), tok)
            for h, f in feats:
                self._tie(fold((fcode, sk, p, h, f), R_CONSTF), tok)
        if mark:
            self._tie(fold((fcode, sk), R_TONE), mark)

    def explain(self, qc: tuple, ac: tuple, memo: dict):
        """The answer lived, explained from the question and from what the store holds: each token a copy of a word
        of the question, what a word brings, the lived answer of a part, what followed in what was told, or a token
        of its own. The fewest tokens of its own, then the fewest pieces. Returns (elements, open places, items) or
        None where nothing of the answer is explained."""
        n, m = len(qc), len(ac)
        runs: Dict[int, list] = {a: [] for a in range(m)}     # answer place -> (length, kind, positions, payload)
        if n > 1:                                             # the lived answers of the question's parts
            parts = [tuple(range(i, j)) for i in range(n) for j in range(i + 1, n + 1) if j - i < n]
            parts += [tuple(range(0, i)) + tuple(range(j, n)) for i in range(1, n) for j in range(i + 1, n)]
            for pos in parts:
                got = self._answer(tuple(qc[i] for i in pos), memo, 1)
                if got is None or not got[0]:
                    continue
                run = got[0]
                for a in range(m - len(run) + 1):
                    if ac[a:a + len(run)] == run:
                        runs[a].append((len(run), "a", pos, None))
        brought = {i: self.brings(qc[i]) for i in range(n)}
        for a in range(m):
            for i in range(n):
                if qc[i] == ac[a]:
                    runs[a].append((1, "c", (i,), None))
                elif brought[i] is not None and brought[i][0] == ac[a]:
                    runs[a].append((1, "r", (i,), None))
            for i in range(n):
                for j in range(i + 1, min(n, i + NGRAM) + 1):
                    both = [v for v in ("s", "n") if (self.next(qc[i:j], v == "n") or (0,))[0] == ac[a]]
                    if both:
                        runs[a].append((1, "x", tuple(range(i, j)), "".join(both)))
        order = {"a": 0, "c": 1, "n": 2, "x": 3, "r": 4, "*": 5}
        INF = (10 ** 9,)
        best: Dict[Tuple[int, bool], tuple] = {(0, False): ((0, 0, 0, 0), None)}
        for a in range(m):
            for seq in (False, True):
                here = best.get((a, seq))
                if here is None:
                    continue
                cost = here[0]
                options = [(1, "*", (), None)] + runs[a]
                if seq:                                      # the told sequence goes on from what was just said
                    both = [v for v in ("s", "n") if (self.next(ac[max(0, a - NGRAM):a], v == "n") or (0,))[0] == ac[a]]
                    if both:
                        options.append((1, "n", (), "".join(both)))
                for length, kind, pos, payload in options:
                    apart = bool(pos) and pos[-1] - pos[0] + 1 != len(pos)
                    new = (cost[0] + (kind == "*"), cost[1] + 1, cost[2] + len(pos) + apart, cost[3] + order[kind])
                    state = (a + length, kind in ("c", "x", "n"))
                    if new < best.get(state, (INF,))[0]:
                        best[state] = (new, (a, seq, kind, pos, payload, length))
        end = min(((best[s][0], s) for s in ((m, False), (m, True)) if s in best), default=None)
        if end is None:
            return None
        chain, state = [], end[1]
        while best[state][1] is not None:
            a, seq, kind, pos, payload, length = best[state][1]
            chain.append([kind, pos, payload, a, length])
            state = (a, seq)
        chain.reverse()
        if all(c[0] == "*" for c in chain):
            return None
        merged: list = []                                    # copies of words that stand together are one copy
        for c in chain:
            if c[0] == "c" and merged and merged[-1][0] == "c" and merged[-1][1][-1] + 1 == c[1][0]:
                merged[-1][1] = merged[-1][1] + c[1]
            else:
                merged.append(c)
        k = len(merged)                                      # a told sequence followed to its end
        while k and merged[k - 1][0] == "n":
            k -= 1
        if k < len(merged):
            both = [v for v in ("s", "n") if all(v in c[2] for c in merged[k:])
                    and (self.next(ac[-NGRAM:], v == "n") or (0,))[0] == END]
            if both:
                merged[k:] = [["e", (), "".join(both), 0, 0]]
        sets: Dict[int, set] = {}
        for c in merged:
            for i in c[1]:
                sets.setdefault(i, set()).add(c[1])

        def lay(open_at):
            elements, holes, where = [], [], {}              # where: position -> its reference in the frame
            i = 0
            while i < n:
                if i in open_at:
                    j = i
                    while j + 1 < n and open_at.get(j + 1) == open_at[i]:
                        j += 1
                    for q in range(i, j + 1):
                        where[q] = "h%d" % len(holes)
                    holes.append((i, j + 1))
                    elements.append(HOLE)
                    i = j + 1
                else:
                    where[i] = "k%d" % len(elements)
                    elements.append(qc[i])
                    i += 1
            return elements, holes, where
        # every word the explanation uses is an open place; where that opens more than two places, the words that
        # two pieces share stay as they are (the frame keeps them)
        elements, holes, where = lay({i: frozenset(s) for i, s in sets.items()})
        if len(holes) > 2:
            elements, holes, where = lay({i: frozenset(s) for i, s in sets.items() if len(s) == 1})
        if not holes:
            return None

        def refs(pos):
            out = []
            for i in pos:
                if not out or out[-1] != where[i]:
                    out.append(where[i])
            return out
        out = []
        for variant in ("s", "n"):                           # what followed most, and what followed last
            items = []
            for kind, pos, payload, _, _ in merged:
                if kind == "*":
                    items.append("*")
                elif kind == "c":
                    items += ["c:" + r for r in refs(pos)]
                elif kind == "r":
                    items.append("r:" + where[pos[0]])
                elif kind == "a":
                    items.append("a:" + ",".join(refs(pos)))
                elif kind == "x":
                    items.append("x:" + "+".join(refs(pos)) + ":" + (variant if variant in payload else payload[0]))
                else:
                    items.append(kind + ":" + (variant if variant in payload else payload[0]))
            if items not in out:
                out.append(items)
        return tuple(elements), tuple(holes), out

    def _born(self, fcode, sk, elements, holes, qc, ac, mk, memo, told) -> None:
        """A way of reading is written, unless the store holds it. The answer as lived (no place open) is a fact
        lived once. A frame is a rule read from one dialogue: it has no confirmation until a dialogue that did not
        make it confirms it, and says nothing until then."""
        if self.S.get(mix(R_KNOWN, fcode), sk):
            return
        told["new"] = told.get("new", 0) + 1
        self._add(mix(R_KNOWN, fcode), sk)
        if not holes:
            self._add(mix(R_WAY, fcode), sk)
            self._add(mix(R_WAYT, fcode), sk)
        else:
            self._add(mix(R_BIRTH, mix(fcode, sk)), mix(F0, memo[("runs", qc)][0][len(qc)]))
        self._write_consts(fcode, sk, [(p, ac[p]) for p in range(len(ac))] if not holes else
                           self._fit(elements, holes, sk, qc, ac, memo) or [], holes, qc, mk)

    def _open_constant(self, said, qc, ac, mk, memo, told) -> None:
        """A token of a skeleton's own that the store can now explain is no longer its own: where the way that
        foresaw the answer keeps a token which is a word of the frame itself, or what such a word brings, the
        frame with that word open and the token read from it is written (a wider frame, read from one that was
        just confirmed)."""
        _, _, _, _, fcode, sk, elements, holes = said
        items = self._items(sk)
        if "*" not in items or len(holes) >= 2:
            return
        consts = self._fit(elements, holes, sk, qc, ac, memo)
        if not consts:
            return
        inside = {i for a, b in holes for i in range(a, b)}
        for i in range(len(qc)):
            if i in inside:
                continue
            got = self.brings(qc[i])
            kinds = {p: ("c" if tok == qc[i] else "r") for p, tok in consts
                     if tok == qc[i] or (got is not None and got[0] == tok)}
            if not kinds:
                continue
            # the old frame's references as places of the question, then as references of the wider frame
            place_of, at = {}, 0
            for e, el in enumerate(elements):
                if el == HOLE:
                    h = sum(1 for x in elements[:e] if x == HOLE)
                    place_of["h%d" % h] = list(range(*holes[h]))
                    at = holes[h][1]
                else:
                    place_of["k%d" % e] = [at]
                    at += 1
            wider = tuple(sorted(holes + ((i, i + 1),)))
            where, els, k = {}, [], 0
            at = 0
            for h, (a, b) in enumerate(wider):
                for q in range(at, a):
                    where[q] = "k%d" % len(els)
                    els.append(qc[q])
                for q in range(a, b):
                    where[q] = "h%d" % h
                els.append(HOLE)
                at = b
            for q in range(at, len(qc)):
                where[q] = "k%d" % len(els)
                els.append(qc[q])

            def again(refs, sep):
                out = []
                for r in refs.split(sep):
                    for q in place_of[r]:
                        if not out or out[-1] != where[q]:
                            out.append(where[q])
                return sep.join(out)
            new_items = []
            for p, item in enumerate(items):
                if p in kinds:
                    new_items.append(kinds[p] + ":" + where[i])
                elif item[0] in ("c", "r"):
                    new_items.append(item[0] + ":" + again(item[2:], ","))
                elif item[0] == "a":
                    new_items.append("a:" + again(item[2:], ","))
                elif item[0] == "x":
                    refs, variant = item[2:].split(":")
                    new_items.append("x:" + again(refs, "+") + ":" + variant)
                else:
                    new_items.append(item)
            run = memo[("runs", qc)]
            self._born(self.frame_code(run, len(qc), wider), self._skeleton(new_items), tuple(els), wider, qc, ac, mk,
                       memo, told)
            return

    def watch(self, q: List[str], a: List[str], mark: str = "") -> dict:
        """A dialogue lived: the question's body, the answer's body, the answer's closing mark."""
        self.clock += 1
        qc, ac = self.codes(q), self.codes(a)
        mk = self.code(mark) if mark else 0
        memo: dict = {}
        fitted = set()
        told = {"claims": 0, "right": 0}
        if qc and ac:
            # 1. the answer is foreseen by every way the store knows, believed or not
            made, fits = [], []
            for fcode, holes in list(self._frames(qc, memo)):
                ways = self.S.vals(mix(R_KNOWN, fcode))
                if not ways:
                    continue
                elements = self.elements_of(qc, holes)
                spec = len(qc) - sum(j - i for i, j in holes)
                for sk, _, _ in ways:
                    claim = self._fill(elements, holes, fcode, sk, qc, memo, 0)
                    if claim is not None:
                        e_way = rule(self.S.get(mix(R_WAY, fcode), sk), self.S.get(mix(R_WAYT, fcode), sk))
                        made.append((spec, e_way, min(e_way, claim[1]), claim[0], fcode, sk, elements, holes))
                    consts = self._fit(elements, holes, sk, qc, ac, memo)
                    if consts is not None:
                        fits.append((fcode, sk, consts, holes, claim is not None and claim[0] == ac))
            said = self.chosen(made)
            right = [c for c in made if c[3] == ac]
            told["claims"] = len(made)
            told["said"] = None if said is None else said[3] == ac
            # 2. the records. Every way that would have been right is believed more, unless this is the very
            # dialogue it was read from (a rule is tested on what did not make it). The way that would have
            # spoken, and wrongly, is believed less.
            here = mix(F0, memo[("runs", qc)][0][len(qc)])
            for c in right:
                if c[7] and self.S.get(mix(R_BIRTH, mix(c[4], c[5])), here):
                    continue
                self._add(mix(R_WAYT, c[4]), c[5])
                self._add(mix(R_WAY, c[4]), c[5])
            if said is not None and said[3] != ac:
                self._add(mix(R_WAYT, said[4]), said[5])
            # 3. a way whose structure gives the answer, but which did not say it, learns the tokens of its own
            # (each way learns from what it did not foresee, whoever else did)
            for fcode, sk, consts, holes, foresaw in fits:
                if not foresaw:
                    self._write_consts(fcode, sk, consts, holes, qc, mk)
            run = memo[("runs", qc)]
            if said is not None and said[3] == ac:
                told["right"] = 1
                if self.widen:
                    self._open_constant(said, qc, ac, mk, memo, told)
            elif not any(not f[3] for f in fits):
                # 4. a surprise, and the store holds no memory of the question that it can correct: the answer as
                # lived is written
                self._born(mix(F0, run[0][len(qc)]), self._skeleton(["*"] * len(ac)), qc, (), qc, ac, mk, memo, told)
            if not any(c[7] for c in right):
                # 5. no frame would have given the answer (the mind may remember it, but that is not to have read
                # it from the question): the frame that explains the answer from the question and from what the
                # store holds now is written
                got = self.explain(qc, ac, memo)
                if got is not None and len(got[1]) <= 2:         # a frame is read with at most two places open
                    for items in got[2]:
                        self._born(self.frame_code(run, len(qc), got[1]), self._skeleton(items), got[0], got[1],
                                   qc, ac, mk, memo, told)
                elif got is not None:
                    told["wider"] = 1
            # 6. what each word of the question brings
            qs, as_ = sorted(set(qc)), sorted(set(ac))
            self._bump(R_REL)
            for x in qs:
                self._bump(mix(R_REL, x))
                for t in as_:
                    self._add(mix(R_REL, x), t)
            for t in as_:
                self._bump(mix(R_RELR, t))
                for x in qs:
                    self._add(mix(R_RELR, t), x)
        # 7. both as sequences
        self.sequence(qc, O_ASKED)
        self.sequence(ac, O_ANSWER)
        return told
