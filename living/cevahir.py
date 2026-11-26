# -*- coding: utf-8 -*-
"""Cevahir: one mind, one layer (research/living_intelligence_2026_11_24/MODEL.md, section 12).

Two bodies of one mind meet in one place.

- **Zihin** (``zihin/``, every mechanism it was built with) lives each word as a moment: its units and ties, its
  frames, families and relations, its wholes, its inferences and conjectures, its ground. It is the mind that
  generalises.
- **The cortex** (``living/fast.py``) lives each letter: its contexts, its episodes, its lines, its short-term
  memory, and the harmony that combines them bit by bit. It is the mind that predicts.

They meet in the harmony. Before every token, what Zihin expects next is a **voice** of the harmony: its candidates,
each in the forms it was lived in (the gap before it and its letters, so that case and spacing come from life, not
from a rule), weighted by how far each is believed above not knowing (e − ½), with how much Zihin knows,
2 (e_best − ½). The harmony trusts the voice by the one rule on the voice's own record (how often its top symbol
was right, per sixth of knowing), and its gate, which learns from its own error, decides with every other origin
how far to follow it. Nothing decides for the harmony: where Zihin is right and the cortex has never lived the
case, Zihin carries the decision; where the cortex knows more (spelling, a line it has lived), the cortex does.

When Cevahir answers, Zihin answers first, as it always did (its race, its speaking line, its wholes answered part
by part): that is the plan, the thought. The harmony then says the answer letter by letter with the plan as the
voice, trusted as Zihin's own decision. ``way`` chooses what speaks, for the measurements: ``"zihin"`` (Zihin
alone), ``"harmony"`` (the cortex alone), ``"cevahir"`` (the one layer).

Nothing here is a mechanism of a test world: no word, no language and no format is named in the code.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from living.analogy import Episodes
from living.fast import FastCortex

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)          # Zihin's tokens (zihin.__main__.words), on the raw text
SEPARATOR = re.compile(r"(?<=[.!?…])\s+|\n+")             # Zihin's sentence split (zihin.__main__.sentences)
CORTEX = dict(depth=8, episode=True, minlen=12, episode_bits=22, mix=True, lr=0.02, lr_min=0.002, words=True,
              two_inputs=True, tree_prior=True, recent=6, lines=True)
TAIL = 8192                                             # characters of the stream kept as the harmony's history
SAY_TAIL = 4096                                         # of which the harmony reads, when it speaks (a whole line)


def zihin_config(variant: str = "full", switches: Optional[Dict[str, bool]] = None):
    """A configuration of Zihin as the chat program builds it (``zihin.__main__.VARIANTS``), without a ledger file."""
    import zihin.__main__ as chat
    made = chat._log_path
    chat._log_path = lambda *a: None
    try:
        cfg = chat.VARIANTS[variant]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for k, v in (switches or {}).items():
        if not hasattr(cfg, k):
            raise ValueError("no such switch: %s" % k)
        setattr(cfg, k, v)
    return cfg


class Cevahir:
    """One mind: Zihin and the cortex, meeting in the harmony.

    ``variant`` and ``switches`` configure Zihin (``zihin_config``); ``window`` is Zihin's focus F (its window);
    ``cortex`` changes the cortex's configuration (``CORTEX``); ``dil`` is Zihin's language convention. Switches of
    Cevahir's own (``OWN_SWITCHES``) may be given among ``switches``."""

    OWN_SWITCHES = ("alone_now",)    # the record of Zihin alone as it speaks, read as far as it believes (off)

    def __init__(self, variant: str = "full", switches: Optional[Dict[str, bool]] = None, window: int = 0,
                 cortex: Optional[dict] = None, dil: str = "tr", seed: int = 0, top: int = 12, forms_per: int = 2,
                 record: bool = True, keep_stream: bool = False, ledger=False):
        import torch
        from zihin import Mind
        from zihin.__main__ import CHANNEL, Session, _log_path
        torch.manual_seed(seed)
        switches = dict(switches or {})
        self.alone_now = bool(switches.pop("alone_now", False))
        cfg_z = zihin_config(variant, switches)
        if ledger:                                       # Zihin keeps its ledger: it can forget exactly. On disk (a
            cfg_z.keep_log = True                        # folder given, or a temporary one), see ``keep_ledger``
            cfg_z.log_path = _log_path(ledger if isinstance(ledger, str) else None)
        self.mind = Mind(cfg_z)
        self.mind.add_channel(CHANNEL, "act")
        if window:
            self.mind.knob["F"] = window
        self.session = Session(self.mind)
        self.session.dil = dil
        self.channel = CHANNEL
        cfg = dict(CORTEX)
        cfg.update(cortex or {})
        self.cortex = FastCortex(voice=True, **cfg)
        self.written: Dict[bool, Dict[str, Counter]] = {False: defaultdict(Counter), True: defaultdict(Counter)}
        #                                                        (in an answer) -> a token -> how it was written
        self.gaps: Dict[Tuple[bool, bool], Counter] = defaultdict(Counter)   # (in an answer, first token) -> gaps
        self.tail = ""                                           # the stream as lived, its last TAIL characters
        self.stream: Optional[List[str]] = [] if keep_stream else None   # all of it, when asked (comparisons)
        self.top, self.forms_per = top, forms_per
        self.counts = Counter()                                  # tokens heard with the voice, without, unaligned
        self.record = record                                     # keep the record of disagreements while living
        self.dis_won = Counter()                                 # disagreements per situation: Zihin's token was it
        self.dis_all = Counter()                                 # disagreements per situation (and agreements apart)
        self.alone_right = Counter()                             # Zihin expected nothing: the harmony's guess was it
        self.alone_all = Counter()                               # Zihin expected nothing, the harmony guessed
        self.az_won = 0                                          # Zihin and the analogy differed, one was right:
        self.az_all = 0                                          # ... the analogy's token was it (per token)
        self.za_won = 0                                          # answers the analogy foresaw nothing of, Zihin
        self.za_all = 0                                          # ... guessing them: every token of it right
        self.za_claims: List[Tuple[float, int]] = []             # ... each claim: (how far Zihin believed it, right)
        self.outcome_good = Counter()                            # the amygdala: answers judged, per (who led, situation)
        self.outcome_all = Counter()
        self.last_leads: List[Tuple[str, int]] = []              # who led each token of the last answer, and where
        self.pending: Optional[Tuple[str, str]] = None           # the last turn, lived by the harmony once judged
        self.alphabets: Dict[str, set] = {self.channel: set()}   # every channel's own alphabet (MODEL.md 4)
        self.entries: Optional[List[dict]] = [] if ledger else None  # what was lived, entry by entry (to forget)
        self.episodes = Episodes()                                   # the hippocampus: dialogues as lived
        self.marks_kept = True                                   # ... what was said heard with its closing marks
        self._open: Optional[dict] = None
        self.others: Dict[str, dict] = {}                        # channels born by the rule: a harmony each

    # ------------------------------------------------------------------------------------------------ the voice --
    def candidates(self, token: str, partial: Optional[str] = None, weight: float = 1.0, gaps: int = 3,
                   where: Tuple[bool, bool] = (False, False)) -> Tuple[List[str], List[float]]:
        """The forms a token may take next: a gap lived before tokens in the same situation (``where``: in an
        answer or not, the first token of the utterance or not) and a way the token was written, weighted by how
        often each was lived. With ``partial`` (the gap already lived), only the written forms."""
        ws = self.written[where[0]].get(token) or self.written[not where[0]].get(token)
        if not ws:
            return [], []
        written = ws.most_common(self.forms_per)
        wt = float(sum(n for _, n in written))
        if partial is not None:
            return [w for w, _ in written], [weight * n / wt for _, n in written]
        lived = (self.gaps.get(where) or Counter({" ": 1})).most_common(gaps)
        gt = float(sum(n for _, n in lived)) or 1.0
        forms, weights = [], []
        for g, gn in lived:
            for w, n in written:
                forms.append(g + w)
                weights.append(weight * (gn / gt) * (n / wt))
        return forms, weights

    def voice_of_zihin(self, where: Tuple[bool, bool] = (False, False)) -> Tuple[Optional[str], float]:
        """Set the voice from what Zihin expects next on its text channel; returns its top token and its e."""
        expected = self.mind.expected(self.channel)
        believed = sorted(((e, s) for s, e in expected.items() if e > 0.5 and isinstance(s, str)), reverse=True)
        believed = believed[:self.top]
        if not believed:
            self.cortex.set_voice([], [], 0.0)
            return None, 0.0
        know = 2.0 * (believed[0][0] - 0.5)
        forms, weights = [], []
        for e, s in believed:
            f, w = self.candidates(s, None, e - 0.5, where=where)
            forms += f
            weights += w
        self.cortex.set_voice(forms, weights, know)
        return believed[0][1], believed[0][0]

    def zihin_would(self) -> Tuple[Optional[str], float]:
        """Zihin's next token as it would say it, and its e: the present context read as when it speaks, with every
        way of reading it has (``predictions`` "specific"); the voice of the harmony is the cheaper reading made
        while it listens (the context's own memory)."""
        expected = self.mind.expected(self.channel, "specific")
        believed = max(((e, s) for s, e in expected.items() if isinstance(s, str)), default=(0.0, None))
        return believed[1], believed[0]

    @staticmethod
    def situation(answer: bool, first: bool, e: float) -> int:
        """The class of a disagreement: in an answer or not, x the first token of the utterance or a later one,
        x Zihin at the speaking line or below it."""
        from living.evidence import SPEAKING_LINE
        return (4 if answer else 0) + (0 if first else 2) + (1 if e >= SPEAKING_LINE else 0)

    def trust_zihin(self, cell: int) -> float:
        """The belief, by the one rule, that Zihin's token is the one when the harmony's differs. Its record: the
        disagreements lived in this situation where one of the two was right (counted for Zihin when it was
        Zihin), and the outcomes of answers (the amygdala, ``feedback``): an answer judged right counts for the one
        that led there, an answer judged wrong counts against it."""
        won = self.dis_won[cell] + self.outcome_good[("zihin", cell)] + \
            (self.outcome_all[("harmony", cell)] - self.outcome_good[("harmony", cell)])
        seen = self.dis_all[cell] + self.outcome_all[("zihin", cell)] + self.outcome_all[("harmony", cell)]
        return (won + 0.5) / (seen + 1.0)

    def trust_alone(self, cell: int) -> float:
        """The belief, by the one rule on its record in this situation, that the harmony's token is the one where
        Zihin expects nothing above not knowing (with the outcomes of the answers it gave alone)."""
        return (self.alone_right[cell] + self.outcome_good[("alone", cell)] + 0.5) / \
            (self.alone_all[cell] + self.outcome_all[("alone", cell)] + 1.0)

    def _begin(self, kind: str) -> Optional[dict]:
        """Open an entry of the history (with a ledger): what is lived until ``_end`` can be forgotten as one."""
        if self.entries is None or self._open is not None:
            return None
        self._open = {"kind": kind, "t0": self.mind.t + 1, "text_from": self.cortex.size()["episode_text"],
                      "history": self.tail[-4096:], "text": [], "forms": [], "forgotten": False,
                      "night": getattr(self, "nights", 0)}
        return self._open

    def _end(self, entry: Optional[dict]) -> None:
        if entry is None:
            return
        entry["t1"] = self.mind.t
        entry["text"] = "".join(entry["text"])
        self.entries.append(entry)
        self._open = None

    def _live_text(self, text: str) -> float:
        bits, _ = self.cortex.stream(text)
        if self._open is not None:
            self._open["text"].append(text)
        self.alphabets[self.channel].update(text)
        self.tail = (self.tail + text)[-TAIL:]
        if self.stream is not None:
            self.stream.append(text)
        return bits

    def tell(self, sentence: str, prov: Optional[int] = None, scene: Optional[dict] = None,
             tone: Optional[str] = None) -> float:
        """Read a sentence: see ``_tell`` (one entry of the history, with a ledger)."""
        self.flush()
        entry = self._begin("tell")
        try:
            return self._tell(sentence, prov, scene, tone)
        finally:
            self._end(entry)

    def _tell(self, sentence: str, prov: Optional[int] = None, scene: Optional[dict] = None,
             tone: Optional[str] = None) -> float:
        """A sentence of a document, read on its own line (as ``Session.read_path`` lives a sentence: heard, its
        end perceived, a pause). With a ground and a ``scene``, what is seen while it is told is perceived."""
        self.flush()
        from zihin.__main__ import IMITATED
        prov = IMITATED if prov is None else prov
        s = self.session
        z = getattr(self.mind, "zemin", None) if scene is not None else None
        s.company(None)
        if z is not None:
            z.see(scene)
        bits = self.hear(sentence, prov, lead=self._lead())
        s.end_of_turn(prov, tone=tone or s.tone(sentence))
        if z is not None:
            z.heard(s.body(sentence), "anlatan")
        self.mind.step([])
        return bits

    def ground(self) -> None:
        """Give Zihin its ground (zihin/zemin.py): perception and words meet in the same calculus."""
        from zihin.zemin import Zemin
        self.mind.cfg.grounded = True
        self.mind.zemin = Zemin(self.channel)

    # --------------------------------------------------------------------------------------------- hearing ------
    def hear(self, text: str, prov: int, speaker: Optional[str] = None, lead: str = "", answer: bool = False,
             foreseen: Optional[List[str]] = None, unforeseen: bool = False) -> float:
        """One utterance heard while not speaking (Zihin's ``Session.listen``): before each of its tokens the
        harmony is asked with Zihin's expectations as the voice and lives the token's form (its gap and its
        letters); then Zihin lives the token as a moment. ``lead``: what comes before the first token (a line end,
        a tab); ``answer``: the utterance answers the one before it; ``unforeseen``: the analogy foresaw nothing of
        it, and how Zihin guesses it is kept as its record there (``trust_unforeseen``). Returns the bits the
        harmony paid."""
        from living.evidence import SPEAKING_LINE
        s = self.session
        body = s.body(text)                                  # what Zihin lives: its words, the closing mark apart
        spans = [(m.start(), m.end()) for m in TOKEN.finditer(text)]
        aligned = [s.lower(text[a:b]) for a, b in spans[:len(body)]] == list(body)
        theta = self.mind.knob["theta"]
        self.mind.knob["theta"] = 1e9                        # listening: the mind does not speak into it
        bits = 0.0
        try:
            if not aligned:                                  # Zihin's words are not the raw tokens: no voice
                self.counts["unaligned"] += 1
                self.cortex.set_voice([], [], 0.0)
                bits += self._live_text(lead + text)
                for symbol in body:
                    self.mind.step(s.events(symbol, prov, speaker))
                return bits
            prev = 0
            on_track = foreseen is not None                  # the analogy's plan, while it agrees with what was said
            own = None                                       # Zihin alone where the analogy foresaw nothing: right
            #                                                  so far (None: it would not say a first token)
            own_e = 1.0                                      # ... and how far it believed what it would say
            for i, (a, b) in enumerate(spans):
                form = (lead if i == 0 else "") + text[prev:a] + text[a:b]
                if i < len(body):
                    h_tok = None
                    if self.record:                          # the harmony's own guess, before the voice is set
                        h_form, _ = self.harmony_token(self.tail)
                        found = TOKEN.findall(h_form)
                        h_tok = s.lower(found[0]) if found else None
                    where = (answer, i == 0)
                    z_tok, z_e = self.voice_of_zihin(where)
                    if self.record and z_tok is not None and h_tok is not None and z_tok != h_tok and \
                            (z_tok == body[i]) != (h_tok == body[i]):    # they differ, and one of them is right
                        cell = self.situation(answer, i == 0, z_e)
                        self.dis_all[cell] += 1
                        self.dis_won[cell] += 1 if z_tok == body[i] else 0
                    # The analogy's plan is made once, whole, before the answer is heard; Zihin guesses each token
                    # knowing those said before it. A plan is a guess of a token only while it has agreed with
                    # every token said before it: once astray, it says nothing of the rest.
                    a_tok = foreseen[i] if on_track and i < len(foreseen) else None
                    if a_tok is not None and z_tok is not None and a_tok != z_tok and \
                            (a_tok == body[i]) != (z_tok == body[i]):    # the analogy and Zihin, one of them right
                        self.az_all += 1
                        self.az_won += 1 if a_tok == body[i] else 0
                    on_track = on_track and a_tok == body[i]
                    if answer and unforeseen and (i == 0 or own):   # a claim begins where Zihin would say the
                        f_tok, f_e = self.zihin_would() if self.alone_now else (z_tok, z_e)   # first token (as it
                        said_it = f_tok is not None and f_e >= SPEAKING_LINE   # speaks, with alone_now), and holds
                        if i == 0:                           # while it would say each token and each is the one
                            own = (f_tok == body[0]) if said_it else None   # said; it is believed as far as the
                            own_e = f_e                      # least believed of them
                        else:
                            own = said_it and f_tok == body[i]
                            own_e = min(own_e, f_e)
                    if self.record and z_tok is None and h_tok is not None:
                        cell = self.situation(answer, i == 0, 0.0)
                        self.alone_all[cell] += 1
                        self.alone_right[cell] += 1 if h_tok == body[i] else 0
                    bits += self._live_text(form)
                    self.gaps[where][form[:len(form) - (b - a)]] += 1
                    self.written[answer][body[i]][text[a:b]] += 1
                    if self._open is not None:
                        self._open["forms"].append((where, form[:len(form) - (b - a)], answer, body[i], text[a:b]))
                    self.mind.step(s.events(body[i], prov, speaker))
                    self.counts["voiced"] += 1
                else:                                        # the closing mark: heard as the end of the turn
                    self.cortex.set_voice([], [], 0.0)
                    bits += self._live_text(form)
                prev = b
            if own is not None:                              # the whole answer, as Zihin alone would have said it
                self.za_all += 1
                self.za_won += 1 if own else 0
                self.za_claims.append((own_e, 1 if own else 0))
            if not spans and lead:
                bits += self._live_text(lead)
            if text[prev:]:
                self.cortex.set_voice([], [], 0.0)
                bits += self._live_text(text[prev:])
        finally:
            self.mind.knob["theta"] = theta
        return bits

    def _lead(self) -> str:
        return "\n" if self.tail else ""

    def watch(self, first: str, answer: str, prov: Optional[int] = None, asker: Optional[str] = None,
              scene: Optional[dict] = None, scene_answer: Optional[dict] = None) -> float:
        """Watch a dialogue: see ``_watch`` (one entry of the history, with a ledger)."""
        self.flush()
        entry = self._begin("watch")
        try:
            return self._watch(first, answer, prov, asker, scene, scene_answer)
        finally:
            self._end(entry)

    def _watch(self, first: str, answer: str, prov: Optional[int] = None, asker: Optional[str] = None,
              scene: Optional[dict] = None, scene_answer: Optional[dict] = None) -> float:
        """Two others talk (Zihin's ``Session.watch``): one says ``first``, another answers. The harmony lives the
        dialogue as one line, the first utterance and the answer apart by a tab. With a ground (``mind.zemin``) and a
        ``scene``, what is seen while each is said is perceived by the ground (``scene_answer`` while the answer is
        said, if it differs), as research/living_mind_2026_11_23/t_m2.py lives its world."""
        self.flush()
        from zihin.__main__ import IMITATED, SOMEONE
        prov = IMITATED if prov is None else prov
        asker = SOMEONE if asker is None else asker
        s = self.session
        z = getattr(self.mind, "zemin", None) if scene is not None else None
        s.company(asker)
        if z is not None:
            z.see(scene)
        bits = self.hear(first, prov, speaker=asker, lead=self._lead())
        s.end_of_turn(prov, tone=s.tone(first))
        if z is not None:
            z.heard(s.body(first), "soran")
            z.see(scene_answer or scene)
        from living.evidence import SPEAKING_LINE
        a_plan, a_e, _ = self.episodes.plan(self._heard(first), parts=False)   # the hippocampus foresees the answer
        foreseen = [self.session.lower(t) for t in a_plan] if a_plan else None
        bits += self.hear(answer, prov, lead="\t", answer=True, foreseen=foreseen,
                          unforeseen=not a_plan or a_e < SPEAKING_LINE)
        s.end_of_turn(prov, tone=s.tone(answer))
        if z is not None:
            z.heard(s.body(answer), "cevaplayan")
        self.mind.step([])
        if getattr(self.mind.cfg, "wholes", False):          # F43: the dialogue lived once more as wholes
            from zihin.butun import consolidate
            consolidate(s, first, answer, prov, asker)
        self.episodes.live(self._heard(first), self._said(answer))   # the hippocampus keeps it as it was said
        if self._open is not None:
            self._open["episode"] = len(self.episodes) - 1
        return bits

    def _heard(self, text: str) -> List[str]:
        """What was said to Cevahir, as the hippocampus keeps it: its tokens as said (case kept) with its closing marks,
        which end the utterance and stand after its last word, so that a part ending at that word can be read with what
        stands after it, as every other part is (a mind saved before marks were kept hears without them)."""
        from living.analogy import tokens
        return tokens(text) if self.marks_kept else self._said(text)

    @staticmethod
    def _said(text: str) -> List[str]:
        """An utterance's tokens as said (case kept), without its closing marks."""
        from zihin.__main__ import FINAL
        from living.analogy import tokens
        out = tokens(text)
        while out and out[-1] in FINAL:
            out = out[:-1]
        return out

    def read(self, text: str, code: bool = False) -> float:
        """Read a document: see ``_read`` (one entry of the history, with a ledger)."""
        self.flush()
        entry = self._begin("read")
        try:
            return self._read(text, code)
        finally:
            self._end(entry)

    def _read(self, text: str, code: bool = False) -> float:
        """A document (Zihin's ``Session.read_path``): the harmony lives every character of it; Zihin lives each
        sentence of at least two words (each line, for source code) as an utterance, its end and a pause."""
        self.flush()
        from zihin.__main__ import IMITATED
        s = self.session
        s.company(None)
        bits = 0.0
        pieces = []
        if code:
            pos = 0
            for line in text.split("\n"):
                pieces.append((pos, pos + len(line)))
                pos += len(line) + 1
        else:
            pos = 0
            for m in SEPARATOR.finditer(text):
                pieces.append((pos, m.start()))
                pos = m.end()
            pieces.append((pos, len(text)))
        done = 0
        for a, b in pieces:
            raw = text[a:b]
            lead = text[done:a]
            if len(s.words(raw)) >= 2 and raw.strip():
                left = len(raw) - len(raw.lstrip())
                sentence = raw.strip()
                bits += self.hear(sentence, IMITATED, lead=lead + raw[:left])
                s.end_of_turn(IMITATED, tone=s.tone(sentence))
                self.mind.step([])
                rest = raw[left + len(sentence):]
                if rest:
                    self.cortex.set_voice([], [], 0.0)
                    bits += self._live_text(rest)
            else:
                self.cortex.set_voice([], [], 0.0)
                bits += self._live_text(lead + raw)
            done = b
        if done < len(text):
            self.cortex.set_voice([], [], 0.0)
            bits += self._live_text(text[done:])
        return bits

    # --------------------------------------------------------------------------------------------- speaking -----
    def belief(self, history: str, text: str) -> float:
        """The harmony's belief in ``text`` coming next after ``history``: 2 to the minus its bits, nothing learned
        (the voice silent)."""
        self.cortex.set_voice([], [], 0.0)
        bits, _ = self.cortex.stream(text, history, learn=False, tail=SAY_TAIL)
        return 2.0 ** -bits

    def harmony_token(self, history: str, most: int = 64) -> Tuple[str, float]:
        """The harmony's own next token after ``history``: its most probable symbols (the voice silent) up to the end
        of one token, with the gap before it. Returns (the form, its probability); ("\n", p) when the line ends
        first, ("", 0) when it would say a symbol it never met."""
        self.cortex.set_voice([], [], 0.0)
        buf, probs = "", []
        while len(buf) < most:
            text, p = self.cortex.say(history + buf, most=1, tail=SAY_TAIL)
            if not text:
                return "", 0.0
            if text == "\n" and not buf.strip():
                return "\n", p[0]
            buf += text
            probs.append(p[0])
            toks = list(TOKEN.finditer(buf))
            if not toks:
                continue
            t = toks[0]
            word = bool(re.match(r"\w", t.group()))
            if not word or t.end() < len(buf):            # a mark is whole at once; a word when something follows
                end = t.end()
                prob = 1.0
                for q in probs[:end]:
                    prob *= q
                return buf[:end], prob
        prob = 1.0
        for q in probs:
            prob *= q
        return buf, prob

    def speak(self, history: str, plan: Sequence[str] = (), plan_e: Sequence[float] = (), start: str = "",
              most: int = 400, line: Optional[float] = None) -> str:
        """The prefrontal decision, one token at a time (MODEL.md 12). Zihin's next token (its decision, believed
        with e) and the harmony's own next token (believed with the product of its symbols' probabilities):

        - when they are the same token, it is said (in the harmony's form): Zihin decided it and the harmony agrees;
        - when they differ, Zihin's token is said (Zihin decided it: it is the plan) unless the harmony's own token
          reaches the speaking line and the record of disagreements in this situation trusts the harmony more (the
          one rule: how often Zihin's token was the one when the two differed while Cevahir lived, with the
          outcomes of judged answers);
        - when Zihin says nothing at all, the harmony speaks alone only if its record of speaking where Zihin
          expected nothing, in this situation, is above one half; once Zihin's plan is said, the harmony adds no
          word of its own (the plan is the thought; its end is part of it);
        - otherwise nothing more is said.

        How Zihin's token is written (its gap, its case) is the harmony's choice among the gaps and the ways it was
        written that it has lived, never a line end: the answer is one line. A token of the plan never written is
        written as it is held, after a gap lived in the situation (without it, a plan by parts that counts a place
        beyond any lived, "x _ 24", was cut there). ``start``: the gap already lived after
        ``history``. Nothing is learned. Returns what was said, without ``start``."""
        from living.evidence import SPEAKING_LINE
        line = SPEAKING_LINE if line is None else line
        said = ""
        partial: Optional[str] = start
        k = 0
        self.last_leads = []
        while len(said) < most:
            here = history + said
            h_form, h_p = self.harmony_token(here)
            h_found = TOKEN.findall(h_form)
            h_tok = self.session.lower(h_found[0]) if h_found else None
            z_tok = plan[k] if k < len(plan) else None
            z_e = (plan_e[k] if k < len(plan_e) else line) if z_tok is not None else 0.0
            if z_tok is not None and z_tok == h_tok:          # Zihin decided it and the harmony agrees
                said += h_form
                k += 1
            else:
                z_form = ""
                if z_tok is not None:
                    forms, _ = self.candidates(z_tok, partial, gaps=6, where=(True, k == 0))
                    forms = [f for f in forms if "\n" not in f]
                    if not forms:                            # a token of the plan never written (a number the
                        #                                      analogy counted, say): written as it is held, after
                        #                                      a gap lived in this situation
                        lived = (self.gaps.get((True, k == 0)) or Counter({" ": 1})).most_common(6)
                        forms = [z_tok] if partial is not None else \
                            [g + z_tok for g, _ in lived if "\n" not in g] or [" " + z_tok]
                    if forms:
                        beliefs = [self.belief(here, f) for f in forms]
                        z_form = forms[max(range(len(forms)), key=lambda i: beliefs[i])]
                lead_z = bool(z_form) and (h_tok is None or h_p < line or
                                           self.trust_zihin(self.situation(True, k == 0, z_e)) >= 0.5)
                if lead_z:
                    said += z_form
                    if h_tok is not None:
                        self.last_leads.append(("zihin", self.situation(True, k == 0, z_e)))
                    k += 1
                elif h_tok is not None and h_p >= line and (z_tok is not None or (
                        not plan and self.trust_alone(self.situation(True, k == 0, 0.0)) > 0.5)):
                    said += h_form                           # alone only where Zihin said nothing at all, and only
                    #                                          as far as its record when alone allows
                    if z_tok is not None:
                        self.last_leads.append(("harmony", self.situation(True, k == 0, z_e)))
                    else:
                        self.last_leads.append(("alone", self.situation(True, k == 0, 0.0)))
                else:
                    break
            partial = None
        return said

    def trust_analogy(self) -> float:
        """The belief, by the one rule, that the analogy's token is the one where it and Zihin's differ while
        Cevahir lives a dialogue and one of the two is right."""
        return (self.az_won + 0.5) / (self.az_all + 1.0)

    def trust_unforeseen(self, believed: Optional[float] = None) -> Optional[float]:
        """The belief, by the one rule, that Zihin's answer is right where the analogy foresees nothing: of the
        answers Cevahir lived that the analogy foresaw nothing of (no plan, or one under the speaking line) and
        whose first token Zihin would have said (as it speaks, at the speaking line), how many it would have said
        whole, each token its guess at the line, knowing the tokens said before it, and the one said. With
        ``believed`` (how far Zihin believes the plan asked about: its least believed word), only the claims it
        believed at least as far are counted: what Zihin came to while it knew less, and guessed less surely, says
        nothing of a plan it is surer of now. None while there are none: nothing lived says how far Zihin goes
        alone so surely."""
        if believed is not None and len(self.za_claims) == self.za_all:
            seen = [won for e, won in self.za_claims if e >= believed]
            if not seen:
                return None
            return (sum(seen) + 0.5) / (len(seen) + 1.0)
        if not self.za_all:
            return None
        return (self.za_won + 0.5) / (self.za_all + 1.0)

    def choose_plan(self, question: str, plan: List[str], plan_e: List[float]) -> Tuple[List[str], List[float], str]:
        """The prefrontal choice of the thought (MODEL.md 12.5): Zihin's plan or the hippocampus's analogy. The
        analogy is a thought only if its own belief reaches the speaking line. Where Zihin has no plan, it is the
        thought; where the two plans differ, the record of their disagreements decides (``trust_analogy``). Where
        the analogy foresees nothing, Zihin's plan is the thought as far as its record alone allows
        (``trust_unforeseen``): under the speaking line it is held back, and nothing is said ("held"). Returns
        (plan, beliefs, whose)."""
        from living.evidence import SPEAKING_LINE
        a_plan, a_e, _ = self.episodes.plan(self._heard(question))
        self.last_choice = None
        if a_plan and a_e >= SPEAKING_LINE:
            low = [self.session.lower(t) for t in a_plan]
            differ = bool(plan) and low != list(plan)
            if not plan or (differ and self.trust_analogy() >= 0.5):
                self.last_choice = ("analogy", differ)
                return low, [a_e] * len(low), "analogy"
            self.last_choice = ("zihin", differ)
            return plan, plan_e, "zihin"
        alone = self.trust_unforeseen(min(plan_e) if plan_e and self.alone_now else None)
        if plan and alone is not None and alone < SPEAKING_LINE:
            self.last_choice = ("held", False)
            return [], [], "held"
        return plan, plan_e, "zihin"

    def _heard_forms(self, question: str) -> None:
        """The written forms of a question's tokens, as heard (so that a word met first in it can be said back)."""
        for m in TOKEN.finditer(question):
            self.written[False][self.session.lower(m.group())][m.group()] += 1

    def reply(self, question: str, way: str = "cevahir", limit: int = 400) -> str:
        """Cevahir answers ``question`` (told); this lives Zihin's part of it (as Zihin's reply does). ``way``:
        "zihin" (Zihin's answer), "harmony" (the cortex alone, under the speaking line), "cevahir" (the prefrontal
        decision between Zihin's answer and the harmony, token by token). Returns the answer as said."""
        history = self.tail + self._lead() + question + "\t"
        if way == "harmony":
            return self.speak(history, (), (), "\t", most=16 * limit).strip()
        decided: List[Tuple[str, float]] = []
        real = self.mind.step

        def step(events):                                # Zihin's decisions, each with its belief e
            acts = real(events)
            if acts:
                expected = self.mind.expected(self.channel)
                for a in acts:
                    if a.channel == self.channel:
                        decided.append((a.symbol, float(expected.get(a.symbol, 0.0))))
            return acts

        self.mind.step = step
        try:
            said = self.session.reply(question, limit=limit)
        finally:
            del self.mind.step
        if way == "zihin":
            return said
        self._heard_forms(question)
        plan = [] if said == "(sessiz)" else self.session.words(said)
        es = [e for w, e in decided]
        plan_e = es if [w for w, _ in decided] == plan else []
        plan, plan_e, whose = self.choose_plan(question, plan, plan_e)
        if whose == "held":                                  # a thought held back: nothing is said
            return ""
        return self.speak(history, plan, plan_e, "\t", most=16 * limit).strip()

    def every_way(self, question: str, limit: int = 400) -> Dict[str, str]:
        """The answers of the three ways to one question (the harmony first: it changes nothing; then Zihin's
        reply, which this lives, and the one layer from it)."""
        out = {"harmony": self.reply(question, "harmony", limit)}
        history = self.tail + self._lead() + question + "\t"
        decided: List[Tuple[str, float]] = []
        real = self.mind.step

        def step(events):
            acts = real(events)
            if acts:
                expected = self.mind.expected(self.channel)
                for a in acts:
                    if a.channel == self.channel:
                        decided.append((a.symbol, float(expected.get(a.symbol, 0.0))))
            return acts

        self.mind.step = step
        try:
            said = self.session.reply(question, limit=limit)
        finally:
            del self.mind.step
        out["zihin"] = said
        self._heard_forms(question)
        plan = [] if said == "(sessiz)" else self.session.words(said)
        es = [e for w, e in decided]
        plan_e = es if [w for w, _ in decided] == plan else []
        from living.evidence import SPEAKING_LINE
        a_plan, a_e, _ = self.episodes.plan(self._heard(question))
        out["analogy"] = " ".join(a_plan) if a_plan and a_e >= SPEAKING_LINE else "(sessiz)"
        plan, plan_e, out["thought"] = self.choose_plan(question, plan, plan_e)
        out["cevahir"] = "" if out["thought"] == "held" else \
            self.speak(history, plan, plan_e, "\t", most=16 * limit).strip()
        return out

    def seen_answer(self, question: str, scene: dict, before: Optional[dict] = None) -> Optional[List[str]]:
        """The ground's answer (research/living_mind_2026_11_23/t_m2.py ``ground_answer``): the scene perceived in a
        moment, then the question understood and answered from what is perceived; None when it does not
        understand. ``before``: the scene in which the question is heard, before the event of ``scene`` begins."""
        from zihin.zemin import BEFORE, answer, understand
        z = getattr(self.mind, "zemin", None)
        if z is None or scene is None:
            return None
        z.pending, z.waiting, z.turn, z.event = [], None, None, None
        body = self.session.body(question)
        if before is not None:
            z.see(before)
            self.mind.step([])
            if understand(self.mind, z, body) is None:
                return None
        z.see(scene)
        self.mind.step([])
        got = answer(self.mind, z, body, when=BEFORE if before is not None else None)
        return list(got) if got else None

    def every_way_seen(self, question: str, scene: Optional[dict], before: Optional[dict] = None,
                       limit: int = 60) -> Dict[str, str]:
        """The three ways with a scene: Zihin's plan is the ground's answer where it understands, else its reply
        (t_m2.py "both"); the harmony alone; and the one layer from Zihin's plan."""
        out = {"harmony": self.reply(question, "harmony", limit)}
        history = self.tail + self._lead() + question + "\t"
        self._heard_forms(question)
        seen = self.seen_answer(question, scene, before) if scene is not None else None
        if seen is not None:
            out["zihin"] = " ".join(seen)
            plan, plan_e = seen, []
        else:
            grounded = getattr(self.mind.cfg, "grounded", False)
            self.mind.cfg.grounded = False
            said = self.session.reply(question, limit=limit)
            self.mind.cfg.grounded = grounded
            out["zihin"] = said
            plan, plan_e = ([] if said == "(sessiz)" else self.session.words(said)), []
        out["cevahir"] = self.speak(history, plan, plan_e, "\t", most=16 * 400).strip()
        return out

    def converse(self, said_to_it: str, limit: int = 80) -> str:
        """A turn of a conversation: Cevahir hears what is said to it and answers (the one layer), and lives both:
        Zihin lives them as its reply does (what it heard, told; what it said, its own act), the harmony lives the
        line (what it heard, a tab, what it answered)."""
        self.flush()
        entry = self._begin("converse")
        answer = self.reply(said_to_it, "cevahir", limit)
        self.pending = (self._lead() + said_to_it, "\t" + answer)
        self._pending_entry = entry
        if entry is not None:
            entry["t1"] = self.mind.t
            self._open = None
        return answer

    def flush(self) -> None:
        """The harmony lives the last turn as it was (nobody judged it)."""
        if self.pending is not None:
            heard, said = self.pending
            self.pending = None
            self.cortex.set_voice([], [], 0.0)
            self._live_judged(heard + said)

    def feedback(self, good: bool) -> None:
        """The amygdala (MODEL.md 12.4): how Cevahir's last answer was received. Zihin learns it as it always did
        (a value at the end of the episode, ``Session.feedback``); every token of the answer counts for the voice
        that led it, in its situation, in the record the prefrontal decision trusts by (``trust_zihin``,
        ``trust_alone``); where Zihin's plan and the analogy's differed, the outcome counts in the record that
        chose between them (``trust_analogy``): for the one chosen if right, for the other if wrong; and the harmony
        lives the turn with the answer only if it was right (a wrong answer is not kept as something said rightly:
        the question alone is lived)."""
        self.session.feedback(good)
        choice = getattr(self, "last_choice", None)
        if choice is not None and choice[1]:                 # the two thoughts differed: the outcome is evidence
            self.az_all += 1                                 # for the one chosen (right) or the other (wrong)
            self.az_won += 1 if (choice[0] == "analogy") == bool(good) else 0
        self.last_choice = None
        for who, cell in self.last_leads:
            self.outcome_all[(who, cell)] += 1
            self.outcome_good[(who, cell)] += 1 if good else 0
        self.last_leads = []
        if self.pending is not None:
            heard, said = self.pending
            self.pending = None
            self.cortex.set_voice([], [], 0.0)
            self._live_judged(heard + (said if good else "\t"))

    def _live_judged(self, text: str) -> None:
        """The harmony lives a conversation turn (into that turn's entry, with a ledger)."""
        entry = getattr(self, "_pending_entry", None)
        self._pending_entry = None
        if entry is not None:
            entry["text_from"] = self.cortex.size()["episode_text"]
            entry["history"] = self.tail[-4096:]
            self._open = entry
            entry["text"] = []
            self._live_text(text)
            entry["text"] = "".join(entry["text"])
            self._open = None
            self.entries.append(entry)
        else:
            self._live_text(text)

    def forget(self, k: int) -> dict:
        """Forget entry ``k`` of the history exactly (with a ledger): Zihin takes back every increment of its moments
        (``Mind.retract``, its ledger), the core takes back the counts its text added and overwrites its episode text
        (``FastCortex.retract``), and the forms it taught are taken back. What depends on the order of life (the
        harmony's gate and calibration, the situations, the records of disagreements) is not taken back. Exact in
        the core for an entry lived since the last sleep; after a night, ``core_exact`` is False: the contexts the
        sleep forgot are not there to take back, and a context made again by a later text loses the counts the
        entry's text gives it (never below zero). Its episode text is overwritten either way."""
        if self.entries is None:
            raise ValueError("Cevahir keeps no ledger (Cevahir(ledger=True))")
        self.flush()
        e = self.entries[k]
        if e["forgotten"]:
            return {"forgotten": False}
        exact = e.get("night", 0) == getattr(self, "nights", 0)  # a night since: contexts it made may be gone, and
        if e["text"]:                                            # made again by later texts (taken back as far as
            self.cortex.retract(e["text"], e["history"], e["text_from"])   # they stand, never below zero)
        moments = list(range(e["t0"], e.get("t1", e["t0"] - 1) + 1))
        for m in reversed(moments):
            self.mind.retract(m)
        if "episode" in e:
            self.episodes.forget(e["episode"])
        for where, gap, answer, token, raw in e["forms"]:
            self.gaps[where][gap] -= 1
            if self.gaps[where][gap] <= 0:
                del self.gaps[where][gap]
            self.written[answer][token][raw] -= 1
            if self.written[answer][token][raw] <= 0:
                del self.written[answer][token][raw]
        e["forgotten"] = True
        return {"forgotten": True, "moments": len(moments), "characters": len(e["text"]), "core_exact": exact}

    def keep_ledger(self, folder: Optional[str] = None) -> str:
        """Keep Zihin's ledger, the experience log that makes forgetting exact, on disk from now on: one file in
        ``folder`` (by default a temporary folder, removed when the process exits), the latest moments in memory
        (``zihin.store.DiskLog``). The ledger is read only to forget, so it need not be in working memory; kept in
        memory it held about five kilobytes a character of natural text, most of what Cevahir held (NOTES.md 8.28).
        What the ledger already holds is moved there, and the history of entries begins if it had not. Returns the
        file's path."""
        from zihin.__main__ import _log_path
        from zihin.store import DiskLog
        S, cfg = self.mind.S, self.mind.cfg
        path = _log_path(folder)
        log = DiskLog(path, keep=max(4, cfg.undo_depth + 2))
        for m in sorted(S.log):
            e = S.log[m]
            log[m] = e
            if getattr(e, "retracted", False):
                log.mark_retracted(m)
        S.log, S.keep_log = log, True
        cfg.keep_log, cfg.log_path = True, path
        if self.entries is None:
            self.entries = []
        return path

    # ------------------------------------------------------------------------------------------------- channels ---
    def newness(self, segment: str) -> Dict[str, float]:
        """For every channel: the belief that ``segment`` is new to it, the rule on its symbols outside the
        channel's alphabet (living/mind.py)."""
        n = len(segment)
        return {name: (sum(1 for x in segment if x not in a) + 0.5) / (n + 1.0) for name, a in self.alphabets.items()}

    def perceive(self, segment: str, channel: Optional[str] = None) -> str:
        """A segment of a stream (MODEL.md 4): on the channel it is told, or on the one the rule decides. A segment
        new to every channel (each belief at the speaking line) is the first of a new channel, born with a harmony
        of its own and an order-0 store beside it (for the structure belief); otherwise it goes to the channel it is
        least new to. The main channel is the whole of Cevahir (``read``: Zihin and the harmony with Zihin's
        voice); a channel born so is lived by its harmony. Returns the channel's name."""
        from living.evidence import SPEAKING_LINE
        if channel is None:
            new = self.newness(segment)
            if min(new.values()) >= SPEAKING_LINE:
                channel = "kanal-%d" % (len(self.others) + 1)
            else:
                channel = min(new, key=lambda k: (new[k], k != self.channel))
        if channel == self.channel:
            self.read(segment)
            return channel
        if channel not in self.others:
            cfg = dict(CORTEX)
            self.others[channel] = {"cortex": FastCortex(**cfg), "plain": FastCortex(depth=0), "won": 0, "seen": 0}
            self.alphabets[channel] = set()
        ch = self.others[channel]
        _, ctx = ch["cortex"].stream(segment, bits=True)
        _, plain = ch["plain"].stream(segment, bits=True)
        ch["won"] += int((ctx < plain).sum())
        ch["seen"] += len(segment)
        self.alphabets[channel].update(segment)
        return channel

    def structure(self, channel: str) -> float:
        """The belief that a channel's context predicts: the rule on how many of its symbols the harmony
        predicted better than the order-0 store living the same stream (noise never passes the speaking line)."""
        ch = self.others.get(channel)
        if ch is None:
            return 0.5
        return (ch["won"] + 0.5) / (ch["seen"] + 1.0)

    def say_on(self, channel: str, prompt: str, most: int = 40) -> str:
        """What Cevahir says on a channel born by the rule: nothing while the channel has shown no structure (it
        watches); then its harmony's guesses, as long as each reaches the speaking line."""
        from living.evidence import SPEAKING_LINE
        if channel == self.channel:
            return self.reply(prompt)
        if self.structure(channel) < SPEAKING_LINE:
            return ""
        text, _ = self.others[channel]["cortex"].say(prompt, most=most, line=SPEAKING_LINE)
        return text

    # ---------------------------------------------------------------------------------------------------- sleep ---
    def sleep(self, keep_min: int = 2, dream: bool = True, cycles: int = 3) -> Dict[str, object]:
        """The hippocampus's night (MODEL.md 8): the cortex of every channel forgets the contexts met fewer than
        ``keep_min`` times (what recurred stays exactly), and Zihin dreams and sorts its dreams against what is real
        (``zihin.dusunce.sleep``, F20–F22) if ``dream``; either way Zihin keeps the day as its night does
        (``Mind.consolidate``: its caches dropped, nothing it knows changed). Returns what was forgotten and dreamt."""
        self.flush()
        self.nights = getattr(self, "nights", 0) + 1
        out = {"forgotten": {self.channel: self.cortex.sleep(keep_min)}}
        for name, ch in self.others.items():
            out["forgotten"][name] = ch["cortex"].sleep(keep_min)
        if dream:
            from zihin.dusunce import sleep as night
            got = night(self.session, cycles)
            out["dreams"] = len(got.get("dreams", []))
        else:
            self.mind.consolidate()
        return out

    # ------------------------------------------------------------------------------------------------- keeping -----
    def save(self, folder: str, compress: bool = False) -> Dict[str, object]:
        """Keep Cevahir in ``folder``: Zihin (``zihin.persist.save``, with its manifest and digests), the core
        (``core.bin``) and what Cevahir itself learned (the forms lexicon, the records, the stream's tail:
        ``cevahir.json``), and a manifest with the SHA-256 of every file (``manifest.json``). Pickle executes code
        when loaded (Zihin's file): load only minds you saved yourself."""
        import hashlib
        import json
        from zihin import persist
        self.flush()
        os.makedirs(folder, exist_ok=True)
        persist.save(self.mind, os.path.join(folder, "zihin.zihin"))
        self.cortex.save(os.path.join(folder, "core.bin"))
        for i, (name, ch) in enumerate(sorted(self.others.items())):
            ch["cortex"].save(os.path.join(folder, "channel-%d.bin" % i))
            ch["plain"].save(os.path.join(folder, "channel-%d-plain.bin" % i))
        state = {
            "version": 1, "channel": self.channel, "dil": self.session.dil, "top": self.top,
            "forms_per": self.forms_per, "record": self.record, "tail": self.tail,
            "written": {str(int(k)): {t: dict(c) for t, c in v.items()} for k, v in self.written.items()},
            "gaps": [[int(a), int(f), dict(c)] for (a, f), c in self.gaps.items()],
            "dis_won": dict(self.dis_won), "dis_all": dict(self.dis_all),
            "alone_right": dict(self.alone_right), "alone_all": dict(self.alone_all), "counts": dict(self.counts),
            "outcome_good": [[w, c, n] for (w, c), n in self.outcome_good.items()],
            "outcome_all": [[w, c, n] for (w, c), n in self.outcome_all.items()],
            "alphabets": {k: "".join(sorted(v)) for k, v in self.alphabets.items()},
            "episodes": self.episodes.state(), "marks": self.marks_kept, "az": [self.az_won, self.az_all],
            "za": [self.za_won, self.za_all], "za_claims": list(self.za_claims), "alone_now": self.alone_now,
            "others": [[name, ch["won"], ch["seen"]] for name, ch in sorted(self.others.items())]}
        with open(os.path.join(folder, "cevahir.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        files = {}
        for name in sorted(os.listdir(folder)):
            if name == "manifest.json":
                continue
            h = hashlib.sha256()
            with open(os.path.join(folder, name), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            files[name] = h.hexdigest()
        manifest = {"what": "Cevahir (living/cevahir.py)", "files": files, "zihin_moments": self.mind.t,
                    "core": self.cortex.size(), "compressed": []}
        if compress:                                     # the stores as .gz (their digests are of what they hold)
            import gzip
            import shutil
            for name in list(files):
                if name.endswith((".bin", ".zihin")):
                    src = os.path.join(folder, name)
                    with open(src, "rb") as fi, gzip.open(src + ".gz", "wb", compresslevel=9) as fo:
                        shutil.copyfileobj(fi, fo, 1 << 20)
                    os.remove(src)
                    manifest["compressed"].append(name)
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        return manifest

    @classmethod
    def load(cls, folder: str) -> "Cevahir":
        """Cevahir read back from ``folder`` after every file is checked against the manifest."""
        import hashlib
        import json
        from zihin import persist
        from zihin.__main__ import Session
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        for name in manifest.get("compressed", []):     # a compressed store is unpacked beside it, once
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                import gzip
                import shutil
                with gzip.open(path + ".gz", "rb") as fi, open(path + ".tmp", "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
                os.replace(path + ".tmp", path)
        for name, digest in manifest["files"].items():
            h = hashlib.sha256()
            with open(os.path.join(folder, name), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise IOError("%s does not match its manifest" % name)
        with open(os.path.join(folder, "cevahir.json"), encoding="utf-8") as f:
            state = json.load(f)
        self = cls.__new__(cls)
        self.mind = persist.load(os.path.join(folder, "zihin.zihin"), ledger=False)
        self.session = Session(self.mind)
        self.session.dil = state["dil"]
        self.channel = state["channel"]
        self.cortex = FastCortex.load(os.path.join(folder, "core.bin"))
        self.top, self.forms_per, self.record = state["top"], state["forms_per"], state["record"]
        self.tail = state["tail"]
        self.stream = None
        self.written = {False: defaultdict(Counter), True: defaultdict(Counter)}
        for k, v in state["written"].items():
            for t, c in v.items():
                self.written[bool(int(k))][t] = Counter(c)
        self.gaps = defaultdict(Counter)
        for a, fst, c in state["gaps"]:
            self.gaps[(bool(a), bool(fst))] = Counter(c)
        self.dis_won = Counter({int(k): v for k, v in state["dis_won"].items()})
        self.dis_all = Counter({int(k): v for k, v in state["dis_all"].items()})
        self.alone_right = Counter({int(k): v for k, v in state["alone_right"].items()})
        self.alone_all = Counter({int(k): v for k, v in state["alone_all"].items()})
        self.counts = Counter(state["counts"])
        self.outcome_good = Counter({(w, c): n for w, c, n in state.get("outcome_good", [])})
        self.outcome_all = Counter({(w, c): n for w, c, n in state.get("outcome_all", [])})
        self.last_leads, self.pending = [], None
        self.entries, self._open, self._pending_entry = None, None, None
        self.episodes = Episodes.from_state(state.get("episodes", {}))
        self.marks_kept = state.get("marks", False)
        self.az_won, self.az_all = state.get("az", [0, 0])
        self.za_won, self.za_all = state.get("za", [0, 0])
        self.za_claims = [tuple(c) for c in state.get("za_claims", [])]
        self.alone_now = state.get("alone_now", False)
        self.alphabets = {k: set(v) for k, v in state.get("alphabets", {self.channel: ""}).items()}
        self.others = {}
        for i, (name, won, seen) in enumerate(state.get("others", [])):
            self.others[name] = {"cortex": FastCortex.load(os.path.join(folder, "channel-%d.bin" % i)),
                                 "plain": FastCortex.load(os.path.join(folder, "channel-%d-plain.bin" % i)),
                                 "won": won, "seen": seen}
        return self

    def answers(self, questions: List[str], way: str = "cevahir", limit: int = 400,
                workers: Optional[int] = None) -> List[str]:
        """What Cevahir answers to each question as it stands now: each in a process that is this mind, which
        answers and ends; nothing of the answers is lived by this mind (Zihin's ``paralel.answers``)."""
        global _ASKED
        if not questions:
            return []
        k = min(workers or os.cpu_count() or 1, len(questions))
        import multiprocessing
        self.episodes.ready()                                    # the kinds and frames, read once for every process
        _ASKED = (self, way, limit)
        try:
            with multiprocessing.get_context("fork").Pool(k, maxtasksperchild=1) as pool:
                return pool.map(_pool_answer, questions, chunksize=1)
        finally:
            _ASKED = None


_ASKED = None


def _pool_answer(question: str) -> str:
    import torch
    torch.set_num_threads(1)
    me, way, limit = _ASKED
    me.mind.S.log = {}                                       # nothing reaches the ledger of the mind that asked
    if way == "every":
        return me.every_way(question, limit)
    if way == "seen":                                        # (question, scene, before)
        q, scene, before = question
        with torch.no_grad():
            return me.every_way_seen(q, scene, before)
    return me.reply(question, way, limit)
