# -*- coding: utf-8 -*-
"""Living with a Zihin mind from the terminal.

    python -m zihin [--load PATH] [--variant members]

Every plain line is something you TELL the mind (provenance "told"), one word per moment, then a silent moment.
Commands (Turkish, with English aliases):
    !gordum <metin>      / !saw     the same, as something the mind SENSES itself (provenance "sensed")
    !izle <soz> || <karsilik> / !watch  the mind WATCHES two others talk (provenance "imitated"): how one answers
    !ogret <dosya>       / !teach   watch every line of a file: "soz<TAB>karsilik" dialogues or plain sentences
    !oku <dosya|klasor>  / !read    watch every text in a file or folder (.json dialogues, .txt .md .tsv; .docx with python-docx,
                                    .pdf with pypdf): sentences of documents as speech of others (imitated)
    !konus               / !chat    chat mode on/off: after each line you write, the mind may answer
    !dogru / !yanlis     / !good / !bad   right after its answer: how its answer was received (a value signal;
                                    the path that led to it is remembered and weighed next time, L5)
Turn taking (while you speak the mind listens, then it may speak until it falls silent) is set by the session, not
learned yet: during your words the act threshold is raised, afterwards it is the mind's own.
    ?<metin>             / ?        what does the mind expect after this beginning (nothing is lived; a clone is asked)
    !neden <metin> => <kelime> / !why   white-box belief: own and effective evidence, per source
    !gecmis              / !history what you have told, with numbers
    !unut <no>           / !forget  retract that utterance exactly (its evidence is removed from the core)
    !boyut               / !size    units, ties, bytes, moments
    !yas                 / !age     developmental profile: moments and nights lived, words known, concepts, phrases
    !kaydet <yol>        / !save    save the mind (with integrity manifest)
    !yukle <yol>         / !load    load a saved mind
    !tasinabilir <yol>   / !portable  save a copy for conversation without the experience ledger (small file)
    !cik                 / !quit
The text adapter splits words and punctuation; the alphabet is open (a new word is a new symbol at any time). The
mind never sees text, only symbols on the "text" channel; meaning is in the ties.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List

from . import IMITATED, SENSED, TOLD, Config, Event, Mind, read
from .events import PROVENANCE
from .persist import export_portable, load, save
from .store import W_NEG, W_POS

CHANNEL = "text"
def _log_path() -> str:
    import os
    import time
    os.makedirs("zihin_logs", exist_ok=True)
    return os.path.join("zihin_logs", "defter-%d.log" % int(time.time() * 1000))


VARIANTS = {"speaking": lambda: Config.changed(support="members", bind_repeats=True, next_only_present=True,
                                               neg_floor=1.0 / 14, backoff=True, log_path=_log_path()),
            "members": lambda: Config.changed(support="members"), "changed": Config.changed}


def lower_tr(text: str) -> str:
    """Turkish lower case (I -> ı, İ -> i) so that a word at the start of a sentence is the same symbol."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def words(text: str) -> List[str]:
    """Text channel adapter: words and punctuation as symbols, in Turkish lower case."""
    return re.findall(r"\w+|[^\w\s]", lower_tr(text), flags=re.UNICODE)


ASK_KEYS = ("soru", "question", "input", "prompt", "söz", "soz")
ANSWER_KEYS = ("cevap", "answer", "output", "response", "karşılık", "karsilik")


def dialogues_of(data) -> List[tuple]:
    """(utterance, answer) pairs from JSON: a list of objects with a question-like and an answer-like field."""
    pairs = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        keys = {k.lower(): v for k, v in item.items() if isinstance(v, str)}
        ask = next((keys[k] for k in ASK_KEYS if k in keys), None)
        answer = next((keys[k] for k in ANSWER_KEYS if k in keys), None)
        if ask and answer:
            pairs.append((ask, answer))
    return pairs


def sentences(text: str) -> List[str]:
    """Split a document into sentences (and lines), keeping only those with at least two words."""
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [p.strip() for p in parts if len(words(p)) >= 2]


def read_document(path: str) -> str:
    """Plain text of a document. .docx needs python-docx, .pdf needs pypdf (only if such files are present)."""
    low = path.lower()
    if low.endswith((".txt", ".md", ".tsv", ".csv")):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    if low.endswith(".docx"):
        import docx  # python-docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    if low.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return ""


def profile(mind: Mind) -> str:
    """Developmental profile measured from the substance (nothing assigned)."""
    from .evidence import conviction
    from .store import ATOMIC, CATEGORY, COMPOSED, EXPECT, SLOTTED
    S = mind.S
    n = S.n_units
    kinds = S.u_kind[:n]
    cid = mind.ch.id(CHANNEL) if mind.ch.has(CHANNEL) else -1
    words_seen = known = 0
    for unit in range(n):
        if not int(kinds[unit]) & ATOMIC or int(S.u_channel[unit]) != cid:
            continue
        words_seen += 1
        ties = S.out.get((unit, EXPECT, 1), [])
        if ties:
            mass = float(max((S.t_w[0, t] * mind.trust).sum() + (S.t_w[1, t] * mind.trust).sum() for t in ties))
            known += conviction(mass, mind.cfg.k) >= 0.5
    phrases = int(((kinds & COMPOSED).bool() & ~(kinds & SLOTTED).bool()).sum())
    lines = ["  yaşanmış an: %d   gece (uyku döngüsü): %d" % (mind.t, getattr(mind, "nights", 0)),
             "  karşılaştığı kelime: %d   kanaat kazandığı kelime (c >= 0.5): %d" % (words_seen, known),
             "  kurduğu kavram (kategori): %d   çerçeve: %d   birleşik birim (dizi): %d" % (
                 int((kinds & CATEGORY).bool().sum()), int((kinds & SLOTTED).bool().sum()), phrases),
             "  bağ: %d   öz: %.1f MB" % (S.n_ties, mind.size()["core_tensors"] / 1e6),
             "  (insan yaşı karşılığı verilmez: kaynaklı bir gelişim normu olmadan sayı uydurulmaz)"]
    return "\n".join(lines)


FACETS = "parca"
TURN, TURN_END = "sira", "⏎"          # the end of someone's utterance (Enter) is perceived, like a pause in speech
VALUE = "deger"
VALUE_SIGN = {(VALUE, "iyi"): 1.0, (VALUE, "kötü"): -1.0}     # the adapter defines what is good and bad


class Session:
    def __init__(self, mind: Mind, pieces=None) -> None:
        self.mind = mind
        self.history: List[tuple] = []        # (number, text, moments)
        self.pieces = pieces                  # optional subword tokenizer: pieces of each word as facets
        if pieces is not None:
            mind.add_channel(FACETS, "facet")
        mind.add_channel(VALUE, "value")
        mind.value_sign = dict(VALUE_SIGN)
        known = mind.ch.has(TURN) and mind.ch.symbol_index(mind.ch.id(TURN), TURN_END, create=False) >= 0
        self.turns = mind.t == 0 or known       # a mind that never lived turn ends keeps the older turn passing
        mind.add_channel(TURN, "sense")
        self.open_turn = False                # the mind has just answered; feedback may still come

    def events(self, symbol: str, prov: int) -> List[Event]:
        """One word as an event on the text channel, with its subword pieces as facets of the same moment."""
        out = [Event(CHANNEL, symbol, prov)]
        if self.pieces is not None and symbol.isalpha():
            parts = [p for p in self.pieces.tokenize(symbol) if p.strip()]
            if len(parts) > 1:
                out += [Event(FACETS, p.strip(), prov) for p in parts]
        return out

    def live(self, text: str, prov: int) -> None:
        """Hear (told) or sense an utterance while listening: the mind does not speak into it."""
        start = self.mind.t + 1
        self.listen(text, prov)
        self.end_of_turn(prov)
        self.mind.step([])
        self.history.append((len(self.history) + 1, text, list(range(start, self.mind.t + 1))))

    def listen(self, text: str, prov: int) -> None:
        """Hear an utterance while not speaking (turn taking set by the session)."""
        theta = self.mind.knob["theta"]
        self.mind.knob["theta"] = 1e9
        try:
            for symbol in words(text):
                self.mind.step(self.events(symbol, prov))
        finally:
            self.mind.knob["theta"] = theta

    def end_of_turn(self, prov: int, listening: bool = True) -> List[Event]:
        """Someone's utterance has ended: a perceived event. While listening the mind does not speak into it."""
        if not self.turns:
            return []
        theta = self.mind.knob["theta"]
        if listening:
            self.mind.knob["theta"] = 1e9
        try:
            return self.mind.step([Event(TURN, TURN_END, prov)])
        finally:
            self.mind.knob["theta"] = theta

    def watch(self, first: str, answer: str) -> None:
        """Two others talk: one says ``first``, another answers ``answer``; the mind only watches. The end of each
        utterance is perceived, so the mind can learn who speaks when (turn taking is watched, not coded)."""
        self.listen(first, IMITATED)
        self.end_of_turn(IMITATED)
        self.listen(answer, IMITATED)
        self.end_of_turn(IMITATED)
        self.mind.step([])
        self.history.append((len(self.history) + 1, "[izledi] %s || %s" % (first, answer), []))

    def reply(self, text: str, limit: int = 12) -> str:
        """You speak (told), then the mind may speak: its own acts, one symbol per moment, until silence.

        The turn passes to the mind with your last word: it decides in that very moment (the context of your
        whole utterance is present), and each symbol it says comes back as its own echo in the next moment.
        """
        self.close_turn()
        start = self.mind.t + 1
        symbols = words(text)
        if not symbols:
            return "(sessiz)"
        if self.turns:                                   # the turn passes when your utterance has ended
            self.listen(" ".join(symbols), TOLD)
            acts = self.end_of_turn(TOLD, listening=False)
        else:                                            # older minds: the turn passes with your last word
            self.listen(" ".join(symbols[:-1]), TOLD)
            acts = self.mind.step(self.events(symbols[-1], TOLD))
        said: List[str] = []
        while acts and len(said) < limit:
            said += [str(e.symbol) for e in acts if e.channel == CHANNEL]
            acts = self.mind.step([])                    # the echo of its own act arrives here
        self.history.append((len(self.history) + 1, text, list(range(start, self.mind.t + 1))))
        self.open_turn = True                            # the turn stays open for "!dogru" / "!yanlis"
        return " ".join(said) if said else "(sessiz)"

    def close_turn(self) -> None:
        if self.open_turn:
            self.mind.step([])
            self.open_turn = False

    def feedback(self, good: bool) -> str:
        """How the mind's last answer was received: a value event at the end of the same episode."""
        if not self.open_turn:
            return "  geri bildirim, zihnin cevabından hemen sonra verilmeli"
        self.mind.step([Event(VALUE, "iyi" if good else "kötü", SENSED)])
        self.mind.step([])
        self.open_turn = False
        return "  not edildi (%s)" % ("iyi" if good else "kötü")

    def read_path(self, path: str) -> str:
        """Watch the texts of a file or of every supported file under a folder; report what was lived."""
        import os
        import time
        files = [path] if os.path.isfile(path) else sorted(
            os.path.join(root, name) for root, _, names in os.walk(path) for name in names)
        start, lived, skipped = time.time(), 0, []
        for name in files:
            try:
                if name.lower().endswith(".json"):
                    import json
                    with open(name, encoding="utf-8") as f:
                        pairs = dialogues_of(json.load(f))
                    for first, answer in pairs:
                        self.watch(first, answer)
                        lived += 1
                    if not pairs:
                        skipped.append(os.path.basename(name))
                    continue
                if name.lower().endswith(".tsv"):
                    with open(name, encoding="utf-8", errors="replace") as f:
                        for raw in f:
                            if "\t" in raw:
                                first, answer = raw.rstrip("\n").split("\t", 1)
                                self.watch(first, answer)
                                lived += 1
                    continue
                text = read_document(name)
            except Exception as exc:
                skipped.append("%s (%s)" % (os.path.basename(name), exc))
                continue
            if not text:
                skipped.append(os.path.basename(name))
                continue
            for sentence in sentences(text):
                self.listen(sentence, IMITATED)
                self.mind.step([])
                lived += 1
            self.history.append((len(self.history) + 1, "[okudu] %s" % os.path.basename(name), []))
        took = time.time() - start
        out = "  %d dosya, %d cümle/diyalog, %d an, %.0f sn (%.1f ms/an)" % (
            len(files) - len(skipped), lived, self.mind.t, took, 1000 * took / max(1, self.mind.t))
        if skipped:
            out += "\n  okunamayan: " + ", ".join(skipped[:10])
        return out

    def ask(self, text: str, top: int = 5) -> str:
        twin = self.mind.clone()
        twin.knob["theta"] = 1e9                         # listening: the clone does not speak into the question
        for symbol in words(text):
            twin.step(self.events(symbol, TOLD))
        expected = sorted(twin.expected(CHANNEL, "specific").items(), key=lambda kv: -kv[1])[:top]
        drive = read.total_drive(twin)
        if not expected or expected[0][1] <= 0.5:
            head = "  (kanaat yok: bu başlangıçtan sonra ne geleceğine dair güçlü bir beklenti oluşmamış; dürtü %.2f)" % drive
        else:
            head = "  beklentiler (e = eyleme esas beklenti):"
        lines = [head] + ["    %-20s e %.3f" % (s, e) for s, e in expected]
        if self.pieces is not None:
            facets = sorted(twin.expected(FACETS, "specific").items(), key=lambda kv: -kv[1])[:top]
            if facets:
                lines.append("  beklenen parçalar (bir sonraki kelimenin kök/ekleri):")
                lines += ["    %-20s e %.3f" % (s, e) for s, e in facets]
        return "\n".join(lines)

    def why(self, text: str, value: str) -> str:
        twin = self.mind.clone()
        twin.knob["theta"] = 1e9                         # listening: the clone does not speak into the question
        for symbol in words(text):
            twin.step(self.events(symbol, TOLD))
        if twin.context is None or not twin.ch.has(CHANNEL):
            return "  bağlam yok"
        cid = twin.ch.id(CHANNEL)
        index = twin.ch.symbol_index(cid, value, create=False)
        target = twin.S.atom_index.get((cid, index), -1) if index >= 0 else -1
        if target < 0:
            return "  '%s' zihin için yeni bir sembol: hiç yaşanmamış" % value
        b = read.belief(twin, twin.context, target)
        out = ["  kendi kanıtı: f %.2f c %.2f e %.2f | etkin (genellemelerle): e %.2f" %
               (b["own_f"], b["own_c"], b["own_e"], b["eff_e"])]
        tie = twin.S.find_tie(twin.context, target, 0, 1)
        if tie >= 0:
            pos, neg = twin.S.t_w[W_POS, tie].tolist(), twin.S.t_w[W_NEG, tie].tolist()
            out.append("  kaynaklara göre (w+ / w-): " + ", ".join(
                "%s %.2f/%.2f" % (name, p, n) for name, p, n in zip(PROVENANCE, pos, neg) if p or n))
            out.append("  kaynak güveni: " + ", ".join("%s %.2f" % (n, float(t)) for n, t in zip(PROVENANCE, twin.trust)))
        return "\n".join(out)

    def forget(self, number: int) -> str:
        for entry in self.history:
            if entry[0] == number:
                removed = 0
                for m in reversed(entry[2]):
                    report = self.mind.retract(m)
                    removed += sum(v for v in report.values() if isinstance(v, int))
                self.history.remove(entry)
                return "  '%s' geri alındı (kanıtı çekirdekten çıkarıldı; %d kayıt)" % (entry[1], removed)
        return "  böyle bir numara yok"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m zihin")
    ap.add_argument("--load")
    ap.add_argument("--parca", help="subword tokenizer (ByteLevelBPE json): word pieces enter as facets")
    ap.add_argument("--defter-yok", dest="no_ledger", action="store_true",
                    help="no experience ledger: smaller and faster, but !unut cannot retract")
    ap.add_argument("--variant", default="speaking", choices=sorted(VARIANTS))
    args = ap.parse_args(argv)
    if args.load:
        mind = load(args.load, ledger=not args.no_ledger)
        if not args.no_ledger and not mind.S.keep_log:      # a mind saved without a ledger: keep one from now on
            from .store import DiskLog
            mind.cfg.keep_log, mind.cfg.log_path = True, _log_path()
            mind.S.keep_log = True
            mind.S.log = DiskLog(mind.cfg.log_path, keep=max(4, mind.cfg.undo_depth + 2))
    else:
        mind = Mind(VARIANTS[args.variant]())
        if args.no_ledger:
            mind.S.log, mind.S.keep_log, mind.cfg.keep_log, mind.cfg.log_path = {}, False, False, None
        mind.add_channel(CHANNEL, "act")
    pieces = None
    if args.parca:
        sys.path.insert(0, ".")
        from tokenizer_management.byte_bpe import ByteLevelBPE
        pieces = ByteLevelBPE.load(args.parca)
    session = Session(mind, pieces)
    chat = False
    print("Zihin — anlat, sor (?...), !neden, !unut, !kaydet, !cik. %d an yaşanmış." % mind.t)
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        line = line.strip()
        if not line:
            continue
        try:
            if not line.lstrip("!").split(" ")[0] in ("dogru", "doğru", "good", "iyi", "yanlis", "yanlış", "bad", "kotu",
                                                       "kötü") or not line.startswith("!"):
                session.close_turn()                     # any other input ends the answered turn first
            if line.startswith("?") and not chat:
                print(session.ask(line[1:].strip()))
            elif line.startswith("!"):
                cmd, _, rest = line[1:].partition(" ")
                if cmd in ("cik", "quit"):
                    break
                elif cmd in ("gordum", "saw"):
                    session.live(rest, SENSED)
                elif cmd in ("izle", "watch"):
                    first, _, answer = rest.partition("||")
                    session.watch(first.strip(), answer.strip())
                elif cmd in ("ogret", "teach"):
                    n = 0
                    with open(rest.strip(), encoding="utf-8") as f:
                        for raw in f:
                            raw = raw.rstrip("\n")
                            if not raw.strip():
                                continue
                            if "\t" in raw:
                                first, answer = raw.split("\t", 1)
                                session.watch(first, answer)
                            else:
                                session.live(raw, IMITATED)
                            n += 1
                    print("  %d satır izlendi, %d an" % (n, session.mind.t))
                elif cmd in ("oku", "read"):
                    print(session.read_path(rest.strip()))
                elif cmd in ("dogru", "doğru", "good", "iyi"):
                    print(session.feedback(True))
                elif cmd in ("yanlis", "yanlış", "bad", "kotu", "kötü"):
                    print(session.feedback(False))
                elif cmd in ("konus", "chat"):
                    chat = not chat
                    print("  sohbet modu %s" % ("açık" if chat else "kapalı"))
                elif cmd in ("neden", "why"):
                    prefix, _, value = rest.partition("=>")
                    print(session.why(prefix.strip(), value.strip()))
                elif cmd in ("gecmis", "history"):
                    for number, text, _ in session.history:
                        print("  %3d  %s" % (number, text))
                elif cmd in ("unut", "forget"):
                    print(session.forget(int(rest)))
                elif cmd in ("yas", "yaş", "age"):
                    print(profile(session.mind))
                elif cmd in ("boyut", "size"):
                    print("  %d an, %s" % (session.mind.t, session.mind.size()))
                elif cmd in ("kaydet", "save"):
                    print("  kaydedildi: %s" % save(session.mind, rest.strip()))
                elif cmd in ("tasinabilir", "portable"):
                    print("  taşınabilir kopya: %s" % export_portable(session.mind, rest.strip()))
                elif cmd in ("yukle", "load"):
                    session = Session(load(rest.strip()), session.pieces)
                    print("  yüklendi: %d an" % session.mind.t)
                else:
                    print("  bilinmeyen komut")
            elif chat:
                print("  zihin: %s" % session.reply(line))
            else:
                session.live(line, TOLD)
        except Exception as exc:  # keep the session alive
            print("  hata: %s" % exc)


if __name__ == "__main__":
    main()
