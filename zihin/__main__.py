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
                                               log_path=_log_path()),
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


class Session:
    def __init__(self, mind: Mind) -> None:
        self.mind = mind
        self.history: List[tuple] = []        # (number, text, moments)

    def live(self, text: str, prov: int) -> None:
        moments = []
        for symbol in words(text):
            self.mind.step([Event(CHANNEL, symbol, prov)])
            moments.append(self.mind.t)
        self.mind.step([])
        self.history.append((len(self.history) + 1, text, moments))

    def listen(self, text: str, prov: int) -> None:
        """Hear an utterance while not speaking (turn taking set by the session)."""
        theta = self.mind.knob["theta"]
        self.mind.knob["theta"] = 1e9
        try:
            for symbol in words(text):
                self.mind.step([Event(CHANNEL, symbol, prov)])
        finally:
            self.mind.knob["theta"] = theta

    def watch(self, first: str, answer: str) -> None:
        """Two others talk: one says ``first``, another answers ``answer``; the mind only watches."""
        self.listen(first, IMITATED)
        self.listen(answer, IMITATED)
        self.mind.step([])
        self.history.append((len(self.history) + 1, "[izledi] %s || %s" % (first, answer), []))

    def reply(self, text: str, limit: int = 12) -> str:
        """You speak (told), then the mind may speak: its own acts, one symbol per moment, until silence.

        The turn passes to the mind with your last word: it decides in that very moment (the context of your
        whole utterance is present), and each symbol it says comes back as its own echo in the next moment.
        """
        start = self.mind.t + 1
        symbols = words(text)
        if not symbols:
            return "(sessiz)"
        self.listen(" ".join(symbols[:-1]), TOLD)
        acts = self.mind.step([Event(CHANNEL, symbols[-1], TOLD)])
        said: List[str] = []
        while acts and len(said) < limit:
            said += [str(e.symbol) for e in acts if e.channel == CHANNEL]
            acts = self.mind.step([])                    # the echo of its own act arrives here
        self.mind.step([])
        self.history.append((len(self.history) + 1, text, list(range(start, self.mind.t + 1))))
        return " ".join(said) if said else "(sessiz)"

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
        for symbol in words(text):
            twin.step([Event(CHANNEL, symbol, TOLD)])
        expected = sorted(twin.expected(CHANNEL, "specific").items(), key=lambda kv: -kv[1])[:top]
        drive = read.total_drive(twin)
        if not expected or expected[0][1] <= 0.5:
            head = "  (kanaat yok: bu başlangıçtan sonra ne geleceğine dair güçlü bir beklenti oluşmamış; dürtü %.2f)" % drive
        else:
            head = "  beklentiler (e = eyleme esas beklenti):"
        lines = [head] + ["    %-20s e %.3f" % (s, e) for s, e in expected]
        return "\n".join(lines)

    def why(self, text: str, value: str) -> str:
        twin = self.mind.clone()
        for symbol in words(text):
            twin.step([Event(CHANNEL, symbol, TOLD)])
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
    ap.add_argument("--variant", default="speaking", choices=sorted(VARIANTS))
    args = ap.parse_args(argv)
    if args.load:
        mind = load(args.load)
    else:
        mind = Mind(VARIANTS[args.variant]())
        mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    chat = False
    print("Zihin — anlat, sor (?...), !neden, !unut, !kaydet, !cik. %d an yaşanmış." % mind.t)
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        line = line.strip()
        if not line:
            continue
        try:
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
                    session = Session(load(rest.strip()))
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
