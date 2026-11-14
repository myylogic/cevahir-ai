# -*- coding: utf-8 -*-
"""Living with a Zihin mind from the terminal.

    python -m zihin [--load PATH] [--variant members]

Every plain line is something you TELL the mind (provenance "told"), one word per moment, then a silent moment.
Commands (Turkish, with English aliases):
    !gordum <metin>      / !saw     the same, as something the mind SENSES itself (provenance "sensed")
    !izle <soz> || <karsilik> / !watch  the mind WATCHES two others talk (provenance "imitated"): how one answers
    !ogret <dosya>       / !teach   watch every line of a file: "soz<TAB>karsilik" dialogues or plain sentences
    !konus               / !chat    chat mode on/off: after each line you write, the mind may answer
Turn taking (while you speak the mind listens, then it may speak until it falls silent) is set by the session, not
learned yet: during your words the act threshold is raised, afterwards it is the mind's own.
    ?<metin>             / ?        what does the mind expect after this beginning (nothing is lived; a clone is asked)
    !neden <metin> => <kelime> / !why   white-box belief: own and effective evidence, per source
    !gecmis              / !history what you have told, with numbers
    !unut <no>           / !forget  retract that utterance exactly (its evidence is removed from the core)
    !boyut               / !size    units, ties, bytes, moments
    !kaydet <yol>        / !save    save the mind (with integrity manifest)
    !yukle <yol>         / !load    load a saved mind
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
from .persist import load, save
from .store import W_NEG, W_POS

CHANNEL = "text"
VARIANTS = {"speaking": lambda: Config.changed(support="members", bind_repeats=True, next_only_present=True),
            "members": lambda: Config.changed(support="members"), "changed": Config.changed}


def words(text: str) -> List[str]:
    """Text channel adapter: words and punctuation as symbols."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


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
            if line.startswith("?"):
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
                elif cmd in ("boyut", "size"):
                    print("  %d an, %s" % (session.mind.t, session.mind.size()))
                elif cmd in ("kaydet", "save"):
                    print("  kaydedildi: %s" % save(session.mind, rest.strip()))
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
