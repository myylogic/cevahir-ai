# -*- coding: utf-8 -*-
"""Living with a Zihin mind from the terminal.

    python -m zihin [--load PATH] [--variant full|speaking] [--kapat a,b] [--ac a,b] [--dil tr|diger] [--pencere N]

A mind of the program lives with everything that was built (--variant full, the default): a new mind is raised so,
and a loaded mind is given every mechanism it was raised without (said in one line). --kapat keeps mechanisms off by
name (remembered with the mind until --ac names them again); a way of reading the mind's own trials put away stays
put away. --variant speaking is the configuration before those mechanisms (a loaded mind then keeps its own settings
and --ac turns single mechanisms on). What leaves this machine stays the owner's choice of each run: the mind reads
the web by itself only with --ag, a folder of documents with --kaynak.

The mind of this computer is kept in $CEVAHIR_HOME/zihin/cevahir.zihin (default ~/.cevahir/zihin; --kayit elsewhere).
A session's experience ledger (what makes !unut exact) is written next to it as defter-<ms>-<pid>.log and deleted at a
clean exit, once the mind has been saved without it (and at exit with --kaydetme); --defter-yok keeps none. --dil sets
how a new mind's text is lower-cased: tr (default, Turkish: I -> ı, İ -> i) or diger (plain lower case); a mind keeps
the convention it was raised with. --pencere sets how many items a new mind holds at once (its window, 3 to 15,
default 7).

Every plain line is something you TELL the mind (provenance "told"), one word per moment, then a silent moment.
Commands (Turkish, with English aliases):
    !gordum <metin>      / !saw     the same, as something the mind SENSES itself (provenance "sensed")
    !izle <soz> || <karsilik> / !watch  the mind WATCHES two others talk (provenance "imitated"): how one answers
    !ogret <dosya>       / !teach   watch every line of a file: "soz<TAB>karsilik" dialogues or plain sentences
    !oku <dosya|klasor> [n] / !read watch every text in a file or folder (.json dialogues, .txt .md .tsv; .docx with python-docx,
                                    .pdf with pypdf): sentences of documents as speech of others (imitated), at most n
                                    of each. Paths (here and in every command and option that takes one): ~ is
                                    expanded, one pair of quotes removed, spaces allowed. Text files: a byte order mark
                                    names the encoding (UTF-8, UTF-16 as Excel "Unicode Text", Notepad "Unicode" and
                                    PowerShell 5.1 write, UTF-32); else UTF-8, a few invalid bytes replaced and counted,
                                    or cp1254 when most non-ASCII bytes are not UTF-8 (said so); a file with NUL bytes
                                    or many control characters is not text and is not read
    !konus               / !chat    chat mode on/off: after each line you write, the mind may answer
    !kim <ad>            / !who     who is talking now (perceived at the start of every utterance)
    !neden               / !why     where each word of the last answer came from (memory, a frame and its relation,
                                    a continuation, a form, similar situations; what the act was expected to lead to)
    !gor <resim>         / !see     show an image with your next utterance (it is seen at its start), e.g.
                                    !gor ben.jpg  then  "ben buyum"; later  !gor baska.jpg  then  "bu kim"
    !dinle <ses.wav>     / !hear    play a sound with your next utterance (it is heard at its start), e.g.
                                    !dinle ben.wav  then  "bu benim sesim"; later  !dinle baska.wav  then  "bu kim"
    !dusun [konu]        / !think   let the mind think about a topic inwardly; you hear its inner voice and its
                                    reality score (nothing leaves the mind otherwise; thoughts are kept as imagined)
    !uyu [n]             / !sleep   sleep: n dreams (each a chain of its own inner voice, about what the day was
                                    about and about what it most wants to know, both by its own evidence: telling and
                                    not yet well known), then sorting: its imaginings weighed against what is real
                                    (trust in its imagination follows), and which earlier dreams life has since shown
                                    true or false
    !kendin              / !introspect  what it can say about itself from its own evidence (its self unit)
    !arastir <konu>      / !research  on your command: search the web (Google Custom Search if GOOGLE_API_KEY and
                                    GOOGLE_CSE_ID are set, otherwise Wikipedia) and read what is found (pages printed)
    !sor_hepsi <dosya>   / !ask_all  every line of a file asked at the same time: each is answered by the mind as
                                    it is now, in a process of its own (as many as the machine has processors); the
                                    answers are exactly what it would give one after another, and none is lived
    !dusun_hepsi a, b, c / !think_all  the same for thoughts: what it thinks about each topic now, at the same time
    !hata                / !faults  (F49) what the mind has learned about its own ways of reading (how
                                    often each held against what it claimed) and where its last faults were: the
                                    context, the word it said or would have said, what was lived, the way that
                                    claimed it
    !deney               / !trials  (F57) the trials of its own ways of reading: which it uses and
                                    which sleep, how often each did better or worse than the mind without it, every
                                    way it took into use or put away, and the last cases (the context, what came,
                                    what it would have said with the way and without it)
    !varsayim            / !conjectures  (F59) the statements it came to by itself: each with
                                    the link and the statement it was carried from, the record it rests on, and what
                                    became of it when it was told more (held, failed, not met yet)
    !ag                  / !net     what this machine can reach: one small request to each service the mind reads
                                    from, with the reason when one cannot be reached (proxy, refusal, no network)
    !oku_url <adres> ... / !read_url  web pages by their addresses, read as documents are: fetched at the same time,
                                    read in the order given; the site's robots.txt is obeyed, every page is printed
    !kendini_gelistir [n] / !improve  the n topics it most wants to know by its own evidence (telling and not yet well
                                    known: tau x drive) are researched; it measures what each way of reading gave it
                                    and chooses its way by that. With open claims (F46), what it was asked about and
                                    found it did not know comes first
    !bak / !bakma        / !look, !nolook  the mind decides by itself (F47), in the silences of its online
                                    life, to look up what it found it did not know (every page read is printed);
                                    !bakma stops that for this session, !bak allows it again. With F50 it
                                    also looks in the turn it is asked in, before it says a word that rests only on
                                    what others share: what it sends for is fetched while it goes on living, and it
                                    answers when it has read it. With F51 it asks itself, in a silence, the
                                    question it knows about what was new in what it was told, and looks up what it
                                    cannot answer (each inner question is printed and is an entry of the history)
    !dogru / !yanlis     / !good / !bad   right after its answer: how its answer was received (a value signal;
                                    the path that led to it is remembered and weighed next time, L5). The answer stays
                                    open for it until the feedback or the next utterance (yours, or the mind's own),
                                    however long the online life ran in between; "not edildi" only when it was learned
Turn taking (while you speak the mind listens, then it may speak until it falls silent) is set by the session, not
learned yet: during your words the act threshold is raised, afterwards it is the mind's own. What the mind says is
only what its own decisions emit; nothing is said or asked for it.
Online life (at a terminal, unless --yasamsiz): between your lines the mind lives silent moments, each as any moment
(what its own decision says in one is said). An innate rhythm, a setting of the program awaiting the owner's decision,
not something the mind learned: once per silence it thinks inwardly after 10 seconds and sleeps after 10 minutes.
    ?<metin>             / ?        what does the mind expect after this beginning (nothing is lived; a clone is asked)
    !neden <metin> => <kelime> / !why   white-box belief: own and effective evidence, per source
    !gecmis              / !history what was lived in this session, with numbers: your lines, the mind's answers,
                                    files read (!oku, one entry per file) or taught (!ogret), watched dialogues (!izle)
    !unut <no>           / !forget  retract that entry exactly (its evidence is removed from the core, every moment
                                    through the ledger); not with --defter-yok
    !boyut               / !size    units, ties, bytes, moments
    !yas                 / !age     developmental profile: moments and nights lived, words known, concepts, phrases
    !kaydet <yol>        / !save    save the mind (with integrity manifest)
    !yukle <yol>         / !load    load a saved mind
    !tasinabilir <yol>   / !portable  save a copy for conversation without the experience ledger (small file)
    !cik                 / !quit
Ctrl+C during a long command (!oku, !ogret, !unut, !arastir, !kendini_gelistir, !uyu, and its sleep while you are
silent) stops it at its next sentence, line or moment, never inside one: what it lived until then is whole and in
the history; then the session ends and the mind is saved. Pressed again before the command has stopped, Ctrl+C does
not wait: the session ends without saving and the mind saved before stays as it was (so does it when the command does
not stop in 30 seconds). A save that does not finish never replaces the mind saved before.
The text adapter splits words and punctuation; the alphabet is open (a new word is a new symbol at any time). The
mind never sees text, only symbols on the "text" channel; meaning is in the ties.
"""
from __future__ import annotations

import argparse
import codecs
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple

from . import ACTED, IMITATED, SENSED, TOLD, Config, Event, Mind, read
from .events import PROVENANCE
from .persist import export_portable, load, save
from .store import W_NEG, W_POS

CHANNEL = "text"


def home_folder() -> str:
    """The folder of this computer's mind: $CEVAHIR_HOME/zihin (default ~/.cevahir/zihin). The ledgers of its sessions
    are kept there too, never in the working folder."""
    home = os.environ.get("CEVAHIR_HOME") or os.path.join(os.path.expanduser("~"), ".cevahir")
    return os.path.join(home, "zihin")


_ISSUED = set()                         # ledger paths given out in this process (two minds never share one)
_TEMPORARY: List[str] = []              # the temporary ledger folder of this process, once made


def temporary_folder() -> str:
    """The ledger folder of minds built in code (tests, research scripts): a temporary folder of this process,
    removed with what is in it when the process exits (by the process that made it, not by a forked child that
    exits before it). Nothing is written into the user's home for them."""
    if not _TEMPORARY:
        import atexit
        import tempfile
        folder = tempfile.mkdtemp(prefix="zihin-defter-")
        atexit.register(_remove_temporary, folder, os.getpid())
        _TEMPORARY.append(folder)
    return _TEMPORARY[0]


def _remove_temporary(folder: str, maker: int) -> None:
    import shutil
    if os.getpid() == maker:
        shutil.rmtree(folder, ignore_errors=True)


def _log_path(folder: str = None) -> str:
    """A new session ledger (the experience log that makes !unut exact): ``folder``/defter-<ms>-<pid>.log. The program
    passes the folder of the kept mind and deletes the ledger at exit (main: the kept mind is saved without it); with
    no folder (a mind built in code) it is made in temporary_folder(), removed when the process exits."""
    folder = folder or temporary_folder()
    os.makedirs(folder, exist_ok=True)
    ms = int(time.time() * 1000)
    while True:
        path = os.path.join(folder, "defter-%d-%d.log" % (ms, os.getpid()))
        if path not in _ISSUED and not os.path.exists(path):
            _ISSUED.add(path)
            return path
        ms += 1


def speaking(folder: str = None) -> Config:
    """The configuration a chat mind is raised with. Its ledger is kept in ``folder`` (the program passes the folder
    of the kept mind); a mind built in code without one keeps it in a temporary folder (see _log_path)."""
    # _log_path() without an argument when no folder is given: research scripts replace it with a function of no
    # arguments to build the configuration without a ledger file (research/living_mind_2026_11_15/t_meaning.py)
    return Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                          backoff=True, resemblance=True, race=True, situations=True, explore=True, starts=True,
                          exclusive=True, exact_relations=True, analogy=True, corroborate=True, paradigmatic=True,
                          bind_relations=True, company=True, replies=True, purpose=True, forms=True, echo=True,
                          exclusive_world=True, log_path=_log_path(folder) if folder else _log_path())


def full(folder: str = None) -> Config:
    """The whole mind: the speaking configuration with every mechanism that was built and measured (STUDIED) in use.
    A mind of the program is raised with it and lives with it (--variant full, the default). ``speaking`` is the
    configuration before those mechanisms: the one every published measurement names its switches from, kept as it
    was so that each of them can be run again."""
    cfg = speaking(folder)
    for name in STUDIED:
        setattr(cfg, name, True)
    return cfg


VARIANTS = {"full": full,
            "speaking": speaking,
            "members": lambda folder=None: Config.changed(support="members"),
            "changed": lambda folder=None: Config.changed()}          # folder: where the session ledger is kept

NEWER = ("backoff", "resemblance", "race", "situations", "explore", "starts", "exclusive",   # a loaded mind may
         "exact_relations", "analogy", "corroborate", "paradigmatic", "bind_relations",      # lack these (--ac)
         "company", "replies", "purpose", "forms", "echo", "exclusive_world")

STUDIED = ("inherit", "witness", "difference", "urge", "known", "wholes",   # built and measured after the speaking
           "told_apart", "open_claims", "look", "think", "faults",           # configuration: in use in the full one
           "silent_claims", "find_out", "wonder", "echo_exact", "families", "lived_apart",   # (the program's
           "asked_place", "whole_kinds", "trials", "built_relations", "conjectures", "unseen_share",   # default)
           "own_race", "echo_told", "relied_on", "shares_tested", "unknown_outside", "trial_ends",
           "split_failed", "deduced_first", "own_turn", "open_window")

NEEDS = (("find_out", ("faults", "look")), ("wonder", ("look",)), ("silent_claims", ("faults",)),   # what does not
         ("built_relations", ("trials",)), ("unseen_share", ("families",)), ("look", ("open_claims",)),   # work
         ("echo_told", ("echo_exact",)), ("relied_on", ("silent_claims",)), ("shares_tested", ("told_apart",)),
         ("trial_ends", ("trials",)), ("split_failed", ("faults",)), ("deduced_first", ("faults", "inherit")),
         ("own_turn", ("silent_claims",)),
         ("told_apart", ("wholes",)), ("whole_kinds", ("wholes",)))
                                                                                                    # without what



def _needs(name: str) -> set:
    """Everything a mechanism cannot work without (``NEEDS``, followed through)."""
    out, todo = set(), [name]
    while todo:
        k = todo.pop()
        for need, wants in NEEDS:
            for w in wants if need == k else ():
                if w not in out:
                    out.add(w)
                    todo.append(w)
    return out


def configure(mind, variant: str = "full", on=(), off=(), loaded: bool = False, look: bool = False,
              say=print) -> bool:
    """Which mechanisms this mind lives with from now on (the program calls it once, for a new mind and for a
    loaded one).

    * ``variant`` "full" (the default): the mind lives with everything that was built. A new mind is raised so; a
      loaded mind is given every mechanism it was raised without, said in one line. Two things stand against it:
      what its owner keeps off (``off``: --kapat, remembered with the mind in ``Mind.owner_off`` until --ac names
      it), and a way of reading the mind's own trials put away (F57: that decision is the mind's; --ac overrules).
    * any other variant: a loaded mind keeps its own settings, and only what ``on`` names (--ac) is turned on.
    * what a mechanism in use cannot work without is turned on with it (``NEEDS``); what is kept off takes with it
      what cannot work without it. ``look``: the owner gave it somewhere to look (--kaynak, --ag).

    A way of reading that is on trial (``deney.WAYS``) is changed through the trials' own record, so that the mind's
    record of its configuration stays whole (``deney.set_from_outside``). Returns whether anything was changed
    (the program then keeps a loaded mind even when nothing was lived in the session)."""
    from . import deney
    cfg = mind.cfg
    names = NEWER + STUDIED
    before = ([bool(getattr(cfg, k, False)) for k in names], sorted(getattr(mind, "owner_off", ())))
    for k in list(on) + list(off):
        if k not in names:
            say("  (--ac / --kapat: bilinmeyen mekanizma '%s'; seçenekler: %s)" % (k, ", ".join(names)))
    on = [k for k in on if k in names]
    named = (set(getattr(mind, "owner_off", ())) | set(k for k in off if k in names)) - set(on)
    if look:
        named.discard("look")
    for k in on + (["look"] if look else []):            # asked for now: what it cannot work without stays
        for w in sorted(_needs(k) & named):
            named.discard(w)
            say("  (%s kapatılmadı: %s onsuz çalışmaz)" % (w, k))
    kept_off = named | set(k for k in names if _needs(k) & named)   # ... and what needs one kept off goes with it

    def put(k: str, value: bool) -> None:
        if k in deney.WAYS:
            deney.set_from_outside(mind, {k: value})
        else:
            setattr(cfg, k, value)

    given, own, without = [], [], []
    for k in names:
        if getattr(cfg, k, False) or k in kept_off:
            continue
        if k in on:
            put(k, True)
            say("  açıldı: %s (senin isteğinle açıldı)" % k)
        elif variant == "full":
            if k in deney.WAYS and deney.own_choice(mind, k) is False:
                own.append(k)                            # put away by its own trials
            else:
                put(k, True)
                given.append(k)
        elif loaded and variant == "speaking" and k in NEWER:
            without.append(k)
    closed = [k for k in names if k in kept_off and getattr(cfg, k, False)]
    for k in closed:
        put(k, False)
    if look and not getattr(cfg, "look", False):
        cfg.look = True                                  # giving it somewhere to look is letting it look: what it
        say("  açıldı: look (ona bakabileceği bir yer verdin; neye, ne zaman bakacağına kendisi karar verir)")
    for k in names:
        for w in sorted(_needs(k)) if getattr(cfg, k, False) else ():
            if not getattr(cfg, w, False):
                put(w, True)
                say("  açıldı: %s (%s onsuz çalışmaz)" % (w, k))
    mind.owner_off = sorted(named)
    if given and loaded:
        say("  tam yapı: bu zihin şu mekanizmalarla büyümedi, şimdi açıldı: %s (birini kapalı tutmak için --kapat ad; "
            "zihnin eski ayarlarıyla yaşaması için --variant speaking)" % ", ".join(given))
    if own:
        say("  (kendi denemeleriyle bıraktığı yollar kapalı kaldı: %s; yine de açmak için --ac %s)" % (
            ", ".join(own), ",".join(own)))
    if closed:
        say("  kapatıldı: %s (senin isteğinle)" % ", ".join(closed))
    if kept_off:
        rest = sorted(kept_off - named)
        say("  (senin isteğinle kapalı: %s%s; açmak için --ac %s)" % (
            ", ".join(sorted(named)), "; onlarsız çalışmadığı için: " + ", ".join(rest) if rest else "",
            ",".join(sorted(named))))
    if without:
        say("  (bu zihin şu mekanizmalarla büyümedi ve onlarsız yaşar: %s; açmak için --ac %s)" % (
            ", ".join(without), ",".join(without)))
    return before != ([bool(getattr(cfg, k, False)) for k in names], mind.owner_off)


DIL = ("tr", "diger")       # how the text adapter lower-cases: Turkish rules (I -> ı, İ -> i) or plain str.lower


def lower_tr(text: str) -> str:
    """Turkish lower case (I -> ı, İ -> i) so that a word at the start of a sentence is the same symbol."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def lower(text: str, dil: str = "tr") -> str:
    """Lower case by a mind's language convention: "tr" Turkish rules (lower_tr), "diger" plain str.lower."""
    return text.lower() if dil == "diger" else lower_tr(text)


def words(text: str, dil: str = "tr") -> List[str]:
    """Text channel adapter: words and punctuation as symbols, lower-cased by the mind's convention (default
    Turkish)."""
    return re.findall(r"\w+|[^\w\s]", lower(text, dil), flags=re.UNICODE)


def clean_path(raw: str) -> str:
    """A path as typed: surrounding spaces and one pair of surrounding quotes removed, ~ expanded (spaces inside
    are part of the path)."""
    path = raw.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
        path = path[1:-1]
    return os.path.expanduser(path)


def path_and_cap(rest: str):
    """The argument of !oku: a path (see clean_path) and an optional last number, the most sentences to read. When
    the whole argument names an existing file or folder ("notlar 2024") it is the path, with no number."""
    whole = clean_path(rest)
    head, _, tail = rest.strip().rpartition(" ")
    if not os.path.exists(whole) and tail.isdigit() and head.strip() and os.path.exists(clean_path(head)):
        return clean_path(head), int(tail)
    return whole, 0


class NotText(ValueError):
    """A file that is not text (NUL bytes, or many control characters): nothing of it is read."""


class TextFile(NamedTuple):
    """How a text file is read: its codec, and how many invalid byte sequences were replaced by U+FFFD."""
    encoding: str
    replaced: int = 0


BOMS = ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"), (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"))  # UTF-32 LE first: it begins like UTF-16 LE
CONTROLS = re.compile("[\x00-\x08\x0e-\x1f\x7f-\x9f]")   # control characters other than tab, line and page breaks
UNDECODED = re.compile("[\udc80-\udcff]")                # bytes a codec could not decode (errors="surrogateescape")
ASCII = bytes(range(0x80))
CONTROL_SHARE = 100     # a file is not text when more than one character in CONTROL_SHARE is a control character


def text_encoding(path: str) -> TextFile:
    """How a text file is read, decided from the whole file (block by block) before anything is lived, so a file is
    never half taught in one encoding:

    * a byte order mark names its codec: UTF-8, UTF-16 little or big endian (Excel "Unicode Text", Notepad
      "Unicode", PowerShell 5.1 redirection) or UTF-32; the mark is not read as a symbol;
    * a file with NUL bytes (NUL characters, when a mark names its codec) is not text, nor one where more than one
      character in CONTROL_SHARE is a control character other than tab, line and page breaks: NotText;
    * without a mark it is UTF-8, and its invalid byte sequences are replaced (U+FFFD) and counted, unless most of its
      non-ASCII bytes are invalid UTF-8 and the whole file decodes as cp1254 (Turkish ANSI, as Excel and Notepad save
      on Turkish Windows): then it is read as cp1254. One stray byte in a UTF-8 file is one replaced character, not a
      whole file read in the wrong encoding."""
    with open(path, "rb") as f:
        head = f.read(4)
        marked = next((codec for bom, codec in BOMS if head.startswith(bom)), None)
        codec = marked or "utf-8"
        f.seek(0)
        replacing = codecs.getincrementaldecoder(codec)("replace")       # how it is read
        ignoring = codecs.getincrementaldecoder(codec)("ignore")         # ... and without what could not be decoded
        escaping = ansi = None
        if not marked:                       # UTF-8: each byte that is not UTF-8 kept as one mark (all are >= 0x80)
            escaping = codecs.getincrementaldecoder(codec)("surrogateescape")
            ansi = codecs.getincrementaldecoder("cp1254")()
        chars = controls = undecoded = replaced = non_ascii = 0
        for block, final in _with_final(iter(lambda: f.read(1 << 20), b"")):
            text = replacing.decode(block, final)
            if "\0" in text:
                raise NotText("NUL baytları var" + ("" if marked else "; UTF-16 ise bayt sırası işaretiyle kaydedin"))
            chars += len(text)
            controls += len(CONTROLS.findall(text))
            replaced += text.count("\ufffd") - ignoring.decode(block, final).count("\ufffd")
            if escaping is not None:
                undecoded += len(UNDECODED.findall(escaping.decode(block, final)))
                non_ascii += len(block.translate(None, ASCII))
                if ansi is not None:
                    try:
                        ansi.decode(block, final)
                    except UnicodeDecodeError:
                        ansi = None
    if controls * CONTROL_SHARE > chars:
        raise NotText("%d denetim karakteri var (%d karakterde birden çok)" % (controls, CONTROL_SHARE))
    if undecoded and ansi is not None and 2 * undecoded > non_ascii:
        return TextFile("cp1254")
    return TextFile(codec, replaced)


def _with_final(blocks):
    """The blocks of a file, each with False, then an empty last block with True (the decoders' final call)."""
    for block in blocks:
        yield block, False
    yield b"", True


def open_text(path: str):
    """A text file opened for reading in its encoding (text_encoding); returns (file, TextFile). NotText when it is
    not text."""
    found = text_encoding(path)
    return open(path, encoding=found.encoding, errors="replace"), found


def encoding_notes(found: Dict[str, TextFile]) -> List[str]:
    """What the program says about how the text files it read were decoded (file name -> TextFile): files that were
    not UTF-8, and the invalid byte sequences it replaced (U+FFFD), with how many, by encoding."""
    notes = []
    ansi = [name for name, f in found.items() if f.encoding == "cp1254"]
    wide = ["%s (%s)" % (name, f.encoding.upper()) for name, f in found.items() if f.encoding in ("utf-16", "utf-32")]
    if ansi:
        notes.append("UTF-8 değil, cp1254 (Türkçe ANSI) olarak okundu: " + ", ".join(ansi[:10]))
    if wide:
        notes.append("bayt sırası işaretine göre okundu: " + ", ".join(wide[:10]))
    for label, codecs_ in (("UTF-8", ("utf-8", "utf-8-sig")), ("UTF-16", ("utf-16",)), ("UTF-32", ("utf-32",))):
        repaired = ["%s (%d)" % (name, f.replaced) for name, f in found.items() if f.replaced and f.encoding in codecs_]
        if repaired:
            notes.append("geçersiz %s dizileri \ufffd ile değiştirildi (kaç tane): %s" % (label, ", ".join(repaired[:10])))
    return notes


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
        if "instruction" in keys:                         # instruction datasets: the instruction, then its input
            ask = (keys["instruction"] + " " + keys.get("input", "")).strip()
        answer = next((keys[k] for k in ANSWER_KEYS if k in keys), None)
        if ask and answer:
            pairs.append((ask, answer))
    return pairs


def sentences(text: str, dil: str = "tr") -> List[str]:
    """Split a document into sentences (and lines), keeping only those with at least two words."""
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [p.strip() for p in parts if len(words(p, dil)) >= 2]


CODE = (".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".ps1", ".sh", ".yml", ".yaml",
        ".toml", ".ini", ".cfg", ".html", ".css", ".sql")      # source code is read as a language too, line by line


def read_document(path: str, encodings: list = None) -> str:
    """Plain text of a document. .docx needs python-docx, .pdf needs pypdf (only if such files are present). Text
    files are decoded as text_encoding decides (NotText when they are not text); how a text file was decoded (a
    TextFile) is appended to ``encodings`` when given."""
    low = path.lower()
    if low.endswith((".txt", ".md", ".tsv", ".csv") + CODE):
        f, found = open_text(path)
        with f:
            text = f.read()
        if encodings is not None:
            encodings.append(found)
        return text
    if low.endswith(".docx"):
        import docx  # python-docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    if low.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return ""


def live_counts(mind: Mind) -> tuple:
    """(units, ties) the mind holds now; those !unut took back are kept in the tables but are dead, not counted."""
    S = mind.S
    return int(S.u_alive[:S.n_units].sum()), int(S.t_alive[:S.n_ties].sum())


def taken_back(log, m: int) -> bool:
    """Whether moment m of a ledger was already retracted (a disk ledger keeps the set, a ledger in memory marks the
    entry)."""
    retracted = getattr(log, "retracted", None)
    if retracted is not None:
        return m in retracted
    entry = log.get(m)
    return entry is not None and bool(entry.retracted)


def profile(mind: Mind) -> str:
    """Developmental profile measured from the substance (nothing assigned)."""
    from .evidence import conviction
    from .store import ATOMIC, CATEGORY, COMPOSED, EXPECT, SLOTTED
    S = mind.S
    n = S.n_units
    alive = S.u_alive[:n]
    kinds = S.u_kind[:n] * alive                        # units !unut took back are not counted (kind 0)
    cid = mind.ch.id(CHANNEL) if mind.ch.has(CHANNEL) else -1
    words_seen = known = 0
    for unit in range(n):
        if not int(kinds[unit]) & ATOMIC or int(S.u_channel[unit]) != cid:
            continue
        words_seen += 1
        ties = S.out.get((unit, EXPECT, 1), [])
        if ties:
            mass = float(max((S.w1(0, t) * mind.trust).sum() + (S.w1(1, t) * mind.trust).sum() for t in ties))
            known += conviction(mass, mind.cfg.k) >= 0.5
    phrases = int(((kinds & COMPOSED).bool() & ~(kinds & SLOTTED).bool()).sum())
    lines = ["  yaşanmış an: %d   gece (uyku döngüsü): %d" % (mind.t, getattr(mind, "nights", 0)),
             "  karşılaştığı kelime: %d   kanaat kazandığı kelime (c >= 0.5): %d" % (words_seen, known),
             "  kurduğu kavram (kategori): %d   çerçeve: %d   birleşik birim (dizi): %d" % (
                 int((kinds & CATEGORY).bool().sum()), int((kinds & SLOTTED).bool().sum()), phrases),
             "  bağ: %d   öz: %.1f MB" % (live_counts(mind)[1], mind.size()["core_tensors"] / 1e6),
             "  (insan yaşı karşılığı verilmez: kaynaklı bir gelişim normu olmadan sayı uydurulmaz)"]
    return "\n".join(lines)


FACETS = "parca"
TURN, TURN_END = "sira", "⏎"           # the end of someone's utterance (Enter) is perceived, like a pause in speech
PERSON, SOMEONE = "kisi", "biri"        # who is speaking is perceived (like a face or a voice); "biri" = someone unknown
from .gorme import EYE, LOOK
from .isitme import EAR, SOUND
TONE = "ton"                          # how an utterance ends (? ! .) is perceived with its end, like intonation
FINAL = (".", "?", "!", "…")          # closing punctuation (heard as the tone of a turn's end in minds that lived it so)
VALUE = "deger"
VALUE_SIGN = {(VALUE, "iyi"): 1.0, (VALUE, "kötü"): -1.0}     # the adapter defines what is good and bad


DAY = 12                                # the recent history entries that are "the day" (dusunce.day_residue)
READ, TAUGHT, WATCHED, CHAT = "read", "taught", "watched", "chat"   # kinds of history entries; a line lived without
SPOKE = "spoke"                                                      # an answer is of its provenance ("told", "sensed");
                                                                     # what the mind said on its own in its life: spoke


@dataclass
class Entry:
    """One entry of a session's history: what was lived, as !gecmis shows it and !unut takes it back.

    ``kind`` says what it was (a line "told" or "sensed", a "chat" line with the mind's answer, a file "read" or
    "taught", a dialogue "watched", what the mind said on its own in a silent moment of its life, "spoke"); ``text``
    is how !gecmis shows it (never parsed); ``moments`` are the moments it
    lived (!unut retracts each through the ledger); ``heard`` is the last utterance the mind heard in it (a thought
    after a silence is about it, dusunce.topic_of); ``counts`` are the words lived in it, heard or said, with how often
    (the day residue weighs them, dusunce.day_residue)."""
    number: int
    kind: str
    text: str
    moments: List[int]
    heard: str = ""
    counts: Dict[str, int] = field(default_factory=dict)


class _indices_rebuilt_once:
    """Retracting many moments with the substance's indices rebuilt once, after the last one.

    ``Substance.retract`` rebuilds every index from the tensors after each moment that removed a tie or a unit, an
    O(ties) pass; for the 1782 moments of a read guide that was 269 s of a 270 s ``!unut``. Nothing in a retraction
    reads the indices (only the ledger entry and the tensors) and the rebuild is a function of the tensors alone, so
    one rebuild at the end gives the same tensors and the same indices as one per moment (pinned by a test). A
    ``retract_many`` in store.py would say this directly; store.py is not changed here."""

    def __init__(self, substance) -> None:
        self.S, self.wanted = substance, False

    def __enter__(self):
        def later() -> None:
            self.wanted = True
        self.S.rebuild_indices = later
        return self

    def __exit__(self, *exc) -> None:
        del self.S.rebuild_indices                       # the class's own method again
        if self.wanted:
            self.S.rebuild_indices()


class Interrupted(Exception):
    """A long command was asked to stop (Ctrl+C): raised by Session.check between two sentences, lines, dialogues or
    moments, never inside a moment, so what was lived until then is whole."""


class Session:
    def __init__(self, mind: Mind, pieces=None, stop: threading.Event = None) -> None:
        self.mind = mind
        self.stop = stop if stop is not None else threading.Event()   # set: long commands stop at their next check
        self.dil = getattr(mind, "dil", "tr")  # the mind's language convention (how its text is lower-cased)
        self.history: List[Entry] = []        # what was lived in this session, entry by entry
        self.numbered = 0                     # the last number given to a history entry (numbers are never reused)
        self.answered = None                  # the path (mind.trace) of the last answer, kept for its feedback
        self.pieces = pieces                  # optional subword tokenizer: pieces of each word as facets
        if pieces is not None:
            mind.add_channel(FACETS, "facet")
        mind.add_channel(VALUE, "value")
        mind.value_sign = dict(VALUE_SIGN)
        known = mind.ch.has(TURN) and mind.ch.symbol_index(mind.ch.id(TURN), TURN_END, create=False) >= 0
        self.turns = mind.t == 0 or known       # a mind that never lived turn ends keeps the older turn passing
        tone_id = mind.ch.id(TONE) if mind.ch.has(TONE) else -1
        lived_tone = tone_id >= 0 and any(mind.ch.symbol_index(tone_id, t, create=False) >= 0 for t in FINAL)
        self.tones = mind.t == 0 or lived_tone  # a mind that lived closing punctuation as words keeps hearing it so
        mind.add_channel(TURN, "sense")
        for name in (PERSON, TONE):             # who speaks and how it ends accompany the words as context: they are
            if mind.ch.has(name) and mind.ch.kind(mind.ch.id(name)) == "facet":   # not what a thing looks like
                mind.ch.kinds[mind.ch.id(name)] = "context"
            mind.add_channel(name, "context")
        mind.add_channel(EYE, "sense")          # a seen thing is present in the episode, its aspects are facets
        mind.add_channel(LOOK, "facet")
        mind.add_channel(EAR, "sense")          # a heard thing likewise
        mind.add_channel(SOUND, "facet")
        self.view = None                        # an image shown for the next utterance: (gist, facets)
        self.seen: Dict[str, str] = {}          # the file names of images shown in this session, by gist
        from .mind import INNER_CHANNEL, SELF_CHANNEL
        if not mind.ch.has(SELF_CHANNEL):       # minds born before the self unit existed get it now
            mind.add_channel(SELF_CHANNEL, "context")
        if not mind.ch.has(INNER_CHANNEL):      # and the inner sense of not knowing
            mind.add_channel(INNER_CHANNEL, "sense")
        self.hearing = None                     # a sound heard with the next utterance: (gist, facets)
        self.partner = SOMEONE                  # who the mind is talking with now (set by --kim or !kim)
        self.explain = False                    # the chat records where each word of an answer came from (!neden)
        self.sources: List[dict] = []           # ... the record of the last answer, word by word
        self.open_turn = False                # the mind has just answered; feedback may still come
        self.background = False               # F50: a look decided in a turn is fetched while the mind goes on living
                                              # (the online life sets it); else it is carried out within the turn
        self.fetching = None                  # ... (what is being fetched: looking, the open situation, the thread, its box)
        self.owed: Dict[int, int] = {}        # ... item -> the situation of a call it has not answered yet
        self.look_notes: List[str] = []       # ... what a look of its own printed (shown by the chat after the answer)
        self.may_look = False                 # F47: the owner lets this session carry out the mind's own looks
        self.look_source = None               # ... from this source (None: web.default_source)
        self.look_gap = self.LOOK_GAP         # ... at least this many seconds apart
        self.looked_at = 0.0
        self.looks: List[dict] = []           # ... the reports of the looks of this session
        self.net = None                       # the session's way out to the network (zihin/ag.py; None: made on demand)
        self.thought = None                   # F48: its last thought of its own: (topic, inner words, reality)
        self.thoughts: List[tuple] = []       # ... and all of this session

    LOOK_GAP = 5.0                            # a setting of the program (not learned): seconds between two looks of
                                              # its own, so that a reading service is never asked in a burst

    @property
    def may_look(self) -> bool:
        return self.__dict__.get("_may_look", False)

    @may_look.setter
    def may_look(self, allowed: bool) -> None:
        self.__dict__["_may_look"] = bool(allowed)
        self.mind.__dict__["can_look"] = bool(allowed)      # F50: the mind knows whether it has somewhere to look

    def find_out(self, looking, say=None, limit: int = 80) -> List[str]:
        """F50: carry out, within the turn, a look the mind decided on when it was called on; then the mind is brought
        back to what it was asked and says what it can say now (``say_again``). With ``background`` (the online
        life) only the fetching is started, on a thread that touches nothing of the mind: the turn ends at once, the
        mind goes on living, and a later silent moment reads what was found and answers (``silent_moment``). Returns
        the words it said after its look ([] when the fetching was only started)."""
        import time
        from .web import fetch, look_up
        context = self.mind.open_where.get(looking[0])
        note = self.look_notes.append
        note("zihin (kendi kararıyla) bakıyor: %s" % (looking[1],))
        first = self.speak(self.mind.step([]), say, limit)   # it perceives that it is looking (an inner event,
                                                             # alone in its moment): what it learned to say then,
        if self.background and self.fetching is not None:    # it says
            return first                                     # (one look at a time: this one waits for its turn)
        if self.background:
            box: Dict[str, object] = {}
            source = self.look_source

            def work() -> None:
                box["fetched"] = fetch(self, str(looking[1]), source)
            thread = threading.Thread(target=work, daemon=True)
            self.fetching = (looking, context, thread, box)
            if context is not None:
                self.owed[looking[0]] = context
            thread.start()
            return first
        first += self.speak(self.mind.step([]), say, limit)   # a pause: what it reads begins a run of its own
        try:
            report = look_up(self, looking, source=self.look_source, say=note)
        except Interrupted:
            raise
        except Exception as exc:                        # a service that does not answer: the look led to nothing
            from .learn import settle_look
            note("%s: bakılamadı (%s)" % (looking[1], type(exc).__name__))
            report = {"topic": str(looking[1]), "read": [], "note": type(exc).__name__,
                      "met": settle_look(self.mind, looking[0])}
        self.looks.append(report)
        self.looked_at = time.time()
        self.company(self.partner)
        return first + (self.speak(self.recall(context), say, limit) if context is not None else [])

    def recall(self, context: int) -> List[Event]:
        """F50: the mind is brought back to a situation it lived (what it was asked): after a pause its items come to
        mind again one after another (imagined; nothing is said for it), and what it decides at the last of them is
        its own. Returns those acts (the caller says them: ``speak``)."""
        from .dusunce import moments_of, take_in
        moments = moments_of(self.mind, context)
        if not moments:
            return []
        self.mind.step([])                               # a pause: what was asked comes to mind as a run of its own
        acts = take_in(self.mind, moments)
        if not acts:                                     # as in a turn: a moment in silence, then the next
            acts = self.mind.step([])
            if not acts:
                acts = self.mind.step([])
        return acts

    def silent_moment(self, say=print) -> List[Event]:
        """A silent moment of its life, lived as any moment; then what the mind decided in it about its own time is
        let happen. It decided to look something up (F47, ``Mind.looking``) and the owner lets this session look
        (``may_look``): the look is carried out (what is found is read, every page printed), the mind settles it,
        and it is an entry of the history (so !unut takes it back). Else it decided to think about what it last took
        in (F48, ``Mind.thinking``): the inner voice thinks it (``dusunce.think``), the mind settles what the thought
        came to, and it is an entry of the history too. Returns the acts the mind decided in the moment (its words,
        said by the caller)."""
        import time
        if self.fetching is not None and not self.fetching[2].is_alive():   # F50: what it sent for has come: it reads
            from .web import look_up                                        # it now, and answers what it was asked
            looking, context, _, box = self.fetching
            self.fetching = None
            start = self.mind.t + 1
            self.close_turn()
            self.mind.step([])                           # a pause: what it reads begins a run of its own
            report = look_up(self, looking, source=self.look_source, say=say, fetched=box.get("fetched"))
            self.looks.append(report)
            self.looked_at = time.time()
            self.remember(READ, "[kendi kararıyla baktı] %s" % (looking[1],), list(range(start, self.mind.t + 1)))
            self.owed.pop(looking[0], None)
            self.company(self.partner)
            return self.recall(context) if context is not None else []
        can = self.mind.__dict__.get("can_look")
        if can and getattr(self.mind.cfg, "own_race", False) and (
                self.fetching is not None or time.time() - self.looked_at < self.look_gap):
            self.mind.__dict__["can_look"] = False       # F61: not in this moment (one look at a time, none in a
        try:                                             # burst): the look takes no part in the race
            acts = self.mind.step([])
        finally:
            if can is not None:
                self.mind.__dict__["can_look"] = can
        if acts:
            return acts
        self.mind.__dict__["_carrying"] = True           # F61: what it decided is carried out: the pauses of that are
        try:                                             # no moments of its own time (nothing is decided in them)
            return self._carry_out(acts, say)
        finally:
            self.mind.__dict__.pop("_carrying", None)

    def _carry_out(self, acts: List[Event], say=print) -> List[Event]:
        """What the mind decided about its own time in a silent moment is let happen (``silent_moment``)."""
        import time
        looking = getattr(self.mind, "looking", None)
        if looking is not None and self.may_look and self.fetching is None \
                and time.time() - self.looked_at >= self.look_gap:   # (one look at a time: none while one is on its way)
            from .web import look_up
            start = self.mind.t + 1
            say("\n  zihin (kendi kararıyla) bakıyor: %s" % (looking[1],))
            try:
                report = look_up(self, looking, source=self.look_source, say=say)
            except Interrupted:
                raise
            except Exception as exc:                    # a service that does not answer: the look led to nothing
                from .learn import settle_look
                say("  %s: bakılamadı (%s)" % (looking[1], type(exc).__name__))
                report = {"topic": str(looking[1]), "read": [], "note": type(exc).__name__,
                          "met": settle_look(self.mind, looking[0])}
            self.looks.append(report)
            if report.get("note") == "kendi bildiğiyle":   # nothing was fetched: it found it can say it already
                say("  bakmasına gerek kalmadı: artık kendisi söyleyebiliyor (beklenti kapandı)")
            else:
                self.looked_at = time.time()
                outcome = {True: "beklenti kapandı", False: "beklenti hâlâ açık",
                           None: "okunacak bir beklenti kalmamış"}
                say("  okundu: %d cümle; %s" % (report.get("sentences", 0), outcome[report.get("met")]))
            self.remember(READ, "[kendi kararıyla baktı] %s" % (looking[1],), list(range(start, self.mind.t + 1)))
            return acts
        if getattr(self.mind, "supposing", None) is not None:
            from .varsayim import carry_out               # F59: it thinks over what it was told of a thing
            start = self.mind.t + 1
            new = carry_out(self, say)
            if new:
                self.remember(SPOKE, "[kendi kendine vardı] %d varsayım" % len(new),
                              list(range(start, self.mind.t + 1)))
            return acts
        wondering = getattr(self.mind, "wondering", None)
        if wondering is not None and getattr(self.mind, "thinking", None) is None:
            from .dusunce import ask_itself              # F51: it asks itself about an item it met for the first time
            start = self.mind.t + 1
            self.mind.met_once.pop(wondering[0], None)
            asked, words_, unknown = ask_itself(self, wondering[0])
            if asked is None:                            # it knows no question to ask yet: nothing to ask itself
                self.mind.met_once.clear()
            else:
                say("\n  zihin (kendi kendine sordu): %s -> %s" % (
                    " ".join(asked), "bilmiyor" if unknown else " ".join(words_) or "(sessiz)"))
                self.thoughts.append((" ".join(asked), words_, 1.0 if not unknown else 0.0))
                self.remember(SPOKE, "[kendi kendine sordu] %s -> %s" % (
                    " ".join(asked), " ".join(words_) or ("(bilmiyor)" if unknown else "(sessiz)")),
                    list(range(start, self.mind.t + 1)))
            return acts
        if getattr(self.mind, "thinking", None) is not None:
            from .dusunce import think, topic_of
            from .learn import settle_thought
            start = self.mind.t + 1
            about = getattr(self.mind, "about", None)      # the item its failed expectation was of, when it is a word
            topic = str(self.mind.symbol_of(about)[1]) if about is not None and \
                self.mind.symbol_of(about)[0] == CHANNEL else topic_of(self)
            words_, reality = think(self, topic) if topic else ([], 0.0)
            unit = self.mind.unit_of(CHANNEL, topic) if topic else -1
            if not words_ and unit >= 0 and getattr(self.mind.cfg, "find_out", False) \
                    and getattr(self.mind.cfg, "open_claims", False):
                from .learn import note_unknown          # F50: a thought that came to nothing is an unknown about
                note_unknown(self.mind, unit, unit)      # its topic (what it may then look up), not a thought that
                self.mind.heard = None                   # was worth nothing
            else:
                settle_thought(self.mind, bool(words_), reality)
            self.thought = (topic, words_, reality)
            self.thoughts.append(self.thought)
            self.remember(SPOKE, "[kendi kararıyla düşündü] %s: %s" % (topic or "-", " ".join(words_) or "(sessiz)"),
                          list(range(start, self.mind.t + 1)))
        return acts

    def words(self, text: str) -> List[str]:
        """The text adapter with this mind's language convention: everything the session reads and hears."""
        return words(text, self.dil)

    def check(self) -> None:
        """Called by long commands between two sentences, lines, dialogues or moments (never inside a moment): when
        the session was asked to stop (Ctrl+C), the command stops here with what it lived so far (Interrupted)."""
        if self.stop.is_set():
            raise Interrupted()

    def lower(self, text: str) -> str:
        return lower(text, self.dil)

    def remember(self, kind: str, text: str, moments: List[int], heard: List[str] = (), said: List[str] = ()) -> int:
        """A new entry of the history (see Entry): its kind, how !gecmis shows it, its moments, the utterances heard
        in it (in order) and the words the mind said in it. Returns its number."""
        counts: Dict[str, int] = {}
        for word in [w for utterance in heard for w in self.body(utterance)] + list(said):
            counts[word] = counts.get(word, 0) + 1
        self.numbered += 1
        self.history.append(Entry(self.numbered, kind, text, list(moments), heard[-1] if heard else "", counts))
        return self.numbered

    def last_heard(self) -> str:
        """The last utterance the mind heard in this session, whatever the entry it belongs to: a line told, the
        partner's line in a chat, the answer of a dialogue watched, the last sentence of a file read or taught."""
        return next((entry.heard for entry in reversed(self.history) if entry.heard), "")

    def events(self, symbol: str, prov: int, speaker: str = None) -> List[Event]:
        """One word as an event on the text channel, with its subword pieces and its speaker as facets."""
        out = [Event(CHANNEL, symbol, prov)]
        if speaker:
            out.append(Event(PERSON, speaker, SENSED))
        if self.pieces is not None and symbol.isalpha():
            parts = [p for p in self.pieces.tokenize(symbol) if p.strip()]
            if len(parts) > 1:
                out += [Event(FACETS, p.strip(), prov) for p in parts]
        return out

    def company(self, person: str = None) -> None:
        """F33 (Relational Memory): who is present while what follows is lived, as this session perceives them: a
        person it was told the name of, or no one in particular (someone unknown, a document)."""
        if getattr(self.mind.cfg, "company", False):
            self.mind.company = person if person and person != SOMEONE else None

    def live(self, text: str, prov: int, keep: bool = True, lesson: bool = False) -> None:
        """Hear (told) or sense an utterance while listening: the mind does not speak into it. ``keep``: the utterance
        is an entry of the history (not when it is part of a larger one, a taught file). ``lesson``: a line of a taught
        file, lived with no one in particular (F33)."""
        self.company(None if lesson else self.partner)
        start = self.mind.t + 1
        self.listen(text, prov, speaker=self.partner)
        self.end_of_turn(prov, tone=self.tone(text))
        self.mind.step([])
        if keep:
            self.remember(PROVENANCE[prov], text, list(range(start, self.mind.t + 1)), [text])

    def listen(self, text: str, prov: int, speaker: str = None) -> None:
        """Hear an utterance while not speaking (turn taking set by the session). ``speaker``: who says it, perceived
        at its start (kept out when the mind never lived speakers, so older minds are read as before)."""
        theta = self.mind.knob["theta"]
        self.mind.knob["theta"] = 1e9
        try:
            if self.view is not None or self.hearing is not None:   # what is shown and heard with this utterance
                sensed = []                                         # comes first, both in the same moment
                if self.view is not None:
                    gist, facets = self.view
                    sensed += [Event(EYE, gist, SENSED)] + [Event(LOOK, f, SENSED) for f in facets]
                if self.hearing is not None:
                    gist, facets = self.hearing
                    sensed += [Event(EAR, gist, SENSED)] + [Event(SOUND, f, SENSED) for f in facets]
                self.view = self.hearing = None
                self.mind.step(sensed)
            for symbol in self.body(text):
                self.mind.step(self.events(symbol, prov, speaker))
        finally:
            self.mind.knob["theta"] = theta

    def body(self, text: str) -> List[str]:
        """The words of an utterance without its closing punctuation (that is perceived with the end, see tone)."""
        symbols = self.words(text)
        if self.turns and self.tones:
            while symbols and symbols[-1] in FINAL:
                symbols = symbols[:-1]
        return symbols

    @staticmethod
    def tone(text: str) -> str:
        symbols = words(text)
        return "?" if symbols and symbols[-1] == "?" else ("!" if symbols and symbols[-1] == "!" else ".")

    def end_of_turn(self, prov: int, listening: bool = True, tone: str = None) -> List[Event]:
        """Someone's utterance has ended: a perceived event, with how it ended (? ! .) as a facet. While listening
        the mind does not speak into it."""
        if not self.turns:
            return []
        theta = self.mind.knob["theta"]
        if listening:
            self.mind.knob["theta"] = 1e9
        try:
            events = [Event(TURN, TURN_END, prov)]
            if tone and self.tones:
                events.append(Event(TONE, tone, prov))
            return self.mind.step(events)
        finally:
            self.mind.knob["theta"] = theta

    def watch(self, first: str, answer: str, prov: int = IMITATED, asker: str = SOMEONE) -> None:
        """Two others talk: one says ``first``, another answers ``answer``; the mind only watches. The end of each
        utterance is perceived, so the mind can learn who speaks when (turn taking is watched, not coded). The one
        who asks is perceived (someone, unless named); the one who answers is the role the mind takes in a chat."""
        self.company(asker)
        self.listen(first, prov, speaker=asker)
        self.end_of_turn(prov, tone=self.tone(first))
        self.listen(answer, prov)
        self.end_of_turn(prov, tone=self.tone(answer))
        self.mind.step([])
        if getattr(self.mind.cfg, "wholes", False):      # F43: the dialogue lived once more as wholes, and in summary
            from .butun import consolidate
            consolidate(self, first, answer, prov, asker)

    def reply(self, text: str, limit: int = 80, say=None) -> str:
        """You speak (told), then the mind may speak: what its own decision emits, one symbol per moment, until silence.

        The turn passes to the mind when your utterance has ended (older minds: with your last word): it decides in
        that very moment, the context of your whole utterance present. When it decides nothing, a silent moment passes
        (in which it may feel its not knowing, L7) and it lives the next one; what it decides then is its own too. The
        words said are exactly the words its decisions emit (speak): no thought is said for it, nothing is asked back
        for it, nothing it decided is left unsaid. The answer's path (its episode trace) is kept for "!dogru" and
        "!yanlis"."""
        self.close_turn()
        start = self.mind.t + 1
        symbols = self.words(text)
        if not symbols:
            return "(sessiz)"
        self.company(self.partner)
        if self.explain:                                 # !neden: each decided word is attributed as it is decided
            self.mind.__dict__["_why"] = []
            self.mind.__dict__["_explain_on"] = True
        said = None
        if self.turns and getattr(self.mind.cfg, "wholes", False):   # F43: an utterance never answered that has a
            from .butun import reply_in_parts                          # summary is heard in it, answered part by part
            said = reply_in_parts(self, text, limit, say)
        if said is None:
            looking = None
            if self.turns:                               # the turn passes when your utterance has ended
                self.listen(text, TOLD, speaker=self.partner)
                acts = self.end_of_turn(TOLD, listening=False, tone=self.tone(text))
                looking = getattr(self.mind, "looking", None)   # F50: it decided to look in the moment it was called on
            else:                                        # older minds: the turn passes with your last word
                self.listen(" ".join(symbols[:-1]), TOLD)
                acts = self.mind.step(self.events(symbols[-1], TOLD))
            if not acts and self.turns:                  # nothing at once: a moment passes in silence (it may feel its
                acts = self.mind.step([])                # not knowing, L7), then the next; what it decides is its own
                looking = looking or getattr(self.mind, "looking", None)
                if not acts:
                    acts = self.mind.step([])
                    looking = looking or getattr(self.mind, "looking", None)
            said = self.speak(acts, say, limit)
            if self.may_look and getattr(self.mind.cfg, "find_out", False):
                looking = getattr(self.mind, "looking", None) or looking   # (or decided in the middle of its answer)
                if looking is not None and looking[0] == self.mind.__dict__.get("_sought"):   # (not a look it
                    said = said + self.find_out(looking, say, limit)   # decided on in a silent moment of the turn,
                                                                       # F47: that one is its own time's)
        if self.explain:
            self.sources = self.mind.__dict__.pop("_why", [])
            self.mind.__dict__.pop("_explain_on", None)
        self.remember(CHAT, text, list(range(start, self.mind.t + 1)), [text], said)
        self.open_turn = True                            # the turn stays open for "!dogru" / "!yanlis"
        self.answered = list(self.mind.trace)            # ... and the path that led to the answer is kept for them
        return " ".join(said) if said else "(sessiz)"

    def speak(self, acts: List[Event], say=None, limit: int = 80) -> List[str]:
        """The mind says what it decided: every word its decisions emit is said (``say``: out as it is said) and comes
        back as its own echo in the next moment, where it decides again (dusunce.go_on). Its speech ends when it decides
        nothing, or when the echo brings back a composed context this episode already lived (it has come round to a
        situation it was in): the echo is heard and it does not decide again there. ``limit`` moments are only a safety
        cap, ended the same way. No echo is ever dropped and nothing it decided is left unsaid. Returns the words
        said."""
        from .dusunce import go_on
        said: List[str] = []

        def out(words_now: List[str]) -> None:
            said.extend(words_now)
            for w in words_now:
                if say is not None:
                    say(w)                               # out as it is said (online life)
        go_on(self.mind, acts, out, limit)
        if said:
            self.open_turn = True                        # its own utterance is a turn: what comes next begins after it
        return said

    def close_turn(self) -> None:
        self.answered = None
        if self.open_turn:
            self.mind.step([])
            if self.mind.episode and getattr(self.mind.cfg, "find_out", False):
                self.mind.step([])                       # F50: an inner event was still to be perceived in that
            self.open_turn = False                       # moment (it had begun to look): the pause comes after it

    def feedback(self, good: bool) -> str:
        """How the mind's last answer was received: a value event at the end of the same episode (L5: every step of
        the path that led to the answer gets evidence toward the value, learn.learn_value).

        The path is the answer's episode trace, kept from the answer until the feedback or the next utterance and
        given back to the mind for the moment the value arrives. The online life lives silent moments meanwhile (and
        may think or sleep), and a silent moment ends an episode (mind.trace is emptied); typed at once, the kept path
        is the trace itself. After silent moments the value reaches the same path with the same evidence as typed at
        once: the lag-2 ties of every step of the path to the value (the consequence, L5). Only that path is the same.
        The value's lag-1 ties are not: typed at once, the value comes in the moment after the answer and the answer's
        last focus is tied to it (as to any event that follows); after silent moments the moment before it is a
        silent one, nothing in it is tied to the value, because time passed. Says "not edildi" only when the value
        evidence grew."""
        if not self.open_turn:
            return "  geri bildirim, zihnin cevabından hemen sonra verilmeli"
        label = "iyi" if good else "kötü"
        spoken_over = self.answered is None              # the mind spoke on its own after its answer
        path = list(self.answered or [])
        self.answered = None
        before = self._path_value(path, label)
        self.mind.trace = list(path)
        self.mind.step([Event(VALUE, label, SENSED)])
        self.mind.step([])
        self.open_turn = False
        if path and self._path_value(path, label) > before:
            return "  not edildi (%s)" % label
        if spoken_over:
            return "  öğrenilmedi (%s): zihin cevabından sonra kendiliğinden konuştu; geri bildirim o cevaba artık " \
                   "ulaşmıyor" % label
        return "  öğrenilmedi (%s): bu cevap zihinde bir yol bırakmadı (sessizlikle bitti); geri bildirim bir şey " \
               "öğretmedi" % label

    def _path_value(self, path: List[int], label: str) -> float:
        """The positive evidence the steps of a path hold toward a value ("later in the episode", EXPECT lag 2)."""
        from .store import EXPECT
        S, value = self.mind.S, self.mind.unit_of(VALUE, label)
        if value < 0:
            return 0.0
        ties = (S.find_tie(unit, value, EXPECT, 2) for unit in dict.fromkeys(path))
        return sum(float(S.w1(W_POS, tie).sum()) for tie in ties if tie >= 0)

    def read_path(self, path: str, cap: int = 0) -> str:
        """Watch the texts of a file or of every supported file under a folder; report what was lived. Each file read
        is an entry of the history (Entry: the moments it lived, so ``!unut`` takes it back, the words it lived and its
        last sentence). Asked to stop (Ctrl+C), it stops before the next sentence or dialogue: what was read until then
        stays lived and in the history."""
        import time
        if not os.path.exists(path):
            return "  böyle bir dosya ya da klasör yok: %s" % path
        files = [path] if os.path.isfile(path) else sorted(
            os.path.join(root, name) for root, _, names in os.walk(path) for name in names)
        start, lived, skipped, t0 = time.time(), 0, [], self.mind.t
        found: Dict[str, TextFile] = {}                  # how each text file was decoded (encoding_notes)
        reached, stopped = 0, False                      # files begun; asked to stop (Ctrl+C) on the way
        self.company(None)                               # F33: a document is no one's in particular
        try:
            for name in files:
                self.check()
                reached += 1
                first_moment, said = self.mind.t + 1, []
                try:
                    try:
                        if name.lower().endswith(".json"):
                            import json
                            f, found[os.path.basename(name)] = open_text(name)
                            with f:
                                pairs = dialogues_of(json.load(f))
                            for first, answer in pairs[:cap] if cap else pairs:
                                self.check()
                                self.watch(first, answer)
                                said += [first, answer]
                                lived += 1
                            if not pairs:
                                skipped.append(os.path.basename(name))
                            continue
                        if name.lower().endswith(".tsv"):
                            f, found[os.path.basename(name)] = open_text(name)
                            with f:
                                watched = 0
                                for raw in f:
                                    if cap and watched >= cap:
                                        break
                                    if "\t" in raw:
                                        self.check()
                                        first, answer = raw.rstrip("\n").split("\t", 1)
                                        self.watch(first, answer)
                                        said += [first, answer]
                                        lived += 1
                                        watched += 1
                            continue
                        encodings = []
                        text = read_document(name, encodings)
                        if encodings:
                            found[os.path.basename(name)] = encodings[0]
                    except Interrupted:
                        raise
                    except NotText as exc:
                        skipped.append("%s (metin dosyası değil: %s)" % (os.path.basename(name), exc))
                        continue
                    except Exception as exc:
                        skipped.append("%s (%s)" % (os.path.basename(name), exc))
                        continue
                    if not text:
                        skipped.append(os.path.basename(name))
                        continue
                    lines = [ln.strip() for ln in text.splitlines() if len(self.words(ln)) >= 2] \
                        if name.lower().endswith(CODE) else sentences(text, self.dil)   # code: each line an utterance
                    for i, sentence in enumerate(lines):
                        if cap and i >= cap:
                            break
                        self.check()
                        self.listen(sentence, IMITATED)
                        # the writer's sentence has ended: its end is perceived, with how it ended, as the end of any
                        # utterance heard (live, watch, !ogret lines); then a pause before the next one
                        self.end_of_turn(IMITATED, tone=self.tone(sentence))
                        self.mind.step([])
                        said.append(sentence)
                        lived += 1
                finally:                                 # what was lived of this file is one entry of the history
                    moments = list(range(first_moment, self.mind.t + 1))
                    if moments:
                        self.remember(READ, "[okudu] %s" % os.path.basename(name), moments, said)
        except Interrupted:
            stopped = True
        took = time.time() - start
        out = "%s  %d dosya, %d cümle/diyalog, %d an, %.0f sn (%.1f ms/an)" % (
            "\n" if stopped else "", reached - len(skipped), lived, self.mind.t, took,
            1000 * took / max(1, self.mind.t - t0))           # stopped: on a new line after the ^C
        if skipped:
            out += "\n  okunamayan: " + ", ".join(skipped[:10])
        for note in encoding_notes({name: f for name, f in found.items() if name not in skipped}):
            out += "\n  " + note
        if stopped:
            out += "\n  yarıda kesildi (Ctrl+C): okunduğu kadarı yaşandı ve geçmişte; dosyaların %d/%d tanesine " \
                   "başlanmıştı" % (reached, len(files))
        return out

    def teach(self, path: str) -> str:
        """``!ogret``: watch every line of a file, "soz<TAB>karsilik" dialogues or plain sentences (told by someone,
        imitated); the whole file is one entry of the history (``!unut`` takes it back). Reports what was lived. Asked
        to stop (Ctrl+C), it stops before the next line; what was watched until then stays lived and in the history."""
        if not os.path.exists(path):
            return "  böyle bir dosya yok: %s" % path
        if os.path.isdir(path):
            return "  bu bir klasör: %s (klasörler !oku ile okunur)" % path
        try:
            f, found = open_text(path)
        except NotText as exc:
            return "  okunmadı: %s metin dosyası değil (%s)" % (os.path.basename(path), exc)
        first_moment, said, n, stopped = self.mind.t + 1, [], 0, False
        try:
            with f:
                for raw in f:
                    raw = raw.rstrip("\n")
                    if not raw.strip():
                        continue
                    self.check()
                    if "\t" in raw:
                        first, answer = raw.split("\t", 1)
                        self.watch(first, answer)
                        said += [first, answer]
                    else:
                        self.live(raw, IMITATED, keep=False, lesson=True)
                        said.append(raw)
                    n += 1
        except Interrupted:
            stopped = True
        finally:
            moments = list(range(first_moment, self.mind.t + 1))
            if moments:
                self.remember(TAUGHT, "[öğretti] %s" % os.path.basename(path), moments, said)
        out = "%s  %d satır izlendi, %d an" % ("\n" if stopped else "", n, self.mind.t)
        for note in encoding_notes({os.path.basename(path): found}):
            out += "\n  " + note
        if stopped:
            out += "\n  yarıda kesildi (Ctrl+C): izlendiği kadarı yaşandı ve geçmişte"
        return out

    def observe(self, first: str, answer: str) -> None:
        """``!izle``: watch two others talk (see watch); the dialogue is one entry of the history."""
        start = self.mind.t + 1
        self.watch(first, answer)
        self.remember(WATCHED, "[izledi] %s || %s" % (first, answer), list(range(start, self.mind.t + 1)),
                      [first, answer])

    def ask(self, text: str, top: int = 5) -> str:
        twin = self.mind.clone()
        twin.knob["theta"] = 1e9                         # listening: the clone does not speak into the question
        for symbol in self.words(text):
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
        for symbol in self.words(text):
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
            pos, neg = twin.S.w1(W_POS, tie).tolist(), twin.S.w1(W_NEG, tie).tolist()
            out.append("  kaynaklara göre (w+ / w-): " + ", ".join(
                "%s %.2f/%.2f" % (name, p, n) for name, p, n in zip(PROVENANCE, pos, neg) if p or n))
            out.append("  kaynak güveni: " + ", ".join("%s %.2f" % (n, float(t)) for n, t in zip(PROVENANCE, twin.trust)))
        return "\n".join(out)

    def forget(self, number: int) -> str:
        """Retract an entry of the history exactly: each of its moments through the ledger, the latest first (a
        typed line, an answer, a file read or taught, a watched dialogue). Each moment is taken back whole, and an
        interrupted !unut is resumed by the next one: the moments already taken back are skipped (retracting one
        twice is refused by the core), so in the end every moment of the entry is taken back once."""
        entry = next((e for e in self.history if e.number == number), None)
        if entry is None:
            return "  böyle bir numara yok (!gecmis numaraları gösterir)"
        moments = entry.moments
        if not moments:
            return "  '%s' geri alınamaz: yaşanmış anı kayıtlı değil" % entry.text
        log = self.mind.S.log
        if any(m not in log for m in moments):
            return "  '%s' geri alınamaz: bu zihin deneyim defteri tutmuyor (--defter-yok ya da defteri olmadan " \
                   "yüklendi)" % entry.text
        left = [m for m in moments if not taken_back(log, m)]
        before = "" if len(left) == len(moments) else \
            "; %d anı önceki !unut geri almıştı" % (len(moments) - len(left))
        removed, done = 0, 0
        try:
            with _indices_rebuilt_once(self.mind.S):
                for m in reversed(left):
                    self.check()                         # asked to stop: between two moments, each taken back whole
                    report = self.mind.retract(m)
                    removed += sum(v for v in report.values() if isinstance(v, int))
                    done += 1
        except Interrupted:
            return "\n  '%s' yarıda kesildi (Ctrl+C): %d/%d anı geri alındı (%d kayıt); kalanını '!unut %d' geri alır" % (
                entry.text, len(moments) - len(left) + done, len(moments), removed, number)
        self.history.remove(entry)
        if not removed and not before:
            return "  '%s': %d an geri alındı ama çekirdekte çıkarılacak kaydı yoktu" % (entry.text, len(moments))
        return "  '%s' geri alındı (kanıtı çekirdekten çıkarıldı; %d kayıt%s)" % (entry.text, removed, before)


def night_report(night, names: Dict[str, str] = None) -> List[str]:
    """What the mind can tell of a night: its dreams, which earlier dreams reality has judged, its trust in its
    imagination before and after."""
    lines = []
    for chain in night["dreams"]:
        lines.append("  rüya:")
        for d in chain:
            new = sum(1 for claim in d["claims"] if claim[3] + claim[4] == 0)
            note = "   (yeni iddia yok: bilinen hatırlandı)" if not new else ""
            for gist in d.get("seen", []):
                lines.append("    görüntü yeniden yaşandı: %s" % (names.get(gist, "bir görüntü") if names else "bir görüntü"))
            if d["other"]:
                lines.append("    (%s yerine %s): %s   [gerçeklik %.2f]%s" % (d["about"], d["other"],
                                                                          " ".join(d["words"]), d["reality"], note))
            else:
                lines.append("    %s%s" % (" ".join(d["words"]), note))
    for v, d in night["judged"]:
        lines.append("  %s: %s" % ("doğru çıktı" if v == "dogru" else "yanlış çıktı", " ".join(d["words"])))
    srt = night["sorting"]
    lines.append("  uyku raporu: hayal kanıtından gerçek kanıtla örtüşen %.1f, çelişen %.1f, sınanmamış %d; hayale "
                 "güven %.2f -> %.2f" % (srt["true"], srt["false"], int(srt["open"]), srt["trust_before"],
                                                       srt["trust"]))
    return lines


class Life:
    """Online life: from the moment the session starts the mind lives in its own time, not only when it is asked.

    What you type arrives as an event of that time and is heard in order; what the mind says comes out word by word
    as it says it. Between your lines it lives on, one silent moment at a time, each lived as any moment: when its own
    decision in one of them emits words, they are said (Session.speak) and it is an entry of the history (kind
    "spoke"); nothing is said for it (no first speech after a silence, no question about a word it wants to know).

    What it does with a silent moment is its own decision where the mind has one (the owner's decision of
    2026-11-22: no ability is to hang on a command or a clock): to look up what it found it did not know (F47, when
    the owner gave it somewhere to look) and to think about what it last took in (F48). Without F48 (--saatli, or an
    older configuration) the program's rhythm thinks for it: once per silence, after THINK_AFTER seconds. Sleep
    (dreams, then sorting) is still the program's rhythm, once per silence after SLEEP_AFTER seconds: what makes a
    mind need sleep is not worked out yet, and a clock is said to be a clock. --yasamsiz turns the whole life off.
    Commands are handled in the same stream.

    Ending it (Ctrl+C, !cik, the end of input) asks the session to stop: a long command stops at its next sentence,
    line or moment (Session.check), its life ends, and only then is the mind saved; a life that cannot be stopped is
    left unsaved (see main)."""

    STOP = object()
    WAIT = 30.0                            # seconds the end of a session waits for a running command to stop
    THINK_AFTER = 10.0                     # innate rhythm (declared, awaiting the owner's decision): one inner thought
    SLEEP_AFTER = 600.0                    # ... and one sleep per silence, after these many seconds of it

    def __init__(self, session, handle, lock, beat: float = 2.0, think_after: float = THINK_AFTER,
                 sleep_after: float = SLEEP_AFTER, pace: float = 0.05) -> None:
        import queue
        import threading
        import time
        self.session, self.handle, self.lock = session, handle, lock
        self.beat, self.think_after, self.sleep_after, self.pace = beat, think_after, sleep_after, pace
        self.inbox = queue.Queue()
        self.idle_since = time.time()
        self.thought, self.slept = None, False
        self.handled = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        session.voice = self.say
        session.background = True                        # F50: in its online life what it sends for is fetched
                                                         # while it goes on living (it reads it in a silent moment)

    def say(self, word: str) -> None:
        import time
        print(" " + word, end="", flush=True)
        time.sleep(self.pace)

    def start(self) -> None:
        self.session.stop.clear()
        self.thread.start()

    def put(self, line: str) -> None:
        self.handled.clear()
        self.inbox.put(line)

    def stop(self, wait: float = None) -> bool:
        """End its life: the session is asked to stop (a running command stops at its next sentence, line or moment),
        then the thread is waited for, at most ``wait`` seconds (Life.WAIT; a second Ctrl+C ends the waiting). True
        when the thread has ended: only then may the mind be saved, nothing changes it any more."""
        self.session.stop.set()
        self.inbox.put(self.STOP)
        deadline = time.time() + (self.WAIT if wait is None else wait)
        try:
            while self.thread.is_alive() and time.time() < deadline:
                self.thread.join(0.1)
        except KeyboardInterrupt:                        # asked again: no more waiting
            pass
        return not self.thread.is_alive()

    def _run(self) -> None:
        import queue
        try:
            while not self.session.stop.is_set():
                try:
                    line = self.inbox.get(timeout=self.beat)
                except queue.Empty:
                    line = None
                if line is self.STOP:
                    break
                if line is not None:
                    with self.lock:
                        going_on = self.handle(line)
                    self.idle_since, self.thought, self.slept = time.time(), None, False
                    if going_on is False:
                        self.done.set()
                    self.handled.set()
                    if going_on is False:
                        break
                    continue
                try:
                    self._silence(time.time() - self.idle_since)
                except Interrupted:                      # asked to stop while it thought or slept
                    break
                except Exception as exc:  # its life goes on
                    print("\n  hata: %s\n> " % exc, end="", flush=True)
        finally:
            self.handled.set()                           # nobody waits for a line its life will not handle

    def _silence(self, quiet: float) -> None:
        from .dusunce import sleep as dream_sleep, think, topic_of
        if self.session.stop.is_set():
            return
        with self.lock:
            start = self.session.mind.t + 1
            before = len(self.session.looks)
            acts = self.session.silent_moment()          # a silent moment of its life, lived as any moment (F47: a
            if acts:                                     # look it decided on is carried out); its own decision
                self._speak(acts, start)                 # emitted something: it is said
            elif len(self.session.looks) > before:
                print("\n> ", end="", flush=True)
        if getattr(self.session.mind.cfg, "think", False):      # F48: when to think is the mind's own decision
            pass                                                   # (Session.silent_moment let it happen above)
        elif self.thought is None and quiet >= self.think_after:   # the program's rhythm: one thought per silence
            with self.lock:
                topic = topic_of(self.session)
                words_, reality = think(self.session, topic) if topic else ([], 0.0)
                self.thought = (topic, words_, reality)
        if not self.slept and self.sleep_after and quiet >= self.sleep_after:   # innate rhythm: one sleep
            with self.lock:
                night = dream_sleep(self.session)
            self.slept = True
            print("\n  (uyudum)\n%s\n> " % "\n".join(night_report(night, self.session.seen)), end="", flush=True)

    def _speak(self, acts: List[Event], start: int) -> None:
        """What the mind decided in a silent moment of its life is said, word by word, until it decides nothing
        (Session.speak); it is its own utterance, an entry of the history, and the answer before it is spoken over
        (feedback no longer reaches that answer)."""
        print("\n  zihin (kendiliğinden):", end="", flush=True)
        self.session.answered = None
        said = self.session.speak(acts, self.say)
        self.session.remember(SPOKE, "[kendiliğinden] %s" % " ".join(said), list(range(start, self.session.mind.t + 1)),
                              said=said)
        self.session.open_turn = True                    # its own utterance is a turn: what comes next begins after it
        print("\n> ", end="", flush=True)


def pieces_for(mind, choice: str = None):
    """Word pieces for this mind: a mind raised with pieces keeps them, a mind raised without them keeps living
    without them; a new mind lives without them unless asked ("cevahir": the repository's own tokenizer, or a
    ByteLevelBPE json). Pieces are opt-in for new minds: raised with them from birth, a mind grew about five times
    as many ties per moment (55 against 11) and ran out of memory at 219k moments (NOTES)."""
    import os
    lived = mind.ch.has(FACETS) and mind.ch.sizes()[mind.ch.id(FACETS)] > 0
    choice = choice or ("cevahir" if lived else "yok")
    if choice == "yok":
        return None
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if choice == "cevahir":
        from .parca import own_pieces
        return own_pieces()
    from tokenizer_management.byte_bpe import ByteLevelBPE
    return ByteLevelBPE.load(choice)


def home_mind() -> str:
    """Where the one mind of this computer lives: $CEVAHIR_HOME/zihin/cevahir (default ~/.cevahir)."""
    folder = home_folder()
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "cevahir")


def ledger_of(mind) -> str:
    """The file of a mind's experience ledger (None when it keeps none, or keeps it in memory)."""
    from .store import DiskLog
    log = getattr(mind.S, "log", None)
    return log.path if isinstance(log, DiskLog) else None


def drop_ledger(mind) -> None:
    """Delete the ledger file of a mind's session: one this program made for it (defter-*.log, or the live copy a
    load makes of a saved ledger, *.live-*), never a saved mind's own ledger."""
    path = ledger_of(mind)
    name = os.path.basename(path) if path else ""
    if (name.startswith("defter-") or ".live-" in name) and os.path.exists(path):
        os.remove(path)


def _exists(path: str) -> bool:
    import os
    base = path[:-6] if path.endswith(".zihin") else path
    return os.path.exists(base + ".zihin") or os.path.exists(base + ".zihin.gz")


def feelings(mind: Mind) -> str:
    """The mind's own state as it computed it in the last moment (functional signals, see design section 2.6)."""
    f = getattr(mind, "felt", None) or {}
    names = (("surprise", "şaşkınlık"), ("relief", "rahatlama"), ("conflict", "çatışma"), ("load", "yük"))
    parts = ["%s %.2f" % (tr, float(f.get(en, 0.0))) for en, tr in names]
    parts.append("dürtü %.1f" % float(f.get("drive", 0.0)))
    return "  (his: " + ", ".join(parts) + ")"


def main(argv=None) -> None:
    import os
    import torch
    torch.set_num_threads(1)                             # the substance's operations are small: one thread is faster
    ap = argparse.ArgumentParser(prog="python -m zihin")
    ap.add_argument("--load", help="a saved mind to live on (otherwise the mind of this computer's home folder)")
    ap.add_argument("--yeni", action="store_true", help="start a new, empty mind")
    ap.add_argument("--kayit", help="where the mind is kept (default: the home folder, see home_mind)")
    ap.add_argument("--kaydetme", action="store_true", help="do not save what was learned in this session")
    ap.add_argument("--his", action="store_true", help="show the mind's feelings after each answer")
    ap.add_argument("--kim", help="who is talking to the mind (asked at the start if not given)")
    ap.add_argument("--yasamsiz", action="store_true", help="no life clock: while you are silent the mind lives no "
                                                            "moments, does not think and does not sleep")
    ap.add_argument("--kaynak", help="a folder of your own documents the mind may turn to by itself, when it decides "
                                     "to; it is asked before the web (which it reads only with --ag)")
    ap.add_argument("--ag", action="store_true", help="the mind may read the web by itself: what it found it did not "
                                                      "know, when it decides to (the word it looks up goes to the "
                                                      "search service; every page read is printed)")
    ap.add_argument("--agsiz", action="store_true", help="its own looks never go to the web (only --kaynak is read; "
                                                         "nothing leaves this machine)")
    ap.add_argument("--saatli", action="store_true", help="the program's clock decides when it thinks in a silence "
                                                          "(one thought ten seconds into each), not the mind itself")
    ap.add_argument("--onbelleksiz", action="store_true", help="do not keep fetched web pages on disk (by default they "
                                                               "are kept under the mind's folder, in 'ag', for 30 days)")
    ap.add_argument("--cevrimdisi", action="store_true", help="read the web only from what was kept before (no request "
                                                              "leaves this machine; what is not kept is not read)")
    ap.add_argument("--parca", help="word pieces as facets: 'cevahir' (the repository's own tokenizer: root, suffixes, "
                                    "stem, last syllable; kept by minds raised with pieces, opt-in for new minds: "
                                    "they multiply the ties per moment), 'yok' (none), or a ByteLevelBPE json")
    ap.add_argument("--defter-yok", dest="no_ledger", action="store_true",
                    help="no experience ledger: smaller and faster, but !unut cannot retract")
    ap.add_argument("--dil", choices=DIL, help="how a NEW mind's text is lower-cased: 'tr' (default; Turkish rules, "
                                               "I -> ı, İ -> i) or 'diger' (plain lower case, for other languages); "
                                               "a mind keeps the convention it was raised with")
    ap.add_argument("--variant", default="full", choices=sorted(VARIANTS),
                    help="'full' (default): the mind lives with every mechanism that was built; a loaded mind is "
                         "given those it was raised without (but for what --kapat keeps off and the ways of reading "
                         "its own trials put away). 'speaking': the configuration before the mechanisms of --ac; a "
                         "loaded mind keeps its own settings")
    ap.add_argument("--ac", default="", help="switch on mechanisms by name (comma separated), for a loaded or a new "
                    "mind: with --variant speaking any of those the full configuration has (%s); with the full "
                    "configuration, one that --kapat or the mind's own trials had put off" % ", ".join(NEWER + STUDIED))
    ap.add_argument("--kapat", default="", help="keep mechanisms off by name (comma separated): remembered with the "
                    "mind, until --ac names them again; what cannot work without one of them is kept off with it")
    ap.add_argument("--pencere", type=int, help="how many items a NEW mind holds at once (its window, 3 to 15; "
                    "default 7); a mind keeps the window it has")
    args = ap.parse_args(argv)
    keep_at = clean_path(args.kayit) if args.kayit else home_mind()    # ~ and quotes as in the commands
    ledgers = os.path.dirname(os.path.abspath(keep_at))   # session ledgers live next to the kept mind
    source = None if args.yeni else (clean_path(args.load) if args.load else (keep_at if _exists(keep_at) else None))
    if source:
        print("zihin yükleniyor: %s ..." % source, flush=True)
        mind = load(source, live_log=_log_path(ledgers), ledger=not args.no_ledger)   # a saved ledger: copied there
        if not args.no_ledger and not mind.S.keep_log:      # a mind saved without a ledger: keep one from now on
            from .store import DiskLog
            mind.cfg.keep_log, mind.cfg.log_path = True, _log_path(ledgers)
            mind.S.keep_log = True
            mind.S.log = DiskLog(mind.cfg.log_path, keep=max(4, mind.cfg.undo_depth + 2))
        if args.dil and args.dil != getattr(mind, "dil", "tr"):
            print("  (bu zihin '%s' harf kuralıyla büyüdü ve onu korur; --dil %s yeni bir zihin içindir)" % (
                getattr(mind, "dil", "tr"), args.dil))
        if args.pencere and args.pencere != mind.knob["F"]:
            print("  (bu zihnin penceresi %d öğe ve onu korur; --pencere yeni bir zihin içindir)" % mind.knob["F"])
    else:
        cfg = VARIANTS[args.variant](ledgers)
        if args.pencere:                                    # the window of a new mind (the range of its own knob)
            from .act import KNOB_RANGE
            cfg.F = max(KNOB_RANGE["F"][0], min(KNOB_RANGE["F"][1], args.pencere))
        mind = Mind(cfg)
        mind.dil = args.dil or "tr"                         # kept with the mind (persist: minds saved before are tr)
        if args.no_ledger:
            mind.S.log, mind.S.keep_log, mind.cfg.keep_log, mind.cfg.log_path = {}, False, False, None
        mind.add_channel(CHANNEL, "act")
    asked = [k.strip() for k in args.ac.split(",") if k.strip()]
    reconfigured = configure(mind, args.variant, asked, [k.strip() for k in args.kapat.split(",") if k.strip()],
                             loaded=bool(source), look=bool(args.ag or args.kaynak))
    asked_look = "look" in asked
    if args.saatli and getattr(mind.cfg, "think", False):
        mind.cfg.think = False                           # the program's clock decides when it thinks (the order before
        print("  (--saatli: bu oturumda ne zaman düşüneceğine programın saati karar verir, zihnin kendisi değil)")  # F48)
    session = Session(mind, pieces_for(mind, args.parca))
    from .ag import Net                                  # the one way out to the network: polite, retrying, kept
    session.net = Net(cache=None if args.onbelleksiz else os.path.join(home_folder(), "ag"), offline=args.cevrimdisi)
    lang = "tr" if getattr(mind, "dil", "tr") == "tr" else "en"
    if getattr(mind.cfg, "look", False):                 # F47: the owner lets the mind look things up by itself
        from .web import Folder, Sources, web_sources
        own = [Folder(clean_path(args.kaynak), getattr(mind, "dil", "tr"))] if args.kaynak else []
        web = (args.ag or asked_look) and not args.agsiz  # the web only when the owner says so (--ag)
        session.look_source = Sources(own + (web_sources(lang, session.net).sources if web else []))
        session.may_look = not args.yasamsiz and bool(session.look_source.sources)
        where = (["klasörün (%s)" % args.kaynak] if args.kaynak else []) + (
            ["ağ: aradığı kelime arama servisine gider (GOOGLE_API_KEY yoksa Vikipedi, sonra Vikisözlük)"]
            if web else [])
        print("  (bu zihin, cevabını bilmediğini kendi fark ettiği şeylere SESSİZLİKTE KENDİ KARARIYLA bakar; baktığı "
              "yer: %s; okuduğu her sayfa yazılır, yalnız okur; !bakma kapatır, !bak açar, Ctrl+C durdurur%s)" % (
                  ", sonra ".join(where) if where else "yok (--ag ya da --kaynak ile bir yer ver)",
                  "" if session.may_look else "; şimdi bakmaz (yaşam saati kapalı ya da bakacak yer yok)"))
    session.explain = True                               # !neden explains the last answer word by word
    interactive = sys.stdin.isatty()
    if interactive and not args.yasamsiz and not args.saatli and not getattr(mind.cfg, "think", False) \
            and "think" not in mind.owner_off:
        mind.cfg.think = True                            # F48: in its online life, when to think is its own decision
        print("  (sessizlikte ne zaman düşüneceğine bu zihin kendisi karar verir: beklediğine aykırı bir şey "
              "duyduğunda düşünür, düşüncelerinin neye vardığından öğrenir; program saati kullanılmaz, --saatli eski "
              "düzeni verir)")
    name = args.kim
    if not name and interactive:
        name = input("Seninle konuşan kim? (adını yaz, boş bırakırsan 'biri') > ").strip()
    if name:
        session.partner = session.lower(name).split()[0]
    chat = interactive                                   # at a terminal you talk to it right away
    autosave = not args.kaydetme
    turns_since_save = 0

    saved_at = [-1 if source and reconfigured else mind.t]   # (a loaded mind whose mechanisms were changed is kept
                                                             # even when nothing was lived)

    def keep(reason: str) -> bool:
        """Save what was learned (the substance as it is now, without the ledger). A save that does not finish
        leaves the mind saved before as it was (persist.save); it is said, and the session goes on to its end."""
        if not autosave or session.mind.t == saved_at[0]:   # nothing new was lived: nothing to keep
            return True
        started = time.time()
        try:
            export_portable(session.mind, keep_at)
        except (Exception, KeyboardInterrupt) as exc:
            print("  kaydedilemedi (%s): %s; en son kaydedilen zihin olduğu gibi duruyor" % (
                reason, str(exc) or type(exc).__name__), flush=True)
            return False
        saved_at[0] = session.mind.t
        print("  zihin kaydedildi (%s): %s.zihin, %d an, %.0f sn" % (
            reason, keep_at, session.mind.t, time.time() - started), flush=True)
        return True

    print("Cevahir — %d an yaşamış bir zihin. %s" % (
        mind.t, "Yaz, konuşalım. Komutlar: !his !yas !dogru !yanlis !unut !cik" if chat else
        "anlat, sor (?...), !konus, !neden, !unut, !kaydet, !cik"))
    if autosave:
        print("  (her sohbetin sonunda öğrendikleri kendiliğinden kaydedilir: %s.zihin)" % keep_at)
    lines = sys.stdin if not interactive else iter(lambda: input("> "), None)
    lock = threading.RLock()
    if interactive and not args.yasamsiz:              # online: it lives from now on, in its own time
        state = {"chat": chat, "turns": 0}

        def handle(line):
            out = _one_line(session, line, state["chat"], args, keep, state["turns"])
            if out is None:
                return False
            if isinstance(out, tuple):
                state["chat"], state["turns"] = out
            else:
                state["turns"] = out
            return True

        life = Life(session, handle, lock)
        life.start()
        print("  (çevrimiçi: oturum açık kaldıkça sessizlikte de an an yaşar; kendi kararı bir şey söylerse söylenir. "
              "Doğuştan ritim, sahibinin kararını bekleyen bir program ayarı: her sessizlikte %g sn sonra bir kez iç "
              "sesle düşünür, %g dk sonra bir kez uyur. Kapatmak: --yasamsiz)" % (Life.THINK_AFTER,
                                                                                  Life.SLEEP_AFTER / 60))
        try:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                life.put(line)
                while not life.handled.wait(0.2) and life.thread.is_alive():   # its answer comes out before the
                    pass                                 # next prompt; Ctrl+C is heard while it waits (on every system)
                if life.done.is_set() or not life.thread.is_alive():
                    break
        except (KeyboardInterrupt, EOFError):
            if life.handled.is_set():                    # a running command's report begins on a new line itself
                print()
        finally:
            if life.stop():                              # its life has ended: nothing changes the mind any more
                keep("sohbet sonu")
                drop_ledger(session.mind)                # kept without it (or not kept: --kaydetme): dead weight
            else:
                print("\n  zihin durdurulamadı: bir komutu %g saniyede bitmedi. Bu oturumda yaşadıkları kaydedilmedi; "
                      "en son kaydedilen zihin olduğu gibi duruyor." % Life.WAIT, flush=True)
                abandon(session.mind)
        return
    interrupt = _Interrupt(session)                      # Ctrl+C during a command: it stops between two moments
    try:
        with interrupt:
            _session_loop(session, lines, chat, args, keep, lock, None, interrupt)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        if interrupt.forced:                             # stopped at once, perhaps inside a moment: not kept
            print("  komut beklenmeden kesildi; zihin bir anın ortasında kalmış olabilir, bu haliyle kaydedilmedi. "
                  "En son kaydedilen zihin olduğu gibi duruyor.", flush=True)
        else:
            keep("sohbet sonu")
        drop_ledger(session.mind)                        # kept without it, or not kept: dead weight either way


def _session_loop(session, lines, chat, args, keep, lock=None, clock=None, interrupt=None) -> None:
    import contextlib
    turns_since_save = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if clock is not None:
            clock.heard()
        with (lock if lock is not None else contextlib.nullcontext()):
            if interrupt is not None:
                interrupt.busy = True
            try:
                turns_since_save = _one_line(session, line, chat, args, keep, turns_since_save)
            finally:
                if interrupt is not None:
                    interrupt.busy = False
            if turns_since_save is None:
                break
            if isinstance(turns_since_save, tuple):         # the chat toggle came back with the new mode
                chat, turns_since_save = turns_since_save
        if session.stop.is_set():                        # asked to stop (Ctrl+C) during a command: the session ends
            break


class _Interrupt:
    """Ctrl+C in a session without a life thread (--yasamsiz, or lines from a pipe or a file), where commands run in
    the main thread. While a command runs, the first Ctrl+C asks it to stop at its next sentence, line or moment
    (Session.check): the mind is not cut inside a moment, the session ends after that command, and the mind is kept.
    A second Ctrl+C while it runs stops it at once: the mind may then be cut inside a moment (``forced``) and is not
    saved. Between commands Ctrl+C ends the session at once, as before. A signal handler can be set only in the
    main thread; anywhere else nothing changes."""

    def __init__(self, session) -> None:
        self.session, self.busy, self.forced, self.installed, self.previous = session, False, False, False, None

    def __enter__(self) -> "_Interrupt":
        if threading.current_thread() is threading.main_thread():
            self.previous = signal.signal(signal.SIGINT, self.heard)
            self.installed = True
        return self

    def __exit__(self, *exc) -> bool:
        if self.installed:
            signal.signal(signal.SIGINT, self.previous if self.previous is not None else signal.default_int_handler)
            self.installed = False
        return False

    def heard(self, signum, frame) -> None:
        if not self.busy:                                # between commands: the session ends at once
            raise KeyboardInterrupt
        if self.session.stop.is_set():                   # asked a second time: at once, wherever it is
            self.forced = True
            raise KeyboardInterrupt
        self.session.stop.set()                          # the command stops at its next check


def abandon(mind) -> None:
    """When the life thread could not be stopped: the mind is not saved, and its session's ledger is not left on
    disk. The ledger is pointed at the null device first, so what the thread still writes goes nowhere and the file is
    not made again once deleted; then it is deleted. A save the thread itself may be making is left alone: the mind
    saved before stays until that save is whole, and what an unfinished save leaves is removed by the next load or
    save of that mind (persist._finish_save)."""
    from .store import DiskLog
    log = getattr(mind.S, "log", None)
    if isinstance(log, DiskLog):
        path, log.path = log.path, os.devnull
        name = os.path.basename(path)
        if (name.startswith("defter-") or ".live-" in name) and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _one_line(session, line, chat, args, keep, turns_since_save):
    """One line of the conversation. Returns the updated counter, (chat, counter) after !konus, None to quit."""
    if True:
        try:
            if not line.lstrip("!").split(" ")[0] in ("dogru", "doğru", "good", "iyi", "yanlis", "yanlış", "bad", "kotu",
                                                       "kötü", "his", "feel") or not line.startswith("!"):
                session.close_turn()                     # any other input ends the answered turn first
            if line.startswith("?") and not chat:
                print(session.ask(line[1:].strip()))
            elif line.startswith("!"):
                cmd, _, rest = line[1:].partition(" ")
                if cmd in ("cik", "quit"):
                    return None
                elif cmd in ("gordum", "saw"):
                    session.live(rest, SENSED)
                elif cmd in ("izle", "watch"):
                    first, _, answer = rest.partition("||")
                    session.observe(first.strip(), answer.strip())
                elif cmd in ("ogret", "teach"):
                    print(session.teach(clean_path(rest)))
                elif cmd in ("oku", "read"):
                    target, cap = path_and_cap(rest)
                    print(session.read_path(target, cap))
                elif cmd in ("dogru", "doğru", "good", "iyi"):
                    print(session.feedback(True))
                elif cmd in ("yanlis", "yanlış", "bad", "kotu", "kötü"):
                    print(session.feedback(False))
                elif cmd in ("konus", "chat"):
                    chat = not chat
                    print("  sohbet modu %s" % ("açık" if chat else "kapalı"))
                    return (chat, turns_since_save)
                elif cmd in ("neden", "why"):
                    if rest.strip():                     # !neden <metin> => <kelime>: the white-box belief
                        prefix, _, value = rest.partition("=>")
                        print(session.why(prefix.strip(), value.strip()))
                    else:                                # !neden: where each word of the last answer came from
                        from .neden import report
                        print(report(session.sources))
                elif cmd in ("gecmis", "history"):
                    for entry in session.history:
                        print("  %3d  %s" % (entry.number, entry.text))
                elif cmd in ("unut", "forget"):
                    if rest.strip().isdigit():
                        print(session.forget(int(rest)))
                    else:
                        print("  !unut bir numara ister: geri alınacak girdinin numarası (!gecmis gösterir), "
                              "örneğin !unut 3")
                elif cmd in ("arastir", "araştır", "research"):
                    from .web import research
                    report = research(session, rest.strip())
                    print("  okudum: %s" % report)
                elif cmd in ("sor_hepsi", "ask_all"):     # many questions at the same time, none of them lived
                    from .paralel import answers
                    target = clean_path(rest)
                    if not os.path.isfile(target):
                        print("  kullanım: !sor_hepsi <dosya>  (her satır bir soru)")
                    else:
                        f, _ = open_text(target)
                        with f:
                            questions = [line.strip() for line in f if line.strip()]
                        started = time.time()
                        said = answers(session, questions)
                        for question, answer in zip(questions, said):
                            print("  %s\n      -> %s" % (question, answer))
                        print("  %d soru, %.1f sn; kopyalar cevapladı: zihin bunları yaşamadı (öğrenmedi, geri bildirim "
                              "alamaz)" % (len(questions), time.time() - started))
                elif cmd in ("dusun_hepsi", "düşün_hepsi", "think_all"):   # many thoughts at the same time
                    from .paralel import thoughts
                    topics = [t.strip() for t in rest.split(",") if t.strip()]
                    if not topics:
                        print("  kullanım: !dusun_hepsi konu1, konu2, ...")
                    else:
                        for topic, (words_, reality) in zip(topics, thoughts(session, topics)):
                            print("  %s: %s  (gerçeklik %.2f)" % (topic, " ".join(words_) or "(sessiz)", reality))
                        print("  kopyalar düşündü: zihin bunları yaşamadı")
                elif cmd in ("varsayim", "varsayım", "conjectures"):   # F59: what it conjectured by itself
                    from . import varsayim
                    if not getattr(session.mind.cfg, "conjectures", False):
                        print("  (varsayım kapalı: --ac conjectures ile açılır)")
                    else:
                        for line in varsayim.report(session.mind):
                            print(line)
                elif cmd in ("deney", "trials"):          # F57: the trials of its own ways of reading
                    from . import deney
                    if not getattr(session.mind.cfg, "trials", False):
                        print("  (deney kapalı: --ac trials ile açılır)")
                    else:
                        for line in deney.report(session.mind):
                            print(line)
                elif cmd in ("hata", "faults"):           # F49: what it learned about its own ways, its last faults
                    from . import hata
                    if not getattr(session.mind.cfg, "faults", False):
                        print("  hata izi kapalı (bu zihin --ac faults ile açılmadı); kayıt:")
                    print("\n".join(hata.report(session.mind)))
                elif cmd in ("ag", "ağ", "net"):          # what can be reached from here (nothing is read into the mind)
                    from .ag import check
                    print("\n".join(check(session.net, "tr" if session.dil == "tr" else "en")))
                elif cmd in ("oku_url", "read_url"):      # web pages by their addresses, read as documents are
                    from .web import read_pages
                    urls = [u for u in rest.split() if u.lower().startswith(("http://", "https://"))]
                    if not urls:
                        print("  kullanım: !oku_url <adres> [adres ...]  (http ya da https)")
                    else:
                        start = session.mind.t + 1
                        report = read_pages(session, urls, session.net)
                        if session.mind.t >= start:
                            session.remember(READ, "[ağdan okudu] %s" % " ".join(urls),
                                             list(range(start, session.mind.t + 1)))
                        print("  %d cümle, %d an" % (report["sentences"], report["moments"]))
                elif cmd in ("kendini_gelistir", "kendini_geliştir", "improve"):
                    from .web import research, wanted_topics
                    topics = wanted_topics(session.mind, CHANNEL, int(rest) if rest.strip().isdigit() else 3)
                    print("  en yüksek konu puanı (ayırt edici ve az bilinen): %s" % ", ".join(topics))
                    for topic in topics:
                        report = research(session, topic)
                        print("  %s: %s yoluyla %s cümle, kazanç %s (yollar: %s)" % (
                            topic, report.get("way"), report.get("sentences"), report.get("gain_per_100_moments"),
                            report.get("ways")))
                elif cmd in ("bak", "bakma", "look", "nolook"):   # F47: may it carry out its own looks?
                    if not getattr(session.mind.cfg, "look", False):
                        print("  bu zihin kendi kararıyla bakmıyor (açmak için programı --ac look ile başlat)")
                    else:
                        session.may_look = cmd in ("bak", "look")
                        print("  kendi kararıyla bakması %s" % ("açık" if session.may_look else "kapalı"))
                elif cmd in ("his", "feel"):
                    print(feelings(session.mind))
                elif cmd in ("gor", "gör", "see"):
                    from .gorme import features
                    image = clean_path(rest)
                    gist, facets = features(image)
                    session.view = (gist, facets)
                    session.seen[gist] = os.path.basename(image)
                    print("  görüyorum (%d yön); bir sonraki sözün bu resimle birlikte algılanacak" % len(facets))
                elif cmd in ("dinle", "hear"):
                    from .isitme import features as hearing
                    gist, facets = hearing(clean_path(rest))
                    session.hearing = (gist, facets)
                    print("  duyuyorum (%d yön); bir sonraki sözün bu sesle birlikte algılanacak" % len(facets))
                elif cmd in ("dusun", "düşün", "think"):
                    from .dusunce import think, topic_of
                    topic = rest.strip() or topic_of(session)
                    if not topic:
                        print("  düşünecek bir konu yok")
                    else:
                        thought, reality = think(session, topic)
                        print("  iç ses (konu: %s): %s   [gerçeklik %.2f]" % (topic, " ".join(thought) or "…",
                                                                           reality))
                elif cmd in ("kim", "who"):
                    if rest.strip():
                        session.partner = session.lower(rest.strip()).split()[0]
                    print("  konuşan: %s" % session.partner)
                elif cmd in ("uyu", "sleep"):
                    from .dusunce import sleep as dream_sleep
                    print("\n".join(night_report(dream_sleep(session, int(rest) if rest.strip().isdigit() else 3),
                                                  session.seen)))
                elif cmd in ("kendin", "introspect"):
                    from .kendilik import self_report
                    print(self_report(session.mind, CHANNEL))
                elif cmd in ("yas", "yaş", "age"):
                    print(profile(session.mind))
                elif cmd in ("boyut", "size"):
                    size = session.mind.size()
                    size["units"], size["ties"] = live_counts(session.mind)    # what !unut took back is not counted
                    print("  %d an, %s" % (session.mind.t, size))
                elif cmd in ("kaydet", "save", "tasinabilir", "portable", "yukle", "yükle", "load") \
                        and not clean_path(rest):
                    print("  bir yol yazın: !%s <yol>" % cmd)
                elif cmd in ("kaydet", "save"):
                    print("  kaydedildi: %s" % save(session.mind, clean_path(rest)))
                elif cmd in ("tasinabilir", "portable"):
                    print("  taşınabilir kopya: %s" % export_portable(session.mind, clean_path(rest)))
                elif cmd in ("yukle", "yükle", "load"):
                    replaced = session.mind
                    fresh = Session(load(clean_path(rest)), session.pieces, stop=session.stop)
                    session.__dict__.update(fresh.__dict__)     # the same session object: autosave follows it
                    drop_ledger(replaced)                       # the mind it replaces is gone, and its ledger
                    print("  yüklendi: %d an" % session.mind.t)
                else:
                    print("  bilinmeyen komut")
            elif chat:
                if getattr(session, "voice", None) is not None:      # online: words come out as they are said
                    print("  zihin:", end="", flush=True)
                    answer = session.reply(line, say=session.voice)
                    print(" (sessiz)" if answer == "(sessiz)" else "", flush=True)
                else:
                    print("  zihin: %s" % session.reply(line))
                for note in session.look_notes:              # F50: what a look of its own did in this turn
                    print("    %s" % note)
                session.look_notes.clear()
                if args.his:
                    print(feelings(session.mind))
                turns_since_save += 1
                if turns_since_save >= 50:               # a long conversation is kept along the way too
                    keep("ara kayıt")
                    turns_since_save = 0
            else:
                session.live(line, TOLD)
        except Interrupted:                              # a long command asked to stop (Ctrl+C) between two moments
            print("\n  yarıda kesildi (Ctrl+C): o ana kadar yaşadıkları yaşandı")
        except Exception as exc:  # keep the session alive
            print("  hata: %s" % exc)
    return turns_since_save


if __name__ == "__main__":
    main()
