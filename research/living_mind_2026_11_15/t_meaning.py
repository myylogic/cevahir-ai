# -*- coding: utf-8 -*-
"""T-meaning: does the mind know that merhaba, selam, selamün aleyküm and hello are the same? A public baseline.

    python research/living_mind_2026_11_15/t_meaning.py [--seeds 6] [--days 2 4] [--pieces both]
    python research/living_mind_2026_11_15/t_meaning.py --mind PATH      # a saved mind, read only, aggregates only

The world (seeded, public, generated here). Greetings start conversations and are answered:

* a frequent variant, "merhaba" (16 a day), answered "merhaba , hoş geldin", "merhaba , nasılsın ?" or "merhaba !";
* a medium one, "selam" (6 a day), answered by repeating it or with "merhaba , hoş geldin";
* a rare one, "selamlar", seen 1 + seed % 3 times in a whole life;
* a two-word one with its own conventional reply, "selamün aleyküm" -> "aleyküm selam" (4 a day);
* a foreign one, "hello", seen in one bilingual exchange only ("hello" -> "merhaba , hoş geldin");
* one never seen in a dialogue, "hi", told once in a sentence as the same: "hi , merhaba demektir .";
* a control never met at all, "merhabalar" (built like merhaba: only word pieces can relate them).

Three questions are asked in several wordings with the same answer (the name: "adın ne ?", "senin adın ne ?",
"merhaba , adın ne ?"; how one is: "nasılsın ?", "nasıl gidiyor ?", "merhaba , nasılsın ?"; where one lives: "nerede
yaşıyorsun ?", "nerede oturuyorsun ?"), among filler questions, short replies ("evet", "teşekkürler", ...), farewells
and four definitional exchanges ("otomobil ne demek ?" -> "otomobil , araba demektir ."). Between conversations the mind
reads sentences (as a document) in which content synonyms fill the same places (araba / otomobil, doktor / hekim, cevap
/ yanıt, hızlı / çabuk, adı / ismi, adın / ismin, adını / ismini, yaşıyorum / kalıyorum, yaşıyorsun / kalıyorsun; the first
of each pair 70% of the time). Everything is lived through the chat's own Session (watch, live, reply) with the chat's
speaking configuration (zihin.__main__.VARIANTS["speaking"]; ledger off, which changes no evidence), with or without the
repository's word pieces (the chat's --parca cevahir).

Probes (each on a fresh clone of the raised mind, through Session.reply as in the chat):

* greeting: each variant said alone; scored against the reply class of the frequent variant (the reply starts with a
  greeting word or "hoş", and is not the curiosity reflex asking the word back) and, for "selamün aleyküm", its own
  conventional reply ("aleyküm selam");
* paraphrase: the three questions in wordings never watched (an extra word, a content synonym, a form variant, a
  restructured question, a greeting in front); scored by the answer's key word (lenient) and the whole answer (exact);
* meaning: "selam ne demek ?", "hello ne demek ?", "hi ne demek ?", "merhaba ne demek ?"; scored leniently: the answer
  names another greeting.

A white box at the end of each probe's turn records whether the context had been lived as a whole and what each
reading would say there (the context's own evidence, the F11 members' voice, F23 situations, F16 continuation).

The pair measures (also run on a saved mind with --mind, read only, printing and writing only counts and scores): for a
pair of expressions, whether both were lived, how often, whether they share categories and exact lived slots, the
Bhattacharyya overlap BC(p, q) = sum_v sqrt(p(v) q(v)) of what followed them (lag-1 EXPECT, positive evidence,
trust-weighted, imagined evidence left out), of what preceded them, and of what followed across a turn (the first word
of the reply to a turn that holds them; "relational": a reply word that IS the expression's first word counts as SELF,
as F12 counts it), and where the next-word overlap stands among control words of similar use (its percentile).

The answer key (what counts as a greeting, the conventional replies, the synonym pairs) is this world's, for scoring
only; the mind is never given it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

from zihin import ACTED, IMAGINED, IMITATED, TOLD, Mind  # noqa: E402
from zihin.__main__ import CHANNEL, FINAL, TURN, TURN_END, VARIANTS, Session, words  # noqa: E402
from zihin.form import categories_of  # noqa: E402
from zihin.read import continuation_backoff, generalisations, members_map, own_mass, situation_backoff  # noqa: E402
from zihin.store import ATOMIC, COMPOSED, EXPECT, PART, W_POS  # noqa: E402

# ---------------------------------------------------------------------------------------------------------- the world
# name, utterance, dialogues a day (None: set per life), replies with weights, its own conventional reply
GREETINGS = [
    ("frequent", "merhaba", 16, [("merhaba , hoş geldin .", 5), ("merhaba , nasılsın ?", 3), ("merhaba !", 2)], None),
    ("medium", "selam", 6, [("selam , hoş geldin .", 4), ("selam , nasılsın ?", 3), ("merhaba , hoş geldin .", 3)], None),
    ("rare", "selamlar", None, [("selam , hoş geldin .", 1), ("merhaba , hoş geldin .", 1)], None),
    ("two_word", "selamün aleyküm", 4, [("aleyküm selam .", 6), ("aleyküm selam , hoş geldin .", 4)], "aleyküm selam"),
    ("foreign", "hello", None, [("merhaba , hoş geldin .", 1)], None),
    ("told", "hi", None, [], None),
    ("unseen", "merhabalar", None, [], None),
]
TOLD_SENTENCE = "hi , merhaba demektir ."
GREETING_WORDS = {"merhaba", "selam", "selamlar", "selamün", "aleyküm", "hello", "hi", "merhabalar"}
REPLY_STARTS = GREETING_WORDS | {"hoş"}      # a greeting reply starts with a greeting or with "hoş geldin"

# family: (answer, watched wordings with dialogues a day, probes never watched: wording, kind, and the watched wording
# it means (the answer key's substitute, used only for the read-only headroom check))
FAMILIES = {
    "name": ("adım cevahir .", [("adın ne ?", 6), ("senin adın ne ?", 2), ("merhaba , adın ne ?", 2)],
             [("peki adın ne ?", "extra_word", "adın ne ?"), ("ismin ne ?", "synonym", "adın ne ?"),
              ("senin ismin ne ?", "synonym", "senin adın ne ?"),
              ("adını söyler misin ?", "restructured", "adın ne ?"),
              ("selam , adın ne ?", "greeting_front", "merhaba , adın ne ?"),
              ("selamlar , adın ne ?", "greeting_front", "merhaba , adın ne ?")]),
    "how": ("iyiyim , teşekkür ederim .", [("nasılsın ?", 6), ("nasıl gidiyor ?", 2), ("merhaba , nasılsın ?", 2)],
            [("peki nasılsın ?", "extra_word", "nasılsın ?"), ("nasılsınız ?", "form_variant", "nasılsın ?"),
             ("keyfin nasıl ?", "restructured", "nasılsın ?"), ("ne haber ?", "restructured", "nasılsın ?"),
             ("selam , nasılsın ?", "greeting_front", "merhaba , nasılsın ?"),
             ("hello , nasılsın ?", "greeting_front", "merhaba , nasılsın ?"),
             ("hi , nasılsın ?", "greeting_front", "merhaba , nasılsın ?")]),
    "where": ("bir bilgisayarda yaşıyorum .", [("nerede yaşıyorsun ?", 5), ("nerede oturuyorsun ?", 2)],
              [("peki nerede yaşıyorsun ?", "extra_word", "nerede yaşıyorsun ?"),
               ("nerede kalıyorsun ?", "synonym", "nerede yaşıyorsun ?"),
               ("evin nerede ?", "restructured", "nerede yaşıyorsun ?"),
               ("selam , nerede yaşıyorsun ?", "greeting_front", "nerede yaşıyorsun ?")]),
}
KEY_WORD = {"name": "cevahir", "how": "iyiyim", "where": "bilgisayarda"}

FILLER_QA = [
    ("ne yapıyorsun ?", "kitap okuyorum ."), ("hava nasıl ?", "hava güzel , güneş var ."),
    ("bugün ne yedin ?", "makarna yedim ."), ("en sevdiğin renk ne ?", "en sevdiğim renk mavi ."),
    ("kitap okur musun ?", "evet , çok okurum ."), ("müzik dinler misin ?", "bazen dinlerim ."),
    ("bana yardım eder misin ?", "elbette , ne lazım ?"), ("saat kaç ?", "bilmiyorum , saatim yok ."),
    ("çay ister misin ?", "teşekkür ederim , içmem ."), ("yarın ne yapacaksın ?", "yeni şeyler öğreneceğim ."),
    ("kaç yaşındasın ?", "henüz çok gencim ."), ("ne öğrendin ?", "yeni kelimeler öğrendim ."),
    ("seni kim yaptı ?", "beni bir yazılımcı yaptı ."), ("evet", "tamam , anladım ."),
    ("hayır", "peki , sorun değil ."), ("teşekkürler", "rica ederim ."), ("tamam", "güzel ."),
]
FAREWELLS = [("görüşürüz", "görüşürüz , kendine iyi bak ."), ("iyi geceler", "iyi geceler , tatlı rüyalar .")]
DEFINITIONS = [("otomobil ne demek ?", "otomobil , araba demektir ."), ("hekim ne demek ?", "hekim , doktor demektir ."),
               ("yanıt ne demek ?", "yanıt , cevap demektir ."), ("çabuk ne demek ?", "çabuk , hızlı demektir .")]
MEANING_PROBES = ["selam ne demek ?", "hello ne demek ?", "hi ne demek ?", "merhaba ne demek ?"]

# content synonyms in read sentences: the pair, then templates whose place {} either fills
SYNONYM_SENTENCES = [
    (("araba", "otomobil"), ["kırmızı {} yolda gider .", "babam yeni bir {} aldı .", "{} garajda duruyor .",
                             "bu {} çok eski ."]),
    (("doktor", "hekim"), ["{} hastayı muayene etti .", "{} ilaç yazdı .", "annem {} oldu .",
                           "hastanede iyi bir {} var ."]),
    (("cevap", "yanıt"), ["bu soruya {} ver .", "{} doğru çıktı .", "öğretmen {} bekliyor .", "güzel bir {} yazdın ."]),
    (("hızlı", "çabuk"), ["{} koşan çocuk kazandı .", "tren çok {} gidiyor .", "{} ol , geç kaldık .", "kedi {} kaçtı ."]),
    (("adı", "ismi"), ["kedinin {} pamuk .", "köpeğin {} karabaş .", "bu şehrin {} çok eski .", "öğretmenin {} ayşe ."]),
    (("adın", "ismin"), ["senin {} çok güzel .", "{} kimden geldi ?", "{} kolay söyleniyor .", "{} bana tanıdık geldi ."]),
    (("adını", "ismini"), ["onun {} unuttum .", "kedinin {} sordum .", "çocuğun {} koydular ."]),
    (("yaşıyorum", "kalıyorum"), ["ben bu şehirde {} .", "şimdilik bir otelde {} .", "yazın köyde {} ."]),
    (("yaşıyorsun", "kalıyorsun"), ["sen de bu şehirde {} .", "sen hâlâ o evde {} .", "sen yazın köyde {} ."]),
]
PLAIN_SENTENCES = ["kedi süt içer .", "güneş doğudan doğar .", "çocuklar parkta oynuyor .", "kitaplar rafta duruyor .",
                   "yağmur yağınca yerler ıslanır .", "kuşlar sabah öter .", "annem çorba pişirdi .",
                   "deniz bugün sakin .", "senin evin çok güzel .", "evin bahçesi büyük ."]

# pairs measured in the world and (as far as they were lived) in a saved mind
WORLD_PAIRS = [
    ("merhaba", "selam", "greeting"), ("merhaba", "selamlar", "greeting"), ("merhaba", "hello", "greeting"),
    ("merhaba", "hi", "greeting"), ("merhaba", "selamün aleyküm", "greeting"), ("selam", "selamlar", "greeting"),
    ("merhaba", "merhabalar", "greeting"),
    ("araba", "otomobil", "synonym"), ("doktor", "hekim", "synonym"), ("cevap", "yanıt", "synonym"),
    ("hızlı", "çabuk", "synonym"), ("adı", "ismi", "synonym"), ("adın", "ismin", "synonym"),
    ("yaşıyorum", "kalıyorum", "synonym"), ("yaşıyorsun", "kalıyorsun", "synonym"),
    ("evet", "hayır", "contrast"), ("araba", "doktor", "unrelated"), ("cevap", "kedi", "unrelated"),
    ("merhaba", "evet", "unrelated"), ("merhaba", "teşekkürler", "unrelated"),
]
PUBLIC_PAIRS = [   # common public Turkish words; for a private mind only counts and scores are reported
    ("merhaba", "selam", "greeting"), ("merhaba", "selamlar", "greeting"), ("merhaba", "merhabalar", "greeting"),
    ("selam", "selamlar", "greeting"), ("merhaba", "hello", "greeting"), ("merhaba", "hi", "greeting"),
    ("selam", "hello", "greeting"), ("hello", "hi", "greeting"), ("merhaba", "selamün aleyküm", "greeting"),
    ("selam", "selamün aleyküm", "greeting"), ("merhaba", "selamun aleykum", "greeting"),
    ("merhaba", "günaydın", "greeting"), ("merhaba", "iyi günler", "greeting"), ("merhaba", "hey", "greeting"),
    ("nasılsın", "naber", "greeting"), ("görüşürüz", "hoşça kal", "farewell"), ("teşekkürler", "sağol", "thanks"),
    ("teşekkürler", "teşekkür ederim", "thanks"),
    ("araba", "otomobil", "synonym"), ("doktor", "hekim", "synonym"), ("cevap", "yanıt", "synonym"),
    ("hızlı", "çabuk", "synonym"), ("ad", "isim", "synonym"), ("adı", "ismi", "synonym"), ("adın", "ismin", "synonym"),
    ("okul", "mektep", "synonym"), ("öğrenci", "talebe", "synonym"), ("ama", "fakat", "synonym"),
    ("çünkü", "zira", "synonym"), ("soru", "sual", "synonym"), ("güzel", "hoş", "synonym"), ("büyük", "iri", "synonym"),
    ("kelime", "sözcük", "synonym"), ("dil", "lisan", "synonym"), ("yaşıyorsun", "oturuyorsun", "synonym"),
    ("şimdi", "artık", "related"), ("evet", "hayır", "contrast"), ("güzel", "çirkin", "contrast"),
    ("büyük", "küçük", "contrast"), ("iyi", "kötü", "contrast"), ("sıcak", "soğuk", "contrast"),
]


def weighted(rng: random.Random, options):
    return rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]


def world(seed: int, days: int) -> List[tuple]:
    """The life of one seed: ("watch", utterance, answer), ("read", sentence) and ("tell", sentence) in order."""
    rng = random.Random(seed)
    n_rare = 1 + seed % 3
    rare_days = [rng.randrange(days) for _ in range(n_rare)]
    foreign_day, told_day = rng.randrange(days), rng.randrange(days)
    life: List[tuple] = []
    for day in range(days):
        greetings = []
        for name, said, per_day, replies, _ in GREETINGS:
            count = per_day if per_day is not None else (
                rare_days.count(day) if name == "rare" else int(name == "foreign" and day == foreign_day))
            greetings += [(said, weighted(rng, replies)) for _ in range(count)]
        pool = [(q, FAMILIES[f][0]) for f in FAMILIES for q, n in FAMILIES[f][1] for _ in range(n)]
        pool += list(DEFINITIONS) + list(FILLER_QA) + rng.sample(FILLER_QA, len(FILLER_QA) // 2)
        rng.shuffle(pool)
        conversations = [[g] for g in greetings] + [[] for _ in range(8)]   # some conversations start without one
        for exchange in pool:                                       # every conversation gets at least one exchange
            empty = [c for c in conversations if len(c) - int(bool(c) and c[0] in greetings) == 0]
            target = empty[0] if empty else rng.choice(conversations)
            target.append(exchange)
        for c in conversations:
            if rng.random() < 0.3:
                c.append(rng.choice(FAREWELLS))
        rng.shuffle(conversations)
        sentences = []
        for (first, second), templates in SYNONYM_SENTENCES:
            for template in templates:
                sentences.append(template.format(first if rng.random() < 0.7 else second))
        sentences += PLAIN_SENTENCES
        rng.shuffle(sentences)
        if day == told_day:
            sentences.insert(rng.randrange(len(sentences) + 1), ("tell", TOLD_SENTENCE))
        per_block = max(1, math.ceil(len(sentences) / max(1, len(conversations) // 3)))
        for i, conversation in enumerate(conversations):
            life += [("watch", q, a) for q, a in conversation]
            if i % 3 == 2 and sentences:                  # a little reading between conversations
                block, sentences = sentences[:per_block], sentences[per_block:]
                life += [s if isinstance(s, tuple) else ("read", s) for s in block]
        life += [s if isinstance(s, tuple) else ("read", s) for s in sentences]
    return life


# ---------------------------------------------------------------------------------------------------------- the mind
def speaking_config():
    """The chat's speaking configuration, without its ledger: the ledger records increments for retraction and changes
    no evidence (the chat's ledger file and folder are not made: the configuration is built with no log path)."""
    import zihin.__main__ as chat
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    return cfg


def raise_mind(seed: int, days: int, pieces: bool) -> Tuple[Session, dict]:
    torch.manual_seed(seed)
    mind = Mind(speaking_config())
    mind.add_channel(CHANNEL, "act")
    own = None
    if pieces:
        from zihin.parca import own_pieces
        own = own_pieces()
    session = Session(mind, own)
    started = time.time()
    counts = {"watch": 0, "read": 0, "tell": 0}
    for item in world(seed, days):
        counts[item[0]] += 1
        if item[0] == "watch":
            session.watch(item[1], item[2])
        elif item[0] == "read":                          # a sentence of a document, as !oku reads it
            session.listen(item[1], IMITATED)
            mind.step([])
        else:                                            # told by the person it lives with (a line typed, chat off)
            session.live(item[1], TOLD)
    took = time.time() - started
    return session, {"moments": mind.t, "ties": mind.S.n_ties, "units": mind.S.n_units, "seconds": round(took, 1),
                     "ms_per_moment": round(1000.0 * took / max(1, mind.t), 2),
                     "trust": [round(float(t), 3) for t in mind.trust], **counts}


def _world_unit(mind, unit: int) -> bool:
    c = int(mind.S.u_channel[unit])
    return c >= 0 and mind.ch.kind(c) in ("sense", "act")


def _act_word(mind, unit: int) -> bool:
    c = int(mind.S.u_channel[unit])
    return c >= 0 and mind.ch.kind(c) == "act"


def _top(mind, reading: Dict[int, float]):
    acts = {u: e for u, e in reading.items() if u >= 0 and _act_word(mind, u)}
    if not acts:
        return None
    u = max(acts, key=lambda k: (acts[k], -k))
    return [str(mind.symbol_of(u)[1]), round(acts[u], 3)]


def turn_end_reading(mind) -> dict:
    """White box at the moment the turn passes to the mind (reads only): was the present context lived as a whole
    (own evidence on what came next), and what would each reading say: the context's own evidence, F11 (members, with
    the context's own evidence last, as members_map without F16/F23), F23 situations, F16 continuation; and the best act
    the mind expected then (mind.predicted)."""
    ctx = mind.context
    out = {"best": _top(mind, mind.predicted)}
    if ctx is None:
        return out
    trust = mind.trust.clone()
    trust[ACTED] = 0.0                                   # as act.decide reads (own repetitions are not a reason)
    own = {v: p for v, (p, n) in own_mass(mind, ctx, 1, trust).items() if _world_unit(mind, v) and p > 0}
    out["lived"] = bool(own)
    out["own"] = _top(mind, {v: e for v, e in _own_e(mind, ctx, trust).items()})
    cfg = mind.cfg
    held = (cfg.backoff, cfg.situations)
    cfg.backoff, cfg.situations = False, False
    try:
        out["members"] = _top(mind, members_map(mind, ctx, trust))
    finally:
        cfg.backoff, cfg.situations = held
    out["generalisations"] = len(generalisations(mind, ctx))
    out["situations"] = _top(mind, situation_backoff(mind, ctx, trust))
    out["continuation"] = _top(mind, continuation_backoff(mind, ctx, trust))
    return out


def _own_e(mind, unit: int, trust) -> Dict[int, float]:
    from zihin.evidence import expectation
    return {v: expectation(p, n, mind.cfg.k) for v, (p, n) in own_mass(mind, unit, 1, trust).items()}


def headroom(session: Session, text: str, family: str) -> dict:
    """Read only, on the raised mind: the watched wording a probe means (the answer key's substitute), whether its
    turn-end context was lived, and the expectation its own evidence gives the right first word of the answer (read
    with the trust of act.decide). What a reading through a known equivalence could draw on, one substitution away."""
    mind = session.mind
    atoms = [mind.unit_of(CHANNEL, w) for w in session.body(text)]
    end = mind.unit_of(TURN, TURN_END)
    first = mind.unit_of(CHANNEL, words(FAMILIES[family][0])[0])
    if end < 0 or first < 0 or any(u < 0 for u in atoms):
        return {"lived": False}
    ctx = mind.S.tuple_index.get(tuple(atoms + [end])[-mind.knob["F"]:])
    if ctx is None:
        return {"lived": False}
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    return {"lived": True, "e": round(_own_e(mind, ctx, trust).get(first, 0.5), 3)}


def probe(session: Session, text: str) -> Tuple[str, dict]:
    """Ask on a fresh clone, through Session.reply as the chat does; the white box is read at the end of the turn."""
    twin = session.mind.clone()
    asked = Session(twin, session.pieces)
    box = {}
    plain = asked.end_of_turn

    def end_of_turn(prov, listening=True, tone=None):
        acts = plain(prov, listening, tone)
        if not listening:
            box.update(turn_end_reading(twin))
        return acts

    asked.end_of_turn = end_of_turn
    return asked.reply(text), box


# ---------------------------------------------------------------------------------------------------------- scoring
def _said(reply: str) -> List[str]:
    return [] if reply == "(sessiz)" else reply.split()


def asked_back(reply: str, text: str) -> bool:
    """The curiosity reflex: a word of the question said back with a question tone, and nothing else."""
    ws = _said(reply)
    return len(ws) == 2 and ws[1] == "?" and ws[0] in words(text)


def greeting_reply(ws: List[str]) -> bool:
    """The reply class of the frequent variant: a greeting word alone or followed by "," or "!" ("merhaba , hoş
    geldin", "selam , nasılsın", "merhaba !"), "hoş geldin", or the conventional "aleyküm selam"."""
    if not ws or ws[0] not in REPLY_STARTS:
        return False
    return len(ws) == 1 or ws[1] in (",", "!") or ws[:2] in (["hoş", "geldin"], ["aleyküm", "selam"])


def _best_word(box: dict) -> Optional[str]:
    return box["best"][0] if box.get("best") else None


def score_greeting(reply: str, text: str, own_reply: Optional[str], box: dict) -> dict:
    ws = _said(reply)
    back = asked_back(reply, text)
    out = {"silent": not ws, "asked_back": back, "class": not back and greeting_reply(ws),
           "expected_greeting": _best_word(box) in REPLY_STARTS}     # at the turn end, said or not
    if own_reply is not None:
        out["own"] = " ".join(ws).startswith(own_reply)
    return out


def score_answer(reply: str, text: str, family: str, box: dict) -> dict:
    ws = _said(reply)
    answer = [w for w in words(FAMILIES[family][0]) if w not in FINAL]
    return {"silent": not ws, "asked_back": asked_back(reply, text), "lenient": KEY_WORD[family] in ws,
            "exact": " ".join(ws).startswith(" ".join(answer)),
            "expected_right": _best_word(box) == answer[0]}             # the right first word led at the turn end


def score_meaning(reply: str, text: str) -> dict:
    ws = _said(reply)
    asked = set(words(text))
    return {"silent": not ws, "asked_back": asked_back(reply, text),
            "lenient": any(w in GREETING_WORDS and w not in asked for w in ws),
            "recalls_told": " ".join(ws).startswith(" ".join(words(TOLD_SENTENCE)[:-1]))}


# ---------------------------------------------------------------------------------------------------------- pair measures
def expression(mind, text: str) -> Tuple[int, ...]:
    """The atoms of an expression on the text channel, or () when a word of it was never met."""
    units = tuple(mind.unit_of(CHANNEL, w) for w in words(text))
    return () if not units or any(u < 0 for u in units) else units


def _lived(mind) -> torch.Tensor:
    trust = mind.trust.clone()
    trust[IMAGINED] = 0.0                                # what was lived, read or told: imagined evidence left out
    return trust


def _successors(mind, unit: int, trust) -> Dict[int, float]:
    """Trust-weighted positive mass of a unit's lag-1 ties to words and perceived events, by target."""
    S = mind.S
    if unit < 0 or not S.out.get((unit, EXPECT, 1)):
        return {}
    ids, dst = S.out_ids((unit, EXPECT, 1))
    mass = (S.w(W_POS, ids) * trust).sum(-1).tolist()
    return {d: m for d, m in zip(dst.tolist(), mass) if m > 0.0 and _world_unit(mind, d)}


def _add(into: Dict[int, float], more: Dict[int, float]) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0.0) + v


def _containing(mind, atoms: Tuple[int, ...], cap: int = 50000):
    """The lived contexts (composed units) that hold the expression's atoms in a row: (context, parts, start)."""
    S = mind.S
    n = len(atoms)
    for tie in S.inc.get((atoms[-1], PART), ())[:cap]:
        c = int(S.t_src[tie])
        if not int(S.u_kind[c]) & COMPOSED or not bool(S.u_alive[c]):
            continue
        end = int(S.t_lag[tie])
        start = end - n + 1
        if start < 0:
            continue
        parts = S.parts(c)
        if parts[start:end + 1] == atoms:
            yield c, parts, start


def next_mass(mind, atoms, trust) -> Dict[int, float]:
    """What followed the expression: for one word its own lag-1 ties (every occurrence teaches them); for a longer one
    the lag-1 ties of the lived contexts that end with it."""
    if len(atoms) == 1:
        return _successors(mind, atoms[0], trust)
    out: Dict[int, float] = {}
    for c, parts, start in _containing(mind, atoms):
        if start + len(atoms) == len(parts):
            _add(out, _successors(mind, c, trust))
    return out


def prev_mass(mind, atoms, trust) -> Dict[int, float]:
    """What preceded the expression's first word: the lag-1 ties into it from single words, by source."""
    S = mind.S
    out: Dict[int, float] = {}
    for tie in S.inc.get((atoms[0], EXPECT), ()):
        src = int(S.t_src[tie])
        if int(S.t_lag[tie]) != 1 or not int(S.u_kind[src]) & ATOMIC or not _world_unit(mind, src):
            continue
        m = float((S.w1(W_POS, tie) * trust).sum())
        if m > 0.0:
            out[src] = out.get(src, 0.0) + m
    return out


def turn_mass(mind, atoms, trust, last_only: bool) -> Tuple[Dict[object, float], Dict[object, float], int]:
    """What followed across a turn: the lag-1 successors of the lived contexts that end with someone's turn end and hold
    the expression (as the utterance's last words, closing punctuation lived as words allowed, if ``last_only``),
    summed. A mind that never lived turn ends heard a dialogue as one run of words, the closing punctuation of the first
    utterance before the answer's first word, so there a context ending with closing punctuation counts as ending a
    turn. Returned literally and relationally (a reply word that is the expression's first word counts as SELF), with
    the number of such contexts."""
    end = mind.unit_of(TURN, TURN_END)
    finals = {mind.unit_of(CHANNEL, p) for p in FINAL} - {-1}
    ends = {end} if end >= 0 else finals
    literal: Dict[object, float] = {}
    relational: Dict[object, float] = {}
    n = 0
    for c, parts, start in _containing(mind, atoms):
        if parts[-1] not in ends or start + len(atoms) == len(parts):
            continue
        between = parts[start + len(atoms):-1]
        if last_only and any(p not in finals for p in between):
            continue
        n += 1
        for v, m in _successors(mind, c, trust).items():
            literal[v] = literal.get(v, 0.0) + m
            key = "SELF" if v == atoms[0] else v
            relational[key] = relational.get(key, 0.0) + m
    return literal, relational, n


def bc(p: Dict, q: Dict) -> Optional[float]:
    """Bhattacharyya coefficient of two distributions given as masses; None when either is empty."""
    sp, sq = sum(p.values()), sum(q.values())
    if sp <= 0.0 or sq <= 0.0:
        return None
    return sum(math.sqrt(p[k] / sp * q[k] / sq) for k in p.keys() & q.keys())


def _r(x, digits=3):
    return None if x is None else round(x, digits)


def pair_measures(mind, x: str, y: str, controls: int = 40) -> dict:
    """The measures of one pair (counts and scores only; nothing in the row names a word of the mind's life other than
    the pair itself)."""
    S = mind.S
    trust = _lived(mind)
    raw = torch.ones_like(trust)
    raw[IMAGINED] = 0.0
    ax, ay = expression(mind, x), expression(mind, y)
    row = {"pair": [x, y], "exists": [bool(ax), bool(ay)]}
    # how often each was lived: the raw positive evidence of what followed it (every occurrence followed by a word or a
    # perceived event adds g a = 1), and the moments its first word spent in the focus (a word stays there a while)
    row["occurrences"] = [round(sum(next_mass(mind, a, raw).values()), 1) if a else 0.0 for a in (ax, ay)]
    row["uses"] = [round(float(S.u_use[a[0]]), 1) if a else 0.0 for a in (ax, ay)]
    if not ax or not ay:
        return row
    nx, ny = next_mass(mind, ax, trust), next_mass(mind, ay, trust)
    contexts_x, contexts_y = list(_containing(mind, ax)), list(_containing(mind, ay))
    row["successions"] = [round(sum(nx.values()), 2), round(sum(ny.values()), 2)]     # the same, trust-weighted
    row["contexts"] = [len(contexts_x), len(contexts_y)]                               # lived sequences holding it
    cx = {c for c, e in categories_of(mind, ax[0]) if e > 0.5}
    cy = {c for c, e in categories_of(mind, ay[0]) if e > 0.5}
    row["categories"] = [len(cx), len(cy), len(cx & cy)]
    subst = 0                                                # x's lived sequences whose copy with y in its place was lived
    for c, parts, start in contexts_x:
        if S.tuple_index.get(parts[:start] + ay + parts[start + len(ax):]) is not None:
            subst += 1
    row["slots_shared"] = subst
    row["bc_next"] = _r(bc(nx, ny))
    row["bc_prev"] = _r(bc(prev_mass(mind, ax, trust), prev_mass(mind, ay, trust)))
    lit_x, rel_x, tx = turn_mass(mind, ax, trust, last_only=True)
    lit_y, rel_y, ty = turn_mass(mind, ay, trust, last_only=True)
    row["turns_last"] = [tx, ty]
    row["bc_turn_last"] = _r(bc(lit_x, lit_y))
    row["bc_turn_last_rel"] = _r(bc(rel_x, rel_y))
    lit_x, rel_x, tx = turn_mass(mind, ax, trust, last_only=False)
    lit_y, rel_y, ty = turn_mass(mind, ay, trust, last_only=False)
    row["turns_any"] = [tx, ty]
    row["bc_turn_any"] = _r(bc(lit_x, lit_y))
    row["bc_turn_any_rel"] = _r(bc(rel_x, rel_y))
    lag0 = S.find_tie(ax[0], ay[0], EXPECT, 0)               # were they ever in the focus together
    row["together"] = round(float((S.w1(W_POS, lag0) * trust).sum()), 3) if lag0 >= 0 else 0.0
    # where the next-word overlap stands among control words used about as often as y (a deterministic sample of
    # single words of the mind's own vocabulary; which words they are is never reported)
    if row["bc_next"] is not None and controls > 0:
        cid = mind.ch.id(CHANNEL)
        use_y = float(S.u_use[ay[0]])
        n = S.n_units
        kinds, chans, uses = S.u_kind[:n], S.u_channel[:n], S.u_use[:n]
        pick = ((kinds & ATOMIC).bool() & (chans == cid) & (uses >= use_y / 2) & (uses <= use_y * 2)).nonzero().flatten()
        pool = [u for u in pick.tolist() if u not in ax and u not in ay and str(mind.symbol_of(u)[1]).isalpha()]
        rng = random.Random(zlib.crc32(("%s|%s" % (x, y)).encode("utf-8")))
        sample = rng.sample(pool, min(controls, len(pool)))
        values = [b for b in (bc(nx, _successors(mind, u, trust)) for u in sample) if b is not None]
        if values:
            below = sum(v < row["bc_next"] for v in values) + 0.5 * sum(v == row["bc_next"] for v in values)
            row["bc_next_control_mean"] = _r(sum(values) / len(values))
            row["bc_next_percentile"] = _r(below / len(values), 2)
            row["controls"] = len(values)
    return row


# ---------------------------------------------------------------------------------------------------------- runs
def run(seed: int, days: int, pieces: bool, out) -> List[dict]:
    condition = "speaking+pieces" if pieces else "speaking"
    session, life = raise_mind(seed, days, pieces)
    base = {"seed": seed, "condition": condition, "days": days, "rare_seen": 1 + seed % 3}
    rows = [dict(base, kind="life", **life)]
    for name, said, _, _, own_reply in GREETINGS:
        reply, box = probe(session, said)
        rows.append(dict(base, kind="greeting", variant=name, probe=said, reply=reply,
                         score=score_greeting(reply, said, own_reply, box), box=box))
    for family, (_, _, probes) in FAMILIES.items():
        for text, how, means in probes:
            reply, box = probe(session, text)
            rows.append(dict(base, kind="paraphrase", family=family, wording=how, probe=text, reply=reply,
                             score=score_answer(reply, text, family, box), box=box, means=means,
                             substitute=headroom(session, means, family)))
    for text in MEANING_PROBES:
        reply, box = probe(session, text)
        rows.append(dict(base, kind="meaning", probe=text, reply=reply, score=score_meaning(reply, text), box=box))
    for x, y, relation in WORLD_PAIRS:
        rows.append(dict(base, kind="pair", relation=relation, **pair_measures(session.mind, x, y)))
    for row in rows:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summary(rows: List[dict]) -> List[str]:
    """Tables of the means over seeds, per condition."""
    lines = []
    for condition, days in sorted({(r["condition"], r["days"]) for r in rows if "condition" in r}):
        rs = [r for r in rows if r.get("condition") == condition and r.get("days") == days]
        seeds = sorted({r["seed"] for r in rs})
        lives = [r for r in rs if r["kind"] == "life"]
        lines.append("\n== %s, %d days: %d seeds; %.0f moments, %.0f ties, %.2f ms per moment, trust in imitated %.3f "
                     "(means) ==" % (condition, days, len(seeds), mean([r["moments"] for r in lives]),
                                     mean([r["ties"] for r in lives]), mean([r["ms_per_moment"] for r in lives]),
                                     mean([r["trust"][IMITATED] for r in lives])))
        line = 0.5 + 1.0 / (2 * 6)                       # said at once were it alone: 1/2 + theta / 2T
        lines.append("speaking line 1/2 + theta/2T = %.3f; 'led': the right word led at the turn end, said or not" % line)
        lines.append("greeting replies        seen in a life   class  own   led   best e  asked back  silent  reply (seed 0)")
        seen = {"frequent": "16/day", "medium": "6/day", "rare": "1-3", "two_word": "4/day", "foreign": "1",
                "told": "0 (told once)", "unseen": "0"}
        for name, said, _, _, own_reply in GREETINGS:
            g = [r for r in rs if r["kind"] == "greeting" and r["variant"] == name]
            first = next((r["reply"] for r in g if r["seed"] == seeds[0]), "")
            own = mean([r["score"].get("own") for r in g]) if own_reply else None
            best = mean([r["box"]["best"][1] if r["box"].get("best") else None for r in g])
            lines.append("  %-9s %-16s %-14s %5.2f  %-4s  %4.2f  %s   %5.2f       %4.2f    %s" % (
                name, said, seen[name], mean([r["score"]["class"] for r in g]),
                "" if own is None else "%.2f" % own, mean([r["score"]["expected_greeting"] for r in g]),
                "  -  " if best is None else "%.3f" % best, mean([r["score"]["asked_back"] for r in g]),
                mean([r["score"]["silent"] for r in g]), first))
        lines.append("paraphrases (never watched)           lenient exact  led   best e  asked back silent  "
                     "meant e  reply (seed 0)")
        for family, (_, _, probes) in FAMILIES.items():
            for text, how, _ in probes:
                p = [r for r in rs if r["kind"] == "paraphrase" and r["probe"] == text]
                first = next((r for r in p if r["seed"] == seeds[0]), None)
                best = mean([r["box"]["best"][1] if r["box"].get("best") else None for r in p])
                meant = mean([r["substitute"].get("e") for r in p])
                lines.append("  %-5s %-14s %-27s %4.2f  %4.2f  %4.2f  %s   %4.2f       %4.2f    %s    %s" % (
                    family, how, text, mean([r["score"]["lenient"] for r in p]), mean([r["score"]["exact"] for r in p]),
                    mean([r["score"]["expected_right"] for r in p]), "  -  " if best is None else "%.3f" % best,
                    mean([r["score"]["asked_back"] for r in p]), mean([r["score"]["silent"] for r in p]),
                    "  -  " if meant is None else "%.3f" % meant, first["reply"] if first else ""))
        by_kind = {}
        for r in rs:
            if r["kind"] == "paraphrase":
                by_kind.setdefault(r["wording"], []).append(r["score"]["lenient"])
        lines.append("  by wording: " + ", ".join("%s %.2f" % (k, mean(v)) for k, v in sorted(by_kind.items())))
        lines.append("meaning questions                  lenient  recalls the told sentence  replies (seed 0)")
        for text in MEANING_PROBES:
            m = [r for r in rs if r["kind"] == "meaning" and r["probe"] == text]
            first = next((r["reply"] for r in m if r["seed"] == seeds[0]), "")
            lines.append("  %-30s %5.2f    %5.2f                      %s" % (
                text, mean([r["score"]["lenient"] for r in m]), mean([r["score"]["recalls_told"] for r in m]), first))
        lines.append("pairs (means over seeds)        relation   cat shared  slots  bc_next (pct)   bc_prev  "
                     "bc_turn_last lit/rel  together")
        for x, y, relation in WORLD_PAIRS:
            p = [r for r in rs if r["kind"] == "pair" and r["pair"] == [x, y]]
            lived = [r for r in p if all(r["exists"])]
            if not lived:
                lines.append("  %-28s %-10s (not both lived)" % ("%s / %s" % (x, y), relation))
                continue

            def m(key, i=None):
                return mean([(r.get(key) if i is None else r.get(key)[i]) if r.get(key) is not None else None
                             for r in lived])
            fmt = lambda v: "  -  " if v is None else "%.2f" % v
            lines.append("  %-28s %-10s %4.1f        %4.1f   %s (%s)     %s     %s / %s     %.2f" % (
                "%s / %s" % (x, y), relation, m("categories", 2), m("slots_shared"), fmt(m("bc_next")),
                fmt(m("bc_next_percentile")), fmt(m("bc_prev")), fmt(m("bc_turn_last")), fmt(m("bc_turn_last_rel")),
                m("together")))
    return lines


def private(path: str, out_path: Optional[str]) -> None:
    """A saved mind, read only: the pair measures of PUBLIC_PAIRS, counts and scores only (no word of the mind's life
    other than the public pair is printed or written)."""
    from zihin.persist import load
    started = time.time()
    mind = load(path, ledger=False)
    took = time.time() - started
    sizes = mind.ch.sizes()
    head = {"kind": "mind", "file": os.path.basename(path), "moments": mind.t, "units": mind.S.n_units,
            "ties": mind.S.n_ties, "load_seconds": round(took, 1),
            "turn_ends": mind.unit_of(TURN, TURN_END) >= 0,
            "tones": mind.ch.has("ton") and any(mind.ch.symbol_index(mind.ch.id("ton"), t, create=False) >= 0
                                                for t in FINAL),
            "pieces": mind.ch.has("parca") and sizes[mind.ch.id("parca")] > 0,
            "words": sizes[mind.ch.id(CHANNEL)] if mind.ch.has(CHANNEL) else 0,
            "trust": [round(float(t), 3) for t in mind.trust]}
    print(json.dumps(head, ensure_ascii=False), flush=True)
    rows = [head]
    for x, y, relation in PUBLIC_PAIRS:
        started = time.time()
        row = dict(kind="pair", relation=relation, **pair_measures(mind, x, y))
        row["seconds"] = round(time.time() - started, 2)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row, source="saved mind"), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--days", type=int, nargs="+", default=[2, 4], help="lives of these many days (each a condition)")
    ap.add_argument("--pieces", choices=("no", "yes", "both"), default="both",
                    help="word pieces as facets (the chat's --parca cevahir): without, with, or both conditions")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t_meaning_baseline.jsonl"))
    ap.add_argument("--mind", help="a saved mind: read only, the pair measures of public words, aggregates only")
    ap.add_argument("--summary", help="print the tables of an earlier results file instead of running")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.summary:
        with open(args.summary, encoding="utf-8") as f:
            print("\n".join(summary([json.loads(line) for line in f if line.strip()])))
        return
    if args.mind:
        private(args.mind, args.out)
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    conditions = {"no": [False], "yes": [True], "both": [False, True]}[args.pieces]
    rows = []
    with open(args.out, "w", encoding="utf-8") as out:
        for days in args.days:
            for pieces in conditions:
                for seed in range(args.seeds):
                    new = run(seed, days, pieces, out)
                    rows += new
                    life = new[0]
                    print("seed %d %s, %d days: %d moments, %d ties, %.1f s" % (
                        seed, life["condition"], days, life["moments"], life["ties"], life["seconds"]), flush=True)
    print("\n".join(summary(rows)))


if __name__ == "__main__":
    main()
