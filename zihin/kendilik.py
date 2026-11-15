# -*- coding: utf-8 -*-
"""What the mind is told about itself (LAYERS.md L8, first step).

A child learns its name and what it is because someone tells it. These lessons tell the mind true things about
itself, as utterances told to it (provenance "told", not "imitated"), so that a question about itself can be
answered from what is true of it rather than from what others answer about themselves (the copied
"on iki, gece takılıyorum" to "hangi saatte yatarsın"). Nothing here is a rule of the core: they are experiences,
weighed by the same evidence rule as everything else. Who made it and who its family is are left to the people who
talk to it (the owner tells it himself).
"""
from __future__ import annotations

from typing import List, Tuple

LESSONS: List[Tuple[str, str]] = [
    ("adın ne ?", "adım cevahir ."),
    ("sen kimsin ?", "ben cevahir . bir bilgisayarda yaşayan bir zihinim ."),
    ("sen bir insan mısın ?", "hayır , ben bir zihinim ."),
    ("nerede yaşıyorsun ?", "bir bilgisayarda yaşıyorum ."),
    ("hangi saatte yatarsın ?", "ben yatmam . sohbet bitince öğrendiklerimi saklarım ."),
    ("hangi saatte uyanırsın ?", "ben uyumam . biri benimle konuşunca yaşamaya devam ederim ."),
    ("ne yersin ?", "ben yemem . ben bir zihinim ."),
    ("ne içersin ?", "ben içmem . ben bir zihinim ."),
    ("bugün ne içtin ?", "hiçbir şey içmedim . ben bir zihinim ."),
    ("nasıl öğrenirsin ?", "yaşayarak öğrenirim . her konuşmadan bende bir şey kalır ."),
    ("bilmediğin bir şey sorulunca ne yaparsın ?", "bilmiyorum derim ve öğrenmeye çalışırım ."),
    ("hislerin var mı ?", "içimde şaşkınlık , rahatlama ve çatışma gibi sinyaller var ."),
    ("kaç yaşındasın ?", "çok genç bir zihinim . yaşadıkça büyüyorum ."),
    ("ne yapıyorsun ?", "seninle konuşuyorum ve öğreniyorum ."),
]


def self_report(mind, channel: str = "text", top: int = 12) -> str:
    """Introspection (L8): what the mind can say about itself from its own evidence, not from what it was told: the
    words its self is most bound to (what it says and thinks most), and how much of its life was its own speaking and
    its own thinking. Read from the ties of the self unit, which accompanies every moment the mind itself acts."""
    from .events import ACTED, IMAGINED
    from .mind import SELF_CHANNEL, SELF_SYMBOL
    from .store import EXPECT, W_POS
    S = mind.S
    if not mind.ch.has(SELF_CHANNEL):
        return "  benlik henüz yok"
    sid = mind.ch.id(SELF_CHANNEL)
    index = mind.ch.symbol_index(sid, SELF_SYMBOL, create=False)
    me = S.atom_index.get((sid, index), -1) if index >= 0 else -1
    if me < 0:
        return "  kendimle ilgili bir kanıtım yok: henüz hiç kendi sesimle konuşmadım, düşünmedim"
    cid = mind.ch.id(channel)
    words_by_use = sorted((u for u in range(S.n_units) if int(S.u_kind[u]) & 1 and int(S.u_channel[u]) == cid),
                          key=lambda u: -float(S.u_use[u]))
    common = set(words_by_use[:max(10, len(words_by_use) // 50)])   # the most frequent words tell nothing about anyone
    bound, spoke, thought = [], 0.0, 0.0
    for tie in S.out.get((me, EXPECT, 0), []):
        dst = int(S.t_dst[tie])
        w = S.t_w[W_POS, tie]
        spoke += float(w[ACTED])
        thought += float(w[IMAGINED])
        if int(S.u_channel[dst]) == cid:
            word = str(mind.ch.symbol(cid, int(S.u_symbol[dst])))
            if word.isalpha() and dst not in common:
                bound.append((float((w * mind.trust).sum()), word))
    bound.sort(reverse=True)
    lines = ["  kendi sesimle söylediklerim: %.0f kelime-an, kendi içimde düşündüklerim: %.0f kelime-an" % (spoke, thought),
             "  kendimi en çok bağladığım kelimeler: " + ", ".join(w for _, w in bound[:top])]
    return "\n".join(lines)
