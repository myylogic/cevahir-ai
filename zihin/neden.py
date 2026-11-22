# -*- coding: utf-8 -*-
"""Why a word was said: which reading of the mind supplied it (the chat's ``!neden``).

When the session asks for it, every word the mind decides is attributed at the moment it is decided, from the same
readings the decision used (``read.members_map``), in their order of precedence:

* the context's own lived evidence (the exact situation was lived and went on with this word: memory);
* a frame: what the members of a category lived in the same place, and by which relation (the same word, the asked
  word itself, its continuation, what stands where it stands, what it was answered with alone), with how many
  members;
* the beginning of a sequence (F26), the lived continuation of the last words (F16), a word never met read as the form
  it is of (F36), the lived situations most like the present one (F23);
* what the act was expected to lead to (good or bad, L5 and F35), when that weighed in the choice;
* a contradiction, when another of these readings expected another word over the speaking line (it would have said
  it): the context's own record against a generalisation (an exception), a generalisation against a continuation, and
  so on. The word said is the decision's; the contradiction says which source it overrode.

It reads only, and nothing in it decides anything: it says where the decision came from.
"""
from __future__ import annotations

from typing import Dict, Optional

from .evidence import conviction, expectation, frequency

RELATION = {-1: "sorulan kelimenin kendisi", -2: "kelimenin devamı", -3: "dizinin başı", -4: "yerine geçen kelime",
            -5: "tek başına aldığı cevap"}


def attribute(mind, unit: int, consequence: float = 0.0) -> Dict[str, object]:
    """The source of the decided word ``unit`` in the present context, as a small record."""
    from . import read as R
    from .events import ACTED
    from .form import parts_of
    ch, symbol = mind.symbol_of(unit)
    record: Dict[str, object] = {"word": str(symbol), "source": "bilinmiyor", "detail": ""}
    if not isinstance(symbol, str) and getattr(mind.cfg, "wholes", False):
        from .butun import show                           # F43: a whole, a pointer or a sequence held as one item
        record["word"] = show(symbol)
    if "_part_depth" in mind.__dict__:
        record["depth"] = max(0, int(mind.__dict__["_part_depth"]))   # F43: how deep in parts it was decided
    context = mind.context
    if context is None:
        record.update(source="bağlamsız", detail="")
        return record
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    own = R.own_mass(mind, context, 1, trust).get(unit)
    if own is not None and expectation(own[0], own[1], mind.cfg.k) > 0.5 and conviction(sum(own), mind.cfg.k) >= 0.5:
        record.update(source="yaşanmış", detail="bu bağlamdan sonra bu kelime yaşandı (hatırlama, kanıt %.1f)"
                      % sum(own))
        if getattr(mind.cfg, "told_apart", False):       # F45: what a word never said alone answers with was inferred
            from .events import INFERRED
            direct = trust.clone()
            direct[INFERRED] = 0.0
            lived = R.own_mass(mind, context, 1, direct).get(unit)
            if lived is None or sum(lived) < 0.5 * sum(own):
                record.update(source="çıkarım", detail="tek kelimesi farklı iki diyalogun cevaplarındaki farktan "
                              "çıkarıldı (fark yöntemi, kanıt %.1f)" % sum(own))
    elif _known_claim(mind, context, unit, trust, record):
        pass                                             # F41: a thing known of the topic, not said yet
    else:
        claim = _frame_claim(mind, context, unit, trust)
        if claim is not None:
            voters, key, e = claim
            relation = RELATION.get(key, "aynı kelime") if isinstance(key, int) else "aynı kelime"
            record.update(source="genelleme", detail="%d üyenin aynı yerde yaşadığı (%s), e=%.2f" % (voters, relation, e))
        else:
            for name, reading, label in (
                    ("dizi başı", lambda: R.start_continuations(mind, context) if R._begins(mind, context) else {},
                     "bu kelimeyle başlayan dizilerin devamı"),
                    ("devam", lambda: R.continuation_backoff(mind, context, trust), "son kelimelerin yaşanmış devamı"),
                    ("biçim", lambda: R.form_reading(mind, context, trust), "hiç görülmemiş kelime, kökü olarak okundu"),
                    ("benzer durum", lambda: R.situation_backoff(mind, context, trust),
                     "en benzer yaşanmış durumların devamı")):
                e = reading().get(unit)
                if e is not None and e > 0.5:
                    record.update(source=name, detail="%s, e=%.2f" % (label, e))
                    break
    if consequence:
        record["detail"] = str(record["detail"]) + "; sonuç beklentisi %+.2f" % consequence
    clash = _contradiction(mind, context, unit, trust)
    if clash is not None:
        source, other, e = clash
        record["conflict"] = "%s '%s' bekliyordu (e=%.2f)" % (source, other, e)
    return record


def _known_claim(mind, context: int, unit: int, trust, record) -> bool:
    """F41: the word was said as what is known of the filler (the frame's members said things known of themselves
    there), or as how the members went on where their own known sequence stood as the topic's does; fills ``record``."""
    from . import read as R
    if not getattr(mind.cfg, "known", False):
        return False
    claim = R._moment_memo(mind).get(("known_claim", context))
    if claim is not None and claim[2] == unit:
        record.update(source="bilinen", detail="'%s' hakkında bilinen ve bu cevapta henüz söylenmemiş (üyeler de kendi "
                      "bildiklerini söyledi)" % str(mind.symbol_of(claim[0])[1]))
        return True
    topic = getattr(mind, "topic", None)
    e = R.sequence_reading(mind, context, trust).get(unit) if topic is not None else None
    if e is not None and e > 0.5:
        record.update(source="bilinen dizi", detail="'%s' hakkında bildiklerini anlatan cevap: üyeler kendi bildikleri "
                      "dizi böyle durduğunda böyle sürdürdü (söylenmemiş pay %.2f), e=%.2f"
                      % (str(mind.symbol_of(topic[0])[1]), R.unsaid_share(mind, topic[0], trust), e))
        return True
    return False


def _contradiction(mind, context: int, unit: int, trust) -> Optional[tuple]:
    """(source, word, e) of the strongest reading that expected another word than ``unit`` over the speaking line, or
    None: the context's own lived record, the frame that would speak (its top value), the lived continuation of the
    last words, the most similar lived situations."""
    from . import read as R
    cfg = mind.cfg
    line = 0.5 + cfg.theta / (2.0 * cfg.T)
    channel = mind.symbol_of(unit)[0]

    def other_top(reading):
        best = None
        for v, e in reading.items():
            if v == unit or e <= line or not R._world(mind, v) or mind.symbol_of(v)[0] != channel:
                continue
            if best is None or e > best[1]:
                best = (v, e)
        return best

    own = {v: expectation(p, n, cfg.k) for v, (p, n) in R.own_mass(mind, context, 1, trust).items()}
    readings = [("yaşanmış kayıt", own), ("genelleme", _frame_values(mind, context, trust)),
                ("devam", R.continuation_backoff(mind, context, trust)),
                ("benzer durum", R.situation_backoff(mind, context, trust))]
    found = None
    for source, reading in readings:
        top = other_top(reading)
        if top is not None and (found is None or top[1] > found[2]):
            other = mind.symbol_of(top[0])[1]
            if not isinstance(other, str) and getattr(cfg, "wholes", False):
                from .butun import show                  # F43: a pointer, a whole or a sequence held as one item
                other = show(other)
            found = (source, str(other), top[1])
    return found


def _frame_values(mind, context: int, trust) -> Dict[int, float]:
    """The values the frame that would speak claims for the context (members' reading, F11), resolved."""
    from . import read as R
    from .form import mv_view, parts_of
    from .store import COMPOSED
    cfg = mind.cfg
    frames = [(g, e, None) for g, e in R.generalisations(mind, context)]
    if getattr(cfg, "bind_relations", False):
        frames += R.bound_frames(mind, context)
    best = None
    for general, e_abs, tail in frames:
        mass, members, where, fixed, voters = R.member_votes(mind, general, context, trust, tail)
        if not mass:
            continue
        c_members = conviction(members, cfg.k)
        score = e_abs * c_members * max(frequency(p, n) for p, n in mass.values())
        if best is None or score > best[0]:
            best = (score, e_abs, c_members, mass, where, fixed, general)
    if best is None:
        return {}
    _, e_abs, c_members, mass, where, fixed, frame = best
    c = cfg.lam * e_abs * c_members
    filler = parts_of(mind, context)[where[0]] if where and mv_view(mind, "u_kind")[context] & COMPOSED else None
    category = R.frame_category(mind, frame)
    out: Dict[int, float] = {}
    for key, (w_pos, w_neg) in mass.items():
        v = R.resolve(mind, key, filler, fixed, category)
        if v is not None:
            out[v] = max(out.get(v, 0.0), c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5)
    return out


def _frame_claim(mind, context: int, unit: int, trust) -> Optional[tuple]:
    """(voters, key, e) of the frame that would speak for ``unit`` in the members' reading, or None."""
    from . import read as R
    from .form import mv_view, parts_of
    from .store import COMPOSED
    cfg = mind.cfg
    frames = [(g, e, None) for g, e in R.generalisations(mind, context)]
    if getattr(cfg, "bind_relations", False):
        frames += R.bound_frames(mind, context)
    best = None
    for general, e_abs, tail in frames:
        mass, members, where, fixed, voters = R.member_votes(mind, general, context, trust, tail)
        if not mass:
            continue
        c_members = conviction(members, cfg.k)
        score = e_abs * c_members * max(frequency(p, n) for p, n in mass.values())
        if best is None or score > best[0]:
            best = (score, e_abs, c_members, mass, where, fixed, voters, general)
    if best is None:
        return None
    _, e_abs, c_members, mass, where, fixed, voters, frame = best
    c = cfg.lam * e_abs * c_members
    filler = parts_of(mind, context)[where[0]] if where and mv_view(mind, "u_kind")[context] & COMPOSED else None
    category = R.frame_category(mind, frame)
    found = None
    for key, (w_pos, w_neg) in mass.items():
        if R.resolve(mind, key, filler, fixed, category) != unit:
            continue
        e = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
        if found is None or e > found[2]:
            found = (voters, key, e)
    return found if found is not None and found[2] > 0.5 else None


def report(why) -> str:
    """The lines ``!neden`` prints for the last answer."""
    if not why:
        return "  son cevapta açıklanacak bir kelime yok (sessiz kaldı ya da açıklama kapalıydı)"
    lines = []
    for r in why:
        inner = "  " * int(r.get("depth", 0))               # F43: a word decided while answering a part, under its part
        lines.append("  %s%-14s %-13s %s" % (inner, r["word"], r["source"], r["detail"]))
        if r.get("conflict"):
            lines.append("  %s%-14s %-13s %s" % (inner, "", "çelişki", r["conflict"]))
    counts: Dict[str, int] = {}
    for r in why:
        counts[str(r["source"])] = counts.get(str(r["source"]), 0) + 1
    lines.append("  özet: " + ", ".join("%s %d" % (k, v) for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    return "\n".join(lines)
