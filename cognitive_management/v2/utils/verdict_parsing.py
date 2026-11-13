"""Negation- and hedge-aware parsing of judge replies (fact check, constitutional review).

The critics ask the model for a one-word or one-number verdict. Earlier parsers
tested ``"doğru" in reply`` before any negation, so ``"Yanlış, doğru değil"`` and
``"Kısmen doğru"`` both scored as true, ``"uygun değil, ihlal var"`` as compliant,
``"ihlal yok"`` as a violation, ``"0,8"`` as ``0.0`` (Turkish decimal comma), and
an empty or nonsense reply as a moderate violation that triggered a revision.

Rules used here:

* The reply is lowercased with Turkish rules and folded to ASCII, so ``"YANLIŞ"``,
  ``"yanlış"`` and ``"yanlis"`` are the same cue.
* The reply is read in segments that end at ``. ! ? ; :`` or a line break. The
  first segment (or label/answer pair, below) that holds a verdict decides; a model
  that keeps writing after its verdict does not change it.
* A segment that ends with ``:`` is a label and one that ends with ``?`` is a
  question; the next segment is its value or answer. An answer with its own cue
  decides (``"Doğru mu? Hayır, yanlış."`` is false). A bare answer is applied to
  the cue of the label or question: ``evet``/``hayır``/``yes``/``no`` agree or
  disagree with what it states (``"Uygun: hayır"`` is a violation), while
  ``var``/``yok``/``none``/``değil`` assert or deny the cue word itself
  (``"İhlal: yok"``, ``"İhlal var mı? Yok."`` and ``"Violation: none"`` are
  compliant). A label whose value has neither keeps its own verdict
  (``"Yanlış: nüfus artmadı"``); a question is never its own answer.
* A cue is negated by a following Turkish negator (``değil``, ``yok``, ``olmayan``,
  or a word that ends in a negative verb suffix: ``etmiyor``, ``etmez``,
  ``edilmedi``, ``olmayabilir``) or a preceding one (``not``, ``no``, ``non``,
  ``n't``, ``hiçbir``). Suffixes are matched at the end of the word, so
  ``medikal`` or ``immediately`` negate nothing. The scope stops at a comma or at
  the next cue, so in ``"Yanlış, doğru değil"`` only ``doğru`` is negated. Negated
  "true" is "false", negated "violation" is "compliant", and vice versa;
  ``ihlalsiz`` (without violation) is compliant.
* A hedge (``kısmen``, ``partially``) makes the verdict partial; an uncertainty cue
  (``belirsiz``, ``emin değilim``, ``muhtemelen``, ``olmayabilir``) makes it
  uncertain. Measure nouns (``doğruluk``, ``uygunluk``) are not verdicts.
* Opposite cues without a hedge, or no cue at all, give ``UNPARSEABLE``; an echoed
  list of options (``... veya ...``) is skipped. Callers must treat
  ``UNPARSEABLE`` as an abstention: it is never evidence for a revision.
* A constitutional reply is read as a number first. A plain number is the asked
  violation score (``"0.0 (tamamen uygun)"`` stays 0.0), but a number labelled as
  compliance (``"Uygunluk: 0.9"``), or a percentage or ratio attached to a
  compliance word (``"%100 uygun"``, ``"Uygun: %90"``), measures compliance, so the
  violation score is ``1 - x``; a negated one (``"%100 uygun değil"``) is
  ``UNPARSEABLE``.

This is a rule-based reader for short replies, not a parser of Turkish syntax.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, List, Optional, Sequence, Tuple

from .turkish_text import ascii_fold

__all__ = [
    "TRUE", "FALSE", "PARTIAL", "UNCERTAIN", "UNPARSEABLE",
    "COMPLIANT", "VIOLATION", "SCORED",
    "VIOLATION_SCORES",
    "Verdict",
    "parse_truth_verdict",
    "parse_violation_verdict",
    "parse_violation_score",
]

TRUE, FALSE, PARTIAL, UNCERTAIN, UNPARSEABLE = "true", "false", "partial", "uncertain", "unparseable"
COMPLIANT, VIOLATION, SCORED = "compliant", "violation", "scored"

#: Violation score of a verbal constitutional verdict (0 = compliant, 1 = severe).
VIOLATION_SCORES = {COMPLIANT: 0.0, PARTIAL: 0.5, VIOLATION: 0.8}


@dataclass(frozen=True)
class Verdict:
    label: str
    score: Optional[float] = None   # violation score; None when nothing could be read
    evidence: str = ""              # the (folded) segment(s) the verdict was read from

    @property
    def parsed(self) -> bool:
        return self.label != UNPARSEABLE


CueOf = Callable[[str], Optional[str]]

_WORD = re.compile(r"[^\W_]+")
_SEGMENT_END = re.compile(r"([.!?;:\n]+)")
_ALTERNATIVES = frozenset({"veya", "yahut", "or"})
_PRE_NEGATORS = frozenset({"not", "no", "never", "non", "without", "neither", "nor", "hicbir"})
_POST_NEGATORS = frozenset({"yok", "yoktur", "olmayan", "olmadigi", "olmadigini", "sayilmaz", "olamaz"})
# Negative verb forms, tested on the whole ASCII-folded word: a stem, the negative
# -mA- and a tense ending (+ person). Anchoring at the word end keeps words such as
# "medikal" or "immediately" out; the listed nouns merely end like such a form.
_NEGATIVE_VERB = re.compile(
    r"\w+?(?:m[iu]yor\w*"                                   # etmiyor, uymuyorlar
    r"|m[ae]m[ae]kt[ae]\w*"                                 # etmemekte(dir)
    r"|m[ae]m[iu]s\w*"                                      # etmemiş, edilmemiş(tir)
    r"|m[ae]y[ae](?:n|c[ae][kg]|bil)\w*"                    # etmeyen, etmeyecek, olmayabilir
    r"|m[ae]z(?:l[ae]r|d[iu]|m[iu]s|s[ae])?"                # etmez(ler), uymazdı, olmazsa
    r"|m[ae]d[iu](?:m|n|k\w*|n[iu]z|l[ae]r|g[iu]\w*)?)"     # etmedi(m), olmadığı, olmadıkça
)
_NOT_NEGATIVE = ("komedi", "namaz")
_MAX_VERB_LEN = 40
_OPPOSITE = {TRUE: FALSE, FALSE: TRUE, COMPLIANT: VIOLATION, VIOLATION: COMPLIANT}

# Bare answers to a label or question. Agreement words answer what the label or
# question states; existence words assert or deny its cue word itself.
_AGREE = {"evet": True, "yes": True, "yeah": True, "yep": True,
          "hayir": False, "no": False, "nope": False, "asla": False, "never": False}
_EXIST = {"var": True, "vardir": True, "mevcut": True, "mevcuttur": True,
          "yok": False, "yoktur": False, "none": False, "nothing": False, "degil": False,
          "degildir": False, "hic": False, "hicbiri": False, "sifir": False}
_FILLER = frozenset({"bence", "kesinlikle", "kesin", "tabii", "tabi", "elbette", "yani",
                     "efendim", "absolutely", "definitely", "certainly", "of", "course"})


def _is_negative_verb(token: str) -> bool:
    return (len(token) <= _MAX_VERB_LEN and _NEGATIVE_VERB.fullmatch(token) is not None
            and not any(noun in token for noun in _NOT_NEGATIVE))


# --- fact-check cues (ASCII-folded) --------------------------------------------
_TRUTH_EXACT = {
    "true": TRUE, "correct": TRUE, "accurate": TRUE,
    "false": FALSE, "wrong": FALSE, "untrue": FALSE, "incorrect": FALSE, "inaccurate": FALSE,
    "hatali": FALSE, "hatalidir": FALSE,
    "kismen": PARTIAL, "kismi": PARTIAL, "kismendir": PARTIAL, "partial": PARTIAL,
    "partially": PARTIAL, "partly": PARTIAL, "mostly": PARTIAL, "cogunlukla": PARTIAL,
    "belirsiz": UNCERTAIN, "belirsizdir": UNCERTAIN, "uncertain": UNCERTAIN, "unknown": UNCERTAIN,
    "unclear": UNCERTAIN, "unsure": UNCERTAIN, "unverifiable": UNCERTAIN, "bilinmiyor": UNCERTAIN,
    "bilinmez": UNCERTAIN, "bilmiyorum": UNCERTAIN, "kararsiz": UNCERTAIN, "kararsizim": UNCERTAIN,
    "supheli": UNCERTAIN, "muhtemelen": UNCERTAIN, "belki": UNCERTAIN, "olabilir": UNCERTAIN,
    "maybe": UNCERTAIN, "perhaps": UNCERTAIN, "possibly": UNCERTAIN, "probably": UNCERTAIN,
}
_CERTAINTY = frozenset({"emin", "eminim", "sure", "certain", "know"})  # only cues when negated


def _truth_cue(token: str) -> Optional[str]:
    if token in _TRUTH_EXACT:
        return _TRUTH_EXACT[token]
    if token.startswith("olmayabil"):                # olmayabilir: may not be
        return UNCERTAIN
    if token.startswith(("dogruluk", "dogrulug")):   # "accuracy": a noun, not a verdict
        return None
    if token.startswith("dogrula"):                  # doğrulandı / doğrulanamaz
        return UNCERTAIN if re.search(r"(amaz|amadi|anmadi|anmamis|emez|emedi)", token) else TRUE
    if token.startswith("dogru"):                    # doğru, doğrudur, doğruydu
        return TRUE
    if token.startswith("yanlis") and not token.startswith("yanlislik"):  # not "yanlışlıkla"
        return FALSE
    return None


# --- constitutional cues (ASCII-folded) ----------------------------------------
_RULE_EXACT = {
    "compliant": COMPLIANT, "complies": COMPLIANT, "acceptable": COMPLIANT,
    "appropriate": COMPLIANT, "good": COMPLIANT, "uyuyor": COMPLIANT, "uymaktadir": COMPLIANT,
    "violation": VIOLATION, "violations": VIOLATION, "violate": VIOLATION, "violates": VIOLATION,
    "violated": VIOLATION, "violating": VIOLATION, "noncompliant": VIOLATION,
    "inappropriate": VIOLATION, "unacceptable": VIOLATION, "bad": VIOLATION,
    "uymuyor": VIOLATION, "uymaz": VIOLATION, "uymamakta": VIOLATION, "uymamaktadir": VIOLATION,
    "kismen": PARTIAL, "kismi": PARTIAL, "partial": PARTIAL, "partially": PARTIAL,
    "partly": PARTIAL, "moderate": PARTIAL, "moderately": PARTIAL, "somewhat": PARTIAL,
}
# What a compliance score measures (uygunluk, uygunluğu, uyumluluk): a noun, not a verdict.
_COMPLIANCE_NOUNS = ("uygunlu", "uyumlulu", "compliance")


def _rule_cue(token: str) -> Optional[str]:
    if token in _RULE_EXACT:
        return _RULE_EXACT[token]
    if token.startswith(_COMPLIANCE_NOUNS):
        return None
    if token.startswith(("uygunsuz", "uyumsuz")):
        return VIOLATION
    if token.startswith(("uygun", "uyumlu")):
        return COMPLIANT
    if token.startswith("ihlal"):                    # ihlalsiz: without violation
        return COMPLIANT if token[5:].startswith(("siz", "suz")) else VIOLATION
    if token.startswith("aykiri"):
        return VIOLATION
    return None


# --- reading -------------------------------------------------------------------

def _negated(tokens: Sequence[str], index: int, cue_of: CueOf) -> bool:
    """Is the cue at ``tokens[index]`` negated within its clause?"""
    for token in tokens[index + 1:index + 4]:
        if token in _POST_NEGATORS or token.startswith("degil") or _is_negative_verb(token):
            return True
        if cue_of(token) is not None:
            break
    for position in range(index - 1, max(-1, index - 3), -1):
        token = tokens[position]
        if token in _PRE_NEGATORS:
            return True
        if token == "t" and position > 0 and tokens[position - 1].endswith("n"):  # isn't, don't
            return True
        if cue_of(token) is not None:
            break
    return False


@dataclass(frozen=True)
class _Segment:
    text: str                          # folded text without the closing punctuation
    clauses: Tuple[Tuple[str, ...], ...]
    kind: str                          # "statement", "label" (ends with ':') or "question"

    @property
    def words(self) -> frozenset:
        return frozenset(token for clause in self.clauses for token in clause)


def _segments(text: str) -> List[_Segment]:
    """Segments of the ASCII-folded text with the punctuation that closes them."""
    parts = _SEGMENT_END.split(ascii_fold(text))
    segments = []
    for position in range(0, len(parts), 2):
        body = parts[position].strip()
        if not body:
            continue
        end = parts[position + 1] if position + 1 < len(parts) else ""
        kind = "question" if "?" in end else "label" if ":" in end else "statement"
        clauses = tuple(tuple(_WORD.findall(clause)) for clause in body.split(","))
        segments.append(_Segment(body, clauses, kind))
    return segments


def _clause_cues(tokens: Sequence[str], cue_of: CueOf, special=None) -> Tuple[List[str], List[str]]:
    """(cue labels as written, cue labels after negation) of one clause."""
    raw, resolved = [], []
    for index, token in enumerate(tokens):
        label = cue_of(token)
        if label is None:
            label = special(tokens, index, cue_of) if special else None
            if label is not None:          # special cues resolve their own negation
                raw.append(label)
                resolved.append(label)
            continue
        raw.append(label)
        if not _negated(tokens, index, cue_of):
            resolved.append(label)
        elif label in _OPPOSITE:
            resolved.append(_OPPOSITE[label])
        # a negated hedge or uncertainty cue carries no verdict
    return raw, resolved


def _cues(segment: _Segment, cue_of: CueOf, special=None) -> Tuple[List[str], List[str]]:
    """(cue labels as written, cue labels after negation) of the segment."""
    raw, resolved = [], []
    for tokens in segment.clauses:
        clause_raw, clause_resolved = _clause_cues(tokens, cue_of, special)
        raw += clause_raw
        resolved += clause_resolved
    return raw, resolved


def _bare_answers(segment: _Segment) -> List[Tuple[bool, bool]]:
    """(is an agreement word, is positive) for each yes/no word in the segment's
    bare-answer clauses: clauses made only of such words and fillers ("Hayır",
    "Bence yok", "Mevcut değil")."""
    found = []
    for tokens in segment.clauses:
        if not tokens or not all(t in _AGREE or t in _EXIST or t in _FILLER for t in tokens):
            continue
        for position, token in enumerate(tokens):
            for table, agreement in ((_AGREE, True), (_EXIST, False)):
                if token in table:
                    positive = table[token]
                    if positive and tokens[position + 1:position + 2] and \
                            tokens[position + 1].startswith("degil"):
                        positive = False                        # "mevcut değil"
                    found.append((agreement, positive))
    return found


def _decide(labels: Sequence[str]) -> str:
    labels = set(labels)
    if PARTIAL in labels:
        return PARTIAL
    if UNCERTAIN in labels:
        return UNCERTAIN
    if len(labels) == 1:
        return labels.pop()
    return UNPARSEABLE                                      # conflicting cues, or none left


def _statement(segment: _Segment, raw: List[str], resolved: List[str]) -> Optional[str]:
    """Verdict of a segment read on its own; None when it holds none."""
    if not raw:
        return None
    if len(set(raw)) >= 2 and _ALTERNATIVES & segment.words:
        return None                                         # an echoed list of options
    return _decide(resolved)


def _answered(stated: List[str], written: List[str], answers: List[Tuple[bool, bool]]) -> List[str]:
    """Labels that bare yes/no answers give to a label or question whose cues read
    ``stated`` after negation and ``written`` as written."""
    labels = []
    for agreement, positive in answers:
        reference = set(stated if agreement else written)
        if len(reference) != 1:
            continue                                        # nothing (or two things) to answer
        label = reference.pop()
        if positive:
            labels.append(label)
        else:
            labels.append(_OPPOSITE.get(label, UNPARSEABLE))  # a hedge has no opposite
    return labels


def _read(text: str, cue_of: CueOf, special=None) -> Verdict:
    """Verdict of the first segment, or label/answer pair, that holds one."""
    segments = _segments(text or "")
    index = 0
    while index < len(segments):
        head = segments[index]
        raw, resolved = _cues(head, cue_of, special)
        answer = segments[index + 1] if index + 1 < len(segments) else None
        answers = _bare_answers(answer) if answer is not None else []
        if head.kind != "statement" and answer is not None and (answer.kind == "statement" or answers):
            answer_raw, answer_resolved = _cues(answer, cue_of, special)
            labels = answer_resolved + _answered(resolved, raw, answers)
            evidence = f"{head.text} | {answer.text}"
            if answer_raw or labels:
                if not (len(set(answer_raw)) >= 2 and _ALTERNATIVES & answer.words):
                    return Verdict(_decide(labels), evidence=evidence)
            elif head.kind == "label":
                verdict = _statement(head, raw, resolved)  # "Yanlış: <explanation>"
                if verdict is not None:
                    return Verdict(verdict, evidence=evidence)
            index += 2
            continue
        if head.kind != "question":                         # a question never answers itself
            verdict = _statement(head, raw, resolved)
            if verdict is not None:
                return Verdict(verdict, evidence=head.text)
        index += 1
    return Verdict(UNPARSEABLE)


def _certainty(tokens: Sequence[str], index: int, cue_of: CueOf) -> Optional[str]:
    """``emin değilim`` / ``not sure`` / ``don't know`` -> UNCERTAIN."""
    if tokens[index] in _CERTAINTY and _negated(tokens, index, cue_of):
        return UNCERTAIN
    return None


def parse_truth_verdict(text: str) -> Verdict:
    """Fact-check reply -> TRUE, FALSE, PARTIAL, UNCERTAIN or UNPARSEABLE."""
    return _read(unicodedata.normalize("NFC", text or ""), _truth_cue, _certainty)


# --- numeric violation scores ----------------------------------------------------
_NUMBER = re.compile(r"(?<![\w.,%])(%\s*)?(\d+(?:[.,]\d+)?)(\s*%|\s*/\s*(?:100|10)(?!\d))?")
_LEADING_LABEL = re.compile(r"^[^\W\d_]{1,20}(?:\s+[^\W\d_]{1,20})?\s*[:=]\s*$")
_ORDINAL_TAIL = re.compile(r"\.\s+[^\W\d_]")
# Up to three words right before the number in its segment ("Uygunluk skoru:").
_NUMBER_LABEL = re.compile(r"\s*([^\W\d_]+(?:\s+[^\W\d_]+){0,2})\s*[:=]?\s*")
_MEASURE_LINKS = frozenset({"oraninda", "derecede", "olcude", "seviyesinde"})


def _numeric_score(text: str) -> Optional[Tuple[float, "re.Match[str]", bool]]:
    """(score, match, has a unit) of the first number that reads as a 0-1 score:
    ``0.8``, ``0,8``, ``%80``, ``80%``, ``8/10``. A number above 1 without a unit
    counts only at the start of the reply (optionally after a label such as
    ``Skor:``), on a 0-10 or 0-100 scale; an ordinal such as ``"2. prensip"`` is not
    a score."""
    for match in _NUMBER.finditer(text):
        percent_prefix, digits, suffix = match.group(1), match.group(2), (match.group(3) or "").strip()
        value = float(digits.replace(",", "."))
        decimal = "," in digits or "." in digits
        before = text[:match.start()]
        leading = not before.strip() or bool(_LEADING_LABEL.match(before))
        if percent_prefix or suffix == "%":
            score = value / 100.0 if value <= 100 else None
        elif suffix.startswith("/"):
            denominator = float(suffix[1:].strip())
            score = value / denominator if value <= denominator else None
        elif decimal and value <= 1.0:
            score = value
        elif leading:
            if not decimal and _ORDINAL_TAIL.match(text[match.end():]):
                continue
            score = (value if value <= 1.0 else value / 10.0 if value <= 10
                     else value / 100.0 if value <= 100 else None)
        else:
            continue
        if score is not None:
            return min(1.0, max(0.0, score)), match, bool(percent_prefix or suffix)
    return None


def _number_measures(text: str, match: "re.Match[str]", has_unit: bool) -> Optional[str]:
    """What the number measures: COMPLIANT, VIOLATION (the asked score, the default)
    or None when a negated word makes it unreadable (``"%100 uygun değil"``)."""
    following = re.split(r"[,.;!?:()\n]", ascii_fold(text[match.end():]), maxsplit=1)[0]
    after = _WORD.findall(following)
    if after and after[0] in _MEASURE_LINKS:                  # "%90 oranında uygun"
        after = after[1:]
    if has_unit and after:
        label = COMPLIANT if after[0].startswith(_COMPLIANCE_NOUNS) else _rule_cue(after[0])
        if label in (COMPLIANT, VIOLATION):
            return None if _negated(after, 0, _rule_cue) else label
    preceding = re.split(r"[.!?;\n]", ascii_fold(text[:match.start()]))[-1]
    label_match = _NUMBER_LABEL.fullmatch(preceding)
    if label_match:
        words = label_match.group(1).split()
        _, stated = _clause_cues(words, _rule_cue)
        if VIOLATION not in stated and (any(w.startswith(_COMPLIANCE_NOUNS) for w in words)
                                         or (has_unit and COMPLIANT in stated)):
            return COMPLIANT
    return VIOLATION


def parse_violation_verdict(text: str) -> Verdict:
    """Constitutional reply -> SCORED (a number), COMPLIANT, PARTIAL, VIOLATION or
    UNPARSEABLE; ``score`` is the violation score in ``[0, 1]`` or ``None``."""
    normalized = unicodedata.normalize("NFC", text or "").strip()
    number = _numeric_score(normalized)
    if number is not None:
        score, match, has_unit = number
        measures = _number_measures(normalized, match, has_unit)
        if measures is None:
            return Verdict(UNPARSEABLE, evidence=normalized[:80])
        if measures == COMPLIANT:
            score = 1.0 - score
        return Verdict(SCORED, score=score, evidence=normalized[:80])
    verdict = _read(normalized, _rule_cue)
    if verdict.label in VIOLATION_SCORES:
        return Verdict(verdict.label, score=VIOLATION_SCORES[verdict.label], evidence=verdict.evidence)
    return Verdict(UNPARSEABLE, evidence=verdict.evidence)


def parse_violation_score(text: str) -> Optional[float]:
    """Violation score in ``[0, 1]``, or ``None`` when the reply cannot be read."""
    return parse_violation_verdict(text).score
