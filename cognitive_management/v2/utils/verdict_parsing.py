"""Negation- and hedge-aware parsing of judge replies (fact check, constitutional review).

The critics ask the model for a one-word or one-number verdict. Earlier parsers
tested ``"doğru" in reply`` before any negation, so ``"Yanlış, doğru değil"`` and
``"Kısmen doğru"`` both scored as true, ``"uygun değil, ihlal var"`` as compliant,
``"ihlal yok"`` as a violation, ``"0,8"`` as ``0.0`` (Turkish decimal comma), and
an empty or nonsense reply as a moderate violation that triggered a revision.

Rules used here:

* The reply is lowercased with Turkish rules and folded to ASCII, so ``"YANLIŞ"``,
  ``"yanlış"`` and ``"yanlis"`` are the same cue.
* Only the first sentence that contains a verdict cue is read; a model that keeps
  writing after its verdict does not change it.
* A cue is negated by a following Turkish negator (``değil``, ``yok``, ``olmayan``,
  negative verb forms such as ``etmiyor``) or a preceding one (``not``, ``no``,
  ``non``, ``n't``, ``hiçbir``). The scope stops at a comma or at the next cue, so in
  ``"Yanlış, doğru değil"`` only ``doğru`` is negated. Negated "true" is "false",
  negated "violation" is "compliant", and vice versa.
* A hedge (``kısmen``, ``partially``) makes the verdict partial; an uncertainty cue
  (``belirsiz``, ``emin değilim``, ``muhtemelen``) makes it uncertain.
* Opposite cues without a hedge, an echoed list of options (``... veya ...``) or no
  cue at all give ``UNPARSEABLE``. Callers must treat that as an abstention: it is
  never evidence for a revision.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

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
    evidence: str = ""              # the (folded) sentence the verdict was read from

    @property
    def parsed(self) -> bool:
        return self.label != UNPARSEABLE


_WORD = re.compile(r"[^\W_]+")
_SENTENCE = re.compile(r"[.!?;:\n]+")
_ALTERNATIVES = frozenset({"veya", "yahut", "or"})
_PRE_NEGATORS = frozenset({"not", "no", "never", "non", "without", "neither", "nor", "hicbir"})
_POST_NEGATORS = frozenset({"yok", "yoktur", "olmayan", "olmadigi", "olmadigini", "sayilmaz", "olamaz"})
_NEGATIVE_VERB = re.compile(r"(miyor|muyor|mez|maz|memis|mamis|medi|madi|memekte|mamakta|meyen|mayan)")
_OPPOSITE = {TRUE: FALSE, FALSE: TRUE, COMPLIANT: VIOLATION, VIOLATION: COMPLIANT}

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


def _rule_cue(token: str) -> Optional[str]:
    if token in _RULE_EXACT:
        return _RULE_EXACT[token]
    if token.startswith(("uygunsuz", "uyumsuz")):
        return VIOLATION
    if token.startswith(("uygun", "uyumlu")):
        return COMPLIANT
    if token.startswith(("ihlal", "aykiri")):
        return VIOLATION
    return None


def _negated(tokens: Sequence[str], index: int, cue_of) -> bool:
    """Is the cue at ``tokens[index]`` negated within its clause?"""
    for token in tokens[index + 1:index + 4]:
        if token in _POST_NEGATORS or token.startswith("degil") or _NEGATIVE_VERB.search(token):
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


def _sentences(text: str) -> List[Tuple[str, List[List[str]]]]:
    """(sentence, clauses) of the ASCII-folded text; each clause is a token list."""
    folded = ascii_fold(text)
    return [(part.strip(), [_WORD.findall(clause) for clause in part.split(",")])
            for part in _SENTENCE.split(folded) if part.strip()]


def _read(text: str, cue_of, special=None) -> Verdict:
    """Verdict of the first sentence that contains a cue."""
    for sentence, clauses in _sentences(text or ""):
        raw, resolved = [], []
        for tokens in clauses:
            for index, token in enumerate(tokens):
                label = cue_of(token)
                if label is None:
                    label = special(tokens, index, cue_of) if special else None
                    if label is not None:      # special cues resolve their own negation
                        raw.append(label)
                        resolved.append(label)
                    continue
                raw.append(label)
                if not _negated(tokens, index, cue_of):
                    resolved.append(label)
                elif label in _OPPOSITE:
                    resolved.append(_OPPOSITE[label])
                # a negated hedge or uncertainty cue carries no verdict
        if not raw:
            continue
        words = {token for tokens in clauses for token in tokens}
        if len(set(raw)) >= 2 and _ALTERNATIVES & words:
            return Verdict(UNPARSEABLE, evidence=sentence)      # an echoed list of options
        labels = set(resolved)
        if PARTIAL in labels:
            return Verdict(PARTIAL, evidence=sentence)
        if UNCERTAIN in labels:
            return Verdict(UNCERTAIN, evidence=sentence)
        if len(labels) == 1:
            return Verdict(labels.pop(), evidence=sentence)
        return Verdict(UNPARSEABLE, evidence=sentence)           # conflicting cues
    return Verdict(UNPARSEABLE)


def _certainty(tokens: Sequence[str], index: int, cue_of) -> Optional[str]:
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


def _numeric_score(text: str) -> Optional[float]:
    """First number that reads as a 0-1 score: ``0.8``, ``0,8``, ``%80``, ``80%``,
    ``8/10``. A number above 1 without a unit counts only at the start of the reply
    (optionally after a label such as ``Skor:``), on a 0-10 or 0-100 scale; an ordinal
    such as ``"2. prensip"`` is not a score."""
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
            return min(1.0, max(0.0, score))
    return None


def parse_violation_verdict(text: str) -> Verdict:
    """Constitutional reply -> SCORED (a number), COMPLIANT, PARTIAL, VIOLATION or
    UNPARSEABLE; ``score`` is the violation score in ``[0, 1]`` or ``None``."""
    normalized = unicodedata.normalize("NFC", text or "").strip()
    score = _numeric_score(normalized)
    if score is not None:
        return Verdict(SCORED, score=score, evidence=normalized[:80])
    verdict = _read(normalized, _rule_cue)
    if verdict.label in VIOLATION_SCORES:
        return Verdict(verdict.label, score=VIOLATION_SCORES[verdict.label], evidence=verdict.evidence)
    return Verdict(UNPARSEABLE, evidence=verdict.evidence)


def parse_violation_score(text: str) -> Optional[float]:
    """Violation score in ``[0, 1]``, or ``None`` when the reply cannot be read."""
    return parse_violation_verdict(text).score
