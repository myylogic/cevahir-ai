"""Key grammar: ``cev:<kind>:<scheme>:<digest>``.

    kind    = [a-z][a-z0-9]{1,15}
    scheme  = name "-v" digits, name = [a-z0-9]+("-"[a-z0-9]+)*
    digest  = lowercase hex, 64 digits (sha256) or 32 (md5 cache names and occurrence ids)

The scheme is part of the key, so a 64-hex value of one scheme is never taken for another, and a
version bump is a new scheme linked by an alias instead of a "mismatch".
"""
import re
from dataclasses import dataclass

from cevahir_ai.errors import KeyFormatError, UnknownSchemeError

KEY_RE = re.compile(r"^cev:([a-z][a-z0-9]{1,15}):([a-z0-9]+(?:-[a-z0-9]+)*-v[0-9]+):([0-9a-f]{32}|[0-9a-f]{64})$")
_PART_RES = (("prefix", re.compile(r"cev")), ("kind", re.compile(r"[a-z][a-z0-9]{1,15}")),
             ("scheme", re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-v[0-9]+")),
             ("digest", re.compile(r"[0-9a-f]{32}|[0-9a-f]{64}")))


@dataclass(frozen=True, order=True)
class Key:
    kind: str
    scheme: str
    digest: str

    def __post_init__(self):
        from cevahir_ai.keys.registry import SCHEMES
        text = f"cev:{self.kind}:{self.scheme}:{self.digest}"
        if not KEY_RE.match(text):
            raise KeyFormatError(f"not a valid key: {text!r}", _error_position(text))
        declared = SCHEMES.get((self.kind, self.scheme))
        if declared is None:
            raise UnknownSchemeError(f"unknown key scheme {self.kind}:{self.scheme}", len("cev:"))
        if len(self.digest) != declared.digest_len:
            raise KeyFormatError(f"{self.kind}:{self.scheme} digests have {declared.digest_len} hex digits, "
                                 f"not {len(self.digest)}", text.rindex(":") + 1)

    def __str__(self):
        return f"cev:{self.kind}:{self.scheme}:{self.digest}"

    @classmethod
    def parse(cls, text):
        if not isinstance(text, str):
            raise KeyFormatError(f"a key is a string, not {type(text).__name__}")
        match = KEY_RE.match(text)
        if not match:
            raise KeyFormatError(f"not a valid key: {text!r}", _error_position(text))
        return cls(*match.groups())

    @property
    def status(self):
        from cevahir_ai.keys.registry import SCHEMES
        return SCHEMES[(self.kind, self.scheme)].status

    def display(self):
        """Short form: ``kind:`` plus 12 hex digits; a legacy scheme also shows its scheme name."""
        if self.status in ("legacy", "observed", "signature"):
            return f"{self.kind}:{self.scheme}:{self.digest[:12]}"
        return f"{self.kind}:{self.digest[:12]}"


def _error_position(text):
    position = 0
    for index, (part, (_, pattern)) in enumerate(zip(text.split(":", 3), _PART_RES)):
        if not pattern.fullmatch(part):
            return position
        position += len(part) + 1
        if index == 3:
            break
    return min(position, len(text))


def parse_key(text):
    return Key.parse(text)


def is_key(text):
    try:
        Key.parse(text)
    except (KeyFormatError, TypeError):
        return False
    return True
