"""CJSON-1: the canonical encoding of structured key payloads, and domain-separated digests.

Rules (they never change; a new meaning is a new scheme):
  - keys sorted, no whitespace, UTF-8, ``ensure_ascii=False``;
  - strings are data and are never normalised (a precomposed and a decomposed token differ);
  - ``bool`` stays ``bool``; integers must fit in 64 bits; floats must be finite, ``-0.0`` becomes
    ``0.0`` and they print with Python's shortest round-trip ``repr``;
  - only ``None``, ``bool``, ``int``, ``float``, ``str``, ``list``/``tuple`` and ``dict`` with string
    keys are accepted; nesting is limited to 64 levels.
Every structured key is ``sha256(b"cev:" + kind + b":" + scheme + b"\\x00" + CJSON-1(payload))``, so
a digest of one scheme can never be mistaken for a digest of another.
"""
import hashlib
import json
import math

from cevahir_ai.errors import CanonicalError

MAX_DEPTH = 64
MAX_BYTES = 256 * 1024 * 1024


def _norm(value, depth=0):
    if depth > MAX_DEPTH:
        raise CanonicalError("nesting deeper than 64")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if not -2**63 <= value < 2**63:
            raise CanonicalError("int out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError("non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, (list, tuple)):
        return [_norm(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalError("non-str key")
        return {key: _norm(item, depth + 1) for key, item in value.items()}
    raise CanonicalError(f"not canonical: {type(value).__name__}")


def cjson(obj):
    """Canonical bytes of ``obj``. A lone surrogate in a string raises (strict UTF-8)."""
    text = json.dumps(_norm(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CanonicalError(f"string is not valid Unicode text: {error.reason}") from error
    if len(data) > MAX_BYTES:
        raise CanonicalError("payload larger than 256 MiB")
    return data


def domain(kind, scheme):
    return b"cev:" + kind.encode("ascii") + b":" + scheme.encode("ascii") + b"\x00"


def structured_digest(kind, scheme, payload):
    """Hex sha256 of the domain prefix followed by the CJSON-1 payload."""
    return hashlib.sha256(domain(kind, scheme) + cjson(payload)).hexdigest()


# ------------------------------------------------------------------------------- typed projections
# Each key kind hashes a typed projection: every field is coerced to its declared type first, so
# 1, 1.0 and True can never produce two keys for the same meaning.
def as_int(value, field="value"):
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise CanonicalError(f"{field} must be an integer, not {value!r}")
    return value


def as_float(value, field="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalError(f"{field} must be a number, not {value!r}")
    return float(value)


def as_bool(value, field="value"):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise CanonicalError(f"{field} must be a boolean, not {value!r}")


def as_str(value, field="value"):
    if not isinstance(value, str):
        raise CanonicalError(f"{field} must be a string, not {value!r}")
    return value


def optional(coerce):
    def inner(value, field="value"):
        return None if value is None else coerce(value, field)
    return inner


COERCE = {"int": as_int, "float": as_float, "bool": as_bool, "str": as_str,
          "int?": optional(as_int), "float?": optional(as_float), "str?": optional(as_str),
          "bool?": optional(as_bool)}
