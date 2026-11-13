"""Training-cache and split keys.

``cache:recipe-v1`` is input-addressed ("do we already have this cache?"): corpus key, tokenizer
key, encoding profile, record format and builder generation. Unlike the legacy md5 file name it
does not depend on the working directory, on newline bytes or on runtime tokenizer settings.
``cache:md5-v2`` recognises today's file names, so no existing cache is invalidated.
"""
import re

from cevahir_ai.keys.canonical import as_bool, as_float, as_int, as_str, structured_digest
from cevahir_ai.keys.grammar import Key

RECORD_FORMAT = 2                 # training_system.cache_identity.CACHE_FORMAT_VERSION (versions table)
BUILDER = "prepare_cache/2"
CACHE_NAME_RE = re.compile(r"^cached_data_([0-9a-f]{32})_([0-9a-f]{64}|)\.pkl$")
SPLIT_METHODS = ("source-aware/1", "random/1")


def cache_recipe_key(corpus, tok, *, encode_mode, include_whole_words, include_syllables, include_sep,
                     max_seq_length, alignment_format="autoregressive_v2", formatted=True,
                     record_format=RECORD_FORMAT, builder=BUILDER):
    """``corpus`` and ``tok`` are keys (``Key`` or string). Returns ``(key, payload)``."""
    corpus_key, tok_key = Key.parse(str(corpus)), Key.parse(str(tok))
    if corpus_key.kind != "corpus" or tok_key.kind != "tok":
        raise ValueError("cache recipe needs a corpus key and a tokenizer key")
    payload = {
        "record_format": as_int(record_format, "record_format"),
        "corpus": str(corpus_key), "tok": str(tok_key),
        "encoding": {
            "encode_mode": as_str(encode_mode, "encode_mode"),
            "include_whole_words": as_bool(include_whole_words, "include_whole_words"),
            "include_syllables": as_bool(include_syllables, "include_syllables"),
            "include_sep": as_bool(include_sep, "include_sep"),
            "max_seq_length": as_int(max_seq_length, "max_seq_length"),
            "alignment_format": as_str(alignment_format, "alignment_format"),
            "formatted": as_bool(formatted, "formatted"),
        },
        "builder": as_str(builder, "builder"),
    }
    return Key("cache", "recipe-v1", structured_digest("cache", "recipe-v1", payload)), payload


def split_assign_key(cache, *, ratio, seed, method="source-aware/1"):
    cache_key = Key.parse(str(cache))
    if cache_key.kind != "cache":
        raise ValueError("split needs a cache key")
    if method not in SPLIT_METHODS:
        raise ValueError(f"unknown split method: {method}")
    payload = {"cache": str(cache_key), "ratio": as_float(ratio, "ratio"), "seed": as_int(seed, "seed"),
               "method": method}
    return Key("split", "assign-v1", structured_digest("split", "assign-v1", payload)), payload


def parse_cache_name(name):
    """``cached_data_<md5>_<datahash>.pkl`` -> (cache:md5-v2 key, corpus:legacy-dd-v1 key or None)."""
    match = CACHE_NAME_RE.match(str(name).replace("\\", "/").rsplit("/", 1)[-1])
    if not match:
        raise ValueError(f"not a training-cache file name: {name}")
    md5, data_hash = match.groups()
    return Key("cache", "md5-v2", md5), (Key("corpus", "legacy-dd-v1", data_hash) if data_hash else None)


def cache_md5_key(name_or_md5):
    text = str(name_or_md5)
    if re.fullmatch(r"[0-9a-f]{32}", text):
        return Key("cache", "md5-v2", text)
    return parse_cache_name(text)[0]
