"""Legacy cache and corpus identities, recognised as keys (nothing is renamed or rebuilt).

``cached_data_<md5>_<datahash>.pkl`` becomes ``cache:md5-v2`` + ``corpus:legacy-dd-v1``. The md5 is
re-derived with the original ``DataCacheV3.get_cache_key``; only a match is an established fact.
A stale ``.sha256`` sidecar is reported and never trusted over the file.
"""
import json
from pathlib import Path

from cevahir_ai._compat.repo import import_legacy
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.schemes.cache import parse_cache_name
from cevahir_ai.keys.schemes.file import file_bytes_key, read_sidecar


def corpus_legacy_dd_key(directory):
    """``training_system.cache_identity.data_directory_digest(directory)``, called unchanged.

    The legacy digest of a missing or empty directory is the empty string; that is not a key.
    """
    digest = import_legacy("training_system.cache_identity").data_directory_digest(directory)
    if not digest:
        raise ValueError(f"legacy data-directory digest is empty (missing or empty directory): {directory}")
    return Key("corpus", "legacy-dd-v1", digest)


def legacy_cache_md5(data_dir, *, encode_mode, include_whole_words, include_syllables, include_sep,
                     max_seq_length, tokenizer_digest, alignment_format="autoregressive_v2", formatted=True):
    """The md5 that names a legacy cache file, computed by the original ``DataCacheV3.get_cache_key``.

    Like the original, an absolute ``data_dir`` is made relative to the current working directory.
    """
    cache_class = import_legacy("training_system.v3.data.cache_v3").DataCacheV3
    stand_in = object.__new__(cache_class)          # no constructor: it would create the cache directory
    stand_in.data_dir = Path(data_dir)
    return Key("cache", "md5-v2", stand_in.get_cache_key(
        encode_mode, include_whole_words, include_syllables, include_sep, max_seq_length,
        tokenizer_digest, alignment_format=alignment_format, formatted=formatted))


def recognise_cache(path, tokenizer_digest=None):
    """Facts about an existing cache file, read without unpickling it.

    Returns a dict with the legacy keys parsed from the name, the file key, the sidecar state
    (``ok`` | ``stale`` | ``missing``), the ``.meta.json`` content and, when ``tokenizer_digest`` is
    given and the metadata is present, whether the original key function reproduces the name.
    """
    path = Path(path)
    md5_key, corpus_key = parse_cache_name(path.name)
    file_key = file_bytes_key(path)
    sidecar_path = path.with_suffix(".sha256")       # DataCacheV3 naming: <stem>.sha256, not <name>.sha256
    sidecar = (sidecar_path.read_text(encoding="utf-8", errors="replace").split() or [""])[0].lower() \
        if sidecar_path.is_file() else read_sidecar(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    facts = {"cache": str(md5_key), "corpus": None if corpus_key is None else str(corpus_key),
             "file": str(file_key), "meta": meta,
             "sidecar": "missing" if sidecar is None else ("ok" if sidecar == file_key.digest else "stale"),
             "rederived": None}
    if meta and tokenizer_digest:
        try:
            again = legacy_cache_md5(meta["data_dir"], encode_mode=meta["encode_mode"],
                                     include_whole_words=meta["include_whole_words"],
                                     include_syllables=meta["include_syllables"], include_sep=meta["include_sep"],
                                     max_seq_length=meta["max_seq_length"], tokenizer_digest=tokenizer_digest,
                                     alignment_format=meta.get("alignment_format", "autoregressive_v2"))
            facts["rederived"] = again == md5_key
        except KeyError:
            facts["rederived"] = None
    return facts
