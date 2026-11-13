"""Tokenizer keys.

``tok:bpe-sem-v1`` identifies the legacy BPE tokenizer by what decides how text maps to ids: the
token->id map, the ordered merges, the specials and the id-semantic config keys. It ignores what the
legacy ``tokenizer_digest`` (identity_version 3) wrongly included: the ``read_only`` flag, CRLF vs
LF bytes of ``merges.txt``, per-token frequency statistics, decode-only and diagnostic settings.

``tok:bytebpe-v1`` is ``ByteLevelBPE.identity()`` verbatim.
"""
import hashlib
import json
from pathlib import Path

from cevahir_ai.keys.canonical import structured_digest
from cevahir_ai.keys.classify import tokenizer_semantic_config
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.text import text_lf

BPE_FAMILY, BPE_PIPELINE = "cevahir-bpe", "cevahir-bpe-pipeline/1"
BPE_SPECIALS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
BYTE_BPE_FORMAT, BYTE_BPE_VERSION = "cevahir-byte-bpe", 1


def read_merges(path):
    """Merges as ordered pairs. Same rule as ``BPEManager._read_merges_file`` (two whitespace-separated
    fields per line, other lines skipped), on TEXT-LF-1 text, so CRLF, LF and a BOM give one result."""
    pairs = []
    for line in text_lf(Path(path).read_bytes()).decode("utf-8").split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            pairs.append([parts[0], parts[1]])
    return pairs


def read_vocab_ids(path_or_vocab):
    """``{token: id}`` from ``vocab.json`` or a ``get_vocab()`` mapping; frequency statistics are dropped."""
    vocab = path_or_vocab
    if not isinstance(vocab, dict):
        vocab = json.loads(text_lf(Path(path_or_vocab).read_bytes()).decode("utf-8"))
    ids = {}
    for token, entry in vocab.items():
        value = entry.get("id") if isinstance(entry, dict) else entry
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"vocabulary entry {token!r} has no integer id")
        ids[token] = value
    return ids


def bpe_sem_payload(vocab, merges, config=None):
    ids = read_vocab_ids(vocab)
    return {
        "family": BPE_FAMILY, "pipeline": BPE_PIPELINE,
        "vocab": ids,
        "merges": [list(pair) for pair in merges],
        "specials": {token: ids[token] for token in BPE_SPECIALS if token in ids},
        "id_config": _plain(tokenizer_semantic_config(config)),
    }


def tok_bpe_sem_key(vocab, merges, config=None):
    """``vocab``: path to vocab.json or a vocab mapping; ``merges``: path to merges.txt or a list of pairs.

    Returns ``(key, explain)``. ``config`` is the tokenizer manager's config; only its id-semantic keys
    enter the key. The same files and semantic config give the same key on every platform.
    """
    pairs = read_merges(merges) if isinstance(merges, (str, Path)) else merges
    payload = bpe_sem_payload(vocab, pairs, config)
    explain = {"family": BPE_FAMILY, "pipeline": BPE_PIPELINE, "vocab_size": len(payload["vocab"]),
               "merges": len(payload["merges"]), "specials": payload["specials"], "id_config": payload["id_config"]}
    return Key("tok", "bpe-sem-v1", structured_digest("tok", "bpe-sem-v1", payload)), explain


def bytebpe_identity(pattern, specials, merges):
    """Byte-for-byte the computation of ``ByteLevelBPE.identity()`` (tokenizer_management/byte_bpe.py)."""
    payload = json.dumps({"format": BYTE_BPE_FORMAT, "version": BYTE_BPE_VERSION, "pattern": pattern,
                          "special": list(specials), "merges": [list(pair) for pair in merges]},
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tok_bytebpe_key(source):
    """Key of a byte-level BPE: a ``ByteLevelBPE`` object, a 64-hex identity, or a saved tokenizer file.

    For a file the identity is recomputed from the stored pattern, specials and merges (torch-free),
    and a stored ``identity`` field that disagrees raises: the file was modified.
    """
    if hasattr(source, "identity") and callable(source.identity):
        return Key("tok", "bytebpe-v1", source.identity())
    if isinstance(source, str) and len(source) == 64 and not Path(source).exists():
        return Key("tok", "bytebpe-v1", source)
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if data.get("format") != BYTE_BPE_FORMAT or data.get("version") != BYTE_BPE_VERSION:
        raise ValueError(f"not a {BYTE_BPE_FORMAT} v{BYTE_BPE_VERSION} file: {source}")
    identity = bytebpe_identity(data["pattern"], data["special_tokens"], data["merges"])
    if data.get("identity") and data["identity"] != identity:
        raise ValueError(f"tokenizer identity mismatch (file was modified): {source}")
    return Key("tok", "bytebpe-v1", identity)


def _plain(value):
    """Config values as canonical data (tuples and sets become lists)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    return value
