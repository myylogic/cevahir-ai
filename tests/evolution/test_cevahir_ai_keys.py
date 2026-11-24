"""Central key system, torch-free part: grammar, canonical encodings, schemes, aliases, the gate.

Every key is ``cev:<kind>:<scheme>:<digest>``. The tests pin the encodings with golden vectors,
show that keys do not depend on CRLF/LF, the working directory or runtime settings, and that the
existing identities (book fingerprints, ByteLevelBPE.identity, cache file names) map to keys
without being recomputed.
"""
import hashlib
import importlib
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path

import pytest

from cevahir_ai.domain.verdicts import CHECKS, IMPLEMENTED_CHECKS, PURPOSES, CheckResult, GatePolicy
from cevahir_ai.errors import CanonicalError, KeyFormatError, UnknownSchemeError
from cevahir_ai.keys import Key, cjson, check_compatibility, structured_digest, text_lf
from cevahir_ai.keys.gate import DEFAULT_GATE_POLICY
from cevahir_ai.keys.registry import SCHEMES, derive
from cevahir_ai.keys.schemes import arch, cache, ckpt, code, corpus, file as file_scheme, runs, tok

ROOT = Path(__file__).resolve().parents[2]
H = "ab" * 32


# ----------------------------------------------------------------------------------------- grammar
def test_key_round_trip_display_and_ordering():
    key = Key.parse(f"cev:tok:bpe-sem-v1:{H}")
    assert (key.kind, key.scheme, key.digest) == ("tok", "bpe-sem-v1", H)
    assert str(key) == f"cev:tok:bpe-sem-v1:{H}" and Key.parse(str(key)) == key and hash(key) == hash(Key("tok", "bpe-sem-v1", H))
    assert key.display() == "tok:abababababab"
    assert Key("tok", "legacy-v3", H).display() == "tok:legacy-v3:abababababab"       # legacy shows its scheme
    assert Key("cache", "md5-v2", "0" * 32).digest == "0" * 32
    assert len({Key("tok", "legacy-v3", H), Key("tok", "bytebpe-v1", H)}) == 2          # same hex, different keys


@pytest.mark.parametrize("text, error", [
    ("cev:tok:bpe-sem-v1:" + "AB" * 32, KeyFormatError),            # upper-case hex
    ("cev:tok:bpe-sem-v1:" + "ab" * 31, KeyFormatError),            # wrong length
    ("tok:bpe-sem-v1:" + H, KeyFormatError),                        # no prefix
    ("cev:tok:bpe-sem:" + H, KeyFormatError),                       # scheme without version
    ("cev:Tok:bpe-sem-v1:" + H, KeyFormatError),
    ("cev:nokind:x-v1:" + H, UnknownSchemeError),                   # unknown scheme is never a silent pass
    ("cev:tok:bpe-sem-v9:" + H, UnknownSchemeError),
    ("cev:cache:md5-v2:" + H, KeyFormatError),                      # md5 scheme has 32 digits
    ("cev:tok:bpe-sem-v1:" + "0" * 32, KeyFormatError),
    (H, KeyFormatError),                                            # a bare digest is not a key
])
def test_malformed_and_unknown_keys_are_rejected(text, error):
    with pytest.raises(error):
        Key.parse(text)


def test_key_error_reports_position():
    with pytest.raises(KeyFormatError) as info:
        Key.parse("cev:tok:bpe-sem-v1:zz")
    assert info.value.position == len("cev:tok:bpe-sem-v1:")


def test_registry_is_complete_and_every_derivation_resolves():
    kinds = {scheme.kind for scheme in SCHEMES.values()}
    assert {"file", "corpus", "tok", "cache", "split", "arch", "wschema", "weights", "optim", "sched", "ckpt",
            "model", "runspec", "run", "eval", "evalset", "code", "env"} <= kinds
    legacy = {scheme.name for scheme in SCHEMES.values() if scheme.status == "legacy"}
    assert legacy == {"tok:legacy-v3", "cache:md5-v2", "corpus:legacy-dd-v1", "arch:ctor-v0", "weights:cev11-fp32-v1"}
    for scheme in SCHEMES.values():
        assert scheme.status in ("primary", "legacy", "signature", "occurrence")
        assert scheme.digest_len == (32 if scheme.name in ("cache:md5-v2", "run:id-v1") else 64)
        module_name, _, function = scheme.derive.partition(":")
        if scheme.torch and "torchkeys" in module_name:
            pytest.importorskip("torch")
        assert callable(getattr(importlib.import_module(module_name), function)), scheme.derive


# --------------------------------------------------------------------------------------- canonical
def test_cjson_golden_vectors():
    data = cjson({"b": 1, "a": [1.0, "ç", None, True], "ı": -0.0, "z": 1e-07})
    assert data.decode("utf-8") == '{"a":[1.0,"ç",null,true],"b":1,"z":1e-07,"ı":0.0}'
    assert hashlib.sha256(data).hexdigest() == "b1fdd533f5e408efe9c003c51840d829a5e713bdb876881bfe102e500825a5a6"
    assert structured_digest("arch", "cfg-v1", {"family": "cevahir-decoder", "fields": {"embed_dim": 32}}) == \
        "d059bbbbe2241f0556aef5b3be080f8d4ef1a2b60b7392de77bb016acda9dc54"


def test_cjson_is_order_free_strict_and_does_not_normalise_strings():
    assert cjson({"a": 1, "b": (1, 2)}) == cjson({"b": [1, 2], "a": 1})
    assert cjson(1) != cjson(1.0) != cjson(True)
    composed, decomposed = "ç", "ç"                    # one token is not the other
    assert unicodedata.normalize("NFC", decomposed) == composed and cjson(composed) != cjson(decomposed)
    for bad in (float("nan"), float("inf"), 2**63, {1: 2}, {"a": object()}, b"x", {"s": {1, 2}}, "\ud800"):
        with pytest.raises(CanonicalError):
            cjson(bad)
    nested = []
    for _ in range(70):
        nested = [nested]
    with pytest.raises(CanonicalError):
        cjson(nested)


def test_domain_separation_makes_schemes_incomparable():
    payload = {"x": 1}
    digests = {structured_digest(kind, scheme, payload) for kind, scheme in (("tok", "bpe-sem-v1"), ("arch", "cfg-v1"),
                                                                             ("arch", "ctor-v0"), ("ckpt", "members-v1"))}
    assert len(digests) == 4
    assert structured_digest("arch", "cfg-v1", payload) == hashlib.sha256(b"cev:arch:cfg-v1\x00" + b'{"x":1}').hexdigest()


def test_typed_projection_gives_one_key_for_one_meaning():
    base = _normalized()
    variants = [{**base, "embed_dim": 32.0}, {**base, "attn_logit_cap": 0}, {**base, "use_moe": 0}]
    assert {arch.arch_cfg_key(variant)[0] for variant in variants} == {arch.arch_cfg_key(base)[0]}
    with pytest.raises(CanonicalError):
        arch.arch_cfg_key({**base, "num_heads": "4"})


def test_text_lf_golden_and_newline_invariance():
    out = text_lf(bytes.fromhex("EFBBBF610D0A620D630A"))            # BOM a CRLF b CR c LF
    assert out == b"a\nb\nc\n"
    assert hashlib.sha256(out).hexdigest() == "880553fca8fcea94e325ee2cfb48e5a985cc797f39a14cc6d3cedecfeb2ae4d2"
    with pytest.raises(UnicodeDecodeError):
        text_lf(b"\xff\xfe")


# --------------------------------------------------------------------------------------- file keys
def test_file_keys_equal_sha256sum_and_ignore_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes("ağaç\nkuş\n".encode("utf-8"))
    crlf.write_bytes("ağaç\r\nkuş\r\n".encode("utf-8"))
    assert file_scheme.file_key(lf) == file_scheme.file_key(crlf)                       # utf8lf-v1 for text suffixes
    assert file_scheme.file_bytes_key(lf) != file_scheme.file_bytes_key(crlf)
    assert file_scheme.file_bytes_key(lf).digest == hashlib.sha256(lf.read_bytes()).hexdigest()
    binary = tmp_path / "w.pth"
    binary.write_bytes(b"\xff\x00\r\n")
    assert file_scheme.file_key(binary).scheme == "bytes-v1"
    not_utf8 = tmp_path / "latin.txt"
    not_utf8.write_bytes(b"\xe7\xfd")
    assert file_scheme.file_key(not_utf8).scheme == "bytes-v1"                        # falls back, never guesses
    assert derive("file", "utf8lf-v1", lf) == file_scheme.file_utf8lf_key(lf)


def test_file_utf8lf_key_equals_the_book_fingerprint_of_every_reviewed_source():
    """Existing identity, no invalidation: the book's utf8-lf fingerprints are file:utf8lf-v1 keys."""
    snapshot = json.loads((ROOT / "docs/book/evidence/reviewed_sources.json").read_text(encoding="utf-8"))
    entries = snapshot.get("files", snapshot)
    checked = 0
    for path, entry in entries.items():
        if not isinstance(entry, dict) or "sha256" not in entry or not (ROOT / path).is_file():
            continue
        scheme = "utf8lf-v1" if entry.get("fingerprint_mode") == "utf8-lf" else "bytes-v1"
        assert file_scheme.file_key(ROOT / path, scheme).digest == entry["sha256"], path
        checked += 1
    assert checked == len(entries) >= 100   # every reviewed source (104 since v7: the Turkish chapters were
                                            # withdrawn by their author and link no sources)
    merges = file_scheme.file_key(ROOT / "data/merges_lib/merges.txt")
    assert str(merges) == "cev:file:utf8lf-v1:b5b0b9fcc04a73ea79ed1d3cee5f4f96860204cad6b020364ec9f20af4b080b1"


def test_sidecar_reader(tmp_path):
    path = tmp_path / "c.pth"
    path.write_bytes(b"abc")
    assert file_scheme.read_sidecar(path) is None
    Path(str(path) + ".sha256").write_text(hashlib.sha256(b"abc").hexdigest().upper() + "  c.pth\n")
    assert file_scheme.read_sidecar(path) == file_scheme.file_bytes_key(path).digest


# ------------------------------------------------------------------------------------------ corpus
def _write_corpus(root, newline="\n", nfd=False):
    name = "şiir.txt"
    name = unicodedata.normalize("NFD", name) if nfd else name
    (root / "alt").mkdir(parents=True)
    (root / name).write_bytes(f"gül{newline}lale{newline}".encode("utf-8"))
    (root / "alt" / "veri.json").write_bytes(f'{{"a": 1}}{newline}'.encode("utf-8"))
    (root / "alt" / "skip.md").write_bytes(b"not part of the corpus")
    (root / "belge.docx").write_bytes(b"PK\x03\x04binary\r\n")


def test_corpus_key_ignores_crlf_cwd_location_and_nfd_names(tmp_path, monkeypatch):
    a, b = tmp_path / "a" / "education", tmp_path / "elsewhere" / "deep" / "education"
    _write_corpus(a)
    _write_corpus(b, newline="\r\n", nfd=True)
    key_a, payload = corpus.corpus_files_key(a)
    assert [entry[0] for entry in payload["files"]] == ["alt/veri.json", "belge.docx", "şiir.txt"]
    assert payload["files"][1][1].startswith("cev:file:bytes-v1:")                    # .docx stays binary
    monkeypatch.chdir(tmp_path)
    assert corpus.corpus_files_key(b)[0] == key_a == corpus.corpus_files_key(Path("a/education"))[0]
    (a / "şiir.txt").write_bytes("gül\nlale\nmenekşe\n".encode("utf-8"))
    assert corpus.corpus_files_key(a)[0] != key_a
    assert corpus.evalset_files_key(b)[0].digest != key_a.digest                      # same files, other kind


def test_corpus_key_refuses_missing_and_empty_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpus.corpus_files_key(tmp_path / "none")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        corpus.corpus_files_key(tmp_path / "empty")


def test_legacy_corpus_digest_is_wrapped_not_changed(tmp_path):
    from cevahir_ai.adapters.caches import corpus_legacy_dd_key
    from training_system.cache_identity import data_directory_digest
    _write_corpus(tmp_path / "c")
    assert corpus_legacy_dd_key(tmp_path / "c").digest == data_directory_digest(tmp_path / "c")
    with pytest.raises(ValueError):
        corpus_legacy_dd_key(tmp_path / "missing")                                    # legacy "" is not a key


# --------------------------------------------------------------------------------------- tokenizer
VOCAB = {"<PAD>": {"id": 0, "total_freq": 0, "positions": []}, "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
         "<BOS>": {"id": 2, "total_freq": 0, "positions": []}, "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
         "a": {"id": 4, "total_freq": 7, "positions": [1]}, "b</w>": {"id": 5, "total_freq": 3, "positions": []},
         "ab</w>": {"id": 6, "total_freq": 2, "positions": []}}


def _tokenizer_files(directory, newline="\n", bom=False, vocab=VOCAB):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_bytes(json.dumps(vocab, ensure_ascii=False, indent=2).replace("\n", newline).encode("utf-8"))
    merges = f"a b</w>{newline}ç ı{newline}{newline}not a pair line{newline}"
    (directory / "merges.txt").write_bytes((b"\xef\xbb\xbf" if bom else b"") + merges.encode("utf-8"))
    return directory / "vocab.json", directory / "merges.txt"


def test_bpe_sem_key_depends_only_on_id_semantics(tmp_path):
    vocab, merges = _tokenizer_files(tmp_path / "lf")
    config = {"include_whole_words": True, "lowercase": False, "read_only": True, "batch_size": 32, "use_gpu": False,
              "capitalize_sentence": True, "max_unk_ratio": 0.01, "max_iter": 60000}
    key, explain = tok.tok_bpe_sem_key(vocab, merges, config)
    assert explain["merges"] == 2 and explain["specials"] == {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    assert explain["id_config"] == {"include_whole_words": True, "lowercase": False}
    # Not identity: read_only, runtime, decode, diagnostic and training settings; CRLF; frequency statistics.
    other = {**config, "read_only": False, "batch_size": 1, "use_gpu": True, "capitalize_sentence": False,
             "max_unk_ratio": 0.5, "max_iter": 1}
    crlf = _tokenizer_files(tmp_path / "crlf", newline="\r\n")
    refreshed = {token: {**entry, "total_freq": entry["total_freq"] + 100} for token, entry in VOCAB.items()}
    stats = _tokenizer_files(tmp_path / "stats", vocab=refreshed)
    assert {tok.tok_bpe_sem_key(vocab, merges, other)[0], tok.tok_bpe_sem_key(*crlf, config)[0],
            tok.tok_bpe_sem_key(*stats, config)[0]} == {key}
    # Identity: an id, a merge, merge order, a semantic setting, an unclassified setting.
    moved = {**VOCAB, "a": {**VOCAB["a"], "id": 5}, "b</w>": {**VOCAB["b</w>"], "id": 4}}
    changed = [tok.tok_bpe_sem_key(moved, merges, config)[0],
               tok.tok_bpe_sem_key(vocab, [["a", "b</w>"]], config)[0],
               tok.tok_bpe_sem_key(vocab, [["ç", "ı"], ["a", "b</w>"]], config)[0],
               tok.tok_bpe_sem_key(vocab, merges, {**config, "lowercase": True})[0],
               tok.tok_bpe_sem_key(vocab, merges, {**config, "brand_new_setting": 1})[0]]
    assert key not in changed and len(set(changed)) == 5
    # A byte-order mark in merges.txt is identity too. The legacy reader keeps it inside the first
    # merge, which then never applies: the tokenizer behaves differently, so it has another key.
    marked = _tokenizer_files(tmp_path / "bom", bom=True)
    assert tok.read_merges(marked[1])[0] == ["\ufeffa", "b</w>"] and tok.read_merges(merges)[0] == ["a", "b</w>"]
    assert tok.tok_bpe_sem_key(*marked, config)[0] != key
    # In vocab.json a BOM makes the file unreadable for the legacy reader; it is no key here either.
    (tmp_path / "bom" / "vocab.json").write_bytes(b"\xef\xbb\xbf" + vocab.read_bytes())
    with pytest.raises(ValueError, match="BOM"):
        tok.tok_bpe_sem_key(*marked, config)


def test_merges_parser_matches_the_legacy_reader_on_lf_crlf_and_bom(tmp_path):
    pytest.importorskip("torch")
    from tokenizer_management.bpe.bpe_manager import BPEManager
    from tokenizer_management.bpe.bpe_manager_utils import read_json
    for name, newline, bom in (("lf", "\n", False), ("crlf", "\r\n", False), ("bom", "\n", True), ("bom-crlf", "\r\n", True)):
        vocab, merges = _tokenizer_files(tmp_path / name, newline=newline, bom=bom)
        stand_in = object.__new__(BPEManager)
        stand_in.merges_file = str(merges)
        legacy = [list(pair) for pair in BPEManager._read_merges_file(stand_in)]
        assert tok.read_merges(merges) == legacy and len(legacy) == 2
        assert legacy[0][0].startswith("\ufeff") == bom
        assert tok.read_vocab_ids(vocab) == {token: entry["id"] for token, entry in read_json(str(vocab)).items()}
    (tmp_path / "bom" / "vocab.json").write_bytes(b"\xef\xbb\xbf" + (tmp_path / "lf" / "vocab.json").read_bytes())
    for reader in (lambda path: read_json(str(path)), tok.read_vocab_ids):         # both refuse a BOM in the vocabulary
        with pytest.raises(ValueError, match="BOM"):
            reader(tmp_path / "bom" / "vocab.json")
    tracked = ROOT / "data/merges_lib/merges.txt"
    stand_in.merges_file = str(tracked)
    assert tok.read_merges(tracked) == [list(pair) for pair in BPEManager._read_merges_file(stand_in)]


def test_tokenizer_key_classes_cover_the_legacy_runtime_denylist():
    from cevahir_ai.keys.classify import TOKENIZER_KEY_CLASS_V1, tokenizer_semantic_config
    from training_system.cache_identity import RUNTIME_ONLY_TOKENIZER_KEYS
    runtime = {name for name, kind in TOKENIZER_KEY_CLASS_V1.items() if kind == "runtime"}
    assert runtime == set(RUNTIME_ONLY_TOKENIZER_KEYS)
    assert TOKENIZER_KEY_CLASS_V1["read_only"] == "mode"
    assert tokenizer_semantic_config({"read_only": True, "unknown_new_key": 1, "lowercase": True}) == \
        {"unknown_new_key": 1, "lowercase": True}                                     # unclassified counts as semantic


def test_bytebpe_key_is_the_existing_identity_verbatim(tmp_path):
    from tokenizer_management.byte_bpe import PRETOKENIZE_PATTERN, SPECIAL_TOKENS, ByteLevelBPE
    tokenizer = ByteLevelBPE.train(["merhaba dünya merhaba", "dünya güzel"] * 20, vocab_size=280)
    path = tmp_path / "t.tokenizer.json"
    tokenizer.save(path)
    assert tok.tok_bytebpe_key(tokenizer) == tok.tok_bytebpe_key(path) == Key("tok", "bytebpe-v1", tokenizer.identity())
    assert tok.bytebpe_identity(PRETOKENIZE_PATTERN, SPECIAL_TOKENS, tokenizer.merges) == tokenizer.identity()
    # Same vocabulary, different merges: the legacy vocab-only digest collides, the keys do not.
    data = json.loads(path.read_text(encoding="utf-8"))
    data["merges"] = data["merges"][:-1]
    data.pop("identity")
    shorter = tmp_path / "shorter.json"
    shorter.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert tok.tok_bytebpe_key(shorter) != tok.tok_bytebpe_key(path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["merges"] = tampered["merges"][:-1]
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        tok.tok_bytebpe_key(bad)


# ---------------------------------------------------------------------------------- cache / split
def test_cache_recipe_and_split_keys(tmp_path):
    corpus_key, tok_key = Key("corpus", "files-v1", H), Key("tok", "bpe-sem-v1", H)
    options = dict(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=False,
                   max_seq_length=512)
    key, payload = cache.cache_recipe_key(corpus_key, tok_key, **options)
    assert payload["record_format"] == 2 and payload["builder"] == "prepare_cache/2"
    assert cache.cache_recipe_key(str(corpus_key), str(tok_key), **{**options, "max_seq_length": 512.0})[0] == key
    for change in ({"max_seq_length": 256}, {"include_sep": True}, {"encode_mode": "inference"}, {"record_format": 3}):
        assert cache.cache_recipe_key(corpus_key, tok_key, **{**options, **change})[0] != key
    assert cache.cache_recipe_key(corpus_key, Key("tok", "bytebpe-v1", H), **options)[0] != key
    with pytest.raises(ValueError):
        cache.cache_recipe_key(tok_key, corpus_key, **options)
    split, _ = cache.split_assign_key(key, ratio=0.1, seed=42)
    assert cache.split_assign_key(key, ratio=0.1, seed=43)[0] != split != cache.split_assign_key(key, ratio=0.2, seed=42)[0]
    with pytest.raises(ValueError):
        cache.split_assign_key(key, ratio=0.1, seed=1, method="unknown")


def test_legacy_cache_names_are_recognised_as_keys():
    md5, data_hash = "0123456789abcdef0123456789abcdef", "cd" * 32
    key, corpus_key = cache.parse_cache_name(f"/any/dir/cached_data_{md5}_{data_hash}.pkl")
    assert key == Key("cache", "md5-v2", md5) and corpus_key == Key("corpus", "legacy-dd-v1", data_hash)
    assert cache.parse_cache_name(f"cached_data_{md5}_.pkl") == (key, None)            # legacy empty data hash
    assert cache.cache_md5_key(md5) == key
    with pytest.raises(ValueError):
        cache.parse_cache_name("bpe_corpus_x.pkl")


def test_legacy_cache_md5_is_reproduced_by_the_original_functions(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from cevahir_ai.adapters.caches import legacy_cache_md5, recognise_cache
    from training_system.data_cache import DataCache
    from training_system.v3.data.cache_v3 import DataCacheV3
    options = dict(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=False,
                   max_seq_length=128)
    monkeypatch.chdir(tmp_path)
    for data_dir in ("education", str(tmp_path / "education")):                       # relative and absolute
        v3 = DataCacheV3(data_dir, cache_dir=str(tmp_path / "c3"))
        v2 = DataCache(data_dir, cache_dir=str(tmp_path / "c2"))
        expected = v3.get_cache_key(vocab_hash=H, **options)
        assert v2._get_cache_key(vocab_hash=H, **options) == expected
        assert legacy_cache_md5(data_dir, tokenizer_digest=H, **options).digest == expected
    # An existing cache file with V3 sidecars: recognised, re-derived, stale sidecar reported.
    path = tmp_path / "c3" / f"cached_data_{expected}_{'cd' * 32}.pkl"
    path.write_bytes(b"payload")
    path.with_suffix(".sha256").write_text(hashlib.sha256(b"payload").hexdigest())
    path.with_suffix(".meta.json").write_text(json.dumps({"data_dir": "education", "alignment_format": "autoregressive_v2", **options}))
    facts = recognise_cache(path, tokenizer_digest=H)
    assert facts["cache"] == f"cev:cache:md5-v2:{expected}" and facts["sidecar"] == "ok" and facts["rederived"] is True
    assert recognise_cache(path, tokenizer_digest="ef" * 32)["rederived"] is False     # another tokenizer: no match
    path.write_bytes(b"rebuilt payload")
    assert recognise_cache(path)["sidecar"] == "stale"


# ------------------------------------------------------------------------------------ architecture
def _normalized(**overrides):
    from cevahir_ai.adapters.model_build import config_schema
    return config_schema().tiny_model_config(**overrides)


def test_arch_classification_covers_every_config_field():
    from dataclasses import fields
    from cevahir_ai.adapters.model_build import config_schema
    from cevahir_ai.keys.classify import ARCH_EXTRA_V1, ARCH_FDF_V1, ARCH_FIELD_CLASS_V1, ARCH_SERVE_V1, ARCH_TRAIN_V1
    names = {entry.name for entry in fields(config_schema().ModelArchConfig)}
    # 24 dataclass fields define the function; the 25th, pe_max_len, is the normaliser's copy of
    # max_seq_length under the name the constructor takes (the positional table length).
    assert len(names) == 40 and names == (set(ARCH_FDF_V1) - {"pe_max_len"}) | set(ARCH_TRAIN_V1) | set(ARCH_SERVE_V1)
    assert (len(ARCH_FDF_V1), len(ARCH_TRAIN_V1), len(ARCH_SERVE_V1), len(ARCH_EXTRA_V1)) == (25, 8, 8, 6)
    assert set(_normalized()) == names | set(ARCH_EXTRA_V1) | {"pe_max_len"} == set(ARCH_FIELD_CLASS_V1)
    assert len(ARCH_FIELD_CLASS_V1) == 47 and "max_seq_length" in ARCH_SERVE_V1


FDF_CHANGES = {
    "vocab_size": 256, "embed_dim": 64, "seq_proj_dim": 64, "num_heads": 8, "num_kv_heads": 2, "num_layers": 3,
    "ffn_dim": 128, "pre_norm": False, "use_rmsnorm": False, "use_swiglu": False, "use_qk_norm": True,
    "parallel_residual": True, "causal_mask": False, "pe_mode": "sinusoidal", "rope_scaling_type": "yarn",
    "sliding_window": 16, "attn_logit_cap": 50.0, "tie_weights": False, "logit_soft_cap": 0.0, "use_moe": True,
}
NON_FDF_CHANGES = {
    "dropout": 0.3, "drop_path_rate": 0.1, "pe_dropout": 0.2, "use_gradient_checkpointing": True,
    "use_advanced_checkpointing": True, "moe_jitter_noise": 0.5, "moe_load_balance_alpha": 0.5,
    "quantization_type": "int8", "max_seq_length": 128, "use_flash_attention": True, "use_pytorch_sdpa": False,
    "use_kv_cache": False, "max_cache_len": 32, "kv_num_sink_tokens": 2, "device": "cuda", "log_level": 10,
    "learning_rate": 1.0, "precision": "fp16",
}


def test_every_function_defining_field_changes_the_arch_key_and_nothing_else_does():
    base = arch.arch_cfg_key(_normalized())[0]
    seen = {base}
    for name, value in FDF_CHANGES.items():
        overrides = {name: value, "embed_dim": 64} if name == "seq_proj_dim" else {name: value}
        overrides = {**overrides, "seq_proj_dim": 64} if name == "embed_dim" else overrides
        key = arch.arch_cfg_key(_normalized(**overrides))[0]
        assert key not in seen or name in ("embed_dim", "seq_proj_dim"), f"{name} did not change the key"
        seen.add(key)
    for name, value in NON_FDF_CHANGES.items():
        assert arch.arch_cfg_key(_normalized(**{name: value}))[0] == base, f"{name} changed the key"
    # Fields of inactive features do not split keys ...
    assert arch.arch_cfg_key(_normalized(num_experts=16, moe_top_k=4))[0] == base
    assert arch.arch_cfg_key(_normalized(rope_scaling_factor=4.0, rope_original_max_len=512))[0] == base
    assert arch.arch_cfg_key(_normalized(num_kv_heads=4))[0] == base                    # None resolves to num_heads
    assert arch.arch_cfg_key(_normalized(ffn_dim=64))[0] == base
    # ... and do once the feature is on.
    moe = arch.arch_cfg_key(_normalized(use_moe=True))[0]
    assert arch.arch_cfg_key(_normalized(use_moe=True, num_experts=16))[0] != moe
    yarn = arch.arch_cfg_key(_normalized(rope_scaling_type="yarn", rope_scaling_factor=2.0))[0]
    assert arch.arch_cfg_key(_normalized(rope_scaling_type="yarn", rope_scaling_factor=4.0))[0] != yarn


def test_context_length_is_architecture_exactly_when_the_positional_table_is_a_weight():
    """max_seq_length sizes pos_encoding.pe_embed.weight under pe_mode="learned"; nowhere else is it a weight."""
    for mode in ("rope", "sinusoidal"):
        short, long = (arch.arch_cfg_key(_normalized(pe_mode=mode, max_seq_length=n))[0] for n in (32, 64))
        assert short == long, mode
        assert arch.function_defining_fields(_normalized(pe_mode=mode, max_seq_length=32))["pe_max_len"] is None
    short, short_fields = arch.arch_cfg_key(_normalized(pe_mode="learned", max_seq_length=32))
    long, long_fields = arch.arch_cfg_key(_normalized(pe_mode="learned", max_seq_length=64))
    assert short != long and (short_fields["pe_max_len"], long_fields["pe_max_len"]) == (32, 64)
    assert arch.diff_fields(short_fields, long_fields) == {"pe_max_len": (32, 64)}
    # The constructor's own name for it gives the same key (a stored model_config carries pe_max_len only).
    stored = {name: value for name, value in _normalized(pe_mode="learned", max_seq_length=32).items() if name != "max_seq_length"}
    from cevahir_ai.adapters.model_build import arch_of_config
    assert arch_of_config(stored)[0] == short
    assert arch.inactive_fields(short_fields) == {"rope_scaling_type", "rope_scaling_factor", "rope_original_max_len",
                                                  "num_experts", "moe_top_k"}
    with pytest.raises(ValueError, match="positional table"):
        arch.function_defining_fields({**_normalized(pe_mode="learned"), "pe_max_len": None, "max_seq_length": None})


def test_learned_positional_table_has_the_declared_length():
    """The claim behind the key field, read from the constructor: the table is [pe_max_len, embed_dim]."""
    pytest.importorskip("torch")
    from cevahir_ai.adapters.model_build import expected_layout
    for length in (32, 64):
        shapes = {name: shape for name, shape, _ in expected_layout(_normalized(pe_mode="learned", max_seq_length=length))["tensors"]}
        assert shapes["pos_encoding.pe_embed.weight"] == [length, 32]
    for mode in ("rope", "sinusoidal"):                                # no persistent positional tensor at all
        layouts = [expected_layout(_normalized(pe_mode=mode, max_seq_length=length)) for length in (32, 64)]
        assert layouts[0]["tensors"] == layouts[1]["tensors"]
        assert not [name for name, _, _ in layouts[0]["tensors"] if name.startswith("pos_encoding.")]


def test_stated_names_follow_the_normaliser():
    """A field counts as stated exactly where the normaliser would take its value from the config."""
    from cevahir_ai.adapters.model_build import config_schema, stated_names, undeclared_fields
    from cevahir_ai.keys.classify import ARCH_FDF_V1
    normalize = config_schema().normalize_model_config
    default = normalize({}, legacy_profile="model_manager")
    probes = {"embed_dim": 96, "num_heads": 2, "num_layers": 3, "ffn_dim": 128, "dropout": 0.25, "max_seq_length": 48}
    spellings = [
        {"d_model": 96, "n_heads": 2, "n_layers": 3, "ff_dim": 128, "drop_rate": 0.25, "pe_max_len": 48},     # old names
        {"arch": {"embed_dim": 96, "num_heads": 2}, "model": {"num_layers": 3, "ffn_dim": 128},
         "training": {"dropout": 0.25}, "extras": {"max_seq_length": 48}},                                    # sections
        dict(probes),                                                                                         # flat
    ]
    for config in spellings:
        normalized, stated = normalize(config, legacy_profile="model_manager"), stated_names(config)
        for name, value in probes.items():
            assert normalized[name] == value != default[name] and name in stated, (name, config)
        assert "pe_max_len" in stated and normalized["pe_max_len"] == 48
        assert not stated & {"vocab_size", "pe_mode", "tie_weights", "use_moe"}
    assert stated_names(None) == set() and stated_names({"arch": "not a mapping"}) == {"arch"}
    # A complete declaration: what tiny_model_config() and a stored model_config carry.
    complete = _normalized()
    fields = arch.function_defining_fields(complete)
    assert undeclared_fields(complete, fields) == [] and undeclared_fields(complete, fields, strict=True) == []
    # Fields of a feature that is off need no statement; every other missing field is named.
    off = {name: value for name, value in complete.items()
           if name not in ("num_experts", "moe_top_k", "rope_scaling_factor", "rope_original_max_len", "pe_max_len", "max_seq_length")}
    assert undeclared_fields(off, fields) == []
    for name in sorted(set(ARCH_FDF_V1) - arch.inactive_fields(fields)):
        partial = {key: value for key, value in complete.items() if key != name}
        assert undeclared_fields(partial, arch.function_defining_fields(normalize(partial)))[0] == f"it does not state {name}", name
    learned = _normalized(pe_mode="learned")
    partial = {key: value for key, value in learned.items() if key not in ("pe_max_len", "max_seq_length")}
    assert undeclared_fields(partial, arch.function_defining_fields(normalize(partial))) == ["it does not state pe_max_len"]
    training = {"batch_size": 8, "epochs": 3}                         # a training config: every architecture field missing
    gaps = undeclared_fields(training, arch.function_defining_fields(normalize(training)), strict=True)
    assert len(gaps) == 1 and "num_heads" in gaps[0] and "vocab_size" in gaps[0] and "batch_size" not in gaps[0]
    foreign = {**complete, "swa_anneal_strategy": "cos"}              # complete, but with a field no model config has
    assert undeclared_fields(foreign, fields) == []
    strict = undeclared_fields(foreign, fields, strict=True)
    assert len(strict) == 1 and "not a model config" in strict[0] and "swa_anneal_strategy" in strict[0]


def test_arch_key_resolves_derived_fields_like_the_constructor():
    from src.neural_network_module.architecture_contracts import resolve_ffn_dim
    for embed_dim in (32, 96, 192, 512, 768):
        for gated in (True, False):
            assert arch.resolve_ffn_dim(embed_dim, None, gated=gated) == resolve_ffn_dim(embed_dim, None, gated=gated)
    assert arch.resolve_ffn_dim(64, 100) == resolve_ffn_dim(64, 100) == 100
    fields = arch.function_defining_fields(_normalized(ffn_dim=None, embed_dim=192, seq_proj_dim=192))
    assert fields["ffn_dim"] == 512 and fields["num_kv_heads"] == 4 and fields["tie_weights"] is True
    wider = arch.function_defining_fields(_normalized(embed_dim=64))                    # seq_proj_dim follows embed_dim
    assert wider["seq_proj_dim"] == 64 and wider["tie_weights"] is True
    projected = arch.function_defining_fields(_normalized(embed_dim=64, seq_proj_dim=32))
    assert projected["tie_weights"] is False                                           # the constructor cannot tie either


def test_arch_key_agrees_with_the_legacy_identity_check():
    """arch:cfg-v1 equality <=> checkpoint_contract.validate_model_identity accepts, on its 22 real fields."""
    pytest.importorskip("torch")
    from model_management.checkpoint_contract import validate_model_identity

    class Stub:
        tie_weights = True

        def __init__(self, config):
            self._ctor_cfg = dict(config)
            self.tie_weights = bool(config["tie_weights"])

    base = _normalized()
    legacy_fields = ("embed_dim", "num_heads", "num_kv_heads", "num_layers", "vocab_size", "use_swiglu", "use_moe",
                     "pre_norm", "use_rmsnorm", "parallel_residual", "pe_mode", "tie_weights", "use_qk_norm",
                     "causal_mask", "attn_logit_cap", "logit_soft_cap", "sliding_window")
    for name in legacy_fields:
        changed = _normalized(**{name: FDF_CHANGES[name]})
        with pytest.raises(ValueError, match="architecture mismatch"):
            validate_model_identity(Stub(base), changed)
        assert arch.arch_cfg_key(changed)[0] != arch.arch_cfg_key(base)[0]
    validate_model_identity(Stub(base), _normalized(dropout=0.5, max_seq_length=128))
    assert arch.arch_cfg_key(_normalized(dropout=0.5, max_seq_length=128))[0] == arch.arch_cfg_key(base)[0]
    # Documented deltas: the legacy check ignores ffn_dim and the length of a learned positional
    # table (both are weight shapes); the key does not.
    validate_model_identity(Stub(base), _normalized(ffn_dim=128))
    assert arch.arch_cfg_key(_normalized(ffn_dim=128))[0] != arch.arch_cfg_key(base)[0]
    learned = _normalized(pe_mode="learned")
    validate_model_identity(Stub(learned), _normalized(pe_mode="learned", max_seq_length=128))
    assert arch.arch_cfg_key(_normalized(pe_mode="learned", max_seq_length=128))[0] != arch.arch_cfg_key(learned)[0]


def test_ctor_v0_and_serve_profile_keys():
    stored = {"vocab_size": 128, "embed_dim": 32, "num_heads": 4, "device": "cuda", "tie_weights": True, "tb_log_dir": None}
    key = arch.arch_ctor_v0_key(stored)
    assert key == arch.arch_ctor_v0_key({**stored, "device": "cpu"}) != arch.arch_ctor_v0_key({**stored, "num_heads": 8})
    profile, payload = arch.serve_profile_key(_normalized())
    assert payload["quantization_type"] == "none" and payload["dtype"] == "float32"
    assert arch.serve_profile_key(_normalized(quantization_type="int8"))[0] != profile
    assert arch.serve_profile_key(_normalized(), dtype="float16")[0] != profile


def test_private_schema_load_gives_the_legacy_normaliser_output():
    """The torch-free loader of config_schema.py must normalise exactly as model_management does."""
    pytest.importorskip("torch")
    import model_management.config_schema as legacy
    from cevahir_ai.adapters import model_build
    spec = importlib.util.spec_from_file_location("_private_schema_probe", ROOT / "model_management/config_schema.py")
    private = importlib.util.module_from_spec(spec)
    sys.modules["_private_schema_probe"] = private
    try:
        spec.loader.exec_module(private)
        cases = [{}, {"device": "cpu"}, {"d_model": 64, "n_heads": 2, "n_layers": 3}, {"arch": {"embed_dim": 96, "num_heads": 4}},
                 {"embed_dim": 64, "seq_proj_dim": None}, {"quant": {"quant_type": "none"}}, {"config_version": 1, "num_layers": 5}]
        cases += [{name: value} for name, value in {**FDF_CHANGES, **NON_FDF_CHANGES}.items() if name != "seq_proj_dim"]
        assert len(cases) >= 40
        for case in cases:
            for profile in (None, "model_manager"):
                assert private.normalize_model_config(case, legacy_profile=profile) == \
                    legacy.normalize_model_config(case, legacy_profile=profile), case
        assert private.tiny_model_config() == legacy.tiny_model_config()
    finally:
        del sys.modules["_private_schema_probe"]
    assert model_build.config_schema() is legacy                                       # loaded package wins


# ------------------------------------------------------------- checkpoint, model, run, evaluation
def test_checkpoint_and_model_keys():
    weights, wschema = Key("weights", "sha256-v1", H), Key("wschema", "layout-v1", H)
    arch_key, tok_key = Key("arch", "cfg-v1", H), Key("tok", "bpe-sem-v1", H)
    full = ckpt.ckpt_members_key({"weights": weights, "wschema": wschema, "arch": arch_key, "tok": tok_key,
                                  "optim": Key("optim", "sha256-v1", H)})
    export = ckpt.ckpt_members_key({"weights": weights, "wschema": wschema, "arch": arch_key, "tok": tok_key})
    assert full != export                                                              # one can resume, the other cannot
    assert ckpt.model_triple_key(weights, arch_key, tok_key) == ckpt.model_triple_key(str(weights), str(arch_key), str(tok_key))
    with pytest.raises(ValueError, match="canonical tokenizer"):
        ckpt.model_triple_key(weights, arch_key, Key("tok", "legacy-v3", H))
    with pytest.raises(ValueError):
        ckpt.ckpt_members_key({"weights": weights})
    with pytest.raises(ValueError):
        ckpt.ckpt_members_key({"weights": weights, "wschema": wschema, "surprise": weights})


def test_signature_keys_are_structural():
    adam = {"state": {0: {"step": 1, "exp_avg": 0, "exp_avg_sq": 0}}, "param_groups": [{"lr": 1e-3, "betas": (0.9, 0.999), "params": [0, 1]}]}
    other_lr = {"state": adam["state"], "param_groups": [{"lr": 5e-5, "betas": (0.8, 0.9), "params": [0, 1]}]}
    sgd = {"state": {0: {"momentum_buffer": None}}, "param_groups": [{"lr": 1e-3, "momentum": 0.9, "params": [0, 1]}]}
    assert ckpt.optim_sig_key(adam)[0] == ckpt.optim_sig_key(other_lr)[0] != ckpt.optim_sig_key(sgd)[0]
    step = {"step_size": 3, "gamma": 0.1, "last_epoch": 4, "base_lrs": [1.0], "_step_count": 5}
    plateau = {"patience": 2, "factor": 0.5, "last_epoch": 4, "best": 1.0, "_last_lr": [1.0]}
    assert ckpt.scheduler_signature(step) == ["gamma", "step_size"]
    assert ckpt.sched_sig_key(step)[0] == ckpt.sched_sig_key({**step, "last_epoch": 99, "_step_count": 1})[0]
    assert ckpt.sched_sig_key(step)[0] != ckpt.sched_sig_key(plateau)[0]


def test_scheduler_signature_rule_is_the_legacy_rule():
    pytest.importorskip("torch")
    from model_management.checkpoint_contract import _GENERIC_SCHEDULER_KEYS, _scheduler_signature
    assert ckpt.GENERIC_SCHEDULER_KEYS == _GENERIC_SCHEDULER_KEYS
    state = {"step_size": 3, "gamma": 0.1, "last_epoch": 4, "base_lrs": [1.0], "_step_count": 5, "verbose": False}
    assert set(ckpt.scheduler_signature(state)) == _scheduler_signature(state)


def test_run_and_evaluation_keys():
    inputs = {"tok": Key("tok", "bpe-sem-v1", H), "cache": Key("cache", "recipe-v1", H), "parent": None}
    spec, payload = runs.runspec_key({"epochs": 3, "learning_rate": 2e-4, "batch_size": 16}, inputs)
    assert "parent" not in payload["inputs"]
    assert runs.runspec_key({"batch_size": 16, "learning_rate": 0.0002, "epochs": 3}, inputs)[0] == spec
    assert runs.runspec_key({"epochs": 4, "learning_rate": 2e-4, "batch_size": 16}, inputs)[0] != spec
    assert runs.runspec_key({"epochs": 3, "learning_rate": 2e-4, "batch_size": 16},
                            {**inputs, "parent": Key("ckpt", "members-v1", H)})[0] != spec
    with pytest.raises(ValueError):
        runs.runspec_key({}, {"checkpoint_dir": Key("tok", "bpe-sem-v1", H)})
    first, second = runs.new_run_key(), runs.new_run_key()
    assert first != second and first.scheme == "id-v1" and len(first.digest) == 32      # two runs of one spec
    assert runs.uuid7_hex(now_ms=0x0123456789AB, random_bytes=bytes(10)) == "0123456789ab70008000000000000000"
    model, evalset = Key("model", "triple-v1", H), Key("evalset", "files-v1", H)
    evaluation, _ = runs.eval_recipe_key(model, evalset, decoding={"temperature": 0.0, "max_new_tokens": 32}, seed=1)
    assert runs.eval_recipe_key(model, evalset, decoding={"max_new_tokens": 32, "temperature": 0}, seed=1)[0] != evaluation
    assert runs.eval_recipe_key(model, evalset, decoding={"max_new_tokens": 32, "temperature": 0.0}, seed=1)[0] == evaluation
    assert runs.eval_recipe_key(model, evalset, decoding={"temperature": 0.0, "max_new_tokens": 32}, seed=2)[0] != evaluation
    with pytest.raises(ValueError):
        runs.eval_recipe_key(evalset, model, decoding={}, seed=1)


def test_code_and_environment_keys(tmp_path):
    for name, newline in (("lf", "\n"), ("crlf", "\r\n")):
        (tmp_path / name / "pkg").mkdir(parents=True)
        (tmp_path / name / "pkg" / "a.py").write_bytes(f"x = 1{newline}y = 2{newline}".encode())
        (tmp_path / name / "b.py").write_bytes(f"z = 3{newline}".encode())
    lf, files = code.code_tree_key_of(tmp_path / "lf", ["pkg/a.py", "b.py"])
    assert [entry[0] for entry in files] == ["b.py", "pkg/a.py"]
    assert code.code_tree_key_of(tmp_path / "crlf", ["b.py", "pkg/a.py"])[0] == lf
    (tmp_path / "lf" / "b.py").write_bytes(b"z = 4\n")
    assert code.code_tree_key_of(tmp_path / "lf", ["pkg/a.py", "b.py"])[0] != lf
    key, payload = code.env_rt_key()
    assert key == code.env_rt_key()[0] and set(payload) == {"python", "implementation", "platform", "machine", "packages"}
    assert payload["python"] == ".".join(str(part) for part in sys.version_info[:3])
    assert not any(os.environ.get(name, "\0") in json.dumps(payload) for name in ("USER", "USERNAME", "HOSTNAME"))


# --------------------------------------------------------------------------------- versions table
def test_versions_table_matches_the_code(tmp_path):
    from cevahir_ai.keys import versions
    assert versions.check(ROOT) == []
    assert {entry.counter for entry in versions.VERSIONS} >= {"cache record format", "tokenizer identity_version",
                                                              "architecture_version", "byte-BPE format version"}
    assert cache.RECORD_FORMAT == next(e.value for e in versions.VERSIONS if e.counter == "cache record format")
    # A bumped counter that is not reflected in the table is reported.
    for entry in versions.VERSIONS:
        target = tmp_path / entry.owner
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry.owner, target)
    owner = tmp_path / "training_system/cache_identity.py"
    owner.write_text(owner.read_text(encoding="utf-8").replace("CACHE_FORMAT_VERSION = 2", "CACHE_FORMAT_VERSION = 3")
                     .replace('"identity_version": 3', '"identity_version": 4'), encoding="utf-8")
    findings = versions.check(tmp_path)
    assert len(findings) == 2 and "cache record format" in findings[0] and "identity_version" in findings[1]


# ------------------------------------------------------------------------------------------- gate
def _facts(**statuses):
    return {check_id: CheckResult(check_id, status, f"{check_id} detail") for check_id, status in statuses.items()}


ALL_PASS = dict(C1="pass", C3="pass", C4="pass", C5="pass", C7="pass", C9="pass", C10="pass", C11="pass", C12="pass")


def test_every_purpose_that_loads_weights_requires_the_shape_architecture_and_dtype_checks():
    for purpose in ("evaluate", "finetune", "resume", "serve", "publish"):
        assert {"C4", "C5", "C7", "C9"} <= set(DEFAULT_GATE_POLICY.required[purpose]), purpose
        for check_id in ("C4", "C5", "C9"):                              # none of them can be waved through as unknown
            assert DEFAULT_GATE_POLICY.unknown_action(purpose, check_id) == "fail", (purpose, check_id)
    assert "C10" in DEFAULT_GATE_POLICY.required["resume"] and "C11" in DEFAULT_GATE_POLICY.required["serve"]


def test_gate_is_pure_and_every_purpose_has_a_requirement_row():
    assert set(DEFAULT_GATE_POLICY.required) == set(PURPOSES)
    for purpose, required in DEFAULT_GATE_POLICY.required.items():
        assert set(required) <= set(IMPLEMENTED_CHECKS) and "C12" in required and "C1" in required
        verdict = check_compatibility(_facts(**ALL_PASS), purpose)
        assert verdict.verdict == "compatible" and verdict.ok and verdict.failing == () and verdict.required == tuple(required)
    assert set(IMPLEMENTED_CHECKS) < set(CHECKS)
    with pytest.raises(ValueError):
        check_compatibility({}, "deploy")
    with pytest.raises(ValueError):
        check_compatibility({"C1": CheckResult("C3", "pass")}, "inspect")
    with pytest.raises(ValueError):
        CheckResult("C99", "pass")


GATE_TABLE = [
    # purpose, overrides, verdict, failing
    ("inspect", {"C1": "unknown"}, "compatible-with-warnings", ()),
    ("inspect", {"C12": "fail"}, "incompatible", ("C12",)),
    ("inspect", {"C12": "unsupported"}, "incompatible", ("C12",)),
    ("inspect", {"C5": "fail", "C7": "fail"}, "compatible", ()),                 # not required for inspect
    ("export", {"C1": "unknown"}, "incompatible", ("C1",)),
    ("evaluate", {"C5": "unknown"}, "incompatible", ("C5",)),                    # architecture unknown: denied
    ("evaluate", {"C5": "fail"}, "incompatible", ("C5",)),                       # wrong head count
    ("evaluate", {"C4": "fail"}, "incompatible", ("C4",)),
    ("evaluate", {"C4": "unknown"}, "incompatible", ("C4",)),
    ("evaluate", {"C1": "unknown", "C3": "unknown", "C7": "unknown"}, "compatible-with-warnings", ()),
    ("evaluate", {"C7": "fail"}, "incompatible", ("C7",)),
    ("evaluate", {"C1": "fail"}, "incompatible", ("C1",)),                       # tampered file
    ("evaluate", {"C3": "fail"}, "incompatible", ("C3",)),
    ("evaluate", {"C10": "unknown", "C11": "fail"}, "compatible", ()),           # resume state and context not required
    ("evaluate", {"C9": "fail"}, "incompatible", ("C9",)),                       # fp16 into fp32: never a silent cast,
    ("finetune", {"C9": "fail"}, "incompatible", ("C9",)),                       # whatever the purpose
    ("resume", {"C9": "fail"}, "incompatible", ("C9",)),
    ("evaluate", {"C9": "unsupported"}, "incompatible", ("C9",)),
    ("inspect", {"C9": "fail"}, "compatible", ()),                               # nothing is loaded
    ("finetune", {"C10": "unknown"}, "compatible", ()),
    ("finetune", {"C5": "unknown"}, "incompatible", ("C5",)),
    ("resume", {"C10": "unknown"}, "incompatible", ("C10",)),                    # weights-only export cannot resume
    ("resume", {"C10": "fail"}, "incompatible", ("C10",)),                       # scheduler type differs
    ("resume", {"C7": "unknown"}, "compatible-with-warnings", ()),
    ("serve", {"C7": "unknown"}, "incompatible", ("C7",)),                       # every unknown denies
    ("serve", {"C1": "unknown"}, "incompatible", ("C1",)),
    ("serve", {"C3": "unknown"}, "incompatible", ("C3",)),
    ("serve", {"C9": "fail"}, "incompatible", ("C9",)),                          # fp16 into fp32
    ("serve", {"C9": "unsupported"}, "incompatible", ("C9",)),                   # quantized
    ("serve", {"C11": "skipped"}, "compatible", ()),
    ("serve", {"C11": "fail"}, "incompatible", ("C11",)),
    ("publish", {"C5": "unknown", "C7": "unknown"}, "incompatible", ("C5", "C7")),
]


@pytest.mark.parametrize("purpose, overrides, expected, failing", GATE_TABLE)
def test_gate_decision_table(purpose, overrides, expected, failing):
    verdict = check_compatibility(_facts(**{**ALL_PASS, **overrides}), purpose)
    assert verdict.verdict == expected and verdict.failing == failing
    assert verdict.ok == (expected in ("compatible", "compatible-with-warnings"))
    for check_id in failing:
        assert check_id in verdict.summary_en and check_id in verdict.summary_tr


def test_gate_missing_fact_is_unverifiable_and_policy_can_tighten():
    partial = {name: status for name, status in ALL_PASS.items() if name in ("C1", "C12")}
    verdict = check_compatibility(_facts(**partial), "evaluate")
    assert verdict.verdict == "unverifiable" and not verdict.ok and set(verdict.failing) == {"C3", "C4", "C5", "C7", "C9"}
    assert check_compatibility(_facts(**partial), "inspect").verdict == "compatible"
    strict = GatePolicy(required={**DEFAULT_GATE_POLICY.required, "evaluate": DEFAULT_GATE_POLICY.required["evaluate"] + ("C11",)},
                        on_unknown={})
    facts = _facts(**{**ALL_PASS, "C7": "unknown", "C11": "fail"})
    assert check_compatibility(facts, "evaluate").verdict == "compatible-with-warnings"
    assert check_compatibility(facts, "evaluate", strict).failing == ("C7", "C11")
    document = check_compatibility(facts, "evaluate", strict).to_dict()
    assert json.loads(json.dumps(document))["checks"][0]["name"] == "envelope.safety"
