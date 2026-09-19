"""Cache identities must match between preparation and strict training."""
from types import SimpleNamespace

import pytest

from training_system.cache_identity import tokenizer_digest
from training_system.data_cache import DataCache
from training_system.v3.data.cache_v3 import DataCacheV3, CacheIntegrityError


def tokenizer(tmp_path):
    merges = tmp_path / "merges.txt"
    merges.write_text("a b\n", encoding="utf-8")
    return SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1},
                           merges_path=merges, tokenizer=SimpleNamespace(config={"mode": "train"}))


def test_content_edits_same_size_invalidate_both_cache_paths(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "sample.txt"
    source.write_text("abc", encoding="utf-8")
    prepared = DataCache(str(data), str(tmp_path / "cache"))
    strict = DataCacheV3(str(data), str(tmp_path / "cache"))
    before = prepared._get_data_dir_hash()
    assert strict._get_data_dir_hash() == before
    source.write_text("xyz", encoding="utf-8")
    after = prepared._get_data_dir_hash()
    assert before != after == strict._get_data_dir_hash()


def test_merge_and_bpe_configuration_changes_invalidate_identity(tmp_path):
    core = tokenizer(tmp_path)
    before = tokenizer_digest(core)
    core.merges_path.write_text("b a\n", encoding="utf-8")
    changed = tokenizer_digest(core)
    assert before != changed
    core.tokenizer.config["mode"] = "inference"
    assert tokenizer_digest(core) != changed


def test_prepare_and_training_keys_agree_and_wrong_cache_is_not_reused(tmp_path):
    core = tokenizer(tmp_path)
    prepared = DataCache(str(tmp_path), str(tmp_path / "cache"))
    strict = DataCacheV3(str(tmp_path), str(tmp_path / "cache"))
    args = ("train", True, False, False, 16, tokenizer_digest(core))
    assert prepared._get_cache_key(*args) == strict.get_cache_key(*args)
    prepared.save_cached_data("wrong", "old", [([1], [2])])
    assert prepared.get_cached_data("right", "new") is None


def test_strict_integrity_requires_checksum_and_detects_corruption(tmp_path):
    cache = DataCacheV3(str(tmp_path), str(tmp_path / "cache"))
    path = cache._get_cache_path("key", "data")
    path.write_bytes(b"not a trusted pickle")
    assert not cache._verify_checksum(path)
    cache._save_checksum(path)
    assert cache._verify_checksum(path)
    path.write_bytes(b"modified")
    with pytest.raises(CacheIntegrityError):
        cache.load_strict("key", "data")
