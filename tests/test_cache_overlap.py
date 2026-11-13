# -*- coding: utf-8 -*-
"""
Cache Overlap Test Suite
========================
Test eder:
1. tokenizer_core.load_training_data() source_id döndürüyor mu?
2. prepare_cache.py cache'e source_id kaydediyor mu?
3. Cache'den yüklendiğinde source_id var mı?
4. source_id bazlı split overlap önlüyor mu?

Korpusun tokenize edilmesi (load_training_data / prepare_cache) her seferinde onlarca saniye
sürer. Bu yüzden ham veri ve cache modül kapsamlı fixture'larda birer kez üretilir; testler
aynı sonucu okur. Cache depo ağacına değil, tmp_path_factory dizinine yazılır.
"""

import os
import sys
import pickle
import random
import hashlib
from typing import List

import pytest

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG
from training_system.prepare_cache import prepare_cache

# Eğitim korpusu (mutlak yol: testler hangi dizinden çalıştırılırsa çalıştırılsın aynı veri)
DATA_DIR = os.path.join(project_root, "education")
MAX_SEQ_LENGTH = TOKENIZER_CONFIG.get("max_seq_length", 768)


def hash_sequence_full(seq: List[int], pad_id: int = 0) -> str:
    """Sequence'in tamamını hash'le (PAD token'ları filtrele)"""
    clean_seq = [t for t in seq if t != pad_id]
    seq_str = str(clean_seq)
    return hashlib.sha256(seq_str.encode()).hexdigest()


def get_special_token_ids(tokenizer_core: TokenizerCore) -> dict:
    """Special token ID'lerini al"""
    vocab = tokenizer_core.get_vocab()
    
    def _id_of(token: str) -> int:
        val = vocab.get(token)
        if isinstance(val, dict):
            return int(val.get("id", 0))
        return int(val or 0)
    
    return {
        "BOS": _id_of("<BOS>"),
        "EOS": _id_of("<EOS>"),
        "PAD": _id_of("<PAD>"),
        "UNK": _id_of("<UNK>"),
        "SEP": _id_of("<SEP>"),
    }


def split_by_source_id(items, seed: int = 42, train_ratio: float = 0.8):
    """
    source_id bazlı (document-level) split: aynı source_id'nin tüm örnekleri aynı tarafa gider.

    Returns:
        (train_source_ids, val_source_ids, train_data, val_data)
    """
    # source_id'lere göre grupla
    source_id_to_examples = {}
    for item in items:
        if len(item) == 3:
            seq_in, seq_tgt, source_id = item
            source_id_to_examples.setdefault(source_id, []).append((seq_in, seq_tgt))
    
    # source_id'leri shuffle et ve split yap
    # (yerel Random(seed): random.seed(seed) + random.shuffle ile aynı sıra, global state değişmez)
    source_ids = list(source_id_to_examples.keys())
    random.Random(seed).shuffle(source_ids)
    
    train_size = max(1, int(train_ratio * len(source_ids)))
    train_source_ids = set(source_ids[:train_size])
    val_source_ids = set(source_ids[train_size:])
    
    # source_id'lere göre örnekleri ayır
    train_data = []
    val_data = []
    for source_id in source_ids:
        examples_for_source = source_id_to_examples[source_id]
        if source_id in train_source_ids:
            train_data.extend(examples_for_source)
        else:
            val_data.extend(examples_for_source)
    
    # Boş bir taraf, overlap kontrollerini anlamsız (her zaman geçen) hale getirir
    assert train_data and val_data, (
        f"Split dejenere: train={len(train_data)}, val={len(val_data)} örnek "
        f"({len(source_ids)} farklı source_id)"
    )
    return train_source_ids, val_source_ids, train_data, val_data


def hash_overlap(train_data, val_data, pad_id: int):
    """Train ve val input sequence'leri arasındaki hash bazlı overlap (PAD'ler filtrelenir)"""
    train_hashes = {hash_sequence_full(seq_in, pad_id=pad_id) for seq_in, _ in train_data}
    val_hashes = {hash_sequence_full(seq_in, pad_id=pad_id) for seq_in, _ in val_data}
    overlap_hashes = train_hashes & val_hashes
    overlap_ratio = len(overlap_hashes) / len(train_hashes) if train_hashes else 0.0
    return overlap_hashes, overlap_ratio


# ---------------------------------------------------------------------------
# Modül kapsamlı fixture'lar (pahalı korpus işlemleri birer kez yapılır)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer_core():
    """TokenizerCore instance"""
    config = {
        "data_dir": DATA_DIR,
        "vocab_path": BPE_CONFIG["vocab_file"],
        "merges_path": BPE_CONFIG["merges_file"],
        "use_gpu": False,
    }
    tokenizer = TokenizerCore(config)
    tokenizer.finalize_vocab()
    return tokenizer


@pytest.fixture(scope="module")
def special_ids(tokenizer_core):
    """Special token ID'leri"""
    return get_special_token_ids(tokenizer_core)


@pytest.fixture(scope="module")
def raw_data_prepare_flags(tokenizer_core):
    """load_training_data(), prepare_cache ile aynı bayraklarla (whole_words, syllables/sep yok)"""
    return tokenizer_core.load_training_data(
        encode_mode="train",
        include_whole_words=True,
        include_syllables=False,
        include_sep=False,
        include_source_id=True,  # ✅ Kritik!
    )


@pytest.fixture(scope="module")
def raw_data_default_flags(tokenizer_core):
    """load_training_data(), include_* bayrakları verilmeden (train modu varsayılanları)"""
    return tokenizer_core.load_training_data(
        encode_mode="train",
        include_source_id=True,
    )


@pytest.fixture(scope="module")
def cached_data(tmp_path_factory):
    """prepare_cache() ile geçici dizinde cache oluştur (source_id ile) ve cache dosyasını yükle"""
    cache_dir = tmp_path_factory.mktemp("cache_overlap")
    prepare_cache(
        data_dir=DATA_DIR,
        cache_dir=str(cache_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        include_whole_words=True,
        include_syllables=False,
        include_sep=False,
    )
    
    # Boş bir dizinde tek bir prepare_cache() çağrısı tam olarak bir cache dosyası üretmeli
    cache_files = sorted(cache_dir.glob("cached_data_*.pkl"))
    assert len(cache_files) == 1, f"Cache dosyası bulunamadı (veya birden fazla): {[f.name for f in cache_files]}"
    
    with open(cache_files[0], "rb") as f:
        return pickle.load(f)


class TestTokenizerCoreSourceID:
    """tokenizer_core.load_training_data() source_id döndürüyor mu?"""
    
    def test_load_training_data_returns_source_id(self, raw_data_prepare_flags):
        """load_training_data() include_source_id=True ile source_id döndürmeli"""
        raw_data = raw_data_prepare_flags
        
        assert len(raw_data) > 0, "Raw data boş olmamalı!"
        
        # İlk örnek formatını kontrol et
        first_item = raw_data[0]
        assert isinstance(first_item, (tuple, list)), "Örnek tuple/list olmalı"
        assert len(first_item) == 3, f"source_id olmalı! Format: {len(first_item)} eleman, beklenen: 3"
        
        inp_ids, tgt_ids, source_id = first_item
        assert isinstance(source_id, int), f"source_id int olmalı! Tip: {type(source_id)}"
        assert source_id >= 0, f"source_id negatif olmamalı! Değer: {source_id}"
    
    def test_source_id_uniqueness(self, raw_data_default_flags):
        """Her dosya/QA için unique source_id olmalı"""
        raw_data = raw_data_default_flags
        
        # source_id'leri topla
        source_ids = []
        for item in raw_data:
            if len(item) == 3:
                _, _, source_id = item
                source_ids.append(source_id)
        
        # En az 2 farklı source_id olmalı (dosya çeşitliliği)
        unique_source_ids = set(source_ids)
        assert len(unique_source_ids) >= 2, f"En az 2 farklı source_id olmalı! Bulunan: {len(unique_source_ids)}"
        
        # source_id'ler 0'dan başlamalı veya offset ile (QA için 1000000+)
        min_source_id = min(unique_source_ids)
        max_source_id = max(unique_source_ids)
        assert min_source_id >= 0, f"source_id negatif olmamalı! Min: {min_source_id}"
        assert max_source_id < 2000000, f"source_id çok büyük! Max: {max_source_id}"


class TestCacheSourceID:
    """prepare_cache.py cache'e source_id kaydediyor mu?"""
    
    def test_cache_format_has_source_id(self, cached_data):
        """Cache dosyası source_id içermeli"""
        assert len(cached_data) > 0, "Cache verisi boş!"
        
        # İlk örnek formatını kontrol et
        first_item = cached_data[0]
        assert isinstance(first_item, (tuple, list)), "Örnek tuple/list olmalı"
        assert len(first_item) == 3, f"Cache'de source_id olmalı! Format: {len(first_item)} eleman, beklenen: 3"
        
        seq_in, seq_tgt, source_id = first_item
        assert isinstance(source_id, int), f"source_id int olmalı! Tip: {type(source_id)}"
    
    def test_cache_source_id_statistics(self, cached_data):
        """Cache'deki source_id istatistikleri kontrol et"""
        # source_id'leri topla
        source_ids = []
        items_without_source_id = 0
        
        for item in cached_data:
            if len(item) == 3:
                _, _, source_id = item
                source_ids.append(source_id)
            else:
                items_without_source_id += 1
        
        # Tüm örneklerde source_id olmalı
        assert items_without_source_id == 0, f"{items_without_source_id} örnekte source_id yok!"
        
        # En az 2 farklı source_id olmalı
        unique_source_ids = set(source_ids)
        assert len(unique_source_ids) >= 2, f"En az 2 farklı source_id olmalı! Bulunan: {len(unique_source_ids)}"


class TestCacheOverlap:
    """source_id bazlı split overlap önlüyor mu?"""
    
    def test_source_id_based_split_no_overlap(self, raw_data_default_flags):
        """source_id bazlı split overlap önlemeli"""
        raw_data = raw_data_default_flags
        assert len(raw_data) > 0, "Veri boş!"
        
        train_source_ids, val_source_ids, _, _ = split_by_source_id(raw_data)
        
        # Train ve val'deki source_id'ler ayrık olmalı
        overlap_source_ids = train_source_ids & val_source_ids
        assert len(overlap_source_ids) == 0, f"source_id bazlı split'te overlap olmamalı! Overlap source_id'ler: {list(overlap_source_ids)[:10]}"
    
    def test_hash_based_overlap_check(self, raw_data_default_flags, special_ids):
        """Hash bazlı overlap kontrolü (source_id bazlı split sonrası)"""
        raw_data = raw_data_default_flags
        assert len(raw_data) > 0, "Veri boş!"
        
        _, _, train_data, val_data = split_by_source_id(raw_data)
        
        # Hash bazlı overlap kontrolü (input sequence'leri, PAD'ler filtrelenmiş)
        overlap_hashes, overlap_ratio = hash_overlap(train_data, val_data, pad_id=special_ids["PAD"])
        
        # source_id bazlı split'te overlap %0 olmalı
        assert overlap_ratio == 0.0, f"Hash bazlı overlap tespit edildi! Overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})"
    
    def test_cache_load_and_split_no_overlap(self, cached_data, special_ids):
        """Cache'den yüklenip split yapıldığında overlap olmamalı"""
        _, _, train_data, val_data = split_by_source_id(cached_data)
        
        overlap_hashes, overlap_ratio = hash_overlap(train_data, val_data, pad_id=special_ids["PAD"])
        
        # Overlap olmamalı
        assert overlap_ratio == 0.0, f"Cache'den yüklenip split yapıldığında overlap tespit edildi! Overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})"


class TestEndToEndOverlap:
    """End-to-end test: tokenizer → cache → split → overlap kontrolü"""
    
    def test_end_to_end_no_overlap(self, cached_data, special_ids):
        """End-to-end test: tokenizer → cache (cached_data fixture'ı) → split → overlap yok"""
        assert len(cached_data) > 0, "Cache verisi boş!"
        
        # Tüm örneklerde source_id olmalı
        items_without_source_id = sum(1 for item in cached_data if len(item) != 3)
        assert items_without_source_id == 0, f"{items_without_source_id} örnekte source_id yok!"
        
        # source_id bazlı split
        train_source_ids, val_source_ids, train_data, val_data = split_by_source_id(cached_data)
        
        # source_id bazlı kontrol
        overlap_source_ids = train_source_ids & val_source_ids
        assert len(overlap_source_ids) == 0, f"source_id bazlı overlap var! Overlap source_id'ler: {list(overlap_source_ids)[:10]}"
        
        # Hash bazlı kontrol
        overlap_hashes, overlap_ratio = hash_overlap(train_data, val_data, pad_id=special_ids["PAD"])
        
        # Overlap %0 olmalı
        assert overlap_ratio == 0.0, f"End-to-end test: Overlap tespit edildi! Overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
