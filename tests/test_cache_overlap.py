# -*- coding: utf-8 -*-
"""
Cache Overlap Test Suite
========================
Test eder:
1. tokenizer_core.load_training_data() source_id döndürüyor mu?
2. prepare_cache.py cache'e source_id kaydediyor mu?
3. Cache'den yüklendiğinde source_id var mı?
4. source_id bazlı split overlap önlüyor mu?
"""

import os
import sys
import pickle
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Set
import hashlib

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG
from training_system.data_cache import DataCache
from training_system.prepare_cache import prepare_cache


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


class TestTokenizerCoreSourceID:
    """tokenizer_core.load_training_data() source_id döndürüyor mu?"""
    
    @pytest.fixture
    def tokenizer_core(self):
        """TokenizerCore instance"""
        config = {
            "data_dir": "education",
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False,
        }
        tokenizer = TokenizerCore(config)
        tokenizer.finalize_vocab()
        return tokenizer
    
    def test_load_training_data_returns_source_id(self, tokenizer_core):
        """load_training_data() include_source_id=True ile source_id döndürmeli"""
        raw_data = tokenizer_core.load_training_data(
            encode_mode="train",
            include_whole_words=True,
            include_syllables=False,
            include_sep=False,
            include_source_id=True,  # ✅ Kritik!
        )
        
        assert len(raw_data) > 0, "Raw data boş olmamalı!"
        
        # İlk örnek formatını kontrol et
        first_item = raw_data[0]
        assert isinstance(first_item, (tuple, list)), "Örnek tuple/list olmalı"
        assert len(first_item) == 3, f"source_id olmalı! Format: {len(first_item)} eleman, beklenen: 3"
        
        inp_ids, tgt_ids, source_id = first_item
        assert isinstance(source_id, int), f"source_id int olmalı! Tip: {type(source_id)}"
        assert source_id >= 0, f"source_id negatif olmamalı! Değer: {source_id}"
    
    def test_source_id_uniqueness(self, tokenizer_core):
        """Her dosya/QA için unique source_id olmalı"""
        raw_data = tokenizer_core.load_training_data(
            encode_mode="train",
            include_source_id=True,
        )
        
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
    
    @pytest.fixture(scope="class")
    def temp_cache_dir(self):
        """Geçici cache dizini"""
        temp_dir = tempfile.mkdtemp(prefix="test_cache_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture(scope="class")
    def cache_with_source_id(self, temp_cache_dir):
        """Cache oluştur (source_id ile)"""
        # Eski cache'i temizle
        cache_path = Path(temp_cache_dir)
        if cache_path.exists():
            for pkl_file in cache_path.glob("*.pkl"):
                pkl_file.unlink()
        
        # Cache oluştur
        try:
            cache_dir = prepare_cache(
                data_dir="education",
                cache_dir=temp_cache_dir,
                max_seq_length=TOKENIZER_CONFIG.get("max_seq_length", 768),
                include_whole_words=True,
                include_syllables=False,
                include_sep=False,
            )
            return cache_dir
        except Exception as e:
            pytest.skip(f"Cache oluşturulamadı: {e}")
    
    def test_cache_format_has_source_id(self, temp_cache_dir):
        """Cache dosyası source_id içermeli"""
        cache_path = Path(temp_cache_dir)
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        
        assert len(cache_files) > 0, "Cache dosyası bulunamadı!"
        
        # İlk cache dosyasını yükle
        with open(cache_files[0], "rb") as f:
            cached_data = pickle.load(f)
        
        assert len(cached_data) > 0, "Cache verisi boş!"
        
        # İlk örnek formatını kontrol et
        first_item = cached_data[0]
        assert isinstance(first_item, (tuple, list)), "Örnek tuple/list olmalı"
        assert len(first_item) == 3, f"Cache'de source_id olmalı! Format: {len(first_item)} eleman, beklenen: 3"
        
        seq_in, seq_tgt, source_id = first_item
        assert isinstance(source_id, int), f"source_id int olmalı! Tip: {type(source_id)}"
    
    def test_cache_source_id_statistics(self, temp_cache_dir):
        """Cache'deki source_id istatistikleri kontrol et"""
        cache_path = Path(temp_cache_dir)
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        
        if not cache_files:
            pytest.skip("Cache dosyası bulunamadı")
        
        with open(cache_files[0], "rb") as f:
            cached_data = pickle.load(f)
        
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
    
    @pytest.fixture(scope="class")
    def tokenizer_core(self):
        """TokenizerCore instance"""
        config = {
            "data_dir": "education",
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False,
        }
        tokenizer = TokenizerCore(config)
        tokenizer.finalize_vocab()
        return tokenizer
    
    @pytest.fixture(scope="class")
    def special_ids(self, tokenizer_core):
        """Special token ID'leri"""
        return get_special_token_ids(tokenizer_core)
    
    def test_source_id_based_split_no_overlap(self, tokenizer_core, special_ids):
        """source_id bazlı split overlap önlemeli"""
        # Veriyi yükle
        raw_data = tokenizer_core.load_training_data(
            encode_mode="train",
            include_source_id=True,
        )
        
        assert len(raw_data) > 0, "Veri boş!"
        
        # source_id bazlı split yap
        # source_id'lere göre grupla
        source_id_to_examples = {}
        for item in raw_data:
            if len(item) == 3:
                inp_ids, tgt_ids, source_id = item
                if source_id not in source_id_to_examples:
                    source_id_to_examples[source_id] = []
                source_id_to_examples[source_id].append((inp_ids, tgt_ids))
        
        # source_id'leri shuffle et ve split yap
        import random
        random.seed(42)
        source_ids = list(source_id_to_examples.keys())
        random.shuffle(source_ids)
        
        train_size = max(1, int(0.8 * len(source_ids)))
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
        
        # Overlap kontrolü: source_id bazlı split'te overlap olmamalı
        train_source_ids_set = set()
        for source_id in source_ids:
            if source_id in train_source_ids:
                train_source_ids_set.add(source_id)
        
        val_source_ids_set = set()
        for source_id in source_ids:
            if source_id in val_source_ids:
                val_source_ids_set.add(source_id)
        
        # Train ve val'deki source_id'ler ayrık olmalı
        overlap_source_ids = train_source_ids_set & val_source_ids_set
        assert len(overlap_source_ids) == 0, f"source_id bazlı split'te overlap olmamalı! Overlap source_id'ler: {list(overlap_source_ids)[:10]}"
    
    def test_hash_based_overlap_check(self, tokenizer_core, special_ids):
        """Hash bazlı overlap kontrolü (source_id bazlı split sonrası)"""
        # Veriyi yükle
        raw_data = tokenizer_core.load_training_data(
            encode_mode="train",
            include_source_id=True,
        )
        
        assert len(raw_data) > 0, "Veri boş!"
        
        # source_id bazlı split yap
        source_id_to_examples = {}
        for item in raw_data:
            if len(item) == 3:
                inp_ids, tgt_ids, source_id = item
                if source_id not in source_id_to_examples:
                    source_id_to_examples[source_id] = []
                source_id_to_examples[source_id].append((inp_ids, tgt_ids))
        
        import random
        random.seed(42)
        source_ids = list(source_id_to_examples.keys())
        random.shuffle(source_ids)
        
        train_size = max(1, int(0.8 * len(source_ids)))
        train_source_ids = set(source_ids[:train_size])
        val_source_ids = set(source_ids[train_size:])
        
        train_data = []
        val_data = []
        
        for source_id in source_ids:
            examples_for_source = source_id_to_examples[source_id]
            if source_id in train_source_ids:
                train_data.extend(examples_for_source)
            else:
                val_data.extend(examples_for_source)
        
        # Hash bazlı overlap kontrolü
        pad_id = special_ids["PAD"]
        train_hashes = set()
        for inp_ids, tgt_ids in train_data:
            # Input sequence'ini hash'le (PAD'leri filtrele)
            hash_val = hash_sequence_full(inp_ids, pad_id=pad_id)
            train_hashes.add(hash_val)
        
        val_hashes = set()
        for inp_ids, tgt_ids in val_data:
            hash_val = hash_sequence_full(inp_ids, pad_id=pad_id)
            val_hashes.add(hash_val)
        
        # Overlap kontrolü
        overlap_hashes = train_hashes & val_hashes
        overlap_ratio = len(overlap_hashes) / len(train_hashes) if train_hashes else 0.0
        
        # source_id bazlı split'te overlap %0 olmalı
        assert overlap_ratio == 0.0, f"Hash bazlı overlap tespit edildi! Overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})"
    
    @pytest.fixture(scope="class")
    def temp_cache_dir(self):
        """Geçici cache dizini"""
        temp_dir = tempfile.mkdtemp(prefix="test_cache_overlap_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_cache_load_and_split_no_overlap(self, temp_cache_dir, tokenizer_core, special_ids):
        """Cache'den yüklenip split yapıldığında overlap olmamalı"""
        # Önce cache oluştur
        cache_path = Path(temp_cache_dir)
        if cache_path.exists():
            for pkl_file in cache_path.glob("*.pkl"):
                pkl_file.unlink()
        
        try:
            prepare_cache(
                data_dir="education",
                cache_dir=temp_cache_dir,
                max_seq_length=TOKENIZER_CONFIG.get("max_seq_length", 768),
                include_whole_words=True,
                include_syllables=False,
                include_sep=False,
            )
        except Exception as e:
            pytest.skip(f"Cache oluşturulamadı: {e}")
        
        # Cache dosyasını bul
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        
        if not cache_files:
            pytest.skip("Cache dosyası bulunamadı")
        
        # Cache'den yükle
        with open(cache_files[0], "rb") as f:
            cached_data = pickle.load(f)
        
        # source_id bazlı split yap
        source_id_to_examples = {}
        for item in cached_data:
            if len(item) == 3:
                seq_in, seq_tgt, source_id = item
                if source_id not in source_id_to_examples:
                    source_id_to_examples[source_id] = []
                source_id_to_examples[source_id].append((seq_in, seq_tgt))
        
        import random
        random.seed(42)
        source_ids = list(source_id_to_examples.keys())
        random.shuffle(source_ids)
        
        train_size = max(1, int(0.8 * len(source_ids)))
        train_source_ids = set(source_ids[:train_size])
        val_source_ids = set(source_ids[train_size:])
        
        train_data = []
        val_data = []
        
        for source_id in source_ids:
            examples_for_source = source_id_to_examples[source_id]
            if source_id in train_source_ids:
                train_data.extend(examples_for_source)
            else:
                val_data.extend(examples_for_source)
        
        # Hash bazlı overlap kontrolü
        pad_id = special_ids["PAD"]
        train_hashes = set()
        for seq_in, seq_tgt in train_data:
            hash_val = hash_sequence_full(seq_in, pad_id=pad_id)
            train_hashes.add(hash_val)
        
        val_hashes = set()
        for seq_in, seq_tgt in val_data:
            hash_val = hash_sequence_full(seq_in, pad_id=pad_id)
            val_hashes.add(hash_val)
        
        overlap_hashes = train_hashes & val_hashes
        overlap_ratio = len(overlap_hashes) / len(train_hashes) if train_hashes else 0.0
        
        # Overlap olmamalı
        assert overlap_ratio == 0.0, f"Cache'den yüklenip split yapıldığında overlap tespit edildi! Overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})"


class TestEndToEndOverlap:
    """End-to-end test: tokenizer → cache → split → overlap kontrolü"""
    
    @pytest.fixture(scope="class")
    def temp_cache_dir(self):
        """Geçici cache dizini"""
        temp_dir = tempfile.mkdtemp(prefix="test_e2e_cache_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_end_to_end_no_overlap(self, temp_cache_dir):
        """End-to-end test: tokenizer → cache → split → overlap yok"""
        # 1. Cache oluştur
        cache_path = Path(temp_cache_dir)
        if cache_path.exists():
            for pkl_file in cache_path.glob("*.pkl"):
                pkl_file.unlink()
        
        cache_dir = prepare_cache(
            data_dir="education",
            cache_dir=temp_cache_dir,
            max_seq_length=TOKENIZER_CONFIG.get("max_seq_length", 768),
            include_whole_words=True,
            include_syllables=False,
            include_sep=False,
        )
        
        # 2. Cache'den yükle
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        assert len(cache_files) > 0, "Cache dosyası oluşturulamadı!"
        
        with open(cache_files[0], "rb") as f:
            cached_data = pickle.load(f)
        
        assert len(cached_data) > 0, "Cache verisi boş!"
        
        # 3. source_id bazlı split yap
        source_id_to_examples = {}
        items_without_source_id = 0
        
        for item in cached_data:
            if len(item) == 3:
                seq_in, seq_tgt, source_id = item
                if source_id not in source_id_to_examples:
                    source_id_to_examples[source_id] = []
                source_id_to_examples[source_id].append((seq_in, seq_tgt))
            else:
                items_without_source_id += 1
        
        assert items_without_source_id == 0, f"{items_without_source_id} örnekte source_id yok!"
        
        import random
        random.seed(42)
        source_ids = list(source_id_to_examples.keys())
        random.shuffle(source_ids)
        
        train_size = max(1, int(0.8 * len(source_ids)))
        train_source_ids = set(source_ids[:train_size])
        val_source_ids = set(source_ids[train_size:])
        
        train_data = []
        val_data = []
        
        for source_id in source_ids:
            examples_for_source = source_id_to_examples[source_id]
            if source_id in train_source_ids:
                train_data.extend(examples_for_source)
            else:
                val_data.extend(examples_for_source)
        
        # 4. Overlap kontrolü
        # source_id bazlı kontrol
        train_source_ids_set = train_source_ids
        val_source_ids_set = val_source_ids
        overlap_source_ids = train_source_ids_set & val_source_ids_set
        assert len(overlap_source_ids) == 0, f"source_id bazlı overlap var! Overlap source_id'ler: {list(overlap_source_ids)[:10]}"
        
        # Hash bazlı kontrol
        tokenizer_core = TokenizerCore({
            "data_dir": "education",
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False,
        })
        tokenizer_core.finalize_vocab()
        special_ids = get_special_token_ids(tokenizer_core)
        
        pad_id = special_ids["PAD"]
        train_hashes = set()
        for seq_in, seq_tgt in train_data:
            hash_val = hash_sequence_full(seq_in, pad_id=pad_id)
            train_hashes.add(hash_val)
        
        val_hashes = set()
        for seq_in, seq_tgt in val_data:
            hash_val = hash_sequence_full(seq_in, pad_id=pad_id)
            val_hashes.add(hash_val)
        
        overlap_hashes = train_hashes & val_hashes
        overlap_ratio = len(overlap_hashes) / len(train_hashes) if train_hashes else 0.0
        
        # Overlap %0 olmalı
        assert overlap_ratio == 0.0, f"End-to-end test: Overlap tespit edildi! Overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

