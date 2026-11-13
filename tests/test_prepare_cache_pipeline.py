# -*- coding: utf-8 -*-
"""
Cache Hazırlama Pipeline Test Suite
====================================
prepare_cache.py sürecini test eder:
1. tokenizer_core.load_training_data() → source_id döndürüyor mu?
2. format_data_func() → source_id alıyor mu? kaydediyor mu?
3. Cache'e kaydedilirken → source_id korunuyor mu?
4. Cache'den yüklendiğinde → source_id var mı?

Korpusun tokenize edilmesi (load_training_data / prepare_cache) her seferinde onlarca saniye
sürer. Bu yüzden ham veri ve hazırlanan cache modül kapsamlı fixture'larda birer kez üretilir;
testler aynı sonucu okur. Cache depo ağacına değil, tmp_path_factory dizinine yazılır.
"""

import os
import sys
import pickle
import random
from collections import Counter

import pytest

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG
from training_system.data_cache import DataCache
from training_system.prepare_cache import prepare_cache

# Eğitim korpusu (mutlak yol: testler hangi dizinden çalıştırılırsa çalıştırılsın aynı veri)
DATA_DIR = os.path.join(project_root, "education")
MAX_SEQ_LENGTH = TOKENIZER_CONFIG.get("max_seq_length", 768)


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
    """prepare_cache() full pipeline'ını geçici dizinde bir kez çalıştır ve cache dosyasını yükle"""
    cache_dir = tmp_path_factory.mktemp("prepare_cache")
    prepare_cache(
        data_dir=DATA_DIR,
        cache_dir=str(cache_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        include_whole_words=True,
        include_syllables=False,
        include_sep=False,
        clear_old_cache=True,
    )
    
    # Boş bir dizinde tek bir prepare_cache() çağrısı tam olarak bir cache dosyası üretmeli
    cache_files = sorted(cache_dir.glob("cached_data_*.pkl"))
    assert len(cache_files) == 1, f"Cache dosyası oluşturulamadı (veya birden fazla): {[f.name for f in cache_files]}"
    
    with open(cache_files[0], "rb") as f:
        return pickle.load(f)


class TestPrepareCachePipeline:
    """prepare_cache.py pipeline'ını test et"""
    
    def test_step1_tokenizer_returns_source_id(self, raw_data_prepare_flags):
        """Adım 1: tokenizer_core.load_training_data() source_id döndürüyor mu?"""
        raw_data = raw_data_prepare_flags
        
        assert len(raw_data) > 0, "Raw data boş olmamalı!"
        
        # Format kontrolü
        first_item = raw_data[0]
        assert isinstance(first_item, (tuple, list)), "Örnek tuple/list olmalı"
        assert len(first_item) == 3, f"Adım 1 BAŞARISIZ: source_id yok! Format: {len(first_item)} eleman, beklenen: 3"
        
        inp_ids, tgt_ids, source_id = first_item
        assert isinstance(source_id, int), f"source_id int olmalı! Tip: {type(source_id)}"
        
        # source_id istatistikleri
        source_ids = []
        items_without_source_id = 0
        for item in raw_data:
            if len(item) == 3:
                _, _, sid = item
                source_ids.append(sid)
            else:
                items_without_source_id += 1
        
        assert items_without_source_id == 0, f"Adım 1 BAŞARISIZ: {items_without_source_id} örnekte source_id yok!"
        
        unique_source_ids = set(source_ids)
        assert len(unique_source_ids) >= 2, f"Adım 1 BAŞARISIZ: En az 2 farklı source_id olmalı! Bulunan: {len(unique_source_ids)}"
    
    def test_step2_format_function_preserves_source_id(self, tokenizer_core, raw_data_default_flags):
        """Adım 2: format_data_func() source_id koruyor mu?"""
        raw_data = raw_data_default_flags
        
        # format_data_func'u simulate et (prepare_cache.py'deki gibi)
        vocab = tokenizer_core.get_vocab()
        
        def _id_of(token: str) -> int:
            val = vocab.get(token)
            if isinstance(val, dict):
                return int(val.get("id", 0))
            return int(val or 0)
        
        BOS_ID = _id_of("<BOS>")
        EOS_ID = _id_of("<EOS>")
        PAD_ID = _id_of("<PAD>")
        max_seq_length = MAX_SEQ_LENGTH
        
        # Formatlama fonksiyonu (prepare_cache.py'deki gibi)
        def format_data_func(raw_data):
            formatted_data = []
            
            for item in raw_data:
                source_id = None
                if len(item) == 3:
                    inp_ids, tgt_ids, source_id = item
                elif len(item) == 2:
                    inp_ids, tgt_ids = item
                else:
                    continue
                
                # Basit formatlama (gerçek formatlama daha karmaşık)
                seq_in = [BOS_ID] + list(inp_ids)[:max_seq_length-1]
                seq_tgt = list(tgt_ids)[:max_seq_length-1] + [EOS_ID]
                
                # Padding
                while len(seq_in) < max_seq_length:
                    seq_in.append(PAD_ID)
                while len(seq_tgt) < max_seq_length:
                    seq_tgt.append(PAD_ID)
                
                # source_id'yi koru
                if source_id is not None:
                    formatted_data.append((seq_in, seq_tgt, source_id))
                else:
                    formatted_data.append((seq_in, seq_tgt))
            
            return formatted_data
        
        # Formatla
        formatted_data = format_data_func(raw_data)
        
        # Kontrol
        assert len(formatted_data) > 0, "Formatlanmış veri boş!"
        
        # source_id korunmuş mu?
        formatted_without_source_id = sum(1 for item in formatted_data if len(item) != 3)
        assert formatted_without_source_id == 0, f"Adım 2 BAŞARISIZ: {formatted_without_source_id} formatlanmış örnekte source_id yok!"
        
        # source_id'ler korunmuş mu?
        original_source_ids = {item[2] for item in raw_data if len(item) == 3}
        formatted_source_ids = {item[2] for item in formatted_data if len(item) == 3}
        
        # source_id'ler aynı olmalı
        assert original_source_ids == formatted_source_ids, f"Adım 2 BAŞARISIZ: source_id'ler korunmamış! Orijinal: {len(original_source_ids)}, Formatlanmış: {len(formatted_source_ids)}"
    
    def test_step3_cache_save_preserves_source_id(self, tmp_path):
        """Adım 3: Cache'e kaydedilirken source_id korunuyor mu?"""
        # Test verisi hazırla (source_id ile)
        test_data = [
            ([1, 2, 3], [1, 2, 3], 0),  # source_id = 0
            ([4, 5, 6], [4, 5, 6], 1),  # source_id = 1
            ([7, 8, 9], [7, 8, 9], 0),  # source_id = 0 (aynı dosya)
        ]
        
        # Cache'e kaydet ve geri yükle: DataCache.save_cached_data / get_cached_data public API.
        # (Eski sürüm save_cached_data'yı private sanıp pickle'ı elle çağırıyordu; böylece DataCache'in
        # kaydetme/yükleme yolu hiç test edilmiyordu.)
        cache = DataCache(data_dir=DATA_DIR, cache_dir=str(tmp_path))
        assert cache.save_cached_data("test_key", "test_hash", test_data), "Adım 3 BAŞARISIZ: Cache kaydedilemedi!"
        
        loaded_data = cache.get_cached_data("test_key", "test_hash")
        assert loaded_data is not None, "Adım 3 BAŞARISIZ: Kaydedilen cache geri yüklenemedi!"
        
        # Kontrol
        assert len(loaded_data) == len(test_data), f"Adım 3 BAŞARISIZ: Veri kaybı! Orijinal: {len(test_data)}, Yüklenen: {len(loaded_data)}"
        
        for i, (original, loaded) in enumerate(zip(test_data, loaded_data)):
            assert len(loaded) == 3, f"Adım 3 BAŞARISIZ: Örnek {i} source_id kaybetmiş! Format: {len(loaded)}"
            
            orig_inp, orig_tgt, orig_sid = original
            load_inp, load_tgt, load_sid = loaded
            
            assert load_sid == orig_sid, f"Adım 3 BAŞARISIZ: Örnek {i} source_id değişmiş! Orijinal: {orig_sid}, Yüklenen: {load_sid}"
    
    def test_step4_prepare_cache_full_pipeline(self, cached_data):
        """Adım 4: prepare_cache() full pipeline source_id koruyor mu? (cache: cached_data fixture'ı)"""
        assert len(cached_data) > 0, "Adım 4 BAŞARISIZ: Cache verisi boş!"
        
        # source_id kontrolü
        items_without_source_id = 0
        source_ids = []
        
        for item in cached_data:
            if len(item) == 3:
                _, _, source_id = item
                source_ids.append(source_id)
            else:
                items_without_source_id += 1
        
        # Tüm örneklerde source_id olmalı
        assert items_without_source_id == 0, f"Adım 4 BAŞARISIZ: {items_without_source_id}/{len(cached_data)} örnekte source_id yok!"
        
        # En az 2 farklı source_id olmalı
        unique_source_ids = set(source_ids)
        assert len(unique_source_ids) >= 2, f"Adım 4 BAŞARISIZ: En az 2 farklı source_id olmalı! Bulunan: {len(unique_source_ids)}"
    
    def test_step5_source_id_based_split_no_overlap(self, tokenizer_core, cached_data):
        """Adım 5: Cache'den yüklenip source_id bazlı split yapıldığında overlap olmamalı"""
        train_source_ids, val_source_ids, train_data, val_data = split_by_source_id(cached_data)
        
        # source_id bazlı overlap kontrolü
        overlap_source_ids = train_source_ids & val_source_ids
        assert len(overlap_source_ids) == 0, f"Adım 5 BAŞARISIZ: source_id bazlı overlap var! Overlap source_id'ler: {list(overlap_source_ids)[:10]}"
        
        # Hash bazlı overlap kontrolü
        pad_id = tokenizer_core.get_vocab().get("<PAD>")
        if isinstance(pad_id, dict):
            pad_id = pad_id.get("id", 0)
        pad_id = int(pad_id or 0)
        
        def hash_sequence(seq, pad_id):
            clean_seq = [t for t in seq if t != pad_id]
            seq_str = str(clean_seq)
            return hash(seq_str)  # Basit hash
        
        train_hashes = {hash_sequence(seq_in, pad_id) for seq_in, _ in train_data}
        val_hashes = {hash_sequence(seq_in, pad_id) for seq_in, _ in val_data}
        overlap_hashes = train_hashes & val_hashes
        overlap_ratio = len(overlap_hashes) / len(train_hashes) if train_hashes else 0.0
        
        # ✅ ENDÜSTRİ STANDARDI: Document-level split kontrolü
        # source_id bazlı split doğru çalışmalı (aynı document train/val'e bölünmemeli)
        overlap_source_ids_cross = train_source_ids & val_source_ids
        source_id_split_ok = len(overlap_source_ids_cross) == 0
        assert source_id_split_ok, f"Adım 5 BAŞARISIZ: source_id bazlı split hatalı! {len(overlap_source_ids_cross)} source_id hem train hem val'de!"
        
        # ⚠️ Hash bazlı overlap: Duplicate content (farklı dosyalarda aynı içerik)
        # Bu normal bir durum olabilir ve kabul edilebilir; yalnızca tanı bilgisi olarak raporlanır
        if overlap_hashes:
            overlap_by_source = Counter(
                source_id for seq_in, _, source_id in cached_data
                if hash_sequence(seq_in, pad_id) in overlap_hashes
            )
            print(f"\n⚠️  Hash bazlı overlap tespit edildi: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})")
            print(f"   Bu, farklı dosya/QA'lardan gelen aynı içeriği gösterir (duplicate content)")
            print(f"   Overlap içeren source_id'ler (ilk 10, örnek sayısı): {overlap_by_source.most_common(10)}")


class TestPrepareCachePipelineIntegration:
    """End-to-end integration test: Tüm pipeline birlikte"""
    
    def test_full_pipeline_no_overlap(self, cached_data):
        """Full pipeline test: prepare_cache() (cached_data fixture'ı) → cache → split → overlap kontrolü"""
        assert len(cached_data) > 0, "Cache verisi boş!"
        
        # source_id kontrolü
        for item in cached_data:
            assert len(item) == 3, f"Cache'de source_id yok! Format: {len(item)}"
        
        # source_id bazlı split
        train_source_ids, val_source_ids, train_data, val_data = split_by_source_id(cached_data)
        
        # Overlap kontrolü
        overlap_source_ids = train_source_ids & val_source_ids
        assert len(overlap_source_ids) == 0, f"Integration test BAŞARISIZ: source_id bazlı overlap! Overlap source_id'ler: {list(overlap_source_ids)[:10]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
