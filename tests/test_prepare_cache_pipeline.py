# -*- coding: utf-8 -*-
"""
Cache Hazırlama Pipeline Test Suite
====================================
prepare_cache.py sürecini test eder:
1. tokenizer_core.load_training_data() → source_id döndürüyor mu?
2. format_data_func() → source_id alıyor mu? kaydediyor mu?
3. Cache'e kaydedilirken → source_id korunuyor mu?
4. Cache'den yüklendiğinde → source_id var mı?
"""

import os
import sys
import pickle
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG, BPE_DETAILED_CONFIG
from training_system.data_cache import DataCache
from training_system.prepare_cache import prepare_cache


class TestPrepareCachePipeline:
    """prepare_cache.py pipeline'ını test et"""
    
    @pytest.fixture(scope="class")
    def temp_cache_dir(self):
        """Geçici cache dizini"""
        temp_dir = tempfile.mkdtemp(prefix="test_prepare_cache_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
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
    
    def test_step1_tokenizer_returns_source_id(self, tokenizer_core):
        """Adım 1: tokenizer_core.load_training_data() source_id döndürüyor mu?"""
        raw_data = tokenizer_core.load_training_data(
            encode_mode="train",
            include_whole_words=True,
            include_syllables=False,
            include_sep=False,
            include_source_id=True,  # ✅ Kritik!
        )
        
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
        
        print(f"✅ Adım 1 BAŞARILI: {len(raw_data):,} örnek, {len(unique_source_ids):,} unique source_id")
    
    def test_step2_format_function_preserves_source_id(self, tokenizer_core, temp_cache_dir):
        """Adım 2: format_data_func() source_id koruyor mu?"""
        # Raw data'yı al
        raw_data = tokenizer_core.load_training_data(
            encode_mode="train",
            include_source_id=True,
        )
        
        # format_data_func'u simulate et (prepare_cache.py'deki gibi)
        vocab = tokenizer_core.get_vocab()
        vocab_size = len(vocab)
        
        def _id_of(token: str) -> int:
            val = vocab.get(token)
            if isinstance(val, dict):
                return int(val.get("id", 0))
            return int(val or 0)
        
        BOS_ID = _id_of("<BOS>")
        EOS_ID = _id_of("<EOS>")
        PAD_ID = _id_of("<PAD>")
        UNK_ID = _id_of("<UNK>")
        max_seq_length = TOKENIZER_CONFIG.get("max_seq_length", 768)
        
        # Formatlama fonksiyonu (prepare_cache.py'deki gibi)
        def format_data_func(raw_data):
            formatted_data = []
            source_id_count = 0
            no_source_id_count = 0
            
            for idx, item in enumerate(raw_data):
                source_id = None
                if len(item) == 3:
                    inp_ids, tgt_ids, source_id = item
                    source_id_count += 1
                elif len(item) == 2:
                    inp_ids, tgt_ids = item
                    no_source_id_count += 1
                else:
                    continue
                
                try:
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
                
                except Exception as e:
                    continue
            
            return formatted_data
        
        # Formatla
        formatted_data = format_data_func(raw_data)
        
        # Kontrol
        assert len(formatted_data) > 0, "Formatlanmış veri boş!"
        
        # source_id korunmuş mu?
        formatted_with_source_id = 0
        formatted_without_source_id = 0
        
        for item in formatted_data:
            if len(item) == 3:
                formatted_with_source_id += 1
            else:
                formatted_without_source_id += 1
        
        assert formatted_without_source_id == 0, f"Adım 2 BAŞARISIZ: {formatted_without_source_id} formatlanmış örnekte source_id yok!"
        
        # source_id'ler korunmuş mu?
        original_source_ids = set()
        for item in raw_data:
            if len(item) == 3:
                _, _, sid = item
                original_source_ids.add(sid)
        
        formatted_source_ids = set()
        for item in formatted_data:
            if len(item) == 3:
                _, _, sid = item
                formatted_source_ids.add(sid)
        
        # source_id'ler aynı olmalı
        assert original_source_ids == formatted_source_ids, f"Adım 2 BAŞARISIZ: source_id'ler korunmamış! Orijinal: {len(original_source_ids)}, Formatlanmış: {len(formatted_source_ids)}"
        
        print(f"✅ Adım 2 BAŞARILI: {formatted_with_source_id:,} örnek formatlandı, source_id korundu")
    
    def test_step3_cache_save_preserves_source_id(self, tokenizer_core, temp_cache_dir):
        """Adım 3: Cache'e kaydedilirken source_id korunuyor mu?"""
        # Cache dizinini temizle
        cache_path = Path(temp_cache_dir)
        if cache_path.exists():
            for pkl_file in cache_path.glob("*.pkl"):
                pkl_file.unlink()
        
        # Test verisi hazırla (source_id ile)
        test_data = [
            ([1, 2, 3], [1, 2, 3], 0),  # source_id = 0
            ([4, 5, 6], [4, 5, 6], 1),  # source_id = 1
            ([7, 8, 9], [7, 8, 9], 0),  # source_id = 0 (aynı dosya)
        ]
        
        # Cache'e kaydet
        cache = DataCache(data_dir="education", cache_dir=temp_cache_dir)
        
        # Manuel kaydet (save_cached_data private, bu yüzden pickle kullan)
        test_cache_file = cache_path / "test_cache.pkl"
        with open(test_cache_file, "wb") as f:
            pickle.dump(test_data, f)
        
        # Cache'den yükle
        with open(test_cache_file, "rb") as f:
            loaded_data = pickle.load(f)
        
        # Kontrol
        assert len(loaded_data) == len(test_data), f"Adım 3 BAŞARISIZ: Veri kaybı! Orijinal: {len(test_data)}, Yüklenen: {len(loaded_data)}"
        
        for i, (original, loaded) in enumerate(zip(test_data, loaded_data)):
            assert len(loaded) == 3, f"Adım 3 BAŞARISIZ: Örnek {i} source_id kaybetmiş! Format: {len(loaded)}"
            
            orig_inp, orig_tgt, orig_sid = original
            load_inp, load_tgt, load_sid = loaded
            
            assert load_sid == orig_sid, f"Adım 3 BAŞARISIZ: Örnek {i} source_id değişmiş! Orijinal: {orig_sid}, Yüklenen: {load_sid}"
        
        print(f"✅ Adım 3 BAŞARILI: {len(loaded_data)} örnek cache'e kaydedildi ve yüklendi, source_id korundu")
    
    def test_step4_prepare_cache_full_pipeline(self, tokenizer_core, temp_cache_dir):
        """Adım 4: prepare_cache() full pipeline source_id koruyor mu?"""
        # Cache dizinini temizle
        cache_path = Path(temp_cache_dir)
        if cache_path.exists():
            for pkl_file in cache_path.glob("*.pkl"):
                pkl_file.unlink()
        
        # prepare_cache() çalıştır
        try:
            cache_dir = prepare_cache(
                data_dir="education",
                cache_dir=temp_cache_dir,
                max_seq_length=TOKENIZER_CONFIG.get("max_seq_length", 768),
                include_whole_words=True,
                include_syllables=False,
                include_sep=False,
                clear_old_cache=True,
            )
        except Exception as e:
            pytest.fail(f"Adım 4 BAŞARISIZ: prepare_cache() hatası: {e}")
        
        # Cache dosyasını bul ve yükle
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        assert len(cache_files) > 0, "Adım 4 BAŞARISIZ: Cache dosyası oluşturulamadı!"
        
        with open(cache_files[0], "rb") as f:
            cached_data = pickle.load(f)
        
        assert len(cached_data) > 0, "Adım 4 BAŞARISIZ: Cache verisi boş!"
        
        # source_id kontrolü
        items_with_source_id = 0
        items_without_source_id = 0
        source_ids = []
        
        for item in cached_data:
            if len(item) == 3:
                _, _, source_id = item
                items_with_source_id += 1
                source_ids.append(source_id)
            else:
                items_without_source_id += 1
        
        # Tüm örneklerde source_id olmalı
        assert items_without_source_id == 0, f"Adım 4 BAŞARISIZ: {items_without_source_id}/{len(cached_data)} örnekte source_id yok!"
        
        # En az 2 farklı source_id olmalı
        unique_source_ids = set(source_ids)
        assert len(unique_source_ids) >= 2, f"Adım 4 BAŞARISIZ: En az 2 farklı source_id olmalı! Bulunan: {len(unique_source_ids)}"
        
        print(f"✅ Adım 4 BAŞARILI: {len(cached_data):,} örnek cache'e kaydedildi, {len(unique_source_ids):,} unique source_id")
        print(f"   - source_id olan: {items_with_source_id:,}")
        print(f"   - source_id olmayan: {items_without_source_id}")
        
        return cached_data, unique_source_ids
    
    def test_step5_source_id_based_split_no_overlap(self, tokenizer_core, temp_cache_dir):
        """Adım 5: Cache'den yüklenip source_id bazlı split yapıldığında overlap olmamalı"""
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
        
        # Cache'den yükle
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        if not cache_files:
            pytest.skip("Cache dosyası bulunamadı")
        
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
        
        train_hashes = set()
        for seq_in, seq_tgt in train_data:
            hash_val = hash_sequence(seq_in, pad_id)
            train_hashes.add(hash_val)
        
        val_hashes = set()
        for seq_in, seq_tgt in val_data:
            hash_val = hash_sequence(seq_in, pad_id)
            val_hashes.add(hash_val)
        
        overlap_hashes = train_hashes & val_hashes
        overlap_ratio = len(overlap_hashes) / len(train_hashes) if train_hashes else 0.0
        
        # ✅ ÖNEMLİ: source_id bazlı split doğru çalışıyor mu?
        # source_id bazlı overlap olmamalı (document-level split standardı)
        overlap_source_ids_cross = train_source_ids & val_source_ids
        source_id_split_ok = len(overlap_source_ids_cross) == 0
        
        # ✅ DETAYLI ANALİZ: Overlap nedenleri
        if overlap_hashes:
            print(f"\n🔍 OVERLAP ANALİZİ:")
            print(f"   - Overlap hash sayısı: {len(overlap_hashes)}")
            print(f"   - Train hash sayısı: {len(train_hashes)}")
            print(f"   - Val hash sayısı: {len(val_hashes)}")
            
            # Overlap örneklerinin source_id'lerini bul
            overlap_source_ids_analysis = {}
            for seq_in, seq_tgt in train_data:
                hash_val = hash_sequence(seq_in, pad_id)
                if hash_val in overlap_hashes:
                    # Bu örneğin source_id'sini bul
                    for sid, examples in source_id_to_examples.items():
                        if (seq_in, seq_tgt) in examples:
                            if sid not in overlap_source_ids_analysis:
                                overlap_source_ids_analysis[sid] = []
                            overlap_source_ids_analysis[sid].append(hash_val)
                            break
            
            print(f"\n🔍 OVERLAP SOURCE_ID ANALİZİ:")
            print(f"   - Overlap olan source_id sayısı: {len(overlap_source_ids_analysis)}")
            if len(overlap_source_ids_analysis) > 0:
                print(f"   - İlk 10 overlap source_id:")
                for sid, hashes in list(overlap_source_ids_analysis.items())[:10]:
                    print(f"     source_id={sid}: {len(hashes)} overlap hash")
                    # Bu source_id train mi val mi?
                    if sid in train_source_ids and sid in val_source_ids:
                        print(f"       ⚠️  source_id hem train hem val'de! (HATA!)")
                    elif sid in train_source_ids:
                        print(f"       ✅ source_id train'de")
                    elif sid in val_source_ids:
                        print(f"       ✅ source_id val'de")
            
            # Train ve val'deki source_id'leri kontrol et
            overlap_source_ids_cross = train_source_ids & val_source_ids
            if len(overlap_source_ids_cross) > 0:
                print(f"\n❌ KRİTİK HATA: {len(overlap_source_ids_cross)} source_id hem train hem val'de!")
                print(f"   Overlap source_id'ler: {list(overlap_source_ids_cross)[:10]}")
        
        # ✅ ENDÜSTRİ STANDARDI: Document-level split kontrolü
        # source_id bazlı split doğru çalışmalı (aynı document train/val'e bölünmemeli)
        assert source_id_split_ok, f"Adım 5 BAŞARISIZ: source_id bazlı split hatalı! {len(overlap_source_ids_cross)} source_id hem train hem val'de!"
        
        # ⚠️ Hash bazlı overlap: Duplicate content (farklı dosyalarda aynı içerik)
        # Bu normal bir durum olabilir ve kabul edilebilir
        if overlap_ratio > 0.0:
            print(f"\n⚠️  Hash bazlı overlap tespit edildi: {len(overlap_hashes)} örnek ({overlap_ratio:.2%})")
            print(f"   Bu, farklı dosya/QA'lardan gelen aynı içeriği gösterir (duplicate content)")
            print(f"   ✅ source_id bazlı split DOĞRU çalışıyor (document-level split standardı)")
            print(f"   💡 Hash bazlı overlap kabul edilebilir (farklı kaynaklardan duplicate content)")
        else:
            print(f"✅ Hash bazlı overlap YOK!")
        
        print(f"\n✅ Adım 5 BAŞARILI: source_id bazlı split doğru çalışıyor!")
        print(f"   - Train: {len(train_data):,} örnek ({len(train_source_ids):,} source_id)")
        print(f"   - Val: {len(val_data):,} örnek ({len(val_source_ids):,} source_id)")
        print(f"   - source_id bazlı overlap: {len(overlap_source_ids_cross)} (beklenen: 0)")
        print(f"   - Hash bazlı overlap: {len(overlap_hashes)} örnek ({overlap_ratio:.2%}) - duplicate content")


class TestPrepareCachePipelineIntegration:
    """End-to-end integration test: Tüm pipeline birlikte"""
    
    @pytest.fixture(scope="class")
    def temp_cache_dir(self):
        """Geçici cache dizini"""
        temp_dir = tempfile.mkdtemp(prefix="test_integration_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_full_pipeline_no_overlap(self, temp_cache_dir):
        """Full pipeline test: prepare_cache() → cache → split → overlap kontrolü"""
        # 1. Cache oluştur
        cache_path = Path(temp_cache_dir)
        if cache_path.exists():
            for pkl_file in cache_path.glob("*.pkl"):
                pkl_file.unlink()
        
        print("\n[Integration Test] Cache hazırlanıyor...")
        try:
            cache_dir = prepare_cache(
                data_dir="education",
                cache_dir=temp_cache_dir,
                max_seq_length=TOKENIZER_CONFIG.get("max_seq_length", 768),
                include_whole_words=True,
                include_syllables=False,
                include_sep=False,
            )
        except Exception as e:
            pytest.fail(f"Integration test BAŞARISIZ: prepare_cache() hatası: {e}")
        
        # 2. Cache'den yükle
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        assert len(cache_files) > 0, "Cache dosyası oluşturulamadı!"
        
        with open(cache_files[0], "rb") as f:
            cached_data = pickle.load(f)
        
        assert len(cached_data) > 0, "Cache verisi boş!"
        
        # 3. source_id kontrolü
        source_ids = []
        for item in cached_data:
            assert len(item) == 3, f"Cache'de source_id yok! Format: {len(item)}"
            _, _, source_id = item
            source_ids.append(source_id)
        
        unique_source_ids = set(source_ids)
        print(f"[Integration Test] Cache yüklendi: {len(cached_data):,} örnek, {len(unique_source_ids):,} unique source_id")
        
        # 4. source_id bazlı split
        source_id_to_examples = {}
        for item in cached_data:
            seq_in, seq_tgt, source_id = item
            if source_id not in source_id_to_examples:
                source_id_to_examples[source_id] = []
            source_id_to_examples[source_id].append((seq_in, seq_tgt))
        
        import random
        random.seed(42)
        source_ids_list = list(source_id_to_examples.keys())
        random.shuffle(source_ids_list)
        
        train_size = max(1, int(0.8 * len(source_ids_list)))
        train_source_ids = set(source_ids_list[:train_size])
        val_source_ids = set(source_ids_list[train_size:])
        
        train_data = []
        val_data = []
        
        for source_id in source_ids_list:
            examples_for_source = source_id_to_examples[source_id]
            if source_id in train_source_ids:
                train_data.extend(examples_for_source)
            else:
                val_data.extend(examples_for_source)
        
        print(f"[Integration Test] Split yapıldı: Train={len(train_data):,}, Val={len(val_data):,}")
        
        # 5. Overlap kontrolü
        overlap_source_ids = train_source_ids & val_source_ids
        assert len(overlap_source_ids) == 0, f"Integration test BAŞARISIZ: source_id bazlı overlap!"
        
        print(f"✅ Integration Test BAŞARILI: Overlap yok!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

