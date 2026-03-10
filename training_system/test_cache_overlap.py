# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ - CACHE OVERLAP TEST
================================================================================

Dosya: test_cache_overlap.py
Modül: training_system
Görev: Cache içindeki source_id'leri kontrol et ve overlap simülasyonu yap

KULLANIM:
    python training_system/test_cache_overlap.py
    
Bu script:
    1. Cache'i yükler
    2. source_id'lerin var olup olmadığını kontrol eder
    3. source_id bazlı split simülasyonu yapar
    4. Overlap olup olmadığını rapor eder

================================================================================
"""

import os
import sys
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Set

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("CacheOverlapTest")


def hash_sequence_full(seq, pad_id=0):
    """Tam token dizisini hash'le (PAD'ler hariç)"""
    if isinstance(seq, list):
        # PAD'leri temizle
        cleaned_seq = [t for t in seq if t != pad_id]
    else:
        # Liste değilse direkt kullan
        cleaned_seq = seq
    
    # Tam diziyi kullan
    seq_str = str(cleaned_seq)
    return hashlib.sha256(seq_str.encode()).hexdigest()


def test_cache_overlap():
    """Cache'deki source_id'leri kontrol et ve overlap simülasyonu yap"""
    logger.info("="*80)
    logger.info("🔍 CACHE OVERLAP KONTROLÜ")
    logger.info("="*80)
    
    # Cache dizini
    cache_dir = Path(".cache/preprocessed_data")
    
    if not cache_dir.exists():
        logger.error(f"❌ Cache dizini bulunamadı: {cache_dir}")
        logger.info(f"\n💡 Önce cache oluştur:")
        logger.info(f"   python training_system/prepare_cache.py")
        return False
    
    # Cache dosyalarını bul
    cache_files = list(cache_dir.glob("cached_data_*.pkl"))
    
    if not cache_files:
        logger.error(f"❌ Cache dosyası bulunamadı: {cache_dir}")
        logger.info(f"\n💡 Önce cache oluştur:")
        logger.info(f"   python training_system/prepare_cache.py")
        return False
    
    logger.info(f"\n📁 Cache dosyaları bulundu: {len(cache_files)} dosya")
    for cf in cache_files:
        size_mb = cf.stat().st_size / (1024 * 1024)
        logger.info(f"   - {cf.name} ({size_mb:.2f} MB)")
    
    # En son cache dosyasını yükle (en büyük dosya)
    cache_file = max(cache_files, key=lambda p: p.stat().st_size)
    logger.info(f"\n📂 Test edilecek cache: {cache_file.name}")
    
    # Cache'i yükle
    logger.info(f"\n[1] Cache yükleniyor...")
    try:
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        logger.info(f"[OK] Cache yüklendi: {len(cached_data):,} örnek")
    except Exception as e:
        logger.error(f"❌ Cache yükleme hatası: {e}")
        return False
    
    # Veri formatını kontrol et
    logger.info(f"\n[2] Veri formatı kontrol ediliyor...")
    
    if not cached_data:
        logger.error(f"❌ Cache boş!")
        return False
    
    first_item = cached_data[0]
    has_source_id = False
    
    logger.info(f"   İlk örnek tipi: {type(first_item)}")
    logger.info(f"   İlk örnek uzunluğu: {len(first_item)}")
    
    if isinstance(first_item, (tuple, list)):
        if len(first_item) == 3:
            has_source_id = True
            logger.info(f"   ✅ Format: (inp, tgt, source_id) - source_id VAR!")
            inp, tgt, src_id = first_item
            logger.info(f"   Örnek source_id: {src_id}")
            logger.info(f"   Input uzunluk: {len(inp)}")
            logger.info(f"   Target uzunluk: {len(tgt)}")
        elif len(first_item) == 2:
            logger.error(f"   ❌ Format: (inp, tgt) - source_id YOK!")
            logger.error(f"   Bu cache ESKI format! Yeniden oluşturulmalı!")
            has_source_id = False
            inp, tgt = first_item
            logger.info(f"   Input uzunluk: {len(inp)}")
            logger.info(f"   Target uzunluk: {len(tgt)}")
        else:
            logger.error(f"   ❌ Bilinmeyen format: {len(first_item)} eleman")
            return False
    else:
        logger.error(f"   ❌ Beklenmeyen tip: {type(first_item)}")
        return False
    
    if not has_source_id:
        logger.error(f"\n❌ ❌ ❌ CACHE ESKİ FORMAT! ❌ ❌ ❌")
        logger.error(f"   source_id YOK - overlap önlenemez!")
        logger.error(f"\n💡 Çözüm:")
        logger.error(f"   1. Eski cache'i temizle:")
        logger.error(f"      python training_system/prepare_cache.py")
        logger.error(f"   2. Bu script'i tekrar çalıştır")
        return False
    
    # source_id'leri analiz et
    logger.info(f"\n[3] source_id analizi...")
    
    source_ids = set()
    source_id_to_count = {}
    
    for item in cached_data:
        if len(item) == 3:
            inp, tgt, src_id = item
            source_ids.add(src_id)
            if src_id not in source_id_to_count:
                source_id_to_count[src_id] = 0
            source_id_to_count[src_id] += 1
    
    logger.info(f"[OK] Unique source_id sayısı: {len(source_ids)}")
    logger.info(f"[OK] Toplam örnek: {len(cached_data):,}")
    logger.info(f"[OK] Ortalama örnek/source_id: {len(cached_data)/len(source_ids):.1f}")
    
    # En çok örneğe sahip source_id'ler
    top_sources = sorted(source_id_to_count.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"\n📊 En çok örneğe sahip 5 source_id:")
    for i, (src_id, count) in enumerate(top_sources):
        logger.info(f"   [{i+1}] source_id={src_id}: {count} örnek")
    
    # Split simülasyonu
    logger.info(f"\n[4] Train/Val split simülasyonu (80/20)...")
    
    import random
    random.seed(42)
    
    # source_id'lere göre grupla
    source_id_to_examples = {}
    for item in cached_data:
        if len(item) == 3:
            inp, tgt, src_id = item
            if src_id not in source_id_to_examples:
                source_id_to_examples[src_id] = []
            source_id_to_examples[src_id].append((inp, tgt))
    
    # source_id'leri shuffle et
    source_ids_list = list(source_id_to_examples.keys())
    random.shuffle(source_ids_list)
    
    # Split
    train_size = int(0.8 * len(source_ids_list))
    train_source_ids = set(source_ids_list[:train_size])
    val_source_ids = set(source_ids_list[train_size:])
    
    # Örnekleri ayır
    train_examples = []
    val_examples = []
    
    for src_id, examples in source_id_to_examples.items():
        if src_id in train_source_ids:
            train_examples.extend(examples)
        else:
            val_examples.extend(examples)
    
    logger.info(f"[OK] Split tamamlandı:")
    logger.info(f"   Train source_id: {len(train_source_ids)}")
    logger.info(f"   Val source_id: {len(val_source_ids)}")
    logger.info(f"   Train örnekler: {len(train_examples):,}")
    logger.info(f"   Val örnekler: {len(val_examples):,}")
    
    # Overlap kontrolü
    logger.info(f"\n[5] Overlap kontrolü...")
    
    # Hash'leri oluştur (PAD=0 varsay)
    PAD_ID = 0
    
    train_hashes = set()
    for inp, tgt in train_examples:
        h = hash_sequence_full(inp, PAD_ID)
        train_hashes.add(h)
    
    val_hashes = set()
    for inp, tgt in val_examples:
        h = hash_sequence_full(inp, PAD_ID)
        val_hashes.add(h)
    
    # Overlap
    overlap_hashes = train_hashes & val_hashes
    overlap_count = len(overlap_hashes)
    overlap_ratio = overlap_count / len(train_hashes) if train_hashes else 0.0
    
    # Sonuçlar
    logger.info(f"\n" + "="*80)
    logger.info("📊 OVERLAP TEST SONUÇLARI")
    logger.info("="*80)
    logger.info(f"\n📊 İstatistikler:")
    logger.info(f"   Toplam source_id: {len(source_ids)}")
    logger.info(f"   Train source_id: {len(train_source_ids)}")
    logger.info(f"   Val source_id: {len(val_source_ids)}")
    logger.info(f"   Train unique hash: {len(train_hashes):,}")
    logger.info(f"   Val unique hash: {len(val_hashes):,}")
    logger.info(f"   Overlap: {overlap_count:,}")
    
    if overlap_count == 0:
        logger.info(f"\n✅ ✅ ✅ MÜKEMMEL! ✅ ✅ ✅")
        logger.info(f"   Overlap: %0")
        logger.info(f"   source_id bazlı split başarılı!")
        logger.info(f"   Aynı dokümanın chunk'ları aynı set'te!")
        logger.info(f"   Endüstri standardına uygun!")
        success = True
    else:
        logger.error(f"\n❌ ❌ ❌ OVERLAP TESPİT EDİLDİ! ❌ ❌ ❌")
        logger.error(f"   Overlap sayısı: {overlap_count:,}")
        logger.error(f"   Overlap oranı: {overlap_ratio:.2%}")
        logger.error(f"   ")
        logger.error(f"   🚨 SORUN: source_id bazlı split overlap önleyemedi!")
        logger.error(f"   🚨 Muhtemelen: Aynı source_id farklı içeriklere sahip")
        logger.error(f"   ")
        logger.error(f"   💡 Çözüm:")
        logger.error(f"      1. tokenizer_core.load_training_data() source_id doğru üretiyor mu?")
        logger.error(f"      2. Her chunk benzersiz source_id alıyor mu?")
        logger.error(f"      3. QA çiftleri için ayrı source_id var mı?")
        
        # İlk 3 overlap'i detaylı göster
        logger.info(f"\n📋 İlk 3 Overlap Örneği:")
        for i, h in enumerate(list(overlap_hashes)[:3]):
            logger.info(f"   [{i+1}] Hash: {h[:16]}...")
        
        success = False
    
    # Endüstri karşılaştırması
    logger.info(f"\n" + "="*80)
    logger.info("📊 ENDÜSTRİ STANDARTLARI")
    logger.info("="*80)
    logger.info(f"   GPT-3: Document-level split → Overlap %0 ✅")
    logger.info(f"   BERT: Document-level split → Overlap %0 ✅")
    logger.info(f"   LLaMA: Exact deduplication → Overlap %0 ✅")
    logger.info(f"   Cevahir-AI: source_id split → Overlap {overlap_ratio:.2%} {' ✅' if overlap_count == 0 else ' ❌'}")
    
    logger.info(f"\n" + "="*80)
    if success:
        logger.info("✅ TEST BAŞARILI - Cache overlap'siz, eğitime hazır!")
    else:
        logger.error("❌ TEST BAŞARISIZ - Cache'de sorun var, yeniden oluştur!")
    logger.info("="*80)
    
    return success


if __name__ == "__main__":
    try:
        success = test_cache_overlap()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

