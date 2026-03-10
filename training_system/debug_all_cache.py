#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: debug_all_cache.py
Modül: training_system
Görev: Comprehensive Cache Debug Script - Hem BPE corpus cache hem de preprocessed
       data cache'i kontrol eder. Cache dosyalarının varlığını, cache key'lerini,
       hash hesaplamalarını kontrol eder ve sorunları tespit eder.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache debug scripti)
- Design Patterns: Script Pattern (standalone debugging tool)
- Endüstri Standartları: Cache debugging ve troubleshooting

KULLANIM:
- Cache sorunlarını tespit etmek için
- BPE corpus cache ve preprocessed data cache kontrolü için
- Debugging ve troubleshooting için

BAĞIMLILIKLAR:
- DataCache: Cache yönetimi
- TokenizerCore: Tokenization işlemleri
- Config modülleri: Yapılandırma yönetimi

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""
import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from training_system.data_cache import DataCache
from tokenizer_management.config import get_bpe_detailed_config, get_turkish_config, BPE_CONFIG
from tokenizer_management.core.tokenizer_core import TokenizerCore

def debug_bpe_cache():
    """BPE corpus cache debug"""
    print("\n" + "="*80)
    print("🔵 BPE CORPUS CACHE DEBUG")
    print("="*80)
    
    # Config parametrelerini al
    bpe_config = get_bpe_detailed_config()
    turkish_config = get_turkish_config()
    merged_config = {**bpe_config, **turkish_config}
    
    data_dir = "education"
    cache_dir = ".cache/bpe_training"
    
    include_whole_words = merged_config.get("include_whole_words", True)
    include_syllables = merged_config.get("include_syllables", False)
    include_sep = merged_config.get("include_sep", True)
    
    print(f"\n📋 Config Parametreleri:")
    print(f"  data_dir: {data_dir}")
    print(f"  cache_dir: {cache_dir}")
    print(f"  include_whole_words: {include_whole_words}")
    print(f"  include_syllables: {include_syllables}")
    print(f"  include_sep: {include_sep}")
    
    # DataCache oluştur
    cache = DataCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        cache_enabled=True
    )
    
    print(f"\n📁 Dizin Kontrolleri:")
    print(f"  data_dir var mı: {Path(data_dir).exists()}")
    print(f"  cache_dir var mı: {Path(cache_dir).exists()}")
    
    # Data hash hesapla
    print(f"\n🔐 Data Hash:")
    data_hash = cache._get_data_dir_hash()
    if data_hash:
        print(f"  Data hash: {data_hash}")
    else:
        print("  ❌ Data hash hesaplanamadı!")
        return
    
    # Cache key hesapla
    cache_key = cache._get_corpus_cache_key(
        include_whole_words,
        include_syllables,
        include_sep,
        data_hash
    )
    
    print(f"\n🔑 Cache Key:")
    print(f"  Cache key: {cache_key}")
    
    # Beklenen cache dosyası
    expected_cache = cache._get_corpus_cache_path(cache_key)
    print(f"  Beklenen cache: {expected_cache.name}")
    print(f"  Beklenen cache var mı: {expected_cache.exists()}")
    
    # Mevcut cache dosyalarını listele
    print(f"\n📦 Mevcut Cache Dosyaları:")
    cache_path = Path(cache_dir)
    if cache_path.exists():
        cache_files = list(cache_path.glob("bpe_corpus_*.pkl"))
        if cache_files:
            print(f"  Toplam {len(cache_files)} cache dosyası bulundu:")
            for cache_file in cache_files:
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                key_from_filename = cache_file.stem.replace("bpe_corpus_", "")
                match = "[OK] EŞLEŞİYOR" if key_from_filename == cache_key else "❌ UYUŞMUYOR"
                print(f"    - {cache_file.name}")
                print(f"      Key: {key_from_filename} {match}")
                print(f"      Size: {size_mb:.2f} MB")
        else:
            print("  ❌ Cache dizininde hiç cache dosyası yok!")
    else:
        print(f"  ❌ Cache dizini mevcut değil: {cache_dir}")

def debug_preprocessed_cache():
    """Preprocessed data cache debug"""
    print("\n" + "="*80)
    print("🟢 PREPROCESSED DATA CACHE DEBUG")
    print("="*80)
    
    data_dir = "education"
    cache_dir = ".cache/preprocessed_data"
    max_seq_length = 512
    include_whole_words = True
    include_syllables = False
    include_sep = False
    alignment_format = "autoregressive_v2"
    
    print(f"\n📋 Config Parametreleri:")
    print(f"  data_dir: {data_dir}")
    print(f"  cache_dir: {cache_dir}")
    print(f"  max_seq_length: {max_seq_length}")
    print(f"  include_whole_words: {include_whole_words}")
    print(f"  include_syllables: {include_syllables}")
    print(f"  include_sep: {include_sep}")
    print(f"  alignment_format: {alignment_format}")
    
    # DataCache oluştur
    cache = DataCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        cache_enabled=True
    )
    
    print(f"\n📁 Dizin Kontrolleri:")
    print(f"  data_dir var mı: {Path(data_dir).exists()}")
    print(f"  cache_dir var mı: {Path(cache_dir).exists()}")
    
    # TokenizerCore oluştur (vocab hash için gerekli)
    try:
        vocab_path = BPE_CONFIG.get("vocab_file", "data/vocab_lib/vocab.json")
        merges_path = BPE_CONFIG.get("merges_file", "data/merges_lib/merges.txt")
        
        tokenizer_core = TokenizerCore({
            "data_dir": data_dir,
            "vocab_path": vocab_path,
            "merges_path": merges_path,
            "bpe_rebuild": False,
        })
        tokenizer_core.finalize_vocab()
        
        # Vocab hash'i manuel hesapla (data_cache.py ile aynı mantık)
        import hashlib
        vocab = tokenizer_core.get_vocab()
        vocab_items = []
        for token, data in vocab.items():
            if isinstance(data, dict):
                token_id = data.get('id', 0)
                vocab_items.append(f"{token}:{token_id}")
            elif isinstance(data, int):
                vocab_items.append(f"{token}:{data}")
        vocab_str = "|".join(sorted(vocab_items))
        vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:16]
        
        print(f"\n🔐 Hash Bilgileri:")
        data_hash = cache._get_data_dir_hash()
        if data_hash:
            print(f"  Data hash: {data_hash}")
        else:
            print("  ❌ Data hash hesaplanamadı!")
            return
        
        print(f"  Vocab hash: {vocab_hash}")
        
        # Cache key hesapla
        cache_key = cache._get_cache_key(
            encode_mode="train",
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
            max_seq_length=max_seq_length,
            vocab_hash=vocab_hash,
            alignment_format=alignment_format
        )
        
        print(f"\n🔑 Cache Key:")
        print(f"  Cache key: {cache_key}")
        
        # Beklenen cache dosyası
        expected_cache = cache._get_cache_path(cache_key, data_hash)
        print(f"  Beklenen cache: {expected_cache.name}")
        print(f"  Beklenen cache var mı: {expected_cache.exists()}")
        
        # Mevcut cache dosyalarını listele
        print(f"\n📦 Mevcut Cache Dosyaları:")
        cache_path = Path(cache_dir)
        if cache_path.exists():
            cache_files = list(cache_path.glob("cached_data_*.pkl"))
            if cache_files:
                print(f"  Toplam {len(cache_files)} cache dosyası bulundu:")
                for cache_file in cache_files[:5]:  # İlk 5'ini göster
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    # Cache dosya adından key'i çıkar
                    parts = cache_file.stem.replace("cached_data_", "").split("_")
                    if len(parts) >= 2:
                        file_key = parts[0]
                        file_data_hash = parts[-1]
                        match = "[OK] EŞLEŞİYOR" if (file_key == cache_key and file_data_hash == data_hash) else "❌ UYUŞMUYOR"
                        print(f"    - {cache_file.name}")
                        print(f"      Key: {file_key} (data_hash: {file_data_hash}) {match}")
                        print(f"      Size: {size_mb:.2f} MB")
                if len(cache_files) > 5:
                    print(f"    ... ve {len(cache_files) - 5} dosya daha")
            else:
                print("  ❌ Cache dizininde hiç cache dosyası yok!")
        else:
            print(f"  ❌ Cache dizini mevcut değil: {cache_dir}")
            
    except Exception as e:
        print(f"  ❌ TokenizerCore yüklenemedi: {e}")
        print("  (Vocab hash hesaplanamadı, cache key kontrol edilemiyor)")

def main():
    """Ana fonksiyon"""
    print("="*80)
    print("COMPREHENSIVE CACHE DEBUG")
    print("="*80)
    print("\nBu script iki cache tipini kontrol eder:")
    print("  1. BPE Corpus Cache (.cache/bpe_training)")
    print("  2. Preprocessed Data Cache (.cache/preprocessed_data)")
    
    try:
        debug_bpe_cache()
        debug_preprocessed_cache()
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("DEBUG TAMAMLANDI")
    print("="*80)
    return 0

if __name__ == "__main__":
    exit(main())

