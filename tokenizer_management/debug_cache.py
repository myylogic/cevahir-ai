#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: debug_cache.py
Modül: tokenizer_management
Görev: BPE Cache Debug Script - Cache sistemini debug etmek için yardımcı script.
       Cache dosyalarının varlığını, cache key'lerini, hash hesaplamalarını
       kontrol eder ve sorunları tespit eder.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache debug scripti)
- Design Patterns: Script Pattern (standalone debugging tool)
- Endüstri Standartları: Cache debugging ve troubleshooting

KULLANIM:
- Cache sorunlarını tespit etmek için
- Cache key uyumluluğunu kontrol etmek için
- Debugging ve troubleshooting için

BAĞIMLILIKLAR:
- DataCache: Cache yönetimi
- get_bpe_detailed_config, get_turkish_config: Config yönetimi

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
from tokenizer_management.config import get_bpe_detailed_config, get_turkish_config

def debug_cache():
    """Cache durumunu debug et"""
    print("="*80)
    print("BPE CACHE DEBUG")
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
    print(f"  cache_dir path: {Path(cache_dir).absolute()}")
    
    # Data hash hesapla
    print(f"\n🔐 Data Hash:")
    print(f"  NOT: Hash artık sadece dosya adı + boyut kullanıyor (mtime YOK)")
    print(f"  Bu sayede Colab/local arasında cache taşınabilir")
    data_hash = cache._get_data_dir_hash()
    if data_hash:
        print(f"  Data hash: {data_hash}")
        print(f"  Data hash length: {len(data_hash)}")
        
        # Hash'in nasıl hesaplandığını göster (ilk birkaç dosya)
        if Path(data_dir).exists():
            sample_files = []
            for ext in [".json", ".txt", ".docx"]:
                # rglob generator döndürür, list() ile sarmalayalım
                file_list = list(Path(data_dir).rglob(f"*{ext}"))
                for file_path in file_list[:3]:  # İlk 3 dosya
                    if file_path.is_file():
                        stat = file_path.stat()
                        try:
                            rel_path = os.path.relpath(file_path, Path(data_dir))
                        except ValueError:
                            rel_path = file_path.name
                        sample_files.append(f"{rel_path}:{stat.st_size}")
                        if len(sample_files) >= 3:
                            break
                if len(sample_files) >= 3:
                    break
            
            if sample_files:
                print(f"  Örnek hash girdileri (ilk 3 dosya):")
                for sf in sample_files:
                    print(f"    - {sf}")
    else:
        print("  ❌ Data hash hesaplanamadı!")
    
    # Cache key hesapla
    if data_hash:
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
        print(f"  Beklenen cache: {expected_cache}")
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
                print(f"    - {cache_file.name}")
                print(f"      Key: {key_from_filename}")
                print(f"      Size: {size_mb:.2f} MB")
                
                if data_hash and cache_key:
                    if key_from_filename == cache_key:
                        print(f"      ✅ BU CACHE KEY EŞLEŞİYOR!")
                    else:
                        print(f"      ❌ Cache key uyuşmuyor")
                        print(f"         Beklenen: {cache_key}")
                        print(f"         Mevcut:   {key_from_filename}")
                        print(f"      💡 ÇÖZÜM: Mevcut cache eski hash yöntemiyle oluşturulmuş")
                        print(f"         Cache'i yeniden oluşturun:")
                        print(f"         python tokenizer_management/prepare_bpe_cache.py")
        else:
            print("  ❌ Cache dizininde hiç cache dosyası yok!")
    else:
        print(f"  ❌ Cache dizini mevcut değil: {cache_dir}")
    
    # Path normalization testi
    print(f"\n🛤️  Path Normalization:")
    cwd = os.getcwd()
    print(f"  Current working directory: {cwd}")
    data_dir_abs = os.path.abspath(data_dir)
    print(f"  data_dir absolute: {data_dir_abs}")
    try:
        data_dir_rel = os.path.relpath(data_dir_abs, cwd)
        print(f"  data_dir relative: {data_dir_rel}")
    except ValueError as e:
        print(f"  ❌ Relative path hesaplanamadı: {e}")
    
    print("\n" + "="*80)
    print("DEBUG TAMAMLANDI")
    print("="*80)

if __name__ == "__main__":
    debug_cache()

