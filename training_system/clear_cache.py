# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: clear_cache.py
Modül: training_system
Görev: Cache Temizleme Scripti - Eski cache dosyalarını temizler (yeni format için).
       Cache dizinini temizleme ve cache yönetimi işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache temizleme scripti)
- Design Patterns: Script Pattern (standalone utility script)
- Endüstri Standartları: Cache cleanup workflow

KULLANIM:
- Eski cache dosyalarını temizlemek için
- Cache dizinini yönetmek için
- Standalone script olarak çalıştırılır

BAĞIMLILIKLAR:
- DataCache: Cache yönetimi

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import sys
import os
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_system.data_cache import DataCache
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ClearCache")


def clear_cache(cache_dir: str = ".cache/preprocessed_data"):
    """Cache'i temizle"""
    logger.info("="*60)
    logger.info("CACHE TEMIZLEME")
    logger.info("="*60)
    
    cache = DataCache(
        data_dir="education",  # Dummy, sadece cache temizleme için
        cache_dir=cache_dir,
        cache_enabled=True
    )
    
    logger.info(f"Cache dizini: {cache_dir}")
    
    try:
        deleted_count = cache.clear_cache()
        logger.info(f"\n{'='*60}")
        if deleted_count > 0:
            logger.info(f"[OK] {deleted_count} cache dosyasi silindi")
        else:
            logger.info("[INFO] Silinecek cache dosyasi yok (zaten temiz)")
        logger.info("="*60)
        return deleted_count
    except Exception as e:
        logger.error(f"[ERROR] Cache temizleme hatasi: {e}", exc_info=True)
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache temizleme scripti")
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default=".cache/preprocessed_data", 
        help="Cache dizini"
    )
    
    args = parser.parse_args()
    
    deleted = clear_cache(cache_dir=args.cache_dir)
    sys.exit(0 if deleted >= 0 else 1)






