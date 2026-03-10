# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: prepare_bpe_cache.py
Modül: tokenizer_management
Görev: BPE Training Cache Preparation Script - BPE training için corpus cache'ini
       hazırlar. train_bpe.py çalıştırılmadan önce bu script ile cache hazırlanabilir.
       Veri yükleme ve corpus oluşturma işlemlerini cache'ler.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache hazırlama scripti)
- Design Patterns: Script Pattern (standalone utility script)
- Endüstri Standartları: Cache preparation workflow

KULLANIM:
- BPE training öncesi cache hazırlama için
- Veri yükleme süresini azaltmak için
- train_bpe.py ile birlikte kullanılır

BAĞIMLILIKLAR:
- DataCache: Cache yönetimi
- DataLoaderManager: Veri yükleme işlemleri
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
import logging
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from training_system.data_cache import DataCache
from data_loader_management.data_loader_manager import DataLoaderManager, DataLoaderConfig, LoadMode
from tokenizer_management.config import get_bpe_detailed_config, get_turkish_config

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrepareBPECache")


def load_training_data(data_dir: str):
    """
    Training verisini yükle (QA format + Raw text chunks)
    
    Returns:
        (qa_data, raw_data): QA çiftleri ve raw text chunks
    """
    logger.info(f"Veri yükleniyor: {data_dir}")
    
    # 1. QA format verilerini yükle (JSON dosyaları)
    qa_loader = DataLoaderManager(DataLoaderConfig(
        data_dir=Path(data_dir),
        mode=LoadMode.QA_TRAIN
    ))
    qa_data = qa_loader.load()
    logger.info(f" QA format: {len(qa_data)} çift yüklendi")
    
    # 2. Raw text chunks yükle (DOCX/TXT dosyaları)
    raw_loader = DataLoaderManager(DataLoaderConfig(
        data_dir=Path(data_dir),
        mode=LoadMode.TEXT_INFER  # veya LoadMode.RAW_TEXT
    ))
    raw_data = raw_loader.load()
    logger.info(f" Raw text: {len(raw_data)} chunk yüklendi")
    
    return qa_data, raw_data


def create_corpus(qa_data: list, raw_data: list) -> list[str]:
    """
    Hibrit corpus oluştur (QA format + Raw text chunks)
    
    Args:
        qa_data: QA çiftleri [(q, a), ...]
        raw_data: Raw text chunks [text, ...]
    
    Returns:
        corpus: Tüm metinlerin listesi
    """
    corpus = []
    
    # QA format verilerini ekle
    for q, a in qa_data:
        if q and a:  # Boş olmayan QA çiftleri
            corpus.extend([q, a])
        elif a:  # Sadece answer varsa
            corpus.append(a)
    
    # Raw text chunks ekle
    corpus.extend(raw_data)
    
    logger.info(f" Corpus oluşturuldu: {len(corpus)} metin")
    return corpus


def prepare_bpe_cache(
    data_dir: str = "education",
    cache_dir: str = ".cache/bpe_training",
    include_whole_words: bool = None,
    include_syllables: bool = None,
    include_sep: bool = None,
):
    """
    BPE training corpus cache'ini hazırla
    
    Args:
        data_dir: Eğitim verisi dizini
        cache_dir: Cache dizini
        include_whole_words: Whole words flag (None ise config'ten alınır)
        include_syllables: Syllables flag (None ise config'ten alınır)
        include_sep: SEP flag (None ise config'ten alınır)
    """
    logger.info("="*80)
    logger.info("BPE TRAINING CACHE HAZIRLAMA BAŞLIYOR")
    logger.info("="*80)
    
    # Config'ten parametreleri al (train_bpe.py ile aynı mantık)
    bpe_config = get_bpe_detailed_config()
    turkish_config = get_turkish_config()
    
    # Config'leri birleştir (train_bpe.py ile aynı)
    merged_config = {
        **bpe_config,
        **turkish_config,
    }
    
    # Parametreleri config'ten al (train_bpe.py ile aynı mantık)
    # Eğer None ise config'ten al, yoksa verilen değeri kullan
    if include_whole_words is None:
        include_whole_words = merged_config.get("include_whole_words", True)
    if include_syllables is None:
        include_syllables = merged_config.get("include_syllables", False)
    if include_sep is None:
        include_sep = merged_config.get("include_sep", True)
    
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Include whole words: {include_whole_words}")
    logger.info(f"Include syllables: {include_syllables}")
    logger.info(f"Include sep: {include_sep}")
    logger.info("")
    
    # DataCache oluştur
    cache = DataCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        cache_enabled=True
    )
    
    # Corpus oluşturma fonksiyonu
    def create_corpus_func():
        """Cache yoksa corpus oluştur"""
        logger.info("\n[1] Veri yükleniyor...")
        qa_data, raw_data = load_training_data(data_dir)
        
        if not qa_data and not raw_data:
            raise RuntimeError(f"Veri dizininde hiç veri bulunamadı: {data_dir}")
        
        logger.info("\n[2] Corpus oluşturuluyor...")
        corpus = create_corpus(qa_data, raw_data)
        
        if not corpus:
            raise RuntimeError("Corpus oluşturulamadı!")
        
        logger.info(f"✅ {len(corpus):,} metin ile corpus hazırlandı")
        return corpus
    
    # Cache'den yükle veya oluştur
    logger.info("\n[3] Cache kontrolü yapılıyor...")
    corpus, from_cache = cache.get_or_create_corpus(
        process_func=create_corpus_func,
        include_whole_words=include_whole_words,
        include_syllables=include_syllables,
        include_sep=include_sep,
    )
    
    if from_cache:
        logger.info("✅ Corpus cache'den yüklendi (zaten mevcut)")
    else:
        logger.info("✅ Corpus oluşturuldu ve cache'e kaydedildi")
    
    logger.info(f"✅ Toplam {len(corpus):,} metin hazır")
    
    # Cache dosyalarını listele
    logger.info("\n[4] Cache dosyaları:")
    cache_path = Path(cache_dir)
    if cache_path.exists():
        cache_files = list(cache_path.glob("bpe_corpus_*.pkl"))
        for cache_file in cache_files:
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"   📁 {cache_file.name} ({file_size_mb:.2f} MB)")
    
    logger.info("\n" + "="*80)
    logger.info("✅ BPE TRAINING CACHE HAZIRLAMA TAMAMLANDI!")
    logger.info("="*80)
    logger.info("\n📤 Sonraki adım:")
    logger.info("   1. Artık 'python tokenizer_management/train_bpe.py' çalıştırabilirsiniz")
    logger.info("   2. train_bpe.py cache'den corpus'u otomatik yükleyecek")
    logger.info("   3. BPE training daha hızlı başlayacak")
    
    return cache_dir


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BPE Training Cache Hazırlama Scripti")
    parser.add_argument("--data-dir", type=str, default="education", help="Eğitim verisi dizini")
    parser.add_argument("--cache-dir", type=str, default=".cache/bpe_training", help="Cache dizini")
    parser.add_argument("--include-whole-words", type=str, default=None, choices=['true', 'false'], help="Include whole words (true/false, yoksa config'ten alınır)")
    parser.add_argument("--include-syllables", type=str, default=None, choices=['true', 'false'], help="Include syllables (true/false, yoksa config'ten alınır)")
    parser.add_argument("--include-sep", type=str, default=None, choices=['true', 'false'], help="Include SEP token (true/false, yoksa config'ten alınır)")
    
    args = parser.parse_args()
    
    # String'den bool'a çevir (None ise None kalır)
    include_whole_words = (args.include_whole_words.lower() == 'true') if args.include_whole_words else None
    include_syllables = (args.include_syllables.lower() == 'true') if args.include_syllables else None
    include_sep = (args.include_sep.lower() == 'true') if args.include_sep else None
    
    try:
        cache_dir = prepare_bpe_cache(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
        )
        print(f"\n✅ Başarılı! Cache dizini: {cache_dir}")
        return 0
    except Exception as e:
        logger.error(f"❌ Hata: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

