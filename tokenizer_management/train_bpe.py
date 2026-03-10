# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: train_bpe.py
Modül: tokenizer_management
Görev: BPE (Byte Pair Encoding) Training Script - Model eğitiminden ÖNCE
       çalıştırılmalıdır. Büyük veri seti ile BPE training yapar ve vocab/merges
       dosyalarını oluşturur. Cache sistemi kullanarak hızlı corpus hazırlama
       sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (BPE training script)
- Design Patterns: Script Pattern (standalone executable script)
- Endüstri Standartları: GPT-2/3/4 BPE training workflow

KULLANIM:
- Model eğitiminden önce çalıştırılır
- Vocab ve merges dosyalarını oluşturur
- Cache sistemi ile hızlı corpus hazırlama

BAĞIMLILIKLAR:
- TokenizerCore: BPE training işlemleri
- DataCache: Corpus cache yönetimi
- DataLoaderManager: Veri yükleme işlemleri

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
from typing import Dict, Any

# Proje kök dizinini sys.path'e ekle
# Script tokenizer_management klasöründe olduğu için iki seviye yukarı çık
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError
from tokenizer_management.config import BPE_CONFIG, get_bpe_detailed_config, get_turkish_config
from training_system.data_cache import DataCache
from data_loader_management.data_loader_manager import DataLoaderManager, DataLoaderConfig, LoadMode

# =========================
# Logging
# =========================
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
bpe_logger = logging.getLogger("BPETraining")
bpe_logger.info("="*80)
bpe_logger.info("BPE TRAINING SCRIPT")
bpe_logger.info("="*80)

# =========================
# Config (Config dosyasından alınır, hardcoded değil)
# =========================
def get_bpe_training_config(data_dir: str = "education") -> Dict[str, Any]:
    """
    BPE training config'ini oluştur (config.py'den alır, hardcoded değil)
    
    Args:
        data_dir: Training verisi dizini
    
    Returns:
        Config dictionary
    """
    # Config'ten al
    bpe_config = get_bpe_detailed_config()
    turkish_config = get_turkish_config()
    
    # Birleştir
    config = {
        **bpe_config,
        **turkish_config,
    }
    
    # Override edilebilir parametreler (komut satırı veya kullanıcı tarafından)
    config.update({
        "data_dir": data_dir,
        # BPE dosya yolları (config'ten al, yoksa default)
        "vocab_path": config.get("vocab_file", "data/vocab_lib/vocab.json"),
        "merges_path": config.get("merges_file", "data/merges_lib/merges.txt"),
        # Vocab size (config'ten al, önce max_vocab_size'a bak, sonra vocab_size'a, yoksa default)
        "vocab_size": config.get("max_vocab_size") or config.get("vocab_size", 60000),
        # Merge operations (config'ten al, yoksa default)
        "merge_operations": config.get("merge_operations", 50000),
        "max_iter": config.get("max_iter", 50000),
        "min_frequency": config.get("min_frequency", 2),
        # Tokenization parametreleri (config'ten al)
        "include_syllables": config.get("include_syllables", False),
        "include_whole_words": config.get("include_whole_words", True),
        "include_sep": config.get("include_sep", True),
        # Diğer parametreler (config'ten al)
        "append_eos": config.get("append_eos", True),
        "protect_specials": config.get("protect_specials", True),
        "sample_ratio": config.get("bpe_sample_ratio", None),
    })
    
    return config

def ensure_dirs(*paths: str) -> None:
    """Dizinleri oluştur"""
    for p in paths:
        if not p:
            continue
        pp = Path(p)
        target = pp.parent if pp.suffix else pp
        target.mkdir(parents=True, exist_ok=True)

def load_training_data(data_dir: str) -> tuple[list, list]:
    """
    Training verisini yükle (QA format + Raw text chunks)
    
    Returns:
        (qa_data, raw_data): QA çiftleri ve raw text chunks
    """
    bpe_logger.info(f"Veri yükleniyor: {data_dir}")
    
    # 1. QA format verilerini yükle (JSON dosyaları)
    qa_loader = DataLoaderManager(DataLoaderConfig(
        data_dir=Path(data_dir),
        mode=LoadMode.QA_TRAIN
    ))
    qa_data = qa_loader.load()
    bpe_logger.info(f" QA format: {len(qa_data)} çift yüklendi")
    
    # 2. Raw text chunks yükle (DOCX/TXT dosyaları)
    raw_loader = DataLoaderManager(DataLoaderConfig(
        data_dir=Path(data_dir),
        mode=LoadMode.TEXT_INFER
    ))
    raw_data = raw_loader.load()
    bpe_logger.info(f" Raw text: {len(raw_data)} chunk yüklendi")
    
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
    
    bpe_logger.info(f" Corpus oluşturuldu: {len(corpus)} metin")
    return corpus

def train_bpe(config: Dict[str, Any]) -> None:
    """
    BPE training yap ve vocab/merges dosyalarını kaydet
    
    Args:
        config: BPE training config
    """
    data_dir = config["data_dir"]
    
    # Veri dizini kontrolü
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Veri dizini bulunamadı: {data_dir}")
    
    # Dizinleri oluştur
    ensure_dirs(
        config["vocab_path"],
        config["merges_path"]
    )
    
    # ✅ BPE Training Cache sistemi (DataCache kullanarak)
    cache_dir = config.get("bpe_cache_dir", ".cache/bpe_training")
    cache_enabled = config.get("enable_bpe_cache", True)  # Default: True
    bpe_cache = DataCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled
    )
    bpe_logger.info(f"BPE Training Cache: {'✅ Aktif' if cache_enabled else '❌ Pasif'} ({cache_dir})")
    
    # Corpus oluşturma fonksiyonu (cache için)
    def create_corpus_func():
        """Cache yoksa corpus oluştur"""
        # 1. Veri yükle
        qa_data, raw_data = load_training_data(data_dir)
        
        if not qa_data and not raw_data:
            raise RuntimeError(f"Veri dizininde hiç veri bulunamadı: {data_dir}")
        
        # 2. Corpus oluştur
        corpus = create_corpus(qa_data, raw_data)
        
        if not corpus:
            raise RuntimeError("Corpus oluşturulamadı!")
        
        return corpus
    
    # Cache'den yükle veya oluştur
    include_whole_words = config.get("include_whole_words", True)
    include_syllables = config.get("include_syllables", False)
    include_sep = config.get("include_sep", True)
    
    corpus, from_cache = bpe_cache.get_or_create_corpus(
        process_func=create_corpus_func,
        include_whole_words=include_whole_words,
        include_syllables=include_syllables,
        include_sep=include_sep,
    )
    
    if from_cache:
        bpe_logger.info("✅ Corpus cache'den yüklendi - hızlı başlangıç!")
    else:
        bpe_logger.info("✅ Corpus oluşturuldu ve cache'e kaydedildi")
    
    # 3. TokenizerCore config'i hazırla
    # BPE config'i al ve override et
    bpe_config = get_bpe_detailed_config()
    bpe_config.update(get_turkish_config())
    bpe_config.update({
        "data_dir": data_dir,
        "vocab_file": config["vocab_path"],
        "merges_file": config["merges_path"],
        "vocab_size": config["vocab_size"],
        "max_iter": config["max_iter"],
        "min_frequency": config["min_frequency"],
        "merge_operations": config["merge_operations"],
        "include_syllables": config["include_syllables"],
        "include_whole_words": config["include_whole_words"],
        "include_sep": config["include_sep"],
        "append_eos": config["append_eos"],
        "protect_specials": config["protect_specials"],
        "bpe_sample_ratio": config.get("sample_ratio"),
    })
    
    # 4. TokenizerCore oluştur
    bpe_logger.info("TokenizerCore başlatılıyor...")
    tokenizer = TokenizerCore(bpe_config)
    
    # 5. BPE training yap
    bpe_logger.info("="*80)
    bpe_logger.info("BPE TRAINING BAŞLIYOR")
    bpe_logger.info("="*80)
    bpe_logger.info(f"Corpus: {len(corpus)} metin")
    # Vocab size: max_vocab_size öncelikli, yoksa vocab_size
    vocab_size_display = config.get("max_vocab_size") or config.get("vocab_size", 60000)
    bpe_logger.info(f"Vocab size (hedef): {vocab_size_display}")
    bpe_logger.info(f"Max iterations: {config['max_iter']}")
    bpe_logger.info(f"Min frequency: {config['min_frequency']}")
    bpe_logger.info(f"Include syllables: {config['include_syllables']}")
    bpe_logger.info(f"Include whole words: {config['include_whole_words']}")
    bpe_logger.info(f"Include sep: {config['include_sep']}")
    bpe_logger.info("="*80)
    
    try:
        tokenizer.train_model(
            corpus,
            method="bpe",
            vocab_size=config["vocab_size"],
            max_iter=config["max_iter"],
            min_frequency=config["min_frequency"],
            include_whole_words=config["include_whole_words"],
            include_syllables=config["include_syllables"],
            include_sep=config["include_sep"],
            append_eos=config["append_eos"],
            protect_specials=config["protect_specials"],
            sample_ratio=config.get("sample_ratio"),
        )
        
        bpe_logger.info("="*80)
        bpe_logger.info("BPE TRAINING TAMAMLANDI")
        bpe_logger.info("="*80)
        
    except TokenizerCoreError as e:
        bpe_logger.error(f"BPE training hatası: {e}")
        raise
    except Exception as e:
        bpe_logger.error(f"Beklenmeyen hata: {e}", exc_info=True)
        raise
    
    # 6. Vocab ve merges dosyalarını kaydet
    bpe_logger.info("Vocab ve merges dosyalari kaydediliyor...")
    tokenizer.tokenizer.save_vocab()
    tokenizer.tokenizer.save_merges()
    
    # 7. Sonuçları göster
    vocab = tokenizer.get_vocab()
    merges = tokenizer.get_merges()
    vocab_size = len(vocab)
    merges_count = len(merges)
    
    bpe_logger.info("="*80)
    bpe_logger.info("BPE TRAINING SONUÇLARI")
    bpe_logger.info("="*80)
    bpe_logger.info(f" Vocab size: {vocab_size:,} token")
    bpe_logger.info(f" Merges: {merges_count:,} merge")
    bpe_logger.info(f" Vocab dosyası: {config['vocab_path']}")
    bpe_logger.info(f" Merges dosyası: {config['merges_path']}")
    bpe_logger.info("="*80)
    
    # Vocab size kontrolü
    target_vocab_size = config["vocab_size"]
    if vocab_size < target_vocab_size * 0.8:
        bpe_logger.warning(
            f"⚠️  Vocab size hedefin altında: {vocab_size:,} < {target_vocab_size:,} "
            f"(%{vocab_size/target_vocab_size*100:.1f})"
        )
        bpe_logger.warning("   Daha fazla veri veya daha fazla merge operation gerekebilir.")
    elif vocab_size > target_vocab_size * 1.2:
        bpe_logger.warning(
            f"⚠️  Vocab size hedefin üstünde: {vocab_size:,} > {target_vocab_size:,} "
            f"(%{vocab_size/target_vocab_size*100:.1f})"
        )
    else:
        bpe_logger.info(f" Vocab size hedef aralığında: {vocab_size:,} ≈ {target_vocab_size:,}")
    
    bpe_logger.info("")
    bpe_logger.info("🎉 BPE training başarıyla tamamlandı!")
    bpe_logger.info("   Şimdi 'train.py' ile model eğitimi yapabilirsiniz.")
    bpe_logger.info("")

def main():
    """Ana fonksiyon"""
    try:
        # Veri dizini (komut satırı veya default)
        data_dir = "education"
        if len(sys.argv) > 1:
            data_dir_arg = sys.argv[1]
            if os.path.isdir(data_dir_arg):
                data_dir = data_dir_arg
                bpe_logger.info(f"Veri dizini komut satırından alındı: {data_dir}")
            else:
                bpe_logger.warning(f"Veri dizini bulunamadı: {data_dir_arg}, default kullanılıyor: {data_dir}")
        
        # Config'i hazırla (config.py'den al, hardcoded değil)
        config = get_bpe_training_config(data_dir)
        
        bpe_logger.info("Config parametreleri (config.py'den alındı):")
        bpe_logger.info(f"  vocab_size: {config['vocab_size']}")
        bpe_logger.info(f"  max_iter: {config['max_iter']}")
        bpe_logger.info(f"  min_frequency: {config['min_frequency']}")
        bpe_logger.info(f"  include_syllables: {config['include_syllables']}")
        bpe_logger.info(f"  include_whole_words: {config['include_whole_words']}")
        bpe_logger.info(f"  include_sep: {config['include_sep']}")
        bpe_logger.info("")
        
        # BPE training yap
        train_bpe(config)
        
    except KeyboardInterrupt:
        bpe_logger.warning("BPE training kullanıcı tarafından durduruldu (KeyboardInterrupt).")
        sys.exit(1)
    except Exception as e:
        bpe_logger.error(f"BPE training başarısız: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

