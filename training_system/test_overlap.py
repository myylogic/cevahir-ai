# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ - OVERLAP TEST SCRIPT
================================================================================

Dosya: test_overlap.py
Modül: training_system
Görev: Train-Validation overlap kontrolü ve raporlama

KULLANIM:
    python training_system/test_overlap.py
    
Çıktı:
    - Overlap sayısı ve oranı
    - Detaylı analiz
    - Endüstri standardı karşılaştırması

================================================================================
"""

import os
import sys
import hashlib
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_system.data_cache import DataCache
from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG, BPE_DETAILED_CONFIG
from training_system.v2.core.data_preparator import DataPreparator
import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("OverlapTest")


def hash_sequence_full(seq, pad_id=0):
    """Tam token dizisini hash'le (PAD'ler hariç)"""
    try:
        import torch
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
    except ImportError:
        pass
    
    # PAD'leri temizle
    cleaned_seq = [t for t in seq if t != pad_id]
    
    # Tam diziyi kullan
    seq_str = str(cleaned_seq)
    return hashlib.sha256(seq_str.encode()).hexdigest()


def test_overlap():
    """Overlap testi yap"""
    logger.info("="*80)
    logger.info("🔍 TRAIN-VALIDATION OVERLAP KONTROLÜ")
    logger.info("="*80)
    
    # Config
    config = {
        "data_dir": "education",
        "vocab_path": BPE_CONFIG.get("vocab_file", "data/vocab_lib/vocab.json"),
        "merges_path": BPE_CONFIG.get("merges_file", "data/merges_lib/merges.txt"),
        "max_seq_length": TOKENIZER_CONFIG.get("max_seq_length", 768),
        "train_val_split": 0.8,
        "split_seed": 42,
        "train_include_whole_words": BPE_DETAILED_CONFIG.get("include_whole_words", True),
        "train_include_syllables": BPE_DETAILED_CONFIG.get("include_syllables", False),
        "train_include_sep": BPE_DETAILED_CONFIG.get("include_sep", False),
    }
    
    logger.info(f"\n📋 Konfigürasyon:")
    logger.info(f"   Data dir: {config['data_dir']}")
    logger.info(f"   Max seq length: {config['max_seq_length']}")
    logger.info(f"   Train/Val split: {config['train_val_split']}")
    logger.info(f"   Split seed: {config['split_seed']}")
    
    # Initialize
    logger.info(f"\n[1] TokenizerCore başlatılıyor...")
    try:
        tokenizer_core = TokenizerCore(config)
        tokenizer_core.finalize_vocab()
        logger.info("[OK] TokenizerCore hazır")
    except Exception as e:
        logger.error(f"❌ TokenizerCore hatası: {e}")
        return
    
    # Cache
    logger.info(f"\n[2] DataCache başlatılıyor...")
    cache = DataCache(
        data_dir=config["data_dir"],
        cache_dir=".cache/preprocessed_data",
        cache_enabled=True
    )
    
    # Data Preparator
    logger.info(f"\n[3] Data hazırlanıyor...")
    preparator = DataPreparator(logger=logger)
    
    try:
        train_data, val_data, vocab_size = preparator.prepare_from_cache(
            data_cache=cache,
            tokenizer_core=tokenizer_core,
            config=config
        )
        
        logger.info(f"\n[OK] Data hazır:")
        logger.info(f"   📊 Total: {len(train_data) + len(val_data):,}")
        logger.info(f"   📊 Train: {len(train_data):,} ({len(train_data)/(len(train_data)+len(val_data))*100:.1f}%)")
        logger.info(f"   📊 Val: {len(val_data):,} ({len(val_data)/(len(train_data)+len(val_data))*100:.1f}%)")
        logger.info(f"   📊 Vocab size: {vocab_size:,}")
        
    except Exception as e:
        logger.error(f"❌ Data hazırlama hatası: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Overlap testi
    logger.info(f"\n" + "="*80)
    logger.info("📊 OVERLAP ANALİZİ")
    logger.info("="*80)
    
    # PAD ID
    vocab = tokenizer_core.get_vocab()
    def _id_of(token: str) -> int:
        val = vocab.get(token)
        if isinstance(val, dict):
            return int(val.get("id", 0))
        return int(val or 0)
    PAD_ID = _id_of("<PAD>")
    
    # Train hash'leri
    logger.info(f"\n[1] Train hash'leri oluşturuluyor...")
    train_hashes = {}  # hash -> (index, length)
    for i, (inp, tgt) in enumerate(train_data):
        try:
            import torch
            if isinstance(inp, torch.Tensor):
                inp = inp.tolist()
        except:
            pass
        h = hash_sequence_full(inp, PAD_ID)
        train_hashes[h] = (i, len([t for t in inp if t != PAD_ID]))
    
    logger.info(f"[OK] {len(train_hashes):,} unique train hash")
    
    # Val hash'leri
    logger.info(f"\n[2] Val hash'leri oluşturuluyor...")
    val_hashes = {}  # hash -> (index, length)
    for i, (inp, tgt) in enumerate(val_data):
        try:
            import torch
            if isinstance(inp, torch.Tensor):
                inp = inp.tolist()
        except:
            pass
        h = hash_sequence_full(inp, PAD_ID)
        val_hashes[h] = (i, len([t for t in inp if t != PAD_ID]))
    
    logger.info(f"[OK] {len(val_hashes):,} unique val hash")
    
    # Overlap kontrolü
    logger.info(f"\n[3] Overlap kontrolü...")
    overlap_hashes = set(train_hashes.keys()) & set(val_hashes.keys())
    overlap_count = len(overlap_hashes)
    overlap_ratio = overlap_count / len(train_hashes) if train_hashes else 0.0
    
    logger.info(f"\n" + "="*80)
    logger.info("📊 SONUÇLAR")
    logger.info("="*80)
    logger.info(f"\n📊 Hash İstatistikleri:")
    logger.info(f"   Train unique hash: {len(train_hashes):,}")
    logger.info(f"   Val unique hash: {len(val_hashes):,}")
    logger.info(f"   Overlap: {overlap_count:,}")
    
    if overlap_count == 0:
        logger.info(f"\n✅ ✅ ✅ MÜKEMMEL! ✅ ✅ ✅")
        logger.info(f"   Overlap: %0")
        logger.info(f"   Train ve Val setleri tamamen ayrı!")
        logger.info(f"   Endüstri standardına uygun!")
    else:
        logger.error(f"\n❌ ❌ ❌ OVERLAP TESPİT EDİLDİ! ❌ ❌ ❌")
        logger.error(f"   Overlap sayısı: {overlap_count:,}")
        logger.error(f"   Overlap oranı: {overlap_ratio:.2%}")
        logger.error(f"   Bu, data leakage anlamına gelir!")
        logger.error(f"   Çözüm: Cache'i yeniden oluştur (prepare_cache.py --clear-cache)")
        
        # İlk 5 overlap örneğini göster
        logger.info(f"\n📋 İlk 5 Overlap Örneği:")
        for i, h in enumerate(list(overlap_hashes)[:5]):
            train_idx, train_len = train_hashes[h]
            val_idx, val_len = val_hashes[h]
            logger.info(f"   [{i+1}] Hash: {h[:16]}...")
            logger.info(f"       Train: idx={train_idx}, len={train_len}")
            logger.info(f"       Val:   idx={val_idx}, len={val_len}")
    
    # Endüstri standardı karşılaştırması
    logger.info(f"\n" + "="*80)
    logger.info("📊 ENDÜSTRİ STANDARTLARI")
    logger.info("="*80)
    logger.info(f"   GPT-3: Overlap %0 (document-level split + deduplication)")
    logger.info(f"   BERT: Overlap %0 (document-level split)")
    logger.info(f"   LLaMA: Overlap %0 (exact deduplication)")
    logger.info(f"   Cevahir-AI: Overlap {overlap_ratio:.2%} {' ✅' if overlap_count == 0 else ' ❌'}")
    
    logger.info(f"\n" + "="*80)
    if overlap_count == 0:
        logger.info("✅ TEST BAŞARILI - Overlap %0, endüstri standardına uygun!")
    else:
        logger.error("❌ TEST BAŞARISIZ - Overlap var, cache'i yeniden oluştur!")
    logger.info("="*80)
    
    return overlap_count == 0


if __name__ == "__main__":
    try:
        success = test_overlap()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

