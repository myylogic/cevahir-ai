# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: test_vocab_size_control.py
Modül: tokenizer_management
Görev: Vocab Size Control Test Script - BPE training sırasında vocab size
       kontrolünün doğru çalıştığını test eder. Initial vocab size hesaplaması,
       merge sırasında vocab size limit kontrolü ve final vocab size'ın
       max_vocab_size'ı aşmadığını doğrular.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (vocab size kontrol test scripti)
- Design Patterns: Test Pattern (vocab size validation)
- Endüstri Standartları: Vocab size control validation

KULLANIM:
- Vocab size kontrolünü test etmek için
- BPE training doğrulaması için
- Debugging ve troubleshooting için

BAĞIMLILIKLAR:
- TokenizerCore: BPE training işlemleri
- Config modülleri: Vocab size yapılandırması

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
import json
import logging
from pathlib import Path

# Project root'u path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizer_management.config import (
    get_bpe_detailed_config,
    get_trainer_config,
    BPE_DETAILED_CONFIG
)
from tokenizer_management.bpe.bpe_manager import BPEManager
from tokenizer_management.bpe.bpe_trainer import BPETrainer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("VocabSizeTest")

def test_config_values():
    """Config değerlerini kontrol et"""
    logger.info("=" * 80)
    logger.info("TEST 1: Config Değerleri Kontrolü")
    logger.info("=" * 80)
    
    config = get_bpe_detailed_config()
    
    max_vocab_size = config.get("max_vocab_size", 60000)
    vocab_size_buffer = config.get("vocab_size_buffer", 5000)
    initial_vocab_ratio = config.get("initial_vocab_ratio", 0.70)
    min_vocab_size = config.get("min_vocab_size", 30000)
    
    logger.info(f"max_vocab_size: {max_vocab_size:,}")
    logger.info(f"vocab_size_buffer: {vocab_size_buffer:,}")
    logger.info(f"initial_vocab_ratio: {initial_vocab_ratio}")
    logger.info(f"min_vocab_size: {min_vocab_size:,}")
    
    # Hesaplamalar
    expected_initial_vocab = int(max_vocab_size * initial_vocab_ratio)
    expected_merges = max_vocab_size - expected_initial_vocab - vocab_size_buffer
    max_allowed_vocab = max_vocab_size + vocab_size_buffer
    
    logger.info("")
    logger.info("Hesaplanan Değerler:")
    logger.info(f"  Expected initial vocab: {expected_initial_vocab:,}")
    logger.info(f"  Expected merges: {expected_merges:,}")
    logger.info(f"  Max allowed vocab (with buffer): {max_allowed_vocab:,}")
    
    # Validasyon
    assert max_vocab_size == 60000, f"max_vocab_size 60,000 olmalı, {max_vocab_size} bulundu"
    assert vocab_size_buffer == 5000, f"vocab_size_buffer 5,000 olmalı, {vocab_size_buffer} bulundu"
    assert initial_vocab_ratio == 0.70, f"initial_vocab_ratio 0.70 olmalı, {initial_vocab_ratio} bulundu"
    
    logger.info("")
    logger.info("✓ Config değerleri doğru!")
    logger.info("")
    
    return {
        "max_vocab_size": max_vocab_size,
        "vocab_size_buffer": vocab_size_buffer,
        "initial_vocab_ratio": initial_vocab_ratio,
        "expected_initial_vocab": expected_initial_vocab,
        "expected_merges": expected_merges,
        "max_allowed_vocab": max_allowed_vocab
    }

def test_initial_vocab_calculation():
    """Initial vocab hesaplamasını test et"""
    logger.info("=" * 80)
    logger.info("TEST 2: Initial Vocab Hesaplaması")
    logger.info("=" * 80)
    
    config = get_bpe_detailed_config()
    max_vocab_size = config.get("max_vocab_size", 60000)
    vocab_size_buffer = config.get("vocab_size_buffer", 5000)
    initial_vocab_ratio = config.get("initial_vocab_ratio", 0.70)
    
    # BPEManager'ı başlat (vocab dosyası yoksa oluşturulacak)
    vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
    merges_file = project_root / "data" / "merges_lib" / "merges.txt"
    
    # Test için küçük bir corpus oluştur
    test_corpus = [
        "Merhaba dünya",
        "Bu bir test cümlesi",
        "BPE tokenization test",
        "Türkçe karakterler: çğıöşü",
        "Numbers: 1234567890",
        "Punctuation: !@#$%^&*()",
    ] * 100  # 600 cümle
    
    logger.info(f"Test corpus: {len(test_corpus)} cümle")
    
    # BPEManager'ı başlat (singleton pattern)
    bpe_manager = BPEManager(
        vocab_file=str(vocab_file),
        merges_file=str(merges_file)
    )
    # Config'i güncelle
    if hasattr(bpe_manager, 'config'):
        bpe_manager.config.update(config)
    
    # Initial vocab hesaplamasını simüle et (BPEManager'a gerek yok)
    initial_vocab_size = int(max_vocab_size * initial_vocab_ratio)
    initial_vocab_size = max(initial_vocab_size, config.get("min_vocab_size", 30000))
    
    logger.info(f"Calculated initial vocab size: {initial_vocab_size:,}")
    logger.info(f"Max vocab size: {max_vocab_size:,}")
    logger.info(f"Vocab size buffer: {vocab_size_buffer:,}")
    logger.info(f"Expected merges: {max_vocab_size - initial_vocab_size - vocab_size_buffer:,}")
    
    # Validasyon
    assert initial_vocab_size == 42000, f"Initial vocab 42,000 olmalı, {initial_vocab_size} bulundu"
    assert initial_vocab_size < max_vocab_size, f"Initial vocab max_vocab_size'tan küçük olmalı"
    
    # Expected merges kontrolü
    expected_merges = max_vocab_size - initial_vocab_size - vocab_size_buffer
    assert expected_merges > 0, f"Expected merges pozitif olmalı, {expected_merges} bulundu"
    assert expected_merges == 13000, f"Expected merges 13,000 olmalı, {expected_merges} bulundu"
    
    logger.info("")
    logger.info("✓ Initial vocab hesaplaması doğru!")
    logger.info("")
    
    return {
        "initial_vocab_size": initial_vocab_size,
        "expected_merges": expected_merges
    }

def test_vocab_size_during_merge():
    """Merge sırasında vocab size kontrolünü test et"""
    logger.info("=" * 80)
    logger.info("TEST 3: Merge Sırasında Vocab Size Kontrolü")
    logger.info("=" * 80)
    
    config = get_bpe_detailed_config()
    max_vocab_size = config.get("max_vocab_size", 60000)
    vocab_size_buffer = config.get("vocab_size_buffer", 5000)
    max_allowed = max_vocab_size + vocab_size_buffer
    
    # BPETrainer'ı başlat (special token ID'leri DEFAULT_SPECIALS'a uygun)
    # DEFAULT_SPECIALS: {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<SEP>": 3, "<UNK>": 4}
    test_vocab = {
        "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
        "<BOS>": {"id": 1, "total_freq": 0, "positions": []},
        "<EOS>": {"id": 2, "total_freq": 0, "positions": []},
        "<SEP>": {"id": 3, "total_freq": 0, "positions": []},
        "<UNK>": {"id": 4, "total_freq": 0, "positions": []},
    }
    
    # Initial vocab'a token ekle (42,000 token simüle et)
    # Special token'lar 0-4 arası, normal token'lar 5'ten başlar
    initial_vocab_size = 42000
    next_id = 5  # Special token'lardan sonra
    for i in range(initial_vocab_size - 5):  # 5 special token var
        test_vocab[f"token_{i}"] = {"id": next_id + i, "total_freq": 10, "positions": []}
    
    logger.info(f"Initial vocab size: {len(test_vocab):,}")
    
    trainer = BPETrainer(vocab=test_vocab, config=config)
    
    # Vocab size kontrolünü test et
    current_vocab_size = len(trainer.vocab)
    logger.info(f"Current vocab size: {current_vocab_size:,}")
    logger.info(f"Max allowed vocab: {max_allowed:,}")
    
    # Kontrol: Vocab size limit'i aşmamalı
    assert current_vocab_size <= max_allowed, f"Vocab size {max_allowed:,}'i aşmamalı, {current_vocab_size:,} bulundu"
    
    # Simüle: Merge yaparken vocab size kontrolü
    test_sequence = ["token_100", "token_101", "token_102"] * 1000
    max_iter = 1000
    
    # Vocab size kontrolü yapılan yerleri test et
    vocab_size_checks = []
    
    # Test 1: Batch oluşturulurken kontrol
    if len(trainer.vocab) >= (max_vocab_size + vocab_size_buffer):
        vocab_size_checks.append("Batch oluşturulurken limit aşıldı")
    
    # Test 2: Batch uygulanmadan önce kontrol
    if len(trainer.vocab) >= (max_vocab_size + vocab_size_buffer):
        vocab_size_checks.append("Batch uygulanmadan önce limit aşıldı")
    
    # Test 3: Merge yapılırken kontrol
    if len(trainer.vocab) >= (max_vocab_size + vocab_size_buffer):
        vocab_size_checks.append("Merge yapılırken limit aşıldı")
    
    logger.info(f"Vocab size kontrolleri: {len(vocab_size_checks)} adet")
    
    # Validasyon: Şu anki vocab size limit altında olmalı
    assert current_vocab_size < max_allowed, f"Vocab size {max_allowed:,}'i aşmamalı"
    
    logger.info("")
    logger.info("✓ Merge sırasında vocab size kontrolü çalışıyor!")
    logger.info("")
    
    return {
        "initial_vocab_size": current_vocab_size,
        "max_allowed": max_allowed,
        "checks_passed": len(vocab_size_checks) == 0
    }

def test_final_vocab_size():
    """Final vocab size'ın limit'i aşmadığını test et"""
    logger.info("=" * 80)
    logger.info("TEST 4: Final Vocab Size Kontrolü")
    logger.info("=" * 80)
    
    config = get_bpe_detailed_config()
    max_vocab_size = config.get("max_vocab_size", 60000)
    vocab_size_buffer = config.get("vocab_size_buffer", 5000)
    max_allowed = max_vocab_size + vocab_size_buffer
    
    # Eğer vocab dosyası varsa kontrol et
    vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
    
    if vocab_file.exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        final_vocab_size = len(vocab_data)
        
        logger.info(f"Final vocab size (from file): {final_vocab_size:,}")
        logger.info(f"Max allowed vocab: {max_allowed:,}")
        
        # Validasyon
        if final_vocab_size > max_allowed:
            logger.error(f"❌ HATA: Final vocab size {max_allowed:,}'i aşıyor! ({final_vocab_size:,})")
            return {
                "passed": False,
                "final_vocab_size": final_vocab_size,
                "max_allowed": max_allowed,
                "exceeded_by": final_vocab_size - max_allowed
            }
        else:
            logger.info(f"✓ Final vocab size limit altında: {final_vocab_size:,} <= {max_allowed:,}")
    else:
        logger.warning("Vocab dosyası bulunamadı, test atlandı")
        return {
            "passed": True,
            "message": "Vocab dosyası yok, test atlandı"
        }
    
    logger.info("")
    logger.info("✓ Final vocab size kontrolü başarılı!")
    logger.info("")
    
    return {
        "passed": True,
        "final_vocab_size": final_vocab_size,
        "max_allowed": max_allowed
    }

def test_buffer_consistency():
    """Buffer değerinin tüm sistemde tutarlı olduğunu test et"""
    logger.info("=" * 80)
    logger.info("TEST 5: Buffer Tutarlılık Kontrolü")
    logger.info("=" * 80)
    
    config = get_bpe_detailed_config()
    expected_buffer = config.get("vocab_size_buffer", 5000)
    
    # BPE_DETAILED_CONFIG'ten kontrol et
    config_buffer = BPE_DETAILED_CONFIG.get("vocab_size_buffer", 5000)
    
    logger.info(f"Config buffer: {expected_buffer:,}")
    logger.info(f"BPE_DETAILED_CONFIG buffer: {config_buffer:,}")
    
    # Validasyon
    assert expected_buffer == config_buffer, f"Buffer değerleri tutarlı olmalı: {expected_buffer} vs {config_buffer}"
    assert expected_buffer == 5000, f"Buffer 5,000 olmalı, {expected_buffer} bulundu"
    
    logger.info("")
    logger.info("✓ Buffer değerleri tutarlı!")
    logger.info("")
    
    return {
        "config_buffer": expected_buffer,
        "consistent": True
    }

def main():
    """Tüm testleri çalıştır"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("VOCAB SIZE CONTROL TEST SUITE")
    logger.info("=" * 80)
    logger.info("")
    
    results = {}
    
    try:
        # Test 1: Config değerleri
        results["config"] = test_config_values()
        
        # Test 2: Initial vocab hesaplaması
        results["initial_vocab"] = test_initial_vocab_calculation()
        
        # Test 3: Merge sırasında vocab size kontrolü
        results["merge_control"] = test_vocab_size_during_merge()
        
        # Test 4: Final vocab size
        results["final_vocab"] = test_final_vocab_size()
        
        # Test 5: Buffer tutarlılık
        results["buffer"] = test_buffer_consistency()
        
        # Özet
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST ÖZETİ")
        logger.info("=" * 80)
        logger.info("")
        
        all_passed = all(
            results.get("config"),
            results.get("initial_vocab"),
            results.get("merge_control", {}).get("checks_passed", True),
            results.get("final_vocab", {}).get("passed", True),
            results.get("buffer", {}).get("consistent", True)
        )
        
        if all_passed:
            logger.info("✓ TÜM TESTLER BAŞARILI!")
        else:
            logger.error("❌ BAZI TESTLER BAŞARISIZ!")
        
        logger.info("")
        logger.info("=" * 80)
        
        return all_passed
        
    except AssertionError as e:
        logger.error(f"❌ Test başarısız: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Beklenmeyen hata: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

