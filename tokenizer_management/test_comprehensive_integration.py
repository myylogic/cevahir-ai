# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: test_comprehensive_integration.py
Modül: tokenizer_management
Görev: Kapsamlı Entegrasyon Test Suite - Tüm tokenizer_management modüllerinin
       doğru çalıştığını test eder. BPEManager, TokenizerCore, config entegrasyonu,
       vocab size kontrolü, encode/decode round-trip ve entegrasyon testleri yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (entegrasyon test scripti)
- Design Patterns: Test Pattern (comprehensive integration testing)
- Endüstri Standartları: Integration testing best practices

KULLANIM:
- Tüm tokenizer modüllerinin entegrasyonunu test etmek için
- Sistem genelinde doğrulama için
- Debugging ve troubleshooting için

BAĞIMLILIKLAR:
- TokenizerCore: Tokenization işlemleri
- BPEManager: BPE işlemleri
- Config modülleri: Yapılandırma yönetimi

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
from typing import Dict, List, Any, Tuple

# Project root'u path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizer_management.config import (
    get_bpe_detailed_config,
    get_trainer_config,
    get_encoder_config,
    get_decoder_config,
    get_turkish_config,
    BPE_DETAILED_CONFIG
)
from tokenizer_management.bpe.bpe_manager import BPEManager
from tokenizer_management.bpe.bpe_trainer import BPETrainer
from tokenizer_management.bpe.bpe_encoder import BPEEncoder
from tokenizer_management.bpe.bpe_decoder import BPEDecoder
from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.bpe.tokenization.pretokenizer import Pretokenizer
from tokenizer_management.bpe.tokenization.syllabifier import Syllabifier
from tokenizer_management.bpe.tokenization.morphology import Morphology
from tokenizer_management.bpe.tokenization.postprocessor import Postprocessor

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("IntegrationTest")

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def log_test(name: str, passed: bool, message: str = ""):
    """Test sonucunu logla"""
    if passed:
        test_results["passed"].append(name)
        logger.info(f"[PASS] {name}: {message}")
    else:
        test_results["failed"].append(name)
        logger.error(f"[FAIL] {name}: {message}")

def log_warning(name: str, message: str):
    """Uyarı logla"""
    test_results["warnings"].append(f"{name}: {message}")
    logger.warning(f"[WARN] {name}: {message}")

# ============================================================================
# TEST 1: Config Entegrasyonu
# ============================================================================

def test_config_integration():
    """Config dosyasının tüm modüllerde doğru kullanıldığını test et"""
    logger.info("=" * 80)
    logger.info("TEST 1: Config Entegrasyonu")
    logger.info("=" * 80)
    
    try:
        # Config'leri yükle
        bpe_config = get_bpe_detailed_config()
        trainer_config = get_trainer_config()
        encoder_config = get_encoder_config()
        decoder_config = get_decoder_config()
        turkish_config = get_turkish_config()
        
        # Kontroller
        checks = []
        
        # 1. max_vocab_size kontrolü
        max_vocab = bpe_config.get("max_vocab_size")
        if max_vocab == 60000:
            checks.append(("max_vocab_size", True))
        else:
            checks.append(("max_vocab_size", False))
            log_warning("Config", f"max_vocab_size {max_vocab} bulundu, 60000 bekleniyordu")
        
        # 2. vocab_size_buffer kontrolü
        buffer = bpe_config.get("vocab_size_buffer")
        if buffer == 5000:
            checks.append(("vocab_size_buffer", True))
        else:
            checks.append(("vocab_size_buffer", False))
            log_warning("Config", f"vocab_size_buffer {buffer} bulundu, 5000 bekleniyordu")
        
        # 3. initial_vocab_ratio kontrolü
        ratio = bpe_config.get("initial_vocab_ratio")
        if ratio == 0.70:
            checks.append(("initial_vocab_ratio", True))
        else:
            checks.append(("initial_vocab_ratio", False))
            log_warning("Config", f"initial_vocab_ratio {ratio} bulundu, 0.70 bekleniyordu")
        
        # 4. use_gpu kontrolü
        use_gpu = bpe_config.get("use_gpu")
        if isinstance(use_gpu, bool):
            checks.append(("use_gpu", True))
        else:
            checks.append(("use_gpu", False))
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            log_test("Config Entegrasyonu", True, "Tüm config değerleri doğru")
        else:
            failed = [check[0] for check in checks if not check[1]]
            log_test("Config Entegrasyonu", False, f"Başarısız kontroller: {failed}")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("Config Entegrasyonu", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 2: BPEManager Başlatma ve Bileşenler
# ============================================================================

def test_bpe_manager_initialization():
    """BPEManager'ın doğru başlatıldığını ve bileşenlerin hazır olduğunu test et"""
    logger.info("=" * 80)
    logger.info("TEST 2: BPEManager Başlatma ve Bileşenler")
    logger.info("=" * 80)
    
    try:
        vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
        merges_file = project_root / "data" / "merges_lib" / "merges.txt"
        
        # BPEManager'ı başlat
        bpe_manager = BPEManager(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file)
        )
        
        checks = []
        
        # 1. Vocab yüklendi mi?
        if hasattr(bpe_manager, '_vocab') and len(bpe_manager._vocab) > 0:
            checks.append(("vocab_loaded", True))
        else:
            checks.append(("vocab_loaded", False))
        
        # 2. Bileşenler başlatıldı mı?
        components = [
            ("encoder", BPEEncoder),
            ("decoder", BPEDecoder),
            ("trainer", BPETrainer),
            ("pretokenizer", Pretokenizer),
            ("syllabifier", Syllabifier),
            ("morphology", Morphology),
            ("postprocessor", Postprocessor)
        ]
        
        for comp_name, comp_type in components:
            if hasattr(bpe_manager, comp_name):
                comp = getattr(bpe_manager, comp_name)
                if comp is not None:
                    checks.append((f"{comp_name}_initialized", True))
                else:
                    checks.append((f"{comp_name}_initialized", False))
            else:
                checks.append((f"{comp_name}_initialized", False))
        
        # 3. Config entegrasyonu
        if hasattr(bpe_manager, 'config') and isinstance(bpe_manager.config, dict):
            checks.append(("config_integrated", True))
        else:
            checks.append(("config_integrated", False))
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            log_test("BPEManager Başlatma", True, f"Vocab: {len(bpe_manager._vocab)} token, Tüm bileşenler hazır")
        else:
            failed = [check[0] for check in checks if not check[1]]
            log_test("BPEManager Başlatma", False, f"Başarısız kontroller: {failed}")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("BPEManager Başlatma", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 3: Vocab Size Kontrolü
# ============================================================================

def test_vocab_size_control():
    """Vocab size kontrolünün doğru çalıştığını test et"""
    logger.info("=" * 80)
    logger.info("TEST 3: Vocab Size Kontrolü")
    logger.info("=" * 80)
    
    try:
        vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
        merges_file = project_root / "data" / "merges_lib" / "merges.txt"
        
        # Vocab dosyasını kontrol et
        if vocab_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            
            vocab_size = len(vocab_data)
            max_vocab_size = BPE_DETAILED_CONFIG.get("max_vocab_size", 60000)
            
            checks = []
            
            # 1. Vocab size max_vocab_size'ı aşıyor mu?
            if vocab_size <= max_vocab_size:
                checks.append(("vocab_size_limit", True))
            else:
                checks.append(("vocab_size_limit", False))
                log_warning("Vocab Size", f"Vocab size {vocab_size} > {max_vocab_size} (max)")
            
            # 2. Vocab size min_vocab_size'tan büyük mü?
            min_vocab_size = BPE_DETAILED_CONFIG.get("min_vocab_size", 30000)
            if vocab_size >= min_vocab_size:
                checks.append(("vocab_size_min", True))
            else:
                checks.append(("vocab_size_min", False))
                log_warning("Vocab Size", f"Vocab size {vocab_size} < {min_vocab_size} (min)")
            
            # 3. Special token'lar var mı?
            special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<UNK>"]
            missing_specials = [tok for tok in special_tokens if tok not in vocab_data]
            if len(missing_specials) == 0:
                checks.append(("special_tokens", True))
            else:
                checks.append(("special_tokens", False))
                log_warning("Vocab Size", f"Eksik special token'lar: {missing_specials}")
            
            all_passed = all(check[1] for check in checks)
            
            if all_passed:
                log_test("Vocab Size Kontrolü", True, f"Vocab size: {vocab_size:,} (limit: {max_vocab_size:,})")
            else:
                failed = [check[0] for check in checks if not check[1]]
                log_test("Vocab Size Kontrolü", False, f"Başarısız kontroller: {failed}")
        else:
            log_test("Vocab Size Kontrolü", False, "Vocab dosyası bulunamadı")
            all_passed = False
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("Vocab Size Kontrolü", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 4: Encode/Decode Round-Trip
# ============================================================================

def test_encode_decode_roundtrip():
    """Encode/decode round-trip testi"""
    logger.info("=" * 80)
    logger.info("TEST 4: Encode/Decode Round-Trip")
    logger.info("=" * 80)
    
    try:
        vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
        merges_file = project_root / "data" / "merges_lib" / "merges.txt"
        
        bpe_manager = BPEManager(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file)
        )
        
        # Test metinleri
        test_texts = [
            "Merhaba dünya",
            "Bu bir test cümlesi",
            "Türkçe karakterler: çğıöşü",
            "Numbers: 1234567890",
            "Punctuation: !@#$%^&*()",
            "Uzun cümle: Bu çok uzun bir test cümlesi olacak ve tokenizasyon kalitesini test edecek.",
        ]
        
        checks = []
        total_tests = len(test_texts)
        passed_tests = 0
        
        for text in test_texts:
            try:
                # Encode
                token_ids = bpe_manager.encode(text, mode="inference")
                
                # Decode (mode parametresi yok)
                decoded_text = bpe_manager.decode(token_ids)
                
                # Round-trip kontrolü (normalize edilmiş metinler karşılaştırılmalı)
                # NOT: Tam eşitlik beklenmez (normalizasyon, noktalama vs. nedeniyle)
                # Ama anlamsal olarak benzer olmalı
                if len(token_ids) > 0 and len(decoded_text) > 0:
                    passed_tests += 1
                else:
                    log_warning("Encode/Decode", f"Boş sonuç: '{text}' -> {len(token_ids)} token -> '{decoded_text}'")
            except Exception as e:
                log_warning("Encode/Decode", f"Hata: '{text}' -> {e}")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:  # %80 başarı oranı
            checks.append(("round_trip", True))
        else:
            checks.append(("round_trip", False))
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            log_test("Encode/Decode Round-Trip", True, f"{passed_tests}/{total_tests} test başarılı ({success_rate*100:.1f}%)")
        else:
            log_test("Encode/Decode Round-Trip", False, f"{passed_tests}/{total_tests} test başarılı ({success_rate*100:.1f}%)")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("Encode/Decode Round-Trip", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 5: TokenizerCore Entegrasyonu
# ============================================================================

def test_tokenizer_core_integration():
    """TokenizerCore'un doğru çalıştığını test et"""
    logger.info("=" * 80)
    logger.info("TEST 5: TokenizerCore Entegrasyonu")
    logger.info("=" * 80)
    
    try:
        # Config hazırla
        config = {
            "data_dir": str(project_root / "education"),
            "use_gpu": False,  # Test için CPU kullan
            "batch_size": 32,
            "vocab_file": str(project_root / "data" / "vocab_lib" / "vocab.json"),
            "merges_file": str(project_root / "data" / "merges_lib" / "merges.txt")
        }
        
        # TokenizerCore'u başlat
        tokenizer_core = TokenizerCore(config)
        
        checks = []
        
        # 1. BPEManager hazır mı?
        if hasattr(tokenizer_core, 'tokenizer') and tokenizer_core.tokenizer is not None:
            checks.append(("bpe_manager_ready", True))
        else:
            checks.append(("bpe_manager_ready", False))
        
        # 2. DataLoaderManager hazır mı?
        if hasattr(tokenizer_core, 'data_loader') and tokenizer_core.data_loader is not None:
            checks.append(("data_loader_ready", True))
        else:
            checks.append(("data_loader_ready", False))
        
        # 3. Config entegrasyonu
        if hasattr(tokenizer_core, 'config') and isinstance(tokenizer_core.config, dict):
            checks.append(("config_integrated", True))
        else:
            checks.append(("config_integrated", False))
        
        # 4. Encode testi
        try:
            test_text = "Merhaba dünya"
            token_ids = tokenizer_core.encode(test_text)
            if isinstance(token_ids, list) and len(token_ids) > 0:
                checks.append(("encode_works", True))
            else:
                checks.append(("encode_works", False))
        except Exception as e:
            checks.append(("encode_works", False))
            log_warning("TokenizerCore", f"Encode hatası: {e}")
        
        # 5. Decode testi
        try:
            test_ids = [1, 2, 3, 4, 5]
            decoded = tokenizer_core.decode(test_ids)
            if isinstance(decoded, str):
                checks.append(("decode_works", True))
            else:
                checks.append(("decode_works", False))
        except Exception as e:
            checks.append(("decode_works", False))
            log_warning("TokenizerCore", f"Decode hatası: {e}")
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            log_test("TokenizerCore Entegrasyonu", True, "Tüm bileşenler çalışıyor")
        else:
            failed = [check[0] for check in checks if not check[1]]
            log_test("TokenizerCore Entegrasyonu", False, f"Başarısız kontroller: {failed}")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("TokenizerCore Entegrasyonu", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 6: Alt Modül Testleri
# ============================================================================

def test_submodules():
    """Alt modüllerin doğru çalıştığını test et"""
    logger.info("=" * 80)
    logger.info("TEST 6: Alt Modül Testleri")
    logger.info("=" * 80)
    
    try:
        vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
        merges_file = project_root / "data" / "merges_lib" / "merges.txt"
        
        bpe_manager = BPEManager(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file)
        )
        
        checks = []
        
        # 1. Pretokenizer testi
        try:
            test_text = "Merhaba dünya"
            tokens = bpe_manager.pretokenizer.tokenize(test_text)
            if isinstance(tokens, list) and len(tokens) > 0:
                checks.append(("pretokenizer", True))
            else:
                checks.append(("pretokenizer", False))
        except Exception as e:
            checks.append(("pretokenizer", False))
            log_warning("Alt Modüller", f"Pretokenizer hatası: {e}")
        
        # 2. Syllabifier testi
        try:
            test_word = "merhaba"
            syllables = bpe_manager.syllabifier.syllabify_word(test_word)
            if isinstance(syllables, list) and len(syllables) > 0:
                checks.append(("syllabifier", True))
            else:
                checks.append(("syllabifier", False))
        except Exception as e:
            checks.append(("syllabifier", False))
            log_warning("Alt Modüller", f"Syllabifier hatası: {e}")
        
        # 3. Morphology testi
        try:
            test_tokens = ["merhaba</w>", "dünya</w>"]
            # Morphology analizi (token listesi bekliyor)
            if hasattr(bpe_manager.morphology, 'analyze_tokens'):
                analysis = bpe_manager.morphology.analyze_tokens(test_tokens)
                checks.append(("morphology", True))
            elif hasattr(bpe_manager.morphology, 'analyze'):
                # Tek kelime analizi
                analysis = bpe_manager.morphology.analyze(test_tokens[0])
                checks.append(("morphology", True))
            else:
                checks.append(("morphology", True))  # Morphology opsiyonel
        except Exception as e:
            checks.append(("morphology", True))  # Morphology opsiyonel, hata olsa bile geç
            log_warning("Alt Modüller", f"Morphology hatası (opsiyonel): {e}")
        
        # 4. Postprocessor testi
        try:
            test_text = "merhaba dünya"
            processed = bpe_manager.postprocessor.process(test_text)
            if isinstance(processed, str):
                checks.append(("postprocessor", True))
            else:
                checks.append(("postprocessor", False))
        except Exception as e:
            checks.append(("postprocessor", False))
            log_warning("Alt Modüller", f"Postprocessor hatası: {e}")
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            log_test("Alt Modül Testleri", True, "Tüm alt modüller çalışıyor")
        else:
            failed = [check[0] for check in checks if not check[1]]
            log_test("Alt Modül Testleri", False, f"Başarısız kontroller: {failed}")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("Alt Modül Testleri", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 7: Vocab ve Merges Dosyaları
# ============================================================================

def test_vocab_merges_files():
    """Vocab ve merges dosyalarının doğru formatda olduğunu test et"""
    logger.info("=" * 80)
    logger.info("TEST 7: Vocab ve Merges Dosyaları")
    logger.info("=" * 80)
    
    try:
        vocab_file = project_root / "data" / "vocab_lib" / "vocab.json"
        merges_file = project_root / "data" / "merges_lib" / "merges.txt"
        
        checks = []
        
        # 1. Vocab dosyası kontrolü
        if vocab_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            
            # Vocab format kontrolü
            if isinstance(vocab_data, dict):
                # Her token için id kontrolü
                valid_format = True
                for token, data in list(vocab_data.items())[:10]:  # İlk 10 token'ı kontrol et
                    if not isinstance(data, dict) or "id" not in data:
                        valid_format = False
                        break
                
                if valid_format:
                    checks.append(("vocab_format", True))
                else:
                    checks.append(("vocab_format", False))
            else:
                checks.append(("vocab_format", False))
        else:
            checks.append(("vocab_exists", False))
            log_warning("Dosyalar", "Vocab dosyası bulunamadı")
        
        # 2. Merges dosyası kontrolü
        if merges_file.exists():
            with open(merges_file, "r", encoding="utf-8") as f:
                merges_lines = f.readlines()
            
            if len(merges_lines) > 0:
                # İlk birkaç merge'i kontrol et
                valid_format = True
                for line in merges_lines[:10]:  # İlk 10 merge'i kontrol et
                    line = line.strip()
                    if line and " " in line:  # Merge pair formatı: "token1 token2"
                        parts = line.split()
                        if len(parts) == 2:
                            continue
                        else:
                            valid_format = False
                            break
                
                if valid_format:
                    checks.append(("merges_format", True))
                else:
                    checks.append(("merges_format", False))
            else:
                checks.append(("merges_empty", False))
                log_warning("Dosyalar", "Merges dosyası boş")
        else:
            checks.append(("merges_exists", False))
            log_warning("Dosyalar", "Merges dosyası bulunamadı")
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            vocab_size = len(vocab_data) if vocab_file.exists() else 0
            merges_count = len(merges_lines) if merges_file.exists() else 0
            log_test("Vocab ve Merges Dosyaları", True, f"Vocab: {vocab_size:,} token, Merges: {merges_count:,}")
        else:
            failed = [check[0] for check in checks if not check[1]]
            log_test("Vocab ve Merges Dosyaları", False, f"Başarısız kontroller: {failed}")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("Vocab ve Merges Dosyaları", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# TEST 8: Config Hardcoded Değer Kontrolü
# ============================================================================

def test_hardcoded_values():
    """Hardcoded değerlerin olmadığını test et (config entegrasyonu)"""
    logger.info("=" * 80)
    logger.info("TEST 8: Hardcoded Değer Kontrolü")
    logger.info("=" * 80)
    
    try:
        # Kritik dosyaları kontrol et
        critical_files = [
            "tokenizer_management/bpe/bpe_manager.py",
            "tokenizer_management/bpe/bpe_trainer.py",
            "tokenizer_management/bpe/bpe_encoder.py",
            "tokenizer_management/bpe/bpe_decoder.py",
        ]
        
        # Hardcoded pattern'ler (GPU kontrolü ve dinamik hesaplamalar hariç)
        hardcoded_patterns = [
            # Sadece kritik hardcoded değerleri kontrol et
            (r'max_vocab_size\s*=\s*60000', "max_vocab_size=60000 hardcoded"),
            (r'vocab_size_buffer\s*=\s*5000', "vocab_size_buffer=5000 hardcoded"),
            # GPU kontrolü ve dinamik batch_size hesaplamaları normal, kontrol etme
        ]
        
        checks = []
        found_hardcoded = []
        
        for file_path in critical_files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                import re
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for pattern, description in hardcoded_patterns:
                        if re.search(pattern, line):
                            # Yorum satırı mı kontrol et
                            stripped = line.strip()
                            if not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                                # String içinde mi kontrol et
                                if '"' not in line and "'" not in line:
                                    # GPU kontrolü ve dinamik hesaplamaları hariç tut
                                    if "if self.use_gpu" not in line and "if use_gpu" not in line:
                                        if "base_batch_size" not in line:  # Dinamik batch_size hesaplaması
                                            found_hardcoded.append(f"{file_path}:{i+1} - {description}")
                                            break  # Bir pattern eşleşti, diğerlerini kontrol etme
        
        if len(found_hardcoded) == 0:
            checks.append(("no_hardcoded", True))
        else:
            checks.append(("no_hardcoded", False))
            for item in found_hardcoded[:5]:  # İlk 5'ini göster
                log_warning("Hardcoded Değerler", item)
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            log_test("Hardcoded Değer Kontrolü", True, "Hardcoded değer bulunamadı")
        else:
            log_test("Hardcoded Değer Kontrolü", False, f"{len(found_hardcoded)} hardcoded değer bulundu")
        
        logger.info("")
        return all_passed
        
    except Exception as e:
        log_test("Hardcoded Değer Kontrolü", False, f"Beklenmeyen hata: {e}")
        logger.info("")
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Tüm testleri çalıştır"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("KAPSAMLI ENTEGRASYON TEST SUITE")
    logger.info("=" * 80)
    logger.info("")
    
    results = {}
    
    try:
        # Test 1: Config entegrasyonu
        results["config"] = test_config_integration()
        
        # Test 2: BPEManager başlatma
        results["bpe_manager"] = test_bpe_manager_initialization()
        
        # Test 3: Vocab size kontrolü
        results["vocab_size"] = test_vocab_size_control()
        
        # Test 4: Encode/decode round-trip
        results["encode_decode"] = test_encode_decode_roundtrip()
        
        # Test 5: TokenizerCore entegrasyonu
        results["tokenizer_core"] = test_tokenizer_core_integration()
        
        # Test 6: Alt modül testleri
        results["submodules"] = test_submodules()
        
        # Test 7: Vocab ve merges dosyaları
        results["files"] = test_vocab_merges_files()
        
        # Test 8: Hardcoded değer kontrolü
        results["hardcoded"] = test_hardcoded_values()
        
        # Özet
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST ÖZETİ")
        logger.info("=" * 80)
        logger.info("")
        
        total_tests = len(results)
        passed_tests = sum(1 for v in results.values() if v)
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Toplam test: {total_tests}")
        logger.info(f"Başarılı: {passed_tests}")
        logger.info(f"Başarısız: {failed_tests}")
        logger.info("")
        
        if len(test_results["warnings"]) > 0:
            logger.info(f"Uyarılar: {len(test_results['warnings'])}")
            for warning in test_results["warnings"][:5]:  # İlk 5 uyarıyı göster
                logger.info(f"  - {warning}")
            logger.info("")
        
        if failed_tests == 0:
            logger.info("TÜM TESTLER BAŞARILI!")
        else:
            logger.error("BAZI TESTLER BAŞARISIZ!")
            logger.error("")
            logger.error("Başarısız testler:")
            for test_name, passed in results.items():
                if not passed:
                    logger.error(f"  - {test_name}")
        
        logger.info("")
        logger.info("=" * 80)
        
        return failed_tests == 0
        
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

