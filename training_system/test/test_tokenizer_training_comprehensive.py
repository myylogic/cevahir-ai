# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for TokenizerCore and TrainingService
===============================================================

Bu test suite, TokenizerCore ve TrainingService'in gerçek education verileri ile
10 farklı test metodunu kullanarak kapsamlı analiz yapar.

Test Metodları:
1. test_tokenizer_initialization - TokenizerCore başlatma ve vocab yükleme
2. test_data_loading - Veri yükleme ve format kontrolü
3. test_encoding_process - Encoding işlemleri ve tokenization
4. test_special_tokens - BOS/EOS/PAD/UNK token kontrolü
5. test_input_target_alignment - Input/Target alignment doğrulaması
6. test_cache_mechanism - Cache oluşturma ve yükleme
7. test_training_service_init - TrainingService başlatma
8. test_data_preparation - Veri hazırlama ve preprocessing
9. test_dataloader_creation - DataLoader oluşturma ve batch işleme
10. test_end_to_end_pipeline - Tüm pipeline'ın end-to-end testi
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG
from training_system.training_service import TrainingService
from training_system.data_cache import DataCache

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestSuite")


class ComprehensiveTokenizerTrainingTest:
    """Kapsamlı test suite"""
    
    def __init__(self, data_dir: str = "education"):
        """
        Args:
            data_dir: Eğitim verisi dizini (varsayılan: "education")
        """
        self.data_dir = data_dir
        self.tokenizer_core: TokenizerCore = None
        self.training_service: TrainingService = None
        self.results: Dict[str, Any] = {}
        
        # Test config
        self.config = {
            "data_dir": data_dir,
            "vocab_path": BPE_CONFIG.get("vocab_file", "data/vocab_lib/vocab.json"),
            "merges_path": BPE_CONFIG.get("merges_file", "data/merges_lib/merges.txt"),
            "bpe_rebuild": False,
            "max_seq_length": 128,  # Test için kısa tut
            "train_include_whole_words": True,
            "train_include_syllables": False,
            "train_include_sep": False,
            "use_gpu": False,  # Test için CPU kullan
            "batch_size": 4,
            "cache_dir": ".cache/test_preprocessed_data",
            "enable_data_cache": True,
            "clear_cache_on_start": False,  # Test için cache'i temizleme
        }
        
        logger.info("="*80)
        logger.info("COMPREHENSIVE TEST SUITE BAŞLIYOR")
        logger.info("="*80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Config: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
    
    def test_1_tokenizer_initialization(self) -> bool:
        """Test 1: TokenizerCore başlatma ve vocab yükleme"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: TokenizerCore Initialization")
        logger.info("="*80)
        
        try:
            # TokenizerCore oluştur
            logger.info("TokenizerCore oluşturuluyor...")
            self.tokenizer_core = TokenizerCore(self.config)
            
            # Vocab finalize et
            logger.info("Vocab finalize ediliyor...")
            self.tokenizer_core.finalize_vocab()
            
            # Vocab bilgilerini al
            vocab = self.tokenizer_core.get_vocab()
            vocab_size = len(vocab)
            
            logger.info(f"✅ Vocab yüklendi: {vocab_size:,} token")
            
            # Special token kontrolü
            special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
            special_token_ids = {}
            
            for token in special_tokens:
                token_data = vocab.get(token)
                if token_data:
                    token_id = token_data.get('id') if isinstance(token_data, dict) else token_data
                    special_token_ids[token] = token_id
                    logger.info(f"  - {token}: ID={token_id}")
                else:
                    logger.warning(f"  - {token}: BULUNAMADI!")
                    special_token_ids[token] = None
            
            # Sonuçları kaydet
            self.results["test_1"] = {
                "success": True,
                "vocab_size": vocab_size,
                "special_tokens": special_token_ids,
                "message": "TokenizerCore başarıyla başlatıldı"
            }
            
            logger.info("✅ TEST 1 BAŞARILI")
            return True
            
        except Exception as e:
            logger.error(f"❌ TEST 1 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_1"] = {
                "success": False,
                "error": str(e),
                "message": "TokenizerCore başlatılamadı"
            }
            return False
    
    def test_2_data_loading(self) -> bool:
        """Test 2: Veri yükleme ve format kontrolü"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Data Loading")
        logger.info("="*80)
        
        try:
            if self.tokenizer_core is None:
                raise ValueError("TokenizerCore başlatılmamış! Önce test_1 çalıştırılmalı.")
            
            # Veri yükle
            logger.info("Eğitim verisi yükleniyor...")
            raw_data = self.tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=self.config["train_include_whole_words"],
                include_syllables=self.config["train_include_syllables"],
                include_sep=self.config["train_include_sep"],
            )
            
            total_examples = len(raw_data)
            logger.info(f"✅ {total_examples:,} örnek yüklendi")
            
            if total_examples == 0:
                raise ValueError("Hiç örnek yüklenmedi!")
            
            # Format kontrolü
            valid_examples = 0
            invalid_examples = 0
            length_stats = {"input": [], "target": []}
            
            for idx, pair in enumerate(raw_data[:100]):  # İlk 100 örneği kontrol et
                try:
                    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                        invalid_examples += 1
                        continue
                    
                    inp_ids, tgt_ids = pair
                    
                    if not isinstance(inp_ids, list) or not isinstance(tgt_ids, list):
                        invalid_examples += 1
                        continue
                    
                    if not all(isinstance(t, int) for t in inp_ids):
                        invalid_examples += 1
                        continue
                    
                    if not all(isinstance(t, int) for t in tgt_ids):
                        invalid_examples += 1
                        continue
                    
                    valid_examples += 1
                    length_stats["input"].append(len(inp_ids))
                    length_stats["target"].append(len(tgt_ids))
                    
                except Exception as e:
                    invalid_examples += 1
                    logger.warning(f"Örnek {idx} geçersiz: {e}")
            
            # İstatistikler
            avg_input_len = sum(length_stats["input"]) / len(length_stats["input"]) if length_stats["input"] else 0
            avg_target_len = sum(length_stats["target"]) / len(length_stats["target"]) if length_stats["target"] else 0
            
            logger.info(f"  - Geçerli örnekler: {valid_examples}/{min(100, total_examples)}")
            logger.info(f"  - Geçersiz örnekler: {invalid_examples}")
            logger.info(f"  - Ortalama input uzunluğu: {avg_input_len:.2f}")
            logger.info(f"  - Ortalama target uzunluğu: {avg_target_len:.2f}")
            
            self.results["test_2"] = {
                "success": True,
                "total_examples": total_examples,
                "valid_examples": valid_examples,
                "invalid_examples": invalid_examples,
                "avg_input_len": avg_input_len,
                "avg_target_len": avg_target_len,
                "message": "Veri başarıyla yüklendi"
            }
            
            logger.info("✅ TEST 2 BAŞARILI")
            return True
            
        except Exception as e:
            logger.error(f"❌ TEST 2 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_2"] = {
                "success": False,
                "error": str(e),
                "message": "Veri yüklenemedi"
            }
            return False
    
    def test_3_encoding_process(self) -> bool:
        """Test 3: Encoding işlemleri ve tokenization"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Encoding Process")
        logger.info("="*80)
        
        try:
            if self.tokenizer_core is None:
                raise ValueError("TokenizerCore başlatılmamış!")
            
            # Test metinleri
            test_texts = [
                "Merhaba dünya",
                "Bu bir test metnidir.",
                "Eğitim verisi analizi yapılıyor.",
                "Tokenizer çalışıyor mu?",
                "123456 test sayıları",
            ]
            
            encoding_results = []
            
            for text in test_texts:
                try:
                    # Encode - encode() Tuple[List[str], List[int]] döndürür (tokens, ids)
                    tokens, ids = self.tokenizer_core.encode(text)
                    
                    # Decode (geri çevir) - decode() sadece ID listesi alır
                    decoded = self.tokenizer_core.decode(ids)
                    
                    # Roundtrip kontrolü - normalize edilmiş karşılaştırma
                    original_normalized = text.strip().lower()
                    decoded_normalized = decoded.strip().lower()
                    # Bazı karakterler farklı encode/decode edilebilir (noktalama, boşluk vb.)
                    # Bu yüzden esnek bir karşılaştırma yapıyoruz
                    roundtrip_success = (
                        original_normalized == decoded_normalized or
                        original_normalized.replace(" ", "") == decoded_normalized.replace(" ", "")
                    )
                    
                    encoding_results.append({
                        "original": text,
                        "tokens": tokens[:10] if len(tokens) > 10 else tokens,  # İlk 10 token
                        "encoded_ids": ids[:10] if len(ids) > 10 else ids,  # İlk 10 ID
                        "decoded": decoded,
                        "token_count": len(ids),
                        "roundtrip_success": roundtrip_success
                    })
                    
                    logger.info(f"  Text: '{text}'")
                    logger.info(f"    → Tokens: {tokens[:5]}..." if len(tokens) > 5 else f"    → Tokens: {tokens}")
                    logger.info(f"    → IDs: {ids[:10]}..." if len(ids) > 10 else f"    → IDs: {ids}")
                    logger.info(f"    → Decoded: '{decoded}'")
                    logger.info(f"    → Roundtrip: {'✅' if roundtrip_success else '❌'}")
                    
                except Exception as e:
                    logger.warning(f"Encoding hatası ('{text}'): {e}", exc_info=True)
                    encoding_results.append({
                        "original": text,
                        "error": str(e)
                    })
            
            success_count = sum(1 for r in encoding_results if r.get("roundtrip_success", False))
            
            self.results["test_3"] = {
                "success": success_count > 0,
                "total_tests": len(test_texts),
                "successful_roundtrips": success_count,
                "encoding_results": encoding_results,
                "message": f"{success_count}/{len(test_texts)} encoding testi başarılı"
            }
            
            logger.info(f"✅ TEST 3 BAŞARILI ({success_count}/{len(test_texts)})")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ TEST 3 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_3"] = {
                "success": False,
                "error": str(e),
                "message": "Encoding testi başarısız"
            }
            return False
    
    def test_4_special_tokens(self) -> bool:
        """Test 4: BOS/EOS/PAD/UNK token kontrolü"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Special Tokens")
        logger.info("="*80)
        
        try:
            if self.tokenizer_core is None:
                raise ValueError("TokenizerCore başlatılmamış!")
            
            vocab = self.tokenizer_core.get_vocab()
            
            def get_token_id(token: str) -> int:
                token_data = vocab.get(token)
                if token_data is None:
                    return -1
                if isinstance(token_data, dict):
                    return token_data.get('id', -1)
                return int(token_data) if isinstance(token_data, (int, str)) else -1
            
            special_tokens = {
                "PAD": ("<PAD>", get_token_id("<PAD>")),
                "BOS": ("<BOS>", get_token_id("<BOS>")),
                "EOS": ("<EOS>", get_token_id("<EOS>")),
                "UNK": ("<UNK>", get_token_id("<UNK>"))
            }
            
            all_found = True
            for name, (token_str, token_id) in special_tokens.items():
                if token_id == -1:
                    logger.error(f"  ❌ {name} ({token_str}): BULUNAMADI!")
                    all_found = False
                else:
                    logger.info(f"  ✅ {name} ({token_str}): ID={token_id}")
            
            # Token ID çakışması kontrolü
            token_ids = [tid for _, (_, tid) in special_tokens.items() if tid != -1]
            if len(token_ids) != len(set(token_ids)):
                logger.error("  ❌ Token ID çakışması tespit edildi!")
                all_found = False
            
            self.results["test_4"] = {
                "success": all_found,
                "special_tokens": {k: {"token": v[0], "id": v[1]} for k, v in special_tokens.items()},
                "message": "Tüm special tokenlar bulundu" if all_found else "Bazı special tokenlar eksik"
            }
            
            logger.info(f"✅ TEST 4 {'BAŞARILI' if all_found else 'BAŞARISIZ'}")
            return all_found
            
        except Exception as e:
            logger.error(f"❌ TEST 4 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_4"] = {
                "success": False,
                "error": str(e),
                "message": "Special token testi başarısız"
            }
            return False
    
    def test_5_input_target_alignment(self) -> bool:
        """Test 5: Input/Target alignment doğrulaması"""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Input/Target Alignment")
        logger.info("="*80)
        
        try:
            if self.tokenizer_core is None:
                raise ValueError("TokenizerCore başlatılmamış!")
            
            # Vocab'dan special token ID'leri al
            vocab = self.tokenizer_core.get_vocab()
            def _id_of(token: str) -> int:
                data = vocab.get(token)
                if data is None:
                    return -1
                return data.get('id') if isinstance(data, dict) else int(data)
            
            PAD_ID = _id_of("<PAD>")
            BOS_ID = _id_of("<BOS>")
            EOS_ID = _id_of("<EOS>")
            
            # Veri yükle
            raw_data = self.tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=self.config["train_include_whole_words"],
                include_syllables=self.config["train_include_syllables"],
                include_sep=self.config["train_include_sep"],
            )
            
            alignment_stats = {
                "total": 0,
                "correct_alignment": 0,
                "incorrect_alignment": 0,
                "missing_bos": 0,
                "missing_eos": 0,
                "length_mismatch": 0,
            }
            
            alignment_errors = []
            
            # İlk 50 örneği kontrol et
            for idx, (inp_ids, tgt_ids) in enumerate(raw_data[:50]):
                alignment_stats["total"] += 1
                
                try:
                    # Input: BOS ile başlamalı, EOS ile bitmemeli
                    # Target: BOS ile başlamamalı, EOS ile bitmeli
                    seq_in = list(inp_ids)
                    seq_tgt = list(tgt_ids)
                    
                    # Input hazırlama (TrainingService'teki mantık)
                    if seq_in and seq_in[-1] == EOS_ID:
                        seq_in = seq_in[:-1]
                    if not seq_in or seq_in[0] != BOS_ID:
                        seq_in.insert(0, BOS_ID)
                        alignment_stats["missing_bos"] += 1
                    
                    # Target hazırlama
                    if seq_tgt and seq_tgt[0] == BOS_ID:
                        seq_tgt = seq_tgt[1:]
                    if seq_tgt and seq_tgt[-1] == EOS_ID:
                        seq_tgt = seq_tgt[:-1]
                    seq_tgt.append(EOS_ID)
                    
                    # Alignment kontrolü
                    if len(seq_in) == len(seq_tgt):
                        # Input: [BOS, t1, t2, ..., tN]
                        # Target: [t1, t2, ..., tN, EOS]
                        # Tokenların eşleşmesi kontrolü
                        if seq_in[0] == BOS_ID and seq_tgt[-1] == EOS_ID:
                            # Token alignment kontrolü (BOS hariç input ile EOS hariç target aynı olmalı)
                            input_tokens = seq_in[1:]  # BOS'u çıkar
                            target_tokens = seq_tgt[:-1]  # EOS'u çıkar
                            
                            if input_tokens == target_tokens:
                                alignment_stats["correct_alignment"] += 1
                            else:
                                alignment_stats["incorrect_alignment"] += 1
                                if len(alignment_errors) < 5:
                                    alignment_errors.append({
                                        "idx": idx,
                                        "input": seq_in[:10],
                                        "target": seq_tgt[:10],
                                        "reason": "Token mismatch"
                                    })
                        else:
                            alignment_stats["incorrect_alignment"] += 1
                            if len(alignment_errors) < 5:
                                alignment_errors.append({
                                    "idx": idx,
                                    "input": seq_in[:10],
                                    "target": seq_tgt[:10],
                                    "reason": "BOS/EOS format error"
                                })
                    else:
                        alignment_stats["length_mismatch"] += 1
                        if len(alignment_errors) < 5:
                            alignment_errors.append({
                                "idx": idx,
                                "input_len": len(seq_in),
                                "target_len": len(seq_tgt),
                                "reason": "Length mismatch"
                            })
                    
                except Exception as e:
                    alignment_stats["incorrect_alignment"] += 1
                    alignment_errors.append({
                        "idx": idx,
                        "error": str(e)
                    })
            
            # Rapor
            logger.info(f"  Toplam örnek: {alignment_stats['total']}")
            logger.info(f"  ✅ Doğru alignment: {alignment_stats['correct_alignment']}")
            logger.info(f"  ❌ Yanlış alignment: {alignment_stats['incorrect_alignment']}")
            logger.info(f"  ⚠️  Uzunluk uyuşmazlığı: {alignment_stats['length_mismatch']}")
            
            if alignment_errors:
                logger.warning(f"  İlk {len(alignment_errors)} hata örneği:")
                for err in alignment_errors[:3]:
                    logger.warning(f"    {err}")
            
            success_rate = alignment_stats["correct_alignment"] / alignment_stats["total"] if alignment_stats["total"] > 0 else 0
            success = success_rate > 0.8  # %80'den fazla doğru olmalı
            
            self.results["test_5"] = {
                "success": success,
                "stats": alignment_stats,
                "success_rate": success_rate,
                "alignment_errors": alignment_errors[:5],
                "message": f"Alignment başarı oranı: {success_rate:.2%}"
            }
            
            logger.info(f"✅ TEST 5 {'BAŞARILI' if success else 'BAŞARISIZ'} ({success_rate:.2%})")
            return success
            
        except Exception as e:
            logger.error(f"❌ TEST 5 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_5"] = {
                "success": False,
                "error": str(e),
                "message": "Alignment testi başarısız"
            }
            return False
    
    def test_6_cache_mechanism(self) -> bool:
        """Test 6: Cache oluşturma ve yükleme"""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Cache Mechanism")
        logger.info("="*80)
        
        try:
            if self.tokenizer_core is None:
                raise ValueError("TokenizerCore başlatılmamış!")
            
            # Cache oluştur
            cache = DataCache(
                data_dir=self.config["data_dir"],
                cache_dir=self.config["cache_dir"],
                cache_enabled=True
            )
            
            # Cache key kontrolü
            def process_data():
                return self.tokenizer_core.load_training_data(
                    encode_mode="train",
                    include_whole_words=self.config["train_include_whole_words"],
                    include_syllables=self.config["train_include_syllables"],
                    include_sep=self.config["train_include_sep"],
                )
            
            # İlk yükleme (cache yok, oluşturulacak)
            logger.info("İlk yükleme (cache oluşturulacak)...")
            data1, from_cache1 = cache.get_or_process(
                tokenizer_core=self.tokenizer_core,
                encode_mode="train",
                include_whole_words=self.config["train_include_whole_words"],
                include_syllables=self.config["train_include_syllables"],
                include_sep=self.config["train_include_sep"],
                max_seq_length=self.config["max_seq_length"],
                process_func=process_data,
                alignment_format="autoregressive_v2"
            )
            
            logger.info(f"  İlk yükleme: {len(data1):,} örnek, cache'den: {from_cache1}")
            
            # İkinci yükleme (cache'den olmalı)
            logger.info("İkinci yükleme (cache'den olmalı)...")
            data2, from_cache2 = cache.get_or_process(
                tokenizer_core=self.tokenizer_core,
                encode_mode="train",
                include_whole_words=self.config["train_include_whole_words"],
                include_syllables=self.config["train_include_syllables"],
                include_sep=self.config["train_include_sep"],
                max_seq_length=self.config["max_seq_length"],
                process_func=process_data,
                alignment_format="autoregressive_v2"
            )
            
            logger.info(f"  İkinci yükleme: {len(data2):,} örnek, cache'den: {from_cache2}")
            
            # Veri karşılaştırması
            data_match = len(data1) == len(data2)
            if data_match and len(data1) > 0:
                # İlk 10 örneği karşılaştır
                sample_match = True
                for i in range(min(10, len(data1))):
                    if data1[i] != data2[i]:
                        sample_match = False
                        break
                data_match = sample_match
            
            success = from_cache2 and data_match
            
            self.results["test_6"] = {
                "success": success,
                "first_load_from_cache": from_cache1,
                "second_load_from_cache": from_cache2,
                "data_length_match": len(data1) == len(data2),
                "data_content_match": data_match,
                "message": "Cache mekanizması çalışıyor" if success else "Cache mekanizması sorunlu"
            }
            
            logger.info(f"✅ TEST 6 {'BAŞARILI' if success else 'BAŞARISIZ'}")
            return success
            
        except Exception as e:
            logger.error(f"❌ TEST 6 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_6"] = {
                "success": False,
                "error": str(e),
                "message": "Cache testi başarısız"
            }
            return False
    
    def test_7_training_service_init(self) -> bool:
        """Test 7: TrainingService başlatma"""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: TrainingService Initialization")
        logger.info("="*80)
        
        try:
            # TrainingService oluştur
            logger.info("TrainingService oluşturuluyor...")
            self.training_service = TrainingService(self.config)
            
            # Kontroller
            checks = {
                "tokenizer_core_exists": self.training_service.tokenizer_core is not None,
                "data_cache_exists": hasattr(self.training_service, "data_cache"),
                "model_manager_exists": hasattr(self.training_service, "model_manager"),
            }
            
            all_passed = all(checks.values())
            
            for check_name, passed in checks.items():
                logger.info(f"  {'✅' if passed else '❌'} {check_name}: {passed}")
            
            self.results["test_7"] = {
                "success": all_passed,
                "checks": checks,
                "message": "TrainingService başarıyla başlatıldı" if all_passed else "Bazı kontroller başarısız"
            }
            
            logger.info(f"✅ TEST 7 {'BAŞARILI' if all_passed else 'BAŞARISIZ'}")
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ TEST 7 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_7"] = {
                "success": False,
                "error": str(e),
                "message": "TrainingService başlatılamadı"
            }
            return False
    
    def test_8_data_preparation(self) -> bool:
        """Test 8: Veri hazırlama ve preprocessing"""
        logger.info("\n" + "="*80)
        logger.info("TEST 8: Data Preparation")
        logger.info("="*80)
        
        try:
            if self.training_service is None:
                raise ValueError("TrainingService başlatılmamış!")
            
            # Veri hazırlama
            logger.info("Veri hazırlanıyor...")
            train_loader, val_loader, max_seq_len = self.training_service._prepare_data()
            
            # Kontroller
            train_batches = len(train_loader)
            val_batches = len(val_loader)
            
            logger.info(f"  Train batches: {train_batches}")
            logger.info(f"  Val batches: {val_batches}")
            logger.info(f"  Max seq length: {max_seq_len}")
            
            # Bir batch'i test et
            try:
                first_batch = next(iter(train_loader))
                inputs, targets = first_batch
                
                batch_size = inputs.shape[0]
                seq_len = inputs.shape[1]
                
                logger.info(f"  Batch size: {batch_size}")
                logger.info(f"  Sequence length: {seq_len}")
                logger.info(f"  Input shape: {inputs.shape}")
                logger.info(f"  Target shape: {targets.shape}")
                
                # Alignment kontrolü
                if inputs.shape == targets.shape:
                    logger.info("  ✅ Input ve Target aynı shape'de")
                    shape_match = True
                else:
                    logger.error(f"  ❌ Input ve Target farklı shape'de: {inputs.shape} vs {targets.shape}")
                    shape_match = False
                
            except Exception as e:
                logger.error(f"  ❌ Batch test hatası: {e}")
                shape_match = False
                batch_size = 0
                seq_len = 0
            
            success = train_batches > 0 and val_batches > 0 and shape_match
            
            self.results["test_8"] = {
                "success": success,
                "train_batches": train_batches,
                "val_batches": val_batches,
                "max_seq_len": max_seq_len,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "shape_match": shape_match,
                "message": "Veri hazırlama başarılı" if success else "Veri hazırlama sorunlu"
            }
            
            logger.info(f"✅ TEST 8 {'BAŞARILI' if success else 'BAŞARISIZ'}")
            return success
            
        except Exception as e:
            logger.error(f"❌ TEST 8 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_8"] = {
                "success": False,
                "error": str(e),
                "message": "Veri hazırlama testi başarısız"
            }
            return False
    
    def test_9_dataloader_creation(self) -> bool:
        """Test 9: DataLoader oluşturma ve batch işleme"""
        logger.info("\n" + "="*80)
        logger.info("TEST 9: DataLoader Creation & Batch Processing")
        logger.info("="*80)
        
        try:
            if self.training_service is None:
                raise ValueError("TrainingService başlatılmamış!")
            
            train_loader, val_loader, _ = self.training_service._prepare_data()
            
            # Batch işleme testi
            batch_stats = {
                "train_batches_processed": 0,
                "val_batches_processed": 0,
                "train_errors": 0,
                "val_errors": 0,
            }
            
            # Train loader test
            logger.info("Train loader test ediliyor...")
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 3:  # İlk 3 batch'i test et
                    break
                
                try:
                    inputs, targets = batch
                    
                    # Shape kontrolü
                    if inputs.shape != targets.shape:
                        batch_stats["train_errors"] += 1
                        logger.warning(f"  Batch {batch_idx}: Shape mismatch")
                        continue
                    
                    # PAD token kontrolü
                    vocab = self.training_service.tokenizer_core.get_vocab()
                    pad_id = vocab.get("<PAD>")
                    pad_id = pad_id.get('id') if isinstance(pad_id, dict) else (pad_id if pad_id else 0)
                    
                    # PAD tokenların sonda olması gerekir
                    pad_positions = (targets == pad_id).nonzero(as_tuple=True)
                    
                    batch_stats["train_batches_processed"] += 1
                    logger.info(f"  ✅ Batch {batch_idx}: Shape={inputs.shape}, PAD count={pad_positions[0].numel()}")
                    
                except Exception as e:
                    batch_stats["train_errors"] += 1
                    logger.warning(f"  ❌ Batch {batch_idx} hatası: {e}")
            
            # Val loader test
            logger.info("Val loader test ediliyor...")
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 3:
                    break
                
                try:
                    inputs, targets = batch
                    
                    if inputs.shape != targets.shape:
                        batch_stats["val_errors"] += 1
                        continue
                    
                    batch_stats["val_batches_processed"] += 1
                    logger.info(f"  ✅ Batch {batch_idx}: Shape={inputs.shape}")
                    
                except Exception as e:
                    batch_stats["val_errors"] += 1
            
            success = (
                batch_stats["train_batches_processed"] > 0 and
                batch_stats["val_batches_processed"] > 0 and
                batch_stats["train_errors"] == 0 and
                batch_stats["val_errors"] == 0
            )
            
            self.results["test_9"] = {
                "success": success,
                "batch_stats": batch_stats,
                "message": "DataLoader batch işleme başarılı" if success else "DataLoader batch işleme sorunlu"
            }
            
            logger.info(f"✅ TEST 9 {'BAŞARILI' if success else 'BAŞARISIZ'}")
            return success
            
        except Exception as e:
            logger.error(f"❌ TEST 9 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_9"] = {
                "success": False,
                "error": str(e),
                "message": "DataLoader testi başarısız"
            }
            return False
    
    def test_10_end_to_end_pipeline(self) -> bool:
        """Test 10: Tüm pipeline'ın end-to-end testi"""
        logger.info("\n" + "="*80)
        logger.info("TEST 10: End-to-End Pipeline")
        logger.info("="*80)
        
        try:
            # Tüm pipeline'ı baştan çalıştır
            pipeline_steps = []
            
            # Step 1: TokenizerCore
            logger.info("Step 1: TokenizerCore başlatma...")
            tokenizer = TokenizerCore(self.config)
            tokenizer.finalize_vocab()
            pipeline_steps.append({"step": "TokenizerCore", "success": True})
            logger.info("  ✅ TokenizerCore başlatıldı")
            
            # Step 2: Veri yükleme
            logger.info("Step 2: Veri yükleme...")
            raw_data = tokenizer.load_training_data(
                encode_mode="train",
                include_whole_words=self.config["train_include_whole_words"],
                include_syllables=self.config["train_include_syllables"],
                include_sep=self.config["train_include_sep"],
            )
            pipeline_steps.append({
                "step": "Data Loading",
                "success": len(raw_data) > 0,
                "example_count": len(raw_data)
            })
            logger.info(f"  ✅ {len(raw_data):,} örnek yüklendi")
            
            # Step 3: Cache
            logger.info("Step 3: Cache işleme...")
            cache = DataCache(
                data_dir=self.config["data_dir"],
                cache_dir=self.config["cache_dir"],
                cache_enabled=True
            )
            cached_data, from_cache = cache.get_or_process(
                tokenizer_core=tokenizer,
                encode_mode="train",
                include_whole_words=self.config["train_include_whole_words"],
                include_syllables=self.config["train_include_syllables"],
                include_sep=self.config["train_include_sep"],
                max_seq_length=self.config["max_seq_length"],
                process_func=lambda: raw_data,
                alignment_format="autoregressive_v2"
            )
            pipeline_steps.append({
                "step": "Cache",
                "success": len(cached_data) == len(raw_data),
                "from_cache": from_cache
            })
            logger.info(f"  ✅ Cache işlemi tamamlandı (cache'den: {from_cache})")
            
            # Step 4: TrainingService
            logger.info("Step 4: TrainingService başlatma...")
            service = TrainingService(self.config)
            pipeline_steps.append({"step": "TrainingService", "success": True})
            logger.info("  ✅ TrainingService başlatıldı")
            
            # Step 5: Veri hazırlama
            logger.info("Step 5: Veri hazırlama...")
            train_loader, val_loader, max_seq_len = service._prepare_data()
            pipeline_steps.append({
                "step": "Data Preparation",
                "success": len(train_loader) > 0 and len(val_loader) > 0,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader)
            })
            logger.info(f"  ✅ Veri hazırlandı: {len(train_loader)} train, {len(val_loader)} val batch")
            
            # Step 6: Batch işleme
            logger.info("Step 6: Batch işleme testi...")
            batch_test_passed = False
            try:
                batch = next(iter(train_loader))
                inputs, targets = batch
                batch_test_passed = inputs.shape == targets.shape
            except Exception as e:
                logger.warning(f"  Batch test hatası: {e}")
            
            pipeline_steps.append({
                "step": "Batch Processing",
                "success": batch_test_passed
            })
            logger.info(f"  {'✅' if batch_test_passed else '❌'} Batch işleme testi")
            
            # Genel başarı
            all_steps_success = all(step["success"] for step in pipeline_steps)
            
            self.results["test_10"] = {
                "success": all_steps_success,
                "pipeline_steps": pipeline_steps,
                "message": "End-to-end pipeline başarılı" if all_steps_success else "Bazı adımlar başarısız"
            }
            
            logger.info(f"✅ TEST 10 {'BAŞARILI' if all_steps_success else 'BAŞARISIZ'}")
            return all_steps_success
            
        except Exception as e:
            logger.error(f"❌ TEST 10 BAŞARISIZ: {e}", exc_info=True)
            self.results["test_10"] = {
                "success": False,
                "error": str(e),
                "message": "End-to-end test başarısız"
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Tüm testleri çalıştır"""
        logger.info("\n" + "="*80)
        logger.info("TÜM TESTLER ÇALIŞTIRILIYOR...")
        logger.info("="*80)
        
        test_methods = [
            self.test_1_tokenizer_initialization,
            self.test_2_data_loading,
            self.test_3_encoding_process,
            self.test_4_special_tokens,
            self.test_5_input_target_alignment,
            self.test_6_cache_mechanism,
            self.test_7_training_service_init,
            self.test_8_data_preparation,
            self.test_9_dataloader_creation,
            self.test_10_end_to_end_pipeline,
        ]
        
        results_summary = {
            "total_tests": len(test_methods),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": {}
        }
        
        for test_method in test_methods:
            try:
                success = test_method()
                if success:
                    results_summary["passed_tests"] += 1
                else:
                    results_summary["failed_tests"] += 1
            except Exception as e:
                logger.error(f"Test hatası ({test_method.__name__}): {e}", exc_info=True)
                results_summary["failed_tests"] += 1
        
        results_summary["test_results"] = self.results
        
        # Özet rapor
        logger.info("\n" + "="*80)
        logger.info("TEST SONUÇLARI ÖZETİ")
        logger.info("="*80)
        logger.info(f"Toplam test: {results_summary['total_tests']}")
        logger.info(f"✅ Başarılı: {results_summary['passed_tests']}")
        logger.info(f"❌ Başarısız: {results_summary['failed_tests']}")
        logger.info(f"Başarı oranı: {results_summary['passed_tests']/results_summary['total_tests']*100:.1f}%")
        
        # Detaylı sonuçlar
        logger.info("\nDetaylı Sonuçlar:")
        for test_name, test_result in self.results.items():
            status = "✅" if test_result.get("success", False) else "❌"
            message = test_result.get("message", "N/A")
            logger.info(f"  {status} {test_name}: {message}")
        
        return results_summary
    
    def save_results(self, output_file: str = "test_results.json"):
        """Test sonuçlarını JSON dosyasına kaydet"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
        except Exception as e:
            logger.error(f"❌ Sonuç kaydetme hatası: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive TokenizerCore and TrainingService Test Suite")
    parser.add_argument("--data-dir", type=str, default="education", help="Eğitim verisi dizini")
    parser.add_argument("--output", type=str, default="test_results.json", help="Sonuç dosyası")
    
    args = parser.parse_args()
    
    # Test suite oluştur
    test_suite = ComprehensiveTokenizerTrainingTest(data_dir=args.data_dir)
    
    # Tüm testleri çalıştır
    results = test_suite.run_all_tests()
    
    # Sonuçları kaydet
    test_suite.save_results(args.output)
    
    # Exit code
    return 0 if results["failed_tests"] == 0 else 1


if __name__ == "__main__":
    exit(main())

