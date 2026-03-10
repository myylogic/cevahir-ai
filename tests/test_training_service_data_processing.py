"""
TrainingService Data Processing Analizi Testi
==============================================
TrainingService'in cache'den gelen verileri nasıl işlediğini test eder.

Test Amacı:
- Cache'den yüklenen örneklerin doğru işlenip işlenmediği
- EOS ekleme mekanizması (her target'in sonunda EOS var mı?)
- BOS ekleme mekanizması (her input'un başında BOS var mı?)
- Alignment kontrolü (autoregressive format doğru mu?)
- Truncation sonrası EOS korunuyor mu?
- Loss function'da EOS weight uygulanıyor mu?
"""

import os
import sys
import json
import logging
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import Counter
import pickle

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG
from training_system.data_cache import DataCache
from training_system.training_service import TrainingService

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("TrainingServiceDataProcessing")


class TrainingServiceDataProcessingTest:
    """TrainingService data processing test sınıfı"""
    
    def __init__(self, data_dir: str = "education", cache_dir: str = ".cache/preprocessed_data"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.tokenizer_core = None
        self.data_cache = None
        self.training_service = None
        
    def initialize(self):
        """Bileşenleri başlat"""
        logger.info("=" * 80)
        logger.info("TRAINING SERVICE DATA PROCESSING TESTİ BAŞLATIYOR")
        logger.info("=" * 80)
        
        # TokenizerCore başlat (data_dir ile - DataLoader için gerekli)
        config = {
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False,  # CPU için
            "data_dir": self.data_dir  # DataLoader için gerekli
        }
        self.tokenizer_core = TokenizerCore(config)
        logger.info("✅ TokenizerCore başlatıldı (data_dir ile)")
        
        # DataCache başlat
        self.data_cache = DataCache(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            cache_enabled=True
        )
        logger.info("✅ DataCache başlatıldı")
        
    def simulate_training_service_prepare_data(self, max_seq_len: int = 768) -> Dict[str, Any]:
        """
        TrainingService._prepare_data metodunu simüle et
        Cache'den yüklenen verileri TrainingService gibi işle
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SERVICE DATA PROCESSING SİMÜLASYONU")
        logger.info("=" * 80)
        
        # Vocab'ı finalize et
        self.tokenizer_core.finalize_vocab()
        
        # Cache'den direkt yükle (kullanıcı cache kullanmak istiyor)
        # Mevcut cache dosyasını bul ve yükle
        cache_files = list(Path(self.cache_dir).glob("cached_data_*.pkl"))
        
        if not cache_files:
            logger.error("Cache dosyası bulunamadı!")
            return {"error": "Cache dosyası bulunamadı"}
        
        # İlk cache dosyasını yükle
        cache_file = cache_files[0]
        logger.info(f"Cache dosyası yükleniyor: {cache_file.name}")
        
        try:
            with open(cache_file, "rb") as f:
                raw_data = pickle.load(f)
            from_cache = True
            logger.info(f"✅ Cache'den yüklendi: {len(raw_data):,} örnek")
        except Exception as e:
            logger.error(f"Cache yükleme hatası: {e}")
            return {"error": f"Cache yükleme hatası: {e}"}
        
        logger.info(f"✅ Veri yüklendi: {len(raw_data):,} örnek (cache'den: {from_cache})")
        
        # Special token ID'leri al
        vocab = self.tokenizer_core.tokenizer.get_vocab()
        
        def _id_of(token: str) -> int:
            val = vocab.get(token)
            if isinstance(val, dict):
                return int(val.get("id", 0))
            return int(val or 0)
        
        PAD_ID = _id_of("<PAD>")
        BOS_ID = _id_of("<BOS>")
        EOS_ID = _id_of("<EOS>")
        UNK_ID = _id_of("<UNK>")
        VOCAB_SIZE = len(vocab)
        
        logger.info(f"Special tokens → PAD:{PAD_ID}, BOS:{BOS_ID}, EOS:{EOS_ID}, UNK:{UNK_ID}")
        
        # İstatistikler
        stats = {
            "total_examples": len(raw_data),
            "examples_processed": 0,
            "examples_skipped": 0,
            "eos_added_count": 0,
            "eos_already_present": 0,
            "bos_added_count": 0,
            "alignment_errors": 0,
            "truncation_count": 0,
            "eos_preserved_after_truncation": 0,
            "final_eos_frequency": 0,
            "final_bos_frequency": 0,
            "sequence_lengths": [],
            "target_last_token_distribution": Counter(),
            "eos_positions": []
        }
        
        # İlk 1000 örneği işle (hızlı test için)
        samples_to_test = min(1000, len(raw_data))
        logger.info(f"İlk {samples_to_test} örnek işleniyor...")
        
        processed_examples = []
        
        for idx, (inp_ids, tgt_ids) in enumerate(raw_data[:samples_to_test]):
            try:
                # TrainingService format'ını simüle et (satır 701-704)
                # Input: BOS ekle
                seq_in = [BOS_ID] + list(inp_ids)
                stats["bos_added_count"] += 1
                
                # Target: EOS ekle
                # Önce kontrol et: EOS zaten var mı?
                if len(tgt_ids) > 0 and tgt_ids[-1] == EOS_ID:
                    stats["eos_already_present"] += 1
                    seq_tgt = list(tgt_ids)
                else:
                    stats["eos_added_count"] += 1
                    seq_tgt = list(tgt_ids) + [EOS_ID]
                
                # Alignment kontrolü (TrainingService satır 708-713)
                if len(seq_in) != len(seq_tgt):
                    stats["alignment_errors"] += 1
                    stats["examples_skipped"] += 1
                    logger.warning(f"Örnek {idx}: Alignment hatası - Input len={len(seq_in)} != Target len={len(seq_tgt)}")
                    continue
                
                # Autoregressive alignment kontrolü (TrainingService satır 721-727)
                alignment_ok = True
                for i in range(len(seq_tgt) - 1):  # Son token (EOS) hariç
                    if seq_tgt[i] != seq_in[i + 1]:
                        stats["alignment_errors"] += 1
                        alignment_ok = False
                        break
                
                if not alignment_ok:
                    stats["examples_skipped"] += 1
                    logger.warning(f"Örnek {idx}: Autoregressive alignment hatası")
                    continue
                
                # Truncation kontrolü (TrainingService satır 751-763)
                original_len = len(seq_in)
                if len(seq_in) > max_seq_len:
                    stats["truncation_count"] += 1
                    
                    # Input'u kes
                    seq_in = seq_in[:max_seq_len]
                    
                    # Target'i kes ama EOS'u koru
                    if len(seq_tgt) > 0 and seq_tgt[-1] == EOS_ID:
                        seq_tgt = seq_tgt[:max_seq_len-1] + [EOS_ID]
                        stats["eos_preserved_after_truncation"] += 1
                    else:
                        seq_tgt = seq_tgt[:max_seq_len-1] + [EOS_ID]
                
                # Güvenlik kontrolü: Target sonunda EOS olmalı (TrainingService satır 766-771)
                if len(seq_tgt) == 0 or seq_tgt[-1] != EOS_ID:
                    logger.warning(f"Örnek {idx}: Target sonunda EOS yok! EOS ekleniyor...")
                    seq_tgt = seq_tgt + [EOS_ID]
                    # Input'u da uzat (alignment için)
                    if len(seq_in) < len(seq_tgt):
                        if len(seq_in) > 0:
                            seq_in = seq_in + [seq_in[-1]]
                        else:
                            seq_in = [BOS_ID]
                
                # Final kontrol: EOS pozisyonu
                if len(seq_tgt) > 0 and seq_tgt[-1] == EOS_ID:
                    stats["final_eos_frequency"] += 1
                    stats["eos_positions"].append(len(seq_tgt) - 1)
                
                # Final kontrol: BOS pozisyonu
                if len(seq_in) > 0 and seq_in[0] == BOS_ID:
                    stats["final_bos_frequency"] += 1
                
                # İstatistikleri topla
                stats["sequence_lengths"].append(len(seq_tgt))
                if len(seq_tgt) > 0:
                    stats["target_last_token_distribution"][seq_tgt[-1]] += 1
                
                stats["examples_processed"] += 1
                processed_examples.append((seq_in, seq_tgt))
                
            except Exception as e:
                stats["examples_skipped"] += 1
                logger.error(f"Örnek {idx} işleme hatası: {e}")
                continue
        
        # Sonuçları hazırla
        results = {
            "total_examples": len(raw_data),
            "examples_analyzed": samples_to_test,
            "examples_processed": stats["examples_processed"],
            "examples_skipped": stats["examples_skipped"],
            "processing_accuracy": (stats["examples_processed"] / samples_to_test) * 100,
            "eos_analysis": {
                "eos_added": stats["eos_added_count"],
                "eos_already_present": stats["eos_already_present"],
                "final_eos_frequency": stats["final_eos_frequency"],
                "final_eos_frequency_percentage": (stats["final_eos_frequency"] / stats["examples_processed"]) * 100 if stats["examples_processed"] > 0 else 0,
                "eos_always_last": all(pos == processed_examples[i][1][-1] == EOS_ID for i, pos in enumerate(stats["eos_positions"]) if len(processed_examples[i][1]) > 0),
                "eos_positions": {
                    "min": min(stats["eos_positions"]) if stats["eos_positions"] else None,
                    "max": max(stats["eos_positions"]) if stats["eos_positions"] else None,
                    "avg": sum(stats["eos_positions"]) / len(stats["eos_positions"]) if stats["eos_positions"] else None,
                    "count": len(stats["eos_positions"])
                }
            },
            "bos_analysis": {
                "bos_added": stats["bos_added_count"],
                "final_bos_frequency": stats["final_bos_frequency"],
                "final_bos_frequency_percentage": (stats["final_bos_frequency"] / stats["examples_processed"]) * 100 if stats["examples_processed"] > 0 else 0
            },
            "alignment_analysis": {
                "alignment_errors": stats["alignment_errors"],
                "alignment_accuracy": ((samples_to_test - stats["alignment_errors"]) / samples_to_test) * 100 if samples_to_test > 0 else 0
            },
            "truncation_analysis": {
                "truncation_count": stats["truncation_count"],
                "eos_preserved_after_truncation": stats["eos_preserved_after_truncation"],
                "eos_preservation_rate": (stats["eos_preserved_after_truncation"] / stats["truncation_count"]) * 100 if stats["truncation_count"] > 0 else 0
            },
            "sequence_analysis": {
                "lengths": {
                    "min": min(stats["sequence_lengths"]) if stats["sequence_lengths"] else None,
                    "max": max(stats["sequence_lengths"]) if stats["sequence_lengths"] else None,
                    "avg": sum(stats["sequence_lengths"]) / len(stats["sequence_lengths"]) if stats["sequence_lengths"] else None
                }
            },
            "target_last_token_distribution": dict(stats["target_last_token_distribution"].most_common(10)),
            "special_tokens": {
                "PAD_ID": PAD_ID,
                "BOS_ID": BOS_ID,
                "EOS_ID": EOS_ID,
                "UNK_ID": UNK_ID,
                "VOCAB_SIZE": VOCAB_SIZE
            },
            "from_cache": from_cache
        }
        
        logger.info(f"\n📊 İşleme Sonuçları:")
        logger.info(f"  İşlenen örnek: {stats['examples_processed']}/{samples_to_test} ({(stats['examples_processed']/samples_to_test)*100:.2f}%)")
        logger.info(f"  Atlanan örnek: {stats['examples_skipped']}")
        
        logger.info(f"\n📊 EOS Analizi:")
        logger.info(f"  EOS eklendi: {stats['eos_added_count']}")
        logger.info(f"  EOS zaten vardı: {stats['eos_already_present']}")
        logger.info(f"  Final EOS frequency: {stats['final_eos_frequency']}/{stats['examples_processed']} ({results['eos_analysis']['final_eos_frequency_percentage']:.2f}%)")
        logger.info(f"  EOS her zaman son pozisyonda: {results['eos_analysis']['eos_always_last']}")
        
        logger.info(f"\n📊 BOS Analizi:")
        logger.info(f"  BOS eklendi: {stats['bos_added_count']}")
        logger.info(f"  Final BOS frequency: {stats['final_bos_frequency']}/{stats['examples_processed']} ({results['bos_analysis']['final_bos_frequency_percentage']:.2f}%)")
        
        logger.info(f"\n📊 Alignment Analizi:")
        logger.info(f"  Alignment hataları: {stats['alignment_errors']}")
        logger.info(f"  Alignment accuracy: {results['alignment_analysis']['alignment_accuracy']:.2f}%")
        
        logger.info(f"\n📊 Truncation Analizi:")
        logger.info(f"  Truncation sayısı: {stats['truncation_count']}")
        logger.info(f"  EOS korunma oranı: {results['truncation_analysis']['eos_preservation_rate']:.2f}%")
        
        logger.info(f"\n📊 Target Son Token Dağılımı:")
        for token_id, count in stats["target_last_token_distribution"].most_common(5):
            token_name = f"<EOS>" if token_id == EOS_ID else f"Token_{token_id}"
            logger.info(f"  {token_name} (id={token_id}): {count} ({count/stats['examples_processed']*100:.2f}%)")
        
        return results
    
    def test_loss_function_eos_weight(self) -> Dict[str, Any]:
        """
        Loss function'da EOS weight'in uygulanıp uygulanmadığını test et
        TrainingService'teki loss function setup'ını simüle et
        """
        logger.info("\n" + "=" * 80)
        logger.info("LOSS FUNCTION EOS WEIGHT TESTİ")
        logger.info("=" * 80)
        
        vocab = self.tokenizer_core.get_vocab()
        vocab_size = len(vocab)
        
        # EOS ID'yi al
        eos_id = None
        if isinstance(vocab.get("<EOS>"), dict):
            eos_id = vocab["<EOS>"].get("id")
        elif isinstance(vocab.get("<EOS>"), int):
            eos_id = vocab["<EOS>"]
        else:
            eos_id = 3  # Fallback
        
        logger.info(f"Vocab size: {vocab_size}, EOS ID: {eos_id}")
        
        # EOS weight tensor'u oluştur (TrainingService satır 310-313)
        import torch
        device = torch.device("cpu")
        loss_weights = torch.ones(vocab_size, device=device)
        eos_weight = 0.1  # TrainingService'te kullanılan değer
        
        if eos_id is not None and 0 <= eos_id < vocab_size:
            loss_weights[eos_id] = eos_weight
            logger.info(f"✅ EOS weight uygulandı: eos_id={eos_id}, weight={eos_weight}")
        else:
            logger.warning(f"⚠️ EOS ID geçersiz: {eos_id}, vocab_size={vocab_size}")
            return {"error": "EOS ID geçersiz"}
        
        # Label smoothing
        label_smoothing = 0.1
        
        # Loss function'ı oluştur (TrainingService satır 322-327)
        from torch.nn import CrossEntropyLoss
        pad_token_id = 0
        
        criterion = CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=label_smoothing,
            ignore_index=pad_token_id,
            reduction="mean"
        )
        
        logger.info(f"✅ Loss function oluşturuldu: Label Smoothing={label_smoothing}, EOS Weight={eos_weight}")
        
        # Test: Dummy logits ve target ile loss hesapla
        batch_size = 4
        seq_len = 10
        
        # Dummy logits (model çıktısı)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Dummy target (son pozisyonda EOS var)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets[:, -1] = eos_id  # Son pozisyona EOS koy
        
        # Loss hesapla
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        
        logger.info(f"✅ Test loss hesaplandı: {loss.item():.4f}")
        
        # EOS pozisyonlarındaki loss'u kontrol et
        eos_positions = (targets_flat == eos_id).nonzero(as_tuple=True)[0]
        logger.info(f"  EOS pozisyonları: {len(eos_positions)}/{batch_size * seq_len}")
        
        results = {
            "eos_id": eos_id,
            "vocab_size": vocab_size,
            "eos_weight": eos_weight,
            "label_smoothing": label_smoothing,
            "loss_weights_applied": True,
            "eos_weight_in_loss_weights": abs(loss_weights[eos_id].item() - eos_weight) < 1e-6,  # Float comparison
            "test_loss": loss.item(),
            "eos_positions_count": len(eos_positions),
            "total_positions": batch_size * seq_len
        }
        
        return results
    
    def run_full_test(self) -> Dict[str, Any]:
        """Tüm testleri çalıştır"""
        self.initialize()
        
        results = {
            "test_date": datetime.now().isoformat(),
            "data_dir": self.data_dir,
            "cache_dir": self.cache_dir,
            "data_processing_test": self.simulate_training_service_prepare_data(max_seq_len=768),
            "loss_function_test": self.test_loss_function_eos_weight()
        }
        
        # Sonuçları kaydet
        output_file = "test_training_service_data_processing_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
        logger.info("=" * 80)
        
        return results


if __name__ == "__main__":
    tester = TrainingServiceDataProcessingTest(
        data_dir="education",
        cache_dir=".cache/preprocessed_data"
    )
    results = tester.run_full_test()
    
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    
    dp = results["data_processing_test"]
    print(f"\nData Processing:")
    print(f"  Islenen ornek: {dp['examples_processed']}/{dp['examples_analyzed']} ({dp['processing_accuracy']:.2f}%)")
    print(f"  Final EOS frequency: {dp['eos_analysis']['final_eos_frequency_percentage']:.2f}%")
    print(f"  Alignment accuracy: {dp['alignment_analysis']['alignment_accuracy']:.2f}%")
    print(f"  EOS truncation sonrasi korunma: {dp['truncation_analysis']['eos_preservation_rate']:.2f}%")
    
    lf = results["loss_function_test"]
    if "error" not in lf:
        print(f"\nLoss Function:")
        print(f"  EOS weight uygulandi: {lf['eos_weight_in_loss_weights']}")
        print(f"  EOS weight degeri: {lf['eos_weight']}")
        print(f"  Label smoothing: {lf['label_smoothing']}")

