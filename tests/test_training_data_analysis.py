"""
Training Data Analizi Testi
============================
Eğitim verisini analiz eder - EOS frequency, token distribution, anlamsız karakterler.

Test Amacı:
- Training data'da EOS frequency kontrolü
- Token distribution analizi
- Anlamsız karakter/harf kombinasyonları tespiti
- EOS pozisyon analizi (her zaman son pozisyonda mı?)
"""

import os
import sys
import json
import logging
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import Counter, defaultdict

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG
from data_loader_management.data_loader_manager import DataLoaderManager

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("TrainingDataAnalysis")


class TrainingDataAnalysis:
    """Training data analiz sınıfı"""
    
    def __init__(self, data_dir: str = "education"):
        self.data_dir = data_dir
        self.tokenizer_core = None
        self.analysis_results = {}
        
    def initialize(self):
        """Tokenizer ve data loader'ı başlat"""
        logger.info("=" * 80)
        logger.info("TRAINING DATA ANALİZ TESTİ BAŞLATIYOR")
        logger.info("=" * 80)
        
        # TokenizerCore başlat
        config = {
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False  # CPU için
        }
        self.tokenizer_core = TokenizerCore(config)
        logger.info("✅ TokenizerCore başlatıldı")
        
    def analyze_raw_data(self) -> Dict[str, Any]:
        """Ham training data'yı analiz et"""
        logger.info("\n" + "=" * 80)
        logger.info("HAM DATA ANALİZİ")
        logger.info("=" * 80)
        
        # DataLoaderManager ile raw data yükle
        data_loader = DataLoaderManager(
            data_dir=self.data_dir,
            file_types=["json", "txt", "docx"],
            mode="raw_text"
        )
        
        raw_texts = data_loader.load()
        logger.info(f"Yüklenen raw text sayısı: {len(raw_texts)}")
        
        # İstatistikler
        total_chars = 0
        total_words = 0
        char_frequency = Counter()
        word_frequency = Counter()
        anlamsiz_patterns = []
        
        # Her text için analiz
        for text in raw_texts[:100]:  # İlk 100'ü analiz et (hızlı test için)
            total_chars += len(text)
            words = text.split()
            total_words += len(words)
            
            # Karakter frekansı
            char_frequency.update(text.lower())
            
            # Kelime frekansı
            word_frequency.update([w.lower() for w in words])
            
            # Anlamsız pattern kontrolü (tekrar eden harfler, karakter kombinasyonları)
            if len(text) > 10:
                # Tekrar eden harfler (örn: "aaaa", "bbbb")
                for i in range(len(text) - 3):
                    if text[i] == text[i+1] == text[i+2] == text[i+3] and text[i].isalpha():
                        anlamsiz_patterns.append(text[max(0, i-5):min(len(text), i+10)])
        
        results = {
            "total_texts_analyzed": min(100, len(raw_texts)),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_text": total_chars / min(100, len(raw_texts)),
            "avg_words_per_text": total_words / min(100, len(raw_texts)),
            "top_10_chars": dict(char_frequency.most_common(10)),
            "top_20_words": dict(word_frequency.most_common(20)),
            "anlamsiz_patterns_count": len(anlamsiz_patterns),
            "anlamsiz_patterns_samples": anlamsiz_patterns[:10]  # İlk 10 örnek
        }
        
        logger.info(f"Toplam karakter: {total_chars}")
        logger.info(f"Toplam kelime: {total_words}")
        logger.info(f"En sık kullanılan 10 karakter: {dict(char_frequency.most_common(10))}")
        logger.info(f"En sık kullanılan 20 kelime: {dict(word_frequency.most_common(20))}")
        logger.info(f"Anlamsız pattern sayısı: {len(anlamsiz_patterns)}")
        
        return results
    
    def analyze_encoded_data(self) -> Dict[str, Any]:
        """Encoding sonrası training data'yı analiz et"""
        logger.info("\n" + "=" * 80)
        logger.info("ENCODED DATA ANALİZİ")
        logger.info("=" * 80)
        
        # Training data'yı yükle (encoding ile)
        try:
            training_examples = self.tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=True,
                include_syllables=True,
                include_sep=True
            )
        except Exception as e:
            logger.error(f"Training data yükleme hatası: {e}")
            return {"error": str(e)}
        
        logger.info(f"Yüklenen training örnek sayısı: {len(training_examples)}")
        
        # İstatistikler
        eos_positions = []  # EOS pozisyonları (target sequence içinde)
        eos_frequency = 0
        token_frequency = Counter()
        sequence_lengths = []
        target_last_tokens = Counter()  # Target'ın son token'ı (EOS olmalı)
        
        # İlk 1000 örneği analiz et
        for inp_ids, tgt_ids in training_examples[:1000]:
            seq_len = len(tgt_ids)
            sequence_lengths.append(seq_len)
            
            # EOS pozisyonu kontrolü
            if len(tgt_ids) > 0:
                last_token = tgt_ids[-1]
                target_last_tokens[last_token] += 1
                
                # EOS ID'yi al
                vocab = self.tokenizer_core.get_vocab()
                eos_id = None
                if isinstance(vocab.get("<EOS>"), dict):
                    eos_id = vocab["<EOS>"].get("id")
                elif isinstance(vocab.get("<EOS>"), int):
                    eos_id = vocab["<EOS>"]
                else:
                    eos_id = 3  # Fallback
                
                if last_token == eos_id:
                    eos_frequency += 1
                    eos_positions.append(seq_len - 1)  # Son pozisyon
                
                # Token frequency
                token_frequency.update(tgt_ids)
            
        # EOS her zaman son pozisyonda mı?
        eos_always_last = all(pos == len(tgt_ids) - 1 for pos in eos_positions) if eos_positions else False
        
        results = {
            "total_examples_analyzed": min(1000, len(training_examples)),
            "eos_frequency": eos_frequency,
            "eos_frequency_percentage": (eos_frequency / min(1000, len(training_examples))) * 100,
            "eos_always_last": eos_always_last,
            "eos_positions": {
                "min": min(eos_positions) if eos_positions else None,
                "max": max(eos_positions) if eos_positions else None,
                "avg": sum(eos_positions) / len(eos_positions) if eos_positions else None
            },
            "sequence_lengths": {
                "min": min(sequence_lengths) if sequence_lengths else None,
                "max": max(sequence_lengths) if sequence_lengths else None,
                "avg": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else None
            },
            "top_20_tokens": dict(token_frequency.most_common(20)),
            "target_last_token_distribution": dict(target_last_tokens.most_common(10))
        }
        
        logger.info(f"EOS frequency: {eos_frequency}/{min(1000, len(training_examples))} ({results['eos_frequency_percentage']:.2f}%)")
        logger.info(f"EOS her zaman son pozisyonda mı: {eos_always_last}")
        logger.info(f"En sık kullanılan 20 token: {dict(token_frequency.most_common(20))}")
        logger.info(f"Target son token dağılımı: {dict(target_last_tokens.most_common(10))}")
        
        return results
    
    def analyze_training_format(self) -> Dict[str, Any]:
        """Training format doğrulaması (BOS/EOS alignment)"""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING FORMAT DOĞRULAMASI")
        logger.info("=" * 80)
        
        # TrainingService format'ını simüle et
        try:
            training_examples = self.tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=True,
                include_syllables=True,
                include_sep=True
            )
        except Exception as e:
            logger.error(f"Training data yükleme hatası: {e}")
            return {"error": str(e)}
        
        # BOS ve EOS ID'leri
        vocab = self.tokenizer_core.get_vocab()
        bos_id = None
        eos_id = None
        
        if isinstance(vocab.get("<BOS>"), dict):
            bos_id = vocab["<BOS>"].get("id")
        elif isinstance(vocab.get("<BOS>"), int):
            bos_id = vocab["<BOS>"]
        else:
            bos_id = 2  # Fallback
        
        if isinstance(vocab.get("<EOS>"), dict):
            eos_id = vocab["<EOS>"].get("id")
        elif isinstance(vocab.get("<EOS>"), int):
            eos_id = vocab["<EOS>"]
        else:
            eos_id = 3  # Fallback
        
        # TrainingService format'ını simüle et (satır 701-704)
        alignment_errors = 0
        correct_alignments = 0
        
        for inp_ids, tgt_ids in training_examples[:100]:
            # TrainingService format: Input = [BOS] + inp_ids, Target = tgt_ids + [EOS]
            seq_in = [bos_id] + list(inp_ids)
            seq_tgt = list(tgt_ids) + [eos_id]
            
            # Alignment kontrolü: Target[i] == Input[i+1] (i < len(tgt_ids)-1)
            # Son token hariç (EOS)
            is_aligned = True
            for i in range(len(seq_tgt) - 1):
                if seq_tgt[i] != seq_in[i + 1]:
                    is_aligned = False
                    break
            
            if is_aligned:
                correct_alignments += 1
            else:
                alignment_errors += 1
        
        results = {
            "total_examples_checked": min(100, len(training_examples)),
            "correct_alignments": correct_alignments,
            "alignment_errors": alignment_errors,
            "alignment_accuracy": (correct_alignments / min(100, len(training_examples))) * 100,
            "bos_id": bos_id,
            "eos_id": eos_id
        }
        
        logger.info(f"Alignment accuracy: {results['alignment_accuracy']:.2f}%")
        logger.info(f"Doğru alignment: {correct_alignments}/{min(100, len(training_examples))}")
        logger.info(f"Alignment hatası: {alignment_errors}")
        
        return results
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Tüm analizleri çalıştır"""
        self.initialize()
        
        results = {
            "test_date": datetime.now().isoformat(),
            "raw_data_analysis": self.analyze_raw_data(),
            "encoded_data_analysis": self.analyze_encoded_data(),
            "training_format_analysis": self.analyze_training_format()
        }
        
        # Sonuçları kaydet
        output_file = "test_training_data_analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
        logger.info("=" * 80)
        
        return results


if __name__ == "__main__":
    analyzer = TrainingDataAnalysis(data_dir="education")
    results = analyzer.run_full_analysis()
    
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    print(f"Raw data analizi: {results['raw_data_analysis'].get('total_texts_analyzed', 0)} text")
    print(f"Encoded data analizi: {results['encoded_data_analysis'].get('total_examples_analyzed', 0)} örnek")
    print(f"EOS frequency: {results['encoded_data_analysis'].get('eos_frequency_percentage', 0):.2f}%")
    print(f"Alignment accuracy: {results['training_format_analysis'].get('alignment_accuracy', 0):.2f}%")

