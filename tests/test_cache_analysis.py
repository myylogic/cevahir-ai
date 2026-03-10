"""
Cache Analiz Testi
==================
Cache'den yüklenen eğitim örneklerini analiz eder - format, EOS pozisyonu, alignment kontrolü.

Test Amacı:
- Cache'deki verilerin formatını kontrol et
- EOS pozisyon analizi (cache'de doğru mu?)
- Alignment kontrolü (BOS/EOS alignment)
- Token distribution (cache'de yanlış pattern var mı?)
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import Counter, defaultdict

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG
from training_system.data_cache import DataCache

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("CacheAnalysis")


class CacheAnalysis:
    """Cache analiz sınıfı"""
    
    def __init__(self, cache_dir: str = ".cache/preprocessed_data", data_dir: str = "education"):
        self.cache_dir = Path(cache_dir)
        self.data_dir = data_dir
        self.tokenizer_core = None
        
    def initialize(self):
        """Tokenizer başlat"""
        logger.info("=" * 80)
        logger.info("CACHE ANALİZ TESTİ BAŞLATIYOR")
        logger.info("=" * 80)
        
        # TokenizerCore başlat
        config = {
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False  # CPU için
        }
        self.tokenizer_core = TokenizerCore(config)
        logger.info("✅ TokenizerCore başlatıldı")
        
    def find_cache_files(self) -> List[Path]:
        """Cache dosyalarını bul"""
        if not self.cache_dir.exists():
            logger.warning(f"Cache dizini bulunamadı: {self.cache_dir}")
            return []
        
        cache_files = list(self.cache_dir.glob("cached_data_*.pkl"))
        logger.info(f"Bulunan cache dosyası sayısı: {len(cache_files)}")
        
        for cache_file in cache_files:
            logger.info(f"  - {cache_file.name}")
        
        return cache_files
    
    def analyze_cache_file(self, cache_file: Path) -> Dict[str, Any]:
        """Cache dosyasını analiz et"""
        logger.info("\n" + "=" * 80)
        logger.info(f"CACHE DOSYASI ANALİZİ: {cache_file.name}")
        logger.info("=" * 80)
        
        try:
            # Cache dosyasını yükle
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            
            logger.info(f"✅ Cache dosyası yüklendi: {len(cached_data):,} örnek")
            
            # İstatistikler
            eos_positions = []
            eos_frequency = 0
            token_frequency = Counter()
            sequence_lengths = []
            target_last_tokens = Counter()
            alignment_errors = 0
            correct_alignments = 0
            
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
            
            logger.info(f"BOS ID: {bos_id}, EOS ID: {eos_id}")
            
            # İlk 1000 örneği analiz et (hızlı test için)
            samples_to_analyze = min(1000, len(cached_data))
            logger.info(f"İlk {samples_to_analyze} örnek analiz ediliyor...")
            
            for idx, (inp_ids, tgt_ids) in enumerate(cached_data[:samples_to_analyze]):
                seq_len = len(tgt_ids)
                sequence_lengths.append(seq_len)
                
                # EOS pozisyonu kontrolü
                if len(tgt_ids) > 0:
                    last_token = tgt_ids[-1]
                    target_last_tokens[last_token] += 1
                    
                    if last_token == eos_id:
                        eos_frequency += 1
                        eos_positions.append(seq_len - 1)
                    
                    # Token frequency
                    token_frequency.update(tgt_ids)
                
                # TrainingService format'ını simüle et (satır 701-704)
                # Input = [BOS] + inp_ids, Target = tgt_ids + [EOS]
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
                    if alignment_errors <= 5:  # İlk 5 hatayı göster
                        logger.warning(f"Alignment hatası (örnek {idx}):")
                        logger.warning(f"  Input[:10]: {seq_in[:10]}")
                        logger.warning(f"  Target[:10]: {seq_tgt[:10]}")
            
            # EOS her zaman son pozisyonda mı?
            eos_always_last = all(pos == len(tgt_ids) - 1 for pos in eos_positions) if eos_positions else False
            
            results = {
                "cache_file": str(cache_file),
                "total_examples": len(cached_data),
                "examples_analyzed": samples_to_analyze,
                "eos_analysis": {
                    "eos_frequency": eos_frequency,
                    "eos_frequency_percentage": (eos_frequency / samples_to_analyze) * 100,
                    "eos_always_last": eos_always_last,
                    "eos_positions": {
                        "min": min(eos_positions) if eos_positions else None,
                        "max": max(eos_positions) if eos_positions else None,
                        "avg": sum(eos_positions) / len(eos_positions) if eos_positions else None,
                        "count": len(eos_positions)
                    }
                },
                "sequence_analysis": {
                    "lengths": {
                        "min": min(sequence_lengths) if sequence_lengths else None,
                        "max": max(sequence_lengths) if sequence_lengths else None,
                        "avg": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else None
                    }
                },
                "token_analysis": {
                    "top_20_tokens": dict(token_frequency.most_common(20)),
                    "target_last_token_distribution": dict(target_last_tokens.most_common(10))
                },
                "alignment_analysis": {
                    "correct_alignments": correct_alignments,
                    "alignment_errors": alignment_errors,
                    "alignment_accuracy": (correct_alignments / samples_to_analyze) * 100
                },
                "bos_id": bos_id,
                "eos_id": eos_id
            }
            
            logger.info(f"\nEOS Analizi:")
            logger.info(f"  EOS frequency: {eos_frequency}/{samples_to_analyze} ({results['eos_analysis']['eos_frequency_percentage']:.2f}%)")
            logger.info(f"  EOS her zaman son pozisyonda mı: {eos_always_last}")
            
            logger.info(f"\nAlignment Analizi:")
            logger.info(f"  Doğru alignment: {correct_alignments}/{samples_to_analyze} ({results['alignment_analysis']['alignment_accuracy']:.2f}%)")
            logger.info(f"  Alignment hatası: {alignment_errors}")
            
            logger.info(f"\nToken Analizi:")
            logger.info(f"  En sık kullanılan 10 token: {dict(token_frequency.most_common(10))}")
            logger.info(f"  Target son token dağılımı: {dict(target_last_tokens.most_common(10))}")
            
            return results
            
        except Exception as e:
            logger.error(f"Cache dosyası analiz hatası: {e}", exc_info=True)
            return {"error": str(e), "cache_file": str(cache_file)}
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Tüm analizleri çalıştır"""
        self.initialize()
        
        # Cache dosyalarını bul
        cache_files = self.find_cache_files()
        
        if not cache_files:
            logger.warning("Cache dosyası bulunamadı!")
            return {"error": "Cache dosyası bulunamadı"}
        
        # Her cache dosyasını analiz et
        results = {
            "test_date": datetime.now().isoformat(),
            "cache_dir": str(self.cache_dir),
            "cache_files_found": len(cache_files),
            "cache_analyses": []
        }
        
        for cache_file in cache_files:
            analysis = self.analyze_cache_file(cache_file)
            results["cache_analyses"].append(analysis)
        
        # Sonuçları kaydet
        output_file = "test_cache_analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
        logger.info("=" * 80)
        
        return results


if __name__ == "__main__":
    analyzer = CacheAnalysis(cache_dir=".cache/preprocessed_data", data_dir="education")
    results = analyzer.run_full_analysis()
    
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    
    if "error" in results:
        print(f"HATA: {results['error']}")
    else:
        for analysis in results.get("cache_analyses", []):
            if "error" in analysis:
                print(f"HATA: {analysis['error']}")
            else:
                print(f"\nCache Dosyası: {Path(analysis['cache_file']).name}")
                print(f"  Toplam örnek: {analysis.get('total_examples', 0):,}")
                print(f"  EOS frequency: {analysis['eos_analysis']['eos_frequency_percentage']:.2f}%")
                print(f"  Alignment accuracy: {analysis['alignment_analysis']['alignment_accuracy']:.2f}%")
                print(f"  Alignment hataları: {analysis['alignment_analysis']['alignment_errors']}")

