"""
Cache Analiz Testi
==================
Cache'den yüklenen eğitim örneklerini analiz eder - format, EOS pozisyonu, alignment kontrolü.

Test Amacı:
- Cache'deki verilerin formatını kontrol et
- EOS pozisyon analizi (cache'de doğru mu?)
- Alignment kontrolü (BOS/EOS alignment)
- Token distribution (cache'de yanlış pattern var mı?)

Cache formatı (prepare_cache.py, alignment_format="autoregressive_v2"): cache artık BOS/EOS/PAD
eklenmiş, formatlanmış kayıtlar saklar:
    (seq_in, seq_tgt, source_id)
    seq_in  = [BOS, t1, ..., tN, PAD, ...]   uzunluk = max_seq_length
    seq_tgt = [t1, ..., tN, EOS, PAD, ...]   uzunluk = max_seq_length
Yani target[i] == input[i+1] ve EOS, hedefteki son PAD olmayan token'dır.

pytest altında cache, prepare_cache() ile geçici bir dizinde (tmp_path_factory) oluşturulur;
depo ağacına hiçbir şey yazılmaz. Script olarak çalıştırıldığında
(python tests/test_cache_analysis.py) yerel .cache/preprocessed_data dizinindeki cache
dosyalarını analiz eder ve JSON raporunu çalışma dizinine yazar.
"""

import os
import sys
import json
import tempfile
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter

import pytest

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG
from training_system.prepare_cache import prepare_cache

# Logger (root logger yalnızca script modunda yapılandırılır; import yan etkisi yok)
logger = logging.getLogger("CacheAnalysis")

# Eğitim korpusu (mutlak yol: testler hangi dizinden çalıştırılırsa çalıştırılsın aynı veri)
DATA_DIR = os.path.join(project_root, "education")


def _summary(values: List[int]) -> Dict[str, Any]:
    """min/max/avg özeti"""
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "avg": sum(values) / len(values) if values else None,
    }


class CacheAnalysis:
    """Cache analiz sınıfı"""
    
    def __init__(
        self,
        cache_dir: str = ".cache/preprocessed_data",
        data_dir: str = "education",
        samples_to_analyze: Optional[int] = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.data_dir = data_dir
        self.samples_to_analyze = samples_to_analyze  # None → tüm örnekler
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
        
        cache_files = sorted(self.cache_dir.glob("cached_data_*.pkl"))
        logger.info(f"Bulunan cache dosyası sayısı: {len(cache_files)}")
        
        for cache_file in cache_files:
            logger.info(f"  - {cache_file.name}")
        
        return cache_files
    
    def _special_id(self, token: str, fallback: int) -> int:
        """Özel token ID'si (vocab'ta yoksa fallback)"""
        val = self.tokenizer_core.get_vocab().get(token)
        if isinstance(val, dict) and isinstance(val.get("id"), int):
            return val["id"]
        if isinstance(val, int):
            return val
        return fallback
    
    def analyze_cache_file(self, cache_file: Path) -> Dict[str, Any]:
        """Cache dosyasını analiz et"""
        logger.info("\n" + "=" * 80)
        logger.info(f"CACHE DOSYASI ANALİZİ: {cache_file.name}")
        logger.info("=" * 80)
        
        # Cache dosyasını yükle
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        
        logger.info(f"✅ Cache dosyası yüklendi: {len(cached_data):,} örnek")
        
        # BOS, EOS ve PAD ID'leri
        bos_id = self._special_id("<BOS>", 2)
        eos_id = self._special_id("<EOS>", 3)
        pad_id = self._special_id("<PAD>", 0)
        logger.info(f"BOS ID: {bos_id}, EOS ID: {eos_id}, PAD ID: {pad_id}")
        
        # İstatistikler
        with_source_id = 0
        without_source_id = 0
        non_int_source_id = 0
        invalid_records = 0
        length_mismatch = 0
        source_ids = set()
        input_lengths = []
        target_lengths = []
        eos_positions = []
        eos_frequency = 0
        targets_with_extra_eos = 0
        empty_targets = 0
        bos_first = 0
        inputs_equal_targets = 0
        pad_inside_content = 0
        token_frequency = Counter()
        target_last_tokens = Counter()
        alignment_errors = 0
        correct_alignments = 0
        error_samples = []
        
        if self.samples_to_analyze is None:
            samples_to_analyze = len(cached_data)
        else:
            samples_to_analyze = min(self.samples_to_analyze, len(cached_data))
        logger.info(f"İlk {samples_to_analyze} örnek analiz ediliyor...")
        
        for idx, item in enumerate(cached_data[:samples_to_analyze]):
            # Kayıt formatı: (seq_in, seq_tgt, source_id); source_id'siz eski cache: (seq_in, seq_tgt)
            if len(item) == 3:
                seq_in, seq_tgt, source_id = item
                with_source_id += 1
                source_ids.add(source_id)
                if not isinstance(source_id, int):
                    non_int_source_id += 1
            elif len(item) == 2:
                seq_in, seq_tgt = item
                without_source_id += 1
            else:
                invalid_records += 1
                continue
            
            seq_in = list(seq_in)
            seq_tgt = list(seq_tgt)
            input_lengths.append(len(seq_in))
            target_lengths.append(len(seq_tgt))
            if len(seq_in) != len(seq_tgt):
                length_mismatch += 1
            
            # Token frequency (PAD hariç)
            token_frequency.update(t for t in seq_tgt if t != pad_id)
            
            # Hedefteki son PAD olmayan pozisyon (içerik + EOS)
            last = len(seq_tgt) - 1
            while last >= 0 and seq_tgt[last] == pad_id:
                last -= 1
            if last < 0:
                empty_targets += 1
                alignment_errors += 1
                if len(error_samples) < 5:
                    error_samples.append(idx)
                continue
            
            # EOS pozisyonu kontrolü: EOS hedefin son PAD olmayan token'ı olmalı (ve tek EOS)
            target_last_tokens[seq_tgt[last]] += 1
            if seq_tgt[last] == eos_id:
                eos_frequency += 1
                eos_positions.append(last)
            if seq_tgt.count(eos_id) > 1:
                targets_with_extra_eos += 1
            
            if seq_in and seq_in[0] == bos_id:
                bos_first += 1
            # "target == input" hizalama hatası (TrainingService ilk kayıtta bunu "Data alignment bug" diye reddeder)
            if seq_in == seq_tgt:
                inputs_equal_targets += 1
            # İçerik içinde PAD olmamalı (PAD yalnızca sondaki dolgu)
            if pad_id in seq_in[1:last + 1]:
                pad_inside_content += 1
            
            # Alignment kontrolü: Input[0] == BOS, Target[i] == Input[i+1] (i < EOS pozisyonu),
            # EOS'tan sonra input da PAD. (Eski analiz cache'te ham (inp, tgt) çiftleri bekleyip
            # BOS/EOS'u kendisi ekliyordu; formatlanmış cache'te bu çift BOS üretirdi.)
            is_aligned = (
                len(seq_in) == len(seq_tgt)
                and bool(seq_in) and seq_in[0] == bos_id
                and seq_tgt[last] == eos_id
                and seq_tgt[:last] == seq_in[1:last + 1]
                and all(t == pad_id for t in seq_in[last + 1:])
            )
            
            if is_aligned:
                correct_alignments += 1
            else:
                alignment_errors += 1
                if len(error_samples) < 5:  # İlk 5 hatayı göster
                    error_samples.append(idx)
                    logger.warning(f"Alignment hatası (örnek {idx}):")
                    logger.warning(f"  Input[:10]: {seq_in[:10]}")
                    logger.warning(f"  Target[:10]: {seq_tgt[:10]}")
        
        valid_records = samples_to_analyze - invalid_records
        # EOS her zaman (içeriğin) son pozisyonunda mı?
        eos_always_last = valid_records > 0 and eos_frequency == valid_records and targets_with_extra_eos == 0
        
        results = {
            "cache_file": str(cache_file),
            "total_examples": len(cached_data),
            "examples_analyzed": samples_to_analyze,
            "format_analysis": {
                "with_source_id": with_source_id,
                "without_source_id": without_source_id,
                "non_int_source_id": non_int_source_id,
                "invalid_records": invalid_records,
                "length_mismatch": length_mismatch,
                "unique_source_ids": len(source_ids),
            },
            "eos_analysis": {
                "eos_frequency": eos_frequency,
                "eos_frequency_percentage": (eos_frequency / samples_to_analyze) * 100 if samples_to_analyze else 0.0,
                "eos_always_last": eos_always_last,
                "targets_with_extra_eos": targets_with_extra_eos,
                "empty_targets": empty_targets,
                "eos_positions": dict(_summary(eos_positions), count=len(eos_positions)),
            },
            "sequence_analysis": {
                "lengths": _summary(target_lengths),
                "input_lengths": _summary(input_lengths),
            },
            "token_analysis": {
                "top_20_tokens": dict(token_frequency.most_common(20)),
                "target_last_token_distribution": dict(target_last_tokens.most_common(10)),
                "inputs_equal_targets": inputs_equal_targets,
                "pad_inside_content": pad_inside_content,
            },
            "alignment_analysis": {
                "bos_first": bos_first,
                "correct_alignments": correct_alignments,
                "alignment_errors": alignment_errors,
                "alignment_accuracy": (correct_alignments / samples_to_analyze) * 100 if samples_to_analyze else 0.0,
                "error_samples": error_samples,
            },
            "bos_id": bos_id,
            "eos_id": eos_id,
            "pad_id": pad_id,
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
    
    def run_full_analysis(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Tüm analizleri çalıştır (output_file verilirse sonuçları JSON olarak kaydet)"""
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
            "cache_analyses": [self.analyze_cache_file(cache_file) for cache_file in cache_files],
        }
        
        # Sonuçları kaydet
        if output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
            logger.info("=" * 80)
        
        return results


# ---------------------------------------------------------------------------
# pytest testleri: prepare_cache() ile geçici dizinde üretilen cache analiz edilir
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prepared_cache_dir(tmp_path_factory):
    """prepare_cache() varsayılanlarıyla (train.py ile aynı config) geçici dizinde cache oluştur"""
    cache_dir = tmp_path_factory.mktemp("cache_analysis")
    prepare_cache(data_dir=DATA_DIR, cache_dir=str(cache_dir))
    return cache_dir


@pytest.fixture(scope="module")
def analyzer(prepared_cache_dir):
    """Tüm cache örneklerini analiz eden CacheAnalysis"""
    analyzer = CacheAnalysis(cache_dir=str(prepared_cache_dir), data_dir=DATA_DIR, samples_to_analyze=None)
    analyzer.initialize()
    return analyzer


@pytest.fixture(scope="module")
def analysis(analyzer):
    """Hazırlanan (tek) cache dosyasının analiz sonucu"""
    cache_files = analyzer.find_cache_files()
    assert len(cache_files) == 1, f"Tek bir cache dosyası bekleniyordu, bulunan: {[f.name for f in cache_files]}"
    return analyzer.analyze_cache_file(cache_files[0])


def test_cache_records_have_expected_format(analysis):
    """Her kayıt (seq_in, seq_tgt, int source_id) olmalı; input/target uzunluğu = max_seq_length"""
    fmt = analysis["format_analysis"]
    assert analysis["total_examples"] > 0, "Cache verisi boş!"
    assert analysis["examples_analyzed"] == analysis["total_examples"]
    assert fmt["invalid_records"] == 0, f"{fmt['invalid_records']} kayıt 2/3 elemanlı değil"
    assert fmt["without_source_id"] == 0, f"{fmt['without_source_id']} kayıtta source_id yok"
    assert fmt["non_int_source_id"] == 0, f"{fmt['non_int_source_id']} kayıtta source_id int değil"
    assert fmt["unique_source_ids"] >= 2, f"En az 2 farklı source_id olmalı! Bulunan: {fmt['unique_source_ids']}"
    assert fmt["length_mismatch"] == 0, f"{fmt['length_mismatch']} kayıtta input/target uzunlukları farklı"
    
    max_seq_length = TOKENIZER_CONFIG["max_seq_length"]
    for key in ("lengths", "input_lengths"):
        lengths = analysis["sequence_analysis"][key]
        assert lengths["min"] == lengths["max"] == max_seq_length, f"{key}: {lengths}, beklenen: {max_seq_length}"


def test_eos_closes_every_target(analysis):
    """EOS her hedefte tam bir kez, son PAD olmayan pozisyonda olmalı"""
    eos = analysis["eos_analysis"]
    assert eos["empty_targets"] == 0, f"{eos['empty_targets']} hedef tamamen PAD"
    assert eos["eos_frequency"] == analysis["examples_analyzed"], (
        f"EOS frequency: {eos['eos_frequency']}/{analysis['examples_analyzed']} "
        f"({eos['eos_frequency_percentage']:.2f}%), son token dağılımı: "
        f"{analysis['token_analysis']['target_last_token_distribution']}"
    )
    assert eos["targets_with_extra_eos"] == 0, f"{eos['targets_with_extra_eos']} hedefte birden fazla EOS var"
    assert eos["eos_always_last"]


def test_targets_are_inputs_shifted_by_one(analysis):
    """BOS/EOS alignment: Input[0] == BOS ve Target[i] == Input[i+1] (next-token hizalaması)"""
    alignment = analysis["alignment_analysis"]
    assert alignment["bos_first"] == analysis["examples_analyzed"], (
        f"Input BOS ile başlamıyor: {analysis['examples_analyzed'] - alignment['bos_first']} kayıt"
    )
    assert alignment["alignment_errors"] == 0, (
        f"Alignment hatası: {alignment['alignment_errors']} kayıt "
        f"(accuracy {alignment['alignment_accuracy']:.2f}%), ilk hatalı örnekler: {alignment['error_samples']}"
    )
    assert alignment["correct_alignments"] == analysis["examples_analyzed"]


def test_token_distribution_has_no_broken_patterns(analysis):
    """Cache'de yanlış pattern olmamalı: target == input yok, içerikte PAD yok, içerik var"""
    tokens = analysis["token_analysis"]
    assert tokens["inputs_equal_targets"] == 0, (
        f"{tokens['inputs_equal_targets']} kayıtta target == input (next-token hedefi kaybolmuş)"
    )
    assert tokens["pad_inside_content"] == 0, f"{tokens['pad_inside_content']} kayıtta içerik arasında PAD var"
    assert tokens["top_20_tokens"], "Cache'de PAD dışında token yok"


def test_full_analysis_report_is_written_where_requested(prepared_cache_dir, tmp_path):
    """run_full_analysis() raporu yalnızca istenen yola (burada tmp_path) yazar"""
    output_file = tmp_path / "cache_analysis_results.json"
    analyzer = CacheAnalysis(cache_dir=str(prepared_cache_dir), data_dir=DATA_DIR)
    results = analyzer.run_full_analysis(output_file=str(output_file))
    
    assert "error" not in results, results.get("error")
    assert results["cache_files_found"] == 1
    assert output_file.exists(), "JSON raporu yazılmadı"
    report = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(report["cache_analyses"]) == 1
    assert report["cache_analyses"][0]["alignment_analysis"]["alignment_errors"] == 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    analyzer = CacheAnalysis(cache_dir=".cache/preprocessed_data", data_dir=DATA_DIR)
    # Rapor repo ağacına değil geçici dizine yazılır.
    _out = Path(tempfile.gettempdir()) / "test_cache_analysis_results.json"
    results = analyzer.run_full_analysis(output_file=str(_out))
    print(f"Rapor: {_out}")
    
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    
    if "error" in results:
        print(f"HATA: {results['error']}")
    else:
        for analysis in results.get("cache_analyses", []):
            print(f"\nCache Dosyası: {Path(analysis['cache_file']).name}")
            print(f"  Toplam örnek: {analysis.get('total_examples', 0):,}")
            print(f"  EOS frequency: {analysis['eos_analysis']['eos_frequency_percentage']:.2f}%")
            print(f"  Alignment accuracy: {analysis['alignment_analysis']['alignment_accuracy']:.2f}%")
            print(f"  Alignment hataları: {analysis['alignment_analysis']['alignment_errors']}")
