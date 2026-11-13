"""
Training Data Analizi Testi
============================
Eğitim verisini analiz eder - EOS frequency, token distribution, anlamsız karakterler.

Test Amacı:
- Training data'da EOS frequency kontrolü
- Token distribution analizi
- Anlamsız karakter/harf kombinasyonları tespiti
- EOS pozisyon analizi (her zaman son pozisyonda mı?)

Eğitim verisi formatı (TokenizerCore.load_training_data):
    (inp_ids, tgt_ids, source_id) — BOS/EOS YOK (add_special_tokens=False) ve tgt_ids == inp_ids.
BOS/EOS ve bir pozisyonluk kaydırma (target[i] == input[i+1]) eğitim formatlamasında eklenir
(prepare_cache.format_data_func / TrainingService: Input = [BOS] + inp_ids, Target = tgt_ids + [EOS]).

pytest altında hiçbir dosya yazılmaz (JSON raporu yalnızca istenen yola, testte tmp_path'e).
Script olarak çalıştırıldığında (python tests/test_training_data_analysis.py) raporu çalışma
dizinine yazar.
"""

import os
import sys
import json
import tempfile
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
from tokenizer_management.config import BPE_CONFIG
from data_loader_management.data_loader_manager import DataLoaderManager, DataLoaderConfig, LoadMode

# Logger (root logger yalnızca script modunda yapılandırılır; import yan etkisi yok)
logger = logging.getLogger("TrainingDataAnalysis")

# Eğitim korpusu (mutlak yol: testler hangi dizinden çalıştırılırsa çalıştırılsın aynı veri)
DATA_DIR = os.path.join(project_root, "education")


def _summary(values: List[int]) -> Dict[str, Any]:
    """min/max/avg özeti"""
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "avg": sum(values) / len(values) if values else None,
    }


class TrainingDataAnalysis:
    """Training data analiz sınıfı"""
    
    def __init__(self, data_dir: str = "education"):
        self.data_dir = data_dir
        self.tokenizer_core = None
        self.analysis_results = {}
        self._training_examples = None
    
    def initialize(self):
        """Tokenizer'ı başlat (tekrar çağrılırsa mevcut tokenizer korunur)"""
        if self.tokenizer_core is not None:
            return
        logger.info("=" * 80)
        logger.info("TRAINING DATA ANALİZ TESTİ BAŞLATIYOR")
        logger.info("=" * 80)
        
        # TokenizerCore başlat (load_training_data için data_dir gerekli)
        config = {
            "data_dir": self.data_dir,
            "vocab_path": BPE_CONFIG["vocab_file"],
            "merges_path": BPE_CONFIG["merges_file"],
            "use_gpu": False  # CPU için
        }
        self.tokenizer_core = TokenizerCore(config)
        logger.info("✅ TokenizerCore başlatıldı")
    
    def load_training_examples(self) -> List[tuple]:
        """Training data'yı yükle (encoding ile). Pahalı olduğu için bir kez yüklenir."""
        if self._training_examples is None:
            self._training_examples = self.tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=True,
                include_syllables=True,
                include_sep=True
            )
            logger.info(f"Yüklenen training örnek sayısı: {len(self._training_examples)}")
        return self._training_examples
    
    def _special_id(self, token: str, fallback: int) -> int:
        """Özel token ID'si (vocab'ta yoksa fallback)"""
        val = self.tokenizer_core.get_vocab().get(token)
        if isinstance(val, dict) and isinstance(val.get("id"), int):
            return val["id"]
        if isinstance(val, int):
            return val
        return fallback
    
    def analyze_raw_data(self, max_texts: Optional[int] = 100) -> Dict[str, Any]:
        """Ham training data'yı analiz et (TXT/DOCX raw text chunk'ları)"""
        logger.info("\n" + "=" * 80)
        logger.info("HAM DATA ANALİZİ")
        logger.info("=" * 80)
        
        # DataLoaderManager ile raw data yükle (güncel API: DataLoaderConfig + LoadMode.RAW_TEXT;
        # JSON dosyaları QA çiftleridir ve QA_TRAIN modunda yüklenir)
        data_loader = DataLoaderManager(DataLoaderConfig(data_dir=Path(self.data_dir), mode=LoadMode.RAW_TEXT))
        
        raw_texts = data_loader.load()
        logger.info(f"Yüklenen raw text sayısı: {len(raw_texts)}")
        texts = raw_texts if max_texts is None else raw_texts[:max_texts]  # İlk N (hızlı analiz için)
        
        # İstatistikler
        total_chars = 0
        total_words = 0
        empty_texts = 0
        char_frequency = Counter()
        word_frequency = Counter()
        anlamsiz_patterns = []
        
        # Her text için analiz
        for text in texts:
            if not text.strip():
                empty_texts += 1
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
        
        analyzed = len(texts)
        results = {
            "total_texts_loaded": len(raw_texts),
            "total_texts_analyzed": analyzed,
            "empty_texts": empty_texts,
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_text": total_chars / analyzed if analyzed else 0.0,
            "avg_words_per_text": total_words / analyzed if analyzed else 0.0,
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
    
    def analyze_encoded_data(self, max_examples: Optional[int] = 1000) -> Dict[str, Any]:
        """Encoding sonrası training data'yı analiz et"""
        logger.info("\n" + "=" * 80)
        logger.info("ENCODED DATA ANALİZİ")
        logger.info("=" * 80)
        
        training_examples = self.load_training_examples()
        examples = training_examples if max_examples is None else training_examples[:max_examples]
        
        bos_id = self._special_id("<BOS>", 2)
        eos_id = self._special_id("<EOS>", 3)
        
        # İstatistikler
        eos_positions = []  # EOS pozisyonları (target sequence içinde)
        eos_frequency = 0
        examples_with_bos = 0
        examples_with_eos = 0
        examples_without_source_id = 0
        empty_examples = 0
        target_differs_from_input = 0
        token_frequency = Counter()
        sequence_lengths = []
        target_last_tokens = Counter()  # Target'ın son token'ı
        
        for item in examples:
            # Format: (inp_ids, tgt_ids, source_id); source_id'siz eski format: (inp_ids, tgt_ids)
            inp_ids, tgt_ids = list(item[0]), list(item[1])
            if len(item) != 3:
                examples_without_source_id += 1
            seq_len = len(tgt_ids)
            sequence_lengths.append(seq_len)
            
            if not inp_ids or not tgt_ids:
                empty_examples += 1
            if tgt_ids != inp_ids:
                target_differs_from_input += 1
            if bos_id in inp_ids or bos_id in tgt_ids:
                examples_with_bos += 1
            if eos_id in inp_ids or eos_id in tgt_ids:
                examples_with_eos += 1
            
            # EOS pozisyonu kontrolü
            if len(tgt_ids) > 0:
                last_token = tgt_ids[-1]
                target_last_tokens[last_token] += 1
                
                if last_token == eos_id:
                    eos_frequency += 1
                    eos_positions.append(seq_len - 1)  # Son pozisyon
                
                # Token frequency
                token_frequency.update(tgt_ids)
        
        analyzed = len(examples)
        results = {
            "total_examples_loaded": len(training_examples),
            "total_examples_analyzed": analyzed,
            "examples_without_source_id": examples_without_source_id,
            "empty_examples": empty_examples,
            "target_differs_from_input": target_differs_from_input,
            "examples_with_bos": examples_with_bos,
            "examples_with_eos": examples_with_eos,
            "eos_frequency": eos_frequency,
            "eos_frequency_percentage": (eos_frequency / analyzed) * 100 if analyzed else 0.0,
            "eos_positions": _summary(eos_positions),
            "sequence_lengths": _summary(sequence_lengths),
            "top_20_tokens": dict(token_frequency.most_common(20)),
            "target_last_token_distribution": dict(target_last_tokens.most_common(10))
        }
        
        logger.info(f"EOS frequency: {eos_frequency}/{analyzed} ({results['eos_frequency_percentage']:.2f}%)")
        logger.info(f"BOS/EOS içeren örnek: {examples_with_bos}/{examples_with_eos}")
        logger.info(f"En sık kullanılan 20 token: {dict(token_frequency.most_common(20))}")
        logger.info(f"Target son token dağılımı: {dict(target_last_tokens.most_common(10))}")
        
        return results
    
    def analyze_training_format(self, max_examples: Optional[int] = 100) -> Dict[str, Any]:
        """Training format doğrulaması (BOS/EOS alignment)"""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING FORMAT DOĞRULAMASI")
        logger.info("=" * 80)
        
        training_examples = self.load_training_examples()
        examples = training_examples if max_examples is None else training_examples[:max_examples]
        
        # BOS ve EOS ID'leri
        bos_id = self._special_id("<BOS>", 2)
        eos_id = self._special_id("<EOS>", 3)
        
        # TrainingService format'ını simüle et
        alignment_errors = 0
        correct_alignments = 0
        eos_only_at_end = 0
        error_samples = []
        
        for idx, item in enumerate(examples):
            inp_ids, tgt_ids = item[0], item[1]
            # TrainingService format: Input = [BOS] + inp_ids, Target = tgt_ids + [EOS]
            seq_in = [bos_id] + list(inp_ids)
            seq_tgt = list(tgt_ids) + [eos_id]
            
            # Alignment kontrolü: Target[i] == Input[i+1] (i < len(tgt_ids)); son token (EOS) hariç
            is_aligned = len(seq_in) == len(seq_tgt) and seq_tgt[:-1] == seq_in[1:]
            
            if is_aligned:
                correct_alignments += 1
            else:
                alignment_errors += 1
                if len(error_samples) < 5:
                    error_samples.append(idx)
            
            # EOS pozisyon analizi: formatlanmış hedefte EOS yalnızca son pozisyonda olmalı
            if seq_tgt.count(eos_id) == 1:
                eos_only_at_end += 1
        
        checked = len(examples)
        results = {
            "total_examples_checked": checked,
            "correct_alignments": correct_alignments,
            "alignment_errors": alignment_errors,
            "alignment_accuracy": (correct_alignments / checked) * 100 if checked else 0.0,
            "error_samples": error_samples,
            "eos_only_at_end": eos_only_at_end,
            "bos_id": bos_id,
            "eos_id": eos_id
        }
        
        logger.info(f"Alignment accuracy: {results['alignment_accuracy']:.2f}%")
        logger.info(f"Doğru alignment: {correct_alignments}/{checked}")
        logger.info(f"Alignment hatası: {alignment_errors}")
        
        return results
    
    def run_full_analysis(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Tüm analizleri çalıştır (output_file verilirse sonuçları JSON olarak kaydet)"""
        self.initialize()
        
        results = {
            "test_date": datetime.now().isoformat(),
            "raw_data_analysis": self.analyze_raw_data(),
            "encoded_data_analysis": self.analyze_encoded_data(),
            "training_format_analysis": self.analyze_training_format()
        }
        self.analysis_results = results
        
        # Sonuçları kaydet
        if output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
            logger.info("=" * 80)
        
        return results


# ---------------------------------------------------------------------------
# pytest testleri (korpus modül boyunca bir kez tokenize edilir)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    """Training data'sı bir kez yüklenmiş TrainingDataAnalysis"""
    analyzer = TrainingDataAnalysis(data_dir=DATA_DIR)
    analyzer.initialize()
    analyzer.load_training_examples()
    return analyzer


def test_raw_data_analysis(analyzer):
    """TXT/DOCX raw text chunk'ları yüklenmeli ve boş chunk olmamalı"""
    results = analyzer.analyze_raw_data(max_texts=None)
    assert results["total_texts_analyzed"] > 0, "education/ altından hiç raw text (TXT/DOCX) chunk'ı yüklenemedi"
    assert results["empty_texts"] == 0, f"{results['empty_texts']} raw text chunk'ı boş"
    assert results["total_words"] > 0, "Raw text'lerde hiç kelime yok"


def test_encoded_data_eos_frequency(analyzer):
    """Encode edilmiş örnekler: BOS/EOS yok, tgt_ids == inp_ids, her örnekte source_id var"""
    results = analyzer.analyze_encoded_data(max_examples=None)
    analyzed = results["total_examples_analyzed"]
    assert analyzed > 0, "Training data boş!"
    assert results["examples_without_source_id"] == 0, f"{results['examples_without_source_id']} örnekte source_id yok"
    assert results["empty_examples"] == 0, f"{results['empty_examples']} örnek boş"
    
    # BOS/EOS'u tokenizer değil eğitim formatlaması ekler (load_training_data: add_special_tokens=False).
    # Ham örnekte EOS bulunsaydı formatlama sonrası hedefte EOS iki kez (içerikte ve sonda) yer alırdı.
    assert results["eos_frequency"] == 0, (
        f"EOS frequency: {results['eos_frequency']}/{analyzed} ({results['eos_frequency_percentage']:.2f}%), "
        f"son token dağılımı: {results['target_last_token_distribution']}"
    )
    assert results["examples_with_eos"] == 0, f"{results['examples_with_eos']} ham örnekte EOS var"
    assert results["examples_with_bos"] == 0, f"{results['examples_with_bos']} ham örnekte BOS var"
    
    # Autoregressive: tgt_ids == inp_ids; bir pozisyonluk kaydırma formatlamada yapılır
    assert results["target_differs_from_input"] == 0, (
        f"{results['target_differs_from_input']} örnekte tgt_ids != inp_ids"
    )


def test_training_format_alignment(analyzer):
    """TrainingService formatında Target[i] == Input[i+1] ve EOS yalnızca son pozisyonda"""
    results = analyzer.analyze_training_format(max_examples=None)
    checked = results["total_examples_checked"]
    assert checked > 0, "Training data boş!"
    assert results["alignment_errors"] == 0, (
        f"Alignment accuracy: {results['alignment_accuracy']:.2f}% "
        f"({results['alignment_errors']} hata, ilk hatalı örnekler: {results['error_samples']})"
    )
    assert results["correct_alignments"] == checked
    assert results["eos_only_at_end"] == checked, (
        f"{checked - results['eos_only_at_end']} formatlanmış hedefte EOS son pozisyon dışında da var"
    )


def test_full_analysis_report_is_written_where_requested(analyzer, tmp_path):
    """run_full_analysis() raporu yalnızca istenen yola (burada tmp_path) yazar"""
    output_file = tmp_path / "training_data_analysis_results.json"
    analyzer.run_full_analysis(output_file=str(output_file))
    
    assert output_file.exists(), "JSON raporu yazılmadı"
    report = json.loads(output_file.read_text(encoding="utf-8"))
    for key in ("raw_data_analysis", "encoded_data_analysis", "training_format_analysis"):
        assert key in report, f"Raporda {key} yok"
    assert report["training_format_analysis"]["alignment_errors"] == 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    analyzer = TrainingDataAnalysis(data_dir=DATA_DIR)
    # Rapor repo ağacına değil geçici dizine yazılır.
    _out = Path(tempfile.gettempdir()) / "test_training_data_analysis_results.json"
    results = analyzer.run_full_analysis(output_file=str(_out))
    print(f"Rapor: {_out}")
    
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    print(f"Raw data analizi: {results['raw_data_analysis'].get('total_texts_analyzed', 0)} text")
    print(f"Encoded data analizi: {results['encoded_data_analysis'].get('total_examples_analyzed', 0)} örnek")
    print(f"EOS frequency: {results['encoded_data_analysis'].get('eos_frequency_percentage', 0):.2f}%")
    print(f"Alignment accuracy: {results['training_format_analysis'].get('alignment_accuracy', 0):.2f}%")
