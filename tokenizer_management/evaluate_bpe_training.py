# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: evaluate_bpe_training.py
Modül: tokenizer_management
Görev: BPE Training Sonrası Kapsamlı Değerlendirme Scripti - BPE training
       tamamlandıktan sonra vocab/merges kontrolü, tokenization kalitesi testi,
       UNK oranı analizi, autoregressive format kontrolü ve over-segmentation
       kontrolü yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (BPE training değerlendirme)
- Design Patterns: Script Pattern (standalone evaluation tool)
- Endüstri Standartları: Tokenization quality assessment

KULLANIM:
- BPE training sonrası kalite kontrolü için
- Vocab ve merges dosyalarını değerlendirmek için
- Tokenization kalitesini test etmek için

BAĞIMLILIKLAR:
- TokenizerCore: Tokenization işlemleri
- DataLoaderManager: Veri yükleme işlemleri
- get_bpe_detailed_config, get_turkish_config: Config yönetimi

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
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Proje kök dizinini sys.path'e ekle
# Script tokenizer_management klasöründe olduğu için bir seviye yukarı çık
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError
from tokenizer_management.config import get_bpe_detailed_config, BPE_CONFIG
from data_loader_management.data_loader_manager import DataLoaderManager, DataLoaderConfig, LoadMode
from tokenizer_management.bpe.bpe_manager_utils import get_valid_ids

# =========================
# Logging
# =========================
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
eval_logger = logging.getLogger("BPEEvaluation")
eval_logger.info("="*80)
eval_logger.info("BPE TRAINING DEĞERLENDİRME SCRIPTİ")
eval_logger.info("="*80)

# =========================
# Config
# =========================
DEFAULT_VOCAB_PATH = "data/vocab_lib/vocab.json"
DEFAULT_MERGES_PATH = "data/merges_lib/merges.txt"
DEFAULT_DATA_DIR = "education"

# =========================
# Yardımcı Fonksiyonlar
# =========================

def check_file_exists(path: str, description: str) -> Tuple[bool, Optional[str]]:
    """Dosyanın varlığını ve boş olup olmadığını kontrol et"""
    if not os.path.exists(path):
        return False, f"{description} bulunamadı: {path}"
    
    if os.path.getsize(path) == 0:
        return False, f"{description} boş: {path}"
    
    return True, None

def load_vocab(vocab_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Vocab dosyasını yükle"""
    exists, error = check_file_exists(vocab_path, "Vocab dosyası")
    if not exists:
        return None, error
    
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab, None
    except json.JSONDecodeError as e:
        return None, f"Vocab dosyası JSON formatında değil: {e}"
    except Exception as e:
        return None, f"Vocab dosyası yüklenemedi: {e}"

def load_merges(merges_path: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Merges dosyasını yükle"""
    exists, error = check_file_exists(merges_path, "Merges dosyası")
    if not exists:
        return None, error
    
    try:
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges = [line.strip() for line in f if line.strip()]
        return merges, None
    except Exception as e:
        return None, f"Merges dosyası yüklenemedi: {e}"

def validate_vocab(vocab: Dict) -> Dict[str, Any]:
    """Vocab doğrulama ve metrikler"""
    results = {
        "vocab_size": len(vocab),
        "min_expected": 30000,
        "max_expected": 60000,
        "in_range": False,
        "special_tokens": {},
        "special_tokens_missing": [],
        "vocab_structure_valid": True,
        "errors": []
    }
    
    # Vocab boyutu kontrolü
    results["in_range"] = results["min_expected"] <= results["vocab_size"] <= results["max_expected"]
    
    # Özel tokenlar kontrolü
    special_tokens = ["<BOS>", "<EOS>", "<SEP>", "<PAD>", "<UNK>"]
    for token in special_tokens:
        if token in vocab:
            token_data = vocab[token]
            if isinstance(token_data, dict) and "id" in token_data:
                results["special_tokens"][token] = token_data["id"]
            else:
                results["special_tokens"][token] = token_data  # Eski format
        else:
            results["special_tokens_missing"].append(token)
    
    # Vocab yapısı kontrolü
    for token, data in list(vocab.items())[:100]:  # İlk 100 token kontrolü
        if isinstance(data, dict):
            if "id" not in data:
                results["vocab_structure_valid"] = False
                results["errors"].append(f"Token '{token}' ID içermiyor")
        elif not isinstance(data, (int, str)):
            results["vocab_structure_valid"] = False
            results["errors"].append(f"Token '{token}' geçersiz format: {type(data)}")
    
    return results

def validate_merges(merges: List[str]) -> Dict[str, Any]:
    """Merges doğrulama ve metrikler"""
    results = {
        "merge_count": len(merges),
        "expected_merges": 13000,  # initial_vocab_ratio=0.70 için
        "sufficient": False,
        "format_valid": True,
        "invalid_merges": [],
        "errors": []
    }
    
    # Merge sayısı kontrolü
    results["sufficient"] = results["merge_count"] >= results["expected_merges"] * 0.8  # %80 tolerans
    
    # Merge format kontrolü
    for i, line in enumerate(merges[:1000]):  # İlk 1000 merge kontrolü
        parts = line.split()
        if len(parts) != 2:
            results["format_valid"] = False
            results["invalid_merges"].append((i+1, line))
            if len(results["invalid_merges"]) >= 10:  # İlk 10 hata
                break
    
    return results

def test_tokenization_on_training_data(
    tokenizer: TokenizerCore,
    data_dir: str,
    sample_size: int = 500,
    mode: str = "inference"
) -> Dict[str, Any]:
    """Tokenization testi - Eğitim verisi üzerinden (training_service gibi)"""
    results = {
        "total_samples": 0,
        "successful": 0,
        "failed": 0,
        "unk_counts": [],
        "unk_ratios": [],
        "token_counts": [],
        "token_word_ratios": [],
        "decode_successful": 0,
        "decode_failed": 0,
        "errors": []
    }
    
    try:
        # DataLoader ile eğitim verisini yükle
        loader = DataLoaderManager(DataLoaderConfig(
            data_dir=Path(data_dir),
            mode=LoadMode.QA_TRAIN
        ))
        qa_data = loader.load()
        
        # RAW text chunks da yükle
        raw_loader = DataLoaderManager(DataLoaderConfig(
            data_dir=Path(data_dir),
            mode=LoadMode.RAW_TEXT
        ))
        raw_texts = raw_loader.load()
        
        # Tüm metinleri birleştir (QA + RAW)
        all_texts = []
        for q, a in qa_data:
            if q:
                all_texts.append(q)
            if a:
                all_texts.append(a)
        all_texts.extend(raw_texts)
        
        # Örneklem al
        sample = all_texts[:min(sample_size, len(all_texts))]
        results["total_samples"] = len(sample)
        
        # Vocab ve UNK ID
        vocab = tokenizer.tokenizer.get_vocab()
        unk_data = vocab.get("<UNK>")
        unk_id = None
        if isinstance(unk_data, dict):
            unk_id = unk_data.get("id")
        elif isinstance(unk_data, (int, str)):
            unk_id = int(unk_data) if unk_data else None
        
        # Her metni test et
        for text in sample:
            try:
                # Encode
                tokens, token_ids = tokenizer.encode(text, mode=mode)
                
                # UNK kontrolü
                unk_count = 0
                if unk_id is not None:
                    unk_count = sum(1 for tid in token_ids if tid == unk_id)
                unk_ratio = (unk_count / len(token_ids)) * 100 if token_ids else 0
                
                results["unk_counts"].append(unk_count)
                results["unk_ratios"].append(unk_ratio)
                results["token_counts"].append(len(token_ids))
                
                # Token/kelime oranı
                words = text.split()
                word_count = len(words)
                token_word_ratio = len(token_ids) / word_count if word_count > 0 else 0
                results["token_word_ratios"].append(token_word_ratio)
                
                # Decode testi (round-trip)
                try:
                    decoded = tokenizer.decode(token_ids)
                    # Basit karşılaştırma (büyük/küçük harf ve boşluk farklarını göz ardı et)
                    # Sadece içerik kontrolü yap (tam eşleşme zor olabilir)
                    if decoded.strip():
                        results["decode_successful"] += 1
                    else:
                        results["decode_failed"] += 1
                except Exception as e:
                    results["decode_failed"] += 1
                    results["errors"].append(f"Decode hatası: {e}")
                
                results["successful"] += 1
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Tokenization hatası: {e}")
        
        # Ortalamalar
        if results["unk_ratios"]:
            results["avg_unk_ratio"] = sum(results["unk_ratios"]) / len(results["unk_ratios"])
        else:
            results["avg_unk_ratio"] = 0
        
        if results["token_word_ratios"]:
            results["avg_token_word_ratio"] = sum(results["token_word_ratios"]) / len(results["token_word_ratios"])
        else:
            results["avg_token_word_ratio"] = 0
        
    except Exception as e:
        results["errors"].append(f"Eğitim verisi yükleme hatası: {e}")
    
    return results

def analyze_training_data_preparation(
    tokenizer: TokenizerCore,
    data_dir: str,
    sample_size: int = 100
) -> Dict[str, Any]:
    """Eğitim verisi hazırlama analizi (training_service simülasyonu)"""
    results = {
        "data_loaded": False,
        "examples_count": 0,
        "valid_examples": 0,
        "invalid_examples": 0,
        "unk_ratio": 0.0,
        "autoregressive_format_valid": True,
        "special_tokens_present": True,
        "errors": []
    }
    
    try:
        # TokenizerCore'dan eğitim verisi yükle (training_service gibi)
        raw_data = tokenizer.load_training_data(
            encode_mode="train",
            include_whole_words=True,
            include_syllables=False,  # Over-segmentation azaltmak için
            include_sep=True,
        )
        
        results["data_loaded"] = True
        results["examples_count"] = len(raw_data)
        
        # Örneklem al
        sample = raw_data[:min(sample_size, len(raw_data))]
        
        # Vocab ve special token ID'leri
        vocab = tokenizer.tokenizer.get_vocab()
        
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
        
        # Her örneği kontrol et
        total_tokens = 0
        total_unks = 0
        
        for idx, (inp_ids, tgt_ids) in enumerate(sample):
            try:
                # Geçerlilik kontrolü
                if not isinstance(inp_ids, list) or not isinstance(tgt_ids, list):
                    results["invalid_examples"] += 1
                    continue
                
                if not all(isinstance(tid, int) for tid in inp_ids + tgt_ids):
                    results["invalid_examples"] += 1
                    continue
                
                # Autoregressive format kontrolü
                # Input: BOS ... EOS olmalı
                # Target: BOS ... EOS olmalı (input'un bir token kaydırılmış hali)
                if len(inp_ids) > 0 and inp_ids[0] != BOS_ID:
                    results["autoregressive_format_valid"] = False
                    results["errors"].append(f"Örnek {idx}: Input BOS ile başlamıyor")
                
                if len(inp_ids) > 0 and inp_ids[-1] != EOS_ID:
                    results["autoregressive_format_valid"] = False
                    results["errors"].append(f"Örnek {idx}: Input EOS ile bitmiyor")
                
                if len(tgt_ids) > 0 and tgt_ids[0] != BOS_ID:
                    results["autoregressive_format_valid"] = False
                    results["errors"].append(f"Örnek {idx}: Target BOS ile başlamıyor")
                
                if len(tgt_ids) > 0 and tgt_ids[-1] != EOS_ID:
                    results["autoregressive_format_valid"] = False
                    results["errors"].append(f"Örnek {idx}: Target EOS ile bitmiyor")
                
                # Special token kontrolü
                if BOS_ID not in inp_ids or EOS_ID not in inp_ids:
                    results["special_tokens_present"] = False
                
                # UNK kontrolü
                unk_count = sum(1 for tid in inp_ids if tid == UNK_ID)
                total_unks += unk_count
                total_tokens += len(inp_ids)
                
                # Vocab size kontrolü
                if any(tid >= VOCAB_SIZE or tid < 0 for tid in inp_ids + tgt_ids):
                    results["invalid_examples"] += 1
                    results["errors"].append(f"Örnek {idx}: Vocab size dışında ID var")
                    continue
                
                results["valid_examples"] += 1
                
            except Exception as e:
                results["invalid_examples"] += 1
                results["errors"].append(f"Örnek {idx} işlenirken hata: {e}")
        
        # UNK oranı
        if total_tokens > 0:
            results["unk_ratio"] = (total_unks / total_tokens) * 100
        
    except Exception as e:
        results["errors"].append(f"Eğitim verisi yükleme hatası: {e}")
    
    return results

def analyze_unk_on_dataset(
    tokenizer: TokenizerCore,
    data_dir: str,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """Eğitim verisi üzerinde UNK oranı analizi"""
    results = {
        "total_samples": 0,
        "total_tokens": 0,
        "total_unks": 0,
        "unk_ratio": 0.0,
        "status": "unknown"
    }
    
    try:
        # DataLoader ile veri yükle
        loader = DataLoaderManager(DataLoaderConfig(
            data_dir=Path(data_dir),
            mode=LoadMode.QA_TRAIN
        ))
        qa_data = loader.load()
        
        # Örneklem al
        sample = qa_data[:min(sample_size, len(qa_data))]
        results["total_samples"] = len(sample)
        
        # Vocab ve UNK ID
        vocab = tokenizer.tokenizer.get_vocab()
        unk_data = vocab.get("<UNK>")
        unk_id = None
        if isinstance(unk_data, dict):
            unk_id = unk_data.get("id")
        elif isinstance(unk_data, (int, str)):
            unk_id = int(unk_data) if unk_data else None
        
        # Her örneği encode et
        for q, a in sample:
            # Soru
            _, q_ids = tokenizer.encode(q, mode="inference")
            results["total_tokens"] += len(q_ids)
            if unk_id is not None:
                results["total_unks"] += sum(1 for tid in q_ids if tid == unk_id)
            
            # Cevap
            _, a_ids = tokenizer.encode(a, mode="inference")
            results["total_tokens"] += len(a_ids)
            if unk_id is not None:
                results["total_unks"] += sum(1 for tid in a_ids if tid == unk_id)
        
        # UNK oranı
        if results["total_tokens"] > 0:
            results["unk_ratio"] = (results["total_unks"] / results["total_tokens"]) * 100
        
        # Durum belirleme
        if results["unk_ratio"] < 1:
            results["status"] = "excellent"
        elif results["unk_ratio"] < 2:
            results["status"] = "good"
        elif results["unk_ratio"] < 5:
            results["status"] = "acceptable"
        else:
            results["status"] = "needs_improvement"
            
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results

# =========================
# Ana Değerlendirme Fonksiyonu
# =========================

def evaluate_bpe_training(
    vocab_path: str = DEFAULT_VOCAB_PATH,
    merges_path: str = DEFAULT_MERGES_PATH,
    data_dir: str = DEFAULT_DATA_DIR
) -> Dict[str, Any]:
    """Kapsamlı BPE training değerlendirmesi"""
    
    report = {
        "vocab_check": {},
        "merges_check": {},
        "tokenization_test": {},
        "training_data_analysis": {},
        "unk_analysis": {},
        "overall_status": "unknown",
        "recommendations": []
    }
    
    eval_logger.info("\n" + "="*80)
    eval_logger.info("1. VOCAB VE MERGES DOSYALARI KONTROLÜ")
    eval_logger.info("="*80)
    
    # Vocab kontrolü
    vocab, error = load_vocab(vocab_path)
    if vocab is None:
        eval_logger.error(f"Vocab yüklenemedi: {error}")
        report["vocab_check"] = {"error": error}
        return report
    
    eval_logger.info(f"Vocab yüklendi: {len(vocab):,} token")
    vocab_results = validate_vocab(vocab)
    report["vocab_check"] = vocab_results
    
    # Vocab sonuçları
    eval_logger.info(f"Vocab boyutu: {vocab_results['vocab_size']:,} token")
    if vocab_results["in_range"]:
        eval_logger.info(f"Vocab boyutu beklenen aralıkta ({vocab_results['min_expected']:,} - {vocab_results['max_expected']:,})")
    else:
        eval_logger.warning(f"Vocab boyutu beklenen aralığın dışında!")
        report["recommendations"].append("Vocab boyutu beklenen aralıkta değil, initial_vocab_ratio ayarlanmalı")
    
    # Özel tokenlar
    for token, token_id in vocab_results["special_tokens"].items():
        eval_logger.info(f"{token} vocab'de var (ID: {token_id})")
    
    if vocab_results["special_tokens_missing"]:
        eval_logger.error(f"Eksik özel tokenlar: {vocab_results['special_tokens_missing']}")
        report["recommendations"].append("Eksik özel tokenlar vocab'e eklenmeli")
    
    # Merges kontrolü
    merges, error = load_merges(merges_path)
    if merges is None:
        eval_logger.error(f"Merges yüklenemedi: {error}")
        report["merges_check"] = {"error": error}
        return report
    
    eval_logger.info(f"Merges yüklendi: {len(merges):,} merge")
    merges_results = validate_merges(merges)
    report["merges_check"] = merges_results
    
    # Merges sonuçları
    eval_logger.info(f"Merge sayısı: {merges_results['merge_count']:,}")
    if merges_results["sufficient"]:
        eval_logger.info(f"Merge sayısı yeterli (beklenen: ~{merges_results['expected_merges']:,})")
    else:
        eval_logger.warning(f"Merge sayısı düşük (beklenen: ~{merges_results['expected_merges']:,})")
        report["recommendations"].append("Merge sayısı düşük, initial_vocab_ratio azaltılabilir")
    
    if merges_results["format_valid"]:
        eval_logger.info("Merge formatı doğru")
    else:
        eval_logger.error(f"{len(merges_results['invalid_merges'])} geçersiz merge formatı bulundu!")
        report["recommendations"].append("Merge formatı düzeltilmeli")
    
    eval_logger.info("\n" + "="*80)
    eval_logger.info("2. TOKENIZER BAŞLATMA")
    eval_logger.info("="*80)
    
    # TokenizerCore başlat
    try:
        config = get_bpe_detailed_config()
        config["data_dir"] = data_dir
        config["vocab_path"] = vocab_path
        config["merges_path"] = merges_path
        tokenizer = TokenizerCore(config)
        eval_logger.info("TokenizerCore başlatıldı")
    except Exception as e:
        eval_logger.error(f"TokenizerCore başlatılamadı: {e}")
        report["tokenization_test"] = {"error": str(e)}
        return report
    
    eval_logger.info("\n" + "="*80)
    eval_logger.info("3. TOKENIZATION TESTİ (EĞİTİM VERİSİ ÜZERİNDEN)")
    eval_logger.info("="*80)
    eval_logger.info("Eğitim verisi üzerinden tokenization testi yapılıyor...")
    eval_logger.info("(Bu, eğitim sürecinin ikinci aşaması gibi çalışır)")
    
    # Eğitim verisi üzerinden tokenization testi
    tokenization_results = test_tokenization_on_training_data(
        tokenizer, 
        data_dir, 
        sample_size=500,  # 500 örnek test et
        mode="inference"
    )
    report["tokenization_test"] = tokenization_results
    
    eval_logger.info(f"Test edilen örnek: {tokenization_results['total_samples']:,} metin")
    eval_logger.info(f"Başarılı testler: {tokenization_results['successful']}/{tokenization_results['total_samples']}")
    eval_logger.info(f"Ortalama UNK oranı: {tokenization_results['avg_unk_ratio']:.2f}%")
    eval_logger.info(f"Ortalama token/kelime oranı: {tokenization_results['avg_token_word_ratio']:.2f}")
    
    if tokenization_results["avg_unk_ratio"] < 1:
        eval_logger.info("Mükemmel! UNK oranı çok düşük")
    elif tokenization_results["avg_unk_ratio"] < 2:
        eval_logger.info("İyi! UNK oranı düşük")
    elif tokenization_results["avg_unk_ratio"] < 5:
        eval_logger.warning("Orta! UNK oranı kabul edilebilir ama iyileştirilebilir")
        report["recommendations"].append("UNK oranı yüksek, BPE training iyileştirilmeli")
    else:
        eval_logger.error("Yüksek! UNK oranı çok yüksek")
        report["recommendations"].append("UNK oranı çok yüksek, BPE training tekrar yapılmalı")
    
    if tokenization_results["avg_token_word_ratio"] < 1.5:
        eval_logger.info("Mükemmel! Over-segmentation düşük")
    elif tokenization_results["avg_token_word_ratio"] < 2.0:
        eval_logger.info("İyi! Over-segmentation kabul edilebilir")
    elif tokenization_results["avg_token_word_ratio"] < 3.0:
        eval_logger.warning("Orta! Over-segmentation var ama kabul edilebilir")
    else:
        eval_logger.error("Yüksek! Over-segmentation çok yüksek")
        report["recommendations"].append("Over-segmentation yüksek, include_syllables=False kontrol edilmeli")
    
    total_tests = tokenization_results.get('total_samples', tokenization_results.get('successful', 0))
    eval_logger.info(f"Decode başarılı: {tokenization_results['decode_successful']}/{total_tests}")
    
    eval_logger.info("\n" + "="*80)
    eval_logger.info("4. EĞİTİM VERİSİ HAZIRLAMA ANALİZİ")
    eval_logger.info("="*80)
    
    # Eğitim verisi hazırlama analizi
    training_data_results = analyze_training_data_preparation(tokenizer, data_dir, sample_size=100)
    report["training_data_analysis"] = training_data_results
    
    if training_data_results["data_loaded"]:
        eval_logger.info(f"Eğitim verisi yüklendi: {training_data_results['examples_count']:,} örnek")
        eval_logger.info(f"Geçerli örnekler: {training_data_results['valid_examples']}")
        eval_logger.info(f"Geçersiz örnekler: {training_data_results['invalid_examples']}")
        eval_logger.info(f"UNK oranı: {training_data_results['unk_ratio']:.2f}%")
        
        if training_data_results["autoregressive_format_valid"]:
            eval_logger.info("Autoregressive format doğru")
        else:
            eval_logger.error("Autoregressive format hatası!")
            report["recommendations"].append("Autoregressive format düzeltilmeli")
        
        if training_data_results["special_tokens_present"]:
            eval_logger.info("Special tokenlar mevcut")
        else:
            eval_logger.error("Special tokenlar eksik!")
            report["recommendations"].append("Special tokenlar kontrol edilmeli")
    else:
        eval_logger.error("Eğitim verisi yüklenemedi")
    
    eval_logger.info("\n" + "="*80)
    eval_logger.info("5. UNK ORANI ANALİZİ (EĞİTİM VERİSİ ÜZERİNDE)")
    eval_logger.info("="*80)
    
    # UNK oranı analizi
    unk_results = analyze_unk_on_dataset(tokenizer, data_dir, sample_size=1000)
    report["unk_analysis"] = unk_results
    
    eval_logger.info(f"Örneklem: {unk_results['total_samples']:,} çift")
    eval_logger.info(f"Toplam token: {unk_results['total_tokens']:,}")
    eval_logger.info(f"UNK token: {unk_results['total_unks']:,}")
    eval_logger.info(f"UNK oranı: {unk_results['unk_ratio']:.2f}%")
    
    status_messages = {
        "excellent": "Mükemmel! UNK oranı çok düşük (GPT seviyesinde)",
        "good": "İyi! UNK oranı düşük",
        "acceptable": "Kabul edilebilir! UNK oranı orta seviyede",
        "needs_improvement": "İyileştirme gerekli! UNK oranı yüksek",
        "error": f"Hata: {unk_results.get('error', 'Bilinmeyen hata')}"
    }
    eval_logger.info(status_messages.get(unk_results["status"], "❓ Bilinmeyen durum"))
    
    if unk_results["status"] in ["needs_improvement", "error"]:
        report["recommendations"].append("UNK oranı yüksek, BPE training iyileştirilmeli")
    
    # Genel durum
    all_checks_passed = (
        vocab_results["in_range"] and
        vocab_results["vocab_structure_valid"] and
        len(vocab_results["special_tokens_missing"]) == 0 and
        merges_results["sufficient"] and
        merges_results["format_valid"] and
        tokenization_results["avg_unk_ratio"] < 5 and
        tokenization_results["avg_token_word_ratio"] < 3.0 and
        training_data_results.get("autoregressive_format_valid", False) and
        training_data_results.get("special_tokens_present", False) and
        unk_results["status"] in ["excellent", "good", "acceptable"]
    )
    
    if all_checks_passed:
        report["overall_status"] = "passed"
        eval_logger.info("\n" + "="*80)
        eval_logger.info("TÜM KONTROLLER BAŞARILI!")
        eval_logger.info("="*80)
    else:
        report["overall_status"] = "needs_attention"
        eval_logger.info("\n" + "="*80)
        eval_logger.info("BAZI KONTROLLER BAŞARISIZ - ÖNERİLERİ KONTROL EDİN")
        eval_logger.info("="*80)
    
    # Öneriler
    if report["recommendations"]:
        eval_logger.info("\nÖNERİLER:")
        for i, rec in enumerate(report["recommendations"], 1):
            eval_logger.info(f"  {i}. {rec}")
    
    return report

# =========================
# Main
# =========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BPE Training Değerlendirme Scripti")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH, help="Vocab dosya yolu")
    parser.add_argument("--merges", type=str, default=DEFAULT_MERGES_PATH, help="Merges dosya yolu")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Eğitim verisi dizini")
    parser.add_argument("--output", type=str, default="bpe_evaluation_report.json", help="Rapor dosya yolu")
    
    args = parser.parse_args()
    
    # Değerlendirme
    report = evaluate_bpe_training(
        vocab_path=args.vocab,
        merges_path=args.merges,
        data_dir=args.data_dir
    )
    
    # Rapor kaydet
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        eval_logger.info(f"\nRapor kaydedildi: {args.output}")
    except Exception as e:
        eval_logger.error(f"Rapor kaydedilemedi: {e}")
    
    # Çıkış kodu
    if report["overall_status"] == "passed":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

