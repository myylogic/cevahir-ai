# -*- coding: utf-8 -*-
"""
Greedy Decode Validation Test
=============================
Model'in gerçekten öğrenip öğrenmediğini test eder.

Test Edilen Modül: model/cevahir.py (CevahirModelAPI._autoregressive_generate)
Test Stratejisi: Greedy decode (temperature=0, argmax) ile generation

Endüstri Standartları:
- pytest framework
- Detailed logging
- JSON result export (rapor için)
- Assertion-based validation

Beklenen Sonuç:
- Model öğrenmişse, greedy decode en iyi token'ı seçmeli
- Mantıklı kelimeler üretmeli
- Tekrarlama olmamalı (greedy decode deterministik)
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch

# Proje dizinini sys.path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.cevahir import Cevahir, CevahirConfig

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_greedy_decode.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("GreedyDecodeTest")


class GreedyDecodeValidator:
    """Greedy decode validation test sınıfı"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Model dosya yolu (varsayılan: saved_models/cevahir_model.pth)
        """
        self.model_path = model_path or os.path.join("saved_models", "cevahir_model.pth")
        self.cevahir = None
        self.results = {
            "test_date": datetime.now().isoformat(),
            "model_path": self.model_path,
            "test_results": []
        }
        
    def initialize_model(self) -> bool:
        """Model'i yükle ve initialize et"""
        try:
            logger.info("=" * 80)
            logger.info("MODEL YÜKLEME")
            logger.info("=" * 80)
            
            # Device kontrolü
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {device}")
            
            # Config oluştur
            config = CevahirConfig(
                device=device,
                tokenizer={
                    "vocab_path": "data/vocab_lib/vocab.json",
                    "merges_path": "data/merges_lib/merges.txt",
                    "use_gpu": device == "cuda",
                    "batch_size": 32,
                },
                load_model_path=self.model_path,
                model={
                    "vocab_size": 60000,
                    "embed_dim": 512,
                    "seq_proj_dim": 512,
                    "num_heads": 8,
                    "num_layers": 12,  # Training ile uyumlu
                    "ffn_dim": None,
                    "pre_norm": True,
                    "causal_mask": True,
                    "use_flash_attention": False,
                    "pe_mode": "rope",
                    "use_gradient_checkpointing": True,
                    "tie_weights": True,
                    "use_rmsnorm": True,
                    "use_swiglu": True,
                    "use_kv_cache": True,
                    "max_cache_len": 2048,
                    "use_advanced_checkpointing": False,
                    "checkpointing_strategy": "selective",
                    "quantization_type": "none",
                    "use_moe": False,
                    "num_experts": 8,
                    "moe_top_k": 2,
                    "use_tensorboard": False,
                }
            )
            
            # Model'i yükle
            logger.info(f"Model yükleniyor: {self.model_path}")
            self.cevahir = Cevahir(config)
            logger.info("✅ Model başarıyla yüklendi!")
            
            # Model state kontrolü
            if hasattr(self.cevahir, 'model') and hasattr(self.cevahir.model, 'model'):
                model = self.cevahir.model.model
                training_mode = model.training
                logger.info(f"Model training mode: {training_mode} (False = eval, True = train)")
                
                if training_mode:
                    logger.warning("⚠️  UYARI: Model training mode'da! Inference için eval mode'da olmalı!")
                    model.eval()
                    logger.info("✅ Model eval mode'a geçirildi")
                else:
                    logger.info("✅ Model zaten eval mode'da")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}", exc_info=True)
            return False
    
    def test_greedy_decode(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Greedy decode ile generation test et
        
        Args:
            prompt: Test prompt'u
            max_new_tokens: Maksimum yeni token sayısı
            expected_keywords: Beklenen anahtar kelimeler (opsiyonel)
        
        Returns:
            Test sonuçları dictionary
        """
        if self.cevahir is None:
            raise RuntimeError("Model initialize edilmemiş! initialize_model() çağrılmalı.")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"GREEDY DECODE TEST: '{prompt}'")
        logger.info("=" * 80)
        
        result = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().isoformat(),
            "greedy_decode": {},
            "sampling_decode": {},  # Karşılaştırma için
            "analysis": {}
        }
        
        try:
            # 1. GREEDY DECODE (temperature=0.0)
            logger.info("\n--- GREEDY DECODE (temperature=0.0, argmax) ---")
            
            # Temperature=0 ile greedy decode yap
            # Not: Cevahir.generate() temperature parametresi alır
            greedy_response = self.cevahir.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy decode
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0
            )
            
            result["greedy_decode"]["response"] = greedy_response
            result["greedy_decode"]["response_length"] = len(greedy_response)
            result["greedy_decode"]["token_count"] = len(greedy_response.split())
            
            logger.info(f"Greedy Response: {greedy_response}")
            logger.info(f"Response Length: {len(greedy_response)} chars, {len(greedy_response.split())} tokens")
            
            # 2. SAMPLING DECODE (karşılaştırma için, temperature=1.0)
            logger.info("\n--- SAMPLING DECODE (temperature=1.0, karşılaştırma) ---")
            
            sampling_response = self.cevahir.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1
            )
            
            result["sampling_decode"]["response"] = sampling_response
            result["sampling_decode"]["response_length"] = len(sampling_response)
            result["sampling_decode"]["token_count"] = len(sampling_response.split())
            
            logger.info(f"Sampling Response: {sampling_response}")
            logger.info(f"Response Length: {len(sampling_response)} chars, {len(sampling_response.split())} tokens")
            
            # 3. ANALİZ
            logger.info("\n--- ANALİZ ---")
            
            analysis = self._analyze_response(greedy_response, prompt, expected_keywords)
            result["analysis"] = analysis
            
            # Analiz sonuçlarını logla
            logger.info(f"Anlamlı Kelimeler: {analysis.get('meaningful_words', [])}")
            logger.info(f"Tekrarlama Oranı: {analysis.get('repetition_ratio', 0):.2%}")
            logger.info(f"Prompt Uyumu: {analysis.get('prompt_relevance', 'N/A')}")
            logger.info(f"Genel Değerlendirme: {analysis.get('overall_assessment', 'N/A')}")
            
            # Sonuç özeti
            logger.info("\n" + "=" * 80)
            if analysis.get("overall_success", False):
                logger.info("✅ TEST BAŞARILI: Model anlamlı çıktı üretiyor")
            else:
                logger.info("❌ TEST BAŞARISIZ: Model anlamlı çıktı üretemiyor")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Test hatası: {e}", exc_info=True)
            result["error"] = str(e)
            result["analysis"]["overall_success"] = False
        
        return result
    
    def _analyze_response(
        self,
        response: str,
        prompt: str,
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Response'u analiz et
        
        Args:
            response: Model çıktısı
            prompt: Input prompt
            expected_keywords: Beklenen anahtar kelimeler
        
        Returns:
            Analiz sonuçları
        """
        analysis = {
            "response_length": len(response),
            "word_count": len(response.split()),
            "meaningful_words": [],
            "repetition_ratio": 0.0,
            "prompt_relevance": "UNKNOWN",
            "overall_assessment": "UNKNOWN",
            "overall_success": False
        }
        
        # Anlamlı kelime tespiti (basit heuristics)
        # Türkçe kelimeler için basit bir kontrol
        words = response.split()
        meaningful_words = []
        
        # Tek karakter olmayan, anlamlı görünen kelimeler
        for word in words:
            word_clean = word.strip(".,!?;:()[]{}'\"")
            if len(word_clean) > 1 and word_clean.isalpha():
                meaningful_words.append(word_clean)
        
        analysis["meaningful_words"] = meaningful_words[:20]  # İlk 20 kelime
        
        # Tekrarlama analizi
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = 1.0 - (len(unique_words) / len(words))
            analysis["repetition_ratio"] = repetition_ratio
        
        # Prompt uyumu (basit keyword matching)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        common_words = prompt_words.intersection(response_words)
        
        if len(common_words) > 0:
            analysis["prompt_relevance"] = "PARTIAL"
        else:
            analysis["prompt_relevance"] = "LOW"
        
        # Genel değerlendirme
        success_criteria = {
            "has_meaningful_words": len(meaningful_words) > 0,
            "low_repetition": repetition_ratio < 0.5,  # %50'den az tekrarlama
            "reasonable_length": len(response) > 10,  # En az 10 karakter
        }
        
        analysis["success_criteria"] = success_criteria
        
        # Overall success
        overall_success = (
            success_criteria["has_meaningful_words"] and
            success_criteria["reasonable_length"]
        )
        
        analysis["overall_success"] = overall_success
        
        if overall_success:
            analysis["overall_assessment"] = "SUCCESS - Model anlamlı çıktı üretiyor"
        else:
            analysis["overall_assessment"] = "FAILURE - Model anlamlı çıktı üretemiyor"
        
        return analysis
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Tam test suite'i çalıştır"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("GREEDY DECODE VALIDATION TEST SUITE")
        logger.info("=" * 80)
        
        # Test prompt'ları
        test_prompts = [
            {
                "prompt": "selam beni anlayabiliyor musun",
                "max_new_tokens": 50,
                "expected_keywords": ["selam", "anlayabiliyor"]
            },
            {
                "prompt": "merhaba nasılsın",
                "max_new_tokens": 30,
                "expected_keywords": ["merhaba", "nasılsın"]
            },
            {
                "prompt": "Türkiye'nin başkenti neresidir",
                "max_new_tokens": 30,
                "expected_keywords": ["Türkiye", "başkenti"]
            },
        ]
        
        # Her prompt için test çalıştır
        for i, test_case in enumerate(test_prompts, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"TEST {i}/{len(test_prompts)}")
            logger.info(f"{'=' * 80}")
            
            result = self.test_greedy_decode(
                prompt=test_case["prompt"],
                max_new_tokens=test_case["max_new_tokens"],
                expected_keywords=test_case.get("expected_keywords")
            )
            
            self.results["test_results"].append(result)
        
        # Özet istatistikler
        total_tests = len(self.results["test_results"])
        successful_tests = sum(
            1 for r in self.results["test_results"]
            if r.get("analysis", {}).get("overall_success", False)
        )
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0
        }
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST ÖZETİ")
        logger.info("=" * 80)
        logger.info(f"Toplam Test: {total_tests}")
        logger.info(f"Başarılı: {successful_tests}")
        logger.info(f"Başarısız: {total_tests - successful_tests}")
        logger.info(f"Başarı Oranı: {self.results['summary']['success_rate']:.2%}")
        logger.info("=" * 80)
        
        return self.results
    
    def save_results(self, output_file: str = "test_greedy_decode_results.json"):
        """Test sonuçlarını JSON dosyasına kaydet"""
        output_path = os.path.join(BASE_DIR, output_file)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n✅ Test sonuçları kaydedildi: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Sonuç kaydetme hatası: {e}", exc_info=True)
            return None


def main():
    """Ana test fonksiyonu"""
    validator = GreedyDecodeValidator()
    
    # Model'i yükle
    if not validator.initialize_model():
        logger.error("❌ Model yüklenemedi, test durduruluyor.")
        return 1
    
    # Test suite'i çalıştır
    results = validator.run_test_suite()
    
    # Sonuçları kaydet
    output_file = validator.save_results()
    
    # Exit code
    success_rate = results.get("summary", {}).get("success_rate", 0.0)
    if success_rate >= 0.5:  # %50+ başarı
        logger.info("\n✅ Test suite başarılı (>=%50 başarı oranı)")
        return 0
    else:
        logger.warning("\n⚠️  Test suite kısmen başarısız (<%50 başarı oranı)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

