"""
Logits Dağılımı Analiz Testi
=============================
Model'in logits dağılımını analiz eder - MODE COLLAPSE tespiti için kritik.

Test Amacı:
- Model'in ürettiği logits'leri görselleştir
- 'selam' token'ının probability'sini kontrol et
- MODE COLLAPSE tespiti (tek token'a aşırı yoğunlaşma)
- Top-k token'ların dağılımını analiz et
"""

import os
import sys
import json
import logging
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.cevahir import Cevahir, CevahirConfig

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("LogitsAnalysisTest")


class LogitsAnalysisTest:
    """Logits dağılımı analiz test sınıfı"""
    
    def __init__(self, model_path: str = "saved_models/cevahir_model.pth"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cevahir = None
        self.results = {
            "test_date": datetime.now().isoformat(),
            "model_path": model_path,
            "device": self.device,
            "test_results": []
        }
        
    def setup(self):
        """Model'i yükle"""
        logger.info("="*80)
        logger.info("MODEL YÜKLEME")
        logger.info("="*80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model yükleniyor: {self.model_path}")
        
        try:
            config = CevahirConfig(
                device=self.device,
                tokenizer={
                    "vocab_path": "data/vocab_lib/vocab.json",
                    "merges_path": "data/merges_lib/merges.txt",
                    "use_gpu": self.device == "cuda",
                    "batch_size": 32,
                },
                load_model_path=self.model_path,
                model={
                    "vocab_size": 60000,
                    "embed_dim": 512,
                    "seq_proj_dim": 512,
                    "num_heads": 8,
                    "num_layers": 12,
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
                },
            )
            
            self.cevahir = Cevahir(config)
            logger.info("✅ Model başarıyla yüklendi!")
            
            # Model mode kontrolü
            is_training = self.cevahir._model_manager.model.training
            logger.info(f"Model training mode: {is_training} (False = eval, True = train)")
            if is_training:
                logger.warning("⚠️ Model training mode'da! Eval mode'a geçiriliyor...")
                self.cevahir._model_manager.eval_mode()
            else:
                logger.info("✅ Model zaten eval mode'da")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}", exc_info=True)
            return False
    
    def analyze_logits(
        self, 
        prompt: str, 
        step: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0
    ) -> Dict[str, Any]:
        """
        Belirli bir prompt için logits dağılımını analiz et
        
        Args:
            prompt: Test prompt'u
            step: Generation step (0 = prompt sonrası ilk token)
            temperature: Temperature değeri
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            repetition_penalty: Repetition penalty
        
        Returns:
            Analiz sonuçları dictionary
        """
        logger.info("="*80)
        logger.info(f"LOGITS ANALİZİ: '{prompt}'")
        logger.info(f"Step: {step}, Temperature: {temperature}, Top-K: {top_k}, Top-P: {top_p}, Repetition Penalty: {repetition_penalty}")
        logger.info("="*80)
        
        try:
            # Encode prompt
            _, token_ids = self.cevahir._tokenizer_core.encode(prompt, mode="inference")
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            
            logger.info(f"Prompt encoded: {len(token_ids)} tokens")
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.cevahir._model_manager.forward(
                    input_tensor,
                    inference=True,
                    return_aux=False,
                    use_cache=True,  # KV Cache kullan
                )
            
            # Son token'ın logits'ini al (ilk generation step için)
            if step == 0:
                next_logits = logits[0, -1, :].clone()  # [vocab_size]
            else:
                # Eğer step > 0 ise, autoregressive generation simüle et
                # (Bu test için şimdilik step=0'a odaklanıyoruz)
                next_logits = logits[0, -1, :].clone()
            
            # Raw logits istatistikleri
            logits_min = next_logits.min().item()
            logits_max = next_logits.max().item()
            logits_mean = next_logits.mean().item()
            logits_std = next_logits.std().item()
            
            logger.info(f"Raw Logits: min={logits_min:.2f}, max={logits_max:.2f}, mean={logits_mean:.2f}, std={logits_std:.2f}")
            
            # Repetition penalty uygula (eğer varsa)
            if repetition_penalty > 1.0:
                for token_id in token_ids[-256:]:  # Son 256 token
                    if 0 <= token_id < next_logits.shape[0]:
                        next_logits[token_id] /= repetition_penalty
            
            # Temperature uygula
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                next_logits = next_logits * float('inf')  # Greedy
            
            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_logits.shape[0])
                top_k_values, top_k_indices = torch.topk(next_logits, top_k)
                filtered_logits = torch.full_like(next_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_values
                next_logits = filtered_logits
            
            # Top-p (nucleus) sampling
            if top_p < 1.0 and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # Probability dağılımı
            probs = torch.softmax(next_logits, dim=0)
            
            # Top-20 tokens ve probability'leri
            top_20_probs, top_20_indices = torch.topk(probs, k=min(20, len(probs)))
            
            # Token ID'lerini kelimelere çevir
            vocab = self.cevahir._tokenizer_core.get_vocab()
            id_to_token = {v: k for k, v in vocab.items() if isinstance(v, int)}
            
            top_20_tokens = []
            for idx, prob in zip(top_20_indices.tolist(), top_20_probs.tolist()):
                token_text = id_to_token.get(idx, f"<UNK:{idx}>")
                top_20_tokens.append({
                    "token_id": idx,
                    "token_text": token_text,
                    "probability": prob,
                    "logit": next_logits[idx].item() if not torch.isinf(next_logits[idx]) else float('-inf')
                })
            
            # 'selam' token'ını bul
            selam_token_id = None
            selam_prob = None
            for token_text, token_id in vocab.items():
                if token_text == "selam" and isinstance(token_id, int):
                    selam_token_id = token_id
                    selam_prob = probs[token_id].item()
                    break
            
            # EOS token ID'sini bul
            eos_id = None
            eos_prob = None
            if isinstance(vocab.get("<EOS>"), dict):
                eos_id = vocab["<EOS>"].get("id")
            elif isinstance(vocab.get("<EOS>"), int):
                eos_id = vocab["<EOS>"]
            
            if eos_id is not None:
                eos_prob = probs[eos_id].item()
            
            # Entropy hesapla (diversity ölçüsü)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            max_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float)).item()
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # MODE COLLAPSE tespiti
            # Eğer en yüksek probability çok yüksekse (>0.5), MODE COLLAPSE var demektir
            max_prob = top_20_probs[0].item()
            mode_collapse_detected = max_prob > 0.5
            mode_collapse_severity = "YOK"
            if max_prob > 0.9:
                mode_collapse_severity = "CIKOK"
            elif max_prob > 0.7:
                mode_collapse_severity = "YUKSEK"
            elif max_prob > 0.5:
                mode_collapse_severity = "ORTA"
            
            # Sonuçları logla
            logger.info(f"\nTop-5 Tokens:")
            for i, token_info in enumerate(top_20_tokens[:5]):
                logger.info(f"  {i+1}. {token_info['token_text']} (id={token_info['token_id']}): prob={token_info['probability']:.4f} ({token_info['probability']*100:.2f}%)")
            
            if selam_token_id is not None:
                logger.info(f"\n'selam' Token Analizi:")
                logger.info(f"  Token ID: {selam_token_id}")
                logger.info(f"  Probability: {selam_prob:.4f} ({selam_prob*100:.2f}%)")
                logger.info(f"  Rank: {sorted(top_20_indices.tolist(), reverse=False).index(selam_token_id) + 1 if selam_token_id in top_20_indices else '>20'}")
            
            if eos_id is not None:
                logger.info(f"\nEOS Token Analizi:")
                logger.info(f"  Token ID: {eos_id}")
                logger.info(f"  Probability: {eos_prob:.4f} ({eos_prob*100:.2f}%)")
            
            logger.info(f"\nEntropy Analizi:")
            logger.info(f"  Entropy: {entropy:.4f}")
            logger.info(f"  Max Entropy: {max_entropy:.4f}")
            logger.info(f"  Normalized Entropy: {normalized_entropy:.4f} (1.0 = perfect diversity, 0.0 = mode collapse)")
            
            logger.info(f"\nMODE COLLAPSE Analizi:")
            logger.info(f"  Max Probability: {max_prob:.4f} ({max_prob*100:.2f}%)")
            logger.info(f"  MODE COLLAPSE Detected: {mode_collapse_detected}")
            logger.info(f"  Severity: {mode_collapse_severity}")
            
            # Sonuçları döndür
            result = {
                "prompt": prompt,
                "step": step,
                "generation_params": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty
                },
                "logits_stats": {
                    "min": logits_min,
                    "max": logits_max,
                    "mean": logits_mean,
                    "std": logits_std
                },
                "top_20_tokens": top_20_tokens,
                "selam_token": {
                    "token_id": selam_token_id,
                    "probability": selam_prob,
                    "rank": sorted(top_20_indices.tolist(), reverse=False).index(selam_token_id) + 1 if selam_token_id is not None and selam_token_id in top_20_indices else None
                } if selam_token_id is not None else None,
                "eos_token": {
                    "token_id": eos_id,
                    "probability": eos_prob
                } if eos_id is not None else None,
                "entropy": {
                    "raw": entropy,
                    "max": max_entropy,
                    "normalized": normalized_entropy
                },
                "mode_collapse": {
                    "detected": mode_collapse_detected,
                    "severity": mode_collapse_severity,
                    "max_probability": max_prob
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Logits analiz hatası: {e}", exc_info=True)
            return {
                "prompt": prompt,
                "error": str(e)
            }
    
    def run_test_suite(self):
        """Test suite'i çalıştır"""
        logger.info("\n" + "="*80)
        logger.info("LOGITS DAĞILIMI ANALİZ TEST SUITE")
        logger.info("="*80 + "\n")
        
        test_prompts = [
            "selam beni anlayabiliyor musun",
            "merhaba nasılsın",
            "Türkiye'nin başkenti neresidir",
        ]
        
        test_configs = [
            {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "name": "Greedy"},
            {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "name": "Sampling (temp=1.0)"},
            {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.0, "name": "Top-K/Top-P"},
            {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.5, "name": "Repetition Penalty 1.5"},
            {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 2.0, "name": "Repetition Penalty 2.0"},
        ]
        
        for prompt in test_prompts:
            logger.info(f"\n{'='*80}")
            logger.info(f"PROMPT: '{prompt}'")
            logger.info(f"{'='*80}\n")
            
            prompt_results = {
                "prompt": prompt,
                "configs": []
            }
            
            for config in test_configs:
                logger.info(f"\n--- {config['name']} ---")
                result = self.analyze_logits(
                    prompt=prompt,
                    step=0,
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    repetition_penalty=config["repetition_penalty"]
                )
                result["config_name"] = config["name"]
                prompt_results["configs"].append(result)
                
                # Kısa özet
                if "error" not in result:
                    logger.info(f"\n✅ {config['name']} - Max Prob: {result['mode_collapse']['max_probability']:.4f}, "
                              f"Entropy: {result['entropy']['normalized']:.4f}, "
                              f"Mode Collapse: {result['mode_collapse']['severity']}")
            
            self.results["test_results"].append(prompt_results)
        
        # Özet analiz
        logger.info("\n" + "="*80)
        logger.info("ÖZET ANALİZ")
        logger.info("="*80)
        
        # Tüm testlerdeki mode collapse durumunu analiz et
        mode_collapse_counts = {"YOK": 0, "ORTA": 0, "YUKSEK": 0, "CIKOK": 0}
        for prompt_result in self.results["test_results"]:
            for config_result in prompt_result["configs"]:
                if "mode_collapse" in config_result:
                    severity = config_result["mode_collapse"]["severity"]
                    mode_collapse_counts[severity] = mode_collapse_counts.get(severity, 0) + 1
        
        logger.info(f"\nMode Collapse Dağılımı:")
        for severity, count in mode_collapse_counts.items():
            logger.info(f"  {severity}: {count}")
        
        # Sonuçları kaydet
        output_file = "test_logits_analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ Test sonuçları kaydedildi: {output_file}")
    
    def run(self):
        """Test'i çalıştır"""
        if not self.setup():
            logger.error("❌ Model yükleme başarısız!")
            return False
        
        self.run_test_suite()
        return True


def main():
    """Ana fonksiyon"""
    test = LogitsAnalysisTest()
    success = test.run()
    
    if success:
        logger.info("\n✅ Test suite başarıyla tamamlandı!")
    else:
        logger.error("\n❌ Test suite başarısız!")
        sys.exit(1)


if __name__ == "__main__":
    main()

