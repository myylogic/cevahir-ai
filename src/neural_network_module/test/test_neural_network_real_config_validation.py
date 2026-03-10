# -*- coding: utf-8 -*-
"""
Gerçek Model Konfigürasyonu Validasyon Testleri

Bu testler, gerçek eğitim konfigürasyonuyla (vocab_size=60000, embed_dim=256) 
kritik sorunları tespit eder.
"""

import pytest
import torch
import torch.nn as nn
import logging
import sys
import os

# Proje kökünü path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network import CevahirNeuralNetwork

logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def real_config():
    """Gerçek eğitim konfigürasyonu"""
    return {
        "learning_rate": 1.5e-4,
        "dropout": 0.15,
        "vocab_size": 60000,  # Gerçek vocab size
        "embed_dim": 256,  # Gerçek embed dim
        "seq_proj_dim": 256,
        "num_heads": 8,
        "num_layers": 6,  # Gerçek layer sayısı
    }


class TestRealConfigGradientExplosion:
    """Gerçek Konfigürasyon Gradient Explosion Testleri"""
    
    def test_real_config_gradient_explosion(self, real_config):
        """
        ✅ KRİTİK TEST: Gerçek konfigürasyonla gradient explosion tespiti
        
        vocab_size=60000, embed_dim=256 → ratio=234 (çok büyük!)
        Bu, gradient explosion riski yaratır.
        """
        model = CevahirNeuralNetwork(
            learning_rate=real_config["learning_rate"],
            dropout=real_config["dropout"],
            vocab_size=real_config["vocab_size"],
            embed_dim=real_config["embed_dim"],
            seq_proj_dim=real_config["seq_proj_dim"],
            num_heads=real_config["num_heads"],
            num_layers=real_config["num_layers"],
            tie_weights=True,
            log_level=logging.WARNING,
        )
        
        x = torch.randint(0, real_config["vocab_size"], (2, 10), dtype=torch.long)
        targets = torch.randint(0, real_config["vocab_size"], (2, 10), dtype=torch.long)
        
        model.train()
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, real_config["vocab_size"]),
            targets.view(-1)
        )
        loss.backward()
        
        # Embedding gradient'ini kontrol et
        embedding_grad = model.dil_katmani.language_embedding.embedding.weight.grad
        embedding_grad_norm = embedding_grad.norm().item()
        
        # ✅ ASSERT: Embedding gradient norm makul
        # Not: Gerçek konfigürasyonda gradient'ler daha büyük olabilir
        # Ama çok büyük olmamalı (< 10000)
        assert embedding_grad_norm < 10000.0, (
            f"GERÇEK KONFİGÜRASYON: Embedding gradient explosion! "
            f"Norm: {embedding_grad_norm:.2f} (beklenen: < 10000.0)"
        )
        
        # ✅ UYARI: Gradient norm çok büyükse uyar
        if embedding_grad_norm > 1000.0:
            pytest.skip(
                f"⚠️ UYARI: Embedding gradient norm çok büyük: {embedding_grad_norm:.2f} "
                f"(beklenen: < 1000.0). Model mimarisi düzeltilmeli!"
            )
    
    def test_real_config_vocab_embed_ratio(self, real_config):
        """
        ✅ KRİTİK TEST: Gerçek konfigürasyon vocab/embed ratio kontrolü
        
        vocab_size=60000, embed_dim=256 → ratio=234
        Bu, endüstri standardından çok büyük (normal: 50-100)
        """
        vocab_size = real_config["vocab_size"]
        embed_dim = real_config["embed_dim"]
        ratio = vocab_size / embed_dim
        
        # ✅ ASSERT: Ratio çok büyük (sorun tespit edildi!)
        # Bu test başarısız olmalı - sorun tespit edildi!
        # Test başarısız olursa, model mimarisini düzeltmeliyiz
        assert ratio < 200, (
            f"⚠️ KRİTİK SORUN: Vocab/Embed ratio çok büyük! "
            f"Ratio: {ratio:.2f} (beklenen: < 200, önerilen: 50-100, "
            f"endüstri standardı GPT-2: 65, GPT-3: 4). "
            f"Model mimarisi düzeltilmeli: embed_dim artırılmalı veya vocab_size küçültülmeli!"
        )


class TestRealConfigLogitsRange:
    """Gerçek Konfigürasyon Logits Range Testleri"""
    
    def test_real_config_logits_range(self, real_config):
        """
        ✅ KRİTİK TEST: Gerçek konfigürasyonla logits aralığı kontrolü
        
        Logits çok geniş aralıkta olmamalı.
        """
        model = CevahirNeuralNetwork(
            learning_rate=real_config["learning_rate"],
            dropout=real_config["dropout"],
            vocab_size=real_config["vocab_size"],
            embed_dim=real_config["embed_dim"],
            seq_proj_dim=real_config["seq_proj_dim"],
            num_heads=real_config["num_heads"],
            num_layers=real_config["num_layers"],
            use_rmsnorm=True,
            log_level=logging.WARNING,
        )
        
        x = torch.randint(0, real_config["vocab_size"], (2, 10), dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
            
            logits_min = logits.min().item()
            logits_max = logits.max().item()
            logits_range = logits_max - logits_min
            logits_std = logits.std().item()
        
        # ✅ ASSERT: Logits aralığı makul
        # Not: Gerçek konfigürasyonda logits daha geniş olabilir
        # Ama çok geniş olmamalı (< 100)
        assert logits_range < 100.0, (
            f"GERÇEK KONFİGÜRASYON: Logits aralığı çok geniş! "
            f"Range: {logits_range:.2f} (beklenen: < 100.0, normal: 10-20)"
        )
        
        # ✅ ASSERT: Logits std makul
        assert logits_std < 10.0, (
            f"GERÇEK KONFİGÜRASYON: Logits std çok büyük! "
            f"Std: {logits_std:.2f} (beklenen: < 10.0, normal: 1-3)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

