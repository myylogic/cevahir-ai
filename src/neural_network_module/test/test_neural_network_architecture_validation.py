# -*- coding: utf-8 -*-
"""
Endüstri Standardı Sinir Ağı Mimarisi Validasyon Testleri

Bu testler, sinir ağı mimarisindeki kritik sorunları tespit eder:
- Weight tying gradient accumulation
- Gradient explosion detection
- Model mimarisi validation (vocab/embed ratio)
- Output layer normalization
- Logits range validation
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
def base_config():
    """Temel model konfigürasyonu"""
    return {
        "learning_rate": 1e-4,
        "dropout": 0.1,
        "vocab_size": 1000,  # Test için küçük vocab
        "embed_dim": 128,
        "seq_proj_dim": 128,
        "num_heads": 8,
        "num_layers": 2,
    }


class TestWeightTyingGradientAccumulation:
    """Weight Tying Gradient Accumulation Testleri"""
    
    def test_weight_tying_gradient_accumulation(self, base_config):
        """
        ✅ KRİTİK TEST: Weight tying'de gradient'lerin doğru accumulate edildiğini kontrol et
        
        PyTorch'ta weight tying kullanıldığında:
        - Embedding layer'dan gelen gradient
        - Output layer'dan gelen gradient
        → Otomatik olarak toplanır (accumulation)
        
        Bu test, gradient'lerin aynı referans olduğunu ve doğru çalıştığını kontrol eder.
        """
        model = CevahirNeuralNetwork(
            learning_rate=base_config["learning_rate"],
            dropout=base_config["dropout"],
            vocab_size=base_config["vocab_size"],
            embed_dim=base_config["embed_dim"],
            seq_proj_dim=base_config["seq_proj_dim"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            tie_weights=True,
            log_level=logging.WARNING,
        )
        
        # Test input
        x = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        targets = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        
        # Forward + Backward (weight tying)
        model.train()
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, base_config["vocab_size"]),
            targets.view(-1)
        )
        loss.backward()
        
        # ✅ ASSERT: Gradient'ler aynı referans (weight tying doğru çalışıyor)
        assert model.output_layer.weight.grad is not None, (
            "Output layer gradient None!"
        )
        assert model.dil_katmani.language_embedding.embedding.weight.grad is not None, (
            "Embedding gradient None!"
        )
        assert model.output_layer.weight.grad is model.dil_katmani.language_embedding.embedding.weight.grad, (
            "Weight tying'de gradient'ler aynı referans olmalı! "
            "Bu, PyTorch'un otomatik gradient accumulation yaptığını gösterir."
        )
        
        # ✅ ASSERT: Weight'ler aynı referans
        assert model.output_layer.weight is model.dil_katmani.language_embedding.embedding.weight, (
            "Weight tying'de weight'ler aynı referans olmalı!"
        )
        
        # ✅ ASSERT: Gradient norm makul (explosion yok)
        grad_norm = model.dil_katmani.language_embedding.embedding.weight.grad.norm().item()
        assert grad_norm < 100.0, (
            f"Gradient explosion tespit edildi! "
            f"Gradient norm: {grad_norm:.2f} (beklenen: < 100.0)"
        )


class TestGradientExplosionDetection:
    """Gradient Explosion Detection Testleri"""
    
    def test_gradient_norm_threshold(self, base_config):
        """
        ✅ KRİTİK TEST: Gradient norm'unun makul bir aralıkta olduğunu kontrol et
        
        Gradient explosion tespiti için:
        - Gradient norm'u çok büyük olmamalı (< 1000)
        - Gradient norm'u çok küçük olmamalı (> 1e-6)
        """
        model = CevahirNeuralNetwork(
            learning_rate=base_config["learning_rate"],
            dropout=base_config["dropout"],
            vocab_size=base_config["vocab_size"],
            embed_dim=base_config["embed_dim"],
            seq_proj_dim=base_config["seq_proj_dim"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            tie_weights=True,
            log_level=logging.WARNING,
        )
        
        x = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        targets = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        
        model.train()
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, base_config["vocab_size"]),
            targets.view(-1)
        )
        loss.backward()
        
        # Tüm gradient norm'larını kontrol et
        max_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                # ✅ ASSERT: Gradient norm makul aralıkta
                assert grad_norm < 1000.0, (
                    f"Gradient explosion tespit edildi! "
                    f"Layer: {name}, Norm: {grad_norm:.2f} (beklenen: < 1000.0)"
                )
                assert grad_norm > 1e-6, (
                    f"Gradient çok küçük! "
                    f"Layer: {name}, Norm: {grad_norm:.6f} (beklenen: > 1e-6)"
                )
        
        # ✅ ASSERT: Max gradient norm makul
        assert max_grad_norm < 100.0, (
            f"Genel gradient explosion! "
            f"Max norm: {max_grad_norm:.2f} (beklenen: < 100.0)"
        )
    
    def test_embedding_gradient_explosion(self, base_config):
        """
        ✅ KRİTİK TEST: Embedding layer gradient explosion tespiti
        
        Weight tying kullanıldığında embedding gradient'leri özellikle büyük olabilir.
        Bu test, embedding gradient'lerinin makul bir aralıkta olduğunu kontrol eder.
        """
        model = CevahirNeuralNetwork(
            learning_rate=base_config["learning_rate"],
            dropout=base_config["dropout"],
            vocab_size=base_config["vocab_size"],
            embed_dim=base_config["embed_dim"],
            seq_proj_dim=base_config["seq_proj_dim"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            tie_weights=True,
            log_level=logging.WARNING,
        )
        
        x = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        targets = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        
        model.train()
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, base_config["vocab_size"]),
            targets.view(-1)
        )
        loss.backward()
        
        # Embedding gradient'ini kontrol et
        embedding_grad = model.dil_katmani.language_embedding.embedding.weight.grad
        embedding_grad_norm = embedding_grad.norm().item()
        
        # ✅ ASSERT: Embedding gradient norm makul
        # Weight tying nedeniyle 2x büyük olabilir, ama çok büyük olmamalı
        assert embedding_grad_norm < 500.0, (
            f"Embedding gradient explosion! "
            f"Norm: {embedding_grad_norm:.2f} (beklenen: < 500.0)"
        )


class TestModelArchitectureValidation:
    """Model Mimarisi Validasyon Testleri"""
    
    def test_vocab_embed_ratio(self, base_config):
        """
        ✅ KRİTİK TEST: Vocab size / Embedding dimension ratio kontrolü
        
        Endüstri standardı:
        - GPT-2: ratio ≈ 65
        - GPT-3: ratio ≈ 4
        - Normal: 50-100 arası
        
        Ratio çok büyükse (> 200) gradient explosion riski var!
        """
        vocab_size = base_config["vocab_size"]
        embed_dim = base_config["embed_dim"]
        ratio = vocab_size / embed_dim
        
        # ✅ ASSERT: Ratio makul aralıkta
        assert ratio < 200, (
            f"Vocab/Embed ratio çok büyük! "
            f"Ratio: {ratio:.2f} (beklenen: < 200, önerilen: 50-100)"
        )
        assert ratio > 1, (
            f"Vocab/Embed ratio çok küçük! "
            f"Ratio: {ratio:.2f} (beklenen: > 1)"
        )
    
    def test_output_layer_normalization(self, base_config):
        """
        ✅ KRİTİK TEST: Output layer normalization'ın doğru çalıştığını kontrol et
        
        Output layer'dan önce normalization olmalı (GPT-2/3/4, LLaMA standardı).
        Bu, logits aralığını kontrol altına alır.
        """
        model = CevahirNeuralNetwork(
            learning_rate=base_config["learning_rate"],
            dropout=base_config["dropout"],
            vocab_size=base_config["vocab_size"],
            embed_dim=base_config["embed_dim"],
            seq_proj_dim=base_config["seq_proj_dim"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            use_rmsnorm=True,
            log_level=logging.WARNING,
        )
        
        # ✅ ASSERT: Output normalization var
        assert hasattr(model, 'output_norm'), (
            "Output layer normalization eksik! (GPT-2/3/4, LLaMA standardı)"
        )
        assert model.output_norm is not None, (
            "Output layer normalization None!"
        )
    
    def test_logits_range(self, base_config):
        """
        ✅ KRİTİK TEST: Logits aralığının makul olduğunu kontrol et
        
        Logits çok geniş aralıkta olmamalı:
        - Normal: -10 ile 10 arası
        - Çok geniş: -50 ile 50 arası (sorun!)
        """
        model = CevahirNeuralNetwork(
            learning_rate=base_config["learning_rate"],
            dropout=base_config["dropout"],
            vocab_size=base_config["vocab_size"],
            embed_dim=base_config["embed_dim"],
            seq_proj_dim=base_config["seq_proj_dim"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            use_rmsnorm=True,
            log_level=logging.WARNING,
        )
        
        x = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
            
            logits_min = logits.min().item()
            logits_max = logits.max().item()
            logits_range = logits_max - logits_min
        
        # ✅ ASSERT: Logits aralığı makul
        assert logits_range < 50.0, (
            f"Logits aralığı çok geniş! "
            f"Range: {logits_range:.2f} (beklenen: < 50.0, normal: 10-20)"
        )
        assert logits_min > -30.0, (
            f"Logits minimum çok küçük! "
            f"Min: {logits_min:.2f} (beklenen: > -30.0)"
        )
        assert logits_max < 30.0, (
            f"Logits maximum çok büyük! "
            f"Max: {logits_max:.2f} (beklenen: < 30.0)"
        )


class TestForwardBackwardConsistency:
    """Forward/Backward Pass Consistency Testleri"""
    
    def test_forward_backward_consistency(self, base_config):
        """
        ✅ KRİTİK TEST: Forward ve backward pass'in tutarlı olduğunu kontrol et
        
        Forward pass'te NaN/Inf olmamalı.
        Backward pass'te gradient'ler NaN/Inf olmamalı.
        """
        model = CevahirNeuralNetwork(
            learning_rate=base_config["learning_rate"],
            dropout=base_config["dropout"],
            vocab_size=base_config["vocab_size"],
            embed_dim=base_config["embed_dim"],
            seq_proj_dim=base_config["seq_proj_dim"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            tie_weights=True,
            log_level=logging.WARNING,
        )
        
        x = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        targets = torch.randint(0, base_config["vocab_size"], (2, 10), dtype=torch.long)
        
        model.train()
        logits, _ = model(x)
        
        # ✅ ASSERT: Forward pass'te NaN/Inf yok
        assert not torch.isnan(logits).any(), "Forward pass'te NaN tespit edildi!"
        assert not torch.isinf(logits).any(), "Forward pass'te Inf tespit edildi!"
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, base_config["vocab_size"]),
            targets.view(-1)
        )
        
        # ✅ ASSERT: Loss finite
        assert torch.isfinite(loss), f"Loss finite değil! Loss: {loss.item()}"
        
        loss.backward()
        
        # ✅ ASSERT: Backward pass'te gradient'ler NaN/Inf yok
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"Gradient NaN tespit edildi! Layer: {name}"
                )
                assert not torch.isinf(param.grad).any(), (
                    f"Gradient Inf tespit edildi! Layer: {name}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

