# -*- coding: utf-8 -*-
"""
Entropy metrikleri – derin inceleme testleri.

AdvancedTokenMetrics.compute_per_token_accuracy içindeki entropy hesabının:
- PAD-dışı (content-only) ortalaması
- Sayısal kararlılık (float32)
- Uniform dağılımda max entropy
doğruluğunu test eder.
"""

import os
import sys
import math
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training_management.v2.metrics.advanced_token_metrics import AdvancedTokenMetrics


@pytest.fixture
def metrics():
    """vocab_size=100, PAD=0, BOS=1, EOS=2, UNK=3"""
    return AdvancedTokenMetrics(
        vocab_size=100,
        special_tokens_dict={0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"},
    )


class TestEntropyContentOnly:
    """Entropy sadece content (PAD-dışı) pozisyonlarda hesaplanmalı."""

    def test_entropy_uniform_logits_gives_max_entropy(self, metrics):
        """Uniform logits -> entropy ≈ ln(vocab_size)."""
        B, T, V = 2, 10, 100
        logits = torch.zeros(B, T, V)
        targets = torch.randint(1, V, (B, T))
        _, _, _, entropy = metrics.compute_per_token_accuracy(logits=logits, targets=targets)
        max_ent = math.log(100)
        assert abs(entropy - max_ent) < 0.01, f"Uniform: entropy={entropy:.4f}, max={max_ent:.4f}"

    def test_entropy_peaked_logits_lower_than_max(self, metrics):
        """Tek token'a yüksek olasılık -> entropy < max."""
        B, T, V = 2, 10, 100
        logits = torch.zeros(B, T, V)
        logits[:, :, 5] = 10.0
        targets = torch.randint(1, V, (B, T))
        _, _, _, entropy = metrics.compute_per_token_accuracy(logits=logits, targets=targets)
        max_ent = math.log(100)
        assert entropy < max_ent * 0.5, f"Peaked entropy should be low: {entropy:.4f}"

    def test_entropy_content_only_excludes_pad(self, metrics):
        """
        Çoğu pozisyon PAD ise: eski davranış (tüm pozisyonlar) ortalama entropy'yi
        PAD'lere doğru çeker (uniform ~ln(V)). Content-only ortalaması farklı olmalı.
        Bu test: PAD'lerde uniform, content'te peaked verince content-only entropy
        düşük, tüm-pozisyon ortalaması yüksek olur.
        """
        B, T, V = 2, 20, 100
        # İlk 2 pozisyon content (peaked), geri kalan PAD (uniform)
        logits = torch.zeros(B, T, V)
        logits[:, :2, 7] = 10.0
        targets = torch.zeros(B, T, dtype=torch.long)
        targets[:, :2] = 7
        _, _, _, content_entropy = metrics.compute_per_token_accuracy(logits=logits, targets=targets)
        # Sadece 2 content pozisyon var, ikisi de peaked -> düşük entropy
        assert content_entropy < 2.0, f"Content-only entropy (peaked) should be low: {content_entropy:.4f}"

    def test_entropy_float32_stability(self, metrics):
        """float16 logits ile çağrı NaN/Inf üretmemeli."""
        B, T, V = 2, 5, 100
        logits = torch.randn(B, T, V, dtype=torch.float16) * 0.1
        targets = torch.randint(0, V, (B, T))
        _, _, _, entropy = metrics.compute_per_token_accuracy(logits=logits, targets=targets)
        assert not math.isnan(entropy) and not math.isinf(entropy), "float16 logits -> entropy NaN/Inf olmamalı"
        assert 0 <= entropy <= math.log(V) + 0.1, f"Entropy [0, ln(V)] aralığında: {entropy}"

    def test_entropy_flattened_shape_BT_V(self, metrics):
        """(B*T, V) shape de desteklenmeli (docstring'deki gibi)."""
        BT, V = 20, 100
        logits = torch.zeros(BT, V)
        targets = torch.randint(1, V, (BT,))
        _, _, _, entropy = metrics.compute_per_token_accuracy(logits=logits, targets=targets)
        max_ent = math.log(100)
        assert abs(entropy - max_ent) < 0.01
        assert not math.isnan(entropy)


class TestEntropyFormatAndWarnings:
    """format_token_metrics ve check_critical_issues entropy kullanımı."""

    def test_format_shows_max_entropy(self, metrics):
        """format_token_metrics entropy satırında max değer gösterilmeli."""
        msg = metrics.format_token_metrics(
            overall_acc=0.5,
            special_accs={"PAD": (0.0, 0), "EOS": (0.1, 10), "BOS": (0.2, 5), "UNK": (0.0, 0)},
            top5_acc=0.6,
            entropy=5.0,
            grad_norm=1.0,
        )
        assert "Entropy (content only)" in msg
        assert "5.0000" in msg

    def test_low_entropy_triggers_mode_collapse_warning(self, metrics):
        """Çok düşük entropy -> mode collapse uyarısı."""
        max_ent = math.log(100)
        warnings = metrics.check_critical_issues(
            overall_acc=0.01,
            special_accs={"EOS": (0.0, 10), "BOS": (0.0, 5), "PAD": (0.0, 0), "UNK": (0.0, 0)},
            entropy=max_ent * 0.05,
        )
        assert any("Mode Collapse" in w for w in warnings)
