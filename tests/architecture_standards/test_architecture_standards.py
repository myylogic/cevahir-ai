# -*- coding: utf-8 -*-
"""
Mimari standart testleri (docs/ARCHITECTURE_TEST_STANDARDS.md).
Endüstri ve akademik kriterlere göre modül ve tam model davranışı doğrulanır.
Testler önce yazılır; mimari düzeltmeler testlerin geçmesini sağlar.
"""
import os
import sys
import math
import logging
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- Standart sabitleri (ARCHITECTURE_TEST_STANDARDS.md ile uyumlu) ----
LOGIT_ABS_MAX = 50.0           # L1: logit patlaması sınırı
MAX_PROB_COLLAPSE = 0.99       # L3: tek token baskınlık eşiği
RMS_MIN, RMS_MAX = 0.01, 100.0 # O1: RMSNorm çıkış RMS aralığı
EMBED_NORM_MAX = 50.0          # O2: embedding vektör norm üst sınırı
GRAD_NORM_EXPLODE = 1e8        # G2: gradient patlaması eşiği
EMBED_GRAD_RATIO_MAX = 500.0   # E1: embedding grad / medyan(grad) üst sınırı
ATTN_WEIGHTS_SUM_TOL = 0.01    # A1: attention satır toplamı 1.0'dan sapma toleransı


def _logit_stats(logits):
    """logits [B,T,V] -> last position entropy, max_prob."""
    last = logits[0, -1, :].float().detach()
    last = last - last.max()
    probs = torch.softmax(last, dim=-1)
    eps = 1e-12
    entropy = - (probs * torch.log(probs + eps)).sum().item()
    max_prob = probs.max().item()
    return entropy, max_prob


def _rms_last_dim(x):
    """Son boyut üzerinde RMS: sqrt(mean(x^2))."""
    return torch.sqrt(torch.mean(x.float() ** 2, dim=-1) + 1e-12)


class TestNumericalStabilityFullModel:
    """S1, S2, L1, L2, L3: Tam model forward - NaN/Inf, logit aralığı, dağılım."""

    def test_s1_s2_no_nan_inf_in_logits(self, minimal_model, batch_minimal):
        """S1/S2: Logit çıkışında NaN ve Inf olmamalı."""
        with torch.no_grad():
            out, _ = minimal_model(batch_minimal)
        assert out is not None
        assert not torch.isnan(out).any().item(), "Logits contain NaN"
        assert not torch.isinf(out).any().item(), "Logits contain Inf"

    def test_l1_logits_bounded(self, minimal_model, batch_minimal):
        """L1: Logit mutlak değeri LOGIT_ABS_MAX (50) içinde olmalı."""
        with torch.no_grad():
            out, _ = minimal_model(batch_minimal)
        abs_max = out.abs().max().item()
        assert abs_max <= LOGIT_ABS_MAX, (
            f"Logit abs max {abs_max} > {LOGIT_ABS_MAX} (softmax patlaması riski)"
        )

    def test_l2_entropy_not_nan(self, minimal_model, batch_minimal):
        """L2: Son pozisyon softmax entropy NaN olmamalı."""
        with torch.no_grad():
            out, _ = minimal_model(batch_minimal)
        entropy, _ = _logit_stats(out)
        assert not math.isnan(entropy), "Last-position entropy is NaN (mode collapse indicator)"

    def test_l3_no_single_token_dominance(self, minimal_model, batch_minimal):
        """L3: Son pozisyonda max_prob < MAX_PROB_COLLAPSE (tek token baskın olmasın)."""
        with torch.no_grad():
            out, _ = minimal_model(batch_minimal)
        _, max_prob = _logit_stats(out)
        assert max_prob < MAX_PROB_COLLAPSE, (
            f"Last position max_prob {max_prob:.4f} >= {MAX_PROB_COLLAPSE} (mode collapse)"
        )


class TestRMSNormStandards:
    """O1: RMSNorm çıkışı sayısal ve ölçek açısından standartlara uygun."""

    @pytest.fixture
    def rms_module(self):
        from src.neural_network_module.ortak_katman_module.rms_norm import RMSNorm
        return RMSNorm(dim=64, eps=1e-6, log_level=logging.WARNING)

    def test_o1_no_nan_inf(self, rms_module):
        """S1/S2 (RMSNorm): Çıkışta NaN/Inf yok."""
        x = torch.randn(2, 8, 64)
        out = rms_module(x)
        assert not torch.isnan(out).any().item()
        assert not torch.isinf(out).any().item()

    def test_o1_output_rms_in_range(self, rms_module):
        """O1: Çıkış RMS (son boyut) RMS_MIN..RMS_MAX aralığında."""
        x = torch.randn(2, 8, 64) * 10  # Büyük girdi
        out = rms_module(x)
        rms = _rms_last_dim(out)
        rms_min, rms_max = rms.min().item(), rms.max().item()
        assert RMS_MIN <= rms_min and rms_max <= RMS_MAX, (
            f"RMSNorm output RMS {rms_min:.4f}..{rms_max:.4f} outside [{RMS_MIN}, {RMS_MAX}]"
        )


class TestEmbeddingStandards:
    """O2: Embedding çıkışı NaN/Inf yok, vektör normu sınırlı."""

    @pytest.fixture
    def embed_module(self):
        from src.neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding
        return LanguageEmbedding(
            vocab_size=2000,
            embed_dim=64,
            init_method="xavier_normal",
            scale_by_sqrt=False,
            log_level=logging.WARNING,
        )

    def test_o2_no_nan_inf(self, embed_module):
        """S1/S2 (Embedding): Çıkışta NaN/Inf yok."""
        x = torch.randint(0, 2000, (2, 8))
        out = embed_module(x)
        assert not torch.isnan(out).any().item()
        assert not torch.isinf(out).any().item()

    def test_o2_norm_bounded(self, embed_module):
        """O2: Token vektör normu EMBED_NORM_MAX (50) altında."""
        x = torch.randint(0, 2000, (2, 8))
        out = embed_module(x)
        norms = torch.norm(out, dim=-1)
        assert norms.max().item() <= EMBED_NORM_MAX, (
            f"Embedding norm max {norms.max().item()} > {EMBED_NORM_MAX}"
        )


class TestGradientFlow:
    """G1, G2, E1: Backward sonrası NaN/Inf, patlama ve embedding baskınlığı."""

    def test_g1_g2_backward_no_nan_explosion(self, minimal_model, batch_minimal):
        """G1/G2: Backward sonrası grad'da NaN/Inf yok, grad norm < GRAD_NORM_EXPLODE."""
        minimal_model.train()
        out, _ = minimal_model(batch_minimal)
        loss = out.mean()
        loss.backward()
        for name, p in minimal_model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            assert not torch.isnan(g).any().item(), f"Grad NaN in {name}"
            assert not torch.isinf(g).any().item(), f"Grad Inf in {name}"
            gn = g.norm().item()
            assert gn < GRAD_NORM_EXPLODE, f"Grad norm {gn} >= {GRAD_NORM_EXPLODE} in {name}"
        minimal_model.eval()

    def test_e1_embedding_grad_not_dominant(self, minimal_model, batch_minimal):
        """E1: Embedding grad norm, diğer parametrelerin medyan grad norm'unun EMBED_GRAD_RATIO_MAX katını geçmemeli."""
        minimal_model.train()
        out, _ = minimal_model(batch_minimal)
        loss = out.mean()
        loss.backward()
        grad_norms = []
        embed_grad_norm = None
        for name, p in minimal_model.named_parameters():
            if p.grad is None:
                continue
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if "embedding.embedding.weight" in name:
                embed_grad_norm = gn
        if embed_grad_norm is None or len(grad_norms) < 2:
            pytest.skip("Embedding grad or multiple params not available")
        import numpy as np
        median_gn = float(np.median(grad_norms))
        if median_gn < 1e-12:
            median_gn = 1e-12
        ratio = embed_grad_norm / median_gn
        assert ratio <= EMBED_GRAD_RATIO_MAX, (
            f"Embedding grad norm ratio {ratio:.0f} > {EMBED_GRAD_RATIO_MAX} (embedding grad dominant)"
        )
        minimal_model.eval()


class TestAttentionStandards:
    """A1: Attention ağırlıkları NaN yok, satır toplamı ~1 (softmax)."""

    def test_a1_attention_weights_valid_and_sum_one(self, minimal_model, batch_minimal):
        """A1: attn_weights (varsa) NaN yok ve her satır toplamı 1'e yakın."""
        with torch.no_grad():
            logits, attn_weights = minimal_model(batch_minimal)
        if attn_weights is None:
            pytest.skip("Model returns no attention weights (e.g. Flash Attention)")
        assert not torch.isnan(attn_weights).any().item(), "Attention weights contain NaN"
        # (B, H, Lq, Sk) -> son eksen üzerinde toplam = 1
        row_sums = attn_weights.sum(dim=-1)
        diff = (row_sums - 1.0).abs()
        assert diff.max().item() <= ATTN_WEIGHTS_SUM_TOL, (
            f"Attention row sum max diff {diff.max().item()} > {ATTN_WEIGHTS_SUM_TOL}"
        )


class TestPositionalStandards:
    """P1: RoPE apply_rotary_pos_emb çıkışı NaN/Inf yok, şekil korunur."""

    def test_p1_rope_apply_no_nan_inf_shape(self, rope_pe):
        """P1: RoPE apply_rotary_pos_emb sonrası NaN/Inf yok, şekil aynı."""
        B, T, D = 2, 16, 32  # head_dim=32 (rope_pe.rope_dim)
        x = torch.randn(B, T, D)
        with torch.no_grad():
            out = rope_pe.apply_rotary_pos_emb(x)
        assert out.shape == x.shape, f"Shape changed: {out.shape} vs {x.shape}"
        assert not torch.isnan(out).any().item(), "RoPE output contains NaN"
        assert not torch.isinf(out).any().item(), "RoPE output contains Inf"

    def test_p1_rope_apply_4d_no_nan_inf_shape(self, rope_pe):
        """P1 (4D): RoPE apply_rotary_pos_emb [B,H,T,D] sonrası NaN/Inf yok, şekil aynı."""
        B, H, T, D = 2, 2, 16, 32
        x = torch.randn(B, H, T, D)
        with torch.no_grad():
            out = rope_pe.apply_rotary_pos_emb(x)
        assert out.shape == x.shape, f"Shape changed: {out.shape} vs {x.shape}"
        assert not torch.isnan(out).any().item(), "RoPE 4D output contains NaN"
        assert not torch.isinf(out).any().item(), "RoPE 4D output contains Inf"


class TestFFNStandards:
    """F1: FFN forward çıkışı NaN/Inf yok, şekil [B,T,embed_dim] korunur."""

    def test_f1_ffn_no_nan_inf_shape(self, ffn_swiglu):
        """F1: FFN (SwiGLU) forward sonrası NaN/Inf yok, şekil korunur."""
        B, T, D = 2, 16, 64
        x = torch.randn(B, T, D)
        with torch.no_grad():
            out = ffn_swiglu(x)
        assert out.shape == x.shape, f"FFN shape changed: {out.shape} vs {x.shape}"
        assert not torch.isnan(out).any().item(), "FFN output contains NaN"
        assert not torch.isinf(out).any().item(), "FFN output contains Inf"


class TestLogitScaleRegression:
    """Logit 1/sqrt(dim) ölçeklemesi: olmadan L1 ihlal edilebilir (regresyon testi)."""

    def test_full_model_logits_use_scale_factor(self, minimal_model, batch_minimal):
        """Tam model logitleri makul aralıkta (scale faktörü sayesinde)."""
        with torch.no_grad():
            out, _ = minimal_model(batch_minimal)
        # Scale uygulanmış modelde logitler ~ [-2, 2] civarı beklenir (minimal config)
        assert out.abs().max().item() <= LOGIT_ABS_MAX
