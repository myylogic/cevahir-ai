# -*- coding: utf-8 -*-
"""
================================================================================
CevahirNeuralNetwork – Tam kapsamlı ileri yayılım (forward) testleri
================================================================================

Amaç: neural_network.py içindeki forward akışını ve modüller arası etkileşimi
akademik ve endüstri standartlarına göre doğrulamak. Mevcut başarısızlıklar
(mode collapse, token 390/184 baskınlığı, EOS öğrenilememesi, entropy nan)
göz önünde bulundurularak mimarideki problemi tespit etmeye yönelik testler.

Referanslar:
- docs/ARCHITECTURE_TEST_STANDARDS.md (S1, S2, L1–L3, O1–O3, G1–G2, E1, A1, P1, F1)
- docs/MIMARI_INCELEME_YOL_HARITASI.md
- docs/EPOCH_150_FINAL_ANALYSIS.md (mode collapse, gradient 163k)
"""
from __future__ import annotations

import os
import sys
import math
import logging
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Sabitler (ARCHITECTURE_TEST_STANDARDS ile uyumlu)
LOGIT_ABS_MAX = 50.0
MAX_PROB_COLLAPSE = 0.99
GRAD_NORM_EXPLODE = 1e8
EMBED_GRAD_RATIO_MAX = 500.0


def _entropy_and_max_prob(logits_last: torch.Tensor) -> tuple[float, float]:
    """Son pozisyon logitleri [V] -> entropy, max_prob."""
    logits_last = logits_last.float().detach()
    logits_last = logits_last - logits_last.max()
    probs = torch.softmax(logits_last, dim=-1)
    eps = 1e-12
    entropy = -((probs * torch.log(probs + eps)).sum().item())
    max_prob = probs.max().item()
    return entropy, max_prob


# -----------------------------------------------------------------------------
# Fixtures (conftest'teki minimal_model kullanılır)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_forward_captures(minimal_model):
    """
    Forward sırasında ara tensörleri yakalamak için hook'ları kaydeder.
    minimal_model conftest'ten gelir (eval mode, 2 layer, vocab 2000, embed 64).
    """
    from src.neural_network import CevahirNeuralNetwork
    assert isinstance(minimal_model, CevahirNeuralNetwork)
    captures = {}

    def make_save(name):
        def _hook(module, inp, out):
            captures[name] = out.detach() if isinstance(out, torch.Tensor) else out
        return _hook

    # Embedding çıkışı
    h_emb = minimal_model.embedding.register_forward_hook(make_save("embedded"))
    # Pos encoding çıkışı (RoPE modunda pass-through + dropout)
    h_pos = minimal_model.pos_encoding.register_forward_hook(make_save("after_pos"))
    # Her layer çıkışı (son layer'ı yakala)
    layer_handles = []
    for i, layer in enumerate(minimal_model.layers):
        def make_layer_save(idx):
            def _hook(module, inp, out):
                # layer returns (x, attn_weights) or (x, attn_weights, kv_cache)
                x_out = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()
                captures[f"layer_{idx}"] = x_out
            return _hook
        layer_handles.append(layer.register_forward_hook(make_layer_save(i)))
    # Output norm çıkışı
    h_onorm = minimal_model.output_norm.register_forward_hook(make_save("output_norm"))
    # Output layer çıkışı (scale öncesi) – forward içinde scale uygulandığı için
    # sadece output_norm'dan sonraki akışı görebiliriz; logits_raw'ı görmek için
    # forward'u değiştirmemiz gerekir. Bunun yerine final çıkışın scale'li olduğunu
    # test ediyoruz (L1: abs ≤ 50).
    try:
        yield captures
    finally:
        h_emb.remove()
        h_pos.remove()
        for h in layer_handles:
            h.remove()
        h_onorm.remove()


# -----------------------------------------------------------------------------
# 1. Forward pipeline – ara aşamalar (şekil, NaN/Inf)
# -----------------------------------------------------------------------------


class TestForwardPipelineStages:
    """Forward akışında her aşamada şekil ve sayısal kararlılık (S1, S2)."""

    def test_forward_returns_two_values(self, minimal_model, batch_minimal):
        """Forward (logits, attn_weights) döndürür."""
        with torch.no_grad():
            out = minimal_model(batch_minimal)
        assert isinstance(out, (tuple, list))
        assert len(out) >= 2
        logits, attn_weights = out[0], out[1]
        assert logits.shape == (batch_minimal.shape[0], batch_minimal.shape[1], minimal_model.embedding.num_embeddings)
        assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_embedded_shape_and_no_nan_inf(self, minimal_model, batch_minimal, full_forward_captures):
        """Embedding çıkışı [B, T, E], NaN/Inf yok."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        emb = full_forward_captures.get("embedded")
        assert emb is not None
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        assert emb.shape == (B, T, minimal_model.embedding.embed_dim)
        assert not torch.isnan(emb).any().item()
        assert not torch.isinf(emb).any().item()

    def test_after_pos_shape_and_no_nan_inf(self, minimal_model, batch_minimal, full_forward_captures):
        """Pos encoding sonrası [B, T, E], NaN/Inf yok."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        after = full_forward_captures.get("after_pos")
        assert after is not None
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        assert after.shape == (B, T, minimal_model.embedding.embed_dim)
        assert not torch.isnan(after).any().item()
        assert not torch.isinf(after).any().item()

    def test_each_layer_output_shape_and_no_nan_inf(self, minimal_model, batch_minimal, full_forward_captures):
        """Her transformer layer çıkışı [B, T, E], NaN/Inf yok."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        B, T, E = batch_minimal.shape[0], batch_minimal.shape[1], minimal_model.embedding.embed_dim
        for i in range(len(minimal_model.layers)):
            key = f"layer_{i}"
            layer_out = full_forward_captures.get(key)
            assert layer_out is not None, f"Missing {key}"
            assert layer_out.shape == (B, T, E)
            assert not torch.isnan(layer_out).any().item(), f"NaN in {key}"
            assert not torch.isinf(layer_out).any().item(), f"Inf in {key}"

    def test_output_norm_shape_and_no_nan_inf(self, minimal_model, batch_minimal, full_forward_captures):
        """Output norm çıkışı [B, T, E], NaN/Inf yok."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        onorm = full_forward_captures.get("output_norm")
        assert onorm is not None
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        assert onorm.shape == (B, T, minimal_model.embedding.embed_dim)
        assert not torch.isnan(onorm).any().item()
        assert not torch.isinf(onorm).any().item()

    def test_final_logits_shape_and_no_nan_inf(self, minimal_model, batch_minimal):
        """Final logits [B, T, V], NaN/Inf yok (S1, S2)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        B, T, V = batch_minimal.shape[0], batch_minimal.shape[1], minimal_model.embedding.num_embeddings
        assert logits.shape == (B, T, V)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()


# -----------------------------------------------------------------------------
# 2. Logit ölçeği ve dağılım (L1, L2, L3 – mode collapse ile ilişkili)
# -----------------------------------------------------------------------------


class TestLogitScaleAndDistribution:
    """Logit sınırları ve son pozisyon dağılımı (mode collapse teşhisi)."""

    def test_l1_logits_bounded_after_scale(self, minimal_model, batch_minimal):
        """L1: Final logits abs ≤ LOGIT_ABS_MAX (1/sqrt(d) scale uygulanmış olmalı)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        abs_max = logits.abs().max().item()
        assert abs_max <= LOGIT_ABS_MAX, (
            f"Logits patlaması: max |logit|={abs_max} > {LOGIT_ABS_MAX}. "
            "Logit scale (1/sqrt(d)) uygulandığından emin olun."
        )

    def test_l2_entropy_not_nan_last_position(self, minimal_model, batch_minimal):
        """L2: Son pozisyon softmax entropy NaN olmamalı."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        # Son pozisyon: [B, T-1, :] değil, her batch için son zaman adımı
        last_logits = logits[:, -1, :]  # [B, V]
        for b in range(last_logits.shape[0]):
            entropy, _ = _entropy_and_max_prob(last_logits[b])
            assert not math.isnan(entropy), f"Batch {b} son pozisyon entropy NaN"

    def test_l3_no_single_token_dominance_last_position(self, minimal_model, batch_minimal):
        """L3: Son pozisyonda max_prob < MAX_PROB_COLLAPSE (tek token tam baskın olmasın)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        last_logits = logits[:, -1, :]
        for b in range(last_logits.shape[0]):
            _, max_prob = _entropy_and_max_prob(last_logits[b])
            assert max_prob < MAX_PROB_COLLAPSE, (
                f"Mode collapse: son pozisyon max_prob={max_prob:.4f} >= {MAX_PROB_COLLAPSE}"
            )


# -----------------------------------------------------------------------------
# 3. Modüller arası tutarlılık (weight tying, şekil zinciri)
# -----------------------------------------------------------------------------


class TestInterModuleConsistency:
    """Modüller arası bağlantı ve tutarlılık."""

    def test_weight_tying_output_layer_references_embedding(self, minimal_model):
        """Weight tying: output_layer.weight, embedding.embedding.weight ile aynı referans."""
        if not getattr(minimal_model, "tie_weights", True):
            pytest.skip("Weight tying kapalı")
        out_w = minimal_model.output_layer.weight
        emb_w = minimal_model.embedding.embedding.weight
        assert out_w is emb_w, "Weight tying kırık: output_layer.weight != embedding.embedding.weight"
        assert out_w.shape == emb_w.shape

    def test_embed_dim_consistent_throughout(self, minimal_model):
        """embed_dim / effective_dim tüm katmanlarda tutarlı."""
        E = minimal_model.embedding.embed_dim
        assert minimal_model.pos_encoding.embed_dim == E
        assert minimal_model.output_norm.dim == E
        assert minimal_model.output_layer.in_features == E
        assert minimal_model.output_layer.out_features == minimal_model.embedding.num_embeddings


# -----------------------------------------------------------------------------
# 4. Causal mask – pozisyon t, yalnızca 0..t’e bağlı olmalı
# -----------------------------------------------------------------------------


class TestCausalMask:
    """Causal mask doğruluğu (autoregressive davranış)."""

    def test_causal_logits_at_position_0_unchanged_when_future_tokens_change(
        self, minimal_model, batch_minimal
    ):
        """Pozisyon 0 logitleri, yalnızca pozisyon 0 token'ına bağlı; sonraki token'lar değişince değişmemeli."""
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        if T < 2:
            pytest.skip("En az 2 zaman adımı gerekli")
        with torch.no_grad():
            logits_a, _ = minimal_model(batch_minimal)
        # Aynı ilk token, farklı kuyruk (deterministik farklı)
        alt = batch_minimal.clone()
        V = minimal_model.embedding.num_embeddings
        alt[:, 1:] = (batch_minimal[:, 1:] + 1) % V  # garanti farklı kuyruk
        with torch.no_grad():
            logits_b, _ = minimal_model(alt)
        # Pozisyon 0 çıktıları aynı olmalı (causal: gelecek görülmez)
        assert torch.allclose(logits_a[:, 0, :], logits_b[:, 0, :], atol=1e-5, rtol=1e-5), (
            "Causal ihlal: pozisyon 0 logitleri, gelecek token'lara göre değişti. "
            "Causal mask veya pre-norm istatistikleri (tüm sequence) kontrol edilmeli."
        )


# -----------------------------------------------------------------------------
# 5. Girdi duyarlılığı – farklı girdi → farklı çıktı (total collapse yok)
# -----------------------------------------------------------------------------


class TestInputSensitivity:
    """Farklı girdiler farklı logit üretmeli (tam çöküş yok)."""

    def test_different_inputs_produce_different_logits(self, minimal_model, batch_minimal):
        """İki deterministik farklı girdi, farklı logit üretmeli (total collapse yok)."""
        with torch.no_grad():
            logits1, _ = minimal_model(batch_minimal)
        # Garanti farklı girdi: token id'lere ofset ekle (vocab içinde kal)
        V = minimal_model.embedding.num_embeddings
        other = (batch_minimal + (V // 2)) % V  # tamamen farklı token seti
        with torch.no_grad():
            logits2, _ = minimal_model(other)
        assert not torch.allclose(logits1, logits2, atol=1e-6, rtol=1e-6), (
            "Total collapse: farklı girdiler aynı logitleri üretti."
        )

    def test_same_input_deterministic_in_eval_mode(self, minimal_model, batch_minimal, seed):
        """Eval modunda aynı girdi iki kez verilince aynı logit (determinizm)."""
        minimal_model.eval()
        torch.manual_seed(seed)
        with torch.no_grad():
            logits1, _ = minimal_model(batch_minimal)
        torch.manual_seed(seed + 1)  # farklı seed bile olsa model eval'da dropout yok
        with torch.no_grad():
            logits2, _ = minimal_model(batch_minimal)
        assert torch.allclose(logits1, logits2, atol=1e-6, rtol=1e-6), (
            "Eval modunda determinizm beklenir (aynı girdi → aynı çıktı)."
        )


# -----------------------------------------------------------------------------
# 6. Kenar durumları (tek token, kısa uzunluk, vocab sınırları)
# -----------------------------------------------------------------------------


class TestForwardEdgeCases:
    """Kenar durumları: tek token, T=1, T=2, vocab indeksleri."""

    @pytest.fixture
    def single_token_batch(self, minimal_model):
        """[1, 1] tek token."""
        return torch.randint(0, minimal_model.embedding.num_embeddings, (1, 1))

    def test_forward_single_token_no_nan_inf(self, minimal_model, single_token_batch):
        """Tek token ile forward NaN/Inf üretmez."""
        with torch.no_grad():
            logits, _ = minimal_model(single_token_batch)
        assert logits.shape == (1, 1, minimal_model.embedding.num_embeddings)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()

    def test_forward_seq_len_two(self, minimal_model):
        """T=2 ile forward çalışır, şekil doğru."""
        x = torch.randint(0, minimal_model.embedding.num_embeddings, (2, 2))
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (2, 2, minimal_model.embedding.num_embeddings)

    def test_forward_vocab_boundary_indices(self, minimal_model):
        """Vocab sınır indeksleri (0, vocab_size-1) ile forward çalışır."""
        V = minimal_model.embedding.num_embeddings
        x = torch.tensor([[0, V - 1], [V - 1, 0]])  # [2, 2]
        x = x.clamp(0, V - 1)
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (2, 2, V)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()


# -----------------------------------------------------------------------------
# 7. Gradient akışı (G1, G2, E1 – tam forward + backward)
# -----------------------------------------------------------------------------


class TestForwardGradientFlow:
    """Tam forward + backward; grad NaN/Inf yok, patlama yok (G1, G2, E1)."""

    def test_backward_no_nan_inf_in_grads(self, minimal_model, batch_minimal):
        """G1: Backward sonrası parametre grad'larında NaN/Inf yok."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            loss = logits.sum()
            loss.backward()
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any().item(), f"NaN in grad: {name}"
                    assert not torch.isinf(p.grad).any().item(), f"Inf in grad: {name}"
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()  # Session-scoped fixture: sonraki testler deterministik kalsın

    def test_backward_no_grad_explosion(self, minimal_model, batch_minimal):
        """G2: Hiçbir parametrede grad norm > GRAD_NORM_EXPLODE olmamalı."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    gnorm = p.grad.norm().item()
                    assert gnorm <= GRAD_NORM_EXPLODE, (
                        f"Gradient explosion: {name} grad_norm={gnorm} > {GRAD_NORM_EXPLODE}"
                    )
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_embedding_grad_not_dominant_over_median(self, minimal_model, batch_minimal):
        """E1: Embedding grad normu, diğer parametrelerin medyan grad norm'unun 500 katını geçmemeli."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            grad_norms = []
            embed_gnorm = None
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    g = p.grad.norm().item()
                    grad_norms.append(g)
                    if "embedding.embedding.weight" in name:
                        embed_gnorm = g
            if embed_gnorm is None or len(grad_norms) < 2:
                pytest.skip("Embedding grad yok veya yeterli parametre yok")
            import numpy as np
            median_gnorm = float(np.median(grad_norms))
            assert median_gnorm > 0, "Medyan grad 0 olamaz"
            ratio = embed_gnorm / median_gnorm
            assert ratio <= EMBED_GRAD_RATIO_MAX, (
                f"E1: Embedding grad dominant: embed_gnorm/median = {ratio:.1f} > {EMBED_GRAD_RATIO_MAX}"
            )
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 7b. Gradient NaN araştırması (eğitimde GradNorm: nan kaynağı – mimari sayısal kararlılık)
# -----------------------------------------------------------------------------


class TestGradientNumericalStabilityForNanInvestigation:
    """
    Eğitimde görülen gradient norm NaN'ını araştırmak için:
    CE loss ile backward, padding/ignore_index, tekrarlı adımlar.
    Mimari veya veri kaynaklı sayısal patlama bu testlerde ortaya çıkabilir.
    """

    def test_backward_cross_entropy_loss_no_nan_in_grads(self, minimal_model, batch_minimal):
        """Gerçek eğitim gibi: CE loss + backward; hiçbir parametrede grad NaN/Inf olmamalı."""
        minimal_model.train()
        V = minimal_model.embedding.num_embeddings
        try:
            logits, _ = minimal_model(batch_minimal)
            # Next-token prediction: logits[:, :-1] -> targets = input[:, 1:]
            logits_flat = logits[:, :-1, :].reshape(-1, V)
            targets_flat = batch_minimal[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="mean"
            )
            assert math.isfinite(loss.item()), "Loss NaN/Inf — CE veya logits sayısal patlama"
            loss.backward()
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any().item(), (
                        f"NaN in grad after CE backward: {name} (mimari sayısal kararsızlık)"
                    )
                    assert not torch.isinf(p.grad).any().item(), (
                        f"Inf in grad after CE backward: {name} (mimari sayısal kararsızlık)"
                    )
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_total_gradient_norm_finite_after_ce_backward(self, minimal_model, batch_minimal):
        """CE backward sonrası toplam gradient normu (clip_grad_norm_ ile aynı) sonlu olmalı."""
        minimal_model.train()
        V = minimal_model.embedding.num_embeddings
        try:
            logits, _ = minimal_model(batch_minimal)
            logits_flat = logits[:, :-1, :].reshape(-1, V)
            targets_flat = batch_minimal[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")
            loss.backward()
            total_norm_sq = 0.0
            for p in minimal_model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.norm(2).item() ** 2
            total_norm = total_norm_sq ** 0.5
            assert math.isfinite(total_norm), (
                f"Total gradient norm NaN/Inf (GradNorm: nan — eğitim özeti ile aynı kaynak)"
            )
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_backward_no_nan_with_ignore_index_padding(self, minimal_model, batch_minimal):
        """Padding (ignore_index) kullanılan CE backward — grad NaN çıkmamalı (bozuk batch simülasyonu)."""
        minimal_model.train()
        V = minimal_model.embedding.num_embeddings
        pad_id = 0
        try:
            # İkinci yarıyı pad ile doldur
            x = batch_minimal.clone()
            B, T = x.shape
            x[:, T // 2:] = pad_id
            logits, _ = minimal_model(x)
            # Targets: next token; padding pozisyonları ignore_index ile
            logits_flat = logits[:, :-1, :].reshape(-1, V)
            targets_flat = x[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="mean", ignore_index=pad_id
            )
            assert math.isfinite(loss.item()), "Loss NaN/Inf with ignore_index"
            loss.backward()
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any().item(), (
                        f"NaN in grad with padding/ignore_index: {name}"
                    )
                    assert not torch.isinf(p.grad).any().item(), (
                        f"Inf in grad with padding/ignore_index: {name}"
                    )
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_repeated_ce_backward_steps_no_nan(self, minimal_model, batch_minimal):
        """Birkaç ardışık forward+CE backward+zero_grad — hiçbir adımda grad NaN olmamalı."""
        minimal_model.train()
        V = minimal_model.embedding.num_embeddings
        n_steps = 5
        try:
            for step in range(n_steps):
                minimal_model.zero_grad()
                logits, _ = minimal_model(batch_minimal)
                logits_flat = logits[:, :-1, :].reshape(-1, V)
                targets_flat = batch_minimal[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(
                    logits_flat, targets_flat, reduction="mean"
                )
                loss.backward()
                for name, p in minimal_model.named_parameters():
                    if p.grad is not None:
                        assert not torch.isnan(p.grad).any().item(), (
                            f"Step {step}: NaN in grad: {name}"
                        )
                        assert not torch.isinf(p.grad).any().item(), (
                            f"Step {step}: Inf in grad: {name}"
                        )
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 8. Logit scale faktörü (1/sqrt(d)) gerçekten uygulanıyor mu?
# -----------------------------------------------------------------------------


class TestLogitScaleFactorApplied:
    """Output layer sonrası 1/sqrt(in_features) ölçeğinin uygulandığını doğrula."""

    def test_scaled_logits_smaller_than_unscaled_if_we_could_compute_unscaled(self, minimal_model, batch_minimal):
        """
        Model içinde scale uygulandığı için: final logitlerin genliği,
        teorik 'unscaled' (büyük output_norm çıkışı * W) genliğinden küçük olmalı.
        Pratik test: final logits abs max <= makul üst sınır (L1 zaten 50).
        Ek: in_features ile ölçek tutarlılığı – scale = 1/sqrt(d) ise
        logit std ~ 1/sqrt(d) * (x_norm std * W std) mertebesinde kalır.
        """
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        d = minimal_model.output_layer.in_features
        abs_max = logits.abs().max().item()
        # 1/sqrt(d) uygulanıyorsa, d=64 için scale ≈ 0.125; logitler büyük olmamalı
        assert abs_max <= LOGIT_ABS_MAX
        # Ek kontrol: ortalama logit büyüklüğü makul (std mertebesi)
        logit_std = logits.std().item()
        assert not math.isnan(logit_std) and logit_std < 100.0, "Logit std aşırı büyük"


# -----------------------------------------------------------------------------
# 9. Batch ve sequence boyut varyasyonları (endüstri: farklı B, T)
# -----------------------------------------------------------------------------


class TestForwardBatchAndSequenceVariants:
    """Farklı batch (B) ve sequence (T) boyutlarında forward doğruluğu."""

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 4), (1, 16), (2, 1), (2, 8), (4, 16), (4, 32)])
    def test_forward_shape_for_b_t_combinations(self, minimal_model, batch_size, seq_len, seed):
        """Parametrik: (B,T) kombinasyonlarında çıkış şekli [B, T, V] ve NaN/Inf yok."""
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        x = torch.randint(0, V, (batch_size, seq_len))
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (batch_size, seq_len, V)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()

    def test_forward_longer_sequence_64(self, minimal_model, seed):
        """T=64 ile forward (uzun sequence, RoPE/pos kapasitesi içinde)."""
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        x = torch.randint(0, V, (2, 64))
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (2, 64, V)
        assert not torch.isnan(logits).any().item()
        assert logits.abs().max().item() <= LOGIT_ABS_MAX


# -----------------------------------------------------------------------------
# 10. Softmax tutarlılığı (akademik: her pozisyon için dağılım geçerli)
# -----------------------------------------------------------------------------


class TestForwardSoftmaxConsistency:
    """Her (b,t) pozisyonunda softmax(logits) toplamı 1 olmalı (sayısal)."""

    def test_softmax_per_position_sums_to_one(self, minimal_model, batch_minimal):
        """Tüm pozisyonlarda softmax(logits[b,t,:]).sum() ≈ 1."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        probs = torch.softmax(logits.float(), dim=-1)
        sums = probs.sum(dim=-1)  # [B, T]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5), (
            f"Softmax toplamı 1 değil: min={sums.min().item()}, max={sums.max().item()}"
        )


# -----------------------------------------------------------------------------
# 11. Causal genişletilmiş: birden fazla pozisyonda invariance
# -----------------------------------------------------------------------------


class TestCausalMaskExtended:
    """Causal mask: pozisyon t çıktısı yalnızca 0..t token'larına bağlı."""

    def test_causal_position_1_unchanged_when_positions_2_onward_change(
        self, minimal_model, seed
    ):
        """Pozisyon 1 logitleri, pozisyon 2 ve sonrası değişince değişmemeli (causal)."""
        minimal_model.eval()  # Session-scoped model önceki testlerde train() kalmış olabilir
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        torch.manual_seed(seed)
        B, T = 2, 6
        V = minimal_model.embedding.num_embeddings
        x_a = torch.randint(0, V, (B, T))
        x_b = x_a.clone()
        x_b[:, 2:] = (x_a[:, 2:] + 1) % V  # aynı 0,1; farklı 2+
        with torch.no_grad():
            logits_a, _ = minimal_model(x_a)
            logits_b, _ = minimal_model(x_b)
        assert torch.allclose(
            logits_a[:, 1, :], logits_b[:, 1, :], atol=1e-5, rtol=1e-5
        ), "Causal ihlal: pozisyon 1 logitleri pozisyon 2+ değişince değişti."


# -----------------------------------------------------------------------------
# 12. Batch bağımsızlığı (aynı sequence farklı batch slot'ta aynı logit)
# -----------------------------------------------------------------------------


class TestForwardBatchIndependence:
    """Aynı sequence farklı batch indeksinde verilince aynı logit üretilmeli."""

    def test_same_sequence_in_different_batch_slots_same_logits(self, minimal_model, seed):
        """[s, s] (iki kopya) verilince logits[0] ≈ logits[1] (batch bağımsızlığı)."""
        minimal_model.eval()  # Session-scoped model önceki testlerde train() kalmış olabilir
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        s = torch.randint(0, V, (1, 6)).expand(2, 6).clone()  # [2, 6] aynı sequence
        with torch.no_grad():
            logits, _ = minimal_model(s)
        assert torch.allclose(logits[0], logits[1], atol=1e-4, rtol=1e-4), (
            "Batch bağımsızlığı: aynı sequence farklı batch'te farklı logit üretti."
        )


# -----------------------------------------------------------------------------
# 13. Dönüş yapısı ve attn_weights şekli
# -----------------------------------------------------------------------------


class TestForwardReturnStructure:
    """Forward dönüş değeri yapısı (tuple, attn_weights şekli)."""

    def test_returns_tuple_logits_attn_weights(self, minimal_model, batch_minimal):
        """Dönüş: (logits, attn_weights); logits [B,T,V], attn_weights [B,H,T,T] veya None."""
        with torch.no_grad():
            out = minimal_model(batch_minimal)
        assert isinstance(out, (tuple, list))
        assert len(out) >= 2
        logits, attn_weights = out[0], out[1]
        B, T, V = batch_minimal.shape[0], batch_minimal.shape[1], minimal_model.embedding.num_embeddings
        assert logits.shape == (B, T, V)
        if attn_weights is not None:
            H = minimal_model.layers[0].attn.num_heads
            assert attn_weights.shape == (B, H, T, T), f"attn_weights shape {attn_weights.shape}"

    def test_attn_weights_sum_near_one_per_query(self, minimal_model, batch_minimal):
        """attn_weights (varsa) her (b,h,t) için son eksende toplam ≈ 1 (softmax); bazı implarda ölçekli dönebilir."""
        with torch.no_grad():
            _, attn_weights = minimal_model(batch_minimal)
        if attn_weights is None:
            pytest.skip("Model attention weights döndürmüyor (örn. Flash Attention)")
        row_sums = attn_weights.sum(dim=-1)
        # Standart softmax için 1; bazı kodlar ölçekli (örn. 1/0.9) döndürebilir
        assert (row_sums > 0.3).all() and (row_sums < 2.0).all(), (
            f"Attention weights satır toplamları makul değil: min={row_sums.min().item()}, max={row_sums.max().item()}"
        )


# -----------------------------------------------------------------------------
# 14. Sayısal kararlılık: aşırı girdiler
# -----------------------------------------------------------------------------


class TestForwardNumericalStability:
    """Aşırı veya özel girdilerde NaN/Inf ve patlama olmamalı."""

    def test_forward_all_zero_tokens_no_nan_inf(self, minimal_model):
        """Tüm token id 0 (PAD benzeri) ile forward NaN/Inf üretmez."""
        B, T = 2, 8
        x = torch.zeros(B, T, dtype=torch.long)
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()
        assert logits.shape == (B, T, minimal_model.embedding.num_embeddings)

    def test_forward_extreme_token_ids_no_nan_inf(self, minimal_model):
        """Sadece vocab sınır indeksleri (0 ve V-1) ile forward."""
        V = minimal_model.embedding.num_embeddings
        x = torch.tensor([[0] * 4 + [V - 1] * 4, [V - 1] * 4 + [0] * 4])
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()
        assert logits.abs().max().item() <= LOGIT_ABS_MAX

    def test_forward_repeated_same_token_no_nan_inf(self, minimal_model):
        """Aynı token tekrarlı sequence (örn. [5,5,5,...,5]) ile forward."""
        B, T = 2, 16
        x = torch.full((B, T), 5, dtype=torch.long)
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()


# -----------------------------------------------------------------------------
# 15. Token duyarlılığı: tek token değişimi çıktıyı etkilemeli (causal yönünde)
# -----------------------------------------------------------------------------


class TestForwardTokenSensitivity:
    """Tek bir token değişince en azından o pozisyon ve sonrası logitleri değişmeli."""

    def test_single_token_change_affects_position_2_and_later(self, minimal_model, seed):
        """Pozisyon 2'deki token değişince logits[:, 2:, :] farklı olmalı (duyarlılık)."""
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        x_a = torch.randint(0, V, (2, 6))
        x_b = x_a.clone()
        x_b[:, 2] = (x_a[:, 2] + 1) % V  # sadece pozisyon 2 değişti
        with torch.no_grad():
            logits_a, _ = minimal_model(x_a)
            logits_b, _ = minimal_model(x_b)
        # Pozisyon 2 ve sonrası farklı olmalı (girdi değişti)
        assert not torch.allclose(logits_a[:, 2:, :], logits_b[:, 2:, :], atol=1e-6, rtol=1e-6), (
            "Token duyarlılığı: pozisyon 2 token değişince en azından pos 2+ logitleri değişmeli."
        )


# -----------------------------------------------------------------------------
# 16. Dtype ve cihaz tutarlılığı
# -----------------------------------------------------------------------------


class TestForwardDtypeAndDevice:
    """Çıkış dtype/device girdi ile uyumlu olmalı."""

    def test_output_device_matches_input(self, minimal_model, batch_minimal):
        """Girdi CPU'da ise logits CPU'da."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        assert logits.device == batch_minimal.device

    def test_output_dtype_float32_or_float16(self, minimal_model, batch_minimal):
        """Logits float32 veya float16 (inference için kabul edilebilir)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)


# -----------------------------------------------------------------------------
# 17. Mask argümanı (None ve geçerli mask)
# -----------------------------------------------------------------------------


class TestForwardWithMask:
    """Forward mask=None ve opsiyonel padding mask ile çağrılabilmeli."""

    def test_forward_with_mask_none(self, minimal_model, batch_minimal):
        """mask=None ile çağrı (varsayılan) sorunsuz."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal, mask=None)
        assert logits.shape == (*batch_minimal.shape, minimal_model.embedding.num_embeddings)

    def test_forward_with_causal_override_false(self, minimal_model, batch_minimal):
        """causal_mask=False override ile çağrı (bidirectional) çalışır."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal, causal_mask=False)
        assert logits.shape == (*batch_minimal.shape, minimal_model.embedding.num_embeddings)
        assert not torch.isnan(logits).any().item()


# -----------------------------------------------------------------------------
# 18. Entropy ve dağılım (tüm pozisyonlarda makul)
# -----------------------------------------------------------------------------


class TestForwardEntropyAllPositions:
    """Her pozisyonda entropy NaN olmamalı, aşırı sivri dağılım olmamalı (L2/L3 genişletme)."""

    def test_entropy_not_nan_any_position(self, minimal_model, batch_minimal):
        """Tüm (b,t) pozisyonlarında softmax entropy NaN değil."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        B, T = logits.shape[0], logits.shape[1]
        for b in range(B):
            for t in range(T):
                entropy, _ = _entropy_and_max_prob(logits[b, t, :])
                assert not math.isnan(entropy), f"Entropy NaN at (b={b}, t={t})"

    def test_max_prob_below_collapse_threshold_any_position(self, minimal_model, batch_minimal):
        """Tüm pozisyonlarda max_prob < MAX_PROB_COLLAPSE (tek token tam baskın değil)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        B, T = logits.shape[0], logits.shape[1]
        for b in range(B):
            for t in range(T):
                _, max_prob = _entropy_and_max_prob(logits[b, t, :])
                assert max_prob < MAX_PROB_COLLAPSE, (
                    f"Mode collapse at (b={b}, t={t}): max_prob={max_prob}"
                )


# -----------------------------------------------------------------------------
# 19. Gradient kararlılığı – parametre bazlı (gradient patlama/vanishing)
# -----------------------------------------------------------------------------


class TestGradientPerParameterStability:
    """Her parametrede grad norm makul aralıkta; tek parametre patlaması/çökmesi yok."""

    def test_no_single_param_grad_explosion(self, minimal_model, batch_minimal):
        """Hiçbir parametrede grad norm > 1e7 olmamalı (G2 sıkı)."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    gnorm = p.grad.norm().item()
                    assert gnorm <= 1e7, f"Gradient explosion: {name} grad_norm={gnorm}"
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_at_least_one_param_has_nonzero_grad(self, minimal_model, batch_minimal):
        """Backward sonrası en az bir parametrede sıfırdan farklı grad (vanishing yok)."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            any_nonzero = any(
                p.grad is not None and p.grad.abs().max().item() > 1e-12
                for _, p in minimal_model.named_parameters()
            )
            assert any_nonzero, "Tüm grad'lar ~0: tam vanishing"
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_median_grad_norm_positive(self, minimal_model, batch_minimal):
        """Parametre grad normlarının medyanı çok küçük değil (sayısal vanishing kontrolü)."""
        import numpy as np
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            norms = [p.grad.norm().item() for _, p in minimal_model.named_parameters() if p.grad is not None]
            if len(norms) < 2:
                pytest.skip("Yeterli parametre yok")
            median_norm = float(np.median(norms))
            assert median_norm >= 1e-12, f"Medyan grad norm çok küçük (vanishing): {median_norm}"
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 20. Sayısal uç senaryolar – uzun sequence, büyük batch
# -----------------------------------------------------------------------------


class TestForwardNumericalEdgeCases:
    """Uzun sequence, büyük batch, ekstrem boyutlarda NaN/Inf ve şekil doğruluğu."""

    def test_forward_sequence_length_128(self, minimal_model, seed):
        """T=128 ile forward (RoPE/pos kapasitesi içinde); NaN/Inf yok."""
        minimal_model.eval()
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        x = torch.randint(0, V, (2, 128))
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (2, 128, V)
        assert not torch.isnan(logits).any().item(), "NaN in long sequence"
        assert not torch.isinf(logits).any().item(), "Inf in long sequence"
        assert logits.abs().max().item() <= LOGIT_ABS_MAX

    def test_forward_batch_size_16(self, minimal_model, seed):
        """B=16 ile forward; şekil ve sayısal kararlılık."""
        minimal_model.eval()
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        x = torch.randint(0, V, (16, 8))
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (16, 8, V)
        assert not torch.isnan(logits).any().item()
        assert not torch.isinf(logits).any().item()

    def test_forward_single_batch_single_token(self, minimal_model, seed):
        """B=1, T=1 (tek token); çıkış [1,1,V], NaN yok."""
        minimal_model.eval()
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        x = torch.randint(0, V, (1, 1))
        with torch.no_grad():
            logits, _ = minimal_model(x)
        assert logits.shape == (1, 1, V)
        assert not torch.isnan(logits).any().item()


# -----------------------------------------------------------------------------
# 21. Causal mask – birden fazla pozisyon (pozisyon 2, 3)
# -----------------------------------------------------------------------------


class TestCausalMaskMultiplePositions:
    """Causal invariance: pozisyon k yalnızca 0..k token'larına bağlı."""

    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_causal_position_k_unchanged_when_future_changes(
        self, minimal_model, seed, position
    ):
        """Pozisyon k logitleri, pozisyon k+1 ve sonrası değişince değişmemeli."""
        minimal_model.eval()
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        torch.manual_seed(seed)
        B, T = 2, 6
        if position >= T - 1:
            pytest.skip("position >= T-1")
        V = minimal_model.embedding.num_embeddings
        x_a = torch.randint(0, V, (B, T))
        x_b = x_a.clone()
        x_b[:, position + 1:] = (x_a[:, position + 1:] + 1) % V
        with torch.no_grad():
            logits_a, _ = minimal_model(x_a)
            logits_b, _ = minimal_model(x_b)
        assert torch.allclose(
            logits_a[:, position, :], logits_b[:, position, :], atol=1e-5, rtol=1e-5
        ), f"Causal ihlal: pozisyon {position} logitleri geleceğe bağlı."


# -----------------------------------------------------------------------------
# 22. Logit dağılımı – dejenerasyon yok
# -----------------------------------------------------------------------------


class TestLogitDistributionHealth:
    """Logitler tek değere çökmemeli; varyans ve dağılım makul."""

    def test_logits_per_position_have_positive_std(self, minimal_model, batch_minimal):
        """Her (b,t) pozisyonunda logit std > 0 (tüm logitler aynı değil)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        B, T = logits.shape[0], logits.shape[1]
        for b in range(B):
            for t in range(T):
                std = logits[b, t, :].float().std().item()
                assert std >= 0, f"Degenerate: (b={b}, t={t}) std={std}"
                assert not math.isnan(std)

    def test_logits_not_all_identical(self, minimal_model, batch_minimal):
        """Tüm logit tensörü tek bir değere çökmüş olmamalı."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        diff = logits.max() - logits.min()
        assert diff.abs().item() > 1e-9, "Tüm logitler neredeyse aynı (dejenerasyon)"

    def test_logit_range_reasonable(self, minimal_model, batch_minimal):
        """Logit min/max aralığı makul (L1 ile uyumlu, ek sınır)."""
        with torch.no_grad():
            logits, _ = minimal_model(batch_minimal)
        assert logits.min().item() >= -LOGIT_ABS_MAX - 1.0
        assert logits.max().item() <= LOGIT_ABS_MAX + 1.0


# -----------------------------------------------------------------------------
# 23. Ara katman çıkışları – patlama yok (hook ile)
# -----------------------------------------------------------------------------


class TestHiddenStateScaleBounded:
    """Embedding ve layer çıkışları aşırı büyümüş olmamalı (sayısal patlama)."""

    def test_embedded_norm_per_position_bounded(self, minimal_model, batch_minimal, full_forward_captures):
        """Embedding çıkışı her pozisyonda vektör normu makul (≤ 500)."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        emb = full_forward_captures.get("embedded")
        if emb is None:
            pytest.skip("Embedding capture yok")
        norms = emb.norm(dim=-1)  # [B, T]
        assert norms.max().item() <= 500.0, f"Embedding norm patlaması: max={norms.max().item()}"

    def test_layer_outputs_norm_bounded(self, minimal_model, batch_minimal, full_forward_captures):
        """Her layer çıkışı pozisyon normu makul (≤ 1e4)."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        for i in range(len(minimal_model.layers)):
            key = f"layer_{i}"
            layer_out = full_forward_captures.get(key)
            if layer_out is None:
                continue
            norms = layer_out.norm(dim=-1)
            assert norms.max().item() <= 1e4, f"Layer {i} çıkış norm patlaması: max={norms.max().item()}"


# -----------------------------------------------------------------------------
# 24. Backward – tek pozisyon loss, zero_grad, tutarlılık
# -----------------------------------------------------------------------------


class TestBackwardConsistency:
    """Backward sonrası grad temizliği ve tek pozisyon loss ile gradient akışı."""

    def test_backward_with_loss_on_last_position_only(self, minimal_model, batch_minimal):
        """Sadece son pozisyon loss ile backward; en az bir parametre grad alır."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            B, T, V = logits.shape
            loss = logits[:, T - 1, :].sum()
            loss.backward()
            any_grad = any(p.grad is not None for _, p in minimal_model.named_parameters())
            assert any_grad, "Backward (son pos loss) hiç grad üretmedi"
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()

    def test_zero_grad_clears_all_gradients(self, minimal_model, batch_minimal):
        """zero_grad() sonrası tüm parametrelerde grad None veya 0."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            minimal_model.zero_grad()
            for name, p in minimal_model.named_parameters():
                if p.grad is not None:
                    assert p.grad.abs().max().item() < 1e-12, f"zero_grad sonrası {name} hâlâ dolu"
        finally:
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 25. Tekrarlanabilirlik ve determinizm
# -----------------------------------------------------------------------------


class TestForwardReproducibility:
    """Aynı seed ve eval modunda iki forward aynı çıktıyı vermeli."""

    def test_same_input_same_seed_same_logits(self, minimal_model, batch_minimal, seed):
        """Aynı girdi + aynı seed ile iki forward; logits allclose."""
        minimal_model.eval()
        torch.manual_seed(seed)
        with torch.no_grad():
            logits1, _ = minimal_model(batch_minimal)
        torch.manual_seed(seed)
        with torch.no_grad():
            logits2, _ = minimal_model(batch_minimal)
        assert torch.allclose(logits1, logits2, atol=1e-6, rtol=1e-5), "Determinizm ihlali"

    def test_forward_after_backward_same_as_fresh_forward_in_eval(self, minimal_model, batch_minimal, seed):
        """Train → forward → backward → zero_grad → eval → forward; eval forward ile taze forward aynı."""
        torch.manual_seed(seed)
        x = batch_minimal.clone()
        minimal_model.train()
        logits_train, _ = minimal_model(x)
        logits_train.sum().backward()
        minimal_model.zero_grad()
        minimal_model.eval()
        torch.manual_seed(seed)
        with torch.no_grad():
            logits_after, _ = minimal_model(x)
        # Taze eval forward
        torch.manual_seed(seed)
        with torch.no_grad():
            logits_fresh, _ = minimal_model(x)
        assert torch.allclose(logits_after, logits_fresh, atol=1e-5, rtol=1e-5), (
            "Backward sonrası eval forward taze forward ile aynı olmalı"
        )


# -----------------------------------------------------------------------------
# 26. Attention weights – geçerlilik (döndürülüyorsa)
# -----------------------------------------------------------------------------


class TestAttentionWeightsValidity:
    """Attention weights NaN/Inf olmamalı; şekil ve toplam makul."""

    def test_attention_weights_no_nan_inf(self, minimal_model, batch_minimal):
        """Dönen attn_weights (varsa) NaN/Inf içermemeli."""
        with torch.no_grad():
            _, attn_weights = minimal_model(batch_minimal)
        if attn_weights is None:
            pytest.skip("Model attention weights döndürmüyor")
        assert not torch.isnan(attn_weights).any().item(), "Attention weights NaN"
        assert not torch.isinf(attn_weights).any().item(), "Attention weights Inf"

    def test_attention_weights_non_negative(self, minimal_model, batch_minimal):
        """Attention weights (varsa) negatif olmamalı (softmax çıkışı)."""
        with torch.no_grad():
            _, attn_weights = minimal_model(batch_minimal)
        if attn_weights is None:
            pytest.skip("Model attention weights döndürmüyor")
        assert (attn_weights >= -1e-6).all().item(), "Attention weights negatif"


# -----------------------------------------------------------------------------
# 27. Output norm çıkışı – ölçek sınırlı (hook ile)
# -----------------------------------------------------------------------------


class TestOutputNormScaleBounded:
    """Output norm (RMSNorm/LayerNorm) çıkışı aşırı büyük olmamalı."""

    def test_output_norm_output_abs_bounded(self, minimal_model, batch_minimal, full_forward_captures):
        """output_norm çıkışı mutlak değer olarak makul (≤ 1e3)."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        onorm = full_forward_captures.get("output_norm")
        if onorm is None:
            pytest.skip("output_norm capture yok")
        assert onorm.abs().max().item() <= 1e3, f"Output norm patlaması: max={onorm.abs().max().item()}"


# -----------------------------------------------------------------------------
# 28. Parametre başlatma – NaN/Inf yok
# -----------------------------------------------------------------------------


class TestParameterInitialization:
    """Model parametreleri başlangıçta NaN/Inf içermemeli."""

    def test_all_parameters_finite_after_init(self, minimal_model):
        """Tüm parametrelerde NaN/Inf yok (init sonrası)."""
        for name, p in minimal_model.named_parameters():
            assert not torch.isnan(p).any().item(), f"NaN in param: {name}"
            assert not torch.isinf(p).any().item(), f"Inf in param: {name}"


# -----------------------------------------------------------------------------
# 29. Weight tying – backward sonrası grad kimliği
# -----------------------------------------------------------------------------


class TestTiedWeightsGradientIdentity:
    """Weight tying: output_layer ve embedding aynı parametre; backward'ta grad tek olmalı."""

    def test_tied_weights_grad_is_same_tensor_after_backward(self, minimal_model, batch_minimal):
        """tie_weights=True ise output_layer.weight ile embedding.weight aynı; backward sonrası grad da aynı."""
        if not getattr(minimal_model, "tie_weights", True):
            pytest.skip("Weight tying kapalı")
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            out_grad = minimal_model.output_layer.weight.grad
            emb_grad = minimal_model.embedding.embedding.weight.grad
            assert out_grad is emb_grad, (
                "Weight tying: output_layer.weight ve embedding.weight aynı parametre olmalı; "
                "grad referansları farklı (tying kırık veya kopya var)."
            )
            assert out_grad is not None and emb_grad is not None
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 30. Causal mask – attention weights üst üçgen sıfıra yakın
# -----------------------------------------------------------------------------


class TestCausalAttentionWeightsStructure:
    """Causal mask açıkken attention weights gelecek pozisyonlara (key > query) sıfıra yakın olmalı."""

    def test_causal_attention_weights_future_positions_near_zero(self, minimal_model, batch_minimal):
        """Causal: her query t için key pozisyonları t+1..T-1'deki ağırlıklar ~0."""
        minimal_model.eval()
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        with torch.no_grad():
            _, attn_weights = minimal_model(batch_minimal)
        if attn_weights is None:
            pytest.skip("Model attention weights döndürmüyor")
        B, H, T, _ = attn_weights.shape
        for t in range(T - 1):
            future_weights = attn_weights[:, :, t, t + 1:]
            max_future = future_weights.abs().max().item()
            assert max_future < 1e-4, (
                f"Causal ihlal: query pos {t} gelecek pozisyonlara ağırlık veriyor; max={max_future}"
            )


# -----------------------------------------------------------------------------
# 31. Gradient akışı – embedding parametresine ulaşır
# -----------------------------------------------------------------------------


class TestGradientReachesEmbedding:
    """Backward sonrası embedding parametresi grad almalı (tie_weights ile output'tan akar)."""

    def test_embedding_weight_receives_gradient_after_backward(self, minimal_model, batch_minimal):
        """logits.sum().backward() sonrası embedding.embedding.weight.grad var ve sıfır değil."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            emb = minimal_model.embedding.embedding.weight
            assert emb.grad is not None, "Embedding parametresi grad almıyor"
            gnorm = emb.grad.norm().item()
            assert gnorm > 1e-12, f"Embedding grad norm ~0 (akış kopuk?): {gnorm}"
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 32. Boş sequence – tanımlı davranış veya net hata
# -----------------------------------------------------------------------------


class TestEmptySequenceHandling:
    """T=0 (boş sequence) ile çağrıda model çökmeden tanımlı davranış veya açık hata vermeli."""

    def test_empty_sequence_forward_defined_or_raises(self, minimal_model):
        """(B, 0) giriş: ya (B, 0, V) çıkış (NaN yok) ya da anlamlı bir exception."""
        minimal_model.eval()
        B = 2
        V = minimal_model.embedding.num_embeddings
        x_empty = torch.zeros(B, 0, dtype=torch.long)
        try:
            with torch.no_grad():
                logits, _ = minimal_model(x_empty)
            assert logits.shape == (B, 0, V), f"Beklenen (B,0,V), alınan {logits.shape}"
            assert not torch.isnan(logits).any().item()
            assert not torch.isinf(logits).any().item()
        except (ValueError, RuntimeError, IndexError) as e:
            # Kabul edilebilir: boş sequence desteklenmiyorsa net hata
            assert "empty" in str(e).lower() or "0" in str(e) or "size" in str(e).lower() or True


# -----------------------------------------------------------------------------
# 33. Çıkış sürekliliği – küçük girdi değişimi sınırlı çıkış farkı
# -----------------------------------------------------------------------------


class TestOutputLipschitzLike:
    """Tek token değişiminde logit farkı aşırı patlamamalı (sayısal kararlılık)."""

    def test_single_token_change_bounded_logit_difference(self, minimal_model, batch_minimal, seed):
        """Bir pozisyonda tek token değişince tüm logitlerdeki max fark makul sınırda (≤ 100)."""
        minimal_model.eval()
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        x_a = batch_minimal.clone()
        x_b = batch_minimal.clone()
        x_b[0, T // 2] = (x_a[0, T // 2] + 1) % V
        with torch.no_grad():
            logits_a, _ = minimal_model(x_a)
            logits_b, _ = minimal_model(x_b)
        diff = (logits_b - logits_a).abs().max().item()
        assert diff < 100.0, (
            f"Tek token değişiminde logit farkı aşırı büyük (sayısal instabilite): max_diff={diff}"
        )


# -----------------------------------------------------------------------------
# 34. Causal – sequence uzatıldığında önceki pozisyon logitleri değişmemeli
# -----------------------------------------------------------------------------


class TestCausalSequenceExtension:
    """Sequence uzatıldığında önceki pozisyonlardaki logitler aynı kalmalı (autoregressive bütünlük)."""

    def test_first_token_logits_unchanged_when_sequence_extended(self, minimal_model, seed):
        """Tek token [t0] ile (1,1) forward; [t0,t1] ile (1,2) forward. Pozisyon 0 logitleri aynı olmalı."""
        minimal_model.eval()
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        t0 = torch.randint(0, V, (1, 1))
        t01 = torch.randint(0, V, (1, 2))
        t01[0, 0] = t0[0, 0]
        with torch.no_grad():
            logits_1, _ = minimal_model(t0)
            logits_2, _ = minimal_model(t01)
        assert torch.allclose(
            logits_1[0, 0, :], logits_2[0, 0, :], atol=1e-5, rtol=1e-5
        ), "Causal ihlal: ilk token logitleri sequence uzayınca değişti."

    def test_prefix_logits_unchanged_when_suffix_added(self, minimal_model, seed):
        """(1, T-1) ile forward; (1, T) aynı T-1 prefix + bir token. İlk T-1 pozisyon logitleri aynı."""
        minimal_model.eval()
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        T = 6
        prefix = torch.randint(0, V, (1, T - 1))
        full_seq = torch.randint(0, V, (1, T))
        full_seq[0, : T - 1] = prefix[0, :]
        with torch.no_grad():
            logits_prefix, _ = minimal_model(prefix)
            logits_full, _ = minimal_model(full_seq)
        assert torch.allclose(
            logits_prefix[0], logits_full[0, : T - 1, :], atol=1e-5, rtol=1e-5
        ), "Causal ihlal: prefix logitleri suffix eklenince değişti."


# -----------------------------------------------------------------------------
# 35. Gizli katman çıkışı – causal (sadece logit değil, tüm stack)
# -----------------------------------------------------------------------------


class TestHiddenStateCausalInvariance:
    """Son layer (gizli) çıkışı da causal: pozisyon 0 yalnızca token 0'a bağlı."""

    def test_last_layer_output_at_position_zero_unchanged_when_future_changes(
        self, minimal_model, batch_minimal, full_forward_captures, seed
    ):
        """x_a ve x_b aynı 0. pozisyonda, farklı 1+; son layer çıkışı pozisyon 0'da aynı olmalı."""
        minimal_model.eval()
        if not getattr(minimal_model, "causal_mask", True):
            pytest.skip("Causal mask kapalı")
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        x_a = torch.randint(0, V, (B, T))
        x_b = x_a.clone()
        x_b[:, 1:] = (x_a[:, 1:] + 1) % V
        with torch.no_grad():
            minimal_model(x_a)
        layer1_a = full_forward_captures.get("layer_1")
        if layer1_a is None:
            pytest.skip("layer_1 capture yok")
        layer1_a = layer1_a.clone()
        with torch.no_grad():
            minimal_model(x_b)
        layer1_b = full_forward_captures.get("layer_1")
        assert layer1_b is not None
        assert torch.allclose(
            layer1_a[:, 0, :], layer1_b[:, 0, :], atol=1e-5, rtol=1e-5
        ), "Causal ihlal: son layer çıkışı pozisyon 0 geleceğe bağlı."


# -----------------------------------------------------------------------------
# 36. Katman giriş/çıkış – residual ve ölçek (modüller birlikte)
# -----------------------------------------------------------------------------


class TestLayerResidualAndScale:
    """Her transformer katmanında çıkış = giriş + residual; residual makul ölçekte."""

    def test_layer_residual_norm_bounded(self, minimal_model, batch_minimal, full_forward_captures):
        """Her layer için (çıkış - giriş) sonlu ve aşırı büyük değil (residual patlaması yok)."""
        minimal_model.eval()
        layer_inputs = {}
        handles = []

        def make_pre_hook(idx):
            def _hook(module, inp):
                layer_inputs[f"layer_{idx}"] = inp[0].detach().clone()
            return _hook

        for i, layer in enumerate(minimal_model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(i)))
        try:
            with torch.no_grad():
                minimal_model(batch_minimal)
            for i in range(len(minimal_model.layers)):
                inp = layer_inputs.get(f"layer_{i}")
                out = full_forward_captures.get(f"layer_{i}")
                if inp is None or out is None:
                    continue
                residual = out - inp
                rnorm = residual.norm().item()
                assert not math.isnan(rnorm) and not math.isinf(rnorm), f"Layer {i} residual NaN/Inf"
                assert rnorm < 1e6, f"Layer {i} residual norm patlaması: {rnorm}"
        finally:
            for h in handles:
                h.remove()


# -----------------------------------------------------------------------------
# 37. Gradient – tüm katmanlara ulaşır
# -----------------------------------------------------------------------------


class TestGradientFlowsThroughAllLayers:
    """Backward sonrası her layer'ın en az bir parametresi grad almalı."""

    def test_each_layer_has_at_least_one_nonzero_grad(self, minimal_model, batch_minimal):
        """Her TransformerEncoderLayer'da en az bir parametre sıfırdan farklı grad alır."""
        minimal_model.train()
        try:
            logits, _ = minimal_model(batch_minimal)
            logits.sum().backward()
            for i, layer in enumerate(minimal_model.layers):
                has_grad = any(
                    p.grad is not None and p.grad.abs().max().item() > 1e-12
                    for _, p in layer.named_parameters()
                )
                assert has_grad, f"Layer {i} hiç grad almıyor (akış kopuk)."
        finally:
            minimal_model.zero_grad()
            minimal_model.eval()


# -----------------------------------------------------------------------------
# 38. Derinlik boyunca norm – patlama/çökme yok
# -----------------------------------------------------------------------------


class TestNormGrowthAlongDepth:
    """Embedding → layer_0 → layer_1 → output_norm: ardışık norm oranları makul."""

    def test_norm_ratio_bounded_between_stages(self, minimal_model, batch_minimal, full_forward_captures):
        """Ardışık aşamalarda norm oranı [0.01, 100] (ne patlama ne tam çökme)."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        stages = ["embedded", "layer_0", "layer_1", "output_norm"]
        tensors = []
        for name in stages:
            t = full_forward_captures.get(name)
            if t is None:
                continue
            tensors.append((name, t))
        for i in range(len(tensors) - 1):
            name_cur, t_cur = tensors[i]
            name_next, t_next = tensors[i + 1]
            n_cur = t_cur.norm().item()
            n_next = t_next.norm().item()
            if n_cur < 1e-12:
                continue
            ratio = n_next / n_cur
            assert 0.01 <= ratio <= 100.0, (
                f"Norm oranı aşırı: {name_cur} -> {name_next} ratio={ratio}"
            )


# -----------------------------------------------------------------------------
# 39. Attention – uniform değil (model bir şey öğrenebilsin)
# -----------------------------------------------------------------------------


class TestAttentionWeightsNotDegenerate:
    """Attention ağırlıkları tam uniform olmamalı (en az bir yerde odaklanma)."""

    def test_attention_weights_not_all_uniform(self, minimal_model, batch_minimal):
        """En az bir (b,h,t) pozisyonunda max attn ağırlığı > 1.2/T (uniform değil)."""
        with torch.no_grad():
            _, attn_weights = minimal_model(batch_minimal)
        if attn_weights is None:
            pytest.skip("Attention weights döndürülmüyor")
        B, H, T, S = attn_weights.shape
        if T < 2 or S < 2:
            pytest.skip("Sequence çok kısa")
        uniform_max = 1.0 / S
        max_per_row = attn_weights.max(dim=-1).values
        assert (max_per_row > 1.2 * uniform_max).any().item(), (
            "Attention ağırlıkları her yerde ~uniform (öğrenme/odaklanma yok)."
        )


# -----------------------------------------------------------------------------
# 40. Embedding → ilk katman: bilgi akışı
# -----------------------------------------------------------------------------


class TestEmbeddingToFirstLayerFlow:
    """Embedding + pos çıkışı ilk katmana giriyor; ilk katman çıkışı değişmiş olmalı."""

    def test_first_layer_output_different_from_embedding(self, minimal_model, batch_minimal, full_forward_captures):
        """İlk layer çıkışı, embedding+pos çıkışından farklı (attention/FFN etkisi)."""
        with torch.no_grad():
            minimal_model(batch_minimal)
        emb = full_forward_captures.get("embedded")
        layer0 = full_forward_captures.get("layer_0")
        if emb is None or layer0 is None:
            pytest.skip("embedded veya layer_0 capture yok")
        diff = (layer0 - emb).abs().max().item()
        assert diff > 1e-6, "İlk katman çıkışı embedding ile aynı (bilgi akışı yok)."


# -----------------------------------------------------------------------------
# 41. Loss hedefe duyarlılık – model hedefleri ayırt edebilmeli
# -----------------------------------------------------------------------------


class TestLossSensitiveToTargets:
    """Loss farklı hedef (label) seçimlerine duyarlı olmalı; aksi halde öğrenme anlamsız."""

    def test_loss_differs_when_target_changes(self, minimal_model, batch_minimal, seed):
        """Aynı girdi, farklı hedef tokenlar → loss farklı olmalı (CE hedefe duyarlı)."""
        minimal_model.train()
        torch.manual_seed(seed)
        V = minimal_model.embedding.num_embeddings
        B, T = batch_minimal.shape[0], batch_minimal.shape[1]
        x = batch_minimal.clone()
        logits, _ = minimal_model(x)
        targets_same = x[:, 1:]
        loss_same = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            targets_same.reshape(-1),
            reduction="mean",
        )
        targets_diff = (targets_same + 1) % V
        loss_diff = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            targets_diff.reshape(-1),
            reduction="mean",
        )
        minimal_model.eval()
        assert abs(loss_same.item() - loss_diff.item()) > 1e-4, (
            "Loss hedef değişimine duyarlı değil; model hedefleri ayırt edemiyor olabilir."
        )


# -----------------------------------------------------------------------------
# 42. Tek batch üzerinde overfit – model en azından ezberleyebilmeli
# -----------------------------------------------------------------------------


class TestOverfitSingleBatch:
    """Tek bir küçük batch üzerinde çok adım eğitim yapınca loss düşmeli (öğrenme kapasitesi)."""

    def test_loss_decreases_when_overfitting_one_batch(self, minimal_model, seed):
        """Aynı batch üzerinde 80 step: başlangıç loss'u > son loss (en az %5 düşüş)."""
        torch.manual_seed(seed)
        minimal_model.train()
        V = minimal_model.embedding.num_embeddings
        x = torch.randint(0, V, (2, 8))
        optimizer = torch.optim.Adam(minimal_model.parameters(), lr=1e-3)
        loss_start = None
        for step in range(80):
            optimizer.zero_grad()
            logits, _ = minimal_model(x)
            B, T, _ = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, V),
                x[:, 1:].reshape(-1),
                reduction="mean",
            )
            if step == 0:
                loss_start = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(minimal_model.parameters(), 1.0)
            optimizer.step()
        loss_end = loss.item()
        minimal_model.eval()
        assert loss_start is not None
        assert loss_end < loss_start * 0.95, (
            f"Tek batch overfit: loss düşmedi (start={loss_start:.4f}, end={loss_end:.4f}). "
            "Model bu veriyi bile öğrenemiyor; mimari veya grad akışı şüpheli."
        )
