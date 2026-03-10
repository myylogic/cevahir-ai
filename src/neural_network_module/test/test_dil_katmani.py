# -*- coding: utf-8 -*-
import logging
import pytest
import torch

from neural_network_module.dil_katmani import DilKatmani

# ---- Ortak sabitler ----
VOCAB = 5000
EMBED_DIM = 128
PROJ_DIM = 64
DROPOUT = 0.1


@pytest.fixture
def dil_katmani():
    return DilKatmani(
        vocab_size=VOCAB,
        embed_dim=EMBED_DIM,
        seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier",
        seq_init_method="xavier",
        log_level=logging.DEBUG,
        dropout=DROPOUT,
    )


def test_forward_output_shape(dil_katmani):
    batch, seq = 32, 50
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    y = dil_katmani(x)
    assert y.shape == (batch, seq, PROJ_DIM)


def test_positional_encoding_effect(dil_katmani):
    batch, seq = 4, 10
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    emb = dil_katmani.language_embedding(x)
    enc = dil_katmani.positional_encoding(emb)
    assert not torch.equal(emb, enc), "Positional encoding embeddings'i değiştirmedi."


def test_layer_norm_statistics(dil_katmani):
    batch, seq = 4, 10
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    emb = dil_katmani.language_embedding(x)
    enc = dil_katmani.positional_encoding(emb)
    normed = dil_katmani.layer_norm(enc)
    mean_val = normed.mean().item()
    std_val = normed.std().item()
    assert abs(mean_val) < 1e-3, f"LayerNorm mean ~0 değil: {mean_val}"
    assert abs(std_val - 1) < 1e-2, f"LayerNorm std ~1 değil: {std_val}"


def test_dropout_randomness_training(dil_katmani):
    dil_katmani.train()
    batch, seq = 4, 10
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    y1 = dil_katmani(x)
    y2 = dil_katmani(x)
    assert not torch.allclose(y1, y2, atol=1e-5), "Train modda dropout rastgelelik katmıyor."


def test_eval_mode_determinism(dil_katmani):
    dil_katmani.eval()
    batch, seq = 4, 10
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    y1 = dil_katmani(x)
    y2 = dil_katmani(x)
    assert torch.allclose(y1, y2, atol=1e-6), "Eval modda çıktı deterministik değil."


def test_seq_projection_output_range(dil_katmani):
    dil_katmani.eval()
    batch, seq = 4, 15
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    y = dil_katmani(x)
    assert y.min() > -10 and y.max() < 10, "Projeksiyon çıktısı beklenen aralıkta değil (-10, 10)."


def test_invalid_input_type_raises():
    m = DilKatmani(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier", seq_init_method="xavier",
        log_level=logging.DEBUG, dropout=DROPOUT
    )
    with pytest.raises(Exception):
        m(torch.rand(32, 50))  # float -> Embedding long ister, hata vermeli


def test_invalid_input_dimension_1d_raises():
    m = DilKatmani(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier", seq_init_method="xavier",
        log_level=logging.DEBUG, dropout=DROPOUT
    )
    # 1D indeks dizisi -> PE ve LN 3D beklediğinden akış kırılır (hata)
    with pytest.raises(Exception):
        m(torch.randint(0, VOCAB, (50,), dtype=torch.long))


def test_logging_output_dilkatmani(caplog):
    caplog.set_level(logging.DEBUG)
    m = DilKatmani(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier", seq_init_method="xavier",
        log_level=logging.DEBUG, dropout=DROPOUT
    )
    batch, seq = 4, 10
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    _ = m(x)
    # Aşamalar loglarda geçmeli (DEBUG)
    txt = caplog.text
    assert ("After LanguageEmbedding" in txt) or ("LanguageEmbedding" in txt)
    assert ("After PositionalEncoding" in txt) or ("PositionalEncoding" in txt)
    assert ("After LayerNorm" in txt) or ("LayerNorm" in txt)
    assert ("After Dropout" in txt) or ("Dropout" in txt)
    assert ("After SeqProjection" in txt) or ("SeqProjection" in txt)


def test_determinism_fixed_input_no_dropout():
    m = DilKatmani(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier", seq_init_method="xavier",
        log_level=logging.DEBUG, dropout=0.0  # Dropout kapalı
    )
    m.eval()
    torch.manual_seed(42)
    batch, seq = 4, 10
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    y1 = m(x)
    y2 = m(x)
    assert torch.allclose(y1, y2, atol=1e-6), "Dropout kapalıyken determinism korunmadı."


def test_learned_positional_encoding_variant():
    # pe_mode="learned" ile de akış çalışmalı
    m = DilKatmani(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier", seq_init_method="xavier",
        log_level=logging.INFO, dropout=DROPOUT,
        pe_mode="learned", pe_max_len=256, pe_dropout=0.05
    )
    batch, seq = 2, 16
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    y = m(x)
    assert y.shape == (batch, seq, PROJ_DIM)


def test_backward_gradient_flow(dil_katmani):
    # Basit bir loss ile backward akışı kopmamalı
    batch, seq = 2, 8
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    out = dil_katmani(x)                # [B, T, PROJ]
    loss = out.pow(2).mean()
    loss.backward()                     # Parametrelerde grad oluşmalı
    has_grad = any(p.grad is not None for p in dil_katmani.parameters())
    assert has_grad, "Backward sonrası gradyan oluşmadı."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
def test_device_cuda_compat():
    m = DilKatmani(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, seq_proj_dim=PROJ_DIM,
        embed_init_method="xavier", seq_init_method="xavier",
        log_level=logging.INFO, dropout=DROPOUT
    ).cuda()
    batch, seq = 2, 12
    x = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long, device="cuda")
    y = m(x)
    assert y.is_cuda and y.shape == (batch, seq, PROJ_DIM)
