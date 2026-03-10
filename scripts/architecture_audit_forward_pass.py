# -*- coding: utf-8 -*-
"""
Mimari inceleme: Tek bir forward geçişinde her aşamanın istatistikleri ve
son pozisyon logit dağılımı (entropy, max prob, top-5). Mode collapse'ın
hangi katmanda başladığını görmek için kullanılır.

Kullanım (proje kökünden):
  python scripts/architecture_audit_forward_pass.py
  python scripts/architecture_audit_forward_pass.py --checkpoint saved_models/checkpoints/checkpoint_epoch_0150.pth
"""
import os
import sys
import argparse
import math
import logging

# Proje kökünü path'e ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    """Tensor min/max/mean/std; NaN/Inf sayısı."""
    t = t.detach().float()
    out = {
        "name": name,
        "shape": tuple(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if t.numel() > 1 else 0.0,
        "nan": int(torch.isnan(t).sum().item()),
        "inf": int(torch.isinf(t).sum().item()),
    }
    return out


def logit_distribution(logits_last: torch.Tensor, vocab_size: int):
    """Son pozisyon logit'leri için softmax, entropy, max prob, top-5."""
    # logits_last: [vocab_size]
    logits_last = logits_last.detach().float()
    if logits_last.dim() > 1:
        logits_last = logits_last.view(-1)
    logits_last = logits_last[:vocab_size]
    # Sayısal kararlılık: max çıkar
    lmax = logits_last.max().item()
    probs = torch.softmax(logits_last - lmax, dim=-1)
    probs = probs.cpu()
    # Entropy: -sum(p*log(p)); p=0 için 0
    eps = 1e-12
    log_p = torch.log(probs + eps)
    entropy = - (probs * log_p).sum().item()
    max_prob = float(probs.max().item())
    top5_probs, top5_ids = probs.topk(min(5, probs.size(0)))
    return {
        "logit_min": float(logits_last.min().item()),
        "logit_max": float(logits_last.max().item()),
        "logit_mean": float(logits_last.mean().item()),
        "entropy": entropy,
        "max_prob": max_prob,
        "top5_token_ids": top5_ids.tolist(),
        "top5_probs": top5_probs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Forward pass mimari teşhis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/cevahir_model.pth",
        help="Model checkpoint veya state_dict dosyası",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Eğitim config ile uyumlu model parametreleri (train.py)
    vocab_size = 60000
    embed_dim = 256
    seq_proj_dim = 256
    num_heads = 4
    num_layers = 6
    dropout = 0.15

    # Model sınıfını yükle
    from src.neural_network import CevahirNeuralNetwork

    model = CevahirNeuralNetwork(
        learning_rate=1e-4,
        dropout=dropout,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        attention_type="multi_head",
        normalization_type="layer_norm",
        device=args.device,
        log_level=logging.WARNING,
        num_layers=num_layers,
        ffn_dim=None,
        pre_norm=True,
        causal_mask=True,
        use_flash_attention=False,
        pe_mode="rope",
        use_gradient_checkpointing=False,
        tie_weights=True,
        use_rmsnorm=True,
        use_swiglu=True,
        use_kv_cache=False,
        max_cache_len=2048,
        use_advanced_checkpointing=False,
        checkpointing_strategy="selective",
        quantization_type="none",
        use_moe=False,
        num_experts=8,
        moe_top_k=2,
    )
    model.eval()

    # Checkpoint yükle
    path = os.path.join(ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if not os.path.isfile(path):
        print(f"[WARN] Dosya yok: {path}. Rastgele ağırlıklarla devam ediyorum.")
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt
        load_ok = model.load_state_dict(state, strict=False)
        if load_ok.missing_keys:
            print("[WARN] Eksik anahtarlar:", load_ok.missing_keys[:5], "...")
        if load_ok.unexpected_keys:
            print("[WARN] Beklenmeyen anahtarlar:", load_ok.unexpected_keys[:5], "...")
        print(f"[OK] State yüklendi: {path}")
    model = model.to(device)

    # Epoch 150 log'daki prompt token_ids (ilk 15): En sevdiğin hayvan nedir?
    prompt_ids = [2, 2934, 390, 26657, 390, 18026, 390, 23655, 390, 184]
    # Seq_len'e tamamla (pad 0 ile)
    if len(prompt_ids) < args.seq_len:
        prompt_ids = prompt_ids + [0] * (args.seq_len - len(prompt_ids))
    else:
        prompt_ids = prompt_ids[:args.seq_len]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # [1, seq_len]

    # Hook ile ara çıktıları topla
    captures = {}

    def make_hook(key):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            captures[key] = out.detach()
        return hook

    model.embedding.register_forward_hook(make_hook("embedding_out"))
    model.output_norm.register_forward_hook(make_hook("output_norm_out"))
    # output_layer çıkışı (logits) forward sonunda alınacak

    with torch.no_grad():
        out, _ = model(x, mask=None, causal_mask=True)
    logits = out  # [1, T, vocab_size]

    # İstatistikler
    print("\n" + "=" * 60)
    print("FORWARD PASS TEŞHİS (Mimari İnceleme)")
    print("=" * 60)
    print(f"Giriş shape: {x.shape}")

    for key in ["embedding_out", "output_norm_out"]:
        if key in captures:
            s = tensor_stats(key, captures[key])
            print(f"  {s['name']}: shape={s['shape']} min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f} nan={s['nan']} inf={s['inf']}")

    s = tensor_stats("logits", logits)
    print(f"  {s['name']}: shape={s['shape']} min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f} nan={s['nan']} inf={s['inf']}")

    # Son pozisyon logit dağılımı
    last_logits = logits[0, -1, :]
    dist = logit_distribution(last_logits, vocab_size)
    print("\n--- Son pozisyon (last token) logit dağılımı ---")
    print(f"  logit min/max/mean: {dist['logit_min']:.4f} / {dist['logit_max']:.4f} / {dist['logit_mean']:.4f}")
    print(f"  entropy: {dist['entropy']:.4f} (0'a yakınsa çökme)")
    print(f"  max_prob: {dist['max_prob']:.6f} (1'e yakınsa çökme)")
    print(f"  top-5 token_id: {dist['top5_token_ids']}")
    print(f"  top-5 prob:     {[f'{p:.4f}' for p in dist['top5_probs']]}")
    if dist["entropy"] < 1.0 or dist["max_prob"] > 0.95:
        print("  [UYARI] Dağılım çok sivri (mode collapse belirtisi).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
