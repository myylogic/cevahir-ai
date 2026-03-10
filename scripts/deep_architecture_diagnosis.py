# -*- coding: utf-8 -*-
"""
Model Mimarisi Derinlemesine Tanı Testi
================================================
Sinir ağının her katmanında:
1. Forward pass çıktıları (activation ranges)
2. Gradient flow (per-layer gradient magnitudes)
3. Dead neurons (zero activations)
4. NaN/Inf kontrolü
5. Weight distribution

Amaç: Mimaride temel sorun bulmak (EOS öğrenememesinin sebebi)
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from model.cevahir import Cevahir, CevahirConfig
from tokenizer_management.core.tokenizer_core import TokenizerCore

print("="*80)
print("MODEL MİMARİSİ DERINLEMESINE TANI")
print("="*80)

# ============================================================================
# 1. MODEL YÜKLEME
# ============================================================================
print("\n[1] Model yükleniyor...")

tokenizer = TokenizerCore({
    "vocab_path": os.path.join(project_root, "data/vocab_lib/vocab.json"),
    "merges_path": os.path.join(project_root, "data/merges_lib/merges.txt"),
})
tokenizer.finalize_vocab()
vocab = tokenizer.get_vocab()

def get_id(name):
    v = vocab.get(name)
    return v.get("id") if isinstance(v, dict) else v

vocab_size = len(vocab)

cevahir = Cevahir(CevahirConfig(
    device="cpu",
    tokenizer={
        "vocab_path": os.path.join(project_root, "data/vocab_lib/vocab.json"),
        "merges_path": os.path.join(project_root, "data/merges_lib/merges.txt"),
    },
    load_model_path=os.path.join(project_root, "saved_models/cevahir_model.pth"),
    model={
        "vocab_size": vocab_size,
        "embed_dim": 256, "seq_proj_dim": 256, "num_heads": 4, "num_layers": 6,
        "ffn_dim": None, "pre_norm": True, "causal_mask": True,
        "tie_weights": True, "use_rmsnorm": True, "use_swiglu": True,
        "use_kv_cache": True,
    }
))

model = cevahir._model_manager.model
model.eval()
print(f"[OK] Model yüklendi: {type(model).__name__}")

# ============================================================================
# 2. DUMMY INPUT
# ============================================================================
print("\n[2] Dummy input oluşturuluyor...")

batch_size, seq_len = 2, 16
x = torch.randint(1, 1000, (batch_size, seq_len))

print(f"  Input shape: {x.shape}, range: [{x.min()}, {x.max()}]")

# ============================================================================
# 3. FORWARD PASS + ACTIVATION MONITORING
# ============================================================================
print("\n[3] Forward pass (activation monitoring)...")

activations_log = {}

def hook_activations(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations_log[name] = {
                "shape": output.shape,
                "dtype": output.dtype,
                "min": output.min().item(),
                "max": output.max().item(),
                "mean": output.mean().item(),
                "std": output.std().item(),
                "nancount": torch.isnan(output).sum().item(),
                "infcount": torch.isinf(output).sum().item(),
                "zeros": (output.abs() < 1e-7).sum().item(),
                "total": output.numel(),
            }
    return hook

# Hook ekle (önemli katmanlar)
layer_names = []
for name, module in model.named_modules():
    if any(x in name.lower() for x in ["embedding", "norm", "attention", "ffn", "layer"]):
        module.register_forward_hook(hook_activations(name))
        layer_names.append(name)

with torch.no_grad():
    logits, _ = model(x)

print(f"  Output logits shape: {logits.shape}")
print(f"  Output logits range: [{logits.min():.4f}, {logits.max():.4f}]")

# ============================================================================
# 4. ACTIVATION ANALIZ
# ============================================================================
print("\n[4] Activation analiz:")

print("\n  [EMBEDDING KATMANI]")
if "embedding" in str(activations_log.keys()).lower():
    for name, act in activations_log.items():
        if "embedding" in name.lower():
            print(f"    {name}:")
            print(f"      Shape: {act['shape']}, Mean: {act['mean']:.6f}, Std: {act['std']:.6f}")
            if act['nancount'] > 0:
                print(f"      [ERROR] NaN: {act['nancount']} / {act['total']}")
            if act['zeros'] > act['total'] * 0.1:
                print(f"      [WARNING] Dead activations: {act['zeros']}/{act['total']} ({100*act['zeros']/act['total']:.1f}%)")

print("\n  [NORMALIZATION KATMANI (RMSNorm)]")
for name, act in activations_log.items():
    if "rmsnorm" in name.lower() or "norm" in name.lower():
        print(f"    {name}:")
        print(f"      Shape: {act['shape']}, Mean: {act['mean']:.6f}, Std: {act['std']:.6f}")
        if act['std'] < 0.1:
            print(f"      [WARNING] Std çok düşük (<0.1): {act['std']:.6f}")

print("\n  [ATTENTION KATMANI]")
for name, act in activations_log.items():
    if "attention" in name.lower() and "output" in name.lower():
        print(f"    {name}:")
        print(f"      Shape: {act['shape']}, Mean: {act['mean']:.6f}, Std: {act['std']:.6f}")

print("\n  [FFN KATMANI]")
for name, act in activations_log.items():
    if "ffn" in name.lower() or "feedforward" in name.lower():
        print(f"    {name}:")
        print(f"      Shape: {act['shape']}, Mean: {act['mean']:.6f}, Std: {act['std']:.6f}")
        if act['zeros'] > act['total'] * 0.2:
            print(f"      [WARNING] Dead neurons: {100*act['zeros']/act['total']:.1f}%")

# ============================================================================
# 5. WEIGHT ANALIZ
# ============================================================================
print("\n[5] Weight distribution analiz:")

weight_stats = {}

for name, param in model.named_parameters():
    if param.requires_grad and len(param.shape) >= 2:
        weight_stats[name] = {
            "shape": param.shape,
            "min": param.min().item(),
            "max": param.max().item(),
            "mean": param.mean().item(),
            "std": param.std().item(),
            "nancount": torch.isnan(param).sum().item(),
        }

print("\n  [EMBEDDING WEIGHTS]")
for name, stats in weight_stats.items():
    if "embedding" in name.lower():
        print(f"    {name}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")

print("\n  [OUTPUT WEIGHTS]")
for name, stats in weight_stats.items():
    if "output" in name.lower() or "lm_head" in name.lower():
        print(f"    {name}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")

print("\n  [LAYER WEIGHTS (sample)]")
count = 0
for name, stats in weight_stats.items():
    if "layer" in name.lower() and count < 5:
        print(f"    {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        count += 1

# ============================================================================
# 6. GRADIENT FLOW TEST
# ============================================================================
print("\n[6] Gradient flow test...")

x_train = torch.randint(1, 1000, (2, 16))
y_train = torch.randint(0, vocab_size, (2, 16))

model.train()
logits, _ = model(x_train)

# Simple loss
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
loss = loss_fn(logits.reshape(-1, vocab_size), y_train.reshape(-1))

loss.backward()

gradient_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        gradient_norms[name] = grad_norm

print(f"  Total loss: {loss.item():.6f}")
print(f"  Layers with gradients: {len(gradient_norms)}")

print("\n  [GRADIENT MAGNITUDES]")
for name, grad_norm in sorted(gradient_norms.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {name}: {grad_norm:.6f}")

# ============================================================================
# 7. TEŞHIS
# ============================================================================
print("\n" + "="*80)
print("TEŞHİS SONUÇLARI")
print("="*80)

issues = []

# Activation range issues
for name, act in activations_log.items():
    if act['nancount'] > 0:
        issues.append(f"[KRITIK] {name} NaN içeriyor: {act['nancount']}/{act['total']}")
    if act['infcount'] > 0:
        issues.append(f"[KRITIK] {name} Inf içeriyor: {act['infcount']}/{act['total']}")
    if act['zeros'] > act['total'] * 0.3:
        issues.append(f"[HATA] {name} %30+ dead activations")
    if act['std'] < 0.01:
        issues.append(f"[HATA] {name} Std çok düşük (<0.01): {act['std']:.6f}")

# Weight initialization issues
for name, stats in weight_stats.items():
    if stats['nancount'] > 0:
        issues.append(f"[KRITIK] {name} weights NaN")
    if stats['std'] == 0:
        issues.append(f"[HATA] {name} tüm weights aynı")

# Gradient issues
if len(gradient_norms) == 0:
    issues.append(f"[KRITIK] Hiçbir layer gradient almıyor!")

zero_grad_count = sum(1 for g in gradient_norms.values() if g < 1e-10)
if zero_grad_count > 0:
    issues.append(f"[HATA] {zero_grad_count} layer'in gradient 0 (vanishing gradient)")

if not issues:
    print("[OK] Mimarı analiz sorun görmedi")
    print("     EOS sorunun kaynağı başka yerde (hyperparameter, data, learning dynamics)")
else:
    print("SORUNLAR BULUNDU:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "="*80)
