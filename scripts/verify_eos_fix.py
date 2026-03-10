# -*- coding: utf-8 -*-
"""
EOS Fix Dogrulama Testi
================================================
Yeni hyperparameter'lar ile dummy batch test:
- eos_weight=10.0 (eskiden 1.0)
- label_smoothing=0.0 (eskiden 0.1)

EOS öğreniminin iyileştiğini kontrol eder.
"""
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from model.cevahir import Cevahir, CevahirConfig
from training_system.v2.core.criterion_manager import CriterionManager
from tokenizer_management.core.tokenizer_core import TokenizerCore

print("="*80)
print("EOS FIX DOGRULAMA TESTI")
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

BOS_ID = get_id("<BOS>")
EOS_ID = get_id("<EOS>")
PAD_ID = get_id("<PAD>")
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
model.train()
print("[OK] Model yüklendi")

# ============================================================================
# 2. CRITERION KONFIGÜRASYONLARI
# ============================================================================
print("\n[2] Criterion'lar oluşturuluyor...")

cm = CriterionManager()

# OLD: label_smoothing=0.1, eos_weight=1.0
criterion_old = cm.create_criterion(
    vocab_size=vocab_size, eos_id=EOS_ID, pad_id=PAD_ID,
    label_smoothing=0.1, eos_weight=1.0
)

# NEW: label_smoothing=0.0, eos_weight=10.0
criterion_new = cm.create_criterion(
    vocab_size=vocab_size, eos_id=EOS_ID, pad_id=PAD_ID,
    label_smoothing=0.0, eos_weight=10.0
)

print("[OK] OLD criterion (smoothing=0.1, weight=1.0)")
print("[OK] NEW criterion (smoothing=0.0, weight=10.0)")

# ============================================================================
# 3. DUMMY BATCH
# ============================================================================
print("\n[3] Dummy batch oluşturuluyor...")

batch_size, seq_len = 4, 20
x = torch.randint(1, 100, (batch_size, seq_len))
x[:, 0] = BOS_ID

# Target: shifted input + EOS at end
y = torch.roll(x, shifts=-1, dims=1)
y[:, -1] = EOS_ID

print(f"  Input shape: {x.shape}, Target shape: {y.shape}")
print(f"  Example: target[-1] = {y[0, -1].item()} (EOS_ID={EOS_ID})")

# ============================================================================
# 4. FORWARD + LOSS KARŞILAŞTIRMASI
# ============================================================================
print("\n[4] Loss hesaplaması karşılaştırıldığı...")

with torch.enable_grad():
    logits, _ = model(x)
    logits_flat = logits.reshape(-1, vocab_size)
    y_flat = y.reshape(-1)
    
    loss_old = criterion_old(logits_flat, y_flat)
    loss_new = criterion_new(logits_flat, y_flat)
    
    print(f"  OLD loss (smoothing=0.1, weight=1.0): {loss_old.item():.6f}")
    print(f"  NEW loss (smoothing=0.0, weight=10.0): {loss_new.item():.6f}")
    print(f"  Fark: {(loss_new - loss_old).item():.6f}")

# ============================================================================
# 5. EOS-SPECİFİK LOSS
# ============================================================================
print("\n[5] EOS-specific loss analiz...")

eos_mask = (y_flat == EOS_ID)
eos_count = eos_mask.sum().item()

if eos_count > 0:
    eos_logits = logits_flat[eos_mask]
    eos_targets = y_flat[eos_mask]
    
    with torch.no_grad():
        loss_eos_old = criterion_old(eos_logits, eos_targets)
        loss_eos_new = criterion_new(eos_logits, eos_targets)
        
        probs = F.softmax(eos_logits, dim=-1)
        eos_prob_mean = probs[:, EOS_ID].mean().item()
    
    print(f"  EOS token sayısı: {eos_count}")
    print(f"  EOS pozisyonlarında EOS prob (mean): {eos_prob_mean:.6f}")
    print(f"  EOS loss (OLD): {loss_eos_old.item():.6f}")
    print(f"  EOS loss (NEW): {loss_eos_new.item():.6f}")
    print(f"  Fark: {(loss_eos_new - loss_eos_old).item():.6f}")

# ============================================================================
# 6. GRADIENT ANALİZİ
# ============================================================================
print("\n[6] Gradient analiz (EOS loss için)...")

model.zero_grad()
with torch.enable_grad():
    logits, _ = model(x)
    logits_flat = logits.reshape(-1, vocab_size)
    loss_new = criterion_new(logits_flat, y_flat)
    loss_new.backward()

if logits.grad is not None:
    eos_logits_grad = logits.grad.reshape(-1, vocab_size)[eos_mask]
    print(f"  EOS pozisyonlarında logits gradient:")
    print(f"    Mean abs: {eos_logits_grad.abs().mean().item():.6f}")
    print(f"    Max abs: {eos_logits_grad.abs().max().item():.6f}")
    print(f"  [OK] Gradient flow çalışıyor")

# ============================================================================
# 7. SONUÇ
# ============================================================================
print("\n" + "="*80)
print("SONUÇ")
print("="*80)

print("[INFO] YENİ HYPERPARAMETER'LAR:")
print("  - label_smoothing: 0.1 -> 0.0")
print("  - eos_weight: 1.0 -> 10.0")
print("\n[INFO] BEKLENEN ETKILER:")
print("  1. EOS loss artar (0.0 smoothing + 10.0 weight yuzunden)")
print("  2. EOS gradient'leri 10x guclenir")
print("  3. Training sirasinda model EOS'u daha hizli ogrenir")
print("  4. EOS probability artmali (inference'da)")
print("\n[INFO] SONRAKI ADIM:")
print("  Egitimi baslatmak icin: python training_system/train.py")

print("="*80)
