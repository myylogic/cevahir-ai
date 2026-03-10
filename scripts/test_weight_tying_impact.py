# -*- coding: utf-8 -*-
"""
Weight Tying Etkisi Testi
================================================
Output layer embedding'i ile weight sharing yapıyor.
Bu durum EOS output logits'inde sorun yaratıyor mu?

Kontrol:
1. Embedding ve output weights gerçekten shared mi?
2. Output logits distribution normal mi?
3. Specific token (EOS) logits'i pattern var mı?
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from model.cevahir import Cevahir, CevahirConfig
from tokenizer_management.core.tokenizer_core import TokenizerCore

print("="*80)
print("WEIGHT TYING ETKISI TESTI")
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
model.eval()
print("[OK] Model yüklendi (tie_weights=True)")

# ============================================================================
# 2. WEIGHT SHARING KONTROL
# ============================================================================
print("\n[2] Weight sharing kontrol...")

# Find embedding and output weights
embed_weight = None
output_weight = None

for name, param in model.named_parameters():
    if "embedding" in name and "weight" in name:
        embed_weight = param
        print(f"  Embedding weight: {name}, shape={param.shape}")
    if "lm_head" in name or ("output" in name and "weight" in name and "layer" not in name.lower()):
        output_weight = param
        print(f"  Output weight: {name}, shape={param.shape}")

# Check if tied
if embed_weight is not None and output_weight is not None:
    # Check if they're the same object (shared)
    if embed_weight.data_ptr() == output_weight.data_ptr():
        print("  [OK] Embedding ve output weights SHARED (same memory)")
    else:
        print("  [WARNING] Embedding ve output weights SEPARATE (different memory)")
        print("           Weight tying çalışmıyor!")
elif embed_weight is not None:
    print("  [INFO] Embedding weight bulundu, output weight aranıyor...")
    # Embedding doğrudan output olarak kullanılıyor mu?
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is embed_weight:
            print(f"         Module {name} embedding weight'i kullanıyor")

# ============================================================================
# 3. DUMMY FORWARD PASS
# ============================================================================
print("\n[3] Forward pass + output analiz...")

x = torch.randint(1, 1000, (2, 16))
with torch.no_grad():
    logits, _ = model(x)

print(f"  Logits shape: {logits.shape}")
print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"  Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")

# ============================================================================
# 4. TOKEN-SPECİFİK LOGITS ANALİZİ
# ============================================================================
print("\n[4] Token-specific logits analiz...")

# Son position'daki logits
last_logits = logits[0, -1, :]  # (vocab_size,)

# Softmax prob
probs = F.softmax(last_logits, dim=-1)

# Special tokens
special_tokens = {
    "BOS": BOS_ID,
    "EOS": EOS_ID,
    "PAD": PAD_ID,
}

print("\n  [SPECIAL TOKENS]")
for token_name, token_id in special_tokens.items():
    logit = last_logits[token_id].item()
    prob = probs[token_id].item()
    print(f"    {token_name} (ID={token_id}): logit={logit:.4f}, prob={prob:.6f}")

# Top tokens
top_probs, top_ids = torch.topk(probs, k=10)
print("\n  [TOP 10 TOKENS]")
for i, (prob, token_id) in enumerate(zip(top_probs, top_ids)):
    print(f"    {i+1}. ID={token_id.item()}: prob={prob.item():.6f}")

# ============================================================================
# 5. EMBEDDING WEIGHT BIAS ANALYZİ
# ============================================================================
print("\n[5] Embedding weight bias analiz...")

if embed_weight is not None:
    # Embedding vec norm distribution
    embed_norms = torch.norm(embed_weight, dim=1)
    
    print(f"  Embedding vector norms:")
    print(f"    Min: {embed_norms.min():.6f}")
    print(f"    Max: {embed_norms.max():.6f}")
    print(f"    Mean: {embed_norms.mean():.6f}")
    print(f"    Std: {embed_norms.std():.6f}")
    
    # Special tokens norms
    print(f"\n  Special token embedding norms:")
    for token_name, token_id in special_tokens.items():
        norm = embed_norms[token_id].item()
        print(f"    {token_name}: {norm:.6f}")

# ============================================================================
# 6. CORRELATION ANALİZİ
# ============================================================================
print("\n[6] Correlation analiz...")

# Logits-to-embedding-weights correlation
# Each logit[i] = hidden @ W[i] (where W[i] is embedding vector for token i)
# Weak embedding weights → weak output logits for that token?

if embed_weight is not None:
    last_hidden = logits.grad is not None  # Just a dummy check
    print(f"  Embedding weights norm distribution bilgisi yukarıda verildi")
    
    # Special tokens için embedding magnitude vs output logit
    print(f"\n  Special tokens: embedding magnitude vs output logit correlation")
    for token_name, token_id in special_tokens.items():
        embed_norm = embed_norms[token_id].item()
        logit = last_logits[token_id].item()
        print(f"    {token_name}: embed_norm={embed_norm:.6f}, logit={logit:.4f}")

# ============================================================================
# 7. TEŞHIS
# ============================================================================
print("\n" + "="*80)
print("TEŞHİS")
print("="*80)

issues = []

# Check weight tying
if embed_weight is not None and output_weight is not None:
    if embed_weight.data_ptr() != output_weight.data_ptr():
        issues.append("[SORUN] Weight tying çalışmıyor - outputs ve embedding separate")

# Check EOS logits
if probs[EOS_ID].item() < 0.001:
    issues.append(f"[SORUN] EOS prob çok düşük: {probs[EOS_ID].item():.6f}")

# Check embedding initialization
if embed_weight is not None:
    if embed_norms.std() < 0.001:
        issues.append("[SORUN] Embedding norms çok uniform - initialization problem?")
    if embed_norms[EOS_ID].item() < embed_norms.mean().item() * 0.5:
        issues.append(f"[SORUN] EOS embedding çok küçük: {embed_norms[EOS_ID].item():.6f} vs avg {embed_norms.mean().item():.6f}")

if not issues:
    print("[OK] Weight tying normal görünüyor")
    print("     Sorun başka yerde (training dynamics, batch effects, etc)")
else:
    print("SORUNLAR:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "="*80)
