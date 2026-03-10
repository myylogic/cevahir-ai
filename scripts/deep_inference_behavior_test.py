# -*- coding: utf-8 -*-
"""
Derinlemesine Inference Davranış Testi
===============================================
Model'in gerçek davranışını analiz ediyor:
- EOS olasılığı
- Probability dağılımı entropy
- Token tekrarı / çeşitlilik
- Softmax çıktıları

Amaç: Label smoothing ve model collapse sorununu teşhis etmek
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from model.cevahir import Cevahir, CevahirConfig
from tokenizer_management.core.tokenizer_core import TokenizerCore

print("="*80)
print("DERINLEMESINE INFERENCE DAVRANIŞI TESTI")
print("="*80)

# ============================================================================
# 1. MODEL YÜKLEME
# ============================================================================
print("\n[1] Model yükleniyor...")

try:
    tokenizer = TokenizerCore({
        "vocab_path": os.path.join(project_root, "data/vocab_lib/vocab.json"),
        "merges_path": os.path.join(project_root, "data/merges_lib/merges.txt"),
    })
    tokenizer.finalize_vocab()
    vocab = tokenizer.get_vocab()
    
    def get_id(name):
        v = vocab.get(name)
        if isinstance(v, dict):
            return v.get("id")
        return v
    
    BOS_ID = get_id("<BOS>")
    EOS_ID = get_id("<EOS>")
    PAD_ID = get_id("<PAD>")
    vocab_size = len(vocab)
    
    cevahir_config = CevahirConfig(
        device="cpu",
        tokenizer={
            "vocab_path": os.path.join(project_root, "data/vocab_lib/vocab.json"),
            "merges_path": os.path.join(project_root, "data/merges_lib/merges.txt"),
        },
        load_model_path=os.path.join(project_root, "saved_models/cevahir_model.pth"),
        model={
            "vocab_size": vocab_size,
            "embed_dim": 256,
            "seq_proj_dim": 256,
            "num_heads": 4,
            "num_layers": 6,
            "ffn_dim": None,
            "pre_norm": True,
            "causal_mask": True,
            "tie_weights": True,
            "use_rmsnorm": True,
            "use_swiglu": True,
            "use_kv_cache": True,
        }
    )
    
    cevahir = Cevahir(cevahir_config)
    model_manager = cevahir._model_manager
    model = model_manager.model
    model.eval()
    
    print("[OK] Model yüklendi")
except Exception as e:
    print(f"[ERROR] Model yükleme hatası: {e}")
    sys.exit(1)

# ============================================================================
# 2. TEST PROMPTLARI
# ============================================================================
print("\n[2] Test promptları encode ediliyor...")

test_prompts = [
    "Merhaba",
    "Nasılsın",
    "En sevdiğin hayvan nedir",
    "Aşk nedir",
    "Mutluluk nedir"
]

test_inputs = []
for prompt in test_prompts:
    try:
        _, token_ids = tokenizer.encode(prompt, mode="inference", add_special_tokens=False)
        token_ids = list(token_ids) if not isinstance(token_ids, list) else token_ids
        # BOS ekle
        input_ids = [BOS_ID] + token_ids
        test_inputs.append((prompt, input_ids))
    except Exception as e:
        print(f"  [ERROR] '{prompt}' encode hatası: {e}")

print(f"[OK] {len(test_inputs)} prompt encode edildi")

# ============================================================================
# 3. LOGITS ANALIZ
# ============================================================================
print("\n[3] Logits analiz ediliyor...")

analysis_results = []

with torch.no_grad():
    for prompt, input_ids in test_inputs:
        print(f"\n--- Prompt: '{prompt}' ---")
        print(f"    Input IDs length: {len(input_ids)}")
        
        x = torch.tensor([input_ids], dtype=torch.long)
        
        # Forward pass
        logits, _ = model_manager.forward(x, inference=True, return_aux=False)
        
        # Son pozisyon logits
        last_logits = logits[0, -1, :].float()
        
        # Softmax
        probs = F.softmax(last_logits, dim=-1)
        
        # EOS olasılığı
        eos_prob = probs[EOS_ID].item()
        
        # Argmax tahmin
        pred_id = last_logits.argmax().item()
        
        # Top-5 tokens
        top5_probs, top5_ids = torch.topk(probs, k=5)
        
        # Entropy (bilgi içeriği)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # Probability dağılımı istatistikleri
        prob_max = probs.max().item()
        prob_mean = probs.mean().item()
        prob_std = probs.std().item()
        
        # Tekrarlayan token sayısı (top-10 en yüksek prob token)
        top10_probs, top10_ids = torch.topk(probs, k=min(10, vocab_size))
        cumsum_top10 = top10_probs.sum().item()
        
        print(f"    EOS olasılığı: {eos_prob:.6f}")
        print(f"    Argmax tahmin ID: {pred_id} (EOS mi? {pred_id == EOS_ID})")
        print(f"    Entropy: {entropy:.4f} (düşük=confident, yüksek=uncertain)")
        print(f"    Prob istatistikleri - Max: {prob_max:.6f}, Mean: {prob_mean:.6f}, Std: {prob_std:.6f}")
        print(f"    Top-10 token'lar toplam olasılık: {cumsum_top10:.4f} (1.0'e yakın = çeşitlilik yok)")
        print(f"    Top-5 tokens:")
        for i, (p, tid) in enumerate(zip(top5_probs, top5_ids)):
            print(f"      {i+1}. ID={tid.item()}: {p.item():.6f}")
        
        analysis_results.append({
            "prompt": prompt,
            "eos_prob": eos_prob,
            "pred_id_is_eos": pred_id == EOS_ID,
            "entropy": entropy,
            "prob_max": prob_max,
            "prob_mean": prob_mean,
            "top10_cumsum": cumsum_top10,
        })

# ============================================================================
# 4. ÖZET VE BULGULAR
# ============================================================================
print("\n" + "="*80)
print("ÖZET VE BULGULAR")
print("="*80)

avg_eos_prob = np.mean([r["eos_prob"] for r in analysis_results])
eos_correct = sum(1 for r in analysis_results if r["pred_id_is_eos"])
avg_entropy = np.mean([r["entropy"] for r in analysis_results])
avg_prob_max = np.mean([r["prob_max"] for r in analysis_results])
avg_top10_cumsum = np.mean([r["top10_cumsum"] for r in analysis_results])

print(f"\nOrtalama EOS olasılığı: {avg_eos_prob:.6f}")
print(f"  -> EOS tahmini yapan prompt: {eos_correct}/{len(analysis_results)}")
if avg_eos_prob < 0.01:
    print(f"  [SORUN] EOS prob çok düşük! Model EOS'u kullanmıyor.")

print(f"\nOrtalama Entropy: {avg_entropy:.4f}")
if avg_entropy < 2.0:
    print(f"  [SORUN] Entropy çok düşük (< 2.0)! Prob dağılımı çok dar (few tokens baskın).")
elif avg_entropy > 8.0:
    print(f"  [SORUN] Entropy çok yüksek (> 8.0)! Prob dağılımı uniform (model çaşkınlık içinde).")
else:
    print(f"  [OK] Entropy makul aralıkta (2-8).")

print(f"\nOrtalama max prob: {avg_prob_max:.6f}")
if avg_prob_max < 0.01:
    print(f"  [SORUN] En yüksek prob < 0.01! Softmax uniform dağılmış (label smoothing çok agresif?).")
else:
    print(f"  [OK] En yüksek prob makul.")

print(f"\nTop-10 token cumsum: {avg_top10_cumsum:.4f}")
if avg_top10_cumsum > 0.95:
    print(f"  [SORUN] Top-10 cumsum > 0.95! Çeşitlilik çok az, collapse var.")
elif avg_top10_cumsum < 0.5:
    print(f"  [OK] Dağılım geniş (sağlıklı entropy).")
else:
    print(f"  [NORMAL] Top-10 normalde %{int(avg_top10_cumsum*100)} olasılık alıyor.")

print("\n" + "="*80)
print("TEŞHIS")
print("="*80)

issues = []

if avg_eos_prob < 0.001:
    issues.append("[KRITIK] EOS prob ~0: Model EOS'u pratikte kullanmıyor")

if avg_entropy < 1.5:
    issues.append("[KRITIK] Entropy çok düşük: Probability collapse var")

if avg_prob_max < 0.01:
    issues.append("[KRITIK] Max prob < 0.01: Label smoothing çok agresif ya da logits degeneratif")

if avg_top10_cumsum > 0.98:
    issues.append("[HATA] Mode collapse: Top-10 token %98+ prob alıyor")

if not issues:
    print("[OK] Davranış testlerinde bariz bir sorun görülmüyor (şaşırtıcı!)")
    print("     Eğer model yine de çökmüştür, sorun daha derinde bir optimization dinamiği olabilir.")
else:
    print("SORUNLAR BULUNDU:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "="*80)
