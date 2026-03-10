# -*- coding: utf-8 -*-
"""
Training Loss EOS İnceleme
================================================
Eğitim sırasında EOS için:
- CrossEntropyLoss ne kadar?
- EOS token gradients ne kadar?
- Label smoothing etkisini görmek
- eos_weight konfigürasyonunu doğrula
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
print("TRAINING LOSS EOS İNCELEMESİ")
print("="*80)

# ============================================================================
# 1. CONFIG & YÜKLEME
# ============================================================================
print("\n[1] Sistem yükleniyor...")

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
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  BOS_ID: {BOS_ID}, EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}")
    
    # Model yükle
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
    model = cevahir._model_manager.model
    model.train()  # Training mode açık
    
    print("[OK] Model yüklendi (training mode)")
    
    # Criterion oluştur
    criterion_manager = CriterionManager()
    criterion = criterion_manager.create_criterion(
        vocab_size=vocab_size,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        device="cpu",
        label_smoothing=0.0,
        eos_weight=1.0  # Mevcut config'teki değer
    )
    print(f"[OK] Criterion oluşturuldu (eos_weight=1.0, label_smoothing=0.0)")
    
except Exception as e:
    print(f"[ERROR] Yükleme hatası: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. DUMMY BATCH OLUŞTUR (EOS'u vurgula)
# ============================================================================
print("\n[2] Dummy batch oluşturuluyor...")

batch_size = 4
seq_len = 20

# Her örnek BOS + tokenlar + EOS şeklinde
dummy_batch_input = torch.randint(1, 100, (batch_size, seq_len))
dummy_batch_input[:, 0] = BOS_ID  # İlk pozisyon BOS

# Target: input'ın shifted version'ı (autoregressive)
dummy_batch_target = torch.randint(1, 100, (batch_size, seq_len))

# İlk pozisyon target'ta shifted (1. token'ın tahmini)
dummy_batch_target[:, 0] = dummy_batch_input[:, 1]

# Son pozisyon target'ta EOS olsun (model EOS tahmin etmeyi öğrensin)
dummy_batch_target[:, -1] = EOS_ID

# PAD mask oluştur (EOS'tan sonra PAD var)
pad_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
for b in range(batch_size):
    for t in range(seq_len):
        if dummy_batch_target[b, t] == EOS_ID:
            pad_mask[b, t+1:] = False  # EOS'dan sonra PAD
            break

print(f"  Input shape: {dummy_batch_input.shape}")
print(f"  Target shape: {dummy_batch_target.shape}")
print(f"  Pad mask shape: {pad_mask.shape}")
print(f"  Örnek input[0]: {dummy_batch_input[0, :10].tolist()}...")
print(f"  Örnek target[0]: {dummy_batch_target[0, :10].tolist()}...")
print(f"  Son token target'ta: {dummy_batch_target[0, -1].item()} (EOS_ID={EOS_ID})")

# ============================================================================
# 3. FORWARD PASS + LOSS HESAPLAMA
# ============================================================================
print("\n[3] Forward pass yapılıyor...")

with torch.enable_grad():
    logits, _ = model(dummy_batch_input)  # (B, T, vocab_size)
    print(f"  Logits shape: {logits.shape}")
    
    # Loss hesapla (CrossEntropyLoss reshape'e ihtiyaç duyar)
    logits_flat = logits.reshape(-1, vocab_size)  # (B*T, vocab_size)
    target_flat = dummy_batch_target.reshape(-1)  # (B*T,)
    
    loss = criterion(logits_flat, target_flat)
    print(f"  Total loss: {loss.item():.6f}")
    
    # Batch-wise loss
    batch_losses = F.cross_entropy(
        logits_flat,
        target_flat,
        reduction="none"
    ).reshape(batch_size, -1)
    
    print(f"  Loss per sequence (first 3):")
    for b in range(min(3, batch_size)):
        seq_loss = batch_losses[b].mean().item()
        print(f"    Sequence {b}: {seq_loss:.6f}")

# ============================================================================
# 4. EOS-SPECIFIC LOSS
# ============================================================================
print("\n[4] EOS-specific loss analiz...")

pad_mask_flat = pad_mask.reshape(-1)

# EOS token'ların indexleri
eos_mask = (target_flat == EOS_ID) & pad_mask_flat

if eos_mask.sum() > 0:
    print(f"  EOS token sayısı: {eos_mask.sum().item()}")
    
    # EOS pozisyonlarında logits
    eos_logits = logits_flat[eos_mask]  # (num_eos, vocab_size)
    eos_targets = target_flat[eos_mask]  # (num_eos,)
    
    # EOS için logits analiz
    eos_prob_for_eos = F.softmax(eos_logits, dim=-1)[:, EOS_ID]
    
    print(f"  EOS pozisyonlarında EOS olasılığı:")
    print(f"    Min: {eos_prob_for_eos.min().item():.6f}")
    print(f"    Max: {eos_prob_for_eos.max().item():.6f}")
    print(f"    Mean: {eos_prob_for_eos.mean().item():.6f}")
    
    # EOS için loss
    eos_loss_each = F.cross_entropy(
        eos_logits,
        eos_targets,
        reduction="none"
    )
    print(f"  EOS pozisyonlarında cross-entropy loss:")
    print(f"    Min: {eos_loss_each.min().item():.6f}")
    print(f"    Max: {eos_loss_each.max().item():.6f}")
    print(f"    Mean: {eos_loss_each.mean().item():.6f}")
    
    # Gradient flow check
    loss.backward()
    
    # EOS logits'in gradients
    if logits.grad is not None:
        eos_logits_grad = logits.grad.reshape(-1, vocab_size)[eos_mask]
        print(f"  EOS pozisyonlarında logits gradients:")
        print(f"    Mean abs grad: {eos_logits_grad.abs().mean().item():.6f}")
        print(f"    Max abs grad: {eos_logits_grad.abs().max().item():.6f}")
else:
    print("[SORUN] Batch'te EOS token yok!")

# ============================================================================
# 5. LABEL SMOOTHING ETKISI
# ============================================================================
print("\n[5] Label smoothing etkisi kontrol...")

# Label smoothing=0 ile loss
criterion_no_smooth = torch.nn.CrossEntropyLoss(reduction="mean")
loss_no_smooth = criterion_no_smooth(logits.reshape(-1, vocab_size), target_flat)

# Label smoothing=0.1 ile loss (manual)
eps = 0.1
n_classes = vocab_size
smooth_target = torch.full_like(logits, eps / (n_classes - 1))
smooth_target.scatter_(-1, dummy_batch_target.unsqueeze(-1), 1.0 - eps)
loss_smooth_manual = -(smooth_target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

print(f"  Loss (label_smoothing=0): {loss_no_smooth.item():.6f}")
print(f"  Loss (label_smoothing=0.1): {loss_smooth_manual.item():.6f}")
print(f"  Fark: {(loss_smooth_manual - loss_no_smooth).item():.6f}")

# ============================================================================
# 6. SONUÇ
# ============================================================================
print("\n" + "="*80)
print("SONUÇ")
print("="*80)

if not eos_mask.sum() > 0:
    print("[SORUN] Batch'te EOS yok!")
elif eos_prob_for_eos.mean().item() < 0.1:
    print("[SORUN] EOS olasılığı çok düşük (<0.1) — training bu batch'te EOS öğrenmede başarısız")
elif eos_loss_each.mean().item() > 2.0:
    print("[SORUN] EOS loss çok yüksek (>2.0) — model EOS'u öğrenmekte zorlanıyor")
else:
    print("[OK] Bu dummy batch'te loss ve gradient flow sağlıklı görülüyor")

print("\n[Ipucu] Eger real training'de hala EOS ogrenmiyorsa:")
print("  - Training data'daki EOS dagilimi kontrol et (cok nadir mi?)")
print("  - eos_weight'i daha artir (1.0 -> 5.0 veya 10.0)")
print("  - Label smoothing'i azalt (0.1 -> 0.0)")
print("  - Learning rate kontrol et (cok dusuk mu?)")
print("="*80)
