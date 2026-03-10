# -*- coding: utf-8 -*-
"""
Training Veri EOS Analiz
================================================
Eğitim verisi'ndeki:
- EOS token'ları ne kadar sık görünüyor?
- Ortalama sequence uzunluğu?
- EOS'un pozisyon dağılımı?
"""
import os
import sys
import pickle
import numpy as np
from collections import Counter
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore

print("="*80)
print("TRAINING VERİSİ EOS ANALİZİ")
print("="*80)

# ============================================================================
# 1. TOKENIZER & TOKEN IDs
# ============================================================================
print("\n[1] Tokenizer yükleniyor...")

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

print(f"[OK] BOS={BOS_ID}, EOS={EOS_ID}, PAD={PAD_ID}")

# ============================================================================
# 2. CACHE VERI YÜKLEME
# ============================================================================
print("\n[2] Cache verileri aranıyor...")

cache_dir = Path(project_root) / "data" / "preprocessed_data"
print(f"  Cache dir: {cache_dir}")
print(f"  Exists: {cache_dir.exists()}")

if not cache_dir.exists():
    print("[ERROR] Cache dizini bulunamadı!")
    sys.exit(1)

# En yakın çıkış (outputs) dosyası ara
cache_files = list(cache_dir.glob("cached_outputs_*.pkl"))
if not cache_files:
    print("[ERROR] Cache dosyaları bulunamadı!")
    print(f"  Dosyalar: {list(cache_dir.glob('*.pkl'))[:5]}")
    sys.exit(1)

# Son dosyayı al
latest_cache = sorted(cache_files)[-1]
print(f"  Yüklenen: {latest_cache.name}")

with open(latest_cache, 'rb') as f:
    cached_data = pickle.load(f)

# Data formatını anla
if isinstance(cached_data, dict):
    keys = list(cached_data.keys())
    print(f"  Format: dict, keys={keys[:5]}")
    token_sequences = [cached_data[k] for k in keys if k != 'metadata']
elif isinstance(cached_data, list):
    print(f"  Format: list, {len(cached_data)} examples")
    token_sequences = cached_data
else:
    print(f"[ERROR] Unknown format: {type(cached_data)}")
    sys.exit(1)

print(f"[OK] {len(token_sequences)} örnek yüklendi")

# ============================================================================
# 3. EOS ANALİZİ
# ============================================================================
print("\n[3] EOS analiz yapılıyor...")

eos_count = 0
eos_positions = []
sequence_lengths = []
eos_is_last = 0
eos_is_not_last = 0

for i, seq in enumerate(token_sequences):
    if not isinstance(seq, (list, np.ndarray)):
        continue
    
    seq_list = list(seq) if isinstance(seq, np.ndarray) else seq
    sequence_lengths.append(len(seq_list))
    
    # EOS'ları bul
    for j, token_id in enumerate(seq_list):
        if token_id == EOS_ID:
            eos_count += 1
            eos_positions.append(j)
            
            # EOS son mı?
            if j == len(seq_list) - 1:
                eos_is_last += 1
            else:
                eos_is_not_last += 1

print(f"[OK] Analiz tamamlandı")

# ============================================================================
# 4. İSTATİSTİKLER
# ============================================================================
print("\n[4] İstatistikler:")

print(f"\nSequence istatistikleri:")
print(f"  Toplam örnek: {len(token_sequences)}")
print(f"  Ortalama uzunluk: {np.mean(sequence_lengths):.2f}")
print(f"  Min uzunluk: {np.min(sequence_lengths)}")
print(f"  Max uzunluk: {np.max(sequence_lengths)}")
print(f"  Median uzunluk: {np.median(sequence_lengths):.2f}")
print(f"  Std dev: {np.std(sequence_lengths):.2f}")

print(f"\nEOS istatistikleri:")
print(f"  Toplam EOS token: {eos_count}")
print(f"  Ortalama EOS per sequence: {eos_count / len(token_sequences):.4f}")
print(f"  EOS oranı (EOS / total tokens): {eos_count / sum(sequence_lengths):.6f}")

if eos_positions:
    print(f"\nEOS pozisyon analizi:")
    print(f"  Ortalama EOS pozisyonu: {np.mean(eos_positions):.2f}")
    print(f"  Min EOS pozisyonu: {np.min(eos_positions)}")
    print(f"  Max EOS pozisyonu: {np.max(eos_positions)}")
    print(f"  Median EOS pozisyonu: {np.median(eos_positions):.2f}")
    
    print(f"\nEOS pozisyon dağılımı:")
    print(f"  EOS sequence'in son token'ı: {eos_is_last}/{eos_count} ({100*eos_is_last/eos_count:.1f}%)")
    print(f"  EOS sequence'in orta token'ı: {eos_is_not_last}/{eos_count} ({100*eos_is_not_last/eos_count:.1f}%)")

# ============================================================================
# 5. TEŞHIS
# ============================================================================
print("\n" + "="*80)
print("TEŞHİS")
print("="*80)

issues = []

if eos_count < 100:
    issues.append(f"[UYARI] EOS sayısı çok az: {eos_count} (beklenen: >1000)")

if eos_count == 0:
    issues.append(f"[KRITIK] EOS yok! Data cache'i yanlış mi?")

if eos_count / len(token_sequences) < 0.5:
    issues.append(f"[UYARI] Ortalama EOS < 0.5 per sequence. Çoğu sequence'de EOS yok mu?")

if eos_is_last < eos_count * 0.8:
    issues.append(f"[UYARI] EOS'lar sequence'in sonunda değil ({100*eos_is_last/eos_count:.1f}%). Data format yanlış olabilir")

if not issues:
    print("[OK] EOS dağılımı sağlıklı görünüyor")
    print("     Sorun training dynamics ya da hyperparameter'lardadır (learning rate, eos_weight, label smoothing)")
else:
    print("SORUNLAR BULUNDU:")
    for issue in issues:
        print(f"  {issue}")

print("="*80)
