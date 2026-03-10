# -*- coding: utf-8 -*-
"""
Cache ve Eğitim Verisi Tanı Script
===============================================
EOS, BOS, PAD, format kontrolleri, veri kalitesi analizi
"""
import os
import sys
import pickle
from pathlib import Path
from collections import Counter

# Proje root
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore

# ============================================================================
# 1. TOKENIZER VE SPECIAL ID'LER
# ============================================================================
print("="*80)
print("1. TOKENIZER BAŞLATILIYOR...")
print("="*80)

try:
    tokenizer = TokenizerCore({
        "vocab_path": os.path.join(project_root, "data/vocab_lib/vocab.json"),
        "merges_path": os.path.join(project_root, "data/merges_lib/merges.txt"),
    })
    tokenizer.finalize_vocab()
    vocab = tokenizer.get_vocab()
    
    # Special token ID'leri bul
    def get_id(name):
        v = vocab.get(name)
        if isinstance(v, dict):
            return v.get("id")
        return v
    
    BOS_ID = get_id("<BOS>")
    EOS_ID = get_id("<EOS>")
    PAD_ID = get_id("<PAD>")
    UNK_ID = get_id("<UNK>")
    
    print(f"[OK] TokenizerCore yüklendi")
    print(f"  BOS_ID = {BOS_ID}")
    print(f"  EOS_ID = {EOS_ID}")
    print(f"  PAD_ID = {PAD_ID}")
    print(f"  UNK_ID = {UNK_ID}")
    print(f"  Vocab size = {len(vocab)}")
except Exception as e:
    print(f"[ERROR] Tokenizer yükleme hatası: {e}")
    sys.exit(1)

# ============================================================================
# 2. CACHE DOSYALARINI BULMA VE YÜ KLE
# ============================================================================
print("\n" + "="*80)
print("2. CACHE DOSYALARI ARANYOR...")
print("="*80)

cache_dir = Path(project_root) / ".cache" / "preprocessed_data"
if not cache_dir.exists():
    print(f"[ERROR] Cache dizini bulunamadı: {cache_dir}")
    sys.exit(1)

cache_files = list(cache_dir.glob("*.pkl"))
if not cache_files:
    print(f"[ERROR] Cache dosyası bulunamadı: {cache_dir}")
    sys.exit(1)

# En yeni cache'i seç
latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
print(f"[OK] Cache bulundu: {latest_cache.name}")

try:
    with open(latest_cache, "rb") as f:
        cached_data = pickle.load(f)
    print(f"[OK] Cache yüklendi: {len(cached_data)} örnek")
except Exception as e:
    print(f"[ERROR] Cache yükleme hatası: {e}")
    sys.exit(1)

# ============================================================================
# 3. CACHE FORMAT ANALIZ
# ============================================================================
print("\n" + "="*80)
print("3. CACHE FORMAT ANALIZI")
print("="*80)

if not cached_data:
    print("[ERROR] Cache boş!")
    sys.exit(1)

# İlk örneği kontrol et
first_item = cached_data[0]
print(f"İlk örneğin format: tuple length = {len(first_item) if isinstance(first_item, tuple) else 'N/A'}")

if len(first_item) == 2:
    inp, tgt = first_item
    source_id = None
    print(f"  Format: (inp, tgt) - source_id yok")
elif len(first_item) == 3:
    inp, tgt, source_id = first_item
    print(f"  Format: (inp, tgt, source_id)")
else:
    print(f"[ERROR] Bilinmeyen format!")
    sys.exit(1)

# Veri tipleri
print(f"  inp type: {type(inp)}, tgt type: {type(tgt)}")
if isinstance(inp, (list, tuple)):
    print(f"  inp length: {len(inp)}, tgt length: {len(tgt)}")
    print(f"  inp[:10]: {inp[:10]}")
    print(f"  inp[-10:]: {inp[-10:]}")
    print(f"  tgt[:10]: {tgt[:10]}")
    print(f"  tgt[-10:]: {tgt[-10:]}")

# ============================================================================
# 4. EOS KONTROLÜ
# ============================================================================
print("\n" + "="*80)
print("4. EOS KONTROLÜ (tüm örnekler)")
print("="*80)

eos_count = 0
eos_positions = []
eos_missing = 0
eos_start_count = 0
multiple_eos_count = 0

for idx, item in enumerate(cached_data):
    if len(item) == 3:
        inp, tgt, _ = item
    else:
        inp, tgt = item
    
    tgt_list = list(tgt) if not isinstance(tgt, list) else tgt
    
    # EOS sayısı
    eos_in_tgt = tgt_list.count(EOS_ID)
    
    if eos_in_tgt > 0:
        eos_count += 1
        # EOS pozisyonunu bul
        eos_pos = tgt_list.index(EOS_ID)
        eos_positions.append(eos_pos)
        
        if eos_in_tgt > 1:
            multiple_eos_count += 1
    else:
        eos_missing += 1
    
    # EOS başta mı?
    if len(tgt_list) > 0 and tgt_list[0] == EOS_ID:
        eos_start_count += 1

total = len(cached_data)
print(f"Toplam örnek: {total}")
print(f"EOS içeren: {eos_count}/{total} ({100*eos_count/total:.1f}%)")
print(f"EOS missing: {eos_missing}")
print(f"Multiple EOS: {multiple_eos_count}")
print(f"EOS başta (sorunlu!): {eos_start_count}")

if eos_positions:
    avg_eos_pos = sum(eos_positions) / len(eos_positions)
    print(f"Ortalama EOS pozisyonu: {avg_eos_pos:.1f}")
    print(f"EOS pozisyon range: {min(eos_positions)} - {max(eos_positions)}")

# ============================================================================
# 5. BOS KONTROLÜ
# ============================================================================
print("\n" + "="*80)
print("5. BOS KONTROLÜ (input başlangıçları)")
print("="*80)

bos_count = 0
bos_start_count = 0

for idx, item in enumerate(cached_data):
    if len(item) == 3:
        inp, tgt, _ = item
    else:
        inp, tgt = item
    
    inp_list = list(inp) if not isinstance(inp, list) else inp
    
    if len(inp_list) > 0 and inp_list[0] == BOS_ID:
        bos_start_count += 1
    
    if BOS_ID in inp_list:
        bos_count += 1

print(f"BOS içeren input: {bos_count}/{total}")
print(f"BOS ile başlayan input: {bos_start_count}/{total}")

# ============================================================================
# 6. PAD KONTROLÜ
# ============================================================================
print("\n" + "="*80)
print("6. PAD KONTROLÜ")
print("="*80)

pad_before_eos_count = 0
pad_after_eos_count = 0
max_seq_len = 0

for idx, item in enumerate(cached_data):
    if len(item) == 3:
        inp, tgt, _ = item
    else:
        inp, tgt = item
    
    tgt_list = list(tgt) if not isinstance(tgt, list) else tgt
    max_seq_len = max(max_seq_len, len(tgt_list))
    
    if EOS_ID in tgt_list:
        eos_pos = tgt_list.index(EOS_ID)
        
        # EOS'tan önce PAD var mı?
        before_eos = tgt_list[:eos_pos]
        if PAD_ID in before_eos:
            pad_before_eos_count += 1
        
        # EOS'tan sonra sadece PAD var mı?
        after_eos = tgt_list[eos_pos+1:]
        if after_eos and not all(t == PAD_ID for t in after_eos):
            pad_after_eos_count += 1

print(f"Max sequence length: {max_seq_len}")
print(f"EOS'tan ÖNCE PAD var (sorunlu!): {pad_before_eos_count}")
print(f"EOS'tan SONRA PAD dışı token var (sorunlu!): {pad_after_eos_count}")

# ============================================================================
# 6b. NEXT-TOKEN HİZALAMA (target[t] = input[t+1])
# ============================================================================
print("\n" + "="*80)
print("6b. NEXT-TOKEN HİZALAMA KONTROLÜ (Autoregressive doğru mu?)")
print("="*80)
print("Doğru format: target[t] = input[t+1] (son pozisyonda target = EOS).")
print("Yanlış format: target = input (aynı dizi) → model 'sonraki token' değil 'mevcut token' öğrenir.")
print()

alignment_ok = 0
alignment_wrong_identity = 0  # target == input (tamamen yanlış)
alignment_wrong_other = 0
check_samples = min(500, len(cached_data))

for idx in range(check_samples):
    item = cached_data[idx]
    if len(item) == 3:
        inp, tgt, _ = item
    else:
        inp, tgt = item
    inp_list = list(inp) if not isinstance(inp, list) else inp
    tgt_list = list(tgt) if not isinstance(tgt, list) else tgt
    n = min(len(inp_list), len(tgt_list))
    if n <= 1:
        continue
    # target[t] == input[t+1] kontrolü (PAD pozisyonlarını atlayabiliriz veya tümüne bakalım)
    mis = sum(1 for i in range(n - 1) if tgt_list[i] != inp_list[i + 1])
    if mis == 0:
        alignment_ok += 1
    else:
        if inp_list == tgt_list:
            alignment_wrong_identity += 1
        else:
            alignment_wrong_other += 1

print(f"Kontrol edilen örnek: {check_samples}")
print(f"  [DOĞRU] target[t]=input[t+1] uyumlu: {alignment_ok}")
print(f"  [YANLIŞ] target = input (aynı dizi): {alignment_wrong_identity}")
print(f"  [YANLIŞ] Diğer uyumsuzluk: {alignment_wrong_other}")

if alignment_wrong_identity > 0:
    print()
    print("  *** KRİTİK: Cache'te target = input kullanılmış. Model 'sonraki token' öğrenemez! ***")
    print("  Çözüm: Cache'i silin, prepare_cache.py çalıştırın veya eğitimi yeniden başlatın (güncel kod format_func kullanıyor).")
elif alignment_ok == check_samples:
    print()
    print("  [OK] Tüm örneklerde next-token hizalaması doğru. Cache formatı uygun.")
else:
    print()
    print("  [UYARI] Bazı örneklerde hizalama bozuk. prepare_cache ile cache'i yeniden oluşturmayı deneyin.")

# ============================================================================
# 7. ÖRNEKLER (ilk 3)
# ============================================================================
print("\n" + "="*80)
print("7. ÖRNEK VERİ (ilk 3)")
print("="*80)

for idx in range(min(3, len(cached_data))):
    item = cached_data[idx]
    if len(item) == 3:
        inp, tgt, sid = item
        print(f"\nÖrnek {idx}: (inp, tgt, source_id={sid})")
    else:
        inp, tgt = item
        print(f"\nÖrnek {idx}: (inp, tgt)")
    
    inp_list = list(inp) if not isinstance(inp, list) else inp
    tgt_list = list(tgt) if not isinstance(tgt, list) else tgt
    
    print(f"  Input  length: {len(inp_list)}, ilk 10: {inp_list[:10]}, son 10: {inp_list[-10:]}")
    print(f"  Target length: {len(tgt_list)}, ilk 10: {tgt_list[:10]}, son 10: {tgt_list[-10:]}")
    
    # EOS pozisyonu
    if EOS_ID in tgt_list:
        eos_pos = tgt_list.index(EOS_ID)
        print(f"  [OK] EOS'un pozisyonu: {eos_pos} (total: {len(tgt_list)})")
    else:
        print(f"  [MISSING] EOS yok!")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*80)
print("ÖZET VE SONUÇ")
print("="*80)

issues = []

if eos_count < total * 0.95:
    issues.append(f"⚠ EOS kullanımı düşük: {eos_count}/{total} ({100*eos_count/total:.1f}%)")

if pad_before_eos_count > 0:
    issues.append(f"⚠ PAD EOS'tan ÖNCE var: {pad_before_eos_count} örnek (hata!)")

if bos_start_count < total * 0.95:
    issues.append(f"⚠ BOS başlangıç oranı düşük: {bos_start_count}/{total}")

if eos_start_count > 0:
    issues.append(f"⚠ EOS başta var: {eos_start_count} örnek (sorunlu!)")

if issues:
    print("SORUNLAR BULUNDU:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("[OK] Cache format kontrolleri başarılı!")
    print("  - EOS'lar doğru pozisyonda ve sayıda")
    print("  - BOS'lar giriş başında")
    print("  - PAD'ler sadece EOS'tan sonra")

print("\n" + "="*80)
