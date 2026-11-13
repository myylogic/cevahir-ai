# -*- coding: utf-8 -*-
"""
Cache ve Eğitim Verisi Tanı Script
===============================================
EOS, BOS, PAD, format kontrolleri, veri kalitesi analizi

Kayıt düzeni (training_system/record_format.py, CACHE_FORMAT_VERSION 2): BOS yalnız belgenin ilk
kaydının girdisinde, EOS yalnız son kaydının hedefinde bulunur; uzun belgenin ara parçaları ikisini de
almaz. Belge sınırları kayıt sırasından okunur (aynı source_id'nin ardışık kayıtları bir belgedir). Bu
yüzden kontroller EOS/BOS'u her kayıtta değil, beklendiği kayıtlarda arar; eski düzende hazırlanmış
(her kaydı EOS ile kapatan) cache "belge sürerken EOS" sorunu olarak raporlanır.
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
from training_system.record_format import record_boundaries

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
# 3b. BELGE SINIRLARI
# ============================================================================
print("\n" + "="*80)
print("3b. BELGE SINIRLARI (aynı source_id'nin ardışık kayıtları bir belgedir)")
print("="*80)

total = len(cached_data)
try:
    boundaries = record_boundaries(cached_data)
except ValueError as e:
    boundaries = None
    print(f"[ERROR] Belge sınırları okunamadı: {e}")
else:
    documents = sum(1 for starts, _ in boundaries if starts)
    print(f"Toplam örnek: {total}")
    print(f"Belge sayısı: {documents}")
    print(f"Belge sürerken biten kayıt (EOS almamalı): {sum(1 for _, ends in boundaries if not ends)}")


def content_end(seq):
    """Son PAD olmayan pozisyon (dizi tamamen PAD ise -1)."""
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] != PAD_ID:
            return i
    return -1


# ============================================================================
# 4. EOS KONTROLÜ
# ============================================================================
print("\n" + "="*80)
print("4. EOS KONTROLÜ (EOS yalnız belgeyi bitiren kaydın hedefinde, son içerik pozisyonunda)")
print("="*80)

eos_count = 0         # hedef içeriği EOS ile biten kayıt
eos_positions = []
eos_expected = 0      # belgeyi bitiren kayıt (EOS beklenir)
eos_missing = 0       # belgeyi bitiriyor ama EOS ile bitmiyor
eos_false = 0         # belge sürerken EOS ile bitiyor (sahte "metin bitti" hedefi)
eos_inner = 0         # içeriğin ortasında EOS

for idx, item in enumerate(cached_data):
    tgt_list = list(item[1])
    last = content_end(tgt_list)
    ends_with_eos = last >= 0 and tgt_list[last] == EOS_ID
    if ends_with_eos:
        eos_count += 1
        eos_positions.append(last)
    if EOS_ID in tgt_list[:max(last, 0)]:
        eos_inner += 1
    if boundaries is not None:
        ends_document = boundaries[idx][1]
        eos_expected += ends_document
        eos_missing += ends_document and not ends_with_eos
        eos_false += ends_with_eos and not ends_document

print(f"EOS ile biten hedef: {eos_count}/{total}")
if boundaries is not None:
    print(f"Belgeyi bitiren kayıt (EOS beklenir): {eos_expected}")
    print(f"  EOS'u eksik (sorunlu!): {eos_missing}")
    print(f"  Belge sürerken EOS (sorunlu!): {eos_false}")
print(f"İçeriğin ortasında EOS (sorunlu!): {eos_inner}")

if eos_positions:
    avg_eos_pos = sum(eos_positions) / len(eos_positions)
    print(f"Ortalama EOS pozisyonu: {avg_eos_pos:.1f}")
    print(f"EOS pozisyon range: {min(eos_positions)} - {max(eos_positions)}")

# ============================================================================
# 5. BOS KONTROLÜ
# ============================================================================
print("\n" + "="*80)
print("5. BOS KONTROLÜ (BOS yalnız belgeyi başlatan kaydın girdisinin başında)")
print("="*80)

bos_start_count = 0   # BOS ile başlayan girdi
bos_expected = 0      # belgeyi başlatan kayıt (BOS beklenir)
bos_missing = 0
bos_false = 0         # belge ortasındaki kayıt BOS ile başlıyor
bos_inner = 0         # girdinin ilk pozisyonu dışında BOS

for idx, item in enumerate(cached_data):
    inp_list = list(item[0])
    starts_with_bos = bool(inp_list) and inp_list[0] == BOS_ID
    bos_start_count += starts_with_bos
    if BOS_ID in inp_list[1:]:
        bos_inner += 1
    if boundaries is not None:
        starts_document = boundaries[idx][0]
        bos_expected += starts_document
        bos_missing += starts_document and not starts_with_bos
        bos_false += starts_with_bos and not starts_document

print(f"BOS ile başlayan input: {bos_start_count}/{total}")
if boundaries is not None:
    print(f"Belgeyi başlatan kayıt (BOS beklenir): {bos_expected}")
    print(f"  BOS'u eksik (sorunlu!): {bos_missing}")
    print(f"  Belge ortasında BOS (sorunlu!): {bos_false}")
print(f"Girdinin ortasında BOS (sorunlu!): {bos_inner}")

# ============================================================================
# 6. PAD KONTROLÜ
# ============================================================================
print("\n" + "="*80)
print("6. PAD KONTROLÜ (PAD yalnız içerikten sonra)")
print("="*80)

pad_inside_content = 0  # girdi/hedef içeriğinin arasında PAD
input_tail_not_pad = 0  # girdide içerik bittikten sonra PAD olmayan token
max_seq_len = 0

for item in cached_data:
    inp_list, tgt_list = list(item[0]), list(item[1])
    max_seq_len = max(max_seq_len, len(tgt_list))
    last = content_end(tgt_list)
    if PAD_ID in tgt_list[:last + 1] or PAD_ID in inp_list[:last + 1]:
        pad_inside_content += 1
    if any(t != PAD_ID for t in inp_list[last + 1:]):
        input_tail_not_pad += 1

print(f"Max sequence length: {max_seq_len}")
print(f"İçerik arasında PAD (sorunlu!): {pad_inside_content}")
print(f"Girdide içerikten SONRA PAD dışı token (sorunlu!): {input_tail_not_pad}")

# ============================================================================
# 6b. NEXT-TOKEN HİZALAMA (target[t] = input[t+1])
# ============================================================================
print("\n" + "="*80)
print("6b. NEXT-TOKEN HİZALAMA KONTROLÜ (Autoregressive doğru mu?)")
print("="*80)
print("Doğru format: target[t] = input[t+1] (EOS yalnız belgenin son kaydının hedefinde).")
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

    # EOS pozisyonu (yalnız belgeyi bitiren kayıtta beklenir)
    ends_document = boundaries[idx][1] if boundaries is not None else None
    last = content_end(tgt_list)
    if last >= 0 and tgt_list[last] == EOS_ID:
        if ends_document is False:
            print(f"  [SORUN] EOS var ama belge sonraki kayıtta sürüyor (pozisyon {last})")
        else:
            print(f"  [OK] EOS'un pozisyonu: {last} (total: {len(tgt_list)})")
    elif ends_document is False:
        print(f"  [OK] EOS yok: belge sonraki kayıtta sürüyor")
    else:
        print(f"  [MISSING] EOS yok!")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*80)
print("ÖZET VE SONUÇ")
print("="*80)

issues = []

# EOS/BOS oranı tek başına ölçüt değildir: uzun belgenin ara parçaları ikisini de almaz.
if boundaries is None:
    issues.append("⚠ Belge sınırları bilinmiyor: aynı source_id'nin kayıtları ardışık değil")
else:
    if eos_missing:
        issues.append(f"⚠ Belgeyi bitiren {eos_missing}/{eos_expected} kayıtta EOS yok")
    if eos_false:
        issues.append(f"⚠ Belge sürerken EOS ile biten kayıt: {eos_false} "
                      f"(eski kayıt düzeni olabilir; prepare_cache.py ile yeniden hazırlayın)")
    if bos_missing:
        issues.append(f"⚠ Belgeyi başlatan {bos_missing}/{bos_expected} kayıtta BOS yok")
    if bos_false:
        issues.append(f"⚠ Belge ortasında BOS ile başlayan kayıt: {bos_false}")

if eos_inner:
    issues.append(f"⚠ İçeriğin ortasında EOS: {eos_inner} örnek (sorunlu!)")

if bos_inner:
    issues.append(f"⚠ Girdinin ortasında BOS: {bos_inner} örnek (sorunlu!)")

if pad_inside_content:
    issues.append(f"⚠ İçerik arasında PAD: {pad_inside_content} örnek (hata!)")

if input_tail_not_pad:
    issues.append(f"⚠ Girdide içerikten sonra PAD dışı token: {input_tail_not_pad} örnek (hata!)")

if issues:
    print("SORUNLAR BULUNDU:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("[OK] Cache format kontrolleri başarılı!")
    print(f"  - EOS yalnız belge sonlarında: {eos_count} kayıt, {documents} belge")
    print(f"  - BOS yalnız belge başlarında: {bos_start_count} kayıt")
    print("  - PAD'ler sadece içerikten sonra")

print("\n" + "="*80)
