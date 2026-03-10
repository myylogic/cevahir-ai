# 🔧 Train-Validation Overlap Sorunu Çözümü

## 🚨 Sorun

**Train ve Validation setlerinde %4.04 overlap vardı!**
- 1198 örnek hem train hem validation'da bulunuyordu
- Bu "data leakage" sorunuydu
- Validation loss gerçek performansı yansıtmıyordu

## ✅ Çözüm

### 📊 Endüstri Standardı

**OVERLAP: %0 OLMALI!**
- Hiçbir örnek hem train hem validation'da olmamalı
- Aynı dokümanın chunk'ları aynı set'te kalmalı (document-level split)
- Hash-based deduplication kullanılmalı

### 🛠️ Yapılan Değişiklikler

#### 1. **prepare_cache.py** - source_id Kaydedilmesi

**Önceki Durum:** source_id cache'e kaydedilmiyordu ❌

```python
# ÖNCE (HATALI)
if len(item) == 3:
    inp_ids, tgt_ids, source_id = item  # source_id okuyor
# ...
formatted_data.append((seq_in, seq_tgt))  # ❌ source_id kayboluyordu!
```

**Yeni Durum:** source_id cache'e kaydediliyor ✅

```python
# ŞIMDI (DOĞRU)
source_id = None
if len(item) == 3:
    inp_ids, tgt_ids, source_id = item  # source_id okuyor
# ...
if source_id is not None:
    formatted_data.append((seq_in, seq_tgt, source_id))  # ✅ source_id saklanıyor!
else:
    formatted_data.append((seq_in, seq_tgt))  # geriye dönük uyumluluk
```

#### 2. **data_preparator.py** - source_id İle Split

**Önceki Durum:** Cache'den yüklerken source_id kaybediliyordu ❌

```python
# ÖNCE (HATALI)
for inp_list, tgt_list in formatted_data:  # ❌ source_id kayboluyordu!
    inp_tensor = torch.tensor(inp_list, dtype=torch.long, device="cpu")
    tgt_tensor = torch.tensor(tgt_list, dtype=torch.long, device="cpu")
    formatted_tensors.append((inp_tensor, tgt_tensor))
```

**Yeni Durum:** source_id korunuyor ve split için kullanılıyor ✅

```python
# ŞIMDI (DOĞRU)
for item in formatted_data:
    if len(item) == 3:
        inp_list, tgt_list, source_id = item  # ✅ source_id var
        formatted_tensors.append((inp_tensor, tgt_tensor, source_id))
    else:
        # source_id yok (geriye dönük uyumluluk)
        formatted_tensors.append((inp_tensor, tgt_tensor))
```

#### 3. **Overlap Kontrolü İyileştirildi**

- **TAM token dizisi** kullanılarak hash oluşturuluyor (önceden sadece 100 token)
- PAD token'ları hash'den çıkarılıyor (daha güvenilir tespit)
- MD5 yerine **SHA256** kullanılıyor (collision riski azaldı)
- Detaylı raporlama eklendi

```python
# ✅ YENİ: Gelişmiş hash fonksiyonu
def hash_sequence(seq):
    # PAD'leri temizle
    cleaned_seq = [t for t in seq if t != pad_id]
    # TAM diziyi kullan (truncate yok!)
    seq_str = str(cleaned_seq)
    return hashlib.sha256(seq_str.encode()).hexdigest()
```

#### 4. **Document-Level Split**

source_id kullanarak aynı dokümanın tüm chunk'ları aynı set'te kalıyor:

```python
# source_id'lere göre grupla
source_id_to_examples = {}
for inp, tgt, source_id in data:
    if source_id not in source_id_to_examples:
        source_id_to_examples[source_id] = []
    source_id_to_examples[source_id].append((inp, tgt))

# source_id'leri shuffle et (örnekleri değil!)
source_ids = list(source_id_to_examples.keys())
random.shuffle(source_ids)

# source_id'lere göre split yap
train_source_ids = set(source_ids[:train_size])
val_source_ids = set(source_ids[train_size:])
```

## 🔄 Çözümü Uygulamak İçin

### Adım 1: Cache'i Yeniden Oluştur

Mevcut cache'de source_id yok, yeniden oluşturmalısın:

```powershell
# Cache'i temizle ve yeniden oluştur
python training_system/prepare_cache.py --clear-cache
```

### Adım 2: Eğitimi Başlat

```powershell
# Yeni cache ile eğitim başlat
python training_system/v2/train.py
```

### Adım 3: Log'ları Kontrol Et

Eğitim başladığında şu mesajları göreceksin:

✅ **Başarılı Durum (source_id var):**
```
[DataPreparator] source_id tespit edildi - overlap önleme aktif
[DataPreparator] source_id bazlı split: 234 source → train, 59 source → val
[DataPreparator] Örnek dağılımı: train=34,260, val=8,565
[DataPreparator] ✅ Overlap kontrolü: %0 overlap - Mükemmel!
```

❌ **Fallback Durum (source_id yok - eski cache):**
```
[DataPreparator] source_id yok - normal split yapılıyor (overlap olabilir)
[DataPreparator] ❌ OVERLAP TESPİT EDİLDİ: 1198 örnek (4.04%)
[DataPreparator] ⚠️  Overlap temizleniyor (val setinden çıkarılıyor)...
[DataPreparator] ✅ Overlap temizlendi: 1198 örnek val'den çıkarıldı
```

## 📊 Beklenen Sonuçlar

### Overlap %0 ile:
- ✅ Validation loss gerçek performansı gösterir
- ✅ Overfitting daha doğru tespit edilir
- ✅ Erken durdurma (early stopping) düzgün çalışır
- ✅ Endüstri standartlarına uygun eğitim

### Overlap %4.04 ile (Eski durum):
- ❌ Validation loss yapay olarak düşük
- ❌ Model val setini ezberlemiş gibi görünür
- ❌ Gerçek performans bilinmiyor
- ❌ Overfitting gizli kalabilir

## 🎯 Doğrulama

### Test Komutu

```powershell
# Overlap kontrolü
python -c "
from training_system.v2.core.data_preparator import DataPreparator
from training_system.data_cache import DataCache
from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG

# Config
config = {
    'data_dir': 'education',
    'vocab_path': BPE_CONFIG.get('vocab_file'),
    'merges_path': BPE_CONFIG.get('merges_file'),
    'max_seq_length': 768,
    'train_val_split': 0.8,
    'split_seed': 42,
}

# Initialize
cache = DataCache('education', '.cache/preprocessed_data', True)
tokenizer = TokenizerCore(config)
tokenizer.finalize_vocab()
preparator = DataPreparator()

# Load data
train, val, vocab_size = preparator.prepare_from_cache(cache, tokenizer, config)

print(f'✅ Train: {len(train)}, Val: {len(val)}')
"
```

### Başarı Kriterleri

- ✅ source_id tespit edilmeli
- ✅ Overlap %0 olmalı
- ✅ Val seti boyutu mantıklı olmalı (~20% total)

## 🔬 Teknik Detaylar

### Hash Fonksiyonu

```python
def hash_sequence(seq):
    # PAD'leri temizle (daha güvenilir hash)
    cleaned_seq = [t for t in seq if t != PAD_ID]
    
    # TAM diziyi kullan (truncate YOK!)
    seq_str = str(cleaned_seq)
    
    # SHA256 kullan (MD5'ten daha güvenilir)
    return hashlib.sha256(seq_str.encode()).hexdigest()
```

### source_id Bazlı Split

```python
# Aynı dokümanın chunk'ları aynı set'te kalır
# source_id: Her dokümana özgü unique ID
# - QA çifti → her çift ayrı source_id
# - RAW text → her dosya ayrı source_id
# - Uzun chunk split → aynı source_id (overlap önlenir)
```

## 📚 Kaynaklar

**Endüstri Standartları:**
- GPT: Document-level split + deduplication
- BERT: Document-level split
- LLaMA: Exact deduplication + document-level split
- Bloom: Multi-stage deduplication

**Best Practices:**
1. Document-level split (chunk-level değil!)
2. Full sequence hashing (truncate yok!)
3. Deterministic splitting (seed kullan)
4. Cross-validation (optional, büyük dataset'lerde)

## ✨ Sonuç

- ✅ source_id artık cache'e kaydediliyor
- ✅ Document-level split aktif
- ✅ Overlap kontrolü geliştirildi (SHA256, full sequence)
- ✅ Detaylı raporlama eklendi
- ✅ Endüstri standartlarına uygun

---

**💡 Öneri:** Cache'i yeniden oluştur ve overlap'in %0 olduğunu doğrula!

```powershell
# 1. Eski cache'i temizle
python training_system/prepare_cache.py --clear-cache

# 2. Eğitimi başlat
python training_system/v2/train.py

# 3. Log'ları kontrol et - "✅ Overlap kontrolü: %0 overlap" görmek zorundasın!
```

