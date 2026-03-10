# Training System V2 - Modüler Mimari

**Tarih:** 2025-12-21  
**Versiyon:** 2.0.0  
**Durum:** 📋 Tasarım Aşaması

---

## 🎯 MİMARİ PRENSİPLERİ

### SOLID Principles
- ✅ **Single Responsibility:** Her modül tek bir sorumluluğa sahip
- ✅ **Open/Closed:** Genişlemeye açık, değişikliğe kapalı
- ✅ **Liskov Substitution:** Interface'ler tutarlı
- ✅ **Interface Segregation:** Küçük, özel interface'ler
- ✅ **Dependency Inversion:** Abstract'lara bağımlılık

### Modüler Yapı
- Her modül kendi dosyasında
- Net sorumluluk dağılımı
- Test edilebilir
- Maintainable ve scalable

---

## 📁 DOSYA YAPISI

```
training_system/v2/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── training_service.py      # Facade/Orchestrator
│   ├── bpe_validator.py          # BPE dosya validasyonu
│   ├── criterion_manager.py      # EOS weight 0.1, loss function
│   ├── data_preparator.py        # Cache-first, autoregressive formatting
│   └── config_manager.py         # V2 TrainingManager config
├── utils/
│   ├── __init__.py
│   └── data_loader_wrapper.py    # Minimal DataLoader wrapper
├── test/
│   ├── __init__.py
│   ├── test_bpe_validator.py
│   ├── test_criterion_manager.py
│   ├── test_data_preparator.py
│   ├── test_config_manager.py
│   └── test_training_service.py
└── ARCHITECTURE.md
```

---

## 🏗️ MODÜL SORUMLULUKLARI

### 1. BPEValidator (`core/bpe_validator.py`)

**Sorumluluk:** BPE dosyalarının varlığını kontrol etme

**Metodlar:**
- `validate_files(vocab_path, merges_path) -> None` - Dosyaları kontrol et, yoksa hata ver

**Özellikler:**
- Sabit vocab stratejisi vurgusu
- Anlamlı hata mesajları (train_bpe.py çalıştırılmalı)

---

### 2. CriterionManager (`core/criterion_manager.py`)

**Sorumluluk:** Loss function (criterion) oluşturma ve yönetimi

**Metodlar:**
- `create_criterion(vocab_size, eos_id, pad_id, device, label_smoothing=0.1, eos_weight=0.1) -> CrossEntropyLoss`

**Özellikler:**
- EOS weight 0.1 uygulama
- Label smoothing 0.1 uygulama
- Endüstri standardı (GPT-4, LLaMA)

---

### 3. DataPreparator (`core/data_preparator.py`)

**Sorumluluk:** Cache-first data preparation ve autoregressive formatting

**Metodlar:**
- `prepare_from_cache(data_cache, tokenizer_core, config) -> Tuple[List, List, int]` - Cache'den veri hazırla
- `_apply_autoregressive_formatting(raw_data, bos_id, eos_id, max_seq_len) -> List[Tuple]` - BOS/EOS ekle
- `_split_train_val(data, train_ratio=0.8, seed=42) -> Tuple[List, List]` - Train/Val split

**Özellikler:**
- Cache-first yaklaşım
- Autoregressive formatting (BOS/EOS ekleme)
- Train/Val split

---

### 4. ConfigManager (`core/config_manager.py`)

**Sorumluluk:** V2 TrainingManager için config hazırlama

**Metodlar:**
- `prepare_training_config(base_config, tokenizer_core, device) -> Dict[str, Any]`

**Özellikler:**
- V2 TrainingManager config formatı
- Config adaptasyonu
- Default değerler

---

### 5. TrainingService (`core/training_service.py`)

**Sorumluluk:** Eğitim orkestrasyonu (Facade Pattern)

**Metodlar:**
- `__init__(config)` - Tüm bileşenleri hazırla
- `train() -> Tuple[float, float]` - Eğitimi başlat

**Bağımlılıklar:**
- BPEValidator
- CriterionManager
- DataPreparator
- ConfigManager
- V2 TrainingManager
- ModelManager
- TokenizerCore
- DataCache

---

### 6. DataLoaderWrapper (`utils/data_loader_wrapper.py`)

**Sorumluluk:** Minimal DataLoader wrapper (V2 TrainingManager için)

**Metodlar:**
- `create_dataloaders(train_data, val_data, batch_size, device) -> Tuple[DataLoader, DataLoader]`

**Özellikler:**
- Minimal overhead
- Cache'den gelen veriyi DataLoader'a çevirir

---

## 🔄 VERİ AKIŞI

```
1. TrainingService.__init__(config)
   ├─> BPEValidator.validate_files()
   ├─> TokenizerCore oluştur
   ├─> DataCache oluştur
   ├─> ModelManager oluştur
   ├─> CriterionManager.create_criterion() (EOS weight 0.1)
   └─> ConfigManager.prepare_training_config()

2. TrainingService.train()
   ├─> DataPreparator.prepare_from_cache() (cache-first)
   │   ├─> Cache'den yükle veya oluştur
   │   ├─> Autoregressive formatting (BOS/EOS ekle)
   │   └─> Train/Val split
   ├─> DataLoaderWrapper.create_dataloaders() (minimal wrapper)
   ├─> V2 TrainingManager oluştur
   │   ├─> criterion (EOS weight 0.1 içeren)
   │   ├─> config (V2 format)
   │   └─> train_loader, val_loader
   └─> TrainingManager.train() (V2 training loop)
```

---

## ✅ AVANTAJLAR

1. **Modülerlik:** Her modül bağımsız test edilebilir
2. **Maintainability:** Kod daha organize ve okunabilir
3. **Testability:** Her modül için unit testler yazılabilir
4. **Scalability:** Yeni özellikler kolayca eklenebilir
5. **SOLID Uyumu:** Clean code prensipleri
6. **Separation of Concerns:** Her modül tek sorumluluğa sahip

---

## 📝 NOTLAR

- V1'den farklı olarak BPE rebuild mantığı YOK (sabit vocab stratejisi)
- DataLoaderManager kullanımı YOK (cache-first yaklaşım)
- Training monitoring V2 TrainingManager'dan geliyor
- Autoregressive formatting DataPreparator'da

**Son Güncelleme:** 2025-12-21

