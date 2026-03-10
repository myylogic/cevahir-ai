# Training System V2 - Modüler Eğitim Sistemi

**Versiyon:** 2.0.0  
**Durum:** 🚧 Geliştirme Aşaması  
**Tarih:** 2025-12-21

---

## 🎯 Genel Bakış

Training System V2, modüler ve SOLID prensiplerine uygun bir eğitim orkestratörüdür. V1'den farklı olarak:

- ✅ **Modüler Yapı:** Her sorumluluk ayrı modülde
- ✅ **SOLID Principles:** Single Responsibility, Open/Closed, etc.
- ✅ **Cache-First:** DataLoaderManager yerine cache-first yaklaşım
- ✅ **Sabit Vocab:** BPE rebuild yok, sadece validasyon
- ✅ **V2 TrainingManager:** Yeni modüler training manager entegrasyonu
- ✅ **EOS Weight 0.1:** Endüstri standardı loss function

---

## 📁 Yapı

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
└── ARCHITECTURE.md
```

---

## 🔧 Modüller

### 1. BPEValidator
**Sorumluluk:** BPE dosyalarının varlığını kontrol etme

**Kullanım:**
```python
from training_system.v2.core import BPEValidator

validator = BPEValidator(logger=logger)
validator.validate_files(vocab_path, merges_path)
```

---

### 2. CriterionManager
**Sorumluluk:** EOS weight 0.1 ve label smoothing 0.1 ile loss function oluşturma

**Kullanım:**
```python
from training_system.v2.core import CriterionManager

manager = CriterionManager(logger=logger)
criterion = manager.create_criterion(
    vocab_size=vocab_size,
    eos_id=eos_id,
    pad_id=pad_id,
    device=device,
    label_smoothing=0.1,
    eos_weight=0.1
)
```

---

### 3. DataPreparator
**Sorumluluk:** Cache-first data preparation ve autoregressive formatting

**Kullanım:**
```python
from training_system.v2.core import DataPreparator

preparator = DataPreparator(logger=logger)
train_data, val_data, vocab_size = preparator.prepare_from_cache(
    data_cache=data_cache,
    tokenizer_core=tokenizer_core,
    config=config
)
```

---

### 4. ConfigManager
**Sorumluluk:** V2 TrainingManager için config hazırlama

**Kullanım:**
```python
from training_system.v2.core import ConfigManager

manager = ConfigManager(logger=logger)
training_config = manager.prepare_training_config(
    base_config=config,
    tokenizer_core=tokenizer_core,
    device=device
)
```

---

### 5. TrainingService
**Sorumluluk:** Eğitim orkestrasyonu (Facade)

**Kullanım:**
```python
from training_system.v2 import TrainingService

service = TrainingService(config)
train_loss, val_loss = service.train()
```

---

## 🚀 Kullanım Örneği

```python
from training_system.v2 import TrainingService

config = {
    "data_dir": "education",
    "vocab_path": "data/vocab_lib/vocab.json",
    "merges_path": "data/merges_lib/merges.txt",
    "batch_size": 8,
    "epochs": 10,
    "max_seq_length": 512,
    "device": "cuda",
    # ... diğer config parametreleri
}

service = TrainingService(config)
train_loss, val_loss = service.train()
```

---

## ✅ V1'den Farklar

| Özellik | V1 | V2 |
|---------|----|----|
| **BPE Rebuild** | ✅ Var | ❌ Yok (sadece validasyon) |
| **DataLoaderManager** | ✅ Kullanılıyor | ❌ Kullanılmıyor (cache-first) |
| **Modüler Yapı** | ❌ Monolitik | ✅ Modüler |
| **TrainingManager** | V1 | V2 |
| **Monitoring** | training_monitor.py | V2 TrainingManager'dan |
| **EOS Weight** | ✅ Var | ✅ Var (CriterionManager'da) |

---

## 📝 Notlar

- **Sabit Vocab Stratejisi:** Vocab sadece `train_bpe.py` ile oluşturulmalı
- **Cache-First:** Veri cache'den gelir, DataLoaderManager kullanılmaz
- **Autoregressive Formatting:** DataPreparator'da yapılır (BOS/EOS ekleme)
- **EOS Weight 0.1:** CriterionManager'da uygulanır

---

**Son Güncelleme:** 2025-12-21

