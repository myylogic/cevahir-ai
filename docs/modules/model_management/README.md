# 🤖 Model Management - Kapsamlı Dokümantasyon

**Versiyon:** V-5  
**Son Güncelleme:** 2025-01-27  
**Durum:** Production-Ready

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Ana Bileşenler](#ana-bileşenler)
4. [Çalışma Prensibi](#çalışma-prensibi)
5. [API Referansı](#api-referansı)
6. [Kullanım Örnekleri](#kullanım-örnekleri)
7. [Modül Detayları](#modül-detayları)

---

## 🎯 Genel Bakış

**Model Management**, Cevahir Sinir Sistemi'nde modelin yaşam döngüsünü (lifecycle) yöneten merkezi modüldür. Model initialization, training tools (optimizer, scheduler, criterion), save/load, update, ve forward pass işlemlerini tek bir yerden yönetir.

### Temel Özellikler

- ✅ **Model Lifecycle Management:** Build, initialize, save, load
- ✅ **Training Tools:** Optimizer, scheduler, criterion management
- ✅ **Forward Pass:** Inference ve training için unified API
- ✅ **TensorBoard Integration:** Tek yerden TensorBoard yönetimi
- ✅ **V-4 Architecture Support:** Tüm V-4 parametrelerini otomatik destekler
- ✅ **Device Management:** GPU/CPU/MPS otomatik seçimi
- ✅ **Flexible Updates:** Freeze/unfreeze, learning rate scheduling, vb.

---

## 🏗️ Mimari Yapı

### Yüksek Seviye Mimari

```
ModelManager (Orchestrator)
├── ModelInitializer
│   ├── build_model()
│   ├── initialize_optimizer()
│   ├── initialize_criterion()
│   └── initialize_scheduler()
│
├── ModelSaver
│   ├── save_checkpoint()
│   ├── save_weights_only()
│   └── save_additional_info()
│
├── ModelLoader
│   ├── load_model()
│   ├── load_optimizer()
│   ├── load_scheduler()
│   └── load_all()
│
└── ModelUpdater
    ├── update_model()
    ├── update_optimizer()
    ├── update_scheduler()
    └── bulk_update()
```

### ModelManager İlişkileri

```
ModelManager
    │
    ├── HAS-A CevahirNeuralNetwork (Model)
    ├── HAS-A Optimizer (AdamW/Adam/SGD/...)
    ├── HAS-A Criterion (CrossEntropyLoss/...)
    ├── HAS-A Scheduler (ReduceLROnPlateau/...)
    │
    ├── USES ModelInitializer (build/initialize)
    ├── USES ModelSaver (save checkpoint)
    ├── USES ModelLoader (load checkpoint)
    └── USES ModelUpdater (update parameters)
```

---

## 🧩 Ana Bileşenler

### 1. ModelManager

**Dosya:** `model_management/model_manager.py`  
**Sınıf:** `ModelManager`

**Sorumluluklar:**
- Model lifecycle orchestration
- Training tools management (optimizer, scheduler, criterion)
- Forward pass coordination
- Save/load operations
- TensorBoard integration
- Device management

**Özellikler:**
- ✅ V-2/V-3/V-4 Architecture support (otomatik parametre geçişi)
- ✅ Pre-norm/Post-norm desteği
- ✅ Causal masking
- ✅ KV Cache support (V-4)
- ✅ TensorBoard integration
- ✅ Multimodal API (audio, vision, text)

---

### 2. ModelInitializer

**Dosya:** `model_management/model_initializer.py`  
**Sınıf:** `ModelInitializer`

**Sorumluluklar:**
- Model initialization (V-2/V-3/V-4 support)
- Optimizer initialization (AdamW/Adam/SGD/RAdam/RMSprop)
- Criterion initialization (CrossEntropyLoss/BCEWithLogits/MSELoss/SmoothL1Loss)
- Scheduler initialization (ReduceLROnPlateau/Cosine/Step/Exponential/OneCycleLR)

**Özellikler:**
- ✅ Automatic parameter filtering (constructor signature-based)
- ✅ Weight decay parameter groups (bias/LayerNorm exclusion)
- ✅ Torch.compile support
- ✅ Seed management (deterministic training)

---

### 3. ModelSaver

**Dosya:** `model_management/model_saver.py`  
**Sınıf:** `ModelSaver`

**Sorumluluklar:**
- Checkpoint saving (model + optimizer + scheduler + metadata)
- Weights-only saving
- Full model saving (pickle)
- Additional info saving (JSON)
- Atomic writes (crash-safe)

**Özellikler:**
- ✅ Atomic checkpoint writes
- ✅ Latest checkpoint marker
- ✅ Old checkpoint pruning
- ✅ Metadata preservation

---

### 4. ModelLoader

**Dosya:** `model_management/model_loader.py`  
**Sınıf:** `ModelLoader`

**Sorumluluklar:**
- Model loading (state_dict veya full checkpoint)
- Optimizer state loading
- Scheduler state loading
- Additional info loading (JSON)
- Device management (automatic)

**Özellikler:**
- ✅ Multiple checkpoint formats support
- ✅ Automatic device mapping
- ✅ Strict/non-strict loading
- ✅ Weights-only loading (PyTorch 2.x)

---

### 5. ModelUpdater

**Dosya:** `model_management/model_updater.py`  
**Sınıf:** `ModelUpdater`

**Sorumluluklar:**
- Model parameter updates (freeze/unfreeze, setattr, device)
- Optimizer updates (learning rate, weight decay, betas, eps, vb.)
- Scheduler updates (factor, patience, step_size, gamma, vb.)
- Bulk updates (model + optimizer + scheduler)

**Özellikler:**
- ✅ Pattern-based freeze/unfreeze (glob + regex)
- ✅ Dry-run mode (preview changes)
- ✅ Automatic frozen param filtering (optimizer)
- ✅ Validation (type checking, range checking)

---

## ⚙️ Çalışma Prensibi

### 1. Model Initialization Flow

```
User creates ModelManager(config)
    ↓
ModelManager.__init__()
    ├── Parse device (cuda/mps/cpu)
    ├── Store config
    └── Initialize components (model=None, optimizer=None, ...)
    ↓
User calls initialize()
    ↓
ModelManager.initialize()
    ├── build_model() → ModelInitializer.build_model()
    │   ├── Filter config params (constructor signature)
    │   ├── Create CevahirNeuralNetwork(**filtered_params)
    │   └── Move to device
    ├── build_optimizer() → ModelInitializer.initialize_optimizer()
    │   ├── Create param groups (weight decay split)
    │   └── Create optimizer (AdamW/Adam/SGD/...)
    ├── build_criterion() → ModelInitializer.initialize_criterion()
    │   └── Create loss function (CrossEntropyLoss/...)
    └── build_scheduler() → ModelInitializer.initialize_scheduler()
        └── Create scheduler (ReduceLROnPlateau/...)
    ↓
Model ready for training/inference
```

### 2. Forward Pass Flow

```
User calls forward(inputs, **kwargs)
    ↓
ModelManager.forward()
    ├── Check model is initialized
    ├── Set model mode (train/eval)
    ├── Move inputs to device
    ├── Apply mask transformations (if needed)
    ├── Call model.forward(**params)
    │   └── CevahirNeuralNetwork.forward()
    │       ├── DilKatmani (embedding + PE)
    │       ├── TransformerEncoderLayer × N
    │       └── Output layer
    ├── Extract logits and aux info
    └── Return (logits, aux)
```

### 3. Save/Load Flow

#### Save Flow:
```
User calls save(save_path, epoch=5)
    ↓
ModelManager.save()
    └── ModelSaver.save_model()
        ├── Collect state_dicts
        │   ├── model.state_dict()
        │   ├── optimizer.state_dict()
        │   └── scheduler.state_dict()
        ├── Collect metadata
        │   ├── epoch
        │   ├── config
        │   └── additional_info
        ├── Create checkpoint dict
        └── Atomic write (temp file → final file)
```

#### Load Flow:
```
User calls load(load_path)
    ↓
ModelManager.load()
    ├── Load checkpoint (torch.load)
    ├── Extract state_dicts
    │   ├── model_state_dict
    │   ├── optimizer_state_dict
    │   └── scheduler_state_dict
    ├── Build model (if None)
    ├── Load state_dicts
    │   ├── model.load_state_dict()
    │   ├── optimizer.load_state_dict()
    │   └── scheduler.load_state_dict()
    └── Update config (epoch, etc.)
```

### 4. Update Flow

```
User calls update(update_params)
    ↓
ModelManager.update()
    └── ModelUpdater.bulk_update()
        ├── Update Model
        │   ├── setattr (if specified)
        │   ├── freeze/unfreeze (pattern matching)
        │   └── device transfer (if specified)
        ├── Update Optimizer
        │   ├── learning_rate
        │   ├── weight_decay
        │   ├── betas, eps, momentum, ...
        │   └── Filter frozen params
        └── Update Scheduler
            ├── ReduceLROnPlateau: factor, patience, ...
            ├── StepLR: step_size, gamma
            └── CosineAnnealingLR: T_max, eta_min
        ↓
Return UpdateReport
```

---

## 📚 API Referansı

### ModelManager API

#### `__init__(config, model_class=None, **kwargs)`

ModelManager'ı başlatır.

**Parametreler:**
- `config` (Dict[str, Any]): Model ve training konfigürasyonu
- `model_class` (Optional[Type[nn.Module]]): Model sınıfı (None ise CevahirNeuralNetwork)
- `device` (Optional[Union[str, torch.device]]): Device override
- `tokenizer` (Optional[Any]): Tokenizer (multimodal için)
- `audio_processor` (Optional[Any]): Audio processor (multimodal için)
- `vision_processor` (Optional[Any]): Vision processor (multimodal için)

**Örnek:**
```python
config = {
    "vocab_size": 60000,
    "embed_dim": 1024,
    "num_heads": 16,
    "num_layers": 12,
    "learning_rate": 1e-4,
    "device": "cuda",
}
manager = ModelManager(config)
```

---

#### `initialize(**kwargs) -> ModelManager`

Model ve training tools'u başlatır.

**Parametreler:**
- `build_optimizer` (bool): Optimizer oluştur (default: True)
- `build_criterion` (bool): Criterion oluştur (default: True)
- `build_scheduler` (bool): Scheduler oluştur (default: True)
- `reset` (bool): Önceki state'i sıfırla (default: False)

**Örnek:**
```python
manager.initialize(
    build_optimizer=True,
    build_criterion=True,
    build_scheduler=True,
)
```

---

#### `build_model() -> nn.Module`

Modeli oluşturur (V-2/V-3/V-4 support).

**Dönüş:**
- `nn.Module`: Oluşturulan model

**Özellikler:**
- ✅ Otomatik parametre filtreleme (constructor signature-based)
- ✅ V-2/V-3/V-4 parametreleri otomatik geçirilir
- ✅ Device'a otomatik taşınır

**Örnek:**
```python
model = manager.build_model()
```

---

#### `forward(inputs, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Forward pass yapar.

**Parametreler:**
- `inputs` (torch.Tensor): Input tensor [B, T]
- `inference` (Optional[bool]): Inference mode (None ise model.training kullanılır)
- `return_aux` (bool): Auxiliary info döndür (default: True)
- `mask` (Optional[torch.Tensor]): Attention mask
- `causal_mask` (Optional[bool]): Causal mask override
- `use_cache` (bool): KV Cache kullan (V-4, default: False)
- `cache_position` (Optional[torch.Tensor]): Cache position (V-4)

**Dönüş:**
- `Tuple[torch.Tensor, Optional[torch.Tensor]]`: (logits, aux_info)

**Örnek:**
```python
# Training
logits, aux = manager.forward(inputs, inference=False)

# Inference with KV Cache
logits, aux = manager.forward(
    inputs,
    inference=True,
    use_cache=True,
    cache_position=torch.arange(seq_len),
)
```

---

#### `predict(inputs, **kwargs) -> Dict[str, Any]`

Prediction yapar (top-k, softmax).

**Parametreler:**
- `inputs` (torch.Tensor): Input tensor
- `topk` (int): Top-k prediction (default: 1)
- `apply_softmax` (bool): Softmax uygula (default: True)
- `return_logits` (bool): Logits döndür (default: False)

**Dönüş:**
- `Dict[str, Any]`: `{"probs": ..., "topk_values": ..., "topk_indices": ..., "logits": ...}`

**Örnek:**
```python
result = manager.predict(inputs, topk=5, apply_softmax=True)
print(result["topk_indices"])  # Top-5 token IDs
```

---

#### `save(save_path=None, **kwargs) -> str`

Model checkpoint'ini kaydeder.

**Parametreler:**
- `save_path` (Optional[str]): Kayıt yolu (None ise default)
- `epoch` (Optional[int]): Epoch numarası
- `additional_info` (Optional[Dict[str, Any]]): Ek bilgiler

**Dönüş:**
- `str`: Kaydedilen dosyanın yolu

**Örnek:**
```python
save_path = manager.save(
    save_path="saved_models/checkpoint_epoch_5.pth",
    epoch=5,
    additional_info={"loss": 2.1, "accuracy": 0.36},
)
```

---

#### `load(load_path=None, **kwargs) -> None`

Model checkpoint'ini yükler.

**Parametreler:**
- `load_path` (Optional[str]): Yükleme yolu (None ise default)
- `strict` (bool): Strict loading (default: True)
- `map_location` (Optional[Union[str, torch.device]]): Device mapping
- `weights_only` (Optional[bool]): Sadece weights (PyTorch 2.x)

**Örnek:**
```python
manager.load(
    load_path="saved_models/checkpoint_epoch_5.pth",
    strict=True,
)
```

---

#### `update(update_params, **kwargs) -> Dict[str, List[str]]`

Model, optimizer, scheduler'ı günceller.

**Parametreler:**
- `update_params` (Dict[str, Any]): Update parametreleri
  ```python
  {
      "model": {
          "freeze": ["encoder.*"],  # Pattern-based freeze
          "unfreeze": ["head.*"],
          "setattr": {"dropout_p": 0.2},
          "device": "cuda:0",
      },
      "optimizer": {
          "learning_rate": 1e-5,
          "weight_decay": 0.01,
      },
      "scheduler": {
          "factor": 0.5,
          "patience": 5,
      },
  }
  ```
- `dry_run` (bool): Sadece preview (default: False)

**Dönüş:**
- `Dict[str, List[str]]`: Update raporu

**Örnek:**
```python
report = manager.update({
    "model": {"freeze": ["dil_katmani.*"]},
    "optimizer": {"learning_rate": 5e-5},
})
```

---

#### `freeze(patterns) -> Dict[str, List[str]]`

Parametreleri dondurur (pattern-based).

**Parametreler:**
- `patterns` (Union[str, List[str]]): Pattern listesi (glob veya regex)

**Örnek:**
```python
# Glob pattern
manager.freeze("encoder.*")

# Regex pattern
manager.freeze("r:^backbone\\.layers\\.(0|1)\\.")

# Multiple patterns
manager.freeze(["dil_katmani.*", "encoder.layers.0.*"])
```

---

#### `unfreeze(patterns) -> Dict[str, List[str]]`

Parametreleri çözer (pattern-based).

**Parametreler:**
- `patterns` (Union[str, List[str]]): Pattern listesi

**Örnek:**
```python
manager.unfreeze("head.*")
```

---

#### `configure_tensorboard(**kwargs) -> None`

TensorBoard'u yapılandırır.

**Parametreler:**
- `writer` (Optional[Any]): External SummaryWriter
- `log_dir` (Optional[str]): Log dizini
- `log_every_n` (Optional[int]): Her N forward'ta log
- `log_histograms` (Optional[bool]): Histogram logging
- `log_attention_image` (Optional[bool]): Attention image logging
- `enable` (Optional[bool]): TensorBoard'u aktifleştir

**Örnek:**
```python
manager.configure_tensorboard(
    log_dir="runs/experiment_1",
    log_every_n=10,
    log_histograms=True,
    enable=True,
)
```

---

#### `train_mode() -> None`

Model'i training mode'a alır.

**Örnek:**
```python
manager.train_mode()
```

---

#### `eval_mode() -> None`

Model'i eval mode'a alır.

**Örnek:**
```python
manager.eval_mode()
```

---

#### `get_tb_writer() -> Optional[Any]`

TensorBoard writer'ı döndürür.

**Örnek:**
```python
writer = manager.get_tb_writer()
if writer:
    writer.add_scalar("loss/train", loss_value, step)
```

---

### ModelInitializer API

#### `build_model(model_class, config, **kwargs) -> nn.Module`

Modeli oluşturur.

**Parametreler:**
- `model_class` (Type[nn.Module]): Model sınıfı
- `config` (Dict[str, Any]): Konfigürasyon
- `extra_kwargs` (Optional[Dict[str, Any]]): Ek parametreler
- `device` (Optional[torch.device]): Device override
- `compile_model` (Optional[bool]): Torch.compile kullan

**Örnek:**
```python
from src.neural_network import CevahirNeuralNetwork

config = {
    "vocab_size": 60000,
    "embed_dim": 1024,
    "num_heads": 16,
    "num_layers": 12,
    "pe_mode": "rope",  # V-4
    "use_rmsnorm": True,  # V-4
}

model = ModelInitializer.build_model(
    CevahirNeuralNetwork,
    config,
    device=torch.device("cuda"),
)
```

---

#### `initialize_optimizer(model, config) -> optim.Optimizer`

Optimizer oluşturur.

**Desteklenen Optimizer'lar:**
- `adamw` (default)
- `adam`
- `sgd`
- `radam`
- `rmsprop`

**Parametreler:**
- `learning_rate` (float): Learning rate (default: 1e-3)
- `weight_decay` (float): Weight decay (default: 0.0)
- `betas` (tuple): Adam betas (default: (0.9, 0.999))
- `eps` (float): Epsilon (default: 1e-8)
- `no_weight_decay_keywords` (List[str]): Weight decay exclusion (default: ["bias", "layernorm", ...])

**Örnek:**
```python
optimizer = ModelInitializer.initialize_optimizer(model, {
    "optimizer": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
})
```

---

#### `initialize_criterion(config) -> nn.Module`

Loss function oluşturur.

**Desteklenen Loss Function'lar:**
- `cross_entropy` (default)
- `bce_with_logits`
- `mse`
- `smooth_l1`

**Parametreler:**
- `criterion` (str): Loss function tipi
- `label_smoothing` (float): Label smoothing (default: 0.0)
- `ignore_index` (int): Ignore index (default: -100)

**Örnek:**
```python
criterion = ModelInitializer.initialize_criterion({
    "criterion": "cross_entropy",
    "label_smoothing": 0.1,
})
```

---

#### `initialize_scheduler(optimizer, config) -> Optional[LRScheduler]`

Learning rate scheduler oluşturur.

**Desteklenen Scheduler'lar:**
- `reduce_on_plateau` (default)
- `cosine`
- `cosine_warm_restarts`
- `step`
- `exponential`
- `onecycle`

**Örnek:**
```python
scheduler = ModelInitializer.initialize_scheduler(optimizer, {
    "scheduler_type": "reduce_on_plateau",
    "lr_decay_factor": 0.5,
    "lr_decay_patience": 5,
})
```

---

### ModelSaver API

#### `save_checkpoint(**kwargs) -> str`

Tam checkpoint kaydeder.

**Parametreler:**
- `model` (nn.Module): Model
- `optimizer` (Optional[Optimizer]): Optimizer
- `scheduler` (Optional[LRScheduler]): Scheduler
- `epoch` (Optional[int]): Epoch
- `config` (Optional[Dict]): Config
- `metadata` (Optional[Dict]): Metadata
- `save_dir` (str): Save dizini
- `filename` (Optional[str]): Dosya adı
- `filename_template` (str): Template (default: "checkpoint_ep{epoch:04d}.pth")
- `create_latest_marker` (bool): Latest marker oluştur (default: True)
- `keep_last_n` (int): Son N checkpoint'i tut (default: 0)

**Örnek:**
```python
save_path = ModelSaver.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=5,
    config=config,
    save_dir="saved_models",
    filename_template="checkpoint_ep{epoch:04d}.pth",
    keep_last_n=3,
)
```

---

#### `save_weights_only(model, **kwargs) -> str`

Sadece model weights'lerini kaydeder.

**Örnek:**
```python
weights_path = ModelSaver.save_weights_only(
    model=model,
    save_dir="saved_models",
    filename="weights.pth",
)
```

---

### ModelLoader API

#### `load_model(model_class, model_path, **kwargs) -> nn.Module`

Modeli yükler.

**Parametreler:**
- `model_class` (Type[nn.Module]): Model sınıfı
- `model_path` (str): Model dosya yolu
- `device` (Optional[Union[str, torch.device]]): Device
- `config` (Optional[Dict]): Config (model reconstruction için)
- `strict` (bool): Strict loading (default: True)
- `weights_only` (Optional[bool]): Weights-only loading

**Örnek:**
```python
model = ModelLoader.load_model(
    CevahirNeuralNetwork,
    "saved_models/checkpoint.pth",
    device="cuda",
    config=config,
)
```

---

#### `load_all(model_class, ckpt_path, **kwargs) -> Tuple`

Tüm checkpoint'i yükler (model + optimizer + scheduler + meta).

**Dönüş:**
- `Tuple[nn.Module, Optional[Dict], Optional[Dict], Dict]`: (model, optimizer_state, scheduler_state, meta)

**Örnek:**
```python
model, opt_state, sch_state, meta = ModelLoader.load_all(
    CevahirNeuralNetwork,
    "saved_models/checkpoint.pth",
    config=config,
)
```

---

### ModelUpdater API

#### `bulk_update(**kwargs) -> UpdateReport`

Toplu güncelleme yapar.

**Parametreler:**
- `model` (Optional[nn.Module]): Model
- `optimizer` (Optional[Optimizer]): Optimizer
- `scheduler` (Optional[LRScheduler]): Scheduler
- `update_params` (Dict[str, Any]): Update parametreleri
- `dry_run` (bool): Dry-run mode (default: False)
- `filter_frozen_params` (bool): Frozen params filtrele (default: True)

**Örnek:**
```python
report = ModelUpdater.bulk_update(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    update_params={
        "model": {"freeze": ["encoder.*"]},
        "optimizer": {"learning_rate": 5e-5},
        "scheduler": {"factor": 0.5},
    },
)
```

---

## 💻 Kullanım Örnekleri

### Örnek 1: ModelManager ile Training Setup

```python
from model_management import ModelManager
import torch

# Config
config = {
    "vocab_size": 60000,
    "embed_dim": 1024,
    "seq_proj_dim": 1024,
    "num_heads": 16,
    "num_layers": 12,
    "ffn_dim": None,  # Auto: 4x seq_proj_dim
    "pre_norm": True,
    "causal_mask": True,
    # V-4 Features
    "pe_mode": "rope",
    "use_rmsnorm": True,
    "use_swiglu": True,
    "use_kv_cache": True,
    # Training
    "learning_rate": 1e-4,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "criterion": "cross_entropy",
    "scheduler_type": "reduce_on_plateau",
    "device": "cuda",
}

# Initialize
manager = ModelManager(config)
manager.initialize()

# Training loop
manager.train_mode()
for epoch in range(10):
    for batch in dataloader:
        inputs, targets = batch
        
        # Forward
        logits, _ = manager.forward(inputs, inference=False)
        
        # Loss
        loss = manager.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward
        loss.backward()
        manager.optimizer.step()
        manager.optimizer.zero_grad()
    
    # Save checkpoint
    manager.save(f"saved_models/checkpoint_epoch_{epoch}.pth", epoch=epoch)
```

---

### Örnek 2: Inference with KV Cache

```python
# Initialize for inference
manager = ModelManager(config)
manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
manager.load("saved_models/checkpoint_epoch_5.pth")
manager.eval_mode()

# First forward: Full sequence
input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
logits, _ = manager.forward(
    input_ids,
    inference=True,
    use_cache=True,
    cache_position=torch.arange(5),
)

# Next forward: Single token (autoregressive)
new_token = torch.tensor([[6]], dtype=torch.long)
logits, _ = manager.forward(
    new_token,
    inference=True,
    use_cache=True,
    cache_position=torch.tensor([5]),
)
```

---

### Örnek 3: Pattern-based Freeze/Unfreeze

```python
# Freeze encoder layers
manager.freeze("dil_katmani.*")  # Freeze language layer
manager.freeze("layers.0.*")     # Freeze first layer
manager.freeze("layers.1.*")     # Freeze second layer

# Unfreeze head
manager.unfreeze("output_layer.*")

# Advanced: Regex patterns
manager.freeze("r:^layers\\.(0|1|2)\\..*")  # Freeze layers 0, 1, 2
```

---

### Örnek 4: Learning Rate Scheduling

```python
# Update learning rate
manager.update({
    "optimizer": {
        "learning_rate": 5e-5,  # Reduce LR
    },
})

# Update scheduler
manager.update({
    "scheduler": {
        "factor": 0.5,      # Reduce LR by 50%
        "patience": 5,      # Wait 5 epochs
        "threshold": 1e-4,  # Minimum improvement
    },
})
```

---

### Örnek 5: Checkpoint Management

```python
# Save checkpoint
save_path = manager.save(
    save_path="saved_models/checkpoint_epoch_10.pth",
    epoch=10,
    additional_info={
        "loss": 2.1,
        "accuracy": 0.36,
        "perplexity": 10.5,
    },
)

# Load checkpoint
manager.load(
    load_path="saved_models/checkpoint_epoch_10.pth",
    strict=True,
)

# Get epoch from config
epoch = manager.config.get("current_epoch", 0)
print(f"Resumed from epoch {epoch}")
```

---

### Örnek 6: TensorBoard Integration

```python
# Configure TensorBoard
manager.configure_tensorboard(
    log_dir="runs/experiment_1",
    log_every_n=10,
    log_histograms=True,
    log_attention_image=True,
    enable=True,
)

# Training loop (TensorBoard automatically logs)
for step, batch in enumerate(dataloader):
    logits, _ = manager.forward(batch["inputs"])
    # TensorBoard automatically logs:
    # - Model weights (histograms)
    # - Attention maps (images)
    # - Forward pass stats
```

---

## 🔧 V-4 Architecture Support

ModelManager, V-4 Architecture özelliklerini otomatik olarak destekler:

### V-4 Parametreleri

```python
config = {
    # V-4 Features
    "pe_mode": "rope",                    # RoPE (Rotary Position Embedding)
    "use_rmsnorm": True,                  # RMSNorm (Root Mean Square Normalization)
    "use_swiglu": True,                   # SwiGLU (Swish-Gated Linear Unit)
    "use_kv_cache": True,                 # KV Cache (Key-Value Cache)
    "max_cache_len": 2048,                # Maximum cache length
    "use_advanced_checkpointing": False,  # Advanced checkpointing
    "quantization_type": "none",          # Quantization ("none" | "int8" | "fp16")
    "use_moe": False,                     # MoE (Mixture of Experts)
    "num_experts": 8,                     # Number of experts
    "moe_top_k": 2,                       # Top-k experts
}
```

### Otomatik Parametre Geçişi

ModelInitializer, config'teki tüm V-4 parametrelerini otomatik olarak model constructor'ına geçirir:

```python
# ModelInitializer otomatik olarak:
# 1. Model constructor signature'ını okur
# 2. Config'teki parametreleri filtreler
# 3. Uygun parametreleri model'e geçirir
# 4. V-4 özellikleri otomatik aktif hale gelir
```

---

## 📊 Performans ve Best Practices

### 1. Device Management

```python
# GPU kullanımı (otomatik)
config = {"device": "cuda"}  # Otomatik GPU seçilir

# CPU kullanımı (test için)
config = {"device": "cpu"}

# MPS (Apple Silicon)
config = {"device": "mps"}
```

### 2. Memory Optimization

```python
# Gradient checkpointing
config = {"use_gradient_checkpointing": True}

# Advanced checkpointing
config = {
    "use_advanced_checkpointing": True,
    "checkpointing_strategy": "selective",
}

# KV Cache (inference)
config = {
    "use_kv_cache": True,
    "max_cache_len": 2048,
}
```

### 3. Training Optimization

```python
# Weight decay parameter groups
config = {
    "weight_decay": 0.01,
    "no_weight_decay_keywords": ["bias", "layernorm", "norm"],
}

# Learning rate scheduling
config = {
    "scheduler_type": "reduce_on_plateau",
    "lr_decay_factor": 0.5,
    "lr_decay_patience": 5,
}
```

---

## 🔗 İlişkiler ve Entegrasyonlar

### ModelManager ↔ CevahirNeuralNetwork

**İlişki:** Composition (Has-A)

```python
ModelManager
    │
    └── HAS-A CevahirNeuralNetwork (self.model)
            │
            ├── FORWARD → ModelManager.forward()
            ├── SAVE → ModelManager.save()
            └── LOAD → ModelManager.load()
```

### ModelManager ↔ Training System

**İlişki:** Used-By

```python
TrainingService
    │
    └── USES ModelManager
            │
            ├── initialize() → Setup
            ├── forward() → Training loop
            ├── save() → Checkpoint saving
            └── load() → Checkpoint loading
```

### ModelManager ↔ Cevahir API

**İlişki:** Used-By (Adapter Pattern)

```python
Cevahir (Unified API)
    │
    └── USES ModelManager (via CevahirModelAPI adapter)
            │
            ├── forward() → Inference
            ├── generate() → Text generation
            └── load() → Model loading
```

---

## 📖 Daha Fazla Bilgi

- **[Neural Network Dokümantasyonu](../neural_network/README.md)**
- **[API Referansı](../../API_REFERENCE.md)**
- **[Sistem Mimarisi](../../ARCHITECTURE.md)**

---

**Son Güncelleme:** 2025-01-27  
**Versiyon:** V-5

