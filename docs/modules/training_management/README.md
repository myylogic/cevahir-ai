# 🎓 Training Management - Kapsamlı Dokümantasyon

**Versiyon:** V-5 (Advanced)  
**Son Güncelleme:** 2025-01-27  
**Durum:** ✅ Production-Ready | Endüstri Standartları

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Çalışma Prensibi](#çalışma-prensibi)
4. [Alt Modüller](#alt-modüller)
5. [API Referansı](#api-referansı)
6. [Training Loop Detayları](#training-loop-detayları)
7. [Validation Loop Detayları](#validation-loop-detayları)
8. [Kullanım Örnekleri](#kullanım-örnekleri)
9. [Entegrasyonlar](#entegrasyonlar)
10. [Best Practices](#best-practices)

---

## 🎯 Genel Bakış

**Training Management**, Cevahir Sinir Sistemi'nin eğitim ve doğrulama süreçlerini yöneten kapsamlı bir modüldür. Endüstri standartlarında özellikler sunar:

### Temel Özellikler

- ✅ **Advanced Training Loop:** AMP, gradient accumulation, gradient clipping
- ✅ **Comprehensive Validation:** Advanced metrics, memory-efficient validation
- ✅ **Checkpoint Management:** Atomic saves, best/last model tracking, rotation
- ✅ **Learning Rate Scheduling:** Multiple schedulers, warmup support
- ✅ **Logging & Monitoring:** File logs, TensorBoard, JSONL events
- ✅ **Performance Tracking:** Memory usage, batch times, throughput
- ✅ **Visualization:** Loss/accuracy plots, custom metrics, CSV/JSON export
- ✅ **Evaluation Metrics:** Precision, Recall, F1, Top-K accuracy, Confusion Matrix
- ✅ **V-4 Feature Validation:** Automatic validation of RoPE, RMSNorm, SwiGLU, etc.
- ✅ **Production-Ready:** Error handling, NaN/Inf detection, progress bars

### Modül Bileşenleri

1. **TrainingManager** - Ana training/validation orchestrator
2. **TrainingLogger** - Comprehensive logging system
3. **TrainingScheduler** - Learning rate scheduling
4. **CheckpointManager** - Model checkpoint management
5. **EvaluationMetrics** - Performance metrics calculation
6. **TrainingVisualizer** - Training visualization

---

## 🏗️ Mimari Yapı

### Dosya Organizasyonu

```
training_management/
├── __init__.py
├── training_manager.py        # Ana training orchestrator (~1549 satır)
├── training_logger.py          # Logging sistemi (~445 satır)
├── training_scheduler.py       # LR scheduling (~361 satır)
├── checkpoint_manager.py       # Checkpoint yönetimi (~464 satır)
├── evaluation_metrics.py       # Metrik hesaplama (~518 satır)
├── training_visualizer.py      # Görselleştirme (~421 satır)
└── test/
    ├── test_training_manager.py
    ├── test_training_manager_comprehensive.py
    └── test_checkpoint_manager.py
```

### Mimari Katmanlar

```
┌─────────────────────────────────────────────────────────┐
│              TrainingService (training_system)          │
│            (Orchestrates TrainingManager)               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   TrainingManager       │
        │   (Main Orchestrator)   │
        └────────────┬────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐  ┌────────▼────────┐  ┌───▼──────────┐
│Logger  │  │   Scheduler     │  │  Checkpoint  │
│        │  │                 │  │  Manager     │
└────────┘  └─────────────────┘  └──────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐  ┌────────▼────────┐  ┌───▼──────────┐
│Metrics │  │  Visualizer     │  │   TensorBoard│
│        │  │                 │  │              │
└────────┘  └─────────────────┘  └──────────────┘
```

---

## ⚙️ Çalışma Prensibi

### 1. Training Flow

```python
TrainingManager.train()
    ↓
1. Initialization & Validation
    ├── V-4 Feature Validation
    ├── Component Initialization
    └── Config Validation
    ↓
2. Training Loop (for each epoch):
    ├── _train_epoch()
    │   ├── Forward pass (AMP)
    │   ├── Loss calculation (PAD masked)
    │   ├── Backward pass (gradient accumulation)
    │   ├── Gradient clipping
    │   ├── Optimizer step
    │   ├── TensorBoard logging
    │   ├── Progress bar update
    │   └── Memory/Performance tracking
    ├── _validate_epoch()
    │   ├── Forward pass (no_grad)
    │   ├── Loss/Accuracy calculation
    │   ├── Advanced metrics (Precision, Recall, F1)
    │   ├── TensorBoard logging
    │   └── Progress bar update
    ├── Scheduler step (LR update)
    ├── Checkpoint save (best/last)
    ├── Early stopping check
    └── Visualization save
    ↓
3. Cleanup & Finalization
    ├── Training history save
    ├── Visualizations export
    └── TensorBoard close
```

### 2. Validation Flow

```python
_validate_epoch()
    ↓
1. Model.eval()
    ↓
2. For each batch:
    ├── Forward pass (no_grad)
    ├── Loss/Accuracy calculation
    ├── Predictions/Targets collection (for metrics)
    ├── Memory cleanup (every 50 batches)
    └── Progress logging
    ↓
3. Advanced Metrics Calculation:
    ├── Precision, Recall, F1
    ├── Top-K Accuracy
    └── TensorBoard logging
    ↓
4. Return (avg_loss, accuracy)
```

### 3. Checkpoint Flow

```python
save_model(epoch, val_loss, is_best)
    ↓
1. CheckpointManager.save()
    ├── Create payload (model, optimizer, history, metadata)
    ├── Atomic save (tmp → replace)
    ├── Update index.json
    ├── Update last.pth alias
    └── Update best.pth alias (if is_best)
    ↓
2. Rotation (if max_checkpoints exceeded)
    └── Delete oldest/worst checkpoints
```

---

## 📦 Alt Modüller

### 1. TrainingManager

**Dosya:** `training_management/training_manager.py`  
**Satır Sayısı:** ~1549  
**Görev:** Ana training/validation orchestrator

#### Ana Özellikler

- **Training Loop:**
  - AMP (Mixed Precision Training)
  - Gradient accumulation
  - Gradient clipping
  - PAD-masked loss/accuracy
  - Progress bars (tqdm)
  - Performance tracking

- **Validation Loop:**
  - Advanced metrics (Precision, Recall, F1, Top-K)
  - Memory-efficient processing
  - Colab-optimized progress logging
  - Error recovery (continue on batch errors)

- **V-4 Feature Validation:**
  - Automatic validation of RoPE, RMSNorm, SwiGLU
  - Gradient checkpointing check
  - KV Cache check
  - Weight tying check

- **Monitoring:**
  - Memory tracking (GPU)
  - Performance tracking (batch times, throughput)
  - NaN/Inf detection
  - Weight update verification

#### Ana Metodlar

```python
class TrainingManager:
    def __init__(model, train_loader, val_loader, optimizer, criterion, config, ...)
    def train(epoch_callback=None) -> Tuple[float, float]
    def _train_epoch() -> Tuple[float, float]
    def _validate_epoch() -> Tuple[float, float]
    def save_model(epoch, val_loss=None, is_best=None)
    def _compute_masked_loss_and_acc(logits, targets, pad_id) -> Tuple[Tensor, float, float]
    def _track_memory_usage(epoch=None) -> Dict[str, float]
    def _track_performance(epoch=None) -> Dict[str, float]
    def _detect_nan_inf(loss, logits=None) -> bool
```

---

### 2. TrainingLogger

**Dosya:** `training_management/training_logger.py`  
**Satır Sayısı:** ~445  
**Görev:** Comprehensive logging system

#### Özellikler

- **File Logging:**
  - Rotating file handlers (5MB max, 5 backups)
  - Separate error log file
  - UTF-8 encoding support
  - Windows/Unix compatible

- **TensorBoard Integration:**
  - Scalar logging (loss, accuracy, LR, etc.)
  - Histogram logging (weights, gradients)
  - Figure logging (attention maps)
  - Text logging (epoch summaries)
  - HParams logging

- **JSONL Event Log:**
  - Structured event logging
  - Machine-readable format
  - Real-time append

- **Console Logging:**
  - Optional console output
  - Formatted messages

#### Ana Metodlar

```python
class TrainingLogger:
    def __init__(run_name=None, log_dir=None, tb_log_dir=None, ...)
    def start_tb(tb_log_dir=None, run_name=None)
    def log_metrics(epoch, training_loss, validation_loss=None, accuracy=None, ...)
    def log_validation_metrics(epoch, validation_loss, validation_accuracy=None, ...)
    def log_scalar(name, value, step)
    def log_histogram(name, values, step)
    def log_figure(name, figure, step)
    def log_text(name, text, step)
    def log_event(event: Dict[str, Any])
    def close()
```

---

### 3. TrainingScheduler

**Dosya:** `training_management/training_scheduler.py`  
**Satır Sayısı:** ~361  
**Görev:** Learning rate scheduling

#### Desteklenen Scheduler Türleri

1. **ReduceLROnPlateau** - Metric-based (default)
2. **StepLR** - Step-based
3. **ExponentialLR** - Exponential decay
4. **CosineAnnealingLR** - Cosine annealing
5. **CosineAnnealingWarmRestarts** - Cosine with warm restarts
6. **OneCycleLR** - One cycle policy
7. **NoOp** - Constant LR

#### Özellikler

- **Linear Warmup:**
  - Configurable warmup steps
  - Start factor (default: 0.1)
  - Seamless integration with base schedulers

- **Gradient Gate:**
  - Skip LR update if gradient norm too low
  - Prevents unnecessary updates

- **Checkpoint Compatibility:**
  - `state_dict()` / `load_state_dict()` support
  - Resume training from checkpoint

#### Ana Metodlar

```python
class TrainingScheduler:
    def __init__(optimizer, scheduler_type="ReduceLROnPlateau", warmup_steps=0, ...)
    def step(metric=None, gradient_norm=None, gradient_gate=None)
    def get_last_lr() -> float
    def state_dict() -> Dict[str, Any]
    def load_state_dict(state: Dict[str, Any])
```

---

### 4. CheckpointManager

**Dosya:** `training_management/checkpoint_manager.py`  
**Satır Sayısı:** ~464  
**Görev:** Model checkpoint management

#### Özellikler

- **Atomic Saves:**
  - tmp file → fsync → os.replace
  - Prevents corruption
  - Windows/Unix compatible

- **Alias Management:**
  - `last.pth` - Latest checkpoint
  - `best.pth` - Best metric checkpoint
  - Automatic updates

- **Checkpoint Rotation:**
  - Top-K rotation (configurable)
  - Sort by ctime or metric
  - Automatic cleanup

- **Index Management:**
  - `index.json` - Metadata database
  - Tracks all checkpoints
  - Best metric tracking

- **Flexible Loading:**
  - Load by filename
  - Load by alias (last/best)
  - Resume support (epoch + 1)

#### Ana Metodlar

```python
class CheckpointManager:
    def __init__(checkpoint_model_dir, max_checkpoints=5, device="cuda", ...)
    def save(model, optimizer, epoch, training_history=None, metric=None, is_best=None, ...) -> str
    def load(model, optimizer=None, filename=None, which="path", ...) -> Tuple[int, Dict]
    def resume(model, optimizer=None, which="last", ...) -> Tuple[int, Dict]
    def list_checkpoints() -> List[str]
    def get_last_checkpoint() -> Optional[str]
    def get_best_checkpoint() -> Optional[str]
```

---

### 5. EvaluationMetrics

**Dosya:** `training_management/evaluation_metrics.py`  
**Satır Sayısı:** ~518  
**Görev:** Performance metrics calculation

#### Özellikler

- **Flexible Input Support:**
  - Sequence classification: `[N, T, C]` logits
  - Flat classification: `[N, C]` logits
  - Class indices: `[N]` or `[N, T]`
  - Automatic shape alignment

- **Metrics:**
  - **Accuracy:** Token-level accuracy (PAD masked)
  - **Top-K Accuracy:** Top-K token accuracy
  - **Precision/Recall/F1:** Macro, Micro, Weighted
  - **Confusion Matrix:** Per-class metrics

- **PAD Masking:**
  - `ignore_index` support
  - Automatic masking in calculations

#### Ana Metodlar

```python
class EvaluationMetrics:
    def calculate_accuracy(predictions, targets, ignore_index=None, from_logits=True, ...) -> float
    def accuracy_topk(predictions, targets, k=5, ignore_index=None, ...) -> float
    def calculate_precision_recall_f1(predictions, targets, average="macro", ...) -> Dict[str, float]
    def confusion_matrix(predictions, targets, num_classes=None, ...) -> np.ndarray
    def calculate_metrics(predictions, targets, top_k=5, ...) -> Dict[str, float]
```

---

### 6. TrainingVisualizer

**Dosya:** `training_management/training_visualizer.py`  
**Satır Sayısı:** ~421  
**Görev:** Training visualization

#### Özellikler

- **Plot Types:**
  - Loss plots (train/val)
  - Accuracy plots (train/val)
  - Custom metric plots
  - Automatic plot generation from history

- **Export Formats:**
  - PNG plots (150 DPI)
  - CSV export
  - JSON export (with merge support)

- **Advanced Features:**
  - EMA smoothing (optional)
  - Epoch axis support
  - Series alignment (auto-trim)
  - Headless support (Agg backend)

#### Ana Metodlar

```python
class TrainingVisualizer:
    def __init__(save_dir="visualizations", style="default", run_name=None, ...)
    def plot_loss(train_losses, val_losses=None, epochs=None, ema_alpha=None, ...) -> str
    def plot_accuracy(train_accuracies, val_accuracies=None, epochs=None, ...) -> str
    def plot_custom_metric(metric_values, metric_name, epochs=None, ...) -> str
    def plot_from_history(history, save_prefix="", ema_alpha=None, ...) -> Dict[str, str]
    def export_history_csv(history, filename="metrics.csv") -> str
    def export_history_json(history, filename="metrics.json", merge_if_exists=True) -> str
```

---

## 📚 API Referansı

### TrainingManager

#### `__init__(model, train_loader, val_loader, optimizer, criterion, config, start_epoch=1, writer=None)`

TrainingManager'ı başlatır.

**Parametreler:**
- `model: torch.nn.Module` → Neural network model
- `train_loader` → Training data loader
- `val_loader` → Validation data loader
- `optimizer` → PyTorch optimizer
- `criterion` → Loss function
- `config: Dict[str, Any]` → Training configuration
- `start_epoch: int` → Starting epoch (default: 1)
- `writer: Optional[Any]` → TensorBoard writer (optional)

**Config Parametreleri:**

```python
config = {
    # Basic training
    "device": "cuda",  # or "cpu"
    "vocab_size": 50000,
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-4,
    
    # AMP & Gradient
    "use_amp": True,  # Mixed precision
    "grad_accum_steps": 1,  # Gradient accumulation
    "max_grad_norm": 1.0,  # Gradient clipping
    "pad_token_id": 0,  # Optional, auto-detected if None
    
    # TensorBoard
    "use_tensorboard": True,
    "tb_log_dir": "runs/training",
    "tb_log_every_n_batches": 20,
    "tb_log_histograms": False,
    "tb_log_attention_image": True,
    "tb_log_train_step": True,
    "tb_log_val_step": False,
    
    # Advanced features
    "calculate_advanced_metrics": True,  # Precision, Recall, F1
    "enable_visualizations": True,
    "visualization_dir": "visualizations",
    "track_memory": True,  # GPU memory tracking
    "track_performance": True,  # Batch time tracking
    "use_progress_bar": True,  # tqdm progress bar
    
    # Checkpoint
    "checkpoint_dir": "checkpoints",
    "training_history_path": "training_history.json",
    "early_stopping_patience": 3,
    
    # Logging
    "log_batches_to_console": True,
}
```

**Örnek:**
```python
config = {
    "device": "cuda",
    "vocab_size": 50000,
    "epochs": 30,
    "use_amp": True,
    "grad_accum_steps": 4,
    "max_grad_norm": 1.0,
    "use_tensorboard": True,
    "calculate_advanced_metrics": True,
    "track_memory": True,
    "track_performance": True,
}

manager = TrainingManager(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config
)
```

#### `train(epoch_callback=None) -> Tuple[float, float]`

Training ve validation döngüsünü çalıştırır.

**Parametreler:**
- `epoch_callback: Optional[Callable]` → Her epoch sonunda çağrılan callback `(epoch, train_loss, val_loss) -> None`

**Dönüş:**
- `Tuple[float, float]` → (final_train_loss, final_val_loss)

**Örnek:**
```python
def on_epoch_end(epoch, train_loss, val_loss):
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    # Model test, custom logging, etc.

train_loss, val_loss = manager.train(epoch_callback=on_epoch_end)
```

#### `save_model(epoch, val_loss=None, is_best=None)`

Model checkpoint kaydeder.

**Parametreler:**
- `epoch: int` → Epoch numarası
- `val_loss: Optional[float]` → Validation loss (best model seçimi için)
- `is_best: Optional[bool]` → En iyi model mi? (val_loss'tan otomatik hesaplanır)

**Örnek:**
```python
manager.save_model(epoch=5, val_loss=2.1, is_best=True)
```

---

### TrainingLogger

#### `__init__(run_name=None, log_dir=None, tb_log_dir=None, enable_tb=True, enable_console=True, enable_jsonl=True, ...)`

TrainingLogger'ı başlatır.

**Parametreler:**
- `run_name: Optional[str]` → Run name for TensorBoard
- `log_dir: Optional[str]` → Log directory path
- `tb_log_dir: Optional[str]` → TensorBoard log directory
- `enable_tb: bool` → Enable TensorBoard (default: True)
- `enable_console: bool` → Enable console logging (default: True)
- `enable_jsonl: bool` → Enable JSONL event logging (default: True)

**Örnek:**
```python
logger = TrainingLogger(
    run_name="experiment-001",
    log_dir="./logs",
    tb_log_dir="./runs",
    enable_tb=True
)
```

#### `log_metrics(epoch, training_loss, validation_loss=None, accuracy=None, step=None)`

Epoch metriklerini loglar.

**Örnek:**
```python
logger.log_metrics(
    epoch=1,
    training_loss=2.5,
    validation_loss=2.3,
    accuracy=85.5
)
```

#### `log_scalar(name, value, step)`

TensorBoard'a scalar loglar.

**Örnek:**
```python
logger.log_scalar("LearningRate", 1e-4, step=100)
```

---

### TrainingScheduler

#### `__init__(optimizer, scheduler_type="ReduceLROnPlateau", warmup_steps=0, warmup_start_factor=0.1, **kwargs)`

TrainingScheduler'ı başlatır.

**Parametreler:**
- `optimizer` → PyTorch optimizer
- `scheduler_type: str` → Scheduler type (see supported types)
- `warmup_steps: int` → Linear warmup steps (default: 0)
- `warmup_start_factor: float` → Warmup start factor (default: 0.1)
- `**kwargs` → Scheduler-specific parameters

**Örnek:**
```python
scheduler = TrainingScheduler(
    optimizer=optimizer,
    scheduler_type="ReduceLROnPlateau",
    warmup_steps=1000,
    warmup_start_factor=0.1,
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=5e-6
)
```

#### `step(metric=None, gradient_norm=None, gradient_gate=None)`

Learning rate'ı günceller.

**Parametreler:**
- `metric: Optional[float]` → Metric for ReduceLROnPlateau (e.g., val_loss)
- `gradient_norm: Optional[float]` → Gradient norm (optional)
- `gradient_gate: Optional[float]` → Skip LR update if gradient_norm < gradient_gate

**Örnek:**
```python
scheduler.step(metric=val_loss, gradient_norm=avg_grad_norm)
```

---

### CheckpointManager

#### `save(model, optimizer, epoch, training_history=None, metric=None, is_best=None, tag=None, extra_state=None, with_optimizer=True) -> str`

Checkpoint kaydeder.

**Parametreler:**
- `model: torch.nn.Module` → Model
- `optimizer: Optional[torch.optim.Optimizer]` → Optimizer
- `epoch: int` → Epoch number
- `training_history: Optional[Dict]` → Training history
- `metric: Optional[float]` → Metric value (for best model selection)
- `is_best: Optional[bool]` → Is best model? (auto-calculated from metric if None)
- `tag: Optional[str]` → Optional tag for filename
- `extra_state: Optional[Dict]` → Additional metadata
- `with_optimizer: bool` → Include optimizer state (default: True)

**Dönüş:**
- `str` → Saved checkpoint path

**Örnek:**
```python
checkpoint_path = checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=5,
    training_history={"train_loss": [2.5, 2.3, ...]},
    metric=2.1,
    is_best=True,
    extra_state={"config": config}
)
```

#### `load(model, optimizer=None, filename=None, which="path", map_location=None, load_optimizer=True, strict=True) -> Tuple[Optional[int], Dict[str, Any]]`

Checkpoint yükler.

**Parametreler:**
- `model: torch.nn.Module` → Model to load into
- `optimizer: Optional[torch.optim.Optimizer]` → Optimizer to load into
- `filename: Optional[str]` → Checkpoint filename (if which="path")
- `which: Literal["last", "best", "path"]` → Which checkpoint to load
- `map_location: Optional[str]` → Device mapping (default: config device)
- `load_optimizer: bool` → Load optimizer state (default: True)
- `strict: bool` → Strict loading (default: True)

**Dönüş:**
- `Tuple[Optional[int], Dict[str, Any]]` → (epoch, training_history)

**Örnek:**
```python
# Load last checkpoint
epoch, history = checkpoint_manager.load(
    model=model,
    optimizer=optimizer,
    which="last"
)

# Load best checkpoint
epoch, history = checkpoint_manager.load(
    model=model,
    optimizer=optimizer,
    which="best"
)

# Load specific checkpoint
epoch, history = checkpoint_manager.load(
    model=model,
    optimizer=optimizer,
    filename="checkpoint_epoch_0005.pth",
    which="path"
)
```

#### `resume(model, optimizer=None, which="last", map_location=None, load_optimizer=True) -> Tuple[int, Dict[str, Any]]`

Eğitime kaldığı yerden devam eder.

**Dönüş:**
- `Tuple[int, Dict[str, Any]]` → (start_epoch, training_history)
- **Not:** `start_epoch = loaded_epoch + 1`

**Örnek:**
```python
start_epoch, history = checkpoint_manager.resume(
    model=model,
    optimizer=optimizer,
    which="last"
)

# Continue training from start_epoch
manager.train(...)
```

---

### EvaluationMetrics

#### `calculate_accuracy(predictions, targets, ignore_index=None, from_logits=True, return_fraction=False) -> float`

Accuracy hesaplar.

**Örnek:**
```python
accuracy = metrics.calculate_accuracy(
    predictions=logits,  # [batch, seq_len, vocab_size]
    targets=targets,  # [batch, seq_len]
    ignore_index=0,  # PAD token ID
    from_logits=True,
    return_fraction=False  # False = percentage (0-100)
)
```

#### `accuracy_topk(predictions, targets, k=5, ignore_index=None, from_logits=True, return_fraction=False) -> float`

Top-K accuracy hesaplar.

**Örnek:**
```python
top5_acc = metrics.accuracy_topk(
    predictions=logits,
    targets=targets,
    k=5,
    ignore_index=0
)
```

#### `calculate_precision_recall_f1(predictions, targets, num_classes=None, average="macro", ignore_index=None, from_logits=True, return_per_class=False) -> Dict[str, float]`

Precision, Recall, F1 hesaplar.

**Örnek:**
```python
metrics_dict = metrics.calculate_precision_recall_f1(
    predictions=logits,
    targets=targets,
    average="macro",  # "macro" | "micro" | "weighted"
    ignore_index=0,
    from_logits=True,
    return_per_class=False
)
# Returns: {"precision": 85.5, "recall": 82.3, "f1_score": 83.9}
```

#### `calculate_metrics(predictions, targets, top_k=5, num_classes=None, average="macro", ignore_index=None, from_logits=None) -> Dict[str, float]`

Tüm metrikleri bir arada hesaplar.

**Örnek:**
```python
all_metrics = metrics.calculate_metrics(
    predictions=logits,
    targets=targets,
    top_k=5,
    average="macro",
    ignore_index=0
)
# Returns: {
#     "precision": 0.855,  # 0-1 range
#     "recall": 0.823,
#     "f1_score": 0.839,
#     "top_k_accuracy": 0.92
# }
```

---

### TrainingVisualizer

#### `plot_loss(train_losses, val_losses=None, epochs=None, ema_alpha=None, save_filename="loss_plot.png", show=False) -> str`

Loss grafiği çizer.

**Örnek:**
```python
plot_path = visualizer.plot_loss(
    train_losses=[2.5, 2.3, 2.1, ...],
    val_losses=[2.4, 2.2, 2.0, ...],
    epochs=[1, 2, 3, ...],
    ema_alpha=0.9,  # Exponential moving average smoothing
    save_filename="training_loss.png"
)
```

#### `plot_accuracy(train_accuracies, val_accuracies=None, epochs=None, ema_alpha=None, save_filename="accuracy_plot.png", show=False) -> str`

Accuracy grafiği çizer.

**Örnek:**
```python
plot_path = visualizer.plot_accuracy(
    train_accuracies=[0.75, 0.80, 0.85, ...],
    val_accuracies=[0.73, 0.78, 0.83, ...],
    epochs=[1, 2, 3, ...]
)
```

#### `plot_from_history(history, save_prefix="", ema_alpha=None, show=False) -> Dict[str, str]`

Training history'den otomatik grafikler oluşturur.

**Örnek:**
```python
history = {
    "train_loss": [2.5, 2.3, 2.1, ...],
    "val_loss": [2.4, 2.2, 2.0, ...],
    "accuracy": [0.75, 0.80, 0.85, ...]
}

paths = visualizer.plot_from_history(
    history=history,
    save_prefix="epoch_5",
    ema_alpha=0.9
)
# Returns: {"loss": "path/to/loss.png", "accuracy": "path/to/accuracy.png"}
```

#### `export_history_csv(history, filename="metrics.csv") -> str`

History'yi CSV olarak export eder.

**Örnek:**
```python
csv_path = visualizer.export_history_csv(
    history={
        "train_loss": [2.5, 2.3, ...],
        "val_loss": [2.4, 2.2, ...],
        "accuracy": [0.75, 0.80, ...]
    },
    filename="training_metrics.csv"
)
```

---

## 🔄 Training Loop Detayları

### Training Epoch Flow

```python
_train_epoch()
    ↓
1. Model.train()
2. Optimizer.zero_grad()
    ↓
3. For each batch (with progress bar):
    ├── Parse batch (inputs, targets)
    ├── Move to device
    ├── Forward pass (with AMP autocast):
    │   ├── model(inputs) → logits
    │   ├── Compute masked loss/accuracy/perplexity
    │   └── NaN/Inf detection
    ├── Backward pass (with gradient accumulation):
    │   ├── loss / grad_accum_steps
    │   ├── scaler.scale().backward() (if AMP)
    │   └── Accumulate gradients
    ├── Optimizer step (every grad_accum_steps):
    │   ├── Unscale gradients (if AMP)
    │   ├── Clip gradients
    │   ├── scaler.step(optimizer) (if AMP)
    │   ├── scaler.update() (if AMP)
    │   └── optimizer.zero_grad()
    ├── TensorBoard logging (every N batches)
    ├── Progress bar update
    ├── Memory tracking (every 10 batches)
    └── Performance tracking (batch time)
    ↓
4. Return (avg_loss, avg_accuracy)
```

### Loss Calculation

```python
_compute_masked_loss_and_acc(logits, targets, pad_id)
    ↓
1. Reshape logits: [B, T, V] → [B*T, V]
2. Reshape targets: [B, T] → [B*T]
3. Compute cross_entropy per token (reduction="none")
4. Create mask: targets != pad_id
5. Apply mask to loss
6. Average over non-padded tokens
7. Calculate accuracy (predictions == targets) & mask
8. Calculate perplexity: exp(min(20.0, loss))
    ↓
Return (loss_tensor, accuracy_float, perplexity_float)
```

---

## 🔍 Validation Loop Detayları

### Validation Epoch Flow

```python
_validate_epoch()
    ↓
1. Model.eval()
2. torch.no_grad()
    ↓
3. For each batch (with progress bar):
    ├── Parse batch (inputs, targets)
    ├── Move to device
    ├── Forward pass (no_grad):
    │   ├── model(inputs) → logits
    │   ├── Compute masked loss/accuracy/perplexity
    │   └── Collect predictions/targets (for metrics)
    ├── Progress logging (every %5)
    ├── Memory cleanup (every 50 batches)
    └── TensorBoard logging (if enabled)
    ↓
4. Advanced Metrics Calculation:
    ├── Concatenate predictions/targets
    ├── Calculate Precision, Recall, F1
    ├── Calculate Top-K Accuracy
    └── TensorBoard logging
    ↓
5. Return (avg_loss, accuracy)
```

### Advanced Metrics Calculation

```python
# Precision, Recall, F1
precision_recall_f1 = metrics.calculate_precision_recall_f1(
    predictions=all_predictions,  # [N, T, V] logits
    targets=all_targets,  # [N, T] class indices
    ignore_index=pad_token_id,
    average="macro",  # or "micro", "weighted"
    from_logits=True
)

# Top-K Accuracy
top_k_acc = metrics.accuracy_topk(
    predictions=all_predictions,
    targets=all_targets,
    k=5,
    ignore_index=pad_token_id,
    from_logits=True
)
```

---

## 💡 Kullanım Örnekleri

### Örnek 1: Basic Training

```python
from training_management import TrainingManager
import torch
import torch.nn as nn

# Model, optimizer, criterion setup
model = YourModel(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Config
config = {
    "device": "cuda",
    "vocab_size": 50000,
    "epochs": 30,
    "use_amp": True,
    "grad_accum_steps": 4,
    "max_grad_norm": 1.0,
    "use_tensorboard": True,
    "tb_log_dir": "runs/training",
    "calculate_advanced_metrics": True,
    "track_memory": True,
    "track_performance": True,
    "checkpoint_dir": "checkpoints",
}

# Initialize TrainingManager
manager = TrainingManager(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config
)

# Train
train_loss, val_loss = manager.train()
print(f"Final - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
```

### Örnek 2: Training with Custom Scheduler

```python
from training_management import TrainingManager, TrainingScheduler

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Custom scheduler with warmup
scheduler = TrainingScheduler(
    optimizer=optimizer,
    scheduler_type="CosineAnnealingLR",
    warmup_steps=1000,
    warmup_start_factor=0.1,
    T_max=10000,
    eta_min=1e-6
)

# TrainingManager will use scheduler automatically
# (scheduler is passed internally)
```

### Örnek 3: Resume Training

```python
from training_management import CheckpointManager, TrainingManager

# Checkpoint manager
checkpoint_manager = CheckpointManager(checkpoint_model_dir="checkpoints")

# Resume from last checkpoint
start_epoch, history = checkpoint_manager.resume(
    model=model,
    optimizer=optimizer,
    which="last"
)

# Continue training
config["start_epoch"] = start_epoch
manager = TrainingManager(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    start_epoch=start_epoch
)

manager.train()
```

### Örnek 4: Advanced Metrics

```python
from training_management import EvaluationMetrics

metrics = EvaluationMetrics()

# Calculate all metrics
all_metrics = metrics.calculate_metrics(
    predictions=logits,  # [batch, seq_len, vocab_size]
    targets=targets,  # [batch, seq_len]
    top_k=5,
    average="macro",
    ignore_index=0  # PAD token
)

print(f"Precision: {all_metrics['precision']:.2%}")
print(f"Recall: {all_metrics['recall']:.2%}")
print(f"F1: {all_metrics['f1_score']:.2%}")
print(f"Top-5 Accuracy: {all_metrics['top_k_accuracy']:.2%}")
```

### Örnek 5: Visualization

```python
from training_management import TrainingVisualizer

visualizer = TrainingVisualizer(
    save_dir="visualizations",
    run_name="experiment-001"
)

# Training history
history = {
    "train_loss": [2.5, 2.3, 2.1, 2.0, 1.9],
    "val_loss": [2.4, 2.2, 2.0, 1.9, 1.8],
    "accuracy": [0.75, 0.80, 0.85, 0.88, 0.90],
    "precision": [0.72, 0.78, 0.83, 0.86, 0.89],
    "recall": [0.73, 0.79, 0.84, 0.87, 0.90],
    "f1_score": [0.725, 0.785, 0.835, 0.865, 0.895]
}

# Generate all plots
paths = visualizer.plot_from_history(
    history=history,
    save_prefix="final",
    ema_alpha=0.9
)

# Export to CSV/JSON
csv_path = visualizer.export_history_csv(history, "metrics.csv")
json_path = visualizer.export_history_json(history, "metrics.json")
```

### Örnek 6: Custom Epoch Callback

```python
def custom_epoch_callback(epoch, train_loss, val_loss):
    print(f"\n=== Epoch {epoch} Summary ===")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    
    # Custom logic
    if val_loss < 2.0:
        print("🎉 Val loss below 2.0!")
    
    # Model testing
    # test_model(model, test_loader)
    
    # Custom checkpoint
    # torch.save(model.state_dict(), f"custom_ckpt_epoch_{epoch}.pth")

# Train with callback
train_loss, val_loss = manager.train(epoch_callback=custom_epoch_callback)
```

### Örnek 7: Checkpoint Management

```python
from training_management import CheckpointManager

# Initialize
checkpoint_manager = CheckpointManager(
    checkpoint_model_dir="checkpoints",
    max_checkpoints=5,  # Keep only 5 checkpoints
    device="cuda",
    sort_key="metric",  # Sort by metric value
    metric_mode="min"  # Lower is better
)

# Save checkpoint
checkpoint_path = checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=5,
    training_history={
        "train_loss": [2.5, 2.3, 2.1, 2.0, 1.9],
        "val_loss": [2.4, 2.2, 2.0, 1.9, 1.8]
    },
    metric=1.8,  # Validation loss
    is_best=True,  # Best model so far
    extra_state={
        "config": config,
        "timestamp": datetime.now().isoformat()
    }
)

# List all checkpoints
all_checkpoints = checkpoint_manager.list_checkpoints()
print(f"Saved checkpoints: {len(all_checkpoints)}")

# Get best/last checkpoint paths
best_path = checkpoint_manager.get_best_checkpoint()
last_path = checkpoint_manager.get_last_checkpoint()
```

---

## 🔗 Entegrasyonlar

### 1. TrainingService Entegrasyonu

**Dosya:** `training_system/training_service.py`

**Kullanım:**
- TrainingService, TrainingManager'ı kullanarak eğitimi yönetir
- ModelManager, TokenizerCore, DataLoaderManager entegrasyonu
- TensorBoard writer yönetimi
- BPE tokenizer entegrasyonu

**API:**
```python
# TrainingService içinde
training_manager = TrainingManager(
    model=model_manager.model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    writer=tb_writer
)

train_loss, val_loss = training_manager.train()
```

### 2. ModelManager Entegrasyonu

**Dosya:** `model_management/model_manager.py`

**Kullanım:**
- ModelManager'dan model, optimizer, criterion alınır
- TrainingManager model'i eğitir
- CheckpointManager model'i kaydeder/yükler

### 3. TensorBoard Entegrasyonu

**Kullanım:**
- Scalar logging (loss, accuracy, LR, etc.)
- Histogram logging (weights, gradients)
- Image logging (attention maps)
- Text logging (epoch summaries)
- HParams logging

**Örnek:**
```python
from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter(log_dir="runs/training")

manager = TrainingManager(
    ...,
    writer=tb_writer
)

# TensorBoard görüntüleme:
# tensorboard --logdir runs/training
```

---

## ✅ Best Practices

### 1. Configuration

```python
# ✅ DO: Comprehensive config
config = {
    "device": "cuda",
    "vocab_size": 50000,
    "epochs": 30,
    "use_amp": True,  # Enable AMP for GPU
    "grad_accum_steps": 4,  # Effective batch size = 32 * 4 = 128
    "max_grad_norm": 1.0,
    "use_tensorboard": True,
    "calculate_advanced_metrics": True,
    "track_memory": True,
    "track_performance": True,
    "checkpoint_dir": "checkpoints",
    "early_stopping_patience": 3,
}

# ❌ DON'T: Minimal config (missing important features)
config = {
    "device": "cuda",
    "vocab_size": 50000,
    "epochs": 10,
}
```

### 2. Checkpoint Management

```python
# ✅ DO: Use CheckpointManager
checkpoint_manager = CheckpointManager(
    checkpoint_model_dir="checkpoints",
    max_checkpoints=5,  # Automatic rotation
    sort_key="metric",  # Keep best models
    metric_mode="min"
)

# Save with metadata
checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metric=val_loss,
    is_best=(val_loss < best_val_loss),
    extra_state={"config": config}
)

# ❌ DON'T: Manual checkpoint saving
torch.save(model.state_dict(), "model.pth")  # No metadata, no rotation
```

### 3. Memory Management

```python
# ✅ DO: Enable memory tracking
config = {
    "track_memory": True,  # GPU memory tracking
    "track_performance": True,  # Batch time tracking
}

# Monitor memory usage
# TrainingManager automatically tracks and logs GPU memory

# ❌ DON'T: Ignore memory
# Memory issues can cause OOM errors
```

### 4. Error Handling

```python
# ✅ DO: Handle errors gracefully
try:
    train_loss, val_loss = manager.train()
except KeyboardInterrupt:
    print("Training interrupted, saving checkpoint...")
    manager.save_model(epoch=manager.current_epoch, val_loss=val_loss)
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    raise

# ❌ DON'T: Ignore errors
train_loss, val_loss = manager.train()  # No error handling
```

### 5. Validation Optimization

```python
# ✅ DO: Optimize for Colab/long validation
config = {
    "tb_log_val_step": False,  # Disable step-level validation logging (memory)
    "calculate_advanced_metrics": True,  # But enable advanced metrics
}

# Progress logging is automatic (every %5)

# ❌ DON'T: Enable all logging in Colab
config = {
    "tb_log_val_step": True,  # Can cause memory issues
    "tb_log_histograms": True,  # Very memory intensive
}
```

### 6. Learning Rate Scheduling

```python
# ✅ DO: Use warmup + scheduler
scheduler = TrainingScheduler(
    optimizer=optimizer,
    scheduler_type="ReduceLROnPlateau",
    warmup_steps=1000,  # Warmup for stable training
    warmup_start_factor=0.1,
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=5e-6
)

# ❌ DON'T: Constant learning rate
# LR should adapt during training
```

---

## 📊 Performance Optimizations

### 1. Mixed Precision Training (AMP)

**Avantajlar:**
- 2x faster training
- 50% memory reduction
- Minimal accuracy loss

**Kullanım:**
```python
config = {
    "use_amp": True,  # Automatically enabled on CUDA
}
```

### 2. Gradient Accumulation

**Avantajlar:**
- Simulate larger batch sizes
- Memory-efficient training
- Stable gradients

**Örnek:**
```python
# Effective batch size = batch_size * grad_accum_steps
config = {
    "batch_size": 32,
    "grad_accum_steps": 4,  # Effective batch size = 128
}
```

### 3. Progress Bars

**Avantajlar:**
- Real-time feedback
- ETA estimation
- Better UX

**Kullanım:**
```python
config = {
    "use_progress_bar": True,  # Requires tqdm
}
```

### 4. Memory Tracking

**Avantajlar:**
- Detect memory leaks
- Optimize batch size
- Monitor GPU usage

**Kullanım:**
```python
config = {
    "track_memory": True,  # GPU memory tracking
}
```

---

## 🔍 Troubleshooting

### Sorun 1: "Training stopped at batch X"

**Olası Nedenler:**
- NaN/Inf in loss or logits
- Memory overflow
- Invalid batch data

**Çözüm:**
```python
# Enable NaN/Inf detection (automatic)
config = {
    "use_amp": True,  # Can help with numerical stability
    "max_grad_norm": 1.0,  # Gradient clipping
}

# Check batch data
for batch in train_loader:
    inputs, targets = batch
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    print(f"Input range: [{inputs.min()}, {inputs.max()}]")
    break
```

### Sorun 2: "Validation takes too long in Colab"

**Çözüm:**
```python
config = {
    "tb_log_val_step": False,  # Disable step-level logging
    "calculate_advanced_metrics": True,  # But keep metrics (uses subset)
}

# Validation automatically:
# - Logs progress every %5
# - Clears GPU cache every 50 batches
# - Uses subset for advanced metrics (memory efficient)
```

### Sorun 3: "Checkpoint save failed"

**Çözüm:**
```python
# Use CheckpointManager (atomic saves)
checkpoint_manager = CheckpointManager(
    checkpoint_model_dir="checkpoints",
    max_checkpoints=5
)

# Atomic save prevents corruption
checkpoint_manager.save(...)
```

### Sorun 4: "Memory errors during training"

**Çözüm:**
```python
config = {
    "use_amp": True,  # 50% memory reduction
    "grad_accum_steps": 4,  # Smaller effective batch
    "tb_log_histograms": False,  # Disable memory-intensive logging
    "tb_log_val_step": False,
}

# Reduce batch size
train_loader = DataLoader(..., batch_size=16)  # Instead of 32
```

---

## 📚 İlgili Dokümantasyon

- [Training System Documentation](../training_system/README.md) - TrainingService, train.py
- [Model Management Documentation](../model_management/README.md) - ModelManager API
- [Neural Network Documentation](../neural_network/README.md) - V-4 architecture
- [API Reference](../../API_REFERENCE.md) - Full API documentation

---

## 🎓 Öğrenme Kaynakları

### Training Best Practices

- **Mixed Precision Training:** [PyTorch AMP Guide](https://pytorch.org/docs/stable/amp.html)
- **Gradient Accumulation:** [Effective Batch Size](https://arxiv.org/abs/1706.02677)
- **Learning Rate Scheduling:** [LR Finder Paper](https://arxiv.org/abs/1506.01186)

### Metrics & Evaluation

- **Precision/Recall/F1:** [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Top-K Accuracy:** [Multi-Class Classification Metrics](https://en.wikipedia.org/wiki/Precision_and_recall)

---

## 📝 Notlar

- ✅ **Production-Ready:** Endüstri standartlarına uygun
- ✅ **Well-Tested:** Comprehensive test suite
- ✅ **Well-Documented:** Full API documentation
- ✅ **Memory-Efficient:** Colab-optimized validation
- ✅ **Error-Resilient:** Graceful error handling
- ✅ **Observable:** Comprehensive logging and monitoring

---

**Son Güncelleme:** 2025-01-27  
**Versiyon:** V-5  
**Durum:** ✅ Production-Ready
