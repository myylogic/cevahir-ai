# Model Management Modülü — Dokümantasyon

**Versiyon:** 4.1.0
**Son Güncelleme:** 2026-03-16
**Durum:** Production-Ready
**Ana Dizin:** `model_management/`

---

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari ve Dosya Yapısı](#mimari-ve-dosya-yapısı)
3. [ModelManager](#modelmanager)
4. [ModelInitializer](#modelinitializer)
5. [ModelSaver](#modelsaver)
6. [ModelLoader](#modelloader)
7. [ModelUpdater](#modelupdater)
8. [ModelProfiler](#modelprofiler)
9. [ModelHealthMonitor](#modelhealthmonitor)
10. [Config Schema](#config-schema)
11. [Exception Hiyerarşisi](#exception-hiyerarşisi)
12. [Kullanım Örnekleri](#kullanım-örnekleri)
13. [Eğitim Entegrasyonu](#eğitim-entegrasyonu)

---

## Genel Bakış

`model_management` modülü, Cevahir-AI modelinin yaşam döngüsünü yönetir: oluşturma, eğitim bileşenlerini başlatma, kaydetme/yükleme, profilleme ve sağlık izleme. SOLID prensiplerini uygular; her dosya tek bir sorumluluğa odaklanır.

```
train.py / cevahir.py
        |
        v
   ModelManager          <- Üst düzey API (facade)
   |-- ModelInitializer  <- Model + Optimizer + Scheduler oluşturma
   |-- ModelSaver        <- Atomik checkpoint kaydetme + SHA-256
   |-- ModelLoader       <- Güvenli checkpoint yükleme + versiyon kontrolü
   |-- ModelUpdater      <- Freeze/Unfreeze, LR güncelleme
   |-- ModelProfiler     <- Parametre sayımı, bellek, FLOP, zamanlama
   +-- ModelHealthMonitor<- Gradient, ağırlık, attention sağlık izleme
```

---

## Mimari ve Dosya Yapısı

```
model_management/
|-- __init__.py              # Herkese açık API (tüm sınıflar burada toplanır)
|-- model_manager.py         # ModelManager -- merkezi facade
|-- model_initializer.py     # ModelInitializer -- model/opt/sched oluşturma
|-- model_saver.py           # ModelSaver -- checkpoint kaydetme
|-- model_loader.py          # ModelLoader -- checkpoint yükleme
|-- model_updater.py         # ModelUpdater -- parametre güncelleme
|-- profiler.py              # ModelProfiler -- profilleme araçları
|-- health_monitor.py        # ModelHealthMonitor -- sağlık izleme
|-- config_schema.py         # Typed config dataclass'ları
|-- exceptions.py            # Exception hiyerarşisi
+-- test/
    |-- test_model_manager.py
    +-- test_model_manager_comprehensive.py
```

### Katman Mimarisi (içten dışa)

| Katman | Dosya | Görev |
|--------|-------|-------|
| 1 | `exceptions.py` | Hata hiyerarşisi |
| 2 | `config_schema.py` | Tip-güvenli konfigürasyon |
| 3 | `profiler.py` | Model profil araçları |
| 4 | `health_monitor.py` | Sağlık izleme |
| 5 | `model_initializer.py` | Model/opt/sched oluşturma |
| 6 | `model_saver.py` | Checkpoint kaydetme |
| 7 | `model_loader.py` | Checkpoint yükleme |
| 8 | `model_updater.py` | Parametre güncelleme |
| 9 | `model_manager.py` | Üst düzey facade |

---

## ModelManager

**Dosya:** `model_management/model_manager.py`

Tüm alt bileşenleri birleştiren merkezi yönetim sınıfı. Eğitim ve inference işlemlerinin giriş noktasıdır.

### `__init__` Parametreleri

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `config` | `Dict[str, Any]` | — | Model ve eğitim konfigürasyonu |
| `model_class` | `Type[nn.Module]` | `CevahirNeuralNetwork` | Model sınıfı |
| `device` | `str\|torch.device\|None` | config'ten | Cihaz |
| `initializer` | sınıf | `ModelInitializer` | DI: oluşturucu |
| `saver` | sınıf | `ModelSaver` | DI: kaydedici |
| `updater` | sınıf | `ModelUpdater` | DI: güncelleyici |
| `tokenizer` | Any | `None` | Multimodal: tokenizer |
| `audio_processor` | Any | `None` | Multimodal: ses |
| `vision_processor` | Any | `None` | Multimodal: görüntü |

### Ana Metodlar

| Metod | Açıklama |
|-------|----------|
| `build_model()` | Modeli oluşturur, cihaza taşır, profil raporunu loglar |
| `build_optimizer()` | Optimizer oluşturur |
| `build_criterion()` | Loss fonksiyonunu oluşturur |
| `build_scheduler()` | LR scheduler oluşturur |
| `initialize(...)` | Tek çağrıda tüm bileşenleri başlatır |
| `forward(input_ids, ...)` | Model forward geçişi; OOM recovery dahil |
| `generate(input_ids, ...)` | Autoregressive token üretimi |
| `predict(input_ids, ...)` | Top-k tahmin, logit/softmax seçeneği |
| `save(epoch, ...)` | Checkpoint kaydeder |
| `load(path, ...)` | Checkpoint yükler |
| `train_mode()` | `model.train()` + dropout aktif |
| `eval_mode()` | `model.eval()` + dropout kapalı |
| `health_check()` | `HealthReport` döndürür |
| `profile()` | Profil raporu döndürür |
| `setup_tensorboard(log_dir)` | TensorBoard writer başlatır |

### Device Seçimi (Öncelik Sırası)

```
1. __init__(device=...) ile açıkça belirtildi
2. config["device"] == "cuda" ve CUDA mevcutsa -> cuda
3. config["device"] == "mps" ve MPS mevcutsa -> mps
4. CUDA otomatik keşfi -> cuda (varsa)
5. Fallback -> cpu
```

### TensorBoard Entegrasyonu

```python
mm = ModelManager(config)
mm.setup_tensorboard(log_dir="runs/cevahir_v6")

# Her model forward'unda otomatik yazılan metrikler:
# - train/loss, train/perplexity
# - grad_norm, lr
# - attn_entropy (model bu degeri sakliyorsa)
```

### Otomatik Profil Raporu

`build_model()` çağrısının ardından otomatik olarak `ModelProfiler.full_report()` çalıştırılır:

```
[Profiler] == Model Raporu ==
  Parametreler : ParamStats(total=85.23M, trainable=85.23M, frozen=0, trainable_mem=325.2 MB)
  Model boyutu : 325.2 MB
  Bellek       : MemorySnapshot(cuda) alloc=1.32 GB / total=8.00 GB (16.5%)
  FLOP (T=512) : FlopEstimate(total=42.3 GFLOPs, attn=18.1, ffn=24.1, seq=512, batch=1)
```

---

## ModelInitializer

**Dosya:** `model_management/model_initializer.py`

Model örneği, optimizer, loss fonksiyonu ve LR scheduler oluşturmak için statik metodlar içerir. Tüm metodlar `@staticmethod`tur; instance oluşturmaya gerek yoktur.

### `build_model()`

```python
model = ModelInitializer.build_model(
    model_class=CevahirNeuralNetwork,
    config=config_dict,
    device=torch.device("cuda"),
    compile_model=True,    # torch.compile (PyTorch 2.0+)
)
```

**Sıralı adımlar:**
1. `_apply_seed(config)` — deterministik başlatma (opsiyonel)
2. `_resolve_device(config)` — cihaz seçimi
3. `_filter_kwargs_for_ctor(model_class, config)` — config'i model imzasına göre süz
4. `model_class(**ctor_kwargs).to(device)` — model oluşturma
5. `torch.compile(model, ...)` — derleme (opsiyonel, `torch_compile=True`)
6. `gradient_checkpointing_enable()` — (opsiyonel)
7. `_apply_quantization(...)` — INT8/INT4 (opsiyonel)
8. `_wrap_distributed(...)` — DDP/FSDP (opsiyonel)

**Güvenli imza süzme:** Config'teki bilinmeyen anahtarlar otomatik filtrelenir; model `__init__` imzası `inspect.signature` ile okunur.

### `initialize_optimizer()`

**Desteklenen optimizer'lar:**

| İsim | Açıklama |
|------|----------|
| `adamw` | PyTorch AdamW (opsiyonel fused) |
| `adamw8bit` / `adamw_8bit` | bitsandbytes 8-bit AdamW — optimizer m/v durumlarini uint8 saklar, ~%75 bellek azalmasi |
| `adam` | Standart Adam |
| `radam` | Rectified Adam |
| `rmsprop` | RMSProp |
| `sgd` | Stochastic Gradient Descent |

**Parametre grupları (3 ayrı grup):**

```
Grup 1 — Embedding: lr = base_lr x embedding_lr_scale (varsayilan 1.0)
Grup 2 — Decay:     lr = base_lr, weight_decay = config degeri
Grup 3 — No-decay:  lr = base_lr, weight_decay = 0.0
         (bias, norm, layernorm, bn gibi parametreler)
```

> **Not:** `embedding_lr_scale` varsayılanı 1.0'dır (base_lr ile aynı). Eski değer 0.1 idi ve EOS/nadir token öğrenimini zayıflatıyordu.

**AdamW8bit (bitsandbytes):**

```python
# Dettmers et al. 2022 -- 8-bit optimizer
# Kurulum: pip install bitsandbytes
# Mevcut degilse otomatik standart AdamW'a fallback
optimizer: str = "adamw8bit"
```

### `initialize_criterion()`

| İsim | Sınıf |
|------|-------|
| `cross_entropy` / `ce` | `nn.CrossEntropyLoss` (label_smoothing, ignore_index destekli) |
| `bce_with_logits` / `bce` | `nn.BCEWithLogitsLoss` |
| `mse` | `nn.MSELoss` |
| `smooth_l1` / `huber` | `nn.SmoothL1Loss` |

```python
# PAD token'larini loss'tan cikarmak icin:
ignore_index: 0   # 0=PAD ignore, -100=hepsini say (varsayilan -100)
```

### `initialize_scheduler()`

| Tür | Config anahtarı | Açıklama |
|-----|-----------------|----------|
| `reduce_on_plateau` | `scheduler_type: "rop"` | Validation loss platoya ulaşınca LR düşür |
| `cosine` | `scheduler_type: "cosine"` | Kosinüs azalma |
| `cosine_warm_restarts` | `scheduler_type: "cawr"` | Warm restart'lı kosinüs |
| `step` | `scheduler_type: "step"` | Sabit adımda LR çarpanı |
| `exponential` | `scheduler_type: "explr"` | Üstel azalma |
| `onecycle` | `scheduler_type: "onecycle"` | OneCycleLR (steps_per_epoch ve epochs zorunlu) |
| `none` | `scheduler_type: "none"` | Scheduler yok |

### `build_training_components()` — Tek Çağrı

```python
optimizer, criterion, scheduler = ModelInitializer.build_training_components(model, config)
```

### Quantization Desteği

| Tür | Açıklama | Gereksinim |
|-----|----------|------------|
| `int8` | LLM.int8() — threshold=6.0, ~%50 VRAM azalması | `bitsandbytes` |
| `int4` | NF4 + double quantization, ~%75 VRAM azalması | `bitsandbytes` |

### Dağıtık Eğitim

| Strateji | Açıklama |
|----------|----------|
| `ddp` | DistributedDataParallel — gradientleri senkronize eder |
| `fsdp` | FullyShardedDataParallel — model parametrelerini GPU'lara parçalar |

```python
config = {
    "distributed_strategy": "ddp",
    "distributed_backend": "nccl",
    "local_rank": 0,
}
# torchrun ile baslatilir
```

---

## ModelSaver

**Dosya:** `model_management/model_saver.py`
**Checkpoint Versiyonu:** `4.1` / Format: `2`

Atomik kayıt (tmp → `os.replace`) ve SHA-256 bütünlük doğrulaması ile güvenli checkpoint yönetimi.

### `save_checkpoint()` — Ana API

```python
path = ModelSaver.save_checkpoint(
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    config=config,
    metadata={"val_loss": 2.34, "train_loss": 1.87},
    save_dir="saved_models/checkpoints",
    filename_template="checkpoint_ep{epoch:04d}.pth",
    create_latest_marker=True,  # latest.txt olusturur
    keep_last_n=5,              # En eski 5 checkpoint disindakileri sil
    prefix_for_prune="checkpoint_",
)
```

**Kaydedilen checkpoint yapısı:**

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": epoch,
    "config": config,
    "metadata": {
        # Kullanici metadata (val_loss, train_loss vb.)
        "cevahir_version": "4.1",
        "checkpoint_format": 2,
        "saved_at": "2026-03-16T10:30:00Z",
        "total_params": 85234567,
        "trainable_params": 85234567,
        "sha256": "abc123..."
    }
}
```

**SHA-256 sidecar dosyası:** `checkpoint_ep0010.pth.sha256` — dışarıdan hızlı bütünlük doğrulaması için.

### Akıllı Checkpoint Budama

```
prefer_best_val_loss=True (varsayilan):
  1. Her checkpoint'in metadata'sindaki val_loss kontrol edilir
  2. En iyi val_loss'a sahip checkpoint KESINLIKLE korunur
  3. Geriye kalan slotlara en yeni checkpoint'ler yerlestirilir
```

### Atomik Yazma

```
1. BytesIO -> serialize et
2. tempfile.mkstemp() -> gecici dosyaya yaz
3. os.replace(tmp, hedef) -> atomik tasi
```

CUDA hatası durumunda tensörler CPU'ya taşınarak yeniden denenir.

### Diğer Metodlar

| Metod | Açıklama |
|-------|----------|
| `save_weights_only(model, ...)` | Sadece state_dict kaydeder |
| `save_full_model(model, ...)` | Tüm model (pickle) — genellikle önerilmez |
| `save_additional_info(info, ...)` | Ek bilgileri JSON olarak kaydeder |
| `save_model(...)` | Eski API (geriye dönük uyumluluk) |

---

## ModelLoader

**Dosya:** `model_management/model_loader.py`

### `load_model()` — Tek Model Yükleme

```python
model = ModelLoader.load_model(
    model_class=CevahirNeuralNetwork,
    model_path="saved_models/checkpoints/checkpoint_ep0010.pth",
    device="cuda",
    config=config,
    strict=True,
    weights_only=None,
)
```

**Yükleme adımları:**
1. `_verify_sha256(path)` — SHA-256 sidecar kontrolü (varsa)
2. `_check_version_compatibility(ckpt)` — format versiyon uyumluluğu
3. `_extract_state_dicts(ckpt)` — model/opt/sch state dict ayırma
4. `model_class(**ctor_kwargs).to(device)` — model örneği oluştur
5. Vocab size kontrolü (`embedding.weight` şekli)
6. `model.load_state_dict(model_sd, strict=strict)` — ağırlık yükleme
7. Eksik/beklenmeyen anahtar uyarıları

### `load_all()` — Tek Çağrıda Hepsi

```python
model, opt_sd, sch_sd, meta = ModelLoader.load_all(
    model_class=CevahirNeuralNetwork,
    ckpt_path="checkpoint_ep0010.pth",
    device="cuda",
    config=config,
)
# meta: {"epoch": 10, "config": {...}}
```

### Checkpoint Format Desteği

| Format | Açıklama |
|--------|----------|
| `{"model_state_dict": ..., "optimizer_state_dict": ..., ...}` | Tam checkpoint (önerilen) |
| `{"state_dict": ...}` | Framework uyumlu |
| `{str: Tensor, ...}` | Düz state_dict |

### Versiyon Uyumluluğu

```
Format 1 -> Format 2: Geriye dönük uyumlu
Format 3+: CheckpointVersionError firlatir
```

---

## ModelUpdater

**Dosya:** `model_management/model_updater.py`

Model parametrelerini ve eğitim bileşenlerini çalışma zamanında güncellemek için statik metodlar.

### Temel Metodlar

| Metod | Açıklama |
|-------|----------|
| `freeze_layers(model, layers)` | Belirtilen katmanları dondurur (`requires_grad=False`) |
| `unfreeze_layers(model, layers)` | Dondurulmuş katmanları serbest bırakır |
| `freeze_all_except(model, patterns)` | Belirtilen desenler hariç tüm katmanları dondurur |
| `update_learning_rate(optimizer, lr)` | Tüm param gruplarında LR günceller |
| `update_weight_decay(optimizer, wd)` | Weight decay günceller |
| `step_scheduler(scheduler, metric)` | Scheduler adımı (plateau için metric gerekli) |
| `apply_weight_noise(model, std)` | Ağırlıklara Gaussian gürültü ekler |
| `reset_parameters(model, layers)` | Seçili katmanların ağırlıklarını sıfırlar |

### Desen Tabanlı Dondurma

```python
# Glob/regex desen destegi
ModelUpdater.freeze_all_except(model, patterns=["layers.7.*", "output_layer"])
# Sadece son katman ve output projection egitilebilir kalir
```

---

## ModelProfiler

**Dosya:** `model_management/profiler.py`

Model boyutu, parametre sayısı, FLOP tahmini ve zamanlama için statik araçlar. Instance oluşturmaya gerek yoktur.

### Veri Sınıfları

| Sınıf | Alanlar |
|-------|---------|
| `ParamStats` | `total`, `trainable`, `frozen`, `trainable_mb`, `by_layer` |
| `MemorySnapshot` | `allocated_mb`, `reserved_mb`, `free_mb`, `total_mb`, `device` |
| `FlopEstimate` | `total_flops`, `attention_flops`, `ffn_flops`, `embedding_flops`, `gflops` |
| `TimingResult` | `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `tokens_per_second` |

### `count_parameters()`

```python
stats = ModelProfiler.count_parameters(model)
print(stats)
# ParamStats(total=85.23M, trainable=85.23M, frozen=0, trainable_mem=325.2 MB)

# Katman bazli dagilim (en buyuk 5 katman):
#   embedding:    30.72M params
#   layers:       54.01M params
#   output_layer:  0.50M params
```

### `memory_snapshot()`

```python
mem = ModelProfiler.memory_snapshot("cuda")
print(mem)
# MemorySnapshot(cuda) alloc=3.72 GB / total=8.00 GB (46.5%)

print(f"Kullanim: {mem.utilization_pct:.1f}%")
print(f"Bos: {mem.free_mb:.0f} MB")
```

### `estimate_flops()`

Kaplan et al. 2020 / PaLM paper formüllerine göre teorik FLOP tahmini:

```
Attention: 4 x B x L x T x D^2   (QKV + Output projeksiyonu)
           2 x B x L x H x T^2   (dikkat skoru)
FFN:       2 x B x L x T x D x F x 2
B=batch, L=num_layers, T=seq_len, D=embed_dim, H=num_heads, F=ffn_dim
```

```python
flops = ModelProfiler.estimate_flops(model, seq_len=512, batch_size=4)
print(flops)
# FlopEstimate(total=42.3 GFLOPs, attn=18.1, ffn=24.1, seq=512, batch=4)
```

### `benchmark_forward()`

```python
sample = torch.randint(0, 32000, (1, 512)).cuda()
timing = ModelProfiler.benchmark_forward(model, sample, n_warmup=3, n_runs=20)
print(timing)
# TimingResult(mean=23.45ms +-0.87, tok/s=21834, runs=20)
```

GPU için `torch.cuda.Event` tabanlı hassas ölçüm kullanır.

### `profile_context()` — torch.profiler Entegrasyonu

```python
with ModelProfiler.profile_context(model, output_path="./profiler_trace") as prof:
    logits, _ = model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# Chrome trace dosyasina kaydedilir (TensorBoard ile goruntulenebilir)
```

### `full_report()` — Özet Rapor

```python
report = ModelProfiler.full_report(model, seq_len=512, run_timing=True)
# Doner: {"params": ParamStats, "memory": MemorySnapshot,
#          "flops": FlopEstimate, "timing": TimingResult, "size_mb": float}
```

---

## ModelHealthMonitor

**Dosya:** `model_management/health_monitor.py`

Gradient akışı, ağırlık dağılımı ve attention entropy patolojilerini tespit eder.

### Tespit Edilen Patolojiler

| Patoloji | Eşik | Seviye |
|----------|------|--------|
| NaN gradient | Herhangi biri | CRITICAL |
| Inf gradient | Herhangi biri | CRITICAL |
| Gradient vanishing | `grad_norm < 1e-8` | INFO |
| Gradient exploding | `grad_norm > 1e4` | WARNING |
| NaN ağırlık | Herhangi biri | CRITICAL |
| Ölü ağırlık | `std < 1e-9` | INFO |
| Ağırlık patlaması | `abs_max > 1e3` | WARNING |
| Attention collapse | `entropy < 0.05` | WARNING |
| Attention uniform | `entropy > 0.99` | INFO |

### Veri Sınıfları

| Sınıf | Açıklama |
|-------|----------|
| `GradientHealth` | Gradient NaN/Inf/vanish/explode bilgisi |
| `WeightHealth` | Ağırlık dağılımı, ölü/patlayan katmanlar |
| `AttentionHealth` | Attention entropy, collapse/uniform katmanlar |
| `HealthReport` | Üç raporu birleştiren birleşik rapor |

### Severity Seviyeleri

```
OK       -> Her sey normal
INFO     -> Vanishing gradient veya olü katman var (izle)
WARNING  -> Exploding gradient veya attention collapse (müdahale et)
CRITICAL -> NaN/Inf (eğitim bozuldu, dur)
```

### `full_health_check()` — Birleşik Kontrol

```python
report = ModelHealthMonitor.full_health_check(
    model,
    sample_input=sample,
    check_gradients=True,
    check_weights=True,
    check_attention=True,
    raise_on_critical=True,     # CRITICAL durumda HealthCheckError firlatir
)

if not report.is_healthy:
    print(report.summary())
```

**Örnek çıktı:**

```
============================================================
  ⚠ MODEL SAGLIK RAPORU -- WARNING
============================================================
GradientHealth [WARNING]
  ⚠  Exploding gradients: ['layers.3.ffn.fc1.weight']
  norm: max=1.23e+05, min=1.23e-06, mean=3.45e+00

WeightHealth [OK]
  global: mean=0.0012, std=0.0345, max_abs=0.8921

AttentionHealth [OK]
  entropy: mean=0.423, min=0.312, max=0.687
============================================================
```

### `quick_gradient_check()` — Her Batch İçin Hızlı Kontrol

```python
# backward() sonrasi her batch'te:
is_safe, msg = ModelHealthMonitor.quick_gradient_check(model)
if not is_safe:
    logger.warning(f"NaN/Inf gradient -- batch atlaniyor: {msg}")
    optimizer.zero_grad()
    continue
```

### `log_gradient_norms()` — TensorBoard Entegrasyonu

```python
norms = ModelHealthMonitor.log_gradient_norms(
    model, step=global_step, tb_writer=writer, top_n=10
)
# En yuksek 10 gradient normunu TensorBoard'a yazar
```

### Attention Entropy İzleme

Model `_last_attn_entropy` attribute'unu tutuyorsa (CevahirNeuralNetwork bunu yapar), her forward sonrası otomatik okunur:

```python
# Modelin forward() sirasinda hesaplanir ve saklanir:
self._last_attn_entropy = normalized_entropy   # [0, 1]

# HealthMonitor bunu okur:
health = ModelHealthMonitor.check_attention_entropy(model)
# Collapse (< 0.05), Normal, Uniform (> 0.99)
```

---

## Config Schema

**Dosya:** `model_management/config_schema.py`

Düz `Dict[str, Any]` yerine tip-güvenli, doğrulanabilir yapılandırma şemaları.

### Sınıflar

#### `ModelArchConfig`

```python
arch = ModelArchConfig(
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    vocab_size=32000,
    ffn_dim=None,
    use_swiglu=True,
    use_rmsnorm=True,
    num_kv_heads=2,
    pe_mode="rope",
    rope_scaling_type="yarn",
    rope_scaling_factor=4.0,
    use_moe=False,
    quantization_type="none",
    tie_weights=True,
)
arch.validate()

print(arch.head_dim)                  # 64
print(arch.effective_ffn_dim)         # 2048
print(arch.parameter_count_estimate)  # yaklasik 85M
```

**Doğrulanan kurallar:**
- `embed_dim % num_heads == 0`
- `num_heads % num_kv_heads == 0` (GQA uyumluluğu)
- `rope_scaling_factor >= 1.0`
- `moe_top_k <= num_experts`
- `tie_weights=True` ise `seq_proj_dim == embed_dim`

#### `TrainingConfig`

```python
training = TrainingConfig(
    learning_rate=2e-4,
    batch_size=72,
    grad_accum_steps=4,
    optimizer="adamw8bit",
    scheduler_type="reduce_on_plateau",
    use_amp=True,
    use_gradient_checkpointing=True,
    use_ema=True,
    ema_decay=0.999,
)
print(training.effective_batch_size)  # 288 (72 x 4)
```

#### `CheckpointConfig`

```python
ckpt_cfg = CheckpointConfig(
    save_dir="saved_models/checkpoints",
    keep_last_n=5,
    save_every_n_epochs=10,
    enable_sha256=True,
)
```

#### `DistributedConfig`

```python
dist_cfg = DistributedConfig(
    enabled=True,
    backend="nccl",
    strategy="ddp",
    world_size=4,
)
```

#### `QuantConfig`

```python
quant_cfg = QuantConfig(
    quant_type="int4",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
```

#### `CevahirConfig` — Birleşik Konfigürasyon

```python
# train.py'deki düz dict'ten olustur:
cfg = CevahirConfig.from_flat_dict(TRAIN_CONFIG)
cfg.validate_all()

print(cfg.arch.embed_dim)
print(cfg.training.effective_batch_size)

d = cfg.to_dict()
```

---

## Exception Hiyerarşisi

**Dosya:** `model_management/exceptions.py`

```
CevahirModelError
|-- ModelNotInitializedError     -> initialize() cagirilmadan kullanim
|-- ModelBuildError              -> model/optimizer/scheduler olusturma hatasi
|    +-- QuantizationError       -> INT8/INT4 quantization basarisizligi
|-- CheckpointError              -> checkpoint I/O taban hatasi
|    |-- CheckpointNotFoundError -> dosya yok
|    |-- CheckpointCorruptError  -> SHA-256 uyumsuz / format bozuk
|    +-- CheckpointVersionError  -> versiyon uyumsuzlugu
|-- ForwardError                 -> model forward pass hatasi
|    +-- OOMRecoveryError        -> CUDA OOM -> kurtarma basarisiz
|-- DeviceError                  -> device secimi / transfer hatasi
|    +-- DeviceMismatchError     -> tensor device uyumsuzlugu
|-- ShapeError                   -> tensor sekil uyumsuzlugu
|    +-- VocabSizeMismatchError  -> vocab_size checkpoint vs model
|-- DistributedSetupError        -> DDP/FSDP kurulum hatasi
+-- HealthCheckError             -> model saglik testi basarisiz
```

### Kullanım

```python
from model_management import (
    CevahirModelError,
    CheckpointNotFoundError,
    OOMRecoveryError,
    VocabSizeMismatchError,
)

try:
    model = ModelLoader.load_model(CevahirNeuralNetwork, path, config=config)
except CheckpointNotFoundError as e:
    print(f"Checkpoint bulunamadi: {e.path}")
except VocabSizeMismatchError as e:
    print(f"Vocab uyumsuz: model={e.model_vocab}, ckpt={e.checkpoint_vocab}")
except CevahirModelError as e:
    print(f"Genel model hatasi: {e.message} | {e.context}")
```

---

## Kullanım Örnekleri

### Hızlı Başlangıç

```python
from model_management import ModelManager

config = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 2,
    "num_layers": 8,
    "ffn_dim": None,
    "use_swiglu": True,
    "use_pytorch_sdpa": True,
    "logit_soft_cap": 30.0,
    "dropout": 0.1,
    "max_seq_length": 2048,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "optimizer": "adamw8bit",
    "scheduler_type": "reduce_on_plateau",
    "criterion": "cross_entropy",
    "ignore_index": 0,
    "device": "cuda",
}

mm = ModelManager(config)
mm.initialize(
    build_model=True,
    build_optimizer=True,
    build_criterion=True,
    build_scheduler=True,
)
```

### Eğitim Döngüsü Entegrasyonu

```python
mm.train_mode()

for batch in dataloader:
    input_ids = batch["input_ids"].to(mm.device)
    labels = batch["labels"].to(mm.device)

    logits, _ = mm.forward(input_ids)
    loss = mm.criterion(
        logits[:, :-1].reshape(-1, vocab_size),
        labels[:, 1:].reshape(-1)
    )
    (loss / grad_accum_steps).backward()

    # NaN kontrol
    is_safe, msg = ModelHealthMonitor.quick_gradient_check(mm.model)
    if not is_safe:
        mm.optimizer.zero_grad()
        continue

    torch.nn.utils.clip_grad_norm_(mm.model.parameters(), 1.0)
    mm.optimizer.step()
    mm.optimizer.zero_grad()

# Epoch sonu checkpoint
mm.save(
    epoch=epoch,
    metadata={"val_loss": val_loss, "train_loss": avg_loss},
    keep_last_n=5,
)
```

### Checkpoint Kaydet / Yükle

```python
# Kaydet
path = ModelSaver.save_checkpoint(
    model, optimizer=optimizer, epoch=10,
    config=config, metadata={"val_loss": 2.34},
    save_dir="saved_models", keep_last_n=5,
)

# Yükle
model, opt_sd, sch_sd, meta = ModelLoader.load_all(
    CevahirNeuralNetwork, path, device="cuda", config=config,
)
optimizer.load_state_dict(opt_sd)
print(f"Epoch {meta['epoch']} yuklendi")
```

### Profil + Sağlık Kontrolü

```python
from model_management import ModelProfiler, ModelHealthMonitor

# Profil
stats = ModelProfiler.count_parameters(model)
mem = ModelProfiler.memory_snapshot("cuda")
flops = ModelProfiler.estimate_flops(model, seq_len=512)

# Saglik kontrolü (epoch sonu)
report = ModelHealthMonitor.full_health_check(
    model, sample_input=sample, raise_on_critical=True
)
if not report.is_healthy:
    print(report.summary())
```

---

## Eğitim Entegrasyonu

### train.py'deki Tipik Kullanım

```python
mm = ModelManager(TRAIN_CONFIG)
mm.initialize(build_model=True, build_optimizer=True,
              build_criterion=True, build_scheduler=True)
mm.setup_tensorboard("runs/cevahir_v6")

if resume_path:
    mm.load(resume_path)

for epoch in range(start_epoch, total_epochs):
    mm.train_mode()
    for batch in train_loader:
        # training step...
        pass

    mm.eval_mode()
    val_loss = validate(mm, val_loader)

    if mm.scheduler:
        mm.scheduler.step(val_loss)

    mm.save(epoch=epoch, metadata={"val_loss": val_loss})

    if epoch % 10 == 0:
        report = mm.health_check()
        if not report.is_healthy:
            logger.warning(report.summary())
```

---

## Bağımlılıklar

| Paket | Versiyon | Zorunlu | Amaç |
|-------|----------|---------|------|
| `torch` | >= 2.0 | Evet | PyTorch çekirdek |
| `bitsandbytes` | >= 0.41 | Hayır | AdamW8bit + INT8/INT4 quantization |
| `torch.distributed` | PyTorch ile gelir | Hayır | DDP/FSDP |

---

*Yazar: Muhammed Yasin Yılmaz — Cevahir-AI Projesi*
*Telif Hakkı: © 2024-2026 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.*
