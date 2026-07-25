# Code Reality — Training Management

> Kanonik Birim #4 · Kaynak: `training_management/` (v2 + v3)
> Eğitim **motorunun** (döngü + güvenlik + izleme) kodundan çıkarılmış mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L2), [search index](../../architecture-search-index.md).
> Komşu birim: [training-system](../training-system/) (bu motoru çağıran servis).

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Training Management (eğitim motoru) |
| **Kaynak dizin** | `training_management/v2/`, `training_management/v3/` |
| **Toplam boyut** | ~18.300 LOC (v2 + v3, testler hariç) |
| **Ana sınıflar** | `TrainingManager` (v2: `v2/core/training_manager.py:69`, v3: `v3/core/training_manager.py:211`) |
| **Çağıran** | `training_system` (`TrainingService` → `TrainingManager`) |
| **Çalışma zamanı** | Batch (eğitim) |
| **Dış bağımlılık** | `torch`; Model Mgmt (`ModelManager`), Neural Network (dolaylı) |

> **v2/v3 birlikte yaşıyor.** İkisi de tam bir `TrainingManager` + alt sistem seti
> içerir. v3 daha yeni (curriculum, optimizers, gelişmiş safety) ama v3'ün bazı
> parçaları hâlâ v2 util'lerine güvenir (bkz. §8). Bu, birimin **en belirgin
> refactor borcudur.**

---

## 2. Sorumluluk

Bir eğitim koşusunun **iç mekaniğini** yürütmek: epoch/adım döngüsü, ileri-geri
geçiş, gradyan yönetimi (clipping/accumulation), kayıp hesaplama, metrikler,
kararlılık güvenliği (NaN/Inf, loss spike, divergence), müfredat (curriculum),
optimizasyon stratejileri (SAM, Lookahead, EMA), TensorBoard izleme, checkpoint.

**Kapsam dışı:** Veri yükleme/cache ve koşu kurulumu (Training System), model
tanımı (Neural Network), model IO (Model Mgmt).

---

## 3. Dosya Envanteri

### 3.1 v3 (güncel) — `training_management/v3/`

| Alt sistem | Dosya(lar) | Rol |
|------------|-----------|-----|
| **core** | `training_manager.py` (1116), `training_loop.py` (964), `loss_manager.py` (783), `gradient_manager.py` (594), `batch_processor.py` (505) | Eğitim döngüsü orkestrasyonu, kayıp/gradyan/batch |
| **curriculum** | `curriculum_manager.py` (676) | Müfredat öğrenimi (zorluk sıralaması) |
| **optimizers** | `sam.py` (333), `lookahead.py` (353) | Gelişmiş optimizasyon sarmalayıcıları |
| **safety** | `nan_recovery.py` (355), `loss_spike_detector.py` (395), `divergence_detector.py` (397), `checkpoint_verifier.py` (414) | Kararlılık koruması |
| **monitoring** | `tensorboard_manager.py` (587), `gradient_health_monitor.py` (424), `token_distribution_monitor.py` (373), `inference_quality_probe.py` (597) | İzleme + kalite probu |
| **utils** | `training_scheduler.py` (711), `checkpoint_manager.py` (610), `training_logger.py` (587), `metrics_tracker.py` (476), `ema.py` (324) | Zamanlayıcı, checkpoint, log, metrik, EMA |

`TrainingManager` yardımcıları: `_EarlyStopper`, `TrainingManagerConfig`,
`from_config`, `resume_from_checkpoint`, `_save_checkpoints`, `_log_to_tensorboard`.

### 3.2 v2 (önceki) — `training_management/v2/`

| Alt sistem | Dosya(lar) | Rol |
|------------|-----------|-----|
| **core** | `training_loop.py` (724), `training_manager.py` (603), `gradient_manager.py` (224), `loss_computation.py` (206), `batch_processor.py` (197) | v2 eğitim döngüsü |
| **metrics** | `advanced_token_metrics.py` (261), `metrics_calculator.py` (112) | Token/eğitim metrikleri |
| **monitoring** | `tensorboard_manager.py` (193) | TensorBoard |
| **safety** | `nan_inf_detector.py` (113), `gradient_explosion_detector.py` (118) | Temel güvenlik |
| **utils** | `training_logger.py` (604), `checkpoint_manager.py` (499), `training_scheduler.py` (450), `training_analytics.py` (446), `training_visualizer.py` (444), `enhanced_training_logger.py` (183) | Log/checkpoint/analitik/görselleştirme |

> v2, `training_system` tarafından hâlâ **fiilen kullanılıyor** (checkpoint/tensorboard/
> logger util'leri v3 servisinde bile import ediliyor).

---

## 4. İç Mimari

```
             training_system.TrainingService  (çağıran — Training System birimi)
                              │  TrainingManager(...) / from_config(...)
                              ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │  TrainingManager  (v3/core/training_manager.py)                         │
   │  · epoch döngüsü, early stopping (_EarlyStopper), best-checkpoint takibi│
   │                                                                         │
   │   her epoch → TrainingLoop.train()                                      │
   │        ├─ BatchProcessor   (batch hazırlama/ileri geçiş)                │
   │        ├─ LossManager      (kayıp + MoE load-balance eklenmesi)         │
   │        ├─ GradientManager  (clip, accumulation, health)                 │
   │        ├─ TrainingScheduler(lr warmup/decay)                            │
   │        │                                                                │
   │   çapraz kesişen:                                                       │
   │        ├─ safety/  (NaN recovery, loss spike, divergence, ckpt verify)  │
   │        ├─ monitoring/ (TensorBoard, gradient health, token dist, probe) │
   │        ├─ curriculum/ (zorluk sıralaması)                               │
   │        ├─ optimizers/ (SAM, Lookahead) + utils/ema                      │
   │        └─ utils/ (checkpoint_manager, training_logger, metrics_tracker) │
   └───────────────────────────────────────────────────────────────────────┘
                              │ eğitir/kaydeder
                              ▼
                   ModelManager / CevahirNeuralNetwork
```

---

## 5. Kontrol Akışı (bir epoch)

```
TrainingManager.train()
  └─ for epoch in range(epochs):
       └─ TrainingLoop.train_epoch()
            └─ for batch in loader:
                 ├─ BatchProcessor → model.forward → logits
                 ├─ LossManager → loss (+ MoE aux loss)
                 ├─ safety: NaN/Inf/loss-spike/divergence kontrol
                 │     └─ tetiklenirse nan_recovery / skip / rollback
                 ├─ backward + GradientManager (clip, accumulate)
                 ├─ optimizer.step (ops. SAM/Lookahead) + scheduler.step + EMA
                 └─ monitoring: TensorBoard + gradient_health + token_dist
       ├─ epoch_callback(epoch, train_loss, val_loss)   ← servise geri bildirim
       ├─ _EarlyStopper.step(val_metric)
       └─ _save_checkpoints() + checkpoint_verifier
```

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni güvenlik dedektörü | `v3/safety/` (dedektör + `TrainingLoop`'a bağla) | Mevcut dedektör arayüzünü izle |
| Yeni optimizer stratejisi | `v3/optimizers/` (SAM/Lookahead deseni) | optimizer sarmalayıcı |
| Müfredat stratejisi | `v3/curriculum/curriculum_manager.py` | zorluk fonksiyonu |
| Yeni metrik | `v3/utils/metrics_tracker.py` + `monitoring/` | TensorBoard tag |
| Kayıp değişikliği | `v3/core/loss_manager.py` | MoE aux loss kontratı |
| Checkpoint politikası | `v3/utils/checkpoint_manager.py` + `safety/checkpoint_verifier.py` | doğrulama simetrisi |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** `torch`; Model Mgmt (`ModelManager` üzerinden model/optimizer).
**Buna bağımlı olanlar:** **Training System** (`TrainingService`/`TrainingServiceV3`
doğrudan `TrainingManager` + v2 util'lerini import eder).

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **v2/v3 ikizliği** | İki ayrı `TrainingManager`, `training_loop`, `gradient_manager`, `checkpoint_manager`, `tensorboard_manager` | **Yüksek** | Kanonik sürüm kararı; ortak çekirdek + sürüm-özel eklentiler |
| **v3 → v2 sızıntısı** | `training_system/v3/...` içinde `from training_management.v2.utils...` import (checkpoint/tensorboard/logger) | **Yüksek** | v3 "temiz" değil; util'ler v3'e taşınmalı veya paylaşılan `common/`'a |
| **Repo-içi eski doclar** | `training_system/v2/ARCHITECTURE.md`, `COMPLETE_STATUS.md`, `FIXES_APPLIED.md` | Orta | Bu doküman güncel referanstır; eskiler arşivlenebilir |
| **Dev sınıflar** | `v3/core/training_manager.py` 1116 LOC, `training_loop.py` 964 | Orta | Sorumluluk bölme |
| **Örtüşen izleme** | v2 ve v3 ayrı `tensorboard_manager` + health monitor (Model Mgmt'te de var) | Orta | İzleme tek çatı altında toplanabilir |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| v3 eğitim yöneticisi | `training_management/v3/core/training_manager.py:211` |
| v3 eğitim döngüsü | `training_management/v3/core/training_loop.py` |
| v3 kayıp yönetimi | `training_management/v3/core/loss_manager.py` |
| v3 gradyan yönetimi | `training_management/v3/core/gradient_manager.py` |
| Early stopping | `training_management/v3/core/training_manager.py:140` (`_EarlyStopper`) |
| Checkpoint'ten devam | `training_management/v3/core/training_manager.py:1048` (`resume_from_checkpoint`) |
| v2 eğitim yöneticisi | `training_management/v2/core/training_manager.py:69` |
| v3 → v2 util sızıntısı | `training_system/v3/core/training_service_v3.py:506` |

---

*Kaynak: `training_management/` — analiz kodun mevcut halinden çıkarılmıştır.*
