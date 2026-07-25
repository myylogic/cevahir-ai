# Phase 1 — Refactoring Plan · Training Management

> **Kanonik Birim #4 · Training Management** — `training_management/` (v2+v3)
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: v2/v3 birleştirme).
> Komşu: [training-system](../training-system/phase-1-refactoring-plan.md).
>
> **Bu sürüm derinleştirilmiştir:** `dosya:satır` çapaları + mimari kesit bağları.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `training_management/v2/`, `training_management/v3/` (~18.300 LOC) |
| Mevcut test | **21 dosya / ~3.256 LOC — tamamı v2** |
| Refactor hedefleri | **v2/v3 ikizliği**, v3→v2 sızıntısı, dev sınıflar, örtüşen izleme |
| Kritik sözleşmeler | epoch döngüsü · checkpoint · `epoch_callback` · MoE aux loss |

> ⚠️ **En kritik boşluk:** v3 (kanonik hedef) **testsiz**; tüm testler v2'de.

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §6](../../master-architecture.md#6-uçtan-uca-akış--eğitim-training) | Eğitim yolu (bu birim motor katmanı) |
| [master-architecture §10](../../master-architecture.md#10-bağımlılık-yönü-ve-kural-i̇hlalleri) | v2/v3 ikizliği kırılma noktası |
| [research §4](README.md#4-i̇ç-mimari) | TrainingManager iç mimarisi |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri |
| [training-system planı](../training-system/phase-1-refactoring-plan.md) | Bu motoru çağıran servis (koordineli birleştirme) |
| [neural-network research §5.5](../neural-network/README.md#55-moe) | MoE aux loss (kayıp toplama sözleşmesi) |

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| v2 core | `v2/test/core/test_{gradient_manager,loss_computation,training_loop}.py` | ✅ Var |
| v2 integration | `v2/test/integration/test_{training_manager_full,comprehensive_training,critical_integrations,silent_error_scenarios}.py` | ✅ İyi |
| **v3 (tümü)** | — | ❌ **Test yok** |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **v3 test kapsamı sıfır** | `v3/core/training_manager.py:211`, `:423` (`train`); `training_loop.py:301`, `:567` (`train_epoch`) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | v3 loop/loss/gradient birim + entegrasyon |
| **T-02** | **v3→v2 sızıntısı** | `training_system/v3/core/training_service_v3.py:506` (v2 util import) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | v3'ün v2'siz çalıştığı testi |
| **T-03** | **Güvenlik dedektörleri** tetikleniyor mu | `v3/safety/{nan_recovery,loss_spike_detector,divergence_detector}.py` | [research §3.1](README.md#31-v3-güncel--training_managementv3) | NaN/spike/divergence senaryoları |
| **T-04** | **Checkpoint verify** | `v3/safety/checkpoint_verifier.py` | [research §3.1](README.md#31-v3-güncel--training_managementv3) | bozuk checkpoint yakalama |
| **T-05** | **Early stopping + best** | `v3/core/training_manager.py:140` (`_EarlyStopper`) | [research §5](README.md#5-kontrol-akışı-bir-epoch) | durma + best takibi |
| **T-06** | **Curriculum/optimizer** (SAM/Lookahead/EMA) | `v3/curriculum/curriculum_manager.py`, `v3/optimizers/{sam,lookahead}.py`, `v3/utils/ema.py` | [research §3.1](README.md#31-v3-güncel--training_managementv3) | strateji birim testleri |

### A.3 Olması Gereken Test Durumu
v3 için v2 ile denk kapsam; güvenlik/curriculum/optimizer birim testli; v3-bağımsızlık
entegrasyon testi (T-02) — **birleştirmenin önkoşulu.**

### A.4 Test Sprint'leri
**T1** v2 yeşil taban + v3 kapsam boşluğu haritası.
**T2** v3 çekirdek (T-01) — `training_manager.py:423`, `training_loop.py:567`.
**T3** güvenlik/curriculum/optimizer + v3-bağımsızlık (T-02..T-06).

---

## B. GELİŞTİRME FAZI *(P0)*

### B.1 Hedef Mimari
```
training_management/
  ├── common/  → paylaşılan util (checkpoint, tensorboard, logger)  [v2'den taşınır]
  └── core/    → tek kanonik TrainingManager (v3 tabanlı)
                 loop · loss · gradient · safety · curriculum · optimizers
  (v2 → _archive veya kaldır)
```
> İç mimari [research §4](README.md#4-i̇ç-mimari) ile uyumlu; ikizlik giderilir.

### B.2 Geliştirme Sprint'leri
**D1 — Kanonik sürüm kararı** *(karar)*: v3 kanonik. Belgelenir
([roadmap §3 Faz 1](../../development-roadmap.md#3-fazlar)).
**D2 — Ortak util çıkarma** *(orta risk)*: v2 `utils/{checkpoint_manager,training_logger}`
+ `monitoring/tensorboard_manager` → `common/`. Önkoşul: T-01/T-02. **Kabul:** hem v2 hem
v3 aynı util'i kullanır.
**D3 — v3→v2 bağını kes** *(orta risk)*: `training_service_v3.py:506` import'larını
`common/`'a yönlendir (**Training System D3 ile koordineli**). Önkoşul: T-02. **Kabul:**
v3-bağımsızlık testi yeşil.
**D4 — v2 emekliye ayır** *(orta risk)*: v2'yi `_archive` veya kaldır.
**D5 — Dev sınıf ayrıştırma** *(orta risk)*: `v3/core/training_manager.py:211` (1116 LOC)
sorumluluklara böl.

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| `epoch_callback` imzası | `v3/core/training_manager.py:423` | servise geri bildirim ([training-system](../training-system/README.md#4-i̇ç-mimari)) |
| checkpoint formatı | `v3/utils/checkpoint_manager.py` | eğitim devamlılığı |
| MoE aux loss toplama | `v3/core/loss_manager.py` | [neural-network §5.5](../neural-network/README.md#55-moe) |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D +
[roadmap izleme](../../development-roadmap.md#5-i̇zleme) (P0 temaları).

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 v2 taban + v3 boşluk | 21 test (v2) | ⏳ |
| A | T2 v3 çekirdek | `training_manager.py:423`, `training_loop.py:567` | ⏳ |
| A | T3 güvenlik/curriculum/bağımsızlık | `v3/safety/*`, `v3/optimizers/*`, `service_v3.py:506` | ⏳ |
| B | D1 kanonik karar | — | ⏳ |
| B | D2 ortak util | `v2/utils/*` → `common/` | ⏳ |
| B | D3 v3→v2 kes | `training_service_v3.py:506` | ⏳ |
| B | D4 v2 emeklilik | `v2/` | ⏳ |
| B | D5 dev sınıf ayrıştırma | `v3/core/training_manager.py:211` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
