# Phase 1 — Refactoring Plan · Training Management

> **Kanonik Birim #4 · Training Management** — `training_management/` (v2+v3)
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: v2/v3 birleştirme).
> Komşu: [training-system](../training-system/phase-1-refactoring-plan.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `training_management/v2/`, `training_management/v3/` |
| Boyut | ~18.300 LOC |
| Mevcut test | **21 dosya / ~3.256 LOC — tamamı v2** (`v2/test/core`, `v2/test/integration`) |
| Refactor hedefleri | **v2/v3 ikizliği**, v3→v2 sızıntısı, dev sınıflar, örtüşen izleme |
| Kritik sözleşmeler | epoch döngüsü kontratı · checkpoint · `epoch_callback` · MoE aux loss |

> ⚠️ **En kritik test boşluğu:** v3 (kanonik hedef) **test edilmiyor**; testlerin
> hepsi v2'de. Birleştirmeden önce v3 test edilmeli.

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| v2 core | `test_gradient_manager`, `test_loss_computation`, `test_training_loop` | ✅ Var |
| v2 integration | `test_training_manager_full`, `test_comprehensive_training`, `test_critical_integrations`, `test_silent_error_scenarios` | ✅ İyi |
| **v3 (her şey)** | — | ❌ **Test yok** |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **v3 test kapsamı sıfır** | [§8](README.md#8-refactor-sinyalleri--tech-debt) | v3 `TrainingManager`, loop, loss/gradient için test |
| T-02 | **v3→v2 sızıntısı** — v3 hâlâ v2 util import ediyor | [§8](README.md#8-refactor-sinyalleri--tech-debt) | v3'ün v2'siz çalıştığını doğrulayan test |
| T-03 | **Güvenlik dedektörleri** — NaN/loss-spike/divergence gerçekten tetikliyor mu | [§3.1 safety](README.md#31-v3-güncel--training_managementv3) | dedektör tetikleme senaryoları |
| T-04 | **Checkpoint verify** doğruluğu | [§3.1](README.md#31-v3-güncel--training_managementv3) | bozuk checkpoint yakalama |
| T-05 | **Early stopping + best takibi** | [§5](README.md#5-kontrol-akışı-bir-epoch) | `_EarlyStopper` davranışı |
| T-06 | **Curriculum/optimizer** (SAM/Lookahead/EMA) doğruluğu | [§3.1](README.md#31-v3-güncel--training_managementv3) | strateji birim testleri |

### A.3 Olması Gereken Test Durumu
v3 için v2 ile denk kapsam; güvenlik dedektörleri, curriculum, optimizer'lar birim
testli; v3'ün v2'den bağımsız çalıştığı entegrasyon testi.

### A.4 Test Sprint'leri
**T1** v2 yeşil taban + v3 kapsam boşluğu haritası.
**T2** v3 çekirdek testleri (loop/loss/gradient) — T-01.
**T3** güvenlik/curriculum/optimizer + v3-bağımsızlık — T-02..T-06.

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
training_management/
  ├── common/   → paylaşılan util (checkpoint, tensorboard, logger)  [v2'den taşınır]
  └── core/     → tek kanonik TrainingManager (v3 tabanlı)
                  · loop · loss · gradient · safety · curriculum · optimizers
  (v2 → _archive veya kaldır)
```

### B.2 Geliştirme Sprint'leri *(P0 — en yüksek öncelik)*
**D1 — Kanonik sürüm kararı** *(karar)*: v3 kanonik (curriculum/optimizers/safety
zengin). Belgelenir.
**D2 — Ortak util çıkarma** *(orta risk)*: v2 util'lerini (`checkpoint_manager`,
`tensorboard_manager`, `training_logger`) `common/`'a taşı. Önkoşul: T-01/T-02.
**D3 — v3→v2 bağını kes** *(orta risk)*: `training_service_v3.py:506` import'larını
`common/`'a yönlendir. Koruyan: v3-bağımsızlık testi (T-02).
**D4 — v2 emekliye ayır** *(orta risk)*: v2'yi `_archive` veya kaldır (Training
System v2 ile koordineli).
**D5 — Dev sınıf ayrıştırma** *(orta risk)*: `v3/core/training_manager.py` (1116)
sorumluluklara böl.

### B.3 Korunacak Sözleşmeler
`epoch_callback` imzası; checkpoint formatı; `ModelManager` ile eğitim arayüzü.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo +
[roadmap durum tablosu](../../development-roadmap.md#5-i̇zleme).

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 v2 taban + v3 boşluk | ⏳ |
| A | T2 v3 çekirdek | ⏳ |
| A | T3 güvenlik/curriculum/bağımsızlık | ⏳ |
| B | D1 kanonik karar | ⏳ |
| B | D2 ortak util | ⏳ |
| B | D3 v3→v2 kes | ⏳ |
| B | D4 v2 emeklilik | ⏳ |
| B | D5 dev sınıf ayrıştırma | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
