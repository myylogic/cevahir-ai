# Phase 1 — Refactoring Plan · Training System

> **Kanonik Birim #5 · Training System** — `training_system/` (v2+v3+kök)
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: v2/v3 birleştirme).
> Komşu: [training-management](../training-management/phase-1-refactoring-plan.md).
>
> **Bu sürüm derinleştirilmiştir:** `dosya:satır` çapaları + mimari kesit bağları.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `training_system/` (kök + v2 + v3) (~8.320 LOC) |
| Mevcut test | **6 dosya / ~2.325 LOC** — v2 servis + kök cache betikleri |
| Refactor hedefleri | v2/v3 servis ikizliği, v3→v2 bağımlılığı, karışık import, dağınık betik, cache formatı |
| Kritik sözleşmeler | cache formatı · kaynak-id-farkında split (sızıntı yok) · `ModelManager` ile init |

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §6](../../master-architecture.md#6-uçtan-uca-akış--eğitim-training) | Eğitim yolu (bu birim servis/koşu katmanı) |
| [master-architecture §7](../../master-architecture.md#7-çalışma-zamanı-yolları-özeti) | `train.py` giriş noktası (facade dışı) |
| [research §4](README.md#4-i̇ç-mimari) + [§5](README.md#5-veri--kontrol-akışı) | Servis akışı + split |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri |
| [training-management planı](../training-management/phase-1-refactoring-plan.md) | Çağrılan motor (koordineli birleştirme) |
| [data research §5](../data/README.md#5-veri--kontrol-akışı--akıllı-bölme) | file_index (kaynak-farkında split girdisi) |
| [model-engine research §4.3](../model-engine/README.md#43-model-yaşam-döngüsü-modelmanager) | `ModelManager` ile model init |

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| v2 servis | `v2/test/test_training_service_comprehensive.py` (402), `test/test_training_service.py` (306) | ✅ Var |
| Tokenizer eğitim entegrasyon | `test/test_tokenizer_training_comprehensive.py` (1017) | ✅ Var |
| Cache overlap | `test_overlap.py`, `test_cache_overlap.py` (kök) | 🟡 Betik |
| v2 import | `v2/test/test_imports.py` | ✅ Var |
| **v3 servis** | — | ❌ Test yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **v3 servis testsiz** | `v3/core/training_service_v3.py:70`, `:408` (`train`), `:212` (`load_data_from_cache`) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | `TrainingServiceV3` yaşam döngüsü |
| **T-02** | **Kaynak-id-farkında split sızıntı** | `v3/core/training_service_v3.py:288` (`_source_id_aware_split`), `:315` (`_split_by_source_id`) | [research §5](README.md#5-veri--kontrol-akışı) · [data §5](../data/README.md#5-veri--kontrol-akışı--akıllı-bölme) | aynı kaynak train/val'e sızmıyor |
| **T-03** | **v3→v2 bağımlılığı** | `v3/core/training_service_v3.py:506` (v2 util import) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | v3'ün v2 olmadan çalıştığı testi |
| **T-04** | **Cache formatı** round-trip | `data_cache.py`, `v3/data/cache_v3.py` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | yaz→oku eşdeğerlik |
| **T-05** | **Config doğrulama** | `config_validator.py` | — | geçersiz config yakalama |
| **T-06** | **Health check** | `health_check.py` | — | koşu öncesi sağlık kontrolü |

### A.3 Olması Gereken Test Durumu
v3 servis için v2 ile denk kapsam; split sızıntı koruması (T-02) ve cache round-trip
(T-04) sözleşme testli; v3-bağımsızlık (T-03) doğrulanmış — birleştirmenin önkoşulu.

### A.4 Test Sprint'leri
**T1** yeşil taban; kök `test_*_overlap.py`'yi `tests/`'e taşı.
**T2** v3 servis (T-01) + split sızıntı (T-02).
**T3** cache round-trip + config + v3-bağımsızlık (T-03..T-06).

---

## B. GELİŞTİRME FAZI *(P0)*

### B.1 Hedef Mimari
```
training_system/
  ├── service/  → tek kanonik TrainingService (v3 tabanlı)
  ├── data/     → cache + dataset + split (tek cache formatı, sürümlü)
  └── (kök betikler → scripts/; prepare_cache tek giriş)
```

### B.2 Geliştirme Sprint'leri
**D1 — Kanonik servis kararı** *(karar)*: v3 servis kanonik.
**D2 — Sürüm seçimi config'e** *(düşük risk)*: `train.py:1`'in v2-kesin/v3-try import
mantığını tek config bayrağına bağla. Önkoşul: T-01. **Kabul:** sürüm config'ten seçilir.
**D3 — v3→v2 bağını kes** *(orta risk)*: `training_service_v3.py:506` → `common/`
(**Training Management D3 ile koordineli**). Önkoşul: T-03.
**D4 — Cache formatı sürümleme** *(orta risk)*: `data_cache`/`cache_v3` tek formatta +
sürüm damgası. Önkoşul: T-04. **Kabul:** tek format; eğitim uyumu korunur.
**D5 — Betik toplama** *(düşük risk)*: `debug_all_cache.py`, `clear_cache.py`,
`test_*_overlap.py` → `scripts/`/`tests/`.

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| Cache formatı | `data_cache.py`/`cache_v3.py` | eğitim uyumu |
| Split sızıntı garantisi | `training_service_v3.py:288` | veri sızıntısı koruması |
| `train.py` giriş noktası | `train.py:1` | [master §7](../../master-architecture.md#7-çalışma-zamanı-yolları-özeti) |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D +
[roadmap](../../development-roadmap.md#5-i̇zleme).

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 yeşil taban | 6 test | ⏳ |
| A | T2 v3 servis + split | `training_service_v3.py:70/288` | ⏳ |
| A | T3 cache/config/bağımsızlık | `cache_v3.py`, `config_validator.py`, `:506` | ⏳ |
| B | D1 kanonik servis | — | ⏳ |
| B | D2 sürüm config | `train.py:1` | ⏳ |
| B | D3 v3→v2 kes | `training_service_v3.py:506` | ⏳ |
| B | D4 cache sürümleme | `data_cache.py`/`cache_v3.py` | ⏳ |
| B | D5 betik toplama | kök `*.py` betikleri | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
