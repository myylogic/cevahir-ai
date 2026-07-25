# Phase 1 — Refactoring Plan · Training System

> **Kanonik Birim #5 · Training System** — `training_system/` (v2+v3+kök)
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: v2/v3 birleştirme).
> Komşu: [training-management](../training-management/phase-1-refactoring-plan.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `training_system/` (kök + v2 + v3) |
| Boyut | ~8.320 LOC |
| Mevcut test | **6 dosya / ~2.325 LOC** — v2 servis + kök cache betikleri |
| Refactor hedefleri | v2/v3 servis ikizliği, v3→v2 bağımlılığı, karışık import, dağınık betik, cache formatı |
| Kritik sözleşmeler | cache formatı · kaynak-id-farkında split (sızıntı yok) · `ModelManager` ile init |

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
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **v3 servis test edilmiyor** | [§8](README.md#8-refactor-sinyalleri--tech-debt) | `TrainingServiceV3` yaşam döngüsü testi |
| T-02 | **Kaynak-id-farkında split sızıntı koruması** | [§5](README.md#5-veri--kontrol-akışı) | aynı kaynak train/val'e sızmıyor testi |
| T-03 | **v3→v2 bağımlılığı** — v3 servis v2 util'lere düşüyor | [§8](README.md#8-refactor-sinyalleri--tech-debt) | v3'ün v2 olmadan çalıştığı testi |
| T-04 | **Cache formatı** — `data_cache` ↔ `cache_v3` uyumu | [§8](README.md#8-refactor-sinyalleri--tech-debt) | cache yaz→oku round-trip |
| T-05 | **Config doğrulama** kapsamı | `config_validator.py` | geçersiz config yakalama |
| T-06 | **Health check** koşu öncesi | `health_check.py` | sağlık kontrolü testi |

### A.3 Olması Gereken Test Durumu
v3 servis için v2 ile denk kapsam; split sızıntı koruması ve cache round-trip
sözleşme testli; v3-bağımsızlık doğrulanmış.

### A.4 Test Sprint'leri
**T1** yeşil taban; kök `test_*_overlap.py`'yi pytest'e/`tests/`'e taşı.
**T2** v3 servis testleri + split sızıntı (T-01, T-02).
**T3** cache round-trip + config + v3-bağımsızlık (T-03..T-06).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
training_system/
  ├── service/  → tek kanonik TrainingService (v3 tabanlı)
  ├── data/     → cache + dataset + split (tek cache formatı, sürümlü)
  └── (kök betikler → scripts/ altına; prepare_cache tek giriş)
```

### B.2 Geliştirme Sprint'leri *(P0)*
**D1 — Kanonik servis kararı** *(karar)*: v3 servis kanonik.
**D2 — Sürüm seçimi config'e** *(düşük risk)*: `train.py`'nin v2-kesin/v3-try import
mantığını tek config bayrağına bağla. Koruyan: T-01.
**D3 — v3→v2 bağını kes** *(orta risk)*: Training Management D3 ile koordineli.
Önkoşul: T-03.
**D4 — Cache formatı sürümleme** *(orta risk)*: `data_cache`/`cache_v3` tek formatta
birleştir, sürüm damgası ekle. Önkoşul: T-04.
**D5 — Betik toplama** *(düşük risk)*: `debug_all_cache.py`, `clear_cache.py`,
`test_*_overlap.py` → `scripts/`/`tests/`.

### B.3 Korunacak Sözleşmeler
Cache formatı (eğitim uyumu); split sızıntı garantisi; `train.py` giriş noktası.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo +
[roadmap](../../development-roadmap.md#5-i̇zleme).

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 yeşil taban | ⏳ |
| A | T2 v3 servis + split | ⏳ |
| A | T3 cache/config/bağımsızlık | ⏳ |
| B | D1 kanonik servis | ⏳ |
| B | D2 sürüm config | ⏳ |
| B | D3 v3→v2 kes | ⏳ |
| B | D4 cache sürümleme | ⏳ |
| B | D5 betik toplama | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
