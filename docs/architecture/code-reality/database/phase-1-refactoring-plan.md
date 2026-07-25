# Phase 1 — Refactoring Plan · Database

> **Kanonik Birim #10 · Database** — `database/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: çift kalıcılık).
> Komşu: [chatting](../chatting/phase-1-refactoring-plan.md), [api](../api/phase-1-refactoring-plan.md).
>
> **Bu sürüm derinleştirilmiştir:** `dosya:satır` çapaları + mimari kesit bağları.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `database/` (~1.985 LOC) |
| Mevcut test | **2 dosya / ~249 LOC** (`test_config.py`, `test_connection.py`) ⚠️ |
| Refactor hedefleri | **çift kalıcılık** (chatting storage ile), boş `schemas/`, entegrasyon belirsizliği, çift UoW tanımı |
| Kritik sözleşmeler | `Repository` CRUD arayüzü · `UnitOfWork` commit/rollback · model şeması |

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §2](../../master-architecture.md#2-katmanlı-görünüm-layered-view) (L0) | Kalıcılık katmanının yeri |
| [master-architecture §9](../../master-architecture.md#9-mimari-kararlar-ve-uygulanan-desenler) | Repository deseni |
| [master-architecture §10](../../master-architecture.md#10-bağımlılık-yönü-ve-kural-i̇hlalleri) | Çift kalıcılık kırılma noktası |
| [research §4](README.md#4-i̇ç-mimari) | Repository + UoW iç mimarisi |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri |
| [chatting planı](../chatting/phase-1-refactoring-plan.md) | Çift kalıcılık birleştirme (koordineli) |

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| Config | `tests/test_config.py` | ✅ Var |
| Connection | `tests/test_connection.py` | ✅ Var |
| Repository'ler, UnitOfWork, Migrations | — | ❌ Yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **Repository CRUD** | `repositories/base_repository.py:48` (`get_by_id`), `:68` (`get_all`), `:92` (`create`), `:112` (`update`), `:132` (`delete`); varlık repo'ları | [research §3.3](README.md#33-repositoryler-repositories) | varlık başına CRUD |
| **T-02** | **UnitOfWork** commit/rollback | `unit_of_work.py:28` (`UnitOfWork`), `:50` (`__enter__`), `:65` (`__exit__`), `:80` (`commit`) | [research §4](README.md#4-i̇ç-mimari) | atomiklik + rollback |
| **T-03** | **Migration** | `utils/migrations.py` | [research §3.4](README.md#34-yardımcılar-utils) | şema migration |
| **T-04** | **Entegrasyon / ölü kod** — API/Chatting kullanıyor mu | `repositories/*` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) · [api §8](../api/README.md#8-refactor-sinyalleri--tech-debt) | fiili kullanım tespiti |
| **T-05** | **Çift UoW tanımı** | `unit_of_work.py:28` vs `interfaces/unit_of_work.py` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | arayüz↔uygulama uyumu |

### A.3 Olması Gereken Test Durumu
Tüm repository'ler CRUD testli (T-01); UoW atomikliği (T-02); migration (T-03);
entegrasyon durumu (kullanılıyor/ölü) netleşir (T-04) — birleştirmenin girdisi.

### A.4 Test Sprint'leri
**T1** repository CRUD (T-01).
**T2** UoW + migration (T-02, T-03).
**T3** entegrasyon/ölü kod + çift UoW (T-04, T-05).

---

## B. GELİŞTİRME FAZI *(P0 — Chatting ile koordineli)*

### B.1 Hedef Mimari
```
database/  → tek kanonik kalıcılık
  ├── interfaces/  → Repository + UnitOfWork (tek tanım)
  ├── repositories/→ CRUD (chatting storage bunun implementasyonu)
  ├── schemas/     → DTO (doldurulur) veya kaldırılır
  └── models.py    → tipli modeller
```

### B.2 Geliştirme Sprint'leri
**D1 — Kanonik kalıcılık kararı** *(karar)*: `database/repositories` kanonik;
`chatting_management/storage` in-memory = test adaptörü ([chatting D1](../chatting/phase-1-refactoring-plan.md#b2-geliştirme-sprintleri) ile).
**D2 — Çift kalıcılık birleştirme** *(orta risk)*: aynı varlıkları tek kaynağa indir.
Önkoşul: T-01, T-04. **Kabul:** user/session/message tek kalıcılıkta.
**D3 — Çift UoW tekilleştirme** *(düşük risk)*: `unit_of_work.py:28` ↔
`interfaces/unit_of_work.py` arayüz/uygulama ayrımını netleştir. Önkoşul: T-05.
**D4 — `schemas/` kararı** *(düşük risk)*: boş `schemas/`'i doldur (DTO) veya kaldır.
**D5 — Ölü kod temizliği** *(düşük risk)*: kullanılmayan repository/yardımcı işaretle/kaldır.
Önkoşul: T-04.

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| `Repository` CRUD arayüzü | `interfaces/repository.py` | tüketiciler (API/Chatting) |
| `UnitOfWork` commit/rollback | `unit_of_work.py:80` | atomiklik |
| model şeması | `models.py` | veri bütünlüğü |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D +
[roadmap](../../development-roadmap.md#5-i̇zleme) (çift kalıcılık teması).

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 repository CRUD | `base_repository.py:48-132` | ⏳ |
| A | T2 UoW/migration | `unit_of_work.py:80`, `utils/migrations.py` | ⏳ |
| A | T3 entegrasyon/çift UoW | `repositories/*`, `interfaces/unit_of_work.py` | ⏳ |
| B | D1 kanonik karar | — | ⏳ |
| B | D2 çift kalıcılık birleştirme | `repositories/*` ↔ `chatting/storage` | ⏳ |
| B | D3 çift UoW tekilleştirme | `unit_of_work.py:28` | ⏳ |
| B | D4 schemas kararı | `schemas/` | ⏳ |
| B | D5 ölü kod temizliği | `repositories/*` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
