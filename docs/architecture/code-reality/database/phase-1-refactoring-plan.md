# Phase 1 — Refactoring Plan · Database

> **Kanonik Birim #10 · Database** — `database/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: çift kalıcılık).
> Komşu: [chatting](../chatting/phase-1-refactoring-plan.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `database/` |
| Boyut | ~1.985 LOC |
| Mevcut test | **2 dosya / ~249 LOC** (`test_config.py`, `test_connection.py`) ⚠️ |
| Refactor hedefleri | **çift kalıcılık** (chatting storage ile), boş `schemas/`, entegrasyon belirsizliği, çift UoW tanımı |
| Kritik sözleşmeler | `Repository` CRUD arayüzü · `UnitOfWork` commit/rollback · model şeması |

> ⚠️ Yalnız config/connection test edilmiş; **repository'ler ve UoW test edilmemiş.**

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| Config | `tests/test_config.py` | ✅ Var |
| Connection | `tests/test_connection.py` | ✅ Var |
| Repository'ler (CRUD) | — | ❌ Yok |
| UnitOfWork | — | ❌ Yok |
| Migrations | — | ❌ Yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **Repository CRUD** (user/session/message/user_memory) | [§3.3](README.md#33-repositoryler-repositories) | varlık başına add/get/update/delete |
| T-02 | **UnitOfWork** commit/rollback | [§4](README.md#4-i̇ç-mimari) | atomiklik + rollback senaryosu |
| T-03 | **Migration** | [§3.4](README.md#34-yardımcılar-utils) | şema migration testi |
| T-04 | **Entegrasyon** — API/Chatting bunları kullanıyor mu | [§8](README.md#8-refactor-sinyalleri--tech-debt) | fiili kullanım / ölü kod tespiti |
| T-05 | **Çift UoW tanımı** tutarlılığı | [§8](README.md#8-refactor-sinyalleri--tech-debt) | arayüz↔uygulama uyumu |

### A.3 Olması Gereken Test Durumu
Tüm repository'ler CRUD testli; UoW atomikliği doğrulanır; migration test edilir;
entegrasyon durumu (kullanılıyor/ölü) netleşir.

### A.4 Test Sprint'leri
**T1** repository CRUD (T-01).
**T2** UoW + migration (T-02, T-03).
**T3** entegrasyon/ölü kod + çift UoW (T-04, T-05).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
database/  → tek kanonik kalıcılık
  ├── interfaces/  → Repository + UnitOfWork (tek tanım)
  ├── repositories/→ CRUD (chatting storage bunun bir implementasyonu olur)
  ├── schemas/     → DTO/serileştirme (doldurulur) veya kaldırılır
  └── models.py    → tipli modeller
```

### B.2 Geliştirme Sprint'leri *(P0 — Chatting ile koordineli)*
**D1 — Kanonik kalıcılık kararı** *(karar)*: `database/repositories` kanonik;
`chatting_management/storage` in-memory = test adaptörü. (Chatting D1/D2 ile.)
**D2 — Çift kalıcılık birleştirme** *(orta risk)*: aynı varlıkları tek kaynağa
indir. Önkoşul: T-01, T-04.
**D3 — Çift UoW tekilleştirme** *(düşük risk)*: kök `unit_of_work.py` ile
`interfaces/unit_of_work.py` arayüz/uygulama ayrımını netleştir. Önkoşul: T-05.
**D4 — `schemas/` kararı** *(düşük risk)*: boş `schemas/`'i ya doldur (DTO) ya kaldır.
**D5 — Ölü kod temizliği** *(düşük risk)*: kullanılmayan repository/yardımcıları
işaretle/kaldır. Önkoşul: T-04.

### B.3 Korunacak Sözleşmeler
`Repository` CRUD arayüzü; `UnitOfWork` commit/rollback; model şeması.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo.

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 repository CRUD | ⏳ |
| A | T2 UoW/migration | ⏳ |
| A | T3 entegrasyon/çift UoW | ⏳ |
| B | D1 kanonik karar | ⏳ |
| B | D2 çift kalıcılık birleştirme | ⏳ |
| B | D3 çift UoW tekilleştirme | ⏳ |
| B | D4 schemas kararı | ⏳ |
| B | D5 ölü kod temizliği | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
