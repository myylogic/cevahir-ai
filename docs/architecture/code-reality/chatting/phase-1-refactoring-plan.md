# Phase 1 — Refactoring Plan · Chatting

> **Kanonik Birim #8 · Chatting** — `chatting_management/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: çift kalıcılık).
> Komşu: [database](../database/phase-1-refactoring-plan.md), [api](../api/phase-1-refactoring-plan.md).
>
> **Bu sürüm derinleştirilmiştir:** `dosya:satır` çapaları + mimari kesit bağları.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `chatting_management/` (~1.917 LOC) |
| Mevcut test | **0 test** ⚠️ |
| Refactor hedefleri | **çift kalıcılık** (storage vs database), manager/servis örtüşmesi, in-memory kalıcılık |
| Kritik sözleşmeler | `send_message` akışı · oturum yaşam döngüsü · `Cevahir.process` çağrısı |

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §2](../../master-architecture.md#2-katmanlı-görünüm-layered-view) (L6) | Servis/erişim katmanı |
| [master-architecture §5](../../master-architecture.md#5-uçtan-uca-akış--çıkarım-inference) | `send_message`→`process` akışı |
| [research §4](README.md#4-i̇ç-mimari) + [§5](README.md#5-veri--kontrol-akışı--send_message) | İç mimari + akış |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri (çift kalıcılık) |
| [database planı](../database/phase-1-refactoring-plan.md) | Kalıcılık birleştirme (koordineli) |
| [api research §8](../api/README.md#8-refactor-sinyalleri--tech-debt) | manager/servis örtüşmesi |
| [model-engine research §4](../model-engine/README.md#4-i̇ç-mimari) | `Cevahir.process` çağrı sözleşmesi |

---

## A. TEST FAZI (sıfırdan)

### A.1 Test Reality
**Mevcut test: yok.** Oturum/konuşma/bağlam mantığı test kalkanından yoksun.

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **Oturum yaşam döngüsü** | `chatting_manager.py:251` (`create_session`), `:284` (`list_sessions`); `components/session_manager.py` | [research §3.1](README.md#31-bileşenler-components) | create/list/get |
| **T-02** | **Konuşma ekleme/geçmiş** | `chatting_manager.py:218` (`get_conversation_history`); `components/conversation_manager.py` | [research §5](README.md#5-veri--kontrol-akışı--send_message) | append + history sırası/limiti |
| **T-03** | **Bağlam inşası** | `components/context_builder.py` | [research §3.1](README.md#31-bileşenler-components) | prompt bağlamı |
| **T-04** | **`send_message` akışı** (Cevahir mock) | `chatting_manager.py:111` (`send_message`) | [research §5](README.md#5-veri--kontrol-akışı--send_message) · [model-engine §4](../model-engine/README.md#4-i̇ç-mimari) | uçtan uca (mock backend) |
| **T-05** | **In-memory kalıcılık** | `storage/memory_storage.py`, `session/conversation/user_storage.py` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | CRUD + izolasyon |

### A.3 Olması Gereken Test Durumu
Oturum/konuşma/kullanıcı/bağlam birim testli; `send_message` mock backend ile uçtan
uca test edilir (T-04); depolama davranışı (kalıcı geçişe hazırlık) doğrulanır (T-05).

### A.4 Test Sprint'leri
**T1** oturum + konuşma + storage (T-01, T-02, T-05).
**T2** bağlam + send_message (T-03, T-04).

---

## B. GELİŞTİRME FAZI *(P0 — Database ile koordineli)*

### B.1 Hedef Mimari
```
chatting_management/
  ├── managers/  → session/conversation/user + context builder
  └── storage → Database Repository arayüzünün implementasyonu
                (in-memory yalnız test/geliştirme adaptörü)
  (api/services ile sorumluluk sınırı net)
```

### B.2 Geliştirme Sprint'leri
**D1 — Depolama arayüzü** *(orta risk)*: `storage/` sınıflarını Database `Repository`
arayüzü arkasına al. Önkoşul: T-05 + [Database T-01](../database/phase-1-refactoring-plan.md#a2-kusur-envanteri).
**D2 — Çift kalıcılık birleştirme** *(orta risk)*: kalıcı yol `database/repositories`;
oturum/kullanıcı/mesaj tek kaynağa iner ([Database D2](../database/phase-1-refactoring-plan.md#b2-geliştirme-sprintleri-p0--chatting-ile-koordineli) ile birlikte).
**D3 — Manager/servis sınırı** *(düşük risk)*: `chatting_management` manager'ları ↔
`api/services` örtüşmesini kaldır ([api D3](../api/phase-1-refactoring-plan.md#b2-geliştirme-sprintleri) ile).

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| `send_message`/`create_session`/`get_conversation_history` imzaları | `chatting_manager.py:111/251/218` | API + terminal bağlı |
| `Cevahir.process` çağrısı | `chatting_manager.py:111` | [model-engine §4](../model-engine/README.md#4-i̇ç-mimari) |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D +
[roadmap](../../development-roadmap.md#5-i̇zleme) (çift kalıcılık teması).

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 oturum/konuşma/storage | `chatting_manager.py:251/218`, `storage/*` | ⏳ |
| A | T2 bağlam/send_message | `context_builder.py`, `chatting_manager.py:111` | ⏳ |
| B | D1 depolama arayüzü | `storage/*` | ⏳ |
| B | D2 çift kalıcılık birleştirme | `storage/*` ↔ `database/repositories` | ⏳ |
| B | D3 manager/servis sınırı | `components/*` ↔ `api/services` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
