# Phase 1 — Refactoring Plan · API

> **Kanonik Birim #9 · API** — `api/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md).
> Komşu: [chatting](../chatting/phase-1-refactoring-plan.md), [database](../database/phase-1-refactoring-plan.md).
>
> **Bu sürüm derinleştirilmiştir:** `dosya:satır` çapaları + mimari kesit bağları.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `api/` (Flask) (~3.300 LOC) |
| Mevcut test | **1 dosya / 104 LOC** (`api/service/tests/test_chat_service.py`) ⚠️ |
| Refactor hedefleri | v2/v3 route ikizliği, `service/` vs `services/`, servis-manager örtüşmesi, ağır app_factory init, DB entegrasyonu belirsiz |
| Kritik sözleşmeler | v3 route sözleşmeleri · JWT auth · standart yanıt formatı |

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §2](../../master-architecture.md#2-katmanlı-görünüm-layered-view) (L6) | En üst erişim katmanı |
| [master-architecture §5](../../master-architecture.md#5-uçtan-uca-akış--çıkarım-inference) | İstek→process akışı |
| [research §4](README.md#4-i̇ç-mimari) + [§5](README.md#5-veri--kontrol-akışı--post-v3chatmessages) | app_factory wiring + istek akışı |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri |
| [chatting research §8](../chatting/README.md#8-refactor-sinyalleri--tech-debt) | servis/manager örtüşmesi |
| [database planı](../database/phase-1-refactoring-plan.md) | Kalıcılık yolu netleştirme |
| [model-engine research §4.1](../model-engine/README.md#41-facade-kompozisyonu-cevahir) | app_factory'nin Cevahir'i kurması (lazy init hedefi) |

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| ChatService | `api/service/tests/test_chat_service.py` | 🟡 Tek test |
| Route'lar (v3), Auth/JWT, Middleware | — | ❌ Yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **Route sözleşmeleri** (v3) | `routes/v3/chat.py:29` (`send_message`), `:92` (`get_messages`); `sessions.py`, `users.py`, `health.py` | [research §3.2](README.md#32-routelar-routes) | endpoint istek/yanıt sözleşmesi |
| **T-02** | **JWT auth** | `routes/v3/chat.py:30` (`@require_auth`); `middleware/auth.py`, `security/jwt.py` | [research §4](README.md#4-i̇ç-mimari) | yetkisiz erişim reddi |
| **T-03** | **İstek doğrulama** | `middleware/validator.py` | [research §3.4](README.md#34-kesişen-i̇lgiler) | geçersiz gövde reddi |
| **T-04** | **Hata yönetimi** | `middleware/error_handler.py`, `utils/response.py` | [research §3.4](README.md#34-kesişen-i̇lgiler) | standart hata formatı |
| **T-05** | **app_factory wiring** | `app_factory.py:56` (`create_cevahir_instance`), `:123` (`create_chatting_manager`) | [research §4](README.md#4-i̇ç-mimari) · [model-engine §4.1](../model-engine/README.md#41-facade-kompozisyonu-cevahir) | composition + blueprint kaydı |
| **T-06** | **v2 route durumu** | `routes/chat_routes_v2.py` (298) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | v2/v3 ayrım (kullanılıyor mu) |

### A.3 Olması Gereken Test Durumu
Tüm v3 route'lar sözleşme testli; auth/validation/error middleware kapsanır;
app_factory wiring doğrulanır (T-05); v2 route durumu netleşir (T-06).

### A.4 Test Sprint'leri
**T1** app_factory wiring + ChatService (T-05).
**T2** route sözleşmeleri + auth (T-01, T-02).
**T3** validation/error + v2/v3 ayrım (T-03, T-04, T-06).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
api/
  ├── app_factory.py  → lazy model init (ağır Cevahir yükü ertelenir)
  ├── routes/v3/      → tek kanonik route seti (v2 kaldırılır)
  ├── services/       → route ↔ manager köprüsü (boş service/ kaldırılır)
  └── (DB entegrasyon yolu netleştirilir)
```

### B.2 Geliştirme Sprint'leri
**D1 — Boş `service/` temizle** *(düşük risk)*: `api/service/` (boş) kaldır; test
`services/`'e. **Kabul:** tek `services/` kalır.
**D2 — v2 route emekliliği** *(düşük risk)*: `chat_routes_v2.py` kullanılmıyorsa kaldır.
Önkoşul: T-06.
**D3 — Servis/manager sınırı** *(orta risk)*: `api/services` ↔ `chatting_management`
örtüşmesini netleştir ([chatting D3](../chatting/phase-1-refactoring-plan.md#b2-geliştirme-sprintleri-p0--database-ile-koordineli) ile).
**D4 — Lazy model init** *(orta risk)*: `app_factory.py:56` Cevahir'i tembel yükle.
Önkoşul: T-05. **Kabul:** uygulama başlangıcı ağır model olmadan ayağa kalkar.
**D5 — DB entegrasyon netleştirme** *(orta risk)*: servislerin `database/` repository
kullanımını netleştir ([Database T-04](../database/phase-1-refactoring-plan.md#a2-kusur-envanteri) ile).

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| v3 route sözleşmeleri | `routes/v3/*` | dış istemciler |
| JWT auth akışı | `middleware/auth.py`, `security/jwt.py` | güvenlik |
| standart yanıt formatı | `utils/response.py` | istemci uyumu |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D.

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 wiring/ChatService | `app_factory.py:56/123` | ⏳ |
| A | T2 route/auth | `routes/v3/chat.py:29`, `middleware/auth.py` | ⏳ |
| A | T3 validation/error/v2v3 | `middleware/*`, `chat_routes_v2.py` | ⏳ |
| B | D1 boş service temizle | `api/service/` | ⏳ |
| B | D2 v2 route emekliliği | `chat_routes_v2.py` | ⏳ |
| B | D3 servis/manager sınırı | `services/*` ↔ `chatting_management` | ⏳ |
| B | D4 lazy model init | `app_factory.py:56` | ⏳ |
| B | D5 DB entegrasyon | `services/*` ↔ `database` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
