# Phase 1 — Refactoring Plan · API

> **Kanonik Birim #9 · API** — `api/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `api/` (Flask) |
| Boyut | ~3.300 LOC |
| Mevcut test | **1 dosya / 104 LOC** (`api/service/tests/test_chat_service.py`) ⚠️ |
| Refactor hedefleri | v2/v3 route ikizliği, `service/` vs `services/`, servis-manager örtüşmesi, ağır app_factory init, DB entegrasyonu belirsiz |
| Kritik sözleşmeler | route sözleşmeleri (v3) · JWT auth · standart yanıt formatı |

> ⚠️ Yalnız `chat_service` test edilmiş; **route'lar, auth, middleware test kalkanından yoksun.**

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| ChatService | `api/service/tests/test_chat_service.py` | 🟡 Tek test |
| Route'lar (v3) | — | ❌ Yok |
| Auth/JWT | — | ❌ Yok |
| Middleware | — | ❌ Yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **Route sözleşmeleri** (v3 chat/session/user/health) | [§3.2](README.md#32-routelar-routes) | endpoint istek/yanıt sözleşmesi |
| T-02 | **JWT auth** — `@require_auth` koruması | [§4](README.md#4-i̇ç-mimari) | yetkisiz erişim reddi |
| T-03 | **İstek doğrulama** | [§3.4](README.md#34-kesişen-i̇lgiler) | geçersiz gövde reddi |
| T-04 | **Hata yönetimi** standart yanıt | [§3.4](README.md#34-kesişen-i̇lgiler) | error_handler formatı |
| T-05 | **app_factory wiring** | [§4](README.md#4-i̇ç-mimari) | composition + blueprint kaydı |
| T-06 | **v2 route durumu** — hâlâ kullanılıyor mu | [§8](README.md#8-refactor-sinyalleri--tech-debt) | v2/v3 ayrım testi |

### A.3 Olması Gereken Test Durumu
Tüm v3 route'lar sözleşme testli; auth/validation/error middleware kapsanır;
app_factory wiring doğrulanır; v2 route'un durumu netleşir.

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
`services/` altına.
**D2 — v2 route emekliliği** *(düşük risk)*: `chat_routes_v2.py` kullanılmıyorsa
kaldır. Önkoşul: T-06.
**D3 — Servis/manager sınırı** *(orta risk)*: `api/services` ↔ `chatting_management`
örtüşmesini netleştir (Chatting D3 ile koordineli).
**D4 — Lazy model init** *(orta risk)*: `app_factory`'de Cevahir'i tembel yükle.
Önkoşul: T-05.
**D5 — DB entegrasyon netleştirme** *(orta risk)*: servislerin `database/`
repository'lerini kullanıp kullanmadığını netleştir (Database ile).

### B.3 Korunacak Sözleşmeler
v3 route sözleşmeleri; JWT auth akışı; standart yanıt formatı.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo.

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 wiring/ChatService | ⏳ |
| A | T2 route/auth | ⏳ |
| A | T3 validation/error/v2v3 | ⏳ |
| B | D1 boş service temizle | ⏳ |
| B | D2 v2 route emekliliği | ⏳ |
| B | D3 servis/manager sınırı | ⏳ |
| B | D4 lazy model init | ⏳ |
| B | D5 DB entegrasyon | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
