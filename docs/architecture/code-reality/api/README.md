# Code Reality — API

> Kanonik Birim #9 · Kaynak: `api/`
> REST erişim katmanının kodundan çıkarılmış mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L6), [search index](../../architecture-search-index.md).

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | API (REST erişim katmanı) |
| **Kaynak dizin** | `api/` |
| **Toplam boyut** | ~3.300 LOC (testler hariç) |
| **Composition root** | `api/app_factory.py` (`create_app`) |
| **Framework** | Flask (Blueprint, JWT auth) |
| **Aktif sürüm** | v3 route'ları (`api/routes/v3/`); v2 (`chat_routes_v2.py`) eski |
| **Çalışma zamanı** | Online (HTTP servis) |
| **Dış bağımlılık** | Model/Engine (`Cevahir`), Chatting (`ChattingManager`), Flask, JWT |

---

## 2. Sorumluluk

Cevahir motorunu **HTTP üzerinden** dışarı açmak: sohbet mesajı, oturum ve kullanıcı
uçları; kimlik doğrulama (JWT), güvenlik başlıkları, istek doğrulama, hata yönetimi,
istek-id/loglama, sağlık ve metrik uçları. Uygulama fabrikası tüm bağımlılıkları
kurar (Cevahir + ChattingManager + servisler + route'lar).

**Kapsam dışı:** İş mantığı (Chatting/Cognitive/Model), kalıcılık (Database).

---

## 3. Dosya Envanteri

### 3.1 Uygulama Çekirdeği

| Dosya | LOC | Rol | Anahtar üyeler |
|-------|-----|-----|----------------|
| `app_factory.py` | 398 | **Composition root** | `create_cevahir_instance`, `create_chatting_manager`, route init, blueprint kaydı |
| `app.py` | 236 | Uygulama giriş/çalıştırma | — |
| `api_config.py` | 251 | API konfigürasyonu | — |

### 3.2 Route'lar (`routes/`)

| Dosya | LOC | Rol |
|-------|-----|-----|
| `routes/v3/chat.py` | 141 | `POST/GET /v3/chat/messages` (`@require_auth`) |
| `routes/v3/sessions.py` | 129 | Oturum uçları |
| `routes/v3/users.py` | 147 | Kullanıcı uçları |
| `routes/v3/health.py` | 70 | Sağlık ucu |
| `routes/chat_routes_v2.py` | 298 | **Eski v2** sohbet route'ları |

### 3.3 Servisler (`services/`)

| Dosya | LOC | Sınıf | Rol |
|-------|-----|-------|-----|
| `chat_service.py` | 117 | `ChatService` | Route ↔ ChattingManager köprüsü |
| `session_service.py` | 97 | `SessionService` | Oturum iş mantığı |
| `user_service.py` | 100 | `UserService` | Kullanıcı iş mantığı |

### 3.4 Kesişen İlgiler

| Dizin | Dosyalar | Rol |
|-------|----------|-----|
| `middleware/` | `auth`, `error_handler`, `request_id`, `security`, `validator` | İstek boru hattı |
| `security/` | `jwt`, `password`, `headers` | Kimlik/güvenlik |
| `monitoring/` | `health`, `metrics` | Gözlemlenebilirlik |
| `utils/` | `response`, `logging`, `exceptions` | Standart yanıt/log |

---

## 4. İç Mimari

```
   HTTP istemci
     │  (JWT bearer)
     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Flask app  (app_factory.create_app)                          │
   │                                                               │
   │  Middleware: request_id → auth(JWT) → validator → security    │
   │              → error_handler                                  │
   │                                                               │
   │  Blueprint: /v3/*                                             │
   │   ┌──────────┐ ┌───────────┐ ┌─────────┐ ┌────────┐         │
   │   │ chat     │ │ sessions  │ │ users   │ │ health │         │
   │   └────┬─────┘ └─────┬─────┘ └────┬────┘ └────────┘         │
   │        │  init_*_routes(service)   │                         │
   │        ▼             ▼             ▼                         │
   │   ┌──────────┐ ┌───────────┐ ┌─────────┐                    │
   │   │ChatService│ │SessionSvc │ │UserSvc  │                    │
   │   └────┬─────┘ └─────┬─────┘ └────┬────┘                    │
   └────────┼─────────────┼────────────┼─────────────────────────┘
            ▼             ▼            ▼
        ChattingManager  [Chatting birimi] → Cevahir [Model/Engine]
```

**Composition root (`app_factory`):**
`create_cevahir_instance()` → `create_chatting_manager(cevahir)` → servisleri kur →
`init_chat_routes(chat_service)` vb. → blueprint kaydet.

---

## 5. Veri / Kontrol Akışı — `POST /v3/chat/messages`

```
istek (JWT) → @require_auth (middleware/auth + security/jwt)
   → validator (gövde doğrula)
   → chat.send_message() route
       → ChatService → ChattingManager.send_message()
           → cevahir.process()  [Model/Engine → Cognitive → NN]
   → utils/response ile standart JSON
   → (hata olursa) error_handler
```

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni uç | `routes/v3/` + `init_*_routes` + blueprint | servis enjeksiyonu deseni |
| Yeni servis | `services/` | route ↔ manager köprüsü |
| Auth değişikliği | `middleware/auth.py` + `security/jwt.py` | `@require_auth` sözleşmesi |
| Yeni middleware | `middleware/` + app_factory zinciri | sıra önemli |
| Metrik/sağlık | `monitoring/` | — |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** Model/Engine (`Cevahir`), Chatting (`ChattingManager`), Flask, JWT.
**Buna bağımlı olanlar:** dış HTTP istemcileri (birim en üstte).

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **v2/v3 route ikizliği** | `routes/chat_routes_v2.py` (298) yanında `routes/v3/chat.py` | Orta | v2 kaldırılabilir mi netleştirilmeli |
| **`service/` vs `services/`** | Boş `api/service/` **ve** dolu `api/services/` | Düşük | İsim karışıklığı; `service/` temizlenmeli |
| **Servis-manager örtüşmesi** | `api/services` ↔ `chatting_management` manager'ları | Orta | Sorumluluk sınırı (bkz. Chatting birimi §8) |
| **Cevahir'i app_factory'de kurma** | Ağır model uygulama başlangıcında yükleniyor | Orta | Lazy init / ayrı yaşam döngüsü |
| **DB entegrasyonu belirsiz** | API servisleri `database/` repository'lerini mi kullanıyor? | Orta | Kalıcılık yolu netleştirilmeli |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Composition root | `api/app_factory.py:56` (`create_cevahir_instance`) |
| ChattingManager kurulumu | `api/app_factory.py:123` |
| Chat route | `api/routes/v3/chat.py:29` (`send_message`) |
| Chat servisi | `api/services/chat_service.py` (`ChatService`) |
| Auth middleware | `api/middleware/auth.py` |
| JWT | `api/security/jwt.py` |
| Standart yanıt | `api/utils/response.py` |

---

*Kaynak: `api/` — analiz kodun mevcut halinden çıkarılmıştır.*
