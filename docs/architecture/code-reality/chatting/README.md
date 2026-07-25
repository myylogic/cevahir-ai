# Code Reality — Chatting

> Kanonik Birim #8 · Kaynak: `chatting_management/`
> Sohbet oturumu / geçmiş yönetiminin kodundan çıkarılmış mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L6), [search index](../../architecture-search-index.md).

> 🔧 **Eşleşen plan:** Bu research ile birlikte okunacak test+geliştirme planı →
> [`phase-1-refactoring-plan.md`](phase-1-refactoring-plan.md)

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Chatting (oturum/sohbet) |
| **Kaynak dizin** | `chatting_management/` |
| **Toplam boyut** | ~1.917 LOC |
| **Facade** | `ChattingManager` — `chatting_management/chatting_manager.py:64` |
| **Çağıran** | API (`app_factory.create_chatting_manager(cevahir)`) ve terminal `chat_pipeline` |
| **Çalışma zamanı** | Online |
| **Dış bağımlılık** | Model/Engine (`Cevahir`) |

---

## 2. Sorumluluk

Kullanıcı ile model arasındaki **konuşma oturumunu** yönetmek: oturum oluşturma/
listeleme, mesaj gönderme (Cevahir'e devredip yanıt alma), konuşma geçmişi tutma,
kullanıcı bağlamı ve prompt bağlamı inşası. Depolama soyutlaması (in-memory) içerir.

**Kapsam dışı:** Model üretimi (Model/Engine), akıl yürütme (Cognitive), REST
taşıma (API), kalıcı veritabanı (Database).

---

## 3. Dosya Envanteri

| Dosya | LOC | Sınıf/rol | Anahtar üyeler |
|-------|-----|-----------|----------------|
| `chatting_manager.py` | 310 | **`ChattingManager`** (Facade) | `send_message`, `get_conversation_history`, `create_session`, `list_sessions` |
| `config.py` | 131 | `ChattingConfig` | — |
| `exceptions.py` | 77 | Hata hiyerarşisi | — |

### 3.1 Bileşenler (`components/`)

| Dosya | LOC | Sınıf | Rol |
|-------|-----|-------|-----|
| `session_manager.py` | 99 | `SessionManager` | Oturum yaşam döngüsü |
| `user_manager.py` | 100 | `UserManager` | Kullanıcı yönetimi |
| `conversation_manager.py` | 103 | `ConversationManager` | Konuşma/mesaj akışı |
| `context_builder.py` | 235 | `ContextBuilder` | Geçmiş → prompt bağlamı |

### 3.2 Depolama (`storage/`)

| Dosya | LOC | Rol |
|-------|-----|-----|
| `memory_storage.py` | 185 | Genel in-memory depo tabanı |
| `session_storage.py` | 202 | Oturum deposu |
| `conversation_storage.py` | 198 | Konuşma/mesaj deposu |
| `user_storage.py` | 185 | Kullanıcı deposu |

---

## 4. İç Mimari

```
   API / terminal chat_pipeline
        │  ChattingManager(cevahir, config)
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  ChattingManager  (Facade)                                    │
   │  send_message / create_session / history / list_sessions      │
   │                                                               │
   │   ┌──────────────┐ ┌───────────────┐ ┌───────────────────┐   │
   │   │ SessionMgr   │ │ ConversationMgr│ │ UserMgr           │   │
   │   └──────┬───────┘ └───────┬───────┘ └────────┬──────────┘   │
   │          │                 │  ContextBuilder   │              │
   │          ▼                 ▼                   ▼              │
   │   ┌──────────────┐ ┌───────────────┐ ┌───────────────────┐   │
   │   │ SessionStore │ │ ConversationSt│ │ UserStore         │   │  (in-memory)
   │   └──────────────┘ └───────────────┘ └───────────────────┘   │
   └───────────────────────────────┬──────────────────────────────┘
                                   │ send_message → cevahir.process()
                                   ▼
                          Cevahir  [Model/Engine → Cognitive]
```

---

## 5. Veri / Kontrol Akışı — `send_message`

```
send_message(user_id, session_id, text)
   │  SessionManager.get/create
   │  ConversationManager.append(user turn)
   │  ContextBuilder.build(history + user context) → prompt bağlamı
   ▼
cevahir.process(prompt)          [Model/Engine → Cognitive → NN]
   │  yanıt
   ▼
ConversationManager.append(assistant turn) → storage
   ▼
yanıt döndür
```

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Kalıcı depolama | `storage/*` yerine Database repository'leri | in-memory → DB geçişi (bkz. §8) |
| Bağlam stratejisi | `context_builder.py` | budama/özetleme |
| Oturum politikası | `session_manager.py` | TTL/limit |
| Çok kullanıcılı akış | `user_manager.py` | yetki |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** Model/Engine (`Cevahir`).
**Buna bağımlı olanlar:** API (`ChatService` → `ChattingManager`), terminal
`chat_pipeline`.

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **Çift kalıcılık modeli** | `chatting_management/storage` (in-memory) **vs** `database/repositories` (kalıcı) | **Yüksek** | İki depolama yaklaşımı; oturum/kullanıcı/mesaj her ikisinde de var — birleştirilmeli |
| **Manager/servis örtüşmesi** | `chatting_management` manager'ları ↔ `api/services` (session/user/chat) | Orta | Sorumluluk sınırı netleştirilmeli |
| **In-memory kalıcılık** | `memory_storage.py` süreç ölünce veri kaybı | Orta | Üretimde DB'ye taşınmalı |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Mesaj gönderme | `chatting_management/chatting_manager.py:111` (`send_message`) |
| Oturum oluşturma | `chatting_management/chatting_manager.py:251` (`create_session`) |
| Geçmiş | `chatting_management/chatting_manager.py:218` |
| Bağlam inşası | `chatting_management/components/context_builder.py` |
| Depolama tabanı | `chatting_management/storage/memory_storage.py` |

---

*Kaynak: `chatting_management/` — analiz kodun mevcut halinden çıkarılmıştır.*
