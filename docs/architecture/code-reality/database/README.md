# Code Reality — Database

> Kanonik Birim #10 · Kaynak: `database/`
> Kalıcılık katmanının kodundan çıkarılmış mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L0), [search index](../../architecture-search-index.md).

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Database (kalıcılık) |
| **Kaynak dizin** | `database/` |
| **Toplam boyut** | ~1.985 LOC (testler hariç) |
| **Desenler** | Repository + Unit of Work + tipli modeller |
| **Çalışma zamanı** | Online |
| **Dış bağımlılık** | ORM/DB sürücüsü (`connection.py` üzerinden), stdlib |

---

## 2. Sorumluluk

Kullanıcı, oturum, mesaj ve kullanıcı-belleği varlıklarını **kalıcı** olarak
saklamak: bağlantı yönetimi, tipli modeller, Repository ile CRUD soyutlaması,
Unit of Work ile işlem (transaction) sınırı, migration yardımcıları.

**Kapsam dışı:** İş mantığı (Chatting/API), vektör bellek (Cognitive'in ChromaDB'si
ayrıdır).

---

## 3. Dosya Envanteri

### 3.1 Çekirdek

| Dosya | LOC | Rol |
|-------|-----|-----|
| `connection.py` | 292 | DB bağlantı/oturum yönetimi |
| `models.py` | 292 | Tipli veri modelleri (User, Session, Message, UserMemory) |
| `unit_of_work.py` | 120 | İşlem sınırı (commit/rollback) |
| `config.py` | 149 | DB konfigürasyonu |
| `exceptions.py` | 49 | Hata hiyerarşisi |

### 3.2 Arayüzler (`interfaces/`)

| Dosya | LOC | Rol |
|-------|-----|-----|
| `repository.py` | 162 | `Repository` soyut arayüzü (CRUD sözleşmesi) |
| `unit_of_work.py` | 45 | `UnitOfWork` arayüzü |

### 3.3 Repository'ler (`repositories/`)

| Dosya | LOC | Rol |
|-------|-----|-----|
| `base_repository.py` | 181 | Ortak CRUD tabanı |
| `user_repository.py` | 79 | Kullanıcı kalıcılığı |
| `session_repository.py` | 115 | Oturum kalıcılığı |
| `message_repository.py` | 121 | Mesaj kalıcılığı |
| `user_memory_repository.py` | 110 | Kullanıcı-belleği kalıcılığı |

### 3.4 Yardımcılar (`utils/`)

| Dosya | LOC | Rol |
|-------|-----|-----|
| `migrations.py` | 120 | Şema migration |
| `helpers.py` | 40 | Yardımcılar |

> **Not:** `database/schemas/` dizini mevcut ama **boş** (`.py` yok).

---

## 4. İç Mimari

```
   Tüketici (API servisleri / Chatting — potansiyel)
        │  UnitOfWork ile
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  UnitOfWork  (unit_of_work.py — işlem sınırı)                  │
   │   with uow:  repo.add(...) ; uow.commit() / rollback()        │
   │                                                               │
   │   ┌──────────────┐ ┌───────────────┐ ┌───────────────────┐   │
   │   │ UserRepo     │ │ SessionRepo   │ │ MessageRepo       │   │
   │   │              │ │               │ │ UserMemoryRepo    │   │
   │   └──────┬───────┘ └───────┬───────┘ └────────┬──────────┘   │
   │          └──── BaseRepository (interfaces/repository.py) ─────┘
   │                          │                                    │
   │                          ▼                                    │
   │                   connection.py  (bağlantı/oturum)            │
   │                          │                                    │
   │                   models.py (User/Session/Message/UserMemory) │
   └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          Kalıcı DB
```

**Desenler:** Repository (varlık başına CRUD soyutlaması), Unit of Work (atomik
işlem), tipli Model (veri şeması). `interfaces/` protokolleri sayesinde repository
implementasyonu değiştirilebilir (DIP).

---

## 5. Veri / Kontrol Akışı

```
İş katmanı (ör. API SessionService)
   → UnitOfWork başlat
       → SessionRepository.add/get/update/delete   (BaseRepository CRUD)
           → connection üzerinden DB oturumu
           → models.Session ile eşle
   → uow.commit()  (hata → rollback)
```

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni varlık | `models.py` + yeni `repositories/*_repository.py` (BaseRepository'den) | `Repository` arayüzüne uy |
| DB backend değişimi | `connection.py` + config | Repository'ler değişmez (DIP) |
| Şema değişikliği | `models.py` + `utils/migrations.py` | migration ekle |
| İşlem stratejisi | `unit_of_work.py` | commit/rollback sözleşmesi |
| Şema doğrulama (DTO) | `schemas/` (şu an boş) | serileştirme katmanı |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** ORM/DB sürücüsü, stdlib.
**Buna bağımlı olan (potansiyel):** API servisleri, Chatting (kalıcı depolamaya
geçerse). **Not:** Mevcut durumda Chatting kendi in-memory `storage/`'ını kullanıyor;
bu birimle **entegrasyon derecesi netleştirilmelidir** (bkz. §8).

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **İki kalıcılık yaklaşımı** | `database/repositories` (kalıcı) **vs** `chatting_management/storage` (in-memory) | **Yüksek** | Aynı varlıklar (user/session/message) iki yerde; tek kaynağa indirgenmeli |
| **Boş `schemas/`** | dizin var, dosya yok | Düşük | Ya doldurulmalı ya kaldırılmalı |
| **Entegrasyon belirsizliği** | API/Chatting bu repository'leri fiilen kullanıyor mu? | Orta | Bağlanım izlenmeli; ölü kod riski |
| **Çift UnitOfWork tanımı** | `unit_of_work.py` (kök) + `interfaces/unit_of_work.py` | Düşük | arayüz/uygulama ayrımı teyit edilmeli |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Repository arayüzü | `database/interfaces/repository.py` (`Repository`) |
| CRUD tabanı | `database/repositories/base_repository.py` |
| Mesaj repository | `database/repositories/message_repository.py` |
| Unit of Work | `database/unit_of_work.py` |
| Modeller | `database/models.py` |
| Bağlantı | `database/connection.py` |
| Migration | `database/utils/migrations.py` |

---

*Kaynak: `database/` — analiz kodun mevcut halinden çıkarılmıştır.*
