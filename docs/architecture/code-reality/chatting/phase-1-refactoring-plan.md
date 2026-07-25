# Phase 1 — Refactoring Plan · Chatting

> **Kanonik Birim #8 · Chatting** — `chatting_management/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P0: çift kalıcılık).
> Komşu: [database](../database/phase-1-refactoring-plan.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `chatting_management/` |
| Boyut | ~1.917 LOC |
| Mevcut test | **0 test** ⚠️ |
| Refactor hedefleri | **çift kalıcılık** (storage vs database), manager/servis örtüşmesi, in-memory kalıcılık |
| Kritik sözleşmeler | `send_message` akışı · oturum yaşam döngüsü · `Cevahir.process` çağrısı |

> ⚠️ **Test yok.** Test Fazı sıfırdan kurulur.

---

## A. TEST FAZI (sıfırdan)

### A.1 Test Reality
**Mevcut test: yok.** Oturum/konuşma/bağlam mantığı test kalkanından yoksun.

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **Oturum yaşam döngüsü** | [§3.1](README.md#31-bileşenler-components) | create/list/get session |
| T-02 | **Konuşma ekleme/geçmiş** | [§5](README.md#5-veri--kontrol-akışı--send_message) | append + history sırası/limiti |
| T-03 | **Bağlam inşası** | [§3.1](README.md#31-bileşenler-components) | `ContextBuilder` prompt bağlamı |
| T-04 | **`send_message` akışı** (Cevahir mock) | [§5](README.md#5-veri--kontrol-akışı--send_message) | uçtan uca akış (mock backend) |
| T-05 | **In-memory kalıcılık davranışı** | [§8](README.md#8-refactor-sinyalleri--tech-debt) | storage CRUD + izolasyon |

### A.3 Olması Gereken Test Durumu
Oturum/konuşma/kullanıcı/bağlam birim testli; `send_message` mock backend ile
uçtan uca test edilir; depolama davranışı (kalıcı geçişe hazırlık) doğrulanır.

### A.4 Test Sprint'leri
**T1** oturum + konuşma + storage (T-01, T-02, T-05).
**T2** bağlam + send_message akışı (T-03, T-04).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
chatting_management/
  ├── managers/  → session/conversation/user + context builder
  └── storage → Database repository arayüzünün bir implementasyonu
                (in-memory yalnız test/geliştirme adaptörü)
  (api/services ile sorumluluk sınırı net)
```

### B.2 Geliştirme Sprint'leri *(P0 — Database ile koordineli)*
**D1 — Depolama arayüzü** *(orta risk)*: `storage/` sınıflarını Database `Repository`
arayüzünün arkasına al; in-memory = test adaptörü. Önkoşul: T-05 + Database T-01.
**D2 — Çift kalıcılık birleştirme** *(orta risk)*: kalıcı yol `database/repositories`
olsun; oturum/kullanıcı/mesaj tek kaynağa iner. (Database D1 ile birlikte.)
**D3 — Manager/servis sınırı** *(düşük risk)*: `chatting_management` manager'ları ile
`api/services` sorumluluğunu netleştir; örtüşmeyi kaldır.

### B.3 Korunacak Sözleşmeler
`send_message`/`create_session`/`get_conversation_history` genel imzaları;
`Cevahir.process` çağrı sözleşmesi.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo.

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 oturum/konuşma/storage | ⏳ |
| A | T2 bağlam/send_message | ⏳ |
| B | D1 depolama arayüzü | ⏳ |
| B | D2 çift kalıcılık birleştirme | ⏳ |
| B | D3 manager/servis sınırı | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
