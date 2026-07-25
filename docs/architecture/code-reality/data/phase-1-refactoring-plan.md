# Phase 1 — Refactoring Plan · Data

> **Kanonik Birim #6 · Data** — `data_loader_management/`, `data_processing/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `data_loader_management/` + `data_processing/` |
| Boyut | ~3.500 LOC |
| Mevcut test | **0 test** ⚠️ (birimin hiç testi yok) |
| Refactor hedefleri | tek dev loader, kaba token tahmini, bağlantısız alt-birimler, hardcoded yollar |
| Kritik sözleşmeler | yükleme çıktısı formatı (QA/text + file_index) · akıllı bölme değişmezleri |

> ⚠️ **En büyük test boşluğu:** Bu birimin **hiç testi yok.** Test Fazı sıfırdan
> kurulur; Geliştirme Fazı bunun üzerine gelir.

---

## A. TEST FAZI (sıfırdan)

### A.1 Test Reality
**Mevcut test: yok.** Kritik veri boru hattı (eğitim verisini besleyen) test
kalkanından yoksun. Öncelik en yüksek.

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **Akıllı bölme** test edilmemiş | [§5](README.md#5-veri--kontrol-akışı--akıllı-bölme) | `_smart_split_text` chunk sınırı + overlap |
| T-02 | **Tekilleştirme** doğruluğu | [§5](README.md#5-veri--kontrol-akışı--akıllı-bölme) | `_deduplicate_chunks` overlap tekrarı temizliği |
| T-03 | **Dosya okuyucular** (docx/json/qa/raw) | [§3.1](README.md#31-yükleme-data_loader_management) | her format için yükleme testi |
| T-04 | **file_index farkındalığı** | [§5](README.md#5-veri--kontrol-akışı--akıllı-bölme) | doğru kaynak indeksi (split sızıntısı için kritik) |
| T-05 | **Token tahmini sapması** | [§8](README.md#8-refactor-sinyalleri--tech-debt) | `_estimate_token_count` vs gerçek |
| T-06 | **Scraper ayrıştırıcılar** (SRT/VTT/Wikipedia) | [§3.2](README.md#32-toplamahazırlık-data_processing) | `parse_srt/parse_vtt` + scrape mock |
| T-07 | **Boş/bozuk girdi** | hata sınıfları | `EmptyTextError`/`UnsupportedFormatError` senaryoları |

### A.3 Olması Gereken Test Durumu
Yükleme boru hattının tamamı birim testli; akıllı bölme + tekilleştirme + file_index
değişmezleri korunur; scraper ayrıştırıcıları (ağ mock'lu) test edilir.

### A.4 Test Sprint'leri
**T1 — Loader çekirdeği:** T-01, T-02, T-04 (bölme/dedup/index — split sızıntısı
için kritik).
**T2 — Okuyucular + hatalar:** T-03, T-07.
**T3 — Processing + tahmin:** T-05, T-06 (scraper mock).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
data_loader_management/
  ├── loader.py        → DataLoaderManager (ince orkestrasyon)
  └── readers/         → DocxReader | JsonReader | QaReader | RawTextReader
                         (tek dev dosya → strateji sınıfları)
  splitter → tokenizer-hizalı token tahmini
data_processing/  → scraper'lar (CLI ↔ kütüphane ayrımı, config'li yollar)
```

### B.2 Geliştirme Sprint'leri
**D1 — Okuyucu ayrıştırma** *(orta risk)*: `data_loader_manager.py`'nin `_read_*` /
`_load_*` metotlarını `readers/` stratejilerine böl. Önkoşul: T-03.
**D2 — Token tahmini hizalama** *(düşük risk, davranış değişir → işaretli)*:
`_estimate_token_count`'ı gerçek tokenizer'a bağla. Önkoşul: T-05.
**D3 — Config'li yollar** *(düşük risk)*: hardcoded çıktı dizinlerini config'e taşı.
**D4 — CLI/kütüphane ayrımı** *(düşük risk)*: `data_processing` sınıf ↔ `main()`
betiği ayrımı; API anahtarı gizli yönetimi.

### B.3 Korunacak Sözleşmeler
Yükleme çıktı formatı (QA/text tuple + file_index); split sızıntı garantisi için
file_index doğruluğu.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo.

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 loader çekirdeği | ⏳ |
| A | T2 okuyucular/hatalar | ⏳ |
| A | T3 processing/tahmin | ⏳ |
| B | D1 okuyucu ayrıştırma | ⏳ |
| B | D2 token tahmini hizalama | ⏳ |
| B | D3 config'li yollar | ⏳ |
| B | D4 CLI/kütüphane ayrımı | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
