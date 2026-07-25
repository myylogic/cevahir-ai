# Phase 1 — Refactoring Plan · Data

> **Kanonik Birim #6 · Data** — `data_loader_management/`, `data_processing/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md).
>
> **Bu sürüm derinleştirilmiştir:** `dosya:satır` çapaları + mimari kesit bağları.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `data_loader_management/` + `data_processing/` (~3.500 LOC) |
| Mevcut test | **0 test** ⚠️ |
| Refactor hedefleri | tek dev loader, kaba token tahmini, bağlantısız alt-birimler, hardcoded yollar |
| Kritik sözleşmeler | yükleme çıktı formatı (QA/text + file_index) · akıllı bölme değişmezleri |

> ⚠️ **En büyük test boşluğu:** birimin hiç testi yok. Test Fazı sıfırdan kurulur.

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §2](../../master-architecture.md#2-katmanlı-görünüm-layered-view) (L0) | Veri katmanının yeri |
| [master-architecture §6](../../master-architecture.md#6-uçtan-uca-akış--eğitim-training) | Veri hazırlığı → eğitim yolu |
| [research §5](README.md#5-veri--kontrol-akışı--akıllı-bölme) | Akıllı bölme + file_index (hedefin temeli) |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri |
| [training-system research §5](../training-system/README.md#5-veri--kontrol-akışı) | file_index'in kaynak-farkında split'te kullanımı (tüketici) |
| [tokenizer research §7](../tokenizer/README.md#7-bağımlılıklar) | Token tahmininin gerçek tokenizer'a hizalanması |

---

## A. TEST FAZI (sıfırdan)

### A.1 Test Reality
**Mevcut test: yok.** Eğitim verisini besleyen kritik boru hattı test kalkanından
yoksun → en yüksek öncelik.

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **Akıllı bölme** | `data_loader_manager.py:130` (`_smart_split_text`), `:240` (`_split_by_words`), `:270` (`_get_overlap_text`) | [research §5](README.md#5-veri--kontrol-akışı--akıllı-bölme) | chunk sınırı + overlap doğruluğu |
| **T-02** | **Tekilleştirme** | `data_loader_manager.py:199` (`_deduplicate_chunks`) | [research §5](README.md#5-veri--kontrol-akışı--akıllı-bölme) | overlap tekrarı temizliği |
| **T-03** | **Dosya okuyucular** | `:366` (`_load_qa_pairs`), `:519` (`_load_text_inputs`), `:552` (`_load_raw_text_chunks`), `:723` (`_read_docx`), `:691` (`_read_json`) | [research §3.1](README.md#31-yükleme-data_loader_management) | her format için yükleme |
| **T-04** | **file_index farkındalığı** | `:344` (`load_with_file_index`), `:420/:490/:588` (index'li varyantlar) | [training-system §5](../training-system/README.md#5-veri--kontrol-akışı) | doğru kaynak indeksi (split sızıntısı için kritik) |
| **T-05** | **Token tahmini sapması** | `:112` (`_estimate_token_count`) | [tokenizer §7](../tokenizer/README.md#7-bağımlılıklar) | tahmin vs gerçek token sapması |
| **T-06** | **Scraper ayrıştırıcılar** | `data_processing/subtitle_processor.py:57` (`parse_srt`), `:76` (`parse_vtt`); `wikipedia_api_scraper.py:108` (`scrape_pages`) | [research §3.2](README.md#32-toplamahazırlık-data_processing) | SRT/VTT + scrape (ağ mock) |
| **T-07** | **Boş/bozuk girdi** | hata sınıfları `:57-61` (`EmptyTextError`, `UnsupportedFormatError`, ...) | [research §3.1](README.md#31-yükleme-data_loader_management) | hata senaryoları |

### A.3 Olması Gereken Test Durumu
Yükleme boru hattı tam birim testli; akıllı bölme + tekilleştirme + file_index
değişmezleri korunur (T-04 kaynak-farkında split için **kritik**); scraper
ayrıştırıcıları ağ-mock'lu test edilir.

### A.4 Test Sprint'leri
**T1 — Loader çekirdeği:** T-01, T-02, T-04 (bölme/dedup/index).
**T2 — Okuyucular + hatalar:** T-03, T-07.
**T3 — Processing + tahmin:** T-05, T-06.

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
data_loader_management/
  ├── loader.py     → DataLoaderManager (ince orkestrasyon)
  └── readers/      → DocxReader | JsonReader | QaReader | RawTextReader
  splitter → tokenizer-hizalı token tahmini
data_processing/  → scraper'lar (CLI ↔ kütüphane ayrımı, config'li yollar)
```

### B.2 Geliştirme Sprint'leri
**D1 — Okuyucu ayrıştırma** *(orta risk)*: `data_loader_manager.py`'nin `_read_*`/
`_load_*` metotlarını `readers/` stratejilerine böl. Önkoşul: T-03. **Kabul:** çıktı
formatı aynı; tek dev dosya küçülür.
**D2 — Token tahmini hizalama** *(davranış değişir → işaretli)*: `:112`
`_estimate_token_count`'ı gerçek tokenizer'a bağla. Önkoşul: T-05.
**D3 — Config'li yollar** *(düşük risk)*: hardcoded çıktı dizinlerini config'e taşı.
**D4 — CLI/kütüphane ayrımı** *(düşük risk)*: `data_processing` sınıf ↔ `main()`
ayrımı; API anahtarı gizli yönetimi.

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| Yükleme çıktı formatı (QA/text tuple + file_index) | `data_loader_manager.py:326/344` | Tokenizer + Training System tüketir |
| file_index doğruluğu | `:344` | split sızıntı garantisi ([training-system §5](../training-system/README.md#5-veri--kontrol-akışı)) |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D.

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 loader çekirdeği | `data_loader_manager.py:130/199/344` | ⏳ |
| A | T2 okuyucular/hatalar | `:366/519/552/723` | ⏳ |
| A | T3 processing/tahmin | `:112`, `subtitle_processor.py:57` | ⏳ |
| B | D1 okuyucu ayrıştırma | `readers/` (yeni) | ⏳ |
| B | D2 token tahmini hizalama | `:112` | ⏳ |
| B | D3 config'li yollar | `data_processing/*` | ⏳ |
| B | D4 CLI/kütüphane ayrımı | `data_processing/*` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
