# Code Reality — Data (Loader + Processing)

> Kanonik Birim #6 · Kaynak: `data_loader_management/` + `data_processing/`
> Veri toplama (offline) ve eğitim verisi yükleme katmanının kodundan çıkarılmış
> mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L0), [search index](../../architecture-search-index.md).

> 🔧 **Eşleşen plan:** Bu research ile birlikte okunacak test+geliştirme planı →
> [`phase-1-refactoring-plan.md`](phase-1-refactoring-plan.md)

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Data (Loader + Processing) |
| **Kaynak dizin** | `data_loader_management/` (yükleme), `data_processing/` (toplama/hazırlık) |
| **Toplam boyut** | ~3.500 LOC (842 loader + ~2.634 processing) |
| **Ana sınıf** | `DataLoaderManager` — `data_loader_management/data_loader_manager.py:97` |
| **Çalışma zamanı** | Loader: Batch (eğitim/tokenizer) · Processing: Offline (elle/betikle) |
| **Dış bağımlılık** | `requests` (scraping), `python-docx` (docx), Wikipedia/OpenSubtitles API'leri |

> **İki farklı yaşam döngüsü:** `data_processing/` **offline** çalışır (ham veriyi
> internetten toplar, dosyaya yazar). `data_loader_management/` **eğitim
> zamanında** bu dosyaları okuyup tokenizer/eğitim boru hattına besler. İkisi
> arasında **kod bağı yoktur**, yalnızca dosya sistemi üzerinden dolaylı bağlıdırlar.

---

## 2. Sorumluluk

- **`data_loader_management/`:** Diskteki eğitim verisini (QA çiftleri, düz metin,
  ham metin parçaları, `.docx`, `.json`) okuyup normalize etmek; uzun metinleri
  token tahmini + örtüşme (overlap) + tekilleştirme ile parçalara bölmek; opsiyonel
  dosya-indeksi ile (kaynak-id farkında split için) döndürmek.
- **`data_processing/`:** Ham korpus toplamak — Wikipedia scraping (kategori/konu/
  rastgele), altyazı indirme+işleme (SRT/VTT), PDF→DOCX dönüştürme, boş dosya
  temizliği, konu konfigürasyonu.

**Kapsam dışı:** Tokenizasyon (Tokenizer), cache üretimi (Training System),
model eğitimi (Training).

---

## 3. Dosya Envanteri

### 3.1 Yükleme (`data_loader_management/`)

| Dosya | LOC | Sınıf/rol | Anahtar üyeler |
|-------|-----|-----------|----------------|
| `data_loader_manager.py` | 842 | **`DataLoaderManager`** + `DataLoaderConfig`, `LoadMode`, hata sınıfları | `load`, `load_with_file_index`, `_load_qa_pairs`, `_load_text_inputs`, `_load_raw_text_chunks`, `_smart_split_text`, `_deduplicate_chunks`, `_estimate_token_count`, `_read_docx`, `_read_json`, `_iter_files` |

**Yükleme modları (`LoadMode`):** QA çiftleri / düz metin girdileri / ham metin
parçaları — her biri düz ve "file_index'li" varyantlarıyla.

### 3.2 Toplama/Hazırlık (`data_processing/`)

| Dosya | LOC | Sınıf/rol | Anahtar üyeler |
|-------|-----|-----------|----------------|
| `wikipedia_api_scraper.py` | 258 | **`WikipediaScraper`** | `get_page_content`, `get_category_pages`, `get_random_pages`, `scrape_pages` |
| `topic_based_scraper.py` | 267 | **`TopicBasedScraper`** | `get_topic_pages`, `scrape_topic` (derinlik kontrollü) |
| `batch_scrape_all_topics.py` | 237 | toplu scraping betiği | `scrape_all_topics`, `main` |
| `topic_config.py` | 588 | konu/kategori konfigürasyonu (veri) | — |
| `subtitle_downloader.py` | 505 | **`SubtitleDownloader`** | `search_subtitles`, `download_subtitle`, `download_dizi_subtitles`, `batch_download`, `normalize_dizi_name` |
| `subtitle_processor.py` | 302 | **`SubtitleProcessor`** | `parse_srt`, `parse_vtt`, `_clean_text`, `process_file`, `save_as_txt/json` |
| `pdf_to_docx_converter.py` | 288 | PDF → DOCX dönüştürücü | — |
| `cleanup_empty_files.py` | 189 | boş/bozuk dosya temizliği | — |

Ayrıca veri/doküman dosyaları: `turkce_dizi_listesi.json`,
`turkce_dizi_onerileri.md`, `scraping_results.json`, `README.md`,
`MANUAL_SUBTITLE_DOWNLOAD.md`.

---

## 4. İç Mimari

```
   OFFLINE  (data_processing/)                EĞİTİM ZAMANI (data_loader_management/)
   ─────────────────────────                  ──────────────────────────────────────
   ┌─────────────────────┐
   │ WikipediaScraper    │──┐
   │ TopicBasedScraper   │  │  ham metin
   │ batch_scrape_*      │  │  dosyaları
   └─────────────────────┘  │  (.txt/.json/
   ┌─────────────────────┐  │   .docx)          ┌──────────────────────────────┐
   │ SubtitleDownloader  │──┼──►  disk  ────────►│  DataLoaderManager           │
   │ SubtitleProcessor   │  │   (data/ ...)      │  · _iter_files (uzantı filtre)│
   └─────────────────────┘  │                    │  · _read_json/_read_docx      │
   ┌─────────────────────┐  │                    │  · _load_qa/_load_text/_raw   │
   │ pdf_to_docx / cleanup│─┘                    │  · _smart_split_text          │
   └─────────────────────┘                       │     (token tahmini+overlap)   │
                                                 │  · _deduplicate_chunks        │
                                                 └───────────────┬──────────────┘
                                                                 │ List[(soru, cevap)]
                                                                 │ | List[str]
                                                                 │ (+ file_index)
                                                                 ▼
                                        Tokenizer (train) · Training System (cache)
```

> **Kod bağı yok, dosya bağı var:** iki alt-birim yalnızca disk üzerinden
> haberleşir; `data_processing` çıktısını `data_loader_management` girdi olarak alır.

---

## 5. Veri / Kontrol Akışı — Akıllı Bölme

`DataLoaderManager._smart_split_text` uzun metinleri model bağlam sınırına uygun
parçalara böler:

```
uzun metin
  │  _estimate_token_count (kaba token sayımı)
  ▼
max_tokens (varsayılan 198) aşılıyorsa
  │  _split_by_words (kelime sınırında böl, overlap=20 token)
  ▼
_get_overlap_text (parçalar arası bağlam köprüsü)
  │
  ▼
_deduplicate_chunks (örtüşmeden doğan tekrarı temizle)
  │
  ▼
parça listesi  → (QA ise) _process_qa_pair_with_splitting
```

**file_index farkındalığı:** `load_with_file_index` her parçaya kaynak dosya
indeksini ekler. Bu, Training System'in **kaynak-id farkında train/val split**'i
için kritiktir (aynı dosyadan örneklerin sızmasını önler).

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni dosya formatı | `DataLoaderManager._iter_files` + yeni `_read_*` | uzantı filtresi + okuyucu |
| Bölme stratejisi | `_smart_split_text` / `_split_by_words` | token tahmini `_estimate_token_count` |
| Yeni veri kaynağı (scraping) | `data_processing/` yeni scraper (Wikipedia deseni) | `scrape_*` sözleşmesi |
| Altyazı formatı | `SubtitleProcessor.parse_*` | SRT/VTT ayrıştırıcı |
| Tekilleştirme mantığı | `_deduplicate_chunks` | overlap parametresi |

---

## 7. Bağımlılıklar

**`data_loader_management` bağımlı olduğu:** `python-docx` (opsiyonel), stdlib.
**Buna bağımlı olanlar:** Tokenizer (`TokenizerCore` eğitim verisi),
Training System (cache hazırlığı).
**`data_processing` bağımlı olduğu:** `requests`, Wikipedia/OpenSubtitles API'leri.
**Buna bağımlı olan (kod):** yok — çıktı dosyaları dolaylı tüketilir.

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **Tek dev dosya** | `data_loader_manager.py` 842 LOC, ~30 metot | Orta | Okuyucular (docx/json/qa/text) ayrı stratejilere bölünebilir |
| **Token tahmini kaba** | `_estimate_token_count` gerçek tokenizer'ı kullanmıyor | Orta | Tokenizer ile hizalanırsa split doğruluğu artar |
| **Bağlantısız iki alt-birim** | processing↔loader yalnızca disk üzerinden | Düşük | Bilinçli; ama sözleşme (dizin/format) belgelenmeli |
| **Betik/kütüphane karışık** | `data_processing` hem sınıf hem `main()` betiği | Düşük | CLI ile kütüphane ayrımı |
| **Hardcoded yollar** | `output_dir="data_processing/subtitles"` vb. | Düşük | config'e taşınabilir |
| **API anahtarı yönetimi** | `SubtitleDownloader(api_key=...)` | Düşük | gizli yönetimi netleştirilmeli |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Veri yükleme girişi | `data_loader_management/data_loader_manager.py:326` (`load`) |
| File-index'li yükleme | `data_loader_management/data_loader_manager.py:344` |
| Akıllı bölme | `data_loader_management/data_loader_manager.py:130` (`_smart_split_text`) |
| Tekilleştirme | `data_loader_management/data_loader_manager.py:199` |
| QA yükleme | `data_loader_management/data_loader_manager.py:366` |
| Wikipedia scraping | `data_processing/wikipedia_api_scraper.py:108` (`scrape_pages`) |
| Konu bazlı scraping | `data_processing/topic_based_scraper.py:139` (`scrape_topic`) |
| Altyazı indirme | `data_processing/subtitle_downloader.py:170` (`download_dizi_subtitles`) |
| Altyazı işleme | `data_processing/subtitle_processor.py:57` (`parse_srt`) |

---

*Kaynak: `data_loader_management/`, `data_processing/` — analiz kodun mevcut halinden çıkarılmıştır.*
