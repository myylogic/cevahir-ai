# Code Reality — Tokenizer

> Kanonik Birim #1 · Kaynak: `tokenizer_management/`
> Bu doküman, tokenizer biriminin **kodundan çıkarılmış** mimarisini ayrıntılı
> olarak haritalar. Üst düzey bağlam için [master-architecture](../../master-architecture.md) (L1),
> gezinme için [search index](../../architecture-search-index.md).

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Tokenizer |
| **Kaynak dizin** | `tokenizer_management/` |
| **Toplam boyut** | ~9.000 LOC (çekirdek), ~42 `.py` (testler dahil) |
| **Giriş noktaları** | `TokenizerCore` (çıkarım/entegrasyon), `train_bpe.py` (sözlük eğitimi) |
| **Çalışma zamanı** | Batch (eğitim) + Online (encode/decode) |
| **Dış bağımlılık** | `torch` (GPU), `data_loader_management` (eğitim verisi), `psutil` (bellek izleme) |

---

## 2. Sorumluluk

Ham metni **token id dizisine** (encode) ve token id dizisini geri **metne** (decode)
çevirmek; ayrıca bir korpustan **BPE sözlüğü (vocab) ve birleştirme kuralları (merges)**
eğitmek. Türkçe'nin sondan eklemeli yapısına özel ön-işleme (heceleme, morfoloji)
içerir; dil-agnostiktir (vocab/merges yeniden eğitilebilir).

**Kapsam dışı:** Model ileri geçişi (Neural Network birimi), embedding (NN), eğitim
döngüsü (Training System). Tokenizer yalnızca metin ↔ id dönüşümü ve sözlük üretimidir.

---

## 3. Dosya Envanteri

### 3.1 Çekirdek + Taban (`core/`, kök)

| Dosya | LOC | Ana sınıf/rol | Anahtar üyeler |
|-------|-----|---------------|----------------|
| `core/tokenizer_core.py` | 1263 | **`TokenizerCore`** — üst düzey facade/wrapper | `encode`, `decode`, `batch_encode`, `_batch_encode_gpu`, `train_model`, `train_from_loader`, `load_training_data`, `finalize_vocab`, `get_vocab_size`, `auto_update_vocab` |
| `base_tokenizer_manager.py` | 105 | **`BaseTokenizerManager`** (ABC) — yönetici arayüzü | soyut: `encode`, `decode`, `train`, `get_vocab`, `get_merges`, `set_vocab`, `update_reverse_vocab` |
| `config.py` | 883 | Tokenizer konfigürasyon şeması/varsayılanları | — |
| `__init__.py` | 1 | Paket girişi | — |

### 3.2 BPE Çekirdeği (`bpe/`)

| Dosya | LOC | Ana sınıf/rol | Anahtar üyeler |
|-------|-----|---------------|----------------|
| `bpe/bpe_manager.py` | **1694** | **`BPEManager`** — merkezi orkestratör (Singleton) | `encode`, `decode`, `train`, `tokenize`, `finalize_vocab`, `load_vocab_and_merges`, `_ensure_special_tokens_in_vocab`, `_ensure_base_alphabet_in_vocab`, `auto_update_vocab` |
| `bpe/bpe_tokenizer.py` | 268 | **`BPETokenizer`** — Encoder+Decoder ince sarmalayıcı | `encode_text`, `encode_tokens`, `decode_ids`; `EncoderProtocol`, `DecoderProtocol` |
| `bpe/bpe_encoder.py` | 690 | **`BPEEncoder`** — token → id + BPE merge uygulama | `encode`, `encode_sequence`, `_bpe_ids_for_token`, `_bpe_ids_for_token_gpu`, `_char_fallback_ids`, `batch_encode_sequence_gpu` |
| `bpe/bpe_decoder.py` | 503 | **`BPEDecoder`** — id → token → metin | `decode`, `_build_reverse_vocab`, `batch_decode_gpu`, `_ensure_special_tokens_exact` |
| `bpe/bpe_trainer.py` | 938 | **`BPETrainer`** — pair istatistiği + merge seçimi | `train`, `_get_pair_stats`, `_select_best_pair`, `_train_gpu_batch`, `_train_cpu_sequential`, `_apply_merges_to_sequence` |
| `bpe/bpe_manager_utils.py` | 606 | Vocab yardımcıları (saf fonksiyonlar) | `normalize_vocab`, `validate_vocab`, `get_base_alphabet`, `default_vocab`, `next_id`, `to_versioned_vocab` |

### 3.3 Tokenizasyon Ön/Son İşleme (`bpe/tokenization/`)

| Dosya | LOC | Ana sınıf/rol | Anahtar üyeler |
|-------|-----|---------------|----------------|
| `pretokenizer.py` | 452 | **`Pretokenizer`** — Unicode/lowercase/noktalama/boşluk bölme | `tokenize`, `_normalize_unicode`, `_separate_punctuation`, `_tokenize_whitespace`, `batch_tokenize_gpu`, `get_token_offsets` |
| `syllabifier.py` | 326 | **`Syllabifier`** — Türkçe heceleme | `syllabify_word`, `split`, `split_into_syllables`, `strip_diacritics`, `batch_syllabify_gpu` |
| `morphology.py` | 416 | **`Morphology`** — kök bulma / ek ayrıştırma | `find_root`, `_split_morpheme`, `_strip_suffix_chain`, `_check_vowel_harmony`, `analyze` |
| `postprocessor.py` | 287 | **`Postprocessor`** — token → düzgün metin | `process`, `_fix_punctuation_spacing`, `_capitalize_sentences`, `_collapse_whitespace` |
| `_syllabifier_utils.py` | 210 | Heceleme yardımcı verileri/fonksiyonları | — |
| `_morphology_utils.py` | 401 | Morfoloji yardımcı verileri (ek listeleri, ünlü uyumu) | — |

### 3.4 Yardımcı/Betik/Test Dosyaları (kök)

`train_bpe.py` (sözlük eğitimi giriş noktası), `prepare_bpe_cache.py`,
`evaluate_bpe_training.py`, `check_vocab.py`, `debug_cache.py`,
`test_*.py` / `tests/` (kapsamlı entegrasyon, OOV fallback, vocab boyutu kontrolü).

---

## 4. İç Mimari

Tokenizer, **iki katlı facade** üzerine kuruludur: `TokenizerCore` (entegrasyon
cephesi) → `BPEManager` (BPE orkestratörü) → alt bileşenler.

```
┌──────────────────────────────────────────────────────────────────────┐
│  TokenizerCore              (core/tokenizer_core.py)                    │
│  · Cihaz (CPU/GPU) yönetimi   · Veri yükleme (DataLoaderManager)        │
│  · batch_encode / GPU batch   · Eğitim akışı sarmalama                  │
│  · Tüm encode/decode mantığını BPEManager'a DELEGE eder                 │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │ self.tokenizer = BPEManager(...)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BPEManager  (Singleton, extends BaseTokenizerManager)                 │
│  bpe/bpe_manager.py — merkezi orkestratör, vocab/merges dosya sahibi   │
│                                                                        │
│   _initialize_components() / _sync_components() ile şunları kurar:     │
│                                                                        │
│   ┌────────────────┐  ┌───────────────┐  ┌───────────────┐            │
│   │  Pretokenizer  │  │  Syllabifier  │  │  Morphology   │  ← ön-işleme│
│   └────────────────┘  └───────────────┘  └───────────────┘            │
│   ┌────────────────┐  ┌───────────────┐  ┌───────────────┐            │
│   │  BPEEncoder    │  │  BPEDecoder   │  │  BPETrainer   │  ← BPE       │
│   └────────────────┘  └───────────────┘  └───────────────┘            │
│   ┌────────────────┐                                                   │
│   │  Postprocessor │  ← son-işleme (decode metni düzeltme)             │
│   └────────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────┘
             │ paylaşılan durum: vocab (Dict), merges (List[Tuple])
             ▼
   vocab.json / merges.txt  (disk — _write_*_atomic ile atomik yazım)
```

> **`BPETokenizer` (bpe_tokenizer.py):** `BPEEncoder` + `BPEDecoder`'ı `EncoderProtocol`
> / `DecoderProtocol` üzerinden birleştiren ince bir sarmalayıcı. Protokol tabanlı
> olduğundan encoder/decoder yerine geçebilir implementasyonlar takılabilir.

### 4.1 Taban Sınıf Sözleşmesi

`BaseTokenizerManager` (ABC), tüm yöneticiler için **kontratı** tanımlar:
`encode`, `decode`, `train`, `get_vocab`, `get_merges`, `set_vocab`,
`update_reverse_vocab`. `BPEManager` bu ABC'yi uygular. Bu, ileride farklı bir
tokenizer ailesinin (ör. WordPiece/Unigram) aynı arayüzle takılabilmesine kapı
açan **Dependency Inversion** noktasıdır.

---

## 5. Veri / Kontrol Akışı

### 5.1 Encode (metin → id)

```
text: str
  │  TokenizerCore.encode(text, add_special_tokens=…)
  ▼
BPEManager.encode()
  │
  ├─► Pretokenizer.tokenize()
  │      · Unicode NFC/NFKC normalize
  │      · (ops.) lowercase / İ-ı Türkçe kuralları
  │      · noktalama ayrımı, boşluk bölme  →  kaba token'lar
  │
  ├─► (include_syllables ise) Syllabifier.split()
  │      · Türkçe hece sınırlarına böl
  │
  ├─► (morfoloji aktifse) Morphology.analyze()
  │      · kök + ek zinciri
  │
  ├─► BPEEncoder.encode()
  │      · merge sıralamasına göre en iyi çifti birleştir (_bpe_ids_for_token)
  │      · GPU yolu: _bpe_ids_for_token_gpu / batch_encode_sequence_gpu
  │      · OOV: _char_fallback_ids → karakter bazlı, en son [UNK]
  │
  ▼
List[int]  (+ ops. token listesi, encode_with_stats ile istatistik)
```

### 5.2 Decode (id → metin)

```
List[int]
  │  TokenizerCore.decode(ids)
  ▼
BPEManager.decode() → BPEDecoder.decode()
  │   · _build_reverse_vocab() ile id→token
  │   · özel token'ları filtrele/koru (_ensure_special_tokens_exact)
  │   · parçaları birleştir (mode'a göre filtre)
  ▼
Postprocessor.process()
  │   · noktalama boşluğu düzelt, whitespace daralt
  │   · cümle başı büyük harf (ops.)
  ▼
text: str
```

### 5.3 Eğitim (korpus → vocab + merges)

```
train_bpe.py  (giriş noktası)
  │
  ▼
TokenizerCore.train_model() / train_from_loader()
  │   · DataLoaderManager ile korpus al, (ops.) örnekle
  ▼
BPEManager.train()
  │   · korpusu chunk'la (_process_corpus_in_chunks / _stream_corpus_processing)
  │   · özel token + temel alfabe garantile
  │   ▼
  │  BPETrainer.train()
  │     · _get_pair_stats (CPU/parallel/GPU varyantları)
  │     · _select_best_pair → en sık çift
  │     · _merge_pair_linear / _apply_merges_to_sequence
  │     · _train_gpu_batch  |  _train_cpu_sequential
  │   ▼
  │  merges listesi + genişletilmiş vocab
  ▼
finalize_vocab() → save_vocab() / save_merges()  (atomik yazım)
```

---

## 6. Genişletme Noktaları (Nasıl geliştirilir?)

| Ne yapmak istiyorsun | Nereye dokunursun | Not |
|----------------------|-------------------|-----|
| Yeni dil ekle | `train_bpe.py` ile yeniden eğit; `tokenization/` kurallarını dilin morfolojisine uyarla | Vocab/merges dile özgüdür |
| Farklı ön-tokenizasyon | `Pretokenizer.tokenize` + `_separate_punctuation` | Offset takibi `get_token_offsets`'i bozmayın |
| Heceleme kuralı değişikliği | `Syllabifier` + `_syllabifier_utils.py` | GPU yolu `batch_syllabify_gpu` ile senkron tutulmalı |
| Morfolojik analiz iyileştirme | `Morphology` + `_morphology_utils.py` (ek listeleri, ünlü uyumu) | `find_root` / `_strip_suffix_chain` |
| Yeni encode/decode stratejisi | `BPEEncoder` / `BPEDecoder`; `EncoderProtocol`/`DecoderProtocol`'e uy | Protokol sayesinde takılabilir |
| Eğitim algoritması (BPE dışı) | `BPETrainer`'ı ayrı bir Trainer ile değiştir; `BaseTokenizerManager` kontratını koru | — |
| Decode sonrası metin kalitesi | `Postprocessor` | Büyük harf/boşluk kuralları |

---

## 7. Bağımlılıklar

**Tokenizer'ın bağımlı olduğu:**
- `data_loader_management` → eğitim korpusunu yüklemek için (`TokenizerCore`).
- `torch` → GPU tokenizasyon/encode/train yolları (opsiyonel; CPU fallback var).
- `psutil` → eğitim sırasında bellek izleme (`_monitor_memory_usage`).

**Tokenizer'a bağımlı olanlar:**
- **Model / Engine** (`Cevahir._init_tokenizer`, `CevahirModelAPI`) — encode/decode.
- **Neural Network** — vocab_size doğrudan embedding boyutunu belirler
  (`Cevahir._init_model`: `vocab_size = tokenizer.get_vocab_size()`).
- **Training System** — eğitim verisini tokenize etmek için.

> **Kritik bağ:** `vocab_size` tokenizer'dan modele akar. Vocab yeniden eğitilirse
> modelin embedding/çıkış katmanı boyutu değişir → checkpoint uyumsuzluğu.

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Konum | Risk | Öneri (bilgi amaçlı) |
|--------|-------|------|----------------------|
| **God class** | `BPEManager` 1694 LOC, ~50 metot | Yüksek | Sorumlulukları ayır: dosya-IO, vocab-bakımı, tokenizasyon-orkestrasyonu ayrı sınıflara |
| **Çift facade** | `TokenizerCore` → `BPEManager` ikisi de "wrapper" | Orta | Katman değeri netleştirilmeli; `TokenizerCore` sadece cihaz+veri, `BPEManager` sadece BPE |
| **Singleton** | `BPEManager.__new__` | Orta | Global durum testleri zorlaştırır; DI ile enjeksiyon düşünülebilir |
| **CPU/GPU kod ikizliği** | Her bileşende `_*_gpu` varyantları (encoder, decoder, trainer, pretokenizer, syllabifier) | Yüksek | İki yol ayrı bakım gerektirir; ortak arayüz + backend seçimi ile birleştirilebilir |
| **Gizli import'lar** | `tokenizer_core.py` içinde metot-içi `import torch/random/time/psutil` | Düşük | Modül başına taşınabilir |
| **`DummyPostprocessor`** | `bpe_decoder.py` | Düşük | Fallback; gerçek Postprocessor her zaman mevcut mu netleştirilmeli |
| **Kök dizinde debug/test betikleri** | `debug_cache.py`, `check_vocab.py`, çok sayıda `test_*.py` kökte | Düşük | `tests/` altına toplanabilir |
| **Vocab-model kaplini** | `vocab_size` sözleşmesi örtük | Orta | Sabit vocab stratejisi kodda var ("Vocab genişletme ATLANDI"); açık kontrat/doğrulama eklenebilir |

---

## 9. Kod Referansları (giriş noktaları)

| Amaç | Referans |
|------|----------|
| Üst düzey encode | `tokenizer_management/core/tokenizer_core.py:424` (`encode`) |
| GPU batch encode | `core/tokenizer_core.py:548` (`_batch_encode_gpu`) |
| Eğitim (core) | `core/tokenizer_core.py:242` (`train_model`), `:310` (`train_from_loader`) |
| BPE encode | `bpe/bpe_manager.py:426` (`encode`) |
| BPE decode | `bpe/bpe_manager.py:760` (`decode`) |
| BPE train | `bpe/bpe_manager.py:928` (`train`) |
| Bileşen kurulumu | `bpe/bpe_manager.py:276` (`_initialize_components`) |
| Merge çekirdeği | `bpe/bpe_trainer.py:142` (`train`), `:430` (`_select_best_pair`) |
| Karakter fallback (OOV) | `bpe/bpe_encoder.py:529` (`_char_fallback_ids`) |
| Ön-tokenizasyon | `bpe/tokenization/pretokenizer.py:273` (`tokenize`) |
| Heceleme | `bpe/tokenization/syllabifier.py:169` (`syllabify_word`) |
| Kök bulma | `bpe/tokenization/morphology.py:222` (`find_root`) |
| Manager kontratı | `base_tokenizer_manager.py:40` (`BaseTokenizerManager`) |

---

*Kaynak: `tokenizer_management/` — analiz kodun mevcut halinden çıkarılmıştır.*
