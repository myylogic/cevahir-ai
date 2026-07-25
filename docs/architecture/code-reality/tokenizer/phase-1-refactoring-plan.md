# Phase 1 — Refactoring Plan · Tokenizer

> **Kanonik Birim #1 · Tokenizer** — `tokenizer_management/`
> Bu doküman, [research (mevcut gerçek durum)](README.md) ile **eşleşir**. Research
> "kod ne durumda"yı; bu plan "önce nasıl test edeceğiz, sonra nasıl geliştireceğiz"i
> tanımlar. Üst plan: [development-roadmap](../../development-roadmap.md).
>
> **Akış:** Önce **Test Fazı (A)** → sonra **Geliştirme Fazı (B)**. Kod değişmeden
> önce test zemini kurulur; her geliştirme sprint'i bir teste bağlanır.
>
> **Tutarlılık kuralı:** Bir sprint bittiğinde (1) kod, (2) [research README §8](README.md#8-refactor-sinyalleri--tech-debt),
> (3) buradaki durum tablosu birlikte güncellenir. Doküman ile kod asla ayrışmaz.

---

## 0. Kapsam ve Bağlam

| Alan | Değer |
|------|-------|
| Birim | Tokenizer (`tokenizer_management/`) |
| Kaynak boyutu | ~9.000 LOC (çekirdek) |
| Mevcut test boyutu | ~5.815 LOC / 17 test dosyası |
| Ana refactor hedefleri | `BPEManager` god-class, çift facade, singleton, CPU/GPU ikizi, token tahmini |
| Kritik sözleşmeler | encode∘decode round-trip · `vocab_size` (Tokenizer→Model) · özel token id'leri · dosya kalıcılığı |

> ⚠️ **Ortam notu:** Bu depoda `torch`/`pytest` kurulu değildir; testler
> **geliştirme ortamında** koşulmalıdır. Aşağıdaki "test reality" kaynaktan
> statik olarak çıkarılmıştır; canlı pass/fail durumu Faz 0'da doldurulacaktır.

---

## A. TEST FAZI (önce test)

### A.1 Test Reality — Mevcut Durum

Birim, kanonik birimler arasında **en olgun test paketine** sahiptir. Mevcut kapsam:

| Bileşen | Test dosyası | Durum |
|---------|--------------|-------|
| `TokenizerCore` | `test_tokenizer_core.py` (980), `test_tokenizer_core_comprehensive.py` (576) | ✅ Kapsamlı |
| `BPEManager` | `test_bpe_manager.py` (611) | ✅ İyi (singleton, roundtrip, finalize) |
| `BPEEncoder` | `test_bpe_encoder.py` (224) | ✅ Var |
| `BPEDecoder` | `test_bpe_decoder.py` (206) | ✅ Var |
| `BPETrainer` | `test_bpe_trainer.py` (204) | ✅ Var |
| `BPETokenizer` | `test_bpe_tokenizer.py` (213) | ✅ Var |
| `Pretokenizer` | `test_pretokenizer.py` (286) | ✅ Var |
| `Syllabifier` | `test_syllabifier.py` (324) | ✅ Var |
| `Morphology` | `test_morphology.py` (181), `test_ai_morphology.py` (126) | ✅ Var |
| `Postprocessor` | `test_postprocessor.py` (254) | ✅ Var |
| Türkçe işlemciler | `test_turkish_processor.py`, `test_processors.py` | ✅ Var |
| Entegrasyon | `test_comprehensive_integration.py` (791) | ✅ Var |
| OOV fallback | `test_oov_fallback_debug.py` (172) | 🟡 Debug betiği (pytest değil) |
| Vocab boyut kontrolü | `test_vocab_size_control.py` (387) | 🟡 Kök betik |

**Doğrulanan davranışlar (kaynaktan):** encode/decode hizası, membership OOB
kontrolü, train→infer no-UNK, kalıcılık tutarlılığı (aynı yol → aynı id),
singleton kimliği, Unicode NFC roundtrip, heceleme bayrağı etkisi, SEP sayımı.

### A.2 Test Boşlukları ve Şüpheli Alanlar (kusur envanteri)

Kaynak analizinden çıkan, **test edilip doğrulanması gereken** noktalar:

| # | Kusur/Boşluk | Kaynak (research) | Test hedefi |
|---|--------------|-------------------|-------------|
| T-01 | **GPU kod yolları test edilmiyor** — `_batch_encode_gpu`, `_bpe_ids_for_token_gpu`, `batch_syllabify_gpu` | [NN/Tokenizer §8 CPU/GPU ikizi](README.md#8-refactor-sinyalleri--tech-debt) | CPU↔GPU **parite** testi: aynı girdi → aynı id |
| T-02 | **OOV fallback pytest'e alınmamış** (debug betiği) | `test_oov_fallback_debug.py` | Karakter fallback + `[UNK]` sınır durumları için pytest |
| T-03 | **Token tahmini gerçek tokenizer'la hizalı değil** (Data biriminde `_estimate_token_count`) | [data §8](../data/README.md#8-refactor-sinyalleri--tech-debt) | Tahmin vs gerçek token sayısı sapma testi |
| T-04 | **Singleton test-izolasyonu** — global durum sızıntısı riski | research §8 singleton | Farklı yollarla instance izolasyonu (kısmen var: `test_instance_isolation`) — genişlet |
| T-05 | **`vocab_size` sözleşmesi** örtük | research §7 kritik bağ | Vocab boyutu değişiminde model uyumu için açık kontrat testi |
| T-06 | **Boş/bozuk vocab/merges dosyası** kurtarma | `_ensure_*` metotları | Eksik dosya/merges silinmiş senaryoları (kısmen var) — genişlet |
| T-07 | **Morfoloji/heceleme doğruluğu** dil-dilbilgisel | `Morphology`, `Syllabifier` | Türkçe altın-standart örnek kümesiyle doğruluk testi |
| T-08 | **Performans regresyonu** (büyük korpus) | `_monitor_memory_usage` | Smoke/perf testi (opsiyonel işaret) |

### A.3 Olması Gereken Test Durumu (hedef)

- **Bileşen başına birim testi** korunur; eksik olanlar (GPU parite, OOV pytest)
  eklenir.
- **Değişmez (invariant) testleri:** `decode(encode(x))` metin-eşdeğerliği
  (özel token/normalizasyon toleransıyla); `encode` id'leri her zaman vocab ∪
  char-fallback ∪ `[UNK]` içinde.
- **Sözleşme testleri:** `vocab_size` sabitliği, özel token id sabitliği,
  checkpoint-uyumlu vocab.
- **Parite testleri:** CPU yolu ile GPU yolu aynı çıktı (GPU yoksa `skip`).
- **CI kapısı:** tüm testler yeşil olmadan Geliştirme Fazı'na geçilmez.

### A.4 Test Sprint'leri

**Sprint T1 — Zemin ve yeşil taban**
- [ ] Geliştirme ortamında mevcut 17 test dosyasını koş, pass/fail'i bu dokümana yaz.
- [ ] `test_oov_fallback_debug.py` ve `test_vocab_size_control.py`'yi pytest'e taşı.

**Sprint T2 — Sözleşme ve invariant testleri**
- [ ] encode∘decode round-trip property testi (T-05, invariant).
- [ ] `vocab_size` + özel token id sözleşme testleri (T-05).
- [ ] Bozuk/eksik dosya kurtarma senaryolarını genişlet (T-06).

**Sprint T3 — GPU parite ve dil doğruluğu**
- [ ] CPU↔GPU parite testleri (T-01, GPU yoksa skip).
- [ ] Türkçe morfoloji/heceleme altın-standart kümesi (T-07).
- [ ] Token tahmini sapma testi (T-03).

> Faz A çıktısı: **yeşil, boşlukları kapatılmış bir test kalkanı.** Ancak bundan
> sonra Faz B'ye geçilir.

---

## B. GELİŞTİRME FAZI (sonra geliştirme)

Her adım, Faz A'daki ilgili testlerle korunur; **davranış değişmez** (aksi
belirtilmedikçe). Sıra, düşük-riskten yükseğe.

### B.1 Hedef Mimari (olması gereken)

```
TokenizerCore (yalnız: cihaz seçimi + veri yükleme köprüsü)
      │  (DI ile enjekte, singleton değil)
      ▼
TokenizerEngine  (eski BPEManager'ın çekirdeği — sadece BPE orkestrasyonu)
      ├── VocabStore        (dosya-IO + vocab/merges bakımı — ayrı sorumluluk)
      ├── TokenizationChain  (pretokenize → syllable → morph)
      └── Backend seçimi:  CpuBackend | GpuBackend  (tek arayüz, ikiz kod yok)
             └── Encoder / Decoder / Trainer (backend-agnostik çekirdek + ince backend)
```

### B.2 Geliştirme Sprint'leri

**Sprint D1 — VocabStore ayrıştırma** *(düşük risk)*
- `BPEManager`'dan dosya-IO ve vocab-bakım metotlarını (`_read/_write_*`,
  `_ensure_*`, `load/save_vocab/merges`) `VocabStore`'a taşı.
- Koruyan testler: T2 sözleşme testleri, mevcut `test_bpe_manager` persistence.
- Kabul: davranış aynı; `BPEManager` LOC belirgin düşer.

**Sprint D2 — Singleton → DI** *(orta risk)*
- `BPEManager.__new__` singleton'ını kaldır; `TokenizerCore` örneği enjekte etsin.
- Koruyan testler: T4 izolasyon testleri (önce genişletilmiş olmalı).
- Kabul: iki bağımsız instance birbirini etkilemez.

**Sprint D3 — CPU/GPU backend birleştirme** *(orta-yüksek risk)*
- Encoder/Decoder/Trainer/Pretokenizer/Syllabifier'daki `_*_gpu` ikizlerini tek
  arayüz + `Backend` seçimiyle birleştir.
- Koruyan testler: T1 parite testleri (**önkoşul**).
- Kabul: parite testleri yeşil; tek kod yolu.

**Sprint D4 — Çift facade netleştirme** *(düşük risk)*
- `TokenizerCore` yalnız cihaz+veri; `TokenizerEngine` yalnız BPE. Örtüşen
  wrapper metotları tekilleştir.
- Kabul: sorumluluk sınırı net; genel API imzası korunur.

**Sprint D5 — Token tahmini hizalama** *(düşük risk, davranış değişir → işaretli)*
- Data birimindeki `_estimate_token_count`'ı gerçek tokenizer'a bağla.
- Koruyan testler: T3 sapma testi (kabul eşiği tanımlanır).

### B.3 Korunacak Sözleşmeler

- `encode`/`decode` genel imzaları ve dönüş tipleri.
- `vocab_size`, özel token id'leri, vocab/merges dosya formatı (checkpoint uyumu).
- `BaseTokenizerManager` ABC kontratı.

---

## C. Kod ↔ Doküman Tutarlılığı

Her sprint sonunda **üçlü güncelleme** zorunlu:
1. **Kod** — değişiklik + testler yeşil.
2. **Research** — [README.md §8](README.md#8-refactor-sinyalleri--tech-debt) ilgili
   sinyal "çözüldü" olarak işaretlenir; §3/§4 gerekiyorsa güncellenir.
3. **Bu plan** — aşağıdaki durum tablosu güncellenir.

Böylece doküman her zaman kodun mevcut halini yansıtır; kaybolmadan ilerlenir.

---

## D. Durum Tablosu

| Faz | Sprint | Durum |
|-----|--------|-------|
| A (Test) | T1 zemin/yeşil taban | ⏳ |
| A | T2 sözleşme/invariant | ⏳ |
| A | T3 GPU parite/dil | ⏳ |
| B (Geliştirme) | D1 VocabStore | ⏳ |
| B | D2 singleton→DI | ⏳ |
| B | D3 CPU/GPU backend | ⏳ |
| B | D4 facade netleştirme | ⏳ |
| B | D5 token tahmini | ⏳ |

> **Durum:** ✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli

---

*Bu plan, [research](README.md) ile birlikte okunur. Öncelik/sıra proje
sahibinin kararıyla güncellenebilir.*
