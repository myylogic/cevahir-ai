# Architecture Search Index

> **Amaç:** Cevahir'in mimari dokümantasyonuna **tek giriş kapısı**. Buradan
> hem üst düzey master mimariye, hem de her kanonik birimin kodundan çıkarılmış
> ayrıntılı `code-reality/` haritasına ulaşırsınız.
>
> **Nasıl kullanılır:** Bir konuda/birimde çalışacaksanız aşağıdaki tablodan
> ilgili satırı bulun, `code-reality/<birim>/` klasörüne gidin. Belirli bir
> kavramı arıyorsanız (ör. "RoPE", "BPE", "critic", "cache") §3 Kavram Dizini'ni
> kullanın.

---

## 1. Doküman Haritası

```
docs/architecture/
├── master-architecture.md          ← Uçtan uca sistem, katmanlar, akışlar
├── architecture-search-index.md    ← (bu dosya) gezinme + arama
└── code-reality/                   ← Kodun mimarideki birebir karşılığı
    ├── tokenizer/
    ├── neural-network/
    ├── model-engine/
    ├── training-management/
    ├── training-system/
    ├── data/
    ├── cognitive/
    ├── chatting/
    ├── api/
    └── database/
```

- **Master mimari:** [`master-architecture.md`](master-architecture.md)
- **Kod gerçekliği kökü:** [`code-reality/`](code-reality/)

---

## 2. Kanonik Birim İndeksi

Her birim, `code-reality/<klasör>/README.md` altında derin analiz + diyagramlarla
belgelenir. **Durum** sütunu dokümantasyon ilerlemesini gösterir.

| # | Kanonik Birim | Kaynak | code-reality | Durum |
|---|---------------|--------|--------------|-------|
| 1 | **Tokenizer** | `tokenizer_management/` | [`code-reality/tokenizer/`](code-reality/tokenizer/) | ✅ |
| 2 | **Neural Network** | `src/neural_network.py`, `src/neural_network_module/` | [`code-reality/neural-network/`](code-reality/neural-network/) | ✅ |
| 3 | **Model / Engine** | `model/`, `model_management/` | [`code-reality/model-engine/`](code-reality/model-engine/) | ✅ |
| 4 | **Training Management** | `training_management/` | [`code-reality/training-management/`](code-reality/training-management/) | ✅ |
| 5 | **Training System** | `training_system/` | [`code-reality/training-system/`](code-reality/training-system/) | ✅ |
| 6 | **Data (Loader + Processing)** | `data_loader_management/`, `data_processing/` | [`code-reality/data/`](code-reality/data/) | ⏳ |
| 7 | **Cognitive** | `cognitive_management/` | [`code-reality/cognitive/`](code-reality/cognitive/) | ⏳ |
| 8 | **Chatting** | `chatting_management/` | [`code-reality/chatting/`](code-reality/chatting/) | ⏳ |
| 9 | **API** | `api/` | [`code-reality/api/`](code-reality/api/) | ⏳ |
| 10 | **Database** | `database/` | [`code-reality/database/`](code-reality/database/) | ⏳ |

> **Durum anahtarı:** ✅ tamam · 🟡 taslak/kısmi · ⏳ planlandı (henüz yazılmadı)

---

## 3. Kavram Dizini (Concept → Birim)

Belirli bir mimari kavramın kodda **nerede** karşılık bulduğunu bulmak için:

| Kavram | Birim | İşaret |
|--------|-------|--------|
| BPE, vocab, merges, encode/decode | Tokenizer | `tokenizer_management/bpe`, `core/tokenizer_core.py` |
| Morfoloji, heceleme (syllabifier) | Tokenizer | `tokenizer_management/bpe/tokenization` |
| OOV fallback | Tokenizer | `core/tokenizer_core.py`, `test_oov_fallback_debug.py` |
| RoPE (positional encoding) | Neural Network | `dil_katmani_module/positional_encoding.py` |
| Embedding | Neural Network | `dil_katmani_module/language_embedding.py` |
| Attention (self/cross/MHA) | Neural Network | `ortak_katman_module/attention_manager_module/` |
| RMSNorm | Neural Network | `ortak_katman_module/rms_norm.py` |
| SwiGLU / FFN | Neural Network | `ortak_katman_module/feed_forward_network.py` |
| MoE (Mixture of Experts) | Neural Network | `ortak_katman_module/mixture_of_experts.py` |
| KV Cache | Neural Network | `ortak_katman_module/kv_cache.py` |
| Quantization | Neural Network | `ortak_katman_module/quantization_manager.py` |
| Checkpointing (activation) | Neural Network | `ortak_katman_module/advanced_checkpointing.py` |
| Unified inference API (facade) | Model / Engine | `model/cevahir.py` |
| Model load/save/lifecycle | Model / Engine | `model_management/model_manager.py` ve kardeşleri |
| Beam search / autoregressive gen | Model / Engine | `model/cevahir.py :: CevahirModelAPI` |
| Chat pipeline (terminal) | Model / Engine | `model_management/chat_pipeline.py` |
| Eğitim döngüsü | Training System/Mgmt | `training_system/train.py`, `training_management/v3/core` |
| Curriculum learning | Training Management | `training_management/v3/curriculum` |
| Optimizer stratejileri | Training Management | `training_management/v3/optimizers` |
| Eğitim güvenliği/kararlılık | Training Management | `training_management/v*/safety` |
| Cache (veri) | Training System | `training_system/data_cache.py`, `prepare_cache.py` |
| Veri yükleme | Data | `data_loader_management/data_loader_manager.py` |
| Scraping / Wikipedia / altyazı | Data | `data_processing/` |
| Strateji (direct/think/debate/tot) | Cognitive | `cognitive_management/v2/core`, `processing` |
| RAG / vektör bellek | Cognitive | `cognitive_management/v2/components` (memory) |
| Critic / güvenlik / doğrulama | Cognitive | `cognitive_management/v2/components` (critic) |
| Araç kullanımı (tool use) | Cognitive | `cognitive_management/v2/components` (tools) |
| Middleware / Event bus / DI | Cognitive | `cognitive_management/v2/{middleware,events,container}` |
| AIOps izleme | Cognitive | `cognitive_management/v2/monitoring` |
| Oturum / geçmiş | Chatting | `chatting_management/` |
| REST uçları | API | `api/routes/v3/` |
| Servisler (chat/session/user) | API | `api/services/` |
| Repository / schema | Database | `database/{repositories,schemas,interfaces}` |

---

## 4. Her `code-reality/<birim>/README.md` Şablonu

Tutarlılık için her birim dokümanı aynı iskeleti izler:

1. **Kimlik** — birim adı, kaynak dizin(ler), LOC/dosya sayısı, çalışma zamanı
2. **Sorumluluk** — birimin tek cümlelik amacı + kapsam sınırı
3. **Dosya Envanteri** — her dosya/alt-modül, sorumluluğu ve anahtar sınıf/fonksiyonları
4. **İç Mimari** — bileşen diyagramı (ASCII/mermaid), sınıf ilişkileri
5. **Veri/Kontrol Akışı** — birim içi akış diyagramı, giriş→çıkış
6. **Genişletme Noktaları** — bir alt-modülü nasıl geliştirir/değiştirirsiniz
7. **Bağımlılıklar** — hangi birimlere bağlı, hangi birimler buna bağlı
8. **Refactor Sinyalleri / Tech-Debt** — v2/v3 ikizliği, coupling, ölü kod, riskler
9. **Kod Referansları** — `dosya:satır` düzeyinde giriş noktaları

---

## 5. Sürüm ve Bakım

- Bu indeks, `code-reality/` altına yeni bir birim dokümanı eklendikçe güncellenir
  (Durum sütunu ⏳ → 🟡 → ✅).
- Kaynak kod değiştiğinde ilgili birim dokümanı ve gerekiyorsa
  [`master-architecture.md`](master-architecture.md) revize edilir.
