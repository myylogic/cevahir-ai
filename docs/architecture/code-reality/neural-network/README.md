# Code Reality — Neural Network

> Kanonik Birim #2 · Kaynak: `src/` (`neural_network.py` + `neural_network_module/`)
> Transformer decoder çekirdeğinin **kodundan çıkarılmış** mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L3b), [search index](../../architecture-search-index.md).

> 🔧 **Eşleşen plan:** Bu research ile birlikte okunacak test+geliştirme planı →
> [`phase-1-refactoring-plan.md`](phase-1-refactoring-plan.md)

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Neural Network (Transformer decoder) |
| **Kaynak dizin** | `src/neural_network.py` (tam model) + `src/neural_network_module/` (yapı taşları) |
| **Toplam boyut** | ~7.450 LOC (944 model + ~6.505 modül; testler hariç) |
| **Tam model sınıfı** | `CevahirNeuralNetwork(nn.Module)` — `src/neural_network.py:85` |
| **İnşa eden** | `model_management/ModelManager.build_model` → `ModelInitializer.build_model(CevahirNeuralNetwork, cfg)` |
| **Çalışma zamanı** | Batch (eğitim) + Online (çıkarım) |
| **Dış bağımlılık** | `torch`, `torch.nn` (opsiyonel TensorBoard `SummaryWriter`) |

> **Önemli isimlendirme notu:** Katman sınıfı `TransformerEncoderLayer` adını taşır
> ama `causal_mask` parametresiyle **decoder-only nedensel (causal) LM bloğu** olarak
> kullanılır. Sistem bir GPT-tarzı decoder'dır; "encoder" adı tarihseldir.

---

## 2. Sorumluluk

Token id dizisini alıp **logit'lere** dönüştüren transformer decoder'ın tüm sinir
ağı bileşenlerini sağlamak: embedding, konumsal kodlama (RoPE dahil), çok-başlı
nedensel dikkat, ileri-besleme (SwiGLU), normalizasyon (RMSNorm), MoE, KV cache,
quantization ve aktivasyon checkpoint'leme.

**Kapsam dışı:** Eğitim döngüsü (Training System/Mgmt), optimizasyon (orada),
checkpoint dosya IO'su (Model Mgmt), tokenizasyon (Tokenizer). Bu birim yalnızca
**ileri geçiş matematiğini** ve bileşen tanımlarını içerir.

---

## 3. Dosya Envanteri

### 3.1 Tam Model (`src/`)

| Dosya | LOC | Sınıf/rol | Anahtar üyeler |
|-------|-----|-----------|----------------|
| `neural_network.py` | 944 | **`CevahirNeuralNetwork`** — tüm bileşenleri montajlayan tam model | `__init__` (montaj), `forward`, `clear_kv_cache`, `apply_quantization`, `get_quantization_info`, `log_gradients`, `set_tb_writer`, `get_last_snapshot` |

### 3.2 Dil Katmanı (`neural_network_module/dil_katmani_module/`)

| Dosya | LOC | Sınıf | Rol / anahtar üyeler |
|-------|-----|-------|----------------------|
| `language_embedding.py` | 329 | **`LanguageEmbedding`** | Token embedding; `forward`, `resize_embedding`, `tie_weights_to`, `load_pretrained`, `freeze_embeddings`, `_initialize_weights` (xavier/normal), `√d` ölçekleme |
| `positional_encoding.py` | 502 | **`PositionalEncoding`** | Sinüzoidal / öğrenilen / **RoPE** / **YaRN-RoPE**; `apply_rotary_pos_emb`, `_build_rope_freqs`, `_build_yarn_rope_freqs`, `_grow_to` (dinamik uzatma) |

### 3.3 Ortak Katman (`neural_network_module/ortak_katman_module/`)

| Dosya | LOC | Sınıf | Rol / anahtar üyeler |
|-------|-----|-------|----------------------|
| `transformer_encoder_layer.py` | 715 | **`TransformerEncoderLayer`** | Decoder bloğu; `forward`, `_forward_impl`, `_parallel_forward_impl` (paralel attn+ffn), `_stochastic_depth`, `get_and_reset_moe_loss` |
| `feed_forward_network.py` | 379 | **`FeedForwardNetwork`** | FFN; `_gated_forward` (SwiGLU/GLU), `_standard_forward`, `_build_activation`, `_init_weights` |
| `rms_norm.py` | 141 | **`RMSNorm`** | RMS normalizasyon; `forward`, `extra_repr` |
| `kv_cache.py` | 382 | **`KVCache`** | Anahtar/değer önbelleği; `update`, `get`, `clear`, `_evict_sliding_window`, `is_full`, `seen_tokens` |
| `mixture_of_experts.py` | 457 | **`Router`**, **`MixtureOfExperts`** | MoE; `forward`, `_compute_load_balance_loss`, `parameter_count` |
| `quantization_manager.py` | 363 | **`QuantizationManager`** | Nicemleme; `quantize_model`, `dequantize_model`, `_apply_fp16/_apply_bf16/_apply_int8_dynamic/_apply_int8_static`, `get_model_size_mb` |
| `advanced_checkpointing.py` | 208 | **`AdvancedCheckpointing`** + `create_checkpointing_strategy` | Aktivasyon checkpoint; `should_checkpoint`, `checkpoint_forward` |

### 3.4 Dikkat Alt Modülü (`ortak_katman_module/attention_manager_module/`)

| Dosya | LOC | Sınıf | Rol / anahtar üyeler |
|-------|-----|-------|----------------------|
| `multi_head_attention.py` | 918 | **`MultiHeadAttention`** | Çok-başlı dikkat; **3 backend**: `_flash_attention_forward`, `_pytorch_sdpa_forward`, `_standard_sdpa_forward`; `_prepare_attention_mask`, `_check_tensor_values` |
| `self_attention.py` | 354 | **`SelfAttention`** | Sade self-attention; `scaled_dot_product_attention`, `combine_heads`, `process_mask`, `validate_inputs` |
| `cross_attention.py` | 367 | **`CrossAttention`** | Çapraz dikkat (encoder-decoder senaryosu) |
| `attention_optimizer.py` | 553 | dikkat optimizasyon yardımcıları | performans yolları |
| `attention_utils_module/attention_initializer.py` | 223 | ağırlık başlatma | — |
| `attention_utils_module/attention_normalizer.py` | 335 | dikkat normalizasyonu | — |
| `attention_utils_module/attention_scaler.py` | 279 | ölçekleme (temperature vb.) | — |

### 3.5 Testler

`neural_network_module/test/` altında ~85 test dosyası: attention (bridge/initializer/
normalizer/optimizer/scaler), embedding, FFN, normalization, MoE (load_balancer),
quantum/quantization, residual, scaling, tensor işleme, memory, parallel execution,
transformer katmanı, uçtan uca ağ doğrulama.

---

## 4. İç Mimari — Tam Model Montajı

`CevahirNeuralNetwork.__init__` (`src/neural_network.py:111`) tüm yapı taşlarını şu
sırayla montajlar:

```
                 input_ids (B, T)  int64
                        │
                        ▼
        ┌──────────────────────────────┐
        │  LanguageEmbedding           │  token → (B, T, D),  ×√D ölçekleme
        │  (dil_katmani_module)        │  (ops.) output katmanına weight-tie
        └───────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │  PositionalEncoding          │  RoPE / YaRN / sinüzoidal / öğrenilen
        │  (RoPE ise attention'da uyg.)│
        └───────────────┬──────────────┘
                        │
        ╔═══════════════▼═══════════════════════════════════════╗
        ║  N × TransformerEncoderLayer  (nedensel decoder bloğu) ║
        ║                                                        ║
        ║   x → RMSNorm → MultiHeadAttention(causal) → +residual ║
        ║     → RMSNorm → FFN(SwiGLU) | MoE          → +residual ║
        ║   (pre-norm; ops. parallel attn+ffn; stochastic depth) ║
        ║   KVCache (çıkarımda), AdvancedCheckpointing (eğitimde)║
        ╚═══════════════┬═══════════════════════════════════════╝
                        │
                        ▼
        ┌──────────────────────────────┐
        │  Final RMSNorm               │
        └───────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │  Output projection           │  (B, T, D) → (B, T, V)
        │  (weight tying ile embedding)│
        └───────────────┬──────────────┘
                        ▼
                  logits (B, T, V)
```

**Weight tying:** `tie_weights=True` iken çıkış projeksiyonu embedding ağırlıklarını
paylaşır (`LanguageEmbedding.tie_weights_to`). `config_schema` bunu zorunlu kılar:
`seq_proj_dim == embed_dim` (bkz. Model/Engine birimi).

---

## 5. Bileşen Detayları

### 5.1 Dikkat — Üç Backend (MultiHeadAttention)

```
scaled_dot_product_attention()
   │
   ├─ use_flash_attention  → _flash_attention_forward   (Flash Attn 2, varsa)
   ├─ PyTorch ≥2 SDPA      → _pytorch_sdpa_forward       (torch.nn.functional.sdpa)
   └─ fallback             → _standard_sdpa_forward      (manuel softmax(QKᵀ/√d)V)
```
Nedensel maske `_prepare_attention_mask` ile hazırlanır; NaN/Inf koruması
`_check_tensor_values` ile yapılır.

### 5.2 Konumsal Kodlama (RoPE + YaRN)

`PositionalEncoding` dört mod destekler: `sinusoidal`, `learned`, `rope`, `yarn`.
RoPE modunda frekanslar `_build_rope_freqs` ile üretilir, `apply_rotary_pos_emb`
attention Q/K'ya döner uygular. YaRN (`_build_yarn_rope_freqs`) uzun bağlam için
frekans ölçekleme sağlar. `_grow_to` dizinin dinamik olarak uzatılmasına izin verir.

### 5.3 FFN — SwiGLU

`FeedForwardNetwork._gated_forward` kapılı aktivasyon (SwiGLU/GLU) uygular:
`(W_gate·x ⊙ act(W_up·x))·W_down`. `_standard_forward` klasik iki-katmanlı MLP'dir.
Aktivasyon `_build_activation` ile isimden kurulur.

### 5.4 KV Cache — Sliding Window

`KVCache.update` yeni K/V'yi biriktirir; `_evict_sliding_window` pencere dolduğunda
en eski token'ları atar. Çıkarımda otoregresif üretimi hızlandırır; `clear_kv_cache`
model üzerinden sıfırlar.

### 5.5 MoE

`Router` token'ları uzmanlara yönlendirir; `MixtureOfExperts.forward` seçili
uzmanları çalıştırır ve `_compute_load_balance_loss` ile yük dengesi kaybı üretir.
Bu kayıp `TransformerEncoderLayer.get_and_reset_moe_loss` üzerinden eğitim kaybına
eklenir.

### 5.6 Quantization

`QuantizationManager` fp16/bf16/int8 (dinamik+statik) destekler; model boyutunu
`get_model_size_mb` ile raporlar. `CevahirNeuralNetwork.apply_quantization`
kalibrasyon verisiyle tetikler.

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni dikkat backend'i | `MultiHeadAttention._*_forward` | Üç yol da aynı imzayı korumalı |
| Yeni konumsal kodlama | `PositionalEncoding` (mod ekle) | `apply_rotary_pos_emb` sözleşmesi |
| Yeni aktivasyon | `FeedForwardNetwork._build_activation` | — |
| Norm değişikliği | `RMSNorm` ↔ yeni norm sınıfı; katmanda değiştir | Pre-norm varsayımı |
| Uzman sayısı/yönlendirme | `mixture_of_experts.py` `Router` | Load-balance kaybı korunmalı |
| Yeni quantization şeması | `QuantizationManager._apply_*` | `is_quantized`/`dequantize` simetrisi |
| Katman montaj sırası | `CevahirNeuralNetwork.__init__` | Tam modelin tek montaj noktası |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** yalnızca `torch`/`torch.nn`. Birim, dış proje modüllerine
bağımlı **değildir** (temiz çekirdek). TensorBoard opsiyoneldir (`_SummaryWriterLike`
protokolü ile gevşek bağ).

**Buna bağımlı olanlar:**
- **Model / Engine** — `ModelManager` bu modelin sınıfını (`CevahirNeuralNetwork`)
  içe aktarır ve `ModelInitializer` ile inşa eder (`model_management/model_manager.py:89`).
- **Training System/Mgmt** — dolaylı (ModelManager üzerinden eğitir).

> Bu birimin dışa bağımlı olmaması, refactor açısından değerli: transformer'ı
> değiştirmek yalnızca `ModelManager`'ın inşa sözleşmesini etkiler.

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Konum | Risk | Not |
|--------|-------|------|-----|
| **Yanıltıcı isim** | `TransformerEncoderLayer` aslında decoder bloğu | Orta | Yeniden adlandırma (`DecoderLayer`) okunabilirliği artırır |
| **Türkçe/İngilizce karışık isimlendirme** | `dil_katmani_module`, `ortak_katman_module` vs. İngilizce sınıflar | Düşük | Tutarlılık kararı |
| **Attention kod ikizliği** | `MultiHeadAttention` vs `SelfAttention` (benzer SDPA mantığı) | Orta | Ortak SDPA çekirdeği paylaşılabilir |
| **`test/` dev boyutu** | ~85 test dosyası, bazıları "quantum_*" gibi bağlamı belirsiz | Düşük | Ölü/deneysel test taraması gerekebilir |
| **Config parametre yayılımı** | `ModelInitializer` eksik parametreleri default'la dolduruyor (`model_manager.py`/`model_initializer.py:207`) | Orta | Örtük default'lar; şema doğrulaması Model/Engine'de sıkılaştırılabilir |
| **`_parallel_forward_impl`** | katmanda ikinci ileri-geçiş yolu | Orta | İki forward yolunun senkron bakımı |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Tam model montajı | `src/neural_network.py:111` (`CevahirNeuralNetwork.__init__`) |
| Model ileri geçiş | `src/neural_network.py:704` (`forward`) |
| KV cache temizleme | `src/neural_network.py:877` |
| Quantization uygula | `src/neural_network.py:888` |
| Embedding forward | `neural_network_module/dil_katmani_module/language_embedding.py:234` |
| RoPE uygulama | `dil_katmani_module/positional_encoding.py:410` (`apply_rotary_pos_emb`) |
| Decoder bloğu forward | `ortak_katman_module/transformer_encoder_layer.py:336` |
| Flash attention | `attention_manager_module/multi_head_attention.py:517` |
| SwiGLU FFN | `ortak_katman_module/feed_forward_network.py:314` (`_gated_forward`) |
| RMSNorm | `ortak_katman_module/rms_norm.py:105` |
| KV cache eviction | `ortak_katman_module/kv_cache.py:342` (`_evict_sliding_window`) |
| MoE load-balance | `ortak_katman_module/mixture_of_experts.py:388` |
| İnşa köprüsü | `model_management/model_manager.py:206` (`build_model`) |

---

*Kaynak: `src/` — analiz kodun mevcut halinden çıkarılmıştır.*
