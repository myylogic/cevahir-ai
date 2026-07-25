# Phase 1 — Refactoring Plan · Neural Network

> **Kanonik Birim #2 · Neural Network** — `src/`
> [research (mevcut gerçek durum)](README.md) ile eşleşir. Akış: önce **Test Fazı (A)**
> → sonra **Geliştirme Fazı (B)**. Üst plan: [development-roadmap](../../development-roadmap.md).
>
> **Tutarlılık kuralı:** Her sprint sonu kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt)
> + durum tablosu birlikte güncellenir.
>
> **Bu sürüm derinleştirilmiştir:** her test ve geliştirme kalemi, `dosya:satır`
> çapası ve ilgili **mimari doküman kesiti** ile ilişkilendirilmiştir. Böylece
> geliştirici, plandan çıkmadan hem kodu hem de "olması gereken"i tek noktadan
> görebilir.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `src/neural_network.py` (944) + `src/neural_network_module/` (~6.505) |
| Boyut | ~7.450 LOC |
| Mevcut test | **67 test dosyası / ~11.276 LOC** |
| Refactor hedefleri | `TransformerEncoderLayer` yanıltıcı isim, attention kod ikizliği, config default doldurma, `_parallel_forward_impl` |
| Kritik sözleşmeler | `forward` çıktı şekli (B,T,V) · weight tying (`seq_proj_dim==embed_dim`) · attention backend parite |

> ⚠️ Ortam notu: `torch`/`pytest` kurulu değil; canlı koşu geliştirme ortamında.

### 0.1 Mimari Referans Haritası

Bu plandaki her iş kalemi, aşağıdaki mimari dokümanlara dayanır. Kaybolmadan
ilerlemek için önce ilgili kesiti okuyun:

| Referans | Ne için |
|----------|---------|
| [master-architecture §2](../../master-architecture.md#2-katmanlı-görünüm-layered-view) (L3b) | Bu birimin sistemdeki yeri (çekirdek sinir ağı katmanı) |
| [master-architecture §9](../../master-architecture.md#9-mimari-kararlar-ve-uygulanan-desenler) | Uygulanan desenler (bu birim `torch`'a bağımlı, temiz çekirdek) |
| [research §4](README.md#4-i̇ç-mimari--tam-model-montajı) | Tam model montaj sırası (hedef mimarinin temeli) |
| [research §5](README.md#5-bileşen-detayları) | Bileşen davranışları (attention 3 backend, RoPE, SwiGLU, KV cache, MoE, quant) |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri (bu planın kaynağı) |
| [research §9](README.md#9-kod-referansları) | `dosya:satır` giriş noktaları |
| [model-engine research §7](../model-engine/README.md#7-bağımlılıklar) | **İnşa köprüsü:** bu ağı `ModelManager` kurar; sözleşme orada |

---

## A. TEST FAZI (önce test)

### A.1 Test Reality — Mevcut Durum

Birim, kanonik birimler arasında **en olgun test paketine** sahiptir
(`src/neural_network_module/test/`, 67 dosya):

| Bileşen | Test | Kaynak (research §3) | Durum |
|---------|------|----------------------|-------|
| Tam ağ | `test_neural_network.py` (381), `_comprehensive.py` (**4882**), `_v2.py` (255), `_architecture_validation.py` (362), `_real_config_validation.py` | [§3.1](README.md#31-tam-model-src) | ✅ Çok kapsamlı |
| Attention | `test_multi_head_attention.py` (384), `test_self_attention.py`, `test_cross_attention.py`, `test_attention_{initializer,normalizer,optimizer,scaler,bridge}.py` | [§3.4](README.md#34-dikkat-alt-modülü-ortak_katman_moduleattention_manager_module) | ✅ Kapsamlı |
| Normalizasyon | `test_{layer,group,instance,batch}_normalization.py`, `test_normalization_*` | [§3.3](README.md#33-ortak-katman-neural_network_moduleortak_katman_module) | ✅ Var |
| FFN | `test_feed_forward_network.py` | [§3.3](README.md#33-ortak-katman-neural_network_moduleortak_katman_module) | ✅ Var |
| MoE | `test_load_balancer.py` | [§5.5](README.md#55-moe) | 🟡 Yalnız load-balancer |
| Embedding | `test_language_embedding.py`, `test_embedding_{initializer,projection,scaler}.py` | [§3.2](README.md#32-dil-katmanı-neural_network_moduledil_katmani_module) | ✅ Var |
| Paralel/tensor/scaling/residual/**quantum**/memory | ilgili `test_*` | — | ✅ Bol (bazıları deneysel/bağlamı belirsiz) |

**KV cache / quantization / RoPE için özel test dosyası görünmüyor** → §A.2'de boşluk.

### A.2 Kusur Envanteri (test edilip doğrulanacak)

Her satır: kusur → **kod çapası** (`dosya:satır`) → **mimari referans** → test hedefi.

| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **Attention backend paritesi** — flash / pytorch-sdpa / standard aynı çıktıyı vermeli | `multi_head_attention.py:517` (`_flash_attention_forward`), `:482` (`_pytorch_sdpa_forward`), `:593` (`_standard_sdpa_forward`), giriş `:449` (`scaled_dot_product_attention`) | [research §5.1](README.md#51-dikkat--üç-backend-multiheadattention) | 3 backend'i aynı Q/K/V ile çağır → çıktı `allclose` (tolerans) |
| **T-02** | **İki forward yolu eşdeğer mi** — standart vs paralel attn+ffn | `transformer_encoder_layer.py:485` (`_forward_impl`) vs `:641` (`_parallel_forward_impl`); giriş `:336` (`forward`) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Aynı girdi + config → iki yol `allclose` |
| **T-03** | **KV cache doğruluğu** — cache'li adım-adım üretim, cache'siz tam-forward ile aynı logit'i vermeli | `kv_cache.py:173` (`update`), `:342` (`_evict_sliding_window`); `neural_network.py:877` (`clear_kv_cache`) | [research §5.4](README.md#54-kv-cache--sliding-window) | Eşdeğerlik + pencere taşınca eviction doğruluğu |
| **T-04** | **Weight tying invariant** — `tie_weights=True` iken embedding ↔ output ağırlığı aynı tensör olmalı | `language_embedding.py:217` (`tie_weights_to`); montaj `neural_network.py:111` | [research §4](README.md#4-i̇ç-mimari--tam-model-montajı) + [model-engine config_schema:205](../model-engine/README.md#9-kod-referansları) | `id(embed.weight)==id(output.weight)`; `seq_proj_dim!=embed_dim` → hata |
| **T-05** | **MoE aux loss** eğitim kaybına doğru ekleniyor mu | `mixture_of_experts.py:388` (`_compute_load_balance_loss`); toplama `transformer_encoder_layer.py:463` (`get_and_reset_moe_loss`) | [research §5.5](README.md#55-moe) | Aux loss > 0 ve reset sonrası sıfırlanıyor |
| **T-06** | **Quantization round-trip** (fp16/bf16/int8) | `quantization_manager.py` (`quantize_model`/`dequantize_model`); tetik `neural_network.py:888` (`apply_quantization`) | [research §5.6](README.md#56-quantization) | quantize→dequantize sonrası çıktı toleransta; `get_model_size_mb` küçülüyor |
| **T-07** | **RoPE sayısal doğruluğu** — özel test yok | `positional_encoding.py:410` (`apply_rotary_pos_emb`), `:294` (`_build_rope_freqs`) | [research §5.2](README.md#52-konumsal-kodlama-rope--yarn) | Bilinen açı → beklenen dönüş; pozisyon kaydırma değişmezliği |
| **T-08** | **Deneysel/ölü testler** — "quantum_*" bağlamı belirsiz | `test/test_quantum_*.py` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Kapsam denetimi; kanonik mi deneysel mi karar |

### A.3 Olması Gereken Test Durumu

- **Şekil/invariant:** her katman için giriş→çıkış boyutu, dtype/device korunumu
  (`forward` çıktısı `(B,T,V)` — [research §4](README.md#4-i̇ç-mimari--tam-model-montajı)).
- **Parite:** attention backend'leri (T-01), iki forward yolu (T-02), KV cache (T-03).
  Bu üç parite testi, Geliştirme Fazı'nın **önkoşuludur** (D2/D3 bunlara dayanır).
- **Sayısal:** RoPE (T-07), RMSNorm (`rms_norm.py:105`), SwiGLU kapısı
  (`feed_forward_network.py:314`) — referans değerlere karşı.
- **Sözleşme:** weight tying (T-04) ve `ModelManager.build_model` inşa uyumu
  (`model_management/model_manager.py:206`).
- Deneysel "quantum_*" testleri ayıklanır (T-08); kanonik kapsam netleşir.

### A.4 Test Sprint'leri

**Sprint T1 — Yeşil taban + envanter**
- [ ] Geliştirme ortamında 67 test dosyasını koş; pass/fail'i §A.1 tablosuna yaz.
- [ ] "quantum_*" ve bağlamı belirsiz testleri işaretle (T-08 girişi).
- [ ] Kapsam boşluklarını doğrula: KV cache / quantization / RoPE için ayrı test var mı?

**Sprint T2 — Parite ve sözleşme (Geliştirme Fazı önkoşulu)**
- [ ] T-01 attention backend parite testi.
- [ ] T-02 iki forward yolu eşdeğerlik testi.
- [ ] T-03 KV cache eşdeğerlik + eviction.
- [ ] T-04 weight tying invariant + negatif (shape) testi.

**Sprint T3 — Bileşen sayısal + MoE/quant**
- [ ] T-05 MoE aux loss; T-06 quantization round-trip.
- [ ] T-07 RoPE + RMSNorm + SwiGLU sayısal doğrulama.

> Faz A çıktısı: **parite testleri yeşil.** D2/D3 ancak bundan sonra başlar.

---

## B. GELİŞTİRME FAZI (sonra geliştirme)

Her adım Faz-A testleriyle korunur; **davranış değişmez** (aksi işaretlenmedikçe).
Hedef mimari [research §4](README.md#4-i̇ç-mimari--tam-model-montajı) montaj sırasını
korur; yalnızca iç yapı sadeleşir.

### B.1 Hedef Mimari

```
CevahirNeuralNetwork  (tek montaj noktası — neural_network.py:111 — DEĞİŞMEZ)
  ├── AttentionCore          (YENİ: paylaşılan SDPA çekirdeği)
  │      ├── MultiHeadAttention  → backend seçimi: flash | sdpa | standard
  │      └── SelfAttention       → aynı çekirdeği kullanır (ikiz SDPA kodu biter)
  ├── DecoderLayer           (eski TransformerEncoderLayer — tek forward yolu)
  ├── PositionalEncoding · LanguageEmbedding · RMSNorm · FFN · MoE · KVCache
  └── (montaj sırası ve dış API korunur)
```
> Hedef, [research §4 diyagramı](README.md#4-i̇ç-mimari--tam-model-montajı) ile birebir
> uyumludur; blok içi sadeleşme dışında akış aynıdır.

### B.2 Geliştirme Sprint'leri

**Sprint D1 — Yeniden adlandırma** *(düşük risk)*
- `TransformerEncoderLayer → DecoderLayer` (`transformer_encoder_layer.py:59`),
  geriye dönük `TransformerEncoderLayer = DecoderLayer` alias'ı ile.
- **Neden:** [research §8](README.md#8-refactor-sinyalleri--tech-debt) "yanıltıcı isim" —
  sınıf `causal_mask` ile decoder bloğu ([research §1 notu](README.md#1-kimlik)).
- **Koruyan testler:** mevcut tam-ağ testleri (`test_neural_network*`).
- **Kabul:** tüm testler yeşil; `DecoderLayer` adı `ModelInitializer`'da da geçerli.

**Sprint D2 — Attention çekirdeği birleştirme** *(orta risk)*
- `MultiHeadAttention` (`multi_head_attention.py:85`) ve `SelfAttention`
  (`self_attention.py:46`) ortak bir SDPA çekirdeğini paylaşsın; üç backend tek yerde.
- **Neden:** [research §8](README.md#8-refactor-sinyalleri--tech-debt) "attention kod ikizliği".
- **Önkoşul:** T-01 parite testleri yeşil.
- **Kabul:** parite testleri yeşil kalır; SDPA mantığı tek kaynak.

**Sprint D3 — Forward yolu netleştirme** *(orta risk)*
- `_parallel_forward_impl` (`transformer_encoder_layer.py:641`) ya kanonik yapılır
  ya kaldırılır; `forward` (`:336`) tek yola sadeleşir.
- **Önkoşul:** T-02 eşdeğerlik testi (hangi yolun kanonik olduğunu belgeler).
- **Kabul:** tek forward yolu; çıktı T-02 ile aynı.

**Sprint D4 — Config sıkılaştırma** *(düşük risk)*
- `ModelInitializer.build_model` (`model_management/model_initializer.py:167`, ~`:207`
  örtük default doldurma) → açık şema doğrulaması.
- **Neden:** [research §8](README.md#8-refactor-sinyalleri--tech-debt) "config default doldurma";
  [model-engine planı D2 (config tek kaynağı)](../model-engine/phase-1-refactoring-plan.md#b2-geliştirme-sprintleri) ile **ortak yürütülür**.
- **Kabul:** eksik parametre → net hata; `config_schema` tek kaynak.

**Sprint D5 — Ölü test/kod ayıklama** *(düşük risk)*
- T-08 çıktısına göre deneysel "quantum_*" testlerini kanonik/deneysel ayır;
  ölü olanı işaretle/kaldır.
- **Kabul:** test envanteri net; kanonik kapsam belgeli.

### B.3 Korunacak Sözleşmeler

| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| `forward` imzası + çıktı `(B,T,V)` | `neural_network.py:704` | Model/Engine ve eğitim buna bağlı |
| weight tying kuralı | `config_schema` ([model-engine §9](../model-engine/README.md#9-kod-referansları)) | checkpoint uyumu |
| attention çıktı eşdeğerliği | T-01 | backend değişimi davranışı bozmamalı |
| `ModelManager.build_model` inşa sözleşmesi | `model_management/model_manager.py:206` | ağı bu köprü kurar ([model-engine §7](../model-engine/README.md#7-bağımlılıklar)) |

---

## C. Kod ↔ Doküman Tutarlılığı

Her sprint sonu **üçlü güncelleme**:
1. **Kod** + testler yeşil.
2. **Research** [README §8](README.md#8-refactor-sinyalleri--tech-debt) ilgili sinyal
   "çözüldü" işaretlenir; yeniden adlandırma sonrası §3/§4/§9 anchor'ları güncellenir.
3. **Bu plan** durum tablosu (§D) + gerekiyorsa
   [roadmap izleme](../../development-roadmap.md#5-i̇zleme).

> Yeniden adlandırma (D1) ve çekirdek birleştirme (D2) research'teki `dosya:satır`
> referanslarını etkiler — bu yüzden research §9 aynı PR'da güncellenir.

## D. Durum Tablosu

| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 yeşil taban + envanter | 67 test dosyası | ⏳ |
| A | T2 parite/sözleşme | `multi_head_attention.py`, `transformer_encoder_layer.py`, `kv_cache.py` | ⏳ |
| A | T3 bileşen/MoE/quant/RoPE | `mixture_of_experts.py`, `quantization_manager.py`, `positional_encoding.py` | ⏳ |
| B | D1 rename → DecoderLayer | `transformer_encoder_layer.py:59` | ⏳ |
| B | D2 attention çekirdeği | `multi_head_attention.py`, `self_attention.py` | ⏳ |
| B | D3 forward yolu | `transformer_encoder_layer.py:641` | ⏳ |
| B | D4 config sıkılaştırma | `model_initializer.py:167` | ⏳ |
| B | D5 ölü test ayıklama | `test/test_quantum_*` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
