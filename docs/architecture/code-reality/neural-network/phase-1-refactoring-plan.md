# Phase 1 — Refactoring Plan · Neural Network

> **Kanonik Birim #2 · Neural Network** — `src/`
> [research (mevcut gerçek durum)](README.md) ile eşleşir. Akış: önce **Test Fazı (A)**
> → sonra **Geliştirme Fazı (B)**. Üst plan: [development-roadmap](../../development-roadmap.md).
>
> **Tutarlılık kuralı:** Her sprint sonu kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt)
> + durum tablosu birlikte güncellenir.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `src/neural_network.py` + `src/neural_network_module/` |
| Boyut | ~7.450 LOC |
| Mevcut test | **67 test dosyası / ~11.276 LOC** (en zengin paketlerden) |
| Refactor hedefleri | `TransformerEncoderLayer` yanıltıcı isim, attention kod ikizliği, config default doldurma, `_parallel_forward_impl` |
| Kritik sözleşmeler | `forward` çıktı şekli (B,T,V) · weight tying (`seq_proj_dim==embed_dim`) · attention backend parite |

> ⚠️ Ortam notu: `torch`/`pytest` kurulu değil; canlı koşu geliştirme ortamında.

---

## A. TEST FAZI (önce test)

### A.1 Test Reality — Mevcut Durum

| Alan | Test | Durum |
|------|------|-------|
| Tam ağ | `test_neural_network.py`, `_comprehensive.py` (4882!), `_v2.py`, `_architecture_validation.py`, `_real_config_validation.py` | ✅ Çok kapsamlı |
| Attention | `test_multi_head_attention.py`, `test_self_attention.py`, `test_cross_attention.py`, `test_attention_*` | ✅ Kapsamlı |
| Normalizasyon | `test_layer/group/instance/batch_normalization.py`, `test_normalization_*` | ✅ Var |
| FFN | `test_feed_forward_network.py` | ✅ Var |
| MoE | `test_load_balancer.py` | 🟡 Kısmi |
| Embedding | `test_language_embedding.py`, `test_embedding_*` | ✅ Var |
| Paralel/tensor/scaling/residual/quantum/memory | ilgili `test_*` | ✅ Bol (bazıları deneysel) |

### A.2 Kusur Envanteri (test edilip doğrulanacak)

| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **Attention backend paritesi** — flash / pytorch-sdpa / standard aynı sonucu vermeli | [§5.1](README.md#51-dikkat--üç-backend-multiheadattention) | 3 backend sayısal parite (tolerans) |
| T-02 | **İki forward yolu** — `_forward_impl` vs `_parallel_forward_impl` eşdeğer mi | [§8](README.md#8-refactor-sinyalleri--tech-debt) | İki yol aynı çıktı testi |
| T-03 | **KV cache doğruluğu** — cache'li vs cache'siz üretim aynı olmalı | [§5.4](README.md#54-kv-cache--sliding-window) | Eşdeğerlik + sliding-window eviction |
| T-04 | **Weight tying** invariant | [§4](README.md#41-tam-model-montajı) | tie=True'da embed↔output ağırlık kimliği |
| T-05 | **MoE aux loss** eğitim kaybına doğru ekleniyor mu | [§5.5](README.md#55-moe) | load-balance loss testi |
| T-06 | **Quantization round-trip** (fp16/bf16/int8) | [§5.6](README.md#56-quantization) | quantize→dequantize davranış |
| T-07 | **Deneysel/ölü testler** — "quantum_*" testlerinin bağlamı belirsiz | [§8](README.md#8-refactor-sinyalleri--tech-debt) | Kapsam denetimi; ölü testi ayıkla |

### A.3 Olması Gereken Test Durumu

- **Şekil/invariant** testleri her katman için (giriş→çıkış boyutu, dtype/device).
- **Parite** testleri: attention backend'leri, iki forward yolu, KV cache.
- **Sayısal** testler: RoPE dönüşü, RMSNorm, SwiGLU kapısı — referans değerlere karşı.
- Deneysel testler ayıklanır; kanonik kapsam netleşir.

### A.4 Test Sprint'leri

**T1 — Yeşil taban + envanter:** 67 dosyayı koş, pass/fail yaz; "quantum_*" ve
şüpheli testleri işaretle.
**T2 — Parite/invariant:** T-01 (attention), T-02 (forward), T-03 (KV cache),
T-04 (tying).
**T3 — Bileşen sayısal + MoE/quant:** T-05, T-06; RoPE/RMSNorm/SwiGLU sayısal doğrulama.

---

## B. GELİŞTİRME FAZI (sonra geliştirme)

### B.1 Hedef Mimari

```
CevahirNeuralNetwork  (tek montaj noktası — değişmez)
  ├── AttentionCore  (paylaşılan SDPA çekirdeği)
  │      ├── MultiHeadAttention (backend seçimi: flash|sdpa|standard)
  │      └── SelfAttention (aynı çekirdeği kullanır — ikiz kod yok)
  ├── DecoderLayer  (eski TransformerEncoderLayer — tek forward yolu netleşir)
  └── (diğer bileşenler aynı)
```

### B.2 Geliştirme Sprint'leri

**D1 — Yeniden adlandırma** *(düşük risk)*: `TransformerEncoderLayer → DecoderLayer`
(geriye dönük alias ile). Koruyan: mevcut tam-ağ testleri.
**D2 — Attention çekirdeği birleştirme** *(orta risk)*: `MultiHeadAttention` ve
`SelfAttention` ortak SDPA çekirdeğini paylaşsın. Önkoşul: T-01 parite testleri.
**D3 — Forward yolu netleştirme** *(orta risk)*: `_parallel_forward_impl`'i ya
kanonik yap ya kaldır. Önkoşul: T-02.
**D4 — Config sıkılaştırma** *(düşük risk)*: `ModelInitializer`'ın örtük default
doldurmasını açık şema doğrulamasına bağla (Model/Engine ile ortak).
**D5 — Ölü test/kod ayıklama** *(düşük risk)*: deneysel "quantum_*" kapsamını netleştir.

### B.3 Korunacak Sözleşmeler
`forward` imzası ve çıktı şekli; weight tying kuralı; attention çıktı eşdeğerliği;
`ModelManager.build_model` inşa sözleşmesi.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint sonu: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) +
aşağıdaki tablo.

## D. Durum Tablosu

| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 yeşil taban | ⏳ |
| A | T2 parite/invariant | ⏳ |
| A | T3 bileşen/MoE/quant | ⏳ |
| B | D1 rename | ⏳ |
| B | D2 attention çekirdeği | ⏳ |
| B | D3 forward yolu | ⏳ |
| B | D4 config sıkılaştırma | ⏳ |
| B | D5 ölü test ayıklama | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
