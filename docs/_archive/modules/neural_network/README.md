#  Neural Network Modülü — V6 Mimari Dokümantasyonu

**Versiyon:** V6
**Son Güncelleme:** 2026-03-16
**Durum:** Production-Ready
**Ana Dosya:** `src/neural_network.py`
**Modül Dizini:** `src/neural_network_module/`

---

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [V6 Değişiklik Özeti](#v6-değişiklik-özeti)
3. [Mimari Yapı](#mimari-yapı)
4. [Bileşenler](#bileşenler)
   - [CevahirNeuralNetwork](#cevahirneuralnetwork)
   - [TransformerEncoderLayer](#transformerencoderlayer)
   - [MultiHeadAttention](#multiheadattention)
   - [FeedForwardNetwork](#feedforwardnetwork)
   - [RMSNorm](#rmsnorm)
   - [KVCache](#kvcache)
   - [PositionalEncoding / RoPE](#positionalencoding--rope)
   - [LanguageEmbedding](#languageembedding)
   - [MixtureOfExperts](#mixtureofexperts)
5. [Parametreler ve Konfigürasyon](#parametreler-ve-konfigürasyon)
6. [İleri Düzey Özellikler](#i̇leri-düzey-özellikler)
7. [Performans Notları](#performans-notları)
8. [Kullanım Örnekleri](#kullanım-örnekleri)
9. [Smoke Test](#smoke-test)
10. [Kaldırılan Bileşenler](#kaldırılan-bileşenler)

---

## Genel Bakış

`CevahirNeuralNetwork`, Cevahir-AI projesinin kalbidir. Decoder-only Transformer mimarisine dayanan bu model; GQA (Grouped-Query Attention), YaRN RoPE, SwiGLU, Flash Attention (F.sdpa), QK-Norm, Logit Soft-Cap ve Scaled Residual Init gibi modern LLM tekniklerini entegre etmektedir.

```
Giriş Tokenları
      │
      ▼
LanguageEmbedding (vocab_size → embed_dim)
      │
      ▼
PositionalEncoding / RoPE
      │
      ▼
embed_dropout
      │
      ▼
┌─────────────────────────────────────────────┐
│  TransformerEncoderLayer × N                │
│  ┌───────────────────────────────────────┐  │
│  │  Pre-Norm (RMSNorm)                   │  │
│  │  MultiHeadAttention (GQA, F.sdpa)     │  │
│  │  Residual Connection                  │  │
│  │  Pre-Norm (RMSNorm)                   │  │
│  │  FeedForwardNetwork (SwiGLU)          │  │
│  │  Residual Connection                  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
output_norm (RMSNorm)
      │
      ▼
output_layer (Linear, weight-tied)
      │
      ▼
Logit Soft-Cap: tanh(x / 30.0) * 30.0
      │
      ▼
Çıktı Logitleri (vocab_size)
```

---

## V6 Değişiklik Özeti

### Eklenen Özellikler

| Özellik | Parametre | Referans |
|---------|-----------|----------|
| PyTorch Flash Attention | `use_pytorch_sdpa=True` | PyTorch 2.0+ F.sdpa |
| QK-Norm | `use_qk_norm=False` | Gemma / PaLM 2 |
| Paralel Residual | `parallel_residual=False` | GPT-J, PaLM |
| Output Logit Soft-Cap | `logit_soft_cap=30.0` | Gemma 2 |
| Attention Logit Cap | `attn_logit_cap=0.0` | Gemma 2 |
| Scaled Residual Init | Otomatik | LLaMA / DeepNet |
| SwiGLU ffn_dim Otomatik | `ffn_dim=None` | LLaMA 3 standardı |
| Attention Entropy İzleme | Otomatik | Eğitim sağlığı |

### Kritik Bug Düzeltmeleri (V5→V6)

| Hata | Düzeltme |
|------|----------|
| MHA içi çift residual+norm | MHA artık ham projeksiyon döndürüyor; residual TransformerEncoderLayer'da |
| Q/K/V dropout (yanlış pozisyon) | Q/K/V dropout kaldırıldı; attention weight dropout (post-softmax) kaldı |
| `time.time()` her forward'da | Timing kodu kaldırıldı |
| NaN/Inf kontrolü her forward'da | `if self.debug:` guard'ına alındı |
| SwiGLU: `gate * sigmoid(gate)` | `F.silu(gate)` (kernel-fused, %15-20 daha hızlı) |
| MoE `_moe_losses` bellek sızıntısı | `get_and_reset_moe_loss()` ile temizleme |

### Kaldırılan Bileşenler

| Bileşen | Kaldırılma Tarihi | Nedeni |
|---------|-------------------|--------|
| `DilKatmani` sınıfı | 2026-01-04 | Yerini `LanguageEmbedding` aldı |
| `SeqProjection` | 2026-01-04 | Kullanılmıyordu |
| `MemoryManager` | 2026-01-04 | Cognitive Management katmanına taşındı |
| Memory Manager Module | 2026-01-04 | Ayrı modül olarak kaldırıldı |
| Tensor Adapter Module | 2026-01-04 | Gereksiz soyutlama |

---

## Mimari Yapı

```
src/
├── neural_network.py                          # Ana model: CevahirNeuralNetwork
└── neural_network_module/
    ├── __init__.py
    ├── dil_katmani_module/
    │   ├── language_embedding.py              # Token embedding + init
    │   └── positional_encoding.py            # Sinusoidal / Learned / RoPE / YaRN
    └── ortak_katman_module/
        ├── transformer_encoder_layer.py       # Tek Transformer katmanı
        ├── feed_forward_network.py            # FFN (SwiGLU, GeGLU, GELU vb.)
        ├── rms_norm.py                        # RMSNorm
        ├── kv_cache.py                        # KV Cache (autoregressive)
        ├── mixture_of_experts.py              # MoE (opsiyonel)
        └── attention_manager_module/
            └── multi_head_attention.py        # MHA / GQA / MQA + Flash Attention
```

---

## Bileşenler

### CevahirNeuralNetwork

**Dosya:** `src/neural_network.py`

Ana model sınıfı. Tüm alt bileşenleri birleştirir ve eğitim/çıkarım akışını yönetir.

#### `__init__` Parametreleri

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `vocab_size` | int | — | Kelime dağarcığı boyutu |
| `embed_dim` | int | — | Gömme boyutu |
| `num_heads` | int | — | Attention head sayısı |
| `num_layers` | int | — | Transformer katmanı sayısı |
| `ffn_dim` | int\|None | `None` | FFN iç boyutu; `None` → SwiGLU otomatik |
| `max_seq_len` | int | 2048 | Maksimum sekans uzunluğu |
| `dropout` | float | 0.1 | Dropout oranı |
| `num_kv_heads` | int\|None | `None` | GQA için KV head sayısı; `None` → MHA |
| `use_swiglu` | bool | `True` | SwiGLU aktivasyonu |
| `use_moe` | bool | `False` | Mixture of Experts |
| `num_experts` | int | 8 | MoE uzman sayısı |
| `top_k_experts` | int | 2 | MoE top-K routing |
| `rope_scaling_type` | str\|None | `None` | RoPE ölçekleme: `"yarn"` |
| `rope_scaling_factor` | float | 1.0 | YaRN ölçekleme faktörü |
| `sliding_window` | int\|None | `None` | Sliding Window Attention |
| `use_pytorch_sdpa` | bool | `True` | F.scaled_dot_product_attention (**V6**) |
| `use_qk_norm` | bool | `False` | QK-Norm (**V6**) |
| `parallel_residual` | bool | `False` | Paralel Attention+FFN (**V6**) |
| `logit_soft_cap` | float | `30.0` | Output logit tanh cap (**V6**) |
| `attn_logit_cap` | float | `0.0` | Attention score tanh cap (**V6**) |

#### `forward()` İmzası

```python
def forward(
    self,
    input_ids: torch.Tensor,          # [B, T] — giriş token ID'leri
    attention_mask: Optional[torch.Tensor] = None,  # [B, T]
    use_cache: bool = False,           # KV cache kullan
    past_key_values: Optional[...] = None,
    output_attentions: bool = False,
) -> tuple
```

**Dönüş Değerleri:**
- `(logits, attn_weights)` — cache yoksa
- `(logits, attn_weights, kv_cache)` — `use_cache=True` ise

#### Önemli Metodlar

| Metod | Açıklama |
|-------|----------|
| `forward()` | İleri geçiş |
| `clear_kv_cache()` | **Zorunlu**: Her yeni üretim öncesi çağrılmalı |
| `get_num_params()` | Parametre sayısı |
| `get_model_size_mb()` | Model boyutu (MB) |

> **⚠️ Kritik:** `use_cache=True` ile autoregressive generation yaparken her yeni sekans öncesinde `model.clear_kv_cache()` çağrılmalıdır. Aksi hâlde önceki sekansın önbelleği bir sonraki üretimi kirletir.

---

### TransformerEncoderLayer

**Dosya:** `src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py`

Tek bir Transformer bloğu. Pre-Norm mimarisi varsayılandır (GPT-2'den itibaren endüstri standardı).

#### Pre-Norm Akışı (Varsayılan)

```
x → Norm1 → MHA → + x → Norm2 → FFN → + x
```

#### Post-Norm Akışı (`pre_norm=False`)

```
x → MHA → + x → Norm1 → FFN → + x → Norm2
```

#### Paralel Residual (`parallel_residual=True`, V6)

```python
x_norm = self.norm1(x)
attn_out, _ = self.attention(x_norm, ...)
ffn_out = self.ffn(x_norm)
x = x + attn_out + ffn_out
```

GPT-J ve PaLM'dan alınan bu yaklaşım, tek norm ile her iki branch'i aynı normalize edilmiş girişten besler; hesaplama paralelleştirilebilir.

#### V6 Yeni Parametreler

| Parametre | Açıklama |
|-----------|----------|
| `use_pytorch_sdpa` | MHA'ya iletilir |
| `use_qk_norm` | MHA'ya iletilir |
| `parallel_residual` | GPT-J tarzı paralel branch |
| `attn_logit_cap` | MHA'ya iletilir |

#### MoE Entegrasyonu

```python
# MoE aktif ise FFN yerine MoE kullanılır
if self.use_moe:
    ffn_out, moe_loss = self.moe(x_norm)
    self._moe_losses.append(moe_loss)
else:
    ffn_out = self.ffn(x_norm)
```

```python
# Bellek sızıntısını önlemek için:
loss = layer.get_and_reset_moe_loss()   # Listeyi temizler
```

---

### MultiHeadAttention

**Dosya:** `src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py`

#### Attention Türleri

| Tür | Koşul | Açıklama |
|-----|-------|----------|
| MHA | `num_kv_heads == num_heads` | Standart çok başlı dikkat |
| GQA | `1 < num_kv_heads < num_heads` | Gruplu sorgu dikkati |
| MQA | `num_kv_heads == 1` | Çok-sorgulu dikkat |

#### GQA (Grouped-Query Attention)

```
Q: [B, num_heads, T, head_dim]       → 8 head
K: [B, num_kv_heads, T, head_dim]    → 2 head (gruplandırılmış)
V: [B, num_kv_heads, T, head_dim]    → 2 head

# Her KV grubu num_heads/num_kv_heads = 4 sorgu başına hizmet eder
K = K.expand(-1, num_heads, -1, -1)  # [B, 8, T, head_dim]'a yayılır
```

**Bellek kazancı:** `num_kv_heads=2` → KV cache boyutu %75 azalır.

#### F.scaled_dot_product_attention (V6)

```python
if self.use_pytorch_sdpa:
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout_rate if self.training else 0.0,
        is_causal=True,   # Causal mask otomatik oluşturulur
        scale=None,       # 1/sqrt(head_dim) otomatik
    )
```

PyTorch 2.0+ ile gelen bu API, donanıma ve duruma göre otomatik olarak:
- **Flash Attention** (GPU, FP16/BF16, causal mask)
- **Memory-Efficient Attention** (uzun sequence)
- **Math** (fallback, CPU)

backend'ini seçer. Harici `flash-attn` paketi gerekmez.

#### QK-Norm (V6)

RoPE uygulanmadan önce Q ve K'ya ayrı ayrı RMSNorm uygulanır:

```python
if self.use_qk_norm:
    q = self.q_norm(q)   # [B, H, T, head_dim]
    k = self.k_norm(k)   # [B, H, T, head_dim]
# Ardından RoPE uygulanır
```

Uzun context'te attention logit patlamasını önler (Gemma / PaLM 2 referansı).

#### Attention Logit Cap (V6)

```python
if self.attn_logit_cap > 0.0:
    scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap
```

Softmax öncesinde attention spike'larını bastırır (Gemma 2). `use_pytorch_sdpa=True` iken manuel path atlandığından bu özellik otomatik olarak devre dışı kalır.

#### Sliding Window Attention

```python
# sliding_window != None ise:
# Her token sadece son `sliding_window` token'a dikkat eder
mask = create_sliding_window_mask(seq_len, sliding_window)
```

---

### FeedForwardNetwork

**Dosya:** `src/neural_network_module/ortak_katman_module/feed_forward_network.py`

#### Desteklenen Aktivasyonlar

| Kategori | Aktivasyonlar |
|----------|---------------|
| Gated (kapılı) | `swiglu`, `geglu` |
| Standart | `gelu`, `gelu_tanh`, `relu`, `silu`, `mish` |

#### SwiGLU (Varsayılan)

```python
# Gate projeksiyon + Up projeksiyon + Down projeksiyon
gate = self.gate_proj(x)    # [B, T, ffn_dim]
gate = F.silu(gate)         # SiLU = x * sigmoid(x), kernel-fused
gate *= self.up_proj(x)     # In-place çarpım: 2 tensor → bellek verimli
out = self.down_proj(gate)  # [B, T, embed_dim]
```

**In-place çarpım neden önemli?** Gated aktivasyonlarda naif implementasyon 4 tensor tutar; `gate *= up` ile 2 tensora düşürülür (%50 VRAM tasarrufu).

#### SwiGLU ffn_dim Otomatik Hesaplama (V6)

```python
if use_swiglu and ffn_dim is None:
    raw = int(2 / 3 * 4 * effective_dim)   # LLaMA 3 formülü
    ffn_dim = (raw + 255) // 256 * 256      # 256 katına yuvarla (tensor core optimal)
    # embed_dim=512 → raw=1365 → ffn_dim=1536
```

| embed_dim | ffn_dim (otomatik) |
|-----------|-------------------|
| 256 | 768 |
| 512 | 1536 |
| 768 | 2048 |
| 1024 | 2816 |

#### Bias

```python
use_bias=False  # Varsayılan — LLaMA / PaLM standardı
```

---

### RMSNorm

**Dosya:** `src/neural_network_module/ortak_katman_module/rms_norm.py`

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * scale
```

LayerNorm'dan farkı: mean çıkarma (centering) yoktur, bu da hem hesaplama hem bellek açısından verimlidir.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
```

---

### KVCache

**Dosya:** `src/neural_network_module/ortak_katman_module/kv_cache.py`

Autoregressive generation için anahtar-değer önbelleği.

#### Tensor Boyutları

```
key_cache:   [B, num_kv_heads, max_cache_len, head_dim]
value_cache: [B, num_kv_heads, max_cache_len, head_dim]
```

#### Kullanım

```python
# Her yeni sekans öncesi TEMİZLE
model.clear_kv_cache()

# Generation döngüsü
for step in range(max_new_tokens):
    logits, _, kv_cache = model(
        input_ids=next_token,
        use_cache=True,
        past_key_values=kv_cache
    )
```

---

### PositionalEncoding / RoPE

**Dosya:** `src/neural_network_module/dil_katmani_module/positional_encoding.py`

| Mod | Açıklama |
|-----|----------|
| `sinusoidal` | Sabit sinüs/kosinüs pozisyon kodlaması |
| `learned` | Öğrenilebilir pozisyon gömmeleri |
| `rope` | Rotary Position Embedding |

#### YaRN RoPE (V5+)

```python
rope_scaling_type="yarn"
rope_scaling_factor=4.0   # 2048 → 8192 token context
```

YaRN (Yet another RoPE extensioN), eğitim sırasında görülmemiş uzunluklara interpolasyon yoluyla genelleşebilir.

---

### LanguageEmbedding

**Dosya:** `src/neural_network_module/dil_katmani_module/language_embedding.py`

Token ID'lerini `embed_dim` boyutlu vektörlere dönüştürür.

```python
# Başlatma yöntemleri: "xavier_normal", "normal", "uniform"
init_method="xavier_normal"
scale_by_sqrt=True    # Embed dim'in kareköküne göre ölçekler
padding_idx=0         # PAD token sıfır vektör
```

**Weight Tying:** `output_layer.weight = embedding.weight` (parametre paylaşımı, ~%20 parametre tasarrufu).

---

### MixtureOfExperts

**Dosya:** `src/neural_network_module/ortak_katman_module/mixture_of_experts.py`

Opsiyonel sparse FFN katmanı.

```python
use_moe=True
num_experts=8    # Toplam uzman sayısı
top_k_experts=2  # Her token için aktif uzman
```

**Router:** `nn.Linear(embed_dim, num_experts, bias=False)` — Softmax + Top-K seçimi.

**Sparse dispatch:** Sadece seçilen uzmanlar token'larını işler (~8x bellek azalması).

**Load Balancing:** Auxiliary loss ile her uzman eşit kullanılmaya teşvik edilir.

> **⚠️ Uyarı:** `use_moe=True` checkpoint'larla `use_moe=False` checkpoint'lar uyumsuz. MoE etkinleştirmek sıfırdan eğitim gerektirir.

---

## Parametreler ve Konfigürasyon

### Önerilen V6 Konfigürasyonu

```python
model_config = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 2,          # GQA: %75 KV cache azalması
    "num_layers": 8,
    "ffn_dim": None,             # SwiGLU otomatik: 1536
    "use_swiglu": True,
    "use_pytorch_sdpa": True,    # Flash Attention (PyTorch 2.0+)
    "use_qk_norm": True,         # Uzun context stabilitesi
    "parallel_residual": False,  # V7'de değerlendir
    "logit_soft_cap": 30.0,      # Gemma 2 standardı
    "attn_logit_cap": 0.0,       # Gerekirse 50.0 dene
    "dropout": 0.1,
    "max_seq_len": 2048,
    "rope_scaling_type": "yarn", # Context extension
    "rope_scaling_factor": 4.0,
    "use_moe": False,
}
```

### Küçük Model (Test/Geliştirme)

```python
model_config = {
    "vocab_size": 32000,
    "embed_dim": 128,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_layers": 2,
    "ffn_dim": None,
    "use_swiglu": True,
    "use_pytorch_sdpa": True,
    "logit_soft_cap": 30.0,
    "dropout": 0.0,
    "max_seq_len": 512,
}
```

---

## İleri Düzey Özellikler

### Scaled Residual Init (V6)

Derin modellerde varyans büyümesini önler:

```python
# num_layers=8 → scale ≈ 0.25
scale = 1.0 / math.sqrt(2.0 * num_layers)
for layer in self.layers:
    nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=scale)
    nn.init.normal_(layer.ffn.fc2.weight, mean=0.0, std=scale)
```

Referans: LLaMA / DeepNet — büyük modellerde eğitim stabilitesini artırır.

### Output Logit Soft-Cap (V6)

```python
if self.logit_soft_cap > 0.0:
    logits = torch.tanh(logits / self.logit_soft_cap) * self.logit_soft_cap
```

`logit_soft_cap=30.0` ile çıktı logitleri `[-30, +30]` aralığına yumuşakça sıkıştırılır. Bu:
- Extreme logit'leri bastırır
- Eğitim stabilitesini artırır
- Softmax'ta sayısal taşmayı önler

### Attention Entropy İzleme

Her eğitim adımında tüm katmanların attention weight entropisi hesaplanır:

```python
# Normalize Shannon entropy: H / log(T)
if normalized_h < 0.05:
    logger.warning(f"Attention collapse! layer={i}, entropy={normalized_h:.4f}")
```

**Durum Etiketleri:**
- `collapse`: entropy < 0.05 → Model tüm dikkati tek bir konuma yöneltiyor
- `normal`: 0.05 ≤ entropy < 0.8 → Sağlıklı aralık
- `uniform`: entropy ≥ 0.8 → Dikkat çok yayılmış (underfitting belirtisi)

`_last_snapshot["attn_entropy"]` ile durum sorgulanabilir.

---

## Performans Notları

### Bellek Karşılaştırması

| Konfigürasyon | VRAM (embed=512, 8 katman, seq=2048) |
|---------------|--------------------------------------|
| MHA (baseline) | ~4.2 GB |
| GQA (num_kv_heads=2) | ~3.1 GB (%26 azalma) |
| GQA + SwiGLU in-place | ~2.8 GB |
| GQA + F.sdpa | ~2.1 GB (%50 azalma) |

### Hız Karşılaştırması

| Özellik | Etki |
|---------|------|
| F.silu vs gate*sigmoid | %15-20 hız artışı (kernel fusion) |
| F.sdpa vs manuel SDPA | %30-40 hız artışı (Flash Attention) |
| Debug guard (NaN kontrol) | 2-3x yavaşlama yok (sadece debug modda) |
| `time.time()` kaldırıldı | Küçük overhead azalması |

---

## Kullanım Örnekleri

### Model Oluşturma

```python
import torch
from src.neural_network import CevahirNeuralNetwork

model = CevahirNeuralNetwork(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,
    num_layers=8,
    ffn_dim=None,           # Otomatik: 1536
    use_swiglu=True,
    use_pytorch_sdpa=True,
    use_qk_norm=True,
    logit_soft_cap=30.0,
    dropout=0.1,
    max_seq_len=2048,
)

print(f"Parametre sayısı: {model.get_num_params():,}")
print(f"Model boyutu: {model.get_model_size_mb():.1f} MB")
```

### İleri Geçiş (Eğitim)

```python
input_ids = torch.randint(0, 32000, (4, 512))  # [batch=4, seq=512]
logits, attn_weights = model(input_ids)
# logits: [4, 512, 32000]

loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, 32000),
    input_ids[:, 1:].reshape(-1)
)
loss.backward()
```

### Autoregressive Generation

```python
model.eval()
model.clear_kv_cache()   # ⚠️ Zorunlu

input_ids = torch.tensor([[1, 234, 567]])  # [1, 3]

with torch.no_grad():
    for _ in range(100):
        logits, _, kv_cache = model(
            input_ids=input_ids[:, -1:],
            use_cache=True,
            past_key_values=kv_cache if _ > 0 else None
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
```

### MoE Loss Toplama (eğer aktifse)

```python
total_loss = task_loss
for layer in model.layers:
    moe_loss = layer.get_and_reset_moe_loss()
    if moe_loss is not None:
        total_loss = total_loss + 0.01 * moe_loss
```

---

## Smoke Test

```python
import torch
from src.neural_network import CevahirNeuralNetwork

# Model oluştur
model = CevahirNeuralNetwork(
    vocab_size=32000, embed_dim=512, num_heads=8, num_kv_heads=2,
    num_layers=2, ffn_dim=None, use_swiglu=True,
    use_pytorch_sdpa=True, use_qk_norm=True, logit_soft_cap=30.0
)

x = torch.randint(0, 32000, (2, 128))  # [batch=2, seq=128]
logits, _ = model(x)

# Temel kontroller
assert logits.shape == (2, 128, 32000), f"Shape hatası: {logits.shape}"
assert not torch.isnan(logits).any(), "NaN tespit edildi!"
assert logits.abs().max().item() <= 30.0 + 1e-3, "Soft cap çalışmıyor!"

# QK-Norm kontrolü
assert hasattr(model.layers[0].attn, "q_norm"), "q_norm yok! use_qk_norm=True olmalı"

# SwiGLU ffn_dim kontrolü
assert model.layers[0].ffn.ffn_dim == 1536, f"ffn_dim beklenen=1536, gerçek={model.layers[0].ffn.ffn_dim}"

# KV head kontrolü (GQA)
assert model.layers[0].attn.num_kv_heads == 2, "GQA num_kv_heads hatası"

print("V6 smoke test: PASSED ✓")
```

---

## Kaldırılan Bileşenler

Aşağıdaki bileşenler V6 öncesinde projeden çıkarılmıştır. Eski kodlarda referans görülürse güncellenmelidir:

### DilKatmani → LanguageEmbedding

```python
# ESKİ (KALDIRILDI):
from src.neural_network_module.dil_katmani_module import DilKatmani

# YENİ:
from src.neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding
```

### SeqProjection (KALDIRILDI)

```python
# ESKİ (KALDIRILDI):
self.seq_projection = SeqProjection(embed_dim, output_dim)

# YENİ: Direkt nn.Linear kullan
self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
```

### MemoryManager → Cognitive Management

```python
# ESKİ (KALDIRILDI):
from src.neural_network_module import MemoryManager

# YENİ: Bellek yönetimi cognitive_management modülündedir
from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
```

---

## Bağımlılıklar

| Kütüphane | Versiyon | Amaç |
|-----------|----------|------|
| `torch` | ≥ 2.0 | `F.scaled_dot_product_attention` |
| `torch` | ≥ 1.9 | Temel model |

> **Not:** V6'nın Flash Attention özelliği `torch >= 2.0` gerektirir. Daha eski PyTorch sürümlerinde `use_pytorch_sdpa=False` ayarlayın.

---

*Yazar: Muhammed Yasin Yılmaz — Cevahir-AI Projesi*
*Telif Hakkı: © 2024-2026 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.*
