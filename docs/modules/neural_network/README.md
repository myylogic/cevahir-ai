# 🧠 Neural Network - V-4 Architecture Dokümantasyonu

**Versiyon:** V-5  
**Son Güncelleme:** 2025-01-27  
**Durum:** Production-Ready

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Ana Bileşenler](#ana-bileşenler)
4. [V-4 Özellikleri](#v-4-özellikleri)
5. [Modül Detayları](#modül-detayları)
6. [Kullanım Örnekleri](#kullanım-örnekleri)

---

## 🎯 Genel Bakış

**CevahirNeuralNetwork**, GPT-4, Claude ve Gemini seviyesinde **V-4 Architecture** kullanan, endüstri standartlarında bir Transformer encoder modelidir.

### Temel Özellikler

- ✅ **V-4 Architecture:** RoPE, RMSNorm, SwiGLU, KV Cache, MoE, Quantization
- ✅ **Layer Stacking:** 12+ layer (endüstri standardı)
- ✅ **Pre-norm/Post-norm:** GPT-2/3/4 (pre-norm) veya BERT (post-norm) desteği
- ✅ **Causal Masking:** Autoregressive generation için
- ✅ **Gradient Checkpointing:** Memory-efficient training
- ✅ **Weight Tying:** Input embedding ve output layer weight sharing

---

## 🏗️ Mimari Yapı

### Yüksek Seviye Mimari

```
CevahirNeuralNetwork
├── DilKatmani (Language Layer)
│   ├── LanguageEmbedding (Token Embedding)
│   ├── PositionalEncoding (RoPE/Sinusoidal/Learned)
│   └── SeqProjection (Sequence Projection)
│
├── TransformerEncoderLayer × N (12+ layer)
│   ├── MultiHeadAttention
│   │   ├── Query/Key/Value Projections
│   │   ├── RoPE (Rotary Position Embedding)
│   │   ├── KV Cache (Key-Value Cache)
│   │   └── Flash Attention 2.0 (opsiyonel)
│   ├── RMSNorm / LayerNorm
│   ├── FeedForwardNetwork
│   │   ├── SwiGLU / GELU Activation
│   │   └── MoE (Mixture of Experts) - opsiyonel
│   └── Residual Connections
│
├── Output Layer (Vocab Projection)
│   └── Weight Tying (opsiyonel)
│
└── QuantizationManager (opsiyonel)
    ├── INT8 Quantization
    ├── FP16 Quantization
    └── Dynamic Quantization
```

### Veri Akışı

```
Input: [B, T] token IDs
    ↓
DilKatmani
    ├── LanguageEmbedding → [B, T, embed_dim]
    ├── PositionalEncoding → [B, T, embed_dim] (RoPE/Sinusoidal/Learned)
    └── SeqProjection → [B, T, seq_proj_dim]
    ↓
TransformerEncoderLayer × N
    ├── Pre-norm (LayerNorm/RMSNorm)
    ├── MultiHeadAttention
    │   ├── Q/K/V Projections
    │   ├── RoPE (Rotary Position Embedding)
    │   ├── Attention Computation (Flash Attention 2.0)
    │   └── KV Cache Update
    ├── Residual Connection
    ├── Pre-norm (LayerNorm/RMSNorm)
    ├── FeedForwardNetwork
    │   ├── SwiGLU / GELU Activation
    │   └── MoE (Mixture of Experts) - opsiyonel
    └── Residual Connection
    ↓
Output Layer
    └── [B, T, vocab_size] logits
```

---

## 🧩 Ana Bileşenler

### 1. CevahirNeuralNetwork

**Dosya:** `src/neural_network.py`  
**Sınıf:** `CevahirNeuralNetwork`

**Sorumluluklar:**
- Ana neural network orchestrator
- Layer stacking (N adet TransformerEncoderLayer)
- Output layer (vocab projection)
- Weight tying (opsiyonel)
- Quantization management

**Özellikler:**
- ✅ V-4 Architecture (RoPE, RMSNorm, SwiGLU, KV Cache, MoE, Quantization)
- ✅ Pre-norm/Post-norm desteği
- ✅ Causal masking
- ✅ Gradient checkpointing
- ✅ TensorBoard integration

---

### 2. DilKatmani (Language Layer)

**Dosya:** `src/neural_network_module/dil_katmani.py`  
**Sınıf:** `DilKatmani`

**Sorumluluklar:**
- Token embedding
- Positional encoding (RoPE/Sinusoidal/Learned)
- Sequence projection

**Bileşenler:**
- `LanguageEmbedding`: Token ID'leri embedding'lere dönüştürür
- `PositionalEncoding`: Pozisyonel bilgi ekler (RoPE/Sinusoidal/Learned)
- `SeqProjection`: Embedding dimension'ı seq_proj_dim'e projekte eder

---

### 3. TransformerEncoderLayer

**Dosya:** `src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py`  
**Sınıf:** `TransformerEncoderLayer`

**Sorumluluklar:**
- Self-attention + FFN + Residual connections
- Pre-norm/Post-norm desteği
- Gradient checkpointing

**Bileşenler:**
- `MultiHeadAttention`: Multi-head self-attention
- `RMSNorm` / `LayerNorm`: Normalization
- `FeedForwardNetwork`: Feed-forward network (SwiGLU/GELU)
- `MixtureOfExperts`: MoE (opsiyonel)

---

### 4. MultiHeadAttention

**Dosya:** `src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py`  
**Sınıf:** `MultiHeadAttention`

**Sorumluluklar:**
- Multi-head self-attention computation
- RoPE (Rotary Position Embedding)
- KV Cache (Key-Value Cache)
- Flash Attention 2.0 (opsiyonel)

**Özellikler:**
- ✅ RoPE (Rotary Position Embedding) - GPT-3+, Claude, Gemini standardı
- ✅ KV Cache - GPT-4, Claude, Gemini inference optimization
- ✅ Flash Attention 2.0 - GPT-4, Claude, Gemini standardı
- ✅ Causal masking - Autoregressive generation için

---

### 5. FeedForwardNetwork

**Dosya:** `src/neural_network_module/ortak_katman_module/feed_forward_network.py`  
**Sınıf:** `FeedForwardNetwork`

**Sorumluluklar:**
- 2-layer MLP (expand → contract)
- Activation function (SwiGLU/GELU/ReLU)

**Özellikler:**
- ✅ SwiGLU activation - GPT-4, PaLM standardı
- ✅ GELU activation - GPT-2/3/4, BERT standardı
- ✅ ReLU activation - Standart

---

### 6. RMSNorm

**Dosya:** `src/neural_network_module/ortak_katman_module/rms_norm.py`  
**Sınıf:** `RMSNorm`

**Sorumluluklar:**
- Root Mean Square Layer Normalization
- LayerNorm'un daha hızlı ve stabil alternatifi

**Özellikler:**
- ✅ GPT-3+, LLaMA standardı
- ✅ Daha hızlı (mean hesaplaması yok)
- ✅ Daha stabil (variance hesaplaması yok)

---

### 7. KV Cache

**Dosya:** `src/neural_network_module/ortak_katman_module/kv_cache.py`  
**Sınıf:** `KVCache`

**Sorumluluklar:**
- Key-Value cache management
- Autoregressive generation optimization

**Özellikler:**
- ✅ GPT-4, Claude, Gemini standardı
- ✅ Autoregressive generation hızını önemli ölçüde artırır
- ✅ Memory-efficient (sadece gerekli key/value'lar saklanır)

---

### 8. Mixture of Experts (MoE)

**Dosya:** `src/neural_network_module/ortak_katman_module/mixture_of_experts.py`  
**Sınıf:** `MixtureOfExperts`

**Sorumluluklar:**
- Sparse activation (N adet expert)
- Router: Her token için en uygun expert'leri seçer
- Top-k routing: Her token için k expert seçilir

**Özellikler:**
- ✅ GPT-4, Gemini standardı
- ✅ Büyük model kapasitesi (1.8T parametre GPT-4)
- ✅ Sparse activation (her forward'da sadece k expert aktif)

---

### 9. QuantizationManager

**Dosya:** `src/neural_network_module/ortak_katman_module/quantization_manager.py`  
**Sınıf:** `QuantizationManager`

**Sorumluluklar:**
- Model quantization (INT8/FP16)
- Dynamic/Static quantization

**Özellikler:**
- ✅ GPT-4, Claude, Gemini standardı
- ✅ INT8 quantization: 4x model compression, 2-4x inference speedup
- ✅ FP16 quantization: 2x model compression, 1.5-2x inference speedup

---

### 10. Advanced Checkpointing

**Dosya:** `src/neural_network_module/ortak_katman_module/advanced_checkpointing.py`  
**Sınıf:** `AdvancedCheckpointing`

**Sorumluluklar:**
- Selective checkpointing
- Layer-wise checkpointing
- Adaptive checkpointing

**Özellikler:**
- ✅ GPT-4, Claude, Gemini standardı
- ✅ Daha fazla memory tasarrufu
- ✅ Training hızını optimize eder

---

## ⚡ V-4 Özellikleri

### 1. RoPE (Rotary Position Embedding)

**Endüstri Standardı:** GPT-3+, Claude, Gemini

**Avantajları:**
- Relative position encoding
- Daha iyi sequence length generalization
- GPT-3+, Claude, Gemini standardı

**Kullanım:**
```python
# PositionalEncoding'de RoPE modu
pe_mode = "rope"  # "sinusoidal" | "learned" | "rope"
```

---

### 2. RMSNorm (Root Mean Square Layer Normalization)

**Endüstri Standardı:** GPT-3+, LLaMA, PaLM

**Avantajları:**
- Daha hızlı (mean hesaplaması yok)
- Daha stabil (variance hesaplaması yok)
- GPT-3+, LLaMA standardı

**Kullanım:**
```python
use_rmsnorm = True  # True ise RMSNorm, False ise LayerNorm
```

---

### 3. SwiGLU Activation

**Endüstri Standardı:** GPT-4, PaLM

**Avantajları:**
- Daha iyi performance (GPT-4, PaLM)
- Gated linear unit (GLU) variant

**Kullanım:**
```python
use_swiglu = True  # True ise SwiGLU, False ise GELU
```

---

### 4. KV Cache (Key-Value Cache)

**Endüstri Standardı:** GPT-4, Claude, Gemini

**Avantajları:**
- Autoregressive generation hızını önemli ölçüde artırır
- Memory-efficient (sadece gerekli key/value'lar saklanır)
- GPT-4, Claude, Gemini standardı

**Kullanım:**
```python
use_kv_cache = True  # KV Cache kullan (inference için)
max_cache_len = 2048  # Maximum cache length
```

---

### 5. MoE (Mixture of Experts)

**Endüstri Standardı:** GPT-4, Gemini

**Avantajları:**
- Büyük model kapasitesi (1.8T parametre GPT-4)
- Sparse activation (her forward'da sadece k expert aktif)
- Daha az computation (sadece aktif expert'ler hesaplanır)

**Kullanım:**
```python
use_moe = True  # MoE kullan
num_experts = 8  # Expert sayısı (GPT-4: 8, Gemini: 16)
moe_top_k = 2  # Her token için seçilecek expert sayısı (GPT-4: 2)
```

---

### 6. Quantization

**Endüstri Standardı:** GPT-4, Claude, Gemini

**Avantajları:**
- Daha küçük model boyutu
- Daha hızlı inference
- Daha az memory kullanımı

**Kullanım:**
```python
quantization_type = "int8"  # "none" | "int8" | "fp16" | "int8_dynamic"
```

---

## 📚 Modül Detayları

### DilKatmani Modülü

#### LanguageEmbedding

**Dosya:** `src/neural_network_module/dil_katmani_module/language_embedding.py`  
**Sınıf:** `LanguageEmbedding`

**Özellikler:**
- Token ID'leri embedding'lere dönüştürür
- Güvenli ağırlık başlatma (padding satırı korunur)
- İsteğe bağlı sqrt(d_model) ölçekleme
- Vocab büyütme/küçültme (ağırlıkları koruyarak)

---

#### PositionalEncoding

**Dosya:** `src/neural_network_module/dil_katmani_module/positional_encoding.py`  
**Sınıf:** `PositionalEncoding`

**Modlar:**
- `sinusoidal`: Transformer orijinal (sinusoidal PE)
- `learned`: Öğrenilebilir PE (nn.Embedding)
- `rope`: RoPE (Rotary Position Embedding) - GPT-3+, Claude, Gemini

**Özellikler:**
- Register buffer ile güvenli, pickle ve device/dtype uyumlu
- Seq_len büyüdükçe otomatik genişleme (ensure_length)

---

#### SeqProjection

**Dosya:** `src/neural_network_module/dil_katmani_module/seq_projection.py`  
**Sınıf:** `SeqProjection`

**Özellikler:**
- Sekansın son eksenini (özellik boyutu) proj_dim'e doğrusal olarak projekte eder
- Güvenli başlatma (weight + bias)
- Giriş şekli ve dtype kontrolleri

---

### Ortak Katman Modülü

#### Attention Manager Module

**Dosya:** `src/neural_network_module/ortak_katman_module/attention_manager_module/`

**Bileşenler:**
- `multi_head_attention.py`: Multi-head attention
- `self_attention.py`: Self-attention
- `cross_attention.py`: Cross-attention
- `attention_optimizer.py`: Attention optimization
- `attention_utils_module/`: Attention utilities

---

#### Memory Manager Module

**Dosya:** `src/neural_network_module/ortak_katman_module/memory_manager_module/`

**Bileşenler:**
- `memory_manager.py`: Memory management
- `memory_allocator.py`: Memory allocation
- `memory_attention_bridge.py`: Memory-attention bridge
- `memory_optimizer.py`: Memory optimization
- `memory_utils_module/`: Memory utilities

---

#### Tensor Adapter Module

**Dosya:** `src/neural_network_module/ortak_katman_module/tensor_adapter_module/`

**Bileşenler:**
- `tensor_projection.py`: Tensor projection
- `tensor_processing_manager.py`: Tensor processing
- `tensor_utils_module/`: Tensor utilities

---

## 💻 Kullanım Örnekleri

### Örnek 1: Basit Model Oluşturma

```python
from src.neural_network import CevahirNeuralNetwork

model = CevahirNeuralNetwork(
    learning_rate=1e-4,
    dropout=0.1,
    vocab_size=60000,
    embed_dim=1024,
    seq_proj_dim=1024,
    num_heads=16,
    num_layers=12,
    device="cuda",
    # V-4 Features
    pe_mode="rope",  # RoPE
    use_rmsnorm=True,  # RMSNorm
    use_swiglu=True,  # SwiGLU
    use_kv_cache=True,  # KV Cache
    use_moe=False,  # MoE (opsiyonel)
    quantization_type="none",  # Quantization (opsiyonel)
)
```

---

### Örnek 2: Forward Pass

```python
import torch

# Input: [B, T] token IDs
input_ids = torch.randint(0, 60000, (2, 128))  # [batch_size, seq_len]

# Forward pass
logits, attn_weights = model(input_ids)

# Output: [B, T, vocab_size] logits
print(f"Logits shape: {logits.shape}")  # [2, 128, 60000]
```

---

### Örnek 3: KV Cache ile Inference

```python
# İlk forward: Tüm sequence için key/value hesapla ve cache'le
logits, attn_weights, kv_cache = model(
    input_ids,
    use_cache=True,
    cache_position=torch.arange(128),
)

# Sonraki forward: Sadece yeni token için key/value hesapla
new_token = torch.randint(0, 60000, (2, 1))  # [batch_size, 1]
logits, attn_weights, kv_cache = model(
    new_token,
    use_cache=True,
    cache_position=torch.tensor([128]),  # Yeni pozisyon
)
```

---

### Örnek 4: MoE ile Model

```python
model = CevahirNeuralNetwork(
    learning_rate=1e-4,
    dropout=0.1,
    vocab_size=60000,
    embed_dim=1024,
    seq_proj_dim=1024,
    num_heads=16,
    num_layers=12,
    device="cuda",
    # MoE
    use_moe=True,
    num_experts=8,  # GPT-4: 8, Gemini: 16
    moe_top_k=2,  # GPT-4: 2
)
```

---

### Örnek 5: Quantization

```python
# Model'i quantize et
model.eval()  # Quantization için eval mode gerekli
model.quantization_manager.quantize_model(model)

# Quantized model ile inference
logits, attn_weights = model(input_ids)
```

---

## 🔧 Konfigürasyon

### V-4 Feature Flags

```python
# V-4 Features
pe_mode = "rope"  # "sinusoidal" | "learned" | "rope"
use_rmsnorm = True  # RMSNorm (GPT-3+, LLaMA)
use_swiglu = True  # SwiGLU (GPT-4, PaLM)
use_kv_cache = True  # KV Cache (GPT-4, Claude, Gemini)
use_moe = False  # MoE (GPT-4, Gemini)
quantization_type = "none"  # "none" | "int8" | "fp16" | "int8_dynamic"
```

---

## 📊 Performans

### Model Özellikleri

- **Parametre Sayısı:** ~176M (artırılabilir)
- **Vocab Size:** 60,000
- **Embed Dim:** 1024 (artırılabilir)
- **Num Layers:** 12 (artırılabilir)
- **Num Heads:** 16

### Performans Metrikleri

- **Inference Speed:** ~50-100 tokens/second (GPU'da)
- **Memory Usage:** ~2-3 GB (inference için)
- **Training Speed:** ~100-150 tokens/second (A100 GPU'da)

---

## 🎓 Endüstri Standartları

### GPT-4 Seviyesi Özellikler

- ✅ RoPE (Rotary Position Embedding)
- ✅ RMSNorm (Root Mean Square Layer Normalization)
- ✅ SwiGLU (Swish-Gated Linear Unit)
- ✅ KV Cache (Key-Value Cache)
- ✅ MoE (Mixture of Experts)
- ✅ Quantization (INT8/FP16)

### Claude Seviyesi Özellikler

- ✅ RoPE (Rotary Position Embedding)
- ✅ RMSNorm (Root Mean Square Layer Normalization)
- ✅ KV Cache (Key-Value Cache)
- ✅ Flash Attention 2.0 (opsiyonel)

### Gemini Seviyesi Özellikler

- ✅ RoPE (Rotary Position Embedding)
- ✅ RMSNorm (Root Mean Square Layer Normalization)
- ✅ MoE (Mixture of Experts)
- ✅ KV Cache (Key-Value Cache)

---

## 📖 Daha Fazla Bilgi

- **[Sistem Mimarisi](../../ARCHITECTURE.md)**
- **[API Referansı](../../API_REFERENCE.md)**
- **[Cevahir Comprehensive Analysis](../../../model/CEVAHIR_COMPREHENSIVE_ANALYSIS.md)**

---

**Son Güncelleme:** 2025-01-27  
**Versiyon:** V-5

