# Cevahir API — Modül Dokümantasyonu

**Versiyon:** V-4 (Unified Inference API)
**Dosya:** `model/cevahir.py`
**Satır Sayısı:** ~2084
**Amaç:** TokenizerCore + ModelManager + CognitiveManager bileşenlerini tek bir inference API'ında birleştiren facade katmanı.

---

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari](#mimari)
3. [Exception Hiyerarşisi](#exception-hiyerarşisi)
4. [Yardımcı Sınıflar](#yardımcı-sınıflar)
5. [CevahirConfig](#cevahirconfig)
6. [CevahirModelAPI](#cevahirmodelapi)
7. [Cevahir Ana Sınıfı](#cevahir-ana-sınıfı)
8. [API Referansı](#api-referansı)
9. [Kullanım Örnekleri](#kullanım-örnekleri)
10. [Tasarım Desenleri](#tasarım-desenleri)
11. [Eğitim Sistemi ile Fark](#eğitim-sistemi-ile-fark)

---

## Genel Bakış

`cevahir.py`, Cevahir-AI sisteminin **tek giriş noktasıdır (Unified Inference API)**. Sadece inference (çıkarım) için tasarlanmıştır. Eğitim sürecinde bu dosya kullanılmaz.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cevahir (Facade)                         │
│              ⚠️  SADECE INFERENCE — EĞİTİM DEĞİL                │
├─────────────────────┬───────────────────┬───────────────────────┤
│    TokenizerCore    │   ModelManager    │   CognitiveManager    │
│  encode / decode    │ forward / predict │  process / memory     │
│  BPE tokenization   │ save / load       │  tools / critic       │
│  GPU batch encode   │ KV Cache          │  Self-Refine          │
└─────────────────────┴───────────────────┴───────────────────────┘
```

### Eğitim vs Inference Ayrımı

| Görev | Kullanılacak Dosya |
|---|---|
| BPE vocab/merges eğitimi | `tokenizer_management/train_bpe.py` |
| Model eğitimi giriş noktası | `training_system/train.py` |
| Eğitim servisi | `training_system/training_service.py` |
| Neural network eğitimi | `model_management/model_manager.py` |
| **Inference / Chat / Üretim** | **`model/cevahir.py`** ← burası |

---

## Mimari

```
cevahir.py
├── Exception Hiyerarşisi
│   ├── CevahirError (kök)
│   ├── CevahirInitializationError
│   ├── CevahirConfigurationError
│   └── CevahirProcessingError (context + suggestion)
│
├── Yardımcı Sınıflar (internal)
│   ├── _ErrorContextBuilder     → Hata mesajı + öneri üretimi
│   └── _InputValidator          → str/list/Tensor → [B, T] Tensor
│
├── Decorator'lar
│   ├── @requires_tokenizer
│   ├── @requires_model_manager
│   ├── @requires_cognitive_manager
│   └── @requires_model_api
│
├── CevahirConfig (dataclass)    → Tüm konfigürasyon
│
├── CognitiveModelAPI (Protocol) → CognitiveManager'ın beklediği interface
│
├── CevahirModelAPI              → Adapter Pattern
│   ├── generate()               → Autoregressive üretim (KV Cache)
│   ├── _autoregressive_generate() → Core üretim döngüsü
│   ├── _generate_with_beam_search() → Beam search üretim
│   ├── score()                  → Log-probability scoring
│   ├── entropy_estimate()       → Shannon entropy ölçümü
│   └── process_audio/image/multimodal()
│
└── Cevahir (ana sınıf)          → Public API
    ├── Tokenization API         → encode / decode / train_tokenizer
    ├── Model API                → forward / generate / predict / freeze
    ├── Cognitive API            → process / generate_batch / memory / tools
    ├── Monitoring               → get_metrics / get_health_status
    └── Properties               → tokenizer / model / device / cognitive
```

---

## Exception Hiyerarşisi

```
CevahirError (Exception)
├── CevahirInitializationError   → Bileşen başlatma hatası
├── CevahirConfigurationError    → Konfigürasyon hatası
└── CevahirProcessingError       → İşlem hatası
     ├── .context: Dict          → Ek bağlam bilgisi
     └── .suggestion: str        → Kullanıcıya önerisini içerir
```

### CevahirProcessingError

```python
class CevahirProcessingError(CevahirError):
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ): ...
```

`suggestion` alanı, hata sonrasında kullanıcıya ne yapması gerektiğini açıklar. Örneğin TokenizerCore hatalarında `"Ensure tokenizer vocab and merges files exist and are valid"` gibi bir öneri iletilir.

---

## Yardımcı Sınıflar

### `_ErrorContextBuilder`

Bileşenlerden gelen Exception'ları yakalayarak yapılandırılmış hata mesajları ve öneriler üretir.

```python
msg, suggestion = _ErrorContextBuilder.build_error_message(
    operation="Forward pass",
    error=exc,
    component="ModelManager"
)
# msg      → "Forward pass failed (component: ModelManager): ..."
# suggestion → "Verify model is initialized and model weights are loaded correctly"
```

Bileşen bazlı öneriler:

| Bileşen | Öneri |
|---|---|
| `TokenizerCore` | vocab/merges dosyalarını kontrol et |
| `ModelManager` | model başlatıldı mı, ağırlıklar yüklendi mi |
| `CognitiveManager` | konfigürasyon ve bağımlılıkları kontrol et |

---

### `_InputValidator`

`forward()` gibi metodlara geçilen girişleri normalize eder.

```python
tensor = _InputValidator.validate_and_convert_input(
    inputs=inputs,       # str | List[int] | torch.Tensor
    device="cuda",
    tokenizer_core=tok,  # str dönüşümü için
    vocab_size=50000     # clamp için
)
```

**Dönüşüm kuralları:**

| Girdi Tipi | İşlem |
|---|---|
| `str` | TokenizerCore ile encode → `[1, T]` Tensor |
| `List[int]` | `torch.tensor([ids])` → `[1, T]` Tensor |
| `torch.Tensor` | `.long()`, 1D ise `unsqueeze(0)` → `[B, T]` Tensor |
| Boş girdi | PAD token ile `[1, 1]` Tensor |

Vocab aralığı dışı token ID'leri `vocab_size - 1`'e kırpılır.

---

## CevahirConfig

Tüm sistemi yapılandıran ana `dataclass`.

```python
@dataclass
class CevahirConfig:
    device: str = "cuda"
    seed: Optional[int] = 42
    log_level: str = "INFO"

    # Tokenizer yapılandırması (dict)
    tokenizer: Dict[str, Any] = field(default_factory=dict)

    # Model yapılandırması — V-4 varsayılanları:
    model: Dict[str, Any] = field(default_factory=lambda: {
        "vocab_size": 50000,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "ffn_dim": 2048,
        "pe_mode": "rope",
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_kv_cache": True,
        "use_moe": False,
        "dropout": 0.1,
        "max_seq_len": 2048,
    })

    # CognitiveManager yapılandırması
    cognitive: Optional[CognitiveManagerConfig] = None

    # Checkpoint yükleme
    load_model_path: Optional[str] = None  # None → auto-detect

    # Config dosyası (hot reload için)
    config_path: Optional[str] = None
    enable_hot_reload: bool = False
```

### `validate()`

Config geçerliliğini kontrol eder; geçersiz değer varsa `CevahirConfigurationError` fırlatır.

### dict'ten CevahirConfig

```python
config = CevahirConfig(**my_dict)
```

---

## CevahirModelAPI

`ModelManager`'ı `CognitiveManager`'ın beklediği `CognitiveModelAPI` Protocol'üne uyarlayan **Adapter sınıfı**.

```
CevahirModelAPI
├── Implements: CognitiveModelAPI Protocol
├── Wraps: ModelManager + TokenizerCore
└── Methods:
    ├── generate(prompt, decoding_cfg) → str
    ├── score(prompt, candidate)       → float
    └── entropy_estimate(text)         → float [0, 1]
```

### `generate(prompt, decoding_cfg)` → `str`

Autoregressive üretim — KV Cache tabanlı.

```
Akış:
1. Prompt → encode → [1, T] Tensor
2. KV Cache temizle (önceki tur kirliliğini önler)
3. İlk forward: tüm prompt → cache initialize
4. Sonraki forward'lar: sadece son token (cache_position ile)
5. Repetition penalty → Temperature → Top-k → Top-p → multinomial
6. EOS gelince dur (minimum 5 token üretme koşullu)
7. Yeni token'ları decode et (prompt token'ları dışarıda)
```

**Parametre sınırları:**

| Parametre | Aralık |
|---|---|
| `max_new_tokens` | `[0, 2048]` |
| `top_p` | `[0.01, 1.0]` |
| `top_k` | `>= 0` (0 = kapalı) |
| `repetition_penalty` | `> 1.0` aktif, son 256 token üzerinde uygulanır |

**EOS davranışı:** Minimum 5 token üretildikten sonra EOS'a izin verilir — erken collapse önlemi.

**NaN koruması:** Tüm prob'lar NaN/sıfır ise uniform fallback, son çare olarak ilk token seçilir.

---

### `_generate_with_beam_search(prompt, max_new_tokens, beam_width)`

```
Akış:
1. beam_width adet beam başlat (hepsi aynı prompt)
2. Her beam için forward → top (beam_width × 2) aday
3. Adayları log-probability toplamına göre sırala
4. En iyi beam_width adayı seç
5. Tüm beam'ler EOS'a ulaşınca veya max_new_tokens dolunca dur
6. En yüksek skorlu beam decode edilir
7. Hata durumunda standard generation'a fallback
```

---

### `score(prompt, candidate)` → `float [0, 1]`

Prompt verildiğinde candidate metnin olasılığını ölçer.

```
avg_log_prob = mean(log P(candidate_token_i | prompt + candidate[:i]))
score = clamp((avg_log_prob + 10) / 10, 0.0, 1.0)
```

Uzunluk bazlı heuristic fallback mevcuttur.

---

### `entropy_estimate(text)` → `float [0, 1]`

Modelin next-token dağılımı üzerinden Shannon entropi ölçümü.

```python
H = -sum(p_i * log(p_i))            # Raw Shannon entropy
normalized = H / log(vocab_size)     # [0, 1]
# 0 → model emin (tek dominant token)
# 1 → model belirsiz (uniform dağılım)
```

**Akademik referans:** Kuhn et al. 2023 (active learning, LLM calibration)

**Heuristic fallback:** Model forward başarısız olursa Type-Token Ratio (TTR) kullanılır.

---

## Cevahir Ana Sınıfı

```python
class Cevahir:
    """Unified API: TokenizerCore + ModelManager + CognitiveManager"""
```

### `__init__`

```python
Cevahir(
    config: Union[CevahirConfig, Dict[str, Any]],
    *,
    tokenizer_core: Optional[TokenizerCore] = None,  # Dependency injection
    model_manager: Optional[ModelManager] = None,    # Dependency injection
    cognitive_manager: Optional[CognitiveManager] = None,  # Dependency injection
)
```

Başlatma sırası:

```
1. Config parse + validate
2. Seed set (reproducibility için)
3. Log level set
4. TokenizerCore init (veya inject)
5. ModelManager init + model auto-load (veya inject)
6. CevahirModelAPI adapter oluştur
7. CognitiveManager init (veya inject)
```

**Model auto-load:** `load_model_path` None ise `saved_models/cevahir_model.pth`'e bakılır. Bulunamazsa random init ile devam edilir (uyarı verilir).

**ModelManager başlatma modu:** Inference only — `build_optimizer=False, build_criterion=False, build_scheduler=False`

---

## API Referansı

### Decorator Güvenceleri

| Decorator | Koşul | Hata |
|---|---|---|
| `@requires_tokenizer` | `_tokenizer_core` mevcut | `CevahirProcessingError` |
| `@requires_model_manager` | `_model_manager` mevcut | `CevahirProcessingError` |
| `@requires_cognitive_manager` | `_cognitive_manager` mevcut | `CevahirProcessingError` |
| `@requires_model_api` | `_model_api` mevcut | `CevahirProcessingError` |

---

### Tokenization API

#### `encode(text, mode="inference") → Tuple[List[str], List[int]]`

```python
tokens, ids = cevahir.encode("Merhaba dünya")
# tokens → ["Mer", "haba", " dü", "nya"]
# ids    → [142, 883, 2041, 567]
```

#### `decode(token_ids) → str`

```python
text = cevahir.decode([142, 883, 2041, 567])
# → "Merhaba dünya"
```

#### `train_tokenizer(corpus)`

Tokenizer'ı verilen corpus üzerinde eğitir. (BPE eğitimi)

---

### Model API

#### `forward(inputs, **kwargs) → torch.Tensor`

```python
logits = cevahir.forward("Merhaba")       # str giriş
logits = cevahir.forward([142, 883])      # list[int] giriş
logits = cevahir.forward(tensor)          # Tensor giriş
# → [B, T, vocab_size]
```

`_InputValidator` ile tip dönüşümü otomatik yapılır.

---

#### `generate(prompt, ...) → str`

```python
text = cevahir.generate(
    prompt="Türkiye'nin başkenti",
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    use_cognitive_pipeline=True,   # Default: True
)
```

**Routing mantığı:**

```
use_cognitive_pipeline=True + CognitiveManager mevcut
  → CognitiveManager.handle(state, input, decoding=...)
     (Self-Refine + Memory + Critic aktif)
  → Hata durumunda: fallback to direct generation

use_cognitive_pipeline=False VEYA CognitiveManager yok
  → CevahirModelAPI.generate(prompt, decoding_cfg)
     (raw autoregressive üretim)
```

---

#### `predict(inputs, topk=1, apply_softmax=True, return_logits=False) → Dict`

```python
result = cevahir.predict("Merhaba", topk=5)
# → {"predictions": [...], "probabilities": [...]}
```

---

#### `freeze(patterns) → Dict`

```python
report = cevahir.freeze(["embedding.*", "layers.0"])
# → {"frozen": [...], "skipped": [...]}
```

#### `unfreeze(patterns) → Dict`

```python
report = cevahir.unfreeze("layers.*")
```

---

#### `save_model(path) / load_model(path)`

```python
cevahir.save_model("checkpoints/v6_best.pt")
cevahir.load_model("checkpoints/v6_best.pt")
```

---

#### `train_mode() / eval_mode()`

```python
cevahir.eval_mode()   # Inference için (dropout kapalı)
cevahir.train_mode()  # Fine-tuning için (dropout açık)
```

---

#### `configure_tensorboard(...)`

```python
cevahir.configure_tensorboard(
    log_dir="runs/v6",
    log_every_n=100,
    log_histograms=True,
    log_attention_image=True,
    enable=True
)
writer = cevahir.get_tb_writer()
```

---

#### `update_model(update_params, dry_run=False) → Dict`

ModelManager üzerinden parametre güncellemesi yapar.

---

### Cognitive API

#### `process(text, state=None) → CognitiveOutput`

```python
output = cevahir.process("Bana yapay zeka hakkında bilgi ver")
print(output.text)
```

Tüm CognitiveManager pipeline'ından (Critic, Memory, Deliberation) geçer.

---

#### `generate_batch(prompts, ...) → List[str]`

```python
results = cevahir.generate_batch(
    ["Soru 1", "Soru 2", "Soru 3"],
    max_new_tokens=64,
    temperature=0.7
)
```

Sequential işleme (şu an). Bir prompt başarısız olursa diğerleri devam eder, hatalı için `""` döner.

---

#### `process_batch(texts, states=None) → List[CognitiveOutput]`

```python
outputs = cevahir.process_batch(["Metin 1", "Metin 2"])
```

Her metin için bağımsız CognitiveState kullanılır.

---

#### `add_memory(note)` / `search_memory(query, limit=5) → List[Dict]`

```python
cevahir.add_memory("Kullanıcı Python tercih ediyor")
results = cevahir.search_memory("programlama dili", limit=3)
```

---

#### `register_tool(name, func, description, parameters)` / `list_tools()`

```python
def hesapla(expr: str) -> float:
    return eval(expr)

cevahir.register_tool(
    name="hesapla",
    func=hesapla,
    description="Matematiksel işlem yapar",
    parameters={"expr": {"type": "string"}}
)
tools = cevahir.list_tools()
```

---

### Training API (DEPRECATED)

```python
cevahir.train(...)
# → DeprecationWarning + CevahirProcessingError
# Eğitim için: python training_system/train.py
```

`Cevahir.train()` her zaman hata fırlatır. Gerçek eğitim için `training_system/train.py` kullanılmalıdır.

---

### Monitoring

```python
metrics = cevahir.get_metrics()
# → CognitiveManager'dan gelen metrik sözlüğü

health = cevahir.get_health_status()
# → {"status": "healthy", ...}
```

---

### Properties

| Property | Dönüş | Açıklama |
|---|---|---|
| `cevahir.tokenizer` | `TokenizerCore` | TokenizerCore'a erişim |
| `cevahir.model` | `ModelManager` | ModelManager'a erişim |
| `cevahir.cognitive` | `CognitiveManager` | CognitiveManager'a erişim |
| `cevahir.device` | `torch.device` | Modelin bulunduğu cihaz |
| `cevahir.is_initialized` | `bool` | Model başlatıldı mı |

---

## Kullanım Örnekleri

### Temel Inference

```python
from model.cevahir import Cevahir, CevahirConfig

config = CevahirConfig(
    device="cuda",
    model={
        "vocab_size": 50000,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "pe_mode": "rope",
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_kv_cache": True,
        "dropout": 0.1,
    },
    tokenizer={
        "vocab_path": "tokenizer_management/data/vocab.json",
        "merges_path": "tokenizer_management/data/merges.json",
    },
    load_model_path="saved_models/cevahir_v6_best.pt",
)

cevahir = Cevahir(config)
```

---

### Metin Üretimi

```python
# Cognitive pipeline ile (varsayılan)
yanit = cevahir.generate(
    "Türkiye'nin başkenti nedir?",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    use_cognitive_pipeline=True,
)

# Ham model üretimi (cognitive bypass)
yanit = cevahir.generate(
    "Türkiye'nin başkenti nedir?",
    use_cognitive_pipeline=False,
)
```

---

### Encode / Decode

```python
tokens, ids = cevahir.encode("Merhaba dünya")
print(f"Tokens: {tokens}")  # ['Mer', 'haba', ' dün', 'ya']
print(f"IDs: {ids}")        # [142, 883, 2041, 567]

text = cevahir.decode(ids)
print(text)  # "Merhaba dünya"
```

---

### Belirsizlik Ölçümü

```python
entropy = cevahir.model_api.entropy_estimate("Bu sorunun cevabı")
# 0.0 → model emin
# 1.0 → model belirsiz
print(f"Entropy: {entropy:.3f}")
```

---

### Dependency Injection

```python
from model_management.model_manager import ModelManager
from tokenizer_management.core.tokenizer_core import TokenizerCore

# Önceden hazırlanmış bileşenlerle
tok = TokenizerCore(tok_config)
mm = ModelManager(model_config)
mm.initialize(build_optimizer=False)

cevahir = Cevahir(
    config,
    tokenizer_core=tok,
    model_manager=mm,
)
```

---

### Batch İşlem

```python
prompts = ["Soru 1", "Soru 2", "Soru 3"]
results = cevahir.generate_batch(prompts, max_new_tokens=50)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}")
    print()
```

---

### TensorBoard

```python
cevahir.configure_tensorboard(
    log_dir="runs/cevahir_v6",
    log_every_n=50,
    log_histograms=True,
    enable=True
)
writer = cevahir.get_tb_writer()
# writer ile manuel log yapılabilir
```

---

## Tasarım Desenleri

### Facade Pattern

`Cevahir`, üç büyük bileşeni (TokenizerCore, ModelManager, CognitiveManager) tek bir tutarlı API altında gizler. Kullanıcı her bileşenin iç detaylarını bilmeden çalışabilir.

### Adapter Pattern

`CevahirModelAPI`, `ModelManager`'ın arayüzünü `CognitiveManager`'ın beklediği `CognitiveModelAPI` Protocol'üne dönüştürür.

```
ModelManager  →→  CevahirModelAPI  →→  CognitiveManager
(karmaşık API)    (protocol uyumu)     (Protocol: generate, score, entropy_estimate)
```

### Decorator Pattern

Bileşen bağımlılıklarını merkezi olarak yönetir. Her metodun başına `if not self._xxx: raise` yazmak yerine:

```python
@requires_model_manager
def forward(self, inputs, **kwargs):
    # Direkt iş mantığı — kontrol zaten yapıldı
    ...
```

### Dependency Injection

```python
Cevahir(config, tokenizer_core=..., model_manager=..., cognitive_manager=...)
```

Bileşenler dışarıdan enjekte edilebilir. Test, profiling ve özel senaryolarda mock/özel bileşen kullanımını kolaylaştırır.

---

## Eğitim Sistemi ile Fark

```
cevahir.py (INFERENCE)          training_system/ (EĞİTİM)
─────────────────────            ─────────────────────────────
ModelManager.initialize(         ModelManager.initialize(
  build_optimizer=False,           build_optimizer=True,
  build_criterion=False,           build_criterion=True,
  build_scheduler=False            build_scheduler=True
)                                )

Optimizer yok                    AdamW / AdamW8bit
Loss hesabı yok                  CrossEntropy / BCE / MSE
Gradient yok                     Gradient accumulation
Cevahir.train() → HATA           training_service.train_epoch()
```

`Cevahir` sınıfı bilinçli olarak eğitim kapasitesinden arındırılmıştır. Yanlış kullanım anında uyarılır:

```python
cevahir.train(data)
# DeprecationWarning: Cevahir.train() DEPRECATED!
# CevahirProcessingError: Eğitim için training_system/train.py kullanın
```

---

## Factory Function

```python
from model.cevahir import create_cevahir, CevahirConfig

config = CevahirConfig(device="cuda", ...)
cevahir = create_cevahir(config)
```

`create_cevahir()` doğrudan `Cevahir(config)` çağrısına eşdeğerdir; dependency injection parametrelerini de destekler.

---

*Yazar: Muhammed Yasin Yılmaz | Telif Hakkı © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.*
