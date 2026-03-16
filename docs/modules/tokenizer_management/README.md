#  Tokenizer Management - Kapsamlı Dokümantasyon

**Versiyon:** V-5  
**Son Güncelleme:** 2025-01-27  
**Durum:** Production-Ready | Türkçe Optimize Edilmiş BPE Tokenizer  
**Lisans:** Public API için hazır

---

##  İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Çalışma Prensibi](#çalışma-prensibi)
4. [Türkçe Özel Özellikler](#türkçe-özel-özellikler)
5. [API Referansı](#api-referansı)
6. [BPE Training Pipeline](#bpe-training-pipeline)
7. [Encoding/Decoding Pipeline](#encodingdecoding-pipeline)
8. [Tokenization Modülleri](#tokenization-modülleri)
9. [Kullanım Örnekleri](#kullanım-örnekleri)
10. [Public API Hazırlığı](#public-api-hazırlığı)

---

## 🎯 Genel Bakış

**Tokenizer Management**, Cevahir Sinir Sistemi'nin Türkçe için özel olarak tasarlanmış Byte Pair Encoding (BPE) tokenizer sistemidir. Bu sistem, Türkçe dilinin morfolojik yapısına uygun olarak geliştirilmiş ve endüstri standartlarında (GPT-4, Claude, Gemini) bir tokenizer kalitesine sahiptir.

### Temel Özellikler

-  **Türkçe Optimize:** Heceleme, morfoloji, ünlü uyumu, ünsüz benzeşmesi
-  **BPE Training:** Deterministik, tekrar üretilebilir merge stratejisi
-  **Hybrid Tokenization:** Whole words + syllables + morphology
-  **GPU Support:** Hızlandırılmış batch processing
-  **Production-Ready:** Atomic file operations, error handling, logging
-  **Config-Driven:** Tamamen yapılandırılabilir parametreler
-  **Public API Ready:** Harici kullanım için hazır

---

##  Mimari Yapı

### Yüksek Seviye Mimari

```
TokenizerCore (Ana API)
├── BPEManager (Orkestrasyon)
│   ├── BPEEncoder (Encoding)
│   ├── BPEDecoder (Decoding)
│   ├── BPETrainer (Training)
│   └── Tokenization Pipeline
│       ├── Pretokenizer (Metin ön işleme)
│       ├── Syllabifier (Türkçe heceleme)
│       ├── Morphology (Kök çıkarımı)
│       └── Postprocessor (Post-processing)
│
├── DataLoaderManager (Veri yükleme)
├── Config System (Yapılandırma)
└── Utilities (Yardımcı fonksiyonlar)
```

### Veri Akışı

```
Raw Text
    ↓
Pretokenizer
    ├── Unicode normalization (NFC)
    ├── Lowercase (Türkçe: İ→i, I→ı)
    ├── Cleanup (non-word removal)
    └── Whitespace tokenization
    ↓
Tokenization Options
    ├── Whole Words (kelime</w>)
    ├── Syllables (he-ce-le-me)
    └── Morphology (kök + ekler)
    ↓
BPE Training / Encoding
    ├── Token frequencies
    ├── Merge operations
    └── Vocab expansion
    ↓
Token IDs
    ├── [BOS] + token_ids + [EOS]
    └── Validated (in-range check)
    ↓
Decoding
    ├── ID → Token mapping
    ├── BPE merge reversal
    ├── </w> removal
    └── Postprocessor (punctuation, spaces)
    ↓
Clean Text Output
```

---

##  Çalışma Prensibi

### 1. TokenizerCore - Ana API

**Dosya:** `tokenizer_management/core/tokenizer_core.py`  
**Sınıf:** `TokenizerCore`

**Sorumluluklar:**
- BPE training orchestrasyonu
- Encoding/Decoding API sağlama
- DataLoaderManager entegrasyonu
- Training data hazırlama
- GPU batch processing

**Lifecycle:**
```
TokenizerCore.__init__()
    ├── Load config
    ├── Initialize DataLoaderManager (optional)
    ├── Initialize BPEManager
    │   ├── Load/create vocab
    │   ├── Load merges
    │   └── Initialize components (Encoder, Decoder, Trainer)
    └── Set default encoding modes
```

---

### 2. BPEManager - Orkestrasyon Katmanı

**Dosya:** `tokenizer_management/bpe/bpe_manager.py`  
**Sınıf:** `BPEManager`

**Sorumluluklar:**
- Encoder/Decoder/Trainer yaşam döngüsü
- Vocab & merges atomik okuma/yazma
- Train/Inference modu yönetimi
- Türkçe tokenization pipeline
- Format conversion (sentiment, instruction → QA)

**Key Methods:**

#### `train(corpus, **kwargs)`
BPE eğitimi yapar:
1. Corpus tokenization (whole words + syllables)
2. Token frequency calculation
3. Initial vocab selection (top N tokens)
4. Base alphabet guarantee
5. BPE merge training (iterative)
6. Vocab expansion
7. Atomic save (vocab.json, merges.txt)

#### `encode(text, mode, **kwargs)`
Metni tokenize edip ID'lere çevirir:
1. Pretokenization (Unicode, cleanup)
2. Tokenization (words/syllables/morphology)
3. BPE encoding (merge application)
4. Special tokens (BOS/EOS)
5. OOV handling (syllable fallback)

#### `decode(token_ids, **kwargs)`
ID'leri metne çevirir:
1. ID → Token mapping
2. BPE merge reversal
3. Prefer mode (word/syllable/auto)
4. Special token removal
5. Postprocessor (punctuation, spaces)

---

### 3. BPETrainer - BPE Eğitimi

**Dosya:** `tokenizer_management/bpe/bpe_trainer.py`  
**Sınıf:** `BPETrainer`

**Algoritma:**
```
1. Tokenized corpus → single sequence
2. Apply existing merges (deterministic)
3. Iterative merge training:
   a. Calculate pair frequencies
   b. Select best pair (freq DESC, lex ASC)
   c. Merge pair in sequence
   d. Add merged token to vocab
   e. Repeat until target_merges or no pairs
4. Save merges (ranked list)
```

**Özellikler:**
-  Deterministik seçim (freq + lexicographic)
-  Special token protection
-  GPU batch processing (optional)
-  Memory-efficient (streaming for large corpus)
-  Progress tracking (log intervals)

---

### 4. BPEEncoder - Encoding

**Dosya:** `tokenizer_management/bpe/bpe_encoder.py`  
**Sınıf:** `BPEEncoder`

**Strateji:**
```
Token Encoding:
1. Direct vocab hit → single ID
2. Special token → single ID
3. BPE merge → sub-tokens → ID list
4. Heuristic split → parts → encode each
5. Character-level fallback → char IDs
```

**Özellikler:**
-  BPE merge application (rank-based)
-  Character-level fallback (OOV handling)
-  GPU acceleration (optional)
-  Reverse vocab mapping (ID → token)

---

### 5. BPEDecoder - Decoding

**Dosya:** `tokenizer_management/bpe/bpe_decoder.py`  
**Sınıf:** `BPEDecoder`

**Strateji:**
```
ID Decoding:
1. ID → Token mapping
2. Segment filtering (word/syllable prefer)
3. </w> removal + space insertion
4. Special token removal
5. Postprocessor:
   - Punctuation spacing
   - Whitespace collapse
   - Sentence capitalization (Türkçe: İ→i)
```

**Prefer Modes:**
- `"word"`: Whole words (kelime</w>) tercih edilir
- `"syllable"`: Syllables tercih edilir
- `"auto"`: Her parça için otomatik seçim

---

## 🇹🇷 Türkçe Özel Özellikler

### 1. Pretokenizer

**Dosya:** `tokenizer_management/bpe/tokenization/pretokenizer.py`

**Özellikler:**
-  Unicode normalization (NFC, Türkçe karakterler korunur)
-  Türkçe lowercase (İ→i, I→ı)
-  Noktalama ayrıştırma (.,!? ayrı token)
-  Whitespace preservation (boşluk tokenları korunur)
-  Valid token filtering (Türkçe karakterler + ASCII)

**Örnek:**
```python
pretokenizer = Pretokenizer()
tokens = pretokenizer.tokenize("Merhaba, nasılsın?")
# → ["Merhaba", ",", "nasılsın", "?"]
```

---

### 2. Syllabifier - Türkçe Heceleme

**Dosya:** `tokenizer_management/bpe/tokenization/syllabifier.py`

**Kurallar:**
1. Her hece bir ünlü çekirdeği içermeli
2. İki ünlü arası sessiz sayısı:
   - 0-1 sessiz: V-CV
   - 2 sessiz: V-C1C2V (izinli onset cluster)
   - ≥3 sessiz: V-C1C2...CV
3. Son hece kalan tüm harfleri alır

**İzinli Onset Cluster'lar:**
```python
["bl", "br", "pl", "pr", "fl", "fr", "kl", "kr",
 "gl", "gr", "dr", "tr", "str", "skr", "spr", "ps", "ks", "ns"]
```

**Örnek:**
```python
syllabifier = Syllabifier()
syllables = syllabifier.syllabify_word("merhaba")
# → ["mer", "ha", "ba"]
```

---

### 3. Morphology - Kök Çıkarımı

**Dosya:** `tokenizer_management/bpe/tokenization/morphology.py`

**Özellikler:**
-  TDK uyumlu kök çıkarımı
-  Ünlü uyumu kontrolü
-  Ünsüz benzeşmesi
-  Yumuşama kuralları (p→b, ç→c, t→d, k→ğ/g)
-  Olumsuzluk ekleri (ma/me)
-  Zaman ekleri (dı/di/du/dü)
-  Kişi ekleri (im/ın/un/ün)
-  Bileşik fiil desteği (yardım etti → yardım et)

**Örnek:**
```python
morphology = Morphology()
root = morphology.find_root("geldim")
# → "gel"
morphemes = morphology.analyze(["geldim"])
# → ["gel", "di", "m"]
```

---

### 4. Postprocessor

**Dosya:** `tokenizer_management/bpe/tokenization/postprocessor.py`

**Özellikler:**
-  Noktalama boşluk düzeltme
-  Whitespace collapse
-  Sentence capitalization (Türkçe: İ→i)
-  Multiple punctuation reduction (... → .)

**Örnek:**
```python
postprocessor = Postprocessor()
text = postprocessor.process(["Merhaba", ",", "nasılsın", "?"])
# → "Merhaba, nasılsın?"
```

---

##  API Referansı

### TokenizerCore

**Dosya:** `tokenizer_management/core/tokenizer_core.py`

#### `__init__(config: Dict[str, Any])`

**Parametreler:**
- `vocab_path` / `vocab_file`: Vocab dosya yolu
- `merges_path` / `merges_file`: Merges dosya yolu
- `data_dir`: Training verisi dizini (optional)
- `use_gpu`: GPU desteği (default: False)
- `batch_size`: GPU batch size (default: 32)

**Örnek:**
```python
config = {
    "vocab_path": "data/vocab_lib/vocab.json",
    "merges_path": "data/merges_lib/merges.txt",
    "data_dir": "education",
    "use_gpu": True,
}
tokenizer = TokenizerCore(config)
```

---

#### `train_model(corpus, **kwargs)`

BPE eğitimi yapar.

**Parametreler:**
- `corpus`: `List[str]` - Eğitim corpus'u
- `method`: `"bpe"` (sadece desteklenen)
- `vocab_size`: `Optional[int]` - Vocab size limiti
- `max_iter`: `Optional[int]` - Max iteration (default: config'ten)
- `min_frequency`: `Optional[int]` - Min pair frequency (default: 2)
- `include_whole_words`: `Optional[bool]` - Whole words ekle
- `include_syllables`: `Optional[bool]` - Syllables ekle
- `include_sep`: `Optional[bool]` - <SEP> ekle
- `sample_ratio`: `Optional[float]` - Corpus sampling (0.0-1.0)

**Örnek:**
```python
corpus = ["Merhaba dünya", "Nasılsın?", ...]
tokenizer.train_model(
    corpus,
    method="bpe",
    vocab_size=60000,
    max_iter=50000,
    include_whole_words=True,
    include_syllables=True,
    sample_ratio=0.1  # %10 sampling
)
```

---

#### `train_from_loader(**kwargs)`

DataLoaderManager ile veri yükleyip eğitim yapar.

**Örnek:**
```python
tokenizer.train_from_loader(
    method="bpe",
    vocab_size=60000,
    include_whole_words=True,
)
```

---

#### `encode(text, mode, **kwargs) -> Tuple[List[str], List[int]]`

Metni tokenize edip ID'lere çevirir.

**Parametreler:**
- `text`: `str` - Girdi metni
- `mode`: `"train"` | `"inference"` - Encoding modu
- `include_whole_words`: `Optional[bool]` - Whole words
- `include_syllables`: `Optional[bool]` - Syllables
- `include_sep`: `Optional[bool]` - <SEP> token

**Dönüş:**
- `Tuple[List[str], List[int]]` - (tokens, token_ids)

**Örnek:**
```python
tokens, ids = tokenizer.encode(
    "Merhaba dünya",
    mode="inference",
    include_whole_words=True,
    include_syllables=False,
)
```

---

#### `decode(ids, **kwargs) -> str`

ID listesini metne çevirir.

**Parametreler:**
- `ids`: `List[int]` - Token ID listesi
- `method`: `"bpe"` | `"raw"` - Decode metodu
- `remove_specials`: `bool` - Special token'ları kaldır (default: True)
- `remove_tags`: `bool` - Tag'leri kaldır (default: True)
- `collapse_spaces`: `bool` - Boşlukları birleştir (default: True)
- `lowercase`: `bool` - Küçük harfe çevir (default: False)
- `prefer`: `"word"` | `"syllable"` | `"auto"` - Token tercihi

**Örnek:**
```python
text = tokenizer.decode(
    [2, 10, 20, 30, 3],  # [BOS, token_ids..., EOS]
    method="bpe",
    remove_specials=True,
    prefer="word",
)
```

---

#### `load_training_data(**kwargs) -> List[Tuple[List[int], List[int]]]`

Training verisi hazırlar (QA pairs → input_ids, target_ids).

**Parametreler:**
- `encode_mode`: `"train"` | `"inference"` - Encoding modu
- `include_whole_words`: `Optional[bool]`
- `include_syllables`: `Optional[bool]`
- `include_sep`: `Optional[bool]`

**Dönüş:**
- `List[Tuple[List[int], List[int]]]` - [(input_ids, target_ids), ...]

**Örnek:**
```python
examples = tokenizer.load_training_data(
    encode_mode="inference",
    include_whole_words=True,
)
```

---

### BPEManager

**Dosya:** `tokenizer_management/bpe/bpe_manager.py`

#### `encode(text, mode, **kwargs) -> Tuple[List[str], List[int]]`

BPEManager encode metodu (TokenizerCore üzerinden kullanılır).

#### `decode(token_ids, **kwargs) -> str`

BPEManager decode metodu.

#### `train(corpus, **kwargs)`

BPE eğitimi (BPEManager üzerinden).

---

## 🔄 BPE Training Pipeline

### 1. Corpus Preparation

```python
# train_bpe.py
qa_loader = DataLoaderManager(DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN
))
qa_data = qa_loader.load()  # List[Tuple[str, str]]

raw_loader = DataLoaderManager(DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.TEXT_INFER
))
raw_data = raw_loader.load()  # List[str]

# Hybrid corpus
corpus = []
for q, a in qa_data:
    corpus.extend([q, a])
corpus.extend(raw_data)
```

---

### 2. Tokenization

```python
# BPEManager.train()
for item in corpus:
    tokens = _tokenize_with_punct(
        item,
        include_whole_words=True,
        include_syllables=True,
        include_sep=False,
    )
    # → ["<BOS>", "merhaba</w>", "dünya</w>", "<EOS>"]
```

---

### 3. Initial Vocab Selection

```python
# Token frequency calculation
token_frequencies = _calculate_token_frequencies(tokenized_corpus)

# Top N tokens seç (initial_vocab_ratio)
initial_vocab_size = int(max_vocab_size * initial_vocab_ratio)
selected_tokens = _select_top_tokens(all_tokens, token_frequencies, initial_vocab_size)

# Base alphabet guarantee
_ensure_base_alphabet_in_vocab()  # Türkçe karakterler + punctuation
```

---

### 4. BPE Merge Training

```python
# BPETrainer.train()
sequence = corpus_to_sequence(tokenized_corpus)  # Single sequence

for iteration in range(max_iter):
    # 1. Pair frequency calculation
    pair_stats = _get_pair_stats(sequence, min_frequency)
    
    # 2. Best pair selection (freq DESC, lex ASC)
    best_pair = _select_best_pair(pair_stats)
    
    # 3. Merge pair
    sequence = _merge_pair_linear(sequence, best_pair)
    
    # 4. Add to vocab
    merged_token = best_pair[0] + best_pair[1]
    nid = next_id(vocab)
    vocab[merged_token] = {"id": nid, "total_freq": 0, "positions": []}
    
    # 5. Save merge
    merges.append(best_pair)
```

---

### 5. Save Results

```python
# Atomic save
save_vocab()  # vocab.json (atomic write)
save_merges()  # merges.txt (atomic write)
```

---

## 🔀 Encoding/Decoding Pipeline

### Encoding Flow

```
Input Text: "Merhaba dünya"
    ↓
Pretokenizer
    → Unicode NFC
    → Lowercase (Türkçe)
    → Cleanup
    → ["merhaba", "dünya"]
    ↓
Tokenization Options
    → Whole words: ["merhaba</w>", "dünya</w>"]
    → Syllables: ["mer", "ha", "ba", "dün", "ya"]
    → Morphology: ["merhaba", "dünya"]
    ↓
BPE Encoding
    → Apply merges (rank-based)
    → ["merhaba</w>", "dünya</w>"]
    → Encode to IDs: [10, 20]
    ↓
Special Tokens
    → [BOS] + [10, 20] + [EOS]
    → [2, 10, 20, 3]
    ↓
Output: (["<BOS>", "merhaba</w>", "dünya</w>", "<EOS>"], [2, 10, 20, 3])
```

---

### Decoding Flow

```
Input IDs: [2, 10, 20, 3]
    ↓
ID → Token Mapping
    → ["<BOS>", "merhaba</w>", "dünya</w>", "<EOS>"]
    ↓
Prefer Mode Filtering
    → prefer="word": ["merhaba</w>", "dünya</w>"]
    ↓
Special Token Removal
    → ["merhaba</w>", "dünya</w>"]
    ↓
</w> Removal + Space Insertion
    → ["merhaba", " ", "dünya"]
    ↓
Postprocessor
    → Punctuation spacing
    → Whitespace collapse
    → Sentence capitalization
    ↓
Output: "Merhaba dünya"
```

---

## 🔧 Tokenization Modülleri

### Pretokenizer

**Amaç:** Raw text → token listesi

**Adımlar:**
1. Unicode normalization (NFC, Türkçe korunur)
2. Lowercase (Türkçe: İ→i, I→ı)
3. Cleanup (non-word removal, punctuation preserved)
4. Alphanumeric separation (letters ↔ digits)
5. Punctuation separation (.,!? ayrı token)
6. Whitespace tokenization
7. Valid token filtering

---

### Syllabifier

**Amaç:** Türkçe kelimeleri hecelere ayırma

**Kurallar:**
- Her hece bir ünlü içermeli
- Onset cluster kontrolü (bl, br, pl, pr, ...)
- Coda-Onset ayrımı (2 sessiz → V-C1C2V)
- Son hece kalan harfleri alır

---

### Morphology

**Amaç:** Türkçe kelimelerden kök çıkarımı

**Özellikler:**
- TDK uyumlu kurallar
- Ünlü uyumu (a-ı-o-u vs e-i-ö-ü)
- Ünsüz benzeşmesi (p-ç-t-k vs b-c-d-g-ğ)
- Yumuşama (p→b, ç→c, t→d, k→ğ/g)
- Olumsuzluk ekleri (ma/me)
- Zaman ekleri (dı/di/du/dü)
- Kişi ekleri (im/ın/un/ün)

---

### Postprocessor

**Amaç:** Token listesi → clean text

**Adımlar:**
1. Special token mapping
2. Unicode normalization
3. Punctuation spacing fixes
4. Whitespace collapse
5. Sentence capitalization (Türkçe: İ→i)

---

## 💻 Kullanım Örnekleri

### Örnek 1: BPE Training

```python
from tokenizer_management.core.tokenizer_core import TokenizerCore
from pathlib import Path

# Config
config = {
    "vocab_path": "data/vocab_lib/vocab.json",
    "merges_path": "data/merges_lib/merges.txt",
    "data_dir": "education",
    "use_gpu": True,
}

# Initialize
tokenizer = TokenizerCore(config)

# Train from data loader
tokenizer.train_from_loader(
    method="bpe",
    vocab_size=60000,
    max_iter=50000,
    include_whole_words=True,
    include_syllables=True,
    sample_ratio=0.1,  # %10 sampling
)
```

---

### Örnek 2: Encoding

```python
# Encode text
tokens, ids = tokenizer.encode(
    "Merhaba, nasılsın?",
    mode="inference",
    include_whole_words=True,
    include_syllables=False,
)

print(f"Tokens: {tokens}")
# → ["<BOS>", "merhaba</w>", ",", "nasılsın</w>", "?", "<EOS>"]

print(f"IDs: {ids}")
# → [2, 10, 5, 20, 6, 3]
```

---

### Örnek 3: Decoding

```python
# Decode IDs
text = tokenizer.decode(
    [2, 10, 5, 20, 6, 3],
    method="bpe",
    remove_specials=True,
    prefer="word",
)

print(text)
# → "Merhaba, nasılsın?"
```

---

### Örnek 4: Batch Processing

```python
# Batch encode
texts = ["Merhaba", "Nasılsın?", "İyi misin?"]
results = tokenizer.batch_encode(
    texts,
    mode="inference",
    include_whole_words=True,
)

for tokens, ids in results:
    print(f"{tokens} → {ids}")
```

---

### Örnek 5: Training Data Preparation

```python
# Load training data
examples = tokenizer.load_training_data(
    encode_mode="inference",
    include_whole_words=True,
)

# Examples: List[Tuple[List[int], List[int]]]
# (input_ids, target_ids) for autoregressive training
for inp_ids, tgt_ids in examples[:5]:
    print(f"Input: {inp_ids}")
    print(f"Target: {tgt_ids}")
```

---

## 🌐 Public API Hazırlığı

### API Endpoints (Önerilen)

```python
# tokenizer_management/api/public_api.py

class PublicTokenizerAPI:
    """
    Public API for external use.
    """
    
    def train(self, corpus: List[str], config: Dict) -> Dict:
        """BPE training endpoint"""
        pass
    
    def encode(self, text: str, options: Dict) -> Dict:
        """Encoding endpoint"""
        pass
    
    def decode(self, token_ids: List[int], options: Dict) -> str:
        """Decoding endpoint"""
        pass
    
    def get_vocab_info(self) -> Dict:
        """Vocab information endpoint"""
        pass
```

---

### REST API (Önerilen)

```yaml
POST /api/v1/tokenizer/train
  Request:
    - corpus: List[str]
    - config: Dict
  Response:
    - vocab_size: int
    - merges_count: int
    - status: "success"

POST /api/v1/tokenizer/encode
  Request:
    - text: str
    - options: Dict
  Response:
    - tokens: List[str]
    - token_ids: List[int]

POST /api/v1/tokenizer/decode
  Request:
    - token_ids: List[int]
    - options: Dict
  Response:
    - text: str

GET /api/v1/tokenizer/vocab
  Response:
    - vocab_size: int
    - merges_count: int
    - special_tokens: Dict
```

---

## 📊 Performans ve Optimizasyonlar

### GPU Acceleration

```python
# GPU batch processing
config = {
    "use_gpu": True,
    "batch_size": 32,  # GPU batch size
}

tokenizer = TokenizerCore(config)
# GPU encoding/decoding otomatik aktif
```

---

### Memory Efficiency

```python
# Streaming corpus processing (büyük dataset için)
config = {
    "chunk_size": 2000,  # Chunk size
    "streaming_mode": True,
}

# Memory monitoring
config = {
    "memory_threshold": 8192,  # 8GB threshold
    "gc_interval": 1000,  # GC her 1000 iterasyonda
}
```

---

### Vocab Size Control

```python
# GPT benzeri yaklaşım (karakter bazlı başlangıç + merge'lerle vocab inşa)
config = {
    "max_vocab_size": 60000,
    "initial_vocab_ratio": 0.15,  # %15 başlangıç, %85 merge'lerle
    "vocab_size_buffer": 5000,  # Buffer
}
```

---

## 🔗 Entegrasyonlar

### TokenizerCore ↔ TrainingService

```python
# training_service.py
tokenizer_core = TokenizerCore(config)
training_data = tokenizer_core.load_training_data()
# → List[Tuple[List[int], List[int]]]
```

---

### TokenizerCore ↔ ModelManager

```python
# model_manager.py
vocab_size = tokenizer_core.get_vocab_size()
# Model embedding layer: nn.Embedding(vocab_size, embed_dim)
```

---

### TokenizerCore ↔ Cevahir API

```python
# cevahir.py
tokens, ids = self.tokenizer_core.encode(text, mode="inference")
# Generation pipeline
```

---

## 📖 Daha Fazla Bilgi

- **[Neural Network Dokümantasyonu](../neural_network/README.md)**
- **[Model Management Dokümantasyonu](../model_management/README.md)**
- **[Data Loader Management Dokümantasyonu](../data_loader_management/README.md)**
- **[API Referansı](../../API_REFERENCE.md)**

---

**Son Güncelleme:** 2025-01-27  
**Versiyon:** V-5  
**Durum:** Production-Ready | Public API Ready

