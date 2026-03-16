# Tokenizer Management - Comprehensive Documentation

**Version:** V-5
**Last Updated:** 2025-01-27
**Status:** Production-Ready | Turkish-Optimized BPE Tokenizer
**License:** Ready for Public API

---

## Table of Contents

1. [General Overview](#general-overview)
2. [Architecture](#architecture)
3. [Working Principle](#working-principle)
4. [Turkish-Specific Features](#turkish-specific-features)
5. [API Reference](#api-reference)
6. [BPE Training Pipeline](#bpe-training-pipeline)
7. [Encoding/Decoding Pipeline](#encodingdecoding-pipeline)
8. [Tokenization Modules](#tokenization-modules)
9. [Usage Examples](#usage-examples)
10. [Public API Readiness](#public-api-readiness)

---

## General Overview

**Tokenizer Management** is the Byte Pair Encoding (BPE) tokenizer system of Cevahir Neural Network, designed specifically for the Turkish language. This system has been developed to match the morphological structure of Turkish and reaches industry-standard tokenizer quality (GPT-4, Claude, Gemini).

### Core Features

- ✅ **Turkish-Optimized:** Syllabification, morphology, vowel harmony, consonant assimilation
- ✅ **BPE Training:** Deterministic, reproducible merge strategy
- ✅ **Hybrid Tokenization:** Whole words + syllables + morphology
- ✅ **GPU Support:** Accelerated batch processing
- ✅ **Production-Ready:** Atomic file operations, error handling, logging
- ✅ **Config-Driven:** Fully configurable parameters
- ✅ **Public API Ready:** Prepared for external use

---

## Architecture

### High-Level Architecture

```
TokenizerCore (Main API)
├── BPEManager (Orchestration)
│   ├── BPEEncoder (Encoding)
│   ├── BPEDecoder (Decoding)
│   ├── BPETrainer (Training)
│   └── Tokenization Pipeline
│       ├── Pretokenizer (Text pre-processing)
│       ├── Syllabifier (Turkish syllabification)
│       ├── Morphology (Root extraction)
│       └── Postprocessor (Post-processing)
│
├── DataLoaderManager (Data loading)
├── Config System (Configuration)
└── Utilities (Helper functions)
```

### Data Flow

```
Raw Text
    ↓
Pretokenizer
    ├── Unicode normalization (NFC)
    ├── Lowercase (Turkish: İ→i, I→ı)
    ├── Cleanup (non-word removal)
    └── Whitespace tokenization
    ↓
Tokenization Options
    ├── Whole Words (kelime</w>)
    ├── Syllables (he-ce-le-me)
    └── Morphology (root + suffixes)
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

## Working Principle

### 1. TokenizerCore - Main API

**File:** `tokenizer_management/core/tokenizer_core.py`
**Class:** `TokenizerCore`

**Responsibilities:**
- BPE training orchestration
- Providing Encoding/Decoding API
- DataLoaderManager integration
- Training data preparation
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

### 2. BPEManager - Orchestration Layer

**File:** `tokenizer_management/bpe/bpe_manager.py`
**Class:** `BPEManager`

**Responsibilities:**
- Encoder/Decoder/Trainer lifecycle
- Atomic read/write of vocab & merges
- Train/Inference mode management
- Turkish tokenization pipeline
- Format conversion (sentiment, instruction → QA)

**Key Methods:**

#### `train(corpus, **kwargs)`
Performs BPE training:
1. Corpus tokenization (whole words + syllables)
2. Token frequency calculation
3. Initial vocab selection (top N tokens)
4. Base alphabet guarantee
5. BPE merge training (iterative)
6. Vocab expansion
7. Atomic save (vocab.json, merges.txt)

#### `encode(text, mode, **kwargs)`
Tokenizes text and converts it to IDs:
1. Pretokenization (Unicode, cleanup)
2. Tokenization (words/syllables/morphology)
3. BPE encoding (merge application)
4. Special tokens (BOS/EOS)
5. OOV handling (syllable fallback)

#### `decode(token_ids, **kwargs)`
Converts IDs to text:
1. ID → Token mapping
2. BPE merge reversal
3. Prefer mode (word/syllable/auto)
4. Special token removal
5. Postprocessor (punctuation, spaces)

---

### 3. BPETrainer - BPE Training

**File:** `tokenizer_management/bpe/bpe_trainer.py`
**Class:** `BPETrainer`

**Algorithm:**
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

**Features:**
- ✅ Deterministic selection (freq + lexicographic)
- ✅ Special token protection
- ✅ GPU batch processing (optional)
- ✅ Memory-efficient (streaming for large corpus)
- ✅ Progress tracking (log intervals)

---

### 4. BPEEncoder - Encoding

**File:** `tokenizer_management/bpe/bpe_encoder.py`
**Class:** `BPEEncoder`

**Strategy:**
```
Token Encoding:
1. Direct vocab hit → single ID
2. Special token → single ID
3. BPE merge → sub-tokens → ID list
4. Heuristic split → parts → encode each
5. Character-level fallback → char IDs
```

**Features:**
- ✅ BPE merge application (rank-based)
- ✅ Character-level fallback (OOV handling)
- ✅ GPU acceleration (optional)
- ✅ Reverse vocab mapping (ID → token)

---

### 5. BPEDecoder - Decoding

**File:** `tokenizer_management/bpe/bpe_decoder.py`
**Class:** `BPEDecoder`

**Strategy:**
```
ID Decoding:
1. ID → Token mapping
2. Segment filtering (word/syllable prefer)
3. </w> removal + space insertion
4. Special token removal
5. Postprocessor:
   - Punctuation spacing
   - Whitespace collapse
   - Sentence capitalization (Turkish: İ→i)
```

**Prefer Modes:**
- `"word"`: Whole words (kelime</w>) are preferred
- `"syllable"`: Syllables are preferred
- `"auto"`: Automatic selection for each segment

---

## Turkish-Specific Features

### 1. Pretokenizer

**File:** `tokenizer_management/bpe/tokenization/pretokenizer.py`

**Features:**
- ✅ Unicode normalization (NFC, Turkish characters preserved)
- ✅ Turkish lowercase (İ→i, I→ı)
- ✅ Punctuation separation (.,!? as separate tokens)
- ✅ Whitespace preservation (space tokens preserved)
- ✅ Valid token filtering (Turkish characters + ASCII)

**Example:**
```python
pretokenizer = Pretokenizer()
tokens = pretokenizer.tokenize("Merhaba, nasılsın?")
# → ["Merhaba", ",", "nasılsın", "?"]
```

---

### 2. Syllabifier - Turkish Syllabification

**File:** `tokenizer_management/bpe/tokenization/syllabifier.py`

**Rules:**
1. Each syllable must contain a vowel nucleus
2. Number of consonants between two vowels:
   - 0-1 consonants: V-CV
   - 2 consonants: V-C1C2V (allowed onset cluster)
   - ≥3 consonants: V-C1C2...CV
3. The last syllable takes all remaining characters

**Allowed Onset Clusters:**
```python
["bl", "br", "pl", "pr", "fl", "fr", "kl", "kr",
 "gl", "gr", "dr", "tr", "str", "skr", "spr", "ps", "ks", "ns"]
```

**Example:**
```python
syllabifier = Syllabifier()
syllables = syllabifier.syllabify_word("merhaba")
# → ["mer", "ha", "ba"]
```

---

### 3. Morphology - Root Extraction

**File:** `tokenizer_management/bpe/tokenization/morphology.py`

**Features:**
- ✅ TDK-compatible root extraction
- ✅ Vowel harmony control
- ✅ Consonant assimilation
- ✅ Softening rules (p→b, ç→c, t→d, k→ğ/g)
- ✅ Negation suffixes (ma/me)
- ✅ Tense suffixes (dı/di/du/dü)
- ✅ Person suffixes (im/ın/un/ün)
- ✅ Compound verb support (yardım etti → yardım et)

**Example:**
```python
morphology = Morphology()
root = morphology.find_root("geldim")
# → "gel"
morphemes = morphology.analyze(["geldim"])
# → ["gel", "di", "m"]
```

---

### 4. Postprocessor

**File:** `tokenizer_management/bpe/tokenization/postprocessor.py`

**Features:**
- ✅ Punctuation spacing correction
- ✅ Whitespace collapse
- ✅ Sentence capitalization (Turkish: İ→i)
- ✅ Multiple punctuation reduction (... → .)

**Example:**
```python
postprocessor = Postprocessor()
text = postprocessor.process(["Merhaba", ",", "nasılsın", "?"])
# → "Merhaba, nasılsın?"
```

---

## API Reference

### TokenizerCore

**File:** `tokenizer_management/core/tokenizer_core.py`

#### `__init__(config: Dict[str, Any])`

**Parameters:**
- `vocab_path` / `vocab_file`: Vocab file path
- `merges_path` / `merges_file`: Merges file path
- `data_dir`: Training data directory (optional)
- `use_gpu`: GPU support (default: False)
- `batch_size`: GPU batch size (default: 32)

**Example:**
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

Performs BPE training.

**Parameters:**
- `corpus`: `List[str]` - Training corpus
- `method`: `"bpe"` (only supported method)
- `vocab_size`: `Optional[int]` - Vocab size limit
- `max_iter`: `Optional[int]` - Max iterations (default: from config)
- `min_frequency`: `Optional[int]` - Min pair frequency (default: 2)
- `include_whole_words`: `Optional[bool]` - Add whole words
- `include_syllables`: `Optional[bool]` - Add syllables
- `include_sep`: `Optional[bool]` - Add <SEP>
- `sample_ratio`: `Optional[float]` - Corpus sampling (0.0-1.0)

**Example:**
```python
corpus = ["Merhaba dünya", "Nasılsın?", ...]
tokenizer.train_model(
    corpus,
    method="bpe",
    vocab_size=60000,
    max_iter=50000,
    include_whole_words=True,
    include_syllables=True,
    sample_ratio=0.1  # 10% sampling
)
```

---

#### `train_from_loader(**kwargs)`

Loads data via DataLoaderManager and runs training.

**Example:**
```python
tokenizer.train_from_loader(
    method="bpe",
    vocab_size=60000,
    include_whole_words=True,
)
```

---

#### `encode(text, mode, **kwargs) -> Tuple[List[str], List[int]]`

Tokenizes text and converts it to IDs.

**Parameters:**
- `text`: `str` - Input text
- `mode`: `"train"` | `"inference"` - Encoding mode
- `include_whole_words`: `Optional[bool]` - Whole words
- `include_syllables`: `Optional[bool]` - Syllables
- `include_sep`: `Optional[bool]` - <SEP> token

**Returns:**
- `Tuple[List[str], List[int]]` - (tokens, token_ids)

**Example:**
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

Converts a list of IDs to text.

**Parameters:**
- `ids`: `List[int]` - Token ID list
- `method`: `"bpe"` | `"raw"` - Decode method
- `remove_specials`: `bool` - Remove special tokens (default: True)
- `remove_tags`: `bool` - Remove tags (default: True)
- `collapse_spaces`: `bool` - Collapse whitespace (default: True)
- `lowercase`: `bool` - Convert to lowercase (default: False)
- `prefer`: `"word"` | `"syllable"` | `"auto"` - Token preference

**Example:**
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

Prepares training data (QA pairs → input_ids, target_ids).

**Parameters:**
- `encode_mode`: `"train"` | `"inference"` - Encoding mode
- `include_whole_words`: `Optional[bool]`
- `include_syllables`: `Optional[bool]`
- `include_sep`: `Optional[bool]`

**Returns:**
- `List[Tuple[List[int], List[int]]]` - [(input_ids, target_ids), ...]

**Example:**
```python
examples = tokenizer.load_training_data(
    encode_mode="inference",
    include_whole_words=True,
)
```

---

### BPEManager

**File:** `tokenizer_management/bpe/bpe_manager.py`

#### `encode(text, mode, **kwargs) -> Tuple[List[str], List[int]]`

BPEManager encode method (used via TokenizerCore).

#### `decode(token_ids, **kwargs) -> str`

BPEManager decode method.

#### `train(corpus, **kwargs)`

BPE training (via BPEManager).

---

## BPE Training Pipeline

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

# Select top N tokens (initial_vocab_ratio)
initial_vocab_size = int(max_vocab_size * initial_vocab_ratio)
selected_tokens = _select_top_tokens(all_tokens, token_frequencies, initial_vocab_size)

# Base alphabet guarantee
_ensure_base_alphabet_in_vocab()  # Turkish characters + punctuation
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

## Encoding/Decoding Pipeline

### Encoding Flow

```
Input Text: "Merhaba dünya"
    ↓
Pretokenizer
    → Unicode NFC
    → Lowercase (Turkish)
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

## Tokenization Modules

### Pretokenizer

**Purpose:** Raw text → token list

**Steps:**
1. Unicode normalization (NFC, Turkish characters preserved)
2. Lowercase (Turkish: İ→i, I→ı)
3. Cleanup (non-word removal, punctuation preserved)
4. Alphanumeric separation (letters ↔ digits)
5. Punctuation separation (.,!? as separate tokens)
6. Whitespace tokenization
7. Valid token filtering

---

### Syllabifier

**Purpose:** Splitting Turkish words into syllables

**Rules:**
- Each syllable must contain a vowel
- Onset cluster control (bl, br, pl, pr, ...)
- Coda-Onset separation (2 consonants → V-C1C2V)
- Last syllable takes remaining characters

---

### Morphology

**Purpose:** Root extraction from Turkish words

**Features:**
- TDK-compatible rules
- Vowel harmony (a-ı-o-u vs e-i-ö-ü)
- Consonant assimilation (p-ç-t-k vs b-c-d-g-ğ)
- Softening (p→b, ç→c, t→d, k→ğ/g)
- Negation suffixes (ma/me)
- Tense suffixes (dı/di/du/dü)
- Person suffixes (im/ın/un/ün)

---

### Postprocessor

**Purpose:** Token list → clean text

**Steps:**
1. Special token mapping
2. Unicode normalization
3. Punctuation spacing fixes
4. Whitespace collapse
5. Sentence capitalization (Turkish: İ→i)

---

## Usage Examples

### Example 1: BPE Training

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
    sample_ratio=0.1,  # 10% sampling
)
```

---

### Example 2: Encoding

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

### Example 3: Decoding

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

### Example 4: Batch Processing

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

### Example 5: Training Data Preparation

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

## Public API Readiness

### API Endpoints (Recommended)

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

### REST API (Recommended)

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

## Performance and Optimizations

### GPU Acceleration

```python
# GPU batch processing
config = {
    "use_gpu": True,
    "batch_size": 32,  # GPU batch size
}

tokenizer = TokenizerCore(config)
# GPU encoding/decoding automatically active
```

---

### Memory Efficiency

```python
# Streaming corpus processing (for large datasets)
config = {
    "chunk_size": 2000,  # Chunk size
    "streaming_mode": True,
}

# Memory monitoring
config = {
    "memory_threshold": 8192,  # 8GB threshold
    "gc_interval": 1000,  # GC every 1000 iterations
}
```

---

### Vocab Size Control

```python
# GPT-like approach (character-based start + vocab built via merges)
config = {
    "max_vocab_size": 60000,
    "initial_vocab_ratio": 0.15,  # 15% initial, 85% via merges
    "vocab_size_buffer": 5000,  # Buffer
}
```

---

## Integrations

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

## Further Reading

- **[Neural Network Documentation](../neural_network/README.md)**
- **[Model Management Documentation](../model_management/README.md)**
- **[Data Loader Management Documentation](../data_loader_management/README.md)**
- **[API Reference](../../API_REFERENCE.md)**

---

**Last Updated:** 2025-01-27
**Version:** V-5
**Status:** Production-Ready | Public API Ready
