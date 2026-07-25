# Data Loader Management - Comprehensive Documentation

**Version:** V-5
**Last Updated:** 2025-01-27
**Status:** Production-Ready

---

## Table of Contents

1. [General Overview](#general-overview)
2. [Architecture](#architecture)
3. [Working Principle](#working-principle)
4. [Load Modes](#load-modes)
5. [API Reference](#api-reference)
6. [Smart Text Splitting](#smart-text-splitting)
7. [Format Support](#format-support)
8. [Usage Examples](#usage-examples)

---

## General Overview

**DataLoaderManager** is a SOLID-compliant module in the Cevahir Neural System that manages data loading operations for training and inference. It loads data from raw text, JSON, and DOCX files and converts different formats (QA pairs, sentiment, instruction) into the standard QA format.

### Key Features

- **SOLID Compliant:** Clean, testable design
- **Multi-format Support:** JSON, TXT, DOCX
- **Smart Text Splitting:** Intelligent text splitting (sentence-based, with overlap)
- **Token Estimation:** Token estimation optimized for Turkish
- **Format Conversion:** Sentiment, instruction → QA format
- **Fail-fast:** Error catching with strict mode
- **Hybrid Training:** QA pairs + Raw text chunks

---

## Architecture

### High-Level Architecture

```
DataLoaderManager
├── DataLoaderConfig (Configuration)
│   ├── data_dir: Path
│   ├── mode: LoadMode
│   ├── strict: bool
│   ├── question_keys: Iterable[str]
│   ├── answer_keys: Iterable[str]
│   ├── extensions: Iterable[str]
│   └── Smart splitting settings
│
├── Load Modes
│   ├── QA_TRAIN → List[Tuple[str, str]]
│   ├── TEXT_INFER → List[str]
│   └── RAW_TEXT → List[str]
│
└── Processing Pipeline
    ├── File Discovery
    ├── Format Detection
    ├── Smart Splitting
    └── Format Conversion
```

### Data Flow

```
Data Directory
    ↓
DataLoaderManager.load()
    ↓
Mode Selection (QA_TRAIN / TEXT_INFER / RAW_TEXT)
    ↓
File Discovery (_iter_files)
    ↓
Format Detection
    ├── JSON → QA pairs / Raw text
    ├── TXT → Raw text
    └── DOCX → Raw text / QA pairs
    ↓
Smart Splitting (if enabled)
    ├── Token estimation
    ├── Sentence-based splitting
    └── Overlap management
    ↓
Format Conversion (if needed)
    ├── Sentiment → QA
    ├── Instruction → QA
    └── Standard QA
    ↓
Output
    ├── QA_TRAIN → List[Tuple[str, str]]
    └── TEXT_INFER/RAW_TEXT → List[str]
```

---

## Working Principle

### 1. Initialization Flow

```
User creates DataLoaderConfig
    ├── data_dir: Path to data directory
    ├── mode: LoadMode (QA_TRAIN / TEXT_INFER / RAW_TEXT)
    ├── strict: bool (fail-fast)
    ├── question_keys: ["Soru", "question", ...]
    ├── answer_keys: ["Cevap", "answer", ...]
    └── Smart splitting settings
    ↓
DataLoaderManager(cfg)
    ├── Store config
    └── Initialize stats tracking
    ↓
Ready for load()
```

### 2. Load Flow

```
DataLoaderManager.load()
    ↓
Check data_dir exists
    ↓
Mode Selection
    ├── QA_TRAIN → _load_qa_pairs()
    ├── TEXT_INFER → _load_text_inputs()
    └── RAW_TEXT → _load_raw_text_chunks()
    ↓
File Discovery (_iter_files)
    ├── Recursive directory walk
    ├── Filter by extensions
    ├── Apply max_files limit
    └── Track stats
    ↓
File Processing
    ├── JSON → Parse JSON
    │   ├── Format detection (sentiment/instruction/QA)
    │   ├── Format conversion
    │   └── Smart splitting (if needed)
    ├── TXT → Read text
    │   └── Smart splitting (if needed)
    └── DOCX → Read DOCX
        └── Smart splitting (if needed)
    ↓
Output Collection
    └── Return List[Tuple[str, str]] or List[str]
```

### 3. Smart Splitting Flow

```
Long Text
    ↓
_estimate_token_count()
    ├── Word count
    └── Token estimation (words * 1.33 for Turkish)
    ↓
Check if splitting needed
    ├── If tokens <= max_tokens → Return as-is
    └── If tokens > max_tokens → Split
    ↓
_smart_split_text()
    ├── Sentence splitting (regex: [.!?]+)
    ├── Sentence token estimation
    ├── Chunk building (sentence-by-sentence)
    └── Overlap management
    ↓
Output: List[str] (chunks with overlap)
```

---

## Load Modes

### 1. QA_TRAIN Mode

**Purpose:** Loads question-answer pairs for training.

**Supported Formats:**
- JSON files (`.json`)

**Expected JSON Structure:**
```json
[
    {
        "Soru": "Türkiye'nin başkenti neresidir?",
        "Cevap": "Türkiye'nin başkenti Ankara'dır."
    },
    {
        "question": "What is the capital of Turkey?",
        "answer": "The capital of Turkey is Ankara."
    }
]
```

**Output:**
- `List[Tuple[str, str]]`: `[(question, answer), ...]`

**Features:**
- Multiple key support ("Soru"/"question"/"instruction"/"prompt")
- Format conversion (sentiment, instruction → QA)
- Smart splitting (splits long answers)

---

### 2. TEXT_INFER Mode

**Purpose:** Loads raw text for inference.

**Supported Formats:**
- TXT files (`.txt`)
- DOCX files (`.docx`)
- JSON files (skipped - already processed in QA_TRAIN)

**Output:**
- `List[str]`: Raw text chunks

**Features:**
- Smart splitting (splits long texts)
- DOCX support (paragraph extraction)
- Token-aware chunking

---

### 3. RAW_TEXT Mode

**Purpose:** Loads raw text chunks (for hybrid training).

**Supported Formats:**
- TXT files (`.txt`)
- DOCX files (`.docx`)

**Output:**
- `List[str]`: Raw text chunks

**Features:**
- Smart splitting
- DOCX support
- Overlap management

---

## Smart Text Splitting

### Token Estimation

**Formula:**
```python
estimated_tokens = word_count * 1.33  # Optimized for Turkish
```

**Features:**
- Optimized for Turkish (1 token ≈ 0.75 words)
- Fast computation (word count based)
- Sentence-level and word-level estimation

---

### Sentence-based Splitting

**Algorithm:**
1. **Sentence Detection:** Split sentences using regex `[.!?]+`
2. **Token Estimation:** Estimate token count for each sentence
3. **Chunk Building:**
   - Accumulate sentence tokens
   - Start a new chunk when max tokens is reached
   - Add overlap (copy last N words to the new chunk)

**Features:**
- Splits at sentence boundaries (context is preserved)
- Overlap management (20 token default)
- No data loss (all text is preserved)

---

### Word-level Splitting

**Usage:** When even a single sentence exceeds max_tokens

**Algorithm:**
1. Split into words
2. Accumulate word tokens
3. Start a new chunk when max tokens is reached
4. Add overlap (copy last N words to the new chunk)

---

## Format Support

### 1. Standard QA Format

**JSON Structure:**
```json
[
    {
        "Soru": "Question text",
        "Cevap": "Answer text"
    }
]
```

**Key Options:**
- Question keys: `["Soru", "question", "instruction", "prompt"]`
- Answer keys: `["Cevap", "answer", "output", "response"]`

---

### 2. Sentiment Format

**JSON Structure:**
```json
[
    {
        "text": "Bu ürün harika!",
        "sentiment": "positive"
    }
]
```

**Conversion:**
- Question: `"Bu metnin duygu analizi nasıldır?"`
- Answer: `"Metin: '{text}' - Duygu: {sentiment}"`

---

### 3. Instruction Format

**JSON Structure:**
```json
[
    {
        "instruction": "Türkçeye çevir",
        "input": "Hello world",
        "output": "Merhaba dünya"
    }
]
```

**Conversion:**
- Question: `"{instruction}\n\nGirdi: {input}"` (if input exists)
- Answer: `"{output}"`

---

### 4. TXT Format

**Structure:** Plain text file

**Processing:**
- The entire file content is read as a single string
- Split into chunks via smart splitting

---

### 5. DOCX Format

**Structure:** Microsoft Word document

**Processing:**
- Paragraphs are read
- Empty paragraphs are skipped
- Paragraphs are joined together
- Split into chunks via smart splitting

**Dependencies:**
- `python-docx` (optional — raises error if absent)

---

## API Reference

### DataLoaderConfig

**File:** `data_loader_management/data_loader_manager.py`
**Class:** `DataLoaderConfig`

#### Parameters

```python
@dataclass(frozen=True)
class DataLoaderConfig:
    data_dir: Path                              # Data directory
    mode: str = LoadMode.QA_TRAIN               # Load mode
    strict: bool = False                        # Fail-fast mode
    question_keys: Iterable[str] = ("Soru", "question", ...)  # Question key options
    answer_keys: Iterable[str] = ("Cevap", "answer", ...)     # Answer key options
    qa_extensions: Iterable[str] = (".json",)   # QA mode extensions
    infer_extensions: Iterable[str] = (".txt", ".json")  # Inference mode extensions
    raw_text_extensions: Iterable[str] = (".txt", ".docx")  # Raw text extensions
    follow_symlinks: bool = False               # Follow symlinks
    max_files: Optional[int] = None             # Max files limit
    max_tokens_per_text: int = 192              # Max tokens per chunk
    enable_smart_splitting: bool = True         # Enable smart splitting
    overlap_tokens: int = 20                    # Overlap tokens
```

---

### DataLoaderManager

**File:** `data_loader_management/data_loader_manager.py`
**Class:** `DataLoaderManager`

#### `__init__(cfg: DataLoaderConfig)`

Initializes DataLoaderManager.

**Parameters:**
- `cfg` (DataLoaderConfig): Configuration

**Example:**
```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
    strict=False,
    enable_smart_splitting=True,
    max_tokens_per_text=192,
)
manager = DataLoaderManager(config)
```

---

#### `load() -> List[Tuple[str, str]] | List[str]`

Loads data (according to mode).

**Returns:**
- `QA_TRAIN`: `List[Tuple[str, str]]` - `[(question, answer), ...]`
- `TEXT_INFER`: `List[str]` - `[text1, text2, ...]`
- `RAW_TEXT`: `List[str]` - `[chunk1, chunk2, ...]`

**Example:**
```python
# QA mode
qa_data = manager.load()  # List[Tuple[str, str]]

# Text inference mode
text_data = manager.load()  # List[str]
```

---

#### `load_data() -> List[Dict[str, Any]]`

**Backward Compatibility API**

Converted output for legacy format.

**Returns:**
- `QA_TRAIN`: `List[Dict]` - `[{"modality": "text", "data": question, "target": answer}, ...]`
- `TEXT_INFER`: `List[Dict]` - `[{"modality": "text", "data": text}, ...]`

---

#### `stats: Dict[str, int]` (Property)

Returns statistics.

**Statistics:**
- `files_seen`: Number of files seen
- `files_loaded`: Number of files loaded
- `examples`: Total number of examples

**Example:**
```python
data = manager.load()
stats = manager.stats
print(f"Files loaded: {stats['files_loaded']}")
print(f"Total examples: {stats['examples']}")
```

---

### LoadMode

**Enum:** Load modes

```python
class LoadMode:
    QA_TRAIN = "QA_TRAIN"      # QA pairs for training
    TEXT_INFER = "TEXT_INFER"  # Raw text for inference
    RAW_TEXT = "RAW_TEXT"      # Raw text chunks
```

---

### Exception Classes

```python
class DataLoaderError(Exception): ...                    # Base exception
class DataDirectoryNotFoundError(DataLoaderError): ...    # Directory not found
class UnsupportedFormatError(DataLoaderError): ...        # Unsupported format
class InvalidStructureError(DataLoaderError): ...         # Invalid JSON structure
class EmptyTextError(DataLoaderError): ...                # Empty text content
```

---

## Usage Examples

### Example 1: QA Training Data Loading

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Config
config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
    strict=False,
    enable_smart_splitting=True,
    max_tokens_per_text=192,
    overlap_tokens=20,
)

# Load
manager = DataLoaderManager(config)
qa_pairs = manager.load()  # List[Tuple[str, str]]

# Stats
stats = manager.stats
print(f"Loaded: {stats['files_loaded']} files, {stats['examples']} pairs")

# Use
for question, answer in qa_pairs[:5]:
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

---

### Example 2: Text Inference Data Loading

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Config
config = DataLoaderConfig(
    data_dir=Path("inference_data"),
    mode=LoadMode.TEXT_INFER,
    strict=False,
    enable_smart_splitting=True,
    max_tokens_per_text=192,
)

# Load
manager = DataLoaderManager(config)
text_chunks = manager.load()  # List[str]

# Use
for chunk in text_chunks:
    print(f"Chunk: {chunk[:100]}...")
```

---

### Example 3: Raw Text Chunks (Hybrid Training)

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Config
config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.RAW_TEXT,
    strict=False,
    enable_smart_splitting=True,
    max_tokens_per_text=192,
    overlap_tokens=20,
)

# Load
manager = DataLoaderManager(config)
raw_chunks = manager.load()  # List[str]

# Use for hybrid training
# QA pairs + Raw chunks
```

---

### Example 4: Hybrid Training (QA + Raw Text)

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# 1. Load QA pairs
qa_config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
)
qa_manager = DataLoaderManager(qa_config)
qa_pairs = qa_manager.load()  # List[Tuple[str, str]]

# 2. Load raw text chunks
raw_config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.TEXT_INFER,
)
raw_manager = DataLoaderManager(raw_config)
raw_chunks = raw_manager.load()  # List[str]

# 3. Build hybrid corpus
corpus = []
for q, a in qa_pairs:
    corpus.extend([q, a])  # Add QA pairs to corpus
corpus.extend(raw_chunks)  # Add raw chunks

print(f"Hybrid corpus: {len(qa_pairs)} QA pairs + {len(raw_chunks)} raw chunks = {len(corpus)} total")
```

---

### Example 5: Custom Keys

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Custom question/answer keys
config = DataLoaderConfig(
    data_dir=Path("custom_data"),
    mode=LoadMode.QA_TRAIN,
    question_keys=("instruction", "prompt", "input"),
    answer_keys=("output", "response", "completion"),
)

manager = DataLoaderManager(config)
qa_pairs = manager.load()
```

---

### Example 6: Strict Mode

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Strict mode: Fail-fast
config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
    strict=True,  # Raises exception on any error
)

manager = DataLoaderManager(config)
try:
    qa_pairs = manager.load()
except Exception as e:
    print(f"Error: {e}")
```

---

### Example 7: Max Files Limit

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Load only the first 100 files
config = DataLoaderConfig(
    data_dir=Path("large_dataset"),
    mode=LoadMode.QA_TRAIN,
    max_files=100,  # Load first 100 files
)

manager = DataLoaderManager(config)
qa_pairs = manager.load()
```

---

### Example 8: Disable Smart Splitting

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Disable smart splitting
config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
    enable_smart_splitting=False,  # No splitting
)

manager = DataLoaderManager(config)
qa_pairs = manager.load()
```

---

## Smart Splitting Details

### Token Estimation Algorithm

```python
def _estimate_token_count(text: str) -> int:
    """
    Token estimation for Turkish:
    - 1 token ≈ 0.75 words
    - word_count * 1.33 ≈ token_count
    """
    words = len(text.split())
    estimated_tokens = int(words * 1.33)
    return estimated_tokens
```

### Sentence-based Splitting Algorithm

```python
def _smart_split_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    1. Split sentences (regex: [.!?]+)
    2. Token estimation for each sentence
    3. Build chunks (sentence-by-sentence)
    4. Add overlap (last N words)
    """
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk)
            # Overlap: Copy last N words to new chunk
            overlap_text = get_last_n_words(current_chunk, overlap)
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += ". " + sentence
            current_tokens += sentence_tokens

    return chunks
```

### QA Pair Splitting

```python
def _process_qa_pair_with_splitting(question: str, answer: str) -> List[Tuple[str, str]]:
    """
    Splits long answers:
    - First part: Original question + first chunk
    - Subsequent parts: "(continued N)" questions + chunks
    """
    if answer_tokens <= max_tokens:
        return [(question, answer)]

    answer_chunks = smart_split_text(answer, max_tokens, overlap)
    qa_pairs = []

    for i, chunk in enumerate(answer_chunks):
        if i == 0:
            qa_pairs.append((question, chunk))
        else:
            continuation_q = f"{question} (devam {i+1})"
            qa_pairs.append((continuation_q, chunk))

    return qa_pairs
```

---

## Format Conversion Examples

### Sentiment Format

**Input:**
```json
{
    "text": "Bu ürün harika!",
    "sentiment": "positive"
}
```

**Output:**
```python
(
    "Bu metnin duygu analizi nasıldır?",
    "Metin: 'Bu ürün harika!' - Duygu: positive"
)
```

---

### Instruction Format

**Input:**
```json
{
    "instruction": "Türkçeye çevir",
    "input": "Hello world",
    "output": "Merhaba dünya"
}
```

**Output:**
```python
(
    "Türkçeye çevir\n\nGirdi: Hello world",
    "Merhaba dünya"
)
```

---

## Integrations

### DataLoaderManager ↔ TrainingService

**Relationship:** Used-By

```python
TrainingService
    │
    └── USES DataLoaderManager
            │
            ├── QA_TRAIN mode → QA pairs
            └── TEXT_INFER mode → Raw text chunks
```

**Usage:**
```python
# training_service.py
qa_loader = DataLoaderManager(DataLoaderConfig(
    data_dir=Path(self.config["data_dir"]),
    mode=LoadMode.QA_TRAIN
))
qa_data = qa_loader.load()

raw_loader = DataLoaderManager(DataLoaderConfig(
    data_dir=Path(self.config["data_dir"]),
    mode=LoadMode.TEXT_INFER
))
raw_data = raw_loader.load()
```

---

### DataLoaderManager ↔ TokenizerCore

**Relationship:** Used-By

```python
TokenizerCore
    │
    └── USES DataLoaderManager (optional)
            │
            └── Auto-initialized if data_dir is provided
```

**Usage:**
```python
# tokenizer_core.py
if data_dir:
    dl_cfg = DataLoaderConfig(
        data_dir=Path(data_dir),
        mode=LoadMode.QA_TRAIN,
        strict=False,
    )
    self.data_loader = DataLoaderManager(dl_cfg)
```

---

## Performance and Best Practices

### 1. Smart Splitting Settings

```python
# Small chunks (more examples)
config = DataLoaderConfig(
    max_tokens_per_text=128,  # Small chunks
    overlap_tokens=10,        # Less overlap
)

# Large chunks (fewer examples)
config = DataLoaderConfig(
    max_tokens_per_text=512,  # Large chunks
    overlap_tokens=50,        # More overlap
)
```

---

### 2. Max Files Limit

```python
# Small dataset for testing
config = DataLoaderConfig(
    max_files=10,  # First 10 files
)

# Full dataset
config = DataLoaderConfig(
    max_files=None,  # Unlimited
)
```

---

### 3. Strict Mode

```python
# Development: Lenient
config = DataLoaderConfig(strict=False)  # Warnings, continue

# Production: Strict
config = DataLoaderConfig(strict=True)  # Errors, fail-fast
```

---

### 4. Format Support

```python
# Custom keys for different formats
config = DataLoaderConfig(
    question_keys=("instruction", "prompt", "input"),
    answer_keys=("output", "response", "completion"),
)
```

---

## Error Handling

### Exception Handling

```python
from data_loader_management import (
    DataLoaderError,
    DataDirectoryNotFoundError,
    UnsupportedFormatError,
    InvalidStructureError,
    EmptyTextError,
)

try:
    data = manager.load()
except DataDirectoryNotFoundError as e:
    print(f"Data directory not found: {e}")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
except InvalidStructureError as e:
    print(f"Invalid structure: {e}")
except EmptyTextError as e:
    print(f"Empty text: {e}")
```

---

## Advanced Features

### 1. DOCX Processing

**Requirements:**
- `python-docx` package

**Usage:**
```python
# DOCX files are automatically detected and processed
config = DataLoaderConfig(
    data_dir=Path("documents"),
    mode=LoadMode.RAW_TEXT,
    raw_text_extensions=(".txt", ".docx"),  # DOCX included
)
```

---

### 2. Custom Extensions

```python
# Custom file extensions
config = DataLoaderConfig(
    qa_extensions=(".json", ".jsonl"),  # JSONL support
    infer_extensions=(".txt", ".md", ".docx"),  # Markdown support
)
```

---

### 3. Follow Symlinks

```python
# Follow symbolic links
config = DataLoaderConfig(
    follow_symlinks=True,  # Follow symlinks in directory walk
)
```

---

## Further Reading

- **[Training System Documentation](../training_system/README.md)**
- **[Tokenizer Management Documentation](../tokenizer_management/README.md)**
- **[API Reference](../../API_REFERENCE.md)**

---

**Last Updated:** 2025-01-27
**Version:** V-5
