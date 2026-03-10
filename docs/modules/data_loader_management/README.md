# 📂 Data Loader Management - Kapsamlı Dokümantasyon

**Versiyon:** V-5  
**Son Güncelleme:** 2025-01-27  
**Durum:** Production-Ready

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Çalışma Prensibi](#çalışma-prensibi)
4. [Load Modes](#load-modes)
5. [API Referansı](#api-referansı)
6. [Smart Text Splitting](#smart-text-splitting)
7. [Format Desteği](#format-desteği)
8. [Kullanım Örnekleri](#kullanım-örnekleri)

---

## 🎯 Genel Bakış

**DataLoaderManager**, Cevahir Sinir Sistemi'nde eğitim ve inference için veri yükleme işlemlerini yöneten, SOLID prensiplere uygun bir modüldür. Raw text, JSON, DOCX dosyalarından veri yükler ve farklı formatları (QA pairs, sentiment, instruction) standart QA formatına dönüştürür.

### Temel Özellikler

- ✅ **SOLID Uyumlu:** Yalın, test edilebilir tasarım
- ✅ **Multi-format Support:** JSON, TXT, DOCX
- ✅ **Smart Text Splitting:** Akıllı metin bölme (cümle bazlı, overlap)
- ✅ **Token Estimation:** Türkçe için optimize token tahmini
- ✅ **Format Conversion:** Sentiment, instruction → QA format
- ✅ **Fail-fast:** Strict mode ile hata yakalama
- ✅ **Hybrid Training:** QA pairs + Raw text chunks

---

## 🏗️ Mimari Yapı

### Yüksek Seviye Mimari

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

### Veri Akışı

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

## ⚙️ Çalışma Prensibi

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

## 📚 Load Modes

### 1. QA_TRAIN Mode

**Amaç:** Eğitim için soru-cevap çiftleri yükler.

**Desteklenen Formatlar:**
- JSON files (`.json`)

**Beklenen JSON Yapısı:**
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

**Çıktı:**
- `List[Tuple[str, str]]`: `[(question, answer), ...]`

**Özellikler:**
- ✅ Multiple key support ("Soru"/"question"/"instruction"/"prompt")
- ✅ Format conversion (sentiment, instruction → QA)
- ✅ Smart splitting (uzun cevapları böler)

---

### 2. TEXT_INFER Mode

**Amaç:** Inference için raw text yükler.

**Desteklenen Formatlar:**
- TXT files (`.txt`)
- DOCX files (`.docx`)
- JSON files (skipped - already processed in QA_TRAIN)

**Çıktı:**
- `List[str]`: Raw text chunks

**Özellikler:**
- ✅ Smart splitting (uzun metinleri böler)
- ✅ DOCX support (paragraph extraction)
- ✅ Token-aware chunking

---

### 3. RAW_TEXT Mode

**Amaç:** Raw text chunks yükler (hibrit eğitim için).

**Desteklenen Formatlar:**
- TXT files (`.txt`)
- DOCX files (`.docx`)

**Çıktı:**
- `List[str]`: Raw text chunks

**Özellikler:**
- ✅ Smart splitting
- ✅ DOCX support
- ✅ Overlap management

---

## 🔧 Smart Text Splitting

### Token Estimation

**Formül:**
```python
estimated_tokens = word_count * 1.33  # Türkçe için optimize
```

**Özellikler:**
- ✅ Türkçe için optimize (1 token ≈ 0.75 kelime)
- ✅ Hızlı hesaplama (word count bazlı)
- ✅ Sentence-level ve word-level estimation

---

### Sentence-based Splitting

**Algoritma:**
1. **Sentence Detection:** Regex `[.!?]+` ile cümleleri ayır
2. **Token Estimation:** Her cümle için token sayısı tahmin et
3. **Chunk Building:**
   - Cümle tokenları topla
   - Max tokens'a ulaşınca yeni chunk başlat
   - Overlap ekle (son N kelimeyi yeni chunk'a kopyala)

**Özellikler:**
- ✅ Cümle sınırlarında böler (bağlam korunur)
- ✅ Overlap management (20 token default)
- ✅ Veri kaybı yok (tüm metin korunur)

---

### Word-level Splitting

**Kullanım:** Tek cümle bile max_tokens'ı aşıyorsa

**Algoritma:**
1. Kelimeleri ayır
2. Kelime tokenları topla
3. Max tokens'a ulaşınca yeni chunk başlat
4. Overlap ekle (son N kelimeyi yeni chunk'a kopyala)

---

## 📄 Format Desteği

### 1. Standard QA Format

**JSON Yapısı:**
```json
[
    {
        "Soru": "Soru metni",
        "Cevap": "Cevap metni"
    }
]
```

**Key Options:**
- Question keys: `["Soru", "question", "instruction", "prompt"]`
- Answer keys: `["Cevap", "answer", "output", "response"]`

---

### 2. Sentiment Format

**JSON Yapısı:**
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

**JSON Yapısı:**
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

**Yapı:** Plain text file

**Processing:**
- Tüm dosya içeriği tek bir string olarak okunur
- Smart splitting ile chunks'a bölünür

---

### 5. DOCX Format

**Yapı:** Microsoft Word document

**Processing:**
- Paragraflar okunur
- Boş paragraflar atlanır
- Paragraflar birleştirilir
- Smart splitting ile chunks'a bölünür

**Dependencies:**
- `python-docx` (opsiyonel - yoksa error)

---

## 📚 API Referansı

### DataLoaderConfig

**Dosya:** `data_loader_management/data_loader_manager.py`  
**Sınıf:** `DataLoaderConfig`

#### Parametreler

```python
@dataclass(frozen=True)
class DataLoaderConfig:
    data_dir: Path                              # Veri dizini
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

**Dosya:** `data_loader_management/data_loader_manager.py`  
**Sınıf:** `DataLoaderManager`

#### `__init__(cfg: DataLoaderConfig)`

DataLoaderManager'ı başlatır.

**Parametreler:**
- `cfg` (DataLoaderConfig): Konfigürasyon

**Örnek:**
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

Verileri yükler (mode'a göre).

**Dönüş:**
- `QA_TRAIN`: `List[Tuple[str, str]]` - `[(question, answer), ...]`
- `TEXT_INFER`: `List[str]` - `[text1, text2, ...]`
- `RAW_TEXT`: `List[str]` - `[chunk1, chunk2, ...]`

**Örnek:**
```python
# QA mode
qa_data = manager.load()  # List[Tuple[str, str]]

# Text inference mode
text_data = manager.load()  # List[str]
```

---

#### `load_data() -> List[Dict[str, Any]]`

**Geriye Dönük Uyum API**

Legacy format için dönüştürülmüş çıktı.

**Dönüş:**
- `QA_TRAIN`: `List[Dict]` - `[{"modality": "text", "data": question, "target": answer}, ...]`
- `TEXT_INFER`: `List[Dict]` - `[{"modality": "text", "data": text}, ...]`

---

#### `stats: Dict[str, int]` (Property)

İstatistikleri döndürür.

**İstatistikler:**
- `files_seen`: Görülen dosya sayısı
- `files_loaded`: Yüklenen dosya sayısı
- `examples`: Toplam örnek sayısı

**Örnek:**
```python
data = manager.load()
stats = manager.stats
print(f"Yüklenen dosyalar: {stats['files_loaded']}")
print(f"Toplam örnek: {stats['examples']}")
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

## 💻 Kullanım Örnekleri

### Örnek 1: QA Training Data Loading

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
print(f"Yüklenen: {stats['files_loaded']} dosya, {stats['examples']} çift")

# Use
for question, answer in qa_pairs[:5]:
    print(f"S: {question}")
    print(f"C: {answer}\n")
```

---

### Örnek 2: Text Inference Data Loading

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

### Örnek 3: Raw Text Chunks (Hybrid Training)

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

### Örnek 4: Hybrid Training (QA + Raw Text)

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# 1. QA pairs yükle
qa_config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
)
qa_manager = DataLoaderManager(qa_config)
qa_pairs = qa_manager.load()  # List[Tuple[str, str]]

# 2. Raw text chunks yükle
raw_config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.TEXT_INFER,
)
raw_manager = DataLoaderManager(raw_config)
raw_chunks = raw_manager.load()  # List[str]

# 3. Hybrid corpus oluştur
corpus = []
for q, a in qa_pairs:
    corpus.extend([q, a])  # QA pairs'i corpus'a ekle
corpus.extend(raw_chunks)  # Raw chunks'ları ekle

print(f"Hybrid corpus: {len(qa_pairs)} QA çifti + {len(raw_chunks)} raw chunk = {len(corpus)} toplam")
```

---

### Örnek 5: Custom Keys

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

### Örnek 6: Strict Mode

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Strict mode: Fail-fast
config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
    strict=True,  # Herhangi bir hata durumunda exception fırlatır
)

manager = DataLoaderManager(config)
try:
    qa_pairs = manager.load()
except Exception as e:
    print(f"Hata: {e}")
```

---

### Örnek 7: Max Files Limit

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Sadece ilk 100 dosyayı yükle
config = DataLoaderConfig(
    data_dir=Path("large_dataset"),
    mode=LoadMode.QA_TRAIN,
    max_files=100,  # İlk 100 dosyayı yükle
)

manager = DataLoaderManager(config)
qa_pairs = manager.load()
```

---

### Örnek 8: Disable Smart Splitting

```python
from pathlib import Path
from data_loader_management import DataLoaderManager, DataLoaderConfig, LoadMode

# Smart splitting'i devre dışı bırak
config = DataLoaderConfig(
    data_dir=Path("education"),
    mode=LoadMode.QA_TRAIN,
    enable_smart_splitting=False,  # Bölme yapma
)

manager = DataLoaderManager(config)
qa_pairs = manager.load()
```

---

## 🔍 Smart Splitting Detayları

### Token Estimation Algorithm

```python
def _estimate_token_count(text: str) -> int:
    """
    Türkçe için token tahmini:
    - 1 token ≈ 0.75 kelime
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
    1. Cümleleri ayır (regex: [.!?]+)
    2. Her cümle için token tahmini
    3. Chunk'lar oluştur (cümle bazlı)
    4. Overlap ekle (son N kelime)
    """
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk)
            # Overlap: Son N kelimeyi yeni chunk'a kopyala
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
    Uzun cevapları böler:
    - İlk parça: Orijinal soru + ilk chunk
    - Sonraki parçalar: "(devam N)" soruları + chunks
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

## 📊 Format Conversion Examples

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

## 🔗 Entegrasyonlar

### DataLoaderManager ↔ TrainingService

**İlişki:** Used-By

```python
TrainingService
    │
    └── USES DataLoaderManager
            │
            ├── QA_TRAIN mode → QA pairs
            └── TEXT_INFER mode → Raw text chunks
```

**Kullanım:**
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

**İlişki:** Used-By

```python
TokenizerCore
    │
    └── USES DataLoaderManager (opsiyonel)
            │
            └── data_dir verilirse otomatik başlatılır
```

**Kullanım:**
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

## 📈 Performans ve Best Practices

### 1. Smart Splitting Ayarları

```python
# Küçük chunks (daha fazla örnek)
config = DataLoaderConfig(
    max_tokens_per_text=128,  # Küçük chunks
    overlap_tokens=10,        # Az overlap
)

# Büyük chunks (daha az örnek)
config = DataLoaderConfig(
    max_tokens_per_text=512,  # Büyük chunks
    overlap_tokens=50,        # Fazla overlap
)
```

---

### 2. Max Files Limit

```python
# Test için küçük dataset
config = DataLoaderConfig(
    max_files=10,  # İlk 10 dosya
)

# Full dataset
config = DataLoaderConfig(
    max_files=None,  # Sınırsız
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

## 🐛 Hata Yönetimi

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
    print(f"Veri klasörü bulunamadı: {e}")
except UnsupportedFormatError as e:
    print(f"Desteklenmeyen format: {e}")
except InvalidStructureError as e:
    print(f"Geçersiz yapı: {e}")
except EmptyTextError as e:
    print(f"Boş metin: {e}")
```

---

## 🔧 Advanced Features

### 1. DOCX Processing

**Requirements:**
- `python-docx` paketi

**Kullanım:**
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

## 📖 Daha Fazla Bilgi

- **[Training System Dokümantasyonu](../training_system/README.md)**
- **[Tokenizer Management Dokümantasyonu](../tokenizer_management/README.md)**
- **[API Referansı](../../API_REFERENCE.md)**

---

**Son Güncelleme:** 2025-01-27  
**Versiyon:** V-5

