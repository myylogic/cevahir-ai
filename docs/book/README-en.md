# Cevahir AI — Anatomy of an AI Engine

[Turkish table of contents](README.md) · [Repository](../../README.md) · [Existing English module guides](../modules/README.md)

Cevahir is being developed as an open-source research and educational resource. This book follows the real engine from text representation and neural computation to training, inference, application orchestration and ongoing living-learning research. It complements the existing engineering documentation; it does not replace the implementation with a tutorial project.

The full chapters are currently in Turkish. This page is an English navigation guide, **not a claim that the complete book has been translated**. Existing English documentation remains available. Source paths and Python symbols are shared across languages.

| Chapters in Turkish | Existing reference material |
|---|---|
| [1. Engine and learning](tr/01-motor-ve-ogrenme.md), [2. Text and tokenization](tr/02-metin-tokenizer.md) | [Tokenizer guide](../modules/tokenizer_management/README-en.md) |
| [3. Neural computation](tr/03-sinir-aglari.md), [4. Attention](tr/04-attention.md), [5. Transformer](tr/05-transformer.md) | [Neural core guide](../modules/neural_network/README-en.md) |
| [6. Training](tr/06-egitim.md), [7. Model lifecycle](tr/07-model-yasam-dongusu.md) | [Training service](../modules/training_system/README-en.md), [active training manager](../modules/training_management/README-en.md) |
| [8. Generation](tr/08-uretim.md), [9. Memory, cognition and tools](tr/09-bellek-bilis-araclar.md), [10. End-to-end paths](tr/10-uctan-uca-sistem.md) | [Cognitive guide](../modules/cognitive_management/README-en.md); older [model-management guide](../modules/model_management/README-en.md) needs the compatibility caveats in chapter 7. |
| [11. Research laboratory](tr/11-arastirma-laboratuvari.md), [12. Open questions](tr/12-acik-sorular.md) | [Preserved research records](../research), [executable isolated experiments](../../research) |

Each chapter connects concepts to actual callers, classes, methods, input/output shapes, configuration gates and tests. Supported, enabled by default, tested and empirically beneficial are different claims. Performance conclusions are restricted to their recorded conditions; no trained-model quality is inferred from a small tensor test.

Chapter 2 now includes a topic-to-file/method map and six executed examples using Cevahir's distributed vocabulary and merges. It follows intermediate pieces, model IDs, decoding, mode-specific defaults and preprocessing loss through the actual implementation. The [recorded outputs](evidence/tokenizer_walkthrough.json) can be reproduced with `python scripts/book_tokenizer_walkthrough.py`; the command also checks that the tokenizer assets remain unchanged.

The book and its source references are reviewed together. Run `python scripts/check_book.py` from the repository root to check local links, Python symbols and reviewed-source fingerprints. A passing result cannot establish semantic correctness. When a source changes, review its chapter before renewing the fingerprint. See the [maintenance rules](BAKIM.md) and [source map](KAYNAK_HARITASI.md).
