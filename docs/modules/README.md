# Cevahir modül rehberleri / Module guides

Güncelleme / Updated: **20 September 2026**.

Modüllerin birlikte katkısı için [sistem bütünlüğü rehberini](../architecture/SYSTEM_OVERVIEW.md), güncel davranış için [sistem sözleşmesini](../architecture/CEVAHIR_ARCHITECTURE_SPEC.md), açık işler için [geliştirme planını](../architecture/NEXT_DEVELOPMENT_ROADMAP.md) kullanın. Aşağıdaki beş modülün giriş rehberleri aktif kodla eşleştirilmiştir. İçlerindeki eski alt rehberler tarihsel ayrıntılar taşıyabilir; çelişkide güncel giriş rehberi ve gerçek çağrı yolu esas alınır.

Use the [system overview](../architecture/SYSTEM_OVERVIEW.md) for the connections between modules, the [architecture contract](../architecture/CEVAHIR_ARCHITECTURE_SPEC.md) for current behavior and the [development roadmap](../architecture/NEXT_DEVELOPMENT_ROADMAP.md) for open work. The five entry guides below have been reconciled with the active code. Older nested guides may describe earlier behavior.

| Modül / Module | Türkçe | English | Kapsam / Scope |
|---|---|---|---|
| Neural network | [Rehber](neural_network/README.md) | [Guide](neural_network/README-en.md) | Decoder, attention, FFN/MoE, V4–V8 history |
| Training system | [Rehber](training_system/README.md) | [Guide](training_system/README-en.md) | Data preparation, cache, split, batching, V3 service |
| Training management | [Rehber](training_management/README.md) | [Guide](training_management/README-en.md) | Active V2 loop, optimizer, checkpoint and resume |
| Cognitive management | [Rehber](cognitive_management/README.md) | [Guide](cognitive_management/README-en.md) | Strategy, context, tools, critic and memory |
| Tokenizer | [Rehber](tokenizer_management/README.md) | [Guide](tokenizer_management/README-en.md) | Turkish-specific BPE infrastructure, language representation and data preparation |

## Önceki modül belgeleri / Earlier module documentation

Bu rehberler bu tur baştan sona yenilenmedi. Özellikle eski sürüm etiketleri, örnek ayarlar ve entegrasyon iddiaları güncel mimariyle karşılaştırılmalıdır.

These guides were not fully refreshed in this pass; check older version labels, sample settings and integration claims against the current architecture.

| Modül / Module | Türkçe | English |
|---|---|---|
| Cevahir facade | [Rehber](cevahir/README.md) | [Guide](cevahir/README-en.md) |
| Model management | [Rehber](model_management/README.md) | [Guide](model_management/README-en.md) |
| Data loaders | [Rehber](data_loader_management/README.md) | [Guide](data_loader_management/README-en.md) |

Geçmiş eğitim çıktıları / Historical training outputs: [Türkçe](../../README-TR.md#gerçek-eğitim-çıktıları), [English](../../README.md#real-training-outputs).
