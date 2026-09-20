# Kavramdan çalışan koda

[İçindekiler](README.md) · [Bakım](BAKIM.md) · [English guide](README-en.md)

Bu harita dosya envanteri değildir. Her satır kitabın bir kavramından gerçek çağrı sınırına ve onun kontrolüne gider. Kaynak incelemesi 20 Eylül 2026'da yapıldı. Satır numaraları birer kolaylıktır; `evidence/` içindeki nitelikli Python sembolleri ve gözden geçirilmiş dosya parmak izleri birlikte denetlenir.

| Konu | Gerçek giriş ve bağlantı | Kitap / mevcut kaliteli ayrıntı | Kontrol |
|---|---|---|---|
| Metin temsili | [`TokenizerCore.encode`](../../tokenizer_management/core/tokenizer_core.py) → `BPEManager` → encoder | [2](tr/02-metin-tokenizer.md), [TR modül rehberi](../modules/tokenizer_management/README.md), [EN](../modules/tokenizer_management/README-en.md) | [Tokenizer sözleşmeleri](../../tests/evolution/test_tokenizer_contracts.py) |
| Tensor hesabı | [`CevahirNeuralNetwork.forward`](../../src/neural_network.py) → embedding/katmanlar/çıkış | [3](tr/03-sinir-aglari.md), [5](tr/05-transformer.md), [TR neural rehberi](../modules/neural_network/README.md), [EN](../modules/neural_network/README-en.md) | [Alt katmanlar](../../tests/evolution/test_lower_layer_contracts.py) |
| Attention ve KV | [`MultiHeadAttention.forward`](../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py) → gerçek cache/mask yolları | [4](tr/04-attention.md), [alt çekirdek sözleşmeleri](../architecture/LOWER_CORE_CONTRACTS.md) | [Cache eşdeğerliği](../../tests/evolution/test_attention_cache.py) |
| Hazırlanmış veri | [`prepare_cache`](../../training_system/prepare_cache.py) → kimlik → [`split_training_records`](../../training_system/data_split.py) | [6](tr/06-egitim.md), [training_system TR](../modules/training_system/README.md), [EN](../modules/training_system/README-en.md) | [Veri/cache sözleşmeleri](../../tests/evolution/test_data_cache_contracts.py) |
| Parametre güncelleme | [`TrainingServiceV3`](../../training_system/v3/core/training_service_v3.py) → V2 manager → [`TrainingLoop`](../../training_management/v2/core/training_loop.py) | [6](tr/06-egitim.md), [training_management TR](../modules/training_management/README.md), [EN](../modules/training_management/README-en.md) | [Eğitim sözleşmeleri](../../tests/evolution/test_training_contracts.py) |
| Kimlik ve devam | [`ModelManager.save/load`](../../model_management/model_manager.py), [`checkpoint_contract`](../../model_management/checkpoint_contract.py), V2 resume | [7](tr/07-model-yasam-dongusu.md), [konsolidasyon kaydı](../architecture/LIFECYCLE_CONSOLIDATION.md) | [Checkpoint yaşam döngüsü](../../tests/evolution/test_checkpoint_lifecycle.py) |
| Token üretimi | [`CevahirModelAPI._autoregressive_generate`](../../model/cevahir.py) → manager → core → token seçimi | [8](tr/08-uretim.md), [sistem bütünlüğü](../architecture/SYSTEM_OVERVIEW.md) | [Runtime yaşam döngüsü](../../tests/evolution/test_runtime_lifecycle.py) |
| Cognition ve bellek | [`CognitiveManager`](../../cognitive_management/cognitive_manager.py), strategy/context/tool sınırları | [9](tr/09-bellek-bilis-araclar.md), [TR cognitive rehberi](../modules/cognitive_management/README.md), [EN](../modules/cognitive_management/README-en.md) | Bölümdeki kapsam/araç testleri |
| Mesaj ve servis | [`ChattingManager`](../../chatting_management/chatting_manager.py) → `Cevahir.process` → cognition | [10](tr/10-uctan-uca-sistem.md), [mimari sözleşme](../architecture/CEVAHIR_ARCHITECTURE_SPEC.md) | Bölümdeki servis/konuşma sınırları |
| Deneyim politikası | [`ResearchController`](../../cognitive_management/research/controller.py) → `ExperienceStore` | [11](tr/11-arastirma-laboratuvari.md), [araştırma sözleşmesi](../research/EXPERIENCE_CONDITIONED_COMPUTE.md) | [Replay karşılaştırması](../../benchmarks/research/experience_replay.py) |
| Bağımsız yaşarken öğrenme | [`research/`](../../research) altında ayrı öğreniciler, güçlü rakipler ve sonuçlar | [11](tr/11-arastirma-laboratuvari.md), [12](tr/12-acik-sorular.md), [korunmuş raporlar](../research) | Her raporun kendi sonuç ve verification dosyası |

## Eski belgelerle nasıl ilişki kuruyoruz?

Mevcut [modül indeksi](../modules/README.md) ve güncel iki dilli neural/tokenizer/training/cognitive rehberleri ayrıntı katmanı olarak kullanıldı. Kitap bunları tek dosyada yeniden üretmez. [Mimari sözleşme](../architecture/CEVAHIR_ARCHITECTURE_SPEC.md), [alt çekirdek sözleşmeleri](../architecture/LOWER_CORE_CONTRACTS.md), [sistem bütünlüğü](../architecture/SYSTEM_OVERVIEW.md) ve [benchmark kaydı](../../benchmarks/README.md) ayrıca başvuru noktalarıdır.

Bazı kayıtlar ise tarihsel tasarım veya önceki denetimdir:

- [Attention audit](../module_audits/4_attention_audit.md) içindeki ölçekleme ve satır toplamı açıklamaları güncel kodla doğrudan eşit sayılmaz. Bugünkü ölçek `1/(sqrt(d_head) * temperature)`; tam maskeli satırlar ve dropout için ayrıntı 4. bölümde.
- [ModelManager rehberi](../modules/model_management/README.md) ve [İngilizcesindeki](../modules/model_management/README-en.md) bazı örnekler eski `setup_tensorboard`, save argümanları ve loss hizalaması kullanır. Güncel `configure_tensorboard`, save/load ve hazırlanmış hedef sözleşmesi 6–7'de gösterilir.
- [Cognitive tasarım belgeleri](../modules/cognitive_management/architecture/README.md) etkin çağrı yolundan daha geniş tasarım hedefleri içerir. SQL öncelik sorgusu, vektör retrieval ve dış araç kaydı 9–10'da ayrı gösterilir.
- [Sonraki geliştirme planı](../architecture/NEXT_DEVELOPMENT_ROADMAP.md) o taramanın bulgularını saklar; bütün maddeleri bugün de açık hata sayılmaz. Örneğin güncel handler sistem talimatını geçirir ve `entropy_details` ID çıktısını kullanır. Araç parametresi çıkarımındaki sınır ise 9. bölümde mevcut kaynak üzerinden gösterilir. Planın geçmiş bulguları sessizce silinmemiştir.
- [Gelişim günlüğü](../development/EVOLUTION_LOG.md) ve eski araştırma raporları o tarihteki kararları/sonuçları taşır. Sonraki düzeltme, önceki kaydı silmek için gerekçe değildir.

Bir tarihsel belgedeki örneği çalıştırmadan önce buradaki güncel sembolün imzasını inceleyin. Eski kaydın varlığını gizlemek yerine kitaptan ona giderken farkını belirtiyoruz.

## Kanıt kapsamı

Bu kitap sürümü için çekirdek/attention/MoE grubunda **82**, eğitim/cache/checkpoint/tokenizer/runtime grubunda **67** hedefli test geçti. Toplam **149 test**, Windows üzerinde Python 3.14.3 ve CPU PyTorch ortamında çalıştırıldı. Altı uyarı eski API/deprecation bildirimleriydi; bütün repository testleri çalıştırılmış veya temiz ilan edilmedi. Ayrıntılı komutlar ve kapsam [verification kaydında](evidence/verification.json).

Yeni düzeltme araştırması ayrıca tam sayı enumeration ve ayrı süreçte devam ile doğrulandı. Önceki araştırma sonuçları tarihli kendi kayıtlarından aktarılır; kitap yazımı sırasında hepsi yeniden çalıştırılmış sayılmaz. GPU başarımı, eğitilmiş Türkçe dil kalitesi, gerçek kullanıcı davranışı ve Mermaid'in bütün görüntüleyicilerde görünümü bu kontrollerin kapsamı değildir.

Bir bağlantının ve metot adının varlığını doğrulamak, anlatımın matematiksel veya davranışsal doğruluğunu ispatlamaz. Bu yüzden mekanik denetim, kaynak incelemesi ve hedefli test farklı kanıt türleri olarak tutulur.
