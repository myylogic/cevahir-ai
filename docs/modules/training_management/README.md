# Training Management — eğitim döngüsü ve durum yönetimi

[Türkçe](README.md) · [English](README-en.md) · [Mimari](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md)

**Kodla eşleştirme:** 20 Eylül 2026. **Etkin arka uç:** `training_management.v2`.

Bu modül modelin nasıl öğrendiğini yönetir: ileri geçiş, kayıp, geri yayılım,
optimizer güncellemesi, doğrulama, öğrenme oranı ve eğitim durumunun kaydı.
Veri hazırlama ve servis kurulumu [training_system](../training_system/README.md) kapsamındadır.

## Sürüm ve çağrı zinciri

```text
training_system/train.py
  → TrainingServiceV3 (içe aktarılamazsa V2 TrainingService)
  → training_backend="v2"
  → training_management.v2.TrainingManager
  → TrainingLoop → model → loss → backward → optimizer
```

| Ad | Bugünkü anlamı |
|---|---|
| TrainingService V3 | Cache, veri yükleme ve eğitim bileşenlerini bir araya getiren üst katman |
| TrainingManager V2 | Servisin kullandığı, geliştirilmeye devam edilen eğitim arka ucu |
| TrainingManager V3 | Ayrı deneysel uygulama; desteklenen servis yoluna bağlanmış değil |
| Neural network V5–V8 notları | Model alt katmanlarının gelişim işaretleri; eğitim yöneticisinin sürümü değil |

V2 adı yalnız eski özellikleri ifade etmez. Güncel accumulation, precision,
MoE yardımcı kaybı ve checkpoint kimliği düzeltmeleri bu etkin yola uygulanmıştır.
Desteklenen arka ucu [contracts.py](../../../training_management/contracts.py) belirler.

## Etkin bileşenler

| Bileşen | Sorumluluk |
|---|---|
| [TrainingManager](../../../training_management/v2/core/training_manager.py) | Epoch akışı, doğrulama, early stopping, callback ve resume |
| [TrainingLoop](../../../training_management/v2/core/training_loop.py) | Batch işleme, autocast, accumulation ve optimizer adımları |
| [LossComputation](../../../training_management/v2/core/loss_computation.py) | Token kaybı, PAD / `-100` maskelemesi, accuracy ve kayıptan türetilen perplexity |
| [GradientManager](../../../training_management/v2/core/gradient_manager.py) | Gradyan kırpma ve norm takibi |
| [TrainingScheduler](../../../training_management/v2/utils/training_scheduler.py) | Warmup ve epoch / metrik tabanlı öğrenme oranı takibi |
| [CheckpointManager](../../../training_management/v2/utils/checkpoint_manager.py) | Eğitim checkpoint'i, best / last kayıtları ve saklama sınırı |

## Öğrenme sözleşmesi

- Modelin logits çıktısı `[batch, sequence, vocabulary]`, hedefler `[batch, sequence]` biçimindedir.
- PAD ve `-100` hedefleri kayıp ve accuracy hesabından çıkarılır.
- Criterion üzerindeki sınıf ağırlıkları, label smoothing ve entropy katsayısı kullanılır.
- Modelin ağırlıklandırılmış MoE yardımcı kaybı aynı ileri geçişin eğitim hedefine bir kez eklenir.
- Accumulation, geçerli hedef token sayısıyla normalize edilir; son eksik grup da güncellenir.
- Warmup sayacı tamamlanan optimizer adımlarını izler; sampler epoch başında güncellenir.
- Sonlu olmayan toplam hedef veya gradyan geçersiz optimizer güncellemesine yol açmamalıdır.

Precision seçenekleri `auto`, `fp32`, `fp16` ve `bf16` değerleridir.
CPU üzerinde `fp16` isteği `fp32`ye döner; açık `bf16` isteği CPU autocast kullanır.
CUDA `bf16` desteği denetlenir; CUDA `fp16` için GradScaler kullanılır.
Bu seçenekler donanıma göre seçilir; burada GPU performans ölçümü iddia edilmez.

## Checkpoint ve eğitime devam

Güncel kayıt modeli, optimizer'ı, model kurulum bilgisini ve tokenizer kimliğini taşır.
Ek eğitim durumu scheduler, scaler, optimizer adımı, en iyi kayıp, early stopping
sayacı ve Python / NumPy / Torch rastgelelik durumlarını kapsar.

`resume_from_checkpoint()` model kimliğini, ağırlık şekillerini ve tokenizer kimliğini
kontrol eder; başarılı yüklemeden sonra sonraki epoch'tan devam eder.
`epochs`, devam edilecek ilave epoch sayısıdır. Batch ortasından devam sunulmaz.
Eski kayıtlar kullanılabilir; ek durum yoksa tam devamın mümkün olmadığı bildirilir.
Eğitim sürdürme, yalnız ağırlıkları yükleyip çıkarım yapmaktan farklı bir işlemdir.

Etkin `TrainingManager.resume_from_checkpoint()` optimizer yükleme hatasını çağırana iletir.
Ayrı `CheckpointManager.load()` yardımcı yolu ise aynı hatayı uyarı olarak kaydedip devam eder.
İki yolun hata davranışı henüz birleşmemiştir; hiçbirinde tüm durum geri yüklemesi atomik değildir.

## Destek sınırı

`training_backend="v3"` etkin servis tarafından kabul edilmez.
EMA, SWA, SAM, Lookahead, LLRD, curriculum, scheduled sampling, focal loss,
AGC, gradient noise ve cosine restarts bayrakları etkinleştirilirse açık hata verilir.
`distributed_strategy` için yalnız `none` desteklenir.
[V3 kaynaklarının](../../../training_management/v3/core/training_manager.py) varlığı
bu özelliklerin desteklenen eğitim zincirine entegre olduğu anlamına gelmez.

## Bir sonraki geliştirme turu

1. Resume sırasında kısmi durum değişimini önlemek ve yardımcı yükleme yolunun hata davranışını eşitlemek.
2. Eğitim checkpoint yazıcısındaki tam bellek tamponunu ve sabit geçici dosya adını kaldırmak.
3. Deneysel V3 yardımcılarının etkin V2 yoluyla ilişkisini haritalamak; seçilecek özellikler için açık entegrasyon ve uyumluluk sözleşmesi hazırlamak.
4. Değişen kayıp bileşenlerini ölçüm raporunda ayırmak; perplexity yorumunu kullanılan hedefle eşleştirmek.

Bu işler [geliştirme yol haritasında](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md)
çekirdek ve bilişsel sistem çalışmalarıyla birlikte önceliklendirilir.

## Kanıt ve kullanım

Projenin [ana README'sindeki](../../../README-TR.md) geçmiş eğitim çıktıları gerçek koşulara aittir.
Küçük sözleşme kontrolleri, bu eğitimlerin yerine geçen model kalite ölçümleri değildir.
Güncel doğrulama kapsamı [benchmarks rehberinde](../../../benchmarks/README.md) izlenir.
Eğitim başlatma, tokenizer ve cache hazırlama için [servis rehberini](../training_system/README.md) kullanın.
