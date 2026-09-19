# Training System — veriden eğitim servisine

[Türkçe](README.md) · [English](README-en.md) · [Mimari](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md)

**Kodla eşleştirme:** 20 Eylül 2026. **Etkin servis:** `TrainingServiceV3`.

Bu katman tokenizer, önceden hazırlanmış veri, model, criterion, optimizer ve
eğitim yöneticisini aynı koşu için birleştirir. Gradyan güncellemeleri
[training_management](../training_management/README.md) tarafından yürütülür.

## Gerçek çağrı zinciri

```text
Mevcut tokenizer veya tokenizer_management/train_bpe.py
  → training_system/prepare_cache.py
  → training_system/train.py
  → TrainingServiceV3
  → TrainingManager V2 → model eğitimi
```

[train.py](../../../training_system/train.py), V3 servis içe aktarılabiliyorsa onu seçer;
içe aktarım başarısızsa V2 servise döner. V3 çalışma zamanı hataları otomatik olarak
deneysel V3 TrainingManager'a veya başka bir eğitim arka ucuna yönlendirilmez.
Her iki servis yolunun desteklenen arka ucu `training_backend="v2"` değeridir.
V3 servis sürümü, neural network V5–V8 gelişim notlarından bağımsızdır.

## Sorumluluklar ve kaynaklar

| Bileşen | İşlev |
|---|---|
| [prepare_cache.py](../../../training_system/prepare_cache.py) | Veriyi tokenizer ile işler; eğitim cache'ini hazırlar |
| [TrainingServiceV3](../../../training_system/v3/core/training_service_v3.py) | Cihaz, tokenizer, veri, model ve eğitim bileşenlerini kurar |
| [ConfigManagerV3](../../../training_system/v3/core/config_manager_v3.py) | Servis ayarlarını model ve eğitim katmanlarına taşır |
| [DataCacheV3](../../../training_system/v3/data/cache_v3.py) | Cache kimliği, metadata ve bütünlük doğrulaması |
| [cache_identity.py](../../../training_system/cache_identity.py) | Veri ve tokenizer içerik kimliklerinin ortak hesabı |
| [data_split.py](../../../training_system/data_split.py) | Kaynak ve aynı içerik gruplarını koruyan train / validation ayrımı |
| [DataLoader V3](../../../training_system/v3/data/dataloader_v3.py) | Dataset, uzunluk kovaları, dinamik padding ve worker ayarları |
| [V2 TrainingService](../../../training_system/v2/core/training_service.py) | Eski servis girişleriyle uyumluluk |

## Veri hazırlama akışı

Komutlar repository kökünden çalıştırılır. Mevcut eğitimle eşleşen tokenizer
varsa yeniden tokenizer eğitmek gerekmez. Yeni tokenizer gerekiyorsa:

```powershell
python tokenizer_management/train_bpe.py education
```

Tokenizer ve veri biçimi ayarlarını seçtikten sonra cache hazırlanır:

```powershell
python training_system/prepare_cache.py --data-dir education --no-clear-cache
```

`--no-clear-cache` mevcut cache dosyalarını topluca temizlemeyi önler.
Bu seçenek verilmezse hazırlama aracı eski cache'i temizler.
`--max-seq-length`, `--include-whole-words`, `--include-syllables` ve `--include-sep`
seçenekleri hazırlanacak veriyi etkiler; eğitim ayarlarıyla eşleşmelidir.

Son aşama, uygun koşu ayarlarıyla `python training_system/train.py` komutudur.
Dosyadaki hazır profil 100 epoch, batch 64 ve büyük model ayarları içerir;
düşük donanımda hızlı kontrol profili olarak kullanılmamalıdır.
Bu belgeyi güncellemek için yeni eğitim çalıştırılmamıştır.

## Cache ve ayrım sözleşmesi

- V3 eğitim servisi hazırlanmış cache kullanır; eksik cache'i ham veri işleyerek tamamlamaz.
- Veri, tokenizer ve biçimlendirme ayarları cache kimliğine katılır.
- Checksum dosya bütünlüğünü denetler; tek başına eğitim verisinin anlamsal doğruluğunu kanıtlamaz.
- Aynı kaynak kimliğine ait parçalar aynı bölüme gider.
- Tam aynı input / target içeriği farklı kaynakları da birbirine bağlar ve aynı bölümde tutar.
- Karışık veya eksik kaynak kimlikleri reddedilir; tüm kimlikler yoksa içerik gruplaması uygulanır.
- En az iki bağımsız grup gerekir; hedef oran grup sayısı üzerinden uygulanır.
- Yakın kopyalar veya kaynak kimliği olmadan farklı parçaları paylaşılan belgeler için tam sızıntı güvencesi yoktur.

Bucket batching benzer uzunlukları birlikte işler; dinamik padding her batch'in
uzunluğuna göre uygulanır. Bu, gereksiz padding işini azaltmayı amaçlar.
CPU ve CUDA veri yükleme ayarları bulunur; kazanç veri ve donanıma bağlıdır.

## Model ve eğitim durumu

Servis tokenizer kimliğini modele ve eğitim ayarlarına bağlar.
Açık resume yolu yoksa checkpoint dizinindeki kayıtları arar; `last.pth`, ardından
`best.pth` önceliklidir. Seçilen kayıt TrainingManager tarafından yüklenir.
Model ve optimizer durumunun devamı [eğitim yöneticisi rehberinde](../training_management/README.md) açıklanır.
Desteklenmeyen optimizasyon bayrakları [ortak sözleşmede](../../../training_management/contracts.py) reddedilir.

## Bir sonraki geliştirme turu

Mevcut `_validate_alignment()` yalnız ilk örnekte input ile target'ın tamamen aynı
olmasını denetler; bütün kayıtlarda doğru next-token ilişkisini kanıtlamaz.
Veri biçimine uygun token aralığı, maskeleme ve hizalama denetimi genişletilmelidir.
Split kökeni ve yakın kopya davranışı ayrıca izlenebilir hale getirilmelidir.

Ayrıntılı öncelikler [geliştirme yol haritasında](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md) yer alır.
[Gerçek geçmiş eğitim çıktıları](../../../README-TR.md) ile
[küçük doğrulama ölçümleri](../../../benchmarks/README.md) ayrı bağlamlarda sunulur.
