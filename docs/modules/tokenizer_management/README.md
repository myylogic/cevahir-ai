# Tokenizer Management — Türkçeye özel geliştirilmiş tokenizer altyapısı

[English](README-en.md) · [Sistem bütünlüğü](../../architecture/SYSTEM_OVERVIEW.md) · [Mimari sözleşme](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md)

**Güncelleme: 20 Eylül 2026.** Cevahir'in tokenizer katmanı, BPE sözlüğünü ve birleşim kurallarını Türkçe odaklı metin işleme bileşenleriyle birleştirir. Tokenizer eğitimi, metin kodlama/çözme ve model eğitimi için kayıt hazırlama aynı altyapı üzerinden yürütülür.

## Sistem içindeki katkısı

Bu modül, metin ile modelin kullandığı sayısal kimlikler arasındaki bağlantıdır. Geliştirici; sözlüğü, BPE birleşimlerini, normalizasyonu ve isteğe bağlı dil işleme seçeneklerini inceleyebilir. Veri hazırlama ve çıkarım aynı token anlamlarını koruyarak bu altyapıyı kullanır. Böylece dil temsili üzerinde yapılan çalışma modelin eğitim girdisine kadar izlenebilir.

Türkçeye özgü geliştirmeler; Türkçe harflerin işlenmesi, isteğe bağlı `I → ı` / `İ → i` dönüşümü, heceleme ve kural tabanlı morfoloji yardımcılarını içerir. Bunlar kendi başlarına modelin dili öğrendiği anlamına gelmez; model token dizilerinden eğitim sırasında öğrenir.

## Bir tokenizer, birden fazla dil

**Farklı bir dilde çalışmak için mutlaka o dile özgü ayrı bir tokenizer gerekmez.** Aynı sözlük ve BPE birleşimleri, temsil edebildikleri metinler üzerinde birden fazla dil için kullanılabilir. Değerlendirilmesi gerekenler ön işlemenin karakterleri koruması, sözlük/alt parça kapsamı ve token sayısının verimidir. Modelin o dildeki yeteneği ayrıca eğitim verisi ve öğrenilen ağırlıklara bağlıdır.

Mevcut Cevahir ön işlemesinde ASCII harfleri, Türkçe harfler ve rakamlar için karakter filtresi bulunur; standart temizlik `.,!?` dışındaki bazı işaretleri de çıkarır. Bu sürümün tüm Unicode metnini kayıpsız temsil ettiği söylenemez. Bu, her dilin ayrı tokenizer gerektirdiği anlamına gelmez; mevcut temsil kapsamının geliştirilmesi gereken bir sınırdır.

`text_loss_policy`, rol etiketi temizliği ve normalizasyondan sonraki metinde pretokenizer'ın düşürdüğü boşluk dışı karakterleri denetler. `warn` varsayılandır, `error` bu girdiyi reddeder, `ignore` denetimi kapatır. Normalizasyon/çözme aşamalarındaki tüm dönüşüm kayıplarını veya karakter sırasını kanıtlamaz. Encoder'ın karakter fallback'i de byte fallback değildir.

## Çalışma akışı

```text
Ham metin
  → TokenizerCore
  → BPEManager: rol etiketi temizliği ve normalizasyon
  → Pretokenizer: ayarlı temizlik ve metin parçaları
  → isteğe bağlı bütün kelime / hece / morfoloji öğeleri
  → BPEEncoder: sözlük, merge sırası ve fallback
  → (metin tokenları, token ID'leri)
  → model girdisi veya eğitim kaydı

Modelin ürettiği ID'ler
  → BPEDecoder ve son metin işlemleri
  → metin
```

| Bileşen | Rolü |
|---|---|
| [TokenizerCore](../../../tokenizer_management/core/tokenizer_core.py) | Eğitim/çıkarım seçenekleri, encode/decode, batch işlemleri ve eğitim kayıtları için ortak API |
| [BPEManager](../../../tokenizer_management/bpe/bpe_manager.py) | Sözlük/merges, encoder/decoder/trainer ve ön işleme bileşenlerini birlikte yönetir |
| [BPETrainer](../../../tokenizer_management/bpe/bpe_trainer.py) | Eğitim dizilerinden birleşim kuralları ve sözlük geliştirme |
| [BPEEncoder](../../../tokenizer_management/bpe/bpe_encoder.py) | Metin parçalarını sözlük ve sıralı birleşimlerle ID'lere dönüştürme |
| [BPEDecoder](../../../tokenizer_management/bpe/bpe_decoder.py) | ID'lerden metin elde etme ve son işleme |
| [Pretokenizer](../../../tokenizer_management/bpe/tokenization/pretokenizer.py) | NFC, ayarlı harf dönüşümü, temizlik ve parçalama |
| [Syllabifier](../../../tokenizer_management/bpe/tokenization/syllabifier.py) | Türkçe heceleme kuralları |
| [Morphology](../../../tokenizer_management/bpe/tokenization/morphology.py) | Kural tabanlı kök/ek ve biçim yardımcıları |

## Hangi dil bileşeni ne zaman çalışır?

`include_whole_words`, `include_syllables` ve `include_sep` seçenekleri temsil dizisini etkiler. Ana tokenizasyon yolunda morfoloji öğeleri yalnız heceleme ve BPE yapılandırmasındaki `include_morphology` birlikte açıkken eklenir; mevcut çağrı morfolojiye heceleri iletir. Bu yol eksiksiz sözcük düzeyinde dilbilimsel çözümleme garantisi olarak sunulmamalıdır.

TokenizerCore'un standart çıkarım profili hecelemeyi kapalı tutar. Eğitim/çıkarım varsayımları ayrı olabilir; çağrıdaki açık seçenekler ve etkin yapılandırma kontrol edilmelidir. Sözlük dışı öğelerin fallback yolları ile açık hece/morfoloji genişletmesi farklı işlemlerdir.

Temel dönüş sözleşmesi:

```python
def encode_model_input(tokenizer, text):
    tokens, token_ids = tokenizer.encode(text, mode="inference")
    # Modelin sayısal girdisi token_ids'dir; tokens metin parçalarıdır.
    return token_ids
```

## Veri ve modelle birlikte kullanım

[DataLoaderManager](../../../data_loader_management/data_loader_manager.py), metin ve soru-cevap kaynaklarını okur. TokenizerCore eğitim için ID dizileri, hedefler ve kaynak bilgisi üretir. [Cache hazırlama](../../../training_system/prepare_cache.py) bu kayıtları sonraki eğitimler için saklar. Yeni tokenizer oluşturma girişi [train_bpe.py](../../../tokenizer_management/train_bpe.py), ayarlar [config.py](../../../tokenizer_management/config.py) içindedir.

Mevcut eğitilmiş modeli kullanırken onun sözlük, merges ve etkin BPE ayarları korunmalıdır. [Cache kimliği](../../../training_system/cache_identity.py) ve [checkpoint sözleşmesi](../../../model_management/checkpoint_contract.py) bu ilişkiyi denetler. Aynı büyüklükte iki sözlük, aynı ID anlamlarına sahip olmayabilir.

Yeni bir dilde çalışmak otomatik olarak sözlüğü değiştirmek veya tokenizer'ı yeniden eğitmek demek değildir. Önce mevcut temsil değerlendirilir. Temsil değiştirilecekse yeni tokenizer kimliği, hazırlanmış veri ve ağırlık uyumluluğu birlikte ele alınır; geçmiş eğitim varlıkları saklanır.

## Geliştirme alanı

Unicode kapsamı, dönüşüm kayıpları, token verimliliği ve isteğe bağlı dil bileşenlerinin etkisi birlikte değerlendirilebilir. Amaç, Türkçe için yapılan geliştirmeleri koruyarak temsil alanını genişletmektir. Ayrıntılı çalışma sırası [geliştirme planında](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md) yer alır. Bu belge herhangi bir başka tokenizer'a karşı ölçülmemiş kalite veya hız üstünlüğü iddia etmez.
