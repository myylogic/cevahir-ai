# 3. Sayılardan öğrenilebilir bir hesaba: sinir ağı

[Kitabın içindekileri](../README.md) · [Önceki: Metin ve tokenizer](02-metin-tokenizer.md) · [Sonraki: Attention](04-attention.md)

Tokenizer metni tamsayılara dönüştürdü. Fakat `42` numaralı token, `21` numaralı tokenın iki katı anlama gelmez. Kimlikler bir sözlüğün adresleridir. Sinir ağının ilk işi bu adreslerden hesap yapılabilecek vektörler seçmektir. Son işi ise bağlamdan çıkardığı vektörü, sözlükteki her olası sonraki token için bir skora dönüştürmektir. Bu bölüm iki uç arasındaki öğrenilebilir hesabı kurar; dikkat mekanizması ve Transformer bloğunun içi sonraki iki bölümde açılır.

## Parametre, aktivasyon ve katman

Bir doğrusal katmanı satır vektörleriyle `y = x Wᵀ + b` diye yazabiliriz. `W` ve varsa `b` öğrenilebilir **parametrelerdir**. `x` ve `y`, o çağrıda hesaplanan **aktivasyonlardır**. Bir girdiye cevap vermek aktivasyonları değiştirir; tek başına parametrelerin öğrenildiği anlamına gelmez. Eğitim, kaybın parametrelere göre türevlerini hesaplayıp bir optimizer aracılığıyla parametreleri günceller.

Cevahir'de bu ayrım elle tutulan sayı listeleri yerine PyTorch'un `nn.Module`, `nn.Parameter`, `nn.Linear` ve `nn.Embedding` nesneleri üzerinden kuruludur. [CevahirNeuralNetwork](../../../src/neural_network.py#L85) bir `nn.Module` alt sınıfıdır. `self.layers` için `nn.ModuleList` kullanılması, alt katmanların parametrelerinin modele kaydolmasını sağlar. Katmanların peş peşe çağrılması ise aynı parametreyi tekrar kullanmak değildir: listedeki her Transformer katmanı ayrı bir nesnedir.

Doğrusal dönüşümleri yalnızca art arda koymak, araya başka işlem girmediğinde yine doğrusal bir dönüşüm verir. Cevahir'in hesap ailesini genişleten işlemler arasında attention içindeki softmax ve FFN içindeki SiLU/GELU vardır. Bunların kod karşılıkları sırasıyla [MultiHeadAttention._standard_sdpa_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L613) ve [FeedForwardNetwork.forward](../../../src/neural_network_module/ortak_katman_module/feed_forward_network.py#L294) içinde görülebilir. Her sayısal işlemi “nöron” benzetmesiyle anlatmak yerine, hangi dönüşümün hangi tensöre uygulandığını takip edeceğiz.

## Embedding: kimlikten vektöre

`V` sözlük büyüklüğü, `D` temsil genişliği olsun. Embedding tablosu `E ∈ R^(V×D)` biçimindedir. Token kimliği `i` için ilk temsil `E[i]` satırıdır. Bu, sıralama veya kimlik numarasının büyüklüğü üzerinden anlam çıkarmaz; hangi satırın okunacağını belirler.

[LanguageEmbedding.__init__](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L56) tabloyu `nn.Embedding` ile kurar. [LanguageEmbedding.forward](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L234) `torch.long` giriş bekler; tabloya bakar, açıksa ölçekler, normalizasyon ve dropout uygular. Ana model bu modülü `scale_by_sqrt=False` ile kurar. `norm_type` verilmediği için embedding çıktısında LayerNorm bulunur. Bu, Transformer katmanlarında RMSNorm seçilmesiyle çelişmez: iki ayrı yerde iki ayrı işlem vardır.

Örneğin iki metnin dörder tokenı için giriş şekli `[2,4]`, embedding çıkışı `[2,4,D]` olur. Aynı token tablodan aynı satırı getirir. Ancak sonraki attention katmanları bağlama göre farklı temsiller üretir; tokenın başlangıç vektörü ile bağlam içindeki son vektörü birbirine karıştırılmamalıdır.

Tablonun başlangıcı da öğrenilmiş anlam değildir. [_initialize_weights](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L262) başlangıç dağılımını oluşturur, satırları normalize edip `0.15` ile ölçekler ve ayarlanmışsa padding satırını sıfırlar. Buradaki `0.15` bir satır ölçeğidir; her bileşenin standart sapmasının `0.15` olduğuna dair sonuç çıkarılamaz. Gerçek kelime ilişkilerinin edinilmesi eğitim verisi ve hedefe bağlıdır.

## Bir ileri geçişi dosyalar arasında izlemek

Modeli üst düzeyden kuran yol [ModelManager.build_model](../../../model_management/model_manager.py#L209) içindedir. Yapılandırma normalize edilir ve initializer aracılığıyla çekirdek oluşturulur. Çalıştırma sınırlarından biri [ModelManager.forward](../../../model_management/model_manager.py#L486), asıl tensör hesabı ise [CevahirNeuralNetwork.forward](../../../src/neural_network.py#L725) metodudur.

| Adım | Girdi → çıktı | Gerçek çağrı ve sonraki tüketici |
| --- | --- | --- |
| Embedding | `[B,T]` → `[B,T,D]` | `self.embedding(x)`; sonuç `self.pos_encoding` girdisidir. |
| Konum ve dropout | `[B,T,D]` → `[B,T,D]` | `self.pos_encoding(...)`, ardından `self.embed_dropout(...)`; sonuç ilk katmana gider. |
| Katman yığını | `[B,T,D]` → `[B,T,D]` | `for i, layer in enumerate(self.layers)`; her çıktı sonraki katmana verilir. |
| Son normalizasyon | `[B,T,D]` → `[B,T,D]` | `self.output_norm(x)`; sonuç vocabulary projeksiyonuna gider. |
| Vocabulary projeksiyonu | `[B,T,D]` → `[B,T,V]` | `self.output_layer(x_normalized)`; isteğe bağlı soft-cap sonrası skorlar döner. |

`B` bağımsız örnek sayısı, `T` bu çağrının token sayısıdır. Önbellekli üretimde `T` yalnız yeni tokenları kapsayabilir. `seq_proj_dim` adı sizi ayrı bir projeksiyon katmanına götürmemeli: [kurucu](../../../src/neural_network.py#L329) etkin genişlik olarak `embed_dim` kullanır; farklı eski `seq_proj_dim` değeri ağırlık paylaşımını kapatabilir, fakat yeni bir sequence projection oluşturmaz.

## Logit neden henüz olasılık değildir?

Son temsil `h` için vocabulary projeksiyonu `z = h W_outᵀ + b` üretir. `z_i` bir **logit**, yani olasılığa dönüştürülebilen sınırlanmamış skordur. Olasılık gerektiğinde `p_i = exp(z_i) / Σ_j exp(z_j)` kullanılır. Çekirdeğin normal çıktısı bu olasılıklar değildir. [Çıkış hesabında](../../../src/neural_network.py#L848) `logit_soft_cap > 0` ise `c tanh(z/c)` uygulanır; değilse ham logit döner. Varsayılan `c=30` değeri skoru yumuşak biçimde sınırlar, olasılık kalibrasyonu veya doğru cevap garantisi vermez.

Hedef token `y` için en temel çapraz entropi `L = −log p_y` olur. Bu formül “doğru tokenın olasılığı düşükse daha büyük ceza” fikrini sayıya çevirir. Gerçek eğitimde padding, etiket yumuşatma ve ek kayıplar sonucu değiştirebilir. Çekirdek `forward` içinde `backward()` veya `optimizer.step()` yoktur; kayıp hesabı ve güncelleme dış eğitim döngüsünün işidir. [LossComputation.compute_loss](../../../training_management/v2/core/loss_computation.py#L76) logits ile hedeflerin buluştuğu somut örnektir. Hedef kaydırma, maskelenmiş ortalama ve güncellemenin ayrıntısı [eğitim bölümünde](06-egitim.md) izlenir.

## Ağırlık paylaşımı neyi değiştirir?

Giriş tablosu ve çıkış projeksiyonu iki farklı görevde kullanılsa da boyutları uygunsa aynı parametreyi paylaşabilir. [Ana model kurucusundaki](../../../src/neural_network.py#L493) gerçek bağlama şöyledir:

```python
self.output_layer.weight = self.embedding.embedding.weight
```

Bu satır bir kopya üretmez; aynı parametre nesnesini iki kullanım yerine bağlar. Böylece ayrı iki `V×D` tablo yerine ortak tablo bulunur ve iki kullanımın türevleri aynı parametreye katkı verir. Mevcut çekirdek, paylaşım açıkken çıkış bias'ı oluşturmaz. Bu bir kapasite ve parametre paylaşımı tercihidir; kaynakta bulunmayan geçmiş bir tasarım kararının gerekçesini burada icat etmiyoruz.

Normal çekirdek çağrısı `(logits, attention_weights)` döndürür. Attention ağırlıkları varsayılan olarak `None` olur. Cache istendiğinde üçüncü değer eklenir; mevcut kodda bu son katmanın cache çıktısıdır, bütün katmanların listesi değildir. Her katmanın gerçek devam durumu kendi attention nesnesinde tutulur.

## Kod okuma deneyi ve kanıt sınırı

[Alt katman sözleşme testleri](../../../tests/evolution/test_lower_layer_contracts.py#L62), açık attention teşhisiyle hızlı yolun çıktı ve gradyanlarını karşılaştırır. [Yapılandırma testi](../../../tests/evolution/test_config_generation.py#L65), facade ile yöneticinin model boyutları ve durum aktarımını sınar. Bunlar hesap bağlantılarını denetler; dil anlama kalitesini ölçmez. Bu bölümde test adı verilmesi tek başına bu yazım oturumunda çalıştırıldığı anlamına gelmez; güncel yürütme kaydı kitabın doğrulama çıktısındadır.

Okuma alıştırması olarak `vocab_size=128`, `embed_dim=32`, `B=2`, `T=5` için tablonun ve logits'in şekillerini çıkarın. Ardından çekirdeğin `forward` metodunun optimizer adımı çağırmadığını, buna karşılık cache ve teşhis durumunun değişebildiğini bulun. Bu ayrım, araştırma bölümlerinde sorulan “deneyim nedeniyle kalıcı kapasite değişimi” ile sıradan bir model çağrısını ayırmak için gereklidir.

Bu anlatım [mevcut sinir ağı modül belgesinin](../../modules/neural_network/README.md) doğrulanmış akışını genişletir. İngilizce eşdeğeri ve önceki araştırma belgeleri yerlerinde korunur.
