# 6. Veriden parametre güncellemesine

[İçindekiler](../README.md) · [Önceki: Transformer](05-transformer.md) · [Sonraki: Model yaşam döngüsü](07-model-yasam-dongusu.md)

Bir Transformer'ın doğru boyutta logits üretmesi, yararlı bir dil modeli olduğu anlamına gelmez. Eğitim, örneklerdeki hedef tokenları daha iyi öngörecek biçimde parametreleri değiştirir. Bu değişimin anlamını veri hazırlama belirler: hedef yanlış hizalanmışsa kusursuz çalışan optimizer yanlış görevi öğrenir. Bu yüzden eğitim anlatımını `backward()` satırından değil, metnin hedefe dönüşmesinden başlatıyoruz.

## Aynı veri, iki ayrı öğrenme işi

Tokenizer eğitimi sözlüğü ve birleşim kurallarını; dil modeli eğitimi bu kimliklerden sonraki kimlikleri öngören ağırlıkları belirler. İkisi aynı işlem değildir. Cevahir'in dil modeli eğitimi sırasında sözlüğü serbestçe büyüttüğünü varsayamayız. [`TokenizerCore.load_training_data`](../../../tokenizer_management/core/tokenizer_core.py#L720) mevcut tokenizer ile metinleri kodlar. Soru-cevap verisinde soru ve cevap arasına SEP yerleştirir, ham metni parçalara ayırır ve kaynak kimliğini taşır. Bu aşamadaki girdi/hedef listeleri hazırlanmış next-token batch'i ile karıştırılmamalıdır.

[`prepare_cache`](../../../training_system/prepare_cache.py#L62) içindeki `format_data_func`, normal kayıtta aynı token dizisinden şu çifti oluşturur:

```text
Metnin ID'leri:  [t1,  t2, t3]
Model girdisi:  [BOS, t1, t2, t3]
Hedef:         [t1,  t2, t3, EOS]
```

Her giriş konumunun çıktısı aynı konumdaki hedefe karşılaştırılır. Model `t2`yi girdide gördüğü konumda `t3`ü öngörür; nedensel attention gelecekteki konumlara erişimi engeller. Hazırlama kodundaki bağlantı:

```python
seq_in = [BOS_ID] + list(inp_ids)
seq_tgt = list(tgt_ids) + [EOS_ID]
```

Ardından uzunluk sınırı, EOS'un korunması ve PAD uygulanır. Uzun diziyi kesip son hedefi EOS yapmak, orijinal belgenin bitişiyle aynı olay değildir; hazırlama politikası öğrenilen durma davranışını etkiler. Eğitim döngüsü hedefleri **yeniden kaydırmaz**. Soru-cevap biçimi de kendiliğinden yalnız cevap tokenlarının kaybını hesaplayan bir maske demek değildir: hangi hedeflerin dışlandığı batch ve loss maskesinden okunur.

Cache, tokenlaştırmayı her epoch'ta yinelememek içindir. [`cache_identity.py`](../../../training_system/cache_identity.py) kaynak içeriğini, vocabulary/merges ve ilgili kodlama ayarlarını parmak izine katar. V3 tüketim yolu cache kimliği ve bütünlüğünü kontrol eder; aynı dosya adı aynı veri anlamına gelmez. Hazırlama komutunun varsayılan temizleme davranışı mevcut cache'i etkileyebilir; ayrıntılar için [hazırlama rehberi](../../modules/training_system/README.md) ve komutun gerçek seçenekleri birlikte okunmalıdır.

## Değerlendirme verisi gerçekten ayrı mı?

Aynı belgenin örtüşen parçalarını rastgele ikiye dağıtırsak validation kaybı yeni belge başarısını abartabilir. [`split_training_records`](../../../training_system/data_split.py#L6) aynı kaynak kimliğinin parçalarını birlikte tutar. Tam eş örnekler farklı kaynakları bağlıyorsa bu grupları da birleştirir. Kimlikler kısmen eksikse hata verir; tümünde yoksa içerik eşitliği üzerinden gruplama yapılabilir. En az iki bağımsız grup gerekir.

Bu yöntem semantik yakın kopyaların tamamını bulmaz. Bölme oranı kaynak grupları üzerinden hesaplandığından tokenların aynı oranda bölüneceği garanti değildir. Bir kaynağın bütün örnekleri çok uzunsa eğitim ve validation hacimleri dengesiz olabilir. Validation sonucunu yorumlarken yalnız oranı değil kaynak dağılımını da bilmek gerekir.

## Etkin çağrı yolu

[`training_system/train.py`](../../../training_system/train.py) kullanılabilir olduğunda V3 eğitim servisini, aksi import yolunda V2 servisini seçer. Ancak V3 servis adı bağımsız V3 optimizer döngüsünün etkin olduğu anlamına gelmez. [`TrainingServiceV3._run_training`](../../../training_system/v3/core/training_service_v3.py#L420), desteklenen **V2 `TrainingManager`** nesnesini kurar. Kaynaktaki eski “fallback” yorumundan daha belirleyici olan bu doğrudan import ve çağrıdır.

```mermaid
flowchart TD
    A[Ham metin ve soru-cevap] --> B[TokenizerCore.load_training_data]
    B --> C[prepare_cache / BOS-hedef-EOS / kaynak kimliği]
    C --> D[TrainingServiceV3.load_data_from_cache]
    D --> E[split_training_records]
    E --> F[create_dataloaders_v3]
    F --> G[V2 TrainingManager.train]
    G --> H[TrainingLoop.train_epoch]
    H --> I[Neural core / logits]
    I --> J[LossComputation + MoE auxiliary loss]
    J --> K[Backward / biriken gradient]
    K --> L[_finish_update / optimizer]
    L --> M[Validation ve checkpoint]
```

| Sınır | Girdi ve dönüşüm | Çıktı ve çağıran |
|---|---|---|
| `TrainingServiceV3.train` | Config, tokenizer, model yöneticisi, doğrulanmış cache | Loader ve eğitim bileşenlerini kurar; `_run_training`e verir. |
| [`create_dataloaders_v3`](../../../training_system/v3/data/dataloader_v3.py#L178) | Hazırlanmış eğitim/validation kayıtları, PAD ve batch ayarları | V3 dataset/collator üzerinden iki loader; servise döner. |
| `TrainingManager.train` | Model, optimizer, criterion, loader ve config | Epoch döngüsü, validation, scheduler/checkpoint koordinasyonu. |
| [`TrainingLoop.train_epoch`](../../../training_management/v2/core/training_loop.py#L283) | `[B,T]` giriş ve hedef batch'leri | Modeli doğrudan çağırır; logits ile kayıp üretir, ağırlıkları günceller. |
| [`LossComputation.compute_loss`](../../../training_management/v2/core/loss_computation.py#L76) | `[B,T,V]` logits ve `[B,T]` hedef | Türevlenebilir skaler kayıp, accuracy, raporlanan perplexity. |

`use_bucket_batching=True` benzer uzunlukları bir araya getirerek dolgu israfını azaltmayı amaçlar. `use_dynamic_padding=True` batch'i kendi uzunluğuna göre hazırlar; her örneği küresel üst sınıra taşımak gerekmez. Bunlar model mimarisi değiştirmez, işlenen PAD miktarını ve batch sırasını etkiler. [`create_dataloader_v3`](../../../training_system/v3/data/dataloader_v3.py#L45) shuffle açıkken bucket sampler seçer; validation sıralı kalır. `batch_size`, `num_buckets`, `max_seq_length`, `num_workers` ve aygıta bağlı `pin_memory` maliyetleri etkiler. Daha az padding her makinede ölçülmüş hız kazancı demek değildir.

## Loss, gradient ve geri yayılım

Basit next-token kaybı, hedef tokenın negatif log olasılığıdır:

$$
L(\theta)=\frac{1}{N}\sum_{b,t}m_{bt}\big[-\log p_\theta(y_{bt}\mid x_{b,\leq t})\big],
\qquad N=\sum_{b,t}m_{bt}.
$$

Burada `m`, geçerli hedef maskesidir. Cevahir `PAD` ve `-100` hedeflerini dışlar. Logits üzerinde float32 cross-entropy hesaplanır; criterion'daki sınıf ağırlıkları ve `label_smoothing` kullanılır. Dolayısıyla EOS ağırlığı veya smoothing etkin olduğunda hedef tam olarak yukarıdaki yalın denklem değildir. `entropy_coeff` ayrıca tahmin entropisini ödüllendiren bir terim ekleyebilir. Bu ayarların etkisi “daha akıllı model” gibi genel bir sonuçla değil, değişen amaç fonksiyonuyla açıklanmalıdır.

Gradient, kaybın her parametreye göre yerel türevidir. Backpropagation zincir kuralıyla bu türevleri hesaplar; **parametreyi kendisi değiştirmez**. PyTorch işlemlerin hesap grafiğini izler, `backward()` ile gradientleri biriktirir; optimizer bunları kullanır. [PyTorch'un autograd anlatımı](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) mekanizmanın bağımsız açıklamasıdır. Cevahir karşılığı `TrainingLoop.train_epoch` içindeki ileri geçiş → kayıp → backward zinciridir.

Yoğun modelde örnek optimizer adımı `θ ← θ − ηg` olabilir. AdamW gibi yöntemler ise geçmiş gradientlerden ek durum taşır ve koordinatlara göre adımı ölçekler. [`ModelManager.build_optimizer`](../../../model_management/model_manager.py#L298), [`ModelInitializer.initialize_optimizer`](../../../model_management/model_initializer.py#L251) üzerinden bu seçimi yapar. Yerel varsayılan `optimizer="adamw"`dir; etkin öğrenme oranı üst config tarafından belirlenebilir. `learning_rate`, `weight_decay`, `betas`, `eps`, `embedding_lr_scale` ve isim tabanlı decay grupları burada kullanılır. Bias/norm grupları ile embedding grubu aynı kuralı paylaşmayabilir. SGD daha az optimizer durumu tutan bir alternatiftir; birinin üstünlüğü bu kodun varlığından çıkarılamaz.

## SGD, Adam ve AdamW hangi hesabı yapar?

### Bir adım ile optimizer durumunun ayrımı

Yalın SGD, minibatch gradyanı $g_t$ ile $\theta_{t+1}=\theta_t-\eta_tg_t$ uygular. Momentum eklendiğinde geçmiş yönün bir özeti de taşınır. Adam, ilk ve ikinci moment tahminlerini ayrı tutar:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}.
$$

Sıfır başlangıç için $\hat m_t=m_t/(1-\beta_1^t)$ ve $\hat v_t=v_t/(1-\beta_2^t)$ düzeltmeleriyle temel adım $-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ olur. İşlemler koordinat bazındadır. İkinci moment, ortalama çıkarılmış istatistiksel varyansla aynı şey değildir. Adam'ın özgün yöntemi [Kingma ve Ba, 2014/2015](https://arxiv.org/abs/1412.6980) içindedir. Bu denklem belirli bir görevde SGD'den üstünlük garantisi vermez.

İlk adımda tek parametre gradyanı `2`, başlangıç momentleri sıfır, `β₁=0.9,β₂=0.999` olsun. `m₁=0.2`, `v₁=0.004`, düzeltmelerden sonra `m̂₁=2`, `v̂₁=4` elde edilir. Epsilon küçükken Adam adımı yaklaşık `−η`, SGD adımı `−2η` olur. Bundan her sonraki Adam adımının yalnız işarete bağlı olduğu sonucu çıkmaz; sonraki momentler bütün geçmişi içerir. Checkpoint yalnız ağırlıkları taşıyorsa bu geçmiş eksiktir.

### Weight decay ile L2 cezası neden aynı olmayabilir?

Loss'a $\frac\lambda2\|\theta\|^2$ eklemek gradyana $\lambda\theta$ ekler. Yalın SGD'de adım $(1-\eta\lambda)\theta-\eta g$ olur. Koordinat bazında ölçeklenen adaptif optimizer'da ceza terimi de bu ölçeklemeden geçer. AdamW, ağırlık küçültmeyi adaptif gradyan adımından ayırır; temel gösterim $\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ biçimindedir. [Loshchilov ve Hutter, 2017/2019](https://arxiv.org/abs/1711.05101).

Cevahir'de hangi parametrenin decay aldığı [`ModelInitializer.initialize_optimizer`](../../../model_management/model_initializer.py#L251) içindeki isim temelli gruplamaya bağlıdır. Embedding önce ayrı gruba alınır; diğer parametreler no-decay anahtarlarına göre ayrılır. Bağlı embedding/çıkış aynı nesneyse iki ayrı parametre olarak varsayılmamalıdır. Bir deneyin tekrarlanması için yalnız “AdamW kullandım” demek yeterli değildir: grup öğrenme oranları, decay, beta, epsilon ve optimizer durumunun geri yüklenmesi belirtilmelidir.

### Adım büyüklüğü, warmup ve clipping

Öğrenme oranı parametre uzayındaki adımın ölçeğidir. Gradient clipping ise örneğin global normu `c` ile sınırlarken $g\leftarrow g\min(1,c/\|g\|)$ yapar; büyük gradientin yönünü koruyup büyüklüğünü değiştirir. Clipping, verinin doğru olmasını veya loss'un her adımda düşmesini garanti etmez. AMP kullanıldığında önce gerçek gradyan ölçeğine dönmek gerekir; ölçeklenmiş türevi doğrudan kırpmak farklı bir işlem olur. Yerel sıra [`_finish_update`](../../../training_management/v2/core/training_loop.py#L246) içinde görülebilir.

Warmup başlangıçtaki öğrenme oranını bir programa göre değiştirir. Birikim varken bir minibatch ile optimizer adımı aynı olay değildir. Cevahir'in `_step_scheduler_for_warmup` çağrısı gerçekleşen optimizer güncellemesine bağlıdır; `TrainingManager` epoch ve validation düzeyindeki programı ayrıca yönetir. Scheduler, scaler ve RNG durumunun sürdürülmesi [checkpoint bölümündeki](07-model-yasam-dongusu.md) devam sözleşmesinin parçasıdır.

## Birikimin doğru paydası

GPU/CPU belleği büyük batch'e yetmezse birkaç küçük batch'in gradientleri biriktirilebilir. Fakat her batch'in ortalama kaybını eşit ağırlıkla toplamak, 2 geçerli tokenla 20 geçerli tokena aynı etkiyi verir. Cevahir'in döngüsü ortalama kaybı batch'in geçerli hedef sayısıyla çarpar:

```python
scaled_loss = loss * valid_tokens
```

Sonra [`_finish_update`](../../../training_management/v2/core/training_loop.py#L246), biriken gradienti toplam geçerli token sayısına böler. Son eksik birikim grubu da işlenir. MoE etkinse [`consume_model_auxiliary_loss`](../../../training_management/contracts.py#L35) çekirdeğin zaten ağırlıklandırdığı yardımcı hedefi bir kez tüketir; router kaybı kullanılmayan bir açıklama alanı değildir. Batch düzeyindeki yardımcı terim de aynı birikim yolundan geçtiğinden, denklemde yalnız dil kaybı varmış gibi değerlendirme yapılmamalıdır.

Adım sırasında varsa scaler önce gradientleri gerçek ölçeğe getirir; sonlu olmayan gradientler reddedilir; clipping uygulanır; optimizer adımı gerçekleştirilir; gradientler temizlenir. Scheduler'ın warmup adımı gerçekleşen optimizer güncellemesine bağlanır. `grad_accum_steps` yalnız hız ayarı değildir: kaç deneyimin bir güncellemede birleştiğini değiştirir.

[`resolve_precision`](../../../training_management/contracts.py#L20), `auto/fp32/fp16/bf16` isteğini aygıt ve AMP koşullarıyla çözer. CPU'da fp16 isteği fp32'ye düşer; açık bf16 isteği CPU autocast kullanabilir. CUDA bf16 desteği ayrıca aranır. CPU testinin geçmesi CUDA veya Flash yolunun sınandığı anlamına gelmez.

### Eşit olmayan minibatch'lerde sayısal karşı örnek

İlk minibatch'in iki geçerli tokenında toplam kayıp `2`, ikincinin sekiz tokenında `24` olsun. Minibatch ortalamaları `1` ve `3` olur. Bunların eşit ağırlıklı ortalaması `2`, token başına gerçek ortalama ise `(2+24)/(2+8)=2.6`'dır. Gradyanlar için de aynı ağırlıklandırma farkı vardır. Cevahir'in `loss * valid_tokens` ve son bölme işlemi, aynı parametrelerde hesaplanan toplam token amacının türevini elde etmeyi hedefler.

Büyük batch ile birebir eşdeğerlik yine koşulludur: batch'e bağlı normalizasyon, dropout rastgeleliği, dinamik router yardımcı hedefi ve ara optimizer adımları sonucu değiştirebilir. Bu nedenle [birikim testi](../../../tests/evolution/test_training_contracts.py) dropout kapalı küçük ağda belirtilen sözleşmeyi sınar. Bütün MoE ve veri sıralama düzenlerinde genel eşitlik iddiası değildir.

### Mixed precision neyi değiştirir?

FP16/BF16 gibi biçimler sayı aralığı ve hassasiyet arasında farklı seçimler yapar. Autocast her işlemi körlemesine tek türe dönüştürmek yerine uygun işlem türlerini seçer. Loss scaling küçük gradyanların dar biçimde kaybolmasını azaltmak için loss'u büyütüp optimizer öncesinde ölçeği geri alır. NVIDIA/Baidu araştırmacılarının özgün karma hassasiyet çalışması bu sayısal meseleleri ve yüksek hassasiyetli ağırlık birikimini açıklar. [Micikevicius vd., 2017/2018](https://arxiv.org/abs/1710.03740).

Yerel uygulama `fp16` CUDA yolunda scaler kurar, `bf16` için aynı scaler'ı varsaymaz. LossComputation logitleri çapraz entropi için float32'ye çevirir. “Düşük hassasiyet kullanıldı” ifadesi hangi işlemin hangi türde çalıştığını söylemez; donanım, çözülmüş precision ayarı ve taşma davranışı ayrıca kaydedilmelidir. Bellek azalması ve hız artışı da aynı ölçü değildir.

## Eğitim bütçesi ve genelleme nasıl raporlanır?

### Model büyüklüğü tek başına deney tanımı değildir

Parametre sayısı, işlenen token sayısı, veri kalitesi/tekrarı, güncelleme sayısı, context uzunluğu ve donanım bütçesi farklı değişkenlerdir. DeepMind'ın Chinchilla çalışması sabit hesap bütçesinde model büyüklüğü ile eğitim tokenlarını birlikte seçmeyi deneysel olarak inceler. Buradan Cevahir için otomatik bir en iyi token/parametre oranı türetmiyoruz; veri, hedef ve maliyet koşulları yeniden değerlendirilmelidir. [Hoffmann vd., 2022](https://arxiv.org/abs/2203.15556).

Bir yöntem karşılaştırmasında yalnız daha düşük son loss değil, bu sonuca hangi bütçeyle gelindiği sorulur. Aynı tokenizer, ayrım, context, amaç ve hesap bütçesi korunmadığında farkın yalnız mimariden geldiği söylenemez. Birden fazla rastgele başlangıç ve dağılım kayması altında değerlendirme de tek seed'deki başarıya göre daha geniş bir soru sorar. Bu bölümün küçük CPU deneyi performans yarışı değildir.

### Kayıp azalması ile çalışırken öğrenme arasındaki bağ

[Gerçek küçük çekirdek örneğinde](../evidence/neural_walkthrough.json) yedi geçerli hedefin loss'u bir SGD adımından sonra `3.7709403 → 3.7681756` oldu. Forward ve backward ağırlıkları değiştirmedi; optimizer adımı değiştirdi. Bu, deneyimden parametre değişimine giden hesap yolunun varlığını gösterir. Öğrenilen değişimin kalıcılığı, eski becerilerin korunması, yeni görevlerde aktarım ve çalışma sırasında hangi deneyimin kullanılacağı ayrıca çözülmelidir. Bu geniş soru [araştırma bölümüne](11-arastirma-laboratuvari.md) taşınır.

## Bir bayrak, uygulanmış yöntem değildir

[`validate_training_backend`](../../../training_management/contracts.py#L5) yalnız entegre V2 backend'ine izin verir. EMA, SWA, SAM, lookahead, curriculum, scheduled sampling ve listelenen diğer desteklenmeyen seçenekler açıkken hata üretir. Distributed eğitim stratejisi de bu uçtan uca hatta desteklenmez. Ayrı modülde sınıf veya config alanı bulunması bu yöntemlerin çalışan eğitim servisine entegre olduğunu göstermez. Kitap bu sınırı korur.

Validation ileri geçiş yapar ve ağırlıkları güncellemez; modelin görmediği ayrılmış kayıtlarda aynı hedefin başarısını ölçer. `compute_loss` tarafından raporlanan perplexity `exp(min(20, loss))`tur. Sınıf ağırlığı, smoothing veya entropy terimi kullanıldığında bu sayı saf negatif log olasılığın standart perplexity'si değildir. Benzer şekilde düşük validation kaybı araç kullanımı, doğruluk veya sürekli öğrenme kanıtı değildir.

## Hangi kanıtı okuyabiliriz?

[Eğitim sözleşmesi testleri](../../../tests/evolution/test_training_contracts.py) token ağırlıklı birikimi, son grubu, gerçek V3 servis–V2 döngü bağlantısını, MoE yardımcı hedefini ve resume davranışlarını küçük koşullarda denetler. [Veri/cache testleri](../../../tests/evolution/test_data_cache_contracts.py) kaynak ayrımı, kimlik ve bütünlük sınırlarını izler. Ayrıntılı kullanım için mevcut [training_system](../../modules/training_system/README.md) ve [training_management](../../modules/training_management/README.md) rehberleri korunmuştur.

Eğitim sonunda ağırlıkların değişmesi bir başlangıçtır. Bu değişimi tekrar yükleyebilmek, aynı tokenizer anlamıyla kullanabilmek ve sonraki eğitime doğru durumdan devam edebilmek ayrı sözleşmelerdir. Şimdi ağırlık dosyasının neden tek başına bütün model yaşamını saklamadığını inceleyeceğiz.

## Çözümlü alıştırmalar

1. İki minibatch'te geçerli token sayıları `3,7`, ortalama kayıplar `2,4` ise birleşik token kaybı nedir? **Yanıt:** `(3×2+7×4)/10=3.4`; minibatch ortalaması olan `3` değildir.
2. Global gradyan normu `10`, clipping sınırı `2` ise çarpan nedir? **Yanıt:** `0.2`. Loss scaling açıksa bu hesap gerçek ölçeğe döndürülmüş gradyan üzerinde yapılır.
3. On minibatch ve `grad_accum_steps=4` için son grup da işleniyorsa kaç optimizer adımı vardır? **Yanıt:** `3`; gruplar `4,4,2` minibatch'tir. Scheduler'ın batch ve optimizer sayacını karıştırması programı değiştirir.
4. Smoothing ve entropy terimi içeren loss'un üssü neden standart perplexity olmayabilir? **Yanıt:** Standart yorum ortalama gerçek hedef negatif log olasılığına dayanır; değiştirilmiş amaç aynı büyüklük değildir.

## Kaynakça

1. Kingma, D. P. ve Ba, J. (2014 ön baskı; ICLR 2015). *Adam: A Method for Stochastic Optimization*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
2. Loshchilov, I. ve Hutter, F. (2017 ön baskı; ICLR 2019). *Decoupled Weight Decay Regularization*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
3. Micikevicius, P. vd. (2017 ön baskı; ICLR 2018). *Mixed Precision Training*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).
4. Hoffmann, J. vd. (2022). *Training Compute-Optimal Large Language Models*. NeurIPS. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).
5. PyTorch contributors. *Automatic Differentiation with torch.autograd*. [Resmî öğretici](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), erişim: 21 Eylül 2026.

[Sonraki: Model yaşam döngüsü](07-model-yasam-dongusu.md) · [İçindekiler](../README.md)
