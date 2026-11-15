# 🇹🇷 Cevahir AI & Engine

[English](README.md) · [Türkçe](README-TR.md)

[Araştırma arşivi DOI: 10.5281/zenodo.24235648](https://doi.org/10.5281/zenodo.24235648) · [Araştırma yayınları](docs/publications/README.md)

**Türkçeye özel geliştirilmiş tokenizer altyapısı, eğitilebilir dil modeli çekirdeği ve bilişsel sistemleri birleştiren yapay zekâ motoru.**

Ben Muhammed Yasin Yılmaz. Cevahir'i, verinin hazırlanmasından modelin öğrenmesine ve konuşma içinde kullanılmasına kadar bütün sürecini geliştirebildiğim bir yapay zekâ altyapısı olarak kurdum. Kendi BPE hattını, yapılandırılabilir PyTorch Transformer decoder çekirdeğini, eğitim ve checkpoint yönetimini, metin üretimini, bilişsel iş akışlarını, konuşma belleğini ve uygulama servislerini aynı kod tabanında bir araya getirdim.

Bu projeyle özellikle Türkiye'deki gençlerin bir dil modelinin iç işleyişini inceleyebilmesini, kendi verileriyle eğitim yapabilmesini ve kendi fikirlerini çalışan mekanizmalara dönüştürebilmesini istiyorum. Cevahir'i geliştiriyor, modeller eğitiyor ve elde ettiğim çıktıları burada paylaşıyorum. Kaynak kodunu, deneylerimi ve öğrendiklerimi aynı yerde erişilebilir tutmak benim için bu çalışmanın önemli bir parçası.

**Cevahir, Türk Gençlerine Armağanımdır. — Muhammed Yasin Yılmaz**

Cevahir'i bundan sonra **açık kaynak bir araştırma ve eğitim mirası** olarak sürdürüyorum; ürünleşme artık projenin hedefi değil. Mühendislik çalışmalarıyla sistemin çalıştığı süre boyunca öğrenmesine yönelik araştırma birlikte devam ediyor.

## Yeni çekirdek: Cevahir Zihin — yaşarken öğrenen bir zihin (Kasım 2026)

Kasım 2026'da Transformer'dan ayrı, yeni bir çekirdek üzerinde çalışmaya başladım: **Zihin**. Aradığım şey sabit bir ağ değil; ağın kendisini yaşarken işleten, deneyimlerinden kavram kuran ve kurduğunu ölçen bir zihin. Eski Transformer motoru ve önceki bütün araştırmalar aşağıda olduğu gibi duruyor.

**Neden Transformer değil:**

| Transformer | Cevahir Zihin |
|---|---|
| eğit → dondur → kullan | öğrenmek yaşamaktır: her an öğrenir, ayrı bir eğitim aşaması yoktur |
| sabit boyut, sabit sözlük, gömme vektörleri | büyüyen birimler ve bağlar, açık alfabe; sabit boyut yoktur |
| softmax ile sonraki token | kanaat: her inancın gücü ve kaynağı vardır; hiçbir şey eşiğe birikmezse susar |
| yeni bilgi eskisini bozar, yeniden eğitim gerekir | yeni bilgi eklenir, eskisi bayt bayt korunur |
| bilgi ağırlıklara dağılmıştır, tek bir örnek geri alınamaz | tek bir deneyim çekirdekten tam olarak geri alınabilir |
| bir bilginin nereden geldiği tutulmaz | her kanıt kaynağıyla tutulur (anlatılan, görülen, taklit, kendi eylemi, çıkarım, hayal); kaynağa güven de yaşanarak öğrenilir |

**Ölçülmüş sonuçlar** (üretilmiş iki dünya, 5 tohum; ayrıntılar [sonuçlar](research/living_mind_2026_11_14/RESULTS.md) ve [notlar](research/living_mind_2026_11_14/NOTES.md) dosyalarında):

- İlk duyuş zayıftır, kanaat dinledikçe artar (1 duyuşta c 0.41, 4+ duyuşta 0.78); bir düzeltme eski kanıtı silmez, tartar (120/120).
- Gizli sınıfları kendisi bulur (Jaccard 1.00).
- Hiç görülmemiş örneğe genelleme yaşadıkça iyileşir: 600 deneyimde %93 / %81, 1500 deneyimde %100 / %90. Önceki okuma kuralı yaşlandıkça kötüleşiyordu; sekiz başarısız deneme ve bunun nedeni kayıtlıdır.
- Unutmaz: erken öğrenilen bilgiler yüzlerce deneyim sonra bayt bayt aynıdır; ikinci bir alanı yaşamak ilkini bozmaz (10/10).
- Tek bir deneyim kesin olarak geri alınabilir (10/10).
- Önce taklit eder, sonra sonuçlarına bakarak kopyaladığından kopar; "başkaları böyle yapar" bilgisini korur (T7).
- Aynı deneyim akışında küçük bir Transformer ile karşılaştırma (5 tohum × 2 dünya, aynı sorular): hiç görülmemiş örneğe genellemede Zihin %90 / %97 ve %75 / %95; yaşarken tek geçişte eğitilen Transformer %67 / %57 ve %60 / %50; akışın tamamında yakınsayana kadar eğitilen Transformer %77 / %67 ve %53 / %53. İkinci bir alan, Transformer'da ilk alanın bilgisini 0.85'ten 0.74'e düşürür, Zihin'de 1.00 kalır. Tek bir deneyimi geri almak Zihin'de 20–77 ms ve tamdır; Transformer'da tam karşılığı sıfırdan yeniden eğitimdir. Bu karşılaştırma az veriyle yaşarken öğrenme koşulundadır; Transformer'lar küçüktür ve çok daha fazla veriyle fark kapanabilir.
- Sohbet: izlediği diyalogları söyler; izlemediği sorulara ilişki kurarak cevap verir ("kuş ne yapar" → "kuş öter"); bilmediğinde susar; "yanlış" denen kopyalanmış bir cevabı bir daha söylemez.
- Görür ve duyar (ilk sürüm): resim ve ses dosyalarını eğitilmiş bir ağ olmadan, piksel ve ses aritmetiğiyle algılar ve benzerlikle tanır. Yeni resimlerde %73 / %83 / %67, yeni seslerde %93 / %97 / %93 doğru (şans %33; benzerlik kapalıyken %0).
- Merak ve hayal (ilk biçimler): söyleyecek bir şeyi yokken bilmediği kelimeyi soru tonuyla geri sorar, açıklananı öğrenip kullanır (T4). Bir düşünceyi yapıca benzediği bir kavrama taşıyarak hayal kurar ve her adımın gerçekliğini yaşanmış kanıta göre ölçer (T8).
- Uyku ve rüya: uykuda son konu ve emin olmadığı kelimeler üzerine rüya görür. Sonra hayallerini yaşadıklarıyla tartar, önceki rüyalarından hangisinin doğru, hangisinin yanlış çıktığını söyler ve kendi hayallerine ne kadar güveneceğini buradan öğrenir. Yanlış çıkan hayal silinmez; gerçek kanıt ona ağır basar.
- Kendilik: konuştuğu ve düşündüğü her anda bulunan doğuştan bir "ben" birimi vardır. İç gözlemle kendi yaşamından neyi ne kadar söyleyip düşündüğünü okur.

**Henüz yapamadıkları:** Öğrenmeyi öğrenmek (T6) olumsuz çıktı. Merak, hayal ve rüya ilk biçimlerindedir: gerçeklik puanı akla yatkın bir kurguyu zayıf bir sıçramadan henüz ayıramıyor; rüyalar yalnız kelimelerle görülüyor. Kendi sesi (konuşma çıktısı) yok. Gerçek veride sohbet büyük ölçüde taklittir; kendisi hakkındaki sorulara kendisine anlatılan gerçeklerle cevap verir, kendilik birimi henüz söylediklerini yönlendirmiyor. Dünyalar küçüktür ve Transformer karşılaştırması yalnızca az veri koşulunda yapıldı. Bu çalışma bir bilinç ya da yapay genel zekâ iddiası değildir; bunlar bugün hiçbir makine için ölçülemiyor. Hedef ve yol haritası açıktır: yaşayarak öğrenen, kendi kavramlarını kuran, edindiğini anlamlandıran ve kendini bilen bir zihin. Tasarımın katmanları (bilinçli odak, otomatik refleksler, iç ve dış ses, yönlendiriciler ve irade, kendini bilmek) [burada](research/living_mind_2026_11_14/LAYERS.md), vizyon [burada](research/living_mind_2026_11_14/design/VISION.md), son çalışmalar (uyku ve rüya, işitme) [burada](research/living_mind_2026_11_15/NOTES.md).

**Denemek için** (ayrıntılı kılavuz: [docs/ZIHIN_KILAVUZ.md](docs/ZIHIN_KILAVUZ.md); PowerShell'den: `.\cevahir.ps1`):

```
python -m zihin                      # yeni bir zihin
!ogret diyaloglar.tsv                # her satır: söz<TAB>karşılık (JSON ve Word belgeleri de okunur: !oku)
!konus                               # sohbet modu
!dogru  /  !yanlis                   # zihnin son cevabının nasıl karşılandığı
!yas   !neden <söz> => <kelime>   !unut <no>   !kaydet <ad>
!gor ben.jpg   !dinle sesim.wav      # bir sonraki sözle birlikte bir resim göstermek / bir ses duyurmak
!dusun [konu]   !uyu   !kendin       # iç sesini dinlemek / rüya görüp hayallerini ayıklamak / iç gözlem
```

## Yapay zeka motorunu kitap olarak okuyun

**[Cevahir AI — Bir Yapay Zeka Motorunun Anatomisi: İçindekiler](docs/book/README.md)**

Modüler kitap, yapay zekanın temellerini gerçek Cevahir koduna bağlar: tokenizer, sinir ağı, Transformer, eğitim, checkpoint, üretim, bellek, bilişsel iş akışları, araçlar ve açık araştırma. Her bölümde çağıran bileşen, girdi/çıktı, yapılandırma ve kanıt zinciri izlenir. [English reading guide](docs/book/README-en.md) · [Kaynak haritası](docs/book/KAYNAK_HARITASI.md) · [Kodla kitabı birlikte güncel tutmak](docs/book/BAKIM.md).

[Araştırma laboratuvarı](docs/book/tr/11-arastirma-laboratuvari.md), başarılı ve olumsuz deneyleri birlikte korur. **[Araştırma yayın dizisi](docs/publications/README.md)**, Muhammed Yasin Yılmaz imzalı dokuz araştırma raporunu özetleri, yöntemleri, kanıtları ve sınırlarıyla sunar. [Son sorgu-durumu deneyi](docs/research/living_learning_query_state_2026_09_21/REPORT_TR.md), gelecekteki sorgu ve düzeltmeler için gereken bilgiyi ayırır. Bu sınırlı sonuçlar genel yaşarken öğrenme probleminin çözümü değildir. [Sürüm ve DOI kaydı](docs/publications/release.json).

<p align="center">
  <img src="image/87E09A64-4E1F-41D5-84AF-7D7C56F6C229.png" alt="Cevahir AI & Engine" style="max-width:100%;">
</p>

## Cevahir'de neyi bir araya getiriyorum?

Cevahir'de en çok önemsediğim şey, **dil modelinin temsilini, öğrenmesini ve kullanımını aynı geliştirilebilir sistemde birleştirmek.** Bir metnin nasıl tokenlara ayrıldığını, modelin bu tokenlardan nasıl öğrendiğini, nasıl cevap ürettiğini ve geçmişi ya da araç sonuçlarını nasıl kullandığını birlikte inceleyebilmek istiyorum. Bu nedenle akışın her katmanını kaynak kodunda izlenebilir ve değiştirilebilir biçimde geliştiriyorum.

**Dil temsili üzerinde çalışma alanı.** Türkçeye özel geliştirilmiş BPE altyapısı; normalizasyonu, sözlüğü, birleşim kurallarını, isteğe bağlı heceleme ve morfoloji bileşenlerini veri hazırlama ile buluşturur. Böylece dilin nasıl temsil edildiği üzerine yapılan çalışma, model eğitimine aynı token kimlikleriyle taşınabilir.

**Mimari tercihleri eğitilebilir modele dönüştüren altyapı.** Attention başlık düzeni, konum kodlaması, normalizasyon, residual yapısı ve yoğun/uzman tabanlı FFN seçenekleri model kurulumuna bağlanır. Eğitim servisi bu modeli veriyle buluşturur; eğitim yöneticisi kaybı, gradyanları ve optimizasyonu yürütür. Checkpoint ve tokenizer kimliği, geliştirilen modeli daha sonra aynı anlamla kullanmanın temelini oluşturur.

**Öğrenilen modeli kullanan bilişsel katman.** Cevahir, modelin üretim ve aday puanlama yeteneklerini strateji seçimi, çoklu aday üretimi, araç kullanımı, bellekten bilgi getirme ve critic revizyonuyla birleştirir. Bu düzen, ağırlıkları yeniden eğitmeden modelin yanıt oluşturma sürecini araştırmaya ve geliştirmeye alan açar.

**İncelenebilir bir sistemden konuşma uygulamasına geçiş.** Model profilleme, gradyan/ağırlık sağlık kontrolleri ve bilişsel izler sistemin iç davranışını görünür kılar. Birleşik `Cevahir` arayüzü, konuşma yöneticisi ve uygulama servisleri aynı motoru kullanıcı ve oturum geçmişi taşıyan uygulamalara bağlar.

Bu altyapı üzerinde Türkçe dil modelleme çalışmalarımı sürdürüyor, farklı model ve bilişsel sistem fikirlerini araştırıyorum. Parçaların birlikte nasıl çalıştığını [sistem bütünlüğü rehberinde](docs/architecture/SYSTEM_OVERVIEW.md), uygulama sözleşmelerini [mimari belgede](docs/architecture/CEVAHIR_ARCHITECTURE_SPEC.md) anlattım.

## Sistem mimarisi

```mermaid
flowchart TD
    Sources[Metin, belge ve soru-cevap verisi] --> Tokens[TokenizerCore / BPE]
    Tokens --> Cache[Hazırlanmış eğitim verisi]
    Cache --> Training[TrainingService / TrainingManager]
    Training --> Core[CevahirNeuralNetwork]
    Core --> Checkpoint[Ağırlıklar, mimari ve tokenizer kimliği]
    Checkpoint --> Manager[ModelManager]
    Manager --> Generation[Üretim ve aday puanlama]
    Tokens --> Generation
    User[Kullanıcı mesajı] --> Chat[Cevahir / ChattingManager]
    Chat --> Cognitive[Bilişsel strateji ve yanıt işleme]
    Cognitive <--> Generation
    Cognitive <--> Memory[Konuşma belleği ve retrieval]
    Cognitive --> Tools[Kayıtlı araçlar]
    Tools --> Cognitive
    Chat <--> Services[HTTP servisleri ve kalıcı oturumlar]
```

Ana model yolu `Cevahir → ModelManager → CevahirNeuralNetwork` şeklindedir. Eğitim başlatıcısı, kullanılabildiğinde V3 eğitim servisini seçer; `training_backend="v2"` ile etkin **V2 TrainingManager** üzerinden eğitim yapar. V2 servis yolu da korunur. Eski yorumlardaki V4, V7 ve V8 adları geliştirme geçmişini anlatır. Güncel model, yetenekleri ayarlar üzerinden seçilen yapılandırılabilir bir çekirdektir.

### Sinir ağı çekirdeği

[Decoder](src/neural_network.py); embedding, üst üste Transformer katmanları, çıkış normalizasyonu ve sözlük projeksiyonunu birleştirir. Yapılandırılabilir bileşenleri şunlardır:

- **MHA, MQA ve GQA:** Sorgu başlıkları ile key/value başlıklarının sayısı ayrı belirlenebilir.
- **Konum ve dikkat:** RoPE, linear/YaRN ölçekleme seçenekleri, nedensel dikkat, kayan pencere maskeleri, QK normalizasyonu ve dikkat/çıkış logitleri için soft-cap.
- **İleri beslemeli katmanlar:** SwiGLU/GELU ile yoğun FFN veya top-k Mixture of Experts; router yardımcı kaybı eğitim hedefine katılır.
- **Katman yapısı:** RMSNorm/LayerNorm, pre/post normalizasyon, paralel residual, ağırlık paylaşımı ve stochastic depth.
- **Bellek ve yürütme:** Artımlı KV önbelleği, sink token'larla sınırlı önbellek yönetimi, PyTorch SDPA, isteğe bağlı harici Flash Attention ve gradient checkpointing.

Normal ileri geçişte dikkat ağırlıklarını ayrıca üretmeden SDPA kullanılabilir. `return_attention_weights=True` inceleme için dikkat ağırlıklarını; `collect_diagnostics=True` ayrıntılı tensör istatistiklerini ister. Normal çağrı `(logits, attention_or_none)` döndürür; önbellekli çekirdek çağrısında üçüncü bir cache değeri bulunur. Harici Flash Attention ve uzun bağlam ayarları hedef donanıma bağlıdır ve kendi değerlendirmelerini gerektirir.

### Tokenizer, veri ve eğitim

[TokenizerCore](tokenizer_management/core/tokenizer_core.py), **Türkçeye özel geliştirilmiş tokenizer altyapısının** ortak girişidir. BPE eğitimi, sözlük/merges yönetimi, kodlama ve çözme aynı bileşenlerle yürütülür. Türkçe `I/İ` dönüşümleri, heceleme, kök/ek üzerine kural tabanlı morfoloji yardımcıları ve özel token yönetimi bu altyapının parçalarıdır. Heceleme ve morfoloji ayara bağlıdır; her çıkarım çağrısında zorunlu olarak uygulanmaz.

Bir tokenizer'ın Türkçe için geliştirilmesi, başka bir dilde kullanılması için mutlaka değiştirilmesini gerektirmez. Aynı sözlük ve birleşim kuralları, kapsadıkları metinler üzerinde farklı diller için kullanılabilir. Tokenizer metni temsil eder; dil modelinin o dilde öğrenmesi, eğitim verisi ve ağırlıklarıyla ilgilidir. Cevahir'deki mevcut karakter filtreleri ve kapsam denetimi [tokenizer rehberinde](docs/modules/tokenizer_management/README.md) açıklanır.

[Veri yükleyici](data_loader_management/data_loader_manager.py), belgeleri ve soru-cevap kayıtlarını kaynak bilgisiyle eğitim hattına taşır. [Veri toplama/dönüştürme araçları](data_processing) ile [altyazı işleme](dataset_subtitle/subtitle_processor.py), eğitimde kullanılabilecek metinlerin hazırlanmasına yardımcı olur; çıktıları seçilen veri dizini üzerinden yüklenir.

[Veri hazırlama](training_system/prepare_cache.py), desteklenen TXT, DOCX ve soru-cevap JSON verilerini okuyup token ID'lerinden girdi/hedef kayıtları üretir. Hazırlanan veri epoch'lar boyunca tekrar kullanılabilir. Önbellek kimliği kaynak içeriğini, sözlüğü, birleşim kurallarını ve kodlama ayarlarını kapsar; V3 tüketimi ayrıca önbellek bütünlüğünü kontrol eder.

Eğitim sistemi; kaynağa göre eğitim/doğrulama ayrımını, birebir tekrar kayıtlarının gruplanmasını, uzunluğa göre batch oluşturmayı, dinamik padding'i, gradyan biriktirmeyi, sayısal hassasiyet seçimini, optimizer/scheduler bağlantısını, doğrulamayı ve checkpoint rotasyonunu içerir. Etkin eğitim yöneticisi epoch sınırından devam etmek için optimizer, scheduler, scaler, rastgele sayı üreteci ve döngü durumunu kaydeder. MoE yardımcı kaybı optimizasyonda gerçekten kullanılır.

### Üretim, bilişsel sistem ve konuşmalar

[Birleşik arayüz](model/cevahir.py); kodlama, çözme, metin üretimi ve bilişsel işlem çağrıları sunar. Üretim; sıcaklıkla örnekleme, greedy çözümleme, top-k/top-p, tekrar kontrolü, EOS sınırları, beam search ve artımlı KV önbelleğini kapsar.

[Bilişsel sistem](cognitive_management); direct, think, debate ve Tree of Thoughts iş akışlarını; bağlam oluşturmayı, araç çalıştırmayı, critic aşamalarını, vektör bellek bağlantısını, yanıt önbelleğini ve izlemeyi sağlar. Bunlar eğitilmiş model, araçlar ve retrieval kurulumu üzerinden çalışan düzenleme bileşenleridir; yanıt kalitesi bu bileşenlerin birlikte kullanımına bağlıdır. Dahili calculator sınırlı aritmetik işlemleri yürütür; dış araçlar araç arayüzü üzerinden kaydedilebilir. Varsayılan olarak katman yalın (lean) profille çalışır: her mesaj bir cevap üretimi, en fazla bir critic revizyonu ve bir entropi ölçümü alır; think/debate/ToT adımları, self-consistency örneklemleri, anayasal eleştiri ve dış iddia doğrulama istekleri kapalıdır. `CognitiveManagerConfig.full()` bunların hepsini, en.wikipedia.org üzerindeki iddia sorgularıyla birlikte yeniden açar; sayaçlı sahte bir modelle bu, sade bir mesajda 2 yerine 16–18 üretim çağrısı (Tree of Thoughts yolunda yaklaşık 70 üretim ve 52 puanlama çağrısı) demektir. Model her anayasal ilkede ihlal bildirirse full profil her ilke için bir öneri çağrısı ve bir revizyon ekler: 15 çağrı daha, sade bir mesajda 31–33. Vektör bellek iki profilde de açıktır: isteğe bağlı `sentence-transformers` paketi kuruluysa kodlayıcı modeli, önbellekte yoksa ilk kullanımda Hugging Face'ten indirilir. Ölçülemeyen entropi artık sıfır belirsizlik gibi okunmaz: mesaj doğrudan (direct) yola gider, cevabın token bütçesi ve sıcaklığı entropi ölçeğinin ortasıyla hesaplanır.

Stratejilerin gerçek çalışma biçimi, senkron/asenkron farklar ve açık geliştirme sınırları [bilişsel modül rehberinde](docs/modules/cognitive_management/README.md) açıklanır.

[ChattingManager](chatting_management) konuşma geçmişini ve bağlamı yönetir. Kapsamlandırılmış bilişsel akışta bellek kayıtları, notlar ve özetler kullanıcı/oturumla ilişkilendirilir. [API servisleri](api) ve [veritabanı repository'leri](database), motoru kimlik doğrulanmış oturumlara, saklanan konuşmalara ve kullanıcı verilerine bağlar.

## Yaptığım eğitimlerden çıktılar

**Aşağıdaki ekran görüntüleri, Cevahir ile yaptığım gerçek model eğitimlerinden ve eğitim sırasında gerçekleştirdiğim üretim kontrollerinden alınmıştır.** Kullandığım prompt'ları, modelin ürettiği yanıtları ve eğitim sırasında aldığım çıktıları burada paylaşıyorum. Bunlar Cevahir'i geliştirirken yürüttüğüm eğitim çalışmalarının bir parçası.

<p align="center">
  <img src="image/1.jpeg" alt="Cevahir gerçek eğitim çıktısı 1" style="max-width:100%;">
  <img src="image/2.jpeg" alt="Cevahir gerçek eğitim çıktısı 2" style="max-width:100%;">
  <img src="image/3.jpeg" alt="Cevahir gerçek eğitim çıktısı 3" style="max-width:100%;">
  <img src="image/4.jpeg" alt="Cevahir gerçek eğitim çıktısı 4" style="max-width:100%;">
  <img src="image/5.jpeg" alt="Cevahir gerçek eğitim çıktısı 5" style="max-width:100%;">
  <img src="image/6.jpeg" alt="Cevahir gerçek eğitim çıktısı 6" style="max-width:100%;">
</p>

Paylaştığım [eğitim verisi koleksiyonuna](https://drive.google.com/drive/folders/19G5uGS5YM3rf42OefjM3KsXRyn0ZEshW?usp=sharing) da ulaşabilirsiniz. Kendi çalışmanız için veri yolunu ve hazırlama ayarlarını belirleyin. Çıkarımda eğitilmiş checkpoint'i, o checkpoint'in eğitildiği sözlük, merges ve BPE ayarlarıyla birlikte kullanın.

## V4'ten sonraki gelişim

Çekirdeği V4'ten sonra da geliştirmeye devam ettim. Kaynak kodundaki sürüm notlarında aynı çekirdeğin üzerine eklenen şu yetenekleri görebilirsiniz:

| Kaynaktaki etiket | Güncel koddaki karşılığı |
|---|---|
| V4 | RMSNorm, SwiGLU, KV önbelleği, gelişmiş checkpointing, quantization ve MoE bağlantıları |
| V5 | GQA/MQA başlık düzeni, kayan pencere dikkati ve YaRN RoPE ölçeklemesi |
| V6 | PyTorch SDPA, QK-Norm, paralel residual, dikkat/çıkış logit soft-cap ve residual başlangıç ölçeklemesi |
| V7 | Stochastic depth, birleştirilmiş SwiGLU gate/up projeksiyonu ve sink token'lı KV yönetimi |
| V8 alt katman notları | KV kapasite sınırları ve MoE yardımcı kaybı gibi alt katman düzeltmeleri |

Bu etiketler ayrı ayrı kurulması gereken model paketleri değildir. Eğitim servisinin **V3**, etkin eğitim yöneticisinin **V2** ve bilişsel altyapının **V2** olması da modelin eski kaldığı anlamına gelmez; her biri farklı bileşenin gelişim hattıdır. `config_version=1` ve `architecture_version="cevahir-capabilities-1"` ise yapılandırma uyumluluğunu tanımlar. Ayrıntılı [çekirdek rehberi](docs/modules/neural_network/README.md) ve [mimari sözleşme](docs/architecture/CEVAHIR_ARCHITECTURE_SPEC.md) bu ayrımı açıklar.

## Sistemi incelemek ve geliştirmek

Modelin [profil araçları](model_management/profiler.py) parametre dağılımını, bellek kullanımını ve hesap maliyeti tahminlerini; [sağlık kontrolleri](model_management/health_monitor.py) gradyan, ağırlık ve dikkat istatistiklerini incelemek için kullanılır. Bilişsel katmanın trace ve metrik arayüzleri ise yanıtın hangi aşamalardan geçtiğini gösterir. Bu araçlar, mimari bir tercihin veya bilişsel adımın etkisini gözlemlemeyi sağlar.

Son altyapı çalışmalarımda bu parçaların birlikte çalışmasını güçlendirdim: ortak model yapılandırması, eğitim hedefine katılan MoE kaybı, artımlı dikkat önbelleği, tokenizer ile checkpoint kimliğinin korunması ve kullanıcı/oturum kapsamlı bellek. Teknik ayrıntıları [alt çekirdek sözleşmelerinde](docs/architecture/LOWER_CORE_CONTRACTS.md) ve [yaşam döngüsü belgesinde](docs/architecture/LIFECYCLE_CONSOLIDATION.md) bulabilirsiniz. Açık geliştirme işlerini [bir sonraki tur planında](docs/architecture/NEXT_DEVELOPMENT_ROADMAP.md) takip ediyorum.

### Güncel araştırma çalışmalarım

Ana soru, sistemin çalışırken yaşadığı deneyimlerle gelecekteki hesaplama ve davranış kapasitesini nasıl kalıcı, kontrollü ve genellenebilir biçimde değiştirebileceği. [Kitabın araştırma laboratuvarı](docs/book/tr/11-arastirma-laboratuvari.md), sonraki kural edinimi, temsil büyümesi, interference, durum taşıma, öğrenilmiş güncelleme ve yinelemeli durum araştırmalarını bir araya getiriyor. [Yeni düzeltme durumu deneyi](docs/research/living_learning_correction_2026_09_20/REPORT_TR.md), önceki deneyimleri düzelterek öğrenmeye devam edebilmek için neyin korunması gerektiğini inceliyor. Genel problem hâlâ açık; yeniden sınama ve hesap yönlendirmesi onun alt problemleri.

[Ağ içi paradigma çalışması (CEV-11)](docs/research/living_learning_paradigm_2026_11_13/REPORT_TR.md) bu soruyu Cevahir'in kendi kod çözücüsünün içine taşıdı. Uydurma bir Türkçe dünyada eğitilmiş 3,4 milyon parametrelik modelde, tek bir çekimli cümleden bir FFN izdüşümüne yapılan ROME tarzı sıra-bir yazım, varlığın kesme işaretli diğer hâl biçimlerine, aynı biçimlerin kesme işaretsiz yazımına göre 24,3 yüzde puanı daha sık ulaştı. Fark, kök ile ekin tek tokende kaynaştığı biçimlerde toplandı; yazım, son kök tokenini paylaşan adlara da sızdı. Önceden kayıtlı üç birincil hipotezin üçü de sekiz doğrulayıcı dünyanın hepsinde desteklendi; ama bu sonuçlar, keşifte iki sızıntı kısıtını da sağlamayan bir yazım yapılandırmasına aittir. **İkincil hipotezlerin hiçbiri desteklenmedi:** biçim-ortalamalı anahtarlar yalnız 3,1 puan kazandırdı (çürütüldü), önek ortalaması karşılaştırması belirsiz kaldı, kök-son tokenindeki anahtarın komşu adlara benzerliğinin derinlikle düşeceği öngörüsü çürütüldü ve ömür boyu defter hipotezi koşulmadı (keşif yaşamları önceden kayıtlı her λ değerinde çökmüştü; doğrulayıcı yaşamlar, işlemci zamanı başka bir programa ayrılınca ertelendi); doğrulayıcı bağlanma yeri haritası ve nedensel izleme de koşulmadı. Keşifte iki sızıntı kısıtını da yalnız ek tokenine yazım ile ince ayar kolları sağladı; ince ayar gündelik biçimlere ad konumuna yapılan yazımlardan daha sık ulaştı (sınanmış bir hipotez değil, betimsel bir gözlem). Model küçük, dünya sentetik ve kalıplıdır; tek tokenleştirici kullanıldı ve her düzenleme tek başına uygulandı.

Cevahir'in bir isteğe ne kadar hesap ayırdığını, dışarıdan doğrulanmış geri bildirimleri nasıl kullandığını ve bilişsel bağlamın sinir ağına nasıl taşındığını deneyebilmek için isteğe bağlı bir araştırma altyapısı ekledim:

- **Ortak istek bütçesi:** Üretim, aday puanlama ve entropi çağrıları aynı sınırları paylaşır; ana yanıt için kaynak ayrılır. Token muhasebesi, ayrılan üst sınırları izler; gerçek token veya FLOP ölçümü değildir.
- **Geri bildirime dayalı strateji tercihi:** Kayıtları kullanıcı/oturum, model ve tokenizer kimliğine göre ayırıyorum. Yeterli dış doğrulama varsa strateji tercihi değişebilir; değerlendirmeler düzeltilebilir, kayıtlar silinebilir ve deneyim ayrı bir dosyada saklanabilir.
- **MoE için açık bağlam girdisi:** Uzman yönlendirmesine sınırlandırılmış bir öncelik tensörü ekledim. Mevcut bağlantı yapılandırılmış alan profillerini kullanır; deneyimden uzman anlamı veya önceliği öğrenmeyi henüz uygulamadım.

Bu araştırma davranışları varsayılan olarak kapalıdır (`research.mode="off"`); MoE önceliği ayrıca etkinleştirilir. Küçük CPU testleri ve sentetik tablo karşılaştırmalarıyla uygulamanın davranışını kontrol ettim. Gerçek görevlerde kalite, hesap tasarrufu ve aktarım kazancını henüz ölçmedim. Mekanizmaları ve deney düzenini [araştırma kaydında](docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md) paylaşıyorum.

Mevcut mimarinin dışına çıkan yetenek sorularını da araştırıyorum. Son keşifte, sistemin kullandığı temsili karşı örneklerle değiştirmesi fikrini inceledim. Küçük karşılaştırmada tanıklarla filtrelenen arama, aynı adayları tamamen inceleyen aramayla aynı temsilleri ve tahminleri buldu. Bazı durumlarda daha az hesap yaptı; bu biçimiyle daha az gözlemle öğrenme iddiasını eledim. [Keşif gerekçelerini](docs/research/FRONTIER_DISCOVERY_2026_09_20.md) ve [tekrar çalıştırılabilir deneyin sonucunu](docs/research/REPRESENTATION_WITNESS_AUDIT.md) birlikte yayımlıyorum.

## Başlangıç

Komutları repo kökünde çalıştırın. Hafif doğrulamalar **Windows, Python 3.14.3 ve PyTorch 2.10.0+cpu** ile **Linux, Python 3.13 ve PyTorch 2.15 (CPU)** ortamlarında kaydedildi. Bağımlılıklar kök düzeyindeki manifestlerde gruplanmıştır: [requirements.txt](requirements.txt) (model, tokenizer, eğitim), [requirements-dev.txt](requirements-dev.txt) (HTTP sözleşme testleri dahil testler), [requirements-api.txt](requirements-api.txt) (HTTP servisleri ve veritabanı) ve [requirements-optional.txt](requirements-optional.txt) (izleme, vektör bellek, veri toplama).

Aşağıdaki küçük çekirdek örneği için ortam oluşturup temel bağımlılıkları kurabilirsiniz. PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
python -m pip install -r requirements.txt
```

Linux/macOS'ta etkinleştirme komutu `source .venv/bin/activate` şeklindedir. Manifestler alt sürüm sınırları verir; sürümleri sabitlenmiş bir ortam değildir. İsteğe bağlı alt sistemler, paketleri yoksa bir uyarıyla kendilerini devre dışı bırakır.

### CPU'da küçük bir model çalıştırma

Bu örnek küçük bir model kurar ve tek ileri geçiş yapar. Sentetik token ID'leri kullanır; veri seti veya checkpoint gerektirmez. Çekirdeği incelemek için hafif bir başlangıçtır.

```python
import torch
from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager

torch.set_num_threads(1)
torch.manual_seed(42)
manager = ModelManager(tiny_model_config())
manager.initialize(
    build_optimizer=False, build_criterion=False, build_scheduler=False
)
logits, attention = manager.forward(
    torch.tensor([[1, 5, 8, 2]]), inference=True
)
print(logits.shape)  # torch.Size([1, 4, 128])
print(torch.isfinite(logits).all().item())  # True
```

### Eğitilmiş modelinizi yükleme

Eğitimde kullanılan tokenizer dosyalarını ve BPE ayarlarını kullanın. Yeni checkpoint'ler model kurulum bilgisini taşır; bu nedenle modeli henüz kurmamış bir `ModelManager`, kayıtlı ayarlardan modeli oluşturabilir:

```python
from tokenizer_management.core.tokenizer_core import TokenizerCore
from model_management.model_manager import ModelManager

tokenizer = TokenizerCore({
    "vocab_path": "data/vocab_lib/vocab.json",
    "merges_path": "data/merges_lib/merges.txt",
    "use_gpu": False,
    # Eğitimde özel BPE ayarları kullanıldıysa bpe_config'i burada verin.
})
manager = ModelManager({"device": "cpu"}, tokenizer=tokenizer)
manager.load("saved_models/checkpoints/last.pth", weights_only=True)
manager.eval_mode()
```

Checkpoint yolunu kendi dosyanızla değiştirin. Kurulum bilgisi içermeyen ham/eski ağırlık dosyaları için eşleşen model ayarlarını açıkça vermek gerekir. Tokenizer kimliği olmayan eski kayıtlar, tokenizer verildiğinde uyarı üretir; token anlamları otomatik olarak doğrulanamaz.

Tam arayüzde ayarlar `CevahirConfig(model={...}, tokenizer={...}, load_model_path=...)` üzerinden verilir. `Cevahir` modeli yüklemeden önce kurduğu için mimari ayarlarının checkpoint ile eşleşmesi gerekir. `load_model_path=None` varsayılan yolda otomatik aramayı etkinleştirir; `load_model_path=""` açıkça yeni bir model başlatır. Seçilmiş checkpoint yoksa veya uyumsuzsa hata verilir. Terminal sohbet girişi [chat_pipeline.py](model_management/chat_pipeline.py) dosyasıdır; onun ayarları da eğitilmiş modelle eşleşmelidir.

## Kendi verinizle eğitim

1. **Tokenizer'ı seçin.** Mevcut model için onun sözlük/merges dosyalarını kullanın. Yeni tokenizer oluşturuyorsanız [tokenizer yapılandırmasında](tokenizer_management/config.py) yolları ve eğitim seçeneklerini belirleyip çalıştırın:

   ```powershell
   python tokenizer_management/train_bpe.py education
   ```

   Tokenizer'ı yeniden eğitmek token ID'lerini değiştirebilir; mevcut model ağırlıkları ve hazırlanmış önbellekler kendi tokenizer'larıyla birlikte kalmalıdır.

2. **Önbelleği hazırlayın.** Tokenizer seçenekleri ve `max_seq_length` eğitim yapılandırmanızla eşleşsin:

   ```powershell
   python training_system/prepare_cache.py --data-dir education --no-clear-cache
   ```

   `--no-clear-cache` mevcut önbellek dosyalarını korur; aksi hâlde script eski önbellekleri temizler. Dizi uzunluğu ve kodlama seçenekleri için `--help` kullanın. V3 eğitim, uyumlu hazırlanmış önbellek gerektirir.

3. **Eğitimi yapılandırıp başlatın.** [training_system/train.py](training_system/train.py) içindeki `TRAIN_CONFIG` üzerinden yolları, model boyutlarını, optimizer'ı, batch boyutunu, epoch sayısını ve cihazı belirleyin:

   ```powershell
   python training_system/train.py
   ```

   Repodaki hazır ayarlar 100 epoch, 64 batch boyutu ve sekiz katmanlı, 512 boyutlu model içeren kapsamlı bir eğitim koşusuna yöneliktir. Başlatmadan önce donanımınıza göre ayarlayın. Hafif başlangıç için yukarıdaki CPU örneğini kullanın.

Mimari ayarları [config_schema.py](model_management/config_schema.py) tarafından normalleştirilir. Sisteme verdiğiniz yapılandırmayı değiştirin; `model/cevahir.py` içindeki varsayılan sınıf tanımlarını düzenlemek gerekmez. Önbellek hazırlama ve eğitimin giriş ayarları hâlâ ayrı yerlerdedir; bunların birbiriyle eşleşmesi gerekir.

## Uygulama bağlantısı

Birleşik Flask giriş noktası [api.app_factory.create_app](api/app_factory.py) işlevidir. Modeli, ChattingManager'ı, servisleri, kimlik doğrulamayı, sağlık kontrollerini ve veritabanı erişimini bağlar. Uygulamayı başlatmadan önce veritabanını, açık bir `JWT_SECRET_KEY` değerini, model yolunu ve eşleşen tokenizer'ı yapılandırın. Model ayarları factory yapılandırmasındaki `CEVAHIR_MODEL_CONFIG` üzerinden verilebilir. Eski `api/app.py` kaldırılmış bir yapılandırma import'u taşır; önerilen giriş noktası değildir.

İstekler bir kullanıcıya yalnız doğrulanmış bir `Authorization: Bearer <JWT>` başlığıyla bağlanır. API token'ları doğrular ama üretmez: giriş, kayıt veya token uç noktası yoktur; token'ları kendi kodunuz aynı `JWT_SECRET_KEY` ile `api.security.jwt.create_access_token` kullanarak üretmelidir. [api/api_config.py](api/api_config.py) modu `FLASK_ENV` değerinden seçer; başka bir değer hata verir:

- **Tanımsız veya `default`:** debug kapalı, hız sınırı açık, CORS yalnız `http(s)://localhost`, `127.0.0.1` ve `[::1]` kökenleri için (her port).
- **`development`:** yerel çalışma için Flask'ın `DEBUG` ayarı açık, hız sınırı kapalı; CORS varsayılandaki gibi. Buna `CEVAHIR_TRUST_X_USER_ID=1` eklenirse kimliği doğrulanmamış `X-User-ID` başlığı çağıranın kimliği sayılır ve her kullanımda uyarı loglanır. Bu durumda herhangi bir çağıran herhangi bir kullanıcı gibi davranabilir; bu yüzden ayar yalnız `development` ve `testing` modlarında kabul edilir, diğer modlarda yapılandırma yüklenmez.
- **`testing`:** test koşuları için: `DEBUG` açık, hız sınırı ve metrikler kapalı; `CEVAHIR_TRUST_X_USER_ID=1`, `development` modundaki gibi kabul edilir.
- **`production`:** `PRODUCTION_SECRET_KEY` ve `JWT_SECRET_KEY` en az 32 karakterlik, birbirinden farklı iki değer değilse yüklenmeyi reddeder; `change-me-in-production` gibi yer tutucular kabul edilmez. Bir değerin rastgele olup olmadığı denetlenemez; ikisini de örneğin `python -c "import secrets; print(secrets.token_urlsafe(48))"` ile üretin. Hız sınırı her zaman açıktır; tarayıcılar farklı kökenden yalnız `CORS_ORIGINS` içinde listelenen kökenler için erişebilir.

Virgülle ayrılmış `CORS_ORIGINS` varsayılan kökenlerin yerine geçer; her girdi ya birebir bir kökendir (IPv6 adresi köşeli parantez içinde yazılır, örneğin `http://[::1]:3000`) ya da `Origin` değerinin tamamıyla eşleşmesi gereken bir düzenli ifadedir. Eşleşmeyen bir `Origin` CORS başlığı almaz, `*` ve `.*` gibi her kökenle eşleşen ifadeler düz `*` olarak gönderilir ve kimlik bilgilerine (credentials) hiçbir zaman izin verilmez. Hız sınırları, `RATELIMIT_STORAGE_URI` paylaşılan bir depo göstermedikçe süreç belleğinde tutulur. Kimlik doğrulaması istemeyen `/api/v3/health/detailed` yalnız genel bileşen mesajları döndürür; hata ayrıntıları sunucu loguna yazılır. `python api/app_factory.py`, Flask geliştirme sunucusunu `development` dahil her modda debug kapalı olarak `127.0.0.1:5000` üzerinde başlatır; `--host`/`CEVAHIR_API_HOST`, `--port`/`PORT` ve `--debug`/`CEVAHIR_API_DEBUG=1` bunu değiştirir, production modunda debug reddedilir. Gerçek dağıtımlar yine de API'nin önünde bir üretim WSGI sunucusu ve TLS gerektirir. Bu davranışlar [tests/evolution/test_api_security.py](tests/evolution/test_api_security.py) sözleşme testleriyle denetlenir; dağıtılmış bir serviste veya sızma testiyle denetlenmemiştir.

## Repo haritası

| Dizin | Sorumluluk |
|---|---|
| [src/](src) | Sinir ağı decoder'ı, dikkat, FFN/MoE, normalizasyon ve KV önbelleği |
| [tokenizer_management/](tokenizer_management) | BPE eğitimi, kodlama/çözme, sözlük ve birleşim yönetimi |
| [data_loader_management/](data_loader_management) | Belge ve soru-cevap okuma, parçalama ve kaynak kimliği |
| [data_processing/](data_processing), [dataset_subtitle/](dataset_subtitle) | Veri toplama, belge dönüştürme ve altyazılardan metin hazırlama araçları |
| [training_system/](training_system) | Veri hazırlama, önbellek, batching ve eğitim servisi |
| [training_management/](training_management) | Eğitim döngüsü, optimizasyon, izleme ve checkpoint'ler |
| [model_management/](model_management) | Ayarlar, kurulum, yükleme, kayıt, profilleme ve çıkarım |
| [model/](model) | Birleşik Cevahir arayüzü ve üretim adaptörü |
| [cognitive_management/](cognitive_management) | Stratejiler, araçlar, critic, kapsamlandırılmış bellek ve middleware |
| [chatting_management/](chatting_management) | Oturumlar, konuşma geçmişi ve bağlam |
| [api/](api), [database/](database) | HTTP servisleri, kimlik doğrulama ve kalıcı saklama |
| [research/](research) | Ana çalışma akışından bağımsız, küçük araştırma ve yanlışlama deneyleri |
| [benchmarks/](benchmarks), [tests/](tests), [scripts/](scripts) | Ölçümler, davranış doğrulaması ve ayrı teşhis araçları |
| [docs/](docs) | Mimari, modül rehberleri ve geliştirme geçmişi |

## Doğrulama ve geliştirme durumu

Yaptığım model eğitimlerinden örnekleri yukarıda paylaştım. Güncel mühendislik kontrollerinde ayrıca küçük CPU modelleriyle önbellekli/tam dizi sonuçlarını, gradyanları, kayıt/yükleme davranışını, tokenizer kimliğini ve durum izolasyonunu karşılaştırıyorum. Bu kontrollerin sentetik kayıp veya süre ölçümleri, eğittiğim modelin dil kalitesini ölçmez.

Küçük bir yaşam döngüsü doğrulama grubu ve ayrı çekirdek ölçümü için:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/evolution/test_lower_layer_contracts.py tests/evolution/test_checkpoint_lifecycle.py -q
python -m pytest
python benchmarks/core.py --label local --output benchmarks/results/core_local.json
```

Daha geniş doğrulama yollarını [benchmark rehberinde](benchmarks/README.md) bulabilirsiniz. `python -m pytest`, `tests/` paketini çalıştırır: 13 Kasım 2026'da 529 geçen, 11 atlanan ve 1 beklenen başarısızlık verdi (Linux, Python 3.13, CPU, yaklaşık 4 dakika). Atlanan testler repoda bulunmayan eğitilmiş bir checkpoint veya üretilmiş eğitim cache'i ister; beklenen başarısızlık bilinen bir checkpoint yükleme hatasını sabitler. GitHub Actions bu paketi her push'ta çalıştırır ve bir test çalışma ağacına dosya bırakırsa başarısız olur. Paketlerin içindeki eski test klasörleri (örneğin `src/neural_network_module/test`) kaldırılmış modüllerin testlerini hâlâ içerir ve bu çalıştırmaya dahil değildir. Kayıpsız Unicode tokenizasyonunun ana eğitim yoluna taşınması (kayıpsız bayt düzeyinde BPE arka ucu artık [tokenizer_management/byte_bpe.py](tokenizer_management/byte_bpe.py) içinde, mevcut tokenizer'dan ayrı olarak bulunuyor), tüm eğitim kayıtlarının hizalama kontrolü, yakın tekrar verilerin ayrılması, eşzamanlı model değiştirme/üretim ve retrieval/critic/ToT kalite değerlendirmesi üzerinde açık işlerim var. GPU çalışmasını, dağıtık eğitimi, uzun bağlam kalitesini ve gerçek derleme performansını ayrıca doğrulamak gerekiyor; yardımcı modüllerin bulunması etkin eğitim yoluna bağlandıkları anlamına gelmez.

## Dokümantasyon, katkı ve iletişim

- [Modüllerin birlikte sunduğu sistem](docs/architecture/SYSTEM_OVERVIEW.md)
- [Mimari ve desteklenen davranışlar](docs/architecture/CEVAHIR_ARCHITECTURE_SPEC.md)
- [Sonraki büyük geliştirme turu: bulgular ve çalışma sırası](docs/architecture/NEXT_DEVELOPMENT_ROADMAP.md)
- [Alt çekirdek sözleşmeleri](docs/architecture/LOWER_CORE_CONTRACTS.md)
- [Checkpoint, veri ve çalışma yaşam döngüsü](docs/architecture/LIFECYCLE_CONSOLIDATION.md)
- [Geliştirme günlüğü](docs/development/EVOLUTION_LOG.md)
- [Modül dokümantasyonu](docs/modules)
- [Deneyimle koşullanan hesaplama araştırması](docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md)
- [Bağımsız yetenek keşfi](docs/research/FRONTIER_DISCOVERY_2026_09_20.md)
- [Temsil değişimi hipotezinin ilk yanlışlama deneyi](docs/research/REPRESENTATION_WITNESS_AUDIT.md)

Çekirdek, veri ve tokenizer hattı, model yaşam döngüsü, uygulama bağlantıları veya değerlendirmeler üzerinde birlikte çalışmak isteyenlerin katkılarını bekliyorum. Bir değişiklik önerirken etkilenen akışı, elde edilen davranışı ve nasıl doğruladığınızı paylaşmanız incelememi kolaylaştırır.

Projeyi [Apache License 2.0](LICENSE) ile paylaşıyorum.

**Bana ulaşın:** [GitHub](https://github.com/myylogic) · [X](https://x.com/myylogic) · [Instagram](https://instagram.com/myylogic)

<p align="center">
  <img src="image/myy.jpeg" alt="Muhammed Yasin Yılmaz, Cevahir'in geliştiricisi" style="max-width:100%;">
</p>

*Bu belgeyi 20 Eylül 2026 tarihinde güncel kaynak koduyla eşleştirdim.*
