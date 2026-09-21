# 3. Sayılardan öğrenilebilir bir hesaba: sinir ağı

[Kitabın içindekileri](../README.md) · [Önceki: Metin ve tokenizer](02-metin-tokenizer.md) · [Sonraki: Attention](04-attention.md)

Tokenizer metni tamsayılara dönüştürdü. Fakat `42` numaralı token, `21` numaralı tokenın iki katı anlama gelmez. Kimlikler bir sözlüğün adresleridir. Sinir ağının ilk işi bu adreslerden hesap yapılabilecek vektörler seçmektir. Son işi ise bağlamdan çıkardığı vektörü, sözlükteki her olası sonraki token için bir skora dönüştürmektir. Bu bölüm iki uç arasındaki öğrenilebilir hesabı kurar; dikkat mekanizması ve Transformer bloğunun içi sonraki iki bölümde açılır.

## Sinir ağı nedir, hangi anlamda öğrenir?

Bir yapay sinir ağı, parametrelerle belirlenen ve birbirine bağlı sayısal dönüşümlerden oluşan bir fonksiyon ailesidir. Girdisi $x$, parametreleri $\theta$ olan hesabı $f_\theta(x)$ ile gösteririz. Mimari hangi işlemlerin bağlandığını; parametreler o işlemlerin sayısal değerlerini; eğitim amacı hangi sonuçların tercih edildiğini belirler. Aynı mimari farklı parametrelerle farklı fonksiyonlar hesaplar. Bu tanım biyolojik hücrelerin kimyasını veya insanın düşünme biçimini birebir taklit etme iddiası taşımaz.

Örneğin $\mathcal D=\{(x_i,y_i)\}_{i=1}^N$ için ampirik risk $\hat R(\theta)=N^{-1}\sum_i\ell(f_\theta(x_i),y_i)$ olabilir. Eğitim verisindeki riski küçültmek ile görülmemiş deneyimlerde başarılı olmak farklıdır. Bir ağ örnekleri ezberleyebilir; parametre değişmesi tek başına genelleme kanıtı değildir. Cevahir'de model hesabı, loss hesabı ve optimizer adımının farklı dosyalarda bulunması bu ayrımı somutlaştırır.

### Nöron benzetmesinden hesap grafiğine

Tek skaler birim $a=\phi(\sum_j w_jx_j+b)$ ile gösterilebilir. $w_j$ bağlantı katsayıları, $b$ bias, $\phi$ aktivasyon fonksiyonudur. Birimleri matris biçiminde birleştirmek aynı hesabı paralel yürütmeyi sağlar. Modern Transformer'da matris çarpımı, softmax, norm, çarpımsal kapı ve residual toplama birlikte kullanılır. Bu işlemleri bir biyolojik benzetmeye zorlamak yerine bağımlılık grafiğini izlemek daha açıklayıcıdır.

Gizli katman, girdilerle çıktı arasındaki ara temsildir; gizli olması erişilemez olduğu anlamına gelmez. Alt modüllere forward hook bağlayarak şekilleri gözlemleyebiliriz. Fakat bir aktivasyon bileşenine isim vermek, onun her girdide aynı insan kavramını temsil ettiğini kanıtlamaz. Ölçülen vektörler ile onlara yüklenen yorum ayrı tutulmalıdır.

### Neden yalnız doğrusal katmanları çoğaltmak yetmez?

İki affine dönüşümün bileşkesi yine affine'dir: $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$. Araya doğrusal olmayan işlem koymadan katman sayısını artırmak yeni tür karar sınırları oluşturmaz. Bunu XOR üzerinde doğrudan görebiliriz. $(0,0)$ ve $(1,1)$ negatif; $(1,0)$ ve $(0,1)$ pozitif olsun. Pozitif sınıfı $w_1x_1+w_2x_2+b>0$ ile ayırdığımızı varsayalım. Pozitif iki örnek $w_1+w_2+2b>0$ gerektirir. Negatifler ise $b\leq0$ ve $w_1+w_2+b\leq0$, dolayısıyla $w_1+w_2+2b\leq0$ verir. Çelişki vardır.

İki ReLU birimiyle $h_1=\max(0,x_1-x_2)$, $h_2=\max(0,x_2-x_1)$ ve $\hat y=h_1+h_2$ dört ikili girdide XOR'u tam hesaplar. Bu elle kurulmuş küçük ağ doğrusal olmayan temsilin işlevini gösterir; bir optimizer'ın bu ağırlıkları her başlangıçtan bulacağını göstermez. Temsil kapasitesi, optimizasyon başarısı ve genelleme üç ayrı sorudur.

## Parametre, aktivasyon ve katman

Bir doğrusal katmanı satır vektörleriyle `y = x Wᵀ + b` diye yazabiliriz. `W` ve varsa `b` öğrenilebilir **parametrelerdir**. `x` ve `y`, o çağrıda hesaplanan **aktivasyonlardır**. Bir girdiye cevap vermek aktivasyonları değiştirir; tek başına parametrelerin öğrenildiği anlamına gelmez. Eğitim, kaybın parametrelere göre türevlerini hesaplayıp bir optimizer aracılığıyla parametreleri günceller.

Cevahir'de bu ayrım elle tutulan sayı listeleri yerine PyTorch'un `nn.Module`, `nn.Parameter`, `nn.Linear` ve `nn.Embedding` nesneleri üzerinden kuruludur. [CevahirNeuralNetwork](../../../src/neural_network.py#L85) bir `nn.Module` alt sınıfıdır. `self.layers` için `nn.ModuleList` kullanılması, alt katmanların parametrelerinin modele kaydolmasını sağlar. Katmanların peş peşe çağrılması ise aynı parametreyi tekrar kullanmak değildir: listedeki her Transformer katmanı ayrı bir nesnedir.

Doğrusal dönüşümleri yalnızca art arda koymak, araya başka işlem girmediğinde yine doğrusal bir dönüşüm verir. Cevahir'in hesap ailesini genişleten işlemler arasında attention içindeki softmax ve FFN içindeki SiLU/GELU vardır. Bunların kod karşılıkları sırasıyla [MultiHeadAttention._standard_sdpa_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L613) ve [FeedForwardNetwork.forward](../../../src/neural_network_module/ortak_katman_module/feed_forward_network.py#L294) içinde görülebilir. Her sayısal işlemi “nöron” benzetmesiyle anlatmak yerine, hangi dönüşümün hangi tensöre uygulandığını takip edeceğiz.

### Tensör şekli ve parametre sayısı

Bir tensör, indislerle erişilen çok boyutlu sayı dizisidir. `[B,T,D]` biçiminde `B` örneği, `T` konumu, `D` temsil bileşenini seçer. `nn.Linear(D,F)` son eksende aynı dönüşümü her örnek ve konuma uygular: $Y_{btf}=\sum_dX_{btd}W_{fd}+b_f$. Ağırlık şekli `[F,D]`, bias şekli `[F]`, parametre sayısı bias açıksa $FD+F$ olur. `B` veya `T` büyümek parametre sayısını değil, aktivasyon ve işlem miktarını artırır.

`reshape` ve eksen değiştirme kendi başına öğrenilmiş dönüşüm değildir. Aynı değerleri başka indis düzeninde okumakla yeni katsayılarla çarpmak farklı işlemlerdir. `nn.ModuleList` ve kayıtlı `nn.Parameter`, optimizer'ın erişeceği nesneleri belirler. `state_dict` ayrıca kalıcı buffer'lar içerebilir; bu sözlükteki her tensörün öğrenilebilir parametre olduğu varsayılmaz. Geçici cache, optimizer momentleri ve ağırlıklar da aynı tür durum değildir.

### Aktivasyonlar ve başlangıç

ReLU, $\max(0,x)$; sigmoid, $1/(1+e^{-x})$; SiLU, $x\sigma(x)$; GELU ise $x\Phi(x)$ biçimindedir. $\Phi$ standart normal dağılımın birikimli dağılım fonksiyonudur. ReLU sıfırın altında sıfır türev, sigmoid büyük mutlak değerlerde küçük türev verir. Bu davranışların katmanlar boyunca bileşimi optimizasyonu etkiler; aktivasyon adına bakarak bütün ağın gradyan kararlılığı çıkarılamaz. GELU'nun özgün tanımı [Hendrycks ve Gimpel, 2016](https://arxiv.org/abs/1606.08415) içindedir. Cevahir'in kapılı çeşitleri [Transformer bölümünde](05-transformer.md) gerçek modül üzerinden açılır.

Bir katmanın birimlerini aynı ağırlıklarla başlatmak, aynı girdiler ve güncelleme altında aynı kalmalarına yol açabilir. Rastgele başlangıç simetriyi kırmanın bir yoludur; tek başına anlam veya öğrenme değildir. Glorot–Bengio aktivasyon ve türev ölçeklerini, He ve arkadaşları doğrultuculara uygun başlangıcı inceler. Bu çalışmalardaki koşullar her Transformer bileşenine aynen taşınmaz. [Glorot ve Bengio, 2010](https://proceedings.mlr.press/v9/glorot10a.html); [He vd., 2015](https://arxiv.org/abs/1502.01852).

Cevahir'in embedding kurucusu başlangıçtan sonra satır normunu ayrıca değiştirdiği için yalnız `init_method="xavier"` adına bakarak son dağılımı yorumlamak yanlıştır. Başlangıç fonksiyonunun tamamı ve çağıranın sonradan yaptığı dönüşümler incelenir. FFN giriş ve residual çıkış ağırlıklarının ölçekleri de farklı seçilebilir. Bir bileşenin son durumu, seçenek adından daha fazla bilgi taşır.

## Embedding: kimlikten vektöre

`V` sözlük büyüklüğü, `D` temsil genişliği olsun. Embedding tablosu `E ∈ R^(V×D)` biçimindedir. Token kimliği `i` için ilk temsil `E[i]` satırıdır. Bu, sıralama veya kimlik numarasının büyüklüğü üzerinden anlam çıkarmaz; hangi satırın okunacağını belirler.

[LanguageEmbedding.__init__](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L56) tabloyu `nn.Embedding` ile kurar. [LanguageEmbedding.forward](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L234) `torch.long` giriş bekler; tabloya bakar, açıksa ölçekler, normalizasyon ve dropout uygular. Ana model bu modülü `scale_by_sqrt=False` ile kurar. `norm_type` verilmediği için embedding çıktısında LayerNorm bulunur. Bu, Transformer katmanlarında RMSNorm seçilmesiyle çelişmez: iki ayrı yerde iki ayrı işlem vardır.

Örneğin iki metnin dörder tokenı için giriş şekli `[2,4]`, embedding çıkışı `[2,4,D]` olur. Aynı token tablodan aynı satırı getirir. Ancak sonraki attention katmanları bağlama göre farklı temsiller üretir; tokenın başlangıç vektörü ile bağlam içindeki son vektörü birbirine karıştırılmamalıdır.

Tablonun başlangıcı da öğrenilmiş anlam değildir. [_initialize_weights](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L262) başlangıç dağılımını oluşturur, satırları normalize edip `0.15` ile ölçekler ve ayarlanmışsa padding satırını sıfırlar. Buradaki `0.15` bir satır ölçeğidir; her bileşenin standart sapmasının `0.15` olduğuna dair sonuç çıkarılamaz. Gerçek kelime ilişkilerinin edinilmesi eğitim verisi ve hedefe bağlıdır.

### One-hot ile aynı lookup hesabı

`i` kimliğinin one-hot satırı $e_i\in\{0,1\}^{V}$ yalnız i'nci bileşende birdir. Dolayısıyla $e_iE=E[i]$. Bu eşitlik embedding lookup'ın anlamını açıklar; uygulamanın dev bir one-hot tensör oluşturmasını gerektirmez. Tamsayı indislerle satır seçmek aynı sonuca daha az gereksiz ara veriyle ulaşır. Lookup sonrasında norm, ölçek veya dropout varsa toplam modül artık yalnız $E[i]$ değildir.

Çalıştırılmış öğretim örneğinde altı satırlı, üç sütunlu tabloya sırayla `0.0,0.1,...,1.7` değerleri atandı. `[1,2,1]` kimlikleri `[[0.3,0.4,0.5],[0.6,0.7,0.8],[0.3,0.4,0.5]]` değerlerini verdi. Gerçek `LanguageEmbedding` içinde norm ve dropout kapatıldığında one-hot çarpımıyla fark sıfır çıktı. Bunlar eğitimden öğrenilmiş kelime anlamları değil, eşitliği denetlemek için atanmış değerlerdir. [Çalıştırma kaydı](../evidence/neural_walkthrough.json).

### Dağıtık temsil, benzerlik ve bağlam

Vektör bileşenleri çoğu zaman elle adlandırılmış özellikler değildir; görev kaybı altında birlikte öğrenilirler. İki sıfır olmayan vektörün cosine benzerliği $u^Tv/(\|u\|\|v\|)$ ile hesaplanır. Yüksek cosine, seçilen uzayda yakın yönü gösterir. Bütün anlam ilişkilerinin bu tek sayıyla ölçüldüğünü göstermez. Normlar, veri dağılımı, görev ve sonraki dönüşümler önemlidir. Rastgele başlangıçta yakın iki vektöre dilsel anlam yüklenmez.

Başlangıç embedding'i token kimliğine bağlıdır. Bağlamsal temsil $h_t=f_\theta(x_{\leq t})$ gibi çevredeki izinli tokenlara da bağlı olabilir. Aynı token iki cümlede aynı satırdan başlar fakat farklı attention sonuçları alır. “Aynı token” ile “aynı sözcük” de eşanlamlı değildir: tokenizer bir sözcüğü birkaç parçaya ayırabilir. Sözcük, token, ilk embedding ve son bağlamsal vektör dört ayrı düzeydir.

Yerel küçük Cevahir örneğinde aynı `3` kimliği aynı konumda farklı önceki tokenlarla işlendi. Ham embedding satırları eşitti; o konumdaki logitlerin en büyük mutlak farkı yaklaşık `0.674958` oldu. Rastgele başlatılmış ağdaki bu fark dil anlayışının kanıtı değildir. Bağlamın hesap sonucuna gerçekten bağlandığını gösteren yapısal bir kontroldür.

### Temsil öğrenmenin literatürdeki yeri

Bengio ve arkadaşlarının sinirsel dil modeli, sözcük temsilleriyle dizi olasılığının birlikte öğrenilmesini ele alır. Mikolov ve arkadaşlarının word2vec çalışması büyük metinlerde sürekli sözcük temsillerini daha düşük hesap maliyetiyle edinmeye yönelik mimariler sunar. Cevahir'in tablosu bu nedenle önceden word2vec ile eğitilmiş sayılmaz; mevcut sınıf yeni tablo kurar, dil modeli eğitimi onun parametrelerini günceller. [Bengio vd., 2003](https://www.jmlr.org/papers/v3/bengio03a.html); [Mikolov vd., 2013](https://arxiv.org/abs/1301.3781).

BERT, Google araştırmacılarının çift yönlü bağlamsal temsilleri ön eğitimle edinen somut bir endüstriyel araştırma örneğidir. Hedefi ve erişebildiği bağlam Cevahir'in nedensel sonraki-token hattıyla aynı değildir. BERT adı bütün embedding yöntemlerinin veya bütün üretici dil modellerinin adı olarak kullanılamaz. [Devlin vd., 2019](https://aclanthology.org/N19-1423/).

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

### Softmax türevi neden öğrenme sinyali verir?

Tek hedef için $L=-z_y+\log\sum_j e^{z_j}$ yazılabilir. Türev $\partial L/\partial z_i=p_i-\mathbf1[i=y]$ olur. Doğru sınıf için $p_y-1$, diğerleri için $p_i$ elde edilir. Bu, doğru sınıf skorunu yükseltme ve rakipleri düşürme yönünü gösterir; parametreler aynı anda birçok skoru etkilediğinden tek bir skor bağımsız ayarlanıyormuş gibi düşünülemez.

Üç sınıfın logitleri sıfırsa olasılıklar `1/3`, doğru sınıf kaybı $\log3\approx1.098612$ olur. Hedef ikinci sınıfsa logit gradyanı `[1/3,-2/3,1/3]`'tür. Logitlerin hepsine aynı sabiti eklemek softmax'ı değiştirmez. Sayısal uygulamada en büyük logiti çıkararak üstel taşma riski azaltılabilir; olasılığı hesaplayıp sonra log almak yerine kararlı log-softmax/cross-entropy yordamı kullanılır.

Türev çıktıda uygulanmış soft-cap'ten önceki skora taşınırken cap'in türeviyle de çarpılır. Kaybın gözlendiği düğüm ile güncellenen parametre arasındaki işlemler önemlidir. Etiket yumuşatma, sınıf ağırlıkları ve maskeler varsa yalın formül buna göre değişir.

## Ağırlık paylaşımı neyi değiştirir?

Giriş tablosu ve çıkış projeksiyonu iki farklı görevde kullanılsa da boyutları uygunsa aynı parametreyi paylaşabilir. [Ana model kurucusundaki](../../../src/neural_network.py#L493) gerçek bağlama şöyledir:

```python
self.output_layer.weight = self.embedding.embedding.weight
```

Bu satır bir kopya üretmez; aynı parametre nesnesini iki kullanım yerine bağlar. Böylece ayrı iki `V×D` tablo yerine ortak tablo bulunur ve iki kullanımın türevleri aynı parametreye katkı verir. Mevcut çekirdek, paylaşım açıkken çıkış bias'ı oluşturmaz. Bu bir kapasite ve parametre paylaşımı tercihidir; kaynakta bulunmayan geçmiş bir tasarım kararının gerekçesini burada icat etmiyoruz.

Normal çekirdek çağrısı `(logits, attention_weights)` döndürür. Attention ağırlıkları varsayılan olarak `None` olur. Cache istendiğinde üçüncü değer eklenir; mevcut kodda bu son katmanın cache çıktısıdır, bütün katmanların listesi değildir. Her katmanın gerçek devam durumu kendi attention nesnesinde tutulur.

### Giriş satırı ve çıkış sınıfı farklı gradyan yollarıdır

Yalnız lookup kullanan bir kayıpta, indislenmeyen embedding satırları o kullanım yolundan gradyan almaz. Ağırlık paylaşımında tablo aynı zamanda bütün sözlük skorlarını üretir. Bu nedenle girdide görülmeyen bir tokenın satırı çıkış yolundan gradyan alabilir. “Embedding yalnız görülen tokenları öğrenir” cümlesi bağlı tam-sözlük çıkışı için eksiktir. PyTorch gradyanının yoğun veya seyrek bellekte saklanması da bu bağımlılık sorusundan ayrıdır.

Altı satırlı örnekte yalnız lookup toplamından gelen gradyanlar `[1,2]` satırlarında görüldü. Aynı tablo çıkışa bağlanıp çapraz entropi hesaplandığında altı satırın tamamında sıfır olmayan gradyan oluştu; girdide bulunmayan `5` de bunlardan biriydi. Amaç burada her batch'te her satırın mutlaka değiştiğini kanıtlamak değil, iki kullanımın farklı türev yollarını göstermektir. Parametre paylaşımının dil modelindeki etkileri için [Press ve Wolf, 2017](https://aclanthology.org/E17-2025/) özgün başvuru kaynağıdır.

## Geri yayılımı bir sayı zinciriyle görmek

### Yerel türevden parametre türevine

Geri yayılım, bileşik fonksiyonun türevini zincir kuralıyla hesaplar. $h=\phi(wx+b)$, $\hat y=vh+c$, $L=\frac12(\hat y-y)^2$ için $\partial L/\partial v=(\hat y-y)h$, $\partial L/\partial w=(\hat y-y)v\phi'(wx+b)x$ olur. Ortak bir parametre birden fazla yolda kullanılırsa katkılar toplanır. Yöntemin sinir ağlarındaki tarihsel ana kaynaklarından biri [Rumelhart, Hinton ve Williams, 1986](https://www.nature.com/articles/323533a0) çalışmasıdır; burada yapılan hesap küçük, bağımsız bir öğretim türetimidir.

`x=2,y=0,w=0.5,b=0,v=2,c=0` ve ReLU seçelim. `h=1`, tahmin `2`, kayıp `2` olur. Gradyanlar sırasıyla `(w,b,v,c)` için `(8,4,2,2)`'dir. Öğrenme oranı `0.01` olan bir SGD adımı ağırlıkları `(0.42,-0.04,1.98,-0.02)` yapar. Yeni `h=0.8`, tahmin `1.564`, kayıp `1.223048` çıkar. [Yürütülebilir örnek](../../../scripts/book_neural_walkthrough.py) bu türevleri autograd ile ve sayıları float64'te denetler. Bu tek adım bütün veri üzerinde ilerleme veya büyük adımlarda kararlılık garantisi değildir.

### Forward, backward ve step aynı olay değildir

Forward, parametrelerin belirlediği çıktıyı hesaplar. Backward, `.grad` alanlarına türevleri biriktirir. Optimizer adımı bu türevleri ve kendi durumunu kullanarak parametreleri değiştirir. `.eval()` dropout gibi eğitim davranışlarını değiştirir; gradyan kaydını kendiliğinden kapatmaz. `torch.no_grad()` ise hesap grafiğinin kurulmasını engeller; modülü otomatik değerlendirme moduna geçirmez. Dondurulmuş parametre, açık gradyan hesabı ve değişen aktivasyon birbirinden ayrılmalıdır.

Gerçek Cevahir çekirdeğinin CPU örneğinde `V=41,D=16,H=4,L=2,F=24`, dropout sıfır ve cache kapalıydı. `[2,4]` token girdisi iki katmandan sonra `[2,4,41]` logit üretti. Modelde `5.184` parametre vardı. Forward ve backward sonrasında bütün parametreler önceki kopyalarıyla aynıydı; `0.001` öğrenme oranıyla SGD adımından sonra 20 parametre tensörü değişti. Yedi geçerli hedef üzerindeki kayıp yaklaşık `3.7709403`ten `3.7681756`ya indi. Bu, eğitilmiş Cevahir checkpoint'i değil; bellekte kurulmuş küçük rastgele modelde hesap ve güncelleme sözleşmesidir.

## Konudan dosyaya ve metoda okuma haritası

| Soru | Gerçek dosya / metot | Girdi → çıktı ve sonraki kullanım |
|---|---|---|
| Kimlikleri hangi tablo karşılıyor? | [language_embedding.py — LanguageEmbedding.forward](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L234) | Tamsayı kimlikleri → lookup/norm/dropout → katmanların girdisi. |
| Başlangıç neden öğrenilmiş bilgi değil? | [LanguageEmbedding._initialize_weights](../../../src/neural_network_module/dil_katmani_module/language_embedding.py#L262) | Başlangıç yöntemi → normalize/ölçeklenmiş parametre satırları. |
| Hesap hangi modüllerden oluşuyor? | [neural_network.py — CevahirNeuralNetwork.__init__](../../../src/neural_network.py#L111) | Yapılandırma → embedding, katmanlar, norm ve bağlı/bağımsız çıkış ağırlıkları. |
| Tokenlar hangi şekilleri izliyor? | [CevahirNeuralNetwork.forward](../../../src/neural_network.py#L725) | `[B,T]` → `[B,T,D]` → `[B,T,V]`. |
| Logit hangi hedefle karşılaştırılıyor? | [loss_computation.py — LossComputation.compute_loss](../../../training_management/v2/core/loss_computation.py#L76) | Logit ve maskeli hedef → skaler kayıp ve raporlanan ölçüler. |
| Gradyanı kim parametreye uyguluyor? | [training_loop.py — TrainingLoop._finish_update](../../../training_management/v2/core/training_loop.py#L246) | Birikmiş türev → ölçek/clipping/optimizer → sonraki çağrının parametreleri. |

## Çözümlü alıştırmalar

1. `V=60000,D=512` embedding'in parametre ve float32 ağırlık belleği kaçtır? **Yanıt:** `30.720.000` parametre, `122.880.000` bayt, yaklaşık `117.19 MiB`. Aktivasyon, gradyan ve optimizer durumları bu sayıya dahil değildir.
2. Bias'lı `Linear(32,48)` girdisi `[2,5,32]` ise çıktı ve parametre sayısı nedir? **Yanıt:** `[2,5,48]` ve `32×48+48=1584`. Batch ve dizi uzunluğu ağırlık sayısını değiştirmez.
3. İki affine katmanı, arada aktivasyon yokken neden tek katmana indirebiliriz? **Yanıt:** Çarpımları yeni ağırlık, taşınmış bias'ların toplamı yeni bias olur. XOR'daki çelişki katman adediyle kalkmaz.
4. Aynı tokenın iki cümlede ilk embedding'inin eşitliği son temsilin eşitliğini gerektirir mi? **Yanıt:** Hayır; attention farklı izinli bağlamlardan içerik toplayabilir. Dropout açıkken embedding modülünün son çıktısı bile farklılaşabilir; burada karşılaştırılan ham tablo satırıdır.
5. Girdide görünmeyen token bağlı çıkış tablosunda gradyan alabilir mi? **Yanıt:** Evet, tam-sözlük loss yolu üzerinden. Sadece lookup yolu için söylenen seyrek bağımlılık yeterli değildir.
6. Bir batch'te kayıp düşmesi genellemeyi kanıtlar mı? **Yanıt:** Hayır; bağımsız veri ve uygun değerlendirme gerekir. Bu bölümün sayısal örneği hesap sözleşmesini denetler.

## Kod okuma deneyi ve kanıt sınırı

[Alt katman sözleşme testleri](../../../tests/evolution/test_lower_layer_contracts.py#L62), açık attention teşhisiyle hızlı yolun çıktı ve gradyanlarını karşılaştırır. [Yapılandırma testi](../../../tests/evolution/test_config_generation.py#L65), facade ile yöneticinin model boyutları ve durum aktarımını sınar. Bunlar hesap bağlantılarını denetler; dil anlama kalitesini ölçmez. Bu bölümde test adı verilmesi tek başına bu yazım oturumunda çalıştırıldığı anlamına gelmez; güncel yürütme kaydı kitabın doğrulama çıktısındadır.

Okuma alıştırması olarak `vocab_size=128`, `embed_dim=32`, `B=2`, `T=5` için tablonun ve logits'in şekillerini çıkarın. Ardından çekirdeğin `forward` metodunun optimizer adımı çağırmadığını, buna karşılık cache ve teşhis durumunun değişebildiğini bulun. Bu ayrım, araştırma bölümlerinde sorulan “deneyim nedeniyle kalıcı kapasite değişimi” ile sıradan bir model çağrısını ayırmak için gereklidir.

Bu anlatım [mevcut sinir ağı modül belgesinin](../../modules/neural_network/README.md) doğrulanmış akışını genişletir. İngilizce eşdeğeri ve önceki araştırma belgeleri yerlerinde korunur.

Bu turdaki küçük örnekleri `python scripts/book_neural_walkthrough.py` komutuyla yeniden çalıştırabilirsiniz. Kayıt [neural_walkthrough.json](../evidence/neural_walkthrough.json) içindedir. Script gerçek sınıfları çağırır; öğretim için atanmış embedding değerlerini ve skaler ağı, gerçek küçük Cevahir çalıştırmasından ayrı kaydeder. Dağıtılan tokenizer, eğitimli ağırlıklar ve geçmiş araştırma dosyaları değiştirilmez.

## Kaynakça

Erişim: 21 Eylül 2026. Kaynakların sonuçları kendi deney koşullarına aittir; Cevahir hakkındaki ifadeler yerel uygulama ve ayrı yürütme kaydına dayanır.

1. Rumelhart, D. E., Hinton, G. E. ve Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature 323, 533–536. [DOI:10.1038/323533a0](https://www.nature.com/articles/323533a0).
2. Bengio, Y., Ducharme, R., Vincent, P. ve Jauvin, C. (2003). *A Neural Probabilistic Language Model*. JMLR 3, 1137–1155. [Özgün yayın](https://www.jmlr.org/papers/v3/bengio03a.html).
3. Mikolov, T., Chen, K., Corrado, G. ve Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).
4. Glorot, X. ve Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS, PMLR 9, 249–256. [Özgün yayın](https://proceedings.mlr.press/v9/glorot10a.html).
5. He, K., Zhang, X., Ren, S. ve Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852).
6. Hendrycks, D. ve Gimpel, K. (2016 ön baskı). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).
7. Press, O. ve Wolf, L. (2017). *Using the Output Embedding to Improve Language Models*. EACL kısa makaleler, 157–163. [Özgün yayın](https://aclanthology.org/E17-2025/).
8. Devlin, J., Chang, M.-W., Lee, K. ve Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 4171–4186. [DOI:10.18653/v1/N19-1423](https://aclanthology.org/N19-1423/).
