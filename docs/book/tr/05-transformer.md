# 5. Transformer: dikkat hesabını öğrenilebilir bir bloğa dönüştürmek

[Kitabın içindekileri](../README.md) · [Önceki: Attention](04-attention.md) · [Sonraki: Eğitim](06-egitim.md)

Transformer, bir dizinin her konumundaki temsili, erişmesine izin verilen diğer konumların bilgisiyle tekrar tekrar dönüştüren bir sinir ağı ailesidir. Attention konumlar arasında bilgi taşır; ileri besleme ağı her konumdaki özellikleri işler; residual bağlantı önceki temsile yeni katkı ekler; normalizasyon hesapta kullanılan ölçeği düzenler. Bu parçaların sırası, maskeleri ve parametre paylaşımı birlikte **mimariyi** tanımlar. Yalnızca attention fonksiyonuna sahip olmak bütün Transformer sistemine sahip olmak anlamına gelmez.

Bu bölümde `B` batch büyüklüğü, `T` token sayısı, `D` temsil genişliği, `H` attention başlığı sayısı, `d=D/H` başlık genişliği, `F` FFN ara genişliği ve `L` katman sayısıdır. Blok girdisi `X ∈ ℝ^(B×T×D)` olsun. Denklemler dropout belirtilmedikçe kapalıyken yazılmıştır; matrisler son eksende işlem görür. Kaynak inceleme tarihi **21 Eylül 2026**'dır. Literatürdeki öneri, Cevahir'deki uygulama ve deneyle gösterilen sonuç ayrı ayrı belirtilir.

## 5.1. Encoder, decoder ve Cevahir'in gerçek hesap grafiği

### Üç mimari düzen

Bir **encoder**, girdiyi bağlama duyarlı temsillere dönüştürür. Çift yönlü self-attention kullanıldığında bir konum kendisinden önceki ve sonraki geçerli konumlara erişebilir. Bir **autoregressive decoder**, sıradaki sembolün dağılımını yalnız eldeki ön eke koşullandırır. **Encoder–decoder** düzeninde ise kaynak dizisini işleyen encoder ile hedef dizisini üreten decoder ayrıdır; decoder'ın cross-attention kısmında sorgular hedef temsilinden, anahtar ve değerler encoder çıktısından gelir. Vaswani ve arkadaşlarının 2017 çalışması encoder–decoder Transformer'ı; decoder'da nedensel self-attention, cross-attention ve FFN bileşimini tanımlar. [Vaswani vd., 2017, §3](https://arxiv.org/html/1706.03762v7#S3).

Mimari türünü dosya adından belirlemeyiz. Cevahir'in [TransformerEncoderLayer](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L60) sınıfı self-attention ve FFN içerir. [CevahirNeuralNetwork.forward](../../../src/neural_network.py#L725) ana akışında bu bloklar `causal_mask=True` varsayılanıyla istiflenir; ayrı bir kaynak encoder çıktısını okuyan cross-attention alt bloğu kurulmaz. Bu nedenle mevcut ana dil modeli **nedensel, decoder-only işlevli** bir yığındır. Kurucudaki `attention_type` gibi bir seçenek veya sınıf adındaki “Encoder”, fiilî çağrı grafiğinin yerini tutmaz.

### Nedensellik, eğitim ve üretim

Dil modellemesinde `p(x₁,…,x_T)=∏ₜ p(xₜ|x_<ₜ)` ayrışımını kullanırız. Eğitimde bütün doğru tokenlar zaten elde olduğundan, hedefler kaydırılarak çok sayıda konumun kaybı tek ileri geçişte hesaplanabilir. Nedensel maske gelecekteki hedefin kendisini görmeyi engeller. Üretimde henüz bilinmeyen sonraki token önce seçilmeli, ardından yeni ön eke eklenmelidir. Eğitimde konum hesabını toplu yapmak, üretimde bütün geleceği aynı anda bilmek anlamına gelmez.

Bu ayrım bir denetim ölçütü verir: dropout kapalıyken aynı ön eke farklı gelecekler eklenirse ön ek logitleri sayısal tolerans içinde aynı kalmalıdır. Buna karşılık nedenselliğin kaldırılması ve gelecek tokenların görünmesi, hedef kaydırması doğru olsa bile bilgi sızıntısı yaratabilir. Nedensellik bir sınıf etiketi değil, girdi bağımlılığına ilişkin sınanabilir bir özelliktir.

## 5.2. Konudan dosyaya ve metoda okuma haritası

| Konu | Cevahir dosyası ve metodu | Girdi, işlem ve çıktı |
| --- | --- | --- |
| Blok kurma | [transformer_encoder_layer.py — `__init__`](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L125) | Attention, norm, yoğun FFN/MoE ve çalışma seçeneklerini bağlar. |
| Eğitim/kullanım ayrımı | [Aynı dosya — `forward`](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L352) | `[B,T,D]`; checkpoint ve cache kararından hesap dalına geçer. |
| Seri residual | [Aynı dosya — `_forward_impl`](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L532) | Attention sonucu FFN girdisinin oluşumuna katılır. |
| Paralel residual | [Aynı dosya — `_parallel_forward_impl`](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L687) | Attention ve FFN aynı normalize edilmiş girdiyi paylaşır. |
| RMSNorm | [rms_norm.py — `RMSNorm.forward`](../../../src/neural_network_module/ortak_katman_module/rms_norm.py#L105) | Son özellik eksenini ölçekler; şekli korur. |
| Konum ekleme | [positional_encoding.py — `forward`](../../../src/neural_network_module/dil_katmani_module/positional_encoding.py#L366) | Sinüsoidal/öğrenilmiş vektör ekler; RoPE modunda temsil geçişini yapar. |
| Q/K döndürme | [Aynı dosya — `apply_rotary_pos_emb`](../../../src/neural_network_module/dil_katmani_module/positional_encoding.py#L419) | `[B,H,T,d]` veya `[B,T,d]` ve konumlar; aynı şekilli dönüşmüş vektör. |
| Kapılı FFN | [feed_forward_network.py — `_gated_forward`](../../../src/neural_network_module/ortak_katman_module/feed_forward_network.py#L314) | `D→2F`, iki parçaya ayırma, kapılama, `F→D`. |
| Genişlik sözleşmesi | [architecture_contracts.py — `resolve_ffn_dim`](../../../src/neural_network_module/architecture_contracts.py#L7) | Açık genişliği korur; otomatik genişliği aktivasyon ailesine göre seçer. |
| Uzman seçimi | [mixture_of_experts.py — `Router.forward`](../../../src/neural_network_module/ortak_katman_module/mixture_of_experts.py#L173) | Token temsillerinden top-k uzman ve ağırlık çıkarır. |
| Uzman birleştirme | [Aynı dosya — `MixtureOfExperts.forward`](../../../src/neural_network_module/ortak_katman_module/mixture_of_experts.py#L349) | Seçilmiş uzman çıktılarını `[B,T,D]` içinde toplar, yardımcı kayıp döndürür. |
| Katman yığını | [neural_network.py — `CevahirNeuralNetwork.forward`](../../../src/neural_network.py#L725) | Token kimlikleri → embedding → L blok → son norm → sözlük logitleri. |

Bu harita dosya sırasından çok veri akışını izler. Bir ayar için önce kurucuya nasıl girdiğine, sonra hangi metot dalını değiştirdiğine, son olarak bu dalı sınayan kanıta bakılır. Kaynak yorumlarında geçen model adları akademik atıf yerine kullanılamaz; örneğin açık teknik kanıt olmadan kapalı bir modelin FFN veya normalizasyon ayrıntısı çıkarılamaz.

## 5.3. Residual bağlantı: temsilin tümünü yeniden üretmek yerine katkı eklemek

### Pre-norm, post-norm ve paralel dal

`A` attention, `F` FFN, `N₁,N₂` normalizasyon olsun. Cevahir'in seri **pre-norm** yolu:

$$U=X+A(N_1(X)),\qquad Y=U+F(N_2(U)).$$

**Post-norm** yolu ise:

$$U=N_1(X+A(X)),\qquad Y=N_2(U+F(U)).$$

Paralel residual yalnız `parallel_residual and pre_norm` koşulunda etkinleşir:

$$Y=X+A(N_1(X))+F(N_1(X)).$$

Seri yolda FFN aynı bloğun attention sonucundan yararlanır. Paralel yolda bu yeni bilgiye ancak sonraki blok erişebilir. Dolayısıyla aynı parametreleri iki düzen arasında taşımak aynı fonksiyonu üretmez. Kodda “paralel”, dalların matematiksel bağımsızlığıdır; Python'daki ardışık çağrılar kendi başına eşzamanlı GPU yürütme garantisi vermez.

```mermaid
flowchart LR
    X["X: B × T × D"] --> N1["norm1"]
    N1 --> A["self.attn"]
    A --> R1["dropout + drop-path"]
    X --> P1["+"]
    R1 --> P1
    P1 --> N2["norm2"]
    N2 --> F["self.ffn: yoğun veya MoE"]
    F --> R2["dropout + drop-path"]
    P1 --> P2["+"]
    R2 --> P2
    P2 --> Y["sonraki katman"]
```

Şema yalnız seri pre-norm içindir. `norm2`, paralel pre-norm dalında kullanılmaz ve kurucu bu parametrelerin gradyanını kapatır. Attention nesnesindeki eski yerel `norm` da checkpoint anahtarlarını korumak için tutulur; residual normalizasyonu katmana ait olduğundan bu parametreler dondurulur. “Toplam parametre” ile “eğitilebilir parametre” sayılarının farklı çıkabilmesinin somut nedenlerinden biri budur.

### Türev yolunu açıkça görmek

Residual öğrenmenin tarihsel dayanağı, alt ağı girdiye eklenecek dönüşüm olarak kuran ResNet çalışmasıdır; ilk deneyleri görüntü tanımadadır. Buradaki Transformer kullanımı bu tasarım fikrinin başka bir hesap grafiğine uygulanmasıdır. [He vd., 2015/2016](https://arxiv.org/abs/1512.03385).

Kendi blok denklemimizden zincir kuralıyla `J_(x+f(x))=I+J_f(x)` çıkar. Pre-norm alt bloğunda türev `I+J_f(N(x))J_N(x)` iken post-normda `J_N(x+f(x))(I+J_f(x))` olur. İkincisinde bütün katkı normalizasyonun Jacobian'ından geçer. Ancak kimlik teriminin varlığı `I+J_f` matrisinin iyi koşullu olacağını garanti etmez: `J_f≈−I` seçilirse toplam yine küçülebilir; katman çarpımları büyüyebilir de. Başlatma, derinlik, öğrenme oranı ve sayısal tür hâlâ önemlidir.

Xiong ve arkadaşları pre/post LayerNorm konumunun başlangıç gradyanlarını ve warmup ihtiyacını etkilediğini belirli kuramsal varsayımlar ve deneylerle inceler. Buradan “pre-norm her görevde daha iyi” veya “warmup gereksizdir” sonucu çıkarılamaz. [Xiong vd., 2020](https://arxiv.org/abs/2002.04745).

## 5.4. LayerNorm ve RMSNorm: hangi eksen, hangi bilgi?

### İşlem ve sayısal örnek

Her token için `x∈ℝᴰ` alalım. LayerNorm'un yaygın affine biçimi:

$$\mu=\frac1D\sum_i x_i,\quad v=\frac1D\sum_i(x_i-\mu)^2,\quad
\mathrm{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i.$$

İstatistikler batch örnekleri arasında değil, tek tokenın özellikleri üzerinde hesaplanır. Öğrenilebilir `γ,β` ölçek ve kaymayı geri sağlar. Ba, Kiros ve Hinton'un çalışması, örnek içi normalizasyonu batch istatistiklerinden ayırır. [Ba vd., 2016](https://arxiv.org/abs/1607.06450).

RMSNorm yeniden merkezlemeyi kaldırır:

$$\mathrm{RMS}(x)=\sqrt{\frac1D\sum_i x_i^2+\epsilon},\qquad
\mathrm{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}.$$

Zhang ve Sennrich bu seçimin maliyet ve öğrenme üzerindeki etkisini araştırır. Cevahir'in `RMSNorm` sınıfında öğrenilebilir ölçek bulunur, ayrı bir `β` bulunmaz. [Zhang ve Sennrich, 2019](https://arxiv.org/abs/1910.07467).

Etkisini görmek için yalnız bu el hesabında `ε=0,γ=1,β=0` alalım: `x=(1,3)` için LN sonucu `(-1,1)`, RMSNorm sonucu `(1/√5,3/√5)≈(0.447,1.342)` olur. `x` yerine `(11,13)` konduğunda LN yine `(-1,1)` verir; RMSNorm aynı sonucu vermez. LN ortak eklemeye değişmezken RMSNorm bu bileşeni korur. Gerçek uygulamada `ε>0` bulunduğundan özellikle küçük büyüklüklerde ölçek değişmezliği yaklaşık olur. Bu hesap hiçbir modelin kalite karşılaştırması değildir.

### Cevahir'deki sayı türü ve ayrı normalizasyonlar

`RMSNorm.forward`, mevcutsa `F.rms_norm` çağırır. Manuel fallback kareler toplamını float32 ile hesaplar, öğrenilebilir ölçekle çarpar ve sonucu giriş türüne döndürür. Float32 kullanmak yarı hassasiyetin bazı taşma risklerini azaltır; sonsuz değer veya bütün sayısal sorunları olanaksız kılmaz. Ayrıca kütüphane fonksiyonunun varlığından kullanılan donanım kernelinin hızı çıkarılamaz.

`use_rmsnorm=True`, bloktaki `norm1/norm2` ile ana modelin `output_norm` seçimini etkiler. [Embedding bölümünde](03-sinir-aglari.md) açıklanan embedding LayerNorm'u ayrı bir işlemdir. Dolayısıyla “model RMSNorm kullanıyor” ifadesi her normalizasyonun aynı tür olduğu anlamına gelmez. Öğrenilmiş normalizasyon ağırlıkları da model kimliğinin parçasıdır; türünü değiştirip aynı checkpoint'in davranışını koruduğumuzu varsayamayız.

## 5.5. Konum bilgisi ve RoPE'nin matematiği

### Toplanan konum vektörü ile Q/K dönüşümünün ayrımı

Sinüsoidal yolda Cevahir'in [_build_sinusoidal_pe](../../../src/neural_network_module/dil_katmani_module/positional_encoding.py#L216) metodu `PE(p,2i)=sin(p·10000^(−2i/D))`, `PE(p,2i+1)=cos(p·10000^(−2i/D))` üretir. Bu tablo token embedding'ine eklenir ve öğrenilen parametre değildir. `learned` yolunda ise her konumun öğrenilebilir bir satırı vardır. İki durumda da `[B,T,D]` şekli korunur, içerik değişir.

RoPE modunda `PositionalEncoding.forward` konum vektörü eklemez; varsa dropout'u uygular. Asıl konum dönüşümü attention içinde projeksiyondan sonraki **Q ve K** üzerinde gerçekleşir; mevcut çağrı yolu V'yi döndürmez. `use_qk_norm=True` seçildiğinde Q/K başlık normalizasyonu dönüşten önce yapılır. Önceki bölümdeki attention hesabını bu sıra ile okumak gerekir.

### Göreli fark neden ortaya çıkar?

Bir koordinat çifti için `R(θ)=[[cosθ,−sinθ],[sinθ,cosθ]]` dönme matrisini tanımlayalım. `p` konumundaki çift `R(pω)x` olur. Cevahir'in ikişerli koordinat düzeni için açık dönüşüm:

$$(a,b)\mapsto(a\cos(p\omega)-b\sin(p\omega),\ a\sin(p\omega)+b\cos(p\omega)).$$

Dönmenin ortogonalliği ve açıların toplanmasıyla:

$$\langle R(p\omega)q,R(s\omega)k\rangle
=q^T R(p\omega)^T R(s\omega)k
=q^T R((s-p)\omega)k.$$

Bu cebir, mutlak konumlarla uygulanan dönüşümün nokta çarpımına göreli fark olarak girdiğini gösterir. Su ve arkadaşlarının RoFormer çalışmasındaki RoPE fikri bu ilişkiyi çok sayıda frekans çiftine taşır. [Su vd., 2021, 2023 revizyonu](https://arxiv.org/abs/2104.09864).

Yerel uygulamada `d` çift olmalıdır. `apply_rotary_pos_emb` son ekseni `d/2` çifte ayırır; `[T]` veya her batch örneği için ayrı `[B,T]` tamsayı konumları kabul eder. Sabit içerik vektörleri ve sabit frekanslarla iki konuma aynı `c` eklenirse `(s+c)−(p+c)=s−p` olur ve Q/K nokta çarpımı korunur. Bütün modelin her koşulda kaydırmaya değişmez olduğu sonucu çıkmaz: maskeler, bağlam sınırları, başka konum yolları ve değişen içerik temsilleri ayrıca hesaba katılmalıdır.

### Daha uzun tablo, öğrenilmiş uzun bağlam değildir

[_grow_to](../../../src/neural_network_module/dil_katmani_module/positional_encoding.py#L317) tablonun kapasitesini büyütür. `linear` ve faktör `>1` yolunda açılar ölçeklenir. Kodun `yarn` adını verdiği [_build_yarn_rope_freqs](../../../src/neural_network_module/dil_katmani_module/positional_encoding.py#L235) düşük frekansları ölçekler, yüksekleri korur, arayı karıştırır. YaRN yayını bir bağlam uzatma yöntemi ve eğitim değerlendirmesi sunar; aynı adın bulunması bütün reçetenin burada aynen uygulandığını kanıtlamaz. [Peng vd., 2023](https://arxiv.org/abs/2309.00071).

Önemli bir uygulama sınırı: kurucu `rope_scaling_type="dynamic"` değerini kabul eder, fakat incelenen tablo kurma ve büyütme dallarında `dynamic` için ayrı bir NTK ölçekleme hesabı bulunmaz; standart frekans üretimine düşer. Yorum veya log mesajından çalışır bir dinamik uzatma özelliği çıkarılamaz. Varsayılan `none` ve `1.0` ise uzatma ölçeklemesini zaten kapatır.

Öğrenilmiş konum tablosunu büyütmek ayrıca farklıdır: mevcut satırlar kopyalanır, yeni satırlar başlatılır ve yeni `nn.Embedding` nesnesi atanır. Bu, yeni satırların daha önce eğitildiği anlamına gelmez. Eğitim sürerken büyütme kullanılırsa optimizer'ın parametre nesnesi ve durum bağlantılarının da denetlenmesi gerekir. Uzun bağlam iddiası için tablo sınırı, cache sınırı ve o uzunlukta görev başarımı ayrı ayrı raporlanmalıdır.

## 5.6. FFN, GELU ve SwiGLU: token içindeki dönüşüm

### Yoğun ileri besleme hesabı

Standart FFN `D→F→D` dönüşümüdür:

$$F(x)=W_2\phi(W_1x+b_1)+b_2.$$

Aynı ağırlıklar bütün token konumlarında kullanılır; bu alt ağ kendi başına başka bir tokenı okumaz. Bağlam bağımlılığı, girdisinin attention ile önceden değişmiş olmasından gelir. [FeedForwardNetwork._standard_forward](../../../src/neural_network_module/ortak_katman_module/feed_forward_network.py#L344) projeksiyon, aktivasyon, dropout ve geri projeksiyon sırasını uygular.

GELU, `GELU(z)=zΦ(z)` biçimindedir; `Φ` standart normal dağılımın birikimli dağılım fonksiyonudur. Negatif girdiyi ReLU gibi her durumda tam sıfıra kesmez. Cevahir `gelu` ile `gelu_tanh` seçeneklerini ayırır; tanh biçimi yaygın bir yaklaştırmadır. Aktivasyonun adı, belirli veri üzerinde üstünlük kanıtı değildir. [Hendrycks ve Gimpel, 2016](https://arxiv.org/abs/1606.08415).

### Kapılama ve parametre hesabı

SwiGLU için `SiLU(z)=zσ(z)` olmak üzere:

$$F(x)=W_2[\mathrm{SiLU}(W_gx+b_g)\odot(W_ux+b_u)]+b_2.$$

İki ayrı öğrenilebilir izdüşüm bileşen bazında çarpılır. “Kapı” burada mutlaka `[0,1]` aralığında bir aç/kapa olasılığı değildir; SiLU değeri negatif de olabilir ve yukarıdan sınırlı değildir. GLU varyantlarının Transformer FFN'lerindeki karşılaştırması Shazeer'in çalışmasındadır. [Shazeer, 2020](https://arxiv.org/abs/2002.05202).

Cevahir iki ilk projeksiyonu `gate_up_proj` içinde birleştirir: `[B,T,D]→[B,T,2F]`; `chunk(2)` ile gate/up ayrılır; SiLU, çarpım, dropout ve `fc2` uygulanır. Birleştirme öğrenilebilir iki farklı matrisi ortadan kaldırmaz; tek büyük matrisin farklı satırlarıdır. Matris çarpımı çağrısını birleştirmek, ölçüm olmadan belirli bir hız kazancı söylemeye yetmez.

Bias kapalıyken yoğun FFN'in parametre sayısı `DF+FD=2DF`, SwiGLU'nunki `2DF+FD=3DF` olur. Bias açıksa sırasıyla `F+D` ve `2F+D` eklenir. Küçük `D=8,F=16` örneğinde bunlar **256** ve **384** ağırlıktır; aynı `F` ile SwiGLU daha fazla parametre taşır. Yoğun yolda `F=4D` için `8D²` bütçesini kapılı yolda yaklaşık korumak istiyorsak `3DF≈8D²`, yani `F≈8D/3` seçeriz.

[resolve_ffn_dim](../../../src/neural_network_module/architecture_contracts.py#L7) bu hesap ailesini kullanır: kapısız otomatik genişlik `4D`; kapılı genişlik `int(8D/3)` değerinin üst 256 katıdır. `D=512` için `F=1536` çıkar; kapılı FFN `2,359,296`, `F=2048` kapısız FFN `2,097,152` bias'sız ağırlık taşır. Yuvarlama nedeniyle bütçeler tam eşit değildir. Küçük öğretim modellerinde `ffn_dim` açık verilerek büyük yuvarlama sıçraması önlenebilir. Ana model `use_swiglu=True` için SwiGLU, kapalıysa GELU seçer; açık `ffn_dim` korunur.

## 5.7. MoE: parametre kapasitesi ile token başına hesabı ayırmak

### Router, seçim ve ağırlıklı çıktı

`use_moe=True` yoğun FFN yerine `E` uzman FFN ve bir router kurar. Bir tokenın temsili `x` için Cevahir'de `r=W_rx+b_context+ξ`, `S=TopK(r,k)` ve seçilen uzmanlar üzerinde `a_e=exp(r_e)/Σ_(j∈S)exp(r_j)` hesaplanır. Çıktı `Σ_(e∈S)a_eF_e(x)` olur. `ξ` yalnız eğitimde ayarlı uniform logit jitter'ıdır. Bu gürültüyü living-learning probleminin cevabı olarak yorumlamayız; burada router düzenlemesinin açık bir bileşenidir.

Mixtral raporu her katmanda sekiz FFN'den iki tanesini token başına seçen sparse MoE düzenini açıklar. Switch Transformer tek uzman yönlendirmesi ve yük dengeleme yaklaşımına odaklanır. Bunlar aynı seçimin iki farklı sayısal ayarı diye bütün ayrıntılarıyla özdeşleştirilemez. [Jiang vd., 2024, §2](https://arxiv.org/html/2401.04088v1#S2); [Fedus vd., 2021/2022](https://arxiv.org/abs/2101.03961).

Cevahir uzmanları bir döngüde gezer, atanan tokenları toplar, yalnız bu tokenları ilgili FFN'den geçirir ve ağırlıklı sonuçları yerlerine ekler. Toplam FFN ağırlığı yaklaşık `E·3DF` iken tokenın etkin FFN ağırlığı yaklaşık `k·3DF` olur; burada kapılı, bias'sız uzmanlar varsayılmıştır, router ve ortak attention ayrıca vardır. Bütün uzman ağırlıklarının tutulması, token gruplama ve dağıtma maliyeti kaybolmaz. Kodda uzman bulunması dağıtık uzman paralelliğinin veya endüstriyel throughput'un kanıtı değildir. “Uzman 3 biyoloji biliyor” gibi bir anlamsal eşleme de davranış ölçümü ister.

### Yardımcı kaybı doğru okumak

[MixtureOfExperts._compute_load_balance_loss](../../../src/neural_network_module/ortak_katman_module/mixture_of_experts.py#L428) geçerli `N` token için tüm uzmanlar üzerinde softmax ortalaması `P_e` ile uzmana atanmış token oranı `f_e` hesaplar:

$$P_e=\frac1N\sum_t\operatorname{softmax}(r_t)_e,\qquad
f_e=\frac1N\sum_t\mathbf1[e\in S_t],\qquad
\mathcal L_{aux}=\alpha E\sum_e f_eP_e.$$

`f_e` sert sayımdır ve gradyanı kesilir; `P_e` türevlenebilirdir. Kod `f_e` değerini `k`'ya bölmez; dolayısıyla `Σ_e f_e=k` olur. Dengeli `f_e=k/E, P_e=1/E` için cebirsel referans değer `αk`'dır. Bu sayı sıfır kayıp hedefi veya bütün router durumları için bir minimum teoremi değildir. `top_k` değiştirilen deneylerde yardımcı kaybın ölçeği de değişebilir. Seçilen uzmanlar üzerindeki karışım softmax'ı ile yardımcı kayıptaki bütün uzmanların softmax'ı farklı hesaplardır.

Buradan ince bir sınır daha çıkar: mevcut router'da `top_k=1` seçilirse tek logitin softmax'ı daima `1` olur ve bu ağırlığın logite türevi sıfırdır. Sert seçimin indeksinden de standart autograd gradyanı geçmez. Bu nedenle ana görev kaybı, mevcut karışım ağırlığı yolu üzerinden router'ı eğitemez; router için yardımcı kayıp gibi başka türev yolları gerekir. Bu yerel kodun cebirsel sonucudur; bütün top-1 MoE yöntemleri için genel bir önerme değildir. Seçilmiş uzman çıktısını tüm uzmanların softmax'ından alınan olasılıkla ağırlıklandırmak farklı bir hesap olurdu.

`valid_token_mask` padding'in **yardımcı kayıp istatistiğine** katılmasını engeller; mevcut uzman dağıtma döngüsünü bütünüyle padding'den arındıran bir yürütme optimizasyonu değildir. Attention maskesiyle de aynı sözleşme değildir. Bütün konumlar geçersizse korunan payda ve sıfır çarpanlar sıfır yardımcı değer üretir. Katman yalnız o forward'ın kaybını saklar; [get_and_reset_moe_loss](../../../src/neural_network.py#L970) çağrısı bir kez tüketir. `α` zaten içeridedir: eğitim hedefinde tekrar çarpmak amaçlanan ağırlığı değiştirir.

### Bağlamsal yönlendirme, cache ve öğrenme ayrımı

`routing_bias` biçimi `[B,E]`'dir; aynı batch satırındaki tokenların router logitlerine aynı önceliği ekler. Böylece aynı ağırlıklar farklı etkin uzman yollarından geçebilir. Bunun tek başına kalıcı öğrenme olduğu söylenemez; dışarıdan verilen geçici bir koşullandırma da olabilir.

[CevahirNeuralNetwork._prepare_routing_bias](../../../src/neural_network.py#L929), dolu KV cache sırasında bu önceliğin değişmesine izin vermez; cache önce temizlenmelidir. Çünkü önceki temsiller eski hesap koşuluyla üretilmiştir. Uzman yönlendirmesi veya model ağırlıkları değiştiğinde geçmiş K/V'lerin otomatik olarak yeni model geçmişi sayılması tutarsızlık yaratabilir. Bu sözleşme hesap doğruluğuna ilişkindir; hangi yönlendirmenin daha iyi olduğuna karar vermez.

## 5.8. Düzenleme ve kaynak yönetimi aynı yöntem değildir

### Dropout ve stochastic depth

Dropout, eğitim sırasında bir katkının elemanlarını rastgele sıfırlayıp yaşayanları ölçekler. Stochastic depth daha geniş bir residual dalı düşürür; temel çalışma derin residual ağlar bağlamında sunulmuştur. [Huang vd., 2016](https://arxiv.org/abs/1603.09382).

Cevahir'in [_stochastic_depth](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L458) metodu `[B,1,1]` Bernoulli maskesi kullanır. Her örneğin bütün `T,D` residual katkısı birlikte tutulur veya sıfırlanır. `m∼Bernoulli(1−p)` ise `r'=mr/(1−p)` ve sabit `r` için `E[r']=r` olur. Bu, bütün doğrusal olmayan ağın beklenen çıktısının aynı olduğu anlamına gelmez. Attention ve FFN katkıları ayrı çağrılarda maskelenir.

Ana model oranı katmanlar boyunca sıfırdan hedef `drop_path_rate` değerine dağıtır; tek katmanda oran sıfır kalır. Değerlendirmede maske uygulanmaz. Katkı bu uygulamada **hesaplandıktan sonra** maskelendiği için sıfırlanan dalın bütün hesap maliyetinin tasarruf edildiği söylenemez.

### Aktivasyon checkpointing, diske checkpoint ve mixed precision

Aktivasyon checkpointing, backward için bazı ara değerleri saklamak yerine yeniden hesaplama karşılığında aktivasyon belleğini azaltma yöntemidir. Chen ve arkadaşları bu zaman–bellek değiş tokuşunu inceler. [Chen vd., 2016](https://arxiv.org/abs/1604.06174). Modeli diske kaydetmek ise başka bir işlem olan kalıcılıktır; [7. bölüm](07-model-yasam-dongusu.md) bunu ele alır.

Katmanın `forward` metodu eğitimde checkpoint yoluna `use_cache=False` ile girer. `_recompute_context` yeniden hesaplama sırasında MoE yardımcı kaybının ikinci kez kaydedilmesini engeller. Yeniden hesaplanan fonksiyonun rastgelelik ve dış durum davranışı da denetlenmelidir; yalnızca aynı tensor şeklini üretmesi yeterli değildir. Ek hesap maliyetinin büyüklüğü ölçüm gerektirir.

Mixed precision ise bazı işlemleri daha dar sayı türleriyle yapar; kaydedilecek aktivasyon sayısını değiştirmek zorunda değildir. Düşük hassasiyet, yüksek hassasiyetli bazı birikimler ve loss scaling ilişkisi Micikevicius ve arkadaşlarının çalışmasında açıklanır. [Micikevicius vd., 2017/2018](https://arxiv.org/abs/1710.03740). Cevahir'in [resolve_precision](../../../training_management/contracts.py#L20) sözleşmesiyle seçilen tür ve cihaz ayrıca kayda geçirilir. Bu üç kavram bir ayar altında birbirinin yerine kullanılamaz.

## 5.9. Tek bloktan logitlere: yürütülebilir iz

Çalıştırılan küçük öğretim modeli `B=2,T=4,D=16,H=4,F=24,L=2,V=41` kullanır; bu ileri geçişte dropout, MoE ve cache kapalıdır. Girdi kimlikleri `[2,4]`, embedding `[2,4,16]`, her attention başlığının genişliği `4` olur. Seri pre-normda sırasıyla norm1, attention, ilk toplama, norm2, FFN ve ikinci toplama aynı dış `[2,4,16]` şeklini korur. İki bloktan sonra son normalizasyon yine `[2,4,16]`, sözlük projeksiyonu `[2,4,41]` üretir. Şeklin korunması içerik değişmediği anlamına gelmez; residual toplamların yapılabilmesini sağlar.

Bu küçük örneğin gerçek Cevahir sınıflarıyla çalıştırılabilir kaydı [book_neural_walkthrough.py](../../../scripts/book_neural_walkthrough.py), sonuç dosyası [neural_walkthrough.json](../evidence/neural_walkthrough.json) içindedir. Komut:

```text
python scripts/book_neural_walkthrough.py
```

Kayıt farklı amaçlı küçük örnekleri ayrı tutar: FFN sayımı `D=8,F=16`; blok yeniden kurma deneyi `B=1,T=3,D=16,H=4,F=24`; uçtan uca iz ise yukarıdaki iki katmanlı modeli kullanır. 21 Eylül 2026 CPU çalıştırmasının Python `3.14.3`, PyTorch `2.10.0+cpu` ortamında kaydedilen sonuçları şöyledir:

| Hesap kontrolü | Kaydedilen sonuç |
| --- | --- |
| Bias'sız GELU / SwiGLU FFN, `D=8,F=16` | `256 / 384` parametre. |
| Otomatik kapılı genişlik, `D=512` | `1536`. |
| Seri pre-norm bloğu ile metotlardan elle kurulan hesap | En büyük mutlak fark `0`. |
| RMSNorm, `(3,4)`, `ε=10⁻⁶`, başlangıç ölçeği `1` | Yaklaşık `(0.8485281,1.1313708)`; açık formülle fark `0`. |
| RoPE konumları `(2,5)→(9,12)` | Nokta çarpımı farkı `2.3842×10⁻⁷`; norm korunumu farkı `0`. |
| İki katmanlı çekirdeğin çıktısı ve toplam parametresi | `[2,4,41]`; `5184` parametre. |

Kaydedilmiş sayılarla yeniden karşılaştırmada mutlak `2×10⁻⁶`, göreli `2×10⁻⁵` tolerans kullanılır; ilgili hesapların kendi assertion kontrolleri de yürütülür. Bunlar rastgele başlatılmış küçük ağda hesap sözleşmesi kontrolleridir; eğitimli modelin dil başarımı, hız karşılaştırması veya uzun bağlam kalitesi değildir. Tek bir çalıştırmada tam sıfır fark görmek bütün cihazlarda bit düzeyinde eşitlik taahhüdü oluşturmaz.

Ana model bu blokları `L` kez uygular; ardından `output_norm` ve `output_layer` ile `[B,T,V]` sözlük logitleri çıkar. `logit_soft_cap=c>0` ise `z↦c·tanh(z/c)` uygulanır. Türevi `sech²(z/c)` olduğundan büyük büyüklüklerde doygunlaşır; capping yalnız bir değer aralığı düzenlemesi değildir, gradyanı da değiştirir. `c=0` uygulamada kapatma işaretidir, denklemde sıfıra bölmek için kullanılmaz. Attention içi `attn_logit_cap` ile çıkış `logit_soft_cap` ayrı seçeneklerdir.

| Özellik | Ortak şemadaki varsayılan | Etkin dalın anlamı |
| --- | --- | --- |
| RMSNorm / SwiGLU / RoPE | Açık / açık / `rope` | İlgili hesap sınıflarına aktarılır. |
| MoE / QK-Norm | Kapalı / kapalı | Bayrak açıldığında ayrı hesap etkinleşir. |
| Paralel residual | Kapalı | Pre-norm ile birlikte aynı girdiyi paylaşan iki dal. |
| Stochastic depth | `0` | Pozitif oran ve eğitim modu gerekir. |
| Aktivasyon checkpointing | Açık | Eğitimde yeniden hesaplama yolu. |
| RoPE ölçekleme | `none`, `1.0` | Varsayılanda uzunluk ölçeklemesi yok. |
| Çıkış / attention soft-cap | `30.0` / `0.0` | Çıkış açık, attention kapalı. |

Kaynaklar [ModelArchConfig](../../../model_management/config_schema.py#L76) ve [çekirdek kurucusudur](../../../src/neural_network.py#L111). Şema sekiz, doğrudan çekirdek on iki katmanı varsayar; `ModelManager` eski profili ayrıca normalize eder. Çalışmanın tekrarlanması için kullanılan kurucu yolu, etkin ayarlar ve checkpoint kimliği birlikte kaydedilmelidir.

## 5.10. Endüstriyel uygulamalarla kaynaklı karşılaştırma

Buradaki raporlar belirli tarihli tasarımlardır; şirketin bütün modelleri veya güncel ürünlerinin açıklaması değildir. Bileşen benzerliği, eğitim verisi, ölçek, başarım veya ağırlık uyumluluğu eşitliği göstermez.

| Birincil kaynak | Raporda açıklanan seçim | Cevahir ile sınırlı karşılaştırma |
| --- | --- | --- |
| [Llama 2 — Touvron vd., 2023, §2.2](https://arxiv.org/html/2307.09288v2#S2.SS2) | Pre-RMSNorm, SwiGLU, RoPE; büyük modellerde GQA. | Bu bileşen aileleri var; boyut, tokenizer, eğitim ve checkpoint aynı değil. |
| [Llama 3 — Dubey vd., 2024, §3.2](https://arxiv.org/html/2407.21783v3#S3.SS2) | Yoğun Transformer; GQA; mimari dışında veri ve eğitim ölçeği de vurgulanır. | Yeni özellik eklenmesi tek başına kalite sıçramasını açıklamaz. |
| [Mixtral — Jiang vd., 2024, §2](https://arxiv.org/html/2401.04088v1#S2) | Katmanda sekiz uzman, token başına iki uzman. | Cevahir'in `E=8,k=2` seçeneği aynı seçim sayılarını verebilir; yürütme sistemi ve eğitim eşit değildir. |
| [Gemma 2 — Gemma Team, 2024, §2](https://arxiv.org/html/2408.00118v3#S2) | Alt blok giriş/çıkış RMSNorm'u, yerel/küresel attention, attention ve çıkış soft-cap. | Cevahir'in pre/post alternatifleri Gemma'nın iki norm yerleşimiyle özdeş değildir; burada attention cap varsayılanda kapalıdır. |

Gemma 2 raporunda attention cap `50`, çıkış cap `30` olarak verilir. Cevahir'in çıkıştaki `30` değeri bu sayıyla örtüşür; bütün Gemma mimarisini kullandığımızı göstermez. Benzer şekilde kaynak yorumundaki “YaRN/Llama 3.1” ifadesi, RoPE ölçekleme dallarının birebir eşleştiğinin kanıtı değildir. Öğretici kaynakta karşılaştırmanın amacı bir marka adı ödünç almak değil, hangi somut denklemin ve hangi çalışma koşulunun karşılaştırıldığını açık tutmaktır.

## 5.11. Kontroller, alıştırmalar ve yanıtlar

### Kaynaktan doğrulamaya

[FFN genişliği testi](../../../tests/evolution/test_lower_layer_contracts.py#L11) boyut sözleşmesini yoğun ve uzman ağırlıklarıyla karşılaştırır. [Paralel dal testi](../../../tests/evolution/test_lower_layer_contracts.py#L50) kullanılmayan normalizasyonun dondurulmasını; [MoE checkpoint testi](../../../tests/evolution/test_contextual_moe_routing.py#L121) yeniden hesaplamada gradyan ve yardımcı kayıp tüketimini; [batched RoPE testi](../../../tests/evolution/test_attention_cache.py#L222) örnekler arasındaki farklı konum dizilerini sınar. Bu kontrolleri, [çekirdek başvuru belgesiyle](../../modules/neural_network/README.md) birlikte okuyun. Geçmeleri araştırma sorusunun çözüldüğünü veya modelin dış görevlerde üstün olduğunu göstermez.

### Alıştırmalar

1. `D=512` için kapısız `F=2048` ile kapılı otomatik `F` değerinin parametre sayılarını bias kapalıyken hesaplayın. Tam eşit olmamalarının nedeni nedir?
2. Seri ve paralel pre-normun aynı şekilli girdileri kullanmasına rağmen aynı fonksiyon olmamasını açıklayın.
3. RoPE'de `p=7,s=12` ile `p=107,s=112` için nokta çarpımı hangi varsayımlarda aynı kalır?
4. `E=8,k=2,α=0.01` ve dengeli kullanımda Cevahir'in yardımcı kaybının referans değerini bulun. Katsayıyı eğitimde yeniden çarparsanız ne olur?
5. KV cache doluyken `routing_bias` değiştirilmesi neden yalnız bir router ayarı değişikliği değildir?
6. `drop_path_rate=0.2` için hayatta kalan residual katkının çarpanı kaçtır? Bunun hesap tasarrufuyla farkı nedir?
7. Bir konum tablosunun 32 bin satıra büyümesi, 32 bin tokenlık metinde doğru bilgi getirmeyi kanıtlar mı? Eksik kanıtları belirtin.

### Yanıtlar

1. Yoğun FFN `2·512·2048=2,097,152`; kapılı genişlik 256 katına yuvarlanarak `1536`, ağırlık sayısı `3·512·1536=2,359,296` olur. Yaklaşık eşitleme ile donanıma uygun genişlik yuvarlaması farklı koşullardır.
2. Seride FFN `N₂(X+A(N₁(X)))`, paralelde `N₁(X)` alır. Birinci ifadenin attention'a bağlılığı ikinci ifadede yoktur; tensör şekilleri bu bağımlılığı söylemez.
3. Aynı içerik Q/K vektörleri, aynı frekanslar ve uygun sayısal tolerans altında iki fark da beştir. Bütün model girdisinin veya maskesinin değişmesi bu küçük cebirsel önermenin dışında kalır.
4. `f=2/8,P=1/8` olduğundan `αEΣfP=0.02`. Bir kez daha `0.01` ile çarpmak katkıyı `0.0002` yapar; amaçlanan katsayı iki kez uygulanmıştır.
5. Eski cache eski hesap koşulundan gelen temsilleri içerir. Yeni önceliklerle sıfırdan işlenen aynı geçmiş farklı üst katman K/V'leri üretebilir; mevcut sözleşme cache temizlemeyi ister.
6. Çarpan `1/0.8=1.25`'tir. Kod katkıyı önce hesaplayıp sonra maskeler; sıfır çıktı görmek o hesabın yapılmadığını göstermez.
7. Kanıtlamaz. Konum hesabının geçerliliği, cache ve bellek sınırları, uygun eğitim/uyarlama, konuma ve mesafeye göre görev başarımı ile maliyet ölçümü gerekir.

## 5.12. Kaynakça ve atıf kapsamı

Kaynaklar özgün makale veya yazarların teknik raporlarıdır; erişim tarihi **21 Eylül 2026**. arXiv kaydının tarihiyle konferans/dergi yılı farklı olduğunda ikisi belirtilmiştir. Atıflar yöntemlerin kökenini ve raporlanan tasarımı destekler; Cevahir hakkındaki davranış ifadelerinin dayanağı yukarıdaki kaynak kodu ve yerel kanıttır.

1. Vaswani, A. vd. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
2. He, K., Zhang, X., Ren, S. ve Sun, J. (2015 ön baskı; 2016 CVPR). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
3. Ba, J. L., Kiros, J. R. ve Hinton, G. E. (2016). *Layer Normalization*. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).
4. Zhang, B. ve Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).
5. Xiong, R. vd. (2020). *On Layer Normalization in the Transformer Architecture*. ICML. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745).
6. Su, J. vd. (2021 ön baskı; burada 2023 revizyonu). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
7. Peng, B., Quesnelle, J., Fan, H. ve Shippole, E. (2023 ön baskı). *YaRN: Efficient Context Window Extension of Large Language Models*. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).
8. Hendrycks, D. ve Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).
9. Shazeer, N. (2020). *GLU Variants Improve Transformer*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).
10. Fedus, W., Zoph, B. ve Shazeer, N. (2021 ön baskı; 2022 JMLR). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).
11. Jiang, A. Q. vd. (2024). *Mixtral of Experts*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
12. Huang, G., Sun, Y., Liu, Z., Sedra, D. ve Weinberger, K. Q. (2016). *Deep Networks with Stochastic Depth*. ECCV. [arXiv:1603.09382](https://arxiv.org/abs/1603.09382).
13. Chen, T., Xu, B., Zhang, C. ve Guestrin, C. (2016). *Training Deep Nets with Sublinear Memory Cost*. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174).
14. Micikevicius, P. vd. (2017 ön baskı; 2018 ICLR). *Mixed Precision Training*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).
15. Touvron, H. vd. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. Meta teknik raporu. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).
16. Dubey, A. vd. (2024). *The Llama 3 Herd of Models*. Meta teknik raporu. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).
17. Gemma Team (2024). *Gemma 2: Improving Open Language Models at a Practical Size*. Google DeepMind teknik raporu. [arXiv:2408.00118](https://arxiv.org/abs/2408.00118).
