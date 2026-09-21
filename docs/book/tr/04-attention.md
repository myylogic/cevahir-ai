# 4. Attention: bir konum başka konumlardan ne alır?

[Kitabın içindekileri](../README.md) · [Önceki: Sinir ağı](03-sinir-aglari.md) · [Sonraki: Transformer](05-transformer.md)

Bir tokenın tablodan alınan vektörü çevresindeki kelimeleri henüz içermez. Attention, bir konumdaki temsilin hangi diğer konumların değerlerinden ne kadar alacağını hesaplar. “Dikkat ediyor” ifadesinin bu kodda karşılığı bir bilinç durumu değil, ağırlıklı vektör toplamıdır. Temel yapı [Transformer makalesindeki](https://arxiv.org/abs/1706.03762) ölçeklenmiş nokta çarpımı hesabıdır; Cevahir bunu farklı head düzenleri, maskeler ve yürütme yollarıyla uygular.

Bu bölümün öğrenme hedefi, attention formülünü okuyabilmenin yanında bir uygulamanın hangi bilgiyi kullanabildiğini, ne kadar bellek ayırdığını ve bir testin hangi iddiayı desteklediğini hesaplayabilmektir. Önceki bölümdeki matris çarpımı, embedding ve gradyan bilgisi yeterlidir. Aşağıdaki küçük sayısal örnekler öğretim için kurulmuştur; eğitilmiş Cevahir modelinin dil başarısını ölçmez. Kaynak inceleme ve literatür kontrol tarihi: **21 Eylül 2026**.

## Attention hangi problemi çözer?

### Sabit temsil ile bağlama göre bilgi taşıma

Bir diziyi tek vektöre indirip bütün çıktıları bu vektörden üretmek, uzun bir girdinin farklı parçalarına farklı anlarda erişmeyi zorlaştırır. Bahdanau, Cho ve Bengio'nun çeviri çalışması, her çıktı adımında kaynak temsillerine yeniden ağırlık verilmesini bu soruna bir çözüm olarak geliştirdi. Buradaki attention, tekrarlayan encoder–decoder ağlarıyla birlikte kullanılıyordu; attention ile Transformer aynı şey değildir. [Bahdanau, Cho ve Bengio, 2015](https://arxiv.org/abs/1409.0473).

İşlemi Cevahir açısından şöyle okuyabiliriz: token kimliği önce bir vektöre dönüşür; bu vektör attention katmanına geldiğinde diğer konumlardaki değerlerin ağırlıklı toplamını alır. Böylece aynı token kimliği, farklı bağlamlarda farklı ara temsiller üretebilir. Ancak bu işlem kendi başına kalıcı öğrenme değildir. Parametreler değişmeden bağlama bağlı hesap yapılabilir. Kalıcı öğrenme için örneğin bir kayıp fonksiyonundan türetilen gradyanların projeksiyon ağırlıklarını değiştirmesi gerekir; bunun ayrıntıları [eğitim bölümündedir](06-egitim.md).

### Self-attention, cross-attention ve bilgi yönü

**Self-attention** aynı dizinin temsilleri arasında bilgi taşır. **Cross-attention** ise query'leri bir kaynaktan, key ve value'ları başka bir kaynaktan alır. Çeviride üretilen hedef dizinin kaynak cümleyi okuması ikinci duruma örnektir. Ayrım, Q/K/V matrislerinin sayısında değil, bunların hangi veri akışından üretildiğindedir. Kaynak dizinin uzunluğu `S`, sorgu dizisinin uzunluğu `T` olabilir; iki uzunluğun eşit olması gerekmez.

**Çift yönlü** attention bir konumun hem önceki hem sonraki konumlara erişmesine izin verir. **Nedensel** attention geleceğe erişimi kapatır. Bu ikinci ayrım, self/cross ayrımından bağımsızdır. Bir self-attention modülü yalnız adı nedeniyle nedensel olmaz; Cevahir'de bunun somut denetimi `causal_mask` ve birleştirilen maskelerdir. Model sınıfında “encoder” sözcüğünün geçmesi de hangi bilgiye erişildiğini belirlemez. Doğru okuma yöntemi, çağıranın geçirdiği maskeyi ve maskenin `forward` içinde kurulmasını izlemektir.

## Q, K ve V aynı girdiden neden üç kez üretilir?

Bir konumun **query** vektörü aradığı ilişkileri, bir aday konumun **key** vektörü eşleşme skorunu, **value** vektörü ise toplanacak içeriği temsil eder. Bunlar elle yazılmış anlam etiketleri değildir. Self-attention'da aynı girdinin üç ayrı öğrenilebilir doğrusal dönüşümüdür:

`Q = X W_Qᵀ`, `K = X W_Kᵀ`, `V = X W_Vᵀ`.

Bu ayrım sayesinde hangi konumun seçileceğini belirleyen özelliklerle taşınacak içeriğin özellikleri aynı olmak zorunda kalmaz. Cevahir'de [MultiHeadAttention.__init__](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L91) `query_proj`, `key_proj`, `value_proj`, `out_proj` adlı, bias içermeyen dört `nn.Linear` kurar. [TransformerEncoderLayer._forward_impl](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L532) attention'a aynı temsili üç kez verir; çağrı bu nedenle self-attention'dır. Genel attention arayüzünün ayrı `key` ve `value` kabul etmesi, ana modelin encoder–decoder cross-attention kullandığını göstermez.

Soft-cap kapalıyken bir head için skor ve çıktı:

`S = QKᵀ / (sqrt(d) τ) + M`, `A = softmax(S)`, `O = AV`.

`d` head genişliği, `τ` attention sıcaklığı, `M` maskedir. Nokta çarpımı benzerliğe dayalı skoru üretir; `sqrt(d)` ölçekleme boyut arttıkça skor büyüklüğünü dengeler. Softmax her query için key ekseni üzerinde normalleştirir. `O_i = Σ_j A_ij V_j`, i'nci konumun aldığı içeriktir. Cevahir'in normal `forward` çağrısı `τ=1` ile çalışır. Metin üretimindeki token örnekleme sıcaklığı, bu iç attention sıcaklığından ayrı bir aşamadır.

### Ölçeklemenin varsayımı ve gradyanı

`q·k = Σ_r q_r k_r` olsun. Öğretim için bileşen çiftlerini birbirinden bağımsız, `q_r` ile `k_r`yi de bağımsız; her birini sıfır ortalamalı ve bir varyanslı varsayalım. O zaman her çarpımın ortalaması sıfır, varyansı birdir. Bağımsız toplamın varyansı `d`, standart sapması `sqrt(d)` olur. `sqrt(d)` ile bölmek bu idealize durumda varyansı bire indirir. Bu, ölçeklemenin arkasındaki boyut hesabıdır; öğrenilmiş Q ve K'nin gerçek dağılımlarının bağımsız ve birim varyanslı olduğunu ispatlamaz. Özellikle Q/K normalizasyonu, korelasyonlar ve öğrenilen ölçekler gerçek skor dağılımını değiştirir.

Bir softmax satırı `a_j=exp(s_j)/Σ_k exp(s_k)` ise türevi

`∂a_j/∂s_k = a_j(δ_jk − a_k)`

olur. Bir olasılık bire, diğerleri sıfıra yaklaşınca birçok türev küçülür. Bu nedenle aşırı büyük skor farkları, bazı yolların öğrenme sinyalini zayıflatabilir. Ölçekleme bu riski kontrol etmeye yardım eder; bütün katmanlarda kararlı gradyan garantisi vermez. Cevahir'in manuel uygulaması ayrıca satır maksimumunu çıkarır: `softmax(s)=softmax(s−max(s))`. Bu eşitlik dağılımı değiştirmeden üstel fonksiyonun büyük pozitif girdilerini azaltır.

Query/key/value ayrımı da öğrenme sırasında anlam kazanır. Bir çıktı kaybı value yoluyla taşınan içeriği, Q/K yoluyla hangi konumdan ne kadar alınacağını değiştirebilir. Sabit attention ağırlıkları altında `∂O_i/∂V_j=A_ij I` olur; fakat self-attention'da girdiye göre tam türev bununla sınırlı değildir. Aynı girdi Q ve K'yi de ürettiği için ağırlıkların türevi de zincir kuralına katılır. “En büyük attention ağırlığı, en büyük toplam etki demektir” çıkarımı bu yüzden genel olarak geçerli değildir.

Bu paragraftaki varyans ve türev hesapları bölüm için açıkça türetilmiştir. Ölçeklenmiş nokta çarpımı ve çok başlıklı düzenin tarihsel referansı [Vaswani ve diğerleri, 2017](https://arxiv.org/abs/1706.03762)'dir; burada incelenen yazılım davranışının referansı ise Cevahir'in bağlantısı verilen kaynak kodudur.

### Üç tokenı elle hesaplamak

Tek head, `d=1`, `τ=1`, dropout kapalı olsun. Projeksiyonlardan **sonra** elde edildiğini varsaydığımız oyuncak vektörleri seçelim:

```text
Q = K = [[1], [2], [0]]
V =     [[1], [3], [2]]
QKᵀ =  [[1, 2, 0],
        [2, 4, 0],
        [0, 0, 0]]
```

Bunlar tokenizer çıktıları veya öğrenilmiş kelime anlamları değildir. Tek boyutlu sayılarla işlemi görünür kılan bir örnektir. Nedensel maske eklendiğinde skorlar şöyle olur:

```text
S = [[1, -∞, -∞],
     [2,  4, -∞],
     [0,  0,  0]]
```

Birinci satır yalnız ilk değeri okuyabilir; softmax sonucu `[1,0,0]` olur. İkinci satırda maksimumu çıkardığımızda `[-2,0,-∞]` kalır. Böylece `a_21=1/(1+exp(2))≈0.119202922`, `a_22≈0.880797078` bulunur. Üçüncü satırın üç skoru eşit olduğundan her değer `1/3` ağırlık alır.

| Query konumu | Attention ağırlıkları | Ağırlıklı value toplamı |
| --- | --- | --- |
| 1 | `[1, 0, 0]` | `1` |
| 2 | `[0.119202922, 0.880797078, 0]` | `0.119202922×1 + 0.880797078×3 ≈ 2.761594156` |
| 3 | `[1/3, 1/3, 1/3]` | `(1+3+2)/3 = 2` |

Sonuç `[1, 2.761594156, 2]`'dir. Üçüncü query'nin sıfır olması çıktıyı sıfır yapmamıştır; sıfır query eşit skorlar üretmiş, value'lar ortalanmıştır. Cevahir'de bunun ardından head'ler birleştirilip `out_proj` uygulanır; tablodaki sayılar bu son projeksiyondan önceki attention çıktısıdır. Maskeyi kaldırmak ilk iki satırın sonucunu değiştirir, üçüncü satır zaten tüm geçmişi gördüğünden aynı kalır.

## Şekilleri görmeden head sayısını anlamak zordur

`H` query head sayısı, `H_kv` key/value head sayısı ve `d=D/H` olsun. [MultiHeadAttention.forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L678) projeksiyonları yeniden şekillendirip eksen değiştirir:

| Tensör | Şekil | Anlamı |
| --- | --- | --- |
| Girdi query | `[B,T,D]` | Bu çağrıda hesaplanacak konumlar. |
| Q | `[B,H,T,d]` | Her query head için sorgular. |
| İlk K ve V | `[B,H_kv,S,d]` | Erişilebilen kaynak konumları; cache ile `S>T` olabilir. |
| Attention skorları | `[B,H,T,S]` | Her query–key konum çifti için skor. |
| Head çıktıları | `[B,H,T,d]` | Value vektörlerinin ağırlıklı toplamı. |
| Birleştirme ve `out_proj` | `[B,T,D]` | Transformer residual dalına geri verilecek temsil. |

`D`, `H` ile; `H` de `H_kv` ile tam bölünmelidir. [Kurucu doğrulaması](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L161) bunları denetler. `H_kv=H` standart MHA, `H_kv=1` MQA, aradaki değerler GQA'dır. Bu adlandırma [GQA çalışmasıyla](https://arxiv.org/abs/2305.13245) örtüşür. Varsayılan `num_kv_heads=None`, `H_kv=H` anlamına gelir; GQA destekleniyor olması varsayılanın GQA olduğu anlamına gelmez.

Örneğin `D=32`, `H=4`, `H_kv=2` seçildiğinde `d=8`, K/V projeksiyon genişliği `16` olur. Kod hesap sırasında `repeat_interleave` ile iki KV head'i dört query head'ine eşler. Cache ise genişletilmeden, iki head biçiminde saklanır. Bu koşullarda K/V depolaması MHA'nın yarısıdır; bütün model belleğinin veya toplam çalışma süresinin yarıya indiği sonucu çıkmaz. Bu uygulama her backend'e yerel GQA yürütmesini devretmez; genişletilmiş çalışma tensörleri ayrıca dikkate alınmalıdır.

### Başlık paylaşımı neyi değiştirir?

Her head'i “bir gramer kuralı” diye adlandırmak gerekmez. Öğrenilebilir farklı projeksiyonlar, aynı girdiden farklı ilişki ve içerik hesapları kurma olanağı verir; bunların insan tarafından önceden belirlenmiş işlevleri yoktur. MQA query başlıklarını korurken K/V başlıklarını paylaşır; GQA paylaşımı gruplara sınırlar. MQA'nın gerekçesi artımlı üretimde büyük K/V tensörlerinin taşınma maliyetidir. GQA çalışması, mevcut MHA checkpoint'lerini dönüştürüp ek eğitim uygulamayı da inceler; Cevahir'de `num_kv_heads` ayarının bulunması bu dönüşüm eğitiminin yapıldığını göstermez. [Shazeer, 2019](https://arxiv.org/abs/1911.02150); [Ainslie ve diğerleri, 2023](https://arxiv.org/abs/2305.13245).

Kaynakta bias içermeyen projeksiyonların toplam ağırlık sayısını doğrudan sayabiliriz:

`N_proj = 2D² + 2D(H_kv d)`.

İlk terim query ve output, ikinci terim key ve value projeksiyonlarıdır. `D=32,H=4,d=8` için MHA'da `4096`, iki KV başlıklı GQA'da `3072`, MQA'da `2560` projeksiyon ağırlığı vardır. Bu yalnız dört doğrusal dönüşümün hesabıdır. İsteğe bağlı Q/K normalizasyonu, sınıfta uyumluluk için tutulan normalizasyon parametreleri ve Transformer'ın diğer katmanları bu sayıya dahil değildir. Bir seçeneği değiştirmek checkpoint tensör boyutlarını da değiştirebilir; eski MHA ağırlıklarını yeni GQA mimarisine doğrudan yüklemek genel bir dönüşüm yöntemi değildir.

## Maske, modelin hangi bilgiye erişebileceğidir

Sonraki tokenı tahmin eden model eğitim sırasında gelecekteki tokenı okuyabilseydi, gerçekte üretim anında bulunmayacak bilgiden yararlanırdı. Nedensel maske, key konumu query konumundan büyükse o bağlantıyı engeller. Cevahir bunu cache varken mutlak `query_positions` ve `key_positions` üzerinden kurar; yalnızca yeni parçanın yerel indislerini kullanmak yeterli olmaz.

[_prepare_attention_mask](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L372) arayüzünde boolean `True` **engelle** demektir. Sonlu, 0–1 aralığındaki ve en az bir pozitif değer içeren sayısal maskede `>0.5` geçiştir; diğer değerler engellenir. Tamamı sıfır sayısal maske bu özel dala girmez ve sıfır additive maske, yani ek engel yok, olarak yorumlanır. Genel additive maskede `0` skoru değiştirmez, `−∞` bağlantıyı engeller.

PyTorch'un doğrudan SDPA boolean maskesi ise `True=katıl` semantiği kullanır. Cevahir'in kendi maskesini additive biçime çevirmesi bu tersliği giderir; iki arayüz arasında boolean maskeyi doğrudan taşımak yanlıştır. [PyTorch SDPA belgesi](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html) bu ayrımı açıkça tanımlar.

İki boyutlu `[B,S]` padding maskesi ile `[T,S]` attention maskesi, `B=T` olduğunda biçimden ayırt edilemez. Böyle bir padding maskesini `[B,1,1,S]` biçiminde vermek gerekir. `sliding_window=w` ayrıca `query_position−key_position >= w` bağlantılarını engeller. Bu koşul tek başına geleceği engellemez; geçmişe dönük yerel attention için causal maske de açık olmalıdır. Kodda yoğun maske oluşturulabildiği için pencere ayarından otomatik `O(Tw)` uygulama maliyeti vaat edilemez.

Tamamen kapalı bir satırda normal softmax ifadesi tanımsız hale gelebilir. [_standard_sdpa_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L613), bu satırların attention ağırlıklarını sıfırlar. Diğer satırlar dropout kapalıyken toplamı yaklaşık bir olan dağılımlardır. Eğitimde attention dropout uygulandığında tek örnekte toplamın tam bir olması beklenmez.

### Padding, nedensellik ve pencere aynı maske değildir

Padding, gerçek veri olmayan doldurma konumlarını engeller. Nedensellik, gerçek veri olsa bile gelecek konumları engeller. Pencere ise belirli uzaklıktaki geçmişi erişim dışına çıkarır. Üçü birlikte kullanılabilir. Örneğin query'nin mutlak konumu `5`, pencere `3` ise nedensellikle birlikte konum `3,4,5` kalır; bunlardan `4` padding ise yalnız `3,5` değerleri toplanır. Bir matrisin üçgensel görünmesi padding'in doğru uygulandığını kanıtlamaz.

Sayısal maskelerin özel 0–1 yorumuna güvenerek farklı arayüzler arasında maske taşımak hata üretir. Açık additive maske kullanılıyorsa engellenen elemanın değeri gerçekten `−∞` olmalıdır; yalnız çok büyük negatif bir sayı, tamamen kapalı satırın tespitinde aynı sözleşme değildir. Cevahir burada `isneginf(...).all(...)` kullanır. Birim test yazarken yalnız çıktı boyutunu kontrol etmek yerine engellenmiş sütunun ağırlıklarının sıfır kaldığını ve geleceğin değiştirilmesinin geçmiş çıktıları değiştirmediğini sınamak gerekir.

## Aynı hesap, üç yürütme yolu

[scaled_dot_product_attention](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L454) hangi hesabın çağrılacağını seçer:

1. `return_attention_weights=True` veya `attn_logit_cap>0` ise manuel yol kullanılır. Soft-cap, skorların üzerine maske eklenmeden önce uygulanır.
2. Aksi halde `use_pytorch_sdpa=True` ise `F.scaled_dot_product_attention` çağrılır. PyTorch uygun backend'i seçer; bu bayrak belirli bir GPU kernelinin çalıştığını kanıtlamaz.
3. PyTorch yolu kapalıyken kullanılabilir harici Flash Attention istenmişse `flash_attn_func` denenir. Float32 veya özel maske varsa doğrudan manuel yola dönülür; ayrıca harici çağrı hata verirse de manuel hesap kullanılır. Harici Flash yolu seçilmemişse yine manuel yol çalışır.

Manuel yolun temel hesabı kaynakta şu satırlarla görülür; arada maskeleme ve kararlılık işlemleri bulunduğu için bu üç satır tek başına tam fonksiyon değildir:

```python
scores = torch.matmul(work_query, work_key.transpose(-2, -1)) / scale
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights.to(value.dtype), value)
```

PyTorch yolu attention matrisini döndürmez. Teşhis için matris istenmesi bu yüzden yalnız ek bir çıktı bayrağı değildir; yürütme yolunu ve bellek maliyetini değiştirir. `_pytorch_sdpa_forward`, değerlendirme modunda `dropout_p=0` geçirir. Bu ayrıntı gereklidir çünkü SDPA fonksiyonunun dropout davranışı kendisine verilen olasılığa bağlıdır; çevre modülün `eval()` durumunu kendiliğinden okumaz.

Attention'ın sayısal çıktısı `out_proj` sonucudur; residual toplama ve katman normalizasyonu burada yapılmaz. Varsayılan `forward` bu tensörü tek başına, attention ağırlıkları istenirse `(output, weights)` çiftini, etkin cache yolunda `(output, weights_or_none, cache)` üçlüsünü döndürür. Sonraki tüketici olan `TransformerEncoderLayer` sonucu açar ve residual/normalizasyon işlerini üstlenir. Sınıfta eski checkpoint uyumluluğu için `norm` bulunsa da mevcut `forward` onu kullanmaz.

### Q/K normalizasyonu ve skor sınırlandırma

`use_qk_norm=True` Q ve K'nin son boyutuna RMSNorm uygular; kaynakta işlem RoPE'den önce yapılır. `attn_logit_cap=c>0` ise ölçeklenmiş skoru `c tanh(s/c)` ile yumuşak biçimde sınırlandırır. Küçük `s/c` için `tanh(s/c)≈s/c`; çok büyük mutlak skorlarda büyüme sınırlanır. Bunlar aynı işlem değildir: ilki skor oluşmadan önce temsil ölçeğini, ikincisi oluşmuş skoru değiştirir.

Maske soft-cap'ten **sonra** eklenir. `−∞` önce tanh içine alınsaydı sonlu bir negatif sayıya dönüşebilir ve engellenmesi gereken bağlantı yeniden pozitif softmax ağırlığı alabilirdi. Bu sıralama matematiksel bir ayrıntının yazılım sözleşmesini nasıl değiştirdiğini gösterir. Ayrıca soft-cap açılınca varsayılan SDPA bayrağı açık kalsa bile manuel yol seçilir. Performans karşılaştırmasında yalnız ayar isimlerini listelemek bu nedenle yeterli değildir; gerçekleşen yolu da kaydetmek gerekir.

## Hesap maliyeti, çalışma belleği ve cache nasıl ayrılır?

### Yoğun attention için işlemleri saymak

Aşağıdaki sayım bu bölümdeki tensör şekillerinden türetilmiştir. Bir çarpma ve bir toplama iki FLOP kabul edilsin; sabit giderler, softmax, maske, normalizasyon, geri yayılım ve output taşıma ayrıca değerlendirilir. `QKᵀ` çarpımı yaklaşık `2BHTSd`, `AV` çarpımı yine `2BHTSd` işlem ister. Toplam ana matris çarpımı maliyeti `4BHTSd=4BTSD` olur. Self-attention'da `S=T` olduğundan bu terim `4BT²D`'dir.

Standart MHA'da aynı dizi üzerindeki dört projeksiyon yaklaşık `8BTD²` FLOP ekler. Dolayısıyla “attention kareseldir” cümlesi yalnız dizi uzunluğuna bağlı terimi anlatır; kısa dizilerde geniş projeksiyonlar önemli olabilir. Nedensel maskede yaklaşık yarı bağlantı kullanılabilse de yoğun manuel matris çarpımı bunların tamamını hesaplayıp sonradan maskeleyebilir. Matematiksel sıfır deseni tek başına yarı süre anlamına gelmez.

Skor veya ağırlık matrisi tek başına `BHTS` sayı içerir. `B=1,H=8,T=S=4096`, float32 örneğinde `1×8×4096²×4 = 536870912` bayt, yani **512 MiB** gerekir. Aynı anda skor ve ağırlıkların tutulması, gradyanlar ve geçici tamponlar toplamı artırabilir. Bu sayı bütün modelin tepe belleği değildir; tek bir yoğun matrisin boyut hesabıdır. Float16 girdiler kullanılması da Cevahir'in manuel skor yolunu otomatik yarıya indirmez: kaynak düşük hassasiyetli Q/K'yi skor hesabı için float32'ye yükseltir.

### FlashAttention hangi maliyeti azaltır?

FlashAttention, yoğun attention'ı küçük bloklarla hesaplayarak büyük ara matrisleri yüksek bant genişlikli GPU belleğine tekrar tekrar yazma/okuma ihtiyacını azaltır. Hedef matematiksel işlem standart attention'dır; yaklaşık seyrek attention ile aynı yaklaşım değildir. İşlem sıralamasının değişmesi sonlu hassasiyette küçük sayısal farklar doğurabilir. Yöntemin “exact” niteliği bit düzeyinde her backend'le aynı sonuç veya genel `O(T)` hesap süresi anlamına gelmez. [Dao ve diğerleri, 2022](https://arxiv.org/abs/2205.14135).

Cevahir için buradan çıkarılabilecek ölçülü sonuç şudur: SDPA arayüzü uygun koşullarda verimli bir gerçekleştirmeye geçiş sağlayabilir; gerçekten hangi kernelin seçildiği cihaz, veri türü, boyutlar, maske ve PyTorch sürümüne bağlıdır. CPU'da doğru sonuç elde etmek GPU FlashAttention hızını ölçmez. Ayrıca ağırlık matrisini kullanıcıya döndürmeyi istemek, zaten `BHTS` elemanlık bir çıktı talep etmektir; bu çıktının belleğini herhangi bir kernel ortadan kaldıramaz. [PyTorch SDPA belgesi](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html).

### KV cache yeniden hesaplamayı azaltır, öğrenme yapmaz

Üretimin yeni adımında geçmiş tokenların K/V temsilleri yeniden hesaplanmadan kullanılabilir. Katman başına kalıcı cache'in asgari veri yükü `2BSH_kv d b` bayttır; `b` saklanan her sayının bayt sayısı, `2` hem K hem V içindir. `L` eş boyutlu katmanda toplam `2LBSH_kv d b` olur. Metadata, kapasite fazlası, geçici kopyalar ve ayırıcı giderleri bu ifadeye dahil değildir.

Öğretim örneği olarak `L=32,B=1,S=4096,H=32,d=128,b=2` seçelim. Bunlar Cevahir'in varsayılan yapılandırması olarak sunulmuyor:

| Düzen | `H_kv` | Asgari K/V veri yükü | MHA'ya oran |
| --- | ---: | ---: | ---: |
| MHA | 32 | `2147483648` bayt = **2 GiB** | 1 |
| GQA | 8 | `536870912` bayt = **512 MiB** | 1/4 |
| MQA | 1 | `67108864` bayt = **64 MiB** | 1/32 |

Query head sayısı bu üç satırda aynıdır. GQA depolaması küçülürken yeni query'nin geçmişle etkileşimi hâlâ hesaplanır. Cevahir'in `repeat_interleave` kullanması çalışma sırasındaki bazı tensörleri yeniden genişletir. Cache'in küçük olması ile tepe çalışma belleği bu yüzden ayrı ölçümlerdir. Önceden ayrılan tampon kapasitesi de kullanılan token sayısından büyük olabilir; uygulamanın gerçek ayrılmış belleği için `S` yerine ayrılmış kapasitenin incelenmesi gerekir.

Cache olmadan her adımda tüm uzunluğu `t` olan öneki yeniden hesaplamak, yalnız attention etkileşimleri açısından `Σ_t O(t²D)=O(N³D)` toplamına yol açar. Cache ile yalnız yeni query ve saklanan geçmiş arasında `O(tD)` etkileşim hesaplanır; toplam `O(N²D)` olur. Bu sayım sıfırdan `N` tokenın art arda üretilmesi, sınırsız geçmiş ve sabit model genişliği varsayımındadır. Bir defalık uzun prompt işleme, projeksiyonlar, ağırlıkların taşınması, FFN ve örnekleme ayrıca maliyet getirir.

Cache, parametre güncellemesi değildir. Ağırlıklar değiştirilirse eski K/V'ler eski fonksiyonun sonucudur; uyumluluk değerlendirilmeden kullanılamaz. Benzer şekilde cache'den geçmiş atılması, o geçmişin hâlâ erişilebilir olduğu tam attention ile genel eşdeğerlik garantisini kaldırır. [KVCache.update](../../../src/neural_network_module/ortak_katman_module/kv_cache.py#L177) ve [üretim bölümü](08-uretim.md), kapasite, mutlak konum ve saklanan geçmiş ayrımını incelemek için sonraki duraktır.

## Endüstri örnekleri nasıl karşılaştırılmalı?

Bir yöntemin endüstride kullanılması, bütün yapılandırmaların aynı başarıyı vereceğini göstermez. Llama 2 raporu GQA'yı 34B ve 70B modeller için belirtir; aynı rapordaki 7B ve 13B modelleri bu özelliğe sahip diye genellemek yanlıştır. Mistral 7B raporu ise GQA ve sliding-window attention'ı birlikte kullanır. Bu örnekler yöntem ailesini yerleştirir; Cevahir'in aynı eğitim verisine, ölçeğe veya ölçülmüş performansa sahip olduğunu söylemez. [Touvron ve diğerleri, 2023, Tablo 1 ve Ek A.2.1](https://arxiv.org/html/2307.09288v2); [Jiang ve diğerleri, 2023](https://arxiv.org/abs/2310.06825).

| Karşılaştırma sorusu | Cevahir'de bakılacak somut yer | Raporlanması gereken sınır |
| --- | --- | --- |
| Query/KV paylaşımı nedir? | `num_heads`, `num_kv_heads`, projeksiyon boyutları | Desteklenen seçenek ile kullanılan ayar farklıdır. |
| Yerel pencere gerçekten etkin mi? | `sliding_window` ve causal maske birleşimi | Yoğun maskeleme seyrek kernel hızını kanıtlamaz. |
| Hangi attention hesabı çalıştı? | `scaled_dot_product_attention` yönlendirmesi | PyTorch bayrağı belirli GPU kernelini kanıtlamaz. |
| Cache ne kadar küçüldü? | KVCache iç boyutları, veri türü, kapasite | Mantıksal K/V yükü tüm model belleği değildir. |
| Kalite korundu mu? | Aynı veri ve eğitim bütçesinde ölçüm | Şekil ve eşdeğerlik testleri dil kalitesi deneyi değildir. |

## Attention ağırlığı bir açıklama mıdır?

Bir ısı haritası, belirli katman ve head'in o girdide hangi konumlardan hangi katsayılarla değer topladığını gösterir. Nihai kararın nedenini tek başına vermez. Jain ve Wallace, bazı NLP görevlerinde farklı attention dağılımlarının benzer tahminler verebildiğini ve ağırlıkların başka önem ölçüleriyle zorunlu olarak örtüşmediğini gösterdi. Wiegreffe ve Pinter ise açıklamanın tanımına ve karşılaştırma tasarımına göre daha dikkatli değerlendirme gerektiğini tartıştı. Akademik sonuç “haritalar anlamsızdır” da değildir; hangi açıklama iddiasının hangi deneyle sınandığı belirtilmelidir. [Jain ve Wallace, 2019](https://aclanthology.org/N19-1357/); [Wiegreffe ve Pinter, 2019](https://arxiv.org/abs/1908.04626).

Bölümün kendi basit karşı örneği bunu görünür kılar: bütün value vektörleri aynı `v` olsun. Dropout kapalı ve satır normalize iken hangi attention dağılımı seçilirse seçilsin `Σ_j a_j v=v` olur. Dolayısıyla farklı haritalar aynı attention çıktısını üretebilir. Gerçek modelde residual yollar, output projeksiyonu ve sonraki katmanlar da etkiyi değiştirir. Cevahir'de bir açıklama deneyi, haritaya ek olarak girdi müdahalesini, çıktıdaki değişimi, uygun karşılaştırmayı ve sınırlılıkları kaydetmelidir. Haritadaki tek büyük hücreden “modelin bu gerekçeyle düşündüğü” sonucu çıkarılamaz.

## Konudan dosyaya ve metoda okuma haritası

Tablodaki her satır bir yazılım sorusuna karşılık gelir. Ana attention dosyası `src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py`'dır; bağlantılar ilgili metoda gider.

| Konu | Dosya / metot | Girdi ve sonuç |
| --- | --- | --- |
| Mimari boyutlar | [multi_head_attention.py — MultiHeadAttention.__init__](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L91) | `D,H,H_kv` → doğrulamalar ve Q/K/V/output ağırlıkları. |
| Tüm attention akışı | [MultiHeadAttention.forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L678) | Query/key/value ve seçenekler → projeksiyon, QK-Norm, RoPE, cache, maskeler, çıktı. |
| Maske sözleşmesi | [_prepare_attention_mask](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L372) | Farklı maske biçimleri → skorlarla birleşebilen additive maske. |
| Yürütme seçimi | [scaled_dot_product_attention](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L454) | Q/K/V ve tanı isteği → manuel, PyTorch veya harici Flash yolu. |
| Manuel matematik | [_standard_sdpa_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L613) | Skor, cap, maske, softmax ve dropout → `AV`. |
| PyTorch köprüsü | [_pytorch_sdpa_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L497) | Düzenlenmiş tensörler → SDPA sonucu; ağırlık matrisi dönmez. |
| Harici kernel ve geri dönüş | [_flash_attention_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L532) | Uyumlu giriş → harici çağrı; uygun olmayan koşul/hata → manuel hesap. |
| Geçmişin tutulması | [kv_cache.py — KVCache.update](../../../src/neural_network_module/ortak_katman_module/kv_cache.py#L177) | Yeni K/V ve konumlar → kullanılabilir geçmiş ve kapasite yönetimi. |
| Sonraki tüketici | [transformer_encoder_layer.py — TransformerEncoderLayer._forward_impl](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L532) | Attention çıktısı → residual/normalizasyon ve FFN akışı. |

Okuma sırası `__init__ → forward → seçilen SDPA metodu → çağıran Transformer katmanı` olmalıdır. Sadece sınıf başındaki özellik listesi, bir ayarın etkin olduğunu veya yorumdaki performans iddiasının ölçüldüğünü göstermez. Örneğin sınıf içinde bulunan `norm` alanının varlığı ile mevcut forward yolunda kullanılması ayrı sorulardır. Bu ayrım, kitapta dosya ve metot bağlantısı vermenin yalnız adres belirtmekten daha fazla anlam taşıdığı yerdir.

## Hangi iddia hangi testle sınanır?

[test_cached_matches_full](../../../tests/evolution/test_attention_cache.py#L34), küçük CPU modelinde MHA/MQA/GQA, iki SDPA yolu ve pencere seçenekleri için parçalı cache hesabını tam diziyle karşılaştırır. [test_causal_prefix_and_masked_rows](../../../tests/evolution/test_attention_cache.py#L89) nedenselliği ve kapalı satırları; [test_softcap_effective_with_default_backend](../../../tests/evolution/test_attention_cache.py#L103) soft-cap'in etkinliğini sınar. [Harici Flash başarısızlığı testi](../../../tests/evolution/test_attention_cache.py#L167) taklit edilen hata sonrası fallback'in nedenselliğini denetler; gerçek GPU Flash kerneline ait hız veya kalite ölçümü değildir.

Bir okuma deneyi olarak üç token için causal izin matrisini çizin. Sonra ikinci tokenın sorgusunun hangi value vektörlerini toplayabildiğini kaynak maskesiyle karşılaştırın. Aynı deneyi cache'e beş token eklenmişken tek yeni token için tekrarlayın: query'nin yerel indisi sıfır olsa da mutlak konumu beştir.

[Mevcut modül belgesi](../../modules/neural_network/README.md) bu bölümün genel haritasını sağlar. [Eski attention incelemesi](../../module_audits/4_attention_audit.md) korunmuştur; ancak oradaki sıcaklık çarpanı ve her attention satırının toplamının bir olduğu yönündeki genellemeler güncel uygulamaya taşınmamıştır. Burada davranışın otoritesi bağlantısı verilen çalışır kaynak ve somut test koşullarıdır.

## Çözümlü alıştırmalar

Bu bölümün üç tokenlık hesabı, [çalıştırılabilir örnekte](../../../scripts/book_neural_walkthrough.py) gerçek `scaled_dot_product_attention` metoduna önceden seçilmiş Q/K/V verilerek yeniden üretildi. Ayrıca iki katmanlı küçük Cevahir modelinde `H=4`, `H_kv=1,2,4` için tam dizi–parçalı cache ve SDPA–manuel çıktı eşdeğerliği kontrol edildi. Cache anahtarlarının şekilleri her katmanda `[2,H_kv,6,4]` oldu. Gelecekteki üç tokenı değiştirmek önceki üç konumun çıktısını değiştirmedi; bütünüyle kapalı attention satırları sıfır çıktı verdi. [Sayısal kayıt](../evidence/neural_walkthrough.json), hata değerlerini ve toleransları içerir. CPU deneyi gerçek GPU FlashAttention yürütmesi değildir.

**1. Sıcaklık:** Elle hesaplanan ikinci satırda `τ=2` olursa ağırlıklar nedir? Skorlar `[1,2]` olur; `a_21=1/(1+e)≈0.268941421`, `a_22≈0.731058579`. Çıktı yaklaşık `2.462117157`'dir. Artan sıcaklık bu örnekte dağılımı eşitliğe yaklaştırır; token örnekleme sıcaklığını değiştirmiş olmayız.

**2. Pencere:** Mutlak query konumu `5`, kaynak konumları `0..5`, `sliding_window=3` ve nedensellik açık olsun. İzin verilen kaynaklar hangileridir? `5−j<3` ve `j≤5` koşullarından `j∈{3,4,5}` çıkar. Query'nin yeni parçadaki yerel indisi `0` olsa da cevap değişmez.

**3. Bellek:** Tablodaki GQA örneğinde batch `1` yerine `4` olursa minimum cache yükü nedir? `4×512 MiB=2 GiB`. Bu değer ağırlıkları veya attention matrisini içermez. Aynı oranda üretim süresi artacağını söylemek için cihaz ve çalıştırma ölçümü gerekir.

**4. Açıklama:** İki izinli value da `[2,−1]` ise `[0.9,0.1]` ve `[0.2,0.8]` ağırlıkları hangi çıktıları verir? İkisi de `[2,−1]`. Ağırlık haritasının değişmesi çıktı değişimi için tek başına yeterli değildir.

**5. Yürütme yolu:** `use_pytorch_sdpa=True`, `attn_logit_cap=0.1`, ağırlık isteme kapalı olsun. Kaynak hangi hesabı seçer? Pozitif cap nedeniyle manuel hesap. Bayrak adı ile gerçekleşen işlem ayrımını `scaled_dot_product_attention` içindeki ilk koşullardan okuyabiliriz.

## Kaynakça

Bu listedeki çalışmalar yöntemin kökeni ve karşılaştırma bağlamı içindir. Cevahir'e ilişkin uygulama iddiaları yukarıdaki yerel kod ve test bağlantılarıyla, öğretim sayıları ise açık hesaplarla desteklenir.

- Bahdanau, D., Cho, K. ve Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR. [Özgün makale](https://arxiv.org/abs/1409.0473).
- Vaswani, A. ve diğerleri (2017). *Attention Is All You Need*. NeurIPS 30. [Özgün makale](https://arxiv.org/abs/1706.03762).
- Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv:1911.02150. [Özgün çalışma](https://arxiv.org/abs/1911.02150).
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F. ve Sanghai, S. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP, 4895–4901. [Konferans makalesi](https://aclanthology.org/2023.emnlp-main.298/).
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. ve Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 35. [Özgün makale](https://arxiv.org/abs/2205.14135).
- Touvron, H. ve diğerleri (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288. [Teknik rapor](https://arxiv.org/abs/2307.09288).
- Jiang, A. Q. ve diğerleri (2023). *Mistral 7B*. arXiv:2310.06825. [Teknik rapor](https://arxiv.org/abs/2310.06825).
- Jain, S. ve Wallace, B. C. (2019). *Attention is not Explanation*. NAACL-HLT, 3543–3556. DOI: 10.18653/v1/N19-1357. [Konferans makalesi](https://aclanthology.org/N19-1357/).
- Wiegreffe, S. ve Pinter, Y. (2019). *Attention is not not Explanation*. EMNLP-IJCNLP. [Özgün makale](https://arxiv.org/abs/1908.04626).
- PyTorch contributors. *torch.nn.functional.scaled_dot_product_attention*. [Resmî API belgesi](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html), erişim: 21 Eylül 2026; API davranışı kullanılan sürümle ayrıca denetlenmelidir.
