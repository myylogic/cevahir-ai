# Yaşam sırasında yararlı temsil ve hesaplama oluşturma: sınırlı deney

20 Eylül 2026. Bu çalışma, önceki doğrusal kural öğrenme tanığının kapsamını genişletir. Eski deneyler ve negatif sonuçlar korunur. Ana Cevahir sisteminde değişiklik yapılmamıştır.

**Sonuç:** Başlangıçta yalnız sabit ve doğrusal özellikler kullanan küçük bir öğrenici, yaşadığı etiketli deneyimlerden çarpım özellikleri oluşturdu; daha sonraki bir görevde önceki çarpımı başka bir girdiye bağlayarak üçlü etkileşimi çalıştırılabilir bir hesaplama olarak edindi. Görülmemiş girdilere aktardı ve bağımsız eski görevlerini korudu. Ancak aynı ifade gücündeki sabit genişletilmiş temsil aynı başarıyı sağladı; aynı etkinleştirme kuralına sahip güçlü sabit karşılaştırma, öğrenme ve hesap sayaçlarında da tam eşleşti. Dolayısıyla sonuç, temsil büyümesinin mümkün ve bu doğrusal başlangıca göre yararlı olduğunu gösterir; sabit bir mimarinin yetersizliğini veya yeni bir temel algoritmayı göstermez.

Kod: [representation_growth.py](../../../research/living_learning_reassessment/representation_growth.py). Makine kaydı: [representation_growth.json](../../../research/living_learning_reassessment/results/representation_growth.json).

## Sorunun daraltılmış biçimi

Önceki affine tanıkta işlemin ailesi verilmişti ve yalnız katsayılar ediniliyordu. Burada değişen durum, çıktı katsayılarıyla birlikte yürütülen özelliklerin tanımları ve aralarındaki bağımlılıklardır. Araştırılan önerme şöyledir:

> Etiketli deneyim, mevcut temsilin açıklayamadığı düzenliliği ortaya çıkarınca sistem, verilmiş bir oluşturma dilinde yeni bir özellik tanımı kurup bunu gelecekteki hesaplamalarında ve sonraki özelliklerin kurulmasında kullanabilir.

Bu önerme yeni ilkel işleç icat edildiğini söylemez. Çarpma işleci, girdi koordinatları, sonlu alan, model seçimi ve doğrulama kuralı araştırmacı tarafından sağlanır. Keşfedilen şey, hangi koordinatların hangi yapı içinde birbirine bağlanacağı ve bunların görevler için hangi katsayılarla kullanılacağıdır.

Bu, eski [tanıkla filtreleme denetiminin](../REPRESENTATION_WITNESS_AUDIT.md) yeniden adlandırılması değildir. O denetimde aynı aday kümesi üzerinde zorunlu bir filtre eklemek tamamlanan aramanın sonucunu değiştirmiyordu. Burada sınırlı başlangıç temsilinin etkin fonksiyon ailesi büyür. Buna rağmen bütün ifadeleri baştan içeren güçlü kontrolün eşitliği de korunur; eski negatif sonuç aşılmış gibi sunulmaz.

## Matematiksel ve hesaplamalı mekanizma

Girdi `x ∈ F_101^5`. Başlangıç özellikleri `Φ₀ = (1, x₀, x₁, x₂, x₃, x₄)`. Görev kimliği `j` gözlenebilir; her görevin ayrı doğrusal başlığı vardır:

```text
f_j(x) = Σ_{φ ∈ Φ_t} w_{j,φ} φ(x)   (mod 101)
```

Etiketli gözlemler tek tek gelir. İlk 16 gözlemde mevcut özelliklerle tam tutarlı bir başlık aranır. Yoksa aday oluşturucu, kabul edilmiş bir özelliği içermediği bir koordinatla çarpar. Üçten yüksek derece ve koordinat tekrarları yasaktır:

```text
G(Φ_t) = { φ(x) · x_i : φ ∈ Φ_t, i ∉ variables(φ), degree(φ) < 3 } \ Φ_t
```

Adaylar derece ve koordinat sırasıyla gezilir. Her aday için sonlu alanda Gauss elemesiyle `Φ_t ∪ {φ_new}` üzerinde başlık uydurulur. İlk tutarlı aday sonraki dört yeni etiketli gözlemde sınanır. Bu dört gözlem arama için kullanılmaz. Doğrulama geçerse özellik ve başlık kabul edilir; öğrenme penceresinin ham satırları silinir. Başarısızlıkta model kabul edilmez. Tek bir görev için en fazla 20 ham satır tutulur; deney görevleri sıralı olduğundan ölçülen toplam tepe de 20 satırdır.

Bir özellik yalnız isim değildir. Öğrenilmiş düğümün iki ebeveyni tutulur ve tahmin sırasında bunlar yürütülüp çarpılır. Örneğin `φ₆ = x₃·x₄`, ardından `φ₇ = φ₆·x₀`. `φ₆` fiziksel olarak silinince hem onu kullanan başlık hem `φ₇` yürütülemez. Değerlendirici bu eksikliği çekimserlik olarak sayar. Bu, belirli temsilin gelecekteki hesaplamaya nedensel katkısını sınar.

Bir görevin yeni etiketi kabul edilmiş başlığıyla çelişirse o başlık hemen geri çekilir; yeni gözlemlerle yeniden edinilir. Diğer görev başlıkları ve kabul edilmiş özellik fonksiyonları değiştirilmez. Buradaki özellikler bir dünya hakkındaki yanlışlanabilir olgular değil, doğruluğu tanımından gelen çarpım fonksiyonlarıdır. Dünya değişince bunların hangi görev için geçerli olduğuna ilişkin katsayılar değişir. Gürültü ve gerçek değişimi ayıran bir mekanizma bu deneyde yoktur.

Doğrusal başlangıcın yetersizliği yalnız ölçümle ileri sürülmez. `a + b x₀ + c x₁` biçimindeki her fonksiyonun `(0,0), (1,0), (0,1), (1,1)` karesi üzerindeki karışık farkı sıfırdır. `x₀x₁` için aynı fark birdir. Dolayısıyla doğrusal başlık bu fonksiyonu alanın tamamında temsil edemez. Bu kanıt, doğrusal başlığın sınırlılığına aittir; girdinin kendisinin bilgi kaybettiğini veya bütün sabit temsillerin yetersiz olduğunu söylemez.

## Önceden belirlenen deney ve kontroller

20 tohumda beş koordinatın rolleri ve sıfır olmayan etkileşim katsayıları değiştirildi. Her yaşam sırası üç görev içerir: bir ikili çarpım, bu çarpımı kullanan üçlü etkileşim, diğer iki koordinatın bağımsız ikili çarpımı. Her görev ayrıca sabit ve doğrusal terimler içerir. Bütün yöntemler aynı 20 etiketli gözlemi alır; ayrı 100 girdi üzerinde değerlendirilir. Bir tohumdaki eğitim ve değerlendirme girdileri, görevler arasında da tam olarak ayrıdır. Değerlendirme sorguları durum güncellemez; etiketler öğreniciye verilmez.

| Yöntem | Başlangıç ve seçme kuralı | Kontrol ettiği açıklama |
|---|---|---|
| Sabit doğrusal | Altı temel özellik; genişleme yok | Ek hesaplama özelliği gerçekten gerekli mi? |
| Büyüyen temsil | Altı temel özellik; kanıt sonrası çarpım düğümü oluşturma | Deneyimle etkin temsil değişiyor mu? |
| Sabit geniş, seyrek arama | Derece ≤3 olan 26 özellik baştan mevcut; aktif küme + bir aday şeklinde aynı seyrek başlık araması | Aynı ifade gücündeki sabit alternatif de öğrenebilir mi? |
| Sabit geniş, aynı etkinleştirme | Aynı 26 özellik baştan mevcut; büyüyen yöntemin aday sırası ve etkinleştirme politikası | Fark yalnız tahsis/etkinleştirme düzeni mi? |

Sabit geniş karşılaştırma, 26 serbest katsayıyı 16 örneğe uydurmak zorunda bırakılmadı: aynı seyrek aday seçimi ve dört gözlemlik doğrulama kullanılır. Böylece gereksiz fazla parametreyle zayıflatılmış bir rakipten avantaj çıkarılmaz. Aynı etkinleştirme karşılaştırması daha güçlüdür ve aynı çalışmayı baştan mevcut bir özellik kaydı üzerinde yürütür.

Ek müdahaleler: başlık ve özellik durumunu sıfırlama; öğrenilmiş çarpım düğümünü silme; etiketleri rastgele bozma; doğru başlığa tek yanlış etiket verme; aynı görev kimliğinde kuralın sabit terimini değiştirme; öncül ikili görev olmadan doğrudan üçlü görev öğrenme. Sonuncusu müfredat bağımlılığını açığa çıkarır.

## Ölçülen sonuçlar

| Yöntem | Ayrılmış girdilerde doğru / toplam | Saklanan özellik tanımı | Ortalama JSON durum boyutu | Toplam aday uydurma |
|---|---:|---:|---:|---:|
| Sabit doğrusal | 0 / 6.000 | 6 | 146 bayt | 60 |
| Büyüyen temsil | 6.000 / 6.000 | 9 | 319,65 bayt | 489 |
| Sabit geniş, seyrek arama | 6.000 / 6.000 | 26 | 573,65 bayt | 565 |
| Sabit geniş, aynı etkinleştirme | 6.000 / 6.000 | 26 | 571,65 bayt | 489 |

Sabit doğrusal koşuldaki sıfır sonuçların tamamı çekimserliktir; rastgele tahmin yapan bir rakibin sıfır doğruluğu iddia edilmez. Bütün yöntemlerin ilk göreve ilişkin 20 gözlem sırasında cevap vermeden çekimser kalmasına izin verilmiştir. Başarı, yeterli gözlem ve doğrulama sonrası kazanımdır; her anda kesintisiz yüksek kalitede hizmet değildir.

Büyüyen temsil 20 yaşamda 15 farklı etkin özellik sırası oluşturdu. Bütün eski görev değerlendirmeleri sonraki görevlerin edinilmesinden sonra da 100/100 kaldı. Ham öğrenme penceresi her ana akışın sonunda boştu; JSON üzerinden geri yüklenen öğrenilmiş durum aynı son sonuçları üretti. Burada JSON gidiş dönüşü aynı süreç içinde yapılmıştır; ayrı işletim sistemi süreciyle yeniden başlatma iddiası yoktur.

**En güçlü negatif sonuç:** Büyüyen ve sabit geniş aynı etkinleştirme koşulları, 20 tohumun tamamında tahmin, edinim olayı, koruma sonucu, etkin özellik sırası ve bütün iş sayaçlarında eşitti. Dolayısıyla büyüyen yöntemin daha az dış deneyimle öğrendiği veya bu sabit alternatifi hesapta geçtiği iddiası desteklenmez. Diğer sabit geniş aramadaki 565→489 aday farkı, farklı aday etkinleştirme sırasına bağlıdır. Büyümenin benzersiz bilgi avantajı değildir.

Durum boyutu, bu açık JSON özellik kayıtlarının boyutudur. Sabit alternatif, aynı özelliği işleç ve indekslerden örtük biçimde üretebilir; 26 açık kaydı saklaması kuramsal olarak zorunlu değildir. Bu nedenle tablo genel bellek alt sınırı veya bütün sabit mimarilere karşı bellek üstünlüğü göstermez.

Çalıştırma **16,28 saniye** sürdü; izlenen Python tahsisatlarının tepe değeri **3.219.802 bayt** oldu. Bu sayı toplam süreç RAM'i değildir. İş sayaçları aday oluşturma girişimlerini, uydurmaları, özellik hesaplamalarını, sonlu alan işlemlerini ve değerlendirme hesaplarını kapsar. Python dallanmaları, bütün nesne tahsisleri, veri üretimi veya CPU komutları tam sayılmaz; sayaçlar FLOP veya ölçülmüş hızlanma diye yorumlanmamalıdır. Kayıttaki kaynak SHA-256 değeri kodla ayrıca doğrulandı.

## Olumsuz koşullar ne gösterdi?

Her tohumdaki öğrenilmiş ikili düğümün fiziksel silinmesi, hem ilgili ikili görevde hem ona bağımlı üçlü görevde 100/100 çekimserliğe neden oldu. Bağımsız diğer ikili görev 100/100 kaldı. Durumu tamamen sıfırlamak bütün görevlerin cevaplarını kaldırdı. Bunlar, kayıtlı düğümlerin mevcut yürütme biçimindeki nedensel önemini gösterir; aynı fonksiyonun başka temsilde hesaplanamayacağını göstermez.

Rastgele etiketli blok hiçbir tohumda doğrulanmış model üretmedi. Ancak **tek bir yanlış etiket**, doğru öğrenilmiş başlığı her tohumda hemen kaldırdı ve yeniden kanıt gelene kadar o görevde bütün cevaplar kayboldu. Diğer görevler korundu. Deney bu nedenle gürültüye dayanıklı kontrollü öğrenmeyi çözmüş değildir.

Aynı görevde dünya kuralı gerçekten değişince sistem değişimi ilk çelişkide fark etti: her tohumda bir yanlış cevap ve 19 çekimserlikten sonra yeni kuralı 100/100 uyguladı. Değişen eski kuralda 0/100, bağımsız iki görevde 100/100 çıktı. Eski dünya kuralının kaybı bu koşulda beklenen düzeltmedir. Dünya değişiminin doğru anda ve doğru nedenle tanındığı yönünde genel bir iddia yoktur; yanlış etikete tepki de aynıdır.

**Müfredat karşı örneği:** Önceki ikili özellik olmadan doğrudan üçlü görev gösterilince büyüyen yaklaşım 0/2.000 başarıyla çekimser kaldı. Sabit geniş seyrek arama 2.000/2.000 başardı. Büyüyen aday oluşturucu yalnız kabul edilmiş bir özelliği bir koordinatla çarptığından, ara ikili terim tek başına gözlemleri açıklamıyorsa üçlü terime ulaşamaz. Gözlenmiş yeterlilik hatası, doğru yeni temsili üretmeye kendi başına yetmez. Daha kapsamlı arama bunu çözebilir; onun maliyeti bu çalışmada ödenmiş veya başarıyla gösterilmiş değildir.

## Önceki çalışmalar ve özgünlük sınırı

Yeni betimleyicilerin gözlemlerden kurallarla türetilmesi, yapıcı tümevarımın eski bir konusudur. Michalski'nin çalışması girdi açıklamalarında bulunmayan değişken, bağıntı ve fonksiyonların oluşturulmasını bu kapsamda tartışır. Buradaki sonlu çarpım dili bu temel fikri yeniden adlandırarak yenilik kazanmaz. [Michalski, Pattern Recognition as Rule-Guided Inductive Inference, 1980](https://pubmed.ncbi.nlm.nih.gov/21868911/).

Cascade-Correlation, küçük bir ağdan başlayıp artık hatayla ilişkili yeni birimleri ekler ve bunların giriş bağlantılarını dondurarak ileride kullanılabilir özellikler oluşturur. Burada o neural algoritma uygulanmadı; deneyimle büyüyen ve korunan özellik fikrinin yeni olmadığını gösteren birincil karşılaştırmadır. [Fahlman ve Lebiere, The Cascade-Correlation Learning Architecture](https://proceedings.neurips.cc/paper_files/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf).

iFDD, ikili özelliklerden yeni birleşimleri çevrimiçi kurar; kalıcı geri bildirim hatalarının görüldüğü bölgelerde temsili genişletir. Çalışma, başlangıç özelliklerinde gerekli bilgi yoksa bu genişlemenin de yardımcı olamayacağını açıklar. Bizim deneyimiz RL veya iFDD uygulaması değildir; artımlı özellik birleşiminin ilgili önceki çalışmasıdır. [Geramifard ve diğerleri, Online Discovery of Feature Dependencies, 2011](https://icml.cc/2011/papers/473_icmlpaper.pdf).

Artık hatayı azaltmak için özellik ekleme, benzer özellikleri birleştirme ve değişen akışlara uyarlanma denoising autoencoder tabanlı çevrimiçi özellik öğrenmede de incelenmiştir. Buradaki tanık o çalışmanın sonuçlarını Cevahir'e taşımaz ve özellik birleştirme uygulamaz. [Zhou, Sohn ve Lee, Online Incremental Feature Learning with Denoising Autoencoders, 2012](https://proceedings.mlr.press/v22/zhou12b.html).

## Desteklenen hüküm ve açık sorular

Bu sınırlı sistemde deneyim; yalnız iki katsayının değerini değil, gelecekte yürütülen özellik ağını da değiştirebilir. Sonradan edinilen bir özellik önceki bir özellikten hesaplanabilir; görev başlıklarını ayrı tutmak ve özellik tanımlarını korumak, bu görev ailesinde eski davranışları koruyabilir. Bunlar kontrollü, tekrarlanabilir varlık tanıklarıdır.

Sabit, yeterince ifade güçlü yapı aynı işlevi gerçekleştirebilir. Bu deney yeni bir hesaplanabilirlik sınıfı, sınırsız temsil icadı veya yeni bir öğrenme ilkesi sağlamaz. Gürültü, bilinmeyen görev sınırları, gecikmiş geri bildirim, algıdan özellik çıkarma, gerekçesiz adaylar üzerinden çok adımlı arama ve sürekli görev akışında sonlu bellek yönetimi açık kalır. Başlıklar ayrı tutulduğundan koruma kolaylaştırılmıştır; ortak esnek başlıkta kontrolsüz girişimi çözdüğümüz söylenemez. Görev sayısı arttığında başlık sayısı da artar.

Bir sonraki ayrıştırıcı soru, büyümenin var olup olmadığı değildir: **aynı öğrenme dili, aynı deneyimler ve aynı toplam kaynak altında hangi deneyime bağlı oluşturma/etkinleştirme politikası işe yarar; hangi dağılım veya müfredat değişiminde zarar verir?** Bu soruyu yanıtlamak için başarı ve başarısızlık bölgeleri birlikte ölçülmelidir.

Yeniden çalıştırma:

```powershell
python -m research.living_learning_reassessment.representation_growth
```

Durum: **MEKANİZMA BİLİNİYOR / SINIRLI TEMSİL BÜYÜMESİ DENEYLE DESTEKLİ / GÜÇLÜ SABİT KARŞILAŞTIRMAYLA EŞDEĞERLİK GÖZLENDİ / MÜFREDAT VE GÜRÜLTÜ BAŞARISIZLIKLARI KORUNDU / GENEL YAŞAM BOYU ÖĞRENME ÇÖZÜLMEDİ.**
