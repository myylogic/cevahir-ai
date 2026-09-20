# Hata neyi değiştirmeyi gerektiriyor? Ayırt edilebilirlik ve korunması gereken geçmiş

20 Eylül 2026 · Önceki araştırmaya eklenen bağımsız teori incelemesi

**Sonuç:** Bir tahmin hatası, mevcut hesabın gözlenen sonuçla uyuşmadığını gösterir. Tek başına; katsayının eskidiğini, geçmişten bir durum değişkeninin eksik bırakıldığını veya geri bildirimin yanlış ölçüldüğünü/yanlış olaya bağlandığını söylemez. Bazı ayrımlar korunmuş olay sırası ve sonraki doğal gözlemlerle yapılabilir. Bazıları aynı gözlem kanalından, geçmişin tamamı korunsa bile yapılamaz. Bununla birlikte bir sistem, fiziksel nedenin tek açıklamasını bulmadan daha iyi bir öngörü durumu öğrenebilir.

Bu not ana yaşarken öğrenme problemine yeni bir sınırlı parça ekler. Başarısızlığın teşhisini genel öğrenmenin tamamı kabul etmez; bütün dünyalarda sıfır hata veya zorunlu yeniden sınama şartı getirmez. Aşağıdaki sonlu örnekler doğrudan türetimdir; yeni bir genel imkânsızlık teoremi iddiası değildir.

## 1. Önce üç farklı soruyu ayırmak gerekiyor

1. **Tahmin yeterliliği:** Erişilebilir geçmiş, bir sonraki sonuç için gereken ayrımı içeriyor mu?
2. **Düzeltme yeterliliği:** Tutulan kayıt, bir aday güncellemeyi veya yeni temsil biçimini değerlendirmeye yetiyor mu?
3. **Nedenin belirlenmesi:** Aynı gözlemi üreten farklı fiziksel açıklamalar arasında seçim yapılabiliyor mu?

İlkinde başarı diğer ikisini otomatik olarak vermez. Önceki durum taşıma deneyindeki “eski tahmin özeti yeni özellik için yeterli olmayabilir” sonucu korunur. Öğrenilmiş güncelleme turu da aynı bugünkü cevabın aynı yarınki öğrenme yolunu belirlemediğini gösterdi. Buradaki ek soru, gözlenen hatanın hangi düzeltme ailesine kanıt sağladığıdır.

“Parametre değişimi” ve “durum değişimi” bütünüyle gözlemden bağımsız doğal türler değildir. `s[t+1] = F(s[t], x[t])` yordamındaki `s`ye değişen parametre denebilir. Parametre güncellemesine sınırsız hesap ve geçmiş erişimi verildiğinde, bir yinelenen durum makinesi de bu biçimde yazılır. Karşılaştırma; isimlerle değil, hangi bilginin tutulduğu, güncellemenin hangi girdilere eriştiği, işlem ailesi ve hesap maliyetiyle tanımlanmalıdır.

## 2. Aynı geri bildirim, iki zıt yararlı düzeltme

İkili girdi `x` ve sistemin değerlendirilmesi gereken gerçek hedef `q` olsun. Öğrenici yalnız `(x, y)` çiftlerini görüyor; `y` geri bildirim kanalıdır. Önceden doğru kural `q=x` idi. Değişimden sonra aşağıdaki dünyalar her girdi dizisi için aynı gözlemi üretir:

| Dünya | Gerçek hedef | Geri bildirim | Öğrenicinin gördüğü |
|---|---|---|---|
| A: hedef değişti | `q = 1-x` | `y=q` | `y=1-x` |
| B: kanal ters çevrildi | `q = x` | `y=1-q` | `y=1-x` |

Amaç `q`yu doğru üretmekse, A'da davranışı değiştirmek; B'de eski davranışı koruyup kanalı düzeltmek yararlıdır. Bir sonraki aynı girdiye verilen tek cevap iki dünyada da doğru olamaz. Bu, mevcut kayıt için teşhis belirsizliğidir; hiçbir öğrenme yapılamayacağı sonucu değildir.

**Kritik ayrım:** Amaç yalnız gözlenen `y`yi öngörmekse, iki dünyada aynı yeni kural doğrudur. O zaman fiziksel açıklamayı ayırmaya gerek yoktur. Dolayısıyla “farklı yararlı düzeltmeler” iddiası, başarı ölçütünün `q` ile ilgili olması sayesinde geçerlidir; görünmeyen bir hedefi sessizce ölçüte eklemiyoruz.

Aynı kanaldan daha uzun geçmiş bu örneği çözmez. Güvenilir referans ölçümü, bilinen kanal davranışı veya başka bir bağlantı varsayımı çözebilir. Örneğin ikinci ölçümün bu iki dünya boyunca `q`yu doğru verdiği biliniyorsa tek uyuşmazlık yeterlidir. Yalnız “ikinci bir sensör var” demek yetmez; iki sensör ortak biçimde bozulabilir. Bu ek güvenilirlik bir varsayımdır, öğrenicinin kendiliğinden bulduğu bilgi değildir.

## 3. Hedef değişimi ile yanlış zaman atfı: geçmiş ne zaman yardım eder?

`x[1:3] = (0,1,0)` olsun. İlk adımın etiketi yok; ikinci ve üçüncü adımda gelen geri bildirim `(0,1)` olsun. İki açıklama bu sonlu kayda tam uyar:

- A: yeni anlık hedef `y[t] = 1-x[t]`.
- B: hedef hâlâ `q[t]=x[t]`, fakat etiket `y[t]=q[t-1]` olarak bir adım gecikiyor; olay kimliği verilmemiş.

Girdi sonsuza kadar `0,1,0,1,...` biçiminde giderse açıklamalar bütün gözlemlerde eşleşir. Daha çok aynı örüntü ve daha büyük katsayı güncellemesi ayrımı üretmez.

Girdi doğal olarak `x[4]=0` olursa tahminler ayrılır: A `1`, B `0` der. Gürültüsüz ve yalnız bu iki adayın geçerli olduğu koşulda, dördüncü geri bildirim ayırıcıdır. Öğrenicinin en azından önceki girdiyi ve gözlem sırasını tutması gerekir. Bu, aktif deney istemeden ortaya çıkan bir ayırıcı devam örneğidir. Gürültü varsa tek örnek kesin teşhis değildir; kanal için olasılıksal varsayım ve tekrarlanan ayrıştırıcı olaylar gerekir.

Gecikmeyi hesaba katan bir tahminci, bu örnekte fiziksel “etiket gecikmesi” açıklamasını kanıtlamadan da hizmet hatasını azaltabilir. Aynı ilişkiler, gerçek bir geçmiş bağımlılığı altında da ortaya çıkabilir.

### Aynı sonlu kayıt, dışarıdan değişen katsayı veya eksik yinelenen durum

Girdi dizisi `x[1:6]=(0,1,0,1,0,1)`, çıktı dizisi `y[1:6]=(0,1,1,0,0,1)` olsun. A dünyasında `z[0]=0`, `z[t]=z[t-1] XOR x[t]`, `y[t]=z[t]` kuralı vardır. B dünyasında `y[t]=x[t] XOR b[t]`; dışarıdan belirlenmiş katsayı dizisi ilk altı adımda `b=(0,0,1,1,0,0)`, sonraki adımlarda 0'dır. İki dünya aynı mevcut kaydı verir.

Sonraki girdiler `(1,0)` olduğunda A'nın çıktısı `(0,0)`, B'ninki `(1,0)` olur. A'da mevcut girdiyi düzeltmek için önceki parity durumunu taşıyan yordam yararlıdır. B'de bu yordamın taşınması yanlış tahmin verir; son geçerli anlık kural `y=x` yararlıdır. İki dünyayı aynı sonlu kayıttan kesin ayırmak mümkün değildir. B'nin katsayısının gelecekte sabit kalması örneğin açık dünya tanımıdır; veriden kanıtlanmış özellik değildir. Model sadeliği veya önsel tercih bir karar sağlayabilir, fakat gözlemsel kanıtla tekil teşhis anlamına gelmez.

## 4. Tutulan özet sonradan gereken zaman ilişkisini silebilir

Aşağıdaki iki dört adımlık kayıt aynı sırasız `(x,y)` çiftlerine sahiptir:

| Kayıt | Girdi dizisi | Geri bildirim dizisi |
|---|---|---|
| A | `0,0,1,1` | `0,0,0,1` |
| B | `0,1,0,1` | `0,0,0,1` |

Her ikisinde `sum(x)=2`, `sum(y)=1`, `sum(x²)=2`, `sum(xy)=1`, `sum(y²)=1`. Sabit terimli güncel doğrusal tahmin için tutulan normal denklem özeti de aynıdır. Fakat bir adım geçmiş özelliği sonradan eklendiğinde gereken

`C1 = sum(t=2..4) x[t-1] y[t]`

A'da `1`, B'de `0` çıkar. A kaydı, başlangıç girdisi `x[0]=0` alınırsa gürültüsüz `y[t]=x[t-1]` kuralına uyar; B uymaz.

Dolayısıyla güncel çiftleri kusursuz özetlemek, gecikmeli adayın yeniden değerlendirilmesini korumaz. Ham sıralı kayıt yeterlidir; bu aday için önceden tutulmuş `C1` de yeterlidir. Her gelecekteki temsil için ham geçmişin tamamının zorunlu olduğu sonucu çıkmaz. Ne saklanması gerektiği, daha sonra desteklenmek istenen düzeltme ailesine bağlıdır.

## 5. Sabit pencereyi büyütmek ile yinelenen durum öğrenmek aynı kapasite değildir

Sonlu bir tanık: girdiler `SET0`, `SET1`, `WAIT`, `QUERY` olsun. Dünyanın bir biti `z` vardır. İlk iki sembol biti sırasıyla 0 ve 1 yapar; `WAIT` ve `QUERY` değiştirmez. `QUERY` çıktısı `z`, diğer bütün çıktıların değeri 0'dır. Başlangıç durumu bu karşı örnek için önemli değildir; `SET` onu belirler.

Her sonlu `K` için iki geçmiş seçilebilir:

`h0 = SET0, WAIT^K`

`h1 = SET1, WAIT^K`.

Bu geçmişlerin son `K` **tam girdi/çıktı çifti** aynıdır: yalnız `(WAIT,0)`. Sonraki girdi `QUERY` olduğunda doğru cevaplar sırasıyla 0 ve 1'dir. İki geçmiş eşit olasılıkla gelirse yalnız bu pencereyi kullanan bir sınıflandırıcının bu sorgudaki beklenen hatası en az `1/2` olur. Olasılık veya kare hata cevabına izin verilirse en iyi ortak cevap `1/2`, kare hata `1/4` olur.

Bir bitlik yinelenen durum ve öğrenilmiş geçiş/çıktı yordamı bütün bekleme uzunluklarını karşılar. **Bu yalnız temsil yeterliliği ispatıdır; yordamın deneyimden edinildiğini tek başına göstermez.** Öğrenme deneyi, sembollerin anlamını veya doğru geçişleri hazır verdiğinde edinim iddiası zayıflar. Algoritmanın mevcut dilinden hangi yordamı ve hangi kanıtla seçtiği ayrıca ölçülmelidir.

Bu örneğin adil karşılaştırma koşulu önemlidir: önceki çıktılar da pencereye verilmiştir ve `WAIT` çıktıları gizli biti açıklamaz. Her adımın etiketi biti açıkça gösterseydi, son etiket tek başına yeterli olabilirdi; yalnız girdileri alan pencereyle yapılan karşılaştırma durum keşfi kanıtı sayılmazdı.

Bilinen doğru yordam ile mevcut bit de farklı şeylerdir. Çalışma bitini silmek, yeni `SET` gelene kadar başarısızlık doğurabilir; yordamı silmek ise sonraki olayların nasıl yorumlanacağını kaldırır. “Yalnız mevcut durumu hatırlama” ile “yeni olayları işlemeyi öğrenme”yi ayırmak için her ikisine ayrı silme/taşıma müdahaleleri uygundur.

## 6. Yapıcı ilke adayı: Geçmişleri gelecekte gerekli ayrımlara göre birleştirmek

Deterministik, sabit bir girdi/çıktı süreç ailesinde `h ~ h'` ilişkisini şöyle tanımlayalım: izin verilen her sonlu girdi devamı, iki geçmişten sonra aynı çıktı dizisini üretir. Bu durumda eşdeğer geçmişleri tek duruma koymak öngörü için bilgi kaybettirmez. Eşdeğerlik güncelleme altında korunur; aynı sonraki girdi ve çıktı, aynı yeni sınıfa geçişi tanımlar. Sonlu sayıda sınıf varsa sonlu bir yinelenen temsil yeterlidir.

Bir çift geçmişten sonra aynı devam farklı çıktı verirse, bunları birleştirmiş temsil yanlışlanır. Bölüm 5'te `QUERY` bu ayırıcı devamdır; en az iki durum gerekir ve iki durum yeterlidir. Olay geçmişi, yalnız tekrar oynatılacak kayıt olmaktan çıkarak yeni olayların hangi duruma geçiş yaptığını belirleyen yürütülebilir bir özete dönüşebilir.

Ancak sınırlı veride hiçbir ayırıcı devam görmemek, geçmişlerin bütün gelecekler için eşdeğerliğini kanıtlamaz. Sonlu bir model sınırı, ayırıcı devamların yeterli kapsamı, süreç kararlılığı veya olasılıksal varsayımlar olmadan “iki durum bulundu, gerçek makine keşfedildi” denemez. Gürültülü süreçlerde eşitlik çıktı değerleri yerine koşullu gelecek dağılımları üzerinden kurulur; bunları öğrenmek ayrı bir istatistiksel problemdir.

Bu yapıcı fikir, bilinen öngörü durumu/otomat öğrenme ailesine aittir. Burada yeni temel ilke olarak sunulmuyor. Ana yaşam boyu öğrenme sorusuna katkısı; yalnız katsayı düzeltmek yerine, bir sonraki gözlemin hangi iç durumu güncelleyeceğini belirleyen yordamın deneyimle edinilebileceğini somutlaştırmasıdır.

## 7. Gürültüye ilişkin olumlu ayrımın varsayımları

Bir aday `f` için artık `e[t]=y[t]-f(x[t])` olsun. Eğer ölçüm gürültüsünün erişilebilir geçmiş ve mevcut girdi koşullu ortalamasının sıfır olduğu, hedef ilişkisinin ilgili aralıkta sabit kaldığı ve yeterince tekrarlanan ayırıcı geçmişler bulunduğu varsayılıyorsa, sistematik `E[e[t] | h[t-1],x[t]] != 0` mevcut tahmincinin eksik kaldığına kanıt verir. Geçmişe bağlı yeni bir tahmin özelliği bu kalanı azaltabilir.

Bu koşullar altında bile “eksik fiziksel değişken kesin budur” sonucu çıkmaz. Geçmişe bağlı sensör yanlılığı, aynı koşullu ortalamayı üretebilir. Ayrıca sonlu veride gözlenen küçük bir ortalama fark gerçek koşullu beklenti farkı değildir. Buradaki ifade belirli bir istatistiksel test veya güven düzeyi garantisi değildir; gözlemin hangi varsayım altında teşhis değeri taşıdığını ayırır.

## 8. Birincil literatür eşleşmeleri ve sınırları

- **Littman, Sutton ve Singh — Predictive Representations of State (NIPS 2001).** Durumu, eyleme koşullu gelecek gözlem tahminleriyle temsil eder; sabit uzunluklu geçmiş ile yinelenen durumun ayrımını açıkça inceler. Makaledeki float/reset örneği, hiçbir sabit pencerenin tam yeterli olmadığı bir süreç verir. Bu nedenle bölüm 5'in kapasite ayrımı yeni değildir. Temsil yeterliliği, çevrimiçi keşif başarısı ve bütün yaşam maliyeti ayrı iddialardır. [Birincil makale](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf).

- **Shalizi ve Crutchfield — Computational Mechanics: Pattern and Prediction, Structure and Simplicity (2001; açık ön baskı 2000).** Geçmişleri aynı gelecek dağılımını vermelerine göre birleştiren durumları ve öngörü/minimallik sonuçlarını geliştirir. Bölüm 6'nın stokastik eşdeğerlik fikri doğrudan bu literatürle eşleşir. Buradaki “causal state” terimi, hedef değişimi ile bozuk sensörü ek varsayımsız fiziksel olarak ayıran bir müdahale teoremi değildir. [Birincil açık metin](https://arxiv.org/pdf/cond-mat/9907176).

- **Allman, Matias ve Rhodes — Identifiability of parameters in latent structure models with many observed variables (2009).** Belirli sonlu, durağan gizli Markov modellerinde gözlenen ardışık değişkenlerin ortak dağılımından parametrelerin genel konumdaki belirlenebilirliğini gösterir. Teorem 6, `r` gizli ve `κ` gözlenebilir durum için `binomial(k+κ-1, κ-1) >= r` koşulunda `2k+1` ardışık değişkenin dağılımını kullanır. Bu, yalnız `2k+1` tekil gözlemle doğru model bulunur demek değildir; dağılım bilgisi, model ailesi ve istisnai parametreler ayrımı gerektirir. Gizli durum adları da gözlemle sabitlenmez. [Birincil açık metin, Teorem 6](https://arxiv.org/pdf/0809.5032).

- **Sampath ve diğerleri — Diagnosability of Discrete-Event Systems (1995).** Kısmen gözlenen olay sistemlerinde arıza teşhisi için biçimsel dil ve tanılayıcı yapılarıyla belirlenebilirlik koşulları verir. Aynı gözlenen davranışla uyumlu arızalı ve arızasız açıklamaların ayrılması, bölüm 2–3'ün doğrudan yakın alanıdır. Fakat verili olay modeli içinde teşhis edilebilirliği denetlemek, yaşarken doğru model dilinin keşfedilmesi değildir. Bu notta makalenin yayıncı özeti doğrulandı; bütün teknik teoremleri yeniden uygulanmadı. [Birincil yayıncı kaydı](https://ieeexplore.ieee.org/document/412626/).

## 9. Araştırma için karar

“Önce hatanın gerçek türünü kesin teşhis et, sonra öğren” zorunlu bir yaşam boyu öğrenme ilkesi olarak desteklenmiyor. Fiziksel açıklama tek olmayabilir; buna rağmen daha iyi işleyen bir öngörü durumu edinilebilir. Tersi de önemlidir: artık hatanın azalması, geri bildirimin anlamının veya yeni açıklamanın doğruluğunu kanıtlamaz.

Bu turun uygun deney iddiası şu ölçekte kalmalıdır: **Verilen deneyim akışından edinilmiş ve saklanmış bir geçiş yordamı, aynı erişilebilir veriyi kullanan belirli sabit pencere sınıflarının koruyamadığı bir gelecek ayrımını koruyabiliyor mu?** Bunun için edinim maliyeti, dilin önceden verilmiş kısmı, etiket erişimi, yordam/durum silme farkı ve yanlış yapı kabulü birlikte raporlanmalıdır. Yanıt olumlu olsa bile, genel temsil keşfi, bozuk geri bildirim teşhisi ve birleşik kaynak yönetimi açık kalır.

Önceki araştırma sonuçları bu notla geri alınmıyor. Yeni ayrım, onların kapsamını genişletirken bir sınır ekliyor: öğrenilmiş bir güncellemenin hangi yönde uygulanacağı kadar, gözlemin hangi geçmiş durumla ve hangi geri bildirim anlamıyla ilişkilendirildiği de gelecekte öğrenilebilir olanı etkiler.
