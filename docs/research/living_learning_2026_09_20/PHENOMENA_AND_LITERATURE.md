# Ek fenomenler: eylem, yeniden sınama ve öğrenmenin kendi verisini değiştirmesi

**Tarih:** 20 Eylül 2026. Bu not kullanıcının 1–22 numaralı gözlemlerini ve ardından eklediği iç durumdan doğan davranış sapması fenomenini araştırmaya dahil eder. Soba ve engel örnekleri düşünce deneyleri olarak kullanılır. Belirli insan, hayvan, sinir sistemi veya kimyasal mekanizma hakkında bunlardan deneysel sonuç çıkarılmaz.

En önemli kapsam değişikliği şudur: İlk afin ve Boole deneylerinde yeni sorguları veya öğretici geri bildirimi değerlendirici veriyordu. Bu deneyler **deneyim → öğrenilmiş durum → davranış** bağlantısını sınadı. **Davranış → gelecekte gözlenebilecek veri → sonraki öğrenme** bağlantısını sınamadı. İlk deneylerdeki başarı, kendi kaçınması nedeniyle veri alamayan bir sistemin eski bir sınırlamayı aşabileceğinin kanıtı değildir. Yeni kapalı döngü deneyi bu eksikliği çok küçük bir karar probleminde hedefler; ayrıntılı çalışma kaydı [CLOSED_LOOP.md](CLOSED_LOOP.md) dosyasındadır.

## 1. Mimari önermeden problemin kapsamını genişletmek

Aşağıdaki değişkenler modül sayısı veya yazılım mimarisi önermez; ayırt edilmesi gereken nedensel rolleri gösterir:

- `w_t`: dış dünyanın durumu;
- `c_t`: sistemin o andaki gerçek yapabilme kapasitesi;
- `m_t`: geçmiş deneyimin sonraki hesaplamayı etkileyen taşınabilir durumu;
- `z_t`: zaman, geçici dürtü, hesaplama evresi gibi içsel durum;
- `o_t`, `a_t`, `r_{t+1}`: gözlem, eylem ve değerlendirme sonucu.

Bir olası model:

\[
o_t\sim O(\cdot\mid w_t,c_t),\qquad
a_t\sim\pi(\cdot\mid o_t,m_t,z_t),
\]
\[
(w_{t+1},c_{t+1},r_{t+1})\sim K(\cdot\mid w_t,c_t,a_t),
\]
\[
(m_{t+1},z_{t+1})=U(m_t,z_t,o_t,a_t,r_{t+1},\kappa_t).
\]

`κ_t` kullanılabilir iç hesaplamayı temsil eder. Dış etkileşim olmayan bir adımda yeni dış sonuç yoktur; buna rağmen `U` eldeki durum üzerinde çalışabilir. `c` ile `m` aynı fiziksel yapıda kodlanabilir. Rollerinin ayrı yazılması, ayrı depolama gerektirmez.

Tam etkileşim geçmişi `H_t` için sonraki verinin yasası eylem seçimine bağlıdır:

\[
P(e_{t+1}\mid H_t)=
\sum_a\pi_t(a\mid H_t)\,P(e_{t+1}\mid H_t,a).
\]

Bu eşitlik, kendisini başarıyla değiştiren bir öğrenicinin gelecekte alacağı örneklerin dağılımını da değiştirdiğini gösterir. Veri akışını sabit kabul eden bir öğrenme sınaması bu etkiyi dışarıda bırakır. Eylemlerin gözlenecek etiketleri seçmesi, önceki çalışmalarda da açık bir problem olarak bulunur: Lakkaraju ve arkadaşları, yalnız mevcut kararların görünür kıldığı sonuçlardan model değerlendirmeyi inceler. Buradaki bariyer örneği onların uygulaması değildir; aynı veri seçimi sorununun daha küçük bir örneğidir. [Lakkaraju ve diğerleri, 2017, giriş ve §3](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf).

## 2. Az gösterim, taklit ve kalıcı öğrenme ayrı ölçülmeli

Bir gösterimden sonra bir hareketi üretmek, o hareketin ertesi gün yeni başlangıç koşullarında yapılabildiğini göstermez. Tersine, davranışın taklitle edinilmesi onu öğrenme olmaktan çıkarmaz. Bu araştırma için yeterli operasyonel ayrım şudur: Gösterim sona erdikten, geçici gösterim erişimi kaldırıldıktan ve belirlenmiş gecikme/ara görevlerden sonra, aynı değerlendirme koşullarında gösterime maruz kalmış sistemin başarısı değişiyor mu? Bu etki öğrenilmiş durumun sıfırlanmasıyla kayboluyor ve aktarılmasıyla taşınıyor mu? Ölçüt sonsuz değişmezlik değil, ilan edilen süre ve koşullarda dayanıklılıktır.

Tek gösterimden robotik göreve uyumun somut önceki örneği vardır. Finn ve arkadaşları önceki görevlerde meta-eğitimden sonra yeni görevlerde bir görsel gösterimle uyumu çalışır. Bir gösterim, çok sayıda zaman adımı içeren bir yörüngedir; tek atomik bit değildir. Önceki görevlerin eğitim maliyeti de sıfır değildir. Makale, kendi hatalarıyla farklı durumlara sürüklenen politikanın hata birikimi sorununu çözmeyi hedeflemediğini açıkça belirtir. Bu sonuç, az gösterimle uyumun mümkün olduğunu destekler; sınırsız yaşam boyunca kalıcılık, kendi deneyimiyle düzeltme veya tüm görev ailelerine aktarım sonucu değildir. [Finn ve diğerleri, CoRL 2017, §2–3](https://proceedings.mlr.press/v78/finn17a/finn17a.pdf).

DAgger, taklitte sonraki durum dağılımının öğrenicinin eylemlerine bağlı olmasını doğrudan ele alır. Öğrenicinin ziyaret ettiği durumlarda uzmandan doğru eylem etiketi alıp veriyi birleştirir; kendi dağılımında iyi davranışı belirli varsayımlarla hedefler. Bu önemli bir öncüldür, fakat sürekli erişilebilir uzman ayrı bir bilgi kaynağıdır. DAgger, yalnız bir defalık gösterimin ardından ödül/sonuçlarla uzman olmadan kendini geliştirme garantisi vermez; özgün algoritması veri biriktirme ve yinelemeli eğitim de kullanır. Burada alınan temel ders belirli eğitim mimarisi değil, politikanın oluşturduğu durumlarda ölçüm ve geri bildirim gereğidir. [Ross, Gordon ve Bagnell, 2011, §2–4 ve Algoritma 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

**Açık test:** Aynı kısa gösterimden başlayan dört kopya: gösterimi geçici bağlamda yeniden üreten; gösterimden kalıcı davranış edinen fakat güncellenmeyen; kendi sonuçlarıyla güncellenen; ziyaret ettiği durumlarda ilave uzman etiketi alan. Yeni başlangıç durumlarında kapalı döngü başarı, kullanılan toplam gösterim/sonuç/uzman etiketi, zarar ve yeniden başlatma sonrası kalıcılık birlikte ölçülmeli. Son grup daha fazla bilgi alır; üstünlüğü bedelsiz öğrenme sayılmamalıdır. Bu taklit testi mevcut iki ilk deney tarafından gerçekleştirilmiş değildir.

**Eklenen küçük çalıştırılmış tanık:** Yeni kapalı döngü dosyasında dört önceden verilmiş ikili politika şablonu, doğru bir gösterimle iki adaya düşer. Sistemin diğer bağlamdaki kendi başarısız eylemi, doğru eylemin yalnız iki seçenekten biri olduğu varsayımıyla tek aday bırakır. Sonraki 20 nesne kimliğinde düzeltilmiş kural 20/20, dondurulmuş ilk taklit 10/20 doğru cevap verir. Ancak nesne kimlikleri yeni olsa da iki bağlam değeri zaten görülmüştür. Bu, kendi geri bildiriminin ilk taklidi değiştirebildiğini gösteren sonlu bir mantık tanığıdır; yeni duyusal özelliklere genelleme veya gecikme/yeniden başlatma sonrasında taklit kalıcılığı testi değildir. Yukarıdaki daha güçlü dört-kopyalı test halen açıktır.

## 3. Tek olayın büyük etkisi: bilgi ile karar eşiği farklı şeylerdir

Olay sayısı ile davranış değişiminin doğrusal olması gerekmez. Bunu göstermek için biyolojik bir iddia gerekmiyor. İki hipotez arasında Bayes oranı:

\[
\frac{P(H_1\mid e)}{P(H_0\mid e)}=
\frac{P(H_1)}{P(H_0)}\,
\frac{P(e\mid H_1)}{P(e\mid H_0)}.
\]

**Kendi sayısal tanığımız:** Önce `P(tehlikeli)=0,01` olsun. Yeni olayın tehlikeli dünyada görülmesi güvenli dünyaya göre 100 kat olasıysa, son olasılık `100/199 ≈ 0,5025` olur. Tek veri, hipotezler arasında ayırt ediciyse güçlü bir güncelleme yapar. Bu sayılar çocuğun gerçek olasılıkları hakkında ölçüm değildir; matematiksel mümkünlük tanığıdır.

Karar değişimi bundan ayrı olabilir. Denemenin güvenli durumda kazancı `G`, tehlikeli durumda kaybı `L`, kaçınmanın değeri sıfır olsun. Deneme ancak

\[
(1-p)G-pL>0
\quad\Longleftrightarrow\quad p<\frac{G}{G+L}
\]

olduğunda tek-adımlık beklenen değerce tercih edilir. `G=1, L=19` için eşik `0,05`'tir. Olasılığın yalnız `0,049`'dan `0,051`'e çıkması kararı tamamen değiştirebilir. Büyük davranış farkı büyük bilgi kazanımı veya büyük iç yapı değişimi gerektirmez. Bir olumsuz sonucun ağır maliyetli olması ile hangi dünya modelini desteklediği ayrı sorulardır.

Beklenmedik her olay güvenilir değildir. Örneğin

\[
P(e\mid H)=(1-\epsilon)P_{\rm signal}(e\mid H)+\epsilon Q(e)
\]

gibi açık bir kirlenme modeli, olayın sensör/gürültü kaynağından gelmiş olabileceğini hesaba katar. `ε`, `Q` ve ana model yanlışsa sonuç yine kötü olabilir; bu formül genel sağlamlık çözümü değildir. İlk afin deneyinin tek yanlış güvenilir etikette doğru kuralı iptal etmesi, yüksek etkiyi otomatik olarak iyi öğrenme saymamak için eldeki negatif örnektir.

Bu yüzden deneyim önemini tek skora indirmek zorunlu değildir. En az üç ayrı değişim ölçülebilir: dünya hakkındaki inanç, gelecekteki eylem dağılımı ve elde edilen değer/risk. Gürültülü bir kaynağın yüksek entropisi bu üçünde fayda sağlamayabilir. Görevle ilgisiz bağımsız bir yazı-tura dizisini öğrenmek, o görevin en iyi eylemini değiştirmiyorsa karar değeri sıfır olabilir.

## 4. Kendi eylemiyle karşılaşmak, nedenselliği otomatik çözmez

Bir kişinin `a` eylemini seçtiği durumlardaki sonuç dağılımı `P(r|a,o)`, o eylemi müdahaleyle yaptırmanın sonucu `P(r|do(a),o)` ile genel olarak aynı değildir. Gizli koşullar hem eylem seçimini hem sonucu etkileyebilir. Gözlenen bir başkasının başarısı, öğrenicinin farklı kapasite ve koşullarda aynı başarıya ulaşacağını da belirlemez. Nedensel etkiyi tanımlama, onu veriden belirlenebilir kılan varsayımlar ve sonlu veriden kestirim birbirinden ayrılmalıdır. [Pearl, 2010, §3.2 ve §4](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/).

Bu araştırmaya özgü kontrol: Benzer görünür durumlarda, sonuçları etkileyen gizli bir zorluk değişkeni ve bu değişkene göre eylem seçen bir gösterici kurulabilir. Yalnız gözlemsel kayıttan elde edilen eylem-sonuç kuralı; koşulları eşleyen kontrollü eylem atamasıyla sınanır. Atama biçimi ve zaman içindeki değişim kaydedilir. Amaç gerçek bir çocuğu veya fiziksel tehlikeyi denemeye yöneltmek değil, küçük simülasyonda alternatif nedensel açıklamaları ayırmaktır.

Sistem kendi eylemini rastgele veya kontrollü seçebildiğinde bazı karıştırıcılar giderilebilir; bu da sınırsız neden keşfi garantisi değildir. Eylemlerin mümkün olduğu durumlara erişim, gözlemlenmeyen iç kapasite, gecikmiş sonuç, zaman içinde değişen dünya ve ölçüm hatası ayrıca hesaba katılmalıdır. Rastgeleleştirme, uygulanmış eylemin gerçek sonucu ile yalnız hayal edilmiş karşı-olgusal sonucu aynı bilgi türüne dönüştürmez.

## 5. Öğrenilmiş kaçınmanın kendini doğrulayabilmesi

İki dünya olsun. `W_0`'da bariyer kapalı kalır. `W_1`'de daha sonra açılır. Kaçınmanın gözlemi her ikisinde de aynıdır; deneme farklı sonucu açığa çıkarabilir.

**Sonlu geçmişler için kendi ayırt-edilemezlik türetimimiz:** Aynı başlangıç durumunu ve rastgelelik kaynağının aynı dağılımını kullanın. Her erişilebilir geçmişte politikanın pozitif olasılıkla seçtiği tüm eylemler için iki dünyanın sonraki gözlem yasaları aynıysa, bir sonraki geçmişin dağılımı da aynıdır. Tümevarımla tüm geçmiş dağılımları aynı kalır. O halde yalnız bu veriye bakan hiçbir karar veya kestirim kuralı hangi dünyada olduğunu güvenilirce ayıramaz. İki dünya eşit önselle verildiğinde son bir ikili dünya sınıflandırmasının ortalama hata alt sınırı `1/2`'dir.

Sorun yalnız tahminin yanlış olması değildir. Politika, tahmini yanlışlayabilecek verinin gelmesini engeller. İçsel daha uzun hesaplama, aynı verinin daha çok tekrarı veya davranışın adının değiştirilmesi bu iki dünya arasında yeni dış kanıt üretmez. İçsel süreç daha sonra bilgilendirici bir eylemi seçerse yukarıdaki varsayımı kırar; yeni sonuç o zaman ayrımı mümkün kılabilir.

Tam olarak bu tür bir öğrenme kapanmasına önceki çalışma vardır. Klenske ve Hennig'in küçük kontrol örneğinde, belirsizlikte eylemi azaltan ihtiyatlı kısa-vadeli kontrol, eylemin sağladığı bilgiyi de azaltır ve öğrenmeyi durdurabilir; makale bunu tarihsel “turn-off phenomenon” adıyla tartışır. Gelecekteki gözlemlerin inanç ve sonraki kararlara etkisini hesaba katmak dual control/Bayesçi pekiştirmeli öğrenmenin merkezindedir. Genel hesaplaması güçtür; makaledeki yaklaşım belirli model varsayımlarıyla yaklaşık çözümdür. Dolayısıyla “öğrenme kendi verisini kapatır; gelecek bilginin karar değerini hesaba kat” düşüncesine bu araştırmada yenilik atfedilemez. [Klenske ve Hennig, JMLR 2016, §3–3.1](https://www.jmlr.org/papers/volume17/15-162/15-162.pdf).

## 6. Yeniden sınamak ne zaman değerlidir?

Aşağıdaki küçük model bu araştırmanın kendi özelleştirmesidir; yukarıdaki makalenin deneyini yeniden üretmez. Bariyer son başarısız denemede kapalıdır. Her zaman adımında, bir daha kapanmamak üzere `h` olasılıkla açılacağı varsayılır. Son başarısızlıktan beri `d` adım geçtiğinde, yeni dış gözlem olmadan öngörülen açıklık olasılığı:

\[
p_d=1-(1-h)^d.
\]

Bu bir değişim modeli altında zaman geçişiyle yapılan kestirimdir. Bariyerin gerçekten açıldığının yeni kanıtı değildir. `h` verilmişse değişimin hızı öğrenilmiş sayılmaz. `h=0` ise başarısızlık sonrasında bu model yeni açılma olasılığı üretmez. Sonlu `T` boyunca hiç açılmama, `h>0` modelinde de `(1-h)^T` olasılıklı bir kuyruk olayıdır; model desteğinin dışında değildir. Bütün test dünyalarının bu olaydan seçilmesi, o kuyruğa önseldekinden fazla ağırlık veren stres koşuludur. Sonsuza kadar hiç açılmayan dünya ise pozitif sabit hazard modelinin dışında kalır. Bu ayrım, koşullu kötü performansı eşleşen model altında beklenen optimalitenin çürütülmesiyle karıştırmamak için gereklidir.

Kalan süre `n`, başarısız deneme maliyeti `C`, başarılı kullanımın her adımdaki kazancı 1 olsun. Başarıdan sonra her adım kullanılabilen açık durum biliniyor olsun. Aşağıdaki Bellman karşılaştırması sonlu bir hesabı tanımlar:

\[
Q_{\rm dene}(n,d)=p_d n+(1-p_d)[-C+V(n-1,1)],
\]
\[
Q_{\rm bekle}(n,d)=V(n-1,d+1),
\qquad V(n,d)=\max\{Q_{\rm dene},Q_{\rm bekle}\},
\quad V(0,d)=0.
\]

İndeksleme, başarısızlığın ardından bir sonraki karar anına kadar bir adım geçtiğini kabul eder. Başarıda kazanç `n`, mevcut adımın 1 birim kazancını da içerir; bu modelde başarıya ayrı maliyet yoktur. Sonlu örnekte bu değer tablosu hesaplanabilir. Verilen model, beklenen toplam kazanç hedefi ve sonlu eylem kümesi için maksimumu alan deterministik eylem seçimi yeterlidir. Rastgelelik burada zorunlu ilke değildir. Adversarial zamanlama, başka güvenlik kısıtları veya farklı hedefler için aynı çıkarım otomatik taşınamaz.

Yeniden sınamaya isteklilik yalnız mevcut başarının olasılığına değil, başarı sonrasında kalan kullanım fırsatlarına da bağlıdır. Saf bir bilgi edinme deneyi için, fiziksel dünyayı ve sonraki eylem kümesini değiştirmeyen gözlem `Y`'nin brüt karar değeri ayrıca şöyle yazılabilir:

\[
\operatorname{EVSI}=
\mathbb E_Y\left[\max_a\mathbb E[U(a,\theta)\mid Y]\right]
-\max_a\mathbb E[U(a,\theta)].
\]

İlk terim gözlem geldikten sonra seçim yapmaya izin verdiğinden, ücretsiz ve yok sayılabilir gözlem için bu değer negatife düşmez. Gerçek bir test maliyetli, tehlikeli veya dünyayı değiştiren bir eylemse bu ifade tek başına yeterli değildir; maliyet, zarar, değişen durum ve kalan fırsatlar birlikte değerlendirilir. Salt Shannon bilgi miktarı, bu karar değerinin yerine konamaz.

**Ayırıcı deney tasarımı:** İlk başarısızlıktan sonra hiç denemeyen politika; sabit aralıkla deterministik yeniden deneme; aynı ortalama sıklıkta rastgele yeniden deneme; yalnız ilgisiz başka hareket yapan iç sayaç; verilen değişim modeli altında değer karşılaştırması. Hepsi aynı dünya örneklerine ve başlangıç bilgisine tabi olmalıdır. Hiç değişmeyen dünya mutlaka bulunmalıdır. Öğrenicinin bariyerin gerçek açılma zamanı veya gizli dünya kimliğine erişimi olmamalıdır. Aynı dış gözlem dizisindeki ilk farklı eylem ile ancak bu eylemden sonra gelen gerçek geri bildirim ayrı kaydedilmelidir.

Ölçütler: açılmayı ilk fark etme gecikmesi, hiç fark edememe oranı, yeniden deneme sayısı, başarısız deneme maliyeti, toplam kazanç, zararın dağılımı, kaçınılan bölgelerdeki yanlış inancın süresi. Rastgele yöntemin daha çok deneme yapmasıyla görülen avantaj keşif stratejisi üstünlüğü sayılmamalıdır. Aynı nominal sıklık, her sonlu koşuda aynı gerçek deneme bütçesi demek değildir.

Bu politika ailesi `closed_loop_experiment.py` dosyasında çalıştırıldı. Kodun bağımsız okunmasında gizli açılma zamanının politika girdisine geçirilmediği ve Bellman tablosunun zaman indekslerinin simülatörle uyumlu olduğu görüldü. Geometrik modelin bütün 240 açılma zamanı ve sonlu ufuk kuyruğu üzerinden tam toplama, tablonun beklenen değerini doğruluyor. Ayrıntılı sayılar [CLOSED_LOOP.md](CLOSED_LOOP.md) dosyasındadır. Sekiz-adımlık geçici açılma kontrolünde, yeniden kapanma sonrasında simülatör başarı inancını kaldırıp son başarısızlıktan yeni geometrik açılma beklemeye döner. Bu belirli yanlış-model tepkisidir; başlangıçtaki kalıcı-açılma modelinde kapanmaya tam tutarlı Bayes çıkarımı yapıldığı söylenmez.

## 7. Sınırsız güvenlik ve sınırsız yeniden açılma birlikte garanti edilemez

Bir güvenli bekleme eyleminin iki dünyada aynı gözlemi verdiğini düşünün. Tek ayırt edici eylem `W_0`'da geri döndürülemez kayıp, `W_1`'de ödül veriyor olsun. Her iki dünya için sıfır felaket garantisi istenirse sistem ayırt edici eylemi pozitif olasılıkla yapamaz. Dolayısıyla ödüllü dünyayı bu yolla keşfetmeyi de garanti edemez. Bu, burada kurulmuş iki-dünya karşı örneğidir; bütün güvenli keşif yöntemlerinin işe yaramadığı iddiası değildir.

Bilinen güvenli başlangıç bölgesi, yan gözlem, doğrulanmış düzenlilik veya geri dönülebilir deneme gibi ek yapı bazı sorunları çözülebilir kılar. Örneğin SafeOpt'un teoremi güvenli başlangıç kümesi, Lipschitz düzenliliği, sınırlı RKHS normu ve koşullu sıfır-ortalamalı sınırlı gürültü varsayar; hedef bütün dünyadaki global optimum değil, güvenli biçimde ulaşılabilen bölgenin optimumudur. Bu garanti keyfî değişen, yanlış modellenmiş veya felaketli her ortama taşınamaz. [Sui ve diğerleri, ICML 2015, §2 ve Teorem 1](https://proceedings.mlr.press/v37/sui15.pdf).

Bu nedenle “geçmiş hiçbir eylemi sonsuza kadar kapatmamalıdır” mutlak koşulu fazla güçlüdür. Savunulabilir araştırma hedefi: Belirlenmiş risk, bilgi, ulaşılabilirlik ve değişim varsayımları altında, yanlış olduğu anlaşılabilir hale gelen geçmiş sınırlamaları uygun maliyetle yeniden sınayabilmek. Belirsizlik bulunduğu için denemek her zaman akılcı değildir; kalıcı kaçınma bazen seçilen hedefe göre doğru davranıştır.

## 8. Kendi kapasitesi ile dünyanın zorluğunu ayırmak

“Başaramadım” gözlemi tek başına “dünya bunu imkânsız kılıyor” anlamına gelmez. Kendi tanımlanabilirlik karşı örneğimizde başarı olasılığı:

\[
P(Y=1)=\sigma(c-d),\qquad \sigma(u)=1/(1+e^{-u}).
\]

Yalnız aynı tür görevin başarı/başarısızlık sonuçları gözleniyorsa, bütün `(c+k,d+k)` çiftleri aynı dağılımı üretir. Daha çok aynı veri bu belirsizliği çözmez. Zaman içinde başarı artışı, kapasitenin artmasıyla veya zorluğun azalmasıyla açıklanabilir. Bir açıklamaya yüksek güven atamak, onu tanımlanabilir yapmaz.

Zorluğu dışarıdan sabitlenmiş bir kalibrasyon görevi, bağımsız zorluk ölçümü veya yalnız kapasiteyi etkilediği savunulabilen bir müdahale eklenirse ayrım mümkün olabilir. Bunlar ilave bilgilerdir; “sistem kendisini anladı” sonucuna bedelsiz girdi sayılamaz. Kalibrasyon aracının kendisi de değişiyorsa ayrım yeniden bozulabilir. Bu, genel bir öz-modelin gereksiz veya imkânsız olduğunu değil, yalnız sonuç kaydının yeterli olmadığını gösterir.

## 9. İçsel sapma: hata mı, öğrenmeyi mümkün kılan bileşen mi?

Son ek fenomen, dış koşullar benzer görünürken içsel fiziksel/kimyasal durumun değişmesiyle farklı eylem yapılması ve ardından eski davranışın dış sonuçlarla yanlışlanmasıdır. Burada iki ayrım zorunludur.

**Aynı gözlem aynı tam durum değildir.** Deterministik `z_{t+1}=F(z_t)` ve `a_t=π(o_t,m_t,z_t)` bile aynı `o_t` karşısında farklı eylemler üretebilir. Bir sayaç, belirlenmiş süre sonra önce kaçınılan eylemi seçebilir. Bu tanık rastgeleliğin zorunlu olmadığını gösterir; insan davranışındaki gerçek biyolojik sebebi belirlemez.

**İç kapasite ile iç tercih aynı şey değildir.** İç durum yalnız politikayı veya geçici maliyet tercihini değiştiriyorsa, aynı eylemin aynı koşullardaki sonucu değişmeden kalabilir. Gerçek kapasite `c_t` değişiyorsa eylemin sonuç yasası da değişebilir. Bu iki koşul ayrı müdahalelerle sınanmalıdır. “Daha çok denedi” ile “artık yapabiliyor” karıştırılmamalıdır.

Sapma tek başına öğrenme değildir. Eski kayıtla aynı veri üzerinde dolaşan iç hareket, ilgili hipotezi yanlışlayamaz. İki dünya arasındaki uyuşmazlığı sınayan bir eylem için gereken temel özellik, en azından bazı mevcut geçmişlerde

\[
P(e\mid H,a,W_0)\ne P(e\mid H,a,W_1)
\]

olmasıdır. Politikanın bu tür bilgilendirici bir eylemi seçmesi yeni kanıtın yolunu açabilir. Sonucun alınması ve gelecekteki tahmin/davranışın ona göre düzeltilmesi öğrenme halkasını tamamlar. İlgisiz rastgele hareket bu eşitsizliği sağlamayabilir; zararlı sapma aynı zamanda net değeri düşürebilir.

Dolayısıyla “sapma ya hatadır ya da temel öğrenme bileşenidir” ikiliği uygun değildir. Bir sapma, mevcut politika altında amaç hatası iken model hakkında bilgilendirici olabilir. Başka bir sapma yalnız bozucu gürültü olabilir. Başka bir sistem doğal kullanım sırasında yeterince ayırt edici deneyim aldığı için özel sapmaya ihtiyaç duymayabilir. Gerekli özellik her eylemde rastgelelik değil, hedeflenen değişimleri saptayabilecek bilgiye erişimin hangi koşullarda korunabildiğidir.

Önerilen sayaç/değer tablosu deneyi yalnız bu matematiksel yeterlilikleri sınar. İnsanlarda fiziksel veya kimyasal dalgalanmanın nasıl oluştuğu, hangi durumlarda faydalı olduğu ve öğrenmeyi ne ölçüde açıkladığı hakkında kanıt oluşturmaz.

## 10. Gözlemlerin tamamının araştırma haritası

“Yerel destek”, mevcut küçük araştırma ailesinin sınırları içindedir; insan fenomene karşılık geldiği iddiası değildir. Önerilmiş testi çalıştırılmış sonuç saymamak için durumlar ayrılmıştır.

| Gözlem | Ayrıştırılan temel soru | Kanıt durumu / ayırıcı sınama |
|---|---|---|
| 1. Az gözlemle davranış edinme | Önceki bilgi sayesinde az yeni gösterim yeterli olabilir mi? | Birincil robotik önceki çalışma ve yerel iki-bağlamlı taklit→ödül tanığı var. Yeni görsel özellikler, gecikmeli kalıcılık ve önceki öğrenme maliyeti ayrı. |
| 2. Soba / tek olay | Bir olay tahmin veya karar eşiğini değiştirebilir mi? | §3 matematiksel tanık. Gürültülü olay kontrolü ve kalıcılık sınaması gerekir. |
| 3. Olayın ötesine genelleme | Öğrenilmiş ilişkinin yeni girdilere etkisi | Afin deneyde bilinen aile içinde yerel destek; aile-dışı polinom karşı örneği var. |
| 4. İlk öğrenmenin değişmesi | Çelişki; bağlam değişimi; gerçek kural değişimi ayrımı | Afin revizyon ve Boole aile değişimi yerel örnekler. Doğru kapsamı otonom keşfetme açık. |
| 5. Eylemin sonucunu yaşama | Tahmin edilen korelasyon mu, müdahalenin sonucu mu? | İlk deneyler yeterli değildi; yeni kapalı döngü tanığı eylemin veri erişimini değiştiriyor. §4'teki gizli karıştırıcı testi ayrıca açık. |
| 6. Deneyimlerin birleşmesi | Bileşen bilgisinden yeni davranış üretme | Verilmiş bileşim semantiğiyle afin tanık var; yeni temsil dilini keşfetme gösterilmedi. |
| 7. Davranışın kullanımına karar verme | Mevcut seçeneklerin sonuçlarına göre seçim | Değer karşılaştırması bunu ifade edebilir; ayrı “metacognition module” zorunluluğu çıkmaz. |
| 8. Kullanmayı bırakma / bastırma | Bilgiyi silmek ile seçimi değiştirmek | Aynı bilgi farklı maliyet veya koşulla kullanılmayabilir; yalnız bilgi kaybı ölçülmemeli. |
| 9. Dış deneyimsiz işleme | Dış dünya hakkında yeni kanıt mı, hesabı yeniden örgütleme mi? | Afin derlemede aynı cevap daha dar çevrimiçi bütçeye sığıyor; ertelenmiş derleme aynı toplam işi yapıyor. |
| 10. Kayıt ile öğrenme farkı | Kalıcı işlevsel etki; kodlama biçiminden bağımsız mı? | Yeniden başlatma/durum aktarımı tanığı var; ham kayıt + aynı çıkarım güçlü rakibi eşit başarılı. |
| 11. Taklit / geçici üretim | Gösterim kaldırıldığında etki sürüyor mu? | Küçük tanık kendi geri bildirimiyle düzeltmeyi gösteriyor. Taklit kuralının gecikme ve yeniden başlatma kalıcılığı ayrıca sınanmadı. |
| 12. Farklı yaşamlar | Aynı başlangıç, farklı deneyim, eşit test | Afin iki dünya tanığı ve yeni süreç kontrolü var; öğrenilmiş durumun kapsamı sınırlı. |
| 13. Etkinin büyüklüğü | Veri miktarı, ayırt edicilik ve karar maliyeti | §3 bunların farklı büyüklükler olduğunu gösterir; her çarpıcı olay doğru değildir. |
| 14. Seçici yoğunluk | Güvenilir sinyal mi, gürültü mü, geç anlamlanan veri mi? | Gürültü modeli ve kaynak güvenilirliği gerekir; genel seçici öğrenme çözülmüş değil. |
| 15. Davranış/yetenek değişimi | Dosya değişikliği yerine müdahaleli işlev ölçümü | Durum sıfırlama/aktarma, yeni girdiler, gecikme ve bütçe kontrolleri; sadece sayaç büyümesi başarı değildir. |
| 16. Eski sınırın eskimesi | Geçmiş başarısızlık güncel imkânsızlık mı? | §5'in iki-dünya tanımlanamazlığı; güncel ayırt edici geri bildirim gerekir. |
| 17. Öğrenilmiş davranıştan sapma | Hangi eylem yeni ayrımı gözlenebilir kılar? | Deterministik zamanlama mümkün; salt rastgelelik gereği türetilmez. |
| 18. Koruma / yeniden açılma | Denemenin bilgi değeri ile zararı | §6 değer hesabı ve §7 karşı örneği; bütün dünyalara sıfır riskli keşif garantisi yok. |
| 19. Değişen öz-kapasite | Dünya ve sistem etkileri tanımlanabilir mi? | §8'de sonuç-only verinin yetersizliği türetildi; kalibrasyon ayrı bilgi kaynağıdır. |
| 20. Yeniden sınanabilir bilgi | Ne, hangi varsayımla, hangi maliyetle sınanabilir? | Tek güven puanı şart değil; test koşulları/değişim modeli/karar değeri gerekir. |
| 21. Kendi verisini kapatma | Politika veri desteğini nasıl daraltıyor? | Seçili etiketler ve dual-control öncülleri; §5'te açık ayırt-edilemezlik. |
| 22. Geçmişe mahkûm olmama | Faydalı geçmişi koruyup yanlış sınırı hangi koşulda değiştirmek? | Bilgiye erişim + maliyet/değişim varsayımları altında şartlı hedef; tek evrensel kural sonucu yok. |
| Son ek fenomen: içsel sapma | Aynı dış gözlemde farklı tam durum ve davranış | §9: deterministik iç dinamik yeterli olabilir; politika değişimi, kapasite değişimi ve dış kanıt ayrılır. |

## 11. Kaynak doğrulaması ve açık sonuçlar

Bu taramada yalnız özgün makaleler ve yazar/yayıncı nüshaları dayanak alındı. Aşağıda belirtilen bölümler okundu; altı makalenin tüm eklerini ve tüm teorem kanıtlarını tükettiğimiz iddia edilmez. Hiçbir biyolojik olay bu kaynaklarla doğrulanmış sayılmadı.

| Doğrulanan kaynak | Okunan kapsam |
|---|---|
| [Finn ve diğerleri, 2017 — One-Shot Visual Imitation Learning via Meta-Learning](https://proceedings.mlr.press/v78/finn17a.html) | Yayın kaydı; PDF giriş, §2–3, meta-eğitim ve test ayrımı. |
| [Ross, Gordon ve Bagnell, 2011 — A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | PDF §2–4, Algoritma 3.1, uzman etiketi ve kendi ziyaret dağılımı koşulları. |
| [Lakkaraju ve diğerleri, KDD 2017 — The Selective Labels Problem](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf) | PDF giriş, §3, kararların etiket görünürlüğünü belirlemesi ve kayıt dışı değişkenler. |
| [Pearl, 2010 — An Introduction to Causal Inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/) | Yayın yılı doğrulandı; §3.2 müdahale/tanımlanabilirlik ve §4 yöntem ayrımları. 2009 tarihli başka derlemeyle karıştırılmadı. |
| [Klenske ve Hennig, JMLR 2016 — Dual Control for Approximate Bayesian Reinforcement Learning](https://www.jmlr.org/papers/volume17/15-162/15-162.pdf) | PDF giriş ve §2–3.1; küçük kontrol örneğindeki öğrenmenin kapanması. |
| [Sui ve diğerleri, ICML 2015 — Safe Exploration for Optimization with Gaussian Processes](https://proceedings.mlr.press/v37/sui15.pdf) | PDF §2, güvenli erişilebilir hedef, Algoritma 1 ve Teorem 1 varsayımları. |

Yeni gözlemler tek bir ek mekanizmayı zorunlu kılmadı. Dört bağımsız eksikliği görünür kıldı: gösterim sonrası kalıcı işlevsel değişimin sınanması; eylemden nedensel sonuç çıkarma varsayımları; politikanın yeni veri erişimini belirlemesi; geçmiş başarısızlığın değişen kapasite ve dünya altında yeniden değerlendirilebilmesi. İçsel değişkenlik bu son sorunu bazen açabilir, fakat bilgilendirici ve güvenilir dış sonuçla birleşmeden öğrenmenin yerini tutmaz.

Araştırmanın yeni kapalı döngü tanığı, bu eksikliklerden bazılarını çok küçük bir karar probleminde sınadı. Gerçek nesnelerden az gösterimli davranış edinimi, görev ailesi keşfi, gerçek kapasite değişimini otonom ölçme ve güvenli açık dünya etkileşimi halen ayrı açık araştırma hedefleridir.
