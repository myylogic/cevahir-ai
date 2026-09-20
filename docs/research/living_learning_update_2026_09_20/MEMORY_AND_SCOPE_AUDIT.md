# Koruma, yenileme ve öğrenilmiş güncelleme: bağımsız kapsam denetimi

20 Eylül 2026. Bu belge önceki [ortak yaşam deneyini](../living_learning_joint_2026_09_20/REPORT_TR.md) ve [kapsam ayrımlarını](../living_learning_reassessment_2026_09_20/SCOPE_AND_FOUNDATIONS.md) koruyarak devam eder. Buradaki matematiksel karşı örnekler analitik sonuçlardır; çalıştırılmış yeni deney sonuçları olarak sunulmaz. Genel yaşarken öğrenme problemi çözülmüş sayılmaz.

**Denetimin ana sonucu:** Geçmiş yaşamlar bir sistemin hangi değişimleri koruyup hangilerine daha hızlı uyacağını etkileyebilir. Bunun gösterilmesi, geçmiş cevapların tekrar kullanılmasından daha güçlü bir aktarım sınaması ister. Yalnız skaler güncelleme hızı öğrenmek; saklanacak ayrımları, bu ayrımların koordinatlarını, uygun geleceği veya bilgi geçerliliğini öğrenmekle aynı değildir. Tam matris bu ifade kısıtının bir kısmını kaldırabilir; kalan araştırma sorunlarını kaldırmaz.

## 1. Önceki deneyin hangi varsayımı gerçekten azaltılabilir?

Önceki ortak akışta hangi eski kayıtların birbirinin yerine geçeceğini sabit bir uzamsal ızgara belirledi. Bu ızgara değişimin gerçekleştiği bölgeyle uyumluydu. Yeni bir yaşam ailesinde değişimin yönü bilinmeyen doğrusal birleşimler boyunca gerçekleşirse, ayrı koordinatları veya baştan verilen hücreleri korumak doğru bağımsızlıkları sağlamayabilir.

Bu nedenle yeni denetimin konusu yalnız daha iyi bir unutma katsayısı değildir. Deneyimin kalıcı bıraktığı bilgi, gelecekteki bir hatanın **hangi hesaplamaları birlikte değiştirmesi gerektiğini** de etkileyebilir. Fakat bu, hesaplama yönlerinin değiştirilmesine ilişkin bir alt problemdir; yeni algısal ayrım oluşturma, geri bildirimin anlamı, yeni yürütülebilir dil, beceri bileşimi ve sınırlı kaynaklarla uzun süre devam etme ayrı açıklar olarak kalır.

## 2. Skaler ve koordinat başına kazancın ifade sınırı

Bir adımda ağırlık değişimi `Δw=K g` olsun. `g`, öğrenicinin o anda kullandığı hata veya gradyan yönüdür. Skaler `K=ηI`, yalnız mevcut yönün boyunu değiştirir. Bu ifade sınıfı, istenen düzeltmenin başka yönde olması halinde yetersizdir.

**Skaler karşı örnek.** Dik birim yönler `u,v` için gelen yön `g=u+v`, uygun düzeltme `v` olsun. Bütün gerçek `η` değerleri için

\[
\|\eta(u+v)-v\|^2=\eta^2+(\eta-1)^2\geq\tfrac12.
\]

`K=vvᵀ` ise istenen değişimi tam üretir. Bu, bir uygun matrisin bulunabilirliğidir; geçmiş deneyimin o matrisi belirleyebileceğinin kanıtı değildir.

**Diyagonal karşı örnek.** `u=(1,1)/√2` korunacak yön, `v=(1,−1)/√2` yenilenecek yön, `g=(1,0)` olsun. Uygun değişim

\[
vv^Tg=(1/2,-1/2).
\]

Her diyagonal `D=diag(d₁,d₂)` için `Dg=(d₁,0)` ve

\[
\|Dg-vv^Tg\|^2=(d_1-1/2)^2+1/4\geq1/4.
\]

Yani koordinat başına ayrı hız vermek bile doğru eşleşmeyi her zaman ifade edemez. Sistem bu tek adımda ikinci koordinatı değiştirmek için sıfır olmayan bir çapraz etkiye ihtiyaç duyar. Birçok adım, farklı girdiler veya başka bir temsil bu engeli aşabilir; sonuç bütün diyagonal öğrenicilerin her görevde başarısızlığı değildir.

**Koordinat uyarısı.** Ortogonal `R` altında hem ağırlıkları hem düzeltmeleri `w'=Rw`, `g'=Rg` olarak değiştirirsek eşdeğer matris `K'=RKRᵀ` olur. Diyagonal matrisler bu dönüşüm altında genel olarak diyagonal kalmaz. Tam simetrik matris sınıfı kalır. Genel tersinir yeniden parametrelemede gradyan ve artık-vektörü farklı taşınır; aynı dönüşüm formülü ikisine gelişigüzel uygulanmamalıdır. Önceki [durum taşıma deneyi](../living_learning_reassessment_2026_09_20/STATE_TRANSPORT.md) bu ayrımı ayrıca sınadı.

## 3. Korunması gereken bilgi, ileride yapılacak işe bağlıdır

Bellekte iki bağımsız, eşit olasılıklı bit `b₁,b₂` hakkında tam bilgi edinildiğini; basit saklama seçeneklerinin yalnız bir bitin değerini kalıcı tutmak olduğunu düşünelim. Gelecek sorgunun `b₁` isteme olasılığı `p`, `b₂` isteme olasılığı `1−p` olsun. Saklanmayan bit için en iyi tahminin hata olasılığı `1/2`dir.

`b₁`i tutmanın beklenen hatası `(1−p)/2`, `b₂`yi tutmanınki `p/2` olur. `p=.9` iken ilk seçim `.05`, ikincisi `.45` hata verir; sorgu dağılımı tersine dönerse sıralama da tersine döner. Bu hesap bütün bir-bit kodların optimumunu ileri sürmez; yalnız iki açık saklama seçeneğinin hangisinin yararlı olduğunun gelecekteki kullanıma bağlı olduğunu gösterir.

Geçmiş sorgular `p` hakkında bilgi vererek sonraki saklamayı iyileştirebilir. Fakat bu aktarım sorgu sürecindeki devamlılığa dayanır. Geçmişin aynı olup gelecekte farklı sorguların geldiği iki dünyayı önceden ayırmak ek bilgi gerektirir. Bu bir genel öğrenme imkânsızlığı iddiası değildir; öğrenilmiş bellek seçiminin genelleme varsayımını görünür kılar.

Önceki ortalama ve temsil genişletme örnekleri ayrıca, bugün yararlı olan özetin yarınki güncellemeye veya düzeltmeye yeterli olmayabileceğini gösterdi. Dolayısıyla bellek seçimi yalnız geçmişi yeniden üretme doğruluğuyla ya da yalnız bugünkü servis kaybıyla sınanmamalıdır. Gelecek kullanım, öğrenme devamı ve geri alma istekleri farklı bilgi talep edebilir.

## 4. Temel ağırlıkları sıfırlamak neyi ayırır?

Durumu `s=(w,m)` şeklinde ayıralım: `w` mevcut tahminci, `m` onun nasıl güncelleneceğini etkileyen kalıcı durum olsun. Önceki yaşamdan öğrenilmiş `m_E` ile aynı başlangıç tahmincisi `w₀` üzerinde `m₀` kontrolünü karşılaştırmak, **aynı yeni akış altında** önceki deneyimin sonraki edinimi değiştirip değiştirmediğini sınar:

\[
w_{t+1}^{E}=U_{m_E}(w_t^E,z_t),\qquad
w_{t+1}^{0}=U_{m_0}(w_t^0,z_t),\qquad w_0^E=w_0^0.
\]

Bu karşılaştırma uygun bir meta-öğrenme müdahalesidir, fakat tek başına kesin izolasyon değildir:

- `m_E` ham örnekler, eski cevaplar veya görev kimlikleri içeriyorsa ağırlık reseti bunları kaldırmaz. Taşınan durumun alanları ve okuma yolları incelenmelidir.
- Yeni hedef katsayıları önceki yaşamdan bağımsız yeniden çekilmeli; yeni etiket öncesi tahminler eşit olmalı; önceki örneklerin doğrudan cevabı vermesi engellenmelidir. Aynı ilişki/geometriyi paylaşmak izinli aktarım varsayımı olarak kaydedilir.
- Yeni akışta `m_E` dondurulmuş kontrol, taşınan yordamın etkisini ayırır. `m`nin aynı yaşamda güncellenmeye devam ettiği ek kontrol ise devamlılığı ölçer. Birini diğerinin yerine koymak iki soruyu karıştırır.
- Aynı yeni örnekler, aynı sıra, aynı temel başlangıç, eşlenmiş gürültü ve aynı servis zamanları kullanılmalı. Geçmiş yaşamda `m_E` üretmenin verisi, hesabı ve belleği ayrıca sayılmalıdır.
- Yeni yaşamın sonuçlarına göre en iyi geçmiş checkpoint'i veya hiperparametreyi seçmek değerlendirme sızıntısıdır. Seçim geçmiş yaşam veya ayrı doğrulama üzerinden yapılmalı; son test seçim için kullanılmamalıdır.

Bu koşullar altında fark, önceki yaşamdan taşınan `m` durumunun sonraki öğrenmeye nedensel etkisine kanıt olur. Bu durumun adı, biyolojik karşılığı veya ağırlık/bellek ayrımı kanıtın esasını değiştirmez. Dar bir öğrenici sınıfında avantaj, genel öğrenmeyi öğrenme mekanizmasının tamamı değildir.

## 5. Üç somut yanlışlama sınaması

**Sınama A — Cevap taşıması mı, edinim biçimi taşıması mı?** Önceki yaşamdan gelen temel ağırlıkları kaldır; yalnız beyan edilen güncelleme durumunu taşı. Yeni hedef değerlerini bağımsız örnekle. Bütün kontroller yeni etiket öncesi aynı cevapları versin. Aynı yeni akışta taşınmış, sıfırlanmış ve uygun rastgele-geometrili durumları karşılaştır. Başarı yalnız eski hedef katsayıları korunduğunda görünürse, yordam aktarımı açıklaması yanlışlanır. Yarar yeni katsayılarda sürer fakat yeni ilişkisel geometride tersine dönerse, aileye özgü yordam aktarımı desteklenir; genel aktarım desteklenmez.

**Sınama B — Doğru koordinatlar araştırmacı tarafından mı verildi?** Önceki ve yeni bütün yaşamlara aynı bağımsız ortogonal dönüşümü uygula; veri üreticisi, tahminci ve değerlendirme birlikte dönüşsün. Her dönüşmüş veri ailesinde öğrenici baştan kendi `m` durumunu edinsin; doğru dönüşmüş matris deneyci tarafından taşınmasın. Diyagonal/skaler ve tam-matris öğrenicilerini hem servis hatası hem koruma/uyum yönleri açısından karşılaştır. Yalnız elverişli koordinatlarda avantaj, koordinattan bağımsız edinim iddiasını yanlışlar. Ortogonal dönüşümün eşdeğer problemi ile önceki geometri sabitken yeni dünyanın geometrisini gerçekten değiştirme kontrolü ayrı tutulmalıdır; ikinci durum aile kaymasıdır.

**Sınama C — Koruma ilişkisi devam mı ediyor, tersine mi dönüyor?** Aynı geçmişten sonra eşlenmiş iki devam kur: birinde eskiden sabit olan yön sabit, değişken olan değişken kalsın; diğerinde bu roller yer değiştirsin. Yeni kanıt geldikçe dondurulmuş ve güncellenmeye devam eden `m` sürümlerinin anlık yanlış koruma, toparlanma süresi ve toplam maliyetini ölç. Öğrenilmiş durum doğru ailede kazanç verip ters ailede zarar veriyorsa bu negatif saklanmalı. Daha güçlü kontrol, aynı geçmişin farklı gelecek sorgu dağılımlarına bağlandığı iki-bit örneğinin çok boyutlu karşılığıdır. Geçmiş iç uyum skoru aynıyken kullanım riskinin ayrışması, iç seçim ölçütünün kendi başarısını tanımlamasını yanlışlar.

## 6. Birincil literatürle somut eşleşmeler

| Kaynak ve incelenen kapsam | Bu araştırmaya eşleşme | Buraya taşınamayacak sonuç |
|---|---|---|
| [Li ve diğerleri, 2021 — Lifelong Learning with Sketched Structural Regularization](https://proceedings.mlr.press/v157/li21b.html); özgün makalenin özet ve yöntem kapsamı | Koruma önemini matrisle temsil etmek, diyagonal yaklaşımın kaybını daha zengin düşük maliyetli özetlerle azaltmak zaten incelenmiştir. | Her eski önemin hâlâ geçerli olduğu, değişim yönlerinin kendiliğinden keşfedildiği veya tam matrisin genel yaşarken öğrenme cevabı olduğu söylenemez. |
| [Denevi, Pontil ve Ciliberto, 2021 — Conditional Meta-Learning of Linear Representations](https://arxiv.org/abs/2103.16277); özgün özet | Tek ortak temsilin farklı görev kümelerine yetmeyebileceği; görev verisi gibi yan bilgiyle koşullanan temsil ediniminin bazı ailelerde üstünlüğü. | Sınırsız yeni dünya, verilen yan bilginin anlamını keşif veya bütün öğrenme durumunun yaşamda güvenli sürdürülmesi gösterilmez. |
| [Zhou ve diğerleri, 2022 — Probabilistic Bilevel Coreset Selection](https://proceedings.mlr.press/v162/zhou22h/zhou22h.pdf); ana metin §3.1 | Hangi örneklerin saklanacağını, bunlarla eğitilen modelin daha geniş veri üzerindeki kaybından öğrenmek bilinen bir yordamdır. İç problem seçilmiş örneklerle eğitim, dış problem bütün mevcut veri kaybıdır. | Dış kaybın bilinmeyen gelecek kullanıma uygunluğu otomatik değildir. Bütün veri kaybını korumak, değişmiş dünyada geçersiz geçmişi bırakmakla aynı amaç değildir. |
| [Tishby, Pereira ve Bialek, 1999; arXiv kaydı 2000 — The information bottleneck method](https://arxiv.org/abs/physics/0004057); özgün özet | Hangi bilginin ilgili olduğu, başka bir değişken hakkında taşıdığı bilgi üzerinden tanımlanır; kısa kod ile ilgili bilginin korunması arasındaki seçim formelleştirilir. | Hangi gelecek değişkenin önemli olacağını kendiliğinden belirleyen genel hedef keşfi veya bilinmeyen yeni görevler için sınırsız yeterlilik değildir. |

Literatür burada araştırma iddiasının yerini ve sınırını belirlemek için kullanıldı; bu yöntemler yeniden uygulanmış veya sonuçları yeniden üretilmiş sayılmaz. Yukarıdaki küçük cebirsel sonuçlar, kaynaklardan alınmış yeni bir evrensel öğrenme ilkesi değildir.

## 7. Bu turun matris deneyi için yorum denetimi

Kök araştırmanın protokolü, `w←w+P x(y−xᵀw)` biçiminde iki boyutlu bir güncelleme kullanıyor. Bilinmeyen iki dik doğrultuda hedef katsayıları farklı hızlarda değişiyor; girişler birim çemberden, çıktı gürültüsü ayrı geliyor. `P` kaynak yaşamın servis hatalarıyla, temel katsayılardan ayrı olarak değiştiriliyor; sonraki yaşamda yalnız `P` taşınıyor. Bu bölüm sayısal sonuçları incelemeden yapılan **protokol yorumu**dur.

**Öğrenilen şey hangi düzeyde?** `P`, önceden verilmiş gradyan biçiminin yönünü ve ölçeğini düzenler. Temiz hedefe karşı hata, doğru hedef katsayısı ve veri üreticisinin gizli yönleri öğreniciye verilmez; gözlenen gürültülü etikete karşı artığını kendisi hesaplar. `P` başarısı böylece araştırmacının doğru katsayı doğrultularını baştan vermesine ihtiyaç olmadığını gösterebilir. Ancak önceki ızgara deneyinin bilinmeyen uzamsal bölgelerini doğrudan çözmez: burada öğrenilen doğrultular model katsayılarının doğrusal uzayındadır. Yeni deney eski dünyadaki doğru bölgelemeyi öğrenmiş diye sunulmamalıdır.

**Matrisin içerdiği bilgi yalnız dünyanın değişim hızı değildir.** Kaynak örnekleme dağılımı, gürültü, sınırlı meta-güncelleme ufku ve verilmiş servis kaybı birlikte hangi `P`nin iyi olacağını belirler. `P`yi dünya için eksiksiz bir model veya genel yeterli istatistik saymak geçersizdir. Veri aynı kalsa yalnız ölçüm gürültüsü değişse bile uygun kazanç değişebilir.

**Geçmiş etkisinin zaman ölçeği.** Sabit `P`, bağımsız birim-çember girdileri ve durağan hedef altında `E[xxᵀ]=I/2`dir. Sıfır ortalamalı bağımsız ölçüm gürültüsüyle, eski başlangıç hatasının koşullu beklentiye katkısı her adımda `I−P/2` ile çarpılır. `P`nin özdeğeri `λ` olan yönde bu katkı `t` adımda `(1−λ/2)^t` olur. Bu dar hesap, küçük kazancın eski durumun etkisini daha uzun tuttuğunu açıklar. Yarılanma süresi `log(.5)/log(1−λ/2)`dir. Tek örnek yolundaki kare risk, değişen hedef, ilişkili girdiler ve çevrimiçi değişen `P` için aynı formül doğrudan risk garantisi değildir.

Bu nedenle burada “koruma”, parametre durumunun önceki etkisinin bazı doğrultularda daha yavaş değiştirilmesi anlamına gelir. Kimlikli kayıtları seçme, hangi etiketin geçerli olduğuna karar verme, hatalı kaydı geri alma veya gelecekteki temsil için gereken ayrımları tutma gösterilmiş olmaz. Özdeğerlere verilen pozitif alt sınır da tam dondurmayı ifade sınıfının dışına çıkarır; bu hazır bir tasarım kararıdır.

**Kontrollerin ayırdığı soru.** Aynı başlangıç katsayısından başlayan yeni yaşam aktarımı, soğuk başlangıç katsayılarının ayrıca değiştirildiği kontrolle birlikte değerlendirilmelidir. `P` ile eş izli skaler, yalnız diyagonal bölüm ve ters çevrilmiş geometri kontrolleri, toplam adım boyundan gelen yararla yönsel yapıdan gelen yararı ayırır. Kaynak veriye bakarak seçilen güçlü sabit matris aynı veya daha iyi sonuca ulaşırsa, çevrimiçi metagüncellemenin benzersizliği veya üstünlüğü gösterilmez. Seçimin kaynak veriyle maliyeti açık yazıldığında bu yine anlamlı bir karşılaştırmadır.

## 8. Ana problem için kalan yükümlülük

Güncelleme geometrisi deneyi başarılı olursa önceki yaşamın yeni öğrenme biçimini, hazır doğru koordinat grupları olmadan değiştirebildiğine ilişkin dar bir sonuç elde edilir. Hâlâ hazır kalanlar: gözlem vektörü, değiştirilebilir model sınıfı, sonuç hizalaması, değerlendirilecek kullanım, hata ölçüsü, meta-güncelleme yordamı ve geçmiş ile gelecek arasındaki aile benzerliği. Bu varsayımları saklamadan daha sonra azaltmak gerekir.

Ana araştırma yalnız hangi hızla veya yönde güncelleme yapılacağına taşınmamalıdır. Şu üç kazanım ayrı ölçülmeye devam etmelidir: yeni dünya ayrımlarını edinmek; edinilmiş bilgiyi sınırlı hesapta kullanılabilir yapmak; sonraki deneyimden öğrenme kapasitesini değiştirmek. Bellek seçimi ve güncelleme geometrisi üçüncüsüne katkı sağlayabilir; diğer ikisini ve bunların aynı yaşamda uyumunu tek başına çözmez.
