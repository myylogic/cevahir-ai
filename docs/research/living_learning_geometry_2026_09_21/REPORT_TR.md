# Öğrenilebilir temsilde eski beceriyi korumak ve yeni kapasite edinmek

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma turu 10

**Ana problem çözülmedi.** Bir yapay sistemin deneyim nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir değiştirmesinin genel ilkesi henüz bulunmuş değil. Bu tur, önceki araştırmaya yeni bir sınama ekledi: temsilin de öğrenildiği bir ağda geçmişi koruma biçimi, gelecekte edinilebilecek becerileri nasıl etkiliyor?

12 eşleştirilmiş sentetik yaşam üzerinde 864 ana koşul ve sonuçlar görüldükten sonra eklenen 216 kontrol çalıştırıldı. Bunlar **1.080 bağımsız dünya değildir**; aynı 12 başlangıç ve akışın farklı koşullarıdır. Matematiksel karşı örnekler, 36 devam müdahalesi, 12 çevrimdışı referans ve bunların yol denetimleri ayrıca tutuldu. Başlıca sonuç: bu deneyde güçlü çıktı koruması eski davranışı saklarken yeni edinimi neredeyse durdurabiliyor. Eski doğru hedeflerdeki hatayı sınırlamak bunu hafifletiyor, fakat kullanılan aday üretimiyle çözmüyor. Daha fazla replay bazı sonuçları iyileştiriyor; kesintisiz koruma garantisi sağlamıyor.

## 1. Önceki araştırmanın neresinden devam edildi?

[Ortak yaşam çalışması](../living_learning_joint_2026_09_20/REPORT_TR.md), hazır özellikler ve araştırmacının verdiği bölgesel bellekle öğrenme, düzeltme ve kaynak sınırlarını birlikte sınamıştı. [Durum taşıma](../living_learning_reassessment_2026_09_20/STATE_TRANSPORT.md) ve [öğrenilmiş güncelleme](../living_learning_update_2026_09_20/REPORT_TR.md), bugünkü eşit tahminlerin gelecekte eşit öğrenme davranışı anlamına gelmediğini göstermişti. [Düzeltme](../living_learning_correction_2026_09_20/REPORT_TR.md) ve [sorguya göre yeterli durum](../living_learning_query_state_2026_09_21/REPORT_TR.md) çalışmaları ise tutulması gereken bilginin gelecekte izin verilen işlemlere bağlı olduğunu belirlemişti.

Bu sonuçların hiçbiri geri alınmadı. Ancak sabit özellikli sonuçlar, özellikleri deneyimle değişen bir ağa kendiliğinden taşınamaz. Burada giriş–gizli katman ağırlıkları da öğreniliyor. Yeni edinim, eski çıktılar ve gelecekte kullanılacak türevler aynı parametre değişiminden etkileniyor.

Önceki [ilke denetiminde](../living_learning_joint_2026_09_20/PRINCIPLE_AUDIT.md) bulunan “ayrı ayrı uygun değişikliklerin birleşimi uygun olmayabilir” karşı örneği de korunuyor. Aşağıdaki çarpımsal örnek bunun yeni bir genel keşfi sayılmıyor; öğrenilebilir temsil geometrisini açıklayan özel bir tanık olarak kullanılıyor.

## 2. Ne sınandı, ne verildi?

[Sonuçlardan önce yazılan yerel protokol](PROTOCOL.md), dış bir kayıt hizmetindeki ön kayıt değildir. Ağ `2→8 tanh→1`, toplam 33 öğrenilebilir parametredir. Sabit ve gürültüsüz hedef, aynı ağ ailesindeki üç etkin tanh birimiyle tam ifade edilebilir. Öğretmen ağırlıkları öğreniciye verilmez.

Her yaşamda koordinatlar rastgele döndürülür. Eski ve yeni örnekler iki ayrı girdi bölgesinden, bağımsız değerlendirme noktaları bu bölgelerden ve aradaki şeritten gelir. Başlangıç, 128 eski örnekte 600 güncelleme ile edinilir. Yeni akış 128 adet sekiz örneklik minibatch içerir. Her minibatch'in tahmini etiketiyle güncellenmeden önce kaydedilir; bu tahminlerin MSE ortalaması **prequential hata**dır. Eski 4, 16 veya 64 etiketli örnek, iç içe bellek kümeleri olarak tutulur. Aşağıda bunlara *ankor* deniyor.

Öğrenme oranları `.03` ve `.12`, ana SVD kesme oranı `10⁻⁸`dir; üç teğet yöntem `10⁻⁶` ile de sınanır. Saklanmış veya güncel eski-çıktı Jacobian'ına dik güncelleme, sıradan online SGD, sabit özelliklerde yalnız çıktı katmanı eğitimi, replay, A-GEM benzeri kayıp yarı-uzayı ve gerçek aday denetimi karşılaştırılır. Yöntemler aynı etiket akışını alır; toplam hesapları eşit değildir.

Eski veri kesimi ve ilk eğitim evresi araştırmacı tarafından sağlanmıştır. Hedef sabittir, ağ boyutu büyümez, çevreye eylem seçilmez ve öğrenme kuralı deneyimden keşfedilmez. Bunlar başarılmış özellikler gibi sunulamaz. Ayrı test etiketleri öğrenme yönüne veya aday kabulüne girmez.

## 3. Matematiksel ayrım: yerel yön, sonlu adım ve erişilebilir çıktı

Tam ispatlar ve kaynak eşleşmeleri [matematik notunda](MATHEMATICS_AND_LITERATURE.md) bulunuyor. Üç ayrım deney yorumunu belirliyor.

**İlk mertebe koruma, sonlu adımda tam koruma değildir.** Eski çıktı `f(a,b)=ab`, eski hedef `1`, başlangıç `(1,1)` olsun. `δ=(h,−h)` için `Jδ=0`, fakat yeni çıktı `1−h²`dir. `h=1/2` için yarım kare kayıp `1/32` olur. Buna karşı `(1+h,1/(1+h))` eğri yolu çıktıyı tam korur. İlk örnek bir türev yaklaşımının sınırını gösterir; uygun bir yolun yokluğunu göstermez.

**İlk mertebede edinim yönü için kesin bir koşul vardır.** Aynı noktadaki eski ve yeni çıktı Jacobian'ları `A` ve `B`, `P=I−A⁺A` olsun. `Aδ=0` ve `Bδ=r` ancak ve ancak `r∈range(BP)` ise birlikte sağlanabilir. Yeni kayıp gradyanı `g` için korunmuş teğet uzayda kesin negatif yönlü türev ancak `Pg≠0` ise vardır. Bu, sabit bir noktada Öklid metriğiyle yapılan lineer cebirdir; yeni bir evrensel öğrenme ilkesi değildir. Deneydeki sayısal SVD kesmesi tam Moore–Penrose tersinden ayrıca ayrılmalıdır.

**İlk mertebede yön bulunmaması, global öğrenilemezlik değildir.** Eski çıktı `a`, yeni çıktı `b²`, başlangıç `(0,0)` ve yeni hedef `1` iken yeni ilk türev sıfırdır. Buna rağmen `(0,h)` eski çıktıyı koruyarak yeni hatayı azaltabilir. Dolayısıyla küçük projekte gradyanı “bu ağda çözüm yok” diye yorumlamak hatalıdır.

Ayrıca eski başlangıç çıktısına göre ortalama kare sapma `D`, başlangıcın doğru etikete göre MSE'si `R₀` ise yeni doğru MSE artışı en fazla `2√(R₀D)+D` ile sınırlanır. `D=0`, başlangıç hatalarını da saklar. Doğru eski beceriyi korumak, her eski tahmini dondurmakla eş anlamlı değildir. Ortalama eski hata sınırı da her eski alt beceri için ayrı güvence vermez.

## 4. Ana sonuçlar

Aşağıdaki tablo 64 ankor, öğrenme oranı `.12`, ana kesme eşiği ve 12 yaşam ortalamasıdır. Son MSE'ler öğrenmede kullanılmayan örnekler üzerindedir; düşük daha iyidir. Başlangıç eski MSE `.00806143`, yeni MSE `.19626949`, şerit MSE `.03491092`dir. Bütün diğer koşullar [ham sonuçta](../../../research/living_learning_geometry_2026_09_21/results/geometry.json) saklıdır.

| Yöntem | Eski MSE | Yeni MSE | Prequential MSE | Ankor çıktı sapması |
|---|---:|---:|---:|---:|
| Donmuş ağ | .00806143 | .19626949 | .20458583 | 0 |
| Yalnız çıktı katmanı | .05148927 | .00653977 | .01221860 | .04315893 |
| Online SGD | .04103444 | .00474944 | .00979798 | .03537611 |
| Replay | .01028929 | .00641810 | .01647353 | .00587973 |
| Kayıp yarı-uzayı | .01045777 | .00399829 | .01109306 | .00698476 |
| Sabit teğet uzay | .00805737 | .19580429 | .20455255 | .00000114 |
| Güncel teğet uzay | .00806143 | .19626921 | .20458569 | ≈0 |
| Güncel teğet + çıktı denetimi | .00806143 | .19626921 | .20458569 | ≈0 |
| Replay + çıktı denetimi | .00802845 | .17455036 | .18265219 | .00009999 |

Güncel teğet yönteminde ortalama rank **31.5/33**; “tam rank, hiçbir yön kalmadı” demek yanlış olur. Yeni ham gradyanın ortalama normu `.70445` iken projekte/ham norm oranı `.00010842`, yani yaklaşık **%0.01084**. Bu koşulda baskın etki, mevcut yeni gradyanın koruma projeksiyonuyla neredeyse silinmesidir. Bütün ağlarda genel bir plastisite kaybı mekanizması saptanmış değildir.

SVD eşiği `10⁻⁶` olduğunda ortalama rank yaklaşık `25.49`, norm oranı `.003694`, yeni MSE `.19604527` oluyor. Daha az yönü korumak güncellemeyi büyütüyor, fakat bu taramada edinim yine çok az. `.03` oranında da yakın durma görülüyor; sonuç yalnız seçilmiş tek orana dayanmıyor.

Çıktı denetimi, sabit başlangıç ankor çıktılarından sapmayı `10⁻⁴` altında tutar. En çok 13 aday denenir, her başarısız denemede adım yarıya iner. Replay ile bu ölçüt 128 adımın ortalama **121.58**'ini tamamen reddetti. Güncel teğet yöntemde ise ilk öneriler zaten çok küçüktü: ek denetim hiç yarılama yapmadı. Dolayısıyla denetimin bütün adayları kabul etmesi öğrenme başarısı değildir.

### Ankor sayısı neyi değiştiriyor?

| Güncel teğet, oran `.12` | Eski MSE | Yeni MSE | Rank | Projeksiyon norm oranı |
|---|---:|---:|---:|---:|
| 4 ankor | .01192201 | .01014484 | 4 | .423692 |
| 16 ankor | .00805568 | .17963712 | 16 | .037724 |
| 64 ankor | .00806143 | .19626921 | 31.5 | .000108 |

Dört ankorda çıktı sapması yalnız yaklaşık `1.09×10⁻⁶` iken bağımsız eski bölge MSE'si artıyor. Tutulan örneklerde koruma, bütün eski bölgeyi koruma değildir. Daha çok ankor ise aynı zamanda daha fazla kısıt getiriyor; bilginin miktarı ile kabul edilen değişim alanı birlikte değiştiğinden bunları ayrı nedensel etkiler gibi ölçtüğümüz söylenemez.

Sabit teğet yöntem 16 ankorda yeni MSE'yi `.02512812`ye indirebiliyor, fakat eski MSE `.04163159`, ankor sapması `.03886594` oluyor. Başlangıç türevlerine dik olmak, temsil değiştikten sonraki türevlere dik olmak değildir.

Eşleştirilmiş olarak güncel teğet yöntemin yeni MSE'si, 16 ve 64 ankorda **her iki öğrenme oranında 12 yaşamın tamamında** replay'den yüksek. 64 ankor, `.12` için farkların ortalaması `.18985111`, aralığı `[.04519730, .35635837]`. Dört ankorda sıralama karışık. Bunlar incelenen yaşam ailesinin betimleyici sonuçlarıdır; bütün görevler için üstünlük testi değildir.

## 5. Aynı ağırlıklar, farklı gelecek edinim

64 ankor ve `.12` sonunda üç yöntemden ayrı devam çiftleri üretildi: toplam 36 çift. Her çift aynı ağırlık, bellek ve sonraki 32 minibatch ile başlıyor. Bir kopya koruma politikasını sürdürürken diğerinde yalnız güncelleme modu online SGD'ye çevriliyor. İlk tahmin farkı bütün çiftlerde tam sıfır.

Güncel teğet yönteminden başlayan çiftlerde koruma sürünce son yeni MSE `.19626914`, kaldırılınca `.00700841`. Fakat eski MSE `.00806143`ten `.04683179`a çıkıyor. Korumanın kaldırılması ücretsiz bir onarım değildir. Replay + çıktı denetiminde de yeni MSE yaklaşık `.17455036` karşısında `.00698898`, serbest kopyanın eski MSE'si `.04681292`.

Bu müdahale, mevcut tahmin işlevinin gelecekteki edinimi tek başına belirlemediğini gösteriyor. Saklanan kanıt ve bu kanıta uygulanacak koruma/güncelleme politikası da öğrenme durumunun parçası. Bu düşünce önceki durum taşıma çalışmalarında vardı; burada öğrenilebilir temsilde başka bir somut neden gösterilmiş oldu. Politikanın sistem tarafından keşfedildiği veya öğrenildiği gösterilmedi.

## 6. Sonuçlardan sonra eklenen kontroller: doğru hedefi korumak yeterli mi?

[Ek kontrol planı](POSTHOC_CONTROLS.md) ana sonuçlar görüldükten sonra yazıldı. Aynı 12 yaşam yeniden kullanıldı; bunlar bağımsız doğrulama verisi değildir. İki yöntem, çıktı yerine eski doğru etiketlere göre MSE'nin başlangıç değerini `10⁻⁴`ten fazla aşmamasını denetledi. Üçüncü yöntem her yeni minibatch'te iki replay güncellemesi yaptı.

| 64 ankor, oran `.12` | Eski MSE | Yeni MSE | Prequential MSE | Tam ret / 128 |
|---|---:|---:|---:|---:|
| Online + doğru-kayıp denetimi | .00830013 | .15360665 | .16201174 | 120.17 |
| Replay + doğru-kayıp denetimi | .00834211 | .14737338 | .15704941 | 117.75 |
| İki replay adımı | .00720101 | .00414022 | .01150301 | Uygulanmıyor |

Doğru-kayıp denetimi başlangıçtaki yanlış cevapların düzeltilmesine izin verir. Buna rağmen eldeki öneri yönlerini yalnız küçülten kabul mekanizması bu deneyde çok sık durdu. Bu, bütün kayıp-kısıtlı optimizasyonun imkânsızlığı değildir; belirli öneri kuralı, tolerans ve sonlu aday bütçesinin negatif sonucudur.

İki replay adımı **12 yaşam ortalamasında** hem eski hem yeni son hatayı başlangıca göre düşürdü. İki hatanın birlikte düştüğü yaşam sayısı 7/12; diğer beş yaşamda eski MSE arttı. Ayrıca eğitim sırasında eski ankor MSE'sinin başlangıcın üzerine en büyük artışının yaşam ortalaması `.00942514`; `10⁻⁴` kesintisiz koruma koşulunu sağlamıyor. Daha çok hesap için güçlü bir kontrol, genel living-learning çözümü veya aynı hesap bütçesinde üstünlük kanıtı değildir. [Ek sonuçların tamamı](../../../research/living_learning_geometry_2026_09_21/results/posthoc_controls.json) korunuyor.

### Uygun son nokta ile uygun yol farklıdır

Çevrimdışı referans, eski 128 ve yeni 1.024 örneğin tamamına aynı anda erişerek 1.000 ortak güncelleme yaptı. Ortalama son eski MSE `.00507431`, yeni MSE `.00289452`, şerit MSE `.00516680`. Bu fazladan veri ve hesap kullanan tanısal referanstır; online yöntemlerle adil yarışmacı değildir.

Sonuçlar görüldükten sonra eklenen [yol tanısı](../../../research/living_learning_geometry_2026_09_21/results/endpoint_diagnostic.json), bu referansın 12 son noktasının **10'unun** 64 ankor üzerindeki doğru-kayıp bütçesine uyduğunu gösterdi. Buna rağmen kullanılan ortak güncelleme yollarının **12'si de** ara adımlarda bütçeyi aştı. Bu 12 örnek yolun başarısızlığı, aynı uç noktalara bütçeyi koruyan başka hiçbir yol olmadığını ispatlamaz.

Öğretmen ağı aynı model sınıfında sıfır hatalıdır; fakat **12 yaşamın tamamında** başlangıç çıktılarına göre `10⁻⁴` sapma bütçesini aşar. Dolayısıyla “ağ öğretmeni ifade edebiliyor” bilgisi, “sıfır hatalı çözüm seçtiğimiz çıktı-koruma kümesinde de var” demek değildir. Kapasite, kısıtın izin verdiği kapasite ve kullanılan hesapla erişilebilen kapasite ayrı incelenmelidir.

## 7. Kaynak maliyeti ve doğrulama

64 ankor koşulunda bir yaşam için online yöntem 1.024, replay 9.216, iki-adımlı replay 18.432 gradyan örnek satırı işledi. Güncel teğet yöntem ayrıca 8.192 Jacobian satırı ve 128 SVD, sabit teğet yöntem 64 Jacobian satırı ve bir SVD kullandı. Replay + çıktı denetimi ortalama 104.389 ek aday ileri-geçiş satırı harcadı. Bu yüksek aday maliyetine rağmen çoğu adım reddedildi.

Bu sayaçlar FLOP veya süre ölçümü değildir. Ortak başlangıç eğitimi, değerlendirme izlemesi, ilk referans ileri geçişi ve geçici matrislerin bütün maliyeti kapsamda değil. Tutulmuş model/ankor/baz skalerleri toplam veya tepe RAM olarak yorumlanmamalı. Teğet olmayan yöntemlerde rank alanındaki sıfır ile sabit teğet yöntemde güncel lineerleştirme değişimi alanındaki sıfır **uygulanmayan ölçümlerdir**.

[Sayısal doğrulama kaydı](../../../research/living_learning_geometry_2026_09_21/results/verification.json):

- Analitik Jacobian ile merkezî sonlu fark arasındaki en büyük hata `8.23×10⁻¹¹`, PyTorch autograd ile fark `2.22×10⁻¹⁶`; kayıp gradyanı ve affine sonlu-adım kontrolü geçti.
- Dokuz ana yöntemin durumları ayrı süreçte yüklenip 17 minibatch sürdürüldü. Tahminler ve bütün JSON durumları birebir eşleşti. Üç post-hoc yöntemin ayrı süreç devamı bu kapsamda değildir.
- Ana ve post-hoc taramalar tümüyle yeniden çalıştırılıp kayıtlarla karşılaştırıldı. Bu aynı kodla yeniden üretimdir; bağımsız bir veri tekrar-deneyi değildir.
- Bütün kayıtlı çıktı ve doğru-kayıp guard koşulları, tanımlandıkları ankor bütçesine uydu. Bu bağımsız test başarımı garantisi değildir.
- Çalışma başındaki **866 mevcut dosya** byte düzeyinde korundu; HEAD değişmedi. Kitap değişiklikleri dahil başlangıçtaki yerel çalışma korunuyor.

Ayrı [uygulama incelemesi](REVIEW.md), türevler, kayıt tamlığı, seçilmiş sekiz hücrenin yeniden hesabı, test etiketi müdahalesi ve devam kopyalarının bağımsızlığını ayrıca denetledi. Kritik uygulama hatası bulunmadı. Bu yerel inceleme, dış hakem değerlendirmesi değildir.

## 8. Literatür eşleşmesi ve geçerli iddialar

Geçmiş kayıp gradyanlarına göre projeksiyon [GEM](https://arxiv.org/abs/1706.08840) ve [A-GEM](https://arxiv.org/abs/1812.00420), çıktı gradyanlarına göre koruma [OGD](https://proceedings.mlr.press/v108/farajtabar20a.html) ile doğrudan ilişkilidir. OGD sıfıra yaklaşan kayıp gradyanını, yerelliği ve yüksek mertebeli etkileri zaten tartışır. Bu deney bunları keşfetmiş veya özgün makalelerin koşullu iddialarını çürütmüş sayılmaz. Sabit teğet özellik varsayımının kapsamı için [Doan vd.](https://proceedings.mlr.press/v130/doan21a/doan21a.pdf), yerel ve gerçek amaç ayrımı için [TRPO](https://proceedings.mlr.press/v37/schulman15.pdf) incelendi; onların garantileri bu probleme doğrudan aktarılmadı.

Araştırma kaydına eklenen geçerli sonuçlar şunlardır: bu küçük öğrenilebilir temsil ailesinde koruma projeksiyonu edinimi baskılayabiliyor; sabit ve güncel türev koruması aynı davranmıyor; sonlu aday denetimi iyi aday üretimini garanti etmiyor; başlangıç çıktısını korumak doğru eski beceriyi korumaktan farklı; aynı tahmin ağırlıklarıyla koruma politikasına müdahale gelecekteki edinimi değiştiriyor. Bu iddiaların sayısal kısımları bu deney ailesiyle sınırlı.

## 9. Genel living-learning probleminde açık kalanlar

| Boyut | Biriken kanıt | Bu turdan çıkarılamayan sonuç |
|---|---|---|
| Deneyimin kalıcı duruma geçmesi | Ağırlık, kanıt ve koruma durumunun saklanıp sürdürülmesi; önceki durum çalışmaları | Her uzun yaşamda yeterli ve ekonomik durumun kendiliğinden keşfi |
| Gelecekteki öğrenmenin değişmesi | Aynı tahminlerden farklı devam edinimi | Öğrenme kuralının deneyimle genel ve güvenilir iyileştirilmesi |
| Temsil ve hesap kapasitesi | Sabit özelliklerin ötesinde öğrenilebilir temsil | Yeni özellik, program, işlem veya modülün açık uçlu edinimi |
| Kontrol | Sonlu ankor bütçelerinin uygulanması; kısıtın edinim maliyeti | Bütün eski becerilerin ve görülmemiş koşulların korunması |
| Genelleme | Aynı sentetik ailede bağımsız değerlendirme bölgeleri | Aile dışı, gürültülü, gecikmeli ve eylemle üretilmiş deneyimde genelleme |
| Kaynaklar | Bazı işlem ve saklanan durum sayaçları | Sonsuz yaşamda sınırlı belleğe genel çözüm |

Sonraki araştırma bu turun koruma problemine hapsedilmemeli. Bir hat, aynı deneyim ve kaynaklarla **yeni bir iç özellik veya hesap işleminin gerçekten edinilip daha sonra başka bir problemde kullanılması**; mevcut aile içinde daha iyi ağırlık bulmaktan nasıl ayırt edileceğidir. İkinci hat, gecikmeli bir sonucun geçmişteki hangi iç hesap değişimine atanabileceği ve tutulmamış ayrımların bu atamayı ne zaman imkânsızlaştırdığıdır. Üçüncü hat, öğrenme durumunun değişmesinin eldeki aileye uyum mu yoksa yeni ailede daha iyi edinim mi olduğunun müdahalelerle ayrılmasıdır.

Bu tur içindeki daha dar açık soru ise: uygun son durum mevcutken iyi bir güncelleme yolunu veriden ve sınırlı hesapla nasıl bulabiliriz? Yalnız bir doğrultuyu küçültmek yerine yön değiştirme, birleşik telafi veya temsil değişimi kullanılabilir; hangisinin hangi varsayımlarla gerekli olduğu henüz gösterilmedi. Rastgelelik bunlardan birinin zorunlu cevabı olarak alınmadı. Sonraki hipotezler, basit replay ve aynı kaynaklı sabit kapasite kontrolleriyle birlikte yanlışlanabilmelidir.

Bu dosya yeni bir yerel araştırma kaydıdır. Önceki sonuçlar ve kitap dosyaları değiştirilmedi; commit, push veya yayın yapılmadı. Yeniden üretim komutları [deney dizininde](../../../research/living_learning_geometry_2026_09_21/README.md) bulunuyor.
