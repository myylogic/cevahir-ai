# Gelecekteki öğrenme işlemlerine göre gerekli durum: kesin sonlu bir denetim

**Muhammed Yasin Yılmaz**

CEV-2026-09 · Sürüm 2026.09.21.3 · Yayın tarihi 2026-09-21

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

Aynı sonlu geçmişlerde bütün Boolean ihlal sorguları, bütün gözlem çoklukları ve kimlikli etiket düzeltmeleri için gereken durum sınıfları karşılaştırılır. Toplam uzunluk ve işaretli etiket farkları bütün gelecekteki Boolean ihlal sorgularına yeterken, kimlikli düzeltme ayrı bir doğrusal bilgi alt sınırı getirir. 61.575 geçmiş, 712.632 etiket yazımı ve 358.020 ekleme tam sayılarla denetlenmiştir. Üç sınıf sayısı formülünde veya replay eşitliğinde uyuşmazlık bulunmamıştır.

## Abstract

We compare state classes for all Boolean loss queries, observation counts and identity-only label edits on the same finite histories. Length plus signed label differences suffices for all future Boolean loss queries; identity edits impose a separate linear information bound. Exact enumeration checks 61,575 histories, 712,632 label writes and 358,020 appends with no count-formula or replay mismatches.

## Bulguların yorumu

Bilinmeyen gelecekteki hipotezler her zaman tam histogram gerektirmez. Gerekli bilgi, hipotezlerin bilinip bilinmemesinden önce hangi sorgu ve işlemlerin destekleneceğine bağlıdır.

Sabit sonlu alan, bütün Boolean riskleri ve tam doğruluk kullanılır. Sınıf sayısı verimli kodlayıcı veya gerçek dünya öğrenme başarımı değildir; genel living-learning sorusu açık kalır.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/living_learning_query_state_2026_09_21/REPORT_TR.md](../../research/living_learning_query_state_2026_09_21/REPORT_TR.md) olarak korunur. Bağlantı yolları bu
edisyonun konumuna uyarlanmıştır. Rapordaki çalışma tarihi ve geçmiş yayın durumu
ifadeleri tarihsel kayda aittir; bu edisyonun tarihi yukarıdadır.

Bu çalışma [yayın dizisi](../README.md) içindeki ayrı bir metindir. DOI, bütün
metinleri, teknik kitabı ve kodu içeren sürüm arşivini tanımlar; her makaleye ayrı
DOI verildiği anlamına gelmez. Atıfta çalışma kimliği ve sürüm DOI'si birlikte
kullanılmalıdır. Hesaplamalı denetimlerin kapsamı [doğrulama kaydında](../evidence/validation.json)
ve [yayın ilkelerinde](../PUBLISHING.md) açıklanır.

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, hesaplamalı deney,
literatür incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır.
Bu katkı açıklaması, DOI kaydı veya otomatik testler dış bilimsel hakemlik sayılmaz.

---

## Araştırma kaydı: Gelecekteki öğrenme işlemlerine göre gerekli durum: kesin sonlu bir denetim

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Cevahir araştırma dizisi, çalışma 09

## Özet

Bir sistemin bugünkü doğru cevapları koruması, yarın yeni deneyimle doğru biçimde değişebilmesine yetmeyebilir. Önceki düzeltme araştırmasının devamında üç farklı sözleşmeyi karşılaştırıyoruz: bütün Boolean hipotezlerin 0–1 ihlal sayısını cevaplamak, bütün gözlem çokluklarını cevaplamak ve yalnız olay kimliği verilen etiket düzeltmelerinden sonra bütün ihlal sayılarını cevaplamak. Aynı geçmişler üzerinde bu sözleşmelerin farklı sayıda ayırt edilebilir duruma ihtiyaç duyduğunu gösteriyoruz. Bilinen sonlu girdi alanındaki bütün gelecekteki Boolean hipotezler için tam histogram zorunlu değildir; toplam kayıt sayısı ve girdi başına işaretli etiket farkı yeterlidir. Kimlikli düzeltme hakkı ise daha fazla bilgi gerektirir. Küçük evrenlerde 61.575 geçmişi tam sayarak türetilen sınıf sayıları, 712.632 içerikli düzeltme ve 358.020 ekleme denetlenmiştir. Uyumsuzluk bulunmamıştır. Bu, genel yaşarken öğrenme probleminin çözümü veya yeni bir genel öğrenme algoritması değildir.

## Araştırma bağlamı ve soru

[Önceki tur](../../research/living_learning_correction_2026_09_20/REPORT_TR.md), yalnız tutarlı adayların korunmasının geçmiş deneyim düzeltildiğinde yetersiz kalabileceğini göstermişti. Sabit adaylara ait ihlal sayaçları ve bütün gözlem histogramı yapıcı rakiplerdi. Yeni soru bu rakiplerden hangisinin gerçekten gerekli bilgiyi taşıdığıdır. Önceki raporda histogramın yeterliliği doğrudur; burada onun her sorgu ailesi için minimal olduğu sonucunun çıkarılamayacağı gösterilir. Önceki deneyler ve raporlar değiştirilmemiştir.

Çalışma yaşam süresince öğrenmenin **gelecekte yeniden değişebilme kapasitesi** bileşenini sınar. Temsil üretimini, öğrenilmiş güncelleyicinin edinilmesini, dünya ile etkileşimin güvenilirliğini ve bütün yaşam boyunca kontrollü genellemeyi kapsamıyor. Bellek yeterliliği bütün ana sorunun yerine geçirilmemelidir.

## Yöntem ve matematiksel sonuç

`X={0,…,m−1}`, `Y={0,1}` ve `N` olaydan oluşan geçmiş olsun. Her `h:X→Y` için toplam ihlal `c_h=Σ_i 1[h(x_i)≠y_i]` sorgulanır. `d_x=n[x,0]−n[x,1]` tanımıyla:

$$c_h=\frac{N-\sum_xd_x}{2}+\sum_xh(x)d_x.$$

Bu açılım `(N,d)` özetinin bütün hipotezlere yeterli olduğunu gösterir. Tersine, sabit-sıfır hipotez ile yalnız `x` noktasında 1 veren hipotezin cevap farkı `d_x`'tir; sabit-sıfır ve sabit-bir cevaplarının toplamı `N`'dir. Dolayısıyla tam risk cevapları ile bu özet aynı eşdeğerlik sınıflarını tanımlar. En iyi adayı seçme gibi daha zayıf sorguların minimal durumu ayrıca araştırılmalıdır.

Yeni kayıt için `N←N+1`, `d_x←d_x+1−2y`; gerçek eski içerik verilen etiket değişiminde `d_x←d_x+2(y−y')` olur. Böylece ortak geçerli, doğru eski içeriği taşıyan işlemler altında özet kapalıdır. Eski içeriğin doğruluğunu bu özet kendi başına denetlemez.

Sabit `N` altında mümkün fark vektörleri `||d||₁≤N` ve `N−||d||₁` çift koşullarını sağlar. Normu `s` olan vektör sayısı, sıfır olmayan koordinatların seçimi, işaretleri ve pozitif parçalanmaları sayılarak bulunur:

$$A_m(0)=1,\qquad A_m(s)=\sum_{k=1}^{\min(m,s)}{m\choose k}2^k{s-1\choose k-1}.$$

$$K_{risk}(N,m)=\sum_{j=0}^{\lfloor N/2\rfloor}A_m(N-2j),\quad K_{hist}(N,m)={N+2m-1\choose2m-1}.$$

Kimlikli sözleşme, `(j,b)` ile j'nci olayın etiketini b yapmayı içerir; aynı etiketi tekrar yazmak da geçerlidir. İki farklı geçmişin bugünkü riskleri farklıysa zaten ayrılırlar. Aynıysa, farklı oldukları bir kimliğin eski etiketleri farklı olduğunda birine etkisiz olan bir yazım ötekinde riski değiştirir. Etiketler aynı, girdiler farklıysa etiketi çevirmek, o iki girdiyi farklı yanıtlayan bir hipotezde zıt risk değişimleri üretir. Bütün farklı kimlik atamaları bu nedenle ayrılır: `K_ID=(2m)^N`. Bu sözleşmede en kötü durum sabit kod uzunluğu en az `ceil(N log₂(2m))` bittir. Kayıtların başka kayıpsız kodlamaları mümkündür; doğal dil metnini ham biçimde tutma zorunluluğu gösterilmemiştir.

Türetim ve karşı örneklerin ayrıntılı ön çalışma kaydı: [matematiksel not](../working-notes/future-state-review.md). Bu iç araştırma değerlendirmesi dış hakemlik değildir.

## Deney düzeni

`m∈{1,2,3}` ve `N∈{0,…,6}` için 21 hücrede bütün `(2m)^N` geçmişler tarandı. İhlaller bir yolda doğrudan kayıtları ve hipotezleri tarayarak, diğer yolda yukarıdaki özet formülüyle hesaplandı. Her kimlikte her iki etiketi yazdıktan sonraki tam yeniden tarama cevapları, bugünkü cevaplarla birlikte imza yapıldı. Sayımda hash özeti değil tam tamsayı cevap baytları kullanıldı; hücre başına farklı imzalar tam olarak sayıldı. İçerikli düzeltmelerin ve tüm olası tek kayıt eklemelerinin her biri replay ile karşılaştırıldı.

Kaynak [query_state.py](../../../research/living_learning_query_state_2026_09_21/query_state.py), ölçüm [query_state.json](../../../research/living_learning_query_state_2026_09_21/results/query_state.json). Yalnız Python standart kütüphanesi gerekir. Gerçekleşen deney ön nottaki temel sayım, JSON boyutu ve negatif kontrolleri uygular; çalışma süresi ve ilkel işlem sayısı benchmark'ı yapmaz.

## Bulgular

| Ölçü | Sonuç |
|---|---:|
| Tam sayılan geçmiş | 61.575 |
| İçerikli etiket yazımı; etkisiz yazımlar dahil | 712.632 |
| Tek kayıt eklemesi | 358.020 |
| Özet–replay sorgu uyuşmazlığı | 0 |
| Üç sınıf sayısı formülünün uyuşmadığı hücre | 0 / 21 |

`m=3, N=4` için 85 risk sınıfı, 126 histogram sınıfı ve 1.296 kimlikli işlem sınıfı bulundu. İdeal sabit kod alt sınırları sırasıyla **7, 7, 11 bit**tir. Daha az sınıf bu küçük örnekte tam bir bit kazanç anlamına gelmez. Sabit m için risk sınıfı sayısı `Θ(N^m)`, histogram sayısı `Θ(N^(2m−1))`; kimlikli sözleşme ise `N log₂(2m)` bitlik doğrusal alt sınır getirir. Bu asimptotik sonuç doğrudan formüllerden çıkar; altı olaylık deney büyük ölçekli performans ölçümü sayılmaz.

Sonuç dosyası dört naif JSON yükünün ortalama boyutunu da kaydeder: bütün ihlaller, işaretli özet, histogram ve kimlik sırasındaki olaylar. Bunlar ideal kod değildir; kod, şema, hipotez tanımları ve Python nesne belleği dahil değildir. Dışarıdan gelen düzeltme paketi eski içeriği taşıyorsa o kaynağın belleği de toplam sistem maliyetinde sayılmalıdır. Bu çalışma onun ücretsiz olduğunu varsayan bir verimlilik iddiası kurmaz.

## Negatif kontroller ve sınırlar

`[(0,0),(0,1)]` ile `[(1,0),(1,1)]` geçmişlerinin `N=2,d=(0,0)` özeti ve bütün risk cevapları aynıdır. Ancak `(0,0)` olayının sayısı 1 ve 0'dır. Gözlem gösterge sorgusu eklendiğinde risk özeti yetersiz kalır. İkinci geçmişte var olmayan `(0,0)` kaydını düzeltme iddiası sayısal risk sınırlarını bile geçebilir. Geçerli içerik varsayımı gerçek bir sınırlamadır.

`[(0,0),(1,0)]` ile ters sıralı geçmiş aynı histogramı taşır; ilk olayın etiketini 1 yapma komutu farklı yeni riskler üretir. Histogram olay kimliğini doğrulamaz. Bu örnekleri ortadan kaldırmak için dış bir kimlik servisi eklemek maliyeti taşır; bilgi gereksinimini yok etmez.

Kesin doğruluk, sabit alan, bütün Boolean riskleri ve bütün kimlikleri düzenleme hakkı güçlü varsayımlardır. Yaklaşık öğrenme, sınırlı düzeltme ufku, unutulmuş kaydı yeniden edinme, açık girdi dili veya dağılıma göre başarım farklı sınırlar verir. Sonuçların hiçbirinden genel bir asgari sinir ağı belleği çıkarılamaz.

## Literatür ve katkı sınırı

[Mitchell (1977), Version Spaces](https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf) tutarlı adaylarla öğrenme bağlamını verir. Buradaki tam ihlal sorgusu, yalnız tutarlı aday kümesini istemekten daha zengindir. [Cao ve Yang (2015), Towards Making Systems Forget with Machine Unlearning](https://www.yinzhicao.org/unlearning/UnlearningOakland15.pdf), toplamlardan katkı çıkarmayı ve adaptif yöntemlerde ek sorunları ele alır. Toplamsal güncellemeye yeni isim verilerek yenilik iddia edilmiyor. Ön notta ilgili birincil bölümlerin inceleme kaydı vardır.

Bu araştırma dizisine somut ek, aynı geçmiş ailesindeki üç gelecek işlem sözleşmesinin ayrı tam sınıf sayılarıyla karşılaştırılmasıdır. Bu özel sunumun literatürde ilk olduğu doğrulanmamıştır. Hesaplamalı tekrar ve iç matematiksel inceleme, dış hakem değerlendirmesinin yerine geçmez.

## Ana problemde kalan açık alan

Gelecek işlem ailesi verilince yeterli durumun nasıl tanımlanacağı burada keskinleşti. Açık dünya öğrenicisi ise hangi gelecekteki işlemleri korumaya değer bulacağını da öğrenmek zorundadır. Deney bu seçimi araştırmacıdan almadı. Sonraki çalışma için ayırıcı soru, sınırlı bellek ve hesap altında **sonradan değişen sorgu/temsil diline karşı hangi geri döndürülemez kayıpların kabul edildiği ve hangi geri edinme yollarının korunduğudur**. Bu yön tek başına ana problemin merkezi ilan edilmiyor: edinilen hesap yordamlarının yürütülebilir hale gelmesi, güncelleme kuralının taşınması, kararlılık ve aktarım birlikte açıklanmayı bekliyor.

Yeniden üretim:

```powershell
python research/living_learning_query_state_2026_09_21/query_state.py --check
```

Bu komut kayıtlı sonuçların üzerine yazmadan tam taramayı tekrarlar. Yayın paketi ayrıca [koruma ve doğrulama kaydını](../evidence/validation.json) taşır. Araştırma ve metin hazırlığında yapay zekâ desteği kullanılmıştır; proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir.
