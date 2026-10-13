# Tur 13 — Deneyimin kalıcı kurala dönüşmesi: konsolidasyon, zamanlama ve bilgi sınırları

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel araştırma kaydı

## Ne elde edildi?

Kullanıcının hızlı edinim ve sonradan bütünleştirme önerisi [24 ayrı hipoteze](HYPOTHESES_TR.md) dönüştürüldü. Bir sonlu yapay modelde **geçici olay belleğinden kalıcı, görülmemiş girdilere uygulanabilir kural edinimi** gösterildi. Aynı veriyi gün içinde işlemek aynı son kurala ulaştığı için, bu modelde ayrı bir “uyku” evresinin zorunlu olduğu iddiası desteklenmedi. Yanlış bağlama uygulama ve gözlem hataları, kusursuz bütünleştirme iddiasının sınırlarını ortaya koydu.

**Bu sonuç, beynin çalışma formülünün veya genel yapay zekânın bulunduğu anlamına gelmez.** Zengin aday dili ve bağlam ayrımı araştırmacı tarafından verildi; ortak bir sinirsel temsil kendi kendine oluşmadı. Bu tur, geniş yaşam-boyu öğrenme problemini çözmek yerine onun belirli iddialarını sınırlar. Önceki bütün geçerli sonuçlar, negatif bulgular ve kitap korundu.

## Biyolojik önermeler nasıl ele alındı?

Kullanıcının son açıklaması [protokolde aynen](PROTOCOL.md) korunuyor. Yeni bilgi edinmeyi her bilgi için yeni bir nöronun doğmasıyla özdeşleştirmedik. İncelenen çalışmalar mevcut sinapsların düzenlenmesini ve deneyimle ilişkili yeniden etkinleşmeyi de içeriyor. Hızlı kayıt ile daha yavaş bütünleştirme ayrımının hesaplamalı literatürde geçmişi var: [McClelland, McNaughton ve O'Reilly, 1995](https://doi.org/10.1037/0033-295X.102.3.419).

Uyku/dinlenme sırasında hipokampal ripple olaylarına müdahale belirli uzamsal bellek görevini bozabiliyor; ancak replay uyanıklıkta da gözlenmiş. Bunlar genel bir “üst bilinç görevcisi”, bilgi başına yeni nöron veya kusursuz aktarım kanıtı değil. [Girardeau vd., 2009](https://doi.org/10.1038/nn.2384); [Foster ve Wilson, 2006](https://doi.org/10.1038/nature04587). Beş kaynağın erişim düzeyi, ölçülmüş bulguları ve yorum sınırları [biyoloji notunda](BIOLOGY_AND_LITERATURE_TR.md) bulunuyor. Buradaki yapay hesaplama biyolojik uyku modeli değildir.

## Deneyin açık önkabulleri

Altı ikili özellik, 64 olası girdi var. Eski kural bir özelliğin değeri veya tersi; yeni kural iki/üç özelliğin paritesi veya tersi. Parite, seçilen bitlerin toplamının ikiye göre kalanıdır. Eski kural 24 gözlemden öğrenilir; ardından yeni kural için 24 farklı gözlem gelir. Yeni ortamın 40 girdisi hiçbir zaman eğitimde kullanılmaz.

Dar dilde sabitler ve tek özellikten oluşan 14 aday; zengin dilde en çok üç özellikli 84 aday önceden tanımlıdır. Öğrenici adayların gözlenen etiketlerdeki hata sayılarını tutar. En az hatalı adayların tamamının ortalama cevabını kullanır. Gizli öğretici, doğru bağlam sınıfı veya temiz hedef etiketi öğreniciye sızdırılmaz. Gürültü rejiminde temiz etiket yalnız değerlendiricidedir.

Üç rejim: doğru ve gözlenebilir iki bağlam; bağlamın gizlenmesi; bağlam görünürken yeni 24 etiketin dördünün bozulması. Her rejimde 24 yaşam ve sekiz yöntem: **576 edinim yolu**, **4.032 durum değerlendirmesi**, **13.824 etiket-öncesi tahmin**. Bunlar 576 bağımsız problem ailesi değil; yöntem ve rejimlerin aynı 24 yaşam üzerinde eşleştirilmiş çalıştırmalarıdır.

“Gündüz” boyunca bazı yöntemler yalnız geçici kayıtlarını günceller; online yöntem kalıcı hata özetini de günceller. “Konsolidasyon” evresinde yeni dış etiket gelmez. Eski/yeni kayıtlarla kural seçimi yapılır; ardından bütün ham olay listeleri kaldırılır. Öğrenilmiş aday kimlikleri ve hata sayıları kalır. `seen/unseen` değerlendirme kümeleri baştan sabit 24/40 ayrımıdır; ara gündüz anlarında `seen` kümesinin tümü henüz gözlenmiş olmayabilir.

Önemli sınır: gözlenebilir bağlamlarda ayrı kural bölmeleri kullanılır. Eski becerinin korunması bu ayrılmış bölmelerden yararlanır. Paylaşılan sinir ağı parametrelerinde girişim giderildiği veya eski ve yeni anlamların ortak temsilde bütünleştiği gösterilmez. Yeni hedef kuralı eski kuralla ortak bir soyutlama öğrenmek zorunda değildir; bu tur eski bilginin yeni görevi hızlandırdığını sınamıyor.

## Bu modelde kalıcı güncellemenin formülü

Verilen aday ailesi \(\mathcal H\), gözlenen bağlam \(c\) ve deneyim \((c_t,x_t,y_t)\) için her adayın hata sayısı:

\[
N_{t+1}(c,h)=N_t(c,h)+\mathbf1[c=c_t]\,\mathbf1[h(x_t)\ne y_t].
\]

En az hatalı aday kümesi ve kalıcı tahmin:

\[
\mathcal H_t^*(c)=\arg\min_{h\in\mathcal H}N_t(c,h),\qquad
p_t(1\mid x,c)=\frac{1}{|\mathcal H_t^*(c)|}\sum_{h\in\mathcal H_t^*(c)}h(x).
\]

Burada kalıcı deneyim izi, yalnız bugünkü cevap değildir: \(N_t\), sonraki olguların hangi adayları öne çıkaracağını da belirler. Ham olaylar olmadan aynı sabit dil ve kayıp için gelecekteki güncelleme sürdürülebilir. Bu özetin sonradan yeni özellik icat etme, tekil kayıt düzeltme veya her farklı sorgu için yeterli olduğu iddia edilmez.

Yeni gözlemlerin hata vektörleri toplamsaldır. Bu nedenle aynı kayıtları bir defada veya tek tek eklemek aynı \(N\)'yi, aynı son aday kümesini ve aynı son tahmini verir. Bu, bu deney için **koşullu bir kanıt**; genel beynin formülü değildir. Ayrı [matematik notu](MATHEMATICAL_LIMITS_TR.md), varsayımları ve zamanlamanın farklı sonuç verebileceği durumları açıklar.

## 1. Olay belleği ile kalıcı genelleme ayrıldı

Görünür, temiz bağlamda gündüz sonu yalnız belleğe dayanan sistem, gördüğü 24 olayın tamamını doğru hatırlar. Tüm 64 yeni girdide doğruluk %68,16; hiç eğitimde kullanılmamış 40 girdide ise %49,06'dır. Ham bellek kaldırılınca yeni görevin tümünde doğruluk %50'ye iner. Bu durumda yüksek görülen-örnek başarısı kalıcı bir kural edinimi değildir.

Aşağıdaki değerler konsolidasyon ve ham belleğin kaldırılmasından **sonraki** ortalama karesel hata (MSE): sıfır iyi, kesin yanlış bir. “Ayrılmış” 40 yeni eğitim-dışı girdiyi belirtir.

| Yöntem | Eski 64 girdide hata | Yeni 64 girdide hata | Ayrılmış 40 girdide hata | Yeni hedefi tam öğrenen yaşam |
|---|---:|---:|---:|---:|
| Donmuş eski kural | 0 | 0,500000 | 0,509375 | 0/24 |
| Yalnız olay belleği | 0 | 0,500000 | 0,509375 | 0/24 |
| 14 adayla doğru kayıt tekrarı | 0 | 0,455729 | 0,540611 | 0/24 |
| **84 adayla doğru kayıt tekrarı** | **0** | **0** | **0** | **24/24** |
| **Aynı 84 adayla gün içinde öğrenme** | **0** | **0** | **0** | **24/24** |
| 84 adayla yalnız yeni kayıtlar | 0 | 0 | 0 | 24/24 |
| Eski modelin kendi etiketleriyle tekrar | 0 | 0,500000 | 0,509375 | 0/24 |
| Etiket–girdi eşleşmesi bozulmuş tekrar | 0 | 0,439062 | 0,427955 | 0/24 |

Doğru etiketler tek başına yeterli değil: dar dil gerçek yeni pariteyi ifade edemiyor. Zengin dil altı kat aday karşılaştırması yapıyor; bu kıyas hesaplama eşitliği altında dilin bağımsız etkisi değildir. Tam hedef kuralının dar dilde bulunmaması doğrudan denetlenebilir; buna rağmen bir aday karışımının kısmi hata azaltabilmesi mümkündür. Yanlış eşlenmiş tekrarın bazı ortalama hataları donmuş modelden düşük çıkması da saklanmadı; bu, doğru genel kuralın edinildiği anlamına gelmez.

Ek önemli negatif sonuç: görünür bağlamda yalnız yeni kayıtlarla öğrenmek, eski kayıtları da tekrar işlemekle aynı sonucu yarı aday-kontrol maliyetinde verdi. Bu modelde eski ve yeni beceriler ayrı bağlam bölmelerindedir; eski kayıt replay'i burada gerekli değildir. Sonuç, tek paylaşılan temsilde eski deneyimlerin gereksiz olduğunu göstermez.

## 2. Aynı bilgi ve aynı temel güncelleme hesabında uyku benzeri evre zorunlu çıkmadı

Online ve ertelenmiş zengin yöntemlerin **72 yaşam–rejim eşleşmesinin tamamında** son hata istatistikleri, aday kümeleri ve çekirdek tahminleri tam aynı. İkisinde de ortak eski edinimden sonra **4.032 aday–örnek hata kontrolü** yapılıyor. Kaynak ediniminin ortak 336 kontrolü buna ayrıca eklenir.

Görünür temiz bağlamda etiket gelmeden ortalama hata ertelenmiş yöntemde **0,484375**, online yöntemde **0,080072**; online yöntem 24 yaşamın tamamında daha iyi. Gürültülü bağlamda karşılıkları **0,484375** ve **0,189135**. Gizli bağlamda **0,484375** ve **0,456163**; 7 yaşamda online iyi, 17 eşit.

Bu, online yöntemin tam CPU eşitliğinde üstünlüğü değildir: görünür bağlamda online 49, ertelenmiş yöntem 2 bağlam-aday seçimi çağrısı yapar. Gündüz kısa yanıt süresi veya enerji kısıtı konulmadı; böyle kısıtlar altında ayrı işleme evresinin faydası hâlâ açık. Günlük girdiler tekrar etmediği için tam anahtarlı bellek etiket-öncesi sorguya isabet etmez; bu tasarım gerçeği gündüz ölçüsünün yorumuna dahildir.

## 3. Bağlam eksikliğini daha çok iç işleme her durumda gideremez

Gizli bağlamda eski ve yeni kurallar 64 girdinin 32'sinde farklı doğru cevap ister. Öğrenici sorgu sırasında hangi hedefin istendiğini göremez. Ortak çekirdeğin konsolidasyon sonrası sonuçları:

| Yöntem | Eski hedef MSE | Yeni hedef MSE | Eşit ağırlıklı birleşik MSE |
|---|---:|---:|---:|
| Donmuş eski kural | 0 | 0,500000 | 0,250000 |
| Eski+yeni kayıtlarla zengin öğrenme | 0,177083 | 0,260417 | 0,218750 |
| Aynı öğrenmenin online sürümü | 0,177083 | 0,260417 | 0,218750 |
| Yalnız yeni kayıtlarla zengin öğrenme | 0,500000 | 0 | 0,250000 |

Tek bir çelişen girdide aynı \(p\) olasılığı için:

\[
\frac{p^2+(p-1)^2}{2}=(p-\tfrac12)^2+\tfrac14\ge\tfrac14.
\]

Girdilerin yarısı çeliştiğinden bütün uzaydaki birleşik MSE en az **1/8 = 0,125**. Bu alt sınır deney algoritmasına değil, sorgudaki bilgi eksikliğine bağlıdır. Yöntemimizin 0,218750 değeri sınıra ulaşmıyor; algoritmanın optimal olduğu gösterilmedi. Bağlam görünür olduğunda iki gereksinim birlikte sıfır hatayla karşılanabiliyor. Bağlamı deneyimden keşfetme ise bu turda yapılmadı.

Bu karşı örnek, eski yanlış bir inancı sonsuza dek korumayı istemiyor. Eski ve yeni iki geçerli görevin aynı anda korunması talebi açıkça konduğunda, onları ayırt etmeye yetecek bilginin gerekliliğini gösteriyor.

## 4. Gürültüde güçlü sonuç var, kusursuzluk yok

24 gündüz kaydının dördü yanlışken zengin doğru-kayıt yöntemi 23/24 yaşamda gerçek yeni kuralı tam belirledi. Ortalama yeni MSE **0,005208**, ayrılmış girdilerde **0,006250**. Bir yaşamda iki aday eşit hatayla kaldı; bu yaşamın yeni MSE'si 0,125, ayrılmış MSE'si 0,15 oldu. Sonuçları gördükten sonra ayrı bir [tanık denetimi](../../../research/living_learning_consolidation_2026_09_22/results/exact_witnesses.json) eklendi; ana protokol değiştirilmedi.

Bu tanıkta iki olası kural `x₂ XOR x₄ XOR 1` ve `x₀ XOR x₂ XOR x₃`. Her biri aynı 24 gözleme tam dört hatalı etiketle uyuyor. İlkine göre bozuk gözlem konumları `[0,6,7,23]`, ikincisine göre `[11,13,18,20]`. Aynı görülen kayıtlar her iki dünyada da mümkün. Kurallar bütün uzayda 32, ayrılmış 40 girdinin 24'ünde farklı. Sıfır girdisi için cevapları sırasıyla 1 ve 0.

Dolayısıyla “dört bozuk etiket var” bilgisi verilse bile tek başına hangi kuralın gerçek olduğunu garantiyle belirlemiyor. Ek önbilgi tercihi yapılabilir, fakat bu bütün mümkün dünyalarda kusursuzluk değildir. Daha fazla ayırt edici gözlem gerektiren durumlar vardır; iç üst denetçi aynı bilgiyle bu sınırı aşamaz.

Gürültülü ham belleğin korunması da otomatik fayda değil: bellek girdiye tam isabet ettiğinde hatalı etiketi kesin cevapla öne çıkarır. Temiz kuralı öğrenmiş çekirdek bile bu seçim kuralıyla yanlış olay belleği tarafından bastırılabilir. Bu deney bir güvenilirlik kapısı öğrenmiyor; hangi kaynağın ne zaman üstün tutulacağı açık bir araştırma sorusu olarak kalıyor.

## Matematikle ne kanıtlandı?

[Matematik notunda](MATHEMATICAL_LIMITS_TR.md) beş koşullu önerme ve kanıtı var: yeterli toplamsal istatistik altında son-nokta zamanlama eşitliği; yeni dış kanıt olmadan ayırt edilemeyen dünyaların sınırı; bağlamsız çelişik görevlerde hata alt sınırı; sabit özelliklerde yeni boyut gerektirmeyen tam öğrenme; temsil dönüşümünün yalnız cevaplarla değil gelecek güncellemelerle de uyumu. Çift yönlü bir döngünün kararlı olmasının doğru olduğu anlamına gelmediği de karşı örneklerle gösteriliyor.

Sabit özellikli ayrı bir ridge örneğinde artımlı ve toplu çözüm dört adımın her birinde tam rasyonel sayılarla eşleşti; son parametreler `(10/11, 3/11)`. Bu, ana parite deneyinin modeli değildir; “her yeni deneyim için yeni boyut gerekir” evrensel iddiasına koşullu karşı örnektir. Sabit sayıda gerçek sayı tutmak, sınırsız bilgiyi sabit bit bütçesine sığdırmak demek değildir.

Bu kanıtlar kullanıcının bütün biyolojik açıklamasını doğru kabul edip sonuç üretmiyor; hangi alt iddianın hangi bilgi ve işlem koşullarında savunulabilir olduğunu ayırıyor. Yalnız frekans tahminini sürekli güncelleyen dar bir sistem de çalışırken öğrenir; bundan genel görev kapasitesi çıkmaz. Yaşarken öğrenme ve genel yapay zekâ eşanlamlı kabul edilemez.

## Hipotezlerin sonuç sonrası durumu

| Hipotez | Durum |
|---|---|
| H18: olay belleği ile kalıcı genelleme ayrımı | Desteklendi; ham belleği silme ve eğitim-dışı sorgu ayrıldı |
| H19: verilen temsil dilinin uygunluğu | Dar/zengin karşılaştırmada destek; daha fazla hesaplama ve hazır dil önbilgisi açık |
| H20: aynı özet altında zamanlama eşitliği | 72/72 eşleşme ve toplamsallık kanıtı; ayrı evrenin genel zorunluluğu desteklenmedi |
| H21: eski modelin kendi etiketlerinin sınırı | Yeni hedef edinimi başarısız; genel bilgi sınırı iki-dünya kanıtıyla ayrıldı |
| H22: bağlam eksikliği ve çelişki | Kesin alt sınır ve görünür/gizli karşılaştırması destekliyor |
| H23: kusursuz denetim sınırı | Gürültüde başarısız tek yaşam + iki uyumlu dünya tanığı; öğrenilmiş denetçi modeli denenmedi |
| H31: ham kayıt olmadan etki | Sabit aday dilinde kural ve güncelleme istatistiği kalıcı; bilinçli hatırlamaya ilişkin sonuç yok |
| H33: girdi–etiket bağının önemi | Eşleşme bozulunca tam yeni kural 0/24; bütün görevlerde aynı öneme genellenmiyor |
| Diğer H24–H41 | Kayda geçti; protokolde açıkça işaretlenen kısmi ölçüler dışında denenmedi |

## Denetim, koruma ve dosyalar

Ana çalıştırma tam tekrarlandı ve JSON sonuçları eşleşti. 5.376 aday çıktısı, 9.216 dünya etiketi ve son durumlarda 73.728 sorgu skaler hesapla denetlendi. 13.824 tahminin etiketten önce üretildiği doğrulandı. 24 ayrı süreçte, ham olay belleği olmadan yeniden başlatılmış durumun sonraki altı gözlemdeki tahmin ve güncellemesi, kesintisiz nesneyle tam eşleşti. Bu devam işlemi ortak bir tanısal istatistik güncellemesidir; donmuş kontrolün özgün ikinci-gün politikasının öğrendiği anlamına gelmez.

Bağımsız kod incelemesinde durum dışa aktarma/yüklemede liste paylaşımı riski düzeltildi. Aynı kayıttan iki nesne oluşturup birini değiştirmenin diğerini veya kayıt anlık görüntüsünü değiştirmediği sınandı. Düzeltilen dosyalar yalnız bu yeni deneyin dosyalarıdır.

Önceki **924 dosyanın SHA256 değerleri ve Git HEAD** korundu. Yerel araştırma; commit, push, sürüm veya DOI yayını yok. Tekrar üretim [README'de](../../../research/living_learning_consolidation_2026_09_22/README.md); [son denetim](../../../research/living_learning_consolidation_2026_09_22/results/final_audit.json) kaynak, sonuç ve dosya bağlarını kaydeder. Yerel denetim dış hakem değerlendirmesi değildir.

Araştırma uzayı şu yönde daraldı: sırf beklemek, eski cevapları tekrar üretmek, çelişik bağlamı görünmez bırakmak veya kusursuz denetçi varsaymak genel çözüm sağlamıyor. Hâlâ açıklanması gerekenler **uygun temsil ve bağlam ayrımının deneyimden oluşması, yeni bilginin kaynağının korunması, gelecek güncellemeyi destekleyen kalıcı durum ve sınırlı kaynak altında güvenilir denetim**. Genel öğrenme problemi bu küçük aday-seçim modeline indirgenmedi.
