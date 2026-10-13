# Tur 19 — Deneyimden yeni ayırt edici devamlar kurmak

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Performans öncesi yerel protokol

## Soru ve önceki araştırmaya ek

Tur 15'te son-ek derinliği ve iki-durum adayları hazırdı; tur 16 çalışan duruma verilen yeni kuralın taşınmasını, tur 18 verilen değişikliklerin etkileşimini sınadı. Burada doğru durum sayısı ve ayırt edici devam listesi verilmez. Aynı görünen geçmişlerin farklı devam sonuçları, öğrenicinin kullandığı ayrımları ve yürütülebilir durum grafiğini genişletebilir mi? Kalıcı yapı, aynı kanıttan yeniden kurmaya göre hangi hesabı önler?

Bu, Angluin'in gözlem tablosu/L* yaklaşımıyla doğrudan örtüşür. Yeni algoritma veya temel ilke icadı iddiası yoktur. Ama önceki sabit aday listemizi kaldıran, deneyimle oluşturulan ayrımları çalışır bir programda sınayan yeni bir araştırma adımıdır. Alfabe, deterministik sonlu-durum varsayımı, doğru ikili çıktı, sıfırlama ve istenen diziyi sorgulayabilme hazır verilmiştir.

## Dünya ve bilgi sınırı

24 tohum × üç dünya. Her dünyada sabit, bilinmeyen bir ikili-girdi/ikili-son-çıktı makinesi vardır. Doğru tablo yalnız ortam ve değerlendiricide bulunur. Öğrenici örnek dizileri ve sonuçları alır; durum kimliği, durum sayısı, aile etiketi, geçiş tablosu veya tam eşdeğerlik sorgusu almaz.

1. `cycle_reset`: 5–8 durum; bir sembol döngüde ilerletir, öteki başlangıca döndürür; yalnız bir durumun çıktısı 1. Sembol kimliği ve durum numaraları tohuma göre karıştırılır.
2. `random_cycle`: 6/10/14 durum; bir sembol erişilebilir döngü, diğer sembol rastgele geçiş; her duruma rastgele çıktı, tüm çıktıların aynı olması durumunda sonuncu terslenir. Semboller ve durum numaraları karıştırılır. Verilen durumların bazıları davranışsal olarak eşdeğer olabilir; değerlendirme erişilebilir minimal sayıyı ayrıca bulur.
3. `delay`: 3/4/5-bit kaydırma belleği; sondan d'inci bit, kısa sözcükler sıfırla doldurulur. 8/16/32 gizli durum. Sembolleri ve durum adlarını karıştırmak yalnız temsil rastlantısını değiştirir.

Her yaşamda önceden üretilmiş 64 dış gözlem dizisi, 16 dört-örnekli blok halinde gelir. Uzunluk 0–24 eşit, bitler bağımsız eşit; tekrarlar reddedilmez. Aynı tohumun dizileri bütün aile ve yöntemlerde aynıdır. Bir blok içinde model parametreleri güncellenmez: her dizinin tahmini kendi etiketi açılmadan yapılır. Blok sonunda dört sonuç birlikte öğrenmeye katılır. Başlangıçta ve her blok sonunda tablo kararlılaştırılır. Dolayısıyla düzeneğin sıfırlanabilir bölümleri vardır; kesintisiz beden/akış güncellemesi gösterilmez.

Öğrenici gerektiğinde başka dizileri seçip ortamdan sonuçlarını alabilir. Her yeni sorgu bir sıfırlama ve dizinin uzunluğu kadar eylem tüketir; önceden aynı dünyada alınmış sonuçlar yeniden kullanılabilir. Sorgu sonuçlarını kendisi üretip doğru etiket saymaz. Farklı yöntemler farklı ek sorgular yapabilir; bu karşılaştırma aynı dış gözlem akışında fakat mutlaka eşit toplam kanıtta değildir. Aynı kanıt kontrolü aşağıdaki yeniden kurma yöntemidir.

## Yapının kurulması

S erişim geçmişleri, E ayırt edici devamlar; başlangıç ikisi de yalnız boş sözcüktür. Satır `row(s)=(y(se):e∈E)` erişilebilir sorgularla doldurulur. S ve S·{0,1} satırları ölçülür. Yeni bir satır mevcut S satırlarından farklıysa onun erişim dizisi S'ye eklenir. Eşit görünen iki S satırı aynı a sembolünden sonra e devamında farklı sonuç veriyorsa `a+e` yeni ayrım olarak eklenir. Böylece yeni devam, gözlenen kısa ayrımların önüne bir eylem bağlanarak oluşur; gizli hedef yapıdan alınmaz.

Kapalı ve tutarlı tablo, satırları durum olan bir makineye derlenir. Bu aday erişilmiş bütün etiketlere karşı denetlenir; uyumsuz bir sözcük varsa bütün önekleri S'ye eklenir ve işlem tekrarlanır. Bilinen sözcüklerde uyuşma, bütün bilinmeyen sözcüklerde doğruluğun sertifikası sayılmaz. Tam eşdeğerlik denetimi yalnız araştırmacıda ve geri bildirim vermeden yapılır.

## Dört yöntem

1. `adaptive`: yukarıdaki klasik gözlem tablosunu, edinilmiş S/E ve sonuç önbelleğini yaşam boyunca taşır.
2. `rebuild`: her blok sonunda başlangıçtan aynı algoritmanın o ana kadarki bloklarını yeniden yürütür. Dünya sonucu önbelleği korunur; daha önce yanıtlanmış sorgu yeniden çevrede yürütülmez. İç öğrenici kendi mantıksal geçmişinde henüz sorgulamadığı kayıtları önceden tarayamaz. Yeterli kaynakta adaptive ile aynı bilgi, sorgu sırası ve son programı vermesi beklenir. Yeniden kurmanın hesap maliyeti yazılır; bu kontrolü yenmek yeni bilgi kazanımı veya yeni algoritmik üstünlük değildir.
3. `exhaustive_tests`: yeni `a+e` ayrımı gerektiğinde yalnız onu eklemek yerine o uzunluğa kadar bütün ikili devamları E'ye ekler. Aynı tablo/derleme/karşı-örnek yordamıdır. Güçlü fakat pahalı bir ayrım kapsamı kontrolüdür; üstelik daha çok dış kanıt toplaması mümkündür.
4. `fixed_tests`: E yalnız boş devamdır. İlk tablo tutarsızlığında son tutarlı programını korur; S ve yeni gözlemleri saklar ama yeni ayrım oluşturamaz. Bu kapasite ablasyonudur; genel amaçlı güçlü öğrenici diye sunulmaz. Diğer esas kontroller klasik adaptive, aynı-kanıt rebuild ve geniş test ailesidir.

16 blok × 24 yaşam × 3 aile × 4 yöntem = 4.608 blok sonucu, 288 öğrenme yolu, 18.432 etiket-öncesi dış tahmin. Başlangıç değerlendirmeleri ayrıca tutulur.

## Kaynak ve değerlendirme

Her yaşam için en çok 100.000 çevre birimi (sıfırlama+sembol), 20.000.000 sanal iç iş birimi. Dış pasif blok akışı bütçe dolsa da gelir ve kendi çevre maliyeti kaydedilir; kalan ek sorgu/öğrenme dondurulur. İç öğrenme sınırında son tamamlanmış model korunur; tamamlanmamış yeni tablo başarı sayılmaz. Dış akışın zorunlu bedeli nedeniyle toplam çevre sayısı sınırı aşarsa ayrı raporlanır. Kaynakların tek skaler fayda sayısına çevrildiği varsayılmaz.

Sınırlandırılan iç öğrenme işi: tablo hücresi okuma, dizi/satır karşılaştırma, yeni ayırıcı ve önek saklama, aday geçiş/çıktı kurma, bilinen sözcüklerin modelle yürütülmesi ve tekrar oynatma. Dış akış tahminlerinin geçiş/çıktı işi ayrı sayaçta tutulur ve öğrenme donduğunda da sayılır. Sözlük/hash, Python nesne ve sıralama maliyetlerinin tamamı sayılmaz; ölçü CPU/RAM değildir. Dış sorgu sayısı, eylem bedeli, cache kayıtları, S/E boyutu, öğrenilmiş durum sayısı, JSON belleği ayrı ölçülür. Kütüphane kaynak kodu JSON boyutuna girmez. Ham sorgu arşivi kaldırılmaz; arşivsiz öğrenme iddiası yoktur.

Başlangıç ve 16 blok sonunda: (a) dış akışla çakışmayan 128 uzun dizide (32–64) hata; (b) öğrenilen ve gizli makinenin erişilebilir durum çarpımında BFS ile **bütün sonlu diziler için tam eşdeğerlik**; (c) eski bilinen kayıtlarla çelişki; (d) geçici olarak eşit görünen geçmişlerin yeni sütunla ayrıldığını gösteren tanıklar. Tam eşdeğerlik ve uzun-dizi etiketleri öğreniciye dönmez. Hata bulunursa değerlendiricinin karşı örneği de öğrenme için kullanılmaz.

Kalıcılık: seçili tohumlarda blok 4 ve 8 durumlarını JSON'dan yeni süreçte devam ettir; sonraki kararlar ve iş toplamları kesintisiz çalışmayla aynı olmalı. Bütün ana sonuç tam yeniden üretilecek. Bağımsız skaler ortam yürütmesi, küçük makinelerde kapsamlı tablo koşulları, öğrenici ile değerlendirici bilgi sınırı, doğru geçmişte farklı devam tanığı, aynı-kanıt yeniden kurma ve önceki dosyaların korunması denetlenecek.

## İddiaları sınırlayan karşı örnekler

Gerçekten bütün devamlar altında eşdeğer geçmişler sabit dünyada daha sonra ayrılmaz; öğrenicinin sınırlı E altında geçici eşitlemesi ayrılır. Dünya bu turda değişmez. Sonlu gözlemlerde doğru model, erişilmeyen uzun devamda yanlış olabilir. Tam bir doğru sonlu makine elde edilirse aynı dünyanın gelecekteki çıktıları için tüm ham geçmiş zorunlu değildir; eksik model ile tam model karıştırılmaz. Sıfırlama ve yeniden sorgu açıksa eski ayrıntının kaybı bazen bilgi kaybından çok yeniden edinim maliyetidir.

Geniş testlerin masrafı, kalıcı tablonun yararı veya durum sayısının artması tek başına yeni living-learning kuramı değildir. Kavram/değer dili, amaç oluşumu, gürültü, değişen dünya, eski becerilerin farklı görevlerde korunması ve kaynakla sınırlı açık uçlu yaşam bu turda çözülmez. Kitap, eski araştırmalar, ana sistem ve yayın dosyaları korunur; commit/push yoktur.
