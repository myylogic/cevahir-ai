# Tur 15 — Gerçek olay dizisinden kullanılabilir bellek yapısı

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Ana sonuçlardan önce yerel protokol

Tur 14'te h hazır bir geçmiş özeti olarak verilmişti. Burada öğrenici bitleri sırayla alır; bölümün sonundaki sorguya cevap verir, sonra tek doğru etiket gelir. Bölüm başında açık sıfırlama vardır. Ara bitlerde doğru durum veya etiket verilmez. Bütün yöntemler aynı bölüm ve sonuçları alır; model parametreleri yalnız terminal etiket geldikten sonra değişir. Öğrenilen parametreleri her bit gelişinde gizli bir bellek olarak değiştiren yöntem bu deneyde yoktur.

961 önceki dosya ve HEAD korunur; kitap/ana sistem entegrasyonu, commit veya push yok. Bu çalışma daha uzun bir çalışma yaşamındaki **bölümler arası öğrenmeyi** inceler. Bellek yapısı değişiminin bölüm ortasında eski durumdan yeni duruma nasıl aktarılacağı açık kalır. Sürekli genel sistem tamamlanmış sayılmaz.

## Sorular

H46: Öğrenici, yalnız terminal sonuçlardan hangi son-geçmiş ayrımlarının yararlı olduğunu öğrenerek değişken derinlikli bir bellek kullanım yapısı edinebilir mi?

H47: Geçmiş ufkunun uzunluğu ile gerekli iç durum sayısı ayrı şeyler midir? Sınırsız ufuk iki durumla, beş-bit gecikme ise 32 durumla ifade edilebilen karşılaştırmalar kullanılacak.

H48: Eğitimdeki uzunlukların dışına çıkınca hangi edinimler sürer? Görülmüş sözcükleri ezberleme ile sonraki akışın kuralını edinme ayrılacak.

H49: Aday bellek biçimi önbilgisi başarıyı ne kadar belirler? Aynı bilgiyi alan bir son-ek ağacı ile iki durumlu yinelemeli model birbirinin bütün görevlerini çözmek zorunda değildir.

## Dünya

İkili girdi bölümleri. Her bölümden sonra tek çıktı sorulur:

- `last_bit`: son bit.
- `delay_5`: sondan beşinci bit.
- `parity`: bütün bölümdeki 1'lerin sayısının ikiye göre kalanı.

24 yaşam; tohumlar 0–23. Her yaşamda eğitim girdileri bütün görev/yöntemler arasında aynı 1.024 farklı sözcüktür. Aday uzunluklar 8–24 aralığından eşit olasılıkla, aday bitler bağımsız eşit olasılıkla önerilir; daha önce kabul edilmiş sözcükler reddedilir. Bu nedenle kabul edilen örneklerin uzunluk dağılımı tam bağımsız ve eşit olasılıklı değildir. Görev yalnız etiketi değiştirir. Gerçek görev adı öğreniciye verilmez.

İki sabit değerlendirme kümesi: eğitim sözcükleriyle çakışmayan 128 yeni sözcük (8–24 uzunluk), 128 uzun sözcük (32–64 uzunluk). Bu kümelerden etiket öğrenilmez, hiperparametre seçilmez. 0/32/128/512/1024 etiket sonrasında dondurulmuş model durumundan sorgulanır. Değerlendirme sırası model parametrelerini değiştirmemeli.

## Yedi yöntem

1. `kt_constant`: girişten bağımsız ikili etiket sayacı, Jeffreys/KT yarım-sayaç düzeltmesi `(n1+.5)/(n0+n1+1)`.
2. `suffix_1`: son bit için ayrı KT sayaçları.
3. `suffix_5`: son beş bit için ayrı KT sayaçları. Gecikme görevi için doğru uzunluğu tesadüfen önceden içerir; bu üstün önbilgi açıkça belirtilecek.
4. `suffix_8`: son sekiz bit için ayrı KT sayaçları.
5. `map_tree_8`: en çok sekiz derinlikte, son bitten geriye dallanan ağaç. Her düğüm kendi bağlamındaki KT marjinal olabilirliğini tutar. İç düğümde durma ve iki çocuğa bölünme ön olasılığı 1/2; derinlik sınırında durma zorunlu. Bütün eldeki gözlemlere göre en yüksek posteriorlu ağaç dinamik programlamayla seçilir; bu seçimin yaprak sayaçları tahmin verir. Eşit puanda durma seçilir. Güncelleme her terminal etiket sonrası yapılır; maksimum derinlik verilmiştir.
6. `mixture_tree_8`: aynı sayaçlar, aynı ağaç ön olasılığı, fakat tek en iyi ağaç seçmek yerine bütün ağaçların tam karışım tahmini. İç düğümün olabilirliği `0.5*KT(node)+0.5*W(left)*W(right)`. Bu, koşullu son-ek ağacı karışımıdır; yeni öğrenme ilkesi iddia edilmez.
7. `two_state`: iki gizli durumlu, iki girdi sembollü deterministik geçiş tablosu ve ikili terminal okuma tablosu. Başlangıç durumu 0. 16 geçiş tablosu × 4 okuma tablosu =64 gösterim hazır verilir; tekdüze gösterim ön olasılığı. Gelen bitler boyunca her adayın durumu taşınır. Terminal etikette eşleşme .95, uyuşmazlık .05 olabilirlik; temiz çıktı aday ortalamasıyla tahmin edilir. Gerçek dünya gürültüsüzdür; .05 öğrenici varsayımıdır. Geçiş/okuma doğru kimlikleri verilmez. Yinelenen işlev gösterimlerinin ön olasılık etkisi yeni bir buluş sayılmaz.

Sonek sayaçları yalnız gerekli son bitleri bölüm içinde tutar; bütün bölüm ham olarak öğrenici belleğinde bırakılmaz. Ağaç yöntemlerinde 511 düğümlük sabit üst kümenin sayaçları tutulur; seçilen ağaç büyüse bile altta tüm bu istatistikler vardır. Bu nedenle minimal bellek tüketimi veya mutlak kapasite artışı iddia edilmeyecek. Değişen şey hangi son-ek ayrımlarının tahmin için kullanıldığıdır. İki durumlu yöntem öğrenilmiş yinelemeli geçişleri kullanır, fakat iki-durum dili hazırdır.

## Ölçü ve maliyet

24×3×7 =504 öğrenme yolu, 516.096 etiket-öncesi tahmin. Her bölümde tahmin önce, dış etiket sonra. İlk 128 ve son 128 eğitim bölümünde hata; 1.024 bölüm toplamı; beş değerlendirme noktasında kısa/yeni ve uzun sözcüklerde MSE ve eşitlikte beklenen doğruluk. Testler dondurulmuş kalıcı durumun kopyalarıyla, gözlem eklemeden yapılır.

Ayrıca MAP ağacının yaprak sayısı ve maksimum seçilen derinliği; karışımın kökte durma ağırlığı; iki durumlu modelin etkin aday sayısı; kalıcı sayaç/dizi ve JSON boyutu; girdi biti başına durum geçişi, sayaç güncellemesi, ağaç düğümü yenilemesi, aday-etiket kontrolü kaydedilir. Başarı eşit etiket akışındadır; bütün modeller aynı işlem veya belleği kullanmaz. Modelin kodunda ve ortak hazır dillerdeki önbilgi, learner JSON boyutuna dahil değildir.

## Matematiksel karşı örnekler

Eğitimden sonraki aynı model durumu kopyalansın. `000000000` ile `100000000` aynı uzunluk ve aynı son sekiz bite sahip, fakat farklı parity sonucu gerektirir. Yalnız son sekiz bit ve uzunluğa erişen sabit-parametreli bir sorgulayıcı aynı p'yi verir; eşit ağırlıklı ortalama MSE en az 1/4. İki durumlu tekrar güncellemesi bu ayrımı taşıyabilir. Bu, bütün sonlu belleklere veya sinir ağlarına imkânsızlık teoremi değildir.

Gecikme-beş fonksiyonunun bütün devamlar için doğru hesaplanması, beş bitlik bütün son-ekleri ayırt etmeyi gerektirir; en az 32 durum. İki durumlu model ailesinin bu görevde yetersiz olması beklenen karşı kontroldür. Parity ve gecikme karşılaştırması “uzun ufuk” ile “büyük durum kapasitesi”ni ayırır.

Geçmişleri bütün gelecek devam sorgularında aynı sonuca götürmelerine göre eşdeğer saymak, öğrenilmiş duruma aday bir tanımdır. Bu, bilinen otomata/öngörü durumu fikirleriyle ilişkilidir; yeni kuram diye sunulmaz. Sonlu gözlemde eşit görünen geçmişlerin tüm geleceklerde eşitliğini garanti etmek ayrıca varsayım gerektirir.

## Denetimler

Skaler öğretici ve otomata yürütmesi; KT oranı tam kesir kontrolü; küçük derinlikte tüm ağaçları numaralandırarak MAP ve karışım doğrulaması; yeni sözcüklerde etiket sızıntısı ve değerlendirme sırası kontrolü; aynı-son-ek parity tanığı; iki-durum ailesinde gecikme-beşin bulunmaması; durum dışa aktarma izolasyonu; 21 ayrı süreçte öğrenme devamı. 504 yolun ana sonucu tam tekrar hesaplanır; 961 dosyanın içeriği ve HEAD korunur.

Bu tur hazır geçmiş özetini kaldırır, ama bütün temsil önbilgisini kaldırmaz. Sistemin uygun **bellek dili**ni kendi deneyiminden icat etmesi, kendi öğrenme hedefini oluşturması, bölüm ortasında temsil dönüşümü ve genel kendi kendine gelişme açık kalır. Mevcut başarılar bu büyük boşluğun çözümü olarak sunulmayacak.
