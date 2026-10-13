# Deneyim gelecekteki hesaplamayı nasıl değiştiriyor? Edinilmiş program parçaları deneyi

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma turu 11

**Genel yaşarken öğrenme problemi hâlâ çözülmedi.** Bu turda, deneyimin gelecekteki hesaplamayı değiştirebildiği başka bir bölge sınandı: önceki örneklerden edinilmiş programların parçaları, daha sonraki bir problemin çözümünde kullanılabiliyor mu? Sonuç, bu küçük program dilinde evet; fakat etki hem olumlu hem olumsuz. Doğru geçmiş birikse ve eski cevaplar bozulmasa bile, aynı kaynak sınırında gelecekte edinilebilen beceriler daralabiliyor.

12 sentetik yaşamda **9.216 ana koşul**, sonuçlardan sonra **384 ek kontrol** çalıştırıldı ve yeniden üretildi. Bunlar 9.600 bağımsız dünya değildir: yaşam başına 16 gelecek görev, aynı geçmiş kopyaları, yöntemler ve bütçeler arasında tekrar sınanıyor. Ana sonuç, yeni bir temel işleç veya hesaplanabilirlik sınıfı icat edildiği değildir. Deneyim, mevcut dille yazılabilen programların hangilerinin sınırlı hesapla önce bulunacağını değiştiriyor.

## 1. Başlangıç noktasını kaybetmeden yön değiştirmek

[Tur 10](../living_learning_geometry_2026_09_21/REPORT_TR.md), öğrenilebilir sinirsel temsilde koruma kısıtlarının yeni edinimi baskılayabildiğini göstermişti. Bu tur o koruma problemine yeni bir varyant eklemek yerine **edinilmiş hesap parçaları ve gelecekteki arama** yönüne geçiyor. Eski araştırmalar, negatif sonuçlar ve kitap dosyaları değiştirilmedi.

Bu yönün tamamı yeni değildir. [Önceki temsil büyütme deneyinde](../living_learning_reassessment_2026_09_20/REPRESENTATION_GROWTH.md) çarpım özellikleri edinilmiş, fakat aynı etkinleştirme sırasını kullanan sabit geniş kontrol tam eşleşmişti. O sonuç geçerli. Burada birden fazla edinilmiş programdaki ortak parçalar çıkarılıyor; onların başka programları edinmeye etkisi ve maliyeti ölçülüyor. Aynı bilgiyi taşıyan güçlü sabit kontrol yine korunuyor.

[Sorguya göre yeterli durum](../living_learning_query_state_2026_09_21/REPORT_TR.md), gelecekte yapılacak işlerin tutulması gereken geçmişi belirlediğini göstermişti. Bu turda saklanan durum yalnız eski soruların cevaplarını değil, yeni bir problemi çözmek için hangi adayların önce üretileceğini de belirliyor. Bu farklılık, geçmiş sonuçları geri almadan araştırma alanını genişletiyor.

## 2. Deneyimden ne ediniliyor?

Girdi ve çıktı `0..100` aralığında, bütün işlemler mod 101. Başlangıç dili dört işlemi biliyor: bir ekleme, ikiyle çarpma, karesini alma ve işaret değiştirme. Program, bunların sonlu bir dizisi. Örneğin `[kare, ekle, ikiyle çarp]`, `2(x²+1)` hesaplar. Yorumlayıcı, dil, arama yordamı ve sayısal etiketin anlamı araştırmacı tarafından verilmiştir.

Her yaşamda iki farklı üç-işlemlik motif seçiliyor. İlk 12 görev bu motiflerden veya başına/sonuna bir işlem eklenmiş hallerinden kuruluyor. İlk iki görev motifleri tek başına içeriyor: **bu elverişli bir müfredattır** ve tam program belleğinin güçlü olmasının nedenlerinden biridir. Motifler iki veya daha az işlemle bütün alanda aynı fonksiyonu vermeyen üçlülerden seçiliyor. Veri oluşturma için kullanılan bu bilgi öğreniciye verilmiyor.

Öğrenici her görevde sekiz girdi–çıktı örneğinden program arıyor. İlk tutarlı aday ayrı sekiz kabul örneğinde sınanıyor. Kabul başarısızsa o görev için çekimser kalıyor; bu kabul etiketleri yeni aday seçmek için yeniden kullanılmıyor. Değerlendirici, programı ayrıca alanın bütün 101 girdisinde denetliyor. Öğrenici hedef programı veya bütün-alan cevaplarını almıyor.

İlk görevlerin hepsi aynı geçmişsiz arama ile ediniliyor. Kabul edilmiş programların kendileri bankaya yazılıyor; öğretmenin programı yerine konmuyor. 144 ilk görevin tamamında edinilen program bütün 101 girdide doğru bulundu. Bu gürültüsüz küçük deneyin sonucu, sonlu örneklerden genel doğruluk garantisi değildir.

Program bankasında en az iki programda görülen, uzunluğu iki veya üç olan bitişik parçalar sayılıyor. Basit bir sıklık/uzunluk puanıyla en fazla dört parça seçiliyor. Puan bir sıkıştırma sezgisidir; bütün parçaların birlikte optimum sıkıştırmasını veya aktarım değerini garanti etmez. Sistem yeni bir primitive keşfetmiyor; **hangi bileşik hesap parçalarının gelecekte öncelikli kullanılacağını deneyimden seçiyor**.

## 3. Gelecekteki edinim nasıl ölçüldü?

[Ana protokol](PROTOCOL.md) ana sonuçlardan önce yerelde yazıldı. 0, 6 ve 12 ilk görev sonrasında durum kopyaları alındı. Her kopyaya sekiz ilgili ve sekiz farklı motifli yeni görev verildi. İlgili görevler eski motiflerin yeni bileşimleri, diğerleri aynı dört işlem dilinden başka motiflerin bileşimleri. Bu ikinci koşul **dilin veya bütün alanın değişmesi değil, motif ailesinin değişmesidir**.

Her yaşamın 12 eski ve 16 yeni hedefi bütün-alan işlevi bakımından birbirinden farklı. Eski programı aynı görev adıyla geri çağırmak yeni görevi çözmüyor; yeniden bileşim veya başka bir program aranması gerekiyor. Yeni probe sonuçları geçmiş bankasına geri yazılmıyor. Dolayısıyla bu çalışma durmadan kendini yeniden düzenleyen bir yaşam akışının tamamını değil, aynı geçmişten sonraki edinimi ölçen müdahale kopyalarını kullanıyor.

Arama genişlik öncelikli; önce daha az çağrıdan oluşan ifadeleri inceliyor. Öğrenilmiş bir parça bir çağrı olarak önerilse de **içindeki bütün primitive işlemler maliyete dahil**. Bütçeler 512 ve 2.048 primitive-vektör adımı. Bir adım aynı sekiz örneğe bir işlem uygulamak demek; sekiz skaler modüler işlem. Kabul sınamasındaki işlemler de bütçeden düşülüyor. Ebeveyn programların ara çıktıları bütün yöntemlerde yeniden kullanılıyor.

Sekiz arama girdisinde aynı çıktı vektörünü veren adaylar tekrar sayılıp budanıyor. Bu eleme yalnız gözlenmiş örneklere göre eşdeğerliktir. Başka girdilerde farklı bir doğru programın erken bir temsilci nedeniyle elenmesi mümkündür. Deneyde yanlış kabul görülmemesi bu tasarımın genel doğruluğunu ispatlamaz.

Sekiz karşılaştırma var: geçmişsiz temel arama; geçmişteki primitive sıklığına göre sıralama; öğrenilmiş parça kütüphanesi; aynı parçaları aynı sırada primitive açılımlarıyla öneren sabit yorumlayıcı; en fazla dört tam eski programı parça olarak kullanan bellek; bütün 80 ikili/üçlü parçayı baştan içeren sabit kütüphane; aynı sayı/uzunluklarda rastgele parçalar; eski ailenin iki motifini baştan bilen ayrıcalıklı referans.

Sonuncusu motif değişmiş görevlerin yeni motiflerini bilmez. Büyük sabit kütüphane belirli bir sıralamayla sınanır; bütün sabit sistemler için en iyi arama sayılmaz. Güçlü eşitlik kontrolü aynı öğrenilmiş öncelikleri primitive açılımlarıyla uygulayan yöntemdir.

## 4. Ana sonuç: olumlu aktarım ve olumsuz aktarım birlikte

Aşağıdaki sayılar **12 görevlik geçmişten sonra**, her sütunda 12 yaşam × 8 görev = 96 görev-koşuludur. Başarı, kabul edilen programın bütün 101 girdide doğru olmasıdır. [Ham sonuçlar](../../../research/living_learning_library_2026_09_21/results/library_learning.json), diğer geçmiş miktarları dahil bütün koşulları içerir.

| Yöntem | İlgili / 512 | Farklı motif / 512 | İlgili / 2.048 | Farklı motif / 2.048 |
|---|---:|---:|---:|---:|
| Geçmişsiz temel arama | 25/96 | 21/96 | 73/96 | 69/96 |
| Primitive sıklık önceliği | 34/96 | 20/96 | 73/96 | 68/96 |
| Öğrenilmiş parçalar | **86/96** | 10/96 | **94/96** | **14/96** |
| Aynı öncelikli primitive açılım | 86/96 | 10/96 | 94/96 | 14/96 |
| Tam program belleği | 80/96 | 7/96 | 95/96 | 15/96 |
| Bütün ikili/üçlü parçalar | 6/96 | 6/96 | 23/96 | 19/96 |
| Rastgele parçalar | 14/96 | 15/96 | 28/96 | 37/96 |
| Eski motifleri bilen referans | 96/96 | 10/96 | 96/96 | 26/96 |

İlgili görevlerde 2.048 bütçeyle öğrenilmiş kütüphane, temel aramanın çözemediği 23 görevi çözüyor; temel aramanın çözdüğü iki görevi kaçırıyor. Toplam fark 21. 512 bütçede fark 62 kazanım ve bir kayıp; 12 yaşamın her birinde toplam başarı artıyor. İlgili ailede bile görev görev mutlak üstünlük yok.

Farklı motiflerde 2.048 bütçede temel aramanın çözdüğü **55 görev kaybediliyor**, ek kazanım yok. 12 yaşamın tamamında toplam başarı daha düşük. Aynı dil içindeki bu aile değişimi bile geçmişten gelen önceliği zararlı kılabiliyor. Rastgele parçaların bu koşulda öğrenilmiş parçalardan daha çok görev çözmesi, doğru geçmişin her gelecek için iyi bir öncelik olmadığına ek uyarıdır; rastgeleliğin genel çözüm olduğunu göstermez.

Tam program belleği güçlü bir basit kontroldür: 2.048 bütçede ilgili görevlerde **95/96**, parça çıkarımı **94/96**. İlk iki görevin motifleri doğrudan içermesi bu belleği özellikle destekliyor. Dolayısıyla ortak alt parçaları çıkarmanın bu müfredatta zorunlu olduğu veya basit yeniden kullanımın ötesinde her zaman kazanç sağladığı savunulamaz.

Sabit büyük kütüphane zayıf kaldı; çok sayıda parça mevcut sıralamada aramayı genişletiyor. Bu, sabit mimarilerin yetersizliği değildir. Aynı parçaları aynı öncelikle açan sabit yorumlayıcı **1.152 eşleştirilmiş koşulun tamamında** program, kabul durumu ve bütün sayaçlarla tam eşleşti. Kütüphane bilgisi farklı bir kodlamayla arama politikasında taşınabiliyor.

## 5. Daha fazla doğru deneyim, her yönde daha çok edinim demek değil

Öğrenilmiş kütüphanenin geçmiş miktarına göre başarısı:

| Geçmiş görev | İlgili / 512 | Farklı motif / 512 | İlgili / 2.048 | Farklı motif / 2.048 |
|---|---:|---:|---:|---:|
| 0 | 25/96 | 21/96 | 73/96 | 69/96 |
| 6 | 84/96 | 9/96 | 90/96 | 24/96 |
| 12 | 86/96 | 10/96 | 94/96 | 14/96 |

Altıdan on iki eski göreve geçerken ilgili görevlerin toplamı artsa da tekil kayıplar var. 512 bütçede sekiz yeni başarıya karşı altı eski probe başarısı kayboluyor. 2.048 bütçede altı kazanıma karşı iki kayıp var. Farklı motiflerde 2.048 bütçede bir kazanım ve 11 kayıp oluşuyor. Ayrıntılı eşleştirmeler [hesaplanabilir özet kaydında](../../../research/living_learning_library_2026_09_21/results/paired_summary.json).

Somut örnek: yaşam 7, `related4`, bütçe 2.048. Altı eski görevden çıkarılan durum, yeni hedef için `[kare, kare, ekle, eksi, ekle, ekle, kare]` programını **478** adımda buluyor; bütün alanda doğru. On iki eski görevin ardından yeniden seçilmiş kütüphane aynı hedefte **2.048** adımı tüketip cevap veremiyor. Yeni yanlış kanıt yok; 12 eski program da doğru. Eski cevapların bozulması da yok. Değişen, gelecekteki adayların önceliği ve bu önceliğin sonlu bütçe içindeki etkisi.

Bu nedenle mevcut beceri kümesinin korunması, gelecekte edinilebilir becerilerin aynı bütçede tek yönde genişlemesini sağlamaz. Bu gözlem belirli bir seçici ve arayıcıya aittir; bütün öğrenme mekanizmaları için kaçınılmaz zarar teoremi değildir. Başka bir seçici, daha fazla bütçe veya ayrı arama yolları bu tanığı değiştirebilir.

## 6. Kalıcı durum ve nedensel müdahale

Durum; kabul edilmiş primitive programları, seçilmiş parça tanımlarını ve çıkarım sayaçlarını içeriyor. Ham girdi–çıktı satırları bu öğrenici durumunda tutulmuyor; deney sürücüsündeki dünya kaydı yeniden üretim içindir ve `token_policy` girdisi değildir. Yeni problemin etiketleri elbette yeni aramaya giriyor.

Kütüphane silme müdahalesinde eski program bankası aynen bırakıldı; yalnız etkin parça listesi boşaltıldı. Eski **144 programın** bütün-alan cevapları değişmedi. Gelecekteki **192** arama koşulu, aynı bütçeli primitive-only sonuçları ve sayaçlarıyla eşleşti. Bu, seçilmiş parçaların sonraki arama üzerindeki nedensel rolünü gösterir. Program bankası durduğu için parçalar yeniden çıkarılabilir; burada yeniden çıkarım kapalı bir müdahale yapılmıştır.

Eski programlar açılmış ve birbirinden bağımsız saklandığı için eski davranışı korumak bu deneyde yapısal olarak kolaydır. Ortak bir sinirsel temsilde unutmayı çözmüş sayılmaz. Parçaların tanımı da dünya hakkında değişen bir inanç değil, verilen işlemlerin değişmez bileşimidir; görevi doğru çözdüğü iddiası ayrı kalır.

Sekiz yöntemin her biri iki gelecek görev ailesinde ayrı süreçte kayıtlı durumdan çalıştırıldı: **16 süreç kontrolü**, tam JSON eşitliği. Bu mekanik kalıcılığı destekler; uzun yaşamda gürültüyle, sınırlı bellekle veya yeni dillerde güvenilirlik garantisi vermez.

## 7. Sonuçlardan sonra eklenen bütçe paylaşımı kontrolü

Olumsuz aktarım görüldükten sonra [ayrı bir planla](POSTHOC_PORTFOLIO.md) 384 ek koşul eklendi. Aynı toplam bütçenin yarısı geçmişsiz, yarısı öğrenilmiş kütüphaneli aramaya ayrıldı. İki dal da çalıştırıldı. Geçmişsiz dal kabul edilmiş program verirse onun sonucu, aksi halde öğrenilmiş dalın sonucu seçildi. Boş kalan bütçe diğer dala aktarılmadı.

| Toplam bütçe | Görev ailesi | Temel yarım bütçe | Öğrenilmiş yarım bütçe | İki dalın seçimi | Temel tam bütçe | Öğrenilmiş tam bütçe |
|---|---|---:|---:|---:|---:|---:|
| 512 | İlgili | 10/96 | 69/96 | 71/96 | 25/96 | 86/96 |
| 512 | Farklı motif | 12/96 | 4/96 | 12/96 | 21/96 | 10/96 |
| 2.048 | İlgili | 46/96 | 93/96 | **96/96** | 73/96 | 94/96 |
| 2.048 | Farklı motif | 37/96 | 13/96 | **40/96** | 69/96 | 14/96 |

Bu paylaşım 2.048 bütçede farklı motiflerdeki 14 başarıyı 40'a çıkarıyor; temel tam bütçenin 69 başarısına ulaşmıyor. İlgili görevlerde iki dal birbirini tamamlayıp 96/96 oluyor. Daha küçük bütçede ilgili edinim 86'dan 71'e düşüyor. Bir önceliğin olumsuz etkisini sınırlamanın hesap bedeli görünür; sonuçlara bakıp genel bir en iyi karışım seçilmedi.

Matematiksel güvence yalnız şu kadar: primitive dalın **yarım bütçede doğru kabul ettiği** her görevi seçim kuralı aynı programla korur. Tam bütçedeki temel başarıları koruduğu söylenemez. Sonlu kabul kümesinde yanlış kabul varsa, yanlış primitive program doğru öğrenilmiş programın önüne geçebilir. Kayıtlı koşullarda yanlış kabul görülmedi. [Ek kontrolün ham sonuçları](../../../research/living_learning_library_2026_09_21/results/portfolio_control.json) ana deneyden ayrı tutuldu.

## 8. Hesap, bellek ve ölçüm sınırları

12 ilk görevin edinimi yaşam başına ortalama **1.374,08** primitive-vektör adımı; aralık `[1.018, 2.154]`. 12 görevden kütüphane çıkarımında ortalama **49,83** alt dizi penceresi taranıyor. Bu tarama, sıralama ve Python nesne giderleri primitive yürütme bütçesine katılmış gibi gösterilmez; ayrı maliyetlerdir. İlk edinim bütün ana yöntemlere ortak aynı yoldan sağlanır. Dolayısıyla deney yalnız yeni görevin arama bütçesini eşler, bütün yaşam hesabının her yöntemde eşit olduğunu iddia etmez.

2.048 bütçe ve ilgili görevlerde öğrenilmiş arama ortalama **231,78**, temel arama **1.152,20** adım harcıyor. Bu ortalamalar başarısızlıkta bütçeyi tüketen görevleri de içerir; çözüm başına hızlanma diye bölünemez. İkisinin de çözdüğü 71 görevde ortalamalar sırasıyla **156,56** ve **876,28**. Bu daha dar karşılaştırma da yalnız ortak çözülebilen görevler içindir. Farklı motiflerde ikisinin de çözdüğü 14 görevde öğrenilmiş arama **489,21**, temel arama **441,93** adım kullanıyor.

12 görev sonundaki kompakt durum JSON'u ortalama **441,25 bayt**, aralık `[431, 450]`; programlar, parçalar ve çıkarım tanıları dahil. Bu Python süreç RAM'i veya en iyi mümkün temsil için alt sınır değildir. Ana koşullarda en büyük kuyruk 969 adaydır. Her adayın vektör/program nesneleri ve ziyaret edilmiş imza kümesi ayrıca bellek tüketir; bunların bayt bütçesi eşlenmedi. İki dallı kontrolde iki arama maliyeti toplanır; durum saklama ve çalıştırma düzeninin bellek bedeli de vardır.

## 9. Matematik ve literatür bakımından ne yeni değil?

[Matematik ve literatür notu](MATHEMATICS_AND_LITERATURE.md), sonlu döngüsüz bir parçanın başlangıç diline açılabildiğinde iki dilin mutlak işlev kümesinin aynı kaldığını ispatlıyor. Fark, sınırlı kaynakla hangi programa ulaşılabildiğinde olabilir. Aynı kütüphane ve arama önceliği verilmiş bir primitive arayıcının açılımları aynı sırada denemesi eşdeğerliği açıklıyor; deney bunu sayaç düzeyinde doğruluyor.

[DreamCoder](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf), program kütüphanesi ve arama politikasını birlikte öğrenerek bu tür erişimi değiştirir. [OOPS](https://sferics.idsia.ch/pub/juergen/oopsmlj.pdf), önceki çözümlerden yararlanan aramayla yeni başlangıçlara ayrılmış zamanı birlikte ele alır. [Stitch](https://arxiv.org/abs/2211.16605), program korpusundan ortak soyutlamalar çıkarmayı inceler. Buradaki küçük sezgisel seçici bu yöntemlerin uygulaması, onların sonuçlarının yeniden üretimi veya yeni bir temel algoritma değildir.

Bu turun katkısı araştırma kaydımıza ait bir ayrıştırmadır: doğru edinilmiş hesap, korunan eski davranış ve gelecekteki edinim birbirine eşit ölçüler değildir. Önceki turda izinli güncelleme geometrisinin oynadığı rolü, burada aday üretim önceliği oynuyor. İki mekanizma farklıdır; tek bir evrensel yasaya indirgenmiş sayılmaz.

## 10. Denetim ve açık problem

[Doğrulama kaydı](../../../research/living_learning_library_2026_09_21/results/verification.json):

- Ayrı skaler yorumlayıcıyla, uzunluğu en fazla üç olan 85 program bütün alanda denetlendi; imza elemesi kullanmayan kaba enumerasyonla seçilen programlar eşleşti. 40 makro bileşimi ayrıca sınandı.
- 48 koşulda gerçek primitive çağrıları sayılarak bütçe sayaçları doğrulandı. Aynı koşullarda öğretmen programı, görev adı ve aile bilgisi değiştirildiğinde öğrenicinin sonucu değişmedi; öğrenici durumu mutasyona uğramadı.
- Kabul edilmiş ana sonuçların bütün-alan doğruluğu ayrı skaler yorumlayıcıyla yeniden hesaplandı. Bütün ana ve ek bütçeler korundu; kayıtlı ana/ek koşullarda yanlış kabul yok.
- İki deney bütünüyle yeniden çalıştırıldı. Bu aynı kodla tekrar üretimdir; yeni bağımsız görev ailesinde dış doğrulama değildir.
- Başlangıçtaki **886 dosya** byte düzeyinde aynı; HEAD değişmedi. Kitap ve önceki araştırma sonuçları korunuyor.

Eksik kalanlar belirleyici: dili ve görev sınırlarını keşfetmek, algıdan yeni ilkel ayrımlar üretmek, çevreye eylemle deneyim oluşturmak, doğru kütüphane seçicisini deneyimden değiştirmek, yanlış eski kanıtı düzeltmek, gecikmiş sonuçların hesabını yapmak ve bellek dolduğunda gelecekteki edinimi güvenilir korumak bu turda çözülmedi. Eski program bankası görev sayısıyla büyüyor; sınırsız yaşam için sabit bellek çözümü yok. Görev dağılımı da bilinçli olarak yeniden kullanılabilir motiflere göre kurulmuş.

Genel probleme taşınabilecek çalışma ilkesi, henüz tamamlanmış çözüm değil: **bir öğrenici durumu yalnız bugün verebildiği cevaplarla değil, aynı yeni kanıt ve aynı kaynak altında yarın hangi değişimleri edinebildiğiyle de değerlendirilmelidir.** Bu bir değerlendirme gereğidir; kendiliğinden iyi öğrenen sistemi kuran yeterli bir algoritma değildir.

Sonraki ayrıştırıcı yön, bu küçük dilden daha çok motif üretmekten önce, deneyimin **aday üreticisini veya güncelleme kuralını hangi kanıtla değiştirmeye hak verdiği** ve bu değişimin yeni bir ailede edinimi destekleyip desteklemediğidir. Diğer açık yön, başlangıç dilinde bulunmayan gözlemsel ayrımların edinimidir: bir programı daha erken bulmak, gerekli bilgiyi hiç temsil etmeyen bir sistemin yeni bilgiyi elde etmesiyle karıştırılmamalıdır. Önceki temsil kaybı ve durum yeterliliği karşı örnekleri burada geçerli kalır.

Bu tur yalnız yerelde tutuldu; commit, push ve yayın yapılmadı. [Kod ve yeniden üretim yönergeleri](../../../research/living_learning_library_2026_09_21/README.md).
