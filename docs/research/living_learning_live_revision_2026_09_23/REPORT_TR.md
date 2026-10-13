# Tur 20 — Öğrenilen yeni yapının çalışan geçmişe bağlanması

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Yerel araştırma raporu

**Yeni modeli öğrenmek, onun mevcut geçmişteki doğru çalışma durumunu kendiliğinden vermez.** Tur 19'da gerçekten edinilen ardışık modelleri sıfırlanmayan ana akışa bağladık. Modelin doğru olduğu evrelerde başlangıç durumunu varsaymak 713 yanlış cevap üretti. Olası durumları koruyan yöntemler bu taşıma hatasını önledi, fakat bazı soruları cevapsız bıraktı. Ayrıca eski öğrenilmiş modeli ve kısa geçmişi birlikte kullanan yöntem, yalnız kısa geçmiş + verilmiş parity kontrolüyle aynı sonuçları daha fazla hesapla üretti. Bu beklenen ek yararı desteklemeyen bir sonuçtur.

**Genel yaşayan öğrenme problemi çözülmedi; yeni temel ilke bulunmadı.** Bu tur, edinilmiş yeni program ile mevcut çalışma durumunun bağlanmasını bir arada sınar. Önceki öğrenme kalibrasyonu hâlâ sıfırlanabilir ayrı bir kanaldan gelir.

## 1. Önceki sonuçlardan kurulan deney

[Tur 19'un](../living_learning_distinctions_2026_09_23/REPORT_TR.md) 72 dünyasının tamamı kullanıldı; yanlış öğrenilen son modeller de korundu. Doğru durum sayısı ve ayırt edici devamlar o turda deneyimden kurulmuştu. Bu tur aynı gerçek başlangıç modeli ve 16 öğrenme-bloku sonrası modellerini kullanır. Kaynak sonuç ve kod hash'leri sabitlendi. Gizli doğruluk bilgisi taşıma yöntemine verilmedi.

[Tur 16](../living_learning_migration_2026_09_22/REPORT_TR.md) hazır aday dilinden edinilmiş yeni bir kuralın tek seferlik katılımını incelemişti. Burada yeni modelin yapısı önceki turun gözlem tablosundan gelir ve aynı yaşamda birden fazla kez değişebilir. Bu, iki araştırma parçasının sınırlı birleşimidir; tamamen yeni bir problem veya literatürden bağımsız yöntem olarak sunulmaz.

24 tohum × üç dünya × iki güncelleme aralığı × altı yöntem = **864 yol**. Model güncellemeleri arasında 32 veya 256 gerçek bit akar; dünya sıfırlanmaz. Her yolda 16 sınırda ve sınır sonrasındaki 16 bitte toplam 272 sorgu: **235.008 tahmin**. Aynı kaynak dünyadaki yöntem ve hızlar eşleştirilmiştir; bağımsızlık birimi 24 tohumdur. Bütün yöntemler ek olarak baştan verilmiş doğru parity yetisini korur; bu yetinin öğrenildiği iddia edilmez.

Gerçekte 72 kaynak dünyanın 14'ünde tablo hiç değişmedi; 48'inde bir, sekizinde iki, ikisinde üç kez değişti. Toplam 70 kaynak revizyonu, iki hızla yöntem başına **140 gerçek taşıma anı** verdi. Bunların 27 kaynak revizyonunda eski model birden fazla durum içeriyordu. Ana kayıt çok sayıda sorgu içerse de model gelişim öyküsü sınırlıdır; uzun ve açık uçlu bir yaşam başarısı sayılmaz.

[Protokol](PROTOCOL.md) ana performans öncesinde yazıldı. Ana model güncellemeleri mantıksal zamanlarda uygulanır; hesapların gerçek zaman son tarihine yetiştiği veya arka planda paralel çalışmanın uygulandığı ölçülmedi.

## 2. Hangi bilgi taşındı?

Altı yöntem: yeni modeli başlangıçtan başlatma; bütün erişilebilir yeni durumları olası tutma; yalnız parity + son sekiz bit; eski/yeni modelin aynı geçmişteki durum ilişkisi + parity; bu ilişki + son sekiz bit; bütün gerçek geçmişi saklayıp yeni modelle yeniden yürütme.

İlişki yöntemleri eski ve yeni programın aynı bit geçmişinde ulaşabileceği durumları hesaplar. Eski mevcut durum kümesi, parity ve varsa son sekiz bit ile uyumlu yeni durumlar kalır. Sonuç birden fazla durum olabilir. Ortak tek çıktı varsa cevap verilir, farklı çıktılar varsa **kararsız** kalınır. Kararsızlık yanlış cevapla aynı ölçüye gömülmez. Ana akış etiketleri bu kümeleri daraltmak veya programı öğrenmek için kullanılmaz.

Değerlendirici ayrıca yeni modelin bütün yaşanmış geçmişi baştan okuduğunda hangi duruma ulaşacağını hesaplar. Bu durumun kapsanması **taşımanın doğruluğu**dur. O modelin gerçek dünyaya uygun cevap vermesi ayrı bir **model doğruluğu** sorusudur. Hedef ve öğrenilen modelin sayısal durum adları karşılaştırılmaz.

## 3. Model doğru olsa bile yanlış başlatma hata üretti

Her yöntemde 39.168 değerlendirme var. Bunların 28.458'i öğrenilen modelin bütün sonlu dizilerde doğru olduğu evrelerdedir:

| Yöntem | Doğru kesin cevap | Yanlış kesin cevap | Kararsız |
|---|---:|---:|---:|
| Başlangıç durumunu varsay | 27.745 | **713** | 0 |
| Bütün erişilebilir durumlar | 26.877 | 0 | 1.581 |
| Parity + son sekiz bit | 27.270 | 0 | 1.188 |
| Eski/yeni model ilişkisi + parity | 26.886 | 0 | 1.572 |
| İlişki + parity + son sekiz bit | 27.270 | 0 | 1.188 |
| Bütün geçmişi yeniden yürüt | **28.458** | **0** | **0** |

Sıfır yanlış cevap veren küme yöntemleri bazı cevapları vermekten kaçındı. Bu yüzden tablodan arşive eşdeğer hizmet veya genel üstünlük sonucu çıkmaz. Tam arşiv bu deneyde doğru modele ait mevcut durumu eksiksiz yeniden kurdu.

Gerçek taşımanın yapıldığı **140 sınır anında**, başlangıç varsayımı 61 yanlış, 79 doğru cevap verdi. Tam arşivde yedi yanlış, 133 doğru vardı; bu yedi hata modelin kendisindendi. Kısa iz kullanan iki yöntem aynı 81 doğru, üç yanlış, 56 kararsız sonucu verdi. Bütün durumlar ve izsiz ilişki yöntemleri bu anların tamamında kararsızdı.

Parity yetisi 235.008 ölçümün tamamında korundu. Bu, paylaşılan ve değiştirilen bir sinirsel temsilde eski becerilerin korunduğu anlamına gelmez: parity ayrı ve değiştirilmeyen bir sayaçtır.

## 4. Eski öğrenilmiş yapı kısa izin üstüne ek yarar sağlamadı

`model_relation_suffix8` ile `parity_suffix8`, **144 eşleştirilmiş yaşamın bütün 39.168 noktalarında aynı olası durum kümelerini ve aynı cevapları** üretti. Teorik olarak eski model ek koşullama sağlayabilirdi; bu kayıtta sağlayamadı. Buna rağmen ilişkili yöntem, aralık 32'de yaşam başına ortalama 721,58, aralık 256'da 653,19 ek sanal iş kullandı. Hiçbir yaşamda daha ucuz değildi; revizyonsuz yaşamlarda eşitti.

İzsiz ilişki yöntemi bütün-durumlar kontrolüne göre toplam yalnız 11 ek kesin cevap verdi; bunlar kısa aralıkta iki yaşamda görüldü. Bu karşılaştırmada ilişki yöntemi parity'yi de koşullarken bütün-durumlar yöntemi koşullamaz. Bu nedenle 11 cevabın kazancını yalnız eski **öğrenilmiş** duruma bağlayamayız. Kısa-izli güçlü kontrol, ilgili iddiayı zaten desteklemedi.

Döngü+sıfırlama ve gecikme ailelerinde kısa iz, değişim anındaki durumu belirledi. Rastgele geçişli ailede iki hızın her birindeki 36 değişimin 28'inde kararsızlık kaldı. “Sekiz bit her yeni yapıyı doğru başlatır” sonucu yoktur. Dünya aileleri ve kısa iz uzunluğu bu deneyin önceden verilmiş kapsamıdır.

## 5. Doğru taşıma yanlış modeli düzeltmez

Doğru ve yanlış model evreleri birlikte, yöntem başına 39.168 noktadaki sonuçlar:

| Yöntem | Doğru | Yanlış | Kararsız | Arşiv durumunu dışarıda bırakan |
|---|---:|---:|---:|---:|
| Başlangıç varsayımı | 38.010 | 1.158 | 0 | 1.471 |
| Bütün durumlar | 36.945 | 305 | 1.918 | 0 |
| Parity + sekiz bit | 37.403 | 324 | 1.441 | 0 |
| İlişki + parity | 36.956 | 305 | 1.907 | 0 |
| İlişki + parity + sekiz bit | 37.403 | 324 | 1.441 | 0 |
| Tam arşiv | 38.750 | **418** | 0 | 0 |

Beş kapsama sağlayan yöntemde, verilen kesin çıktı hiçbir zaman aynı modelin arşiv hesabıyla çelişmedi. Buna rağmen yanlış model nedeniyle kesin yanlış cevaplar vardı. Küme yöntemlerinin daha az yanlış sayısı, kısmen modelin yanlış cevap vereceği yerde kararsız kalmalarından gelir; modeli daha doğru öğrendikleri anlamına gelmez. Bütün taşıma yöntemleri aynı öğrenilmiş programları aldı.

Başlangıç varsayımı, arşiv doğruyken 782 cevabı bozdu. Arşivin model hatası yaptığı 42 yerde ise yanlış çalışma durumu tesadüfen doğru çıktı verdi. Muhasebe:

**1.158 = 418 + 782 − 42.**

Bu nedenle toplam doğru cevap, taşımanın doğru olduğunun yeterli göstergesi değildir. Durum kapsaması, model doğruluğu ve çıktı doğruluğu birlikte incelenmelidir. [Matematik notu](MATHEMATICS_AND_PRIOR_ART_TR.md) bu ayrımı ve sonlu taşıma koşulunu türetir.

## 6. Hesap ve bellek bedeli

Aşağıdaki ortalamalar, aynı model ediniminin üstüne eklenen ana akış/taşıma işidir; öğrenmenin kendisi dışarı atılmamıştır, ortak kaynak maliyeti ayrıca gösterilir.

| Yöntem | 32 bit aralığında ek iş | 256 bit aralığında ek iş | Son ana geçmiş izi |
|---|---:|---:|---|
| Başlangıç varsayımı | 1.328,97 | 8.496,97 | 0 bit |
| Bütün durumlar | 1.512,10 | 8.911,35 | 0 bit |
| Parity + sekiz bit | 2.227,86 | 13.205,11 | 8 bit |
| İlişki + parity | 1.871,24 | 9.270,93 | 0 bit |
| İlişki + parity + sekiz bit | 2.949,44 | 13.858,31 | 8 bit |
| Tam arşiv | 1.922,75 | 13.135,19 | 528 / 4.112 bit |

Ortak kalibrasyonun son iç öğrenme bedeli yaşam başına ortalama **39.979,58** işti. Önceki sorgu/çevre bedeli ve kalibrasyon JSON'u da sonuçlara taşındı. Ana akış durumunun JSON boyutu ayrı; kalibrasyon arşivi bütün yöntemlerde ayrıca bulunur. “Yalnız sekiz bit bellekle bütün sistem öğreniyor” iddiası kurulamaz. Sıfır ana geçmiş izi de sıfır bellek değildir: program, parity ve olası durumlar saklanır.

İlişki kurma 356 üçlüye kadar geçici alan kullandı; kısa çift kontrolü en fazla 64 çift kullandı. İlişkiler her değişimde yeniden kuruldu, kalıcı önbelleğe alınmadı. Tam arşiv bu sonlu öykülerde ortalamada ilişkili kısa izden daha az iş istedi ve daha çok cevap verdi; buna karşılık daha çok ham geçmiş sakladı. Dolayısıyla arşivsiz yöntemin genel hesap üstünlüğü gösterilmedi.

Sanal iş, belirtilen geçiş/filtre/çıktı/bit-yazma sayımıdır. Python nesne, hash, birleştirme, sıralama ve program kopyalama giderlerinin bütünü veya gerçek CPU/RAM değildir. Bu tur bütçe kesilmesi veya toplam bellek sınırında rekabet deneyi değildir. Çok uzun yaşam için asimptotik üstünlük yalnız bu kısa ölçümlerden çıkarılamaz.

## 7. Matematik, literatür ve denetim

Eski küme eski programın geçmiş-yürütme durumunu kapsıyorsa, ortak ulaşılabilirlik ilişkisinin görüntüsü yeni programın geçmiş-yürütme durumunu kapsar. Bu koşullu sonuç yanlış modeller için de geçerlidir. Bir ara temsile geçişin daha sonra gerekli ayrımı kaybettirebildiği ayrı modulo-3/parity tanığı da kuruldu. Bu son tanık ana performans verisi değildir.

Dinamik yazılım güncelleme ve durum gözlemcisi literatürüyle eşleşmeler [kaynak notunda](MATHEMATICS_AND_PRIOR_ART_TR.md) açıkça verildi. Yeni programın yanında mevcut durumun dönüştürülmesi veya olası durum kümesi tutulması yeni bir genel fikir sayılmıyor. Kalibrasyon reset'i kaldırılmadığı için sıfırlamasız tek-kanallı öğrenme başarısı da iddia edilmiyor.

Doğrulama tamamlandı:

- 864 yol bütünüyle tekrar üretildi; ham sonuç içerikleri birebir eşleşti.
- 235.008 noktada gerçek dünya, modelin bağımsız geçmiş yürütmesi, parity ve verilen cevap denetlendi. Beş sound yöntemde **195.840 kapsama kontrolü** geçti.
- Küçük makinelerde 258.048 kısa-geçmiş ilişki ve 258.048 son-ek ilişki tanığı geçti.
- 39.168 noktada eski-model+iz kümesinin güçlü kısa-iz kontrolünün altkümesi olduğu doğrulandı; burada hepsi eşitti.
- Durum adları yeniden numaralandırılarak 18 yol sınandı; cevaplar, dönüştürülmüş kümeler ve sayılan iş değişmedi.
- 36 yol, yaşamın ortasındaki JSON kaydından yeni süreçte sürdürüldü; kalan sonuçlar ve iş toplamları aynı çıktı.
- Önceki **1.039 dosya** ve Git HEAD korundu. Kitap ve ana sistem değişmedi; commit, push veya yayın yapılmadı.

## 8. Ana problemde kalan

Bu tur, öğrenilmiş yapı ile çalışan geçmişin birleşmesinde üç şeyi ayırdı: programın doğru olması, mevcut durumunun doğru kapsanması ve bu belirsizlik altında hangi cevabın verilebileceği. Edinim kanalını değiştirmeden bu üçlü birlikte ölçüldü; güçlü kontrol ek yapının yararını desteklemedi.

Ana sorun hâlâ temsil/işlem dilinin edinimi, kavramların yeniden örgütlenmesi, amaç ve değer, eski becerilerle yeni öğrenmenin ortak yapıda sürmesi ve sınırlı kaynakla yaşam boyunca gelişmedir. Bu çalışmayı yalnız durum taşıma problemine indirgemiyoruz. [Ana problem ve sonraki yön](WHOLE_PROBLEM_TR.md) bu sınırı korur.

Kayıtlar: [yeniden üretim dizini](../../../research/living_learning_live_revision_2026_09_23/README.md), [özet](../../../research/living_learning_live_revision_2026_09_23/results/summary.json), [doğrulama](../../../research/living_learning_live_revision_2026_09_23/results/verification.json), [hata ayrıştırma](../../../research/living_learning_live_revision_2026_09_23/results/diagnostics.json), [son denetim](../../../research/living_learning_live_revision_2026_09_23/results/final_audit.json).
