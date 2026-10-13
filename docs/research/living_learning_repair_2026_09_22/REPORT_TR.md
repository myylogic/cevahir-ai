# Tur 17 — Edinilmiş düzeltme yordamı yararlı; temel yenilik iddiası desteklenmedi

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel araştırma kaydı

## Ana sonuç

Sistem kendi aramasında bulduğu onarımlardan değişken bağlı düzeltme şemaları çıkardı. Bu şemaları yürütmek, aynı kaynak bilgisiyle başlayan fakat şema yürütmesi kapalı kopyaya göre uyumlu yeni görevlerde daha fazla doğru edinim sağladı: **192 hedefte 84 doğruya karşı 54 doğru**. Fark 24 yaşamın 19'unda olumlu, beşinde sıfırdı. Kaynak ve şema edinim maliyetleri iki kola da yazıldı.

Fakat güçlü karşılaştırma aynı 84 görevi çözdü. Genel toplu düzenleme işlemlerini geçmiş başarıyla sıralayan yöntem ile öğrenilmiş şemaları kullanan yöntem, üç rejim ve 24 yaşamın **72 eşleştirmesinin tamamında aynı kabul edilmiş program dizilerini** üretti. Bu deney, yeni bir temel öğrenme ilkesi bulunduğu veya anti-unification ile yeni şema çıkarmanın gerekli olduğu iddiasını desteklemiyor. Etki, bu alanda bilinen bağlamsal toplu düzenlemelerle açıklanabiliyor.

Görev ailesi değiştiğinde şemalar ek doğru görev getirmedi, hesap maliyetini artırdı. Kısmi düzeltme isteyen görevlerde de daha sade öğrenilmiş yerel sıralama aynı başarıya daha düşük ortalama maliyetle ulaştı. Olumlu aktarım, olumsuz maliyet ve güçlü kontrolün eşleşmesi birlikte korunuyor.

**Genel yaşayan öğrenme problemi hâlâ açık.** Bu tur deneyimin gelecekteki değişiklik önerilerini nasıl etkileyebildiğini sınırlı bir dilde sınadı; kendi temsil dilini, hedefini, geri bildirimini veya öğrenme yordamını açık uçlu biçimde kuran bir sistem göstermedi.

## Önceki çalışmadan devam

[Tur 16](../living_learning_migration_2026_09_22/REPORT_TR.md), öğrenilmiş kuralın çalışan akışa yerleştirilmesini ve eski becerinin korunmasını incelemişti. [Sonraki sınama taslağı](../living_learning_migration_2026_09_22/WHOLE_PROBLEM_AND_NEXT_TEST_TR.md), deneyimin düzeltme yöntemini de değiştirmesini öneriyordu. Bu tur o ayrımı denetledi; liste programları yerine iki değişkenli aritmetik ağaçları seçildi. Böylece anlam, bütün 289 alan girdisinde bağımsız denetlenebildi. Liste öğrenimi yapılmış sayılmadı.

Bu turun başlangıcındaki 994 dosya ve HEAD korundu. Kitap ve ana uygulama değişmedi. Önceki olumlu/olumsuz sonuçlar yeniden yazılmadı.

## Ne edinildi, ne hazır verildi?

İfade dili `x`, `y`, sabitler, toplama, çıkarma ve çarpmadır; bütün işlemler 17 elemanlı alanda yapılır. Öğreniciye hatalı aday ağaç ve 16 girdi–çıktı örneği verilir. Öğretmenin doğru ağacı ve görev rejimi verilmez. Öğrenici 16 örneğin hepsini geçen ilk adayı kabul eder; gizli doğrulama bütün `(x,y)` çiftlerinde yapılır ve öğreniciye geri dönmez.

Her yaşamda dört kaynak görevde `(u+v)*w` ifadesi, örneklerle uyumlu `u*w+v` hesabına onarılır. Çözüm önceden teslim edilmez: aynı genel yerel arama bunu bulur. 96 kaynak onarımının tamamı örnekleri geçti; bağımsız alan kontrolünde hepsi 289 girdide doğruydu.

Kaynak onarımlarının önce/sonra ağaçları ikişer genelleştirildi. Farklı altifadeler değişken deliklerine dönüştürüldü; önce ve sonra tarafındaki aynı altifade çifti aynı deliğe bağlandı. Bütün yaşamlarda üç şema çıktı: ikisi x/y'ye özel, biri genel:

\[
\operatorname{mul}(\operatorname{add}(A,B),C)
\ \longmapsto\
\operatorname{add}(\operatorname{mul}(A,C),B).
\]

Bu ifade **cebirsel özdeşlik değildir**. Belirli kaynak hatasını düzeltmek için edinilmiş bir öneridir. Her yeni görevde kendi kanıtlarıyla sınanması gerekir. Örneğin doğru hedef gerçekten `(A+B)*C` ise bu öneri doğru programı bozar.

Genelleştirme algoritması, değişken bağlama, ağaç gezinme, hazır 108 genel yerel yeniden kurma işlemi ve tek geçişte bütün eşleşmeleri düzenleme yorumlayıcısı tasarımcı tarafından verildi. Oluşan desen ağaçları deneyimden geldi; fakat bunları çıkaran öğrenme algoritması deneyimden icat edilmedi. Girdi ve etiket kanalı da hazırdır.

## Deney ve karşılaştırmalar

[Sonuç öncesi protokol](PROTOCOL.md): 24 yaşam × üç rejim × beş yöntem =360 yol. Her yolda sekiz yeni görev, toplam 2.880 hedef denemesi. Aynı yaşamın yöntemleri ve rejimleri eşleşir. Bağımsız birim 24 yaşamdır; 2.880 bağımsız dünya iddiası yoktur.

Hedefler bir, iki veya dört ekilmiş hata noktası ve kaynakta bulunmayan derin altifadeler içerir. Uyumlu rejimde kaynak düzeltmesi bütün hata noktalarına uygundur. Değişmiş rejimde gereken düzeltme farklıdır. Kısmi rejimde son nokta zaten doğrudur; topluca değiştirmek yanlış olur. Tek noktalı kısmi görevler baştan doğru kontrolüdür. Rastgele altifadelerde aynı işlem biçimi tesadüfen bulunabilir; toplu uygulama bu iç noktaları da yanlışlıkla değiştirebilir.

| Yöntem | Kullanılan geçmiş bilgi / öneri |
|---|---|
| Sabit yerel arama | Aynı eski çözümler, sabit yerel öneri sırası |
| Öğrenilmiş yerel sıra | Başarılı kaynak düzenlemesi ilgili işlem bağlamında öne alınır |
| Güçlü toplu kontrol | Aynı başarılı genel düzenleme bütün uygun işlem bağlamlarına uygulanır; sonra yerel arama |
| Öğrenilmiş şemalar | Edinilmiş desenlerle toplu/tekli öneriler; sonra aynı yerel arama |
| Şemalar kapalı, bedeli ödenmiş | Şemalı yöntemle aynı başlangıç belleği ve edinim maliyeti; yürütme kapalı |

Bütün yöntemler kabul edilmiş çözümleri kütüphanede tutar. Hedeflerde yeni şema çıkarılmaz; yalnız yeni kabul edilen çözüm kütüphaneye eklenir. Bu nedenle kaynak sonrası ablas­yon toplam etkiyi ölçer: ilerleyen hedeflerde farklı kabul edilen çözümler kütüphaneleri de farklılaştırabilir. Her sonraki hedefte aynı kütüphaneyle doğrudan etki ölçüldüğü ileri sürülmez.

## Doğru edinim sayıları

Her hücrenin paydası 24×8 =192 hedef denemesidir. Doğru, kabul edilen programın bütün 289 alan girdisinde hedefle aynı sonucu vermesidir.

| Yöntem | Uyumlu | Değişmiş | Kısmi |
|---|---:|---:|---:|
| Sabit yerel arama | 48 | 48 | 51 |
| Öğrenilmiş yerel sıra | 54 | 48 | 120 |
| Güçlü toplu kontrol | 84 | 48 | 120 |
| Öğrenilmiş şemalar | 84 | 48 | 120 |
| Şemalar kapalı, bedeli ödenmiş | 54 | 48 | 120 |

Toplam 1.095 kabul edilmiş hedef programının hepsi gizli tam alan kontrolünü geçti; bu deneyde yanlış kabul görülmedi. Ancak 16 örnek uyumunun genel garanti olmadığı matematiksel karşıörnekle ayrıca gösterildi. Başarılı kabul edilmiş programların bu örneklemde doğru çıkması o karşıörneği ortadan kaldırmaz.

Uyumlu rejimde şemalı yöntemin 30 ek edinimi, kapalı kolun bellek ve edinim bedeli aynı tutulduğu müdahalede ortaya çıktı. Fakat güçlü toplu kontrol de aynı 30 ek edinimi sağladı. Üç rejimin tamamında şemalı ve güçlü toplu yöntem yalnız sayıda değil, kabul edilen program kimliklerinde de eşleşti.

Derinlik aktarımı sınırsız çıkmadı. Uyumlu rejimde dört hata noktası ve derinlik üç içeren 72 görevden şemalı ve toplu yöntem yalnız **iki** tanesini çözdü. Yerel sıralama bu 72 görevde sıfır doğruya ulaştı. Yani iki örnekte daha derin aktarım var, fakat derin bileşim sorunu çözülmüş değil. Yeni üretilen iç bağlamlar ve hedef başına arama sınırı önemli kısıtlar olarak kaldı.

Kısmi rejimdeki 120 doğrunun 48'i baştan doğru kontrol, 72'si tek gerekli onarımı olan iki-noktalı görevlerdir. Bu sayı “120 yeni beceri sıfırdan edinildi” şeklinde yorumlanmamalıdır. Dört-noktalı kısmi görevler bütün yöntemlerde başarısız kaldı.

## Bütün yaşam maliyeti ve olumsuz aktarım

Kaynak arama/öncelik kurma her yaşamda 5.690 soyut birim; şema çıkarma ek 468 birim tuttu. Şemalı ve kapalı kollara toplam 6.158 yazıldı. Kalıcı kaynak bilgisi iki kolda aynıdır. Pratik karşılaştırmalar, kullanmadıkları şema çıkarımını ödemedi.

| Yöntem | Uyumlu ortalama toplam iş | Değişmiş | Kısmi |
|---|---:|---:|---:|
| Öğrenilmiş yerel sıra | 691.659,75 | 721.066,75 | 370.224,92 |
| Güçlü toplu kontrol | 548.624,79 | 721.758,00 | 373.897,83 |
| Öğrenilmiş şemalar | 548.820,58 | 725.126,46 | 373.331,04 |
| Şemalar kapalı, bedeli ödenmiş | 692.127,75 | 721.534,75 | 370.692,92 |

Uyumlu rejimde şemalı yöntem kapalı kola göre ortalama 143.307,17 daha az iş harcayıp daha çok doğruya ulaştı. Güçlü toplu kontrole göre aynı doğru sayısında ortalama 195,79 daha fazla iş harcadı. Bu küçük fark her yaşamda aynı yönde değildir.

Değişmiş rejimde şemalı yöntem kapalı kola göre bütün 24 yaşamda daha pahalıydı: ortalama 3.591,71 ek birim, sıfır ek doğru. Kısmi rejimde ortalama 2.638,13 ek birim, yine sıfır ek doğru vardı. Bu deneyde olumsuz aktarım esas olarak ek hesap maliyetidir; şemalar yanlış kabulü artırmış gibi sunulmaz.

Şemalı yöntemin kanıtları geçmeyen, değerlendirilmiş şema adayları uyumlu/değişmiş/kısmi rejimlerde sırasıyla 776/934/684 adetti. Bunlar doğruluk garantili dönüşümler değildir. Aynı adayı yeniden üreten öneriler arama kümesinde elenebilir; öneri sayısıyla değerlendirme sayısı aynı ölçü değildir.

**Bütçe yorumu:** Bütün yaşam üst sınırı 1.000.000, hedef başına ek sınır 120.000 idi. Hiçbir yol bütün yaşam tavanına ulaşmadı; en büyük toplam 726.737 oldu. Başarısızlıklar ağırlıkla hedef başına sınırda gerçekleşti. Sonuç, bu sabit bütçe paylaşımı altındadır; kullanılmayan bütçeyi görevler arasında başka biçimde dağıtmak sınanmadı. Eşit global tavanda her mümkün bütçe yöneticisine üstünlük iddiası yoktur.

İş birimi gerçek CPU zamanı değildir. İfade değerlendirmesi, ağaç dolaşması, desen eşleştirme, değişken kontrolleri, şema çıkarımı ve açıkça tanımlanmış sanal kopyalama maliyetidir. Python'daki paylaşılan alt ağaçlar, önbellek ve nesne ek yükü nedeniyle literal tahsis sayısı da değildir. Kalıcı bellek ayrıca JSON boyutuyla kaydedildi: uyumlu rejimde toplu kontrol ortalama 2.078, şemalı yöntem 2.447 bayt; tam çalışan süreç belleği bundan ibaret değildir.

## Güçlü kontrol neden aynı sonuca ulaşabildi?

Edinilmiş genel şemanın sol tarafındaki üç değişken birbirinden bağımsızdır. Dolayısıyla bu şemanın eşleşmesi, kökün `mul`, sol çocuğun `add` olmasına eşdeğerdir. Sağ taraf, aynı üç altifadeyi `A*C+B` düzeninde kurar. Genel düzenleme üreticisi bunu `L/add/mul/(0,2,1)` kaydıyla zaten ifade edebilir; başarılı kayıt kaynak aramada edinilir.

Bu iki işlemin aynı düğümde aynı çıktıyı verdiği doğrudan görülebilir. Aynı alttan üste tek-geçiş yorumlayıcısıyla, bütün ağaç üzerindeki toplu sonuç da yapısal tümevarımla aynıdır. Yeni oluşturulan düğümler aynı geçişte yeniden ziyaret edilmez. 40 bağımsız ağaç kontrolü bu eşdeğerliğin uygulamasını sınadı.

Bu, bütün öğrenilmiş düzeltme şemalarının her genel düzenleme diline indirgeneceği teoremi değildir. Yalnız bu deneyde edinilen şema için bilinen güçlü kontrolün aynı dönüşümü yapabildiğini açıklar. Kaynak deneyimin yararı gerçektir; yeni bir hesaplama ilkesinin gerekliliği sonucu çıkmaz.

## Matematik, literatür ve doğrulama

[Matematik ve literatür notu](MATHEMATICS_AND_LITERATURE_TR.md), 16 örnekle uyumlu farklı alan fonksiyonlarının varlığını, alan eşitliğinin tamsayı eşitliği olmadığını, ortak değişkenli genelleştirmenin sözdizimsel niteliğini ve nedensel karşılaştırmanın koşullarını açıklar. [Getafix](https://software-lab.org/publications/Getafix_arXiv_1902.06111.pdf) ve [REFAZER](https://cseweb.ucsd.edu/~ldantoni/papers/icse17.pdf) doğrudan yöntemsel öncüllerdir; [SOAR](https://proceedings.mlr.press/v267/pourcel25a.html) kendi arama girişimlerinden sonraki üretim/düzeltme yeteneğini geliştirme düşüncesinin de öncülü olduğunu gösterir. Bu eşleşmeler araştırma katkısını sınırlayan bilgiler olarak korundu.

Performans ölçülmeden önce bağımsız kod denetimi, ağaç yapımının bütçe kontrolünden önce yapılmasını ve şema doğrulama taramalarının sayılmamasını buldu. Bunlar ana çalıştırma öncesinde düzeltildi; sanal maliyet modeli protokolde açıklaştırıldı. Ana sonuç görüldükten sonra bütçe, görevler veya öğrenici değiştirilmedi.

[Doğrulama](../../../research/living_learning_repair_2026_09_22/results/verification.json): 360 yolun tam tekrarı; 13.872 bağımsız skaler alan çıktısı; 1.095 kabulün 289 girdide ayrı skaler denetimi; ortak/değişken tekrarları ve serbest sağ-değişken yasağı; sonlanma; bütçe muhasebesi; 15 ayrı süreçte dördüncü hedeften sonra aynı devam. [Son bütünlük denetimi](../../../research/living_learning_repair_2026_09_22/results/final_audit.json), kaynak sonuçlarını, güçlü kontrol eşleşmesini, rapor sayılarını ve eski 994 dosyayı ayrıca kontrol eder.

[Ham sonuç](../../../research/living_learning_repair_2026_09_22/results/repair_learning.json) ve [yeniden hesaplanan özet](../../../research/living_learning_repair_2026_09_22/results/summary.json) yereldir. Commit, push veya yayın yapılmadı.

## Ana araştırmaya kalan sonuç

Desteklenen dar sonuç: deneyimden kurulan bir değişiklik önericisi, aynı kaynak bilgisine sahip bir kopyanın daha sonra hangi hesapları verilen kaynak sınırında edinebileceğini değiştirebilir. Bu, yalnız bugünkü cevabı saklamak ile gelecekteki öğrenme tepkisini değiştirmek arasındaki ayrımı somutlaştırıyor.

Desteklenmeyen geniş sonuç: bu örnekte yeni temel öğrenme ilkesi gerektiği, genel düzeltme şemalarının güçlü bilinen yöntemi geçtiği veya açık uçlu kendini geliştirme sağlandığı. Daha derin görevlerin çoğu hâlâ başarısız; kaynak değiştiğinde ek iş boşa gidebiliyor; görev/etiket/dil/yorumlayıcı dışarıdan veriliyor.

Sonraki araştırmanın yükü yalnız daha fazla şema üretmek değildir. Sistemin başarısızlığın hangi iç hesap adımından kaynaklandığını deneyimle ayırması, gerekli yeni değişiklik biçimini oluşturması ve bunu başka görevlerde toplam maliyet ile eski becerileri koruma açısından sınaması gerekiyor. Bu da tek başına bütün yaşayan öğrenme problemi olmayacak. Önceki süreklilik, bilgi koruma, çapraz aktarım ve deneyim erişimi gereklilikleri açık araştırma haritasında kalıyor.
