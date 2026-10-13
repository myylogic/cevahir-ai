# Tur 18 — Birlikte değişen hesaplar ve koşulsuz aktarımın bedeli

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel deney raporu

**Ana yaşayan öğrenme problemi çözülmedi.** Bu tur, önceki deneyimde edinilmiş bir onarımın yeni hesapta hangi yerlere birlikte uygulanacağını inceledi. Tekil etkilerin yetmediği durumlar gösterildi; hazır ikinci derece model kapsamı artırdı. Buna karşılık bu tercihi sonraki bütün görevlere kalıcı olarak taşımak, aynı doğrulukta toplam maliyeti artırdı. Yeni temel matematiksel ilke veya AGI sonucu yoktur.

## 1. Önceki araştırmadan devam

[Tur 17](../living_learning_repair_2026_09_22/REPORT_TR.md), kendi bulunan program düzeltmelerinden şema çıkarılmasını sınadı. Öğrenilen şemanın kazancını güçlü toplu düzenleme kontrolü de sağladı; bu negatif yenilik sonucu korunur. Tur 18, aynı genel şemayı bütün yöntemlere vererek onun **uygulama yerlerinin birlikte etkisini** ayırır. Kaynak şemanın edinim kazancı yeniden ölçülmüyor.

[Protokol](PROTOCOL.md) performans çalıştırmasından önce yazıldı; deney sırasında değişmedi. 24 yaşam, yedi eşleştirilmiş yöntem, yaşam başına 16 hedef: toplam 168 yol ve 2.688 hedef denemesi. Bağımsız dünya üretimi 24 yaşam düzeyindedir; yöntemler ve aynı yaşamdaki hedefler bağımsız tekrarlar gibi sunulmaz.

Her hedefte sekiz ayrık potansiyel düzenleme yeri bulunur; hedef bunların dördünü değiştirir. Öğrenici hedef programı veya doğru maskeyi bilmez. Kendi adayını, 16 girdi–çıktı örneğini ve geçmişten alınmış `(A+B)*C → A*C+B` onarımını bilir. Bu dönüşüm cebirsel eşitlik değildir; kanıta göre önerilen bir değişikliktir. Başlangıç AST'sinin ve işlemlerin anlamının görülebilir olması ayrıca bir kolaylıktır.

Yaşam sırası dört toplamsal, dört ikili çarpımlı, tekrar dört toplamsal, dört üçlü çarpımlı hedeftir. İkinci evrenin ilk hedefi özellikle tekil etkileri sıfırlayan önceden belirlenmiş tanıktır. Bu sıra ve tanık deneyci tarafından kurulmuştur; doğal yaşam dağılımı olarak yorumlanmaz. Diğer affine parçalar rastgele çekilir, fakat birbirlerinden mutlaka farklı olmaları dayatılmaz.

## 2. Tam alan doğruluğu

Her kabul edilen program, öğrenicinin görmediği değerlendirmede F17²'nin 289 girişinin tamamında sınandı. Aşağıdaki her hücre 96 hedef üzerindedir; toplam sütunu 384 hedef üzerindedir.

| Yöntem | İlk toplam | İkili | Tekrar toplam | Üçlü | Toplam doğru | Ortalama yaşam işi |
|---|---:|---:|---:|---:|---:|---:|
| Bütün yerleri birden değiştir | 0 | 0 | 0 | 0 | 0 | 88.186,00 |
| Tekli açgözlü düzeltme | 2 | 8 | 3 | 21 | 34 | 667.716,71 |
| Birinci derece etki modeli | 96 | 18 | 96 | 0 | 210 | 598.782,58 |
| İkinci derece etki modeli | 96 | 96 | 96 | 84 | 372 | 1.733.919,83 |
| Her görevde yeniden uyarlama | 96 | 96 | 96 | 84 | 372 | 1.217.804,83 |
| Kalıcı derece tercihi | 96 | 96 | 96 | 84 | 372 | 1.450.551,83 |
| Bütün altkümeleri gerçek yürütmeyle ara | 96 | 96 | 96 | 96 | 384 | 4.820.016,29 |

Sanal iş toplamları, bütün yöntemlere aynı 6.158 kaynak edinim bedelini de yazar. Ölçü gerçek CPU süresi, enerji veya RAM değildir. Düğüm değerlendirme, tam sanal ağaç yapımı, eşleşme ve model vektör hesabını içerir; Python nesne, önbellek, karşılaştırma ve listeleme giderlerinin tamamını kapsamaz. Programı büyük dünyalara ölçeklerken bu yaklaşık hesap modeli yeterli kabul edilemez.

Hedef başına bir milyon iş sınırı vardı. **Hiçbir yöntem bu sınıra çarpmadı**; en yüksek harcama 396.593 oldu. Dolayısıyla burada gerçekleşmiş bütçe kesilmesi nedeniyle doğruluk farkı gösterilmedi. Fark, modellerin öneri kapsamı ve bu kapsam için harcanan iştedir. Yaşam toplamına ayrıca bütçe koyulmadı.

Toplam 1.744 kabulün tamamı tam alanı geçti; yanlış kabul gözlenmedi. Bu, 16 örneğin her olası program ailesinde yeterli olduğu anlamına gelmez. Dört kabul kaydında bulunan maske ekilmiş maskeden farklıydı: hepsi aynı dünyanın dört yöntemle çözümüdür. Programlar 289 girişte eşdeğerdi. Doğru davranış ile hedefin iç yapısını birebir yeniden bulmak ayrıldı.

## 3. Etkileşim neyi açıkladı?

Özel ikili tanıkta başlangıç ve bütün tekil düzenlemeler sıfır çıktı üretir. Gerekli çiftler birlikte değişince sıfırdan farklı hedef ortaya çıkar. Birinci derece ve açgözlü yöntem bu 24 tanığın hiçbirini çözemedi; ikinci derece, iki uyarlamalı yöntem ve tam arama 24'ünü de çözdü. Bu, etkisiz görünen tekil değişikliklerin birlikte gereksiz sayılmasına somut karşı örnektir.

Birinci derece yöntem, diğer ikili hedefler dâhil toplam 18 ikili hedefi çözdü. Sonradan yapılan açıklayıcı denetim bunların tam olarak hiçbir çarpım çiftinin iki yerini birden değiştirmeyen maskeler olduğunu doğruladı. Etkileşimleri olan bir dünya içinde bile belirli bir düzeltmenin tekil etkileri yeterli olabilir.

İkinci derece yöntem üçlü evrede 84/96 doğru verdi. Başarısız 12 hedefin her birinde ekilmiş hedef maskesi bir üçlü grubun tamamını içeriyordu. Bu maskelerde ikinci derece açılımın 16 erişilebilir örnek üzerindeki kalanı sıfır değildi. Diğer 84 maskede tam bir üçlü seçilmemişti ve ikinci derece yeterliydi. Bütün altküme kontrolü 12 kaçırılan hedefi de çözdü. Böylece “ikinci derece her yerde yeterlidir” sonucu reddedilir; “üçlü bir yapı varsa bütün hedefler üçüncü derece gerektirir” sonucu da reddedilir.

Bu açıklama performans sonrası tanı denetimidir. Öğreniciye gizli maske verilmedi; program, sıra, bütçe ve yöntemler sonuca göre değiştirilmedi. [Matematik notu](MATHEMATICS_AND_PRIOR_ART_TR.md) açılımı ve karşı örnekleri verir.

## 4. Kalıcılık ne kazandırdı, ne kaybettirdi?

Kalıcı yöntem her yaşamda tanık görevde bir kez 1→2 geçişi yaptı: toplam 24 geçiş. Yeniden uyarlayan yöntem toplam 174 kez genişledi. Fakat daha az genişleme daha az toplam iş demek olmadı. Kalıcı yöntem 24 yaşamın **hepsinde** daha pahalıydı; ortalama fark 232.747 iş, yaklaşık %19,1 ek maliyetti. İkinci derece, yeniden uyarlama ve kalıcı yöntem her yaşamda aynı kabul maskesi dizisini üretti; doğrulukları da eşitti.

| Evre | Kalıcı eksi yeniden uyarlayan: yaşam başına ortalama iş | Kalıcının daha ucuz olduğu yaşam |
|---|---:|---:|
| İlk toplamsal görevler | 0 | 0/24; hepsi eşit |
| İkili görevler | +10.939 | 9/24 |
| Tekrar toplamsal görevler | +303.728 | 0/24 |
| Üçlü görevler | −81.920 | 24/24 |
| Bütün yaşam | +232.747 | 0/24 |

Üçlü evrede kalıcı tercih, sonunda işe yaramayan birinci derece aramasını atladı. Yeniden toplamsal görevlere dönüldüğünde ise gereksiz çift ölçümleri yaptı. İkili evrede bile 18 hedef için birinci derece yeterli olduğundan, kalıcı tercih evre toplamında daha pahalı kaldı. “Aynı tür görevlerde her zaman kazandırır” iddiası da desteklenmedi.

Kalıcı öğrenilen nesne burada **bir bitlik inceleme tercihi**dir; yeni etkileşim dili icat edilmez. Katsayılar görevden göreve aktarılmaz, yeniden ölçülür. Kabul edilen programlar ve örnekleri saklanır, fakat bu kütüphane sonraki çözüm aramasında kullanılmaz. Bu turun kalıcılık sonucu, geçmiş deneyimin sonraki hesap yolunu değiştirebildiğini ve bunun olumsuz maliyet aktarımı yapabildiğini gösterir.

## 5. Kontroller ve iddia sınırı

- Bütün yerleri değiştiren yöntem dört/sekiz hedeflere yapısal olarak uyumsuzdur; sıfır başarısı güçlü yenilik kanıtı değildir. Sonlu alan üzerinde tam çıktı eşitliğine dayalı açgözlü kayıp da zayıf bir sezgiseldir.
- Anlamlı karşılaştırma birinci/ikinci derece, yeniden uyarlama, kalıcılık ve temsil-uyumlu tam aramadır. Tam arama daha pahalıdır ama bütün hedefleri çözer. Görülebilir AST'den etkileşimleri doğrudan türeten sembolik analiz/faktörleme kontrolü bu turda yoktur; en güçlü bütün olası rakiplerin aşıldığı söylenemez.
- Kaydedilen yanlış model öngörüsü sayısı sıfırdır. Derece genişlemesi yanlış çıkan bir aday öngörüsünden değil, doğrulanmış aday bulunamamasından doğdu. “Kendi yanlış tahminini doğrudan çürüttü” anlatımı bu sonuç için kullanılmamalıdır.
- Düzenleme yerleri ayrık, sayısı sekiz, işlemler bilinen aritmetik, geri bildirim doğru ve göreve bağlıdır. Gürültü, gecikme, iç içe dönüşüm, bilinmeyen gözlem dili ve sınırsız program ailesi sınanmadı.
- Deney ardışık görevlerde çalışan ve durumunu taşıyan küçük bir öğrenicidir. Kesintisiz dış ortamla etkileşirken eşzamanlı güncelleme, yeni nöron, uyku, bilinç veya genel zekâ modeli değildir. Eski becerilerin yeni ortak bir temsilde korunması bu turda ölçülmedi.

Birlikte etki açılımı klasik Möbius matematiğiyle, programı değiştirerek sınama da mevcut hata ayıklama/onarım literatürüyle eşleşir. İncelenen birincil kaynaklar ve erişim sınırları [literatür notunda](MATHEMATICS_AND_PRIOR_ART_TR.md) verildi. Geniş bir “ilk kez” veya yeni temel computational ilke iddiası yoktur.

## 6. Doğrulama ve koruma

- Deneyin tamamı yeniden çalıştırıldı; kayıt içerikleri birebir aynı üretildi.
- 94.110 gerçek aday sorgusu bağımsız skaler hesaplamayla denetlendi; 1.744 kabul ayrıca bütün 289 girişte kontrol edildi.
- Üç temsilci dünyada 3.072 bağımsız müdahale çıktısı ve 768 tam Möbius yeniden kurması kontrol edildi. Tekil etkileri gizlenen ikili tanık, sıfır olmayan üçüncü derece ve bütçenin atomik kesilmesi doğrulandı.
- Açıklayıcı tanı denetiminde 16.896 bağımsız skaler katsayı çıktısı hesaplandı; 18 ikili başarı ve 12 üçlü başarısızlığın maske koşulları doğrulandı.
- Yedi yöntemin her biri için iki kontrol noktasından yeni süreçte devam edildi: 14 devamın sonraki adayları, maliyetleri ve son durumları kesintisiz çalışmayla aynıydı.
- Önceki 1.007 dosyanın içerik hash'leri ve Git HEAD değişmedi. Kitap, ana uygulama ve eski araştırma kayıtları korundu. Commit, push veya yayın yapılmadı.

Geliştirme kaydı: performans sonrası ek tanı betiğinin ilk çalıştırması JSON'daki listeleri AST tuple biçimine dönüştürmediği için kendi doğrulamasında durdu. Önceden var olan geri yükleme yordamını çağıracak şekilde düzeltildi ve yeniden geçti. Son denetimin ilk çağrısı da önceki turun aynı adlı analiz modülünü yüklediği için durdu; bu turun analiz dosyasını açık yoluyla yükleyerek düzeltildi. İki yardımcı hata da başarı kaydı üretmeden yakalandı. Ana deney kodu, protokol ve ham sonuç değiştirilmedi. Bağımsız okuma denetimi hedef sızıntısı veya sonuçları geçersiz kılan somut hata bulmadı; yukarıdaki yorum sınırlarını özellikle belirledi.

## 7. Ana problemdeki yeri

Bu turun genel araştırmaya katkısı, **öğrenilecek iç değişimin kapsamı** ile **o değişikliği geleceğe taşımanın kapsamını** ayırmasıdır. Bir hesapta birlikte gerekli olan düzeltmelerin bulunması, bu arama biçiminin bütün gelecekte uygun olacağını söylemez. Geçmişten kalıcı bir nesne taşımanın varlığı ile kalıcı olarak yararlı gelişme farklı ölçülmelidir.

Kullanıcının kavramların büyümesi, bölünmesi, çapraz bağlanması ve deneyimle yeniden örgütlenmesi penceresi bundan ibaret değildir. Temsil ve işlem dilinin edinimi, hedeflerin oluşumu, bellek seçimi, yeni becerilerin birleşmesi ve mevcut hesap sürerken güncelleme hâlâ açıktır. [Araştırma haritası ve sonraki ayırıcılar](WHOLE_PROBLEM_AND_NEXT_TEST_TR.md) bu genişliği korur.

Yeniden üretim dosyaları ve ham kayıtlar: [deney dizini](../../../research/living_learning_interactions_2026_09_22/README.md), [özet](../../../research/living_learning_interactions_2026_09_22/results/summary.json), [doğrulama](../../../research/living_learning_interactions_2026_09_22/results/verification.json), [açıklayıcı tanı](../../../research/living_learning_interactions_2026_09_22/results/coverage_diagnostics.json), [son denetim](../../../research/living_learning_interactions_2026_09_22/results/final_audit.json).
