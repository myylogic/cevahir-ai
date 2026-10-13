# Tur 22 — matematik, literatür ve açık kalan araştırma yükü

Muhammed Yasin Yılmaz · 23 Eylül 2026

## Aday listesinin kapalı biçimde hesabı

X ve Y, n öğeli kümeler; M, gözlenen k ayrı birebir eşleşme olsun. Başka yapısal kısıt kullanmayan modelde uygun bijeksiyon kümesi Ω(M)'nin büyüklüğü (n−k)!'dir. Gözlenmiş x'in görüntüsü kesindir. Gözlenmemiş x için kullanılmamış her y, eşit ağırlıklı adayların (n−k−1)! tanesinde görüntüdür. Dolayısıyla:

P(π(x)=y | M) = 1/(n−k), x ve y henüz eşlenmemişse.

Bu nedenle yalnız bir π veya π⁻¹ girdisine bağlı sorguların olasılıkları, bütün bijeksiyonları sıralamadan hesaplanabilir. Yerel f ve g gözlenmişse π(f(x)) veya π⁻¹(g(y)) de bu biçimdedir. Bir kategori sorusunda kalan y'lerin kategori sayıları toplanır; farklı kategori cevaplarına körlemesine eşit ağırlık verilmez.

Bu sonuç bir yeni öğrenme ilkesi değil, seçili aday sınıfının simetrisi ve sorgunun biçimidir. Eşleme kaydı, hazır bijeksiyon modeliyle birlikte bütün ortak aday kısıtlarını tekrar üretmeye yeter. “Yalnız marjinal olasılık vektörü de her şeye yeter” sonucu çıkmaz: iki eşlenmemiş x ve iki y kaldığında iki legal eşleme vardır. Ayrı marjinallerde her eşleşme 1/2 olsa da iki x'in aynı y'ye gitmesi imkânsızdır. Marjinalleri bağımsız çarpmak çakışmaya 1/2 verir. [Tam tanık](../../../research/living_learning_crossmap_2026_09_23/results/model_scope_audit.json) bu yanlışı denetler. Sıkıştırılmış eşleme kaydıyla daha karmaşık ortak sorgu yeniden çıkarılabilir; mevcut tek-girdi yordamı bu sorguyu otomatik çözmez.

## Bilgi yokluğu ancak açık varsayımla kanıtlanır

Yerel tablolar sabit tutulup alanlar arası bijeksiyon tamamen serbest bırakılan soyut ailede, hiç köprü yokken her π aynı yerel geçmişi üretir. Aynı geçmiş iki farklı hedef cevabıyla uyumluysa yalnız bu geçmişten daima doğru cevabı garanti eden yöntem yoktur. Eşit ağırlıklı modelde tek çapraz çıktı n olasılığa eşit dağılır; n=6 için 1/6.

Bu aile, ana deneyin bilinen üretim yasasıyla aynı değildir. Ana üretecin kategori/döngü kuralı ek bilgi taşır. [Kapsam denetimi](../../../research/living_learning_crossmap_2026_09_23/results/model_scope_audit.json) bu kuralla köprüsüz 6 aday kaldığını ayrıca gösterir. Sonuçlar şu üç şeyi ayırmalıdır: gerçekten erişilebilir kanıt; öğrenicinin kullandığı varsayımlar; değerlendiricinin bildiği gizli üretim yasası. Bunların karıştırılması hem sahte öğrenme başarısı hem sahte imkânsızlık sonucu üretebilir.

## Tek ağaç için doğru ve yanlış teorem

Bir köklü ağaçta iki düğümün altındaki yaprak kümeleri ya ayrıktır ya da biri diğerini kapsar. Dolayısıyla kısmen kesişen iki kategori aynı anda birer düğümle temsil edilemez. Buradaki X bölümünün her sınıfı Y bölümünün her sınıfıyla kısmen kesişiyor; tek düğüm olabilen en fazla iki sınıf vardır. Öteki ikisi en az ikişer düğüm birleşimi ister, dolayısıyla toplam referans alt sınırı 1+1+2+2=6'dır. Rapordaki ağaç bu alt sınıra ulaşır.

Ancak yalnız yaprakların birleşimine bile izin verilirse her sonlu alt küme temsil edilebilir. Bu yüzden “kesişen kavramlar ağacı imkânsız kılar” savı fazla güçlüdür. Temsilin düğümleri, izinli okuma işlemi ve maliyet birlikte değerlendirilmelidir. Bu matematiksel tanık kullanıcının düşüncesindeki olası örgütlenme biçimlerini daraltır; beyin anatomisini veya bilinci kanıtlamaz.

## Birincil literatürle ilişki

[Gentner'ın 1983 yapı eşleme makalesi](https://groups.psych.northwestern.edu/gentner/papers/Gentner83.2b.pdf), alanlar arasında nesnelerin yüzey özelliklerinden çok ilişkisel örgünün taşınmasını ele alır. Yazarın üniversite kopyasının metni bu turda incelendi. Bizim doğrudan kimlik çiftleri alan deneyi, bu kuramın uygulaması veya yeni bir analoji algoritması değildir; kaynak ve hedefte hangi bağıntının eşleneceğini kendisi bulmamaktadır.

[Whittington ve arkadaşlarının TEM çalışması](https://discovery.ucl.ac.uk/id/eprint/10115119/), yapısal bilginin duyusal içerikten ayrılması ve yeni bağlamlarda kullanılmasını araştırır. Bu turda üniversite/yazar kaydı ve özeti doğrulandı; PMC tam metin sayfası doğrudan açıldığında erişim kontrolüne takıldı. Bu kaynak kullanıcının gözlemine yakın bir araştırma hattı olduğunu gösterir; Cevahir deneyinin beyin mekanizmasını doğruladığı veya TEM'i yeniden ürettiği anlamına gelmez. Nörobiyolojik kesinlik iddiası yoktur.

Önceki [tur 12](../living_learning_relations_2026_09_22/REPORT_TR.md) kaynak ilişkilerini yeni hedefe taşımış, ortak arayüzü hazır vermişti. Tur 22 başlangıçtaki daha güçlü niyetine rağmen bu açığı kapatamadı: köprü semantiği yine hazırdır. [Tur 11](../living_learning_library_2026_09_21/REPORT_TR.md) öğrenilmiş program parçalarıyla arama önceliğini değiştirdi; güçlü kontrol aynı hesabı üretebildi. Burada da daha fazla öğrenilmiş mekanizma eklemek, iyi bir aynı-bilgi kontrolünün ötesinde kapasite getirmedi. Bu eşleşmeler korunmalı; yeni isim altında yenilik iddiasına dönüştürülmemeli.

## Yeni hipotezler: aynı alt problemi büyütmeden

**H22.6 — Edinilmiş bir güncelleme işlemi içerikten ayrılabilir.** Aynı eski cevapları ve olay kayıtlarını veren iki durumda, aynı yeni deneyim sonrası gelecekteki hesap kapasitesi farklılaşabilir. Önceki yeterlilik tanıkları bunu mümkün kıldı; açık yük, değişen güncelleme işleminin hangi deneyimden edinildiği ve hazır katalog önceliğine indirgenip indirgenmediğidir. Ayırıcı karşı olgu: içerik, kaynak kanıt ve temsil sabitken yalnız edinilmiş güncelleme işlemini değiştirmek; aynı geçmişi yeniden işleyen güçlü yöntemi karşılaştırmak. Henüz yeni deney yapılmadı.

**H22.7 — Kavramı yalnız saklanan küme değil, görevde kullanılabilen işlemler belirler.** Aynı ağaç farklı okuma işlemleriyle farklı görev kapasiteleri verir; tur 22 bunun sonlu tanığıdır. Açık hipotez, okuma işleminin farklı görev deneyimlerinden edinilebilmesidir. Hazır birleşim işlemini açmakla onu edinmeyi ayıran müdahale gerekir. Görev doğru kategoriyi zaten etiketiyle veriyorsa, kavramın keşfedildiği iddiası kurulamaz.

**H22.8 — Yeni bir ilişki türünü açıklama olarak üretmek ile mevcut dilde parametre bulmak ayrı tutulmalıdır.** Tamamen sınırsız hesap bakımından yorumlayıcının yazabildiği programlar baştan vardır; bu, operasyonel olarak yeni yordam edinilemeyeceği anlamına gelmez. İddia, yeni yordamın aynı hesap/bellek koşulunda neyi erişilebilir yaptığı ve ne kadar edinim bedeli ödediğiyle sınanmalıdır. Daha fazla sabit primitive vermek veya doğru programı üreticiden sızdırmak çözüm sayılmaz. Sonraki tasarım önce bu sızıntıyı ve sabit güçlü sentez kontrolünü belirtmelidir.

**H22.9 — Uygun hipotez diline ilişkin belirsizlik, dil içindeki belirsizlikten ayrıdır.** Bu deneyde 24 aday kalması gerçek üreteç altında zorunlu değildi. Kaynak deneyimlerinden aday dilinin de değişebilmesi hızlı öğrenme getirebilir; yanlış dilde erken kesinleşme ise zarar verebilir. Sınama, yanlış yapısal varsayım ve doğru geniş model kontrollerini beraber gerektirir. Bu henüz çıkarılmış yeni bir algoritma veya kanıtlanmış AGI yolu değildir.

Bu dört hipotez tek bir mimarinin zorunlu parçaları diye sunulmuyor. Ana soru, **çalışan sistemde deneyimin gelecekte hangi hesap ve güncellemelerin edinilebilir olduğunu nasıl kalıcı ve kontrollü değiştirdiği** olarak kalıyor. Bir sonraki turdan önce sorguyu, güçlü kontrolü ve edinilmiş/yazılı mekanizma sınırını ayırmak, küçük bir modeli büyütmekten daha belirleyici olacaktır.
