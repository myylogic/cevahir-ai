# Tur 22 — iki bilgi alanının bağlanması: başarı, gereksiz mekanizma ve yanlış iddia sınırı

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Yalnız yerel araştırma

**Ana living-learning problemi çözülmedi.** Bu turun yararlı sonucu, yeni bir keşif iddiasından çok üç ayrımın doğrulanmasıdır: deneyimle verilen eşleşmeyi kullanmak ortak anlamı keşfetmek değildir; bütün adayları kalıcı saklamak aynı bilgiyi daha küçük durumda tutmaktan üstün olmayabilir; kesişen kavramlar tek bir ağacın her türlü kullanımını imkânsız kılmaz.

Sonuçlar düzeltme sonrası çalışmaya aittir. İlk uygulama ve ilk düzeltmenin analizinde maliyet/denetim hataları vardı. Bunlar saklandı, geçerli kanıt olarak kullanılmıyor. [Düzeltme protokolü](AMENDMENT.md), özgün protokolün hangi maddelerinin değiştiğini ve sonuçlar görüldükten sonra yapıldığını açıklar. Bu çalışma dış ön kayıt, dış hakem değerlendirmesi veya bağımsız bilimsel tekrar değildir.

## Kullanıcının gözleminden sınanabilir parçaya

Kullanıcının önerisi, ayrı deneyim kümelerinin bağlanıp yeni düşünceyi ve öğrenmeyi kolaylaştırabilmesiydi. Önceki tur 12'nin hazır ortak arayüz sınırını kaldırmayı amaçladık; bu amaç **tam karşılanmadı**. Bu deneyde köprü örneği açıkça “X alanındaki bu sembol, Y alanındaki şu sembolle aynı nesneyi gösterir” diyor. Öğrenici bütün eşlemeyi başlangıçta bilmiyor; fakat köprünün anlamı, birebir eşleme varsayımı ve kullanılacak bileşim işlemi araştırmacı tarafından veriliyor.

Her yaşamda altı nesne iki ayrı adlandırmayla sunuluyor. X'te altılı döngü, Y'de üç ikili döngü gözleniyor. Her alanda altı geçiş ve altı kategori kaydı var: toplam 24 kaynak kaydı. Ardından önce iki, sonra birer yeni köprü çifti geliyor; son evrede beş çift biliniyor. 24 yaşamın her birinde 24 sorgu, yedi yöntemle değerlendiriliyor. **168 yol ve 4.032 cevap noktası, 168 bağımsız dünya değildir.** Eşleştirilmiş temel birim 24 yaşamdır.

Sorgular X'te bir işlem yaptıktan sonra Y'deki adı; Y'de bir işlemden sonra X'teki adı; iki kategori etiketini birlikte soruyor. Son evre ilk altı soruyu tekrar ediyor. Dolayısıyla yöntem başına 576 sorgunun 432'si yeni soru, 144'ü tekrardır. Yeni bir soru, mutlaka yeni bir eşleşme gerektirmez: çoğunda bileşimin ihtiyaç duyduğu köprü zaten gösterilmiştir. Kaynak tabloları tam ve gürültüsüzdür; algıdan nesne/kavram çıkarımı araştırılmamıştır.

## Güçlü karşılaştırmanın sonucu

| Yöntem | Kesin doğru | Kararsız | Kesin yanlış | Ortalama toplam yaşam işi |
|---|---:|---:|---:|---:|
| Köprüyü kullanmama (`separate`) | 0 | 576 | 0 | 394 |
| Bütün uygun adayları kalıcı tutma (`persistent`) | 374 | 202 | 0 | 6.612 |
| Her soruda kaynak ve eşlemeleri yeniden kurma (`recompute`) | 374 | 202 | 0 | 130.170 |
| Bilinen çiftler ve kalan adlarla doğrudan tamamlama (`completion`) | 374 | 202 | 0 | **694** |
| İlk uygun adayı kesin sanma (`point`) | 450 | 0 | 126 | 6.009 |
| Kullanımı kapatıp son evrede yeniden açma (`disabled`) | 144 | 432 | 0 | 5.840 |
| Doğru eşlemenin baştan verilmesi (`oracle`) | 576 | 0 | 0 | 625 |

“İş”, açıklanan skaler işlem sayaçlarının toplamıdır; CPU süresi değildir. Edinim, kaynak okuma, kaynak tablolarını kurma, eşleme denemeleri ve malzeme yazımları dahildir. Geçici tabloların yazımları, kalıcı durum yazımlarından ayrıca ayrılır. Python nesne yönetimi, bazı başlatma/muhasebe işlemleri, sıralama ve tanı JSON'u maliyetin tamamına dahil değildir. Byte yazımı ayrıca raporlanır; sessizce işlem sayısıyla toplanmaz. [Özet veri](../../../research/living_learning_crossmap_2026_09_23/results/summary_corrected.json), ağırlık 0/1/4 için ayrı byte duyarlılığını ve bütün kategorileri içerir.

Kalıcı aday belleği, geçmişten yeniden kurmaya göre 24 yaşamın tamamında daha ucuzdu. Fakat doğrudan tamamlama da **aynı 576 olasılık dağılımını tam kesirlerle** verdi ve bütün yaşamların her birinde kalıcı aday yönteminden daha ucuzdu. Ortalama iş oranı 6.612/694 ≈ 9,53. Tamamlama, sayaçların hiçbir kategorisinde kalıcı aday yönteminden daha fazla iş yapmadı. Bu negatif sonuç, bu kategorilere farklı negatif olmayan ağırlıklar vermekle tersine dönmez; bütün gerçek donanımlar için hız teoremi değildir.

Tamamlama da deneyimle kalıcı durum değiştiriyor: gözlenen çiftleri ve yerel ilişkileri tutuyor. Dolayısıyla sonuç “öğrenmeye gerek yok” değildir. Sonuç, **bu sorgu ailesinde öğrenilmiş bütün olasılık listesini saklamanın gerekli olmadığıdır**. Yöntemin çıkarım kuralı hazır verildi; öğrenici bu kısa kuralı deneyimden icat etmedi.

## Ne genellendi, ne yalnız yeniden kullanıldı?

Üç belirsizlik taşıyan yöntem, 432 yeni sorunun 230'una kesin doğru cevap verdi; 202'sinde kararsız kaldı. İhtiyaç duyulan çiftin doğrudan gösterilmediği 240 sorgunun yalnız 38'i kesin cevaplanabildi: son evrede bijeksiyon varsayımıyla kalan son eşleşmenin çıkarılması ve bazı kategori sorularında kalan adayların aynı kategoriye düşmesi buna izin verdi. 240'ın diğer 202'si belirsiz kaldı. Bileşim sorusunun doğrudan etiketi öğreniciye hiç verilmedi; gereken yerel işlem ve köprü bilgisi verildi.

Belirsizlik taşıyan yöntemlerde Brier kaybı 0,237847; `point` yönteminde 0,4375. `point` daha fazla kesin doğru cevap da verdiği için sadece 374'e karşı 450 sayısını karşılaştırmak yanıltır: bunun karşılığı 126 kesin yanlıştır. Eşit maksimumlarda eşit seçim varsayımıyla belirsizlik yönteminin beklenen argmax doğruluğu %76,215; `point`'inki %78,125'tir. Bu dağılımda kalibrasyon/kesin yanlış riskinin korunması, en yüksek zorunlu seçim doğruluğuyla aynı sonuç değildir.

Kullanımı kapatılan yöntem kanıt ve adayları tutmaya devam etti. Son altı sorguda kullanım açılınca 144/144 doğru cevap geri geldi. Bu, kullanım yolunun nedensel etkisini gösterir; iç yapının yanlışlığını kendiliğinden düzeltme mekanizması değildir. `separate` zayıf bir “öğrenemeyen model” rakibi değil, bilerek kapatılan bağlantı kontrolüdür. `oracle` ise altı doğru eşlemeyi fazladan alan tavan kontrolüdür. Bilgi eşitliği sağlanmış rakip sayılamaz.

## Modelin belirsizliği ile dünyanın belirsizliği aynı değil

Ana öğrenici, yalnız köprülerle sınırlanan bütün bijeksiyonları eşit ağırlıklandırıyor. İki/üç/dört/beş çift sonrası aday sayıları 24/6/2/1. Bu, o varsayım sınıfında tam hesap; dünyanın üretim yasasını kullanan Bayes hesabı olduğu iddia edilmiyor.

Sonuç sonrası [üreteç kapsam denetimi](../../../research/living_learning_crossmap_2026_09_23/results/model_scope_audit.json), sabit kategori/döngü kurulumunu bilen bir çıkarıcının daha fazla bilgi kullanabileceğini doğruladı. X'in döngüsü ve ardışık 3/3 etiket yapısı konumları ayırıyor; Y'nin tek/çift eşleştirmesi üç çiftin yerini belirsiz bırakıyor. Bu ek üretim yasasıyla başlangıçta 6 aday var. İki köprü sonrasında 20 yaşamda tek, dört yaşamda iki aday kalıyor; üçüncü köprüyle bütün yaşamlar tek adaya iniyor. Bu denetim, ana yöntemlere verilmeyen bir üretim yasasını kullanır; yeni bir adil rakip veya yöntem başarısı değildir. Ana aday sayısını “hiçbir sistem daha az deneyimle öğrenemez” sınırı diye okumamızı engeller.

Ayrıca her evrede soru türü değişiyor. Evreler arasındaki hata düşüşü yalnız yeni köprü sayısının nedensel etkisi diye sunulamaz. Aynı evrede yöntemler arası eşleştirilmiş karşılaştırma geçerlidir.

## Kesişen kategoriler bir ağacı gerçekten dışlıyor mu?

Kategori kümeleri X için {0,1,2}/{3,4,5}; Y için {0,2,4}/{1,3,5}. Altı farklı etiketli yaprağı olan **945 köklü ikili ağacın tamamı** tarandı. Hiçbir ağaç bu dört sınıfın her birini tek bir düğümün altındaki yapraklar olarak gösteremiyor; en fazla ikisi tek düğüm olabiliyor.

Fakat “tek ağaç görevlerden birini mutlaka kaybeder” daha güçlü iddiası yanlıştır. Ağaç düğümlerinin birleşimini okumaya izin verilirse bütün sınıflar, nesne kopyalamadan gösterilebilir. En iyi iki ağaçta dört sınıf için toplam altı düğüm referansı yeterli:

- X sınıfları doğrudan iki ana alt ağaçtır.
- Y'nin ilk sınıfı {0,2} düğümü ile {4} yaprağının birleşimidir.
- Y'nin ikinci sınıfı {1} yaprağı ile {3,5} düğümünün birleşimidir.

Bu bir temsil/okuma yordamı sonucudur; kavramların deneyimden oluştuğu bir öğrenme deneyi değildir. Kullanıcının çapraz bağ önerisini dışlamaz; hangi kısıt altında çapraz bağların gerekli veya ekonomik olabileceğini daha kesin sorar. Bir ağacı, onu nasıl okuyabildiğinden bağımsız değerlendiremeyiz.

## Denetim ve saklama

Bağımsız serbest-koordinat eşleme sayımıyla 4.032 cevap; 3.024 olasılık skoru; 672 durum anlık kaydı; 1.728 yöntemler arası tam olasılık eşliği ve 4.032 eski yerel cevap kontrolü denetlendi. 168 yolun tümünde başlangıç + edinim + hizmet sayaçları son toplamla eşleşti. 12 yeniden adlandırma × 6 yöntem = 72 yolun olasılıkları değişmedi. `point`'in sayısal ada bağlı eşitlik bozması bu invarians iddiasına dahil değildir.

84 ayrı süreçten yükleme denetiminin 63'ünde kalan sorgular/gelecek köprüler işlendi; 21'i son noktadaki boş kuyruk yüklemesidir. Kanıt kaydına göre bir bozuk aday ve bir bozuk köprü reddedildi; bu çalışma zamanı yükleyicisinin saldırgan kayıtlara karşı güvenlik sertifikası değildir. Tam motor yeniden çalıştırması ve 3.780 ayrı ağaç-kategori birleşim hesabı geçti. Önceki **1.075 dosya** ve Git HEAD değişmedi. Ana uygulama, kitap, commit, push ve yayın işlemi yoktur.

## Araştırmaya etkisi

Bir yapay sistemin yaşadığı deneyim nedeniyle gelecekte farklı hesap yapması bu örnekte mümkün; fakat mekanizma hazır bir dille eşleşme kaydetme ve çıkarımdır. Ortak anlamı veya öğrenme işlemini icat etme başarısı yoktur. Yeni turu başka bir permütasyon sorusuyla uzatmak bu açıklama açığını kapatmaz. [Matematik ve açık hipotezler](MATHEMATICS_AND_NEXT_TR.md), “edinilmiş içerik”, “edinilmiş hesap yolu” ve “edinilmiş güncelleme/temsil dili” ayrımını koruyarak sonraki soruyu belirler.

Önceki tur 21'in tablosundaki “185 lemma/yaşam ortalaması” ifadesi de bir yazım hatasıdır: ham veride 185, `scoped` yönteminin **24 yaşam toplamıdır**; ortalama 7,7083'tür. Önceki dosya korunmuştur; bu düzeltme eski deney sonuçlarını değiştirmez.
