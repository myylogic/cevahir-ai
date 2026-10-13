# Tur 20 sonrası: birleşen parçalar ve kapanmayan ana açıklama

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Araştırma yönü kaydı

Ana soru hâlâ: **Bir yapay sistem çalıştığı süre boyunca yaşadığı deneyimler nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir biçimde nasıl değiştirebilir?** Yeni bir modelin çalışan geçmişe bağlanması bu sorunun gerekli olabilecek bir tarafıdır; tanımı veya bütünü değildir.

## Bu tur neyi birleştirdi?

[Tur 19](../living_learning_distinctions_2026_09_23/REPORT_TR.md), hazır doğru durum sayısı olmadan yeni devam ayrımları ve hesap grafikleri kurdu. [Tur 16](../living_learning_migration_2026_09_22/REPORT_TR.md), yeni bir kuralı akan geçmişte doğru durumdan başlatmanın ayrı sorun olduğunu gösterdi. [Tur 20](REPORT_TR.md), tur 19'un gerçek model değişimlerini ayrı ve sıfırlanmayan bir akışa bağladı. Böylece modelin hazır doğru verilmesi kolaylığı kaldırıldı; yanlış modellerle doğru taşımanın da mümkün olduğu ölçüldü.

Fakat kalibrasyon sorguları hâlâ başlangıca dönebilen ayrı bir kanaldadır. Gelen yeni yapı, canlı akışın kendi geri bildiriminden öğrenilmiyor. Birlikte çalışan bu iki parçanın varlığı, kendiliğinden gelişen açık uçlu bir sistemin kurulduğu anlamına gelmez.

## Korunan ve daraltılan iddialar

| İddia | Tur 20 sonrası durum |
|---|---|
| Daha doğru yeni program, geçmişi kendiliğinden doğru yorumlar. | Yanlış: doğru program evrelerinde yanlış başlangıç 713 yanlış cevap verdi. |
| Ham geçmişi saklamadan hiçbir doğru taşıma mümkün değildir. | Yanlış genelleme: mevcut özet/son iz yeterli olduğunda doğru durum veya doğru çıktı bulunabiliyor. |
| Yeni durumu tam bilemiyorsak kesin yanlış cevap zorunludur. | Yanlış: kapsayıcı küme ve kararsızlık bazı yanlış kesin cevapları önleyebilir; hizmet kapsamının bedeli vardır. |
| Eski öğrenilmiş yapı her yeni taşımaya ek yarar sağlar. | Bu deney desteklemedi: eski-model+sekiz-bit, güçlü sekiz-bit/parity kontrolüyle bütün kümelerde aynıydı. |
| Daha az yanlış cevap veren küme yöntemi daha iyi model öğrenmiştir. | Burada yanlış: bütün yöntemler aynı modelleri aldı; bazı farklar cevap vermemeden doğdu. |
| Arşivden yeni programı yeniden yürütmek genel olarak en pahalıdır. | Gösterilmedi: bu küçük yaşamların ortalamasında ilişkili kısa-iz hesabından daha ucuzdu, ama daha çok ham geçmiş sakladı. |

Bu bulgular önceki sonuçları geri almaz. Özellikle tur 17'de güçlü kontrolün öğrenilmiş onarımı eşlemesi, tur 18'de koşulsuz kalıcılığın ek maliyeti, tur 19'da kusursuz örnek başarısının yanlış modeli gizlemesi korunur. Giderek daha çok yapı eklemek otomatik olarak daha iyi öğrenme sağlamıyor.

## Araştırma alanının yeniden açılması

Son iki tur deterministik sonlu makineler kullandı. Bunun yararı tam doğrulama ve açık karşı örneklerdir. Riski, yaşayan öğrenmeyi giderek yalnız makine öğrenimi ve durum taşıma olarak görmektir. Ana problemi bu alanda çözmüş saymayacağız. Kullanıcının kavramların dallanması, birleşmesi, çapraz bağlanması ve eski olayların yeni yorum kazanması önerisinin büyük bölümü bu dünyalarda bulunmuyor.

Üç alan özellikle ayrılıyor:

**A. Öğrenilen içerik kadar öğrenme biçiminin de deneyimle gelişmesi.** Bu turdaki taşıma ilişkisi ve tur 19'daki ayrım kurma yordamı araştırmacı tarafından yazıldı. Bir adayın gelecekteki hesap kapasitesini değiştirmesi ile o adayın nasıl üretileceğini öğrenmek aynı başarı değildir. Önceki program-kütüphanesi ve onarım deneyleri burada kısmi örnekler sağladı, fakat güçlü kontroller yeni ilke iddiasını desteklemedi. Yeni çalışmada aynı şeyi farklı isimle yeniden uygulamak yerine, edinilmiş bir yordam çıkarılınca aynı yeni deneyimden hangi hesabın artık kurulamadığı veya pahalılaştığı ayrılmalı.

**B. Ortak anlamın birden fazla görevde yeniden örgütlenmesi.** Şimdiye kadarki makinelerde hedef çıktı fonksiyonu sabittir. Yeni bir ihtiyaç eski ayrımların yeterliliğini değiştirmedi; öğrenici yalnız sabit dünyanın eksik açıklamasını tamamladı. Aynı geçmiş olayların bir görev için birleştirilip başka görev için ayrılması, iki ayrı kaynakta öğrenilen yapının ortak bir yeni işlemde kullanılabilmesi ve bunun eski becerilere etkisi daha geniş araştırma yüküdür. [Kullanıcının ilk gözlemi](../living_learning_concepts_2026_09_22/OBSERVATION_AND_HYPOTHESES_TR.md) burada canlı tutulmalıdır. Hazır doğru görev/bağlam etiketleri sonucu kolaylaştırıyorsa açıkça sayılmalı.

**C. Yaşamın ortak kaynak kararı.** Sekiz bit mi, tam arşiv mi, eski çalışma durumu mu korunacağı bu turda hazır yöntem seçimi oldu. Gerçek araştırma açığı, yeni görev bilinmeden hangi izi tutacağına ve hangi hesabı yeniden kullanacağına deneyimle karar verilmesidir. Önceki sorguya-göre yeterlilik sınırları, bütün olası gelecekler için bedelsiz eksiksiz özet olmadığını gösterir; fakat bu, gerçekçi dağılımlarda işe yarayan kontrollü seçimlerin bulunamayacağı anlamına gelmez. Ortak bütün-yaşam hesap ve ayrı bellek sınırı, edinim/taşıma/düzeltme bedelleri birlikte ölçülmeli.

Bu alanlar yeni bir mimarinin zorunlu modülleri olarak önerilmiyor. Araştırılan kapsam ile açık kalan iddiaları ayırıyor. Bir sonraki aşamada yalnız ilişki ürününü daha hızlı hesaplamak veya sonek boyunu artırmak, ana açıklama açığını tek başına kapatmaz.

## Sonraki ayırıcı çalışma için ölçütler

Sonraki deneysel yön, **farklı deneyim kümelerinden edinilmiş yapının yeni bir görevin hesap biçimini değiştirmesi** olmalı. Tur 12'de verilen ortak ilişki diliyle bileşim zaten gösterildi; aynı başarı yeni buluş sayılamaz. Yeni yük, hangi eşlemenin veya yeniden kullanılabilir işlemin seçileceğinin doğru kimlik verilmeden edinilmesi ve yanlış aktarımda kendi etkisinin düzeltilebilmesidir.

Karşılaştırmalarda güçlü tek-seferlik çıkarım/sentez, aynı geçmişten yeniden hesaplama, öğrenilmiş parçayı kullanımı kapatılmış ama kanıtı korunmuş karşı olgu ve bütün yaşam kaynakları birlikte bulunmalı. Ortak arayüzün ne kadarının hazır verildiği açıkça yazılmalı. Daha fazla veri, daha büyük başlangıç dili, hedefin gizli yapısını veren görev üreticisi veya ucuzlaştırılmış maliyet hesabı kazanımı açıklıyorsa iddia daraltılmalı.

Bu belge yeni deneyi olmuş gibi anlatmaz; kesin algoritma ve performans protokolü sonraki çalışmada sonuçlardan önce yazılmalıdır. Temel ilke hâlâ bulunmamıştır. Sırf teoriye yeni isim vermek veya biyolojik anlatımı yazılım kutularına çevirmek bu eksikliği kapatmaz. Yerel araştırma, önceki bütün olumlu ve olumsuz sonuçlarla devam eder.
