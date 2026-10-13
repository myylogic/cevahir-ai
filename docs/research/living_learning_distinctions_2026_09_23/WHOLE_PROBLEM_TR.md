# Tur 19 sonrası ana problem: ayrım edinimi nereye kadar açıklıyor?

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Araştırma yönü kaydı

Ana problem değişmedi: **Çalışan bir yapay sistem, deneyimleri nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir biçimde nasıl değiştirebilir?** Yeni ayrımlar oluşturmak bu problemin bir parçasıdır. Sonlu makine öğrenimini bütün sorunun yanıtı kabul etmiyoruz.

## Bu turla değişen araştırma durumu

[Tur 18 sonrası değerlendirmede](../living_learning_interactions_2026_09_22/WHOLE_PROBLEM_AND_NEXT_TEST_TR.md), hazır bütün bağlam kapıları yerine ayrımın içeriğinin edinilmesi önceliklendirilmişti. [Tur 19](REPORT_TR.md) bu boşluğa dar, çalışan bir örnek ekledi: doğru durum sayısı ve doğru devam listesi verilmeden, gözlenen sonuçlardan yeni devam testleri ve bunlarla yürütülebilir bir durum grafiği oluşturuldu.

Bu açıdan önceki deneyi aynı hazır aday listesiyle tekrar etmedik. Fakat yeni ayrımın üretim yordamı klasik otomata öğreniminden geldi. Dolayısıyla “şimdi bir örneğini çalıştırdık” ile “yeni bir temel ilke bulduk” ayrılır. İkincisi gerçekleşmedi.

| Araştırma gereksinimi | Bu turda olan | Açık kalan |
|---|---|---|
| Deneyimin kalıcı etkisi | S/E ayrımları, öğrenilen geçişler ve etiket arşivi süreçler arasında taşındı. | Hangi kayıtların güvenle atılabileceği ve sınırlı bellekle uzun yaşam. |
| Yeni hesap ayrımı | Gözlenen tutarsızlıktan yeni devam kuruldu; doğru durum sayısı verilmedi. | Yeni işlem, gözlem ve değerlendirme dilinin edinilmesi. |
| Genelleme | 57 kalıcı model bütün sonlu dizilerde doğru; 15'i eksik. | Aynı sabit dünya dışındaki görevlere aktarım ve yeniden kullanım. |
| Kontrollü güncelleme | Bilinen kanıtla uyum denetlendi; başarısız model tam doğru sayılmadı. | Gürültü, çelişkili/yanlış sonuçlar ve kanıtın hangi eski parçaya ait olduğu. |
| Sonraki öğrenmenin değişmesi | Kalıcı tablo aynı kanıtla yeniden kurulacak hesabı önledi. | Öğrenme yordamının kendisinin edinilmesi veya güçlü farklı algoritmaya üstünlük. |
| Sürekli çalışma | Ardışık sıfırlanabilir bölümlerde öğrenme sürdü. | Dünyayı sıfırlamadan ve mevcut çalışan durumu kaybetmeden yeni temsil kurma. |
| Kavramların yeniden örgütlenmesi | Geçici olarak birleştirilmiş geçmişler ayrıldı. | Birleştirme, çapraz görev anlamı, değer değişimi, kavramın küçülmesi ve farklı eski kavramların birleşmesi. |

## Birbirine indirgenmeyen üç eksik

**1. Ayrımı mümkün kılan içerik.** Bazı dünyalarda gereken devam hiç yaşanmadı. Bu, yeni deneyime erişim meselesidir ve önceki araştırmada zaten bir bölgeydi. Tur 19 bunu başka bir yapıda yeniden gösterdi. Yalnız daha çok sorgu seçmeye odaklanmak ana problemi tekrar keşif problemine daraltır; bunu yapmamalıyız.

**2. Ayrımı kuracak hesap biçimi.** Deneyim geldiğinde yeni ayrımın nasıl kurulacağı bu turda hazırdı: devamın başına eylem ekle, satırları böl, geçiş tablosu kur. Klasik yaklaşımın çalışması bu yordamın deneyimle edinildiği anlamına gelmez. Tur 17'de öğrenilmiş düzeltme şemalarının güçlü toplu kontrolle aynı sonuç vermesi de korunur. Yeni yöntem iddiası, aynı geçmiş ve toplam kaynakla daha güçlü bir karşılaştırmaya dayanmalı.

**3. Yeni yapının mevcut yaşama bağlanması.** Burada her sözcük başında dünya sıfırlanabildi. Tur 16'da yeni kuralın çalışan duruma taşınmasının ayrı sorun olduğu görülmüştü; fakat o tur yeni kuralın kendisini hazır verdi. Deneyimle kurulmuş yeni yapının sıfırlamasız çalışma ortasında devreye alınması, bu iki parçanın birlikte henüz sınanmadığı bir açıklık. “Yeni modeli öğrendim” mevcut konumunun doğru taşındığı anlamına gelmez.

Bu üç eksik, zorunlu ayrı yazılım modülleri önerisi değildir. Aynı yapı birden fazlasını sağlayabilir; ayrım, iddiaları ölçülebilir kılmak içindir.

## Kullanıcının önerisinden kalan araştırma hipotezleri

Kullanıcının kavramlar arasında çapraz bağlantı ve yaşam boyu yeniden örgütlenme açıklaması [ilk gözlem kaydında](../living_learning_concepts_2026_09_22/OBSERVATION_AND_HYPOTHESES_TR.md) korunur. Tur 19 o açıklamanın bütünü için biyolojik kanıt değildir. Devam araştırması şu iki ayırıcı üzerinde kurulabilir:

**H19-A — Yeni ayrımın yararı, yalnız mevcut hedefin daha iyi çözülmesinde olmayabilir.** Aynı mevcut çıktılara sahip iki geçmiş örgüsü, yeni görev geldiğinde farklı edinim/taşıma maliyeti doğurabilir. Bunun bilgi yönündeki matematiksel ayrımı önceki [kapsam değerlendirmesinde](../living_learning_reassessment_2026_09_20/SCOPE_AND_FOUNDATIONS.md) zaten vardı. Yeni araştırma yükü, gerekli ayrımların hazır etiketle verilmeden edinilmesi ve daha sonra yeni görevde gerçekten kullanılmasıdır. Yalnız durum sayısının büyümesi olumlu sonuç sayılmamalı.

**H19-B — Tam bir eski model ile gelecekteki öğrenme için yeterli durum farklı olabilir.** Tur 19'da tam doğru makine, aynı sabit dünyanın sonraki çıktıları için yeterlidir. Yeni gözlemlenebilir sonuçlar veya yeni işlemler eklendiğinde eski modelin bazı ayrımları artık yetersiz kalabilir. Burada iki koşul ayrılmalı: ayırt edici bilgi eski deneyimde erişilebilir biçimde duruyor mu, yoksa hiç gözlenmedi veya kayboldu mu? İkinci durumda temsil büyütmek bilgiyi yoktan çıkaramaz. Bu da önceki bilgi sınırının yeni uygulamasıdır; yeni teorem diye sunulmaz.

Bu hipotezler için kesin performans protokolü henüz yazılmadı; başarı iddiası yoktur. Kontroller aynı eski deneyimi, aynı gözlem hakkını, edinim/taşıma bedelini ve bütün yaşam kaynaklarını almalı. Yeni doğru kavramın adını veya görev sınırını öğretmenin vermesi, çözülmek istenen boşluğu gizlememeli. Görev değişimi olmadığı durum ve gereksiz ayrımın maliyeti negatif kontrol olarak bulunmalı.

## Yenilik konusunda elde kalan

Yeni bir terim, tablo, grafik veya birden fazla bilinen yöntemi bir araya getirmek tek başına yeni temel ilke değildir. Bu turun katkısı daha dar ve denetlenebilirdir: bir önceki hazır-ayrım varsayımı kaldırıldı; edinilen ayrımların bütün sonlu dizilerde yeterli olduğu ve yetersiz kaldığı koşullar ölçüldü; aynı kanıt kontrolü yeni bilgi ile hesap tekrarını ayırdı.

Sıfırlamayı kaldırmak da başlı başına keşif sayılmaz: homing dizileriyle öğrenmenin birincil literatürü [matematik notunda](MATHEMATICS_AND_PRIOR_ART_TR.md) kontrol edildi. Devam çalışmasının yükü, eldeki farklı sonuçları belirli varsayımlar ve maliyetlerle birleştirip hâlâ açıklanmayan farkı göstermektir. Beynin tam çalışma ilkesi veya genel zekâ çözümü bulunduğu iddiası yoktur. Eski olumlu/negatif kayıtlar korunarak yerel araştırma sürer.
