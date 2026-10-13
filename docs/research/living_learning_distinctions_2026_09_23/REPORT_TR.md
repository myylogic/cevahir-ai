# Tur 19 — Deneyimle yeni ayrımlar kurmak ve eksik ayrımda kalmak

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Yerel araştırma raporu

**Doğru durum sayısı ve ayırt edici devam listesi verilmeden, gözlenen sonuçlardan yeni ayrımlar ve yürütülebilir bir durum grafiği kuruldu.** Kalıcı yöntem 72 dünyanın 57'sinde bütün sonlu girdi dizileri için doğru modele ulaştı. Fakat bu mekanizma klasik gözlem tablosu öğrenimidir; yeni bir temel computational ilke bulunduğu anlamına gelmez. Aynı kanıttan yeniden kurma aynı modellere ulaştı. Daha geniş devamları deneyen kontrol bir dünyayı daha çözdü. Ana yaşayan öğrenme problemi hâlâ açıktır.

## 1. Önceki araştırmaya eklenen şey

[Tur 18](../living_learning_interactions_2026_09_22/REPORT_TR.md), verilmiş düzenlemelerin birlikte etkisini sınamıştı. Bu tur, kullanıcının “deneyimle yeni dallar/ayrımlar oluşması” düşüncesine geri döner. [Tur 15'te](../living_learning_temporal_2026_09_22/PROTOCOL.md) son-ek üst sınırı ve iki-durum aday listesi hazırdı. Burada doğru durum sayısı ve doğru ayırıcılar verilmedi. Sistem, önceki devamların önüne bir eylem ekleyip gözlenen tutarsızlıktan yeni bir ayırıcı kurdu.

Yine de alfabe, ikili çıktı, sabit deterministik sonlu-durum varsayımı, başlangıca sıfırlama, istenen diziyi sorgulayabilme ve tabloyu güncelleme algoritması hazır verildi. Öğrenici yeni bir algı dili, amaç, değer veya temel işlem icat etmedi. “Önceden bütün doğru kavramlar verildi” ile “hiçbir önbilgi verilmedi” arasında açık bir konum var: ayırıcı **içerik** edinildi; onu oluşturma yordamı verildi.

[Protokol](PROTOCOL.md) ana performans çalıştırmasından önce kaydedildi. 24 tohumda üç dünya ailesi ve dört yöntem: 72 dünya, 288 öğrenme yolu, 4.608 blok sonucu, 18.432 dış etiket-öncesi tahmin. Aynı tohumun aileleri aynı dış dizileri kullanır; yöntemler eşleştirilmiştir. Bunlar 288 bağımsız dünya tekrarı değildir. Her yaşamda 64 dış gözlem, dört gözlemli 16 blok halinde gelir; model blok sonunda güncellenir.

## 2. Öğrenilen ayrımın ne olduğu

Sistem iki geçmişi, şimdilik bildiği devamların sonuçları aynıysa aynı durumda tutar. Aynı eylemden sonra farklı bir sonuç görülürse eski devamın önüne o eylemi bağlayarak yeni bir test oluşturur. O test önce birlikte tutulan geçmişleri ayırır; ayrım yeni programın durumlarına ve geçişlerine katılır.

Örneğin küçük doğrulama dünyasında boş geçmiş ile `0` geçmişi o an sıfır çıktı verir. Fakat `1` devamı birinde sıfır, diğerinde bir üretir. Yeni ayrım böyle doğar. Performans kayıtlarında da her bölünmenin önceki eşit satırı ve onu ayıran devam/çıktılar saklandı. Kalıcı yöntemde 72 dünyanın 58'inde en az bir yeni test kuruldu; bunların 57'sinde tam model elde edildi.

Bu, aynı dünyanın sınırlı kanıt altında eksik bölümlemesinin düzelmesidir. Gerçekten bütün devamlarda eşdeğer geçmişler sabit dünyada sonradan farklılaşmaz. Kullanıcının anlattığı değişen değerler, anlamların yeniden yorumlanması ve kavramların daralması bu turda gösterilmedi. Matematik ve örnek [ayrı nottadır](MATHEMATICS_AND_PRIOR_ART_TR.md).

## 3. Bütün sonlu diziler için doğruluk

Gizli makine ile öğrenilen makinenin erişilebilir durum çiftleri incelendi. Çıktılar bütün çiftlerde eşitse, mümkün **bütün sonlu dizilerde** eşdeğerlik doğrulandı. Bu, yalnız 128 testin geçmesi değildir. Değerlendirici hedefi bilir; öğreniciye bu denetimin sonucu veya bulduğu karşı örnek verilmez.

| Yöntem | Döngü + sıfırlama: doğru dünya /24 | Rastgele geçişli döngü /24 | Gecikmeli çıktı /24 | Toplam /72 |
|---|---:|---:|---:|---:|
| Deneyimle ayrım kuran kalıcı tablo | 10 | 23 | 24 | **57** |
| Aynı kanıttan baştan yeniden kurma | 10 | 23 | 24 | **57** |
| Gerekli uzunluğa kadarki bütün devamları ekleme | 10 | 24 | 24 | **58** |
| Yeni ayrım oluşturması kapatılmış kontrol | 0 | 0 | 0 | **0** |

Son satır kapasite ablasyonudur; güçlü genel öğrenici değildir. Asıl karşılaştırmalar klasik kalıcı yöntem, aynı-kanıt yeniden kurma ve geniş devam kontrolüdür. Geniş kontrol, tutarsızlık oluştuğunda o uzunluğa kadarki bütün devamları ekler; başlangıçtan keyfî uzun devamları bağımsız olarak taramaz.

Gecikme ailesinde 8/16/32 durumlu yapılar öğrenildi; sayı öğreniciye verilmedi. Doğru çıkan toplam 172 son modelin durum sayıları, hedeflerin bağımsız hesaplanan minimal durum sayılarıyla eşitti. Bu 172, 57+57+58 eşleştirilmiş yöntem sonucudur; 172 bağımsız dünya değildir.

Sabit uzun test kümesi, yaşam başına görülmemiş 128 dizi ve uzunluk 32–64 içeriyordu. Öğrenmede sorgulanmış en uzun dizi 28'di; uzun testlerin tamamının sorgu arşivi dışında kaldığı denetlendi. Kalıcı yöntemin bu 9.216 uzun sorgudaki hatası yalnız **26** idi; buna rağmen **15 dünyadaki modeli tam doğru değildi**. Yüksek örnek başarısı ile doğru yapının edinimi ayrıldı.

## 4. Başarısızlığın somut yeri

Döngü+sıfırlama ailesindeki 14 başarısız dünyada bütün alınmış etiketler sıfırdı. Kalıcı yöntem tek durumlu, sürekli sıfır veren modelde kaldı. Topladığı bütün kanıtı doğru açıklıyordu; dünya ise daha uzun veya farklı düzenlenmiş bir devamda bir üretebiliyordu. Örneğin tohum 2'de `111111` yanlış modelin karşı örneğidir. Bu sözcük öğreniciye verilmedi ve öğrenme arşivinde yoktu.

Bu 14 dünyanın dördünde, görülmemiş uzun test kümesi de **sıfır hata** verdi. Tam durum-çarpımı denetimi yine karşı örnek buldu. Bu bulgu “sonlu testler işe yaramaz” demek değildir; seçilen örnek dağılımında nadir davranışların gözden kaçabileceğini somutlaştırır.

Rastgele geçişli ailede tek başarısız kalıcı yaşam tohum 19'du. Hedef 10 minimal durumlu iken öğrenilen model 9 durumdaydı; bilinen kayıtlarla hiç çelişmiyordu. Görülmemiş `000000011` dizisi farkı gösterdi. Geniş devam kontrolü bu dünyayı da öğrendi. Daha seçici ayrım üretimi burada daha ucuz olsa bile doğru ayrımı kaçırdı; koşulsuz üstünlük yoktur.

Bu tanılar performans sonrasında, yalnız değerlendiricide yapıldı. Yeni karşı örnekleri öğretip aynı deneyin başarısını yükseltmedik; dünya, sıra, kaynak sınırları ve öğrenici kodu sonuçlara göre değiştirilmedi.

## 5. Kalıcı yapının hesap etkisi

Kalıcı tablo ile her blokta geçmişten yeniden kuran kontrol, 72 dünyanın tamamında ve 1.152 eşleştirilmiş blokta **aynı öğrenici durumunu, aynı dış sorgu sırasını ve aynı sonuç arşivini** verdi. Kalıcılığın katkısı burada ek bilgi veya farklı nihai çıkarım değil, tamamlanmış hesabın yeniden kullanılabilmesidir.

| Dünya ailesi | Kalıcı: ortalama iç öğrenme işi | Yeniden kurma: ortalama iç öğrenme işi | Yeniden kurma / kalıcı |
|---|---:|---:|---:|
| Döngü + sıfırlama | 15.417,17 | 104.718,67 | 6,79 |
| Rastgele geçişli döngü | 57.136,00 | 470.542,17 | 8,24 |
| Gecikmeli çıktı | 47.385,58 | 419.849,96 | 8,86 |

Oranlar aile ortalamalarının oranıdır; yaşam başına oranların ortalaması değildir. Kalıcı yöntem 72 yaşamın hepsinde daha az iç iş kullandı. Baştan yeniden hesaplamanın pahalı oluşu yeni bir algoritmik buluş değildir; bu kontrol kalıcı örgünün etkisini izole eder.

Geniş devam kontrolünün aynı ailelerde ortalama iç işi 58.090,96 / 82.335,67 / 200.964,54 oldu. Çevreyi sıfırlama ve sembol yürütme bedeli de ayrı sayıldı:

| Dünya ailesi | Kalıcı: çevre birimi | Geniş devam kontrolü: çevre birimi |
|---|---:|---:|
| Döngü + sıfırlama | 1.226,83 | 4.002,33 |
| Rastgele geçişli döngü | 2.828,08 | 4.001,29 |
| Gecikmeli çıktı | 2.186,17 | 7.421,79 |

Geniş kontrol daha çok veya farklı ek sorgu yapar. Bu yüzden bu ikili, “aynı kanıttan daha iyi öğrenme” karşılaştırması değildir. Üstelik rastgele geçişli 24 dünyanın altısında kalıcı yöntemin çevre bedeli geniş kontrolden daha yüksek çıktı. Daha az yeni test eklemek tek başına daha az toplam sorgu garantilemez.

Sanal iç iş CPU zamanı, enerji veya tam bellek maliyeti değildir. Tablo okuma/karşılaştırma, kayıt, geçiş kurma ve model yürütme sayılır; Python nesneleri, sıralama ve hash giderlerinin tamamı sayılmaz. Dış akış tahmini ayrı sayaçta, bütün yöntemlerde ortalama 841 işti. Kalıcı JSON boyutu aileye göre ortalama yaklaşık 6,7 / 17,3 / 13,6 KB'dı; sorgu arşivi ve yaşam kaydı dâhildir, kaynak kodundaki önbilgi dâhil değildir. Bellek üst sınırı veya arşiv silme yöntemi sınanmadı.

## 6. Kaynak sınırı ve protokol açıklaması

Hiçbir yol kaynak sınırına ulaşmadı. En yüksek iç öğrenme işi 1.057.573, sınır 20.000.000; en yüksek çevre bedeli 33.256, sınır 100.000 idi. Dolayısıyla bu tur eşit sınırlı kaynak nedeniyle gerçekleşmiş başarı farkı göstermiyor; farklı miktarda kullanılan kaynakları raporluyor.

Bağımsız kod denetiminde protokolün bir ifadesiyle uygulama arasında sınır-durumu belirsizliği bulundu: protokol çevre bütçesi dolunca öğrenmenin donduğunu söylerken, kod **yeni, önbellekte olmayan sorgu sınırı aşacaksa** durur. Önbellekten yürüyen iç hesap o anda sürebilir; zorunlu dış gözlemler de akmaya devam eder. Ana çalıştırma bu denetim sürerken başlamıştı. Kodu veya protokolü geriye dönük değiştirmedik; gerçek uygulamayı burada açıkladık ve ayrıca sınır davranışını kontrol ettik. Hiçbir gerçek yol sınıra yaklaşmadığından raporlanan sonuçlar bu farktan etkilenmedi. Gelecek bütçe-kesilmesi deneyinden önce kural tek anlamlı hâle getirilmeli.

Öğrenicinin `known` kayıtları ile ortamın tüm fiziksel önbelleği de kavramsal olarak farklıdır; yarıda kesilen bir yeniden kurmada ayrılabilirler. Bu deneyde kesilme olmadı. Son modeller her iki kanıt kümesinde bağımsız denetlendi: üç uyarlanabilir yöntemde sıfır, sabit-ayrım kontrolünde toplam 1.677 uyuşmazlık bulundu.

## 7. Doğrulama ve korunan kayıt

- Bütün 288 yol yeniden çalıştırıldı; ham sonuç içerikleri birebir aynı üretildi.
- 4.608 blok modelinin tam eşdeğerlik sonucu, ayrı bölümleme algoritmasıyla doğrulandı; 589.824 uzun-test çıktısı bağımsız skaler hesaplandı.
- 57.799 son arşiv etiketi ve model çıktısı denetlendi. İlk bloktaki 1.152 tahmin ayrıca etiket açılmadan oluşan başlangıç modelinden kontrol edildi.
- 592 kaydedilmiş bölünme tanığında eski devamların eşitliği ve yeni devamın farklı sonuç üretmesi doğrulandı. Bu sayı yöntem tekrarları içerir.
- Üç durumlu mekanizma tanığı; uzunluk sekize kadarki 511 sözcük; bütün 64 ikili iki-durum makinesinin 4.096 eşleştirmesi ve bütçe kontrolü geçti.
- Dört yöntem × üç aile × iki noktadan toplam **24 yeni-süreç devamı**, sonraki kayıtlar ve maliyetlerle birebir eşleşti.
- Önceki **1.023 dosyanın** hash'i ve Git HEAD değişmedi. Kitap, ana uygulama, eski deneyler ve yayın dosyaları korundu. Commit, push veya yayın yapılmadı.

Bağımsız okuma denetimi, yeniden kurmanın gelecekteki kanıtı erken okumadığını ve öğreniciye hedef tablo verilmediğini kontrol etti. Belirtilen denetim boşlukları ek tanı betiğinde kapatıldı. Ana deneyin kaynak ve protokol hash'leri sonuçlarla uyuşur. Bu yerel, yeniden üretilebilir bir araştırma kaydıdır; dış önkayıt veya hakem değerlendirmesi değildir.

## 8. Ana probleme katkı ve kalan açıklama yükü

Bu tur, **deneyimle oluşturulan bir ayrımın gelecekteki hesap yapısına kalıcı katılabildiğini** çalışır biçimde gösterir. Aynı mekanizmanın eksik kanıt altında gelişmeden kalabildiğini ve doğru ayrımı bulmanın maliyetini de gösterir. Başarı, önceden verilen sonlu-durum dili ve deneyim arayüzü içindedir.

Angluin'in klasik gözlem tablosuyla doğrudan eşleşme nedeniyle bunu yeni temel ilke olarak adlandırmıyoruz. Ayrıntılı birincil kaynaklar ve kesin sınırlar [matematik/literatür notunda](MATHEMATICS_AND_PRIOR_ART_TR.md). Birbirinden bağımsız bilgi kümelerinin yeni işlemlerle birleşmesi, kavramların küçülüp yeniden örgütlenmesi, değer/amaç değişimi ve sıfırlamasız çalışma sırasında yeni temsilin kurulması açık. [Ana problem değerlendirmesi](WHOLE_PROBLEM_TR.md) bu turu bütün sorunun yerine koymadan korur.

Dosyalar: [yeniden üretim dizini](../../../research/living_learning_distinctions_2026_09_23/README.md), [özet](../../../research/living_learning_distinctions_2026_09_23/results/summary.json), [doğrulama](../../../research/living_learning_distinctions_2026_09_23/results/verification.json), [tanı kaydı](../../../research/living_learning_distinctions_2026_09_23/results/diagnostics.json), [son denetim](../../../research/living_learning_distinctions_2026_09_23/results/final_audit.json).
