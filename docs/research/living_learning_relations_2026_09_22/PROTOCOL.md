# Ayrı edinilmiş ilişkilerden yeni bileşim ve hedef öğrenimi — yerel ön plan

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Araştırma turu 12

Bu protokol ana deneyin başarım sonuçları hesaplanmadan yazıldı. Dış bir ön kayıt değildir. Önceki araştırma, kullanıcı gözlemleri ve kitap korunur; commit/push/yayın yapılmaz.

## Çıkış noktası

Kullanıcının [anlam örgüsü gözlemi](../living_learning_concepts_2026_09_22/OBSERVATION_AND_HYPOTHESES_TR.md) ve [çapraz ilişki açıklaması](../living_learning_concepts_2026_09_22/CROSS_CONNECTIONS_AND_TRANSFER_TR.md) korunur. Bu deney onların tümünü veya beyin mekanizmasını modellemez. Belirli bir hesaplama yükümlülüğünü sınar: **ayrı ortamlarda deneyimle öğrenilmiş ilişkileri, yeni adlandırılmış bir ortamda eşleştirerek doğrudan öğretilmemiş bileşik sonuçları öngörmek ve hedefi daha az örnekle edinmek**.

## Hipotezler

- H8: İki ayrı kaynak ilişkisi, hedefteki doğru eşleme verilmeden birlikte kullanıldığında, her bir kaynağın tek başına taşınmasına göre yeni bileşimleri daha az hedef deneyimiyle öngörebilir.
- H9: Kaynakla yüzeysel değil yapısal uyum gereklidir. Doğru kaynak bilgisinin kapsamı değişmiş hedefe taşınması erken yanlış kesinlik doğurabilir; çelişkiyle gözden geçirme gerekir.
- H10: Aynı hedef kanıtında, kaynak deneyimini korumak ile kaynak–hedef eşlemesini bilmek farklı avantajlardır. Ad değiştirme kontrolü eşleme sızıntısını dışlamalı; birden fazla olası eşleme varken tahmin dağılımı korunmalı.
- H11: Kaynak ilişkilerin birlikte kullanılması, hedefte iki işlemli sonuçlar için doğrudan eğitim etiketi almadan yarar sağlayabilir. Bu, verilen işlem-bileşimi yordamının kullanımıdır; bileşim ilkesinin sıfırdan keşfi diye sunulmayacak.

## Dünya, kaynaklar ve açık önbilgi

Her ortamda altı gözlenebilir nesne/durum vardır. Öğreniciye altı nesnenin listesi ve eylemlerinin kimlikleri verilir; nesnelerin gizli rolleri verilmez. Kaynak A'da bir eylem altı durumluk bir döngü oluşturur. Kaynak B'de başka bir eylem üç ayrı ikili döngü oluşturur. Kaynakların adları bağımsız karıştırılır. Her kaynakta durum başına iki gerçekleşmiş geçiş, toplam 12 gözlem gelir. Öğrenici bu gözlemlerden geçiş tablosunu kurar; gizli tablodan eksik bilgi tamamlamaz. Herhangi bir eksik kaynak kenarı varsa sınıf çıkarımı yapılmaz.

Kaynak modeli edinildikten sonra, altı nesnenin 720 bijektif yeniden adlandırması denenir ve yinelenen geçiş fonksiyonları elenir. Bu işlem, aynı ilişkisel yapının farklı nesnelerdeki adaylarını üretir. **Bir kaynağın hedefe nesne bijeksiyonuyla taşınabileceği hipotez dili araştırmacının önkabulüdür.** Model bunu deneyimden icat etmiş sayılmaz. Kaynaklardan sınıf adları veya gerçek hedef eşlemeleri alınmaz; sınıfın geçiş yapısı gözlemlerden edinilir.

Hedefte iki eylem vardır. Bunların kimlikleri de karıştırılır; hangi eylemin hangi kaynağa benzediği verilmez. Her eylem herhangi bir kaynak sınıfına ait olabilir; “mutlaka biri A, diğeri B” bilgisi öğreniciye verilmez. İki eylemin kaynak nesnelerinden hedef nesnelerine eşlemeleri birbirinden bağımsızdır. Dolayısıyla kaynaklardaki hazır bir ortak nesne kimliğiyle bağ kurma yoktur. Eşleme kısıtları yalnız hedefte gerçekleşmiş geçişlerden gelir.

İki hedef eyleminin bileşimi kaynak deneyimlerinde birlikte görülmemiştir. Değerlendirmede `eylem0→eylem1` ve `eylem1→eylem0` dizileri, altı başlangıcın her birinde sorulur. Ara sonucun sonraki eyleme girdi olması yorumlayıcının verilen işlevidir; bunun öğrenildiği iddia edilmez.

## Üç hedef rejimi

1. **Uyumlu yeni bileşim:** Hedef eylemler bağımsız yeniden adlandırılmış A ve B ilişkileridir.
2. **İlişki ailesi değişimi:** B yerine iki ayrı üçlü döngü gelir. Altı nesne, eylem sayısı ve bijektiflik aynıdır; yalnız ayrıntılı ilişkisel yapı farklıdır. A hâlâ geçerlidir.
3. **Yerel istisna:** Uyumlu hedefte B'nin tek geçişi başka bir sonuca yönlendirilir; hedefin bu eylemi artık bijektif değildir. Eski yapıyı zorlamak yanlış olabilir. A yine geçerlidir.

24 yaşam (`0..23`), üç rejim. Aynı yaşamda nesne/eylem adları, hedef gözlem sırası ve değişmeyen kaynaklar yöntemler arasında eşleştirilir. Hedefte 12 farklı durum–eylem hücresinin her biri bir kez gözlenir. Her yeni sonuç verilmeden önce tahmin kaydedilir; sonra güncelleme yapılır. Hedef tamamen açığa çıktığında denetlenebilir bitiş kontrolü vardır.

## Sekiz yöntem

- `table`: Görülmüş hedef geçişini tutar; görülmemiş deterministik hücreler için altı sonuç üzerinde eşit olasılık. Kaynaksız temel kontrol.
- `permutation_prior`: Her eylem için bütün 720 bijektif fonksiyonu baştan olası sayar; çelişkide o eylem için hedef tablosuna döner. Bijektiflik önbilgisi verilmiş güçlü kontrol, kaynak bilgisini öğrenmiş değildir.
- `source_a`: Yalnız A'dan edinilen aday sınıfını iki hedef eyleme de dener; çelişkide ilgili eylem için tabloya döner.
- `source_b`: Yalnız B'den edinilen sınıf; aynı geri dönüş.
- `source_both`: Her eylem için A/B sınıflarına eşit ön olasılık, sınıf içinde farklı geçiş fonksiyonlarına eşit olasılık. Hedef kanıtı adayları ve sınıf olasılıklarını günceller. Bütün adaylar elenirse yalnız o eylem tabloya döner.
- `unrelated_sources`: Eşit sayıda doğru kaynak gözleminden edinilmiş farklı sınıflar: iki üçlü döngü ve bir sabit nokta ile beşli döngü. Aynı aktarım ve geri dönüş.
- `frozen_source_both`: `source_both` başlangıcı; yeni kanıt mevcut son adayların tümünü eliyorsa yapısal durumu değiştirmeyi reddeder. Kaynak doğruluğunu hedefte değişmez kabul etmenin bedeli için kontrol.
- `cycle_catalog`: Altı nesnede mümkün bütün permütasyon döngü türlerini baştan içerir. Türlere eşit, tür içindeki fonksiyonlara eşit ön olasılık; çelişkide tabloya döner. Bu da ayrıcalıklı yapısal dil kontrolüdür; bütün sabit mimariler adına en iyi yöntem diye sunulmaz.

Gözlenmiş hücrelerde deterministik ve gürültüsüz sonuç varsayımı kullanılır. Tahmin, uyumlu adayların posterior ortalamasıdır; ilk aday keyfî seçilmez. Hedef etiketlerine göre hiperparametre ayarlanmaz.

İlk başarım çalıştırmasından önce uygulama açıklığı: `frozen_source_both` gerçekleşmiş hedef olgusunu denetim tablosuna yine yazar; yalnız yapısal aday havuzunu boşaltan güncellemeyi reddeder. Sonraki filtrelemeler son boş olmayan havuz üzerinde sürer. Tahmin için olgu tablosuyla sessizce düzeltme yapmaz; bu nedenle gördüğü bir olguyu da yanlış tahmin edebilir. Diğer yapısal yöntemler ilk boşalmada o eylem için kalıcı olarak olgu tablosuna geçer. Yanlış kesinlik, en büyük tahmin olasılığı en az `1−1e−12` olup bu kesin sonucun yanlış olmasıdır. Maliyet sayaçları yöntem başına mantıksal işleri sayar; paylaşılan önhesaplamaların gerçek süre tasarrufunu CPU eşitliği diye yorumlamaz.

## Ölçüler

24×3×8 = **576 edinim yolu**. Her yolda başlangıç ve her 12 güncelleme sonrasında değerlendirme; aynı dünyadaki bu zaman noktaları bağımsız örnek sayılmaz.

Öncelikli ölçüler: gelecek etiket gelmeden 12 adımın ortalama Brier hatası; henüz gözlenmemiş tek-adım hücreler üzerindeki Brier hatası; iki farklı eylemden oluşan 12 bileşik sorgunun Brier hatası. Brier, altı sonuç üzerindeki kare olasılık hatalarının toplamıdır; sıfır iyi, yanlış kesin tahmin iki verir. Başlangıç, 2/4/6/8/10/12 kanıt noktaları özetlenecek. Eşit en yüksek olasılıklarda kimliğe bağımlı seçim yapmamak için başarı, eşitlikler arasında uniform rastgele seçimin beklenen doğruluğuyla ölçülecek.

Bir ve iki eylem için aday posteriorları bağımsız tutulur. Bu nedenle iki farklı eylemli sorgunun tam olasılığı, iki ortalama geçiş matrisinin çarpımıdır. Aynı bilinmeyen eylemin tekrarlandığı dizilerde bu çarpım genel olarak doğru olmayacağı için onlar bu deneyde yoktur; sonuç uzun zincirlere genişletilmez.

Ayrıca yanlış kesin tahmin sayısı, kalan aday sayısı, elenen adaylar, reddedilen güncellemeler, tabloya dönüş zamanı ve her adımda sınıf posteriorları kaydedilecek. Eşleme sayısı tahmin belirsizliğiyle eş tutulmayacak. Maliyet: kaynak gözlemleri, 720'lik yeniden adlandırma girişimleri, benzersiz aday tabloları, filtrelemede sınanan aday/hücre sayısı ve saklanan dizi boyutu. Bunlar toplam CPU/FLOP/RAM eşitliği değildir.

## Kontroller ve sınırlar

Kaynaklar yalnız gerçekleşmiş geçiş kayıtlarından yeniden kurulacak; sıra/ad değiştirme sonrası tahminler uygun biçimde dönüştürüldüğünde aynı kalmalı. Kaynak–hedef gerçek eşlemeleri, hedef rejimi ve gizli tablonun görülmemiş hücreleri öğreniciye verilmeyecek. İki-adım olasılık hesabı küçük aday kümelerinin tüm kombinasyonlarıyla bağımsız doğrulanacak. Kaynak ve hedef durumu ayrı süreçte saklanıp sonraki güncellemelerle birlikte yeniden başlatılacak. Ana sonuçlar tekrar hesaplanacak.

Bu model gelişim, duygu/değer, algıdan özellik oluşumu, görev sınırı keşfi, sınırsız kavram derinliği veya insan beyninin genel mekanizmasını çözmez. Hedef ilişkilerini sonlu geçiş fonksiyonlarıyla ifade etme ve birleştirme dili verilmiştir. Kullanıcının geniş açıklaması bu dar deneyle özdeşleştirilmez; diğer hipotezler açık kalır.
