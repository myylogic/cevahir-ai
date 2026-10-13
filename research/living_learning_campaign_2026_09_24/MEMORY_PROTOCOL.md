# Bellek kolu: önceden belirtilmiş protokol

Bu kayıt, katalog ve kod ana performans tohumları çalıştırılmadan hazırlanır.
MEM, CON ve RES aileleri yirmişer koşullu araştırma önermesi içerir. Yalnız
MEM01–MEM12 ilk dalgada çalıştırılır; kalan 48 önerme kuyruktadır. Tohumlar,
parametreler veya aynı mekanizmanın koşulları bağımsız temel ilkeler sayılmaz.

## Problem ve erişilebilir bilgi

16 anahtarın ikili etiketleri öğrenilir. Her adımda önce rastgele anahtar sorgusu
cevaplanır, sonra ayrı bir anahtara ait etiketli gözlem veya kimlikli düzeltme
verilir. Anahtarlar ve kimlikler hazırdır; kategori oluşturma veya yeni yordam
keşfi bu kolun konusu değildir. Gizli doğru etiket yalnız puanlayıcıdadır.
Öğreniciye evre sınırı, hata olup olmadığı veya gürültü oranı verilmez.
İlk altı anahtarın yanlı hatalı kaydedildiği düzeltme koşulunda daha sonra her
kaydın doğru etiketi kimliğiyle açıkça gönderilir; bu güçlü dış denetim açık
varsayımdır. Düzeltmeler sistemin kendi doğrulama keşfi değildir.

Her tohumu paylaşan aday/kontrol aynı sorgu ve gözlemleri alır. Kaynaklar:
1024 adım durağan rastgele, bloklu, seyrek kapsamalı, yarıda 8 anahtarı değişen
ve anahtarların 256+32k anlarında tek tek değiştiği akışlar; 512 adım düzeltme
akışı. Bütün normal koşullarda gözlem etiketleri bağımsız 0,25 olasılıkla bozulur.
Bloklu ve seyrek koşulların ilk 128 adımı tüm anahtarları 8 kez gösterir.
Bloklu devam her anahtarı 56 adım tutar; seyrek devamın %96'sı ilk 4 anahtardır.
Sorgular bütün anahtarlara tekdüze dağılır. Düzeltmede ilk 256 gözlemin hata
olasılığı ilk 6 anahtarda 0,85, diğerlerinde 0,10'dur. Sonraki 256 adım bu
kayıtların rastgele sıralı, tam kimlikli ve doğru düzeltmeleridir.

## Yöntemler ve maliyet

- count: anahtar başına iki birikimli sayı; durağan bağımsız gürültü için güçlü
  basit çoğunluk kontrolü.
- fifo/reservoir: toplam 96 gözlem; sırasıyla son kayıtlar ve standart tekdüze
  rezervuar seçimi. Her ikisi de saklanan kayıt sayılarıyla çoğunluk verir.
- window: her anahtarın son 6 gözlemi; toplam en çok 96 kayıt. Anahtar başına
  kota hazırdır, öğrenilmiş kategori değildir.
- last: her anahtarın son etiketi.
- decay: her gözlemde bütün sayıların 0,99 ile çarpılması, sonra yeni gözlem.
- revision_delta: kimlikli kaydı ve sayıları tutar, eski katkıyı çıkarıp yeniyi
  ekler. revision_rebuild aynı güncel kimlikli kayıtları her sorguda yeniden
  sayar. revision_append eski katkıyı silmeden düzeltmeyi ekler;
  revision_ignore düzeltme mesajını işlemeden ilk gözlemlerin sayılarını tutar.

Tahmin çoğunluk etiketi 0 veya 1, beraberlikte 0,5'tir. Birincil hata metriği
gizli gürültüsüz etikete karşı bütün yaşamın ortalama karesel kaybıdır [0,1].
Bu değer kalibre edilmiş olasılık yorumu gerektirmez. Yanlış kesin cevaplar,
beraberlikler, birinci/ikinci evre ve değişen/değişmeyen anahtar grupları ayrı
kaydedilir. Düzeltme koşulunda grup ilk 6 ve kalan 10 anahtardır. Ani değişimde
çift numaralı 8 anahtar değişir; kademeli değişimde 16 anahtarın hepsi değişir.
Durağan, bloklu ve seyrek koşullarda değişen anahtar grubu boştur.
MEM11'in birincil metriği adım başına sanal işlem maliyetidir; eşit
kayıp ayrıca zorunlu kontrol edilir. Çalışan kodun üst sınırı 900 işlem/adımdır.

Okuma, karşılaştırma, sayı güncelleme, dizi kaydırma ve RNG çağrıları kodda
belirtilmiş sanal adımlarla ücretlenir. Bunlar Python çalışma süresi değildir.
Gözlem edinimi tüm yöntemlere aynı olay sayısında verilir; bu dış maliyet
tabloya ayrıca olay sayısı olarak yazılır, iç işlem sayısına keyfi fiyatla
eklenmez. Bellek atomik alan sayısıdır; bayt veya eşit bellek bütçesi değildir.
Rezervuarın RNG durumu, kimlikli düzeltmenin kayıt belleği ve anahtar kotası
dahildir. Kod, yorumlayıcı ve performans günlükleri bu tanımda hariçtir.
Sayıların büyüyen bit uzunluğu maliyetlendirilmez. Hata üstünlüğü kaynak
üstünlüğü şeklinde yorumlanamaz; maliyet ve bellek ayrıca raporlanmalıdır.

## Yanlışlama ve sınırlar

Katalogdaki on iki karşılaştırma, belirtilen koşullarda daha düşük yaşam kaybı
(MEM11'de aynı cevapla daha düşük maliyet) öngörür. Sıfır veya ters farklar
saklanır. Katalog pratik eşiği ve kök orkestratörün çoklu test düzeltmesi
kullanılır. 24 keşif ve 48 dokunulmamış doğrulama tohumu arasında kod, eşik veya
hipotez değiştirilmez. Bu sayılar 72 gerçek dünya veya 72 bağımsız ilke değildir.
Bu kol tek başına yeni matematik, biyolojik model veya genel yaşayan öğrenme
çözümü olarak sunulamaz. Bilinen istatistik, rezervuar örnekleme, kayan pencere,
unutma ve artımlı görünüm güncelleme aileleriyle doğrudan kesişir.

Özdenetim, bağımsız skaler sayım, üstel ağırlıkların kapalı biçim hesabı,
kimlikli düzeltme sonrası yeniden sayım, düzeltmeyi kapatma müdahalesi ve JSON
durumundan aynı sonraki çıktılar/işlemlerle devamı kapsar. Rastgele örneklem
kalitesini özdenetim sonuçlarından seçmek veya ayarlamak yasaktır.
