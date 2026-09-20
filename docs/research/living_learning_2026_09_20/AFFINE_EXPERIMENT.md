# Yaşam döngüsü, deneyimlerin birleşmesi ve içsel derleme deneyi

20 Eylül 2026. Standart Python ile tamamlanan sonlu, gürültüsüz bir yapıcı örnek. Özgün yöntem veya ana Cevahir uygulaması değildir. [Kod](../../../research/living_learning/affine_lifecycle.py), [ham sonuçlar](../../../research/living_learning/results/affine_lifecycle.json), [matematiksel gerekçe](FOUNDATIONS.md).

## Protokol

Her biri `0,…,24` seed'iyle belirlenen 25 tekrar; her tekrarda iki farklı dünya kuruldu. Her dünyada dört opak araç kimliği ve `fᵢ(x)=aᵢx+bᵢ mod 101` fonksiyonu vardır. Katsayılar rastgele seçilir, öğreniciye verilmez. Dünya ailesi, asal sayı, araç kimlikleri ve genel bileşim işlemi araştırmacı tarafından sağlanır. Bu güçlü başlangıç bilgisi rapordan çıkarılmamalıdır.

Başlangıçta öğrenilmiş kural yoktur. `x=0,1,2` için dört aracın sonuçları sırayla, iç içe geçirilerek alınır; toplam 12 doğru geri bildirim. Öğrenici her olayda etiketi görmeden önce tahmin etmeye çalışır, sonra durumu günceller. İki farklı örnekten katsayı çıkarır, üçüncü farklı örnek tutarlıysa kuralı kullanıma alır. Örnekleri öğrenilmiş durumdan çıkarır. Gelecek davranış dört katsayı çiftiyle yürür. Üçüncü örnek yalnız sonlu tutarlılık kontrolüdür, dünya ailesinin doğru olduğunu kanıtlamaz.

Değerlendirme girdileri `3,…,100`dür. Dört tekli araç ve 16 sıralı ikili bileşim için toplam `20×98=1.960` sorgu vardır. Eğitim sırasında hiçbir ikili bileşim çıktısı verilmemiştir. Bileşim semantiği öğrenilmez; öğrenilmiş işlemler bu hazır semantik altında yeni sorgularda birleştirilir. Bütün tahminler etiketlerle karşılaştırılmadan önce üretilir. Soyut aritmetik bütçesi iki çağrıyı da yürütmeye yeterlidir.

## Gerçekleşen sonuç ve güçlü kontroller

| Koşul | Her tekrardaki sonuç | Ne gösterir? |
|---|---:|---|
| A deneyimli sistem / A dünyası | 1.960/1.960 doğru | Verilen ailede yeni giriş ve bileşimlere aktarım |
| B deneyimli sistem / B dünyası | 1.960/1.960 doğru | Başka yaşam aynı mekanizmayla başka katsayıları edinir |
| Öğrenilmiş durum sıfırlanmış | 0 doğru, 1.960 çekimser | Hazır yürütücü tek başına görevi çözmez |
| Yalnız ham örneğe tam eşleşme | 0 doğru, 1.960 çekimser | Sonuç yalnız görülmüş `(araç,girdi)` çiftlerini geri getirmek değildir |
| Ham örnekler + aynı çıkarım yordamı | 1.960/1.960 doğru | Öğrenilmiş biçim güçlü bellek+çıkarım rakibinden bilgi açısından üstün değildir |
| A'nın öğrenilmiş durumunu temiz örneğe aktarma | A ile birebir aynı | Davranış farkı öğrenilmiş durumla birlikte taşınır |

25 tekrarda iki dünyanın her biri toplam **49.000** değerlendirme cevabını doğru verdi. Bu 49.000 bağımsız gerçek dünya deneyi değildir: küçük, kapalı bir cebir ailesinde tam giriş taramalarının toplamıdır. Asıl genelleme gerekçesi aile koşullu cebirsel ispat, sayısal çalışma ise uygulamanın buna uygunluğudur.

A ve B'nin aynı sorgudaki cevapları tekrar başına 1.940–1.943 yerde, toplam 48.523/49.000 sorguda farklıdır. Yalnız farklı dosyalar değil, geçmişe bağlı farklı yürütülen fonksiyonlar oluşmuştur. Bu fark tek başına iyilik ölçütü sayılmadı; her sistem kendi dünyasının gerçek etiketleriyle ayrıca değerlendirildi.

Son seed'deki öğrenilmiş durum ayrı bir Python işlemine yazılıp yüklenerek **1.960 sorguda** yeniden sınandı. Çocuk işlem yalnız kural katsayıları, sürümler, boş derlenmiş işlem listesi ve etiketsiz sorguları aldı. Ham olay listesi, bekleyen örnek ve olay izi yoktur; başlangıç/bitiş arasında birebir tahmin eşitliği sağlandı. Bu gerçek işlem yeniden başlatma denetimidir; 25 restart, elektrik kesintisine dayanıklı saklama veya uzun süreli kesintisiz servis denetimi değildir.

Güçlü ham-bellek rakibi, aynı 12 örnek üzerinde **değerlendirmeden hemen önce bir kez** aynı çıkarımı yapar; her sorguda baştan indüksiyon yapması zorunlu tutulmaz. Aynı cevapları bulduğu için öğrenilmiş katsayı biçimi ayrı bir bilgi kaynağı veya yeni temel öğrenme yöntemi olarak sunulamaz. Bu kontrol, yalnız zayıf bir retrieval rakibini yenerek başarı ilan etmeyi önler. Çıkarım maliyeti bu doğruluk karşılaştırmasında ölçülmedi.

## Dış geri bildirim olmadan yeniden düzenlenme

`seed=7001` ile dört araç öğrenildikten sonra 12 farklı sekiz-adımlı işlem dizisi önceki iş yükü olarak kaydedildi. Derleyici yalnız bu geçmiş işlem dizilerini ve öğrenilmiş kuralları görür; sonraki sorgu girişlerini veya etiketlerini görmez. Her sekizli birleşimin katsayılarını hesaplamak yedi soyut birleştirme işlemi ister. Toplam iç işlem **84** birimdir. Dört temel kuralın katsayıları bu sırada değişmez ve **sıfır** dış geri bildirim alınır.

Sonra aynı 12 işlem biçimi `3,…,100` yeni girişlerinde kullanılır: 1.176 sorgu. Tekrarlayan iş yükü derlemeye elverişli olacak şekilde seçilmiştir; gelecekte hangi dizilerin geleceğini tahmin etme problemi çözülmemiştir.

| Koşul | Cevap başına bütçe | Doğru / sorgu | Derleme + değerlendirme toplam işi |
|---|---:|---:|---:|
| Derlemeden sekiz adımı yorumla | 1 | 0/1.176; hepsi çekimser | Yeterli bütçe yok |
| İç dönemde derle, sonra uygula | 1 | 1.176/1.176 | 84+1.176 = **1.260** |
| Derlemeden sekiz adımı yorumla | 8 | 1.176/1.176 | **9.408** |
| İlk yeni istekte derle, tekrar kullan | İlkinde 8, sonra 1 | 1.176/1.176 | **1.260** |

Bir affine uygulama veya bir affine birleştirme bir soyut iş birimidir. Bunlar aynı CPU süresine sahip işlemler değildir. Sözlük erişimi, bağımlılık taraması, saklama, öğrenme ve önceki istekler sayılmaz; tabloda bütün yaşam FLOP'u veya ölçülmüş hızlanma iddiası yoktur. Derlemenin kendi işi sayıldığı için ücretsiz iç işlem yanılgısı önlenir; güçlü tembel derleyiciyle toplam işin **eşitliği** temel negatif kontroldür.

Burada iç işlem, bütçeye göre gerçekleştirilebilir davranışı değiştirmiştir. Bilgi avantajı veya uykuya özgü mekanizma yoktur. Yeni girişlerde çalıştığı için yalnız önceki tam cevapları önbellekten getirmek de değildir; öğrenilmiş fonksiyonların bilinen cebirsel derlemesidir. Bilgi zaten vardır, uygulama maliyeti değişmiştir. Bu deney uyku fazının gerekli olduğunu desteklemez.

## Düzeltme ve eski yararlı işlemlerin korunması

İlk aracın sabit terimi dünyada 1 artırıldı. Yeni doğru sonuç eski kuralı çürütünce kural hemen geri çekildi; o araca bağlı bütün derlenmiş diziler iptal edildi. Yeniden doğrulanana kadar ilgili isteklerde çekimser kalındı. `x=3,4,5` üzerindeki üç yeni doğru örnekten sonra yeni kural doğrulandı ve birleşimler yeniden derlendi. Diğer üç kuralın katsayıları bit düzeyinde aynı kaldı. Yeni dünyanın 1.176 bileşim sorgusunun tamamı doğru çıktı.

Burada bileşim çıktıları verilmemiştir; ancak düzeltmeden sonraki değerlendirmede kullanılan 3–5 girişleri atomik düzeltme örneklerinde de vardır. Dolayısıyla revizyon testinin her başlangıç girdisi görülmemiş diye sunulmuyor. Ana aktarım testinde 3–100 girişleri başlangıç eğitiminden tamamen ayrıdır.

Bu, tek bir doğru etiketli değişim olayında eksiksiz bağımlılıkla seçici geçersizleştirme örneğidir. Hangi kaynağın doğru olduğu, gürültü, gizli bağlam ve bütün becerilerin paylaşılan parametrelerle korunması çözülmemiştir. Değişmiş dünyanın eski kuralını kullanmayı bırakmak yararlı düzeltmedir; bu kuralın eski bağlamda da unutulmaması istenirse ek bağlam ayrımı gerekir.

## Olumlu sonucu bozan karşı örnekler

1. `g(x)=2x+3+x(x−1)(x−2) mod 101`, eğitimdeki bütün üç örnekte `2x+3` ile aynıdır. Öğrenici üçüncü kontrolü de geçer; buna rağmen `3,…,100` üzerinde **0/98** doğru sonuç üretir. Sonlu kontrol, doğru hipotez ailesinin yerini tutmaz.
2. Üçüncü örnek affine yapıyla açıkça çelişirse kural kabul edilmez. Görünür çelişkiyi tespit etmek, görünmeyen aile uyuşmazlığını tespit etmekten daha kolaydır.
3. Doğru öğrenilmiş `2x+3` kuralına `x=3` için yanlış ama güvenilir sayılan 10 sonucu verilince kural geri çekilir. **Bir hatalı etiket doğru yeteneği geçici olarak yok edebilir.** Gürültüye dayanıklılık iddiası elenir.
4. On iki kez yalnız `x=0,y=3` görülmesi eğimi belirlemez. İçsel derleme eksik eğimi tamamlayamaz. Ayırt edici deneyim gereksinimi hesapla ortadan kalkmaz.

## Yeniden üretim ve kapsam

Repository kökünde `python -m research.living_learning.affine_lifecycle`. Standart kütüphane dışında bağımlılık yüklenmez. İlk çalışma yaklaşık 0,61 saniye sürdü; süre sonuç JSON'unda yeniden ölçülür ve makineye bağlıdır. Sürecin toplam RAM tüketimi ölçülmedi. Kaynak SHA-256, seed'ler, denetimler ve olumsuz sonuçlar JSON'da korunur. Deney parametreleri ilk çalıştırmadan önce kodda belirlendi; yayımlanmış bir önkayıt yoktur. Sonradan yapılan açıklama düzeltmeleri araştırma günlüğünde tutulur.

Bu çalışma kapalı ailede öğrenilebilirlik, ham kayıt olmadan kalıcılık, bileşim, içsel derleme ve bir seçici düzeltmenin birlikte mümkün olduğunu gösterir. Doğal dil, yeni temsil ailesi, bilinmeyen görev sınırları, otonom aktif deney seçimi, sabit kaynakla sınırsız yaşam veya Cevahir'in bu becerileri edindiği sonucunu göstermez.
