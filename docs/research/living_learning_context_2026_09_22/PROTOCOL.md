# Tur 14 — Hazır bağlam etiketinden gözlenebilir geçmiş ilişkisine

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel sonuç-öncesi protokol

Ana başarım sonuçları hesaplanmadan yazıldı. Dış ön kayıt değildir. Tur 13'ün bağlam kimliğini vermesiyle açık bıraktığı H24'ün sınırlı bir parçasını sınar. Önceki 941 dosya, kitap ve HEAD korunur; commit/push/yayın yok. Genel yaşam-boyu öğrenme problemi bu sonlu deneye indirgenmez.

## Araştırma sorusu

Sistem, farklı iş kurallarını hangi gözlenebilir geçmiş durumunda kullanacağını deneyimden öğrenebilir mi? Bu ilerleme, yalnız görünür bir yüzey ipucunu izlemekten ve araştırmacının doğru bağlam kimliğini vermesinden ayrılabilir mi?

H42: Gizli bağlam etiketine erişmeden, gözlenen geçmiş ile sonuç arasındaki ilişkiyi ve bağlama bağlı iş kurallarını birlikte edinmek bu aday dilinde mümkün olabilir.

H43: Bütün edinim boyunca geçmiş ilişkisi ve yüzey ipucu örtüşüyorsa, bu deneyim doğru açıklamayı tek başına belirlemeyebilir. Başarılı eğitim tahmini bağlamın gerçek kaynağını keşfetmek sayılmaz.

H44: İki ipucunu ayıran yeni deneyimler, eski kayıtlar dururken bile kullanım ilişkisinin kalıcı biçimde düzelmesine katkıda bulunabilir. Çelişkili yeni etiketler yalnız ezberlenmek yerine daha önce görülmemiş girdilere taşınmalıdır.

H45: Aynı aday işlevleri kapsayan iki modelin başlangıç ağırlıkları, belirsiz dönemdeki genellemesini değiştirebilir. Temsil sayısı ile farklı işlev sayısı aynı değildir.

## Dünya ve verilen yapı

Gözlenen girdi `(h,s,x)`:

- h: hazır geçmiş özeti olarak yorumlanan dört gözlenebilir bit. Bu dört öğe araştırmacı tarafından girdi olarak verilir; örnekler arasında kayan pencere tutarlılığı kurulmaz. Ham zaman dizisinden bu pencerenin seçilmesi veya oluşturulması öğrenilmez.
- s: tek yüzey ipucu biti.
- x: mevcut iş girdisinin dört biti.

Toplam 16×2×16 = 512 farklı gözlenen girdi. Gerçek bağlam `c=g(h)`; g, h'nin seçilen iki veya üç bitinin paritesidir. Kullanılacak iş kuralı `f_c(x)`. Her uzman x üzerinde bir veya iki bitli affine paritedir. f0 ve f1 birbirinden ve birbirinin tümleyeninden farklıdır; bütün x uzayının yarısında farklı sonuç verirler. Hangi bitlerin seçildiği ve çıktı tersleme bitleri yaşamlar arasında değişir.

Öğreniciye g veya c verilmez. Her gözlem yalnız `(h,s,x,y)` girdisini ve etiketini verir. Ancak aşağıdaki sonlu model dili hazırdır: h üzerinde 15 boş olmayan homojen parite kapısı; ayrıca s kapısı; her dal için x üzerinde 32 affine parite. Kapı ve uzman birlikte seçilir. Bu dilin icadı, yeni nöron üretimi veya bilinç modeli iddia edilmez. h=0 için tüm geçmiş kapıları 0 verir; kapı yönü bu konvansiyonla sabittir. Uzmanlara ayrıca sıralama kısıtı konmaz.

16×32² = 16.384 parametre gösterimi, 15.904 farklı bütün-girdi işlevi oluşturur. f0=f1 olan 32 işlev 16 kapıda yinelenir; başlangıç ön olasılığı parametre gösterimleri üzerindedir. İşlevler üzerinde tekdüze olduğu söylenmez.

## Edinim ve üç devam rejimi

24 yaşam, tohumlar 0–23. Her yaşamın ilk 48 farklı girdisinde `s=g(h)`; geçmiş ve yüzey kapıları aynı uzmanlarla aynı doğru etiketleri verir. İlk bölüm bütün yöntem/rejimler arasında aynıdır. Öğreniciye bölüm sınırı veya rejim kimliği verilmez.

Sonraki 16 yeni ve farklı girdi:

1. `aligned`: s her zaman g(h) ile aynı kalır.
2. `balanced`: sekiz örnekte aynı, sekizinde ters; sıra karıştırılır.
3. `reversed`: s her zaman g(h)'nin tersidir.

Gerçek y her rejimde geçmiş kapısından üretilir; gerçek görev değişmez. Buradaki müdahale, hangi girdilerin gözlendiğini değiştirir. Etiketler gürültüsüzdür. İlk 48 girdi ve sonraki her 16 girdi tekrar içermez; gözlenen ham kayıtlar öğrenici durumunda tutulmaz.

## Yedi yöntem

1. `full_table`: her tam `(h,s,x)` anahtarında 0/1 gözlem sayısı; görülmemiş anahtarda 1/2, görülmüşte ampirik etiket ortalaması. Yeni birleşimlere yapısal transfer yapmayan bellek kontrolü. Sayaçlar deneyimin sıkıştırılmış izidir; bellek yokmuş gibi sunulmaz.
2. `pooled_x`: aynı sayaç yordamı yalnız x anahtarında; bütün gözlenen bağlamları karıştırma kontrolü.
3. `surface_only`: kapı s olarak verilmiş, 32² uzman çifti deneyimle ağırlıklandırılır.
4. `history_only`: 15 geçmiş kapısı ve uzman çiftleri birlikte ağırlıklandırılır. Doğru kapı verilmez; yüzey ipucunun ilgili olmadığı önbilgisi verilmiştir.
5. `joint_uniform_gate`: 16 kapıya eşit başlangıç ağırlığı, kapı içinde 32² çifte eşit ağırlık. Geçmiş/yüzey ailesi toplam ağırlıkları 15/16 ve 1/16.
6. `joint_balanced_family`: geçmiş ve yüzey ailelerine 1/2; geçmiş içindeki 15 kapıya eşit, bütün kapılarda uzman çiftlerine eşit ağırlık. Aynı 16.384 gösterim ve aynı işlevleri kapsar; yalnız ön olasılık farklıdır.
7. `oracle_gate`: doğru geçmiş kapısı araştırmacı tarafından verilmiş; iki uzman yine yalnız gözlemlerden öğrenilir. Bu ayrıcalıklı ölçüt ana öğrenici başarısı veya uygulanabilir yöntem diye sunulmaz.

Finite aday yöntemlerinde her yeni etiket için aday uyuşuyorsa 0,95, uyuşmuyorsa 0,05 olabilirlik çarpanı kullanılır. 0,05 ilk çalıştırma öncesinde sabitlendi; sonuçlara göre ayarlanmaz. Bu, yöntemin varsaydığı etiket bozulma modelidir; gerçek veri gürültüsüz olduğundan model varsayımıyla dünya aynı değildir. Boş aday kümesi ve keyfî geri dönüş yordamı yerine bütün yöntemler aynı pozitif olabilirliği kullanır. Sayısal kararlılık için tamsayı hata sayıları ve bunlardan hesaplanan log ağırlıklar tutulur.

Değerlendirme temiz doğru işlevi sorgular: tahmin `Σ_h q(h|D) f_h(input)`; öngörüye ayrıca 0,05 etiket gürültüsü eklenmez. Aday ağırlıkları ve değerlendirme etiketi birbirine karıştırılmaz. Table yöntemleriyle aynı ön olasılık iddia edilmez. İki joint yönteminin karşılaştırması aynı bilgi, aynı aday dizisi ve aynı işlem sayısında yalnız ön olasılık değişimidir.

## Ölçüler ve maliyet

24×3×7 = 504 edinim yolu. Her etiket gelmeden tahmin ve hata kaydedilir. Durum noktaları 0,12,24,48,49,52,56,64 gözlemdir; 48→64 aralığı iki ipucunun ayrıştırılması için önemlidir. Noktalar bağımsız örnekler değildir.

Gizli doğru etiketlerle bütün 512 girdi, s=g(h) olan 256 girdi, s≠g(h) olan 256 girdi ve henüz gözlenmemiş girdiler ayrı değerlendirilir. 48. noktada bütün ters-ipucu girdileri eğitim dışıdır. 64. noktada tüm-küp ölçüsü görülmüş girdileri de içerir; ayrı görülmemiş metriği raporlanacak. Ana hata `(p−y)²`; eşitlikte beklenen sınıflama doğruluğu da ölçülür.

Asıl ölçüler: ilk 48'den sonra ve son 64'ten sonra ters-ipucu ve görülmemiş girdi hatası; ilk 48 ve sonraki 16 etiket öncesi ortalama hata; doğru geçmiş kapısının ve yüzey kapısının toplam ağırlığı; etkin aday sayısı. Doğru kapının kimliği yalnız değerlendirici içindir. Düşük görev hatası doğrudan benzersiz iç bağlam kimliği keşfi sayılmaz.

Maliyet: her yeni etikette hesaplanan aday uyuşmazlıkları, ağırlık güncelleme sayısı, aday sayısı, kalıcı sayaç/dizi ve JSON boyutu. Paylaşılan hazır 512-girdi tahmin bankası ayrı boyutla bildirilir; learner JSON boyutuna dahil değil. Değerlendirme tam-küp çarpımları eğitim maliyeti diye sayılmaz. Bu sayaçlar toplam FLOP/CPU eşitliği değildir.

## Doğrulama ve yanlışlanma koşulları

- Skaler bit işlemleriyle bütün kapı/uzman/tüm-girdi çıktıları bağımsız doğrulanacak.
- İlk 48'de gerçek geçmiş ve yüzey açıklamalarının uyuşmazlık sayısı eşit olmalı; posterior oranları sırf bu veriyle değişmemeli. Sonraki örnekler ayrıştırabilir.
- Her rejimde ilk 48 kaydın ve yöntemlerin girdilerinin eşitliği, örnek tekrarının yokluğu denetlenecek.
- Uzmanlar farklı/tümleyen değil; tam x uzayında 1/2 anlaşmazlık koşulu doğrulanacak.
- Gizli doğru kapı/öğretici kimliği ana öğrenici arayüzünde bulunmayacak. Oracle ayrıcalığı açık ayrı yöntemde kalacak; ortak hazır aday bankasının doğru kapıya ait aralığını kullanır.
- Tam kayıtlar olmadan ayrı süreçte yeniden başlatıp kalan gözlemlerle kesintisiz nesneye aynı güncelleme sağlanmalı. Durum dışa aktarma/yüklemede nesne listeleri paylaşılmayacak.
- İki joint modelin aday bankaları aynı olmalı; başlangıç ön olasılıkları ilgili tanıma uymalı.
- Birinci 48 gözlemin aynı olduğu iki farklı dünya ve ters-ipucundaki farklı doğrular, genel yanlışsız belirleme iddiasına karşı örnek olarak ayrıca kurulacak.
- Ana sonuçlar yeniden çalıştırılacak; önceki dosyalar korunacak.

Başarı, hazır bir sonlu dilde hangi gözlenebilir bilginin hangi iş kuralına bağlanacağını öğrenme düzeyindedir. Ham geçmişten hangi özelliğin çıkarılacağı, ilişki dilinin oluşturulması, öğrenicinin kendi ayrıştırıcı deneyimini seçmesi, nedensellik, duygular ve bilinç bu turda çözülmez. İlk karışık veriden doğru açıklamayı garantiyle bulamamak kullanıcının bütün hipotezini çürütmez; hangi ilave deneyimin gerektiğini sınırlar.
