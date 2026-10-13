# Yaşarken öğrenme için çok yönlü deney kampanyası: ilk dalga

**Muhammed Yasin Yılmaz**

CEV-2026-10 · Sürüm 2026.10.13 · Yayın tarihi 2026-10-13

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

Dört deney motorunda bellek, temsil, güncelleme ve eylemle deneyim üretimi karşılaştırıldı. On aileye yayılan 200 katalog kaydının 48 koşullu karşılaştırması, 24 keşif ve 48 doğrulama tohumu ile 3.456 eşleştirilmiş sonuç üretti. 30 iddia desteklendi, 9 ters sonuç verdi ve 9 belirsiz kaldı. Güçlü kontroller, öğrenilen temsil ve güncelleme davranışının dağılım değişiminde sınırlı ve bazen zararlı olabildiğini gösterdi.

## Abstract

A first broad campaign compares memory, representation, update behavior and action-produced experience in four local engines. Of 200 catalog entries across ten families, 48 conditional comparisons yielded 3,456 paired outcomes over discovery and untouched confirmation seeds. Thirty claims were supported, nine contradicted and nine inconclusive under the preregistered synthetic decision rule. Strong controls show that learned representation and update behavior can be conditional and harmful under shift; the general living-learning problem remains open.

## Bulguların yorumu

The campaign maps conditional mechanisms and negative controls rather than discovering a general computational principle. Experience can change later computation in these supplied languages, but a stronger pre-supplied representation or a shifted environment can remove the apparent advantage.

The worlds, languages, actions, metrics and candidate families are finite and researcher supplied. Catalog entries overlap and are not 200 independent mechanisms. The results are local synthetic evidence, not external peer review, biological validation, AGI evidence or proof of novelty.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/living_learning_campaign_2026_09_24/REPORT_TR.md](../../research/living_learning_campaign_2026_09_24/REPORT_TR.md) olarak korunur. Bağlantı yolları bu
edisyonun konumuna uyarlanmıştır. Rapordaki çalışma tarihi ve geçmiş yayın durumu
ifadeleri tarihsel kayda aittir; bu edisyonun tarihi yukarıdadır.

Bu çalışma [yayın dizisi](../README.md) içindeki ayrı bir metindir. DOI, bütün
metinleri, teknik kitabı ve kodu içeren sürüm arşivini tanımlar; her makaleye ayrı
DOI verildiği anlamına gelmez. Atıfta çalışma kimliği ve sürüm DOI'si birlikte
kullanılmalıdır. Hesaplamalı denetimlerin kapsamı [doğrulama kaydında](../evidence/validation.json)
ve [yayın ilkelerinde](../PUBLISHING.md) açıklanır.

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, hesaplamalı deney,
literatür incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır.
Bu katkı açıklaması, DOI kaydı veya otomatik testler dış bilimsel hakemlik sayılmaz.

---

## Araştırma kaydı: Tur 23 — geniş deney kampanyasının ilk dalgası

Muhammed Yasin Yılmaz · 24 Eylül 2026 · Yerel araştırma kaydı

## Sonuç ve kapsam

Çalışan sistemin deneyim nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir biçimde değiştirmesi ana problem olmaya devam ediyor. Bu turda tek bir küçük düzeneğin tekrarları yerine dört farklı deney motoru kuruldu ve toplu çalıştırıldı. **Genel living-learning ilkesi bulunmuş veya kanıtlanmış değildir.** İlk dalga, bilinen mekanizmaların nerede işe yaradığını ve nerede yetersiz kaldığını ölçen bir başlangıç taramasıdır.

200 kayıtlık katalog on konu ailesini kapsıyor: bellek, deneyimin yeniden işlenmesi, kaynak seçimi, temsil, alanlar arası aktarım, yordam dili, güncelleme, gecikmiş sonuçların önceki işlemlere bağlanması, bağlam ve eylemle deneyim üretme. **48 koşullu karşılaştırma yürütüldü; 152 önerme henüz sınanmadı.** Katalog girdileri konu bakımından örtüşebilir. Bunlar 200 benzersiz mekanizma veya 200 yeni bilimsel buluş değildir.

| Kol | Ne değişiyor? | Çalıştırılan iddia | Destek | Ters sonuç | Belirsiz |
|---|---|---:|---:|---:|---:|
| MEM | Deneyimin saklanması, unutulması ve düzeltilmesi | 12 | 12 | 0 | 0 |
| REP | Sonraki tahminde kullanılacak özellikler | 12 | 8 | 4 | 0 |
| UPD | Yeni gözlemin canlı durumu değiştirme katsayısı | 12 | 5 | 3 | 4 |
| ENV | Hangi eylemin hangi yeni gözlemi üreteceği | 12 | 5 | 2 | 5 |
| Toplam | Dört ayrı motor | 48 | 30 | 9 | 9 |

Her motorda 24 keşif ve 48 doğrulama tohumu çalıştırıldı: **288 motor-yaşamı ve 3.456 eşleştirilmiş karşılaştırma sonucu**. Bir motor-yaşamı birden çok rejim ve yöntem koşusu içerir. Aynı yaşamı paylaşan karşılaştırmalar bağımsız veri sayılmadı. Yazılım kontrolü sayısı deney veya hipotez sayısına eklenmedi.

## Nasıl sınandı?

Öngörüler, rakipler, dünya üreticileri, pratik eşikler ve kaynak sayaçları sonuçlardan önce yerel dosyalarda tanımlandı. 17 kaynak/katalog/protokol dosyası SHA-256 ile donduruldu. Bu kamuya açık bir önkayıt değildir; yerel çalıştırma sırasını ve içerik bütünlüğünü belgeler. Keşiften doğrulamaya geçerken hiçbir yöntem veya eşik ayarlanmadı.

Birincil karar yalnız 48 doğrulama tohumundan gelir. Her tohumda Δ = kontrol − aday; küçük ölçüt daha iyidir. Eşleştirilmiş işaret testinin iki yönü, 48 iddia için toplam 96 testlik ailede Holm düzeltmesi aldı. Destek için düzeltilmiş p≤0,05, ortalama farkın önceden belirlenen pratik eşiği aşması ve bütün doğrulama tohumlarının en az üçte ikisinde olumlu fark birlikte arandı. Ters sonuç simetrik kuralla belirlendi. Diğer sonuçlar belirsizdir.

İşaret testi ortalama kazancı veya pratik marjın aşılmasını sınamaz; fark yönünün sıklığını sınar. Pratik marj betimsel ek filtredir. Çok küçük p değerleri dünya yasasının gerçekliğini, yöntemin yeniliğini veya genel zekâyı kanıtlamaz. İstatistik yöntemi ve bilinen algoritma aileleri [birincil kaynaklarla](../../research/living_learning_campaign_2026_09_24/SOURCES_TR.md) konumlandırıldı.

## 1. Bellek: saklamak, unutmak ve geri çekmek farklı işler

Durağan ve gürültülü etiketlerde bütün geçmişi anahtar başına sayımlarla biriktirmek 0,02299 yaşam kaybı verdi. Son etiketi tutmak 0,24871; sınırlı rezervuar 0,09239 verdi. Basit bir yeterli özet, bu dünyada daha çok ham kayıt saklamaktan yararlı olabiliyor. Anahtarlar ve etiket dili hazır olduğu için bu sonuç kavram keşfi değildir.

Anahtarların yarısı aniden değiştiğinde son altı gözlemi anahtar başına tutan pencere 0,10151, birikimli sayım 0,23188 kayıp verdi. Aynı birikimli sayım durağan dünyada bu pencereyi geçiyor: 0,02299 karşısında 0,07859. Dolayısıyla burada ne sınırsız birikim ne de sürekli unutma bütün koşulların kazananıdır. Bu, yalnız sınanan yöntem ve akışların karşılaştırmasıdır.

Kimliği verilmiş eski gözlemi düzeltirken eski katkıyı çıkarıp yeni katkıyı eklemek, düzeltmeyi yeni bir bağımsız olay gibi eklemekten daha iyi çıktı: 0,25800 karşısında 0,30690. Tam güncel kayıtları her sorguda yeniden sayan güçlü kontrolle **aynı cevap dizisi** elde edildi. Artımlı yöntemin toplam sanal işi 7.680, yeniden saymanın 317.056: yaklaşık 41,3 kat fark. Bu fark gerçek işlemci hızlanması değildir; bildirilen sayaç ve tekrar eden sorgu iş yükü içindir. Düzeltme etiketleri dışarıdan doğru olarak verildi; sistem doğruluğu kendisi keşfetmedi.

## 2. Temsil: doğru ayrımı eklemek yararlı, en iyi çözüm olmak ayrı

İkili etkileşimli dünyada hata izinden özellik ekleyen adayın ortalama karesel hatası **0,22858**, yalnız doğrusal özellik kullanan kontrolün **1,16095** oldu. Rastgele özellik ekleyen kontrol 0,76208 verdi. Deneyimle seçilen ayrımın daha sonraki tahminleri etkilediği ölçüldü.

Ancak aynı 11 özelliği baştan kullanan güçlü kontrol **0,09332** ile daha iyi. Seçici yöntemin üstünlüğü iddiası bu karşılaştırmada ters sonuç verdi; 48 doğrulama tohumunun tamamında tam kontrol daha düşük hata verdi. Etkileşimin değiştiği, sonradan başladığı ve girdi ilişkilerinin değiştiği diğer karşılaştırmalarda da tam kontrol avantajlı kaldı. Buna karşılık salt doğrusal ve gürültülü doğrusal dünyalarda seçici alt küme daha iyi çıktı.

Kaynaklar başka bir tablo gösteriyor: sürekli etkileşimde seçici aday ortalama 72.271, tam kontrol 160.662 sanal işlem kullandı. Aday daha az işlemle daha yüksek hata üretiyor; tek başına üstün yöntem değildir. Ayrıca adayın örnek arşiviyle sayılan belleği ortalama **1.009 sayısal hücre**, tam kontrolünki **132**. Az etkin özellik, az toplam bellek anlamına gelmedi.

Yeni özelliği gözlenmiş geçmişle yeniden kurmak, aynı özelliğe yalnız sıfır geçmişle başlama ablasyonunu geçti: 0,22858 karşısında 0,26545. Toplam hesap da bu akışta daha düşüktü; çünkü müdahale sonraki edinim kararlarını da değiştirdi. Bu sonuç yeniden işlemenin yerel adım maliyetinin düşük olduğunu göstermez. Hazır özellik bankası dışındaki bir işlem keşfedilmedi.

## 3. Güncelleme: öğrenilen katsayı aktarılabiliyor, fakat yanlış ortamda zarar verebiliyor

Ortak canlı durumdan alternatif güncellemeler üretip gözlenmiş hataya göre kazanç seçen aday, düşük gürültülü durağan, sıçramalı, yön değiştiren ve karışık koşullarda sabit katsayı kontrolünü geçti. Yüksek gürültülü durağan ortamda ise kayıp **0,003244**, sabit kontrolünki **0,001214** oldu: aday daha kötü.

Her kazanç için ayrı durum tutan çevrimiçi filtre bankası daha güçlü bir karşılaştırmaydı. Aday sürüklenmede bankayı geçti (0,000526 / 0,001280), yüksek gürültüde bankaya yenildi (0,003244 / 0,002058). Sıçramalarda yön çoğunlukla olumlu olsa da önceden belirlenmiş pratik fark eşiğine ulaşılmadı; karışık ortamda da destek çıkmadı. “Kendi güncellemesini değiştirmek her zaman daha iyidir” sonucuna ulaşılamaz.

Aktarım deneyinde eski canlı tahmin sıfırlandı; yalnız edinilmiş katsayı yeni göreve taşındı. Benzer sürüklenme hedefinde küçük kazanç vardı, fakat bütün yaşam farkı **0,000222**, önkayıt eşiği **0,0005** altında kaldı. Bu nedenle UPD11 belirsizdir; 48/48 olumlu işaret tek başına yeterli sayılmadı.

Uyumsuz, yüksek gürültülü durağan hedefe aktarımda yalnız hedef evresindeki hata **0,005236**, sabit kontrolünki **0,002921** oldu. Edinim evresindeki yarar bu zararı kapatmadı; bütün yaşam iddiası ters sonuç verdi. Edinim maliyeti de görünür: aday 28.444, kontrol 1.282 sanal işlem harcadı; hedef evresinde ikisi de 1.282. Öğrenilmiş bir güncelleme katsayısının kalıcı olması, her yeni ortamda yararlı olacağını garanti etmiyor.

## 4. Eylem: ortalama kazanç ile çoğu yaşamda kazanç aynı şey değil

Tekrarlı terslenen iki-eylemli dünyada belirsizlik bonuslu yöntem 0,11353, yalnız mevcut en iyi görünen eylemi seçen yöntem 0,30112 ortalama fırsat kaybı verdi. Yakın geçmişli yöntem bu koşulda 0,06638'e indi, fakat daha çok hesap kullandı. Durağan dünyalarda aynı pencere yöntemi tam geçmişli yönteme yenildi.

Durağan büyük fark koşulunda özellikle öğretici bir ayrışma var. Belirsizlik bonuslu yöntem ortalamada çok daha iyi: **0,00562 / 0,05127**. Buna rağmen yalnız **4/48** yaşamda daha iyi; kalan 44 yaşamda küçük bir ek kayıp taşıyor. Greedy'nin bazı yaşamlarda yanlış eyleme uzun süre bağlanması büyük ortalama zararı üretiyor. Önkayıtlı çoğunluk ve ortalama koşulları zıt yönlere baktığı için sonuç **belirsiz** olarak kaldı. Bu yöntem işe yaramıyor demek değildir; burada “başarı”nın hangi yaşam dağılımı özelliğiyle tanımlandığı önemlidir. Sonucu destek göstermek için ölçüt sonradan değiştirilmedi.

## Önceki araştırmayla bağlantı

[Önceki eşleme turunda](../../research/living_learning_crossmap_2026_09_23/REPORT_TR.md) açık köprü deneyimi bileşik cevapları mümkün kılıyor, fakat kompakt hazır çıkarım yordamı daha pahalı aday saklamayı geçiyordu. Bu turdaki tam özellikli kontrol benzer bir denetim sağlıyor: öğrenilmiş parçanın basit rakibi geçmesi, güçlü hazır yordamı geçtiği anlamına gelmiyor.

[Tümdengelim turundaki](../../research/living_learning_deduction_2026_09_23/REPORT_TR.md) bağımlılık ve kalıcı hesap sonucu sorusu burada kimlikli düzeltme karşılaştırmasıyla farklı bir görevde sınandı. Eski sonuçlar geri alınmadı veya bunlarla değiştirilmedi. Aynı kavramların başka koşullarda sınanması özgün ilke keşfi diye sayılmadı.

## Ana problemde açık kalanlar

Bu dört motor birleşik, sürekli çalışan bir genel öğrenici değildir. Temsil dili, etiketler, eylemler, aday güncelleme ailesi ve değerlendirme ölçütleri araştırmacı tarafından verilmiştir. Gerçek dil, algı, açık uçlu görevler, çok uzun ömür, fiziksel beden veya bilinç sınanmadı. Kullanıcının gelişen kavramlar, bağlam kümeleri ve çapraz bağlar önerileri katalogda korunuyor; bu ilk dalga onların tamamını uygulamadı.

Öncelikli sonraki iş, 152 kaydın tamamını hemen çalışmış gibi göstermek yerine farklı müdahaleler gerektiren motorları kurmaktır. Özellikle hazır özellik dili dışındaki yeni ilişkiler (REP16/LAN03), aynı mevcut cevaplara rağmen farklı sonraki öğrenme kapasitesi (REP19/CTX20), bağlamı bölme/birleştirme (CTX03/CTX04), gecikmiş sonucun doğru geçmişe bağlanması (CRD05/CRD10), kesintiler arasında deneyimin yeniden işlenmesi (CON02/CON10) ve hazır kimlik köprüsü olmadan aktarım (XFR01) henüz açıktır. Sonraki dalga bunlar için yeni önkayıt ve dokunulmamış tohumlar gerektirir.

## Denetim ve dosyalar

28.811 önkontrol; kapalı biçim hesapları, bağımsız sayım/normal denklem referansları, bileşen kapatma, kaydedilmiş durumdan devam ve istatistik örneklerini kapsıyor. Bunlar 28.811 bilimsel deney değildir. **288 motor-yaşamının tamamı yeniden hesaplandı ve ilk kayıtlarla birebir eşleşti.** Bu aynı uygulamanın tekrarlanabilirliğidir, bağımsız laboratuvar tekrarı değildir.

Kaynak hash'leri, tüm kayıtların kapsanması, sonlu ve sınırlar içindeki ölçütler, özetin ham veriden yeniden hesaplanması ve önceki **1.114 dosyanın** korunması doğrulandı. Git HEAD değişmedi; commit, push, yayın, kitap veya ana uygulama değişikliği yapılmadı.

- [200 maddelik katalog](../../research/living_learning_campaign_2026_09_24/CATALOG_TR.md)
- [Sonuç öncesi protokol](../../research/living_learning_campaign_2026_09_24/PROTOCOL_TR.md)
- [48 karşılaştırmanın tamamı ve kaynak tablosu](../../research/living_learning_campaign_2026_09_24/RESULTS_TR.md)
- [Birincil kaynaklar ve özgünlük sınırı](../../research/living_learning_campaign_2026_09_24/SOURCES_TR.md)
- [Makinece okunur sonuç özeti](../../../research/living_learning_campaign_2026_09_24/results/summary.json)
- [Tekrar ve koruma denetimi](../../../research/living_learning_campaign_2026_09_24/results/verification.json)
- [Çalıştırma yönergeleri](../../../research/living_learning_campaign_2026_09_24/README.md)
