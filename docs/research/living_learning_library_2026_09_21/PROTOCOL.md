# Deneyimden program parçaları edinme — sonuçlardan önceki yerel protokol

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma turu 11

Bu dosya ana deney çalıştırılmadan yazılmıştır; dış bir ön kayıt hizmeti değildir. Önceki araştırma ve kitap korunur, commit/push/yayın yapılmaz. Ana problem hâlâ deneyimin gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir değiştirmesidir. Bu tur yalnız **edinilmiş hesap parçalarının sonraki edinime etkisini** ayrıştırır.

## Önceki sonuç ve yanlışlanacak iddialar

Önceki temsil büyütme deneyi, çarpım özellikleri edinimini ve aynı etkinleştirmeli sabit geniş alternatifle tam eşitliği gösterdi. Bu negatif eşitlik geçerli kalır. Yeni deney, farklı programların ortak alt dizilerinden yararlanarak sınırlı arama bütçesinde başka programlar edinmeyi sınar.

1. Deneyimden çıkarılmış parçalar ilgili sonraki görevlerde edinimi kolaylaştırabilir; bunu bütün görevlerde garanti ettiği kabul edilmeyecek.
2. Ek kütüphane arama dallanmasını büyüttüğünden alakasız görevlerde maliyet veya başarısızlık artabilir. Artış ölçülmezse varmış gibi raporlanmayacak.
3. Parçaların primitive açılımlarını aynı sırada öneren sabit yorumlayıcı aynı davranışı verebilir. Bu güçlü eşitlik kontrolü zorunludur; kütüphane kazanımı yeni bir hesaplanabilirlik sınıfı diye yorumlanmayacak.
4. Tam programların tekrar kullanımı ve primitive sıklığına göre sıralama daha basit rakiplerdir; yalnız deneyimsiz rakiple kıyas yapılmayacak.

## Dünya ve önbilgi

Girdi ve çıktı `F_101`; tek girdili programlar dört bilinen işlemin dizileridir: `x+1`, `2x`, `x²`, `−x`, hepsi mod 101. Bir görev belirli bir dizinin girdi–çıktı ilişkisidir. Yorumlayıcı, arama, etiket anlamı ve dil başlangıçtan verilmiştir; sistem yeni primitive işleç icat etmez.

Her 12 yaşamda (`0..11`) iki üç-işlemlik ortak motif seçilir. Motifler, bütün 101 girdide semantiği uzunluğu iki veya daha az bir programla aynı olmayan farklı üçlüler arasından seçilir. Bu dünya üretim koşuludur, öğreniciye aday filtresi olarak verilmez. 12 ilk görev bu motifleri tek başına veya bir primitive ön/son ekle kullanır. Öğrenici yalnız sekiz arama örneği ve aday çıktıktan sonra sekiz ayrı kabul örneği alır. Bütün alanın kalan noktaları değerlendiricide tutulur.

İlk görevler aynı primitive-only aramayla, görev başına 32.768 primitive-vektör adımı sınırıyla edinilir. Kabul edilen programlar gelecekteki öğrenme durumu olarak saklanır. Öğretmen programları bunların yerine konmaz. Ortak edinim yolu, sonraki politika farklarının farklı ilk kanıtlardan kaynaklanmasını önler; tamamen serbest bir ömür boyu ajan deneyi değildir.

0, 6 ve 12 deneyimden sonraki durumlar kopyalanır. Her kopya üzerinde sekizer ilgili ve aile değiştirmiş yeni görev sınanır. İlgili görevler önceki iki motifin yeni bileşimleri; aile değişimindeki görevler aynı dilden farklı motiflerin aynı biçimde bileşimleridir. Yarısı iki üçlü motif, yarısı buna bir primitive ön/son ek içerir. İlk görevlerle ve birbirleriyle bütün-alan semantiği aynı olan hedefler yeniden seçilir. Bu ayrım öğretmenin üretim dağılımına aittir; öğrenici aile kimliğini veya motifleri görmez.

Her yeni görevin çözümü yeni sekiz etiketli örnekten aranır. Probe sonuçları geçmiş deneyim bankasına geri yazılmaz: bunlar aynı geçmişten gelecekteki edinimi ölçen müdahale kopyalarıdır. Görev sınırları ve geri bildirim ilişkilendirmesi dışarıdan verilmiştir.

## Deneyimden çıkarılan kütüphane

Kabul edilmiş primitive programlardaki uzunluğu 2 veya 3 olan bitişik parçalar sayılır. Bir parça her programda en çok bir kez sayılır. En az iki programda görülen parçalar `count·(length−1)−length` puanı ile sıralanır; yalnız pozitif puanlı en fazla dört parça alınır. Eşitlikler uzunluk ve leksikografik sırayla çözülür. Bu puan üst üste binen parçaların toplam sıkıştırma kazancının ispatı değildir; açık bir seçim sezgisidir. Araştırmacı motif kimliğini seçime vermez.

Kütüphane sonlu primitive dizileri saklar. Eski kabul edilmiş programlar primitive açılımlarıyla korunur; kütüphanenin yeniden sıralanması eski programların anlamını değiştirmez. Bu kolaylaştırma ve saklanan geçmişin boyutu raporlanacak. Genel sonlu bellek veya yanlış kanıttan kendini düzeltme çözümü iddia edilmeyecek.

## Sekiz yöntem

- `primitive_only`: sabit dört primitive, geçmişsiz arama sırası.
- `frequency_prior`: kabul edilmiş programlarda sık geçen primitive'leri önce önerir; yeni parça yok.
- `learned_library`: seçilmiş en fazla dört parçayı önce, sonra primitive'leri önerir.
- `expanded_proposals`: aynı seçilmiş parçaları yorumlayıcıya yeni işlem eklemeden açılmış primitive dizileri olarak, aynı arama önceliğiyle önerir. Anlam ve bütçe eşitliği denetlenir.
- `whole_program_cache`: en sık görülen en fazla dört kabul edilmiş tam programı önce önerir, sonra primitive'ler.
- `static_all_chunks`: bütün 2 ve 3 uzunluklu parçaları baştan içerir; hepsi aynı semantik tekrar elemesine tabidir. Bu belirli geniş-sabit arama, mümkün olan en iyi sabit arama değildir; aynı etkinleştirmeyi kullanan güçlü kontrol `expanded_proposals`dır.
- `random_library`: öğrenilmiş kütüphaneyle aynı parça uzunlukları ve sayısı, bağımsız rastgele parçalardan; etiket seçimi yok.
- `oracle_motifs`: dünyayı üreten iki motif baştan verilidir; ayrıcalıklı tanısal referans, adil öğrenilmiş rakip değildir.

Sıfır geçmişte öğrenilmiş kütüphane ve sıklık önceliği primitive-only ile aynı olmalıdır. Arama BFS ile ifade token sayısını izler. Her adayın sekiz arama girdisindeki çıktı imzası üzerinden tekrarlar elenir. Bir makronun yürütülmesi primitive açılımının tamamıyla ücretlendirilir; bedelsiz çalıştırma yoktur. İlk tutarlı aday bağımsız sekiz kabul örneğinde sınanır; başarısızsa bu görev reddedilir. Kabul örnekleri aramayı yönlendirmek için yeniden kullanılmaz. Bütün-alan değerlendirme öğrenmeye girmez.

## Kaynak sınırı ve sonuç sayıları

Ana bütçeler görev başına 512 ve 2.048 **primitive-vektör adımı**dır. Bir adım aynı sekiz örneğe bir primitive uygulanmasıdır; sekiz skaler işlem demektir. Adayın kabul sınamasındaki bütün açılmış program adımları da bu bütçeye dahildir. Arama kuyruğu, aday sayısı, tekrar elemeleri, maksimum kuyruk boyutu, kütüphane çıkarımı için taranan parça/primitive sayıları, saklanan JSON baytları ayrıca kaydedilir. Bu sınır CPU zamanı, toplam FLOP veya toplam RAM eşitliği değildir. Kütüphane ediniminin maliyeti gizlenmeyecek; önceki ortak eğitim maliyeti de ayrı verilecek.

12 yaşam × 3 geçmiş miktarı × 2 bütçe × 16 görev × 8 yöntem = **9.216 koşul**. Aynı görevler geçmiş miktarı ve yöntemler arasında eşleştirilir. Oracle geçmiş miktarından yararlanmadığı için tekrarlanan koşulları yeni bağımsız kanıt sayılmayacak.

Her koşulda kabul, bütçe tükenmesi, kabulden sonra bütün-alan doğruluğu, görülmemiş girdilerde doğruluk, arama ve kabul adımları kaydedilir. Sonuç tablosundaki ana başarı, bütün 101 girdide doğru program edinimidir; yalnız eğitim eşleşmesi değil. Eşleştirilmiş farklar ve yaşam bazındaki dağılımlar raporlanır. Gürültüsüz ve küçük sonlu dünya açık sınırlılıktır.

## Doğrulama ve değiştirmeme

Primitive yorumlayıcısı ve bileşim sırası doğrudan formüllerle bağımsız denetlenecek. BFS küçük tam taramayla karşılaştırılacak; learned/expanded eşitliği bütün koşullarda sınanacak. Öğreniciye test hedefleri verilmediği, görev geçmişi öğretmen kodlarından kurulmadığı ve kabul bütçesi aşılmadığı kontrol edilecek. Durum ayrı süreçte yüklenip aynı gelecekteki arama sonucu doğrulanacak. Kütüphane sıfırlama müdahalesi gelecekteki aramayı değiştirirken eski açılmış programların cevaplarını değiştirmemeli.

İlk kayıtlı tarama öncesi yalnız uygulama doğruluğu kontrolleri yapılabilir. Sonuçlara göre eklenen tasarımlar ayrı post-hoc notla kaydedilecek. Önceki 886 dosyanın SHA-256 manifesti alındı; kitap ve önceki araştırma aynı kalacak.

## İlk çalıştırmadan önce uygulama ayrıntıları

Motif listesi bütün-alan semantiğine göre üçlüler arasında kanonikleştirilir. 12 farklı ilk görev, dört farklı çıplak motif bileşimi ve dört farklı ön/son ekli bileşim üretilemeyen dünya çifti yeniden seçilir. Bu yalnız veri kümesi boyutu ve semantik tekrar koşuludur; öğrenicinin başarımına bakılmaz. 12 yaşamın yapısal denetiminde yalnız bir yaşamda ikinci çift seçimi gerekmiştir. Her yaşamın toplam 28 hedefi bütün-alan semantiğinde birbirinden farklıdır.

`oracle_motifs`, ilk deneyim ailesinin iki motifini baştan alır; aile değişmiş probelarda geleceğin farklı motifleri kendisine verilmez. Dolayısıyla o koşulda eski aile için oracle, yeni aile için ayrıcalıklı çözüm oracle'ı değildir.

Tekrarlanan arama imzaları yalnız sekiz arama girdisindeki çıktılardan oluşur. Bu eleme gözlenen örneklerdeki eşdeğerliği kullanır, bütün alan için eşdeğerlik garantisi vermez. Yanlış bir erken temsilci, başka girdilerde farklı bir doğru programı budayabilir. Kabulde yanlış çıkan ilk adaydan sonra daha fazla aday denenmez; bu tasarımın sınırı saklanacaktır. Kuyruktaki ebeveyn çıktıları bütün yöntemlerde aynı biçimde yeniden kullanılır. Dolayısıyla primitive yürütme bütçesi her tam programı sıfırdan yeniden çalıştırmanın maliyeti değildir; genişletilen kenarların açılım maliyetidir.
