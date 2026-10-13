# Tur 13 — Geçici edinim, konsolidasyon ve aynı bilgiyle güncelleme

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel sonuç-öncesi protokol

Ana ölçümler hesaplanmadan yazıldı; dış bir ön kayıt değildir. Önceki 924 dosya ve Git HEAD korunur. Kitap, ana sistem, commit/push/yayın kapsam dışında. Geniş yaşam-boyu öğrenme probleminin yerine bu deney geçirilmeyecek.

## Kullanıcının açıklaması — aynen

> insan zihninde yeni bilgiler yeni nöronlar beyne kaydedilir gün içinde ve insan uyuduğu zaman bu bilgiler beyne regüle edilir yani ana nöral sisteme bağlanır. yani aslında insan nöral olarak gün içerisinde yine edinilmiş bağlam uzayını kullanır ve uyurken yeni öğrendiği veya farklı bir bağlam uzayı varsa bu edinimlerde mevcut bilince dahil edilir. yukarı ve aşağı yönlü paralel işleyen bilinç yapısı üst görevciler tarafından denetlenir ve sistem kusursuz bir biçimde çalışır. işte zihin ve beyin tam olarak böyle çalışır. bunlardanda hipotezler çıkar gerekirse 50 farklı hipotez çıkar seriden araştırmaya devam et problem uzayını olabildiğince daralt. genel yapay zeka yaşarken öğrenen bir yapay zekayı tamınlamıyor mu işte bende sana yaşarken öğrenmenin formülünü veriyorum matematiksel olarak bunu kanıtla. hadi bakalım

Bu metin araştırma önerisi olarak korunur; biyolojik olgu veya kanıtlanmış formül diye kabul edilmez. Uyku benzetmesi burada yeni dış etiket alınmadan ayrılan hesaplama evresini anlatır; uyku fizyolojisi veya bilinç modellenmez.

## Ayrıştırılacak iddialar

1. Olayı geçici bellekte tutmak ile gözlenmemiş durumlara uygulanabilir kalıcı kural edinmek farklıdır.
2. Doğru eşlenmiş deneyimlerin verilen daha zengin bir temsil diline aktarılması, basit dilde yapılamayan genellemeyi sağlayabilir.
3. Aynı istatistik, aynı amaç ve aynı hesaplama kullanıldığında gün içi ve ertelenmiş güncelleme aynı sonuca ulaşabilir. Böyle bir sonuç, ayrı bir “uyku” evresinin bu modelde zorunlu olmadığını gösterir; uykunun biyolojik işlevini çürütmez.
4. Kendi eski tahminlerini yeniden üretmek, silinen yeni dış etiketlerin yerine geçmeyebilir.
5. Eski ve yeni kurallar aynı gözlenen girdide çelişiyorsa bağlam ayrımı olmadan kusursuz koruma ve edinim birlikte mümkün olmayabilir.
6. Gürültülü veriye uyum ve gerçek hedef doğruluğu farklıdır; bir denetçi tüm yanlışları gizli doğru etiketleri görmeden kesin ayıklayamaz.

## Dünya ve verilen temsil dili

Girdi altı gözlenebilir ikili özellik: x∈{0,1}⁶, toplam 64 farklı girdi. İki gerçek bağlam var; öğreniciye bağlamın verilip verilmemesi rejime bağlı. Eski kural tek özelliğin değeri veya tersi; yeni kural iki ya da üç özelliğin paritesi veya tersi. Parite, seçilen bitlerin toplamının ikiye göre kalanıdır. Yeni kural eski kuraldan farklıdır.

Araştırmacının verdiği dar dil, sabitler ve tek özellikten oluşan 14 fonksiyon; zengin dil, en çok üç özellikli tüm pariteler ve terslerinden oluşan 84 fonksiyon. Bu dili öğrenici icat etmiyor. Yeni nöron üretilmiyor; sabit aday bankasında deneyimle kullanılabilir kural değişiyor. Mutlak ifade kapasitesi artışı iddia edilmeyecek.

Önceki evrede eski bağlamdan 24 doğru gözlem verilir. Sıfır girdisi ve altı birim vektörü bunların içinde bulunur; kalan 17 girdi farklı rastgele girdilerdir. Bu yedi temel girdi, bu aday dilindeki eski pariteyi tekil belirler. Başlangıç eski kuralı gizli öğreticiden alınmaz; 14 adayın gözlem hatalarıyla edinilir. İki gözlenebilir bağlam için başlangıç kuralı aynı edinilmiş eski kuraldır; yeni bağlam henüz uzmanlaşmamıştır.

Gündüz yeni gerçek bağlamdan 24 farklı girdi gözlenir. Öğrenici her etiket verilmeden önce tahmin eder. Geçici bellek yalnız gündüz olaylarını tutar. Gece/konsolidasyon evresinde yeni dış etiket yoktur. Sonrasında geçici ham olay belleği ve ham eski kayıtlar kaldırılır; kalıcı kural ve varsa hata sayısı özeti korunur. Ayrı süreçte bu durumdan yeni gözlemlerle devam edilebilirlik denetlenecek.

## Üç rejim, 24 yaşam

- `visible_context`: eski gözlem bağlamı 0, yeni bağlam 1 olarak sunulur; etiketler doğru.
- `hidden_context`: her iki bağlam da 0 olarak gözlenir. Girdi ve kural aynı yaşamın görünür rejimiyle aynı, yalnız bağlam ayrımı kaldırılmıştır. Aynı gözlenen girdide çelişki doğabilir.
- `noisy_day`: bağlam görünür; 24 yeni etiketin önceden belirlenmiş dört tanesi ters çevrilir. Doğru etiketler yalnız değerlendiricidedir.

Rastgelelik tohumları 0–23. Aynı yaşamın kaynakları, hedef girdileri ve öğreticileri yöntemler arasında aynıdır; rejim değişiklikleri kontrollüdür. 24×3×8 = 576 edinim yolu; bağımsızlık birimi 24 yaşamdır, değerlendirme noktaları bağımsız örnek sayılmaz.

## Sekiz yöntem

1. `frozen_core`: edinilmiş eski kural; gündüz belleği ve konsolidasyon yok.
2. `cache_only`: eski kural + tam anahtarlı gündüz olay belleği. Konsolidasyon yok. Bellek kaldırıldıktan sonra yeni kural kalmaz.
3. `sleep_linear_replay`: geçici bellek; evre sonunda eski ve yeni doğru kaydedilmiş olaylar 14 adaylı dilde birleştirilir.
4. `sleep_rich_replay`: aynı olaylar 84 adaylı dilde birleştirilir.
5. `online_rich`: aynı 84 adayın eski kayıt hata sayıları başta hesaplanır; her yeni olayın hataları birer kez eklenir. Güncel en iyi adaylar her olaydan sonra kullanılabilir. Toplam aday–örnek hata kontrolleri `sleep_rich_replay` ile aynı; en iyi aday seçme sıklığı farklı olduğundan tam CPU eşitliği iddiası yok.
6. `sleep_rich_recent`: yalnız gündüzün 24 olayıyla ilgili gözlenen bağlam yeniden öğrenilir. O bağlamın eski olay özeti bırakılır; gündüz etkinleşmemiş bağlamın eski kuralı korunur.
7. `sleep_self_replay`: gündüz girdileri için gerçek yeni etiketler yerine başlangıçtaki eski kuralın ürettiği etiketler kullanılır; bunlar eski kayıtlarla birleştirilir. Aynı girdi ve işlem miktarında yeni dış bilginin çıkarılması kontrolü; eşit bilgilendirilmiş yöntem diye sunulmaz.
8. `sleep_mispaired_replay`: gündüz etiketleri sıra üzerinde bir konum döndürülür ve yanlış girdilerle eşlenir; eski kayıtlar doğru kalır. Olaylar arası bağın korunması için bilgi bozulması kontrolü.

Model, her gözlenen bağlam için adayların tüm kayıtlar üzerindeki tamsayı hata sayılarını hesaplar. En az hatalı bütün adaylara eşit ağırlık verir. Örtük karmaşıklık cezası, test etiketinden model seçimi veya keyfî ilk adayı seçme yoktur. Olasılık, adayların 0/1 çıktılarının ortalamasıdır. Aynı gözlenen bağlamda çelişik etiketler hata olarak birikir; çelişkiyi reddeden özel işlem yoktur. Gürültü rejiminde bu, gerçek gürültü dağılımının tam Bayes posterioru diye sunulmaz.

## Ölçüler

Eski 64 girdi, yeni 64 girdi, yeni bağlamda henüz görülmemiş 40 girdi ve gözlenmiş 24 girdi ayrı değerlendirilir. Ana hata MSE: (tahmin olasılığı−doğru ikili etiket)²; sıfır iyi, kesin yanlış bir. Eşitliklerde beklenen doğruluk raporlanır. Gizli bağlamda eski/yeni riskler eşit ağırlıklı ortalanır.

İlk ana çalıştırmadan önce değerlendirme açıklığı: `seen` ve `unseen` anahtarları bütün gündüz boyunca kullanılacak sabit 24 eğitim girdisi ve hiç kullanılmayacak 40 ayrılmış girdiyi belirtir. Ara anlarda `seen` kümesinin tamamı henüz görülmemiştir; o anın görülmüş/görülmemiş bölümü diye yorumlanmayacak. Bu değerlendirme kümeleri öğreniciye verilmez. Gündüz girdileri tekrar etmediği için tam anahtarlı belleğin etiket-öncesi sorguda isabet etmemesi beklenir; bellek yararı ayrı sorgular ve evre sonuyla ölçülür. Yeniden başlatma denetimi, tutulan istatistiğe yeni kayıt ekleyen ortak tanısal devam işlemidir; donmuş/bellek yönteminin özgün politikasının ikinci gün boyunca öğrenmeye başladığını göstermez.

Gündüz etiket öncesi ortalama hata; 0/6/12/18/24 gündüz noktasında kalıcı çekirdek ve bellekli davranış; konsolidasyon sonrası bellek varken ve kaldırıldıktan sonra aynı ölçüler. Konsolidasyondan önceki ve sonraki fark ile online yöntemin aynı sonuca ulaşması ayrı raporlanır. Yalnız eski örnekleri hatırlamak transfer başarısı sayılmaz.

Maliyetler: ortak eski edinim 24×14 aday–örnek kontrolü; yöntem başına sonraki kontroller; tutulan ham olay adedi; aday hata özeti boyutu; JSON durum baytı; aday seçme çağrısı. Bunlar toplam süreç RAM/FLOP/CPU eşitliği değildir. Rich replay'in toplamı 48×84, linear replay'in 48×14, recent'in 24×84 kontrolüdür. Online ile rich replay aynı 48×84 sayıya ulaşmalıdır. Sentezlenen etiketler yeni dış kanıt sayılmaz.

## Matematik ve doğrulama yükümlülükleri

- Bağımsız skaler parite hesabı ile tüm aday çıktıları ve gerçek değerlendirme etiketleri doğrulanacak.
- Sonuçlar incelenmeden belirlenen eski edinim tekilliği, etiket bozulma sayısı ve yöntem eşleştirmeleri denetlenecek.
- Online ve rich replay bütün rejim/yaşamlarda aynı hata istatistiğini ve aynı son çekirdek tahminini vermeli. Toplam öğrenme kontrolleri eşleşmeli.
- Geçici bellek boşaltma sonrası kalıcılık ve ayrı süreçten kesintisiz çalışmaya eşit devam kontrolü yapılacak.
- Gizli bağlamda eski ve yeni kurallar 64 girdinin 32'sinde çelişir. Aynı gözlenen girdiye tek olasılık veren sistemin eşit ağırlıklı birleşik MSE'si en az 1/8 olmalıdır. Bağlam sunulursa bu alt sınır zorunlu değildir.
- Bilgisiz iç işleme hakkında iki-ayrışmayan-dünya karşı örneği ve sabit dilde toplamsal istatistik yeterliliği ayrıca ispatlanacak.
- Tam ana çalıştırma tekrar hesaplanacak; 924 eski dosyanın içeriği korunacak.

Bu deney uykunun nörobiyolojik zorunluluğunu, üst bilinç görevcilerinin varlığını, kendiliğinden temsil dilinin oluşmasını, açık uçlu hedef üretimini veya genel yapay zekâyı kanıtlamaz. Bu sınırlara rağmen geçici deneyimin kalıcı hesaplamaya dönüşmesi, veri/hesaplama/zamanlama ayrımı ve çelişik gereksinimlerin sınırları hakkında yanlışlanabilir sonuçlar üretir.
