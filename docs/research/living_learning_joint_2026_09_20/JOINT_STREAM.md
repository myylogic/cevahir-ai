# Edinim, koruma ve düzeltme aynı yaşamda

20 Eylül 2026. [Önceki ayrı deneylerden](../living_learning_reassessment_2026_09_20/REPORT_TR.md) devam. [Kod](../../../research/living_learning_joint_2026_09_20/joint_stream.py), [tam sonuçlar](../../../research/living_learning_joint_2026_09_20/results/joint_stream.json).

**Sonuç:** Görev adı veya değişim bildirimi almayan tek bir tahminci; verilmiş küçük bir özellik dilinde ilişkiler edindi, hâlâ geçerli eski bölgede becerisini korudu, başka bölgede değişen ilişkiyi düzeltti ve bütün öğrenme durumunu taşıyarak öğrenmeye devam etti. Ancak aynı dilin tamamını başlangıçtan kullanan güçlü yöntem aynı son becerileri daha az hesapla elde etti. Yerel koruma, görülmemiş bölgenin de değiştiği dünyada yanlış eski ilişkiyi korudu. Gürültülü etiket dizisi ayrıca gerçek değişim gibi işlendi.

Bu, bazı mekanizmaların artık **aynı değişen durum üzerinde birlikte** çalıştırıldığı bir varlık tanığıdır. Algı dilini keşfeden veya bütün yanlış değişimleri ayıran genel bir sistem değildir.

## Protokol ve verilen bilgi

24 eşleştirilmiş yaşam, her birinde 1.792 gözlem. Öğreniciye yalnız iki sürekli girdi `(x,z)` ve o girdinin gürültülü sayısal sonucu verilir. Önce tahmin, sonra sonuç ve güncelleme. `x` sırayla `[-1,-.2]` veya `[.2,1]`, `z∈[-1,1]` aralığından gelir. Ayrı görev başlıkları, dünya kimliği ve sınır bildirimi yoktur; bütün cevaplar aynı katsayı vektörünü kullanır. Öğrenicinin kendi örnek sayacı vardır.

İlk ilişki:

\[
f_0(x,z)=.5+.7x-.4z+.8xz.
\]

1.025'inci gözlemden itibaren sağ tarafta ilişki değişir:

\[
f_1(x,z)=f_0(x,z)-1.6\max(x,0)z.
\]

Sıra, 256 gözlemlik yedi dönemdir: sol edinim, sağ edinim, sol dönüş, sağda yanlış etiket dizisi ve toparlanma, sağda gerçek değişim, sol koruma, değişmiş sağa dönüş. Bu sıra değerlendiricinin kaydıdır; öğreniciye iletilmez. Bütün etiketlerde bağımsız `.1` standart sapmalı Gauss gürültüsü vardır. Ayrıca 769–800 arasındaki 32 etikette, gerçek dünya henüz değişmemişken değişmiş sağ hedefin değeri verilir. Bu düzenli yanlış bilgi, bağımsız sıfır ortalamalı gürültüden daha zor bir negatiftir.

**Hazır verilenler:** `[1,x,z,xz,max(x,0)z]` özellik dili, koordinatların anlamlılığı, sayısal etiket ve aynı girdiye hizalanması, düzenli `4×4` uzamsal saklama ızgarası, ridge çözümü ve özellik kabul kuralı. `x=0` ayrımı hem dil hem saklama ızgarası için verilmiştir ve dünyanın değişim bölgesiyle uyumludur. Görev kimliğini kaldırmak, bütün yararlı yapısal önbilgiyi kaldırmak değildir.

Her hücre en yeni sekiz gözlemi tutar; toplam en fazla 128 ham kayıt. Böylece ziyaret edilmeyen bölgenin örnekleri kalırken ziyaret edilen bölgede eski etiketler yenileriyle yer değiştirir. Her 16 gözlemde mevcut bellekten ortak katsayılar yeniden hesaplanır. Bu, çalışma sırasında aralıklı küçük yeniden uydurmadır; her cevapta bütün geçmişi tekrar getirmez.

## Edinim kuralı ve güçlü kontroller

`growth_coverage`, başlangıçta yalnız `[1,x,z]` kullanır. Bellekte en az 48 örnek olduğunda, örneklerin iç sayacının tek/çift oluşuyla iki eğitim katı kurulur. Bir özellik eklenince iki yöndeki ayrı-kat tahmin hatası da en az `.001`, ortalama hata en az %10 azalırsa özellik etkinleştirilir. Sonra katsayılar bütün bellekte yeniden hesaplanır. Bunlar önceden belirlenmiş sezgisel eşiklerdir; tekrar kullanılan katlar adaptif arama nedeniyle bir güven aralığı garantisi sağlamaz. İzinli iki ek terim hazırdır; yeni ilkel işlem veya sınırsız dil icat edilmez. Etkinleşen özellik kaldırılmaz; etkisi katsayıyla değişebilir.

Kontroller aynı akışı ve temiz ayrı değerlendirmeleri paylaşır:

| Yöntem | Özellikler | Geçmişin etkisi |
|---|---|---|
| Büyüyen + bölgesel bellek | 3→en fazla 5 | Her ızgara hücresinde en yeni 8 kayıt |
| Baştan geniş + bölgesel bellek | Baştan 5 | Aynı hücreler ve aynı güncelleme sıklığı |
| Baştan geniş + kayan pencere | Baştan 5 | Bölgesine bakılmadan en yeni 128 kayıt |
| Baştan geniş + tüm geçmiş istatistiği | Baştan 5 | `G=Σφφᵀ`, `b=Σφy`; ham örnek biriktirmez |
| Sabit temel + bölgesel bellek | Sabit 3 | Aynı hücre belleği |

Bütün yöntemlerde ridge `.001`. Tüm geçmiş yöntemi aynı örneklerin etkisini 25+5 istatistik skalarıyla toplar; her yeniden hesaplamada aynı beş katsayıyı çözer. Farklı saklama politikaları farklı zamansal/bölgesel ağırlıklandırmalar yapar; aynı amaçta yalnız daha iyi sayısal çözüm karşılaştırması değildir. İşlem sayıları ayrıca verilir; tüm yöntemler tam FLOP veya toplam bilgi miktarı eşit değildir.

Her yaşamda her taraf için ayrı 400 sürekli girdi vardır. Etiketleri hiçbir öğrenme adımına girmez. Son riskle birlikte, bütün yaşamda tahminden önce oluşan temiz servis hatası ölçülür; yalnız son başarılı dönemi seçme yoktur.

## Sonuçlar

24 yaşam ortalaması, temiz hedefe karşı ortalama kare hata:

| Yöntem | Bütün yaşam servis hatası | Son sol, eski ilişki geçerli | Son sağ, ilişki değişmiş |
|---|---:|---:|---:|
| Büyüyen + bölgesel | `.012425` | `.000478` | `.000417` |
| Baştan geniş + bölgesel | `.012216` | `.000478` | `.000417` |
| Baştan geniş + kayan | `.029270` | `.204387` | `.000293` |
| Baştan geniş + tüm geçmiş | `.048933` | `.000052` | `.078358` |
| Sabit temel + bölgesel | `.056672` | `.012557` | `.012463` |

Son iki bölgede büyüyen ve baştan geniş bölgesel modeller bütün yaşamların her birinde aynı risklere ulaşmıştır. Büyümenin servis farkı `+.000208656`, eşleştirilmiş ortalama için yaklaşık normal %95 aralık `[+.000134131,+.000283182]`: bu düzende baştan geniş model lehine. Bu aralık yalnız seçilmiş dünyalarda örnekleme değişimini kapsar.

Büyüyen yöntem her yaşamda `xz` ve `max(x,0)z` terimlerini etkinleştirir. Fakat maliyeti ortalama 320,17 küçük doğrusal çözüm, baştan geniş yöntemin 112 çözümüdür. Eğitim momentlerinde sayılan çarp-topla işlemleri yaklaşık 613.810 ve 392.106. Büyümenin seçme maliyeti ödenmiştir; daha az hesapla daha iyi genel öğrenme iddiası desteklenmez.

**Sayaç sınırı:** Kodun `features` yordamı beş aday değeri de hesaplayıp etkin bileşenleri seçer. `service_feature_components` gibi sayaçlar kullanılan vektör/başlık bileşenlerini sayar; bütün özellik üretim işlemlerini saymaz. Bunların azalması gerçek CPU hızlanması kanıtı değildir. Bellek sınırı ham kayıt ve sonlu özellik sayısınadır; JSON boyutu Python nesneleri, bütün yürütme belleği veya keyfî uzun yaşam için sabit bit garantisi değildir. Tutulan tanısal sayaçlar ve örnek sayacı ayrıca yer kaplar.

## Negatifler ve bulgu sonrasında yapılan ek kontrol

**Kısa yanlış kanıt, doğru kuralı bozdu.** 768'inci gözlem sonunda büyüyen modelin sağ hatası `.000464`; 32 yanlış etiketten sonra `.132789`; doğru akış döndükten sonra 896'da `.000673`. Sistem toparlanır, ama yanlış etiket dizisini gerçek değişimden başlangıçta ayıramamıştır. İki eğitim katında iyileşme aramak bu sorunu çözmez: aynı yanlış düzenlilik iki katta da bulunur. Tüm geçmiş istatistiği bu kısa dizide daha dayanıklıdır (`.005082`), fakat gerçek değişime daha yavaş uyar.

Özellik etkinleşmelerinin **16/24'ü yanlış etiket ve toparlanma aralığında** ikinci terimi eklemiştir. Bu, sonraki gerçek değişime uyumu yapay olarak kolaylaştırmış olabilir. Bunun üzerine, başlangıç protokolünden ayrı ve **sonuç görüldükten sonra seçilmiş** bir mekanizma kontrolü yapıldı: aynı girdiler, aynı gerçek değişim, aynı gürültü ve eşikler korunup yalnız 32 yanlış etiket etkisi çıkarıldı. İlk çıktı veya kod değiştirilmedi. [Ek yordam](../../../research/living_learning_joint_2026_09_20/joint_ablation.py), [ayrı sonuç](../../../research/living_learning_joint_2026_09_20/results/joint_no_burst.json).

Bu kontrolde gerçek değişimden önce ikinci terimi etkinleştiren yaşam sayısı 1/24 oldu. Bütün yaşamlar sonunda yine aynı son sağ MSE `.000417`ye ulaştı. Gerçek değişim döneminin servis hatası `.042377`den `.044044`e yükseldi. Böylece yanlış etiketler bazı yaşamlarda sonraki uyumu hazırlamış olsa da **son becerinin edinimi bu yanlış etiket dizisine bağımlı değildi**. Ek deney yeni bağımsız doğrulama kümesi değildir; aynı dünyalarda bir neden açıklamasını sınar. Yalnız olağan Gauss gürültüsüyle bir yaşamda erken etkinleşme kalması da kabul kuralının tam yanlış-pozitif kontrolü olmadığını gösterir.

**Korunan bölge de değişmiş olabilir.** 1.025–1.280 arasında yalnız sağ bölge görülür. Sağda değişen dünya ile aynı anda her iki tarafta değişen dünya, bu ana kadar tamamen aynı gözlemleri verir. Fakat soldaki doğru hedef farklıdır. Aynı durumu üreten öğrenici için iki dünyanın sol risk ortalaması en az hedefler arası kare farkın dörtte biridir; değerlendirme girdilerinde ortalama alt sınır `.088415`. Bölgesel öğrenici soldaki eski ilişkiyi koruduğundan, gizli global değişim yorumunda 1.280 anında hata `.355531` olur. Bu bir ikinci global dünya boyunca tamamlanmış öğrenme deneyi değildir; ortak gözlem öneki ve karşıolgusal sol değerlendirmesiyle kurulmuş karşı örnektir. Sonraki sol kanıt geldiğinde durum değişebilir.

Bu sonuç, bölgesel saklamanın temel eksik varsayımını gösterir: ziyaret edilmeyen bölgedeki eski kayıtların geçerliliği veriden kendiliğinden onaylanmış değildir. Saklama geometrisi, dünyanın değişim geometrisiyle uyumlu olduğunda faydalıdır.

## Kalıcılık ve ilke iddiası

896'da bütün öğrenme durumu JSON üzerinden yeniden kuruldu; sonraki 896 gözlem aynı sırayla işlendi. Her yaşam/yöntemde bütün tahminler ve son durumlar tam aynı kaldı. Bu ilk kontrol aynı işlem içindedir. Ayrı doğrulama ayrıca beş yöntemin her biri için yeni işletim sistemi sürecinde 64 güncellemelik devamı sınar. Ham örnekler ve çözüm istatistikleri kaldırıldığında mevcut tahminler korunur; çünkü tahmin yalnız etkin özellikleri ve ağırlıkları okur. Öğrenmeye devam etmenin bilgisi ile mevcut beceriyi kullanmanın bilgisi yine ayrıdır.

Veri akışından yapıyı değiştirme ve sınırlı istatistiklerle edinim yeni değildir; [Domingos ve Hulten, 2000](https://alchemy.cs.washington.edu/papers/pdfs/domingos-hulten00.pdf) karar ağacı büyümesini araştırır. Görev sınırları olmadan öğrenme de önceden incelenmiştir; [Zeno ve diğerleri](https://arxiv.org/abs/1803.10123). Bu deney bu algoritmaları uygulamaz; “görev kimliği yok” veya “özellik etkinleşti” tek başına özgünlük oluşturmaz.

Yeni ve yerel sonuç: önceki ayrı deneylerdeki bazı işlevler tek ortak katsayı/bellek durumu üzerinde birlikte yürütüldü. Çözülmeyenler: hazır dil ve bölgelemenin edinilmesi, yanlış etiket ile gerçek değişimin ayrımı, görünmeyen bölgelerin geçerliliği, daha uzun ve değişik sıralı yaşamlar, dil dışı beceriler, kaynakları öğrenilmiş bir stratejiyle dağıtma. Bu deney genel yaşarken öğrenme mekanizması olarak sunulmuyor.
