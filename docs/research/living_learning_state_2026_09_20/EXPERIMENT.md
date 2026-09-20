# Deneyimden öğrenilen yinelenen durum: protokol, sonuç ve sınırlar

20 Eylül 2026. Önceki dört araştırma turuna ek. Bu deney bilinen otomata çıkarımı ailesini kullanır; yeni bir temel öğrenme algoritması iddiası yoktur.

## 1. Önceki açık ve yeni sınama

Önceki zamansal deneyde geçmişi temsil edecek gecikmeler araştırmacı tarafından verildi. Sonraki deney güncelleme matrisini öğrendi, fakat temsil edilen doğrusal fonksiyon sınıfı sabit kaldı. Burada deneyim; hangi geçmişlerin aynı iç duruma gitmesi, hangi yeni sembolün bu durumu nasıl değiştirmesi gerektiğini belirler.

Sınanan iddia: **Kısa dizilerin terminal sonuçları, ham eski dizileri yeni sorguda okumadan, çok daha uzun ve yeni dizileri işleyen kalıcı bir geçiş yordamının edinilmesini sağlayabilir.** Bu, fiziksel nedenin teşhisi veya sınırsız hesaplama dili keşfi iddiası değildir.

## 2. Verinin ve hesabın hakları

Her yaşam 256 farklı ikili dizi içerir; uzunlukları 1–12. Dizi bitince öğrenici önce tahmin verir, sonra yalnız terminal etiketi alır. Ara gizli durumlar veya ara çıktı etiketleri verilmez. Bölüm başlangıcı ve sonu gözlenebilir, her bölüm aynı başlangıçtan başlar.

Örnekleme, uzunluğu 1–12 arasında eşit seçip bitleri bağımsız seçer; daha önce görülmüş tam diziler reddedilir. Dolayısıyla ilerleyen yaşamda uzunluk dağılımı tekil-dizi koşulu nedeniyle değişir. Sorgunun yeni tam dizilere aktarımı bilinçli olarak sınanır; bu bütün doğal yaşamların dağılımı değildir.

Öğrenilmiş grafik 16, 32, 64, 128 ve 256 etiket sonrasında yeniden çıkarılır. Arada son grafik hizmet verir. Yenileme yalnız gözlenen bölüm sınırında yapılır; anlık durumun yeni grafiğe taşınması öğrenilmedi. Bütün geçmiş etiketli diziler yeniden çıkarım için saklanır. Bu, sınırlı süreli veriyi giderek alan ve hizmetle güncellemeyi iç içe yürüten bir protokoldür; gerçek zaman gecikme üst sınırı veya sabit ömürlük bellek garantisi değildir.

Her yenilemeden sonra 128 yeni kısa dizi ve uzunlukları 32, 64, 128, 256 olan toplam 64 yeni uzun dizi değerlendirilir. Aynı değerlendirme kümeleri kontrol noktalarında kullanılır; öğrenici bunları veya skorlarını güncellemede görmez. Ayarlar bu sonuçlarla seçilmedi. Son grafiğin gerçek hedefle bütün dizilerde eşdeğerliği, ayrı değerlendirici tarafından çarpım grafiğiyle sınanır. Karşı örnekler öğreniciye verilmez.

| Dünya ailesi | Yaşam sayısı | Gizli hedef / bozulma |
|---|---:|---|
| Üç durumlu farklı kurallar | 16 | Erişilebilir, minimal, üç durumlu ikili makinelerden bağımsız seçim; gerçekleşen 16 hedef farklı. |
| Parite | 4 | Dizideki birlerin tek/çift sayısı; doğru iki durumlu yordam öğreniciye verilmez. |
| Üç durum sınırı dışı | 4 | Birlerin sayısının 4'e bölümünden kalan sıfır mı? Dört durum gerekir. |
| Bozulmuş farklı etiketler | 8 | İlk ailenin ilk sekiz hedefi; 256 farklı dizinin 32 terminal etiketi ters çevrilir (%12,5). |

Bozulma örnekleri benzersizdir: aynı diziye iki farklı etiket verilmiş değildir. Dolayısıyla daha büyük deterministik bir makine bütün gözlemleri açıklayabilir. Ayrı karşı örnekte aynı diziye iki karşıt etiket verilmesi ayrıca sınanır.

## 3. Öğreniciler

**Durum birleştirme (`merged`).** Etiketli bütün dizilerin ön ek ağacı kurulur. Etiketi gözlenmemiş ön ekler bilinmiyor olarak tutulur. Kısa sözlük sırasıyla aday durumlar birleştirilir; deterministik geçişi korumak için gerekli diğer birleşmeler de yürütülür. Gözlenmiş iki farklı terminal etiketi aynı duruma düşerse birleşme reddedilir. Sonra eksik geçişler reddeden duruma, bilinmeyen çıktılar 0'a tamamlanır. Her öğrenme sonunda bütün eğitim örnekleriyle tutarlılık denetlenir.

Bu, RPNI tarzı açgözlü bir yöntemdir. Klasik RPNI ile birebir aynı uygulama olduğu veya her örnek kümesi için en küçük makineyi bulduğu iddia edilmez. Model durum sayısına 3 gibi bir üst sınır verilmez; sınır mevcut veri ve hesap kaynaklarıdır. İkili alfabe, deterministik sonlu durum dili, ön ek kurma/birleştirme işlemleri ve sıra tercihi verilidir.

**Kesin aday eleme (`enumerated_compiled`, `enumerated_vote`).** Başlangıçta en fazla üç durumlu bütün farklı ikili DFA dilleri bulunur. Her terminal etiketiyle çelişen adaylar elenir. Bir yöntem kalan ilk adayı tek grafik olarak çalıştırır; diğeri kalan adayların 1 cevabı oranını verir. Bunlar aynı aday eleme işini paylaşan iki okuma biçimidir. Ana ailede doğru durum sınırı bu karşılaştırmaya verilmiş güçlü bir ön bilgidir; dört durumlu kontrolde aynı sınır yanlıştır.

İlk etiketli betim uzayı 5.898 tanedir. Erişilmeyen durumlar ve dil kopyaları çıkarılınca 1.054 farklı dil kalır: 1 durumlu 2, 2 durumlu 24, 3 durumlu 1.028. Bu sayım, farklı bir tablo-doldurma küçültmesiyle bütün 5.898 betim üzerinde bağımsız doğrulandı. Sonuçlar ilk adayın sırasına veya oy modelinin önseline de bağlı olabilir; bütün model sınıfları üzerinde tarafsız önsel iddiası yoktur.

**Son sekiz sembol (`suffix8`).** Geçmiş eğitimde aynı son sekiz sembolün kaç kez 0/1 etiketi aldığını sayar; birer ek sayımla olasılık verir. Görülmemiş son ek için `.5`. Öğrenilmiş yinelemeli sınıfın kendisine karşı güçlü bir alternatif olduğu iddia edilmez; açık bir sabit pencere sınıfıdır. Parite için bu sınıfın en iyi olası okuyucusunun sınırı ayrıca analitik olarak verilir.

**Tam dizi tablosu (`word_table`).** Yalnız gördüğü tam dizinin etiketini kullanır; yeni dizide `.5`. Bütün eğitim dizileri farklı ve test yeni olduğundan bu koşulda aktarım yapamaz. Tek başına ona üstünlük edinilmiş yapı lehine güçlü kanıt sayılmadı.

Aday sınıf boşalınca aday eleme yöntemleri cevap vermekten kaçınır. Skor tablosunda bu olay `.5` olasılığının Brier bedeliyle hesaplanır; kapsam ayrıca sıfır olarak kaydedilir. `.25` skoru, başarısız aday sınıfının geçerli yeni kural ürettiği anlamına gelmez.

## 4. Sonuçlar

Ana skor, ikili terminal etiket için `(p−y)²` Brier kaybıdır. 0/1 kesin cevaplarda sınıflandırma hatasına eşittir; olasılıklı cevaplarda değildir. Aşağıda uzun dizilerdeki son skorlar veriliyor.

| Yöntem | Üç durumlu aile | Parite | Dört durumlu kural | Yanlış etiketler |
|---|---:|---:|---:|---:|
| Öğrenilmiş birleştirilmiş grafik | 0 | 0 | 0 | .287109 |
| Aday eleme, seçilmiş tek grafik | 0 | 0 | .250000* | .250000* |
| Aday eleme, adayların oyu | 0 | 0 | .250000* | .250000* |
| Son sekiz sembol | .206508 | .264526 | .240724 | .227182 |
| Tam dizi tablosu | .250000 | .250000 | .250000 | .250000 |

`*` Aday kalmadığı için kaçınmanın `.5` ile puanlanması. Ham çıktı, kapsamı ve ilk boşalma adımını içerir.

**Kullanım alanının tamamı için denetim:** Ana ailede öğrenilmiş 16 grafiğin tamamı, yalnız test edilen dizilerde değil, **boş olmayan bütün ikili dizilerde** gerçek hedefle eşdeğerdir. Paritede 4/4, dört durumlu kontrolde 4/4 için aynı sonuç vardır. Bu sonlu çarpım grafiği denetimi, bu seçilmiş hedefler hakkında kesin bir sonuçtur. Her gelecekteki yeni dünyaya uygulanacak bir garanti değildir.

Ana ailede güçlü aday eleme karşılaştırması da aynı son başarıyı sağladı. Kaynak yaşam boyunca temiz hizmet Brier kaybı birleştirmede `.050781`, seçilmiş adayda `.016113`, adayların oyunda `.009146` oldu. Dolayısıyla yapı birleştirmenin bu ailede genel üstünlüğü gösterilmedi. Aday eleme her etiketle, grafik birleştirme beş kontrol noktasında yenilenir; verilen ön bilgi ve çalışma programı da farklıdır. Bunlar eşit hesaplı algoritma yarışı olarak sunulmaz.

Dört durumlu kontrolde öğrenici durum sayısını dört olarak edindi; üç durumla sınırlandırılmış aday ailesi dört yaşamda da boşaldı. Bu, yeni makine boyutunun öğrenilebilir olduğunu gösterir. Her tür yeni hesaplama dilinin kendiliğinden oluştuğunu göstermez: ikisi de hâlâ verilen sonlu durum dilindedir.

### Tam dil denetiminde iki görünür uyuşmazlık nasıl çözümlendi?

İlk kayıt, boş diziyi de içeren bütün dil için ana ailede 14/16 eşdeğerlik verdi. Ayrı kanıt denetimi farkın iki yaşamda **yalnız boş dizi** olduğunu buldu. Eğitim ve kullanım protokolü uzunluğu en az 1 seçmişti; boş bölümün etiketi hiç gözlenmemişti. Diğer bütün uzunluklarda tam eşdeğerlik vardı.

Bu nedenle ilk 14/16 sayısı korunur; fakat “iki öğrenici uzun dizilerde başarısız” diye yorumlanmaz. Gerçek kullanım alanındaki sayı 16/16'dır. Araştırmacının sağladığı ek boş-dizi etiketiyle yapılan tanısal yeniden öğrenme, bu iki tam-dil farkını da kaldırdı. Bu ek etiket **sonuç sonrası ve ayrıcalıklı** bir denetimdir; özgün deneyin başarısına katılmadı. Özgün JSON değiştirilmedi.

## 5. Negatif: yanlış geri bildirimi karmaşık yapı diye açıklamak

32 farklı etiket bozulunca birleştirme sekiz yaşamın hepsinde gözlenen 256 etiketi tam açıklayan bir grafik kurdu. Ancak bunlar 36–42 durum taşıdı (ortalama 39,375); temiz üç durumlu gerçek hedeflerden hiçbiriyle eşdeğer değildi. Uzun temiz sorgularda ortalama kayıp `.287109`; sekiz yaşamın aralığı `.109375–.609375`.

Son sekiz sembol modeli aynı bozulmuş etiketlerle `.227182` verdi. Bu küçük ve eşlenmiş örneklemde genel yöntem sıralaması ilan edilmiyor. Gösterilen negatif daha nettir: **eğitim tutarlılığı ve yapı büyümesi, doğru bir yeni ayrımın edinildiği anlamına gelmedi.** Sabit sınırlı aile ise boşaldı; bu da hatanın bozuk etiketten mi, daha büyük gerçek kuraldan mı geldiğini tek başına söylemedi. Temiz dört durumlu dünya aynı boşalma belirti­sini üretmişti.

Bozulmuş etiketler araştırmacı tarafından doğru değerlerle değiştirildiğinde, saklanan dizilerden yeniden kurulan grafik sekiz yaşamın tamamında boş olmayan kullanım alanında doğru oldu. Boş diziyi de içerince 6/8; diğer iki fark yukarıdaki başlangıç-sorgusu kapsamına ait. Bu, korunmuş verinin **güvenilir bir düzeltme verildiğinde** onarıma yettiği tanığıdır. Hangi etiketin yanlış olduğunun sistem tarafından keşfedildiği gösterilmedi.

Ayrı tam karşı örnek: aynı başlangıçtan `(0,1)` dizisine hem 0 hem 1 zorunlu etiketi verilirse hiçbir deterministik DFA ikisini de sağlayamaz. Uygulama bu çelişkiyi reddeder; daha çok durum çözüm değildir. Eksik gözlem, değişmiş hedef, stokastik çıktı ve bozuk etiket bu belirti altında hâlâ farklı olası açıklamalardır.

## 6. Neden uzun geçmiş için uzun anlık bellek şart değil?

Parite hedefinde her `k` için `0^(k+1)` ve `1 0^k` dizileri aynı uzunluğa ve aynı son `k` bite sahiptir, doğru etiketleri farklıdır. Dengeli seçimde yalnız bu pencereyi kullanan her tahminci için en az `1/2` sınıflandırma hatası veya `.25` Brier kaybı vardır. Bu sınır; okuyucunun eğitimi ve büyüklüğünden bağımsız, verilen gözlem hakkına ilişkindir.

Öğrenilmiş iki durumlu grafik ise yeni bit geldikçe mevcut durumunu günceller. Sonuç, bütün dizinin ham kopyasını saklamadan gerekli ayrımı korur. Burada öğreniciye `XOR` devresi veya doğru geçiş tablosu verilmedi; doğru geçişler terminal deneyimlerden çıktı. Fakat sonlu durum kurma ve birleştirme işlemleri verildi. “Sıfır başlangıç yapısı” iddiası yoktur.

Temel matematiksel ilişki, deterministik ve sabit terminal hedef `f` için:

\[
h\sim h' \iff \forall u,\ f(hu)=f(h'u).
\]

Bu ilişki sağdan devamla korunur: `h~h'` ise `ha~h'a`. Böylece eşdeğerlik sınıfları üzerinden `δ([h],a)=[ha]` geçişi iyi tanımlıdır. Sonlu sınıf varsa sonlu yinelenen hesap yeterlidir. Bugünkü terminal etiketlerin eşitliği ise tek başına buna yetmez; aynı sonraki sembol etiketleri ayırabilir.

İlişkinin sınırlı veriden öğrenilmesi, onu tanımlamakla aynı başarı değildir. Her sonlu eğitim kümesi için görülmemiş daha uzun bir `w*` seçip yalnız o dizinin etiketini ters çevirmek, yine sonlu durumla ifade edilebilir farklı bir hedef verir. Ayrı denetimde parite ile yalnız `0^13` dizisinde farklı olan bir makine kuruldu: özgün eğitim verisi iki dünyada aynı, bu yeni sorgunun cevabı farklıdır. Ek sınıf/kanıt varsayımı olmadan koşulsuz kesinlik yoktur; yararlı uzunluk genellemesi bununla olanaksız kılınmış olmaz.

Bu, bilinen düzenli dil ve öngörü durumu fikridir. Önceki genel güncellenebilirlik koşuluna somut bir öğrenme yordamı ve çalışma devresi ekler; onu yeni adla yeniden keşfedilmiş temel yasa diye sunmaz.

## 7. Kalıcılık, silme ve hesap maliyeti

Öğrenilmiş grafik yeni diziyi sembol başına bir geçişle işler. Tahmin yolu saklanan eğitim dizilerini okumaz. Parite grafiğinin iki durumu ve dört geçişi vardır. Ayrı silme denetiminde aynı 128 sıfırlık son ekten önce gelen 0/1 ayrımı, grafik ve anlık durum korununca doğru taşındı. Grafik silme karşılığı açıkça başlangıçtaki sürekli 0 cevaplı yordama dönmek olarak hesaplandı; eksik grafikle çalışan bir program sınaması değildir. Bölüm içindeki durum bitini sıfırlama ise gerçekten farklı başlangıç durumu kullanılarak yürütüldü ve o bölümün geçmiş ayrımını kaybettirdi. Durum biti kaybolsa bile grafiğin korunması, yeni başlayan tam bölümleri yeniden öğrenmeden işlemeye yeter.

Yeni süreçte iki devam denetimi yapıldı:

- Öğrenmenin 37. bölümünde bütün öğrenme durumu yüklendi; 53 sonraki bölüm, 64'teki yeniden çıkarım dahil aynı tahmin ve son durumu verdi.
- Bölüm içindeki 127 sembollük ön ek saklanmadan yalnız grafik ve mevcut durum yüklendi; kalan 385 sembollük işlem birebir aynı oldu.

İlk devam örnek arşivini tutar; ikinci yalnız çalışma durumunu tutar. Küçük çalışma belleği, küçük öğrenme belleği değildir. Ana ailede öğrenme arşivi yaşam başına 256 dizi, ortalama 2.094 semboldür; son grafikte ortalama 5,875 geçiş girişi bulunur.

| İşlem sayacı, yaşam ortalaması | Temiz üç durumlu aile | Yanlış etiketler |
|---|---:|---:|
| Birleştirme kapanışında ziyaret edilen çiftler, beş çıkarım toplamı | 1.546,3 | 7.460,1 |
| Birleştirme denemelerinde kopyalanan durumlar | 8.530,6 | 440.572,8 |
| Aday elemenin eğitim geçişleri | 18.551,6 | 14.572,0 |

Sayaçlar aynı birim değildir; FLOP veya süre üstünlüğü diye karşılaştırılmaz. Aday kütüphanesini oluşturma ve ayrı değerlendirme hesabı bu hizmet/eğitim sayaçlarına dahil değildir. Bozulmada aday elemenin erken boşalması onun hesabını düşürürken yararlı cevap kapsamını da yok eder. Birleştirme büyüyen bir yapıyla devam edip daha çok hesap harcar. Kaynak sorunu sonuç yorumunun parçasıdır.

## 8. Kaynaklar ve açık kalanlar

Bu durum birleştirme yaklaşımı [Oncina ve García'nın RPNI çalışmasıyla](https://grfia.dlsi.ua.es/repositori/grfia/pubs/77/inferring.pdf) eşleşir. Sınırlı pencere ile öğrenilmiş yinelenen öngörü durumu ayrımı, [CSSR çalışmasında](https://arxiv.org/abs/cs/0406011) da ele alınır. Öğrenicinin kendi seçtiği sorgulara doğru üyelik cevabı ve eşdeğerlik karşı örneği aldığı [Angluin L* düzeni](https://homepages.math.uic.edu/~lreyzin/papers/angluin87.pdf) ise farklı bilgi haklarına sahiptir; buradaki pasif öğreniciye böyle bir öğretmen verilmedi.

Deney; anlamlı alfabeyi, bölüm sınırlarını, terminal geri bildirimin atfını, deterministik sonlu durum dilini ve eski örneklerin saklanmasını hazır alır. Fiziksel neden teşhisi, yeni amaç edinimi, gürültülü devamda güvenilir yapı kabulü, bölüm ortasında temsil taşıma ve uzun ömürde kaynak sınırlaması çözülmedi. Temiz sınırlı dünyada daha az hazır durum bilgisiyle edinim gösterildi; genel yaşarken öğrenme cevabı bulunmuş sayılmadı.

[Ham deney](../../../research/living_learning_state_2026_09_20/results/recurrent_state.json), [ayrı kanıt denetimi](../../../research/living_learning_state_2026_09_20/results/evidence_audit.json), [teori](FAILURE_IDENTIFIABILITY.md), [literatür denetimi](STATE_DISCOVERY_LITERATURE.md).
