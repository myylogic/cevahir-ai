# Öğrenmenin kendi kanıtını kapatması ve iç durumla yeniden sınama

20 Eylül 2026. İlk iki deneyden sonra gelen 1–22 numaralı gözlemler ve biyolojik iç durum hakkındaki ek soru üzerine araştırmanın kapsamı genişletildi. Önceki deneyler dışarıdan sağlanan sorgulara dayanıyordu. **Öğrenicinin kendi gelecekteki verisini seçmesini sınamıyorlardı.** Bu belge o açığı ayrı, çalıştırılmış bir deney ve bilgi sınırlarıyla ele alır.

**Sonuç:** Davranıştan sapma yalnız hata olmak zorunda değildir. Eski öğrenmenin engellediği ayırt edici bir eylemi mümkün kılarsa, o eylemin ürettiği gerçek sonuç öğrenmenin düzeltilmesini sağlayabilir. Rastgelelik bunun gerekli koşulu değildir. Bununla birlikte davranış çeşitliliği, belirsizlik veya içsel hareket tek başına yeterli değildir; yeni davranışın ilgili varsayımı sınaması, sonucun gözlenmesi ve güncellemede kullanılması gerekir. Maliyetli yanlış denemeler de aynı mekanizmanın olası sonucudur.

## 1. Dış koşulun aynı olması bütün durumun aynı olması değildir

Bir açıklama için şu ayrımı yapabiliriz:

\[
z_{t+1}=F(z_t,m_t,o_t),\qquad a_t=\pi(m_t,z_t,o_t),
\qquad y_t\sim K_{\theta_t,\kappa_t}(\cdot\mid o_t,a_t),
\qquad m_{t+1}=U(m_t,o_t,a_t,y_t).
\]

`m` geçmişten öğrenilmiş durum, `z` güncel iç işleyiş, `θ` çevre, `κ` eylemi gerçekleştiren sistemin mevcut kapasitesidir. Bu dört zorunlu modül önerisi değildir; aynı toplam durumun açıklayıcı ayrımıdır. Yalnız `o`yu aynı tutmak `z`, `m` veya `κ`yı sabitlemez. Deterministik `F` altında bile zamanla değişen `z`, aynı dış gözlemde farklı eylem üretebilir. Özdeş bütün durumlar ve özdeş geçişlerde farklı sonuç beklemek başka bir iddiadır; burada gerekmez.

İki ayrı fenomen vardır. İç durum eylem seçimini değiştiriyorsa aynı eylemin dünya üzerindeki etkisi sabit kalabilir. Gerçek kapasite değişiyorsa aynı eylemin sonucu da değişebilir. Daha çok yeniden deneme isteği ile gerçekten daha güçlü bir yürütücünün oluşmasını aynı başarı diye ölçmemek gerekir. Biyokimyasal mekanizma hakkında bu denklemlerden sonuç çıkmaz.

## 2. Öğrenme yalnız veriyle değişmez; veriyi de değiştirir

Bir politikanın ürettiği geçmiş dağılımı yaklaşık olarak:

\[
P_\theta^\pi(h_T)=\prod_t\pi(a_t\mid h_t)K_\theta(y_t,o_{t+1}\mid h_t,a_t).
\]

Öğrenilmiş durum `π`yi değiştirdiğinde hangi gözlemlerin alınacağı da değişir. Dolayısıyla yalnız sabit bir veri kümesindeki doğruluk yeterli değerlendirme değildir. Kaçırılan fırsatlar, yeniden keşif süresi, hatalı deneme bedeli ve gelecekte kanıt alabilme ayrıca ölçülmelidir. Bu bağlantı imitation learning'de de önemlidir: öğrenicinin kendi eylemleri uzman gösterimlerindekinden farklı durumlara götürebilir. [Ross, Gordon ve Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

**Ayırt edilemezlik önermesi:** `W₀`da engel daima kapalı, `W₁`de bir tarihten sonra açık olsun. Kaçınma her ikisinde aynı sabit gözlemi üretsin. Yalnız kaçınan herhangi bir öğrenicinin iki dünyadaki gözlem geçmişi dağılımı aynıdır. Bu geçmişten değişimi güvenilir biçimde belirleyemez. İçsel olarak "değişmiş olabilir" tahmini üretebilir; o tahmin yeni dış kanıt değildir.

Bu kilidi açmak için iki dünyada farklı sonuç dağılımları veren erişilebilir bir müdahale veya yeni gözlem kanalı gerekir. İki hipotez `h,h'` için bir eylemde

\[
\operatorname{TV}(K_h(\cdot\mid a),K_{h'}(\cdot\mid a))>0
\]

olması böyle bir ayrımın en basit biçimidir. Farklılık sıfırsa eylem ne kadar sıra dışı olursa olsun bu iki hipotezi ayırmaz. Tek adım yetmiyorsa erişilebilir eylem dizileri değerlendirilir. Pozitif fark, sonlu gürültülü örnekte kesin tanımlama garantisi de değildir.

"Öğrenilmiş sınırı aşabilme" için teknik gereklilik, ilgili koşullarda **yanlışlama olanağına erişimin tamamen kaybolmamasıdır**. Bu, aynı tehlikeli eylemi körlemesine tekrar etmeyi gerektirmez; başka bir ölçüm, güvenilir dış sonuç veya kalibrasyon da yeterli olabilir. Güvenli hiçbir ayırt edici yol yoksa hem hiç risk almama hem mutlaka keşfetme garantileri birlikte verilemez.

Bu yeni bir isimle sahiplenilebilecek fikir değildir. Dual control, eylemin hem görev sonucunu hem gelecekteki bilgi durumunu değiştirmesini ele alır. Klenske ve Hennig'in basit örneğinde ihtiyatlı kontrolün bilgi sağlayan eylemleri azaltıp öğrenmeyi durdurması açıkça gösterilir. Bizim deney, aynı tür sorunun sonlu ve kolay denetlenebilir bir örneğidir. [JMLR 2016, §3.1](https://www.jmlr.org/papers/volume17/15-162/15-162.pdf).

## 3. Yeni kapalı döngü deneyi

[Kod](../../../research/living_learning/closed_loop_experiment.py) ve [tam sonuçlar](../../../research/living_learning/results/closed_loop.json) standart Python ile üretildi. Her yaşam 240 adım sürer. Bütün politikalar aynı başlangıç deneyimini edinmiştir: `t=0`daki bir başarısız deneme. Sonraki dış gözlem her zaman aynı semboldür. **Beklemek yeni geri bildirim üretmez.** Eylem gerçekleştirildiğinde başarı/başarısızlık gözlenir. Başarı +1, başarısızlık `−c`, bekleme 0 değerdedir. `c=1` ve `c=20` ayrı koşullardır. İlk ortak başarısızlığın maliyeti bütün karşılaştırmalarda ortak geçmiş bedeli olduğu için sonraki toplamdan çıkarılır.

Gizli dünya koşulları: hiç açılmayan engel; her adım `.01` olasılıkla açılıp açık kalan engel; aynı açılma tarihinden sonra yalnız sekiz adım açık kalan engel. Sonuncusu planlayıcının kalıcı-açılma varsayımını bozan negatif kontroldür. Açılma tarihi politikaya verilmez. 200 eşleştirilmiş seed'de aynı dünya ve maliyetler bütün politikalar için kullanılır; 181 yaşamda açılma 240 adım içinde gerçekleşir.

| Politika | Yeniden denemeyi ne sağlar? | Başarı sonrası ne olur? |
|---|---|---|
| `frozen_failure` | Hiçbir şey; eski başarısızlık eylemi kapatır | Başarıya erişemez |
| `internal_timer` | Son başarısızlıktan itibaren iç sayaç 20'ye ulaşır | Her adım yararlı eylemi yapar; yeni başarısızlık olursa sayaç yeniden başlar |
| `random_retest` | Başarı bilinmiyorken her adım bağımsız `.05` deneme olasılığı | Aynı güncelleme |
| `irrelevant_motion` | İç sayaç aynı sıklıkta farklı ama engel hakkında bilgi vermeyen hareket üretir | Engel hakkında bilgi edinmez |
| `belief_planning` | Zamanla değişen model inancı ve gelecekteki fayda hesabı | Aynı başarı/başarısızlık güncellemesi |

Sayaçlı ve rastgele politika yalnız başarısız evrede beklenen deneme aralığı bakımından eşlenmiştir; gerçekleşen maliyetleri eşit değildir ve ayrıca raporlanır. Öğrenilmiş başarı durumundan sonra her adım denemek bilgiyi gelecekteki davranışa geçirir. Dış sembol değişmemesine rağmen iç sayaç farklı eylem üretebildiği için biyolojik kimyayı taklit etmeyen deterministik bir varlık örneği vardır. Sayaç tasarımcı tarafından verilmiştir; sistem kendi yeniden sınama yordamını öğrenmemiştir.

### Gelecek bilgi durumunu hesaba katan politika

Son başarısızlıktan beri `d` adım geçtiyse, verilen geometrik değişim modelinde `p_d=1−(1−h)^d`. Kalan adım `n` için:

\[
Q_{\mathrm{dene}}(n,d)=p_d n+(1-p_d)(-c+V(n-1,1)),
\]
\[
Q_{\mathrm{bekle}}(n,d)=V(n-1,d+1),\qquad
V(n,d)=\max\{Q_{\mathrm{dene}},Q_{\mathrm{bekle}}\},\quad V(0,d)=0.
\]

Başarı, modelde engelin kalıcı açıldığını gösterdiği için kalan `n` adımın her birinden +1 alınabilir. Başarısızlık inancı yeniden kapalıya döndürür. Bu Bellman hesabı gözlenmemiş etiket kullanmaz; olayın iki olası sonucunu ve sonraki kararların onlardan nasıl yararlanacağını hesaba katar. Ayrı bir keyfi "merak puanı" eklenmemiştir. Hesaplanan politika, bu **verilmiş** sonlu model ve ödüller için optimaldir. Planlama 28.920 inanç-durum hücresi kullanır; gerçek CPU maliyeti çevresel fayda hesabına dahil değildir. Sonsuz veya bilinmeyen dünyada optimalite iddiası yoktur.

İçsel olarak değişen `p_d` yeni ölçüm değildir; zaman ve varsayılan değişim modeli altında bir öngörüdür. Eski bilginin doğrulanması/çürütülmesi ancak denemenin gerçek sonucuyla gelir. `h=0` varsayılsaydı ve başka bilgi yolu olmasaydı bu modelde yeniden denemek faydalı olmayacaktı. Yanlış sıfır değişim varsayımı bir öğrenme kilidi oluşturabilir.

## 4. Ölçülen sonuçlar

Aşağıdaki faydalar 200 yaşamın ortalamasıdır. Keşif oranı yalnız 240 adım içinde gerçekten değişen 181 yaşam üzerinden hesaplanır.

| Politika | Kalıcı değişim, c=1: fayda / keşif | Kalıcı değişim, c=20: fayda / keşif | Değişim yok: fayda c=1 / c=20 |
|---|---:|---:|---:|
| Eski başarısızlığa bağlı kal | 0 / %0 | 0 / %0 | 0 / 0 |
| Deterministik iç sayaç | 140,695 / %100 | 66,405 / %100 | −12 / −240 |
| Rastgele yeniden deneme | 130,565 / %97,24 | 50,290 / %97,24 | −11,695 / −233,900 |
| İlgisiz davranış değişikliği | 0 / %0 | 0 / %0 | 0 / 0 |
| Model ve gelecek sonuçlarla planlama | 141,395 / %96,69 | 104,025 / %93,37 | −16 / −60 |

İç sayaç 240 olası kalıcı açılma tarihinin **tamamında** ayrıca tarandı; her değişimi en fazla 19 adım gecikmeyle fark etti. Bu garanti ufkun 20'nin katı olmasına ve açılmanın kalıcı kalmasına bağlıdır. Rastgelelik bu garantinin nedeni değildir. Daha fazla keşif her zaman daha iyi karar da değildir: yüksek maliyette planlayıcı bazı geç fırsatları kaçırmayı göze alıp daha yüksek fayda sağlar.

Geometrik dünya için yalnız örneklem ortalamalarına güvenilmedi. Bütün `τ=1,…,240` açılma tarihleri ve ufuk sonrasına kalan olasılık tam toplandı. Planlayıcının hesapladığı değer ile bu bağımsız ortam taraması `10⁻⁸` toleransta aynı çıktı:

| Başarısızlık maliyeti | Bellman değeri | Bütün açılma tarihlerinden beklenen değer | Sayaçlı politikanın beklenen değeri |
|---|---:|---:|---:|
| 1 | 137,855843 | 137,855843 | 136,831575 |
| 20 | 100,032756 | 100,032756 | 59,138433 |

Bu 200 seed ortalamalarının tam değerlerden farklı olması örnekleme farkıdır. Model dışında bir genel üstünlük kanıtı değildir. "Hiç değişmiyor" koşulu sonlu ufukta modelin kuyruğuyla aynı gözlemlere sahiptir; bu koşulu tek başına sürekli tekrarlamak dağılımı değiştirir. O koşulda zararlı deneme yapmak, tüm geometrik dağılımdaki beklenen optimaliteyi çürütmez, model duyarlılığını gösterir.

### Negatif sonuç: geçici fırsatlar

Engel yalnız sekiz adım açık kalınca `c=1` için ortalama fayda; sayaçta **−10,225**, rastgelede **−9,945**, planlayıcıda **−13,910** olur. Planlayıcı değişimlerin %50,83'ünü saptasa da keşif maliyeti elde edilen ödülü aşar. `c=20`de değerler sırasıyla **−231,860**, **−230,250**, **−59,735**tir. Hiç denememenin faydası 0'dır.

Dolayısıyla "sapma yeni bilgi üretti, öyleyse yararlı öğrenmedir" çıkarımı elenir. Bir bilginin değerini onun sayısı, sürprizi veya davranış farklılığı tek başına belirlemez. Kalıcı-açılma modeline göre optimal politika, geçici-açılma dünyasında iyi politika olmak zorunda değildir. Bu stres koşulu ilk çalıştırmadan önce belirlenmiştir; olumsuz sonuç gizlenmedi.

Geçici ortamda başarıdan sonra yeniden başarısızlık gözlenmesi kalıcı-açılma modelinin sıfır olasılık verdiği bir olaydır. Uygulama bu durumda başarı durumunu geri çekip yeni bir kapalı-durum döngüsü başlatır. Bu açıkça tanımlanmış bir yanlış-model tepkisidir; aynı emici model altında tutarlı Bayes güncellemesi veya genel model keşfi değildir.

## 5. Tek deneyim, taklit ve kendi kapasitesi: küçük ayırt edici kontroller

**Bir olayın büyük etkisi olabilir.** Verilmiş bir modelde yüksek-riskli sınıfın önseli `.05`, zarar olasılıkları `.9` ve `.01` olsun. Başlangıç zarar olasılığı `.0545`tir. Bir zarar gözlemi sonrası yüksek-riskli sınıfın olasılığı `.825688`, bir sonraki aynı sınıftaki nesnenin zarar olasılığı yaklaşık `.744862` olur. Başarı faydası 1 ve zarar bedeli 9 ise eylem eşiği `.1`dir: tek olay eşiği geçirir ve gelecekteki karar değişir. Bu basit Bayes hesabı kodda doğrulandı. Öğrenme büyüklüğü ile olay sayısı doğrusal olmak zorunda değildir. Sınıf benzerliği, olasılıklar ve kayıp verilmiştir; çocukların sobayı nasıl öğrendiğine ilişkin deneysel açıklama değildir.

**Taklit son karar değildir.** İki gözlenebilir bağlam ve iki eylem için dört hazır davranış şablonu kullanıldı: hep 0, hep 1, bağlamla aynı, bağlamın tersi. Tek doğru gösterim `(bağlam=0, eylem=0)` iki şablonu uyumlu bırakır. İlkini seçen öğrenici kendi başka-bağlam denemesinde başarısız olur. İkili eylem ve tek doğru eylem varsayımıyla bu başarısızlık yanlış şablonu eler. Yeni nesne adlarıyla 20 sorguda güncellenen davranış **20/20**, ilk taklidi donduran davranış **10/20** doğrudur. Yeni nesne adları yeni özellik değerleri değildir. Bu yalnız küçük bir varlık örneğidir; görsel taklit, hareket koordinasyonu veya geniş önbilgisiz tek-atış yetenek öğrenimi değildir. Örnek, doğru gösterimden sonra da kendi eylem sonuçlarının öğrenmeyi değiştirebildiğini ayırır.

**Eski başarısızlık nedenini tek başına söylemez.** Sonuç `başarı=1[kapasite≥zorluk]` olsun. `(kapasite=0,zorluk=1)` ve `(kapasite=1,zorluk=2)` on tekrar boyunca aynı başarısızlıkları üretir. Bu kayıtlar hangi nedenin sorumlu olduğunu ayıramaz. Bağımsız olarak zorluğu 1 olduğu bilinen kalibrasyon görevinde ilk sistem başarısız, ikincisi başarılı olur. Ayırım ancak ek bilgiyle sağlanır. Kalibrasyonu kendiliğinden edinilmiş gibi saymadık; gerçek kapasite değişimini öğrenen gelişimsel bir sistem çalıştırılmadı.

## 6. Sürekli yeniden sınama neyi garanti edebilir?

Kalıcı değişimde, ayırt edici denemeler arasındaki aralık en fazla `K` ise keşif gecikmesi en fazla `K−1` olur. Bedeli, değişim hiç yokken de en az yaklaşık `T/K` denemedir. Denemeler zarar verebiliyorsa her dünyada hem sıfır zarar hem sınırlı keşif gecikmesi istenemez.

Seyrekleşen deterministik zamanlar (`2,4,8,…`) sonsuza dek kalıcı bir değişimi sonunda sınayabilir; bekleme süresi yaşla büyür. "Her eylemin olasılığı pozitiftir" tek başına yeterli değildir: deneme olasılıkları toplamı sonluysa hiç yeniden denememe olasılığı pozitif kalabilir. Bağımsız Bernoulli denemelerde uygun koşullarla `Σ p_t=∞` sonsuz yeniden denemeye izin verir; sonlu beklenen gecikme veya sonlu toplam zarar bundan çıkmaz. Bu teoremler yeni deney sonucu değildir, seçilen programların sınırlarını açıklayan matematiksel ayrımlardır.

Sekiz adımlık geçici fırsat, 20 adımlık deterministik aralığın arasına tamamen sığabilir. Bağımsız `.05` denemede böyle bir pencereye en az bir giriş olasılığı `1−.95^8≈.337`dir; bu da garanti değildir. Ortamın eylem zamanlarını görerek karşıt davranabildiği dünyada daha güçlü varsayımlar gerekir. "Rastgelelik gereksizdir" sonucu bütün ortamlara değil, bu kalıcı-değişim varlık örneğine aittir.

Öğrenilmiş bilginin yeniden sınanabilir olması için yalnız güven skoru yeterli değildir. Hangi koşullarda geçerli olduğu, onu hangi gözlemin çürüteceği, o gözleme hangi eylemle ulaşılabileceği ve çürüme halinde hangi davranışın güncelleneceği hesaplanabilir kalmalıdır. Bunlar tek bir veri yapısı zorunluluğu değil, test edilebilir işlevlerdir. Hiçbir erişilebilir güvenli deney iki olasılığı ayırmıyorsa yeniden sınama sorusu daha fazla düşünmekle çözülemez.

## 7. Bu ekin araştırmayı değiştirdiği yer

Önceki `deneyim → kalıcı durum → yeni görev performansı` tanımı başarılı öğrenmeyi ölçmeye devam eder; fakat yaşam mekanizmasını tamamlamak için `davranış → sonraki deneyimlere erişim` oku da gereklidir. Öğrenici, değişimi saptamayı engelleyen kendi veri toplama düzenini de değerlendirmelidir. Bunu ayrı bir metacognition modülü diye adlandırmak çözüm değildir.

İç durum değişimi, uygun durumlarda bu döngüyü yeniden açabilecek bir hesaplama kaynağıdır. **Sapma, bilgi edinmeyi mümkün kılan bir müdahale olduğunda öğrenmenin işlevsel parçası olabilir; her öğrenme için zorunlu veya her sapmada faydalı değildir.** Bütün bu özellikleri aynı genel Cevahir yaşamında, bilinmeyen temsil diliyle ve sınırlı bellek altında güvenilir biçimde birleştirmek hâlâ gösterilmemiştir.

Yeniden üretim: repository kökünde `python -m research.living_learning.closed_loop_experiment`. Çalıştırma yaklaşık 0,56 saniye sürdü. Kod ve JSON kaynak özeti, bütün bölüm sonuçları, örnek eylem izleri, kesin toplama denetimi ve negatif koşulları saklar. Verilen fayda çevreyle etkileşim maliyetidir; süre/GPU/RAM benchmark'ı değildir.
