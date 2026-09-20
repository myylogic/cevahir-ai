# Gecikmiş deneyimin doğru yere yazılması ve ortak yapıdaki becerilerin korunması

20 Eylül 2026. Önceki araştırmaya ek deney ailesi; önceki sonuçların yerine geçmez. Ana sistem değiştirilmedi.

**Sonuç:** Çalışırken alınan geri bildirim doğru geçmiş hesaplamayla ilişkilendirilse bile, ortak parametrelerin güncellenmesi hâlâ geçerli eski becerileri bozabilir. Bu deneyde hem küçük bir örnek belleğiyle tekrar işleme hem de ham örnek tutmadan yeterli istatistik biriktirme bu bozulmayı azaltıyor. Aynı koruma mekanizmaları gerçek dünya değişiminde artık yanlış olan eski ilişkiyi koruyarak güncel başarıyı düşürüyor. Bunlar yaşarken öğrenmenin tamamı için çözüm değildir; deneyimden kalıcı değişime giden yolun birbirinden ayrı üç koşulunu ayırır: **ilişkilendirme, bilgi biriktirme ve geçerlilik**.

## Araştırma sorusu ve önceki çalışmayla farkı

Eski affine deneyinde araç kimliği ve ayrı kurallar düzeltmeyi yerelleştiriyordu. Burada bütün beceriler **aynı üç parametreyi** kullanır; yeni bir aracı ayrı bir tabloda açmak yoktur. Geri bildirim aynı anda gelmez. Sistem yeni isteklere cevap verirken eski cevapların etiketlerini alır ve hemen güncellenir. Çalışma sonrasında ayrı eğitim veya toplu yeniden öğrenme yapılmaz.

Sınanan hipotezler:

1. Geri bildirimi geldiği andaki girdiye bağlamak ile onu üreten deneyime bağlamak eşdeğer değildir.
2. Doğru ilişkilendirme, geçerli geçmiş becerinin korunması için tek başına yeterli değildir.
3. Ham deneyimleri tekrar işlemek korumaya yardımcı olabilir; zorunlu olduğu sonucu çıkmaz.
4. Geçmişi daha iyi korumak her durumda daha iyi öğrenme demek değildir.

## Önceden belirtilen düzenek ve erişim sözleşmesi

32 eşleştirilmiş seed (`0..31`), yaşam başına 1.200 istek; ilk ve sonraki 600 istekte farklı girdi dağılımları. Öğreniciye bu sınır, görev etiketi veya dünya rejimi verilmez. Tek istisna olan `frozen_after_old`, eski becerinin ediniminden sonra donmanın sonucunu ölçen, sınırı bilen tanısal kontroldür.

Her istek `x=(1,u,v)`; `u` kesintisiz düzgün `[-1,1]` dağılımından gelir. Ana dünyada ilk dönemde `v=0`, ikinci dönemde `v=u`. Bütün yaşamda doğru ilişki sabittir:

\[
f(x)=0.3+1.2u-1.7v.
\]

Ortak doğrusal tahminci `wᵀx`, başlangıçta üç sıfır katsayıdan oluşur. Her istekte önce cevap verilir, sonra o zamana ulaşmış geri bildirim işlenir. Etiket `y=f(x)+ε`, bağımsız `ε~N(0,0.1²)`; etiket gürültülüdür, fakat bütün değerlendirmeler temiz doğru değere karşı MSE hesaplar. Bu, seyrek ödülden eylem dizisi öğrenme değil, sayısal denetimli geri bildirimdir.

Ana koşulda gecikme `1..31` arasından eşit olasılıkla seçilir; geliş sırası sıkça bozulur. Geri bildirim **doğru episode ID'siyle** gelir. Öğrenici en fazla 64 bekleyen girdiyi ID→x kaydında tutabilir; eski kayıt kapasite aşımında çıkarılır. Bu ilişkilendirme dışarıdan sağlanmıştır: deney hangi geçmiş eylemin etkili olduğunu veya nedensel zaman aralığını kendisi bulmuyor. Gradyanın doğru örnekten hesaplanması için yeterli iz, bu sabit modelde `x`'tir. İzin bütünüyle yokluğu ve aynı güncel gözleme sahip farklı geçmişlerin farklı güncelleme gerektirmesi, bir ilişkilendirme problemi yaratır; buradan her öğrenicinin aynı ID sistemine ihtiyaç duyduğu çıkarılamaz.

Bağımsız rastgelelik akışları girdiyi/gürültüyü, gecikmeyi ve örnek belleğini üretir. Böylece anlık-gecikmeli koşullar aynı istekleri ve aynı etiketleri paylaşır. Her seed için bağımsız 1.024 eski, 1.024 yeni ve 1.024 bileşim değerlendirme girdisi üretilir. Son kümede `u,v` birbirinden bağımsızdır; bu girdiler eğitimdeki iki doğrunun üzerinde değildir. Değerlendirme kümesi hiçbir güncellemeye girmez. Buradaki bileşim doğrusal değişkenlerin yeni birleşimidir; program bileşimi veya yeni temsil dili değildir.

Tek öğrenme hızı `.12` ve tek 32-örneklik tekrar belleği kullanıldı. Parametre taraması veya heldout sonucuna göre yöntem seçimi yapılmadı. Kod denetiminde gecikmenin aynı girdi akışını bozmaması için rastgelelik akışları ayrıldı; ilk deneme çıktıları bu nihai protokolün sonuçları olarak sunulmadı.

## Karşılaştırmalar ve maliyet

| Yöntem | Her kabul edilen geri bildirimde işlem |
|---|---|
| `initial_frozen` | Hiç güncelleme yok |
| `frozen_after_old` | İlk dönemde iki doğru örnek güncellemesi; sonra donma |
| `current_two` | Gelen eski etiketi şimdiki girdiye bağlayıp iki güncelleme |
| `anchored_one` | ID ile bulunan geçmiş girdiden bir güncelleme |
| `anchored_two` | Aynı doğru geçmiş örnekten iki güncelleme |
| `anchored_replay` | Bir doğru geçmiş örnek güncellemesi ve 32 kayıtlık reservoir belleğinden bir güncelleme |
| `online_rls` | Online ridge / recursive least squares yeterli-istatistik güncellemesi |

Gradyan adımı, `w←w−.12(wᵀx−y)x/‖x‖²`. Tekrarlanan ikinci adım ilk adımın güncel katsayılarını kullanır. Reservoir örneklemesi geçmiş kabul edilen örneklerden sabit boyutlu örneklem tutar; görev etiketine bakmaz. Belleğe gelen örnek dahil edildikten sonra bir örnek çekildiği için bazen yine o örnek işlenebilir.

`current_two`, `anchored_two` ve `anchored_replay` **aynı sayıda kabul edilen etiketi** ve tam aynı iki gradyan değerlendirmesini kullanır: ana koşulda ortalama 2.368,3125. Yanlış ilişkilendirme kontrolüne fazladan etiket avantajı vermemek için bekleyen kayıt kabul filtresi onda da uygulanır. Tekrar kullanılan örnek güncellemesinin avantajı yalnız daha fazla gradyan işleminden gelmez. Reservoir erişimi ve rastgele seçim maliyeti vardır; bu, tam FLOP veya duvar saati eşitliği iddiası değildir.

Bekleyen iz kaydı ID ve üç girdi skalarıdır. Replay kaydı üç girdi skaları ve etikettir; en fazla 32 kayıt. Diğer gradyan yöntemleri bu kapasite sınırını aşmaz ama replay kullanmaz. RLS üç katsayıya ek olarak dokuz skalar içeren ters Gram matrisini tutar; ham etiketli örnek biriktirmez. RLS güncellemesi `O(d²)` olup gradyanla FLOP eşleştirilmemiş **güçlü istatistik referansıdır**. Bütün modeller aynı sabit temsil ve aynı kabul edilmiş kanıtla çalışır. RLS'nin iyi sonucu, yalnız zayıf rakibe karşı replay üstünlüğünden genel ilke çıkarılmasını engeller.

RLS'nin biriktirdiği içerik `G=I+Σxxᵀ`, `b=Σxy` ve `w=G⁻¹b` ile ifade edilir; uygulama eşdeğer `G⁻¹,w` durumunu taşır. Her yeni etiket geldiğinde Sherman–Morrison güncellemesiyle aynı sonlu veri ridge çözümü elde edilir. [Cothren ve diğerleri, 2022, §3.2](https://proceedings.mlr.press/v168/cothren22a/cothren22a.pdf) bu bilinen recursive least squares / ridge araçlarını özetler; onların kontrol sistemi burada uygulanmaz. Üç örneklik bağımsız cebir denetiminde `G⁻¹G=I` hatası en fazla `2.22e−16`, `w=G⁻¹b` hatası `1.39e−16` bulundu. Bellek rakamları algoritmanın ihtiyaç duyduğu skalarları belirtir; Python nesne başlıkları veya deney sürücüsünün ortak izleme kopyaları ölçülmüş değildir.

## Çalıştırılmış sonuçlar

Ana gecikmeli dünyada yaşam sonu temiz heldout MSE; 32 seed ortalaması:

| Yöntem | Hâlâ geçerli eski girdiler | Yeni girdiler | Görülmemiş değişken birleşimleri |
|---|---:|---:|---:|
| Hiç öğrenmeyen | 0.562479 | 0.174732 | 1.536633 |
| Eski edinimden sonra donan | 0.000905 | 0.975463 | 0.958291 |
| Şimdiki girdiye yazan | 0.495545 | 0.101597 | 1.494124 |
| Doğru girdiye bir adım | 0.231043 | 0.000618 | 0.466139 |
| Doğru girdiye iki adım | 0.216505 | 0.001233 | 0.434709 |
| Doğru girdiye + replay | 0.000957 | 0.000767 | 0.001725 |
| Online RLS | 0.000070 | 0.000047 | 0.000241 |

Eski beceri `anchored_two` için öğrenim sonundaki **0.000905'ten 0.216505'e** bozulur. Hedef değişmediği ve tek gerçek `w=(.3,1.2,−1.7)` her iki dağılımda da çalıştığı için bu fark çelişen doğru cevapların kaçınılmaz bedeli değildir. Replay ile aynı artış `0.000521→0.000957` olur; sıfır unutma veya tekil her yaşam için garanti yoktur.

Eşleştirilmiş farkların iki taraflı %95 Student-t aralıkları, 31 serbestlik derecesiyle:

- Yeni hata, yanlış ilişkilendirme eksi doğru ilişkilendirme: **0.100364 [0.086869, 0.113858]**.
- Eski hata, doğru iki adım eksi replay: **0.215548 [0.208600, 0.222497]**.
- Yeni hata, doğru iki adım eksi replay: **0.000466 [0.000074, 0.000859]**. Bu küçük ortalama fark, replay'in her seed'de yeni görevi iyileştirdiği demek değildir.

Anlık geri bildirim koşulunda `current_two` ile `anchored_two` bütün seed'lerde **tam aynı katsayılara** ulaşır; kod bunu doğrular. Dolayısıyla gecikmeli koşuldaki bu karşılaştırma ilişkilendirme farkını hedefler. Gecikmenin kendisi yine maliyetlidir: RLS'nin ikinci dönem servis boyunca ortalama hatası anlık `.006461`, gecikmeli `.022569`.

**Kazanımın bedeli:** İkinci dönem boyunca cevap verme hatası replay için `.051818`, iki doğru adım için `.034694`. Finalde eski ve yeni başarının birlikte daha iyi olması, yeni dağılıma ilk uyumun daha hızlı veya bütün yaşam toplamının daha iyi olduğunu göstermez. Hangi risk ve zaman ufkunun korunacağı problem sözleşmesinin parçasıdır.

Gecikmeli ana koşulda ortalama 1.184,156 geri bildirim yaşam sonuna yetişir, hiçbiri iz kaybından atılmaz; ortalama 462,25 geliş sırası tersliği gözlenir. Son 15,844 geri bildirim henüz ulaşmamıştır; deney bitince kuyruğu boşaltıp fazladan öğrenme yapılmaz.

## Negatif rejimler ve matematiksel ayrımlar

**Güncellemeyi doğru örneğe bağlamak korumayı sağlamıyor.** Gürültüsüz anlık sınırda ilk dönem ideal olarak `w₁=1.2,w₂=0` edinilir. İkinci dönemde `v=u` olduğundan her gradyan bu iki katsayıyı eşit değiştirir; `w₁−w₂=1.2` değişmezi korunur. Yeni verinin istediği `w₁+w₂=−.5`, sonuçta `w₁=.35,w₂=−.85` verir. Eski popülasyon hatası `.85²/3≈.240833`, bağımsız bileşim hatası `2·.85²/3≈.481667` olur. Anlık deneyin `.241632` ve `.482351` sonuçları bu analitik mekanizmayla uyumludur. Bu limit sonlu, gürültülü deneyin kesin eşitlik iddiası değildir. Eksik olan ikinci dönemin tek başına belirlemediği eski doğrultudur; ilk dönemden kalan istatistik veya örnekler onu kısıtlayabilir.

**Bellek ve gecikme uyumsuzluğu:** Gecikme `1..127`, bekleyen iz kapasitesi dört olunca ortalama yalnız **30,031** etiket ilişkilendirilebilir; 1.107,219 ulaşmış etiket izleri silindiği için atılır. Replay bileşim MSE'si `.827401`, RLS `.214673`; ana koşuldaki kazanım kaybolur. Replay yeni MSE'yi doğru iki adıma göre `.053602→.134564` kötüleştirir. Bu evrensel dört-kayıt alt sınırı değildir; mevcut iz silme politikası, dış geri bildirim protokolü ve süre altında başarısızlıktır. Geri bildirim girdiyi de taşısaydı farklı bir bellek hesabı gerekir.

**Geçersiz geçmişi koruma:** Çelişkili dünyada bütün girdiler `(1,u,0)` kalırken ikinci dönemde eğim `+1.2→−1.2` değişir; ayırt edici zaman/durum özelliği modele verilmez. Yeni heldout MSE doğru iki adımda **.001106**, replay'de **.181190**, unutmasız RLS'de **.509150**. Replay eksi doğru iki adım farkı `.180084`, %95 aralığı `[.131957,.228211]`. Burada eski hedefe karşı hata artışı tek başına bir başarısız unutma ölçütü değildir: eski ilişki artık yanlıştır. Her aynı x için iki hedef arasındaki fark Δ iken, tek tahminin iki karesel hatasının eşit ağırlıklı ortalaması en az `Δ²/4` olur. Bu dünyada popülasyon alt sınırı `.48`'dir. Koruma politikası, geçerliliği değerlendiren bir mekanizma olmadan kontrolü tamamlamaz. RLS'ye unutma faktörü veya değişim algılaması eklemek mümkündür; bu deneyde uygulanmadı.

**Temsil yanlışlığı:** Gerçek ilişkiye `.8u²` eklenirken model hâlâ üç doğrusal katsayıyla sınırlanır. Simetrik düzgün dağılımda en iyi doğrusal tahminin temiz MSE alt sınırı

\[
.8^2\operatorname{Var}(u^2)=.64(1/5−1/9)=.056888\ldots
\]

olur. RLS bileşim MSE'si **.057802**, replay `.079437`. Daha iyi ilişkilendirme, daha uzun yaşam veya doğru geçmişi koruma bu temsil sınıfının hatasını sıfıra indiremez. Bu, genel yapay zekâya veya belirli Cevahir ağına ilişkin ifade sınırı değildir; seçilmiş doğrusal sınıfın açık sınırıdır.

## Kalıcılık, bilinen yöntemler ve çözülmemiş bölüm

Her koşul/seed/yöntemde yalnız üç katsayı JSON'a yazılıp tekrar okunarak çıplak tahminci yeniden kurulur. Bekleyen iz, replay ve RLS istatistikleri tahminciye verilmez; bütün heldout riskleri tam aynıdır. Bu kontrol **edinilmiş davranışın ham kayıt okumadan katsayılarda taşındığını** gösterir. Aynı işlem içinde serileştirmedir; bağımsız işletim sistemi sürecinde restart değildir. Öğrenmeye devam etmek için gerekli tarih istatistikleriyle, mevcut beceriyi kullanmak için yeterli katsayılar burada farklıdır.

Gecikmiş etiketin doğru geçmiş örneğe bağlanması, yeni bir temel algoritma değil, gecikmeli çevrimiçi öğrenmenin bilinen konusudur. [Joulani, György ve Szepesvári, 2013](https://proceedings.mlr.press/v28/joulani13.html), gecikmenin etkisini farklı çevrimiçi problem sınıflarında inceler; burada o çalışmanın meta-algoritmaları uygulanmadı. Replay'in eski beceriyi desteklemesi [Rolnick ve diğerleri, 2019](https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) ve [Chaudhry ve diğerleri, 2019](https://arxiv.org/abs/1902.10486) ile eşleşir; CLEAR veya bu çalışmaların deneyleri yeniden üretilmiş sayılmaz. RLS standart çevrimiçi en küçük karelerdir; bu ad altında özgünlük iddiası yoktur.

Genel living-learning problemi açısından temel olan, belirli reservoir veya matris güncellemesi değil, **geçmişin gelecekteki değişime hangi kısıtları taşıdığıdır**. Üç farklı şey karıştırılmamalıdır: gelecekte cevap vermek için yeterli durum, gelecekte öğrenmeye devam etmek için yeterli durum ve daha sonra yanlış çıkabilecek varsayımların geçerlilik kaydı. Bu deneyde ilk ikisi somut ayrılır; üçüncüsü çözülmez. Ayrıca öğrenici doğru temsil dilini, öğrenme kuralını ve sayısal etiket anlamını başlangıçtan bilir. Öğrenme yordamını, zamansal nedenselliği veya daha önce ifade edemediği yeni işlemi deneyimden keşfetmemiştir. Bu açık kısımlar ana problemin parçası olarak kalır.

## Yeniden üretim

Kaynak: [credit_assignment.py](../../../research/living_learning_reassessment/credit_assignment.py). Ham seed sonuçları, aralıklar, parametreler, kaynak özeti ve bilimsel sonuç özeti: [credit_assignment.json](../../../research/living_learning_reassessment/results/credit_assignment.json). Tekrar çalıştırma kaydı: [credit_assignment_verification.json](../../../research/living_learning_reassessment/results/credit_assignment_verification.json).

Repository kökünde `python -m research.living_learning_reassessment.credit_assignment`. Yalnız Python standart kütüphanesi gerekir. Tüm sayılar bu dosyadaki koşullara aittir; aralıklar yalnız bu sentetik dünyalar ve seed değişkenliğini kapsar, gerçek sistemlere genellenebilirlik garantisi değildir.
