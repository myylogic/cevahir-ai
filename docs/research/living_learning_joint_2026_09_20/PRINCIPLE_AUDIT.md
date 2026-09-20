# Yaşarken öğrenme: atıf, ortak yaşam ve ilke iddiasının denetimi

20 Eylül 2026. Önceki [geniş kapsam raporu](../living_learning_reassessment_2026_09_20/REPORT_TR.md) ve [matematiksel ayrımlar](../living_learning_reassessment_2026_09_20/SCOPE_AND_FOUNDATIONS.md) korunur. Bu belge yeni bir deney başarısı bildirmez. Aşağıdaki önermeler ve karşı örnekler analitik; literatür eşleşmeleri birincil kaynaklara dayanır. Bu turdaki deneylerin neyi gösterebileceği ayrıca sınırlandırılır.

**Ana problem çözülmedi.** Deneyimin kalıcı durumu değiştirmesi gerekli bir açıklamadır, fakat hangi değişimin neden öğrenme olduğu ve nasıl edinileceği sorusuna tek başına cevap değildir. Buradaki ilerleme, verilmiş deneyim kimliğinin kaldırılmasını bilgi yokluğu ile karıştırmamak; tahmine katkı ile nedensel sorumluluğu ayırmak; ayrı modüllerdeki başarının aynı yaşamda birleşeceğini varsaymamaktır.

## 1. Sonucun bir eski olaya bağlanması her zaman doğru hesaplama modeli değildir

Bir çıktı birkaç eski eylemin veya gözlemin birleşik sonucu olabilir. Örneğin

\[
y_t=\sum_{k=1}^{L} a_k x_{t-k}+\xi_t.
\]

Burada her `y_t` için bir adet doğru olay kimliği bulunması gerekmez. Öğrenilecek şey, gecikmiş girdilerin birlikte oluşturduğu ilişkidir. `z_t=(x_{t-1},...,x_{t-L})` geçmiş özelliği tutulursa, `y_t=θ_*^T z_t+ξ_t` doğrusal ilişkisi kurulabilir. Görev adı veya geri bildirimin ait olduğu bölüm kimliği verilmeden de bu ilişki öğrenilebilir. Ancak sıralı gözlemler, zaman hizası, gecikme üst sınırı ve uygun özellik ailesi hâlâ verilmiştir. Bunlar kaldırılmış varsayımlar gibi sunulmamalıdır.

**Önerme 1 — Kimliksiz akışta koşullu ilişki edinimi.** `G_n=Σ_{t≤n} z_tz_t^T`, `b_n=Σ_{t≤n}z_ty_t`, sabit `λ>0` ve

\[
\widehat θ_n=(λI+G_n)^{-1}b_n
\]

olsun. Her gerçekleşmiş akışta tam eşitlik

\[
\widehat θ_n-θ_*=(λI+G_n)^{-1}
\left(\sum_{t≤n}z_t\xi_t-λθ_*\right)
\]

ve dolayısıyla

\[
\|\widehat θ_n-θ_*\|_2\le
\frac{\|\sum_{t≤n}z_t\xi_t\|_2+λ\|θ_*\|_2}
{λ+λ_{\min}(G_n)}
\]

geçerlidir. Eğer yeterince büyük `n` için `λ_min(G_n)≥κn`, `κ>0` ve `||Σz_tξ_t||=o(n)` ise katsayı hatası sıfıra gider. **İspat:** `b_n=G_nθ_*+Σz_tξ_t` yerine konur; pozitif tanımlı ters matrisin operatör normu kullanılır. Yakınsama doğrudan sağ taraftan çıkar.

Bu, “iç durum değişir” ifadesinden daha yapıcıdır: saklanacak nicelikleri, güncellemeyi ve onu başarılı kılan gözlenebilir ayrışma koşulunu verir. `G,b` her örnekte güncellenebilir; tam geçmiş örneklerini saklamak gerekmez. Basit uygulama `O(L²)` sayı saklar; matrisi her seferinde sıfırdan terslemek yerine RLS ile güncelleme `O(L²)` aritmetik işlemle yapılabilir. Bu kaynaklar gecikme sayısı ile artar; sınırsız tarih problemi çözülmüş değildir.

Bu önermede gürültüye ilişkin `o(n)` koşulu açık varsayımdır. Uygun moment koşulları ve geçmişe göre sıfır ortalamalı hatalar onu sağlayabilir; fakat geri beslemeli her gerçek süreçte kendiliğinden geçerli değildir. Katsayıları tam ayırmak için verilen alt özdeğer koşulu yeterlidir; bütün tahmin problemleri için zorunlu değildir. Gelecek sorgular yalnız gözlenmiş alt uzayda kalacaksa tek tek katsayılar belirsizken tahmin yeterli olabilir.

Tek gözlem yolu üzerinde doğrusal sistem belirleme için sonlu örnek garantileri literatürde vardır. Simchowitz ve diğerlerinin teoremleri belirli doğrusal dinamikler, süreç gürültüsü ve uyarılabilirlik koşulları kullanır; bağımsız eğitim bölümlerinin zorunlu olmadığını gösterir. Yukarıdaki basit cebirsel sınır onların bütün teoreminin yeniden üretimi değildir. [Simchowitz ve diğerleri, 2018](https://proceedings.mlr.press/v75/simchowitz18a.html).

**Karşı örnek 1 — Çok veri, ayrım yok.** Her gözlenen anda `x_{t-1}=x_{t-2}` olsun. `θ=(1,0)` ve `θ'=(0,1)` aynı çıktıları verir. Akış ne kadar uzarsa uzasın yalnız bu koşullarda hangi gecikmenin etkili olduğu ayırt edilemez. Sonradan `z=(1,0)` gelirse tahminleri ayrılır. Genel biçimde, bütün eski `z_t` için `z_t^Tv=0` olan sıfırdan farklı `v`, `θ` ve `θ+v` arasında gözlemsel eşdeğerlik oluşturur. Bu örnek, kontrollü bir öğrenme sisteminin imkânsızlığını değil, belirli geçmişten çıkarılabilecek bilginin sınırını gösterir.

**Deney yükümlülüğü:** Kimliksiz gecikme deneyinde iyi sonuç, doğru geçmiş ilişkilerinin hazır episode ID olmadan edinilebildiğini gösterebilir. Gizli nedensel grafiğin, sınırsız gecikmenin veya doğru temsil dilinin bulunduğunu göstermez. Bağımsız girdili olumlu koşulun yanında, yüksek korelasyonlu girdiler ve gecikme aralığı dışındaki sonuçlar karşı örnek olarak tutulmalıdır. Rastgeleliğin zorunluluğu çıkarılamaz; uygun deterministik girdi dizileri de ayrıştırıcı olabilir.

## 2. Tahmine katkı, müdahale sonucu ve tek olayın sorumluluğu farklı sorulardır

Bir geçmiş özelliğinin gelecek çıktıyı iyi tahmin etmesi, o özelliği değiştirmenin çıktıyı değiştireceğini kanıtlamaz. En küçük tanık:

| Dünya | Yapısal ilişkiler | Normal gözlem | `do(A=1)` altında `P(Y=1)` |
|---|---|---|---|
| D1 | `U` eşit olasılıklı bit, `A=U`, `Y=U` | `A=Y`, iki değer eşit olasılıklı | `1/2` |
| D2 | `U` eşit olasılıklı bit, `A=U`, `Y=A` | Aynı gözlem dağılımı | `1` |

Normal akışın bütün uzunlukları aynı dağılıma sahiptir. Birinde `A` ortak nedenin işaretidir; diğerinde sonucu etkiler. Dolayısıyla yalnız bu gözlemsel akışı kullanan bir yordam iki müdahale cevabını doğru ayıramaz. Bu, türetilmiş bir özdeşlenemezlik karşı örneğidir; yeni bir nedensellik teoremi değildir. Gözlem dağılımı ile müdahale dağılımını ayıran yapısal yaklaşımın doğrudan örneğidir. [Pearl, 2009](https://www.cs.columbia.edu/~blei/fogm/2018F/materials/Pearl2009a.pdf).

Bu yüzden raporda üç farklı kullanım ayrılmalıdır:

1. **Öngörüsel atıf:** Hangi geçmiş değişkenleri sonucu tahmin etmeye katkıda bulunuyor?
2. **Müdahaleye ilişkin atıf:** Hangi eylemin değiştirilmesi sonuç dağılımını değiştiriyor?
3. **Belirli olayın karşıolgusal sorumluluğu:** Bu tek sonuç, şu eski eylem farklı olsaydı yine oluşur muydu?

Birinciye ilişkin katsayı öğrenimi ikinciyi ancak ek yapısal/dışsallık varsayımları altında destekler; üçüncü daha güçlü bir sorudur. Sayısal simülatörde neden mekanizmasını araştırmacının bilmesi, öğrenicinin onu veriden ayırt ettiğini göstermez. Öte yandan her yararlı öğrenme nedensel grafik keşfi gerektirmez: tahmin öğrenmek kendi başına geçerli bir edinimdir. Ana problem bu ayrımlardan yalnız birine indirgenmemelidir.

Zamansal fark öğrenmesi, ardışık tahminler arasındaki farkları geçmiş kestirimlere aktararak temporal credit probleminin bazı biçimlerine yapıcı cevap verir. Bu, doğru episode etiketi taşımayan sıradan gözlem dizilerinden yararlanabilen önemli bir eşleşmedir. Ancak temporal credit ile müdahale etkisini özdeşlemek doğru değildir. [Sutton, 1988](https://jmvidal.cse.sc.edu/library/sutton88a.pdf).

## 3. Ayrı güncellemelerin iyi olması birleşik güncellemeyi iyi yapmaz

**Karşı örnek 2 — Eski ortak duruma göre iki iyi değişim.** Durum `s=(a,b)`, başlangıç `(0,0)` ve gerçek değerlendirme

\[
R(a,b)=(a+b-1)^2
\]

olsun. Bir yordam `a←1.5`, diğeri `b←1.5` önerir. Her öneri başlangıç durumunda tek başına değerlendirilince risk `1→0.25` düşer. İkisi uygulanınca durum `(1.5,1.5)` ve risk `4` olur. Yordamlar ayrı bellek alanlarını yazsa bile ortak çıktı üzerinden etkileşir. Hesaplama sırası değişmez; ikinci önerinin eski duruma göre kabulü sorun yaratır. Bu tam aritmetik karşı örnektir.

Bu sonuç “modülerlik kötüdür” veya “bütün öğrenme tek optimize edicide yapılmalıdır” demez. Yalnız her bileşenin diğerleri dondurulmuşken başarılı olması, aynı yaşayan durumda birleşik başarı kanıtı değildir. Önceki üç deneyin sonuçlarını toplayıp tek sistemin toplam kapasitesi sayamamızın matematiksel bir tanığıdır.

**Önerme 2 — Sonlu yaşamda koşullu bileşim.** Gerçekte uygulanan sıralı durumlar `s_0,...,s_T` ve yaşam boyunca sabit değerlendirme fonksiyonları `R_1,...,R_m` seçilsin. Her adım için ortak bir olay `E_t` üzerinde

\[
R_j(s_t)-R_j(s_{t-1})\le\epsilon_{t,j}\quad\forall j
\]

olsun. Geçmişe koşullu `P(E_t^c|F_{t-1})≤δ_t`, sayısal bütçeler `δ_t` ve `ε_{t,j}` verilsin. O zaman en az `1−Σδ_t` olasılıkla bütün `j` için

\[
R_j(s_T)\le R_j(s_0)+\sum_{t=1}^{T}\epsilon_{t,j}
\]

geçerlidir. **İspat:** Koşullu hata sınırından toplam olasılıkla `P(E_t^c)≤δ_t`; birleşim sınırı bütün olayları birlikte korur; risk artışları teleskopik toplanır. Bağımsız güncellemeler varsayılmaz.

Bu ilke sıfır risk istemez; açık toleransları sonlu yaşamda biriktirir. Fakat tek başına öğrenme algoritması değildir. Hiç güncellemeyen sistem de bozulma sınırını karşılar. Bu yüzden ayrıca yeni kapasite edinimi, servis maliyeti, değişen gerçeğe uyum ve sonraki öğrenebilme ölçülmelidir. Garantinin önkoşulu olan gerçek risk artışı sınırları da bedelsiz değildir: yeniden kullanılan doğrulama verisi üzerinde uyarlamalı aday seçmek, uygun düzeltme olmadan bu koşullu olasılık iddiasını sağlamaz. Değerlendirme hedefi zamanla değişirse eski sabit `R_j`'leri otomatik korumak doğru hedef olmayabilir; önerme o hedef değişimini çözmez.

Ortak yaşam deneyinin görevi yalnız ardışık görevleri tek dosyadan geçirmek değildir. Temsil edinimi, düzeltme ve sıkıştırma aynı kaydı ve aynı kaynak bütçesini paylaşmalı; sonraki güncellemenin girdisi gerçekten önceki güncellemeden çıkmış durum olmalıdır. Her deney aşamasında gizlice eski temiz modele dönmek ortak yaşam kanıtını bozar. Ayrı veriyle ölçülen her becerinin toplamı kadar, etkileşimden doğan başarısızlıklar da raporlanmalıdır.

## 4. Sayısal amaç optimizasyonu öğrenmenin mantıksal zorunluluğu değildir

Öğrenme tanımı “tek bir skaler kaybı en aza indirme” ile özdeşlenirse araştırma daha başta bir mekanizma ailesiyle sınırlanır. Bunun gerekli olmadığını gösteren eski, açık ve koşullu yapıcı örnek vardır.

Sonlu yürütülebilir aday kümesi `H`, sabit gerçek ilişki `h_*∈H`, hatasız gözlem çiftleri `(x_t,y_t)` olsun. Başlangıç `V_0=H`; kalıcı güncelleme

\[
V_t=\{h\in V_{t-1}:h(x_t)=y_t\}
\]

olsun. Sistem yalnız bütün kalan adayların aynı cevabı verdiği sorguları yanıtlasın; diğerlerinde çekimser kalsın. Bir skaler ödülün gradyanı hesaplanmaz. Deneyim, ileride hangi hesaplamaların geçerli kullanılacağını değiştirir.

**Önerme 3 — Koşullu, tutarlı genişleme.** Belirtilen varsayımlarla `h_*∈V_t` bütün zamanlarda kalır; verilen her kesin cevap doğrudur; daha önce cevaplanabilir bir sorgu daha sonra cevaplanamaz veya farklı cevaplı hale gelmez. **İspat:** Gerçek aday hiçbir doğru gözlemle elenmez. Oybirliği varsa onun cevabı da bu oybirliğindedir. `V_t⊆V_{t-1}` olduğu için önceki oybirliği korunur. Ayırıcı deneyimler her yanlış adayı sonunda elerse, sonlu `H` nedeniyle sonlu bir zamanda tek aday kalır; bu süre için önceden sabit bir sınır iddia edilmez.

Bu klasik version-space/candidate-elimination yaklaşımıdır; yeni temel ilke olarak sunulamaz. Mitchell'in özgün çalışması aday dilini ve doğru örnek sınıflandırmasını açıkça varsayar; örneğe doğru kredi atanmasının ayrıca zor bir problem olduğunu da belirtir. [Mitchell, 1977](https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf).

Örneğin sınırları esaslıdır: büyük aday kümesi hesap ve bellek maliyeti doğurur; yanlış bir gözlem gerçek adayı geri alınamaz biçimde silebilir; dünya değişimi önceki oybirliğini yanlış kılabilir; dil dışında yeni yararlı program otomatik oluşmaz. Bu nedenle bu örnek geniş probleme çözüm değil, **optimizasyon biçimi ile öğrenme olgusunun farklı olduğunu gösteren varlık örneğidir**. Bir amaç fonksiyonuyla eşdeğer biçimde yeniden yazılabilmesi de onun çalışması için açık skaler optimizasyon yordamı gerektiği anlamına gelmez.

Öte yandan her kalıcı değişimi “faydalı öğrenme” sayamayız. Yararlı olma, hangi sorularda, hangi maliyet ve toleransla işe yaradığına bağlıdır. Bu değerlendirme tek sayı olmak zorunda değildir: doğru yeni cevaplar, korunan geçerli cevaplar, düzeltme gecikmesi, bellek, işlem maliyeti ve sonraki edinim ayrı raporlanabilir. Hedeflerin nereden geldiğini öğrenmek veya hedefleri yaşam içinde değiştirmek de araştırılabilir; fakat bir sistem kendi değerlendirmesini değiştirerek daha başarılı görünürse, bu tek başına kapasite artışı kanıtı değildir. Değerlendiren ölçütün değişimi ile değerlendirilene ait beceri değişimi ayrı izlenmelidir.

## 5. Literatürün kapattığı ve açık bıraktığı alanlar

| Birincil çalışma | Bu araştırmaya somut eşleşme | Buraya taşınamayacak sonuç |
|---|---|---|
| [Sutton, 1988](https://jmvidal.cse.sc.edu/library/sutton88a.pdf) | Zaman içinde oluşan tahmin farklarıyla artımlı güncelleme ve temporal credit. | Gizli nedenlerin veya bütün yeni temsillerin otomatik keşfi. |
| [Simchowitz ve diğerleri, 2018](https://proceedings.mlr.press/v75/simchowitz18a.html) | Tek bağımlı gözlem yolundan doğrusal dinamik belirleme; bilgi içeriğinin uyarılabilirlik ile ilişkisi. | Genel değişen nonlinear yaşam için aynı garantiler. |
| [Mitchell, 1977](https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf) | Deneyimle tutarlı yürütülebilir adayların güncellenmesi; açık skaler optimize edici gerekmeyen edinim örneği. | Gürültü, yapı dışı hedef, yanlış örneğin geri alınması veya bilinmeyen kredi eşlemesi. |
| [Domingos ve Hulten, 2000](https://alchemy.cs.washington.edu/papers/pdfs/domingos-hulten00.pdf) | Veri akışından karar ağacının yeterli istatistikler ve istatistiksel bölünme testleriyle büyütülmesi. | Yeni algı dilinin keşfi; her türlü hedef kaymasının kendiliğinden doğru çözülmesi. |
| [Fahlman ve Lebiere, 1990; teknik rapor sürümü 1991](https://www.cs.cmu.edu/~eugene/refs/f-curves/Fahlman-lebiere-90.pdf) | Deneyimden yeni ara özellikler oluşturup sonraki özelliklere girdi yapmak; mevcut özellikleri dondurarak eşzamanlı değişim sorununu azaltmak. | Donmuş özelliklerin her yeni dünya için uygun kalacağı veya sınırlı bellekte süresiz büyümenin çözüldüğü. |
| [Zeno ve diğerleri, 2018](https://arxiv.org/abs/1803.10123) | Eğitimde görev sınırlarının verilmediği continual learning'in zaten araştırılmış olması. | Görev sınırı yokluğunun; etiket yokluğu, bilinmeyen temporal eşleme ve sınırsız temsil keşfiyle aynı olması. Bu belge yalnız kaynak özetini doğruladı, yöntemi yeniden üretmedi. |

Bu eşleşmeler “ana cevap bu bilinen kavramdır” sonucunu üretmez. Ama başarı iddialarının yeniliği konusunda sınır koyar: görev kimliği kullanmama, artımlı yapı büyütme veya çalışma sırasında istatistik güncelleme tek başına yeni ilke değildir. Yeni katkı aranacaksa, hangi varsayımın kaldırıldığı veya hangi önceki çatışmanın aynı kaynaklarla çözüldüğü gösterilmelidir.

## 6. Yeni, yanlışlanabilir hipotezler ve deney yorum sınırı

**H1 — Olay kimliği yerine öğrenilmiş ilişki yeterliliği.** Doğru sıra ve uygun sonlu geçmiş özellikleri verildiğinde, yeterince ayrıştırıcı bir akışta geri bildirim başına hazır episode ID olmadan yeni tahmin kapasitesi edinilebilir. Bu dar iddia yukarıdaki önerme ile desteklenir. Gecikme desteği veya değişkenler yanlış verildiğinde, aynı güncellemenin başarısız olacağı kontrol edilmelidir. Başarı causal responsibility keşfi diye adlandırılmamalıdır.

**H2 — Ortak durum etkileşimleri, bağımsız skorların açıklayamadığı kayıplar üretir.** Aynı temsili veya bellek bütçesini paylaşan edinim/düzeltme/sıkıştırma, ayrı çalıştırmalarda görülmeyen girişimlere yol açabilir. Matematiksel karşı örnek bu olasılığı kesin gösterir. Yaşam deneyinde ölçülmesi gereken büyüklük, her parçanın skorunu toplamak değil, parçaların birlikte çalışmasının tek bir parçanın dondurulduğu koşula göre yararı ve zararıdır.

**H3 — Hedefe uymayan temsil ile geçersiz geçmiş farklı izler bırakabilir, ancak her zaman ayrılmaz.** Aynı güncel girdide kalıcı tutarsızlık ile yeni bölgede sistematik hata farklı aday değişiklikleri gerektirebilir. Bunun her dağılımda ayırt edilebilir olduğu iddia edilmemeli. Deneyde hem hedefin gerçekten değiştiği hem yalnız ziyaret dağılımının değiştiği yaşamlar kullanılmalı; aynı saklama/güncelleme bütçesinde yanlış düzeltme ve düzeltmeme ayrı ölçülmelidir. Bu hipotez burada kanıtlanmadı.

**H4 — Bir kapasite artışının kanıtı, mekanizmanın kendi ölçütünden bağımsız olmalıdır.** Temsil ve kabul eşiği değiştiren aday, kendi eğitim uyumunu artırabilir ama görülmemiş devam görevlerinde kötüleşebilir. Aynı öğrenilmiş duruma karşı yeni girdi, gecikmiş geri bildirim, ilgisiz akış, düzeltme ve yeniden başlatma kontrolleri bu ayrımı sınamalıdır. Bu, yeni bir amaç fonksiyonu önerisi değil, ölçümün kendi kendini doğrulamamasını isteyen yöntemsel koşuldur.

Bu turda bölüm kimliği olmadan uzun bir akışta birkaç mekanizmanın birlikte çalışması gösterilirse, bu önceki ayrı düzeneklere göre gerçek bir ilerlemedir. Yine de giriş dili, izinli işlemler, zaman gözlemi, aday üretim kuralı ve doğrulama sinyali hâlâ araştırmacı tarafından veriliyorsa genel yaşam mekanizması bulunmuş sayılmaz. Başarısızlık da bütün yaşarken öğrenme olasılığını çürütmez; mevcut mekanizma ve deney sözleşmesi hakkında bilgi verir.

**Savunulabilir koşullu edinim ilkesi:** Yukarıdaki kestirim ve tutarlı aday eleme ailelerinde, deneyimin adaylar arasında gerçekten ayrım taşıması ve kalıcı güncellemenin bu ayrımı sonraki hesaplamada kullanması, belirtilen varsayımlar altında genellenen kapasite oluşturur. Açık algoritma ve ispat bu ailelere aittir. Bağımsız değerlendirme ise araştırmacının başarı iddiasını denetleme koşuludur; her öğrenicinin içinde ayrı bir değerlendirici bulunması gerektiği iddiası değildir.

Bu ilke bütün kapasite değişiminin zorunlu biçimi olarak sunulamaz. Önceki deneylerde dış kanıt almadan derleme, aynı bilgiyi dar hesap bütçesinde kullanılabilir kılmıştı; orada yeni dünya hipotezleri ayırt edilmedi. Dolayısıyla bilgi edinimi, hesaplamanın düzenlenmesi ve öğrenme yordamının değişimi birbirine indirgenmemelidir. Bilinmeyen yararlı temsili, geçerli geçmişi ve değişmiş hedefi aynı sınırlı yaşamda birlikte yönetme yükümlülüğü sürer.
