# Öğrenilebilir temsil altında koruma: matematik ve literatür sınırları

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma turu 10

Bu inceleme [ön kayıt protokolüne](PROTOCOL.md) eşlik eder. **Bu dosya yazılırken ana deneyin istatistiksel sonuçları henüz yoktur.** Aşağıdaki cebirsel tanıklar ve koşullu önermeler, sinir ağı deneyinin sonuçları olarak sunulmaz. Kesir örnekleri geçici bir Python işleminde `fractions.Fraction` ile ayrıca hesaplandı; bu kontrol ağ deneyi veya yöntemlerin başarım karşılaştırması değildir. Önceki araştırma, kitap ve yayın dosyaları değiştirilmedi.

## 1. Devam noktası ve yeni iddianın sınırı

[Önceki bileşim incelemesi](../living_learning_joint_2026_09_20/PRINCIPLE_AUDIT.md), ayrı değerlendirildiğinde yararlı güncellemelerin ortak durumda zararlı olabileceğini zaten gösterdi. [Durum taşıma incelemesi](../living_learning_reassessment_2026_09_20/STATE_TRANSPORT.md), bugünkü fonksiyonun korunmasıyla gelecekteki öğrenme yolunun korunmasını ayırdı; eski temsile yeterli bir özetin yeni temsile yetmeyebileceğini de gösterdi. Bunları yeniden keşfetmiyoruz.

Bu turdaki daha dar soru, öğrenilebilir bir temsilde **korumaya yeterli bilgi ile yeni edinime açık yönlerin birlikte nasıl değiştiğidir**. Bir yöntem eski çıktı değişimini azaltırken, uygun çözümü ifade edebilen aynı ağda yeni becerinin edinimini de durdurabilir. Bunun varlığını veya büyüklüğünü henüz deney sonucu olarak ilan etmiyoruz. Yerel cebir, hangi niceliklerin ölçülmesi gerektiğini belirliyor; başarıyı peşinen belirlemiyor. Ana yaşarken öğrenme problemi bu alt probleme indirgenmiş değildir.

## 2. Sonlu adım için iki tam karşı örnek

### Düzenli bir noktada teğet yön, koruyan sonlu adım değildir

Eski çıktı `f_o(a,b)=ab`, eski doğru hedef `1`, kayıp `L_o=(ab−1)²/2` olsun. Başlangıç `(1,1)` noktasında çıktı Jacobian'ı `(1,1)`dir. Öneri `δ=(h,−h)` için:

\[
J_o\delta=0,\qquad f_o(1+h,1-h)=1-h^2,
\qquad L_o(1+h,1-h)=\frac{h^4}{2}.
\]

Her sıfırdan farklı, ne kadar küçük olursa olsun `h`, eski kaybı artırır. Burada çıktı Jacobian'ı sıfır değildir; başarısızlık yalnız türevin sıfır olduğu tekil bir noktaya bağlanamaz. Aynı ilk mertebe yönünü izleyen eğri

\[
(a(h),b(h))=(1+h,(1+h)^{-1}),\quad h\ne-1
\]

eski çıktıyı tam korur. `b(h)=1−h+h²+…` açılımındaki ikinci ve sonraki mertebeler, düz teğet adımda bulunmayan telafidir. `h=1/2` için düz adımın kaybı `1/32`, eğri üzerindeki kayıp sıfırdır. Bu, koruyan yolun varlığı ile seçilen güncellemenin o yolu izlemesinin farklı iddialar olduğunu gösterir.

### Ayrı güvenli edinimler ve aynı kapasitedeki telafi

Eski çıktı `f_o(a,b,c)=ab+c`, hedef `0`, başlangıç `(0,0,0)` olsun. İki yeni beceri `a=h` ve `b=h` gerektirsin; ilgili kayıplar `(a−h)²/2` ve `(b−h)²/2`dir. `δ_A=(h,0,0)` ve `δ_B=(0,h,0)` kendi becerisini edinir, diğer yeni becerinin kaybını değiştirmez ve eski kaybı ayrı ayrı tam korur. Birleşimde:

\[
L_o(h,h,0)=h^4/2,
\qquad L_o(h,h,-h^2)=0.
\]

İkinci nokta iki yeni hedefi de karşılar. Dolayısıyla bu tanıktaki zarar, üç hedefin aynı parametre kapasitesinde bağdaşmamasından kaynaklanmaz. Başlangıç Jacobian'ı `(0,0,1)` iki öneriyi de kabul eder. Eski kayıp gradyanı sıfırdır; başlangıçtaki kayıp Hessian'ının `(a,b)` düzlemindeki ikinci derece biçimi de sıfırdır. Kayıptaki etkileşim dördüncü derecede ortaya çıkar. Bu, her sonlu mertebeli yöntemin başarısız olduğunu değil, bu noktadaki ikinci derece kayıp modelinin yeterli olmadığını gösterir.

| `h=1/2` durumları | Eski kayıp | Yeni A kaybı | Yeni B kaybı |
|---|---:|---:|---:|
| Başlangıç | 0 | 1/8 | 1/8 |
| Yalnız A | 0 | 0 | 1/8 |
| Yalnız B | 0 | 1/8 | 0 |
| İkisinin toplamı | 1/32 | 0 | 0 |
| Telafili `(1/2,1/2,−1/4)` | 0 | 0 | 0 |

Örnekte telafi araştırmacı tarafından bilinmektedir; sistemin onu deneyimden bulduğu gösterilmez. Ardışık her adımda Jacobian'ı yenilemek, başlangıçta doğrulanmış iki öneriyi birlikte uygulamaktan farklıdır. Bu fark da protokolde sabit ve güncel teğet yöntemlerini ayırmanın nedenidir.

## 3. İlk mertebe erişilebilirliğin gerekli ve yeterli koşulu

Seçilmiş eski girdilerdeki çıktı vektörü `F_o(θ)∈Rᵐ`, yeni girdilerdeki çıktı vektörü `F_n(θ)∈Rᵏ` olsun. Aynı mevcut noktada `A=J_o(θ)`, `B=J_n(θ)` ve Moore–Penrose tersiyle

\[
P=I-A^+A
\]

tanımlansın. `P`, Öklid metriğinde `ker(A)` üzerine dik projeksiyondur. **Önerme:** İstenen ilk mertebe yeni çıktı değişimi `r` için

\[
A\delta=0,\ B\delta=r
\quad\Longleftrightarrow\quad r\in\operatorname{range}(BP).
\]

**İspat:** `Aδ=0` ise `δ=Pδ`, dolayısıyla `r=BPδ`. Tersine `r=BPz` sağlayan `z` varsa `δ=Pz` iki eşitliği sağlar. Uygun `r` için bir çözüm `δ=(BP)⁺r`dir; bu vektör `range(PBᵀ)⊆ker(A)` içindedir. En fazla `d−rank(A)` parametre yönü kalır; erişilebilir yeni çıktı yönlerinin sayısı ise `rank(BP)`dir. Bunlar aynı sayı olmak zorunda değildir.

Yeni kaybın mevcut gradyanı `g` olduğunda, korunmuş teğet uzayda **kesin negatif yönlü türev** bulunması `Pg≠0` ile eşdeğerdir. Varsa `δ=−Pg` için `gᵀδ=−‖Pg‖²<0`; yoksa bütün izinli `δ` için yönlü türev sıfırdır. Bunun önkoşulu aynı noktada hesaplanan türevlerdir. Saklanmış eski Jacobian'a diklik, güncel Jacobian'a diklik değildir.

Bu önerme sonlu veya global öğrenilemezlik teoremi değildir. Örneğin eski çıktı `a`, yeni çıktı `b²`, başlangıç `(0,0)`, yeni hedef `1` olsun. `BP=0` olmasına rağmen `(0,h)` eskiyi tam korur ve `0<|h|<√2` için yeni kare kaybı azaltır. Değişim ikinci mertebedendir. Buna karşı affine çıktı modellerinde Jacobian ifadeleri tam olduğundan erişilebilirlik koşulu sonlu çıktı değişimleri için de geçerlidir. Nonlinear modelde aynı geçiş yapılamaz.

Öğrenilmiş temsilin değişmesi iki ayrı sorun doğurur. Önce, saklanmış türev yönleri mevcut fonksiyonun türevleri olmaktan çıkabilir. Aynı eski girdiler tutuluyorsa bunları yeniden hesaplamak mümkündür; yalnız eski Jacobian tutulmuşsa bu işlem kendiliğinden yapılamaz. İkinci olarak, yeniden hesaplanan güncel türevler bile yalnız o noktadaki ilk mertebeyi taşır. `tangent_current` ilk sorunu azaltan bir kontrol olsa da ikinciyi ortadan kaldırmaz. Sonlu aday denetimi ikinci sorunun ankorlar üzerindeki gerçekleşmesini gözler; geçmişte tutulmamış girdileri geri getirmez.

Temsil değişimi yalnız koordinatların tersinir yeniden adlandırılmasıysa, aynı fonksiyon ailesi farklı Öklid projeksiyonlarıyla farklı güncellemeler üretebilir. Önceki durum taşıma çalışmasındaki metrik ayrımı burada da geçerlidir. Bu dosyadaki `P` açıkça mevcut parametrelerin Öklid metriğini seçer; her parametrizasyonda aynı öğrenme yolunu verdiği iddia edilmez. Bununla birlikte doğrusal sistemin `Aδ=0, Bδ=r` biçimindeki çözülebilirliği uygun tersinir koordinat dönüşümünde korunur. **Bir yönün var olması ile o yönler arasından en yakın olanı seçmek farklıdır.** Deneyin aynı mimari ve başlangıçtan karşılaştırma yapması, bu seçimi kontrol altında tutar; araştırmacının verdiği parametrizasyonu ortadan kaldırmaz.

## 4. Sonlu adımda gerçekten yeterli olan sınır

Eski kaybın gradyanı, `θ` ile `θ+δ` arasındaki doğru parçasını kapsayan bölgede `β`-Lipschitz olsun. Gradyanı doğru boyunca integre ederek:

\[
L_o(\theta+\delta)-L_o(\theta)
\le\nabla L_o(\theta)^T\delta+\frac{\beta}{2}\|\delta\|^2.
\]

Dolayısıyla sağ tarafın sıfırdan büyük olmaması, o kayıpta artmama için yeterlidir. Yalnız ilk terimin pozitif olmaması yeterli değildir. `δ=ηd` ve `∇L_oᵀd<0` için `0<η≤−2∇L_oᵀd/(β‖d‖²)` yeterli bir aralıktır. `β`nin bilinmesi veya güvenilir üst sınırının bulunması ek yükümlülüktür; tek noktadaki Hessian normu bütün yol için otomatik üst sınır değildir.

Çıktı Jacobian'ının aynı bölgede `K`-Lipschitz olması halinde benzer hesap:

\[
\|F_o(\theta+\delta)-F_o(\theta)-A\delta\|_2
\le\frac K2\|\delta\|_2^2
\]

verir. `Aδ=0` ise ortalama kare çıktı sapması en fazla `K²‖δ‖⁴/(4m)` olur. Küçük sapma sınırı, tam eşitlik değildir. Ayrıca sınır seçilmiş eski girdiler içindir; görünmeyen eski bölgeye genelleme için ayrıca dağılım, temsil veya düzenlilik varsayımları gerekir. Gerçek aday ileri geçişini denetlemek yerel yaklaşım hatasını o ankorlarda ölçer, bu ek genelleme sorununu çözmez.

## 5. Eski çıktıyı korumak ile doğru eski kaybı korumak

Eski etiketler `y`, başlangıç çıktıları `F₀` ve aday çıktıları `F₀+d` olsun. Ortalama kare doğru-hedef riski `R`, çıktı sapması `D=‖d‖²/m` ile:

\[
R-R_0=\frac{2}{m}(F_0-y)^Td+D
\le2\sqrt{R_0D}+D.
\]

Tam çıktı koruması başlangıç hatalarını da korur. Küçük çıktı sapması doğru kaybı artırabilir veya azaltabilir; doğru kaybı azaltmak ise çıktıyı değiştirmeyi gerektirebilir. Bu nedenle `replay_guard` doğru etiketlerle yön üretse bile, protokoldeki çıktı kabul ölçütü ayrı bir hedef uygular. Sabit başlangıca göre `D≤10⁻⁴` ile her önceki adıma göre aynı tolerans farklı sözleşmelerdir; ikincisindeki küçük değişimler birikebilir.

Ortalama doğru kaybın korunması da bütün eski becerilerin ayrı korunması değildir. `L_A=(θ−1)²/2`, `L_B=(θ+1)²/2` için `θ:1/2→1/4` ortalamayı `5/8→17/32` düşürürken `L_A`yı `1/8→9/32` artırır. Burada sonlu adım yaklaşım hatası yoktur; değerlendirme hedeflerinin farkı vardır.

## 6. Birincil literatür ve özgünlük denetimi

Kaynakların ilgili bölümleri 21 Eylül 2026'da doğrudan incelendi. Aşağıdaki eşleşmeler, yöntem adlarını genel problemin cevabı saymak veya uygulanmamış bir benchmark'ın yeniden üretildiğini ileri sürmek için kullanılmaz.

| Birincil kaynak | Eşleşme ve bu araştırmadaki sınır |
|---|---|
| López-Paz ve Ranzato (2017), [Gradient Episodic Memory](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), §3, (6)–(8) | Geçmiş görevlerin bellek kayıplarını eşitsizliklerle koruma; uygulanabilir gradyan projeksiyonu. Kaynak yerel doğrusallığı ve belleğin temsil ediciliğini açıkça varsayar. Karşı örnekler makalenin bu koşullu iddiasını çürütmez. |
| Chaudhry vd. (2019), [Efficient Lifelong Learning with A-GEM](https://arxiv.org/pdf/1812.00420), §4 | Görev başına kısıtların yerine ortalama episodik kayba dayalı tek kısıt getirir. Protokoldeki `loss_halfspace`, bunun tanısal benzeridir; özgün veri akışı ve bellek örneklemesinin birebir uygulaması değildir. |
| Farajtabar, Azizan, Mott ve Li (2020), [Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf), §3, §5–6 | Kayıp yerine çıktı gradyanlarından koruma uzayı kurar. Sıfıra yaklaşan kayıp gradyanları, yerellik, büyük öğrenme oranı ve yüksek mertebeli bilginin maliyeti zaten tartışılır. `tangent_current`, eski girdilerle tekrar Jacobian hesapladığı için saklanmış yönlerden farklı maliyete sahiptir. |
| Doan vd. (2021), [NTK Overlap Matrix](https://proceedings.mlr.press/v130/doan21a/doan21a.pdf), §3.3 | Teğet özellik haritasını zaman boyunca sabit kabul eden rejimde unutmayı inceler. Öğrenilebilir tanh temsilin sonlu değişimlerine bu varsayım doğrulanmadan aynı garanti taşınamaz. |
| Schulman vd. (2015), [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.pdf), §3–4 | Yerel amaç ile gerçek amaç arasındaki farkı kontrol eder. Teorik maksimum-KL sınırı ile pratik ortalama-KL yaklaşımını ayırır. Bu politika optimizasyonu sonucunu eski bütün becerilerin korunması teoremi olarak kullanmıyoruz. |

## 7. Deneyin cevaplaması gereken ve henüz cevaplamadığı

33 parametreye karşı 64 ankor, Jacobian tam ranka erişirse teğet uzayı kapatabilir; bu beklenen lineer cebirdir. Rank yanında tekil değerler, SVD kesme eşiği, `‖Pg‖/‖g‖` ve yeni gradyanın normu ölçülmelidir. Tanh doygunluğu nedeniyle zaten küçük olan gradyan ile projeksiyonun sildiği yön ayrılmalıdır. Aynı sınıftaki öğretmen kapasite tanığıdır; onun varlığı SGD'nin verilen sürede o çözüme ulaşacağını kanıtlamaz.

Ankor sayısı değişirken tutulan bilgi ve koruma kısıtları birlikte değişir. Jacobian, SVD ve aday denemelerinin maliyeti replay'in ortalama kayıp hesabıyla eşit sayılamaz. Ankor kabulünün iyi olması bağımsız eski bölge başarımı değildir; donma da yeni edinim değildir. Sonuçlar geldiğinde bu ölçüler birlikte raporlanacaktır. Şimdilik savunulabilir ilerleme, **koruma için seçilmiş yerel hesapla gerçekten erişilebilir yeni beceriyi birbirinden ayıran, yanlışlanabilir bir deney sorusu** elde etmektir; genel öğrenme ilkesi bulunduğu iddiası yoktur.
