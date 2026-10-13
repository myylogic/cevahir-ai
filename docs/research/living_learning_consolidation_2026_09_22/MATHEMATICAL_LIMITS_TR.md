# Edinim, bütünleştirme ve süreklilik: kanıtlanabilir önermeler ve sınırlar

22 Eylül 2026 · Araştırma turu 13 · Bağımsız matematik notu

Bu metin, ana deney kodu ve sonuçları görülmeden hazırlanmıştır. Önceki araştırma kayıtlarını değiştirmez. Buradaki önermeler, belirtilen yapay modellerin koşullu sonuçlarıdır; yeni teorem önceliği, insan beyninin tam açıklaması, bilinç ölçütü veya genel yapay zekâ kanıtı iddiası değildir. Kanıtlar aşağıda verildiği için doğrulukları biyolojik benzetmeye dayanmaz. İlgili biyolojik iddiaların deneysel durumu ayrı literatür incelemesinde değerlendirilmelidir.

Kullanıcının önerisinin araştırılabilir çekirdeği şudur: **Yeni deneyimler ilk anda kullanılabilir bir iz oluşturabilir; daha sonra mevcut bilgilerle bütünleştirilerek gelecekteki çıkarım ve öğrenmeyi değiştirebilir. Bu dönüşüm eski yetkinliklerle ve sonraki güncellemelerle bağdaşmalıdır.** Bunun belirli bir sistemde mümkün olduğunu göstermek ile bütün beyinlerde gerçekleşen mekanizmayı saptamak farklı yükümlülüklerdir.

## 1. Ortak tanım: deneyimden sonra hangi şey değişiyor?

Bir deneyim kaydı e_t; sistemin o ana kadar eriştiği kayıt dizisi H_t=(e_1,…,e_t) olsun. Kalıcı hesaplama durumu

\[
z_t=(m_t,\theta_t,r_t)
\]

ile gösterilsin. Burada m hızlı erişilebilen kayıt/özet, θ daha yavaş değişen ortak temsil veya parametreler, r ise gelecekteki güncelleme için gerekli diğer durumdur. r; kaynak kimliği, belirsizlik, optimizasyon durumu, sayaç veya denetim bilgisi içerebilir. Bu ayırımlar işlevsel yer tutuculardır; ayrı biyolojik organlar, üç zorunlu bellek veya baştan seçilmiş bir mimari değildir. Bazı modellerde bileşenlerden biri boş olabilir.

Yeni deneyim güncellemesi U_e:Z→Z, dış deneyim gelmeyen bütünleştirme işlemi C:Z→Z, bütçeli cevap üretimi ise Q_B(z,x) olsun. “Uyku” bu notta biyolojik uyku diye tanımlanmaz: yalnız daha önce edinilmiş durum üzerinde ayrılan bir hesaplama dönemine benzetilebilir. U ve C'nin farklı zaman ölçeklerinde çalışması bir hipotezdir; genel öğrenmenin tanımı değildir.

İncelenmesi gereken değişiklikler birbirinden ayrılır:

1. Aynı sorgudaki cevap değişebilir.
2. Aynı cevap daha az zaman veya bellekle üretilebilir.
3. Henüz görülmemiş bir sorguda doğru ilişki kurulabilir.
4. Aynı yeni deneyim, önceki duruma göre farklı ve yararlı bir sonraki güncellemeyi mümkün kılabilir.

Bu ayrım önemlidir: bütünleştirme birinci ölçütte değişiklik yaratmadan diğer üçünde etkili olabilir. Rastgele cevap değişmesi ise tek başına bunların hiçbirinin öğrenildiğini göstermez.

## 2. Zamanlama eşdeğerliği: tam olarak hangi koşullarda?

### Önerme 1 — Aynı yeterli özetin tam çözümü

Sabit bir deneyim çoklukümesi D için T(D) bir özet olsun. Ayrık parçaların birleştirilmesi için birleşmeli ve değişmeli bir ⊕ işlemi bulunsun:

\[
T(D_1\uplus D_2)=T(D_1)\oplus T(D_2).
\]

Sabit amaç fonksiyonu yalnız bu özetten hesaplansın: J_D(θ)=j(θ;T(D)). Son durumda F(T(D)) bu amacın tam çözümünü versin. Çözüm tek olsun veya her çalıştırmada aynı çözümü seçen, zamanlamadan bağımsız bir bağlayıcı kural bulunsun. Bütün programlar aynı başlangıç önbilgisiyle aynı D'yi işlesin, hiçbir gerekli özet kaybolmasın ve karşılaştırmadan önce F son kez çalıştırılsın.

Bu koşullarda, D'nin gün içinde tek tek işlenmesi, bloklar halinde biriktirilmesi veya son bir bütünleştirme döneminde çözülmesi, **son çözümü** değiştirmez.

**Kanıt.** Birleşme ve değişme özellikleri bütün bölme ve işleme sıralarının aynı T(D)'yi üretmesini sağlar. F aynı girdiden aynı çözümü seçer. Dolayısıyla son θ eşittir. Aradaki θ değerlerinin veya yanıtların eşit olması gerekmez. □

Önerme yalnız θ eşitliğini kanıtlar. Gelecekteki güncelleme için T(D) de korunuyorsa devam ettirilebilir durum da eşdeğerdir. Sadece θ'yı tutup yeterli özeti silmek, son cevabı korurken sonraki öğrenmeyi değiştirebilir.

**Koşulların neden gerekli olduğu:**

- Tam çözüm yerine sınırlı adım kullanılıyorsa optimizasyon yolu sonucu değiştirebilir. Örneğin her gözlemde θ←(θ+y)/2 ve θ_0=0 kuralı, y dizisi (0,1) için 1/2; (1,0) için 1/4 verir. Aynı verinin farklı sırası eşit sonuca yetmez.
- Zamanla değişen hedef, ağırlıklandırma, unutma veya doğrusal olmayan ve yerel çözümlere giden optimizasyon varsa aynı sabit j varsayımı sağlanmayabilir.
- Sistemin ara davranışı sonraki gözlemlerini değiştiriyorsa D artık aynı değildir. Kapalı çevrimde farklı zamanlamalar farklı deneyim üretebilir.
- Sıkıştırma bir programda gerekli ayrımı kaybediyorsa iki program aynı T'yi korumaz.
- Aynı doğru sonuca geç ulaşmak, işlem sırasında kullanılan yanlış cevabın maliyetini ortadan kaldırmaz. Son nokta ve yaşam boyunca başarım ayrı ölçülmelidir.

Bu nedenle bir deneyde tam çözüm zamanlamalarının eşit çıkması “uykunun faydası yok” anlamına gelmez. Eşitlik, belirlenmiş bir modelde zamanlamanın dışında bilgi veya hesaplama farkı aranması gerektiğini gösterir.

## 3. Dış kanıt eklemeyen hesaplama ne yapabilir, ne yapamaz?

### Önerme 2 — Ayırt edilemeyen iki dünya

İki olası dünya W_0 ve W_1, bütünleştirmeden önce öğrenicinin eriştiği aynı z durumunu üretmiş olsun. Bir sonraki x sorgusunun doğru cevabı W_0'da 0, W_1'de 1 olsun. Bütünleştirme, bu dünyaları ayıran yeni bir dış gözlem alamasın. C'nin kullandığı rastgelelik varsa bunun dağılımı da gerçek dünyadan bağımsız olsun.

O zaman C ve ardından gelen cevap üretimi iki dünyada aynı cevap dağılımına sahiptir. Her ikisinde de kesin doğru olmak mümkün değildir.

**Kanıt.** Deterministik durumda aynı z, aynı C(z)'yi; aynı C(z) ve x de aynı cevabı verir. İki farklı doğru cevapla aynı anda eşleşemez. Rastgele durumda aynı girdi ve aynı rastgelelik dağılımı aynı çıktı dağılımını üretir; dünya etiketi hakkında bilgi oluşmaz. □

İki dünya eşit ağırlıklı ve çıktı a gerçek sayı olduğunda ortalama karesel kayıp için daha güçlü bir nicel sınır vardır:

\[
\frac12\mathbb E[a^2]+
\frac12\mathbb E[(a-1)^2]
=\mathbb E[(a-\tfrac12)^2]+\frac14
\geq\frac14.
\]

Rastgelelik bu sınırı düşürmez. İkili bir cevap verilmesi zorunluysa eşit ağırlıklı iki dünyadaki ortalama sınıflama hatası 1/2'dir. Eşit dünya ağırlığı burada açık varsayımdır; farklı ön olasılıklar farklı ortalama risk verir.

**Bu sonuç iç hesaplamanın yararsız olduğu anlamına gelmez.** z içinde bir problemin çözümünü mantıksal olarak belirleyen bütün bilgi bulunabilir, fakat kısa cevap bütçesi bunu açığa çıkarmaya yetmeyebilir. C bir kanıtı bulabilir, bir program parçasını derleyebilir, bir arama indeksini kurabilir veya ayrı kayıtlar arasındaki çıkarımı hesaplayabilir. Sonraki Q_B bu önceden hesaplanmış sonucu aynı B bütçesinde kullanabilir. Bütçeli performans artışı gerçektir; C'nin önceki hesaplama zamanı ve kalıcı alanı toplam maliyete dahil edilmelidir.

Dolayısıyla iki iddia birlikte doğrudur: dış kanıtsız bütünleştirme **ayırt edilemeyen dünyaların hangisinin gerçek olduğunu öğrenmeyi garanti edemez**; mevcut bilgiyi kullanarak **daha önce hesaplanmamış sonuçları erişilebilir hale getirebilir**. İlki bilgi sınırıdır, ikincisi hesaplama kapasitesidir. Bunları aynı ölçüye indirgemek kullanıcı hipotezinin önemli kısmını kaçırır.

## 4. Eski ve yeni gereksinimler çelişirse kusursuzluk mümkün mü?

### Önerme 3 — Gözlenebilir bağlam olmadan eşzamanlı çelişki

Aynı güncel durumda, aynı gözlenebilir x girdisi için iki geçerli değerlendiricinin eşzamanlı olarak sırasıyla 0 ve 1 çıktısını istediğini varsayalım. Sistemden tek bir a çıktısı alınsın. İki yükümlülükteki karesel kayıplar L_0=a² ve L_1=(a−1)² olur.

\[
\min_{a\in\mathbb R}\max\{a^2,(a-1)^2\}=\frac14.
\]

**Kanıt.** a≥1/2 ise a²≥1/4; a≤1/2 ise (a−1)²≥1/4. a=1/2 her ikisini de 1/4 yapar. □

Bu, “hiç unutma ve her yeni talebi tam karşıla” sözünün bağlamsız ve çelişkili sürümünü çürütür. Bir öğrenme algoritması eksikliğinden kaynaklanmaz; istek kümesinin tutarsızlığıdır. Rastgele çıktıda da iki beklenen kaybın en büyüğü en az 1/4'tür; ortalamaları yukarıdaki alt sınırı sağlar.

Gözlenebilir c∈{0,1} bağlamı doğru şekilde sunulursa a(x,c)=c iki işi de sıfır kayıpla yapabilir. Bağlam yalnız görünür bir etiket olmak zorunda değildir: gözlenen geçmiş, zaman, kişiler veya olay düzeninden güvenilir biçimde çıkarılabilir. Fakat ilgili bütün gözlenebilir geçmiş de aynıyken iki farklı gizli anlamı doğru ayırt etmek garanti edilemez. “Üst görevci” adı verilen bir birim bu bilgiyi yoktan üretemez; ek duyum veya geçerli önbilgi edinmesi gerekir.

**Kapsam:** Bir olgu önce yanlış öğrenilip sonradan düzeltilmişse eski yanlış cevabı korumak zaten doğru hedef olmayabilir. Önerme, her eski cevabın sonsuza dek korunmasını savunmaz. Hangi eski yetkinliğin hâlâ geçerli olduğu, hangisinin düzeltildiği ve hangi bağlamda değerlendirildiği ayrı tanımlanmalıdır. Böyle bir ayrım, araştırma problemini çözmekten çok onu tutarlı hale getirir.

## 5. Her yeni deneyim için yeni temsil boyutu zorunlu değildir

### Önerme 4 — Sabit özelliklerde tam, birleştirilebilir öğrenme

Bir girdinin sabit ve deneyim sırasında değişmeyen özelliği v=φ(x)∈ℝ^d olsun. Skaler hedef y ve λ>0 için amaç

\[
J_D(\theta)=\lambda\|\theta\|_2^2+
\sum_{(v,y)\in D}(y-v^\top\theta)^2
\]

olarak sabitlensin. Şu iki veri özeti yeterlidir:

\[
G_D=\sum_Dvv^\top,\qquad b_D=\sum_Dvy.
\]

\[
A_D=\lambda I+G_D,\qquad
\theta_D=A_D^{-1}b_D.
\]

**Kanıt.** Amaç açıldığında θ'ya bağlı kısım θᵀA_Dθ−2b_Dᵀθ'dır. Gradyanın sıfır olması A_Dθ=b_D verir. λ>0 olduğundan A_D pozitif tanımlıdır ve çözüm tektir. Veri bloklarının G ve b özetleri toplanır; bu nedenle tam blok çözümü ve tam artımlı çözüm aynı θ'yı verir. □

Amaç fonksiyonunun sayısal değerini de tam hesaplamak için ayrıca c_D=Σ_D y² saklanır. Örnek sayısına göre ortalama kayıp veya başka normalizasyon kullanılacaksa n_D de korunur. Farklı normalizasyonlarda λ'nın anlamı aynı tutulmalıdır.

**Birleştirme ayrıntısı:** Bir blok için saklanan A_i zaten λI içeriyorsa A_1+A_2, düzenlileştirmeyi iki kez sayar. Doğru birleştirme A_1+A_2−λI veya daha açık biçimde G_1+G_2'den yeni A üretmektir. Çoklu bloklarda önbilgi her blokta yeniden eklenmemelidir.

Artımlı hesap için P=A^{-1} saklanabilir. Yeni (v,y) geldiğinde

\[
k=\frac{Pv}{1+v^\top Pv},\qquad
\theta^+=\theta+k(y-v^\top\theta),\qquad
P^+=P-\frac{Pvv^\top P}{1+v^\top Pv}.
\]

P_0=λ⁻¹I ve θ_0=0 başlangıcından bu güncelleme, aynı sabit amacın tam çözümünü sürdürür. Formül, yeni A^+=A+vvᵀ tersinin doğrudan çarpımla doğrulanmasından ve b^+=b+vy ilişkisinden çıkar. Deneyde iki çözümün eşitliği sonlu hassasiyet nedeniyle uygun toleransla sınanmalıdır; kayan nokta aritmetiğinde bit düzeyinde eşitlik garanti edilmez.

Bu modelde deneyim sayısı artarken özellik boyutu d değişmeden kalabilir; G ve b için saklanan skaler sayısı O(d²), bir artımlı güncellemenin aritmetik işlem sayısı O(d²)'dir. Bu, **sabit sayıda gerçek sayı** sonucudur; sınırsız deneyimi sabit sayıda bit ile kayıpsız saklama kanıtı değildir. Tam sayı/rasyonel hesapta değerlerin bit uzunluğu büyüyebilir; sonlu hassasiyette de sayısal koşulluluk ayrıca incelenmelidir.

**Ne kanıtlandı, ne kanıtlanmadı?**

- Her yeni gözlemin mutlaka yeni bir temsil boyutu gerektirdiği evrensel iddia için karşı örnek elde edildi.
- Yeni ayrımın mevcut φ içinde ifade edilebildiği varsayıldı. φ'nin öğrenilmesi, yeni kavramların doğması veya bilinmeyen gelecekteki görevler çözülmedi.
- Bu özet, bu amaç ve sabit özellikler için yeterlidir. Sonradan yeni özellik hesaplama, tekil eski kaydı kimliğiyle silme, eski olayı yeniden yorumlama veya farklı hedeflere geçiş için genel olarak yeterli değildir.
- θ tek başına gelecekteki tam güncelleme için yeterli olmayabilir; P veya eşdeğer G,b bilgisi de gerekir. Aynı bugünkü cevap ile aynı sonraki öğrenme kapasitesinin ayrımı burada yeniden görünür.

Dolayısıyla bu sınırlı model, genel yaşarken öğrenme sorununu doğrusal regresyona indirgemek için kullanılamaz. Görevi, “boyut büyümesi her deneyimde zorunludur” ile “hiçbir zaman gerekmez” iddialarının ikisini de dikkatle ayırmaktır.

## 6. Eski cevabı korumak, gelecekteki öğrenmeyi korumaya yetmez

### Önerme 5 — Durum dönüşümünün güncellemeyle uyumu

Eski durum uzayı Z, yeni düzenlenmiş uzay Z′ ve dönüşüm T:Z→Z′ olsun. Eski ve yeni cevap yordamları sırasıyla f ve f′, deneyim güncellemeleri U_e ve U′_e olsun. Erişilebilir bütün z durumları ve izinli deneyimler için

\[
f'(Tz,x)=f(z,x),\qquad
T(U_e(z))=U'_e(Tz)
\]

eşitlikleri sağlansın. O zaman T dönüşümünden sonra, izinli herhangi bir sonlu deneyim dizisinde bütün sonraki cevaplar eski sistemle aynıdır.

**Kanıt.** Sıfır adımda durum Tz ve cevap eşitliği varsayılmıştır. Bir adımda ikinci eşitlik yeni durumu yine eski durumun T görüntüsü yapar. Tümevarımla bu ilişki her sonlu önekte sürer; birinci eşitlik her adımın cevaplarını eşitler. □

Ham parametrelerin veya nöron sayılarının aynı olması gerekmez. Bu yüzden bir temsilin yeniden örgütlenmesi süreklilikle bağdaşabilir. Ancak yalnız ilk eşitliği ölçmek yeterli değildir.

**Karşı örnek:** Eski z=(a,b), f(z)=a ve deneyim e için U_e(a,b)=(a+b,b) olsun. T(a,b)=a dönüşümü bugünkü cevabı tam korur. Buna karşın z_0=(0,0) ve z_1=(0,1) aynı T durumuna iner; bir sonraki deneyimden sonra eski cevaplar sırasıyla 0 ve 1 olur. Tek bir U′_e(0) ikisine birden uyamaz. Kaybolan b bugünkü cevap için gereksiz, sonraki öğrenme için gereklidir.

Bu nottaki T, mevcut davranış ve gelecekteki güncelleme anlamını korumayı amaçlayan yeniden düzenleme işlemidir. C'nin her kullanımının tam olarak bu T olması gerekmez: bütünleştirme bazı cevapları veya öğrenme stratejisini bilinçli olarak iyileştirebilir. O durumda hedef tam eşitlik yerine hangi işlevlerin hangi hata sınırıyla korunacağı şeklinde yazılmalıdır.

### Yaklaşık uyumda küçük hata kendiliğinden küçük kalmaz

Yeni uzayda bir d metriği, U′_e için ortak L-Lipschitz sınırı ve bütünleştirme/güncelleme uyumsuzluğu için

\[
d\big(U'_e(Tz),T(U_ez)\big)\leq\varepsilon
\]

varsayalım. Eski sistemin t adım sonraki durumunun T görüntüsü ile yeni sistemin gerçek durumu arasındaki fark Δ_t olsun. Üçgen eşitsizliğiyle

\[
\Delta_{t+1}\leq L\Delta_t+\varepsilon,
\qquad
\Delta_t\leq L^t\Delta_0+
\varepsilon\sum_{j=0}^{t-1}L^j.
\]

L<1 ise biriken ek hata ε/(1−L) ile sınırlanabilir; L=1 ise tε büyür; L>1 ise geometrik büyüme mümkün olur. Cevap yordamı durum açısından K-Lipschitz ve T üzerindeki davranış hatası en fazla η ise çıktı farkı en fazla KΔ_t+η olur. Bu sınırlar belirtilen bölgede ortak sabitler bulunduğu varsayımına dayanır; genel sinir ağında otomatik verilmez.

İşlevsel eşitlik tek başına kullanılabilirliği de garanti etmez. Yeni gösterim aynı cevabı üretmek için bellek veya süre sınırını aşabilir. Bütçeli sistemlerde ayrıca cevap, güncelleme ve bütünleştirme maliyetleri ile deneyimler arası kullanılabilir zaman ölçülmelidir. Bir dönüşüm yalnız sınırsız süreyle eşdeğerse gerçek çalışma koşullarında eşdeğer olmayabilir.

## 7. Çift yönlü etkileşim ve üst denetçi neden tek başına garanti değildir?

Bir alt düzeyin veriden h üretmesi ve bir üst düzeyin geri bildirimle onu düzenlemesi, soyut olarak h_{k+1}=F_x(h_k) yinelemesiyle temsil edilebilir. Bu bir olası matematiksel araçtır; beynin mimarisi diye kabul edilmez.

Sabit x altında, tam bir metrik uzayda F_x'in o uzayı kendi içine taşıdığı ve bütün h,h′ için

\[
d(F_x(h),F_x(h'))\leq q\,d(h,h'),\quad 0\leq q<1
\]

olduğu sağlanırsa tek bir sabit noktaya yakınsama elde edilir. Neden: ardışık adımların farkı en çok q^k oranıyla küçülür; geometrik kuyruk toplamı sonludur, dolayısıyla dizi Cauchy'dir. Tamlık limiti sağlar; Lipschitz sürekliliği limitin sabit nokta olduğunu, q<1 ise iki ayrı sabit noktanın olamayacağını gösterir.

Fakat çift yönlü bağlantı veya bir denetçinin varlığı bu koşulu kendiliğinden sağlamaz. ℝ üzerinde F(h)=−h, h_0≠0 iken salınır; F(h)=2h uzaklaşır. F(h)=1 ise ilk adımda tamamen kararlıdır, fakat doğru hedef 0 ise kararlı biçimde yanlıştır. **Kararlılık, doğruluk ve uyum sağlayabilme farklı ölçütlerdir.** Sabit bir yanlış cevaba kilitlenmek de kararlı olabilir.

Dolayısıyla “üst görevci denetliyor, öyleyse kusursuz” bir kanıt değildir. Denetçinin gözlediği sinyaller, müdahale alanı, yanlış kararları, ek maliyeti ve değişen hedeflerde gecikmesi belirtilmelidir. Denetçinin de öğrenmesi gerekiyorsa kendi güncellemesi aynı incelemeye dahildir. Bu, hiyerarşiyi reddetmez; hangi koşullarda yarar sağlayacağının ölçülebilir hale gelmesini sağlar.

## 8. Ana deney için sonuçtan bağımsız denetim listesi

Bu maddeler başarı eşiğini sonuçlara göre değiştirmek için kullanılmamalıdır. Ana deneyin seçtiği model yukarıdaki özel koşulları sağlamıyorsa, ilgili eşitlik zorla beklenmemelidir.

| Denetim | Gerekçe ve beklenen yorum |
|---|---|
| Aynı sabit veri, aynı amaç, tam çözüm: tek tek/blok/sonda birleştirme | Önerme 1'in koşulları altında son çözüm sayısal tolerans içinde eşit olmalı; ara cevaplar ayrıca raporlanmalı |
| Önbilginin bloklarda tekrar eklenmemesi | G ve b toplama ile A toplamanın farklılığı, yanlış bir “zamanlama etkisi” üretmemeli |
| Önce tahmin, sonra yeni etiketle güncelleme | Yeni gözlemin kendi önceki tahminine sızması engellenmeli |
| Aynı kalan durum, farklı saklı doğru cevap | Dış kanıtsız bütünleştirme veya denetçi saklı dünya etiketini kullanmamalı |
| Bugünkü tahmin ve sonraki güncelleme ayrı karşılaştırma | θ eşitliği veya eski görev başarısı, tam kalıcı durum eşdeğerliği diye sunulmamalı |
| Sınırlı hesaplama ile tam çözüm ayrı kontrol | Uykuya ayrılmış ek hesaplama, zamanlama tek başına yarar sağlamış gibi raporlanmamalı |
| Kaynağa göre eşit deneyim, hesaplama ve kalıcı alan | Bütünleştirme önhesaplaması ve yeniden oynatılan kayıtlar toplam maliyete dahil olmalı |
| Bağlamlı ve bağlamsız çelişki ayrımı | Gerçekte gözlenmeyen doğru bağlam öğreniciye ayrıcalıklı etiket olarak verilmemeli |
| Yeniden başlatma sonrasında gelecek deneyim devamı | Sadece eski tahminlerin değil, aynı yeni veri altında sonraki durum ve cevapların da sürmesi sınanmalı |
| Son nokta, yol boyunca bozulma ve uyum gecikmesi | Sonunda doğru olmanın çalışma sırasında yapılan hataları gizlemesi önlenmeli |

Bu sonuçlar araştırma uzayını belirli yönlerden daraltır: dış kanıtsız kusursuz dünya tanıma, bağlamsız çelişkilerin birlikte karşılanması ve yalnız bugünkü cevapları koruyarak tüm sonraki öğrenmenin korunduğunu varsayma geçerli genel çözümler değildir. Buna karşılık deneyimin yeniden örgütlenmesi, hesaplanabilir sonuçların erişilebilir hale gelmesi, gerekli ayrımları koruyan sıkıştırma ve gelecekteki güncelleme kapasitesinin sürdürülmesi hâlâ açık ve anlamlı araştırma yönleridir.

**Hiçbir önerme, yaşarken öğrenmenin bütün matematiğini veya genel yapay zekânın yeterli koşullarını vermez.** Bir biyolojik açıklamanın doğru olup olmadığı da yalnız bu önermelerden çıkmaz. Bu notun sağladığı şey, yeni hipotezlerin hangi koşullarda gerçekten yeni bir yetkinlik açıkladığını ve hangi iddiaların mevcut bilgiyle kanıtlanamayacağını açıklaştırmaktır.
