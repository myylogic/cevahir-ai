# Gelecekteki öğrenme işlemleri için yeterli durum: sorgu sınıfları ve kimlik belleği

20 Eylül 2026 · Yayın öncesi iç matematiksel değerlendirme · Muhammed Yasin Yılmaz

**Durum:** Bu not doğrudan türetimler ve yazıldığı sırada henüz çalıştırılmamış bir deney önerisi içerir. Not tek başına sayısal deney veya yeni bir genel öğrenme ilkesi keşfi sayılmamalıdır. Önceki [düzeltme turunun raporu](../../research/living_learning_correction_2026_09_20/REPORT_TR.md), [deney çıktısı](../../../research/living_learning_correction_2026_09_20/results/correction_state.json) ve [doğrulama kaydı](../../../research/living_learning_correction_2026_09_20/results/verification.json) okunmuştur. O doğrulamada tam yeniden üretim eşitliği ve önceki 88 dosyanın değişmediği kaydedilmiştir. Burada önceki dosyalara müdahale edilmez. Yapay zekâ destekli iç incelemedir; dış hakemlik değildir.

**21 Eylül eki:** Önerinin tam sınıf sayımı, ekleme/düzeltme ve negatif kontrol kısmı sonradan çalıştırıldı. Ölçülmüş sonuçlar ayrı [deney raporundadır](../../research/living_learning_query_state_2026_09_21/REPORT_TR.md). Aşağıdaki öneri geçmişteki statüsünü korur; bütün önerilen performans ölçülerinin yapılmış olduğu anlamına gelmez.

Ana bulgu iki parçalıdır. **Bilinmeyen gelecek hipotezler, bilinen sonlu bir girdi alanında her zaman tam histogram gerektirmez.** Öte yandan yalnız olay kimliğiyle yapılacak keyfi etiket düzeltmelerini tam doğru desteklemek, aynı toplu istatistiklere sahip geçmişlerin olaylarla eşlemesini korumayı gerektirebilir. Bu iki sonuç farklı işlem sözleşmelerine aittir; tek bir bellek zorunluluğu diye birleştirilemez.

## 1. “Yeterli” hangi çıktılar ve hangi işlemler için?

Sonlu gözlem evreni `Z`, `N` kayıtlı geçmiş `D=(z₁,…,z_N)` ve gelecekte sorulabilecek sorgular ailesi `Q` olsun. Yalnız bugünkü sorguları cevaplamak için doğal eşdeğerlik:

\[
D\equiv_Q D'\quad\Longleftrightarrow\quad
q(D)=q(D')\quad\text{her }q\in Q\text{ için}.
\]

Her sınıf için ayrı kod tutmak teorik olarak yeterlidir. Farklı sınıfları aynı koda koyan kesin deterministik özet ise onları ayıran en az bir sorguyu yanlış cevaplar. Bu nedenle sınıf sayısı `K` ise, en kötü durumda sabit uzunluklu durum kodu en az `ceil(log₂ K)` bit ister. Bu, sonlu ayırt edilebilir durumların sayılmasından çıkan doğrudan sonuçtur; verimli kodlayıcı bulunduğunu söylemez.

Güncelleme de isteniyorsa eşdeğerlik genişletilmelidir. İzinli işlem dizileri `u` için `q(uD)=q(uD')` aranır. İşlemler kısmi tanımlıysa ve sistem geçersiz işlemi reddetmekle de yükümlüyse, **hangi işlemlerin geçerli olduğunun eşitliği** de gerekir. Önceki turun “ortak geçerli düzeltmeler” koşulu, sistemin işlemin geçerliliğini ayrıca kanıtlamak zorunda olmadığı sözleşmeydi. Bu ayrım, aşağıdaki küçük özetin neyi sağlamadığını belirler.

## 2. Toplamsal sorgularda en güçlü başlangıç rakibi histogramdır

Bir sorgu `q_g(D)=Σ_i g(z_i)` biçimindeyse, `n_z` gözlem çokluklarıyla `q_g=Σ_z g(z)n_z` yazılır. Dolayısıyla sonlu `Z` üzerindeki histogram bütün böyle sorgulara yeter. Sorgu satırlarından oluşan `A` matrisi için iki histogramın eşdeğerliği `An=An'` olur. Satırların doğrusal bağımlılığı, bütün kutuları saklamadan sorguları hesaplamayı mümkün kılabilir.

Tam minimal bit sayısı yalnız matrisin rank'ı değildir: erişilebilir tamsayı histogramları üzerinde `An` kaç farklı değer alıyorsa sınıf sayısı odur. Sayaç sayısı, sayaçların basamakları ve hipotez tanımlarının belleği ayrı bedellerdir. Sonsuz hassasiyetli tek bir gerçek sayıya bütün geçmişi kodlamak burada sabit bellek çözümü sayılmaz.

Gelecek sorguları bütün gözlem göstergelerini `g_z(w)=1[w=z]` içeriyorsa histogram da zorunludur: bu sorguların cevapları zaten bütün `n_z` değerleridir. `r=|Z|` ve `N` sabitken histogram sayısı

\[
K_{\mathrm{hist}}(N,r)=\binom{N+r-1}{r-1}
\]

olur. Buna karşılık “gelecekte bütün Boolean hipotezlerin ihlal sayıları sorulabilir” koşulu, bütün gözlem gösterge sorgularını içermek zorunda değildir. Önceki histogram rakibi bu daha dar koşulda minimal olmayabilir.

## 3. Bütün gelecek Boolean hipotezlere daha küçük bir özet yeter

`X={1,…,m}`, `Y={0,1}`, `Z=X×Y` olsun. Geçmişte `n_{x,0}` sıfır, `n_{x,1}` bir etiketi bulunuyor. Gelecek hipotez dilinin tamamı `H_all={0,1}^X`; hangi hipotezin ne zaman ekleneceği önceden bilinmiyor. İstenen sorgu, her hipotezin toplam 0–1 ihlali:

\[
c_h(D)=\sum_x n_{x,1-h(x)}.
\]

Şu işaretli çokluk farkını tutalım:

\[
d_x=n_{x,0}-n_{x,1},\qquad N=\sum_x(n_{x,0}+n_{x,1}).
\]

Doğrudan açılım verir:

\[
\boxed{c_h(D)=\frac{N-\sum_x d_x}{2}+\sum_x h(x)d_x.}
\]

Dolayısıyla `N` ve `m` işaretli sayaç, mevcut veya **sonradan eklenecek bütün Boolean hipotezlerin** ihlal sayısını verir. Her aday için ayrı sayaç gerekmez; `2m` histogram kutusu da bu sorgu ailesinde gereğinden fazla bilgi taşıyabilir.

Bu özet sorgu eşdeğerlik sınıfını tam belirler. Ters yönde, sabit-sıfır hipotezin ihlali `a=c₀`; yalnız `x` girdisinde 1 veren hipotezin ihlali `c_{e_x}` olsun. O zaman `d_x=c_{e_x}−c₀`. `N` bilinmiyorsa sabit-bir hipotezle `N=c₀+c₁` elde edilir. Böylece bütün `c_h` cevapları tam olarak `(N,d)` bilgisini belirler; histogramın geri kalanını belirlemez.

Bu, en iyi adayı seçme veya yalnız sıfır ihlalli adayları bulma görevinden **daha güçlü** bir sorgu sözleşmesidir. Bu görevler için çıkarılan alt sınır burada otomatik olarak aynı sayılmaz.

### Geçerli düzeltme içeriği verilirse kapanış

Yeni `(x,y)` kaydında `N←N+1`, `d_x←d_x+1−2y` yapılır. Eski gerçek `(x,y)` yeni `y'` etiketiyle değiştirilirse `N` değişmez ve

\[
d'_x=d_x+2(y-y')
\]

olur; diğer koordinatlar değişmez. Her ortak geçerli, gerçek eski içeriği taşıyan işlem dizisi için bu özet kapanır. Bu nedenle hipotezler aynı sabit gözlem alanında genişlese de eski ham kayıtları yeniden taramak gerekmez.

Ancak `(N,d)` bir düzeltme paketinin gerçekten var olan kaydı anlattığını doğrulamaz. Bir koordinata eşit sayıda 0 ve 1 gözlemi eklenmesi `d_x`'i değiştirmez. Bu çelişkili çiftlerin **hangi koordinatta** bulunduğu toplam ihlal sorgularından kaybolabilir. Doğrulama için histogram, tekil olay doğrulaması için de kimlik bilgisi gerekebilir. Özetin geçerlilik denetimi yapmaması cebirsel güncelleme eşitliğini bozmaz; sözleşmesini daraltır.

### Tam sonlu sınıf sayısı

`N` sabitken erişilebilir işaretli vektörler tam olarak

\[
\mathcal D_{N,m}=\{d\in\mathbb Z^m:\|d\|_1\le N,
\quad N-\|d\|_1\text{ çift}\}
\]

kümesidir. Gereklilik açıktır: her koordinatta karşıt etiketler ikişer kayıt tüketip farkı değiştirmez. Yeterlilik için `|d_x|` kaydı `d_x`'in işaretine uygun etiketle koyar, kalan çift sayıyı herhangi bir koordinata karşıt etiket çiftleri olarak eklersiniz.

`A_m(s)` normu tam `s` olan tamsayı vektör sayısı olsun. `A_m(0)=1`; `s>0` için sıfır olmayan `k` koordinatı, işaretlerini ve pozitif büyüklüklerin toplamını seçerek

\[
A_m(s)=\sum_{k=1}^{\min(m,s)}\binom{m}{k}2^k\binom{s-1}{k-1},
\qquad
K_{\mathrm{risk}}(N,m)=\sum_{j=0}^{\lfloor N/2\rfloor}A_m(N-2j).
\]

Örneğin `m=3,N=4` için `K_risk=A₃(4)+A₃(2)+A₃(0)=66+18+1=85`; histogram sayısı `C(9,5)=126`'dır. Bunlar **formülden hesaplanan değerlerdir**, yeni taramanın ölçülmüş çıktısı değildir. İki temsilde de bu küçük örneğin ideal sabit kod alt sınırı 7 bit olur; “85<126” eşitsizliğini “en az bir bit daha az” diye sunmak yanlıştır. Fark daha büyük `N` altında veya basit sayaç temsillerinde ayrıca değerlendirilmelidir.

Sabit `m` için signed-count sınıfı büyüklüğü `N^m`, histogram sınıfı büyüklüğü `N^(2m−1)` mertebesindedir. Bu da yalnız bu sabit alan ve kesin toplamsal sorgu sözleşmesinin asimptotik sayımıdır; açık dünyadaki gözlem büyümesini sınırlandırmaz.

## 4. Kimlikli düzeltmenin ayrı bellek alt sınırı

Şimdi düzeltme paketi yalnız `(j,y')` taşısın: “j kimlikli olayın etiketini y' yap.” Girdi `x_j` değişmez, ancak eski `(x_j,y_j)` dışarıdan verilmez. Kimlikler `1,…,N` sabittir ve sistem mevcut bütün `c_h` sorgularını ve bu düzeltmeden sonraki sorguları **tam doğru** yanıtlamak zorundadır. Aynı etiketi yeniden yazmak da geçerli bir işlem sayılır; yeni etikete zaten sahip kayıt için etkisi sıfırdır.

Sabit bir histogramdaki farklı ID→kayıt atamalarının sayısı

\[
K_{\mathrm{ID}\mid n}=\frac{N!}{\prod_{z\in Z}n_z!}
\]

olur. Bunların her ikisi bir kimlikte ayrışır. O kimlikteki eski kayıtlar `(x,y)` ve `(x',y')` olsun:

- Eski etiketler farklıysa yeni etiketi birincinin eski etiketi yapın. Bir geçmişte etki sıfır, diğerinde ihlal sayısına etki `+1` veya `−1` olur.
- Etiketler aynı, girdiler farklıysa etiketi tersine çevirin. Bütün Boolean hipotezlerin arasında, bu iki girdiden birinde eski etiketi veren, diğerinde vermeyen bir hipotez vardır. İki geçmişte ihlal değişimi sırasıyla `+1` ve `−1` olur.

Başlangıç sorguları aynı olduğuna göre, bu tek düzeltmeden sonra ayırt edici sorgular farklıdır. Dolayısıyla aynı histogramlı geçmişler de kimlikli düzeltme durumu olarak ayrı tutulmalıdır. En az `ceil(log₂ K_ID|n)` bitlik **eşleme bilgisi** gerekir. Bu, histogramın üstüne naif biçimde aynı sayıda bit eklenmesi gerektiği anlamına gelmez; ortak kodlama mümkün olabilir. Sabit-histogram alt ailesinin kaç farklı duruma ihtiyaç duyduğu gösterilmiştir.

Daha güçlü sayım da çıkar. İki keyfi geçmişin mevcut ihlal cevapları farklıysa zaten ayrıdır; aynıysa yukarıdaki kimlik müdahalesi onları ayırır. Böylece uzunluğu `N` olan `(2m)^N` olay atamasının tamamı farklı davranış sınıflarına düşer. Bu sözleşmede durumun en kötü durum alt sınırı

\[
\boxed{\lceil N\log_2(2m)\rceil\text{ bit}}
\]

olur. Kayıtları kimlik sırasıyla saklamak bu bilgiyi taşır; başka kayıpsız kodlama biçimleri de taşıyabilir. “Sistemin mutlaka ham metin arşivi tutması gerekir” sonucu çıkmaz: olaylar zaten sonlu `X×Y` kodlarıdır. Ayrıca bu sınır geçmişte verilmiş bütün ara cevapları dışarıdan tekrar okuyabilen sisteme uygulanmaz; o kayıt dış bellek olarak sayılmalıdır.

`m=3,N=4` örneğinde 1.296 ID-ataması vardır, alt sınır 11 bittir. Bunun 85 veya 126 toplu sınıftan farklılaşması, geçmişin yalnız **ne kadar** değil **hangi kimlikle** tutulduğunun gelecek düzeltme kapasitesini değiştirmesidir. Bu sonuç rastgele/hatalı düzeltme teşhisinden bağımsızdır; doğru düzeltme komutunu yerine getirme problemidir.

## 5. Bilinmeyen gelecek genişleme hakkında doğru sonuç

“Gelecek hipotezler bilinmiyor, öyleyse her şey saklanmalı” fazla güçlüdür. Sabit `X` üzerindeki bütün Boolean hipotezleri baştan sorgu ailesi içine almak mümkündür; `(N,d)` onların 0–1 ihlal sayılarını korur. Gelecekte hangi adayın seçileceği bilinmese de bu yeterlilik sürer.

Fakat gelecekte sorgulanabilecek şey, etiket kaybı dışında bir gözlem özelliği veya kayıt sırası olabilir. Bütün gözlem göstergelerine izin verilirse tam histogram gereklidir. Her kimlikli etiketi eski içeriği vermeden değiştirme hakkı ve bütün ihlal sorguları istenirse ID eşlemesi gerekir. Metindeki daha önce atılmış bir özelliğe sonradan erişmek istenirse, sabit `X` varsayımı artık geçerli değildir.

Bu nedenle araştırmada önce “gelecekte hangi işlemler desteklenecek?” sorusu açık yazılmalıdır. Buradaki en ağır sözleşme yapay olarak tam doğruluk ve bütün kimlikleri içerir; ana yaşarken öğrenme problemine evrensel gereklilik olarak yüklenmez. Yaklaşık doğruluk, sınırlı düzeltme ufku, bilinen görev dağılımı, bazı kayıtların yeniden edinilebilirliği veya dış provenance hizmeti farklı bellek sınırları verebilir.

## 6. Yeni deney önerisi — henüz çalıştırılmadı

Öneri, önceki elenen-adayı-diriltme örneğini yeniden saymak yerine **minimal davranış sınıflarını** saymaktır. `m∈{1,2,3}`, `N=0,…,6` seçilsin. Her geçmiş için şu üç imza bağımsız üretilecek:

1. Bütün Boolean hipotezlerin mevcut toplam ihlal vektörü.
2. Histogram.
3. Mevcut ihlal vektörü ve her `(j,b)`, `b∈{0,1}` ID-etiket yazımından sonraki bütün ihlal vektörleri.

Birinci imzanın farklı değer sayısı `K_risk(N,m)`, ikincinin `C(N+2m−1,2m−1)`, üçüncünün `(2m)^N` olmalıdır. Üçüncü için `N=0` ayrıca tek boş geçmiş olarak ele alınır. Sayaç formülüyle replay aynı fonksiyonu paylaşmamalı; biri doğrudan kayıt taraması, diğeri signed-count hesabı olmalıdır.

Yöntem karşılaştırması: bütün hipotez sayaçları; `(N,d)`; histogram; histogram+ID eşlemesi; ham ID kayıtları. İçerikli güvenilir düzeltme ve yalnız-ID düzeltmesi ayrı koşullardır. Dış eski-içerik servisi kullanılırsa gönderilen paket boyutu ve servisin belleği ayrıca yazılmalıdır. ID-only girdiyi içerikli girdiye dönüştürüp bu maliyeti gizlemek karşılaştırmayı geçersizleştirir.

Ölçüler: sorgu uyuşmazlığı; sınıf sayısı; ideal kod alt sınırı; gerçek serileştirilmiş bayt; güncelleme ve sorgu başına temel işlemler. Bunların biri diğerinin yerine geçmez. Python sayaç nesnelerinin sayısından byte tasarrufu veya teorik sınıf sayısından hızlı kodlama sonucu çıkarılmamalıdır.

Negatif kontrol: histogramın yokluk denetimini geçebileceği ama yanlış ID'nin kullanıldığı paket; signed-count durumunun bazı geçersiz eski içerikleri ayırt edememesi; izinli sorgu ailesine gözlem-gösterge sorguları eklenince eski signed-count sınıflarının ayrışması. Başarı ölçütü gerçek dünya öğrenme kalitesi değil, açıkça tanımlı bilgi ve işlem sözleşmesine göre yeterliliktir.

## 7. Birincil literatür ve yayın kapsamı

[Mitchell, 1977, *Version Spaces: A Candidate Elimination Approach to Rule Learning*](https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf) tutarlı aday kümelerinin eklenen gözlemlerle güncellenmesini ve verilen dilin/verinin yetersizliği sınırlarını ele alır. Burada tam ihlal sorguları daha zengin çıktı sözleşmesidir. Bu not Mitchell'in algoritmasının yeni sürümü olarak sunulmamalıdır. Tam birincil PDF, özellikle bölüm 2 ve 3.1, 20 Eylül 2026'da okundu.

[Cao ve Yang, 2015, *Towards Making Systems Forget with Machine Unlearning*](https://www.yinzhicao.org/unlearning/UnlearningOakland15.pdf), bölüm IV-A'da tutulan toplamlardan eski gözlemin katkısını çıkarmayı kullanır. Bölüm IV-B, adaptif öğrenicilerde sonraki sorguların ve yakınsama durumunun ek güçlüklerini ayırır. Buradaki cebirsel kapanış bilinen toplamsal güncelleme fikriyle eşleşir. O makalenin bütün öğreniciler için tam aynı yeniden-eğitim dağılımı garanti ettiği veya bu notun modern unlearning değerlendirmesini tamamladığı söylenmez. İlgili birincil PDF bölümleri 20 Eylül 2026'da doğrulandı.

Yayın için savunulabilir katkı adayı, **tek bir küçük deney ailesinde farklı gelecek-işlem sözleşmelerinin ayrı minimal durum gereksinimlerini açıkça karşılaştırmak** olabilir. Özellikle histogramın her sorgu için minimal olmadığı ve ID düzeltmesinin farklı bir bedel getirdiği ayrımı önceki araştırmayı keskinleştirir. Özgünlük taraması ve önerilen bağımsız tarama tamamlanmadan “yeni genel alt sınır” veya “yaşarken öğrenmenin temel ilkesi” şeklinde bir yayın iddiası uygun değildir. Yeni deney veya hakem değerlendirmesi bu notla yapılmış sayılmaz.
