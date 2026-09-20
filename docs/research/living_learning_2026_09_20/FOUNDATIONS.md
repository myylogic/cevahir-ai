# Yaşarken öğrenme: tanım, yapıcı sonuç ve sınırlar

20 Eylül 2026. Bu metin bir yeni teori veya evrensel çözüm iddiası değildir. Aşağıdaki tanımlar bu araştırmanın ölçüm sözleşmesidir; önermeler belirtilen varsayımlardan türetilmiştir. Literatürün ayrı katkıları [kaynak kaydında](SOURCES.md) gösterilir.

İlk formalizasyondan sonra gelen eylem–veri döngüsü ve iç durum gözlemleri [kapalı döngü devam çalışmasında](CLOSED_LOOP.md) ayrıca işlendi. Aşağıdaki sabit görev riski öğrenme kazanımını tanımlar; tek başına hangi gelecekteki kanıtların alınacağını açıklamaz.

## 1. Öğrenenin sınırı ve nedensel tanım

Öğrenen yalnız sinir ağı çekirdeği olarak tanımlanırsa dışarıdaki öğrenilmiş programlar çekirdeğin öğrenmesi sayılmaz. Öğrenen bütün çalışan sistem olarak tanımlanırsa aynı programlar sistemin öğrenilmiş durumuna dahildir. Bu sınır deneyden önce sabitlenmelidir. Dosyanın içeride veya dışarıda olması bilimsel ayrım değildir; çıktıyı hangi kalıcı durumun nasıl değiştirdiği ayrımdır.

Bir yaşamı şu etkileşimli süreçle gösterelim:

\[
a_t\sim\pi_{s_t}(\cdot\mid o_t;b_t),\qquad
z_t=(o_t,a_t,f_t),\qquad
s_{t+1}=U(s_t,z_t;c_t).
\]

`o` gözlem, `a` eylem/cevap, `f` sonuç veya geri bildirimdir; `f` eksik/gecikmeli olabilir. `b` cevaplama, `c` güncelleme hesap bütçesidir. Durum `s`; sayıları, olasılıkları, programları, grafikleri, geçerlilik koşullarını ve güncelleme yordamını kodlayabilir. Tek vektör ya da belirli sayıda katman varsayılmaz. Değişken büyüklüklü sonlu kodlamalar için `s ∈ ∪ₙ {0,1}ⁿ` yeterli bir soyutlamadır. Bu gösterim tek başına öğrenme algoritması değildir: `U` sabit durum döndürürse hiçbir öğrenme gerçekleşmez.

Deneyim `E` verilen ve verilmeyen iki özdeş başlangıcın kalıcı durumları `s_E` ve `s_0` olsun. Geçici sohbet bağlamını sıfırlayan, ham geçmişi erişilemez yapan ve yalnız ilan edilmiş öğrenilmiş durumu yükleyen müdahaleye `J` diyelim. Ayrılmış görev dağılımı `Q`, kayıp `ℓ` ve aynı cevaplama bütçesi `B` altında:

\[
R_{Q,B}(s)=\mathbb E_{(x,y)\sim Q}[\ell(\operatorname{Exec}_B(J(s),x),y)],
\quad G_{Q,B}(E)=R_{Q,B}(s_0)-R_{Q,B}(s_E).
\]

Deneyime nedensel bağlı kalıcı davranış değişimini **uyarlanma**, pozitif `G`yi bu ölçütte **başarılı öğrenme** diye adlandırıyoruz. Olumsuz `G` yanlış/zararlı öğrenmedir. Her davranış farkını başarı saymıyoruz. Başarı için yalnız ortalama fark yeterli değildir: eşleştirilmiş deneyler, etiket sızıntısı denetimi, belirsizlik ve korunan görevlerde bozulma ayrıca raporlanır. Bu tanım gelecekteki bütün dağılımlarda iyileşmeyi iddia etmez.

Yetenek de görev ve bütçeye görelidir:

\[
\mathcal C_{\epsilon,B}(s)=\{Q\in\mathcal Q:R_{Q,B}(s)\le\epsilon\}.
\]

Deneyimden sonra yeni bir `Q` bu kümeye giriyorsa ölçülmüş yetenek edinimi vardır. Yeni örneğe aktarım, yeni görev ailesine aktarım ve aynı cevabı daha az hesapla üretmek ayrı sonuçlardır. Bir olguyu hatırlamak dar anlamda öğrenme olabilir; bu çalışmadaki güçlü başarı ölçütü görülmemiş girdilere/işlem birleşimlerine aktarımı da ister.

## 2. Kalıcılık sonsuza kadar donmak değildir

Kalıcılık bir yaşam müdahalesine dayanıklılıktır. Yeniden başlatma sonrası, ham kayıt olmadan `R(J(s_E))`nin korunması en küçük sınamadır. Daha güçlü sınama, `k` ilgisiz etkileşimden sonra da kazanımın korunmasıdır; bunun için müdahale dağılımı ve `k` açıkça belirtilmelidir. Her olası gelecek deneyime karşı değişmezlik istemek düzeltilebilirlikle çelişir.

Önerilen genel sözleşme: bir bilginin **geçerlilik koşulları sürdükçe** tutulması, koşullar çürütüldüğünde geri çekilebilir olması. Onaylanmamış aday, kullanımda olan kural ve gerekçesi geri çekilmiş kural aynı statü değildir. Bunun ayrı üç fiziksel bellek gerektirdiği sonucu çıkmaz; etiketli tek yapı da yeterlidir. Hangi değişimin tutulacağı, beklenen gelecek fayda ve saklama/hata bedeliyle ilgilidir; bilinmeyen gelecekte bunu hatasız seçen genel bir yordam yoktur.

Bir kaydı silmenin eski öğrenmeyi sildiği de varsayılamaz. O kayıt bir katsayıya, programa veya başka bir sonuca etki etmişse yalnız kaydın silinmesi etkisini geri almaz. Tam geri alma için yeterli istatistiklerin ters güncellemesi, bağımlılık izleme veya yeniden hesaplama gerekir. Keyfi veride hem kayıpsız sıkıştırma hem bütün geçmiş güncellemeleri bedelsiz geri alma beklenemez.

## 3. Yapıcı yeterlilik sonucu: çevrimiçi sonlu kural öğrenme

Varsayımlar: `p` asal; her gözlenebilir araç kimliği `i` için dünya kararlı bir `fᵢ(x)=aᵢx+bᵢ mod p` fonksiyonudur; girişler farklı seçilebilir; geri bildirim doğru ve kimliğe bağlıdır; hesap ve saklama bu sonlu problem için yeterlidir.

İki farklı girişin sonuçları geldiğinde:

\[
a_i=(y_1-y_0)(x_1-x_0)^{-1}\pmod p,\qquad b_i=y_0-a_ix_0\pmod p.
\]

**Önerme:** Bu katsayılar aynı ailedeki gerçek fonksiyonu bütün girişlerde belirler. Çünkü iki adayın farkı derece en fazla bir polinomdur; iki ayrı kökü varsa sıfır polinomudur. Bu bir sonlu-cisim cebiri sonucudur; deney başarısından türetilmiş genelleme varsayımı değildir.

Üçüncü farklı örnek uygulamada ek tutarlılık kontrolüdür. Aile varsayımı zaten doğruysa üçüncü örnek bilgi açısından gerekli değildir. Aile bilinmiyorsa üçüncü örnek onun doğruluğunu kanıtlamaz.

İki öğrenilmiş işlem birleşebilir:

\[
f_j(f_i(x))=(a_ja_i)x+(a_jb_i+b_j)\pmod p.
\]

İndüksiyonla her sonlu birleşim yürütülebilir. Yeni katsayılar çalışma olayları geldikçe hesaplanıp davranışa katılır; modeli durdurma, eğitim veri kümesi, gradyan veya yeniden dağıtım zorunlu değildir. Gerçek makinede ardışık olay döngüsü yeterlidir. Paralel servis ve sert gerçek zaman garantileri bu sonuçtan çıkmaz.

Bağımsız `i` için kural değişimi diğer kuralların katsayılarını değiştirmiyorsa o araçlardaki davranış korunur. Değişen kurala bağlı birleşimler sürüm bağımlılıkları üzerinden iptal edilirse eski türevler kullanılmaz. Bu sonuç bağımlılıkların eksiksizliği ve doğru araç kimliği varsayımına bağlıdır. Ana Cevahir'in ortak ağırlıkları için otomatik koruma kanıtı değildir.

Bu yapıcı sonuç, çalışma ile öğrenmenin ayrılmasının **mantıksal zorunluluk olmadığını** gösterir. Mekanizma bilinen çevrimiçi çıkarım/cebirsel kestirimdir; yeni mimari değildir. Katsayılar geniş anlamda parametredir. Deney, "parametre diye kodlanamayan öğrenme"yi kanıtlamaz; gradyanla objective optimizasyonunun zorunlu olmadığını gösterir. Her programın kodunu parametre sayarsak bütün sonlu öğrenme durumları parametreleştirilebilir ve başlangıçtaki ayrım içeriksizleşir.

## 4. Daha genel uygulanabilir örnek: hipotez güncellemesi

Sonlu hipotez sınıfı `H` ve doğru geri bildirim altında:

\[
V_{t+1}=\{h\in V_t:h(x_t)=y_t\},\qquad
q_{t+1}(h)=\frac{q_t(h)\mathbf 1[h(x_t)=y_t]}{\sum_gq_t(g)\mathbf 1[g(x_t)=y_t]}.
\]

Birinci denklem sürüm uzayı elemesi, ikincisi gürültüsüz Bayes güncellemesidir. Bunlar kaydedilebilir, çalışırken güncellenebilir ve gelecekteki yeni girişlerde kullanılabilir. İkili çıktı, gerçek `h* ∈ H` ve eşit ağırlıklı çoğunluk tahmini altında her hata kalan adayların en az yarısını eler; dolayısıyla hata sayısı `≤ floor(log₂|H|)` olur. Bu sınır hesap maliyetini, gürültüyü, bilinmeyen aileyi veya dünya değişimini çözmez.

Pozitif farklı önseller aynı gözlemler ve aynı sorular altında aynı destek kümesini bırakır. Dolayısıyla tek adayı **kesin belirlemek** için gereken etiket sayısı değişmez. Fakat belirsizlik sürerken verilen tahmin ve beklenen kayıp değişebilir. İkinci deney bu ayrımı ölçer; önceki temsil filtresi çalışmasının eşdeğerlik sınırı korunur.

Yanlış ailede boş sürüm uzayı gerçek çelişkiyi bildirir; bunu körlemesine "yeni öğrenme" diye olumlamak doğru değildir. Gürültü için sıfır/bir eleme yerine bir gözlem modeli, dünya değişimi için zaman/bağlam ya da değişim modeli gerekir. Örneğin `q⁻ₜ=(1−ρ)qₜ+ρp₀` ardından olasılıksal güncelleme, eski hipotezin tekelini kırabilir; ancak `ρ` seçimi ve bozucu gözlemlere dayanıklılığı bu çalışmada deneysel olarak doğrulanmış değildir.

## 5. Yeni dış veri olmadan ne değişebilir?

Tüm başlangıç bilgisi, algoritması ve erişilebilir deneyimi `S`; dünya hakkında bilinmeyen nicelik `W`; dışarıdan bilgi taşımayan rastgelelik `R` olsun. İçsel işlem `S'=F(S,R)` ve `R ⟂ W | S` ise `W → S → S'` Markov zinciridir:

\[
I(W;S')\le I(W;S).
\]

Bu standart veri işleme eşitsizliğidir; [Cover ve Thomas, Elements of Information Theory](https://doi.org/10.1002/0471200611). Buradaki sonlu değişkenler için kısa türetim: `I(W;S,S')=I(W;S)+I(W;S'|S)=I(W;S)` ve aynı zincir kuralıyla `I(W;S,S')=I(W;S')+I(W;S|S')≥I(W;S')`. `S`nin dışında bırakılmış bir dosya, başka model veya araç daha sonra okunursa varsayım değişir; o kaynak bilgi girişidir. Bilgisayarın dünyada bir eylem gerçekleştirip sonucunu gözlemesi de "kapalı iç işlem" değildir.

Buna rağmen sabit `B` için `R_{Q,B}(S') < R_{Q,B}(S)` olabilir. Bir kanıtı aramak, birleşimi hesaplamak veya bir programı derlemek sonraki kısa hesapta erişilebilir olanı değiştirir. Örneğin sekiz affine çağrı, yedi birleştirme işlemiyle tek affine çağrıya indirgenebilir. Yeni dünya kanıtı gelmez; sonuçların kullanılabilirliği ve maliyeti değişir. Bu, bütçeye bağlı yetenek ölçütünde gelişme, bilginin içsel erişilebilirliği açısından öğrenme; geleneksel terminolojide ise derleme veya planlama olarak adlandırılabilir. İsim seçimi yeni bir mekanizma yaratmaz.

Yeterli hesap ve aynı bilgiye sahip güçlü rakip aynı sonucu buluyorsa "uyku yeni bilgi sağladı" veya "yalnız uyku ile mümkün" iddiası elenir. Toplam yaşam hesabı da sayılmalıdır. İç işlem fırsatının değeri, ertelenen dış işin fırsat maliyetiyle birlikte değerlendirilir. Böyle bir işlem için biyolojik uykuya benzeyen zorunlu bir faz kanıtlanmış değildir. Dyna ve DreamCoder önceden farklı biçimlerini araştırmıştır; [kaynak karşılaştırması](SOURCES.md).

## 6. Evrensel garantilerin neden olamayacağı

**Görülmeyen ayrım:** İki dünya bütün alınmış gözlemlerde aynı, gelecekteki `x*`te farklı doğru cevaba sahip olabilir. Aynı geçmişe sahip öğrenici aynı çıktı dağılımını üretir. `x*`te 1 cevabının olasılığını artırmak, 1'in doğru olduğu dünyada yararlı ve 0'ın doğru olduğu dünyada zararlıdır. Bu nedenle varsayımsız, her dünyada sıkı iyileşme yoktur. Bu yerel karşı örnektir; herhangi bir No Free Lunch teoreminin bütün varsayımlarını buraya taşımıyoruz.

**Çelişkili eski/yeni istek:** Aynı görünür koşulda eski görev 0, yeni görev 1 istiyorsa tek kesin cevabın ikisini birden koruması imkânsızdır. Dengeli ve her sorguda geçmişten bağımsız gizli bağlam için en düşük ortalama hata `1/2`dir. Ayırt edilebilir bağlam veya hedefin değiştiğine dair bilgi gerekir. Zamansal olarak öngörülebilir bağlamda geçmiş yararlı olabilir; imkânsızlık o farklı duruma uygulanmaz.

**Sonlu kalıcı kapasite:** `M` bit durumun en fazla `2^M` farklı hali vardır. `N` bağımsız ikili olguyu her sorguda hatasız hatırlamak `2^N` ayırt edilebilir hali gerektirir. `N>M` ise güvercin yuvası ilkesi gereği iki farklı geçmiş aynı duruma düşer ve en az bir soruda ayırt edilemez. Dünya sıkıştırılabilir olduğunda genelleme yardımcıdır; keyfi bağımsız olgular için sınırsız kayıpsız yaşam belleği sağlamaz.

**Temsil kaybı:** Sabit `φ(x)=φ(x')` fakat gerekli çıktılar farklıysa yalnız `g(φ(x))` ailesindeki hiçbir başlık ikisini doğru yapamaz. Bu, o temsile yönelik somut yetersizlik kanıtıdır. Çözüm daha fazla gözlenebilir bilgi veya ayrım sağlayan temsil/güncellemedir. Bunun mutlaka embedding boyutunu artırması gerekmez: aynı bit bütçesini daha uygun kodlamak da mümkün olabilir. Sonsuz hassasiyetli tek reel sayıyı sınırsız fiziksel bellek gibi kullanmak geçerli kaynak modeli değildir.

**Hesaplanabilirliğin sınırı:** Sonlu deneyimden yeni programlar üretmek, ilk sabit programın pratikte yapamadığı işlerin yapılmasını sağlayabilir. Evrensel yorumlayıcı bakımından bunlar zaten ifade edilebilir olabilir. Bu, öğrenmenin yokluğu anlamına gelmez; ifade edilebilirlik ile sınırlı zamanda erişilebilirlik farklıdır. Fiziksel kaynak/oracle değişimi olmadan hesaplanamaz fonksiyon öğrenildiği iddia edilemez.

## 7. Kontrollü değişime koşullu bir kabul ölçütü

Deneylerimiz genel bir koruma algoritmasını doğrulamadı. Ancak "kontrollü" kelimesi somut bir kabul kuralına dönüştürülebilir. Sabit aday `s'` ile referans `s_ref` için, yeni ve korunan `K` görev dağılımında bağımsız doğrulama örnekleri alalım. `[0,1]` kayıpta fark `d=ℓ(s')−ℓ(s_ref) ∈ [−1,1]`dir. Aday bu örnekler görülmeden seçilmiş olsun. Hoeffding ve birleşim sınırıyla, aynı `n` için olasılık en az `1−δ` iken bütün `j`lerde:

\[
R_j(s')-R_j(s_{ref})\le\bar d_j+\sqrt{2\log(K/\delta)/n}.
\]

Yeni görevde sağ taraf `<−γ`, korunan görevlerde `≤ε_j` olursa aday ilgili dağılımlar için kabul edilebilir. Bu istatistiksel ölçüt aynı anda her girdide doğruluğu kanıtlamaz. Adaptif seçilmiş çok sayıda aday için taze holdout veya açık çoklu-aday düzeltmesi gerekir; aynı holdout'u sınırsız tekrar kullanmak garantiyi bozar. Sürekli güncellemelerde örneğin `δ_t=6δ/(π²t²)` ile toplam hata bütçesi sınırlanabilir. Her adımda küçük bozulmaya izin verilirse bozulmalar birikir: sabit referanslarla değerlendirmek veya toplam `Σₜ ε_{j,t}`yi sınırlamak gerekir.

Örnek: `K=20, δ=.05` ve `.05` güven payı bu kaba dağılımsız sınırda dağılım başına yaklaşık **4.794** bağımsız örnek ister. Birkaç olumlu örnekten güvenli yaşam boyu öğrenme garantisi çıkarılamamasının somut nedenidir. Bu büyük doğrulama yapılmadı. Çevrimdışı cebirsel eşdeğerlik gibi ispatlanabilir güncellemeler için örnek yerine ispat kullanılabilir; gerçek dünyaya uygunluk yine ayrıca gerekir.

## 8. Öğrenmeyi öğrenme ve eksik formül

Bir yaşamın önceki görevleri, sonraki **yeni görevlerden** öğrenme eğrisini iyileştiriyorsa meta-öğrenme vardır. Aynı görevleri daha iyi hatırlamak yeterli değildir. Güncellemenin hiper-durumu `η_t` de `s_t`nin bir parçası olabilir; böylece `U_{η_t}` deneyimle değişir. Bu, sonsuz bir öğrenici zinciri gerektirmez; sabit bir yorumlayıcı güncelleme kurallarını da veri olarak yürütebilir.

Ancak değişebilir olmak doğru yönde gelişmek değildir. Deneyimiz yalnız önceden verilmiş aile ayrımı üzerinde önsel kütlelerini öğrenir. Ayrımın kendisini keşfetmez, genel bir yeni öğrenme algoritması üretmez. Aile değişince negatif aktarım vardır.

Sonuç olarak bir denklem ailesi ve yapıcı yeterli mekanizma bulunmuştur; **tek doğru ve bütün dünyalarda garantili bir yaşam boyu öğrenme formülü bulunmamıştır**. Böyle bir formül için önce dünya sınıfı, kanıt erişimi, hedef görevler, kabul edilen hata, kaynak bütçesi ve değişim rejimi belirtilmelidir. Araştırmanın esas açık sorusu bunların bilinmediği veya öğrenilmesi gerektiği durumda güvenilir ayrımları ve güncellemeleri nasıl keşfedeceğimizdir.
