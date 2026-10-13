# Sembol akışından öğrenilen bellek: matematiksel sınırlar ve literatür

Tarih: 22 Eylül 2026. Araştırma turu: 15. Bu belge deney sonuçları görülmeden hazırlanmıştır. Aşağıdaki önermeler koşulları belirtilmiş matematiksel sonuçlardır; deneyin başarılı olduğu veya genel yaşarken öğrenme probleminin çözüldüğü iddiasını içermez. Önceki araştırma kayıtlarının yerine geçmez.

## 1. Bu turun sınadığı ayrım

Öğreniciye hazırlanmış bir geçmiş özeti vermek ile öğrenicinin gelen semboller boyunca kendi çalışma durumunu oluşturması farklı işlemlerdir. Bu turda bir bölüm sıfırlamayla başlar, bir bit dizisi gelir, bölüm sonunda tek bir sorgu yapılır. Doğru etiket ancak tahminden sonra verilir. Bölüm içinde ara çıktı etiketi yoktur.

Üç hedef, son bit, sondan beşinci bit ve bütün bölümün tek/çift toplamıdır. Kısa bölümlerde gecikmiş bitin nasıl tanımlandığı uygulama protokolüne bağlıdır; aşağıdaki gecikme kanıtı en az beş bitlik sözcükleri kullanır ve bu ayrıntıdan etkilenmez.

Deney dili önceden verilmiştir: en fazla sekiz bitlik bağlamlar ve iki durumlu deterministik otomatalar. Bağlam ağacının seçilmesi veya otomata geçişlerinin deneyimle belirlenmesi, bu dil içindeki öğrenmedir. Dilin veya öğrenme algoritmasının deneyimden keşfi değildir. Özellikle sonlu otomata kuramının aşağıdaki temel sonuçları yeni bir buluş olarak sunulamaz.

## 2. Geçmişler ne zaman aynı çalışma durumunu paylaşabilir?

Bit alfabesi `Σ={0,1}` ve sonlu sözcükler kümesi `Σ*` olsun. Bilinen, değişmeyen bir ortamın terminal çıktısını

\[
F:\Sigma^*\longrightarrow\{0,1\}
\]

ile gösterelim. İki geçmişi, bütün olası devamların terminal sonuçları aynıysa eşdeğer sayalım:

\[
h\sim_F h'
\quad\Longleftrightarrow\quad
\forall z\in\Sigma^*,\;F(hz)=F(h'z).
\]

Burada `z` boş sözcük de olabilir. Eşdeğerlik yalnız şu anki cevabı karşılaştırmaz; aynı gelecek girişlerin ardından verilecek bütün cevapları karşılaştırır.

**Önerme 1 — Eşdeğerlik sınıfları yinelemeli bir durum tanımlar.** `h∼Fh'` ise her bit `b` için `hb∼Fh'b` olur. Dolayısıyla

\[
\delta([h],b)=[hb],\qquad o([h])=F(h)
\]

iyi tanımlıdır.

**Kanıt.** İlk eşdeğerlik, `bz` dahil her devam için eşit çıktı verir. Bu, `hb` ve `h'b` için gereken koşuldur. Boş devam için çıktı eşit olduğundan `o` da temsilci seçiminden bağımsızdır.

**Önerme 2 — Tam doğru deterministik bir öngörücü farklı sınıfları birleştiremez.** Aynı iç duruma düşen iki geçmişin bütün devamları boyunca deterministik durum geçişleri ve çıktıları aynı olur. Eğer geçmişler eşdeğer değilse onları ayıran bir devam vardır; en az bir cevap yanlış olur. Bu nedenle her yerde doğru bir deterministik öngörücünün erişilebilir durum sayısı, eşdeğerlik sınıflarının sayısından az olamaz. Bölüm başı durumu ve geçişleri sınıflar üzerinden kurulan öngörücü bu sınıra ulaşır.

Bu, belirli bir hedef için neyin yeterli bellek olduğunu anlatır. Hedef henüz bilinmiyorken o belleğin sonlu deneyimden nasıl güvenilir ve ekonomik olarak bulunacağını tek başına çözmez. Öğrenici, ortamın çalışma durumundan başka aday modeller hakkındaki belirsizliği ve güncelleme için gerekli istatistikleri de taşıyabilir.

## 3. Sınırsız geçmiş etkisi, sınırsız durum sayısı değildir

Tek/çift toplamını

\[
P(h)=\bigoplus_{i=1}^{|h|}h_i
\]

ile tanımlayalım. `P(hz)=P(h) XOR P(z)` olduğundan geçmişler yalnız tek/çift toplamlarına göre iki sınıfa ayrılır. İki durumlu hesaplama yeterlidir:

\[
q_0=0,\qquad q_{t+1}=q_t\oplus b_{t+1},\qquad \hat y=q_t.
\]

Sınıflar farklıdır; boş devam bile onları ayırır. Dolayısıyla iki durum hem yeterli hem zorunludur. Çok eski bir bitin etkisi bu bir bitlik çalışma durumunda sürebilir. Ham geçmişin saklanması gerekmez.

**Önerme 3 — Hiçbir sabit sonlu sonek bütün uzunluklarda parity için yeterli değildir.** Bir `D≥1` için şu eşit uzunluklu iki sözcüğü alalım:

\[
h_0=0^{D+1},\qquad h_1=1\,0^D.
\]

Son `D` bitleri ve uzunlukları aynıdır; parity etiketleri sırasıyla 0 ve 1'dir. Eğitimden sonraki tahmini yalnız bu soneke ve uzunluğa bağlı olan bir model ikisi için aynı `p` olasılığını üretir. Eşit ağırlıklı çift üzerindeki skaler karesel kayıp:

\[
\frac{p^2+(p-1)^2}{2}
=\left(p-\frac12\right)^2+\frac14
\geq\frac14.
\]

İki sınıfın olasılıklarını birlikte toplayan vektör Brier tanımı kullanılırsa alt sınır `1/2` olur. Bu iki ölçü birbiriyle karıştırılmamalıdır. Çifte aynı karar dağılımını veren bir sınıflandırıcının beklenen doğruluğu da en fazla `1/2` olur.

Bu alt sınır her test dağılımının ortalaması için otomatik bir iddia değildir. Karşıörnek çiftine veya eski önekin parity bilgisinin gözlenen sonekten bağımsız ve dengeli olduğu uygun dağılıma aittir. Ayrıca belirli sonlu azami uzunluk için bütün sözcüğü saklayan yeterince büyük bir pencereye karşı olanaksızlık iddiası değildir.

## 4. Kısa ufuk bazen çok daha fazla durum gerektirir

`D_k(h)` sondan `k`'ıncı biti veren görev olsun. Son `k` biti taşıyan kaydırma belleği `2^k` durumla yeterlidir.

**Önerme 4 — Gecikme-k için en az `2^k` deterministik durum gerekir.** Uzunluğu tam `k` olan farklı iki sözcük `u,v` seçelim. Soldan `i` konumunda farklı bir bitleri bulunsun. İkisine de `i−1` sıfır ekleyelim. Ortaya çıkan sözcüklerin sondan `k`'ıncı bitleri, başlangıçtaki `i`'inci bitleridir ve farklıdır. Böylece uzunluğu `k` olan bütün `2^k` sözcük çiftler halinde ayırt edilebilir. Önerme 2 alt sınırı verir.

Sonuç: parity sınırsız zaman ufkuna rağmen iki durumla; gecikme-5 yalnız beş bitlik ufka rağmen 32 durumla gerçekleştirilir. “Bellek daha uzağa uzanıyor”, “daha çok durumu ayırt ediyor” ve “daha fazla öğrenilmiş bilgi taşıyor” ayrı kapasite özellikleridir.

Bu sonuç, her biri iki durumlu adaylardan oluşan otomata sınıfında gecikme-5'in bulunmadığını gösterir. Fakat bütün öğrenicinin yalnız iki durumluk bellek kullandığını göstermez.

## 5. Aday otomata karışımı ile tek otomata aynı nesne değildir

Başlangıç durumu 0 olarak sabitlenmiş iki durumlu, iki giriş sembollü deterministik bir otomatanın dört geçiş girdisi vardır. `2^4=16` geçiş tablosu ve `2^2=4` ikili çıktı tablosu, toplam 64 etiketli aday verir. Bazı adaylar erişilemeyen durumlara sahip olabilir; farklı tablolar aynı girdi-çıktı işlevini gerçekleştirebilir. Dolayısıyla 64 tabloya eşit önsel vermek, farklı işlevlere eşit önsel vermek değildir.

Bayes karışımı bu adayların ağırlıklarını ve çalışma durumlarını tutar. Toplam depolama, tek otomatanın durum sayısından büyüktür. Kaynak karşılaştırmasında ağırlıklar, adayların çalışma durumları ve güncelleme maliyetleri sayılmalıdır.

**Koşullu ek sonuç.** Bir bölüm boyunca sabit kalan `w_j≥0`, `Σw_j=1` ağırlıkları ve ikili aday çıktıları için

\[
p(h)=\sum_j w_j F_j(h)
\]

karışımı, deterministik bir hedef `F` için her sözcükte tam doğru 0 veya 1 üretiyorsa, pozitif ağırlıklı her aday bütün sözcüklerde `F_j=F` olmalıdır. Çünkü toplamın 0 olduğu yerde tüm pozitif terimler 0; toplamın 1 olduğu yerde tüm pozitif ağırlıklı çıktılar 1 olmalıdır. Böylece iki durumlu adayların sabit konveks karışımı da gecikme-5'i her sözcükte tam 0/1 olasılıkla gerçekleştiremez.

Bu sonuç, daha genel bir birleştiriciye, bölüm içinde değişen ağırlıklara veya aday durumları üzerinde yeni bir okuma fonksiyonuna otomatik olarak genişletilemez. Olasılıkların çıktı aşamasında ayrıca gürültüyle yumuşatılması durumunda tam 0/1 çıktılar zaten tasarım gereği elde edilemez. Deneydeki `0.05` olabilirlik gürültüsü bir modelleme tercihidir; doğru çevrenin gerçekten bu oranda gürültülü olduğunu göstermez.

## 6. Sonlu deneyim, sınırsız geleceğin sertifikası değildir

**Önerme 5 — Sınırsız sonlu-otomata ailesinde sonlu terminal örnekler genel doğruluğu sertifikalayamaz.** Eğitim sözcükleri kümesi `S` sonlu olsun. `S`'de bulunmayan daha uzun bir sözcük `w*` seçelim ve

\[
G(h)=P(h)\oplus\mathbf{1}[h=w^*]
\]

tanımlayalım. `G` ile parity, eğitimdeki her örnekte aynı cevabı verir, `w*` üzerinde ayrılır. `G` de sonlu durumla hesaplanabilir: parity otomatası ile yalnız `w*` sözcüğünü tanıyan sonlu otomatanın çarpımı yeterlidir. Birincisi iki; ikincisi en fazla `|w*|+2` durum kullanır. Çarpım en fazla `2(|w*|+2)` durumla gerçekleştirilir.

Dolayısıyla sonlu örneklerde aynı sonuçları veren bütün geçmişleri veya modelleri birleştirmek, onların bütün gelecekte aynı olduğunun kanıtı değildir.

Burada öğrenmenin olanaksızlığı ileri sürülmez. Bilinen sonlu bir aday sınıfı, yeterli ayırt edici örnekler ve uygun varsayımlar altında belirlenebilir. Sınırlı bir ufukta bütün örnekleri sınamak o ufku doğrulayabilir. Bir dağılım için olasılıklı hata güvencesi de bütün olası sözcüklerde kesin doğruluktan farklıdır. Karşıörnek, varsayımsız ve sınırsız gelecek sertifikasına yöneliktir.

## 7. Bilgi erişimi deneyin parçasıdır

Yukarıdaki sonek karşıörneğinin geçerli kullanımı şu koşullara bağlıdır:

1. **Ara hedef etiketleri verilmez.** Her bitten sonra gerçek parity etiketi verilirse son etiket zaten bütün eski parity bilgisini taşır. “Yalnız kısa geçmişe erişim” denilerek bu yan kanal göz ardı edilemez.
2. **İki karşıörnek aynı eğitimli durumdan başlatılır.** Önce `0^9` üzerinde etiket verip modeli güncellemek, sonra `1·0^8` sormak aynı fonksiyonu karşılaştırmaz. Bağımsız kopyalar kullanılmalı ve testler öğrenmeye dahil edilmemelidir.
3. **Bölüm içi parametre güncellemesi sınırlandırılır.** Model etiketsiz her bitten sonra parametrelerini değiştiriyorsa ağırlıkları bir parity biti saklayabilir. Bu durumda nominal sonek penceresi toplam belleğin sınırı değildir. Bu turdaki parametre güncellemeleri terminal etiket anındadır; çalışma durumu güncellemeleri ise gelen bitleri işler.
4. **Uzunluk sayacı tek başına yeterli değildir.** Karşıörnekler eşit uzunluktadır. Bununla birlikte sayaç veya başka değişkenler için izin verilen güncelleme kuralı gerçekten tanımlanmalıdır; “sayaç” adı altında bit içeriğini kodlayan sınırsız bir tamsayı tutulamaz.
5. **Girdi ve etiket geçmişleri aynı erişim koşuluyla kıyaslanır.** Bir yönteme ham bölüm, diğerine yalnız son etiket verilmesi temsil öğrenmesi farkı olarak sunulamaz. Yöntemlerin aynı akışı farklı biçimlerde saklamasına izin verilebilir; bu saklama maliyeti görünür olmalıdır.

## 8. Bellek kuralını değiştirmek, mevcut durumu taşıma sorununu doğurur

Eski geçmiş özeti `φ(h)`, yeni özet `ψ(h)` olsun. Ham geçmiş olmadan eski durumdan yenisini çıkaracak deterministik `T` dönüşümü için gerekli ve yeterli koşul:

\[
\psi=T\circ\phi
\quad\Longleftrightarrow\quad
\bigl(\phi(h)=\phi(h')\Rightarrow\psi(h)=\psi(h')\bigr).
\]

**Kanıt.** Bir `T` varsa aynı girdiye aynı çıktıyı verir. Tersine, sağdaki koşul sağlanıyorsa `T(s)=ψ(h)`, `φ(h)=s` olan herhangi bir `h` seçilerek tanımlanabilir; koşul bu seçimin sonucu değiştirmediğini garanti eder.

Örneğin yalnız parity saklanmışsa son bit sonradan kesin olarak çıkarılamaz: `00` ve `11` aynı uzunluğa ve aynı parity'ye sahiptir; son bitleri farklıdır. Bellek kuralının yeni ayrımlara ihtiyaç duyması halinde eski özet yeterli olmayabilir. Ham izin, yardımcı istatistiğin, yeniden oynatmanın veya yeni gözlemin rolü burada somuttur.

Bu turda her bölüm başında sıfırlama yapılması, değişmiş modelin yeni bir bölümde temiz başlangıçla kullanılmasını sağlar. Bunun bedeli, uzun ve kesintisiz bir bölüm ortasında temsil değiştirmenin bütün sorunlarının sınanmamasıdır. Öğrenmenin program çalışırken sürmesi ile her temsil dönüşümünün kesintisiz biçimde mümkün olması aynı iddia değildir.

## 9. Birincil literatürle eşleşme ve ayrım

**Shalizi ve Crutchfield, 2001.** *Computational Mechanics: Pattern and Prediction, Structure and Simplicity*, Journal of Statistical Physics 104, 817–879. [Yazar PDF'si](https://csc.ucdavis.edu/~cmg/papers/cmppss.pdf), [DOI](https://doi.org/10.1023/A:1010388907793). Geçmişlerin koşullu gelecek dağılımları üzerinden sınıflanması ve minimal öngörü durumları, bu araştırmadaki eşdeğerlik düşüncesinin yakın öncülüdür. Çalışmanın VII.B bölümü, bütün blok uzunluklarının kesin ortak olasılıklarının bilinmesi ve durağanlık gibi varsayımları açıkça sıralar. H.4 bölümü Nerode eşdeğerliğiyle ilişkisini belirtir. Buradaki sonlu, deterministik sorgu deneyi o genel stokastik kuramın bir kanıtı değildir; literatürdeki temsil sonucunu yeni bir kuram diye adlandırmak da doğru olmaz. PDF'nin ilgili tanım, varsayım ve karşılaştırma bölümleri kontrol edilmiştir.

**Littman, Sutton ve Singh, 2001.** *Predictive Representations of State*, Advances in Neural Information Processing Systems 14. [Resmî bildiri PDF'si](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). Çalışma, durumu geçmişin ham kaydı yerine eyleme koşullu gelecek testlerinin tahminleriyle temsil eder. Teorem 1, sonlu POMDP ile temsil edilen bir çevre için, en küçük POMDP'nin durum sayısından daha fazla test gerektirmeyen lineer bir öngörü durumu temsili bulunduğunu gösterir. Kanıt, verilmiş bir POMDP'den temsil kurar. Bu varlık ve temsil sonucu, sonlu pasif deneyimden kendiliğinden keşif veya sınırsız gelişme garantisi değildir. Resmî sayfanın kısa yazar listesi eksik görünse de PDF'de üç yazar vardır; burada PDF esas alınmıştır. Teorem ve kurulum bölümleri kontrol edilmiştir.

Bu kaynaklar deney tasarımını hazır bir genel zekâ çözümüne dönüştürmez. Araştırmanın yerel sorusu, verilen küçük hesaplama dilleri içinde hangi geçmiş ayrımlarının deneyimle kullanılabilir duruma geldiği, hangi maliyetle sürdüğü ve hangi ailelerde yetersiz kaldığıdır.

**Willems, Shtarkov ve Tjalkens, 1995.** *The Context-Tree Weighting Method: Basic Properties*, IEEE Transactions on Information Theory 41(3), 653–664. [Yazarların üniversitesindeki yayımlanmış PDF](https://pure.tue.nl/ws/portalfiles/portal/1383848/Metis122608.pdf), [DOI](https://doi.org/10.1109/18.382012). Çalışma, önceden sınırlanmış bellek derinliğinde ağaç modellerini yinelemeli olarak ağırlıklandırır. Bölüm V, KT yarım-sayaç kestirimini; bölüm VI.A, düğümün kendi kanıtı ile iki çocuk kanıtının çarpımını birleştiren ağaç hesabını tanımlar. Bu turdaki karışımın matematiksel işlemi bu bilinen yapıyla ilişkilidir. Burada ardışık kaynak bitlerinin kodlanması yerine, dışarıdan gelen bölüm sonekine koşullu terminal etiketler öğrenilir. Bu uygulama farkı yeni bir genel öğrenme ilkesi iddiası değildir. Makalenin kodlama başarımı sınırları, deneyimizin MSE'si veya sınırsız ufukta öğrenme başarısı için doğrudan aktarılmamıştır. Üniversite arşivindeki yayımlanmış PDF'nin ilgili kestirim ve yöntem bölümleri kontrol edilmiştir.

## 10. Bu turun yapabileceği ve yapamayacağı çıkarım

Desteklenebilecek sınırlı iddia: terminal geri bildirimle öğrenilen bir geçiş/çıktı modeli, bölüm içinde gerçek sembol akışından durumu ilerletebilir; böylece bütün olay dizisini saklamadan uzun geçmiş etkilerini gelecekteki tahmine taşıyabilir. Bağlam ağacı ve otomata örnekleri bunun farklı, önceden verilmiş temsil aileleridir.

Açık kalanlar: temsil dilinin kendisinin edinimi, farklı diller arasında kaynak hesabıyla seçim, bellek dönüşümü sırasında kaybedilen ayrımlar, görev değişiminin bilinmediği ortamlar, yararlı deneyimin sistemce üretilmesi, birden fazla yetkinliğin birlikte korunması ve genel problem çözme. Matematiksel bir alt problemin açıklanması bu geniş yaşayan öğrenme probleminin tamamının çözümü olarak sayılamaz.
