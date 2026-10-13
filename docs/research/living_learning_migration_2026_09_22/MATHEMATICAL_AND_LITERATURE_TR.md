# Tur 16 — Kural öğrenimi, durum taşıma ve belirsizlik

Muhammed Yasin Yılmaz araştırma kaydı · 22 Eylül 2026. Bu belge ana başarım sonuçları görülmeden, [önceden yazılmış protokole](PROTOCOL.md) göre hazırlanmıştır. Önermeler koşullu matematiksel sonuçlardır; genel yaşarken öğrenmenin çözümü veya yeni bir kuram iddiası değildir.

## 1. Bir kuralı öğrenmek, şu anki durumunu bilmek değildir

Eski geçmiş `h`, saklanan eski özet `φ(h)` ve yeni kuralın gerektirdiği özet `ψ(h)` olsun. Yeni kuralın doğru ve artık sabit olduğu, `ψ(hb)=δ(ψ(h),b)` eşitliğini sağladığı kabul edilir. Eski özetten yeni duruma kesin bir dönüşüm ancak aynı eski özeti paylaşan geçmişler aynı yeni özeti de paylaşıyorsa mümkündür:

\[
\exists T:\psi=T\circ\phi
\iff
\forall h,h',\;\phi(h)=\phi(h')\Rightarrow\psi(h)=\psi(h').
\]

Gereklilik, fonksiyonun aynı girdiye aynı çıktıyı vermesinden gelir. Yeterlilik için `T(s)`, `φ(h)=s` olan herhangi bir geçmişin `ψ(h)` değeri olarak tanımlanır; sağdaki koşul seçimin etkisini kaldırır. Bu, tur 15'teki taşıma sınırının devamıdır.

Kesin dönüşüm yoksa, eldeki bütün bilgi `I` ile uyumlu yeni durumlar korunabilir:

\[
C_0(I)=\{\psi(h):h\text{, }I\text{ ile uyumlu}\},\qquad
C_t=\delta^*(C_0,w_{1:t}).
\]

Burada bilgi, eski parity'yi, bilinen geçmiş uzunluğunu ve yöntemin sakladığı yardımcı izi içerir. Tam küme keyfî bir başlangıç tahmini yapmaz. Sabit deterministik geçiş altında `|C_{t+1}|≤|C_t|`; çünkü bir fonksiyonun bir küme üzerindeki görüntüsü kümeden büyük olamaz. Gerçek durum başlangıçta kümedeyse, aynı gerçek girdiler işlendiği sürece kümede kalır.

Fakat küçülme zorunlu değildir. Ayrıca bu küme bir olasılık dağılımı değildir. Programın kendisi belirsizse durumlar, ait oldukları programla birlikte tutulmalıdır; farklı modellerin durumlarını model kimliğini silerek birleştirmek genel olarak doğru değildir. Deneyde doğru programın kalibrasyondan elde edilip edilmediği bu nedenle ayrıca denetlenir.

## 2. Gelecek girişler kayıp geçmişi geri getirmeden doğru mevcut durumu sağlayabilir

Belirli bir devam `w` sonrasında eski özetten kesin yeni durum hesaplanabilmesi için gerekli ve yeterli koşul, her eski bilgi lifi için

\[
|\delta^*(C_0(I),w)|=1
\]

olmasıdır. Bu, başlangıçta farklı olan yeni durumların aynı devam altında birleşmesidir. Kaybolmuş geçmişi yeniden kurmakla aynı işlem değildir: geçmişin artık gerekli olmayan etkisi ortadan kalkmış olabilir.

**Gecikme-k.** Yeni durum son `k` bitten oluşan kaydırma belleği olsun. Başlangıçta bütün `2^k` durumlar mümkünse, herhangi bir `t` yeni bitinden sonra

\[
|C_t|=2^{\max(k-t,0)}.
\]

Kanıt: `t<k` iken son `t` konum gerçek yeni bitlerle belirlenir, baştaki `k−t` konum serbesttir. `t≥k` olduğunda bütün konumlar yeni girişlerle belirlenmiştir. Böylece gecikme-5 için her devamın beşinci bitinde mevcut durum kesinleşir. Bu işlemde eski beş bit öğrenilmiş olmaz; onların yerini artık gözlenmiş yeni bitler alır.

Eski bilginin yalnız parity olduğu, geçmiş uzunluğunun `N>k` olduğu durumda her `k`-bit sonek her parity ile uyumludur: sonek dışındaki en az bir bit uygun seçilerek parity ayarlanabilir. Dolayısıyla bu deneyin tam başlangıç kümesi gerçekten bütün kaydırma durumlarıdır.

## 3. Bazı eksik başlangıç bilgileri yalnız giriş akışıyla asla giderilemez

**Modulo-3.** Durum `q∈{0,1,2}`, geçiş `δ(q,b)=(q+b) mod 3` olsun. Her bitin geçişi bir permütasyondur. Her devam için

\[
\delta^*(q,w)=q+\sum_i w_i\pmod 3.
\]

Başlangıçtaki farklı durumlar farklı kalır. Başlangıç kümesi üç durumsa, bütün devamlar boyunca üç durumdur. Sıfırı doğru durum sanmak gerçek başlangıç sıfır değilse hatayı gidermez; yanlış fark korunur.

En az beş bitlik geçmişte her parity altında üç kalan da mümkündür. Çift parity için bir sayıları `0,2,4`, tek parity için `1,3,5` yeterlidir; daha uzun sözcükler sıfırla tamamlanabilir. `reserve_8` için geride kalan uzun önekin parity'si bilinse de aynı argüman üç kalanı mümkün kılar. `reserve_mod6` için son beş bitin her düzeni, gerideki önekin bir sayısını uygun altıya göre kalana getirerek uyumlu kılınabilir; protokolün en kısa öneki bile buna yeterlidir.

**Dağılımın karışması, mevcut durumun bilinmesi değildir.** Modulo-3 sayaca bağımsız eşit olasılıklı bitler verilirse, girişler üzerinden ortalaması alınmış Markov zinciri indirgenemez ve aperiodiktir; dağılımı uniform sınıra gider. Oysa gözlenen gerçek devam `w`'ye koşullanınca başlangıç dağılımı yalnız döndürülür. Girdiler bilinmeyen başlangıç durumundan bağımsızsa

\[
H(Q_t\mid I,W=w)=H(Q_0\mid I).
\]

Bu nedenle marjinal yakınsama, gözlemcinin eksik başlangıç kalanını öğrenmesi sayılmaz. Girişlerin olasılığı gizli duruma bağlıysa, gözlenen girişlerin ayrıca kanıt sağlaması mümkündür; yukarıdaki bağımsızlık koşulu kaldırılarak aynı sonuç ileri sürülemez.

## 4. Olası çıktılara eşit ağırlık: posterior değil, koşullu minimax seçim

Olası durumların çıktı kümesi `O={o(q):q∈C}` ve boyutu `r` olsun. Protokol, `O` içindeki farklı çıktılara `1/r`, dışındakilere sıfır olasılık verir. Bu, olası durumları eşit ağırlıklandırmakla da genel olarak aynı değildir; farklı sayıda durum aynı çıktıyı verebilir.

Gerçek çıktının `O` içinde olduğu biliniyor, başka istatistiksel bilgi varsayılmıyorsa, üç sınıflı Brier kaybı için bu seçim minimax'tır:

\[
\min_{p\in\Delta_3}\max_{y\in O}
\sum_{j=0}^2(p_j-\mathbf1[j=y])^2
=1-\frac1r.
\]

Kanıt: En büyük kayıp, `O` içindeki gerçek çıktıların uniform ortalamasından az olamaz. Bu ortalama karesel kayıp, hedefin uniform dağılımında en küçüktür ve değeri `1−1/r` olur. Uniform tahmin her `y∈O` için bu aynı değere ulaştığından alt sınır sağlanır. Böylece `r=1,2,3` için sırasıyla `0,1/2,2/3` elde edilir. Yanlış kesin tahminin kaybı `2`'dir.

Bu güvence gerçek çıktının kümede olmasına bağlıdır. Yanlış başlangıç tahmini veya yanlış öğrenilmiş model bu koşulu bozabilir. Ayrıca uniform seçim, gerçek geçmişlerin koşullu olasılıklarının hesaplandığı anlamına gelmez. Örneğin `N` bağımsız adil geçmiş biti ve bilinen parity `p` için modulo-3 posterioru

\[
\Pr(Q=r\mid p,N)=
\frac{\sum_{0\le j\le N,\;j\bmod3=r,\;j\bmod2=p}\binom Nj}{2^{N-1}}
\]

olur ve sonlu `N` için zorunlu olarak uniform değildir. Deney bu posterioru kullandığını iddia etmez.

## 5. Dış etiketin sağladığı bilgi ayrı sayılmalıdır

Doğru çıktı etiketi `y` alındığında küme

\[
C^+=\{q\in C:o(q)=y\}
\]

olarak süzülür. Modulo-3'ün öğrenilmiş okuması üç kalanın bir permütasyonu olduğundan birebirdir. Tek bir doğru etiket mevcut kalanı tam belirler; bundan sonraki bilinen girişler doğru durumu sürdürür. Bu, giriş geçişlerinin durumları birleştirmesi değildir: dış gözlem alternatifleri eler.

`t=16` tahmini etiketten önce kaydedilmelidir. O etiketten sonraki düzelme, önceki tahmine mal edilemez. Yanlış singleton kümesinin boşalması üzerine okuma tablosundaki bütün uygun durumlara genişleme, açıkça bir onarımdır. Kümenin bu adımda büyüyebilmesi bölüm 1'deki küçülme önermesiyle çelişmez; önerme sabit kümenin yalnız deterministik giriş geçişine uygulanıyordu.

## 6. Yeni yetkinlik eklerken eski yetkinlik ne zaman korunur?

Eski durum uzayı `S`, genişletilmiş durum uzayı `S'` ve eski durumu okuyan izdüşüm `π:S'→S` olsun. Eski geçiş `δ`, yeni geçiş `δ'` için

\[
\pi(\delta'(s',b))=\delta(\pi(s'),b),
\qquad\pi(s'_0)=s_0
\]

sağlanırsa, her devamda yeni sistemin eski duruma izdüşümü özgün eski sistemin durumuna eşittir. Kanıt, devam uzunluğu üzerinde doğrudan tümevarımdır. Eski çıktı da bu izdüşümden okunuyorsa bütün eski cevaplar korunur. Yeni durum bir küme ise bütün adayların aynı doğru eski duruma projekte olması yeterlidir.

Protokolde eski parity'nin ayrı korunup yalnız yeni koordinatın belirsiz bırakılması bu koşulu sağlar. `reset_all`, gerçek eski parity 1 iken başlangıç koşulunu bozar. Sonraki parity geçişleri permütasyon olduğundan fark, doğru eski etiket veya geçmişi yeniden kuran bilgi olmadan kalır. Bu negatif kontrol sinir ağlarına özgü bir unutma sonucu değildir; doğru devam durumunun silinmesinin sonucudur.

Bu koruma yapısal bir koşuldur. Deneyde eski yetkinliğin korunması, bu ayrımı sistemin kendiliğinden keşfettiğini veya genel kaynak paylaşımında her yetkinliği koruyabileceğini göstermez. Arşivin doğruluğu ve yardımcı izlerin başarısı da onların önceden tutulmasına bağlıdır; bunların saklama ve yeniden işleme maliyetleri değerlendirmeden çıkarılamaz.

## 7. Kontrol edilen birincil literatür ve iddia sınırı

**Rabin ve Scott (1959), _Finite Automata and Their Decision Problems_.** IBM Journal of Research and Development 3(2), 114–125. [Makalenin üniversite sunucusundaki kopyası](https://www.cs.miami.edu/~burt/learning/csc427.222/docs/rabin-scott.pdf), [DOI](https://doi.org/10.1147/rd.32.0114). Tanım 11 ve Teorem 11'deki altküme yapısı kontrol edildi: durum kümelerini tek bir deterministik yapının durumları olarak ilerletmek yerleşik bir fikirdir. Burada başlangıç belirsizliğini taşımak için kullanılan küme hesabı yeni bir genel öğrenme ilkesi değildir. Makalenin dil kabulü sonucu ile deneyimizin tahmin başarımı aynı iddia değildir.

**Eppstein (1990), _Reset Sequences for Monotonic Automata_.** SIAM Journal on Computing 19(3), 500–510. [Yazarın üniversite sunucusundaki PDF](https://www.ics.uci.edu/~eppstein/pubs/Epp-SJC-90.pdf), [DOI](https://doi.org/10.1137/0219033). Giriş bölümündeki `δ(w,X)` küme görüntüsü ve bunun tek elemanlı olmasıyla tanımlanan reset dizisi kontrol edildi. Bu araştırmada “reset sözcüğü” literatür terimi, programın veya gerçek çevrenin dışarıdan sıfırlanması anlamına gelmez; olağan girişlerin olası durumları birleştirmesidir. Makale verilen geçişlerden senkronlaştırıcı diziler bulmayı inceler. Bizim deneyimiz temsil dilinin veya geçişlerin genel biçimde icat edilmesini bu sonuçtan çıkarmıyor.

**Travers ve Crutchfield (2011), _Exact Synchronization for Finite-State Sources_.** Journal of Statistical Physics 145(5), 1181–1201. [Yazar PDF'si](https://csc.ucdavis.edu/~cmg/papers/esfs.pdf), [yazarların yayın kaydı](https://csc.ucdavis.edu/~cmg/compmech/pubs/esfs.htm). PDF'nin II.E bölümünde gözlemcinin **doğru modele zaten sahip olması**, durum-belief'i ve sonlu zamanda kesin senkronizasyon tanımları kontrol edildi. III.C'deki belirsizlik yakınsaması, makalenin sonlu durumlu epsilon-makinesi ve kesin senkronizasyon koşulları altındadır. Bu sonuçlar, rastgele her öğrenicide veya burada kullanılan modulo sayacın yalnız dış girişlerini gözlemleyerek durumun mutlaka öğrenileceğini söylemez. Modeli öğrenmek ile bilinen modelin mevcut durumunu öğrenmek arasındaki ayrım bu kaynakta da açıktır.

## 8. Araştırma açısından geçerli çıkarım

Bir yeni kuralın öğrenilmiş olması üç ayrı işi bitirmez: kuralın doğruluğunu belirlemek, mevcut akıştaki doğru çalışma durumunu kurmak ve önceki yetkinlikleri korumak. Bu tur, ikinci ve üçüncü işteki farklı bilgi gereksinimlerini denetler. Önceden saklanmış yardımcı özet, arşivden yeniden yürütme, girişlerin etkisiyle birleşme ve dış etiketle belirsizliği giderme ayrı olanaklardır; hiçbiri bedelsiz geçmiş yeniden inşası değildir.

Temsil dilinin kendisinin edinimi, hangi yardımcı ayrımların gelecekte gerekeceğinin deneyimden seçilmesi, belirsiz programlar arasında güvenli genişleme ve sınırlı kaynakta çok sayıda yetkinliğin birlikte sürmesi açık kalır. Bu taşıma alt probleminin kapanması, ana yaşayan öğrenme probleminin kapanması sayılmaz.
