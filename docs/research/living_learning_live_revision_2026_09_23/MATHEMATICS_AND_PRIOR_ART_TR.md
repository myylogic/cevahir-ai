# Tur 20 — Öğrenilen model ile çalışan durumun ayrı doğruluğu

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Matematik ve birincil literatür

Bu not [protokolü](PROTOCOL.md) açıklar; deney sonucu [rapordadır](REPORT_TR.md). Model edinimi, durum taşıma ve dünya doğruluğu ayrı iddialardır. Aşağıdaki ulaşılabilirlik ve küme taşıma hesapları, bilinen otomata/durum gözlemcisi/program güncelleme fikirleriyle ilişkilidir; yeni bir temel ilke olarak sunulmaz.

## 1. Aynı geçmişte iki farklı programın durumu

A eski, B yeni deterministik makine; başlangıç durumları a₀,b₀ ve geçişleri δ_A,δ_B olsun. Gerçekten yaşanmış bit geçmişi h için

\[
q_A(h)=\delta_A(a_0,h),\qquad q_B(h)=\delta_B(b_0,h).
\]

P(h) ise bütün bitlerin parity'si, bu deneyde verilen ve korunmuş eski bir beceridir. Öğrenilmiş modelin dünyayı doğru açıklaması burada varsayılmaz. q_B(h), **B'nin kendi hesabına göre** h sonrasındaki durumdur; gizli dünyanın durumu değildir.

Eski çalışan durum kümesi U, q_A(h)'yi içeriyorsa şu ilişkiyi kullanabiliriz:

\[
R_{A,B}=\{(P(w),q_A(w),q_B(w)):w\in\{0,1\}^*\}.
\]

Bu ilişki, (0,a₀,b₀)'dan iki sembolle ulaşılabilen üçlülerdir. En çok 2|A||B| farklı üçlü vardır. Tekil bir durum adının aynı numarayla yeni modelde sürmesi anlam taşımadığından, sayısal kimlik eşleme kullanılmaz.

Mevcut parity p biliniyorsa taşıma

\[
T_R(p,U)=\{b:\exists a\in U, (p,a,b)\in R_{A,B}\}
\]

ile yapılır. **Kapsama önermesi:** q_A(h)∈U ve p=P(h) ise q_B(h)∈T_R(p,U). İspat doğrudandır: h'nin kendisi ilgili üçlünün R içinde tanığıdır. Bu sonuç A ve B'nin dünyaya ilişkin doğruluğundan bağımsızdır. Sonraki gerçek bitin bütün adaylara uygulanması kapsama özelliğini korur. Dolayısıyla başlangıçta doğru kapsama ve her değişimde bu taşıma varsa, ardışık yanlış/doğru hipotezlerde dahi modelin geçmiş-yürütme durumu kümede kalır.

Bu, bütün adayların mümkün en dar liste olduğu iddiası değildir. U daha önce genişlemiş olabilir; toplam geçmiş uzunluğu ve bazı eski kısıtlar unutulmuş olabilir. Ulaşılabilir ilişki bütün uzunlukları kapsar. Deneyci zaman indeksini bilir, fakat yöntem onu ek durum filtresi olarak kullanmaz.

## 2. Kısa geçmişin eklediği bilgi

Son s bit biliniyorsa h=v·s yazılabilir. İlişkiyi s boyunca her üç bileşende ilerletelim:

\[
R_{A,B}^{s}=\{(p\oplus P(s),\delta_A(a,s),\delta_B(b,s)):(p,a,b)\in R_{A,B}\}.
\]

Sonra **şimdiki** parity ve eski durum kümesiyle filtreleriz. Gerçek v ilişki içinde, h=v·s de onun görüntüsü içinde olduğundan kapsama sürer. Parity bileşenini s boyunca ilerletmeden şimdiki parity ile filtrelemek yanlış olur; uygulama bunu birlikte yapar.

Eski A'yı kullanmayan güçlü kontrol aynı hesabı yalnız (P(w),q_B(w)) çiftleriyle yapar. Aynı parity ve s altında

\[
T_{A,B}^{s}(p,U)\subseteq T_B^{s}(p)
\]

olur. Sol tarafın tek ilave filtresi A'nın mevcut kümesidir. Bu nedenle aynı s ile iki yöntem farklı kesinlik sağlarsa, fark eski öğrenilmiş durum bilgisinin koşullamasından gelebilir. Deneyde bütün ölçülen kümeler eşit çıktı; bu olası ek yarar gösterilmedi.

Arşiv yöntemi bütün h'yi bilir; δ_B(b₀,h)'yi doğrudan yeniden hesaplar. Tam geçmiş, bu sonlu deterministik program için tek durumu verir. Bunun doğru **dünya** çıktısını vermesi ise B'nin o geçmişte doğru olmasına bağlıdır. Ayrıca tam geçmiş saklama ve yeniden yürütme bedeli vardır.

## 3. Tek değerli taşımanın koşulu

Yalnız (P(h),q_A(h)) okunarak q_B(h)'yi kesin veren bir f varsa, aynı eski özete sahip iki geçmişin yeni durumları eşit olmalıdır:

\[
(P(h),q_A(h))=(P(h'),q_A(h'))\Rightarrow q_B(h)=q_B(h').
\]

Tersi de yeterlidir: her eski özetin gerçekleşmiş temsilcilerinden birini seçip onun yeni durumunu yazmak, bu koşul sayesinde temsilciden bağımsızdır. Bu, önceki araştırmadaki temsil taşıma koşulunun burada sonlu makinelere uygulanmasıdır; yeni teorem diye sunulmaz. Ham durum kimliği yerine bütün gelecek çıktılar için eşdeğerlik isteniyorsa, sağ taraf eşitlik yerine davranışsal eşdeğerlik olarak zayıflatılabilir.

Koşul başarısızsa seçenekler yeni bilgi almak, ek geçmiş izi kullanmak veya birden fazla durum taşımaktır. “Daha güçlü model öğrendik” cümlesi eski özetin kaybettiği ayrımı kendiliğinden geri getirmez.

## 4. Ara temsilden geçerken kayıp

A, birlerin sayısını üçe göre sayan; B, tek durumlu sabit; C yine A olsun. Verilen parity de korunsun. A→C doğrudan taşımada eski modulo durumu yeterlidir. A→B→C yapılırken eski modulo durumu gerçekten bırakılmışsa, yalnız parity altında üç modulo kalanı da mümkündür. Geri dönüşte {0,1,2} kümesi gerekir.

Sekiz bitlik aynı s devamının önüne ε veya `11` eklemek parity'yi aynı bırakır, modulo-3 sonucunu değiştirir. Bu yüzden eşit parity + eşit son sekiz bit de genel olarak kaybı kapatmaz. Uygulamadaki yapısal sınama bu tanığı doğrular. Bu, ana deneyde böyle bir ara-kayıp olayı mutlaka yaşandı demek değildir; yöntemin genellenmesine sınır koyan ayrı karşı örnektir.

Kayıp bütün öğrenilmiş kodun silinmesi anlamına gelmek zorunda değildir: eski programın kodu saklansa bile o anda bulunduğu modulo durumu saklanmıyorsa, mevcut konum geri getirilemeyebilir. Program belleği ile çalışma durumu belleği ayrılır.

## 5. Kesin cevap, doğru cevap değildir

U kümesinin bütün durumları aynı b çıktısını veriyorsa yöntem b der; aksi hâlde kararsızdır. Birden fazla durum aynı mevcut çıktıyı verebilir, fakat ileride ayrılabilir. Bu nedenle tek durum ve tek çıktı ayrı ölçülür.

U, q_B(h)'yi kapsıyorsa ve ortak çıktı varsa, bu çıktı B'nin bütün geçmişi yeniden okumasıyla verdiği çıktıya eşittir. Dolayısıyla kapsama sağlayan yöntemin **kesin** cevabı arşiv hesabıyla çelişmez. Bu, arşiv hesabının dünya açısından doğru olduğunu söylemez.

Zayıf başlangıca dönme kontrolünde taşıma hatası bazen model hatasını örtebilir. İkili çıktıda:

\[
\text{yanlış toplam}=\text{arşiv model hatası}
+\text{arşiv doğruyken bozulan}
-\text{arşiv yanlışken tesadüfen düzelen}.
\]

Deneyde bu eşitlik 1.158 = 418 + 782 − 42 şeklindedir. Sırf doğru cevap yüzdesiyle taşımanın doğruluğunu değerlendirmek, 42 tesadüfi düzeltmeyi yanlış biçimde yarar sayabilir. Kararsız yöntemlerde cevap vermeme ayrıca tutulur; daha az yanlış cevap, aynı kapsama oranında üstünlük anlamına gelmez.

## 6. Birincil kaynaklar ve erişim sınırı

**Dinamik program güncelleme.** Hicks ve Nettles'ın *Dynamic Software Updating* çalışması (TOPLAS 27(6), 2005), çalışan yazılımın kod/veri/tür değişimini ve eski sürümden yenisine geçiş kodunu birlikte ele alır. Dolayısıyla program değiştirmenin yanında çalışma durumunu dönüştürme gereği yeni bir genel fikir değildir. Buradaki deneyimden edinilen küçük makineler, o yazılım sisteminin yeniden üretimi değildir. [Yazarın yayın kaydı ve özeti](https://www.cs.umd.edu/~mwh/papers/HicksNettles03.html) okundu; URL'deki 03 yayın yılı olarak alınmadı. Performans ayrıntıları burada karşılaştırılmadı.

**Durum gözlemcisi.** Shu ve Lin'in *Generalized Detectability for Discrete Event Systems* çalışması (Systems & Control Letters 60(5), 2011), gözlemlerle mevcut/sonraki durumu belirleme ve belirli durum çiftlerini ayırma sorularını inceler. Bu, durum kümesi ile tek durum bilmenin yerleşik ayrımına karşılık gelir. [PubMed kayıt özeti](https://pubmed.ncbi.nlm.nih.gov/21691432/) ve [PMC yazar metninin özeti](https://pmc.ncbi.nlm.nih.gov/articles/PMC3117620/) arama aracının indekslenmiş çıktısında kontrol edildi; iki sayfanın doğrudan açılışı CAPTCHA verdi. Bu turun farklı-model taşıma teoreminin o makaleden alındığı veya bütün yöntem bölümünün incelendiği iddia edilmez.

**Sıfırlamasız otomata edinimi.** Rivest ve Schapire'ın [1989 genişletilmiş özeti](https://people.csail.mit.edu/rivest/pubs/RS89.pdf), homing dizileriyle reset verilmeden öğrenimi ele alır; ana düzeneğinde karşı örnek kaynağı bulunur. Bu kaynak önceki turda okunmuştur. Bu turda 49 sayfalık [dergi metni PDF'si](https://www.schapire.net/papers/homing.pdf) açıldı, fakat araç metin çıkarmadı; bu PDF'nin ayrıntılı okunduğu söylenmez. Bizim taşıma düzeneği homing öğrenimi değildir; ana akıştan yeni etiket gelmez ve kalibrasyon kanalı reset kullanır.

Bu eşleşmeler, “yeni kod + doğru çalışma durumu + kontrollü kullanım” bütünlüğünün literatürden bağımsız keşfedildiği iddiasını engeller. Kullanıcının daha geniş yaşayan öğrenme sorusunu da kapatmaz: gözlem/işlem dilinin edinimi, hedef ve değer oluşumu, ortak temsilde becerilerin yeniden örgütlenmesi hâlâ açıklanmalıdır.
