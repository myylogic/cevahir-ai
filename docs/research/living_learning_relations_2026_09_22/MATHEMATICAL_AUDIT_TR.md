# Ayrı edinilmiş ilişkilerin aktarımı: sonuç öncesi matematiksel denetim

22 Eylül 2026 · Araştırma turu 12 · Yerel kayıt

Bu not, [PROTOCOL.md](PROTOCOL.md) okunarak, ana deney sonuçları görülmeden yazıldı. Deney kodu incelenmedi veya çalıştırılmadı. Aşağıdaki sonuçlar protokolün tanımladığı sonlu modelden yapılan çıkarımlardır; gözlenmiş başarım, biyolojik mekanizma veya genel living-learning çözümü değildir. Önceki araştırmaların sonuçlarını değiştirmez.

## 1. Öğrenilen şey ve baştan verilen ilişki dili

Durum kümesi S altı elemanlıdır. Tek bir deterministik eylem, f:S→S fonksiyonuyla temsil edilir. Öğreniciye durumların gözlenebilir kimlikleri, eylem kimlikleri, geçiş kaydının anlamı, deterministiklik varsayımı ve fonksiyon bileşimi yordamı verilir. Her kaynakta her durumun sonucu gözlendiği için kaynak fonksiyonu deneyimden tamamen belirlenir. Durum başına ikinci aynı gözlem, gürültüsüz modelde yeni bir mantıksal kısıt eklemez; yine de gerçekleşmiş deneyim maliyetine dahildir.

Bir kaynak fonksiyonunun taşınabilir adayları

\[
\mathcal C(f)=\{\pi\circ f\circ\pi^{-1}:\pi\in\operatorname{Sym}(S)\}
\]

olarak üretilir. Öğrenilen kaynak fonksiyonu değişirse bu sınıf değişir. Fakat hangi yeniden adlandırmaların kabul edileceği ve aynı yapının yeni nesnelere taşınabileceği varsayımı öğrenilmiş değildir. Durumların altı ayrık nesne olarak tanınması da çözülmüş sayılmaz.

İki kaynakta ortak nesne kimliği bulunmaması önemli bir kontroldür. Bununla birlikte hedefte iki eylemin de aynı S üzerindeki fonksiyonlar olarak sunulması ortak bir arayüz verir: ilk eylemin çıktısı ikinci eylemin girdisi olabilir. Bu bağlama kuralının kendisi keşfedilmez. Başarı, verilen dilde deneyimle edinilmiş içeriklerin yeni kullanımını gösterebilir; ilişki dilinin veya ortak anlam alanının sıfırdan doğuşunu göstermez.

## 2. Sınıf büyüklükleri ve ön olasılığın gerçek anlamı

Altı durum üzerinde 720 bijeksiyon vardır; bunların hepsi farklı taşınmış fonksiyon üretmez. Döngü yapısında uzunluğu j olan m_j döngü bulunuyorsa farklı fonksiyon sayısı

\[
N=\frac{6!}{\prod_j j^{m_j}m_j!}
\]

olur. Dolayısıyla altılı döngü sınıfında 120, üç ikili döngü sınıfında 15 fonksiyon vardır. Her fonksiyon ilkinde altı, ikincisinde 48 farklı yeniden adlandırmadan elde edilir. Yinelenen adayların giderilmesi, nesne eşleme sayısını farklı geçiş hipotezi sayısıyla karıştırmayı önler.

`source_both`, sınıflara 1/2 ön olasılık veriyorsa her altılı döngünün başlangıç ağırlığı 1/240, her üç-ikili-döngü fonksiyonununki 1/30 olmalıdır. Birleştirilmiş 135 adaya eşit ağırlık vermek farklı bir modeldir: sınıf olasılıklarını 8/9 ve 1/9 yapar. “İki kaynak eşit” ile “bütün fonksiyonlar eşit” aynı tarafsızlık değildir.

D kanıtıyla uyumlu adayların sınıf içindeki sayısı n_k(D), başlangıç sınıf büyüklüğü N_k ise

\[
\Pr(k\mid D)=
\frac{\Pr(k)\,n_k(D)/N_k}
{\sum_\ell\Pr(\ell)\,n_\ell(D)/N_\ell}.
\]

Sınıf posteriorunun değişmesi yeni bir sınıfın öğrenildiği anlamına gelmez; mevcut alternatifler arasında kanıtla yeniden ağırlıklandırmadır. Ham aday sayılarıyla normalizasyon yapmak başlangıç ön olasılığını değiştirir. Bir sınıfın sıfıra düşmesi, deterministik ve kapalı modelde kesin elenmedir; bu deneyde tekrar belirsizliğe dönüşmesi beklenmez.

`permutation_prior` ve `cycle_catalog` aynı 720 bijektif fonksiyonu kapsayabilir, fakat farklı ön olasılıklar verir: ilki fonksiyonlara, ikincisi döngü türlerine eşittir. Dolayısıyla aralarındaki fark ifade kapasitesi artışı değildir. İki üçlü döngü sınıfının 40, bir sabit nokta ve bir beşli döngü sınıfının 144 fonksiyonu olması, ilgisiz kaynakların aday maliyetini de değiştirir. Eşit kaynak deneyimi, eşit arama maliyeti sağlamaz.

## 3. İki farklı eylemde ortalama matris çarpımı

Satır yönelimini sabitleyelim: P_a(i,j)=1, eylem a durum i'yi j'ye götürüyorsa; aksi halde sıfırdır. İki eylem sırasıyla 0 ve 1 uygulanırsa bileşik matris P_0P_1'dir. Bir başlangıç dağılımı satır vektörü olarak soldan çarpılır.

Öğrenici bağımsız ön olasılıklar kullanıyor ve geçiş kanıtları eylem bazında ayrılıyorsa,

\[
q(P_0,P_1\mid D_0,D_1)=q_0(P_0\mid D_0)q_1(P_1\mid D_1).
\]

Sonlu toplamların dağıtılmasıyla

\[
\mathbb E_q[P_0P_1]
=\mathbb E_{q_0}[P_0]\,\mathbb E_{q_1}[P_1].
\]

Bu yaklaşık bir yöntem değil, **tanımlanan bağımsız posterior altında** tam iki-adım tahminidir. Ters sıra P_1P_0'dır; matrisler genel olarak değişmeli değildir. Kaynak fonksiyonlarının tam öğrenilmiş, sabit kabul edilmesi de bu ayrışmanın koşuludur. İki eylemi etkileyen ortak, belirsiz bir kaynak değişkeni sonradan eklenirse yalnız ayrı marjinalleri tutmak yeterli olmayabilir.

Önemli kapsam ayrımı: uyumlu dünyayı üreten düzenek gerçekte “biri A, diğeri B” kısıtını kullanır. Bu bilgiyi içeren üretici posteriorunda eylem sınıfları bağımlıdır. Protokol bu bilgiyi öğreniciye vermediği için çarpım, üreticinin bütün bilgisine koşullanan posterior diye sunulmamalıdır. Bu, seçilmiş bir önbilgi kısıtıdır; modelin kendi hesabında hata değildir. Sonuç raporu “bağımsız öğrenici modelinde tam” demelidir.

Eylem bazında tabloya dönüş de bağımsızlığı koruyacak biçimde tanımlanabilir. Bileşik doğruluk etiketleri, ortak bir kaynak seçimi veya öğreniciye “iki eylem farklı sınıflardandır” bilgisi verilirse ayrışma yeniden denetlenmelidir.

## 4. Aynı eylemin tekrarında kesin karşı örnek

Genel olarak E[P²]≠E[P]²; ikinci uygulama yeni ve bağımsız bir fonksiyon çekmez, aynı bilinmeyen fonksiyonu tekrar kullanır.

Protokolün B sınıfı doğrudan karşı örnek verir. Üç ikili döngüden oluşan 15 fonksiyonun her biri için P²=I. Hepsine eşit olasılık verildiğinde, tek-adım ortalaması K'nin köşegeni sıfır, diğer hücreleri 1/5'tir. Buna rağmen

\[
\mathbb E[P^2]=I,\qquad
(\mathbb E[P])^2=K^2,
\]

ve K²'nin köşegen hücreleri 1/5, köşegen dışı hücreleri 4/25 olur. Gerçek posterior iki uygulama sonunda başlangıca kesin dönüş öngörürken ortalamaları çarpmak bunu kaybeder. Bu, sayısal hassasiyet sorunu değil, aynı gizli fonksiyona ait bağımlılığın silinmesidir.

Dolayısıyla mevcut iki-farklı-eylem hesabı, `0→0`, `1→1` veya `0→1→0` için otomatik olarak geçerli değildir. Böyle sorgular aynı aday fonksiyonu bütün tekrarlarında koruyan ortak beklentiyle hesaplanmalıdır.

Ek sonuç: kanıt gelmeden A ve B sınıflarının tek-adım ortalamaları aynı K'dir. Bu nedenle `source_a`, `source_b` ve `source_both`, bağımsız iki eylem için başlangıçta aynı tek-adım ve K² bileşik tahminlerini vermelidir. Sınıfları ayıran yarar hedef kanıtıyla ortaya çıkabilir; kaynak sayısının artması tek başına başlangıç üstünlüğünü zorunlu kılmaz. Bu bir deney sonucu değil, modelin simetrisinden çıkan denetlenebilir öngörüdür.

## 5. Çelişki, geri dönüş ve donmuş kontrol

Kapalı deterministik aday sınıfındaki bütün fonksiyonlar elendiğinde normalizasyon sabiti sıfırdır. Aynı model içinde sıradan Bayes koşullaması artık tanımlı değildir. Hedef tablosuna dönüş, önceden verilmiş bir model değiştirme kuralıdır. Görülmemiş hücreleri eşit olasılıklı tutan tablo, bağımsız ve uniform deterministik hücre değerlerine sahip daha geniş bir fonksiyon modeliyle yorumlanabilir; fakat geçişin kendisi tek bir değişmez ön olasılığın koşullaması olarak iddia edilmemelidir.

İlk başarım çalıştırmasından önce protokole eklenen açıklığa göre `frozen_source_both`, son adayları eleyen gözlemi denetim tablosuna yazar, fakat yapısal güncellemeyi reddeder. Tahmin son boş olmayan aday havuzundan gelir; gözlenen hücreler tabloyla sessizce düzeltilmez. Sonraki kanıtlar bu havuzu filtrelemeyi sürdürür. Böylece kontrol, bazı yeni kanıtları bilinçli olarak kullanmaz ve gördüğü bir olguyu da yanlış tahmin edebilir. Bu, “gözlenmiş hücrelerde deterministik sonuç” kuralına amaçlanan istisnadır. Kontrolün dağılımına bütün gözlenmiş verilere koşullanan posterior denmemelidir; yalnız kabul edilmiş kısıtları korur. Adaylar tutulup gözlenen hücrelerin tabloyla ezildiği melez tahminci başka bir kontrol olurdu. Bu not protokoldeki tanımı denetler; kod uygulamasının buna uyduğunu doğrulamaz.

Tam hedef gözlemiyle, doğru sınıfı kapsayan yöntem deterministik hedefi belirleyebilmelidir. Yanlış sınıfta kalan esnek yöntem en geç bu noktada çelişkiyi görüp tabloya dönmelidir. Donmuş kontrolün bu son kontrolü geçmesi ise garanti değildir. Sonuçlara bakmadan bu ayrım kayda geçirilmiştir.

## 6. Çıkarım, yeni öğrenme ve sızıntı sınırları

Kaynak fonksiyonunu gerçekleşmiş geçişlerden edinmek öğrenmedir. Hedef kanıtıyla uyumlu adayları değiştirmek ve bu durumu sonraki tahminlerde kullanmak da sonlu modelde öğrenme/çıkarım sürecidir. Henüz gözlenmemiş bileşik sonucu hesaplamak yeni bir çıkarımdır; bileşik etiketten ayrıca öğrenme değildir. Kaynakta birlikte görülmemiş iki ilişkiyi hedefte kullanmak önemli bir birleşim sınamasıdır, ancak matris çarpımını veya yeni işlem üretme kuralını keşfetmek değildir. Daha az hedef gözleminde düşük hata, bu dil ve kaynak deneyimi altında edinim avantajıdır; toplam yaşam maliyetinin azaldığı ayrıca gösterilmelidir.

Kimlik denetimi yalnız görünen adları karıştırmamalıdır: hedef eyleminin kaynak ailesi, gizli permütasyon, rejim etiketi, üretici indisleri ve rastgelelik durumu öğreniciye taşınmamalıdır. Kaynak yeniden adlandırması, kaynak kayıt sırası ve hedef yeniden adlandırması ayrı denetlenmelidir. Tahmin vektörleri de yeni kimliklere uygun çevrilerek karşılaştırılmalıdır; ham indis eşitliği doğru değişmezlik ölçütü değildir.

Hedefteki ortak nesne kimliği verilen girdidir; bu yüzden iki eylem arasındaki bağlantının tamamının deneyimden keşfedildiği söylenmemelidir. Aynı yaşamda yöntemleri aynı gözlem sırasıyla karşılaştırmak uygundur. Değerlendiricinin gizli geçişleri ve bileşik sonuçları öğreniciye veya model seçimine dönmemelidir. 576 edinim yolu, 576 bağımsız dünya değildir; zaman noktaları ve aynı yaşamın yöntemleri eşleştirilmiş ölçümlerdir.

Bu denetim başarı varsaymaz. Olumlu sonuç, verilen ilişki dilinde edinilmiş kaynak yapılarının yeniden adlandırmaya dayanıklı aktarımını ve belirli yeni bileşimlerde yararını destekleyebilir. Kavram sınırlarının kendiliğinden doğması, duygusal değer değişimi, temsil dilinin gelişmesi veya genel yaşarken öğrenme ilkesi hakkında tek başına karar vermez.
