# Tur 18 — İç değişikliklerin etkileşimi: matematik ve yenilik sınırı

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel araştırma notu

Bu not [önceden yazılan protokolü](PROTOCOL.md) açıklar. Deneyin sonuçları [ayrı rapordadır](REPORT_TR.md). Matematik, verilen sekiz düzenleme kararının etkisini betimler; genel yaşayan öğrenmenin tamamını çözmez. Aşağıdaki açılım klasik sonlu fark / Möbius açılımıdır. Yeni kuram olarak sahiplenilmez.

## 1. Değişiklik ile etkisinin ayrılması

Sabit bir aday program, bir girdi x ve n tane ayrık düzenleme yeri verilsin. A ⊆ {1,…,n}, uygulanmış düzenlemeleri göstersin. Değişmiş programın çıktısını f_x(A) ile gösterelim. Deneyde çıktı F17 içindedir; x aslında iki bileşenli bir girdidir. Öğretmenin hedef fonksiyonunu bilmek, f_x'i ölçmek için gerekli değildir: sistem kendi adayının değiştirilmiş hâlini erişebildiği girdide yürütür.

Her S altkümesi için

\[
m_x(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}f_x(T)
\]

tanımlansın. O zaman bütün A için

\[
f_x(A)=\sum_{S\subseteq A}m_x(S).
\]

İspat: İkinci toplamda bir f_x(T)'nin katsayısı

\[
\sum_{T\subseteq S\subseteq A}(-1)^{|S|-|T|}
=(1-1)^{|A|-|T|}
\]

olur. T=A dışında sıfır, T=A için birdir. Katsayılar tektir: boş kümenin katsayısı doğrudan f_x(∅), diğerleri daha küçük kümelerin katsayıları çıkarılarak belirlenir. İspat yalnız toplama ve çıkarma kullanır; F17'de de geçerlidir. Bölme veya gerçel sayılarda sıralama gerektirmez.

Derecesi en çok k olan yaklaşım

\[
\widehat f_{x,k}(A)=\sum_{\substack{S\subseteq A\\|S|\leq k}}m_x(S)
\]

olsun. Sabit x için bu yaklaşımın **bütün maskelerde** doğru olması, k'dan yüksek bütün katsayıların sıfır olmasına denktir. Fakat yalnız gereken tek bir A'da doğru olması daha zayıf bir koşuldur: o A'nın içermediği etkileşimler sonucu etkilemez; içerilen terimler de bazı girdilerde sıfırlanabilir veya birbirini götürebilir.

Bu yüzden “programda üçüncü dereceden etkileşim var” ile “bu hedefi bulmak için üçüncü derece zorunlu” aynı önerme değildir. Deney bu ayrımı özellikle gösterir. On altı kanıt girdisindeki açılımın uygunluğu, denenmemiş girdilere otomatik garanti de vermez.

## 2. Tek başına etkisiz görünen değişim nasıl gerekli olabilir?

Bir düzenleme yerindeki önceki çıktı a_i(x), düzenleme farkı d_i(x), ikili seçim z_i ∈ {0,1} olsun. Yerel çıktı a_i+z_i d_i'dir. İki yerelin çarpımında

\[
(a_i+z_i d_i)(a_j+z_j d_j)
=a_i a_j+z_i d_i a_j+z_j d_j a_i+z_i z_j d_i d_j.
\]

Son terim tekil ölçümlerden görünmeyebilir. Özellikle a_i=a_j=0 iken hiçbir tekil düzenleme çıktıyı değiştirmez; ikisi birlikte d_i d_j üretir. d_i=d_j=1 seçimi, F17'de de açık karşı örnektir. Sıfır olan f ile z_i z_j olan f, boş ve bütün tekil maskelerde aynı ölçümü verir. Yalnız bu ölçümlere erişen bir yordam ikisini genel olarak ayıramaz. İç yapı bilgisi veya başka kanıt varsa bu sınırlama aynı biçimde uygulanmaz.

Bu karşı örnek “tek tek işe yaramıyor, o hâlde birlikte de gereksiz” çıkarımını çürütür. Rastgelelik gerekmez; gereken düzenleme deterministic olarak da bulunabilir. Fakat karşı örnek hangi iki yerin seçileceğini bedelsiz söylemez.

Üç yerin çarpımı ayrıca z_i z_j z_k d_i d_j d_k terimini içerir. Genel olarak ∏z_i terimi, tam ilgili kümeden daha küçük bütün müdahalelerde görünmez. Sabit düşük derece, keyfî yüksek dereceli değişim etkileşimlerinin evrensel açıklaması değildir. Bu bir bilinen sonlu-fark gözlenebilirlik sınırıdır; genel öğrenmenin imkânsızlığı veya her öğrenicide böyle bir model zorunluluğu değildir.

Deneyde toplama ile birleşen yereller en çok birinci dereceyi; çift çarpımlar en çok ikinciyi; üçlü çarpımlar en çok üçüncüyü üretir. Sekiz yer birbirinden ayrı olduğu için bir yerdeki değiştirme diğer eşleşmenin yapısını bozmaz. İç içe düzenlemelere geçildiğinde bu kolaylığın yeniden değerlendirilmesi gerekir.

## 3. Etki, doğruluk ve köken aynı şey değildir

f_x(A) sistemin kendi adayının sonucudur. Hedef etiketi y(x) ile uyuşmak ise görev başarısının erişilebilir ölçüsüdür. İç etki modelini öğrenmek, y'nin bütün alanını öğrenmek anlamına gelmez. Öğrenici bu turda 16 etiketi bilir; 289 girişin geri kalanı yalnız araştırmacı değerlendirmesidir.

İki farklı A ve B bütün girişlerde aynı çıktı üretebilir. Bu durumda doğru programı bulmak, öğretmenin hangi iç yerleri değiştirdiğini teşhis etmekle eşdeğer değildir. Deneyde doğru ama ekilmiş maskeden farklı cevaplar bu nedenle ayrı sayılır; hatalı kabul olarak etiketlenmez. Biyolojik neden, dış dünyanın nedensel yapısı veya zihindeki gerçek temsil yolları hakkında buradan sonuç çıkarılmaz.

Birinci derece arama doğrulanmış aday bulamayınca ikinci dereceye geçiş, burada hazır yazılmış bir öğrenme politikasıdır. Geçiş için doğrudan yanlış çıkan model öngörüsü şart koşulmamıştır. Aday bulunamaması; yetersiz model, yetersiz değişiklik ailesi veya kısıtlı arama nedeniyle de olabilir. Deneyde hiçbir kabul öngörüsü gerçek çalıştırmayla çürütülmedi; genişlemeler aday yokluğundan doğdu. Bu nedenle “öğrenici etkileşim derecesinin doğru olduğunu kanıtladı” denmez.

## 4. Kalıcı olarak öğrenilen şeyin kapsamı

`adaptive_persistent` geçmişten bir tercih taşır: bundan sonraki görevlerde etkileri derece 2 ile incele. Bu seçim bir bitlik kalıcı durumdur. Etki katsayıları her yeni görev için yeniden ölçülür. Öğrenilen programlar da saklanır, fakat sonraki aramaya aday üretmez; bu kütüphaneden aktarım iddiası yoktur. Kaynak onarım şeması zaten tur 17'de edinilmiştir ve bütün yöntemlerde aynıdır.

İkinci dereceye gereksinim duyulan bir görevde, kalıcı seçim birinci derecenin başarısız tarama bedelini önleyebilir. Birinci derece zaten yeterliyse ise ek çift ölçümleri gereksiz olabilir. Aynı ikili çarpım ailesinde bile bazı hedef maskeleri tekil etkilerle bulunabilir. Dolayısıyla “aynı aileye dönmek” dahi global tercihin faydasını tek başına garanti etmez.

Yaşam düzeyindeki maliyet farkı görev başına farkların toplamıdır:

\[
\Delta C=\sum_t(C_{\mathrm{kalıcı},t}-C_{\mathrm{sıfırlanan},t}).
\]

Bu özdeşlik bir yeni optimizasyon teoremi değildir. Hangi gelecekteki görevlerde hangi işin atlandığını veya gereksiz yapıldığını hesaplamayı zorunlu kılar. Bu tur global bütçe yönetimi, bellek optimizasyonu, tercihi geri alma veya bağlama göre ayrı tercihler öğrenme yöntemini sınamaz.

## 5. Birincil literatür ve kontrol edilmeyen iddialar

**Klasik açılım.** Couceiro, Lehtonen ve Waldhauser'ın 2012 araştırma makalesinin 3. bölümü, gerçel değerli ikili giriş fonksiyonlarının tekil çokdoğrusal gösterimini ve Möbius dönüşümünü bilinen sonuç olarak verir; Teorem 3.1 Hammer–Rudeanu'ya atıflıdır. Bizim F17 hesabımız yukarıdaki cebirsel türetimdir. Kaynak bu turdaki öğrenme başarısını veya özgünlük iddiasını desteklemez. [Yazar arşivindeki tam metin](https://www.math.u-szeged.hu/~twaldha/cikkek/2012_3.pdf), Discrete Applied Mathematics 160, 383–390.

**Programın içine müdahale.** Zeller'ın FSE 2002 çalışması, yürütme durumlarını değiştirip test ederek hata oluşumuna katkı veren durum farklarını izler. Dolayısıyla “kendi programında değişiklik yapıp sonucu ölçmek” yeni bir genel fikir değildir. Buradaki sabit dönüşüm maskeleri, hedef örnekleri ve kalıcı derece tercihi o çalışmanın birebir uygulaması olarak sunulmaz. [Yazarın bildiri sayfası ve özeti](https://www.st.cs.uni-saarland.de/papers/fse2002/) kontrol edildi.

**Testlerden onarım.** Roychoudhury'nin araştırma grubunun proje sayfası, SemFix (ICSE 2013) ve Angelix (ICSE 2016) dâhil testler ve anlamsal çıkarımla onarım çalışmalarını kaydeder. Bu, verilen örneklerle program değişikliği üretmenin yerleşik bir araştırma alanı olduğunu doğrular. SemFix PDF erişimi 403 verdi; bu turda ayrıntılı algoritma, performans veya eşdeğerlik karşılaştırması yapıldığı iddia edilmez. [Yazar grubunun araştırma kaydı](https://abhikrc.com/projects/Repair/).

Deney, bu yayınları yeniden üretmez ve rakiplerini yenmez. Özellikle AST ve işlemlerin anlamı erişilebilirken, yapıyı doğrudan açımlayan bir sembolik analiz kontrolü eklenmemiştir. Tam altküme araması bu turun temsil-uyumlu güçlü kontrolüdür; mümkün bütün program analizi yöntemlerinin en iyisi değildir.

## 6. Ana araştırmada kalan sonuç

Şu dar önerme desteklenebilir: **İç değişikliklerin tekil etkileri, bazı birlikte düzeltmeleri açıklamaya yetmez; geçmişte yararlı bulunan daha zengin bir değişim modelini kalıcı kılmak da bütün yeni görevlerde yarar sağlamaz.** İlk bölüm klasik matematiksel karşı örnek, ikinci bölüm bu protokolde deneysel sonuçtur.

Ana soru hâlâ daha geniştir: deneyimle gelecekteki hesaplama kapasitesinin kalıcı, kontrollü ve genellenebilir değişimi. Yeni kavram ve ayrım üretimi, eski anlamların yeniden örgütlenmesi, edinilmiş parçaların yeni görevlerde birleşmesi, bellek ve hesap bütçesi, geri bildirimin kaynağı, aynı anda çalışırken güncelleme ve önceki yetilerin korunması bu açılımdan çıkmaz. Etkileşim ölçümü bunları birleştiren temel computational ilke olarak ilan edilmez.
