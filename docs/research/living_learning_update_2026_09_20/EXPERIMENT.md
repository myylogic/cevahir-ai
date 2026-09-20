# Deneyimden edinilen güncelleme yönü: deney ve bağımsız hesap

20 Eylül 2026. Önceki araştırmaya ek; yeni bir genel optimizasyon yöntemi iddiası yok.

## 1. Sınanan hipotez

Geçmiş deneyim, mevcut cevapları taşımadan, aynı yeni kanıtın gelecekteki hesaplama durumunu nasıl değiştireceğini etkileyen bir durum bırakabilir. Bu durum yeni problemleri öğrenmeye yardımcı olabilir; yararın geçmiş ile gelecek arasındaki hangi ortaklığa bağlı olduğu ayrıca sınanmalıdır.

Bu hipotez daha önceki deneylerden farklıdır: sabit güncelleme yöntemleri arasında yalnız araştırmacı seçim yapmıyor. Öğrenici çalışırken kendi güncelleme matrisinin üç bağımsız bileşenini gözlenen sonraki hataları kullanarak değiştiriyor. Ancak kullanılabilecek matris ailesi, meta-güncelleme yöntemi ve değerlendirme sinyali hâlâ verilidir.

## 2. Dondurulmuş protokol

24 bağımsız kaynak yaşam; her birinde 16.384 örnek. Her kaynaktan sonra dört rejimin her birinde dört yeni yaşam; her yeni yaşam 1.536 örnek. Toplam 384 yeni yaşam, 589.824 yeni örnek. Her yeni yaşam aynı örnekler ve etiketlerle dokuz yöntem tarafından işlenir. Yöntemlerin dokuz kez çalıştırılması yeni bağımsız dünya sayısını artırmaz.

Kaynak başına gizli yön farklıdır. Girdiler birim çemberden bağımsız ve eşit dağılımlı gelir. Gizli iki boyutlu hedef, birbirine dik iki yönde rastgele yürür: adım standart sapmaları `.04` ve `.001`. Etiket gürültüsünün standart sapması `.15`. Her adımda önce hedef değişir, sonra girdi görülür ve tahmin yapılır, sonra gürültülü etiket alınır. Temiz hedef ve gizli yön yalnız değerlendiricide bulunur.

Kaynak yaşamda tahminci ve hedef sıfırdan başlar. İlk üç yeni rejimde de başlangıç hedefi sıfırdır; bağımsız yeni yürüyüşler başlar. Bu nedenle bu rejimler özellikle **değişen bir ilişkiyi takip etme** deneyleridir. Dördüncü rejimde hedefin başlangıç bileşenleri bağımsız `N(0,.75²)` dağılımından gelir. Bu, bilinmeyen bir ilişkiyi sıfırdan edinmenin başlangıç maliyetini ekler.

| Yeni rejim | Kaynağa göre değişen |
|---|---|
| Aynı yapı (`same_geometry`) | Yeni hedef yürüyüşü, yeni girdiler ve gürültü; hızlı/yavaş değişim yönleri ortak. |
| Ters yapı (`reversed_geometry`) | Hızlı ve yavaş değişen yönler yer değiştirir. Yavaş yön tam sabit değildir. |
| Her yönde hızlı değişim (`isotropic_change`) | İki yönde de değişim standart sapması `.04` olur. |
| Bilinmeyen başlangıç (`cold_restart`) | Aynı hızlı/yavaş yapı, fakat başlangıç hedefi sıfır değildir ve öğreniciye verilmez. |

Bu rejimler ve bütün ayarlar ilk tam deneyden önce kodlandı; yeni yaşam sonuçlarıyla ayar seçimi yapılmadı. Sonradan eklenen ikinci-moment hesabı bir açıklama/denetimdir; yeni bir bağımsız deney örneklemi değildir.

## 3. Tam olarak ne öğreniliyor?

Tahmin ve temel güncelleme:

\[
\widehat y_t=x_t^Tw_t,\quad e_t=y_t-x_t^Tw_t,\quad
w_{t+1}=w_t+Px_te_t.
\]

`P=[[a,b],[b,d]]` simetriktir. Başlangıç `.08 I`; özdeğerler her meta-güncellemeden sonra `[.002,.6]` aralığına izdüşürülür. Her 64 örneklik blok boyunca `P` sabittir; `w` her örnekte değişir. Bloğun gözlenen ortalama yarım kare hatası, matris için meta-kayıptır. Meta-adım `.1`dir.

Blok başında duyarlılıklar sıfırlanır; `w` sıfırlanmaz. `m=(a,d,b)`, `E_j=∂P/∂m_j`, `s_t^j=∂w_t/∂m_j` için:

\[
s_{t+1}^j=(I-Px_tx_t^T)s_t^j+E_jx_te_t,
\qquad g_j=\frac1{64}\sum_t -e_tx_t^Ts_t^j.
\]

Simetrik matrisin Frobenius gradyanı `[[g_a,g_b/2],[g_b/2,g_d]]` olur. Köşegen dışı `1/2` zorunludur: bağımsız `b` iki matris girişini temsil eder. Önerilen meta-adımın ardından spektral izdüşüm yapılır.

**Türev kapsamı:** Blok başlangıç ağırlığı sabit kabul edildiğinde, blokta sabit kullanılan `P`ye göre tam türev. Önceki bütün değişken matrislere, bütün kaynak yaşamına veya izdüşümden geçen bütün uyarlamalı yola göre tam türev değildir. 64 adımlık ufuk ve duyarlılık sıfırlaması verilmiş tercihlerdir.

Kaynak bitince yalnız `P`nin üç bağımsız sayısı taşınır. Yeni yaşamda `w=0`; duyarlılıklar, birikmiş gradyan ve blok sayacı da sıfırdır. Ham örnek, kaynak tahmin katsayısı veya kaynak hedefi taşınmaz. İlk tahmin dokuz yöntemde tam olarak aynıdır. `P`nin geçmişten bilgi taşıması dışlanmaz; sınanan şey zaten bu bilginin **yeni kanıtın etkisini değiştirmesidir**.

## 4. Karşılaştırmalar ve maliyet

| Yöntem | Yeni yaşama taşınan/uyarlanan |
|---|---|
| Başlangıç sabit (`initial_fixed`) | `.08 I`, meta-öğrenme yok. |
| Öğrenilmiş sabit (`learned_frozen`) | Kaynakta edinilmiş `P`, yeni yaşamda sabit. |
| Öğrenilmiş ve devam (`learned_continue`) | Aynı `P`, yeni yaşamda meta-öğrenme sürer. |
| Yeni başlayan meta (`fresh_meta`) | `.08 I` ile yeni yaşamda meta-öğrenme. |
| Yalnız köşegen (`diagonal_only`) | Edinilmiş `P`nin köşegen dışı bileşeni silinir. |
| Aynı toplam hız (`isotropic_same_trace`) | `tr(P)/2 I`; yön ayrımı silinir. |
| 90° döndürülmüş (`rotated_90`) | Edinilmiş matrisin yönleri değiştirilir. |
| Kaynaktan seçilmiş skaler (`train_selected_scalar`) | Sekiz sabit skalerin kaynakta gözlenen ortalama tahmin hatası en düşük olanı. |
| Kaynaktan seçilmiş matris (`train_selected_matrix`) | Toplam 32 sabit adayın aynı kaynak ölçütüyle seçileni. |

Skaler adaylar `.004,.008,.016,.032,.064,.128,.256,.512`. Diğer 24 aday sekiz yön ile üç özdeğer çiftinden oluşur: `(.12,.012),(.36,.004),(.36,.04)`. Seçim yalnız kaynakta etiket görülmeden yapılan tahminlerin gözlenen etiket hatasına dayanır. Yeni yaşam veya temiz hedef skorları kullanılmaz.

Bu son iki yöntem de geçmişten öğrenir. Öğrenmeyen karşılaştırma diye adlandırılmaz. Ayrıca kaba bir aday ızgarasıdır; tüm sabit matrisler arasındaki en iyi seçenek değildir.

Kaynak meta-öğrenici 16.384 temel güncelleme, 49.152 iki bileşenli duyarlılık güncellemesi ve 256 meta-blok işler. Aday karşılaştırması aynı 16.384 etiketi 32 ayrı tahminciyle işleyerek 524.288 temel güncelleme yapar. Yeni yaşamda tüm yöntemler 1.536 temel güncelleme yapar; iki uyarlanan yöntem ayrıca 4.608 duyarlılık güncellemesi ve 24 meta-blok işler. Bunlar işlem türü sayaçlarıdır; eşit FLOP veya eşit toplam yaşam maliyeti iddiası yoktur. Edinilmiş matris yalnız üç sayı olsa da edinim bedeli sıfır değildir.

## 5. Bütün ana sonuçlar

Tablo temiz hedefe karşı, yeni yaşamın **bütün 1.536 adımı boyunca** etiket öncesi ortalama kare hatadır. Daha düşük daha iyi. Önce aynı kaynak içindeki dört yaşam, sonra 24 kaynak ortalanır.

| Yöntem | Aynı yapı | Ters yapı | Her yönde hızlı | Bilinmeyen başlangıç |
|---|---:|---:|---:|---:|
| Başlangıç sabit | .011426 | .011251 | .021592 | .016174 |
| Öğrenilmiş sabit | .005199 | .034495 | .036181 | .013223 |
| Öğrenilmiş ve devam | .005179 | .011996 | .015551 | .012988 |
| Yeni başlayan meta | .008774 | .008623 | .016619 | .013219 |
| Yalnız köşegen | .007207 | .014041 | .019104 | .011162 |
| Aynı toplam hız | .007762 | .007699 | .013704 | .010516 |
| 90° döndürülmüş | .034933 | .005220 | .038364 | .045334 |
| Kaynaktan seçilmiş skaler | .006868 | .006868 | .010456 | .008538 |
| Kaynaktan seçilmiş matris | .005333 | .067916 | .063193 | .023619 |

Kaynakta öğrenilen ortalama güncelleme kazancı, değerlendiricinin bildiği gerçek hızlı yönde `.261987`, yavaş yönde `.025604`. Yön öğreniciye söylenmemişti. Kaynak başlangıcında ikisi de `.08`di.

**Benzer yapıda aktarım var.** Öğrenilmiş sabit matris, yeni başlayan meta-öğreniciden `.003575` daha düşük hata verdi; eşlenmiş farkın yaklaşık %95 normal aralığı `[-.003693,-.003456]`. Kaynaktan seçilmiş skalerden fark `-.001670`, aralık `[-.001707,-.001632]`. Kaba matris aramasına karşı fark küçük: `-.000134`, aralık `[-.000229,-.000039]`. Bu fark sonlu ızgaraya aittir; meta-yordamın genel optimalitesi değildir.

**Yön bilgisinin etkisi ayrışıyor.** Köşegen dışı bileşeni veya yön ayrımını silmek benzer yapıda hatayı artırdı. Matrisi 90° döndürmek aynı dünyada zararlı, ters dünyada yararlı oldu. Fayda yalnız üç sayı saklanmasından veya toplam adım büyüklüğünden kaynaklanmıyor; taşınan dönüşümün yeni ilişkiyle eşleşmesine bağlı.

**Olumsuz aktarım güçlü.** Ters yapıda öğrenilmiş sabit matris yeni başlayan metadan `.025872` daha kötü; aralık `[.023307,.028438]`. Meta-öğrenmeyi sürdürmek zararı azaltıyor, fakat tüm yaşam ortalamasında yeni başlayan metadan hâlâ `.003373` kötü; aralık `[.003277,.003469]`. Başarılı eski öğrenme alışkanlığının yanlış yerde uygulanması, öğrenilebilirlik için maliyet oluşturdu.

**Başlangıç bilgisi sonucu değiştiriyor.** Bilinmeyen başlangıçta öğrenilmiş sabit ile yeni başlayan meta arasındaki fark `+.00000377`; aralık `[-.001554,.001561]`. Bu karşılaştırmada üstünlük gösterilmedi. Kaynaktan seçilmiş skaler ise öğrenilmiş sabitten `.004685` daha iyi; aralık `[.002856,.006514]` (öğrenilmiş eksi skaler). Hızlı/yavaş değişimi takip etmeyi öğrenmek, her yönde henüz bilinmeyen başlangıç katsayılarını hızla edinmeyi otomatik olarak sağlamadı.

Aralıklar 24 bağımsız kaynak birimi üzerinden yaklaşık normal aralıklardır. Dört yeni yaşam bağımsız meta-durum diye sayılmadı. Çoklu karşılaştırma için eşzamanlı kapsam veya önceden ilan edilmiş hipotez kabul testi yok. Bütün tekil yaşamlar, zaman bölümleri ve son sorgu skorları [ham sonuçta](../../../research/living_learning_update_2026_09_20/results/learned_update.json) korunur.

## 6. Simülasyondan bağımsız ikinci-moment hesabı

Sonuçlar görüldükten sonra, yeni bir ayar aramadan, sabit `P` yöntemlerinin hata dinamiği ayrı kodla hesaplandı. Öğrenici kodu bu hesapta içe aktarılmıyor.

Etiket öncesi hata `d_t=θ_t−w_t`, ikinci moment `C_t=E[d_td_t^T]`, hedef artışı kovaryansı `Q`, gözlem gürültüsü varyansı `r` olsun. Bağımsız ve birim çemberde eşit dağılımlı `x` için:

\[
E[xx^T]=I/2,\qquad
E[xx^TCxx^T]=(\operatorname{tr}(C)I+2C)/8.
\]

Güncellemeden hemen sonraki ikinci moment:

\[
M_P(C)=C-\tfrac12(PC+CP)
 +\tfrac18P(\operatorname{tr}(C)I+2C)P+\tfrac r2P^2.
\]

Sonraki etiketten önce `C_{t+1}=M_P(C_t)+Q`; beklenen temiz tahmin hatası `tr(C_t)/2`. İlk hedef artışı ilk tahminden önce olduğundan, başlangıç `C_0=Q`; bilinmeyen başlangıçta `C_0=.75²I+Q`. Son sorgu son güncellemeden hemen sonra aynı son hedefe karşı ölçülür, bu nedenle orada `tr(M_P(C_last))/2` kullanılır; fazladan `Q` eklenmez.

Bu eşitlikler, kaynakta edinilmiş `P` sabitlenince her yeni yaşamın sonlu zaman beklenen hatasını tam belirler. Yeni yaşamda `P`yi değiştiren iki yöntem için aynı kapalı yineleme kullanılmadı; `P` ve hata birbirine bağımlı olur.

| Sabit matrisli karşılaştırma | Tam beklenen hata | Deneyde gözlenen |
|---|---:|---:|
| Öğrenilmiş — aynı yapı | .00519828 | .00519868 |
| Öğrenilmiş — ters yapı | .03453095 | .03449503 |
| Öğrenilmiş — her yönde hızlı | .03792460 | .03618068 |
| Öğrenilmiş — bilinmeyen başlangıç | .01367785 | .01322295 |
| Seçilmiş skaler — bilinmeyen başlangıç | .00851145 | .00853796 |

Bu hesap, olumlu aktarımı, ters yapıda zararını ve başlangıç belirsizliğinin farklı bedelini aynı veri üretim modeli altında açıklar. İkinci-moment formülü ayrıca 16 yönlü doğrudan çember integraliyle karşılaştırıldı; en büyük fark `5.55×10⁻¹⁷`. Deney ortalamasının tam beklentiye tam eşit olması beklenmez. Her karşılaştırmanın kaynak düzeyindeki Monte Carlo farkı ve yaklaşık aralığı [ayrı hesapta](../../../research/living_learning_update_2026_09_20/results/analytic_risk.json) yer alır. Aralıklar bilimsel genel geçerlilik testi değildir.

## 7. Kontrol ve kapsam sınırı

Birim normlu girdiler ve özdeğer aralığı sayesinde tek etiket güncellemesinde aynı örneğin artığı `e⁺=(1−xᵀPx)e` olur; `xᵀPx∈[.002,.6]`. Bu, o etiketin hatasını azaltır. Etiket yanlışsa temiz dünya hatası artabilir. İzdüşüm; geri bildirimin güvenilirliğini, eski becerilerin korunmasını veya meta-adımın sonraki riskini garanti etmez.

Bu deneyde ham geçmiş saklama seçimi, temsil keşfi, görev anlamı, gecikmiş sonucun atfı, eylem yoluyla deneyim üretimi veya sınırsız yeni beceri edinimi öğrenilmedi. Matris değişse de doğrusal tahmincinin temsil edebildiği fonksiyon sınıfı aynı kaldı. Kazanılan şey, belirli yeni veri bütçesinde sonraki öğrenme yolunun değişmesidir.

Temel yordamlar deterministiktir; rastgelelik deney dünyası ve örneklemededir. Bu sonuç biyolojik iç kimyanın eşdeğerini keşfetmez ve rastgeleliğin gerekli olup olmadığını sınamaz.

## 8. İlgili birincil literatür

Öğrenme oranını geçmiş öğrenme deneyiminden edinme ve değişen doğrusal görevlerde sınama IDBD'de doğrudan ele alınır. Bu yüzden bu turun olumlu sonucu yeni ilke diye sunulamaz. [Sutton, 1992](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf).

Hem görev içindeki öğrenmenin hem görevler boyunca öğrenme yordamının uyarlanmasının çevrimiçi yürütülmesi de incelenmiştir; parametrik mirror descent için koşullu hata sınırları vardır. Buradaki küçük matris deneyi o teoremlerin yeniden üretimi değildir. [Denevi ve diğerleri, 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/e0e2b58d64fb37a2527329a5ce093d80-Abstract.html).

Diğer eşleşmeler ve algoritmalar arasındaki farklar [kuram notunda](UPDATE_THEORY.md), [bellek/kapsam denetiminde](MEMORY_AND_SCOPE_AUDIT.md) ve [bağımsız kod değerlendirmesinde](REVIEW.md) kayıtlıdır. Kaynakların geniş deneyleri burada tekrar üretilmedi.
