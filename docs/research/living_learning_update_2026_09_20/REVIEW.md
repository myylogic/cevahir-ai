# Güncelleme durumunun öğrenilmesi — bağımsız denetim

20 Eylül 2026. Bu dosya yeni deneyin tasarım ve kod denetimidir. Önceki araştırma dosyaları değiştirilmez. **İlk kayıt deney kodu oluşmadan yazıldı; aşağıdaki maddeler bulgu veya geçmiş sonuç değil, yanlışlama koşullarıdır.** Kod ve sonuç denetimi tamamlandıkça son bölüm eklenir.

## 1. Asıl iddia ve doğru nedensel kontrol

Sınanabilir iddia şudur: geçmiş deneyim, yalnız mevcut tahmin katsayılarını değil, daha sonra aynı yeni deneyimden nasıl yararlanıldığını etkileyen kalıcı bir durumu değiştirebilir. Bu iddia, genel bir kendini geliştiren sistem veya yeni optimizasyon ilkesi bulmakla eşdeğer değildir.

Yeni katsayılara geçerken temel tahminci `w` bütün yöntemlerde aynı değere sıfırlanmalı; yalnız güncelleme durumu `η` aktarılmalıdır. İlgili artık gradyan, momentum, meta-türev izi, eski örnek, gizli durum ve rastgelelik de açıkça sayılmalıdır. Yalnız iki öğrenme oranı aktarılıyorsa dosyada gerçekten yalnız bu iki sayı olmalıdır. İlk yeni girdide, herhangi bir geri bildirim alınmadan önce bütün yöntemlerin cevabı aynı olmalıdır. Sonraki fark, eşlenmiş aynı örnek/gürültü/sıra altında ortaya çıkmalıdır.

Bu kontrol eski cevabın doğrudan taşınmasını dışlar; taşınan durumun geçmiş dünyaya dair hiçbir bilgi taşımadığını iddia etmez. Öğrenme oranı, hangi koordinatın daha oynak veya hangi verinin daha güvenilir olduğuna ilişkin bir yanlılık taşıyabilir. Zaten sınanan aktarımın olası içeriği budur. Yeni rastgele katsayıların eski katsayılardan bağımsız olması gerekir; yalnız yeni örnekler seçmek aynı işi görmez.

## 2. Meta-türevin hangi nesnenin türevi olduğu açık olmalı

`ℓ_t(w)=1/2(wᵀx_t−y_t)²`, `g_t=∇ℓ_t`, `H_t=x_tx_tᵀ`, `D=diag(α)` ve `w_{t+1}=w_t−Dg_t` olsun. Sabit bir `α` vektörüne göre tam ileri duyarlılık `S_t=∂w_t/∂α` ise

\[
S_{t+1}=(I-DH_t)S_t-\operatorname{diag}(g_t).
\]

Sonraki kaybın türevi `S_tᵀg_t` olur. `α_i=exp(β_i)` parametrelemesinde doğrudan terim `−diag(α⊙g_t)` olur. Öğrenme oranı lojistikle veya başka bir kısıtla kodlanmışsa bu terime onun doğru türevi girmelidir. İleri duyarlılıkta çapraz koordinatları atmak, Hessian'ı atmak veya yalnız önceki gradyanı kullanmak seçilmiş bir yaklaşım olabilir; tam türev diye adlandırılmamalıdır.

Bir ek ayrım önemlidir: bu eşitlik, tek bir sabit `α` parametresinin bütün geçmiş adımlarda kullanıldığı yola göre tam türevdir. `α_t` her adımda meta-yordamla değişiyorsa, eski duyarlılık izini sürdürmek otomatik olarak bütün uyarlamalı algoritmanın başlangıç durumuna göre tam türevi olmaz. Kodun türev iddiası, yerel adım, dondurulmuş parametreli blok veya tam uyarlamalı yol seçeneklerinden hangisine ait olduğunu belirtmelidir.

En az bir küçük deterministik akışta, tam olarak iddia edilen nesne için merkezi sonlu farkla sayısal karşılaştırma yapılmalıdır. Deney ortalamasının iyi çıkması yanlış meta-türevi doğrulamaz. Oran kesilmesi, normalize edilmesi veya adımın reddi varsa bunların türevdeki konumu ayrıca incelenmelidir.

## 3. Karşılaştırma ve kaynak denetimi

| Kontrol | Hangi açıklamayı sınar? | Yorum sınırı |
|---|---|---|
| Güncelleyici durumunu başlangıca sıfırla | Kazanım önceki deneyimin kalıcı güncelleyici etkisini gerektiriyor mu? | Kötü seçilmiş tek başlangıca üstünlük güçlü optimalite iddiası vermez. |
| Öğrenilmiş oranların koordinatlarını yer değiştir | Fayda koordinatla doğru eşleşmeden geliyor mu? | Bu kontrol genel algoritmik zekâyı ölçmez. |
| Aynı geçmiş veriyle seçilmiş güçlü tek skaler oran | Fayda yalnız daha uygun genel adım boyundan mı geliyor? | Sabit oran seçimine ayrılan arama hesabı sayılmalı. |
| Aynı geçmiş veriyle seçilmiş koordinat oranları | Fayda çevrimiçi meta-yordama özgü mü, yoksa geçmişten sabit bir hiperparametre tahmini yeterli mi? | Bu kontrol de öğrenmedir; 'öğrenmeyen sistem' denmemeli. |
| Ters oynaklık veya değişmiş koordinatlar | Aktarım geçmiş aileye bağımlı mı? | Kötüleşme aktarımın yokluğunu değil, kapsamını gösterebilir. |
| Sonradan en iyi yeni-test oranını seçen referans | Erişilebilir skor zarfını gösterir. | Gerçek kullanılabilir rakip veya ücretsiz öğrenici sayılmamalı. |

Geçmişte tüketilen etiket, akış uzunluğu, meta-güncelleme işlemi, temel güncelleme işlemi ve aktarılmış durum boyutu ayrı verilmeli. Yeni görevde eşit veriden daha iyi yararlanmak, toplam yaşamda daha az veri tüketildiği anlamına gelmez. Toplam maliyet iddiası varsa geçmiş öğrenme maliyetinin kaç yeni yaşam/görevde geri kazanıldığı gösterilmeli. Tam çıkarım yöntemi veya analitik en iyi sabit kontrol varsa bunların da gerçek veri ve model bilgisi hakkı yazılmalıdır.

## 4. Stabilite ve koordinat bağımlılığı

Tek gürültüsüz doğru örneğin hatası, `D` o adımda sabitse,

\[
e_{t+1}(x_t)=(1-x_t^TDx_t)e_t(x_t)
\]

olur. Dolayısıyla `0<x_tᵀDx_t<2`, o örnekte hata küçülmesi için yeterli ve sıfır olmayan hata için gereklidir. Her `α_i`'yi ayrı ayrı üstten kesmek, girdiler sınırsızsa bu koşulu garantilemez. Bu koşul da tek başına yeni örneğe genelleme, gürültülü asimptotik kararlılık veya değişen hedef takibi garantisi değildir. Deney yalnız sayısal taşma yokluğunu sınadıysa 'kararlı' sözcüğü bu anlamla sınırlandırılmalıdır.

Koordinat başına oran, başlangıç temsilinde bir güncelleme geometrisidir. `x'=Ax` altında aynı çıktıyı ve aynı öğrenme yolunu korumak için ağırlıkla birlikte `D'=A^{-T}DA^{-1}` gerekir; genel dönüşümde köşegen matris köşegen kalmaz. Bu önceki durum taşıma sonucuyla tutarlıdır. Koordinat değişiminde başarısızlık, edinilmiş mekanizmanın işe yaramadığını göstermez; bulunduğu temsilin yanlılığını gösterir. Yeni deney bu açıdan önceki sonucu geri almamalıdır.

## 5. Literatür ve yenilik sınırı

Öğrenme oranının geçmiş deneyimden uyarlanması ve bunun drifting hedeflerde sınanması doğrudan IDBD'nin konusudur. Sutton, girdi başına öğrenme oranlarını öğrenilen bir yanlılık olarak kurar. Bu nedenle yeni deney benzer biçimde başarılı olursa, 'sistem deneyimden güncelleme biçimini değiştirebilir' için nedensel bir tanık olabilir; yeni temel optimizasyon yasası iddiası vermez. [Sutton, 1992](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf).

Schraudolph'un stochastic meta-descent çalışması, öğrenme oranı uyarlamasında ardışık gradyan bağıntısının sınırlarını tartışır ve daha zengin duyarlılık yaklaşımı geliştirir. Burada yalnız başarım benzerliği, matematiksel aynı yordam veya aynı garanti anlamına gelmez. [Schraudolph, 1999](https://n.schraudolph.org/pubs/Schraudolph99b.pdf).

Öğrenme oranını güncelleme kuralının türeviyle çevrimiçi değiştirmek, daha yeni hypergradient descent literatüründe de doğrudan ele alınır. Bu eşleşme yeni deneyi gereksiz kılmaz; deneyin rolünü 'bilinen mekanizma ailesinde, eski tahmini sıfırlayan ve yalnız güncelleyiciyi taşıyan kontrollü araştırma' olarak doğru sınırlar. [Baydin ve diğerleri, 2018](https://gbaydin.github.io/assets/pdf/baydin-2018-hypergradient.pdf).

## 6. Kod ve sonuç denetimi

İlk kayıt anında yeni deney kodu henüz mevcut değildi. Bu bölümde sonuç veya yeniden üretim iddiası yoktur. Kod hazır olduğunda; aktarılan durum, türev uygulaması, veri eşleme, güçlü rakipler, değerlendirme sızıntısı ve maliyet kaydı okunarak denetim eklenecektir.

### Sonraki kayıt: tamamlanan kodun bağımsız okunması

Kod ve sonuçlar oluştuktan sonra ayrı incelemeci `learned_update.py` ve kayıtlı JSON'u okudu; dosya değiştirmedi. Sonucu geçersiz kılan hata bildirmedi. Bu incelemeci tüm deneyi tekrar çalıştırmadı; yeniden üretim ayrı doğrulama yordamının işidir.

- 77–101. satırlardaki duyarlılık yinelemesi, blokta sabit `P` için doğru. Köşegen dışı Frobenius gradyanının `1/2` katsayısı doğru. Tam geçmiş türevi iddiası yapılmamalı.
- 174–204. satırlarda yeni yaşamda bütün tahmin katsayıları sıfır; yalnız üç bağımsız matris girişi taşınıyor. İlk bölümün eski iki-oran örneği tasarım kontrolüydü; gerçekleşen deney tam simetrik iki boyutlu matristir.
- Kaynak seçimi gözlenen kaynak hatasıyla yapılıyor. Temiz hedef ve gizli yön öğreniciye verilmemiş. Yeni yaşam örnekleri yöntemler arasında eşlenmiş.
- Hedef artışı ilk tahminden önce; son sorgu son güncellemeden sonra aynı hedefe karşı. Bağımsız ikinci-moment hesabının başlangıç ve bitişi bunu korumalı.
- Dört yeni yaşam önce kaynak içinde ortalanıyor; bağımsız birim 24 kaynak. Yaklaşık normal aralıklar eşzamanlı güvence değil.
- Bilinmeyen başlangıçta öğrenilmiş sabit ile yeni başlayan meta arasındaki fark `+.00000377`, aralık `[-.00155376,.00156131]`; üstünlük gösterilmedi. Kaynaktan seçilmiş skalerden fark `+.00468499`, aralık `[.00285645,.00651354]`; bu negatif korunmalı.
- Ortogonal döndürme testi, genel tersinir koordinat dönüşümü altında meta-yordam eşdeğerliği değildir. İzdüşüm ve Frobenius adımı genel dönüşümde aynı kalmaz.
- Doğrusal temsil sınıfı büyümüyor. Kanıt, aynı yeni deneyimin gelecekteki öğrenme yolunda farklı etkisi olmasıyla sınırlı.

Ana rapor bu sınırları içeriyor. Matematiksel ikinci-moment hesabı sonuçlar görüldükten sonra eklendi; yeni ayar seçilmedi, yeni bağımsız deney örneklemi gibi sunulmadı. Önceki üç turun dosyaları değiştirilmedi.
