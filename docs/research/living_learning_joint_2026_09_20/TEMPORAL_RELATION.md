# Sonucun geçmişle ilişkisini hazır deneyim kimliği olmadan öğrenmek

20 Eylül 2026. [Kod](../../../research/living_learning_joint_2026_09_20/temporal_relation.py), [bütün sonuçlar](../../../research/living_learning_joint_2026_09_20/results/temporal_relation.json), [koşullu ispat ve nedensellik ayrımı](PRINCIPLE_AUDIT.md).

**Sonuç:** Sıralı bir akışta gözlenen sonucun hangi geçmiş girdilerle ilişkili olduğu, sonuç başına tek bir doğru deneyim kimliği verilmeden öğrenilebilir. Gözlenen sonuç birden fazla geçmiş girdinin birleşimiyse, onu tek olaya bağlamak zaten yanlış problem tanımı olabilir. Bu deney, bilinen sonlu gecikme özelliklerinde doğrusal ilişkiyi öğrenir; gizli nedensel sorumluluk veya keyfî gecikmeli etiket eşlemesi keşfetmez.

## Soru ve hesaplama

Her anda yeni bir sayısal girdi `u_t` gelir. Sistem güncel sensör çıktısını tahmin eder, ardından gerçek gürültülü `y_t` değerini gözler. İlk dünya:

\[
y_t=.2+1.1u_{t-2}-.7u_{t-5}+\xi_t,\quad \xi_t\sim N(0,.05^2).
\]

Öğreniciye bu katsayılar veya etkili iki gecikme verilmez. `φ_t=(1,u_t,u_{t-1},...,u_{t-6})` özellik sözlüğü ve örnek sırası verilir. Kalıcı `w,P` ile recursive least squares güncellemesi kullanılır:

\[
k_t=\frac{P_t\phi_t}{\rho+\phi_t^TP_t\phi_t},\quad
w_{t+1}=w_t+k_t(y_t-w_t^T\phi_t),\quad
P_{t+1}=\frac{P_t-k_t\phi_t^TP_t}{\rho}.
\]

Başlangıç `w=0,P=I`. `ρ=1` bütün geçmişin ridge etkisini biriktirir; `ρ=.995` eski etkileri zamanla azaltır. Bu bilinen bir kestirim güncellemesidir, yeni öğrenme algoritması değildir. Sayısal etiket, sonlu doğrusal geçmiş dili, zaman hizası ve pozitif deneyde dışsal girdi verilmiştir.

Önceki [gecikmiş etiket deneyinden](../living_learning_reassessment_2026_09_20/CREDIT_ASSIGNMENT.md) kaldırılan varsayım, her sonucun bir eski isteğin ID'siyle gelmesidir. **Bu aynı ortamdan yalnız ID alanını silmek değildir:** şimdi bir sensör çıktısının birden fazla geçmiş girdiden oluştuğu ayrı bir sistem belirleme problemi kurulmuştur. Bilinmeyen permütasyonla gelen denetimli etiketlerin eşlemesi çözülmüş sayılmaz.

## Deney ve kontroller

Her rejimde 24 eşleştirilmiş yaşam, 1.800 gözlem. Bütün yöntemler aynı girdi ve gürültüyü paylaşır. Güncel sonuç önce tahmin edilir, sonra öğrenilir. Bağımsız 512 yeni geçmiş vektörüyle yaşam sonu değerlendirme yapılır; bu cevaplar öğreniciye verilmez. Ayrıca bütün yaşam ve son 200 adımın servis hatası ölçülür.

Yöntemler: hiç öğrenmeyen; yalnız güncel girdiyi kullanan iki katsayılı RLS; bütün 0–6 gecikmelerini kullanan sekiz katsayılı RLS; aynı dilde `.995` unutmalı RLS; gerçek etkili iki gecikmeyi önceden bilen üç katsayılı referans. Sonuncusu **ayrıcalıklı bilgiye sahip referanstır**, adil olarak aynı önbilgili rakip değildir. Değişim anını ve yeni katsayıyı o da bilmez.

Rejimler:

- Bağımsız düzgün `[-1,1]` girdiler, sabit ilişki.
- Aynı girdiler; 900 sıfır-tabanlı zamanda katsayılar `1.1,−.7→−.4,1.2` değişir; bildirim yok.
- Girdiler sürekli `+1,−1` dönüşümlüdür; ilişki sabittir.
- İkinci etki gerçekte gecikme 10'dadır; aday dil 0–6 kalır. Ayrıcalıklı referans 2 ve 10'u bilir ve daha uzun tarih okur.

Bu seçimler sonuçlar görülmeden yapıldı; öğrenme oranı/unutma katsayısı veya değerlendirme üzerinde arama yapılmadı. Girdiler öğrenicinin politikasıyla seçilmez; kapalı döngü kontrol veya yeniden sınama deneyi değildir.

## Sonuçlar

Temiz hedefe karşı ortalama kare hata; bağımsız yeni geçmişlerde 24 yaşam ortalaması:

| Yöntem | Sabit, bağımsız girdi | Katsayı değişimi | Dönüşümlü girdiden sonra yeni bağımsız geçmiş | Etki aday gecikmelerin dışında |
|---|---:|---:|---:|---:|
| Hiç öğrenmeyen | `.609923` | `.570297` | `.609923` | `.613588` |
| Yalnız güncel girdi | `.569447` | `.531311` | `1.651309` | `.572413` |
| Bütün gecikmeler, tüm geçmiş | `.00001344` | `.488246` | `.415140` | `.165005` |
| Bütün gecikmeler, unutmalı | `.00005410` | `.00029226` | `.415222` | `.167184` |
| Gerçek gecikmeleri bilen referans | `.00000565` | `.487417` | `.026810` | `.00000587` |

Sabit bağımsız dünyada bütün gecikmeli yöntemin katsayı vektörü hatası ortalama `.005906`; model hangi eski girdilerin ne ölçüde kullanılması gerektiğini deneyimden edinmiştir. Bütün yaşam servis MSE'si `.002912`. Sadece güncel girdi geçmiş katkılarını ayıramaz; daha çok aynı biçimde güncelleme yapmak yeterli değildir.

Katsayı değişiminde unutmalı yöntem bütün yaşam servis hatasını `.492554→.114709` azaltır; eşleştirilmiş fark `−.377846`, yaklaşık %95 aralık `[−.383780,−.371911]`. Fakat sabit dünyada son yeni-geçmiş hatası tüm-geçmiş yöntemine göre `+.00004066` daha yüksektir; aralık `[+.00003160,+.00004973]`. Unutma her koşulda gelişme değildir. Unutma parametresi bu deneyde öğrenilmez; uygun olması araştırmacının seçimine ve değişim rejimine bağlıdır.

## Başarıyı yanlış yorumlatan iki karşı örnek

**Aynı örüntüye uyum, genel geçmiş ilişkisini edinmek değildir.** Dönüşümlü girdilerde tüm gecikmeli model son 200 gerçek adımda yalnız `.00000318` hata verir. Yeni bağımsız geçmişlerde hata `.415140` olur. Çünkü çok sayıda gecikme, gözlenen tek örüntüde aynı veya işareti ters girdi üretir; doğru katsayıların ayrı ayrı belirlenmesine yetecek ayrım yoktur.

Tam küçük tanık: `u_t=−u_{t−1}` iken `y_t=u_t` ile `y_t=−u_{t−1}` bütün gözlemlerde aynıdır. Yeni `(u_t,u_{t−1})=(1,0)` için doğru cevapları `1` ve `0` olur. Gözlemlenen düzenlilik gerçek ve kullanışlı olabilir; onun görülmemiş devamlar için yeterli olduğu sonucu çıkmaz. Gerçek iki gecikmeyi bilen referansın da yeni bağımsız girdilerde sıfıra inmemesi, yalnız destek bilgisinin kollinear katsayıları ayırmaya yetmediğini gösterir.

**Doğru ilişki dili dışında kalan tarih.** Eksik `u_{t−10}` bağımsız ve sıfır ortalamalıyken, 0–6 tarihini kullanan mevcut sabit tahmincinin bağımsız yeni-geçmiş MSE'si en az

\[
.7^2\operatorname{Var}(u)=.49/3=.163333\ldots
\]

olur. Ölçülen `.165005` bunun yakınındadır. Bu sınır herhangi bir durumlu öğrenicinin bütün akışlar için sınırı değildir: daha uzun tarihi ayrıca kodlayan bir yöntem eksik değişkene erişebilir. Bizim değerlendirmemizde yeni bağımsız geçmişin 10'uncu gecikmesini bu yöntemin tahmin fonksiyonu okumaz. Etkiyi dilin dışına koyduğumuz kontrol tam bu kısıtı sınar.

**Öngörüsel ilişki ile neden farklıdır.** Ek tam yapısal tanıkta dışsal `U` eşit olasılıklı bit, normal davranış `A=U`. Bir dünyada `Y=U`, diğerinde `Y=A`. Bütün normal gözlemler iki dünyada aynıdır. Fakat `do(A=1)` altında `P(Y=1)` birinde `.5`, diğerinde `1`. Bu deneyde müdahale öğrenicisi çalıştırılmadı; aynı gözlemlerin iki nedensel açıklamayı ayıramadığını gösteren hesap verildi. Simülatörün denklemini bizim bilmemiz, öğrenicinin nedenselliği bulması değildir. [Pearl'in yapısal yaklaşımı](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/).

## Kalıcı davranış ve kalıcı güncelleme

900 gözlemde `w,P` ve gerekli son girişler JSON'a çevrilip tekrar yüklenir; kalan 900 gözlemde bütün tahmin ve son durumlar her rejim/yöntemde aynıdır. Bir ayrı fixture yeni Python sürecinde tahmin ve sonraki güncellemeyi de birebir üretir. Yalnız `w`yi saklayıp `P`yi yeniden başlatan sürüm başlangıçta bütün sorgulara aynı cevabı verir, daha sonra farklı öğrenir. Sabit dünyada tam gecikmeli model için en büyük gelecek tahmin farkının yaşam ortalaması `.067295`; değişen dünyada `2.742197`. Bu fark kötüleşme ölçüsü değildir; geçmişin güncellemeyi de belirlediğini gösterir. Matris sıfırlamanın bazı değişimlerde daha hızlı uyum sağlaması mümkündür.

Tam gecikmeli öğrenici sekiz katsayı, 64 matris skaları ve yedi giriş değeri kullanır. Ortam ve ayrıcalıklı kontrol 11 giriş saklar; standart yöntemin özellik yordamı yalnız yedisini okur. Kod, belirgin çarp-topla ve bölme işlemlerini sayar; döngüler, bütün veri üretimi ve tanısal kopyalar gerçek zaman hızlanması olarak hesaplanmaz. Tam yaşam bit sayısının sınırsız sabit olduğu iddia edilmez.

Tek bağımlı gözlem yolundan doğrusal dinamikleri belirleme literatürde koşullu garantilere sahiptir; [Simchowitz ve diğerleri, 2018](https://proceedings.mlr.press/v75/simchowitz18a.html). Unutmalı RLS ve yeterince ayrıştırıcı girdinin ilişkisi de bilinen sistem belirleme konusudur; [Johnstone ve diğerleri, 1982](https://doi.org/10.1016/S0167-6911(82)80014-5). Bizim küçük sensör örneği onların bütün algoritma ve teoremlerinin yeniden üretimi değildir.

Rastgele girdi bir mantıksal zorunluluk değildir: her sekiz adımda bir tek darbe, diğerlerinde sıfır girdiden oluşan deterministik dizi 0–6 gecikmeleri ve sabit terim için tam-rank sekiz satır üretir. Sıfır tarih satırını diğerlerinden çıkarmak yedi birim vektörü verir. Bu cebirsel kontrol, olumlu sonucun “rastgelelik öğrenmenin temelidir” diye okunmasını engeller; gürültülü deterministik yaşam performansı ayrıca denenmedi.

Ana probleme katkı, her sonucun tek bir olaya bağlanması gerektiği varsayımının gevşetilmesidir. Çözülmeyenler: geçmiş dilinin, gecikme üst sınırının, sayısal sonuç anlamının ve uygun unutma politikasının edinilmesi; bilinmeyen nedenlerden güvenilir eylem geliştirme; diğer öğrenme işlevleriyle aynı kaynakları paylaşma. Kimliksiz ilişki öğrenimi bu soruların yerine geçmez.
