# Bugünkü beceriyi korumak, sonraki öğrenmeyi korumaya yetiyor mu?

20 Eylül 2026. Yeni araştırma kolu; önceki deneylerin yerine geçmez. [Kapsam ve genel önermeler](SCOPE_AND_FOUNDATIONS.md). [Çalıştırılan deney](../../../research/living_learning_reassessment/state_transport.py), [bütün sonuçlar](../../../research/living_learning_reassessment/results/state_transport.json).

**Sonuç:** Aynı bugünkü cevapları veren iki kalıcı durum, aynı sonraki deneyimden farklı biçimde öğrenebilir. Temsil değiştiğinde yalnız cevap üreten ağırlıkları taşımak yeterli olmayabilir; güncellemenin kullandığı durum da uyarlanmalıdır. Ayrıca eski görevde tam yeterli olan bir özet, sonradan oluşturulan bir özellik için gerekli geçmiş bilgisini içermeyebilir. Bunlar iki farklı engeldir: uyumsuz okuma/güncelleme ve gerçekten kaybedilmiş bilgi.

Bu sonuç bir evrensel öğrenme algoritması değildir. Deneyde dönüşüm ve doğru matematik araştırmacı tarafından verilmiştir. Sistemin kendisinin yararlı dönüşümü bulduğu veya her dönüşümü güvenilir yönettiği gösterilmedi.

## 1. Hipotez ve matematik

Sınanan iddia: “Bir yeniden düzenleme bütün mevcut tahminleri aynı bırakıyorsa öğrenme durumu da korunmuştur.” Bu iddia yanlıştır.

Sütun vektörlü özellik `x`, ağırlık `w`, tahmin `wᵀx` olsun. Yeni kodlama `z=Ax`, `A` tersinir. `v=A⁻ᵀw` ile bütün girdiler için `vᵀz=wᵀx`; bu, yalnız mevcut fonksiyon eşitliğidir.

`P`, gradyanı güncelleme yönüne çeviren matristir (pozitif tanımlı olduğunda ters metrik/preconditioner). Kare kayıpta eski çevrimiçi güncelleme:

\[
w^+=w-\eta P(w^Tx-y)x.
\]

Yeni koordinatta aynı gelecek hesaplama yolunu elde etmek için:

\[
P_z=A^{-T}PA^{-1},\qquad
v^+=v-\eta P_z(v^Tz-y)z=A^{-T}w^+.
\]

**İspat:** `vᵀz=wᵀx` ve `P_z z=A⁻ᵀPx` eşitliklerini yeni güncellemede yerine koymak yeterlidir. Her adım için indüksiyonla yol eşitliği sürer. Hedefin sabit veya değişken olması gerekmez; eşleştirilmiş aynı `(x,y)` akışı yeterlidir. Sayısal yuvarlama pratik uygulamada küçük fark bırakır.

Yalnız ağırlıkları taşıyıp yeni kodlamada `P_z=I` kullanılırsa, eski koordinattaki etkin güncelleme `w⁺=w−ηAᵀA(wᵀx−y)x` olur. Öğrenme dinamiği değişmiştir. Bu değişim bazı problemlerde faydalı da olabilir. Dönüşümün sırf fonksiyonu koruduğu için öğrenmeyi de aynı bırakacağını varsayamayız.

Buradaki matris hesabı doğal gradyan/koordinat geometrisiyle ilişkilidir; yeni optimize edici olarak sunulmuyor. Fisher matrisi kestirmiyoruz ve doğal gradyanın genel bir uygulamasını sınamıyoruz. [Amari, 1998](https://doi.org/10.1162/089976698300017746). Mevcut fonksiyonu koruyarak ağ yapısını değiştirme de bilinen bir fikirdir; [Net2Net](https://arxiv.org/abs/1511.05641) böyle dönüşümler kullanır. Bizim karşı örneğimiz Net2Net'in vaatlerini çürütmez; mevcut fonksiyon eşitliği ile sonraki güncellemelerin eşitliğini ayırır.

## 2. Sayısal protokol

Her rejimde 32 eşleştirilmiş başlangıç: `x=(1,u,v)`, `u,v∈[-1,1]`. İlk 96 canlı gözlemle üç ağırlık edinilir. Sonra kodlama araştırmacının seçtiği

\[
A=\begin{pmatrix}1&0&0\\0&3&1\\0&0&1/4\end{pmatrix}
\]

ile değiştirilir. 192 yeni gözlem boyunca üç sürüm izlenir: eski koordinat; yalnız ağırlıkları doğru taşınmış yeni koordinat; ağırlık ve güncelleme metriği birlikte taşınmış yeni koordinat. Öğrenme oranı `.04`, etiket gürültüsü standart sapması `.02`; her başlangıçta 256 ayrı sürekli girdiyle değerlendirme. Değerlendirme cevapları öğreniciye verilmez. İki rejim, hedefin aynı kaldığı ve değiştiği dünyalardır. Dönüşüm doğrulama skoruna göre seçilmedi; parametre taraması yapılmadı.

Her sürümde üç öğrenilen ağırlık var. Tam taşıma ilave dokuz matris elemanı saklar ve yoğun matris-vektör işlemi yapar; kimlik matrisli sürümün bunu saklaması gerekmez. Bu bir eşit işlem bütçesinde üstünlük deneyi değildir. Amaç cebirsel eşitlik ve karşı örneğin fiilî yürütülmesini sınamaktır.

| Ölçüm | Sabit hedef | Değişmiş hedef |
|---|---:|---:|
| Taşıma anında en büyük tahmin farkı, bütün başlangıçlarda üst sınır | `4.44×10⁻¹⁶` | `4.44×10⁻¹⁶` |
| Yalnız ağırlık taşımasında yaşam boyunca en büyük tahmin farkının başlangıç ortalaması | `0.440868` | `2.568257` |
| Ağırlık+metrik taşımasında bütün yaşamlar boyunca en büyük fark | `2.22×10⁻¹⁵` | `1.11×10⁻¹⁵` |
| Son temiz hedef MSE, eski koordinat | `0.000351` | `0.010339` |
| Son temiz hedef MSE, yalnız ağırlık taşınmış | `0.027625` | `1.110122` |

Son MSE farkı için eşleştirilmiş başlangıç ortalamasının yaklaşık normal %95 aralığı sabit hedefte `[.023536,.031012]`, değişmiş hedefte `[1.051264,1.148302]`. Bunlar yalnız seçilen akışlardaki rastgele örnekleme değişimini kapsar. Evrensel üstünlük kanıtı cebirsel eşitlikten de, bu aralıklardan da çıkmaz. Ters yönde daha uygun bir yeniden ölçekleme sıradan güncellemeyi hızlandırabilir.

Her yaşam sonunda yalnız ağırlık, özellik dönüşümü ve metrik JSON'a çevrilip geri okunur; ham deneyim olmadan sonraki güncelleme eşitliği sınanır. Ayrıca bir başlangıcın bu durumu yeni bir Python işlemine verilir: tahmin ve bir sonraki güncelleme tam aynı JSON sayılarıyla yeniden üretilir. Bu bir yeniden başlatma tanığıdır; uzun dönem dosya arızası veya bütün algoritmaların durum taşımasını sınamaz.

## 3. Tam bir eski özet, yeni temsil için yetersiz olabilir

Önceki deneyde bilgi duruyordu; doğru dönüşüm onu kullanabildi. Şimdi bilgi kaybını ayırıyoruz.

Dört girdi `(x₁,x₂)∈{−1,+1}²` için iki geçmiş kur:

- `H₊`: her doğru çıktı `y=x₁x₂`.
- `H₋`: her doğru çıktı `y=−x₁x₂`.

Eski modelin özellikleri `φ=(1,x₁,x₂)`. Doğrusal en küçük kareler için bütün geçmişin etkisini tam tutan `G=Σφφᵀ`, `b=Σφy`, `Σy²` ve örnek sayısı iki geçmişte de aynıdır:

\[
G=4I_3,\quad b=(0,0,0),\quad \sum y^2=4,\quad n=4.
\]

Bu özet eski modelin optimumunu ve kare kaybını, hatta aynı özelliklerde sonraki eklemeleri tam hesaplamaya yeter. Eski optimum iki dünyada da `w=0`, MSE `1`.

Şimdi `φ'=(1,x₁,x₂,x₁x₂)` oluşturulsun. İki yeni Gram matrisi de `4I₄` fakat:

\[
b'_+=(0,0,0,4),\qquad b'_-=(0,0,0,-4).
\]

Doğru yeni ağırlıklar sırasıyla `(0,0,0,+1)` ve `(0,0,0,−1)`; MSE sıfır. **Aynı eski özetten iki farklı doğru yeni özet üreten bir dönüşüm yoktur.** Bu, optimizasyonun az çalıştırılması değildir. Ham dört kayıt veya eksik `Σx₁x₂y` çapraz momenti saklanmışsa yeni çözüm bulunur; bedeli ek saklama ve işlemdir. Gelecek yeni etiketler de ayrımı yeniden öğretebilir. Bu örnek gelecekte öğrenmeyi ebediyen imkânsız kılmaz; geçmişten tam geri kazanımı imkânsız kılar.

Daha genel alt sınır: ek bilgi almadan iki dünyada aynı tahmini `v(x)` veren herhangi bir yöntem için her noktada iki dünya kare kayıp ortalaması `[(v−q)²+(v+q)²]/2=v²+1≥1`, `q=x₁x₂`. Dolayısıyla en az bir dünyada MSE en az `1`; rastgele tahmin de beklentide bu sınırı kıramaz. Bu iddia eski özet dışında ayırt edici erişim olmayan bu iki dünya içindir.

Sınanan güçlü iddia “sabit model için tam yeterli istatistik, daha sonra edinilen bütün temsillere taşınmaya yeter” böylece yanlışlandı. “Ham veriyi hep tutmak zorunludur” sonucu çıkmaz: uygun ek istatistik bu belirli eksikliği daha küçük kayıtla kapatır. Açık sorun, gelecekte hangi ayrımların yararlı olacağı bilinmezken neyin tutulacağıdır.

## 4. Ana probleme katkı ve açık kalan

Mevcut cevap, eklemeli öğrenme, eski deneyimin düzeltilmesi ve yeni temsil oluşturma farklı yeterlilik sözleşmeleri olabilir. Bu ayrım, yalnız yeniden sınama ailesine bağlı değildir. Fakat bütün öğrenicilerin bir metrik matrisi, geçmiş arşivi veya büyüyen özellik listesi tutması gerektiğini de göstermez.

Yeni araştırma yükümlülüğü: aynı yaşama ait durumu yalnız son test skoruyla değil, daha sonra edinim, düzeltme ve yeniden düzenleme sırasında koruduğu hesaplama imkânlarıyla değerlendirmek. Sonraki öğrenme kapasitesinin korunması literatürde de ayrı bir araştırma konusudur; derin ağlarda uzun öğrenmeyle plastisite kaybı gösteren çalışma vardır. [Dohare ve diğerleri, 2024](https://www.nature.com/articles/s41586-024-07711-7). O çalışmanın kullandığı rastgele birim yenilemesini bütün olası öğreniciler için zorunlu kabul etmiyoruz; bu doğrusal deneyde o mekanizma yoktur.

Çalıştırma: `python research/living_learning_reassessment/state_transport.py`. Tekrar üretimde ayrı çıktı için `--output <dosya>` kullanılabilir. Dış paket, GPU, ana sistem veya ağ gerektirmez.
