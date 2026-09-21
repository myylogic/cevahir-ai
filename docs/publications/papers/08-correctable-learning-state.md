# Düzeltilebilir öğrenme durumu: bugünkü tam aday kümesinin sınırı

**Muhammed Yasin Yılmaz**

CEV-2026-08 · Sürüm 2026.09.21.3 · Yayın tarihi 2026-09-21

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

Aynı bugünkü tutarlı aday kümesini veren iki geçmiş, aynı geçerli etiket düzeltmesinden sonra farklı ve boş olmayan kümeler gerektirir. Bütün adayların ihlal çokluklarını koruyan özet, doğru eski içerik verildiğinde tam replay ile eşleşir. 1.507.050 sınıf–düzeltme durumunda sıfır uyuşmazlık bulunmuştur; gözlem histogramı da eşleşir. Yalnız olay kimliği veya sonradan genişleyen aday dili, daha dar özetlerin yeterliliğini bozabilir.

## Abstract

Two histories with the same complete current version space can require different nonempty spaces after the same valid correction. Violation multiplicities match replay when truthful old contents are supplied, with zero mismatches over 1,507,050 projected class-correction cases. Histograms also match. ID-only edits and expanding hypothesis languages expose additional information requirements.

## Bulguların yorumu

Bugünkü bütün doğru adayları korumak, yarın geçmiş deneyimi doğru düzeltmeye yetmez; korunacak durum izinli gelecekteki işlemlere bağlıdır.

Düzeltmenin doğruluğu dış varsayımdır. Sayaçlar sabit sayıda olsa bile bit belleği geçmiş uzunluğuyla büyür. Alt sınıf projeksiyonları bağımsız deney dünyaları değildir.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/living_learning_correction_2026_09_20/REPORT_TR.md](../../research/living_learning_correction_2026_09_20/REPORT_TR.md) olarak korunur. Bağlantı yolları bu
edisyonun konumuna uyarlanmıştır. Rapordaki çalışma tarihi ve geçmiş yayın durumu
ifadeleri tarihsel kayda aittir; bu edisyonun tarihi yukarıdadır.

Bu çalışma [yayın dizisi](../README.md) içindeki ayrı bir metindir. DOI, bütün
metinleri, teknik kitabı ve kodu içeren sürüm arşivini tanımlar; her makaleye ayrı
DOI verildiği anlamına gelmez. Atıfta çalışma kimliği ve sürüm DOI'si birlikte
kullanılmalıdır. Hesaplamalı denetimlerin kapsamı [doğrulama kaydında](../evidence/validation.json)
ve [yayın ilkelerinde](../PUBLISHING.md) açıklanır.

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, hesaplamalı deney,
literatür incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır.
Bu katkı açıklaması, DOI kaydı veya otomatik testler dış bilimsel hakemlik sayılmaz.

---

## Araştırma kaydı: Düzeltilebilir öğrenme durumu: bugünkü cevap neyi korumaz?

20 Eylül 2026. Önceki [durum yordamı turunun](../../research/living_learning_state_2026_09_20/REPORT_TR.md) devamıdır. Eski raporlar, deneyler ve negatifler değiştirilmedi. Bu tur ana sistem entegrasyonu yapmaz; genel yaşarken öğrenme problemi hâlâ açıktır.

## Soru ve önceki sonuçla bağlantı

Önceki turda küçük bir geçiş grafiği uzun yeni dizileri işleyebildi; daha sonraki etiketlerle yeniden öğrenmek için ise örnek arşivi taşındı. Burada soruyu dar ama ayırıcı bir deneyle açıyoruz: **Bugünkü bütün tahminleri ve bütün tutarlı adayları koruyan durum, geçmiş bir deneyim düzeltildiğinde doğru öğrenme durumunu üretmeye yeter mi?**

Bu alt soru, ana problemin yerine geçmez. Kontrollü kalıcı öğrenmenin hangi geçmiş ayrımlarına ihtiyaç duyabileceğini gösterir. Deneyimden yeni hesaplama dili oluşturmayı, düzeltmenin güvenilirliğini keşfetmeyi veya amaç edinmeyi çözmez.

## Tam karşı örnek

Girdi kümesi `X={0,1,2}` ve hipotez sınıfı aşağıdaki üç fonksiyon olsun. Sütunlar `x=0,1,2` cevaplarıdır; kodda hipotez kimliği doğruluk tablosunun bit kodudur.

| Hipotez | 0 | 1 | 2 |
|---|---:|---:|---:|
| `h0` | 0 | 0 | 0 |
| `h3` | 1 | 1 | 0 |
| `h5` | 1 | 0 | 1 |

İki geçmiş:

`D_A = [(0,0), (1,0)]`, `D_B = [(0,0), (2,0)]`.

Her ikisinin uzunluğu aynıdır. Her ikisinde tutarlı aday kümesi yalnız `{h0}`dur: tüm girdilerdeki bugünkü tahminler aynıdır. Her ikisine **aynı kimlikteki ilk olayı, aynı eski `(0,0)` içeriğinden `(0,1)`e düzeltme** gelir. Yeni tutarlı kümeler sırasıyla `{h5}` ve `{h3}`tür. İki sonuç da boştan farklıdır.

Yalnız eski version space'i tutan deterministik bir güncelleyici, aynı durum ve aynı düzeltmeden iki ayrı doğru sonuç üretemez. Rastgele seçim de her iki geçmişte kesin doğruluğu sağlayamaz. Kaybolan bilgi, elenen adayların hangi başka kanıtlarla çeliştiğidir. Bu, yalnız bugünkü seçilmiş modelin değil, **bugünkü tam tutarlı aday kümesinin bile** düzeltme için yetersiz olabileceğini gösterir.

Genel ifade: özet `c(D)`, istenen sonuç `F(D)`, izinli işlem `u` olsun. `F(uD)=G(c(D),u)` biçiminde doğru cevap hesaplanacaksa ortak geçerli her işlem için

$$
c(D)=c(D')\Longrightarrow F(uD)=F(uD')
$$

gerekir. Özetin kendisi tekrar kullanılacaksa daha güçlü kapanış koşulu `c(uD)=c(uD')` gerekir. Bu, önceki ekleme altında yeterlilik koşulunun düzeltme işlemine uygulanmasıdır. Verimli özetin nasıl bulunacağını söyleyen yeni bir öğrenme algoritması veya özgün temel yasa değildir.

## Koşullu yapıcı çözüm ve güçlü rakip

Sabit, bilinen sonlu `H` için **elenmiş olanlar dahil her hipotezin ihlal sayısını** tutalım:

$$
c_h(D)=\sum_i \mathbf1[h(x_i)\ne y_i],\qquad
V(D)=\{h:c_h(D)=0\}.
$$

Gerçek eski içerik `(x_j,y_j)` ile yeni etiket `y'_j` verilirse:

$$
c'_h=c_h-\mathbf1[h(x_j)\ne y_j]+\mathbf1[h(x_j)\ne y'_j].
$$

Toplamın eski terimini çıkarıp yenisini eklediğimiz için tam geçmişi yeniden taramakla eşitlik bütün sonlu geçmişlerde cebirsel olarak geçerlidir. Çoklu düzeltmeler için de her adımda gerçek güncel eski içerik verilirse tümevarımla sürer. Eşitlik deney kapsamının ötesindeki uzunluklar için bu varsayımlar altında matematiksel bir sonuçtur; bilinmeyen hipotez ailelerine genellenmez.

**Daha basit güçlü rakip:** Bu küçük girdi kümesinde yalnız altı `(x,y)` kutusunun sayısını tutmak yeterlidir. `c_h = Σ_x n[x,1-h(x)]` ile herhangi bir Boolean hipotezin ihlali hesaplanır. Tam sekiz aday için altı sayaç, sekiz ihlal sayacından bile azdır ve sonradan aday eklemeyi de destekler. Böylece deney, önerilen sayaç yöntemine üstünlük atfetmez; hangi bilginin tutulduğunu karşılaştırır.

## Gerçek uygulama zinciri

Kaynak: [correction_state.py](../../../research/living_learning_correction_2026_09_20/correction_state.py).

| Sembol | Girdi → işlem → çıktı |
|---|---|
| `CorrectionSummary.append` | `(x,y)` → bütün sabit adayların ihlal sayılarını artırır → kalıcı sayaç durumu. |
| `CorrectionSummary.replace_label` | Gerçek eski `(x,y)` ve yeni etiket → çıkar/ekle → yeni sayaçlar. |
| `ObservationHistogram` | Altı kutu → etiket düzeltmesinde iki kutuyu değiştirir → aday sınıfına göre ihlal sayıları. |
| `replay_counts` | Düzeltilmiş tam geçmiş → bağımsız baştan tarama → referans ihlal vektörü. |
| `exact_counterexamples` | Üç kesin çakışma tanığı → iddiaları doğrulayan assertions. |
| `exhaustive_single_corrections` | Sonlu bütün durumların taraması → yöntemlere göre aday kümesi uyuşmazlık sayıları. |
| `repeated_corrections` | Art arda üç düzenleme → her ara adımın referansla karşılaştırması. |
| `verify_fresh_process` | Serileştirilmiş sayaçlar ve içerikli düzeltme akışı → ayrı süreçte devam → son durum eşitliği. |

Bu bir öngörü servisi veya dil modeli eğitimi değildir; öğrenme durumunun düzenleme sözleşmesini tam sayılarla sınayan deneydir. “Deneyim” burada kimlikli, doğru/yanlış etiketlenebilen küçük sembolik kayıttır.

## Deney tasarımı ve sayılar

Üç girdide sekiz Boolean fonksiyon vardır. Bunların **255 boş olmayan hipotez altkümesi**, uzunluğu 1–4 olan **1.554 geçmiş**, her geçmişte her olayın ikili etiketinin çevrilmesi tarandı. Tam evrende 5.910 düzeltme, altkümelerle **1.507.050 sınıf–düzeltme durumu** üretir. Alt sınıf sonuçları tam evrenin sayımlarından projekte edilir; bu adaylar birbirinden bağımsız değerlendirildiği için geçerlidir. 1,5 milyon bağımsız dünya veya istatistiksel örnek yoktur.

| Yöntem | Tam yeniden taramadan farklı aday kümesi |
|---|---:|
| Bütün adayların ihlal sayaçları | **0 / 1.507.050** |
| Altı kutulu gözlem histogramı | **0 / 1.507.050** |
| Yalnız daha önce hayatta kalan adaylar | **360.000 / 1.507.050** |
| İhlal var/yok bitini sayıya benzeterek çıkar/ekle | **1.338.336 / 1.507.050** |

Yalnız hayatta kalanları tutan ablation, gerçek bir etiket çevirme sonrasında her zaman boş döner: bütün eski tutarlı adaylar eski etiketi vermişti. Dolayısıyla bu 360.000 başarısızlık, **doğru yeni kümenin boş olmadığı 360.000 durumun tamamıdır**. Bu beklenen yapısal sonuçtur; zayıf rakibi yenerek performans keşfi yapılmış değildir. Hem önce hem sonra kümenin boş olmadığı 78.534 durum da bu kapsamdadır.

“İhlal var” biti tekrar sayısını kaybeder. Bir çelişkiyi geri almak başka çelişkileri ortadan kaldırmaz. Bit üzerinden naif çıkar/ekle, 1.338.336 durumda yanlış aday kabul eder. Bu bütün bit temsillerinin olanaksızlığı teoremi değil, belirtilen ablation'ın sonucudur.

6.126 üç-düzeltmeli akışta **18.378 ara durum** tam yeniden taramayla eşitti. Ayrı süreçte 80 gözlemden kaydedilen durum üzerine 50 yeni düzeltme uygulandı; sayaçlar aynıydı. Düzeltme paketleri eski içeriği dışarıdan taşıdı. Bu dış belleği deney maliyetinden veya kavramsal kapsamdan çıkarmıyoruz.

Tam sonuç: [correction_state.json](../../../research/living_learning_correction_2026_09_20/results/correction_state.json). Yeniden üretim ve koruma: [verification.json](../../../research/living_learning_correction_2026_09_20/results/verification.json).

## Negatifler ve maliyet

**Olay kimliği tek başına yeterli değil.** `[(0,0),(1,0)]` ile ters sıralı geçmiş aynı tam ihlal vektörünü ve histogramı verir. “0 numaralı olayın yeni etiketi 1” komutu farklı girdileri düzeltir; doğru yeni kümeler farklıdır. Yalnız özetle ID→eski içerik çözülemez. İçerik pakette verilmeli veya ayrı kaynak kaydı korunmalıdır. `replace_label` basit sayı sınırlarını kontrol eder; dış paketin gerçekten o olaya ait olduğunu doğrulayan provenance sistemi değildir.

**Yeni hipotezler eski özeti aşabilir.** Yalnız sabit-0 ve sabit-1 için üç sıfır etiketli geçmişin sayaçları `[0,3]`tür. Geçmiş `[(0,0)]×3` ya da `[(0,0),(1,0),(1,0)]` olabilir. Sonradan eklenen `[0,1,0]` hipotezinin ihlali sırasıyla 0 ve 2'dir. Eski sayaçlar bu ayrımı taşımaz. Altı kutulu histogram bu kapalı girdi alanında taşır; alan veya deneyim tanımı değişince aynı garanti verilmez.

**Sabit sayaç sayısı, sabit bit belleği değildir.** İhlal yöntemi `|H|` adet `0…n` sayısı taşır: `O(|H| log(n+1))` bit, her ekleme/düzeltmede `O(|H|)` aday değerlendirmesi. Hipotez tanımlarının maliyeti ayrıca vardır. Histogram `2|X|` sayısıyla güncellenir; bir aday kümesi üretmek için onu adaylarda değerlendirmek gerekir. Ham replay `O(n)` kayıt ve `O(n|H|)` değerlendirme kullanır. Bunlar soyut maliyetlerdir; Python nesne baytı veya zaman benchmark'ı yapılmadı.

**Düzeltmenin doğru olması dış varsayım.** Hatalı yeni etiketle yöntem yalnız hatalı kaydın tam yeniden taramasını taklit eder. Veri zaten tutarsızsa version space boş olabilir. Sayaçların korunması, hangi deneyimin güvenilir olduğunun keşfi değildir. Ayrıca bu çalışma veri silme sonrası modelin bütünüyle unutmasını veya yasal bir unlearning garantisini sınamaz.

## Literatürdeki yeri

Tutarlı aday kümesi, [Mitchell'in version-space çalışmalarıyla](https://www.cs.cmu.edu/afs/cs/usr/mitchell/ftp/publications.html) aynı öğrenme çerçevesindedir; burada amaç aday üretme yöntemi önermek değil mevcut kümenin düzeltme açısından yeterliliğini sınamaktır.

[Doyle'un Truth Maintenance System çalışması](https://www.sciencedirect.com/science/article/pii/0004370279900080), inançların nedenlerini koruyarak geri çekme ve revizyonu ele alır. Bu tarihsel ilişki doğrudandır; bizim sayaçlarımız tam bir gerekçelendirme sistemi uygulamaz.

[Gupta, Mumick ve Subrahmanian'ın incremental view maintenance çalışması](https://doi.org/10.1145/170036.170066), ekleme/silme/güncelleme altında türetilmiş bilgiyi ve destek çokluklarını korur. Buradaki toplamsal çıkar/ekle ilişkisi bu bilinen hesap ailesine yakındır. Bu eşleşme bir yorumdur; deney o makalenin algoritmalarının yeniden uygulaması değildir. Birincil kaynak sayfaları 20 Eylül 2026'da kontrol edildi. Yeni temel computational principle bulunduğu iddia edilmiyor.

## Genel problem için kalan sonuç

Kalıcı öğrenme durumunu yalnız “şimdi neyi biliyor?” ile değerlendirmek eksiktir. “Hangi yeni deneyim ve hangi düzeltmeler altında yeniden öğrenebilir?” sorusu da gerekir. **Yeterli durum, gelecekte izin verilen işlemlere göre yeterlidir.** Sabit ailede içerikli düzeltme için toplamsal özet yapıcı çözüm verir; yeni temsil/hipotez dili ve kaybolmuş kaynak bağları bu çözümün dışına çıkar.

Bir sonraki açık yön, bilinmeyen gelecekteki temsil genişlemeleri ve düzeltmeler arasında sınırlı belleğin hangi bilgiyi koruyacağıdır. Hazır üç-girdi evrenini büyütmek tek başına bu soruyu çözmez. Yeni araştırma, maliyeti eşit güçlü özetlerle ve hipotez dilinin değiştiği negatiflerle ilerlemelidir.

Yeniden çalıştırma (repository kökünden):

```powershell
python research/living_learning_correction_2026_09_20/correction_state.py
python research/living_learning_correction_2026_09_20/verify_correction.py
```

Bu komutlar yalnız bu turun sonuçlarını yazar; önceki araştırma dosyalarını değiştirmez.
