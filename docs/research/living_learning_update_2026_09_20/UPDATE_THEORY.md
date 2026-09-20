# Cevahir — Deneyimin sonraki öğrenmeyi değiştirmesi

20 Eylül 2026 · Yeni kuramsal inceleme · Önceki kayıtların yerine geçmez

## Kapsam ve ana probleme bağlantı

Önceki turda hangi geçmişin ne hızla yenileneceği araştırmacının verdiği bir kuralla belirleniyordu. Bu turda incelenen yeni parça, geçmiş deneyimin bu güncelleme davranışını değiştirebilmesidir. Bu, **yaşarken öğrenmenin tamamı değildir**. Yeni temsil edinimi, hesaplama derleme, gecikmiş sonuç ilişkileri, geçerli eski beceriler ve kaynak yönetimi önceki araştırma haritasında ayrı sorular olarak kalır.

Bir öğrenicinin durumu iki işlevsel parçayla gösterilsin:

\[
s_t=(w_t,m_t),\qquad \hat y_t=f_{w_t}(x_t),\qquad
w_{t+1}=U(w_t,e_t;m_t).
\]

Burada `w` mevcut tahmini, `m` aynı yeni deneyimin `w`yi nasıl değiştireceğini etkiler. Bu ayrım depolamanın fiziksel türüne ilişkin değildir. Her ikisi de sayılar, ağırlıklar veya yürütülebilir kod olarak saklanabilir. İstenirse bütün süreç tek bir büyük tahmin fonksiyonu olarak yeniden kodlanabilir. İncelenebilir ayrım **nedensel erişim yoludur**: edinilmiş bilgi doğrudan mevcut cevabı mı değiştiriyor, yoksa yeni kanıt geldikten sonraki öğrenme yolunu mu değiştiriyor?

Bu ikinci değişim için bütün güncelleme algoritmasının yeniden yazılması gerekmez. Deneyimden edinilmiş birkaç katsayı bile aynı kanıtın hangi yönlerde ve ne kadar etkili olacağını değiştiriyorsa, ilerideki öğrenme hesabı değişmiştir. Bu bulgu tek başına yeni hipotez dili veya daha geniş sınırsız ifade gücü anlamına gelmez.

## Nedensel taşıma deneyi neyi ayırabilir?

Geçmiş yaşam `H` sonunda edinilen `m_H`yi saklayıp, yeni ve bağımsız bir problemde yalnız bu durumu taşımak güçlü bir kontroldür. Eski `w`, örnekler, momentum, artıklar, gradyan-duyarlılık izleri ve deneyim kimlikleri aktarılmaz. Karşılaştırılan sistemler aynı `w_0`, aynı yeni veri sırası, aynı yeni etiketler ve aynı hizmet bütçesiyle başlar. Taşınmış `m_H` yeni sınama boyunca sabitlenirse, etkisi yeni problemde yeniden edinilen bilgiyle karışmaz.

Doğrusal özel durumda:

\[
f_w(x)=x^\top w,\qquad
U(w,(x,y);m)=w+P_m x(y-x^\top w).
\]

Yeni etiket gelmeden iki sistemin tahminleri eşittir. Etiket geldikten sonra ise

\[
\frac{\partial w_{t+1}}{\partial y_t}=P_m x_t
\]

olduğundan, gelecekteki deneyime tepkileri farklı olabilir. Başlangıç tahminleri eşitken, yalnız `m`yi taşıma/silme yeni veriyle kazanılan başarıyı değiştiriyorsa **öğrenme biçiminin deneyimle edinilmiş bir bileşeni** için nedensel kanıt elde edilir. Bir katsayıyı doğrudan hatırlamanın katkısı bu düzende dışlanır.

Sınırlar:

- `m_H` yalnız hazır birkaç rejim arasından seçim yapıyorsa sonuç o seçim kapasitesini gösterir. Sürekli bir güncelleme matrisi öğreniliyorsa daha geniş bir güncelleme ailesi kullanılır; yine aile araştırmacı tarafından verilmiştir.
- Yeni katsayıların farklı olması aktarım için gerekli ama tek başına yeterince güçlü değildir. Yeni problemler geçmişle aynı değişim yönünü, ölçeği, gürültüyü ve gözlem geometrisini paylaşabilir. Bunlar hangi genellemenin sınandığını açıklar.
- Eski parametreler sıfırlanmadığında, daha iyi başlangıç tahmini ile daha iyi güncelleme yöntemi ayrışmaz. Sadece son tahmini karşılaştırmak da hız, gürültüye duyarlılık ve koruma etkilerini ayırmaz.
- Yeni sınama verisiyle en iyi sabit kuralı sonradan seçmek uygulanabilir çevrimiçi rakip değildir; ayrıcalıklı referanstır. Buna karşı iyi veya kötü olmak, gerçek zamanlı üstünlükle aynı iddia değildir.
- Geçmiş yaşamların edinim maliyeti raporlanmalıdır. Eşit yeni etiket sayısı, toplam eşit deneyim veya toplam eşit hesap anlamına gelmez.

Bu kontroller sınırlı bir görev ailesinde, belirli bütçede ileri öğrenme kapasitesinin değiştiğini gösterebilir. Bütün yaşarken öğrenme ilkesini veya biyolojik bir zorunluluğu kanıtlamaz.

## Güncelleme parametresinin deneyimden öğrenilmesi

Aşağıdaki türetim standart zincir kuralının bu doğrusal düzene uygulanmasıdır; yeni bir matematik ilkesi olarak sunulmuyor. `m`yi geçici olarak sabit kabul edelim. Her bileşeni için

\[
J_t^{(j)}=\frac{\partial w_t}{\partial m_j},\qquad
e_t=y_t-x_t^\top w_t
\]

tanımlansın. Verinin `m`den bağımsız olduğu açık çevrim akışında tam yineleme:

\[
\boxed{J_{t+1}^{(j)}=(I-P_mx_tx_t^\top)J_t^{(j)}
 +\frac{\partial P_m}{\partial m_j}x_te_t.}
\]

Önce tahmin, sonra etiket protokolünde `\ell_t=\tfrac12e_t^2` için:

\[
\frac{\partial\ell_t}{\partial m_j}=-e_tx_t^\top J_t^{(j)}.
\]

Dolayısıyla gözlenmiş sonraki hata, önceki güncelleme tercihinin hangi yönde değiştirilmesinin yararlı olacağına ilişkin bir sinyal üretebilir. Bu, etiketin gelecekteki gerçek kaybı doğrudan söylediği anlamına gelmez: sinyal gürültülü, seçilmiş temsil ve zaman ufkuna bağlıdır.

**Tamlık koşulu önemli:** Bu türev, aynı `m` geçmiş boyunca sabit tutulan hesap yolunun türevidir. `m` her adımda değiştirilirken bu izi olduğu gibi taşımak, değişen bütün meta-güncelleme geçmişini tam türevlemek değildir. Kısa açılım, iz sönümü veya çapraz etkilerin ihmali kullanılıyorsa yaklaşım açıkça adlandırılmalıdır. Ayrıca eylemler gözlemleri değiştiriyorsa yalnız bu yineleme, veri dağılımının `m`ye bağımlı etkisini kapsamaz. Xu ve diğerlerinin çevrimiçi meta-gradyan türetimi sabit meta-parametre varsayımını, değişen parametrede iz yaklaşımını ve bir sonraki örnekte değerlendirmeyi açıkça tartışır. [Xu, van Hasselt ve Silver, 2018, §1](https://proceedings.neurips.cc/paper_files/paper/2018/file/2715518c875999308842e3455eda2fe3-Paper.pdf).

Koordinat başına `P=diag(exp(\beta_i))` seçilmesi, öğrenilebilir oranların temel örneğidir. Daha genel `P`, bir girdiden gelen düzeltmenin başka katsayılara taşınmasını da belirleyebilir. Bunun hangi yönlerde yararlı olduğu dünyanın değişim yapısına, gözlemlerin ayırt ediciliğine ve gürültüye bağlıdır. Verinin ayırt etmediği yeni bir ilişkinin yalnız `P`yi değiştirerek elde edildiği ileri sürülemez.

## Hız ve gürültü arasındaki kesin küçük tanık

Öğrenilmiş güncelleme tercihi neden hem yararlı hem zararlı olabilir? Bir boyutta, gizli hedefin ve gözlemin

\[
\theta_{t+1}=\theta_t+\nu_{t+1},\quad
y_t=\theta_t+\epsilon_t,\quad
w_{t+1}=w_t+\alpha(y_t-w_t)
\]

olduğunu varsayalım. `\nu` ve `\epsilon` sıfır ortalamalı, zamansal olarak bağımsız; birbirinden ve mevcut kestirim hatasından bağımsız olsun. Varyansları sırasıyla `q` ve `r`dir. Öğrenicide hedef değişimi bildirimi yoktur. `d_t=\theta_t-w_t` için

\[
d_{t+1}=(1-\alpha)d_t+\nu_{t+1}-\alpha\epsilon_t.
\]

Böylece temiz hedefe karşı, etiket alınmadan önceki hata ikinci momenti

\[
S_{t+1}=(1-\alpha)^2S_t+q+\alpha^2r
\]

eşitliğini sağlar. `0<\alpha<2` için kalıcı hata:

\[
\boxed{S_\infty(\alpha)=\frac{q+\alpha^2r}{2\alpha-\alpha^2}.}
\]

Bu formül yalnız varsayılan skaler takip düzeni için burada doğrudan türetildi. Çok boyutlu deneyin kapalı çözümü olduğu iddia edilmiyor. Gözlenmiş yeni etiketin tahmin hatası istenirse buna bağımsız gözlem gürültüsü `r` eklenir.

`q,r>0` olduğunda optimum pozitif oran

\[
r\alpha^2+q\alpha-q=0,\qquad
\alpha_* =\frac{2q}{q+\sqrt{q^2+4qr}}
\]

eşitliğini sağlar. Büyük gerçek değişim, daha hızlı yenilemeyi yararlı kılabilir; yüksek etiket gürültüsü aynı hızın maliyetini artırır. Dolayısıyla geçmiş yaşamdan “bu yöndeki yeni kanıta hızlı tepki ver” tercihinin edinilmesi matematiksel olarak anlamlıdır. Dünya yönleri yer değiştirirse aynı tercih geçerli eski bilgiyi daha çok gürültüyle bozabilir.

**Sonlu yaşam ayrımı:** `a=(1-\alpha)^2` ile

\[
S_t=a^tS_0+S_\infty(1-a^t).
\]

İlk `T` adım ortalaması

\[
\overline S_T=S_\infty+(S_0-S_\infty)
\frac{1-a^T}{T(1-a)}.
\]

Bu nedenle en iyi sonlu-ufuk oranı başlangıç bilgisinin kalitesine ve değerlendirme süresine de bağlıdır. `q=0` için sonsuz zaman formülünün `\alpha\downarrow0` limitinde sıfıra gitmesi, baştan `\alpha=0` seçmenin yeni bir beceriyi öğreteceği anlamına gelmez. Sıfır oran başlangıç hatasını korur. Sonsuz zaman ile sıfır öğrenme oranı limitlerini yer değiştirmek yanlış sonuca götürür.

Bu tanık, ana problemin olanaksızlığını göstermiyor. Kontrolün neden “her şeyi koru” veya “her zaman hızlı değiş” kuralına indirgenemediğini, açık bir hata bedeli üzerinden açıklıyor. Kullanıcının talep etmediği bütün dünyalarda sıfır zarar şartı eklenmiyor.

## Yerel düzeltmenin sınırlı güvencesi

Bir örneğe uygulanan güncellemeden sonra aynı örneğin artığı

\[
e_t^+=(1-x_t^\top P_mx_t)e_t
\]

olur. `0\le x_t^\top P_mx_t\le2` ise bu **aynı örnekteki** kare hata artmaz. Sınır, `P` ve gözlem normları kontrol edilerek sağlanabilir. Fakat bu, daha önceki geçerli becerilerin veya temiz dünya riskinin korunması değildir.

Tam karşı örnek: gerçek hedef sıfır, `w=0`, `x=1`, tek yanlış etiket `y=1`, `P=1` olsun. Güncelleme aynı örneğin hatasını `1→0` düşürür; gerçek hedefe karşı hatayı `0→1` yükseltir. Yerel cebirsel kararlılık ile geri bildirimin geçerliliği ayrı sorunlardır. Öğrenilmiş oranları sınırlandırmak denetimli bir mekanizma parçasıdır; yanlış bilginin güvenilir ayırt edilmesini tek başına çözmez.

## Birincil literatür eşleşmeleri ve açık farklar

| Kaynak | Doğrudan eşleşen fikir | Bu araştırma için sınırı |
|---|---|---|
| [Sutton, 1992, IDBD](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf) | Akıştaki doğrusal öğrenicinin girdi başına oranlarını önceki öğrenme deneyiminden uyarlama; değişen görevlerde uygun öğrenme yanlılığını edinme. | Diğer ağırlıklar üzerindeki meta-parametre etkileri yaklaşık alınır. Basit doğrusal ailede sonuç, genel dil/amaç edinimi değildir. “Öğrenme oranı da öğrenilebilir” yeni buluş sayılamaz. |
| [Andrychowicz ve diğerleri, 2016](https://papers.nips.cc/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf) | Güncelleme yordamını parametreli bir öğrenme problemi yapmak; öğrenilmiş optimizasyonu benzer yapılı yeni problemlere taşımak. | Bildirilen aktarım görev ailesi ve eğitim ufkuyla sınırlıdır; yeni bir yordamın her gelecekte iyi kalacağı gösterilmez. Bu tur çok küçük ve incelenebilir bir özel durum kullanır. |
| [Xu, van Hasselt ve Silver, 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/2715518c875999308842e3455eda2fe3-Paper.pdf) | Etkileşim sürerken güncellemenin meta-parametrelerini, sonraki örnekteki ölçüte göre değiştirmek. | Bir üst değerlendirme ölçütü ve meta-güncelleme yöntemi verilidir. Seçilmiş iç hedefin uyarlanması, sistemin bütün değerlendirme anlamını kökensiz üretmesi değildir. |
| [Bruce, Goel ve Bernstein, 2020](https://dsbaero.engin.umich.edu/wp-content/uploads/sites/441/2021/03/BruceRLSVRF.pdf) | Değişken oranlı unutmalı RLS; yakınsama ve gürültü altında tutarlılığın açık koşulları. | Sonuçlar uyarım ve gürültü koşullarına bağlıdır; keyfî unutma seçimlerinin koşulsuz doğru olduğu söylenmez. Her sabit unutma oranı gürültü altında tutarlı kestirim sağlamaz. |
| [Bruce, Goel ve Bernstein, 2020, matrix forgetting](https://dsbaero.engin.umich.edu/wp-content/uploads/sites/441/2021/03/RLSMatrixForgetting.pdf) | Unutmanın hem zaman oranı hem parametre yönleri üzerinden düzenlenmesi; değişim ve yetersiz uyarımı aynı çerçevede ele alma. | Unutmanın yönlü olması literatürde vardır. Bu turun öğrenilmiş `P` matrisiyle özdeş algoritma değildir: biri geçmişin ağırlıklandırılmasını, diğeri doğrudan güncellemenin dönüşümünü parametreler. |

Literatürde “unutma” çoğu kez eski gözlemin güncel kestirimdeki ağırlığını azaltmaktır. Ham geçmişin geri döndürülemez biçimde silinmesiyle özdeş değildir. Sonradan düzeltme için saklama ile güncel davranışta ne kadar kullanma ayrı kararlardır; önceki yeterlilik karşı örnekleri bu ayrımı korur.

Bu inceleme kaynaklardaki büyük deneyleri yeniden üretmedi. Birincil yayınlar kavramsal eşleşmeleri ve hangi iddiaların zaten bilindiğini belirlemek için okundu. Yeni deneyin başarısı, bu yayınların geniş deneysel iddialarının doğrulanması olarak sunulmamalıdır.

## Araştırma kararı

Sınanabilir yeni hipotez şudur: **Geçmiş deneyim, doğrudan eski cevapları aktarmadan, aynı yeni kanıtın kalıcı iç durumu hangi biçimde değiştireceğini öğrenebilir; bu değişim belirli yeni problemlerde daha iyi edinim sağlayabilir.** Tersi geometri, gürültü ve ufuk karşı örnekleriyle birlikte sınanmalıdır.

Olumlu sonuç, önceki beceri edinimi/koruma/derleme sonuçlarına yeni bir işlev ekler; onların yerine geçmez. Negatif sonuç, seçilen küçük meta-yordamın sınırını gösterir; öğrenme yordamının deneyimle değişmesinin olanaksızlığını göstermez. Bu iki yöndeki kapsam sınırını korumak, ana yaşarken öğrenme problemini yeniden tek alt probleme daraltmamak için gereklidir.
