# Tur 14 — Bağlamın öğrenilmesi: bağımsız matematik ve literatür denetimi

Tarih: 2026-09-22. Bu not performans sonuçları görülmeden, tur 14'ün deney tasarımını denetlemek için yazıldı. Sayısal sonuç raporu değildir. Önermeler aşağıdaki sonlu model ailesine koşulludur; beynin, bilincin veya genel yapay zekânın açıklaması oldukları ileri sürülmez.

## 1. Tam olarak ne öğreniliyor?

Girdi `h ∈ {0,1}⁴`, `s ∈ {0,1}`, `x ∈ {0,1}⁴` biçimindedir. Geçmiş kapıları, sıfır olmayan 15 maskenin pariteleridir:

\[
g_a(h)=a^\top h\pmod 2,\qquad a\in\mathbb F_2^4\setminus\{0\}.
\]

On altıncı kapı yüzey ipucu `s`'dir. Her uzman, 32 olası afin pariteden biridir:

\[
f_{u,b}(x)=u^\top x\oplus b.
\]

Bir aday yapı `θ=(g,f₀,f₁)`, `Fθ(h,s,x)=f_{g(h,s)}(x)` işlevini üretir. Gerçek ortamda `g`, iki veya üç geçmiş bitinin paritesidir. Gerçek uzmanlar farklıdır ve birbirinin tümleyeni değildir. Bunun sonucu `d(x)=f₀(x)⊕f₁(x)` sıfır olmayan doğrusal kısmı bulunan afin paritedir; tekdüze `x` altında `P(d=1)=1/2`.

Burada geçmişin dört bitlik temsili, iki uzmanlı ayrışma, parite adayları ve uzman ailesi tasarımcı tarafından verilir. Öğrenilen şey, bu önceden tanımlı aile içindeki olasılık ağırlıklarıdır. Geçmişten ilgili değişkenleri çıkarmanın, sınırsız kavram yaratmanın veya bağlam kavramının kendisinin keşfedildiği söylenemez.

## 2. Gösterim sayısı, işlev sayısı ve etiket simetrisi

`16 × 32² = 16.384` aday gösterim vardır. Ancak bunlar `15.904` farklı tam-girdi işlevi üretir:

\[
16(32^2-32)+32=15904.
\]

Gerekçe: `f₀=f₁` olan 32 işlev her kapıda aynıdır. Uzmanlar farklıysa iki farklı kapı aynı tam-girdi işlevini veremez. İki farklı geçmiş paritesi bağımsız iki doğrusal formdur ve dört kapı-değeri çifti de gerçekleşebilir. Bu çiftlerde işlevlerin eşitliği, iki uzmanı eşit olmaya zorlar. Yüzey kapısı ile geçmiş kapısı için de tam küpte aynı dört çift bulunur.

Dolayısıyla her kapıya ve o kapıdaki her uzman çiftine eşit önsel vermek, farklı işlevlere eşit önsel vermek değildir. Kapıdan bağımsız bir afin işlev, tek bir bağlama bağlı işlevden 16 kat önsel kütle alır. Bu tercih geçerlidir; fakat açıkça bildirilmelidir. Aileleri eşit ağırlıklandıran ikinci önselde de aynı işlevin birden fazla gösterimi bulunur; oranlar ayrıca o önsele bağlıdır.

Bütün geçmiş kapıları homojendir: `g(0)=0`. Bu, uzman etiketlerinin yönünü zaten sabitler. Ek olarak `f₀<f₁` gibi bir sıralama şartı koymak, tümleyen kapılar adaylarda yokken bazı geçerli işlevleri dışarı atar. Genel `g ↔ 1−g`, `f₀ ↔ f₁` simetrisi ancak tümleyen kapı da temsil edilebiliyorsa geçerlidir. Uzmanların eşit olması durumunda kapı kimliği işlevden belirlenemez.

**Bağımsız sonlu denetim:** Tam 512 girdilik küpte bütün 16.384 gösterim ayrı bir kısa numaralandırma ile üretildi; farklı işlev sayısı 15.904 bulundu. Bu, performans deneyi değildir.

## 3. Tam karışmış eğitim verisi iki açıklamayı ayıramaz

İki olası dünya düşünelim:

\[
H(h,s,x)=f_{g(h)}(x),\qquad S(h,s,x)=f_s(x).
\]

Eğitimde her zaman `s=g(h)` ise, `H` ile `S` bütün eğitim örneklerine aynı yanıtı verir. Bu, sonlu veri azlığına özgü değildir: hizalı altkümenin 256 girdisinin tamamı bilinse de iki açıklama ayırt edilemez.

Üstelik bu altkümenin tamamındaki hatasız etiketler, yukarıdaki 16.384 gösterim içinde tam olarak bu iki gösterime uyar. Yanlış bir geçmiş paritesi, gerçek pariteyle dört kapı-değeri çiftinin tamamını alır; eşit olmayan uzmanları her çiftte açıklayamaz. Doğru geçmiş kapısında ve yüzey kapısında ise iki uzman, bütün `x` değerleriyle belirlenir. Bu ifade, gerçek uzmanların farklı olması varsayımını kullanır.

**Önemli ayrım:** Gerçek kapının yalnız geçmişten gelebileceği baştan kabul edilirse, yüzey açıklaması model ailesinden çıkarılmış olur. O zaman geçmiş kapısı belirlenebilir. Bu başarı, verinin yanında doğru bir önvarsayıma da dayanır. Veri tek başına yüzey açıklamasının yanlışlığını göstermez.

Yumuşak olabilirlik kullanan asıl deneyde hiçbir adayın olasılığı tam sıfır olmaz. Bu nedenle “iki aday kaldı” ifadesi asıl deneyin sonlu posterioru için kullanılamaz; yukarıdaki ifade hatasız uyum kümesi hakkındadır.

## 4. Karışmanın tam MSE sınırı

Testte `h` ve `x` tekdüze olsun. Yüzey ipucu `s`, geçmişten bağımsız ve tekdüze ise:

\[
\Pr[H\ne S]=\Pr[s\ne g(h)]\Pr[d(x)=1]=\tfrac14.
\]

Testte her zaman `s=1−g(h)` ise aynı anlaşmazlık oranı `1/2` olur. Herhangi bir `p∈[0,1]` tahmini için, iki dünya farklı yanıt veriyorsa:

\[
\frac{(p-H)^2+(p-S)^2}{2}
=\left(p-\frac12\right)^2+\frac14\ge\frac14.
\]

Dolayısıyla, aynı eğitim geçmişini üreten bu iki dünya üzerindeki eşit ağırlıklı ortalama ikili MSE en az:

| Test girdisi dağılımı | İki dünyanın anlaşmazlığı | Ortalama MSE alt sınırı |
|---|---:|---:|
| `s` bağımsız ve tekdüze | `1/4` | `1/16` |
| `s` sürekli ters | `1/2` | `1/8` |

Bu, **tek bilinen gerçek dünya için zorunlu hata** değildir. Daima geçmiş açıklamasını seçen sistem `H` dünyasında sıfır hataya ulaşabilir; aynı veriyle ayırt edilemeyen `S` dünyasında bunun bedelini öder. Kanıt, her iki dünyada birden veriden kaynaklanan kusursuz garanti verilemeyeceğini söyler. İki koordinatın hatalarını toplayan Brier ölçüsü kullanılırsa bu sayılar ikiyle çarpılır; burada ölçü `(p−y)²`'dir.

Bu oranlar, `g(h)=h₀⊕h₁`, `f₀(x)=x₀`, `f₁(x)=x₁⊕1` tanığı üzerinde tam küp numaralandırmasıyla da doğrulandı. İki dünyanın veri dağılımları eğitimde aynı tutuldu; yalnız sınama etiketleri farklılaşır.

## 5. Önselin etkisi ve yeni deneyimin tam katkısı

Asıl deneyde veri hatasız üretilmesine rağmen adayların olabilirliği önceden sabitlenen `ε=.05` bozulma olasılığını kullanır:

\[
L_\theta(y\mid z)=
\begin{cases}.95,&F_\theta(z)=y,\\.05,&F_\theta(z)\ne y.\end{cases}
\]

Tahmin ise temiz işlevlerin posterior ortalamasıdır; tahmin değerine ayrıca yüzde beş etiket gürültüsü eklenmez. Bu seçimin amacı adayları tek çelişkide geri dönülmez biçimde silmemektir. Gerçekte gürültü bulunduğunun keşfi değildir.

`H` ile `S`, her hizalı gözlem için aynı olabilirliği alır. Posterior oranları bu gözlemlerle hiç değişmez:

\[
\frac{P(H\mid D)}{P(S\mid D)}=\frac{P(H)}{P(S)}.
\]

Kapıların 16'sına eşit önsel verildiğinde bu oran `1`'dir. Önsel kütlenin yarısı yüzey ailesine, yarısı 15 geçmiş kapısına bölündüğünde doğru geçmiş kapısı ile yüzey kapısının oranı `1/15` olur. Uzman çiftlerinin iki tarafta eşit koşullu önsel aldığı varsayılmıştır.

Gerçek dünya `H` olsun. Bir yeni gözlem ancak **hem** `s≠g(h)` **hem** `f₀(x)≠f₁(x)` olduğunda bu belirli iki açıklamayı ayırır. Böyle her gözlem oranı:

\[
\frac{1-\epsilon}{\epsilon}=19
\]

ile çarpar. `k` ayırt edici gözlemden sonra oran sırasıyla `19ᵏ` ve `19ᵏ/15` olur. Diğer gözlemler bu aday çiftinin oranını değiştirmez. Bu sonuç, tüm geçmiş adaylarının toplam kütlesi ile tüm yüzey adaylarının toplam kütlesi için otomatik olarak geçerli değildir: başka uzman çiftleri de hesaba katılır.

Dolayısıyla yüzey ipucunu değiştirmek tek başına yeterli olmayabilir. Uzmanların aynı cevabı verdiği girdide iki açıklama hâlâ aynı sonucu üretir. Kanıt sağlayan deneyim, rakip açıklamaların sonuçlarını birbirinden ayıran deneyimdir.

## 6. İşlev belirlenmesi ile yapısal açıklamanın belirlenmesi

Doğru geçmiş ailesinin içinde ve tam `h,x` küpü biliniyorsa, eşit olmayan uzmanlara sahip yapı belirlenebilir. Daha cebirsel bir tanık da vardır. `f₀=b₀⊕u₀ᵀx` ve `f₀⊕f₁=δb⊕δuᵀx` yazılırsa:

\[
F(h,x)=b_0\oplus u_0^\top x
\oplus \delta b(a^\top h)
\oplus\sum_{ij}a_i\delta u_jh_ix_j.
\]

`hᵢxⱼ` etkileşim katsayılarının matrisi `M=aδuᵀ`'dir. Gerçek uzmanlar tümleyen olmadığından `δu≠0`; `a≠0` olduğu için `M` sıfır olmayan, `F₂` üzerinde rankı 1 olan bir matristir. Burada sıfır olmayan skaler yalnız 1 olduğundan bu çarpanlara ayırma tektir. `h=0` kesiti `f₀`'ı verir; etkileşim matrisi `a` ve `δu`'yu, kalan doğrusal geçmiş katsayıları `δb`'yi belirler. Böylece `f₁` de bulunur.

Bu kanıt, tam işlev biliniyorsa **verilen aile içinde** bir ayrışmanın tanımlanabildiğini gösterir. Sonlu eğitimden bütün küpün zaten bilindiğini, bu ailenin doğru biyolojik açıklama olduğunu veya insanın aynı ayrışmayla düşündüğünü göstermez.

## 7. Deney sonuçlarına uygulanacak denetimler

1. Gerçek kapı kimliği, uzman kimliği ve ortamın aşama etiketi normal öğrenici güncellemesine verilmemeli. Ayrıcalıklı gerçek-kapı karşılaştırması açıkça ayrı tutulmalı.
2. Tam 512-girdi küpü başarısı, eğitimde görülmeyen girdilerin başarısıyla ayrıca raporlanmalı. Tam küp, öğrenilmiş etiketleri de içerir; doğrudan “yeni örneklere genelleme” sayılamaz.
3. İki önselin karşılaştırılması, bilgi ile varsayımın katkısını ayırır. En iyi sonucu veren önseli sınama sonuçlarından seçmek doğrulama sızıntısı olur.
4. Her gözlemde tahmin güncellemeden önce ölçülmeli. Yüzey ile geçmişi ayıran gözlem sayısı, toplam yeni gözlem sayısından ayrıca ayrılmalı.
5. Dört bitin sırasını yeniden adlandırmak ve uzman etiketlerini uygun eşdeğerlikle dönüştürmek, işlevsel sonuçları beklenmedik biçimde değiştirmemeli. Gerçek parite maskelerinin sabit kodlanması engellenmeli.
6. `ε=.05` ile bütün adaylar pozitif kütleyi korur. Bellek ve işlem maliyeti 16.384 gösterimin tutulmasına dayanıyorsa bu maliyet, küçük tablolara karşı başarıyla birlikte yazılmalı.
7. Aynı posteriorun yeniden yüklenmesi gelecekteki güncellemeleri de korumalı. Yalnız son tahminleri saklamak, öğrenme durumunun korunduğu anlamına gelmez.
8. “Geçmiş bağlamını öğrenme” sonucu, hazır dört bitlik geçmiş kodu üzerinde verilmelidir. Ham olay zincirinden bu kodun oluşması bu turda sınanmamaktadır.

## 8. İki birincil literatür bağlantısı

**Gershman, Blei ve Niv (2010), “Context, Learning, and Extinction”.** Gizli neden çıkarımıyla bağlama bağlı edinim ve sönmeyi modelleyen birincil kuramsal çalışmadır. Önceden verilmiş üretici varsayımlar altında gözlemlerin farklı nedenlere atanmasını inceler; neden sayısı da modelde genişleyebilir. Bu turla ortak nokta, aynı girdinin hangi öğrenilmiş yapıyla değerlendirilmesi gerektiğinin çıkarılmasıdır. Bizim 15 parite kapımız, onların genişleyebilen karışım modeli değildir. Makalenin kendisi de çıkarılan kümelerin gerçek nedenlerle aynı olmak zorunda olmadığını belirtir. Çalışma, bu turun parite varsayımını veya bilinç hakkında bir iddiayı doğrulamaz. DOI: [10.1037/a0017808](https://doi.org/10.1037/a0017808). [Yazarın kurumunda tam metin](https://www.cs.columbia.edu/~blei/papers/GershmanBleiNiv2010.pdf). Erişim: 2026-09-22; özellikle giriş ve model varsayımları incelendi.

**Rosenfeld, Ravikumar ve Risteski (2021), “The Risks of Invariant Risk Minimization”.** Dağılım dışı genelleme için değişmezlik hedeflerinin hangi koşullarda başarısız olabileceğini inceleyen birincil teorik çalışmadır. Buradaki ilişki, eğitimde başarılı olmanın sınamada değişen korelasyonlara karşı güvence vermemesi ve garantilerin veri/model varsayımlarına bağlı olmasıdır. Bu tur IRM algoritmasının yeniden uygulaması değildir; yukarıdaki iki-dünya kanıtı bu sonlu model için bağımsız ve daha basit bir tanıktır. Literatür bağlantısı, bu fikrin bütünüyle yeni olduğu iddiasını da desteklemez. [Tam metin](https://arxiv.org/html/2010.05761). Erişim: 2026-09-22; özet ve teorik kapsam incelendi.

## 9. Genel araştırma bakımından sonuç sınırı

Bu tur, öğrenilmiş deneyimin mevcut yorum yapısını nasıl değiştirebileceğine ilişkin bir parçayı ayırır: geçmişle uyumlu birden fazla yapı tutulabilir; onları farklı sonuçlara götüren yeni deneyim, bu yapıların göreli ağırlığını değiştirebilir. Bunun için hangi geçmiş değişkenlerinin tutulduğu, hangi aday yapılarının erişilebilir olduğu ve hangi varsayımların önceden verildiği önemlidir.

Bu parçanın başarıyla gösterilmesi; temsil dilinin deneyimden oluşmasını, kavramların sınırsız genişlemesini, uykuya özgü bütünleştirme mekanizmasını veya genel zekâyı açıklamış olmaz. Ana living-learning problemi bu alt probleme indirgenmemelidir.
