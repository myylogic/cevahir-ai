# Edinilmiş program kütüphanesi: ifade gücü, arama ve düzeltme

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma turu 11

Bu not, deneyimden edinilen işlem tanımlarının sonraki program edinimine nasıl katkıda bulunabileceğini inceler. **Bu dosya yazılırken yeni deneyin istatistiksel sonuçları henüz bulunmamaktadır.** Aşağıdaki tanıklar analitiktir; DreamCoder, OOPS veya Stitch'in yeniden uygulanması ve genel yaşarken öğrenme probleminin çözümü değildir.

## 1. Önceki negatif sonuç korunuyor

[Temsil büyümesi araştırmasında](../living_learning_reassessment_2026_09_20/REPRESENTATION_GROWTH.md), deneyimle oluşturulan çarpım özellikleri sonraki bir görevin hesabında kullanılmıştı. Ancak aynı özellikleri baştan içeren ve aynı etkinleştirme sırasını kullanan güçlü sabit karşılaştırma, tahminler ve bütün iş sayaçlarında eşleşmişti. Ayrıca öncül ikili görev verilmeden üçlü göreve geçilince büyüyen yöntem başarısız, sabit geniş arama başarılıydı. Bunlar geri alınmaz.

Yeni soru “kütüphane büyüyebilir mi?” değildir. **Aynı temel işlem dili ve aynı deneyimler verildiğinde, edinilmiş kütüphane ile arama politikasının hangi kısmı, hangi toplam bütçede sonraki çözüme erişimi değiştirir?** Aranan fark mevcut matematiksel fonksiyon ailesinden çıkmak olmayabilir. Daha önce izin verilen ama bulunamayan bir programı şimdi bulabilmek, kaynakları belirtilmiş gerçek bir kapasite değişimidir; bunun yeni ifade gücü olduğunu söylemek farklı ve daha güçlü bir iddiadır.

## 2. Açılabilen makrolar mutlak ifade gücünü artırmaz

Temel program dili `P₀`, kütüphane `L={m₁,…,m_k}` olsun. Her makronun gövdesi temel işlemlerden ve yalnız daha önce tanımlanmış makrolardan oluşsun. Bağımlılıklar sonlu ve döngüsüz olsun; değişken bağlama ve çağrı anlamı korunarak açılım yapılabilsin. `expand_L(p)`, bütün makro çağrılarını temel dile çevirsin. Bu koşullarda:

\[
\forall p\in P_L,\quad
\llbracket p\rrbracket_L=\llbracket\operatorname{expand}_L(p)\rrbracket_0.
\]

**İspat:** Kütüphane tanım sırasına göre indüksiyonla her makro temel bir programa açılır. Sonra programın yapısı üzerinde aynı yerine koyma uygulanır. Temel programlar genişletilmiş dilde hâlâ kabul ediliyorsa iki dilin hesaplayabildiği fonksiyon kümeleri eşittir. Dildeki program uzunluğuna ek sınır konulmadığı varsayımı önemlidir.

Bu önerme salt sözdizimsel ve anlamsal bir derleme eşdeğerliğidir; açılımın kısa, hızlı veya az bellekli olduğunu söylemez. Yan etkiler, rastgelelik veya çağrı sırasında ortak durum varsa naif metinsel yerine koyma anlamı değiştirebilir; değerlendirme sırası ve paylaşım ayrıca korunmalıdır. Temel dilde bulunmayan dış araç veya oracle eklemek de bu önermenin kapsamında değildir. Sonlu döngüsüz makro koşulu bırakılırsa temel dilin yeni özyinelemeyi ifade edebilmesi ayrıca gösterilmelidir.

Sabit bir yorumlayıcı, değişen `L`yi veri olarak saklayıp yürütebilir. Bu nedenle “sistem yeni işlem adı ekledi” gözlemi, bütün sabit mimarilere karşı olanaksızlık kanıtı değildir. Aynı deneyimden aynı kütüphane ve öncelikleri edinen sabit yorumlayıcı, aynı öğrenme sürecinin başka bir kodlaması olabilir. Deneyimin edinilmiş iç durum üzerindeki nedensel etkisi yine vardır; mimarinin fiziksel büyümesi zorunlu sonuç değildir.

## 3. Bütçe altında erişim aynı kalmak zorunda değildir

`S(L,π,E,B)`, kütüphane `L`, arama politikası `π`, gözlenen örnekler `E` ve bütçe `B` ile sınanan program dizisini belirlesin. Bütçe; aday sayısı, primitive yürütme adımı, bellek veya süre olabilir. Bunlar ayrı ölçülerdir. Fonksiyon kümesi eşitliği, aynı bütçe içinde ziyaret edilen aday kümelerinin eşitliğini gerektirmez. Gözlenen örneklere uymak da hedef fonksiyonu bütün girdilerde doğru öğrenmekle aynı değildir; deneyde ayrı değerlendirme gerekir.

**Erişim tanığı:** Temel iki işlem, girdiye sırasıyla `a` veya `b` karakterini eklesin. Programlar önce uzunluk, sonra `a<b` sırasıyla gezilsin ve boş program da aday sayılsın. `x↦xaaa` hedefinin doğru programı şu sırada sekizincidir: boş, `a`, `b`, `aa`, `ab`, `ba`, `bb`, `aaa`. Daha önce edinilmiş `M=aaa` makrosu işlem sırasında ilk gelirse, makrolu sıralamada boş programdan sonra `M` ikinci adaydır. Aday bütçesi iki iken erişim değişir; temel dilin ifade gücü değişmez.

Makro çalışırken üç temel ekleme hâlâ yürütülür. Her aday sıfırdan yürütülürse bu basit örnekte hedefe kadar primitive işlem toplamı temel sırada 13, makrolu sırada 3'tür; fark hedef programın hızlı yürütülmesi değil, gereksiz adayların daha az denenmesidir. Ebeveyn çıktısını paylaşan deney BFS'si farklı bir maliyet hesabı kullanır; bu 13 sayısı onun ölçümü değildir. Makronun önce nasıl edinildiği ve bu edinimin maliyeti bu sayılara dahil değildir. Dolayısıyla bu rakamlar toplam yaşam üstünlüğü veya öğrenilmiş makro seçiminin başarılı olduğu yönünde deney sonucu sayılmaz.

**Eşleştirilmiş açılım kontrolü:** Aynı `L` ve `π` bilgisine sahip temel-dil arayıcısı, makrolu arayıcının önerdiği `p₁,p₂,…` dizisi yerine `expand_L(p₁),expand_L(p₂),…` dizisini aynı sırayla önerebilir. Yürütme maliyeti primitive açılımla ölçülüyorsa aday sayıları ve bu hesaplar eşleşir; derleme/çağrı giderleri ayrıca kalır. Bu kontrol öğrenilmiş bilgiyi kaldırmaz, onun makro sözdizimi yerine aday sırasına kodlanabildiğini gösterir. Makrolu yöntemin bu kontrolle eşitliği, deneyimin yararını çürütmez; yararın zorunlu olarak yeni primitive semantiğinden geldiği iddiasını sınırlar.

## 4. Kısa betimleme, ucuz edinim veya olumlu aktarım değildir

Önceki tanıkta yeni hedef `x↦xb` olsun. Temel sıra boş, `a`, `b` ile üç adayda çözer. `M,a,b` işlem sırasıyla kütüphaneli yöntem boş, `M`, `a`, `b` olmak üzere dört aday harcar. Üç adaylık bütçede önceden yararlı olan makro artık başarıyı engeller. Makroyu kullanma zorunluluğu yoktur; zarar arama önceliğinden gelir. Bu, her kütüphane algoritmasının kaçınılmaz zararı değildir: temel aramaya ayrılmış ayrı bütçe veya uygun yeniden sıralama bu özel sorunu sınırlar.

Genel olarak `b` işlemden oluşan bir sözdiziminde uzunluğu en fazla `d` olan dizilerin sayısı `Σ_{j=0}^d b^j`dir. `k` makro eklenince aynı sözdizimsel sayım `Σ_{j=0}^d(b+k)^j` olur. Bazı hedeflerin derinliği kısalırken dallanma ve anlamsal tekrar artabilir. Türler, anlamsal tekilleştirme veya güdümlü öneriler bu kaba sayımı değiştirir; formül genel çalışma süresi teoremi değildir.

Sıkıştırmanın kendisi de kütüphane giderini ödemelidir. Tanımı `ℓ`, çağrısı `c`, kayıt gideri `h` birim olan bir makro `k` kez aynı biçimde kullanılıyorsa basitleştirilmiş betimleme kazancı:

\[
k\ell-(\ell+h+kc)=(k-1)\ell-h-kc.
\]

Tek kullanımda pozitif giderler altında kazanç yoktur. Bu hesap argümanları ve kodlamayı açıkça sabitleyen bir örnektir; gerçek AST/bit maliyetini temsil ettiği varsayılamaz. Kısa kaynak kodu, aynı aritmetik işlemleri yapan makronun yürütmesini kendiliğinden hızlandırmaz.

Edinim/madencilik/doğrulama toplamı `C_build`, sonraki görev maliyetleri kütüphaneyle `C_L,i`, temel yöntemle `C_0,i` olsun. Toplam hesapta kazanç ancak

\[
C_{build}+\sum_i C_{L,i}<\sum_i C_{0,i}
\]

ise vardır. Sabit görev başına tasarruf `s>0` için eşik `n>C_build/s` olur. Gelecek dağılım değişince `s` negatif olabilir. Yalnız sınama zamanındaki kazancı raporlamak yararlı bir ölçüdür; onu edinim maliyeti dahil yaşam kazancı diye adlandırmamak gerekir. Açık kütüphane boyutu da en iyi örtük temsil için bellek alt sınırı değildir.

## 5. Aynı kütüphaneyi baştan vermek neyi sınar?

Sonradan edinilen tam kütüphaneyi başlangıçta veren kontrol, erken aşamada geleceğin bilgisini taşıdığı için ayrıcalıklıdır. Online öğrenciyle toplam edinim yarışında eşit koşullu rakip değildir. Buna karşı sonraki görevler başladığında aynı kütüphane, ağırlıklandırma ve aday sırası verildiğinde davranışın eşitlenmesi beklenebilir. Bu, kütüphane ediniminin maliyetini ve rolünü ayıran güçlü bir kontroldür; sırf kolaylaştırıldığı için dışlanmamalıdır.

Başlangıçtan verilen büyük bir kütüphaneyi kötü sıralamayla bütün adayları taramaya zorlamak, dinamik yöntemi gereksiz yere avantajlı gösterebilir. Önceki çalışmadaki gibi aynı etkinleştirmeyi kullanan kontrol korunmalıdır. Aynı sayıda rastgele makro, başka yaşamdan edinilmiş makro ve öğrenilmiş makroları aynı sırayla primitive programlara açan kontrol, bilgi miktarı, kaynak ve gösterim etkilerini ayrı sınar.

Deneyimin gelecekteki aramaya katkısı için yeni görev başlamadan önce kütüphaneyi dondurup aynı görev/etiket akışında taşıma, silme ve yanlış kütüphane müdahaleleri yapılabilir. Önceki görevlerin yalnız son cevaplarını taşıyan kontrol, taşınan avantajın yeniden kullanılabilir işlemden mi yoksa görev yanıtlarını hatırlamadan mı geldiğini sınar. Görev adı, hedef program gövdesi veya geleceğin motif listesi öğreniciye gizlice verilmemelidir.

## 6. Yanlış kanıt: program tanımı ile doğruluk iddiası ayrılmalı

Sonlu örneklerde tutarlı bulunan bir programın başka girdilerde doğru olduğu garanti değildir. Örnekler yalnız `(0,0),(1,1)` ise `x` ve `x²` aynı kanıtı açıklar. Bu veriden seçilen `F(x)=x` tanımı yanlış yürütülmüş değildir; onun bilinmeyen hedefin bütün girdilerindeki çözümü olduğu iddiası desteklenmemiş olabilir.

Yeni kanıt hedefin `x²` olduğunu gösterirse `G(x)=F(x)+1` ile öğrenilmiş `x²+1` görevi de sorgulanmalıdır: `x=2` için eski `G` üç verirken doğru cevap beştir. Fakat bağımsız bir görev gerçekten `x+1` istiyor ve aynı `F`yi kullanıyorsa, `F` gövdesini sessizce `x²` yapmak bu doğru programı bozar. Sözdizimsel bağımlılık, bütün bağımlı programların aynı nedenle yanlış olduğunu söylemez.

İki uygulanabilir ayrım vardır: değişmez sürümlü tanımlar eski programın semantiğini korur; kanıt/iddia bağımlılıkları ise hangi görevlerin yeniden doğrulanması gerektiğini izler. Düzeltme gerektiğinde bağımlı doğruluk kayıtlarının yeniden sınanması gerekebilir, fakat tüm alt programların silinmesi zorunlu değildir. Programlar tamamen açılmışsa yürütme bağımlılığı kaybolabilir; eski kanıta dayalı seçimin bilgi bağımlılığı yine sürebilir. Bu düşünce deneyi gürültü ile gerçek hedef değişimini ayıran tamamlanmış bir düzeltme algoritması değildir.

## 7. Doğrulanan üç birincil literatür eşleşmesi

İlgili makaleler 21 Eylül 2026'da doğrudan açılıp belirtilen bölümleri incelendi. Buradaki cebirsel notlar yöntemlerin yenilik iddialarını yeniden sahiplenmez.

**Ellis vd. (2021), [DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf), PLDI, DOI [10.1145/3453483.3454080](https://doi.org/10.1145/3453483.3454080).** §2 temel dilin eğitim problemlerini prensipte çözebilecek kadar güçlü olmasını ister; edinilen soyutlamalar daha kısa programlar ve sınırlı aramada erişim sağlar. Kütüphane ile sinirsel arama politikası birlikte öğrenilir. Dolayısıyla yalnız kütüphaneyi değiştiren küçük deney, bütün DreamCoder mekanizmasının tekrarı değildir. Çalışmanın uygun müfredat/alan paylaşımı koşulu, ilgisiz sonraki görevlerde otomatik olumlu aktarım varsaymamamızı gerektirir.

**Schmidhuber (2004), [Optimal Ordered Problem Solver](https://sferics.idsia.ch/pub/juergen/oopsmlj.pdf), Machine Learning 54, 211–254; [yazarın HTML sürümü](https://people.idsia.ch/~juergen/oopsweb/oopsweb.html).** §2.1'de optimalite verilmiş aday öncelikleri ve aday oluşturma/çalıştırma/sınama zamanı açısından tanımlanır. §3–4 eski çözüm programlarından yararlanırken yeni başlangıçlara da arama zamanı ayırır. Dolayısıyla “önceki programları gelecekte çağırmak” veya “sıfırdan aramayı korumak” yeni fikir değildir. Makalenin koşullu yakın bias-optimalite sonucu, keyfî motif çıkarıcıya veya gerçek donanımda her görev için üstünlüğe taşınamaz.

**Bowers vd. (2023), [Top-Down Synthesis for Library Learning](https://arxiv.org/pdf/2211.16605), PACMPL 7, POPL; [yayımlanmış sürüm kaydı](https://dspace.mit.edu/entities/publication/ad86e996-94bb-4326-9757-7d686da8fde5), DOI [10.1145/3571234](https://doi.org/10.1145/3571234).** Stitch, program korpusundaki ortak soyutlamaları üstten aşağı arar. §4'teki sıkıştırma hedefi soyutlamanın kendi maliyetini de düşer; §6 kütüphane kalitesi, kaynak verimliliği ve ölçeklenmeyi değerlendirir. Kütüphane üretiminde hızlanma, sonraki bütün problem aramalarında veya program yürütmelerinde aynı hızlanma değildir. Kullanacağımız küçük makro seçiciyi Stitch'in algoritması veya sonuçlarıyla özdeşlemiyoruz.

## 8. Yeni deney için ayrıştırıcı ölçüler

Başarı yalnız daha kısa bulunan programa bağlanmamalıdır. Aynı yaşamda edinim örnekleri, kütüphane çıkarımı, aday üretimi, primitive yürütme, doğrulama, saklanan tanımlar ve sonraki görev başarısı ayrı kaydedilmelidir. Aday sayısına eşit bütçe ile toplam temel işlem maliyetine eşit bütçe farklı soruları yanıtlar. İlkinde avantaj bulunması ikincisini sağlamaz.

Olumlu aktarıma uygun yaşam yanında ilgisiz görev, ters müfredat, öncül görevin çıkarılması ve yanlış kanıttan edinilmiş makro koşulları bulunmalıdır. Aynı edinilmiş kütüphaneyi baştan sunan kontrolün ve aynı aday sırasını primitive dilde üreten kontrolün eşitliği saklanmamalıdır. Sıradan sabit yorumlayıcının aynı davranışı taklit edebilmesi öğrenmenin yokluğu değildir; yeni hesaplanabilirlik sınıfı iddiasını sınırlar.

Bu turun olası katkısı, deneyimin hangi yeniden kullanılabilir hesap durumunu oluşturduğunu ve bunun sonraki edinim maliyetini hangi koşullarda değiştirdiğini nedensel olarak ayırmaktır. Kitap, önceki araştırma ve sabit geniş karşılaştırmanın eşitlik sonucu korunur. Genel yaşarken öğrenme problemi, program kütüphanesi öğrenimi olarak yeniden adlandırılıp çözülmüş kabul edilmez.

Editoryal denetim notu: Ana sonuçlardan sonra kaynak bölümünün Stitch §5 atfı §6 olarak düzeltildi ve analitik örnekteki 13 işlem sayısının her adayı sıfırdan yürütme varsayımı açıklandı. Önceki tur dosyaları ve deney sonuçları değiştirilmedi; bu açıklamalar yeni başarım hipotezi veya deney seçimi değildir.
