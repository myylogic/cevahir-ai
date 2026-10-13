# Tur 17 — Matematiksel iddialar ve literatür sınırı

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Başarım sonuçlarından önce hazırlanmış denetim

Bu belge [protokolün](PROTOCOL.md) neyi sınayabileceğini belirler. Deneyin başarılı olduğu, yeni bir temel ilke bulunduğu veya genel yaşayan öğrenmenin çözüldüğü iddia edilmez. Önceki araştırma sonuçlarını değiştirmez.

## 1. Öğrenilen nesne ve verilmiş araçlar

Bir görevdeki aday hesap `p`, iki değişkenli sonlu bir ifade ağacıdır. Erişilebilir kanıt `E`, 16 girdi–çıktı çiftidir. Öğrenilmiş bir düzeltme şeması `r`, bir giriş deseni ile bu desendeki değişkenlere bağlı sonuç ağacından oluşur. Verilen eşleştirme ve uygulama yordamı `r` ile yeni adaylar üretir; aynı kanıt bu adayları sınar.

Kaynak çözümün doğru AST'si verilmez: adayın kendi aramasıyla kabul ettiği önce/sonra çiftleri genelleştirilir. Bu özellik, hazır dört onarım kimliğinden birini seçmekle aynı değildir. Bununla birlikte başlangıç AST dili, yerel 108 yeniden kurma işlemi, genelleştirme algoritması, değişken bağlama ve toplu geçiş yorumlayıcısı araştırmacı tarafından verilmiştir. Öğrenici bunların anlamını veya üretim yöntemini icat etmez.

Bir şemanın daha derin altifadelere uygulanması, edinilmiş desenin hazır yapısal yorumlayıcıyla aktarımıdır. Özyinelemeyi keşfetme sonucu değildir. Hedeflerde şema çıkarımı dondurulduğundan bu tur, yeni hedef deneyimlerinin düzeltme dili veya genelleştirme yöntemini yeniden örgütlediğini de sınamaz.

Sözdizimsel ifade uzayı sınırsız ağaç boyutlarına izin verebilir; fakat sabit dünyada `F17² → F17` fonksiyonlarının sayısı sonludur: `17^289`. Buradaki olası kapasite kazanımı, verilen bütçe içinde doğru fonksiyona ulaşma yeteneğidir. Hesaplanabilir fonksiyon sınıfının genişletilmesi veya sınırsız bilişsel kapasite sonucu çıkarılamaz.

## 2. Kanıta uyma ve tam alan doğruluğu

Bir ifade için `[[p]](a,b)` onun F17 çıktısı olsun. Kabul koşulu

\[
\forall (a,b,z)\in E:\quad \llbracket p\rrbracket(a,b)=z
\]

iken araştırmacının tam doğruluk ölçüsü

\[
\forall (a,b)\in\mathbb F_{17}^{2}:\quad
\llbracket p\rrbracket(a,b)=\llbracket p_*\rrbracket(a,b)
\]

olur. İkincisi birinciden genel olarak çıkmaz. `p_*` dünya tarafındaki referanstır; öğreniciye verilmez. Tam tabloyu yalnız sonradan ölçmek, öğrenicinin tam tabloyla doğrulanmış bir kabul yaptığını söyleme hakkı vermez. Kaynak görevlerde de örnekleri geçen çözüm, ancak ayrıca raporlanan gizli değerlendirme açısından tam doğru olabilir.

**Dil düzeyinde karşıörnek.** Gözlenen 16 farklı çiftin ilk koordinatlar kümesi `S_x` olsun. En fazla 16 farklı değer olduğundan `S_x ⊊ F17` olur. Her `f` için

\[
g(x,y)=f(x,y)+\prod_{a\in S_x}(x-a)
\]

aynı 16 örneği açıklar. `x` gözlenmiş ilk koordinatlardan değilse çarpımdaki her terim sıfırdan farklıdır; alanın sıfır böleni bulunmadığı için çarpım sıfır değildir. Dolayısıyla `f` ve `g` bazı gizli girdilerde ayrılır. Bu ifade verilen toplama/çıkarma/çarpma dilinde kurulabilir.

Bu karşıörnek, söz konusu alternatifin protokoldeki sınırlı yerel arama ve bütçe içinde mutlaka bulunacağını söylemez. Gösterdiği, erişilebilir kanıtın genel dilde tek bir fonksiyonu belirlemeye yetmediğidir. Yanlış kabul oranı bu nedenle gerekli bir ölçüdür; yalnız kabul sayısı ilerleme sayılmaz.

**Tam alanın da sınırı vardır.** `x^17=x`, F17'nin bütün elemanlarında geçerlidir. Böylece `f(x,y)` ve `f(x,y)+(x^17-x)` 289 girdinin tamamında aynıdır. Kaynak AST'leri farklı olsa da bu deneyde aynı doğru fonksiyondurlar. Tam alan denetimi, amaçlanan kaynak programı veya tam sayılardaki davranışı özdeşleştirmez. F17 dışındaki sınama ayrı bir görev ve ayrı bir iddiadır.

Sayısal artıkların sıradan karesini kullanmak ayrıca bir arama tercihidir. Alan kalıntılarının `0,...,16` olarak temsilinde 0 ile 16 uzak görünür; bunun cebirsel olarak ayrıcalıklı bir hata geometrisi olduğu ileri sürülmez. Ortak kullanımı yöntem karşılaştırmasını belirginleştirir, tercih edilen arama geometrisinin genel doğruluğunu kanıtlamaz.

## 3. Ortak değişkenli genelleştirmenin koşulu

İki başarılı önce/sonra çifti `(p₁,q₁)` ve `(p₂,q₂)` genelleştirilirken aynı altifade çifti, giriş ve sonuç taraflarında aynı desen değişkenini kullanmalıdır. Bir çift için iki tarafta ayrı ayrı yeni değişkenler açmak, hangi parçanın taşındığını kaybettirir.

Örneğin `mul(add(x,2),y) → add(mul(x,y),2)` ve `mul(add(y,3),x) → add(mul(y,x),3)` çiftleri şu şemayı destekler:

\[
\operatorname{mul}(\operatorname{add}(A,B),C)
\longmapsto
\operatorname{add}(\operatorname{mul}(A,C),B).
\]

Burada `A`, `B`, `C` girişte bağlanır; sonuç aynı bağları kullanır. Sonuçtaki değişkenler girişte bağlananların altkümesi olmalıdır. Girişte tekrar eden bir değişken, eşleştirmede aynı AST'yi gerektirir. Başka bir ifadede değişken adlarının benzemesi veya iki parçanın yalnız örneklerde aynı çıktı vermesi bu yapısal eşitlik yerine geçmez.

Bu şema bir cebirsel özdeşlik değildir: `(u+v)w` ile `uw+v` genel olarak farklıdır. Şema, bazı görevlerde istenen davranış değişikliğine adaydır. Kaynak başarı, bütün sonraki görevlerde aynı değişikliğin isteneceğini garanti etmez. Anti-unification sözdizimsel bir genelleştirme sağlar; anlam doğruluğu veya doğru uygulanma bağlamını tek başına ispatlamaz.

Toplu uygulamanın sonlanması da öğrenilmiş bir başarı değildir. Protokolde eski ağacın düğümleri bir alttan üste geçişte ziyaret edilir ve üretilen düğümler aynı geçişte yeniden gezilmez. Bu uygulama semantiği sonluluğu sağlar. Sonuç, sınırsız biçimde sabit noktaya kadar uygulanan herhangi bir yeniden yazım sisteminin sonlanacağına genellenemez.

## 4. Nedensel ablasyonun doğru yorumu

Kaynak-sonrası ortak durum `s`, yöntem müdahalesi `I∈{açık,kapalı}`, eşleştirilmiş hedef yaşamı `L` olsun. `Y(s,I,L,B)` bütçe `B` altında edinilen tam doğru hedef sayısını göstersin. Karşılaştırma

\[
\Delta(L,B)=Y(s,\text{açık},L,B)-Y(s,\text{kapalı},L,B)
\]

ancak iki kol aynı `s`, aynı görev kanıtları ve aynı bütçe sayacıyla başlayıp yalnız şema yürütme müdahalesinde ayrılıyorsa bu müdahalenin etkisi olarak yorumlanabilir. `rules_disabled_paid` şema verisini, eski çözümleri ve edinim bedelini koruduğu için bu soruya yöneliktir. Kopyaların değişebilir belleği paylaşmaması ve gizli doğrulamanın sonraki kararlara sızmaması denetlenmelidir.

**Ardışık hedeflerde aracılı etki:** Hedeflerde kabul edilen çözümler kütüphaneye eklendiğinden, iki kolun kütüphaneleri ilk farklı sonuçtan sonra ayrışabilir. Sekiz hedefin toplamındaki fark, şema kullanımının bu sonraki kütüphane etkilerini de içeren toplam etkisidir. Her sonraki hedef için aynı eski çözümler korunmuşken yalnız anlık kural yürütmesinin etkisi ölçüldüğü söylenemez. Böyle bir doğrudan etki için her hedef öncesinde ayrıca ortak durumdan dallanmak gerekir; mevcut protokol bu ek deneyi içermez.

24 yaşam esas eşleştirme birimidir. Rejimler ve yöntemler aynı kaynakları/adayları paylaştığından 360 yol veya 2.880 hedef denemesi bağımsız örnek sayısı gibi sunulmamalıdır. Yalnız toplam doğru sayısı değil, yaşam içi eşleştirilmiş farklar, yanlış kabul ve maliyet de gösterilmelidir.

Şema açık kolun ablasyondan üstün olması, şemanın verilmiş sistemde kullanılmasının yararını destekleyebilir. Bunun en iyi uygulanabilir yöntem olduğu sonucunu vermez: şema üretmeyen bir yöntem aynı maliyeti başka aramalara ayırabilir.

## 5. Bütün yaşam maliyeti ve güçlü toplu karşılaştırma

Bir yöntemin yaşam maliyeti kaynak arama, durum kurma, şema üretme ve bütün hedef girişimlerinden oluşur:

\[
C_{\rm yaşam}=C_{\rm kaynak}+C_{\rm kurulum}
+C_{\rm şema}+\sum_t C_{{\rm hedef},t}.
\]

Şemalı yöntem için `C_şema` ve başarısız eşleştirmeler de sayılır. Diğer yöntemlere kullanılmayan şema üretim bedeli yüklenmez; bunun istisnası bilinçli olarak maliyeti eşleyen nedensel ablasyondur. Aynı aday sayısı aynı maliyet değildir: derin bir ağaçta toplu değişiklik yapmak, küçük bir ağaçta tek değişiklik yapmaktan daha çok iş gerektirebilir.

Protokolün düğüm temelli iş sayımı, tanımlanan soyut makine altında karşılaştırmadır. Python hashing, sıralama ve işletim sistemi giderleri dışarıda olduğundan gerçek CPU süresi veya toplam RAM eşitliği iddia edilemez. Kalıcı JSON boyutu ayrıca raporlanmalıdır; bu da tepe çalışma belleğinin yerine geçmez.

`bulk_ranked` temel kontroldür: kendi kaynak başarılarından öncelik öğrenir ve aynı işlem/parantez bağlamındaki bütün uygun düğümlere hazır toplu işlem uygulayabilir. Dolayısıyla şemalı yöntemin yalnız tekli yerel aramayı geçmesi, öğrenilmiş desenin gerekli olduğunu göstermez. Şema kazancı bu güçlü karşılaştırmada kaybolursa, sınanan ailede toplu uygulama ve deneyimle sıralama yeterli bir açıklamadır.

Şema çıkarma bedelinin kısa yaşamda karşılanmaması geçerli bir negatif sonuçtur. Değişmiş veya kısmi rejimde eski şemanın yanlış adayları öne alması da koşulsuz gelişim yerine olumsuz aktarım gösterebilir. Birincil rejimi veya bütçeyi bu sonuçlar görüldükten sonra değiştirmek yeni bir deney olur; eski protokol sonucu yerine sunulmamalıdır.

## 6. Birincil literatürle eşleşme

**Getafix — Bader, Scott, Pradel ve Chandra, 2019.** Makalenin §4.1–4.3 bölümleri önce/sonra AST değişikliklerini anti-unification ile genelleştirir; ortak delik eşlemeleri değişen taraflar arasındaki bağı korur. Öğrenilen desenlere bağlam eklenir ve aday düzeltmeler sıralanır. Bu, bizim değişken bağlı düzenleme çıkarımımızla doğrudan yöntemsel eşleşmedir. Getafix'in insan değişiklikleri ve statik analiz uyarılarıyla kurulan düzeni, küçük deneyimizin kendi aramasından elde ettiği çiftlerle aynı deney değildir. Bu ayrılık anti-unification veya dönüşüm öğrenmeyi yeni temel ilke yapmaz. Birincil yazar PDF'sinin ilgili yöntem bölümleri okundu. [Getafix makalesi](https://software-lab.org/publications/Getafix_arXiv_1902.06111.pdf)

**REFAZER — Rolim ve diğerleri, ICSE 2017.** §III, verilen AST dönüşüm dili içinde örneklerden eşleştirme ve değiştirme kuralları sentezler; çıkarımsal arama ve sıralama kullanır. Örneklerle uyumlu dönüşümlerin aşırı genellenebilmesi veya fazla özel kalabilmesi açık bir sorundur. Hazır somut onarım listesi vermemek, dönüşüm dilinin ve sentez yönteminin araştırmacı tarafından sağlandığı gerçeğini kaldırmaz. Bu ayrım mevcut protokol için de geçerlidir. Birincil yazar PDF'sindeki yöntem bölümü okundu. [REFAZER makalesi](https://cseweb.ucsd.edu/~ldantoni/papers/icse17.pdf)

**SOAR — Pourcel, Colas ve Oudeyer, ICML 2025.** Resmî bildiri özeti, arama girişimlerini geriye dönük eğitim örneklerine dönüştürüp sonraki program örnekleme ve düzeltme yeteneğini geliştiren bir döngü tanımlar. Dolayısıyla kendi deneyiminden sonraki düzeltme becerisini geliştirme fikri tek başına yeni değildir. Bu kaynak için resmî özet doğrulandı; bu denetimde PDF açılamadığından ayrıntılı algoritma veya maliyet eşdeğerliği ileri sürülmüyor. [Resmî SOAR kaydı](https://proceedings.mlr.press/v267/pourcel25a.html)

Bu kaynaklar bütün yenilik araştırmasının tamamlandığını göstermez. Olumlu deney, mevcut yöntem aileleriyle bağlantılı küçük ve yeniden üretilebilir bir nedensel sonuç olabilir. “İlk”, “genel öğrenme yasası”, “yeni matematiksel temel” veya “AGI çözümü” ifadeleri için dayanak olmaz.

## 7. Ana probleme geri bağlantı

Bu tur, yaşantının yalnız saklanan cevabı değil, ileride yeni cevapları bulmak için kullanılan aday üretme işlemini değiştirmesinin sınırlı bir örneğini sınar. Kalıcılık yeniden başlatma ile, aktarım görülmemiş derin ifadelerle, kontrol yanlış kabul ve olumsuz aktarım ölçüleriyle incelenir.

Temsil dilinin oluşumu, doğrulama ölçütünün edinimi, hedeflerin ve deneyimlerin seçimi, aynı anda çok sayıda beceriyi koruma, yeni düzeltme yöntemleri üretme yönteminin değişmesi ve uzun süreli kaynak paylaşımı açık kalır. Buradaki program onarım sonucu bu bileşenlerin birleşik cevabı değildir. Kullanıcının kavramların yeniden örgütlenmesi ve deneyimlerle yeni bağların oluşması penceresi korunur; küçük alanın başarı ölçüsüne indirgenmez.
