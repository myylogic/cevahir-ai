# Yaşarken öğrenme: kapsamın geri açılması ve matematiksel ayrımlar

20 Eylül 2026. Bu belge [önceki araştırma raporuna](../living_learning_2026_09_20/REPORT_TR.md) ve [önceki matematiksel temellere](../living_learning_2026_09_20/FOUNDATIONS.md) ek niteliğindedir. Eski deneyler, negatif sonuçlar ve literatür eşleşmeleri korunur. Burada bir genel öğrenme algoritmasının bulunduğu iddia edilmez. Aşağıdaki küçük karşı örnekler analitik olarak türetilmiştir; ayrı bir geniş deney başarısı değildir.

**Ana problem hâlâ açıktır:** Bir sistemin çalışırken aldığı deneyim, gelecekte hangi hesaplamaları hangi maliyetle yapabildiğini nasıl kalıcı, kontrollü ve yeni durumlara aktarılabilir biçimde değiştirebilir? Yeniden sınama, bu sorunun deneyime erişim tarafında bir alt problemdir. Öğrenilebilir yapının edinilmesi, deneyimin ilgili eski hesaplamalara bağlanması, temsilin geliştirilmesi, kalıcılık, düzeltme, aktarım ve sonraki öğrenmenin değiştirilmesi aynı sorunun başka parçalarıdır.

## 1. Korunan sonuçların kapsamı

| Sonuç | Bilimsel statüsü | Ana problemdeki görevi | Çıkarmadığı sonuç |
|---|---|---|---|
| Deneyimden sonra gelecekteki davranış değişiyorsa, davranışa nedensel olarak katılan erişilebilir durum değişmiştir. | Sistem sınırı açık tutulduğunda genel gerekli koşul. | Öğrenme iddiasının maddi ve nedensel yerini saptar. | Her durum değişimi faydalı öğrenme değildir; nasıl güncelleneceğini söylemez. |
| Eşit bilgi, farklı hesap düzeniyle sabit bütçede farklı kapasite verebilir. | Genel kavramsal ayrım; affine derleme dar yapıcı örnek. | Bilgi edinimi ile edinilmiş bilginin kullanılabilirliğini ayırır. | Her derleme yeni bilgi veya toplam yaşam hesabında üstünlük değildir. |
| Ayrı bir eğitim dönemi mantıksal zorunluluk değildir. | Çevrimiçi affine öğrenme bir varlık kanıtıdır. | Çalışma ve güncellemenin aynı süreçte gerçekleşebildiğini gösterir. | Bilinmeyen işlem ailesinin, temsilin veya geri bildirim eşlemesinin keşfi değildir. |
| Eksik temsilde aynılaştırılan iki gerekli ayrım, yalnız o temsili okuyan bir başlıkla geri getirilemez. | Açık temsil ve görev için genel işlevsel sınır. | Temsil kaybını daha fazla optimizasyonla karıştırmayı önler. | Her başarısızlık temsil kaybı değildir; daha büyük vektör zorunlu değildir. |
| Doğru kimlikli bağımsız kural ve eksiksiz bağımlılık izleme seçici düzeltmeye izin verir. | Belirtilen ailede yapıcı yeterlilik. | Koruma ve düzeltmenin birlikte mümkün olduğunu gösterir. | Ortak ağırlıklarda, bilinmeyen kimliklerde ve eksik neden kayıtlarında aynı garanti yoktur. |
| Geçmiş görevler sonraki görevlerde önseli yararlı değiştirebilir; aile değişiminde zarar verebilir. | Verilmiş hipotez aileleri altında deneysel sonuç. | Aktarımın yönü ve kapsamı ölçülmelidir. | Güncelleme dilinin veya yeni ayrımların kendisinin edinildiğini göstermez. |
| Politika ayırt edici kanalı kapatırsa yeni karşı kanıt gelmeyebilir. | Gözlem erişimi varsayımlarına bağlı yapısal sonuç. | Veri üretiminin davranışa bağlılığını görünür kılar. | Bütün öğrenme eylemle yeniden sınama gerektirir demek değildir. Pasif yeni veri de öğrenme sağlayabilir. |
| Deterministik iç değişim ilgili eylemi tekrar açabilir. | Kalıcı fırsatlı küçük ailede yapıcı örnek. | Rastgeleliğin o ailede zorunlu olmadığını gösterir. | İç değişim, öğrenilecek yapıyı veya doğru güncelleme kuralını kendiliğinden üretmez. |
| Sonlu yanlış-deneme maliyetiyle keyfî geç bütün değişimleri ortak garanti altında saptama çatışır. | Yalnız maliyetli sınamanın bilgi verdiği ailede koşullu sınır. | O keşif sözleşmesinin gerçek bedelini açıklar. | Uygulanabilir yaşarken öğrenmenin imkânsızlığı değildir. |

Bu sınıflandırma eski sonuçları geri almaz. Bir sonucun geçerli olduğu bölgeyi, ana problemin tamamına uygulanmasından ayırır.

**Gereksinim şişirmesi düzeltilmelidir.** Kullanıcının “kontrollü, kalıcı ve genellenebilir” talebi; bütün dünyalarda sıfır hata, sıfır risk, sınırsız bağımsız bilgi veya sonsuz zaman boyunca değişmezlik talebi değildir. Bu aşırı garantilerin karşı örnekleri araştırma kaydında kalır. Fakat onlardan “ana problem için önce kullanıcı bütün dünya sınıfını tam belirtmeli” ya da “genel ilke bulunamaz” sonucu çıkmaz. Araştırmacı gerçekçi aileler, ölçülebilir toleranslar ve kaynak sınırları seçebilir; bunları aşamalı genişletebilir. Seçilen sözleşmenin gerekçesini ve kalan açığı açıklamak araştırmanın yükümlülüğüdür.

## 2. Bir anlık başarıdan yaşam boyunca öğrenebilirliğe

Önceki risk ve bütçe tanımları korunur. Bununla birlikte yalnız `R_Q(s_t)` ölçmek, iki durumun gelecekteki öğrenme kapasitesini ayırt etmeyebilir. Aynı anlık davranışı üreten iki durum; gelecek veri geldiğinde, eski bir bilgi düzeltildiğinde veya temsil değiştiğinde farklı gelişebilir.

`h` alınmış geçmiş, `c(h)` saklanan durum, `z` yeni bir deneyim olsun. Tam geçmişe göre hedeflenen güncellemenin sıkıştırılmış durumdan yürütülebilmesi için bir `u` yordamı şu eşitliği sağlamalıdır:

\[
c(hz)=u(c(h),z).
\]

**Önerme 1 — Tam çevrimiçi güncellenebilirlik.** Böyle bir işlevsel `u`, erişilebilir geçmişler üzerinde ancak ve ancak

\[
c(h)=c(h')\ \Longrightarrow\ c(hz)=c(h'z)
\]

her ortak izinli `z` için geçerliyse vardır.

**İspat.** `u` varsa eşit girdilere aynı çıktıyı verir. Tersine, koşul varsa her erişilebilir `s=c(h)` için `u(s,z)=c(hz)` tanımlanabilir; seçilen `h` temsilcisinden bağımsızdır. Bu bir işlevin varlık sonucudur. `u`nun sonlu kaynakla hesaplanabilir, veriden öğrenilebilir veya verimli olduğunu kanıtlamaz. Stokastik durumlarda uygun karşılık eşit çıktı dağılımları ve güncelleme çekirdekleriyle kurulmalıdır.

Şimdi yalnız ortalama tahmini yapan sistemi ele alalım. `h=(0,2)` ve `h'=(1)` için mevcut ortalama aynıdır: `1`. Yeni `4` geldiğinde doğru ortalamalar sırasıyla `2` ve `2.5` olur. **Mevcut cevabı vermeye yeterli tek sayı, sonraki öğrenmeyi doğru sürdürmeye yeterli değildir.** `(toplam,adet)` saklamak bu örnekte gerekli güncellemeyi sağlar. Kalıcı bir cevabın saklanmasıyla kalıcı öğrenme durumunun saklanması aynı şey değildir.

Bu ayrımı daha genel değerlendirmek için, gelecekteki izinli deneyim devamlarına `z`, sorgulara `q`, tam geçmişe göre değerlendirilmek istenen hedefe `F(hz,q)` diyelim. İki geçmişi şu şekilde eşdeğer sayabiliriz:

\[
h\equiv_{\mathcal Z,\mathcal Q}h'
\quad\Longleftrightarrow\quad
F(hz,q)=F(h'z,q)\quad\forall z\in\mathcal Z,\ q\in\mathcal Q.
\]

Boş devam `z=\epsilon` için eşitlik yalnız mevcut davranışı sınar. Bütün izinli devamlar için eşitlik, gelecekteki güncellemelerde hangi geçmiş ayrımlarının korunması gerektiğini söyler. Bu tanımda gelecek cevaplar zaten biliniyormuş gibi bir oracle kullanılarak algoritma yazılamaz; tanım, önerilen sıkıştırmanın hangi karşı örnekle yanlışlanacağını belirtir. Tüm devamları eşit ağırlıkla korumak da kaynak hedefi değildir: yaklaşık uygulamada önemli devamların dağılımı, tolerans ve maliyet açıkça seçilmelidir.

Literatürde geçmişleri gelecek davranışları üzerinden eşdeğerleştirme yeni değildir. Causal-state yaklaşımı, aynı gelecek dağılımını veren geçmişleri birleştirerek öngörü için yeterli ve uygun koşullarda minimal bir durum kurar. Bu, bilinen süreç açısından öngörü yeterliliği sonucudur; keyfî yeni görevlerde doğru temsilin veriden kolay öğrenileceği anlamına gelmez. [Shalizi ve Crutchfield, 2001](https://arxiv.org/abs/cond-mat/9907176).

Predictive state representations, eylem koşullu çok adımlı gelecek gözlem olasılıklarıyla durum kurar; bilinen uygun modellerde durum özyinelemeli güncellenir. İlk makalenin temsil varlığı ve modelden inşa sonucu ile bilinmeyen modelin deneyimden öğrenilmesi ayrımı özellikle önemlidir. Bizim ortalama örneğimiz, tam gelecek için yeterli bir PSR'nin eksik olduğunu söylemez; yalnız mevcut cevap veya birkaç mevcut test için yeterliliğin aynı güçte olmadığını gösterir. [Littman, Sutton ve Singh, 2001/2002](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf).

**Araştırma hipotezi H1:** Aynı anlık doğruluk ve benzer saklama maliyetinde, gelecekteki güncelleme gereksinimlerini temsil eden durum, yalnız bugünkü çıktıyı koruyan durumdan daha geniş devam ailelerinde öğrenmeyi sürdürebilir. Ortalama örneği bunun dar varlık/karşı örneğidir. Gürültülü, temsili değişen gerçek görevlerde yararının ve bedelinin ölçülmesi açıktır.

## 3. Düzeltilebilirlik, eklemeli öğrenmeden farklı bir bilgi ister

Ortalama için `(toplam,adet)` yeni örnek eklemeye yeter. Fakat kayıtlar kimlikli ve sonradan yanlış bir kaydın etkisi kaldırılacaksa aynı durum yetmeyebilir:

| Geçmiş | Saklanan `(toplam,adet)` | “A kaydını çıkar” sonrası ortalama |
|---|---|---|
| `A=0, B=2` | `(2,2)` | `2` |
| `A=1, B=1` | `(2,2)` | `1` |

Silme isteği yalnız `A` kimliğini içeriyorsa sıkıştırılmış durum iki doğru cevaptan hangisinin gerektiğini belirleyemez. İstek eski değeri de içerirse veya başka yerde kimlik-değer bilgisi tutulmuşsa engel kalkar. Aynı nedenle “öğrenilen durum küçük, dolayısıyla ham deneyim artık bütünüyle gereksiz” sonucu geçersizdir.

**Araştırma hipotezi H2:** Edinim, kullanma ve düzeltme için yeterli bilgi kümeleri bazı ailelerde farklıdır. En iyi yaşam durumu, yalnız cevap üretme maliyetine göre seçilemez; düzeltme maliyeti ve beklenen yeni ayrımlar da hesaba katılmalıdır.

Bu, her örneğin sonsuza dek saklanması önerisi değildir. Toplanabilir yeterli istatistikler, bağımlılık kayıtları, seçilmiş eski örnekler, yeniden edinim veya yaklaşık geri alma farklı maliyetlerle farklı sözleşmeleri karşılayabilir. Bunlar araştırılacak alternatiflerdir; tek bir mimariye zorunlu modüller olarak çevrilmemelidir.

## 4. Temsil değiştiğinde eski öğrenmenin taşınması

Eski temsil `φ₀(x)`, yeni temsil `φ₁(x)` olsun. Yeni temsilin eski koddan tam üretilebilmesi için bir `T` dönüşümü aranır:

\[
\phi_1(x)=T(\phi_0(x)).
\]

**Önerme 2 — Tam temsil taşıma koşulu.** Böyle bir `T`, ancak ve ancak `φ₀(x)=φ₀(x')` iken `φ₁(x)=φ₁(x')` ise vardır. İspat Önerme 1 ile aynı temsilci bağımsızlığı argümanıdır. Koşul yine hesaplanabilirlik veya düşük maliyet garantisi vermez.

Örneğin eski kod iki gözlemin yalnız ilk bitini tutuyorsa, yeni kodun iki biti de tutması eski koda uygulanan bir dönüşümle başarılamaz. İkinci bit ham kayıtta, başka bir öğrenilmiş istatistikte veya yeniden gözlemlenebilir dünyada hâlâ bulunabilir. Bütün erişilebilir kaynaklardan kaybolmuşsa “temsil yükseltmesi” adı altında geri yaratılamaz. Aynı önerme girişler yerine bütün geçmişler ve öğrenilmiş durumları için de geçerlidir.

Temsil değişimi her zaman bilgi kaybı değildir. `φ₁=Aφ₀`, `A` tersinir ve eski çıktı `w₀ᵀφ₀` ise `w₁=A^{-T}w₀` taşıması aynı çıktıyı korur. Yeni kodu eski `w₀` ile kullanmak ise başarımı bozabilir. Böylece üç ayrı olay ortaya çıkar: gerekli bilgi silinmiştir; bilgi vardır ama okuma yordamı uyumsuzdur; uyum mümkündür fakat mevcut hesap bütçesinde bulunamamaktadır.

Bu ayrım yalnız cebirsel oyuncak değildir: temsil değişimiyle eski başlığın başarım kaybını, yeni bir başlığın erişebildiği eski görev bilgisinden ayıran deneysel çalışma vardır. Bu makalenin bulguları seçilmiş görsel görevler içindir; temsil değişimi genel olarak zararsızdır sonucuna genişletilemez. [Davari ve diğerleri, 2022](https://arxiv.org/abs/2203.13381). Başka bir çalışma, temsil düzeyinde eski özellik kaybını özellikle ayırt eden protokoller ve sentetik örnekler sunar; iki makale birlikte, “unutma”nın tek bir ölçü olmadığını destekler. [Zhang, Dou ve Wu, 2022](https://arxiv.org/abs/2205.13359).

**Araştırma hipotezi H3:** Yeni temsilin tek başına yeni göreve yararı, yaşam boyu yararını öngörmez; önceki edinimlerin taşınması ve sonraki düzeltmeler için gerekli ayrımların korunması birlikte ölçülmelidir. Tam geçmişle yeniden eğitim yalnız üst referans olabilir; adayın aynı hesap ve veri hakkına sahip olması gerekir.

## 5. Araştırılmamış mekanizmaların haritası

| Açık mekanizma | Önceki düzenek neyi hazır verdi? | Ayırt edici araştırma yükümlülüğü |
|---|---|---|
| Deneyimden yeni yürütülebilir yapı edinmek | Affine sınıf, araç kimliği, doğru çıktı eşleşmesi. | Aday ayrım veya işlem, elle etiketlenmeden deneyimden oluşmalı; yeni giriş ve birleşimlere aktarımı, aynı veri ve toplam bütçeye sahip güçlü çıkarım rakibine karşı ölçülmeli. |
| Gecikmiş sonucu doğru geçmiş hesaplamaya bağlamak | Geri bildirim ilgili araca doğrudan bağlandı. | Değişken gecikme, araya giren ilgisiz olaylar ve eksik geri bildirim altında hangi kalıcı parçanın değiştirilmesi gerektiği öğrenilmeli. Rastgele gecikme tek başına neden bilgisini silmez; tam gözlem ve gizli eşleme ayrı kontroller olmalı. |
| Yeni yararlı ayrımları oluşturmak | Başlangıçta doğru fonksiyon dili ve hipotezler bilindi. | Bilgi kaybı ile mevcut okuyucu/hesap sınıfının kullanamadığı ilişki ayrılmalı; uygun yeni ayrım veya hesaplamanın nasıl edinileceği bulunmalı. Yapıyı yalnız büyütmenin genelleme sağlamadığı gösterilmeli. |
| Değişmiş temsil altında eski beceriyi sürdürmek | Bağımsız sabit kimlikler ve sabit kodlama vardı. | Bilgi kaybı, eski okuyucunun uyumsuzluğu ve sınırlı hesap nedeniyle erişilemeyen bilgi ayrı ölçülmeli. |
| Sıkıştırma ile düzeltmeyi birlikte sürdürmek | Katsayıların hangi veriden geldiği ve değişen aracın kimliği yeterince açıktı. | Ekleme için yeterli durumun geri alma, aykırı veri ve yeni ayrımlar karşısında nerede yetmediği ve hangi küçük ek bilginin işe yaradığı ölçülmeli. |
| Öğrenme yordamını deneyimle değiştirmek | Verilmiş aileler üzerinde önsel değişti. | Aynı yeni görev verisi altında gerçekten farklı güncelleme davranışı gösterilmeli; farkın yalnız eski cevapların saklanması olmadığı nedensel olarak sınanmalı. |
| Beceri birleştirme ve büyüyen bağımlılıklar | Affine işlemler birleşim altında kapalıydı. | Yeni birleşimin geçerlilik koşulları, ara hata büyümesi, taşınan varsayımlar ve bir temel beceri düzeltilince türevlerin davranışı ölçülmeli. |
| Tek bir yaşamda eşzamanlı devamlılık | Dört deney ayrı düzeneklerdi. | Edinim, koruma, temsil değişimi, düzeltme ve aktarım aynı durum üzerinde, yeniden başlatmalar ve ilgisiz uzun akışlar arasında birlikte gerçekleşmeli. |

Gecikmiş geri bildirime ilişkin bilinen mekanizma ailesi zamansal fark öğrenmesidir: ardışık tahmin farkları geçmiş kestirimleri güncelleyebilir. Bu, hazır verilen durum/özelliklerle temporal credit probleminin bazı biçimlerini çözer; gerçek nedensel sorumluluğu, yeni temsilin edinimini ve yaşam boyu korumayı tek başına çözmez. [Sutton, 1988](https://jmvidal.cse.sc.edu/library/sutton88a.pdf).

Öğrenilmiş işlem dili ve bileşim için DreamCoder doğrudan bir eşleşmedir: örneklerden program arama, tekrar kullanılabilir soyutlamalar ve arama yordamının geliştirilmesi birlikte çalışır. Bu literatür, başlangıçta sabit operatörleri yalnız birleştirmekle dili geliştirmeyi ayırmamıza yardım eder. Buradaki araştırma yükümlülüğü sürekli, gürültülü ve düzeltilen bir yaşamda bu edinimin hangi varsayımlar altında korunacağıdır. [Ellis ve diğerleri, 2020/2021](https://arxiv.org/abs/2006.08381).

Güncelleme yordamının öğrenilmesi de yeni bir fikir değildir: Andrychowicz ve diğerleri optimize ediciyi öğrenilen bir yordam olarak kurar ve benzer yapılı yeni problemlere aktarımı inceler. Bu sonuçlar, rastgele yeni dünyalarda sürekli kendi güncellemesini güvenle geliştirme garantisi değildir. [Andrychowicz ve diğerleri, 2016](https://papers.nips.cc/paper_files/paper/2016/hash/fb87582825f9d28a8d42c5e5e5e8b23d-Abstract.html).

Yordam değişimi sonsuz bir “öğrenmeyi öğrenen öğrenici” zinciri gerektirmez: `s=(θ,η)` ve `θ'=U_η(θ,z)` biçiminde güncelleme kodu/parametresi duruma dahil olabilir; sabit bir yorumlayıcı bu kodu çalıştırabilir. Esas açık soru bunu yazabilmek değil, `η` değişimini hangi kanıtın haklı çıkardığı ve hangi yeni görevlerde yararının sürdüğüdür. Meta-öğrenme sınaması bu nedenle aynı yeni görev örneklerine erişen, yalnız `η` durumu farklı iki sürümü karşılaştırmalı; güncelleme durumu aktarımı farkı taşımalı, sıfırlama kaldırmalıdır.

## 6. Genellemenin matematiksel yeri

Sonlu geçmişten görülmemiş durumlara geçiş, geçmiş örneklerin salt kopyalanmasıyla belirlenmez. Hangi olasılıkların veya programların önce deneneceği; temsil, işlem dili, simetriler, kaynak sınırı, veri geliş biçimi ve geçmiş görevlerle değişebilir. Bunlardan hangisinin seçileceğini bulmak ana araştırmanın bir parçasıdır. “Önceden hiçbir yapı olmasın” talebi burada kullanılmamalıdır: çalışan bir hesaplayıcı zaten gözlem kodu, işlem kuralları ve kaynak kısıtları taşır.

Gold'un biçimsel dil öğrenme çerçevesi, belirlenebilirliğin hem aday sınıfa hem de bilginin nasıl sunulduğuna bağlı olduğunu açık biçimde gösterir. Bu örnek, doğal dilin veya yaşayan öğrenmenin imkânsızlığı olarak kullanılmamalıdır: asimptotik tam özdeşleme ölçütü, kabul edilen veri sunumları ve hipotez sınıfı özgüldür. Bizim çıkardığımız yöntem dersi, bir başarısızlığın “öğrenme yetersizliği” sayılmadan önce hangi bilgi ve hangi genelleme sözleşmesi altında ortaya çıktığının yazılmasıdır. [Gold, 1967](https://langev.com/pdf/gold67limit.pdf).

**Araştırma hipotezi H4:** Yaşam sırasında edinilen ayrım ve işlem yapısı, yeni görevlerin öğrenme maliyetini azaltabilir; fakat bu kazanım yalnız daha iyi cevap vermekle değil, aynı yeni kanıt altında daha iyi güncellenebilmekle gösterilmelidir. Aile dışı aktarımın zararını ve hangi eski varsayımın taşındığını açıklamayan bir ortalama skor yeterli değildir.

Bir evrensel yorumlayıcıda bütün sonlu güncellemeleri en baştan ifade etmek mümkün olsa bile yaşam içi öğrenme anlamını kaybetmez. Önceden ifade edilebilir fakat mevcut bütçede bulunamayan/çalıştırılamayan hesaplamanın deneyim nedeniyle erişilebilir hale gelmesi gerçek bir kapasite değişimi olabilir. Yeni işlem dili, salt sözdizimsel genişleme değil; arama, yürütme, paylaşma ve düzeltme maliyetlerinde ölçülen bir değişim olarak sınanmalıdır.

## 7. Kontrolün uygulanabilir anlamı ve ana problem için başarı eşiği

Kontrollü değişim, geçmişte doğru bilinen her şeyin her koşulda aynen kalması değildir. Uygulanabilir sözleşme; fayda ve bozulmanın nerede ölçüldüğü, kabul toleransı, müdahale ve düzeltme olanağı, hesap/saklama maliyeti ve belirsizliğin raporlanmasıdır. Önceki bağımsız doğrulama ve birikimli hata bütçesi argümanları bu sözleşmenin bir yolu olarak korunur. Bir doğrulama filtresinin varlığı yeni yapıyı edinme algoritmasının yerini tutmaz; aday üretimiyle kabul denetimi iki ayrı bilimsel sorudur.

Bir sonraki sonuç aşağıdaki üç ölçekte raporlanmalıdır:

1. **Anlık yetenek:** Deneyimden sonra temiz bağlamda hangi yeni sorgular çözülebiliyor, hangi maliyetle?
2. **Devam yeteneği:** İlgisiz ve ilgili sonraki deneyimler, yanlış bilgi düzeltmeleri, yeni temsil ve görev birleşimleri sonrasında aynı sistem ne öğrenebiliyor ve neyi koruyor?
3. **Öğrenme değişimi:** Aynı yeni deneyimden, önceki yaşamı nedeniyle daha az veri/hesap veya daha az girişim hatasıyla yararlanabiliyor mu? Aile değiştiğinde ne kaybediyor?

Bu üç ölçeğin hiçbiri tek başına genel yaşarken öğrenmenin tamamı değildir. H1–H4 de dört zorunlu modül veya bulunmuş temel yasa değildir. Önceki kanıt erişimi ekseninden bağımsız, yanlışlanabilir araştırma yönleridir.

Şu anda savunulabilecek geniş araştırma ilkesi şudur: **Yaşarken öğrenme değerlendirilirken deneyimin gelecekteki çıktıya etkisiyle birlikte, sonraki deneyimlerden öğrenme olanaklarına etkisi de incelenmelidir.** Bu, her öğrenme olayında öğrenme yordamının, temsilin veya boyutun değişmesini gerektiren bir tanım değildir. Temsil değişimi boyut büyümesiyle aynı değildir; aynı boyutta yeni kodlama veya mevcut kod üzerinde yeni okuyucu da farklı hesaplama imkânı yaratabilir. Bu ilke hangi durumun hangi yöntemle edinileceğini tam belirlemez. Genel, kontrollü ve kaynakları hesaplanmış bir Cevahir mekanizması hâlâ gösterilmemiştir. Önümüzdeki yükümlülük, bu açık bölümlere gerçek mekanizmalar ve karşılaştırmalı kanıt üretmek; tek bir alt problemdeki başarıyı bütün problemin cevabı saymamaktır.

Bu turda ilgili üç kol fiilen çalıştırıldı: [temsil ve işlem bileşimi](REPRESENTATION_GROWTH.md), [gecikmiş geri bildirim ve ortak parametrelerde koruma](CREDIT_ASSIGNMENT.md), [durum taşıma ve yeni temsil için eksik istatistik](STATE_TRANSPORT.md). İlk iki kol da hazır dil/geri bildirim varsayımlarını bütünüyle kaldırmadı. Üçüncü kol öğrenicinin yararlı dönüşümü kendisinin bulduğunu göstermedi. Böylece haritadaki açık alanlar deney eklenince kapanmış sayılmıyor.
