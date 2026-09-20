# Deneyimden yinelemeli durum çıkarımı: literatür ve kapsam denetimi

Tarih: 2026-09-20. Bu belge, önceki araştırmayı değiştirmeden yeni deneyin kavramsal ve literatür denetimini yapar. Deney sonuçları içermez. Aşağıdaki öneriler başarılı oldukları varsayılarak yazılmamıştır.

## 1. Bu tur hangi boşluğu doldurabilir?

Önceki zamansal deney, araştırmacının sağladığı sonlu gecikmeler arasındaki ilişkileri öğrendi. Yeni deneyde sorgulanan şey, bir dizinin hangi geçmişlerini aynı hesaplama durumunda birleştirmek, hangilerini ayırmak gerektiğinin terminal sonuçlardan öğrenilmesidir. Böylece yalnız bir tahmin katsayısı değil, sonraki sembollerin nasıl işleneceği de değişir.

Bu, genel yaşayan öğrenmenin çözümü değildir. İkili gözlem alfabesi, bölüm başlangıçları, terminal etiket anlamı, deterministik durum geçişi ve durum birleştirme işlemi hâlâ araştırmacı tarafından sağlanır. Kazanılabilecek daha dar sonuç: deneyimin, daha önce ayrı ayrı görülmemiş uzun geçmişlerde tekrar kullanılabilen bir hesaplama devresini belirleyebilmesi.

## 2. Önceden bilinen sonuçlarla doğrudan eşleşmeler

### Gold: öğrenme, bilgi sunum biçimine bağlıdır

Gold'un limitte tanımlama çerçevesi, bir hipotezin sınırsız veri akışında sonunda doğru temsilde sabitlenmesini inceler. Pozitif örneklerden oluşan metin ile hem üyelik hem üyelik-dışı bilgiyi sunan informant farklı koşullardır. Bütün sonlu dilleri ve en az bir sonsuz dili içeren bir sınıf, keyfî tam pozitif metinlerden bu anlamda öğrenilemez. Bu, ikili doğru terminal etiketleri alan mevcut deneye doğrudan uygulanmış bir imkânsızlık teoremi değildir. Ayrıca sonunda doğru hipotezde sabitlenmek, öğrenicinin o anda doğru olduğunu bilmesi veya sonlu zamanda sertifika üretmesi demek değildir. [Gold, 1967, özgün makale](https://homepages.math.uic.edu/~lreyzin/papers/gold67.pdf).

### Angluin L*: bilgi erişim hakları bir algoritma ayrıntısı değildir

L*, seçilmiş sözcükler için doğru üyelik cevabı ve bütün dilin eşdeğerliğini sınayıp yanlış hipoteze karşı örnek veren bir öğretmen kullanır. Bu erişimle minimal DFA'nın durum sayısı ve en uzun karşı örneğe göre polinom zamanda öğrenme sonucu vardır. Makale rastgele örneklemeyle yaklaşık eşdeğerlik değerlendirmesini de inceler. Pasif terminal etiket akışında aynı haklar verilmiş olmaz. Özellikle değerlendiricinin gizli DFA'ya erişip eşdeğerlik ölçmesi, bu bilgiyi öğreniciye geri vermediği sürece L* öğretmeni değildir. [Angluin, 1987, §1–2](https://homepages.math.uic.edu/~lreyzin/papers/angluin87.pdf).

### RPNI: yeni keşfedilmiş bir yapı öğrenme ilkesi değil

RPNI, pozitif örneklerin ön ek ağacındaki durumları, negatif örneklerle çelişmeyen biçimde birleştiren pasif düzenli dil çıkarımıdır. Yapı, doğru durum sayısı doğrudan verilmeden öğrenilebilir. Yeterli karakteristik örnek kümesini içeren temiz veride doğru kanonik otomata ulaşma sonucu, her rastgele küçük örnek kümesinde doğruluk garantisi değildir. Rastgele bir örnek kümesine uyan en küçük DFA'yı bulma problemiyle de aynı optimizasyon problemi değildir: RPNI her örnek kümesi için küresel minimumu bulduğunu iddia etmez. Dolayısıyla bu algoritmayla olumlu sonuç, durum birleştirme ve düzenli çıkarımın yeniden gösterimidir. [Oncina ve García, 1992, özgün kurum kopyası](https://grfia.dlsi.ua.es/repositori/grfia/pubs/77/inferring.pdf).

### Hesaplamalı mekanik: geçmişleri gelecekteki sonuçları bakımından birleştirmek

Shalizi ve Crutchfield, aynı koşullu gelecek dağılımına sahip geçmişleri eşdeğer kabul eden nedensel durum temsillerini inceler. Çalışma, doğru öngörü için minimal yeterli temsille ilgili sonuçlar verir. Buradaki “nedensel” adının, deneyimizde müdahale etkilerinin veya organizmanın iç biyolojik nedenlerinin tanımlandığı anlamına gelmesine izin verilmemelidir. Koşullu gözlemsel geleceği eşlemek ile dışarıdan değiştirilen bir eylemin etkisini tanımlamak farklı iddialardır. Mevcut terminal-sınıflama deneyinin çıkardığı durumlar, bütün çevresel gelecek için yeterlilik iddiası taşımaz. [Shalizi ve Crutchfield, 2001](https://csc.ucdavis.edu/~cmg/papers/cmppss.pdf).

### CSSR: öğrenilmiş yinelemeli durum, sonlu son-ek belleğinden farklıdır

CSSR, son-eklerin sonraki sembol dağılımlarını karşılaştırır, gerekirse durumları ayırır ve yinelemeli geçişlerin tutarlılığını kurar. Yakınsama için koşullu durağanlık, sonlu nedensel durum kümesi ve her duruma sonlu bir eşzamanlama son-ekiyle ulaşılabilmesi varsayılır. Her sonlu gizli durum süreci otomatik olarak bu koşulları sağlamaz. Makalenin Even Process örneği, sonlu yinelemeli durumla temsil edilip herhangi bir sabit son-ek uzunluğuyla tam temsil edilemeyen bağımlılığın zaten bilindiğini gösterir. Bu, planlanan parite tanığına çok yakın bir öncüldür; terminal etiketli bölüm öğrenimiyle yöntemsel olarak aynı deney değildir. [Shalizi ve Shalizi, 2004, §3–5](https://arxiv.org/pdf/cs/0406011).

### Değişken uzunluklu bellek ve gürültü

Olasılıksal son-ek otomatalarını öğrenme, ayrı ve köklü bir model ailesidir. Öğrenilmiş son-ek uzunluğunu sabit bir pencereye karşı güçlü bir karşılaştırma olarak eklemek değerlidir; fakat bu aile bütün sonlu yinelemeli otomatalarla aynı değildir. [Ron, Singer ve Tishby, 1996](https://cs.brown.edu/courses/csci2840/spring-2025/resources/fp_ideas/power_of_amnesia.pdf).

Ye ve arkadaşları, L*'ın PAC sürümünü çeşitli bozulmuş aygıtlarda inceleyerek rastgele ve yapılandırılmış bozulmalar altında farklı sonuçlar bulur. Bu çalışma RPNI'ya genel gürültü dayanıklılığı vermez. Bizim tekrarlı aynı sözcük için karşıt etiket denetimimiz ile her sözcüğün etiketinin bir kez rastgele değiştirilip daha sonra tutarlı kalması da farklı gürültü modelleridir. Deney hangisini kullandığını söylemelidir. [Ye ve arkadaşları, LMCS 2024](https://lmcs.episciences.org/13257/pdf).

## 3. Deney için matematiksel ayrımlar

Aşağıdakiler deney denetimi için burada doğrudan türetilen sonuçlardır; yeni teorem oldukları iddia edilmez.

### Geçerli sıkıştırma, yalnız mevcut etiketi korumaz

Bir deterministik terminal hedefi `f` ve geçmişler `h, h'` için ilgili eşdeğerlik:

`h ~ h'  ancak ve ancak  her devam u için f(hu) = f(h'u)`.

İki geçmişin şimdiki terminal etiketi aynı olabilir, ancak bir sonraki sembolden sonra ayrılmaları gerekebilir. Bu yüzden `f(h)=f(h')` tek başına onları aynı yinelemeli durumda birleştirmeyi haklı çıkarmaz. Bir durum kodu `q` ve sabit geçiş `δ` kullanılıyorsa:

`q(h)=q(h') => q(ha)=q(h'a)`

her sembol için zorunludur. Bu devam-tutarlılığı, yeni grafiğin temel hesaplama şartıdır; genel dünyada hedefin durağan olduğunu göstermez.

### Parite tanığı: uzun geçmiş mutlaka büyük anlık bellek değildir

Uzunluğu `n > k` olan eş olasılıklı ikili dizilerde hedef bütün dizideki birlerin paritesi olsun. Son `k` bit verilince görülmeyen ön ekin paritesi hâlâ eş olasılıklıdır. Yalnız son `k` bite ve toplam uzunluğa dayanan her tahmincinin beklenen 0–1 hatası `1/2` olur. İki durumlu bir yineleme ise `q <- q XOR bit` ile bütün uzunluklarda sıfır hata verir.

Bu, sonlu pencereyle yinelemeli hesap arasındaki ifade gücü farkıdır. Parite devresini öğrenicinin kendisinin bulduğunu, verinin bunu tanımlamaya yettiğini veya bütün yinelemeli modellerin kolay öğrenildiğini tek başına kanıtlamaz.

### Sabit küçük DFA sınıfını tüketmek: güçlü ama ayrıcalıklı karşılaştırma

İkili alfabe, sabit başlangıç durumu ve etiketlenmiş `n` durum için `n^(2n) · 2^n` DFA betimi vardır. `n <= 3` için toplam `2 + 64 + 5832 = 5898` betim bulunur. Bunların arasında erişilemeyen durumları olan ve aynı dili farklı adlarla anlatan kopyalar vardır.

Bu sınıfı tüketip örneklerle çelişen adayları elemek uygulanabilir bir kontrol sağlar. Ancak üç durum sınırı doğruysa öğreniciye faydalı ön bilgi verilmiştir. Bir adayın seçilmesi, aday üretme mekanizmasının deneyimden doğduğu anlamına gelmez. Tüm adayları paralel izlemek, tek seçilmiş grafiği yürütmekten çok daha fazla bellek ve işlem de gerektirebilir.

### Tam eşdeğerlik, uzun örneklerde başarıdan daha güçlüdür

`n` ve `m` durumlu iki tam DFA'nın çarpım grafiği en fazla `nm` düğümlüdür. Çıktıları farklı olan erişilebilir bir çift varsa, en kısa yol düğüm tekrarı yapmaz; uzunluğu en fazla `nm-1` olur. Çarpım grafiğini taramak tam eşdeğerliği veya somut ayırıcı sözcüğü verir.

Değerlendirici bu işlemi gizli hedefle yapabilir. Sonucu, karşı örneği veya hedef durum sayısını öğreniciye geri vermemelidir. Böyle bir değerlendirme “test edilen uzunluklarda iyi” ile “bu belirli hedefle bütün sözcüklerde aynı” sonuçlarını ayırır.

### Sonlu kanıtın dışında görünmez alternatif

Her sonlu eğitim kümesi `D` ve düzenli dil `L` için, `D` içinde bulunmayan daha uzun bir `w*` seçilsin. `L' = L △ {w*}` yine düzenlidir ve eğitim kümesinde `L` ile aynıdır. Bu nedenle sınırsız düzenli dil ailesinde yalnız `D`'ye bakarak hangisinin gerçek olduğu belirlenemez. Doğru ve sabit küçük durum sınırı bu alternatifin bir kısmını dışlayabilir; sonuç sınırdan bağımsız değildir.

Bu karşı örnek, uzun dizilerde gerçek genellemenin mümkün olmadığını söylemez. Sonlu kanıtla koşulsuz, bütün olası hedeflere geçerli kesinlik iddiasını sınırlar.

### Gözlem çakışmasını durum büyüterek çözemezsin

Aynı başlangıçtan aynı görünür sözcük iki farklı zorunlu terminal etiketle eşlenirse, sözcüğün deterministik fonksiyonu olan hiçbir DFA ikisini de sağlayamaz. Dengesiz tekrar sayıları `n0,n1` için o sözcükte en az `min(n0,n1)` eğitim hatası gerekir. Daha çok durum, aynı görünür girişe gizli bilgi eklemez.

Buradan çevrenin bozuk olduğu sonucu çıkmaz: eksik gözlem, stokastik çıktı, hedef değişimi veya hatalı geri bildirim olasılıkları ayrılmamış olabilir. Deneyde bunlar ayrı negatif koşullar olarak tutulmalıdır.

## 4. Seçilen küçük deney için gerekli kontroller

1. **Bilgi erişimi:** Öğrenici yalnız sembolleri, açık bölüm sınırlarını ve bölüm sonundaki etiketi görmeli. Ara durum etiketleri, gerçek durum sayısı, seçici öğretmen karşı örnekleri veya test etiketleri güncellemeye girmemeli. Bağımsız değerlendirici daha fazla bilgi kullanabilir.
2. **Canlı hizmet:** Tahmin yeni etiket gelmeden kaydedilmeli. Yeni model geçmiş terminal etiketlerle kurulmalı. Veri akışından sonra toplu öğrenim ve son test tek başına çalışma boyunca öğrenmeyi göstermeye yetmez.
3. **Güçlü karşılaştırmalar:** Tam sözcük tablosu, aynı etiketleri alan sabit/öğrenilmiş son-ek modeli ve `<=3` durumlu sınıfın kesin aday elemesi kullanılmalı. İkincisinin ifade sınırı ve üçüncüsünün doğru sınıf ön bilgisi açık olmalı.
4. **Çalıştırma ayrımı:** Öğrenilmiş grafiği ham eğitim sözcükleri olmadan, yeni sembolleri birer birer alarak çalıştırmak gerekir. Bu, uzun örnek başarısının arka plandaki sözcük aramasından gelmesini dışlar.
5. **Kalıcı etki:** Grafiği silme ve farklı süreçte yalnız grafiği yükleme karşılaştırılmalı. Bölüm ortası devamlılık iddiası varsa anlık durum da doğru grafik sürümüyle birlikte saklanmalı.
6. **Grafik değişince durum taşıma:** Eski grafikteki sayısal durum kimliği yeni grafikte aynı anlama gelmez. Güncelleme yalnız bölüm sonunda yapılırsa bu açıkça belirtilmeli. Bölüm ortasında güncelleme isteniyorsa ön ek tekrarı veya doğrulanmış durum taşıma gerekir.
7. **Genelleme:** Eğitimden çok uzun diziler ayrı tutulmalı; rastgele uzun test doğruluğuna ek olarak gizli hedefle çarpım-grafik eşdeğerliği raporlanmalı. Test başarısına göre eğitim dağılımı veya birleştirme sırası seçilmemeli.
8. **İşlem ve bellek:** Ham arşiv boyutu, ön ek ağacı boyutu, birleştirme denemeleri, tutarlılık taramaları ve hizmet sırasında geçiş sayıları ayrı ölçülmeli. Küçük son grafik, ucuz öğrenme veya sınırlı ömürlük bellek demek değildir.
9. **Negatifler:** Üç durum sınırı dışındaki temiz hedef; aynı görünür sözcüğe çelişkili etiket; temiz hedefin rastgele/yapılandırılmış etiket bozulması; kısa veride ayırt edilemeyen uzun istisna ayrı sonuçlar üretmeli.

## 5. Genel living-learning problemi için çıkarılabilecek ve çıkarılamayacak şey

Olumlu bir sonuç, deneyimden çıkarılan durumların gelecekte hangi geçmiş ayrımlarının taşındığını değiştirerek yeni hesaplamaları mümkün kılabildiğini gösterebilir. Bu bulgu, önceki kalıcı parametre, temsil, bellek taşıma ve öğrenilmiş güncelleme sonuçlarını tamamlar.

Ancak bu deney hâlâ bölüm sonu doğru denetim, sağlanan sembolik dil ve yeniden eğitilebilir bir örnek arşivi kullanır. Algının kendisini edinme, hangi sonuçların amaç bakımından anlamlı olduğunu öğrenme, belirsiz geri bildirimi kendi eylemine atfetme, kesintisiz durum geçişinde eski yeteneği koruma ve sınırlı kaynakla yeni hesaplama türlerini oluşturma açık kalır. RPNI veya aday eleme başarısı bunların cevabı yerine geçirilmemelidir.

Kaynak denetimi: Yukarıda listelenen özgün makaleler ve kurum/yazar kopyaları 2026-09-20 tarihinde web aracılığıyla kontrol edildi. RPNI PDF'si görüntü tabanlıdır; arama aracının aynı özgün PDF için çıkardığı metin de incelendi. Literatür sonuçları deneyin henüz elde edilmemiş sonuçları olarak sunulmadı.
