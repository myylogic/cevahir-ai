# Cevahir — Yaşarken öğrenmenin bütün problemine dönüş

20 Eylül 2026 · Önceki araştırmaya ek kayıt · Üç yeni deney ailesi · Genel problem açık

**Ana problem çözülmedi.** Önceki araştırma, öğrenilmiş davranışın karşı kanıta erişimi kapatması ve yeniden sınama üzerinde yoğunlaştı. Bu sonuçlar geçerlilik koşullarıyla korunuyor; yaşarken öğrenmenin tamamının cevabı sayılmıyor. Bu turda araştırma deneyimden hesaplama edinimi, ortak yapıda öğrenme ve koruma, temsil değişimi ve gelecekte güncellenebilirlik yönlerinde ilerletildi.

Kullanıcının düzeltmesi araştırmanın hedefini yeniden açık hale getirir: **Bir sistemin çalışma sırasında yaşadığı deneyimler, gelecekteki hesaplama ve davranış kapasitesini nasıl kalıcı, kontrollü ve genellenebilir biçimde değiştirebilir?** Kullanıcı yeni bir mimari veya hipotez önermedi. Buradaki sorular, karşı örnekler ve deneyler araştırma tarafından üretildi. Ana sisteme entegrasyon yapılmadı; GitHub'a gönderim yapılmadı.

## 1. Önceki çalışma nasıl korundu, ne düzeltildi?

Önceki rapor, matematik, dört deneyin kodları/sonuçları, literatür kaydı, kullanıcının özgün metni ve gözlemleri aynı dosyalarda kaldı. Daha eski tanık filtresi ve deneyime bağlı hesaplama araştırması da koruma kapsamına alındı. Bu tur başında **30 dosyanın** boyutları ve SHA-256 özetleri kaydedildi; tur sonunda değişmedikleri doğrulandı. [Koruma dökümü](../../../research/living_learning_reassessment/results/prior_preservation_manifest.json), [doğrulama kaydı](../../../research/living_learning_reassessment/results/verification.json).

Eski [ana rapor](../living_learning_2026_09_20/REPORT_TR.md), [temeller](../living_learning_2026_09_20/FOUNDATIONS.md), [araştırma günlüğü](../living_learning_2026_09_20/RESEARCH_LOG.md) ve [literatür eşleşmeleri](../living_learning_2026_09_20/PHENOMENA_AND_LITERATURE.md) tarihsel kayıttır. Yeni kapsam değerlendirmesi bunların üzerini yazmaz.

**Düzeltilen yorum:** “Kontrollü, kalıcı, genellenebilir” talebi; her dünyada sıfır hata, sonsuz belleksiz öğrenme veya eski her cevabı değişmez tutma talebi değildir. Eski imkânsızlık örnekleri bu aşırı garantilerin sınırlarını gösterir. Kullanıcının gerçek problemine cevap olarak kullanılamaz. Gerçekçi tolerans, kaynak ve dünya ailelerini seçip araştırmak bizim sorumluluğumuzdur.

**Korunan genel ayrımlar:** Deneyimin etkisinin gelecekteki hesaplamaya nedensel olarak taşınması; bilgi içeriği ile o bilgiyi belli hesap bütçesinde kullanabilmenin ayrılığı; görülmemiş görevlere ilişkin ölçüm; bilgi kaybıyla okuyucu/hesap sınıfı yetersizliğinin ayrılığı; varsayım ve kaynakların açık tutulması.

**Aileye bağlı kalan sonuçlar:** Affine kuralların belirlenmesi/bileşimi; verilmiş görev aileleri üzerinde önsel öğrenme; doğru kimlikle bağımsız kuralları düzeltme; tek bilgilendirici eylemle kalıcı değişimi saptama ve bunun maliyeti. Bunlar dar yapıcı örnekler veya koşullu sınırlardır. [Ayrıntılı sınıflandırma ve araştırma haritası](SCOPE_AND_FOUNDATIONS.md).

## 2. Yeni deneyler ve kendi karşı örnekleri

| Araştırılan yön | Elde edilen sonuç | Sonucun sınırını gösteren kontrol |
|---|---|---|
| Deneyimle yeni yürütülen özellik/bileşim edinimi | 20 yaşamda öğrenilmiş çarpım düğümleri ve bunların birleşimleriyle 6.000/6.000 ayrılmış girdi çözüldü. Başlangıçtaki altı özellik dokuz oldu. | Aynı etkinleştirme politikasına sahip sabit geniş alternatif bütün başarı ve hesap sayaçlarını eşitledi. Gerekli öncül görev kaldırılınca büyüyen yöntem 0/2.000, geniş alternatif 2.000/2.000. |
| Gecikmiş geri bildirim ve ortak yapıda beceri koruma | 32 yaşamda doğru deneyime bağlama yeni beceriyi öğretti. Küçük replay veya ham kayıt tutmayan yeterli istatistik eski ve yeni becerinin birlikte sürmesini sağladı. | Doğru bağlama tek başına eski beceriyi korumadı. Hedef gerçekten değişince geçmişi koruyan yöntemler güncel hedefte kötüleşti. |
| Temsil değişiminde devam eden öğrenme | İki rejimde 32'şer yaşam: ağırlık ve güncelleme matrisi birlikte taşındığında bütün gelecek tahmin yolları `2.22×10⁻¹⁵` sayısal tolerans içinde aynı kaldı. | Yalnız ağırlık taşımak bugünkü cevapları aynı tuttu ama sonraki öğrenmeyi değiştirdi. Bu, seçilmiş dönüşüm için cebirsel sonuç; genel optimize edici üstünlüğü değil. |
| Eski özetten yeni temsil oluşturma | İki farklı geçmişin eski tam yeterli istatistikte birleştiği, yeni özellik için ise farklı istatistik gerektirdiği tam karşı örnek kuruldu. | Yeni özellik eklemek geçmişte silinmiş ayrımı geri yaratamadı. Ham kayıt veya uygun ek moment bu belirli açığı kapattı; her kaydı sonsuza dek tutmak gerektiği sonucu çıkmadı. |

Son iki satır aynı deney ailesinin sayısal ve tam cebirsel kısımlarıdır. Başarı sayıları farklı düzenekler arasında ortak bir genel başarı oranında toplanamaz.

[Temsil oluşturma deneyi](REPRESENTATION_GROWTH.md), [gecikmiş deneyim ve koruma deneyi](CREDIT_ASSIGNMENT.md), [durum taşıma deneyi](STATE_TRANSPORT.md).

Ortak yapı deneyinin önemli nicel ayrımı şudur: üç katsayılı tek modelde yeni girdilerden doğru güncelleme yapmak, hâlâ geçerli eski girdilerdeki ortalama kare hatayı `.000905→.216505` artırdı. Hedef değişmedi; hem eskiyi hem yeniyi doğru yapan tek bir üç-katsayılı çözüm var. Aynı iki güncellemeden birini küçük geçmiş örneklemine ayırmak eski hatayı `.000957` düzeyinde tuttu. Tam istatistik biriktiren online RLS `.000070` elde etti. Dolayısıyla “daha fazla güncelleme” ve “ham deneyimi tekrar oynatma zorunluluğu” açıklamaları yeterli değil.

Bu kazanım bedelsiz değildir. Replay yeni dağılımın ilk dönemindeki toplam servis hatasını artırdı. Gerçek eğim değişiminde güncel MSE, geçmişi hızlı bırakan güncellemede `.001106`, replay'de `.181190`, unutmasız RLS'de `.509150` oldu. Geçerli geçmişi korumak ile geçersiz geçmişi sürdürmek ayrı davranışlardır; bu deney bunları kendiliğinden ayırt eden bir yöntem bulmadı.

Temsil oluşturma deneyinde ayrı görev kimlikleri, hazır çarpma dili ve sayısal doğru etiketler verildi. Ortak yapı deneyinde görev sınırı öğreniciye verilmedi, ama gecikmiş etiketin doğru deneyim kimliği verildi. Bu iki düzenekteki kolaylaştırmalar birbirini tamamlayan genel bir çözüm sayılmaz. Ayrıca bir yanlış etiket, oluşturulmuş doğru başlığı hemen geçersiz kıldı; gürültüye dayanıklılık iddiası yoktur.

## 3. Matematikte ilerleyen nokta

**Bugün cevap vermeye yeterli durum ile yarın doğru öğrenmeye yeterli durum aynı şey olmayabilir.** En küçük örnek: geçmişler `(0,2)` ve `(1)` aynı ortalamaya sahiptir. Yalnız `1` saklanırsa yeni `4` geldiğinde doğru yeni ortalamanın `2` mi `2.5` mi olacağı bilinemez. Toplam ve adet tutmak bu sorunu çözer. Ama daha sonra “A kaydının etkisini çıkar” istenirse toplam ve adet, A'nın değerini saklamadığı için yetersiz olabilir.

Genel ifade: geçmişi `c(h)` durumuna dönüştüren bir özetin tam çevrimiçi güncellenebilmesi için

\[
c(h)=c(h')\Rightarrow c(hz)=c(h'z)
\]

her ortak izinli yeni deneyim `z` için sağlanmalıdır. Koşul gerekli ve yeterlidir: aynı mevcut durum ve aynı deneyim tek bir sonraki duruma gitmelidir; ters yönde bu özellik, temsilciden bağımsız bir güncelleme tanımlamayı sağlar. Bu bir **işlevsel varlık önermesi**dir. Verimli veya öğrenilebilir bir güncelleme algoritması vermez.

Benzer şekilde, eski özetten yeni özete kayıpsız bir taşıma `T(c₀(h))=c₁(h)` ancak eski özetin birleştirdiği her geçmiş çiftini yeni özet de birleştiriyorsa mümkündür. Parity karşı örneğinde bu koşul ihlal edilir. Ağırlık ve güncelleme matrisi deneyinde ise bilgi vardır; doğru taşıma bulunarak eşitlik sağlanır. Tam ispatlar ve koşullar [kapsam belgesinde](SCOPE_AND_FOUNDATIONS.md) ve [durum taşıma kaydında](STATE_TRANSPORT.md).

Bu önermeler yeni bir evrensel öğrenme yasası değildir. Gelecek davranış üzerinden durum yeterliliği, causal states ve predictive state representations ile doğrudan ilişkilidir. [Shalizi ve Crutchfield](https://arxiv.org/abs/cond-mat/9907176), [Littman, Sutton ve Singh](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). Online öğrenmede yeterli istatistiklerden belirli regret garantilerine ulaşan yöntemler de vardır; bu, koşulları verildiğinde salt betimleme ötesinde algoritmik sonuç alınabildiğini gösterir. Burada Burkholder yöntemi uygulanmadı ve onun garantileri yeni deneylere taşınmadı. [Foster, Rakhlin ve Sridharan, 2018](https://proceedings.mlr.press/v75/foster18b.html).

Araştırmaya eklenen ölçüm önerisi, yalnız son riski `R_Q(s)` incelememektir. Aynı yeni deneyim akışını alan durumlar için, belirlenmiş `k` örnek ve güncelleme bütçesinden sonra

\[
A_{\mathcal P,k,B,C}(s)=\mathbb E_{f,E_k}\big[R_{Q_f,B}(U^k_C(s,E_k);f)\big]
\]

ölçülebilir. `P` seçilmiş gelecek görev ailesi, `Q_f` o görevde ayrılmış sorgular, `B` cevaplama ve `C` toplam güncelleme bütçesidir. Bu tanısal karşılaştırmada gelecek kanıt eşleştirilir; eylemin veriyi değiştirdiği serbest etkileşim ayrı sınanır. İlk riskler eşitken `A` farklı olabilir. Sabit bir `U` altında da bu fark oluşabilir; her öğrenme olayında güncelleme kodunun kendisinin değişmesi gerekmez. İfade yeni amaç fonksiyonu dayatmaz, öğrenebilirliği ölçmek için bir deney sözleşmesi önerir. Bu tur geniş bir `P` üzerinde böyle bir meta-deney yapılmadı.

## 4. Yeni hipotezlerin statüsü

| İddia | Bu turdaki durum |
|---|---|
| Mevcut tahminler korunursa sonraki öğrenme de korunur. | Tersinir koordinat değişimiyle **yanlışlandı**. |
| Sabit modelde tam yeterli özet her yeni temsile tam taşınır. | İki geçmişin çakışmasıyla **yanlışlandı**. |
| Deneyimin doğru yere bağlanması eski geçerli beceriyi korumaya yeter. | Ortak üç-katsayılı modelde **yanlışlandı**. |
| Ham örnekleri tekrar işlemek korumanın zorunlu mekanizmasıdır. | Sabit ailede online yeterli istatistik kontrolü bu zorunluluğu **yanlışladı**. |
| Temsilin büyümesi eşit ifade güçlü sabit yapıya kendiliğinden üstünlük sağlar. | Güçlü kontrol eşitliği ve müfredat karşı örneğiyle **desteklenmedi; genel biçimi yanlış**. |
| Bir özellik bugün işe yaramıyorsa hiçbir gelecek beceri için değer taşımaz. | Öncül ikili beceri olmadan üçlüye ulaşamama, mevcut aday aramasında bu varsayımın tehlikesini gösterdi; verimli genel çözüm açık. |
| Gelecekte edinim ve düzeltme ihtiyaçlarını gözeten durum, yalnız bugünkü cevabı saklamaktan daha geniş devam ailelerine hizmet edebilir. | **Dar karşı örnekler ve yapıcı durumlarla destekli araştırma hipotezi**; açık dünyada hangi bilginin tutulacağını seçen çözüm yok. |

Bu sonuçlar ne sinir ağlarını ne gradyanı ilke gereği dışlar. Sabit mimari içinde öğrenilen ağırlıklar yeni ara özellikler oluşturabilir; bunun klasik birincil örneği [Rumelhart, Hinton ve Williams, 1986](https://doi.org/10.1038/323533a0). Güncel bir ağda plastisitenin zamanla kaybı ise ayrı bir deneysel problem olarak araştırılmıştır; mevcut beceri ve yeni beceri edinme kapasitesi tek metrikle değerlendirilmemelidir. [Dohare ve diğerleri, 2024](https://www.nature.com/articles/s41586-024-07711-7). Bu kaynaklar yeniden üretilmedi ve yöntemleri zorunlu mimari parçalara çevrilmedi.

## 5. Ana problemde ne açık kaldı?

Araştırma artık yalnız hangi eylemle yeni kanıt alınacağını sormuyor. Aşağıdaki açıklar, şimdiye kadarki deneylerden bağımsız önem taşır:

1. **Gözlemden yararlı değişken ve ilişki edinimi:** Hazır doğru etiket, görev kimliği ve sınırlı işlem dili olmadan hangi ayrımlar öğrenilecek? Yeni temsil adayı nereden gelecek; daha fazla arama ile daha iyi dil nasıl karşılaştırılacak?
2. **Deneyimin değişime atanması:** Uzak bir sonucun hangi eski hesaplamaya, eyleme veya varsayıma bağlanacağı nasıl bulunacak? Gecikmiş doğru ID bu sorunu çözmüş sayılmaz.
3. **Düzeltilebilir sıkıştırma:** Bugün gereksiz görünen hangi deneyim ayrımı yarın önemli olur? Sonlu bellekte edinim, yeniden kullanım, geri alma ve yeni temsil ihtiyacı nasıl dengelenir? Tek bir görevde yeterli özet bütün yaşamın özeti değildir.
4. **Koruma ile yenilenmenin aynı sistemde seçilmesi:** Yeni fark gerçek dünya değişimi mi, gürültü mü, yanlış temsil mi? Yeni öğrenmenin eski geçerli beceriyi bozduğu ile eski artık yanlış kuralın bırakıldığı nasıl ayrılacak?
5. **Öğrenme yordamının deneyimle gelişmesi:** Önceki yaşamdan öğrenilen yapı aynı yeni kanıtı daha az maliyetle kullanılabilir kılabilir mi? Kazanım, taşınan eski cevaplarla veya seçilmiş kolay aileyle açıklanıyor mu?
6. **Bir yaşam boyunca birlikte gerçekleşme:** Edinim, bileşim, düzeltme, sıkıştırma, yeniden düzenleme ve sonraki öğrenme kapasitesi aynı öğrenilmiş durum üzerinde, uzun ilgisiz akışlar ve sınırlı kaynaklar arasında birlikte sınanmadı. Ayrı düzeneklerin başarıları böyle bir ortak sistemin kanıtı değildir.

Bu bir modül veya mimari listesi değildir. Bunlar farklı araştırma yöntemlerinin karşılaması gereken sorulardır. Sonraki çalışma bunların yalnız birini ana problem ilan etmemeli. Özellikle yalnız yeterli istatistik veya yalnız temsil büyümesi etrafında yeni bir daralma, önceki yeniden sınama daralmasını başka isimle tekrar eder.

Mevcut çalışma, yaşarken öğrenmenin bazı biçimlerinin hesaplanabilir ve fiilen yürütülebilir olduğunu gösteren örnekler sağlamıştır. Fakat bilinmeyen yararlı yapıyı deneyimden edinip, gerektiğinde düzeltip, gelecek öğrenme kapasitesini sınırlı kaynaklarla sürdüren genel mekanizma henüz bulunmuş değildir. Araştırma bu açıkla devam eder; bu tur bir çözüm ilanı değildir.

## 6. Doğrulama ve devam noktası

Üç yeni deney Python standart kütüphanesiyle çalıştırıldı. Kaynak özetleri ve ayrı çıktılara tekrar üretimler kontrol edildi. Kredi deneyinin bağımsız yeniden çalıştırması bütün JSON'u birebir üretti; diğer deneylerde zaman/bellek gibi ortam ölçümleri dışındaki bilimsel içerik karşılaştırılır. Bu kontroller sonuçların doğrulanabilirliğini sağlar; genellik veya özgünlüğü kanıtlamaz. [Doğrulama yordamı](../../../research/living_learning_reassessment/verify_reassessment.py).

Devam ederken başlangıç kayıtları: bu rapor, [tam kapsam haritası](SCOPE_AND_FOUNDATIONS.md), üç deney raporu ve önceki araştırma. Korunmuş negatifler yeni hipotezler için karşılaştırma olarak kalmalıdır. En belirgin deneysel açık, hazır görev kimlikleri ve ayrı deney düzenekleriyle kolaylaştırılmış başarıların aynı bellek ve hesap bütçesini paylaşan bir yaşamda birlikte sürdürülmesidir; bunun mümkün olmadığı veya yalnız bir mevcut öğrenme kavramıyla çözüleceği peşinen kabul edilmiyor.
