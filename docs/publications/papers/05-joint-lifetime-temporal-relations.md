# Tek yaşam akışında temsil, bellek ve zamansal ilişkiler

**Muhammed Yasin Yılmaz**

CEV-2026-05 · Sürüm 2026.09.21.2 · Yayın tarihi 2026-09-21

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

Temsil genişletme, bölgesel bellek ve değişen hedefler tek yaşam akışında birleştirilir. Yirmi dört yaşamda incelenen yöntem sabit geniş temsilin doğruluğuyla eşleşirken daha az yeniden çözüm yapar; bu genel maliyet üstünlüğü değildir. Hatalı etiket patlaması ve eski istatistiklerin korunması olumsuz kontrollerdir. Ayrı zamansal görevde hazır gecikme özellikleri uygun dağılımda başarılı, ayırt etmeyen akıştan yeni dağılıma geçişte başarısızdır.

## Abstract

A joint stream combines representation expansion, regional memory and target change across 24 lives. A fixed expanded baseline matches final accuracy; fewer solves do not establish total efficiency. Label corruption and stale statistics remain negative controls. Temporal prediction with supplied lag features succeeds on informative input but fails after an uninformative training stream shifts.

## Bulguların yorumu

Bileşenlerin ayrı ayrı çalışması tek bir yaşamın bütün koşullarında güvenilir öğrenme garantisi sağlamaz; veri akışının ayırt ediciliği belirleyicidir.

Hazır gecikmelerde iyi tahmin, zamansal ilişkinin veya nedensel yapının genel keşfi değildir. Sonradan hatalı veri patlamasının çıkarıldığı koşul ayrı tutulmuştur.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/living_learning_joint_2026_09_20/REPORT_TR.md](../../research/living_learning_joint_2026_09_20/REPORT_TR.md) olarak korunur. Bağlantı yolları bu
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

## Araştırma kaydı: Cevahir — Ortak bir yaşamda öğrenme ve hazır kimliklerin kaldırılması

20 Eylül 2026 · Önceki iki araştırma turuna ek · İki yeni deney ailesi ve bir tanısal ek deney

**Ana problem hâlâ açık. Bu turdaki ilerleme:** Daha önce ayrı düzeneklerde gösterilen edinim, koruma ve düzeltmenin bazıları tek bir ortak öğrenme durumunda, görev adları verilmeden birlikte yürütüldü. Ayrıca bir sonucun hangi geçmiş girdilerle ilişkili olduğu, sonuç başına hazır deneyim kimliği olmadan öğrenildi. İki başarı da açık, sınırlı dillerde elde edildi; bunların genel yaşarken öğrenme mekanizması olduğu iddia edilmiyor.

Önceki [ilk araştırma](../../research/living_learning_2026_09_20/REPORT_TR.md) ve [kapsamı yeniden açan araştırma](../../research/living_learning_reassessment_2026_09_20/REPORT_TR.md) korunur. Bu tur başında iki turun ve daha eski ilgili araştırmanın **46 dosyasının** içerik özetleri alındı. Eski deneyler, olumsuz sonuçlar ve literatür eşleşmeleri değiştirilmedi. [Koruma dökümü](../../../research/living_learning_joint_2026_09_20/results/prior_manifest.json), [doğrulama](../../../research/living_learning_joint_2026_09_20/results/verification.json). Ana Cevahir koduna entegrasyon ve GitHub'a push yok.

## 1. Araştırma yönünde gerçekten ne değişti?

Önceki yeniden sınama sonuçları deneyime erişimin bir alt problemini çözüyordu. Son tur, tek başına mevcut cevabı korumanın gelecekte doğru öğrenmeyi veya düzeltmeyi korumaya yetmediğini gösterdi. Bu tur, bu ayrımları yalnız teorik sınıflandırmada bırakmamak için iki hazır varsayımı azalttı:

1. Ortak öğrenicide her beceriye ayrı bir görev kimliği ve ayrı başlık verilmedi. Aynı katsayı vektörü, özellik listesi ve sınırlı bellek yeni ve eski ilişkileri taşıdı.
2. Zamansal ilişkide her sonuca tek bir geçmiş deneyim ID'si verilmedi. Sonuç birden fazla eski girdinin birleşimiydi; öğrenilecek şey bir eşleştirme tablosundan daha genel bir geçmiş ilişkisiydi.

**Kaldırılmayan varsayımlar:** Sayısal gözlem, uygun özellik veya gecikme dili, zaman sırası, belirlenmiş güncelleme yöntemi ve değerlendirme ailesi hâlâ hazırdır. İlk deneyde kullanılan uzamsal bölgeleme dünyanın değişim bölgesiyle uyumludur. İkinci deney, anonim gecikmiş etiketlerin bilinmeyen permütasyonunu çözmez; bir sensör zaman serisinin ilişkisini öğrenir. Bu farklar yeni başarıların kapsamını belirler.

## 2. Aynı durumda edinim, koruma ve düzeltme

24 yaşamın her birinde 1.792 sürekli gözlem. Sistem önce iki girdi arasındaki çarpım ilişkisini edinir; sonra yalnız girdilerin sağ bölgesindeki ilişki değişir. Sol bölgedeki eski ilişki geçerlidir. Model en fazla beş etkin özellik ve 128 ham gözlemle çalışır; bir bölgenin eski örnekleri, aynı bölgeden gelen yenilerle yer değiştirir. Görev veya değişim bildirimi yoktur.

24 yaşam sonunda büyüyen modelin temiz ortalama kare hatası eski geçerli bölgede `.000478`, değişmiş bölgede `.000417` oldu. Bütün yaşam boyunca cevap verme hatası `.012425`. Ham geçmişi tahmin sırasında okumadan, edinilmiş katsayılarla yeni girdilere cevap verilebiliyor. Öğrenme durumu taşınınca sonraki güncellemeler de korunuyor.

Fakat bu başarı yeni yapı büyümesinin zorunluluğu anlamına gelmedi. Aynı özellikleri baştan içeren ve aynı bölgesel belleği kullanan model aynı son risklere, ortalama 320 yerine 112 küçük doğrusal çözümle ulaştı. Bütün yaşam servis hatası da biraz daha düşüktü: `.012216`. Büyüme, bu güçlü rakibe karşı genel veya hesap açısından üstünlük sağlamadı.

Karşılaştırmalar aynı zamanda geçmişin **nasıl** tutulduğunun etkisini ayırdı. Yalnız en yeni 128 örneği tutan kayan pencere güncel sağ bölgeyi iyi öğrenirken, soldaki hâlâ geçerli beceriyi bozdu: son sol MSE `.204387`. Bütün geçmişin istatistiğini biriktiren yöntem solda çok iyi kaldı (`.000052`), fakat artık yanlış sağ geçmişi bırakmakta yavaştı (`.078358`). Bölgesel yenileme bu belirli dünyada ikisini birlikte yürütebildi.

**Başarısızlıklar korundu.** Kısa bir düzenli yanlış etiket dizisi, henüz değişmemiş doğru sağ ilişkiyi bozdu; hata `.000464→.132789` yükseldi. Daha sonra doğru veriyle toparlandı. Yani gürültülü öğrenme sırasında çalışabilmek ile yanlış bilgiyi güvenilir biçimde tanımak aynı başarı değildir.

Dahası 16/24 yaşamda, sonraki gerçek değişime yarayacak özellik bu yanlış etiket döneminde etkinleşmişti. Bunun başarının açıklaması olup olmadığını sınamak için yalnız yanlış etiket dizisi çıkarılan ayrı kontrol yapıldı. Sistem gerçek değişimden sonra yine aynı son beceriyi edindi; ilk uyumu biraz yavaşladı. Önceki çıktı korunarak yapılan bu ek deney, sonuç görüldükten sonra tasarlanmış bir mekanizma denetimidir; bağımsız doğrulama diye sunulmuyor.

**Gizli varsayım ortaya çıktı:** Sistem ziyaret edilmeyen sol bölgedeki eski kayıtları koruyor. Eğer aynı anda sol dünya da değişmişse, yalnız sağdan gelen aynı kanıt bunu göstermez. Bu alternatifte koruma yanlış geçmişi sürdürür. Dolayısıyla bölgesel yenileme, geçmişin geçerliliğini genel olarak çözmedi; dünyanın değişim biçimine uygun bir saklama varsayımı kullandı.

[Denklemler, erişim hakları, bütün karşılaştırmalar ve negatifler](../../research/living_learning_joint_2026_09_20/JOINT_STREAM.md).

## 3. Sonucun birden fazla geçmiş girdiye bağlı olması

İkinci deneyde sensör çıktısı iki eski girdinin toplam etkisinden oluşuyor:

\[
y_t=.2+1.1u_{t-2}-.7u_{t-5}+\xi_t.
\]

Öğrenici hangi gecikmelerin etkili olduğunu bilmeden, 0–6 gecikmelerini içeren verilmiş dilde ilişkiyi kestiriyor. Her rejimde 24 yaşam, yaşam başına 1.800 gözlem. Bağımsız girdili sabit dünyada yeni geçmişlerde MSE `.00001344`; yalnız güncel girdiyi kullanan modelde `.569447`. Deneyim, gelecekteki hesabın hangi geçmiş girdileri hangi katsayıyla kullanacağını değiştirdi.

Bu sonuç tek tek olayların neden olduğu sonucu bulmakla aynı değildir. Öğrenilen şey, belirli varsayımlar altında bir **öngörü ilişkisi**dir. Birden fazla geçmiş girdinin katkısı varken, “bu geri bildirimin ait olduğu tek deneyimi bul” ifadesi esas soruyu yanlış daraltabilir.

**En güçlü negatif:** Girdiler sürekli `+1,−1` dönüşümlü geldiğinde öğrenici son 200 gerçek adımda yalnız `.00000318` hata verdi; fakat yeni bağımsız geçmişlerde `.415140` hata yaptı. Birçok farklı gecikme açıklaması gözlenen örüntüde aynı cevapları veriyordu. Mevcut davranış akışında başarı, öğrenilmiş ilişkinin görülmemiş devamlar için yeterliliği değildir.

Katsayılar değiştiğinde geçmiş etkilerini azaltan yöntem daha iyi uyum sağladı; ama sabit dünyada son doğruluğu daha kötüydü. Gerçek etki gecikme 10'a taşındığında ise 0–6 dili hata tabanını aşamadı. Ne doğru unutma kuralı ne geçmiş dilinin genişliği bu deneyde öğrenildi.

Gözlemsel öğrenme ve nedensellik ayrıca ayrıldı: iki dünya aynı normal girdileri ve sonuçları üretirken, eyleme müdahale edildiğinde farklı sonuç verebilir. Dolayısıyla iyi bir zamansal kestirimden nedensel keşif sonucu çıkarılmadı. Rastgeleliğin zorunluluğu da çıkarılmadı; uygun deterministik bir darbe dizisinin aynı sonlu gecikme dilini ayrıştırabildiği cebirsel olarak gösterildi.

[Tam sonuçlar, ayrıcalıklı karşılaştırmalar ve karşı örnekler](../../research/living_learning_joint_2026_09_20/TEMPORAL_RELATION.md).

## 4. İlke arayışında ilerleme ve düzeltilen ifadeler

**Ayrı iyi güncellemeler birlikte iyi olmayabilir.** `R(a,b)=(a+b−1)²`, başlangıç `(0,0)` olsun. Bir güncelleme yalnız `a`yı `1.5`, diğeri yalnız `b`yi `1.5` yapıyor. Her biri başlangıçta tek başına riski `1→.25` düşürüyor; ikisi birlikte risk `4` yapıyor. Ayrı bellek alanları kullanmaları bile ortak çıktıdaki etkileşimi kaldırmıyor. Bu, önceki küçük deneylerin başarılarını neden tek genel sistem başarısı olarak toplayamayacağımızın tam karşı örneği.

**Kimliksiz edinim için koşullu, yapıcı bir sonuç var.** Doğrusal geçmiş özellikleri `φ_t` için `G=Σφφᵀ`, `b=Σφy` tutmak ve `(λI+G)⁻¹b` hesaplamak, yeterince ayrıştırıcı gözlemler ve uygun gürültü koşullarında ilişkiyi edinmeye yeter. Hata açıkça gözlemlerin ayırt ediciliğine ve gürültüyle ilişkisine bağlanabiliyor. Bu, yalnız “durum değişir” demekten daha somut bir algoritma ve ispat; fakat belirli doğrusal aile için bilinen sistem belirleme ilkesidir. Tek gözlem yolundan öğrenme de literatürde incelenmiştir. [Simchowitz ve diğerleri, 2018](https://proceedings.mlr.press/v75/simchowitz18a.html).

**Açık skaler optimizasyon zorunlu değil.** Hatasız ve sabit dünyada yürütülebilir tutarlı adayları elemek, kalanlar aynı cevabı verdiğinde cevaplamak; deneyimle cevaplanabilir yeni sorguların kalıcı biçimde artmasını sağlayabilir. Bunun koşullu ispatı verildi. Bu klasik version-space yaklaşımıdır, yeni ilke değildir. [Mitchell, 1977](https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf). Büyük dil, yanlış etiket veya değişen gerçek altında aynı garanti yoktur. Yararlı öğrenmeyi değerlendirmek de tek bir sayısal hedefle sınırlı değildir; edinim, koruma, düzeltme, süre ve kaynak ayrı ölçülebilir.

**Tek bir yeni daraltma yapılmadı.** Bilgi ediniminde adayların ayrıştırılması önemlidir; ama önceki derleme deneyi yeni kanıt almadan hesap bütçesinde kapasiteyi artırmıştı. Dolayısıyla bütün yaşarken öğrenmeyi yalnız hipotez eleme, yalnız yeterli istatistik veya yalnız yerel bellek yenilemeye indirgemek de yanlış olur. Deneyimi bilgiye dönüştürme, mevcut bilgiyi yürütülebilir hale getirme ve sonraki öğrenme biçimini değiştirme ilişkili fakat özdeş olmayan süreçlerdir.

[İspatlar, karşı örnekler ve literatür denetimi](../../research/living_learning_joint_2026_09_20/PRINCIPLE_AUDIT.md).

## 5. Güncellenmiş araştırma haritası

| Soru | Şimdiye kadar gösterilen | Hâlâ açık |
|---|---|---|
| Deneyim gelecekteki hesabı kalıcı değiştirebilir mi? | Kurallar, katsayılar, etkin özellikler ve bunlara bağlı yeni sorgu başarısı; silme/taşıma kontrolleri. | Geniş ve başlangıçta bilinmeyen beceri ailesi. |
| Birkaç öğrenme işlevi aynı yaşamda birlikte çalışabilir mi? | Ortak katsayı/bellekte sınırlı edinim, koruma, düzeltme ve devam. | Doğru dil ve bölgeleme olmadan, değişik uzun yaşamlar ve kaynak baskısı altında devam. |
| Her sonuca hazır deneyim kimliği gerekli mi? | Sonlu geçmiş dilinde birleşik zamansal ilişki kimliksiz öğrenilebildi. | Bilinmeyen zaman ölçekleri, eylem dizileri, anonim etiketler ve nedensel sorumluluk. |
| Daha iyi koruma daha iyi öğrenme midir? | Geçerli geçmiş ile geçersiz geçmiş farklı sonuçlar doğuruyor. | Hangisinin geçerli olduğunu öğrenicinin güvenilir biçimde ayırması. |
| Yapının büyümesi temel cevap mı? | Etkinleşen özellikler işe yarayabiliyor. | Güçlü sabit alternatiflere genel üstünlük yok; bu turda daha fazla hesap kullandı. |
| Gelecekte öğrenebilirlik nasıl korunur? | Tahminciyle birlikte güncelleme durumunu taşıma; salt cevap saklamanın eksikliği. | Hangi bilgiyi ne kadar süre ve hangi maliyetle tutacağını deneyimden seçme. |
| Öğrenme yordamı yaşamla gelişiyor mu? | Önceki aile-önsel deneyi ve bu tur farklı sabit güncellemelerin sınırları. | Verilmiş eşiği/unutmayı/dili kendi deneyiminden güvenilir geliştiren ortak sistem. |
| Dış kanıt olmadan ne değişebilir? | Önceki derleme tanığıyla sınırlı hesapta erişilebilir kapasite. | Açık görevlerde yararlı yeniden düzenlemenin seçimi, bedeli ve bozulmanın kontrolü. |

**En belirgin yeni araştırma açığı:** Ortak deneyde hangi geçmişin korunup hangisinin yenileneceğini belirleyen yerellik ve zaman ağırlıkları araştırmacı tarafından verildi. Zamansal deneyde hangi geçmiş dilinin tutulacağı da verildi. Bu seçimlerin deneyimle edinilmesi, elde edilmiş becerileri ve sonraki öğrenmeyi birlikte nasıl etkiler? Bu soru araştırılmalı; fakat ana problemi yalnız bu seçime indirgememek gerekir. Algıdan yeni değişken oluşturma, hedef/değerlendirme sinyalinin anlamı, bileşim ve kaynakların uzun yaşamda paylaşımı da açık kalır.

## 6. Doğrulama ve negatiflerin saklanması

İki deney ailesi ve yanlış etiket dizisini çıkaran ek kontrol ayrı çıktılara tekrar çalıştırıldı; bilimsel sonuçlarla birlikte bütün JSON aynı çıktı. Çalıştırılan kaynakların özetleri doğrulandı. Ek denetimde RLS ile farklı doğrudan ridge çözümü arasındaki en büyük fark `3.33×10⁻¹⁶` oldu. Ortak öğrenici değerlendirme sırasında değişmedi; ham belleksiz aynı tahminleri verdi. Beş yöntemin her biri için yeni işletim sistemi sürecinde 64 güncellemelik devam birebir aynı sonuçlandı.

İlk ayrı-süreç denetiminde, denetleyicinin durum kopyasını yanlışlıkla paylaşmasından kaynaklanan hata bulundu ve ayrık kopyayla düzeltildi. Son kayıt aşamasında henüz yazılmamış doğrulama dosyasına öz-bağlantı ayrıca düzeltildi. Başarılı tekrar üretim aşamaları korunarak yalnız ilgili denetimler yeniden yürütüldü; deney çıktıları değiştirilmedi. Önceki 46 dosya aynı kaldı; izlenen uygulama kodunda fark yok. [Denetim kaydı ve kapsam sınırları](../../research/living_learning_joint_2026_09_20/ADVERSARIAL_REVIEW.md), [doğrulama yordamı](../../../research/living_learning_joint_2026_09_20/verify_joint.py).

Genel yaşarken öğrenme ilkesi bulunduğu ilan edilmiyor. Bu tur ana probleme ilişkin iki hazır varsayımı azalttı, bazı işlevleri aynı yaşamda birleştirdi ve bu birleşmenin dayandığı yeni varsayımları görünür kıldı. Önceki sonuçlarla birlikte bunlar sonraki araştırmanın başlangıç durumudur.
