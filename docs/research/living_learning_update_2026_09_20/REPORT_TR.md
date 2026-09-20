# Cevahir — Deneyim yalnız cevabı değil, sonraki öğrenmeyi de değiştirebilir

20 Eylül 2026 · Önceki üç araştırma turunun devamı

**Ana yaşarken öğrenme problemi hâlâ açık.** Bu turda yeni olarak, eski cevapları taşımadan yalnız deneyimden edinilmiş güncelleme durumunu taşıyan bir deney tamamlandı. Aynı yeni kanıttan daha iyi yararlanma belirli bir yeni yaşam ailesinde gösterildi. Aynı aktarım başka ailelerde zarar verdi; bilinmeyen başlangıç becerisini edinmede üstünlük göstermedi. İki sonuç birlikte korunuyor.

Önceki [ilk araştırma](../living_learning_2026_09_20/REPORT_TR.md), [kapsam değerlendirmesi](../living_learning_reassessment_2026_09_20/REPORT_TR.md) ve [ortak yaşam araştırması](../living_learning_joint_2026_09_20/REPORT_TR.md) değiştirilmedi. Bu tur başlangıcında ilgili **61 dosyanın** içerik özeti alındı; son denetim bunları karşılaştırır. [Koruma kaydı](../../../research/living_learning_update_2026_09_20/results/prior_manifest.json). Ana sisteme entegrasyon ve GitHub'a push yapılmadı.

## 1. Ana probleme geri bağlanan soru

Önceki araştırma, deneyimin yürütülebilir kurallar, katsayılar, etkin özellikler ve zaman ilişkileri bırakabildiğini göstermişti. Bunların bir kısmı aynı sınırlı yaşamda birlikte çalıştı. Fakat o deneylerde neyin ne kadar hızlı değiştirileceğini belirleyen önemli tercihler araştırmacı tarafından verilmişti.

Bu turdaki soru şuydu: **Sistem geçmişte yaşadıkları nedeniyle, henüz karşılaşmadığı aynı yeni deneyimden farklı biçimde öğrenebilir mi?** Bu, mevcut cevabın değişmesinden farklı bir ölçümdür. İki sistem bugün aynı cevapları verirken yeni etiketlerden bir süre sonra farklı beceriler edinebilir.

Bu ayrım, 'öğrenme kapasitesi'ni yalnız bugünkü test başarısıyla ölçmenin eksikliğini görünür kılar. Bir yandan da sınırı vardır: ilerideki öğrenme yolunun değişmesi, kullanılabilir bütün fonksiyon sınıfının genişlediği veya genel bir öğrenme ilkesinin bulunduğu anlamına gelmez.

## 2. Ne yapıldı ve ne bulundu?

24 kaynak yaşamda doğrusal bir tahminci çalışırken, aynı gözlemin katsayıları hangi yönde ve hangi büyüklükte değiştireceğini belirleyen iki boyutlu bir matris öğrendi. Gerçek ilişkiler bir gizli yönde hızlı, dik yönde yavaş değişiyordu. Yön öğreniciye verilmedi. Kaynak başına 16.384 örnek kullanıldı.

Sonra bütün eski tahmin katsayıları ve çalışma izleri sıfırlandı. Yalnız matrisin üç bağımsız sayısı yeni yaşama taşındı. Her kaynaktan 16 yeni yaşam başlatıldı; dört rejim, rejim başına dört yaşam, yaşam başına 1.536 yeni örnek. Bütün karşılaştırmalar aynı yeni veriyi gördü; ilk cevapları tam aynıydı.

| Yeni yaşam koşulu | Öğrenilmiş sabit kural | Yeni başlayan meta-öğrenici | Aynı geçmişten seçilmiş skaler kural |
|---|---:|---:|---:|
| Hızlı/yavaş değişim yapısı aynı | .005199 | .008774 | .006868 |
| Hızlı/yavaş yönler yer değiştirmiş | .034495 | .008623 | .006868 |
| İki yönde de hızlı değişim | .036181 | .016619 | .010456 |
| Yapı aynı, başlangıç ilişkisi bilinmiyor | .013223 | .013219 | .008538 |

Sayılar yeni yaşamın bütün adımları boyunca temiz tahminin ortalama kare hatasıdır; düşük daha iyi. Yeni başlayan meta-öğrenici, aynı meta-yordamı geçmişten edinilmiş matris olmadan çalıştırır. Skaler karşılaştırma da geçmişten öğrenmiştir; yalnız daha kısıtlı bir güncelleme ailesi kullanır.

**Olumlu sonuç:** Benzer değişim yapısında geçmiş deneyim, ilk cevabı değiştirmeden yeni kanıttan daha iyi yararlanmayı sağladı. Matrisin yön bilgisini silmek veya döndürmek bu yararı azalttı. Kazanılan bilgi yalnız eski hedefin hatırlanması değildi.

**Güçlü olumsuz sonuç:** Yönler yer değiştirince aynı öğrenilmiş kural ciddi biçimde zararlı oldu. Kuralı yeni yaşamda öğrenmeye devam etmek hatayı `.034495→.011996` düşürdü, fakat tüm yaşam boyunca yeni başlayan meta-öğreniciden daha kötü kaldı. Öğrenilmiş bir öğrenme alışkanlığı da düzeltme maliyeti yaratabiliyor.

**Kapsamı daraltan sonuç:** Başlangıç ilişkisi bilinmediğinde öğrenilmiş kuralın yeni başlayan meta-öğreniciye üstünlüğü gösterilmedi. Kaynaktan seçilmiş skaler daha iyiydi. Kaynak yaşamın öğrettiği değişim takibi ile yeni bir ilişkinin başlangıçtan edinimi farklı ihtiyaçlar doğurdu. Bu sonuç, 'deneyim artık genel olarak daha iyi öğrenmeyi sağlıyor' ifadesini desteklemiyor.

Dokuz karşılaştırmanın tamamı, yaklaşık belirsizlik aralıkları, veri hakları ve maliyetler [deney raporunda](EXPERIMENT.md). Kaynak edinim maliyeti hesaba katılmadan toplam yaşam verimliliği ilan edilmedi. Sonradan yapılan bağımsız matematiksel hesap, sabit matrisli yöntemlerin beklenen hatasını türeterek olumlu ve olumsuz aktarımı aynı varsayımlar altında açıkladı.

## 3. Temel ilke arayışı için anlamı

Bir sistemin kalıcı durumunu işlevsel olarak `s=(w,m)` diye ayıralım. `w` mevcut cevabı, `m` aynı yeni deneyimin `w`yi nasıl değiştireceğini etkilesin. Bu deneyde:

\[
w_{t+1}=w_t+P_mx_t(y_t-x_t^Tw_t),\qquad
\frac{\partial w_{t+1}}{\partial y_t}=P_mx_t.
\]

Dolayısıyla aynı bugünkü tahmin, aynı yarınki öğrenme kapasitesi demek değildir. Geçmişin etkisi, henüz alınmamış kanıtın gelecekteki davranışa nasıl dönüşeceğinde saklı olabilir. Bu işlevsel ayrım önceki 'tahmin durumunu taşımak sonraki güncellemeyi korumaya yetmez' sonucunu tamamlıyor.

**Bu sonuç yeni bir zorunlu mimari getirmiyor.** `w` ve `m` tek bir büyük durum olarak yazılabilir; onları güncelleyen sabit bir program, deneyime bağlı farklı öğrenme yollarını gerçekleştirebilir. Sistemin kaynak kodunu sürekli değiştirmesi veya sonsuz sayıda üst öğrenici katmanı kurması gerekmez. Ayrım, depolanan şeyin adıyla değil, silme/taşıma müdahalesinin hangi gelecek hesapları değiştirdiğiyle sınanır.

**Tek başına durum değişimi yeterli ölçüt değil.** Rastgele bozulma ve yanlış etiket de geleceği değiştirir. Yararlı öğrenme iddiası; hangi yeni görevlerde, hangi geri bildirim ve kaynak koşullarında edinim, koruma ve düzeltmenin gerçekleştiğini göstermelidir. Bu göreli değerlendirme, kullanıcıdan gelen probleme bütün dünyalarda kusursuzluk şartı eklemez. Sadece başarının kapsamını açıklar.

**Öğrenilmiş güncelleme de ana cevap değil.** Öğrenme oranının önceki öğrenme deneyiminden edinilmesi bilinen bir konudur; IDBD doğrudan bu soruyu ele alır. [Sutton, 1992](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf). Tamamen çevrimiçi görev içi ve görevler arası uyarlama da parametrik algoritma ailelerinde incelenmiştir. [Denevi ve diğerleri, 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/e0e2b58d64fb37a2527329a5ce093d80-Abstract.html). Bu tur bilinen aile içinde nedensel bir ayrım ve sınırlı tanık üretiyor; yeni temel yasa bulduğunu iddia etmiyor.

## 4. Genel sonuç ile yerel sonuç nasıl ayrılıyor?

| Önceki ve yeni sonuç | Taşıdığı genel ders | Kanıtın sınırlı kaldığı yer |
|---|---|---|
| Deneyimden kalan yürütülebilir durum gelecekteki hesabı değiştiriyor. | Kalıcılık, durum silme ve taşıma müdahaleleriyle ölçülebilir. | Sonlu kurallar, özellikler ve görev aileleri kullanıldı. |
| Aynı mevcut cevap farklı gelecek öğrenme yollarıyla uyumlu. | Bugünkü tahmin yeterliliği, yarınki öğrenme/düzeltme yeterliliği değildir. | Bu tur iki boyutlu güncelleme dönüşümüyle gösterildi. |
| Geçmişi korumak bazı dünyalarda yararlı, bazılarında zararlı. | Koruma ve yenileme değerlendirmesi geçmişin geçerliliğine bağlı. | Önceki bölgesel bellek ve bu turun değişim yönleri evrensel değil. |
| Derleme, yeni dış kanıt olmadan sınırlı hesapta yarar sağlayabiliyor. | Bilgi kazanımı ile mevcut bilginin kullanılabilir hale gelmesi farklı. | Derleyici, dil ve ölçülen bütçe verilmişti. |
| İyi ayrı güncellemeler birlikte zarar verebiliyor. | Parça başarıları birleşik sistem başarısı olarak toplanamaz. | Genel birleşik denetim mekanizması henüz yok. |
| Gözlenen aynı geçmiş farklı gerçek açıklamalarla uyumlu. | Ek varsayım veya yeni ayırt edici deneyim gerekebilen durumlar var. | Her öğrenme probleminin yalnız yeniden sınama problemi olduğu sonucu çıkmaz. |

Bu ortak dersler bir mimari tarifi değil; sonraki adayların hangi iddiaları kanıtlamak zorunda olduğunu belirleyen araştırma kısıtlarıdır. Bunlardan genel yapıcı bir yaşarken öğrenme mekanizması henüz çıkarılmadı.

## 5. Hâlâ araştırılmamış alanlar ve yön kararı

Bu tur güncellemenin bir kısmını deneyime açtı; fakat aşağıdaki boşlukları kapatmadı:

1. **Deneyimin nasıl temsil edileceğinin edinilmesi.** Önceki yapı büyümesinde işlem dili, zamansal deneyde gecikme dili ve burada doğrusal gözlem uzayı hazırdı. Yeni bir hesap türüne ne zaman ve nasıl ihtiyaç duyulduğunu sistemin kendisinin belirlemesi sınanmadı.
2. **Geri bildirimin ne hakkında ve ne kadar güvenilir olduğu.** Bu tur etiketlerin aynı adımın doğru anlamlandırılmış hedefi olduğu verilidir. Hedef değişimi, gözlem bozulması, eksik değişken ve yanlış atıf; aynı artık hatayı üretebilir. Bunların ayrımı hız matrisiyle çözülmedi.
3. **Gelecekte düzeltme yapmaya yetecek geçmişin seçilmesi.** Burada geçmişin ham saklama bütçesi öğrenilmedi. Güncel iyi tahminle uyumlu bir sıkıştırma, gelecekte gerekli bir ayrımı silebilir. Önceki karşı örnekler geçerlidir.
4. **Beceri edinimi, koruma, düzeltme ve hesap düzenlemesinin ortak kaynak kullanımı.** Sınırlı ortak yaşam deneyimiz bir başlangıçtır. Yeni güncelleme sonucunu ona eklemek, ikisinin otomatik olarak güvenli birlikte çalışacağını göstermez.
5. **Deneyimi üretme ve iç durumun davranışı değiştirmesi.** Eski yeniden sınama çalışması bu alanın belirli parçasını araştırdı. Bu tur ise dışarıdan verilen akışla çalışıyor; biyolojik iç durum sorusunu yeni bir adla çözmüş olmuyor.

**Sonraki yön kararı:** Güncelleme hızını daha fazla iyileştirmeyi araştırmanın merkezi yapmamak gerekir. Bu turda başlangıç bilgisinin değişmesi bile edinilmiş tercihi tersine çevirdi. Daha geniş açık soru, sistemin bir başarısızlığın mevcut katsayıya mı, gözlem/atıf biçimine mi, temsil diline mi ait olduğuna ilişkin ayrımı nasıl edinebileceğidir. Bu ayrım için geçmişte ne tutulması gerektiği de birlikte araştırılmalı. Bu bir sonraki araştırma hipotezi alanıdır; henüz denenmiş sonuç veya zorunlu mimari değildir. Derleme, beceri bileşimi ve eylemle deneyim üretimi paralel açık alanlar olarak korunur.

## 6. Doğrulama

Meta-türev, bloktaki sayısal sonlu farkla karşılaştırıldı; en büyük fark `4.94×10⁻¹²`. Bütün iki boyutlu problem ortogonal döndürülünce 512 adımlık öğrenme yolunun farkı en çok `1.94×10⁻¹⁶` oldu. Bu kontrol genel tersinir koordinat dönüşümü altında değişmezlik iddiası vermez.

Blok ortasında kaydedilen tüm öğrenme durumu yeni işletim sistemi sürecinde yüklendi; 149 adımlık devam, tahminler ve son durum dahil aynı oldu. Bu devam için yalnız `P` değil, mevcut `w`, duyarlılıklar, birikmiş gradyan ve blok konumu gerekir. Yeni bir yaşama yalnız `P` taşıma deneyi ile aynı yaşamı kaldığı yerden sürdürme deneyi farklıdır.

Bağımsız kod incelemesi veri sızıntısı veya sonucu geçersiz kılan güncelleme hatası bulmadı; bilinmeyen başlangıç negatifinin ve temsil sınıfının sabit kalmasının raporda yer almasını istedi. Tam yeniden üretim ve koruma denetiminin makine tarafından yazılan sonucu [doğrulama kaydındadır](../../../research/living_learning_update_2026_09_20/results/verification.json). İncelemenin kapsamı ve sınırları [denetim notunda](REVIEW.md) korunur.

Son denetim geçti: tam deney ve bağımsız ikinci-moment hesabı ayrı çıktılara yeniden çalıştırıldı; iki JSON da bütün içerikleriyle aynı çıktı. Analitik beklentinin döndürme simetrileri en çok `2.22×10⁻¹⁶` farkla korundu. Önceki 61 dosyanın içeriği aynı; izlenen uygulama dosyalarında değişiklik yok. Bu doğrulama deneyin tekrarlanabilirliğini ve belirtilen uygulama kontrollerini destekler; genel bilimsel çözüm garantisi değildir.
