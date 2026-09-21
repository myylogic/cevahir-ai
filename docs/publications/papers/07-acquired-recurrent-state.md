# Deneyimden edinilen yinelemeli durum yordamı ve yeniden öğrenme belleği

**Muhammed Yasin Yılmaz**

CEV-2026-07 · Sürüm 2026.09.21.3 · Yayın tarihi 2026-09-21

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

İkili dizilerden sonlu geçiş grafikleri edinilerek görülmemiş uzun dizilerde yürütme, ayrı süreçte devam ve sonradan öğrenme ayrıştırılır. Temiz koşullarda uzun dizilere aktarım gerçekleşir; aynı küçük model ailesindeki güçlü sayımlı rakip çevrimiçi hatada daha iyidir. Bozuk etiketler büyük ve kötü genelleyen grafikler oluşturur. Küçük yürütme grafiğiyle devam etmek, yeni etiketlerden yeniden öğrenmek için tutulan örnek arşivinin gereksiz olduğunu göstermez.

## Abstract

Finite transition graphs acquired from binary strings are tested on long unseen strings, fresh-process continuation and later relearning. Clean conditions generalize, while a strong bounded enumeration baseline has fewer online errors. Corrupted labels induce larger, poorly generalizing graphs. Compact execution state does not remove the archive used for subsequent learning.

## Bulguların yorumu

Deneyimden yürütülebilir bir durum yordamı edinilebilir; yürütme belleği, öğrenme belleği ve hatayı teşhis etme bilgisi birbirinden farklıdır.

Boş dizi değerlendirmesi ile boş olmayan dillerin değerlendirmesi ayrı raporlanmıştır. Gerçek etiketlerin dışarıdan verilerek düzeltilmesi kendi kendine hata teşhisi sayılmaz.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/living_learning_state_2026_09_20/REPORT_TR.md](../../research/living_learning_state_2026_09_20/REPORT_TR.md) olarak korunur. Bağlantı yolları bu
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

## Araştırma kaydı: Cevahir — Deneyim, hangi geçmiş ayrımlarının taşınacağını değiştirebilir

20 Eylül 2026 · Önceki dört araştırma turunun devamı

**Ana problem hâlâ açık.** Bu turda sistem, kısa olay dizilerinin yalnız sonundaki sonuçlardan, daha uzun yeni dizileri işleyen kalıcı bir iç durum yordamı edindi. Önceki gecikme deneyinden farklı olarak hangi geçmiş ayrımlarının aynı duruma gideceği hazır verilmedi. Ancak otomata çıkarımı bilinen bir yöntem ailesi; sonuç yeni bir genel öğrenme ilkesi diye sunulmuyor.

Önceki [ilk tur](../../research/living_learning_2026_09_20/REPORT_TR.md), [kapsam değerlendirmesi](../../research/living_learning_reassessment_2026_09_20/REPORT_TR.md), [ortak yaşam](../../research/living_learning_joint_2026_09_20/REPORT_TR.md) ve [öğrenilmiş güncelleme](../../research/living_learning_update_2026_09_20/REPORT_TR.md) korunur. Yeni tur başında **74 eski dosyanın** içerik özeti alındı. Negatifler ve literatür eşleşmeleri geri alınmadı. Ana sisteme entegrasyon ve GitHub'a push yapılmadı.

## 1. Araştırma yönünde ilerleme

Son turda aynı bugünkü cevabın farklı yarınki öğrenme yollarıyla uyumlu olduğunu göstermiştik. Fakat yalnız öğrenme hızını/yönünü değiştirmek, eksik bir geçmiş ayrımını kendiliğinden oluşturmaz. Bu tur iki ilişkili soruyu araştırdı:

1. Sistem yaşadığı hatanın nedenini kesin olarak teşhis etmeden, daha iyi çalışan bir hesaplama durumu edinebilir mi?
2. Deneyim, geçmişin ham tekrarından farklı olarak, gelecekteki olayları işleyen yeni bir geçiş yordamına nasıl dönüşebilir?

İlk soru için açık karşı örnekler kuruldu: hedef değişimi ile geri bildirim kanalının bozulması aynı gözlem akışını üretebilir. Zamansal atıf hatası da belirli dizilerde hedef değişimiyle aynı görünebilir. Buna karşılık tutulan olay sırası ve daha sonra doğal biçimde gelen ayırıcı örnekler, bazı ayrımları mümkün kılar. **“Önce hatanın fiziksel türünü kesin bul, sonra öğren” zorunlu bir genel ilke olarak desteklenmedi.** Daha iyi öngörü durumu öğrenmek ile fiziksel açıklamayı belirlemek farklı başarılar.

[Tam karşı örnekler, yapıcı koşullar ve kaynaklar](../../research/living_learning_state_2026_09_20/FAILURE_IDENTIFIABILITY.md).

## 2. Yeni deney: geçmişten bir yürütme yordamı edinmek

Her yaşamda 256 farklı ikili dizi gösterildi; uzunluklar 1–12. Öğrenici dizi sonunda cevap verip sonra tek etiketi aldı. Ara durumlar, gerçek geçişler ve gerekli durum sayısı verilmedi. 16, 32, 64, 128 ve 256 etiket sonunda eski deneyimlerden bir durum grafiği çıkardı; arada son grafiğiyle çalışmayı sürdürdü.

Bu grafik yeni dizilerde ham eski kayıtları okumadan, sembol başına bir geçişle çalışıyor. Eğitimden çok daha uzun, 32–256 sembollük dizilerle sınandı. Bunun yanında, gizli gerçekle bütün olası dizilerde eşdeğerliğini denetleyen bağımsız bir hesap kullanıldı; bu hesabın sonuçları öğreniciye verilmedi.

- **16 farklı üç durumlu hedef:** Öğrenilmiş grafiklerin tamamı, kullanılan boş olmayan dizi alanının bütününde doğru çıktı. Doğru üç-durum sınırını baştan bilen güçlü aday-eleme yöntemi de aynı son başarıya ulaştı ve öğrenme sırasında daha az hata yaptı.
- **Parite:** Dört yaşamda da iki durumlu kural edinildi. Geçmişin ne kadar uzağında kaldığından bağımsız bir ayrım taşınabildi; herhangi bir sabit uzunluktaki son-sembol penceresi için bunun mümkün olmadığı tam karşı örnekle gösterildi.
- **Dört durum gerektiren yeni kural:** Durum sayısına üç sınırı verilmeyen öğrenici dört yaşamda da doğru dört durumlu yordamı edindi. Üç durumlu aday ailesi ise boşaldı.

Bu, her türlü yeni temsilin keşfi değildir. İkili alfabe, bölüm sınırları, doğru terminal atfı ve sonlu durum hesaplama dili hâlâ verilidir. Yine de önceki hazır gecikme listesinden farklı olarak, **yeni olayların hangi iç durumu nasıl değiştireceği deneyimden çıkarıldı**.

## 3. En önemli negatif: büyüyen yapı, yanlış deneyimi açıklayabilir

Her biri farklı bir diziye ait 256 etiketin 32'si bozulunca, öğrenici bütün geçmiş etiketlerle tutarlı kaldı. Fakat üç durumlu gerçek ilişkileri açıklamak için **36–42 durumlu** yapılar kurdu. Sekiz yaşamın hiçbirinde gerçek kurala eşdeğer olmadı; yeni uzun dizilerde ortalama Brier kaybı `.287109` oldu.

Burada aynı diziye iki farklı etiket verilmediği için, salt tutarlılık denetimi bozulmayı yakalayamadı. Daha çok yapı, yanlış etiketlere açıklama üretti. Sabit üç durumlu aday ailesinin boşalması da tek başına teşhis sağlamadı: hem yanlış etiketler hem temiz dört durumlu gerçek aynı belirtiyi üretti.

**Çıkarım:** Hata azalması, eğitim tutarlılığı ve yapı büyümesi; yeni yapının geçerli olduğuna tek başına kanıt değildir. Fakat bundan yapı öğrenmenin yararsız olduğu sonucu da çıkmaz; temiz dört durumlu koşulda büyüme gerekliydi ve doğru sonucu verdi. Açık problem, bu iki durumu hangi erişilebilir kanıt ve kaynaklarla ayırabildiğimizdir.

Saklanan kayıtların etiketleri güvenilir dış düzeltmeyle değiştirildiğinde sekiz model de tanımlı kullanım alanında onarıldı. Bu, geçmişin onarıma yettiğini gösterir; hatalı etiketlerin sistem tarafından keşfedildiğini göstermez.

## 4. Sonuç yorumunda düzeltilen bir sınır

İlk tam-dil denetimi 16 ana öğrenicinin ikisinde uyuşmazlık bildirdi. Sonraki inceleme ikisinin de **yalnız boş diziye** ait olduğunu buldu. Deneyde hiçbir boş bölüm kullanılmamıştı; boş olmayan bütün dizilerde modeller doğruydu.

Bu nedenle özgün 14/16 tam-dil kaydı korunuyor, fakat kullanım alanındaki sonuç 16/16 olarak ayrılıyor. Bu bulguyu “uzun dizilerde görünmeyen iki başarısızlık” diye sunmak yanlış olurdu. Ek boş-dizi etiketiyle yapılan tanısal onarım ayrı sonuç sonrası denetim olarak kaydedildi. Böylece ne ilk sonuç silindi ne değerlendirme alanı sessizce değiştirildi.

## 5. Temel hesaplama ilkesi açısından ne öğrendik?

İki geçmişin şu an aynı cevabı vermesi, onları aynı iç duruma koymaya yetmez. Aynı sonraki olay, ileride farklı cevaplar gerektirebilir. Sabit deterministik bir hedef için anlamlı ilişki:

\[
h\sim h'\quad\Longleftrightarrow\quad
\text{her izinli devam }u\text{ için }f(hu)=f(h'u).
\]

Bu ilişki yeni sembol eklenince korunuyorsa, geçmişlerin sınıfları bir geçiş yordamı oluşturabilir. Sistem her eski olayı ayrı ayrı taşımak yerine, gelecekteki kullanım açısından gereken ayrımı taşır. Bu tur, belirli temiz ailelerde bu yordamın terminal deneyimlerden edinilebildiğini gösterdi.

Bu ilke bilinen otomata ve öngörü-durumu kuramıyla eşleşir; yeni bir keşif iddiası yoktur. Durum birleştirme için [Oncina ve García, 1992](https://grfia.dlsi.ua.es/repositori/grfia/pubs/77/inferring.pdf); öğrenilmiş yinelenen öngörü durumu için [Shalizi ve Shalizi, 2004](https://arxiv.org/abs/cs/0406011). Sonlu veride henüz ayrıştırıcı devam görülmemesi ise bütün geleceklerin eşdeğer olduğunu kanıtlamaz. Öğrenmenin genelleme tercihleri ve varsayımları burada da gerekli.

**Önceki sonuçlarla bağlantı:** Deneyim bir cevabı, yeni kanıta tepkiyi veya geçmiş olayları işleyen yordamı değiştirebilir. Bu üç etki ayrı mekanizmalarla gösterildi. Hepsi deneyimden kalan yürütülebilir durumun gelecekteki hesaplara etkisini içeriyor; fakat ortak tanımın yazılabilmesi, genel edinim/koruma/düzeltme algoritmasının bulunması demek değil.

## 6. Kalıcılık ve bedel

Öğrenilmiş grafiği ve bölüm içindeki mevcut durumunu yeni bir süreçte yükleyince, eski ham ön ek olmadan kalan 385 sembol aynı sonucu verdi. Öğrenmenin tamamını kaldığı yerden sürdürmek için ise örnek arşivi de taşındı; sonraki 53 bölüm ve bir yeniden çıkarım aynı kaldı. **Küçük çalışma durumu ile küçük öğrenme belleği aynı başarı değil.**

Ana ailede son grafik ortalama yaklaşık üç durum kullanırken, yeni yapı çıkarımı için 256 dizi ve ortalama 2.094 sembol saklandı. Gürültü altında birleştirme denemelerinin kopyalama hesabı belirgin büyüdü. Yaşam boyu sınırlı bellek veya üstün toplam hesap verimliliği gösterilmedi. Eski grafikteki sayısal durum kimliğini yeni grafiğe aynen taşımak da yanlış olabilir; güncellemelerin bölüm sonunda yapılması bu sorunu bu deneyde erteliyor.

[Bütün yöntemler, skorlar, sayaçlar ve deney ayrıntıları](../../research/living_learning_state_2026_09_20/EXPERIMENT.md).

## 7. Araştırma hâlâ nereye açık?

Bu turdan “yaşarken öğrenme otomata çıkarmaktır” sonucu çıkmıyor. Dünyanın gözlemlerini anlamlı sembollere dönüştürme, değerlendirme/amaç sinyali, gecikmiş veya belirsiz atıf, açık hesaplama dili, bilgi derleme, beceri bileşimi ve eylem yoluyla deneyim üretimi hâlâ açık alanlar. Önceki yeniden sınama ve meta-öğrenme sonuçları kendi yerlerinde korunuyor.

Somut yeni açık, **bugünkü iyi çalışmayı korurken yarın gerekli olabilecek ayrımları kaybetmeden öğrenmeye devam etmek**. Grafik yürütmek için birkaç durum yeterliyken grafiği güvenilir değiştirmek için çok daha zengin kayıt gerekti. Veri bozulunca aynı yapı çıkarımı yanlış büyüdü. Bu nedenle sonraki adım yalnız daha büyük bir makine veya daha iyi bir durum-birleştirme skoru olmamalı: hangi deneyim bilgisinin hem yeni yapı edinimini hem yanlış yapının düzeltilmesini desteklediği, daha farklı temsil ve bileşim aileleriyle sınanmalı. Ana problem bu alt soruya indirgenmiyor.

## 8. Kayıt ve doğrulama

[Özgün deney çıktısı](../../../research/living_learning_state_2026_09_20/results/recurrent_state.json), [ayrı kanıt denetimi](../../../research/living_learning_state_2026_09_20/results/evidence_audit.json), [eski dosyaların koruma dökümü](../../../research/living_learning_state_2026_09_20/results/prior_manifest.json), [doğrulama sonucu](../../../research/living_learning_state_2026_09_20/results/verification.json).

Birleştirme yordamının beş bilinen dilde tam eşdeğerliği, 10 uzunluğa kadar bütün dizilerde davranışı, rastgele altı büyük örnek kümesinde eğitim tutarlılığı, örnek sırası ve JSON taşınması ayrı kontrollerle sınandı. Aday evreni bağımsız küçültmeyle doğrulandı. Tüm deney ve kanıt denetimi yeniden üretimi, önceki 74 dosyanın içerik kontrolü ve bağlantı denetiminin sonucu doğrulama kaydında yer alır. Bunlar belirtilen deneyin uygulama güvenilirliğine ilişkindir; ana bilimsel problemin çözüldüğüne ilişkin sertifika değildir.

Son doğrulama geçti: özgün deney ve ayrı kanıt denetimi baştan, geçici ayrı çıktılara çalıştırıldı; her iki JSON da bütün içerikleriyle aynı çıktı. Önceki 74 dosyanın tamamı değişmeden kaldı; izlenen uygulama dosyalarında fark yok. Bağımsız kod okuması, hizmet tahmininin etiketten önce geldiğini ve değerlendirme bilgisinin öğreniciye sızmadığını doğruladı; sayaçların tam kaynak maliyeti olmaması ve düzeltmelerin dış bilgi kullanması rapora açıkça işlendi.
