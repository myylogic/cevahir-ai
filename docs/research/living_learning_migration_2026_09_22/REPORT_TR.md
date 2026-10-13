# Tur 16 — Doğru öğrenilmiş kuralı yaşayan akışta çalıştırmak

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel araştırma kaydı

## Sonuç

Yeni kuralı doğru öğrenmek, çalışan sistemin o kural için gereken mevcut iç durumunu bilmesini sağlamadı. Deneyde eski parity kuralı ve iki yeni kural ailesi, bütün 24 yaşamda kalibrasyon deneyimlerinden doğru seçildi. Buna rağmen yeni kuralı sıfır durumundan başlatmak, geçmişe bağlı yanlış cevaplar üretti. Doğru başlatma için gereken bilgi görevden göreve değişti.

Sondan beşinci biti isteyen yeni beceride eksik başlangıç bilgisi beş yeni bit geldikten sonra gereksizleşti. Toplam bir sayısını üçe göre izleyen beceride ise yeni bitleri görmek eksik başlangıç kalanını gidermedi. Bir kez dışarıdan verilen doğru çıktı etiketi bu belirsizliği giderdi; bu iyileşme yeni dış bilginin etkisidir.

Eski becerinin iç durumunu koruyan beş yöntem bütün eski parity cevaplarını korudu. Her şeyi sıfırlayan kontrol, yeni becerisi düzelse bile eski becerinin geçmiş bilgisini geri kazanamadı. Böylece kural edinimi, mevcut duruma yerleştirme ve eski beceriyi sürdürme ayrı gereklilikler olarak sınandı.

**Ana yaşayan öğrenme problemi çözülmüş değildir.** Bu deneyin aday programları, yardımcı bellek biçimleri ve kalibrasyon kanalı önceden verilmiştir. Bunların sistemce icadı, kendi deneyimini üretmesi, açık uçlu kavram geliştirmesi veya genel zekâ gösterilmedi. Bu tur bir gerekli koşulu inceler; bütün araştırmanın yerine geçmez.

## Kurulum ve bilgiye erişim

[Tur 15](../living_learning_temporal_2026_09_22/REPORT_TR.md) öğrenilmiş kuralı her bölümde sıfırdan kullanmıştı. Bu turda ana bit akışı devam ederken yeni kural eklenir. [Önceden yazılan protokol](PROTOCOL.md) 24 yaşam, 256 veya 4.096 bitlik önceki akış, iki yeni görev ve altı durum taşıma yöntemi belirledi. Toplam 576 yol, her yolda 65 an ve iki yetkinlik cevabı vardır: 37.440 değerlendirme anı, 74.880 yetkinlik çıktısı. Bunlar 24 eşleştirilmiş yaşamdan gelir; bağımsız deney sayısı olarak yorumlanmaz.

Eski parity kuralı, iki-durumlu 64 aday arasından; yeni kural, sekiz gecikme ve altı modulo-3 okuma permütasyonundan oluşan 14 aday arasından 128 etiketli örnekle edinilir. Kalibrasyon bölümleri 8–24 bit uzunluktadır ve sıfır başlangıçlıdır. Bu referans örnekler ana akışın geçmişini içermez. Ana akış sıfırlanmadan sürer, fakat **bütün veri erişimi sıfırlamasız tek bir dünyadan gelmiş değildir**: ayrı kalibrasyon kanalı deneyin açık varsayımıdır.

72 kalibrasyonun tamamında tek tutarlı ve doğru aday kaldı: 24 eski parity, 24 gecikme-beş, 24 modulo okuma kuralı. Görev kimliği öğreniciye verilmedi; doğru aday kimliği yalnız denetimde kontrol edildi. Öğrenme sonrası referans örnekler ve elenmiş adaylar derlenen çalışma denetleyicisinde tutulmadı. Bu aktarım gelecekte yeniden kural öğrenmeye yeterli bütün kanıtın korunduğu anlamına gelmez; tur 15'teki öğrenmeye devam etme ayrımı açık kalır.

Yeni kural kurulduğu an `t=0`; 64 yeni bit daha geldiğinde `t=64` olur. Her tahmin değerlendirilir ama doğru etiket öğreniciye verilmez. Tek istisna `t=16`: yeni becerinin doğru etiketi, o anki tahminden sonra bir kez verilir. Eski beceriye ana akış boyunca düzeltici etiket verilmez.

## Aynı kural, farklı geçmiş bilgisi

Altı yöntem aynı kalibrasyon sonuçlarını alır:

| Yöntem | Kurulumdan önce saklanan ek bilgi | Yeni durumun kurulması |
|---|---|---|
| Sıfır varsayımı | Yok | Yeni durumun sıfır olduğu varsayılır; eski durum korunur |
| Olası durumlar | Yok | Eski bilgiyle uyumlu bütün yeni durumlar taşınır |
| Son 8 bit rezervi | Son sekiz ham bit | Gecikme tam kurulur; modulo belirsiz bırakılır |
| Modulo-6 rezervi | Bir sayısının altıya göre kalanı | Modulo-3 tam kurulur; gecikme belirsiz bırakılır |
| Tam arşiv | Ana akışın bütün önceki bitleri | Öğrenilmiş programla geçmiş yeniden yürütülür |
| Her şeyi sıfırlama | Yok | Eski ve yeni iç durum sıfırlanır; gerçek çevre sıfırlanmaz |

Rezervler, hangi ek bilginin tutulacağının önceden seçildiği karşılaştırmalardır. Bunların ileride faydalı olacağını sistemin öğrendiği söylenmiyor. Olası durum kümesi gerçek geçmişlerin olasılık dağılımı değildir; farklı olası çıktılara eşit ağırlık verilir. Bu tercih, doğru çıktı kümede olmak şartıyla kullanılan Brier kaybında minimax'tır; [kanıt](MATHEMATICAL_AND_LITERATURE_TR.md) ve koşulları ayrı belgededir.

## Ölçülen sonuçlar

Yeni beceri için üç sınıflı Brier toplamı kullanılır: doğru kesin cevap `0`, yanlış kesin cevap `2`; iki olası çıktıya eşit dağılım `0.5`, üçüne eşit dağılım `2/3`. Eski parity'nin ikili skaler MSE'si ayrı hesaplanır. Bu ölçekler tur 15'in skaler MSE'siyle doğrudan karşılaştırılmamalıdır.

Aşağıdaki değerler 24 yaşam ortalamasıdır. Aralıktaki bütün anlar da ortalamaya katılmıştır. `256 / 4096`, kurulumdan önceki iki akış uzunluğunu gösterir.

| Yöntem | Gecikme, t=0–4 | Gecikme, t=5–15 | Modulo-3, t=0–15 |
|---|---:|---:|---:|
| Sıfır varsayımı | 1.016667 / 0.833333 | 0 / 0 | 1.000000 / 0.916667 |
| Olası durumlar | 0.5 / 0.5 | 0 / 0 | 0.666667 / 0.666667 |
| Son 8 bit rezervi | 0 / 0 | 0 / 0 | 0.666667 / 0.666667 |
| Modulo-6 rezervi | 0.5 / 0.5 | 0 / 0 | 0 / 0 |
| Tam arşiv | 0 / 0 | 0 / 0 | 0 / 0 |
| Her şeyi sıfırlama | 1.016667 / 0.833333 | 0 / 0 | 1.000000 / 0.916667 |

Gecikmede tam olası durum kümesi sırasıyla `32→16→8→4→2→1` oldu. Eski beş bitin etkisi beş yeni bit sonra sona erdi. Sıfır varsayımı da bu noktada doğru duruma geldi; erken yanlış cevaplar geçmiş kayıtlarından silinmedi.

Modulo görevinde olası durum kümesi `t=0` ile etiket öncesi `t=16` arasında üç elemanda kaldı. Sıfır varsayımı 256-bit geçmişte 12/24, 4.096-bit geçmişte 11/24 yaşamda yanlış başladı; aynı yanlış fark dış etiket gelene kadar sürdü. `t=16` etiketi bu yolları onardı. `t=17–64` arasında bütün yöntemler yeni beceride hatasızdı. Bu sonuç etiket öncesi başarısızlığı ortadan kaldırmaz; tek etiketin mevcut durumu belirlediği birebir okuma varsayımına bağlıdır.

Eski parity, sıfırlama dışındaki bütün yöntemlerin bütün sorgularında doğru kaldı. Her şeyi sıfırlamak, 256-bit geçmişte 11/24, 4.096-bit geçmişte 9/24 yaşamda eski cevabı ters çevirdi. Eski hata oranları sırasıyla `0.458333` ve `0.375` olarak tüm devam boyunca kaldı. Yeni beceriye gelen etiket eski parity hakkında bilgi içermediği için bu kaybı düzeltmedi.

Bu eski-beceri başarısı, ayrı korunan koordinatın yapısal sonucudur. Paylaşılan bir temsilin bütün becerileri otomatik koruduğu veya sinir ağlarında unutmanın çözüldüğü iddia edilmez.

## Maliyet ve denetimde bulunan düzeltme

Tam arşiv 256 veya 4.096 ham bit tuttu ve yeni program kurulurken aynı sayıda biti yeniden işledi. Son-sekiz rezervi sekiz ham bit tuttu; gecikme kurulurken sekiz biti yeniden işledi. Modulo-6 rezervi altı değerli ek sayaç tuttu; basit sabit kodlamada üç bit yeterlidir. Eski parity ile arasındaki bilgi örtüşmesi nedeniyle bu, minimal ortak kodlama iddiası değildir.

Başlangıç belirsizliğini taşımak da bedelsiz değildir: 32 gecikme durumunu ilerletip birleştiren yöntem 64 yeni bitte 121 aday geçişi yaptı; tek durumla başlayan yöntem 64 geçiş yaptı. Üç modulo durumunu etikete kadar koruyan yöntem 96, tek durumlu yöntem 64 aday geçişi yaptı. Bunlar yalnız yeni çalışma koordinatının geçiş sayılarıdır. Eski koordinatın ilerletilmesi, rezervlerin her eski bitte güncellenmesi, program kodu, yorumlayıcı ve nesne ek yükü toplam CPU/RAM ölçüsünde ayrıca yer alır; burada tam sistem maliyeti ölçülmedi.

Bağımsız denetim, ilk çalıştırmada sekiz-bit rezervinin kurulum tekrarının maliyet sayacına eklenmediğini buldu. Sayaç düzeltildi; ilk kaynak ve sonuç [düzeltme kaydıyla](CORRECTIONS.md) korundu. Tahminler değişmedi. Kaynak değişirken devam eden ilk tekrar geçerli doğrulama sayılmadı; düzeltilmiş kaynak ve sonuç birlikte yeniden doğrulandı.

## Kuramsal ayrım ve literatür

Öğrenilmiş yeni geçişi `δ`, eski bilgiden mümkün yeni durumları `C₀` ile gösterelim. Görülen yeni dizi `w` sonrasında `C_w={δ*(q,w):q∈C₀}` olur. Sabit deterministik geçiş altında küme boyutu artmaz, fakat mutlaka bire düşmez. Gecikme geçişleri farklı eski durumları birleştirir; modulo geçişleri permütasyon olduğu için farklı durumları farklı tutar.

Eski durumu yeni yapıdan okuyan bir izdüşüm `π` için `π(δ'(s',b))=δ(π(s'),b)` ve doğru başlangıç izdüşümü sağlanırsa eski cevaplar bütün devamlar boyunca korunur. Bu koşulun tümevarımla kanıtı, belirsizlikle taşıma koşulları ve olasılık dağılımının karışmasıyla mevcut durumu öğrenmenin farkı [matematik notundadır](MATHEMATICAL_AND_LITERATURE_TR.md).

Küme halinde durum ilerletmenin [Rabin–Scott altküme yapısında](https://www.cs.miami.edu/~burt/learning/csc427.222/docs/rabin-scott.pdf), girişlerin durumları birleştirmesinin [Eppstein'ın senkronizasyon çalışmasında](https://www.ics.uci.edu/~eppstein/pubs/Epp-SJC-90.pdf), doğru modeli bilmek ile mevcut durumu belirlemenin ayrımının [Travers–Crutchfield çalışmasında](https://csc.ucdavis.edu/~cmg/papers/esfs.pdf) öncülleri vardır. İlgili tanımlar ve varsayımlar kontrol edildi. Bu hesaplamalara yeni ad vererek temel buluş iddia edilmedi.

## Yeniden üretim ve açık kalan ana problem

[Ham kayıt](../../../research/living_learning_migration_2026_09_22/results/migration_learning.json), [özet](../../../research/living_learning_migration_2026_09_22/results/summary.json), [doğrulama](../../../research/living_learning_migration_2026_09_22/results/verification.json) ve [son denetim](../../../research/living_learning_migration_2026_09_22/results/final_audit.json) yereldir. Denetimler: 576 yolun tam tekrarı; 9.216 kalibrasyon tahmini; 37.440 ana akış kaydı; `t=3` ve `t=16` sonrası 24 ayrı süreçte devam; 8.192 küçük geçmişin tam taranması; durum kapsama, etiket sırası, eski beceri ve hash kontrolleri. Önceki 977 dosya ve HEAD korunur. Kitap/ana sistem değişikliği, commit, push veya yayın yapılmadı.

Bu turdan ana probleme taşınabilen genel kısıt: **Yeni hesaplama kapasitesi edinildiğinde, o kapasitenin mevcut yaşam geçmişiyle tutarlı çalışmasını sağlayacak bilgi de gereklidir.** Kuralın doğruluğu tek başına yeterli değildir. Eski yetkinlikleri koruma ve yeni durumu oluşturma yükümlülükleri ayrı sınanmalıdır.

Deneyin aileye özgü sonuçları beş-bit gecikmenin beş adımda düzelmesi, modulo-3'ün bir etiketle belirlenmesi ve verilen rezervlerin başarısıdır. Her gelecekteki kavram dönüşümünün böyle olacağı söylenmiyor.

Geniş araştırmada açık kalanlar: deneyimin hangi yeni hesaplama biçimlerinin üretileceğini değiştirmesi; gelecekte öğrenmeye yarayacak ayrımların bütçeyle seçilmesi; edinilmiş kavram/işlem bileşimlerinin başka görevlerde öğrenmeyi hızlandırması; kaynak ve beceri çatışmalarında kontrollü genişleme; sistemin yararlı geri bildirimi kendi eylemleriyle bulması. Bu tur bunları bir senkronizasyon problemine indirgemedi; yalnız artık atlanamayacak bir süreklilik koşulunu kayda geçirdi.
