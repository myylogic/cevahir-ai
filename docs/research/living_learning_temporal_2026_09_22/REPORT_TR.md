# Tur 15 — Deneyimden bellek kullanımı ve gelecekte öğrenmeye yeterlilik

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel araştırma kaydı

## Sonucun kapsamı

Sistem, gerçek bir sembol akışını işlerken, yalnız bölüm sonundaki sonuçlardan bazı geçmiş ayrımlarını kullanmayı öğrenebildi. Son biti isteyen görevde iki yapraklı, sondan beşinci biti isteyen görevde 32 yapraklı bir bağlam ağacı seçildi. İki durumlu geçiş adayları arasından öğrenen ayrı yöntem, bütün dizinin tek/çift toplamını eğitimden uzun dizilerde de doğru sınıflandırdı. Başarı, önceden verilmiş bellek ailelerinin içinde gerçekleşti.

Aynı yöntemlerin birbirini ters yönde başarısızlığa uğratması belirleyici: sekiz-bit sonek ailesi sınırsız ufuklu parity kuralını temsil edemiyor; iki-durum ailesi gecikme-beş kuralını temsil edemiyor. İkinci yöntemde yanlış aile içinde öğrenmeye devam etmek, düşük hataya yakınsamayı sağlamadı. Uzun testte ortalama karesel hata başlangıçtaki `0.25` düzeyinden son değerlendirmede `0.470584` düzeyine çıktı.

**Ana yaşayan öğrenme problemi hâlâ çözülmedi.** Bu tur hazır geçmiş özetini kaldırıyor; temsil dilinin kendiliğinden üretilmesini, açık uçlu kavram oluşumunu, kesintisiz görev/temsil değişimini veya kendi kendine sınırsız gelişmeyi göstermiyor. Yeni bir temel algoritma keşfedildiği de ileri sürülmüyor.

## Önceki araştırmayla bağ

[Tur 14](../living_learning_context_2026_09_22/REPORT_TR.md), erişilebilir bağlam ve ayırıcı deneyim seçimini incelemişti. Oradaki geçmiş özeti hazır verilmişti. Burada aynı özet varsayımını sürdürmek yerine bitler sırayla okutuldu. [Tur 13](../living_learning_consolidation_2026_09_22/REPORT_TR.md) ile önceki program, ilişki ve temsil deneylerinin sonuçları yerinde duruyor. Bu turun başlangıcındaki 961 dosyanın içerik özeti korunuyor.

Yeni soru, geçmişin ne kadarının depolanacağı kadar, geçmişin **hangi hesaplamaya dönüştürüleceği** oldu. Buna ek olarak, bugünkü cevaba yeterli durumun yarınki öğrenmeye yeterli olup olmadığı matematiksel olarak ayrıldı.

## Deney

[Önceden yazılmış protokol](PROTOCOL.md): 24 eşleştirilmiş yaşam, üç görev, yedi yöntem; toplam 504 öğrenme yolu. Bunlar 504 bağımsız veri kümesi değildir. Aynı yaşamın girdileri bütün görev ve yöntemlerde kullanılır. Her yol 1.024 farklı eğitim bölümü içerir; toplam 516.096 etiket-öncesi tahmin kaydedildi.

Bölümlerde 8–24 bit sırayla gelir. Öğrenici cevabını verdikten sonra tek doğru etiket sağlanır. Ara etiket verilmez. Öğrenme parametreleri yalnız bu geri bildirim anında güncellenir; bölüm içindeki çalışma durumu her bit geldiğinde ilerler. Bölüm başlangıcında çalışma durumu sıfırlanır, öğrenilmiş sayaçlar/ağırlıklar korunur.

Her yaşamda 128 görülmemiş kısa bölüm ve 128 uzun bölüm (32–64 bit) vardır. Bunlar öğrenmeye girmez. Uzunluk ve bitler eşit olasılıkla önerilir, tekrar sözcükler reddedilir; kabul edilen örnekler tam bağımsız bir örneklem değildir. Beş kontrol noktasında dondurulmuş model kopyalarıyla değerlendirme yapıldı.

Yöntemler: girdisiz KT sayacı; bir, beş ve sekiz bitlik sabit sonek sayaçları; en çok sekiz derinlikte MAP bağlam ağacı; aynı ağaç ailesinin tam karışımı; 64 hazır iki-durumlu geçiş/çıktı tablosunun ağırlıklı karışımı. Görev adı öğreniciye verilmedi. Maksimum ağaç derinliği, iki-durum sınırı, güncelleme kuralı ve aday aileler verildi.

## Uzun dizilerde son sonuçlar

Aşağıdaki değerler 1.024 geri bildirim sonrası 24 yaşamın ortalama skaler MSE'sidir; küçük değer daha iyidir. Her hücrede toplam 3.072 uzun test sorgusu vardır. Her zaman `0.5` cevabını veren sistemin hatası tam `0.25` olur.

| Yöntem | Son bit | Sondan beşinci bit | Bütün dizinin parity'si |
|---|---:|---:|---:|
| Girdisiz sayaç | 0.250189 | 0.250134 | 0.250231 |
| Son 1 bit | 0.000000952 | 0.250008 | 0.250606 |
| Son 5 bit | 0.000249592 | 0.000249592 | 0.257862 |
| Son 8 bit | 0.019317581 | 0.019317581 | 0.289163 |
| Öğrenilen MAP ağacı | 0.000000952 | 0.000249592 | 0.250231 |
| Ağaçların tam karışımı | 0.000000978 | 0.000301700 | 0.250234 |
| İki-durumlu aday karışımı | Sayısal sıfır | 0.470584 | Sayısal sıfır |

“Sayısal sıfır”, sonlu hassasiyetli hesaplamanın kaydıdır. Pozitif olabilirlikli yanlış adayların matematiksel posterior ağırlığının sonlu örnekte tam sıfır olduğu söylenmiyor; bazı değerler makine hassasiyetinin altına düşer. Son bit ve parity için iki-durumlu yöntem uzun testte 24 yaşamın tamamında %100 sınıflandırma doğruluğuna ulaştı. Bu örneklem sonucu bütün olası çevreler için bir sertifika değildir.

MAP ağacı bütün yaşamlarda son-bit görevinde derinlik 1 ve 2 yaprak; gecikme-beş görevinde derinlik 5 ve 32 yaprak seçti. Her iki görevde bütün uzun test sınıflandırmaları doğruydu. Sabit sekiz-bit tabloya göre MSE her iki görevde 24 yaşamın tamamında daha düşüktü. Gecikme-beşte doğru uzunluğu baştan verilen beş-bit tabloyla son tahminleri eşleşti. Böylece faydalı ayrım derinliği verilmiş üst sınır içinden edinildi; daha büyük tablo otomatik daha iyi öğrenme sağlamadı.

Parity görevinde MAP ağacı her yaşamda tek kök yaprakta kaldı. Uzun test doğruluğu %50.39, MSE `0.250231` oldu. Sabit sekiz-bit tabloda doğruluk %50.55, MSE `0.289163` idi. Daha çok sonek ayrımı bu görevde yararlı önek bilgisinin yerini tutamadı.

İki-durumlu yöntem gecikme-beşte %51.95 uzun test doğruluğuyla kaldı. MSE'nin yüksek olması, yanlış tahminlerin çoğunun aşırı kesin hale gelmesiyle uyumludur. Aday ağırlıklarından hesaplanan etkin aday sayısı ortalama `1.8867`'ye indi; aile daralırken gerçek kural ailenin dışında kaldı. Bu deney, ailenin yanlışlığını fark edip genişletme mekanizması içermiyor.

## Öğrenmenin zaman içindeki maliyeti

Gecikme-beş görevinde uzun test MSE:

| Alınan etiket | MAP ağacı | Ağaç karışımı | İki-durumlu karışım |
|---:|---:|---:|---:|
| 0 | 0.250000 | 0.250000 | 0.250000 |
| 32 | 0.256645 | 0.247308 | 0.422312 |
| 128 | 0.046826 | 0.053830 | 0.476094 |
| 512 | 0.001017 | 0.001546 | 0.458616 |
| 1024 | 0.000250 | 0.000302 | 0.470584 |

Karışım her ölçüde ve her zamanda daha iyi çıkmadı. Gecikme-beşte ilk 128 eğitim bölümünün **cevap verildikten sonra ölçülen** MSE'si MAP için `0.232339`, tam karışım için `0.177188` idi; bu eğitim tahminleri etiketten önce üretilmişti. Son uzun testte MAP'ın MSE'si daha düşüktü. Erken belirsizlik ve son uzmanlaşma farklı karşılaştırmalardır.

Parity'de iki-durumlu yöntemin uzun test MSE'si 32 etiketten sonra ortalama `4.1284e-17`; doğruluğu bütün yaşamlarda %100 oldu. Bu kuvvetli sonuç doğru iki-durumlu algoritmanın küçük ve hazır aday ailesinde bulunmasına bağlıdır.

Bir öğrenme yolu için ağaç yöntemleri 9.216 sayaç güncellemesi ve 9.216 düğüm yenilemesi yaptı. Her biri 511 düğümün istatistiklerini baştan sona tuttu; sayısal diziler 20.951 bayt. Seçilen iki yaprak, bütün öğrenicinin iki durum veya iki sayaç kadar yer kapladığı anlamına gelmez.

İki-durumlu yöntem ortalama 1.051.210,67 aday-durum geçişi ve 65.536 aday-etiket karşılaştırması yaptı; öğreniciye ait sayısal diziler 1.040 bayt. Bunun yanında kodda ortak geçiş/çıktı tabloları bulunur. Model dili, kod, Python nesne ek yükü ve işletim sistemi belleği bu dizi ölçüsünün dışındadır. Ölçüler eşit işlem süresi karşılaştırması değildir; değerlendirme sorgularının maliyeti eğitim sayaçlarına eklenmemiştir.

## Matematiksel sonuç

[Ayrıntılı kanıtlar](MATHEMATICAL_AND_LITERATURE_TR.md) iki ayrı boyutu ayırıyor:

- Parity'nin çok eski bitlere bağımlılığı sınırsızdır, fakat doğru çalışma durumunu iki değerle taşımak yeterlidir.
- Gecikme-beş yalnız beş-bit ufka bağlıdır, fakat bütün gelecek devamlar için en az 32 deterministik çalışma durumu gerektirir.
- Aynı son sekiz bite sahip `000000000` ile `100000000` farklı parity sonuçları gerektirir. Aynı eğitimli modelden başlayan bir sonek sorgulayıcısı için bu çiftte ortalama MSE en az `1/4` olur.
- Sonlu gözlemlerle uyumlu olmak, sınırsız gelecekte doğruluk kanıtı değildir. Eğitimin dışında tek bir uzun sözcükte parity'den ayrılan sonlu otomata, bütün eğitim sonuçlarıyla uyumlu kalabilir.

[Ek matematik notu](LEARNING_STATE_TR.md) ana problem için farklı bir gerekliliği açıyor: **şimdiki davranışı korumak ile gelecekteki öğrenme tepkisini korumak farklıdır.** Aynı `0.5` cevabını veren, fakat farklı miktarda kanıt toplamış iki sayaç, aynı yeni deneyimden sonra `0.75` ve `0.5025` cevaplarına geçebilir. Yalnız bugünkü cevabın saklanması öğrenme devamını korumaz.

Ayrıca eski parity özetinden bölüm ortasında `(parity, son beş bit)` özetine kayıpsız geçiş için, genel durumda beş ek bitlik ayrımın önceden korunması gerektiği gösterildi. Sonradan yeni temsil kurmak, daha önce kaybedilmiş bilgiyi kendiliğinden geri getirmez. Bu koşullu sayma sonucu, beynin mekanizmasını veya hazır bir genel mimariyi kanıtlamaz.

## Literatür ve yenilik sınırı

Ağaç karışımının düğüm hesabı [Willems, Shtarkov ve Tjalkens'in CTW çalışmasıyla](https://pure.tue.nl/ws/portalfiles/portal/1383848/Metis122608.pdf) ilişkilidir. Burada dış bölüm sonekine koşullu terminal etiketler öğrenilir; özgün kaynak kodlama sınırları MSE başarısı gibi sunulmadı. Geçmişleri gelecek sonuçlarına göre eşdeğer saymanın [hesaplamalı mekanikte](https://csc.ucdavis.edu/~cmg/papers/cmppss.pdf) ve [öngörü durum temsillerinde](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) açık öncülleri vardır. Kaynakların varsayımları matematik notunda ayrıştırıldı.

Bu turun katkısı proje içindeki varsayımı sınamak, karşılaştırmaları gerçek akışta yürütmek ve genel amaç için eksik kalan koşulları görünür kılmaktır. Bilinen ailelerin küçük deneyde işe yaraması yeni buluş veya genel zekâ başarısı sayılmadı.

## Denetim ve koruma

[Doğrulama kaydı](../../../research/living_learning_temporal_2026_09_22/results/verification.json): 504 yolun tam tekrarı; 516.096 kayıtlı tahminin yeniden yürütülmesi; 2.520 değerlendirme anı; 21 ayrı süreçte kesintisiz nesneyle aynı öğrenme devamı. Küçük derinliklerde bütün ağaçları ayrı ayrı sayarak 168 kontrol; otomataları skaler olarak yürüterek 32.640 çıktı kontrolü. Testlerin parametreleri değiştirmediği, etiket sırası, kopya izolasyonu ve görev ailesi karşıörnekleri sınandı.

[Kuramsal tanıkların denetimi](../../../research/living_learning_temporal_2026_09_22/results/learning_state_verification.json) 100 eşit-bugünkü-cevap durumunun yeni etiketten sonra ayrıldığını tam kesirlerle kontrol etti; ortak parity/gecikme durumları için 64 geçmişin 2.016 çiftini ayırdı. Sonlu denetimler metindeki genel kanıtların yerine geçmez.

[Ham sonuç](../../../research/living_learning_temporal_2026_09_22/results/temporal_learning.json), [yeniden hesaplanan özet](../../../research/living_learning_temporal_2026_09_22/results/summary.json) ve [son bütünlük denetimi](../../../research/living_learning_temporal_2026_09_22/results/final_audit.json) yereldir. Kaynak/protokol özetleri, eski 961 dosya ve HEAD denetlenir. Kitap veya ana sisteme entegrasyon, commit, push ve yayın yapılmadı.

## Ana problemin kalan kısmı

Genel araştırma için bu turdan kalan temel kısıtlar: kalıcı öğrenilmiş durum geçici çalışma durumundan ayrılmalı; yeni deneyime verilecek öğrenme tepkisi için gerekli ayrımlar korunmalı; temsil değişiminde kaybolmuş bilginin bedeli hesaba katılmalı; bir aday aile içinde eminleşme, ailenin yeterliliğinin göstergesi sayılmamalı.

Belirli ailelere ait sonuçlar ise iki-durumlu otomataların parity başarısı, sekiz-derinlik ağaçların gecikme başarısı ve bu sayısal maliyetlerdir. Bunlar kavramların bütün yaşam boyunca gelişmesini tek başına açıklamıyor.

Bir sonraki araştırmanın ayırıcı sorusu: **Görev veya gerekli temsil değiştiğinde sistem, hem eski yetkinliklerini sürdürüp hem yeni yetkinliğe geçmesini sağlayacak hangi ayrımları koruyacağını nasıl öğrenebilir?** Hazır ailelerden birini seçmek bir karşılaştırma olarak tutulmalı; öğrenicinin aileyi genişletmesi, mevcut durumunu dönüştürmesi ve bunu kaynak/başarı hesabıyla denetlemesi ayrıca gösterilmelidir. Geniş yaşayan öğrenme problemi bu sorulardan yalnız birine indirgenmemelidir.
