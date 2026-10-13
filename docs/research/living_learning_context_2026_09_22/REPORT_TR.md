# Tur 14 — Bağlamı deneyimle ayırmak ve yeni deneyimi seçmek

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Yerel araştırma kaydı

## Sonuç ve araştırmanın ana hedefi

Tur 13'te doğru bağlam kimliği hazır verilmişti. Bu turda normal öğreniciye bu kimlik verilmedi: hazır dört gözlenebilir geçmiş biti, bir yüzey işareti ve mevcut iş girdisinden sonuç üreten yapılar birlikte öğrenildi. Başlangıçta geçmiş ilişkisiyle yüzey işareti her zaman örtüştüğünde, başarılı eğitim tahmini doğru açıklamayı seçmeye yetmedi. İpuçlarını ayıran yeni gözlemlerle yanlış aktarım düzeltilebildi. Ek deneyde sistem, hangi yeni girdiyi sorgulayacağını kendi mevcut belirsizliğiyle seçti; dış gözlem başına kazanım arttı, fakat seçim hesabı çok daha pahalıydı.

**Bu, yeni temel kuram veya yeni bir öğrenme ilkesi bulunduğu iddiası değildir.** Hazır adayların olasılıklarını güncelleme ve belirsizliğe göre sorgu seçme bilinen yaklaşımlardır. Kullanıcının asıl hedefi, çalışan sistemin temsilini, kullanabileceği hesapları ve sonraki öğrenme yollarını deneyimle geliştirmesidir. Bu hedef [sürekli sistem notunda](CONTINUOUS_SYSTEM_TARGET_TR.md) korunuyor; mevcut sinir ağına ek yordamlar takmakla sınırlandırılmıyor. [Kuramsal yenilik sınırı](THEORY_NOVELTY_BOUNDARY_TR.md), DreamCoder, Gödel machines ve öğrenilmiş optimizasyonla eşleşmeleri ve açık kalan gereksinimleri ayırır.

## Kurulum ve hazır verilenler

Girdi `(h,s,x)`: geçmiş özeti olarak yorumlanan dört bit, tek yüzey biti, dört iş girdisi biti. **h, gerçek akışın önceki dört olayından çıkarılmıyor;** örnekler arasında kayan zaman penceresi yok. Geçmişi seçme, algıdan kavram oluşturma veya temsil dilini keşfetme bu deneyde yapılmadı.

Gerçek bağlam h'nin iki/üç bitinin paritesi; iki iş kuralından hangisinin kullanılacağını belirliyor. Her kural x üzerinde bir/iki bitli parite veya tersi. Öğreniciye doğru bağlam, doğru parite veya uzman kimliği verilmiyor. Ancak 15 geçmiş-parite kapısı, bir yüzey kapısı ve her dal için 32 aday iş kuralı baştan veriliyor. Böylece 16.384 gösterim, 15.904 farklı tüm-girdi işlevini kapsıyor. Aynı işlevin birden fazla gösterimi bulunması ön olasılığa etki eder.

24 yaşamın ilk 48 farklı girdisinde geçmiş ve yüzey ipuçları aynı şeyi gösteriyor. Sonraki 16 gözlemde ipuçları hep aynı, yarı yarıya farklı veya hep ters olacak şekilde üç devam koşulu var. Doğru işlev değişmiyor; gözlenen girdi dağılımı değişiyor. Öğrenici bölüm sınırını ve koşul etiketini almıyor. Yedi yöntemle **504 edinim yolu**, **4.032 durum değerlendirmesi**, **32.256 etiket-öncesi tahmin** oluştu. Aynı yaşamın ölçümleri eşleştirilmiştir; bağımsızlık birimi 24 yaşamdır.

Sonuç öncesi [ana protokol](PROTOCOL.md) ve [matematik denetimi](MATHEMATICAL_AUDIT_TR.md) ayrı tutuldu. Olasılıksal yöntemler her etiket için uyuşmaya 0,95, uyuşmazlığa 0,05 olabilirlik verir. Gerçek etiketler temizdir; bu yüzde beş, öğrenicinin önceden seçilmiş hata varsayımıdır. Tahmin, temiz aday çıktılarının ağırlıklı ortalamasıdır. Hata ölçüsü MSE `(p−y)²`; sıfır iyi, kesin yanlış bir.

## 1. Eğitimde doğru görünmek, doğru bağlam açıklamasını bulmak değildir

İlk 48 hizalı gözlemden sonra yüzey kapısını kullanan yöntem hizalı girdilerde neredeyse hatasızdır. Yüzey işareti terslendiğinde ortalama hatası 0,5 olur. Olayların bütün anahtarını saklayan tablo, görülmemiş girdilerde 0,25 hata verir.

| Yöntem | İlk 48 sonrası ters-ipucu MSE | 8 hizalı + 8 ters ek gözlem sonrası ters-ipucu MSE | 16 ters ek gözlem sonrası ters-ipucu MSE |
|---|---:|---:|---:|
| Tam girdi tablosu | 0,250000 | 0,242188 | 0,234375 |
| Yalnız iş girdisi x'i kullanan tablo | 0,179817 | 0,163687 | 0,165193 |
| Yalnız yüzey kapısı | 0,500000 | 0,500000 | 0,499905 |
| Yalnız geçmiş kapısı ailesi | ≈0 | ≈0 | ≈0 |
| Geçmiş+yüzey, kapılara eşit ön olasılık | 0,125000 | 0,000053 | ≈0 |
| Geçmiş+yüzey, iki aileye eşit ön olasılık | 0,439453 | 0,004155 | ≈0 |
| Doğru kapı verilmiş ayrıcalıklı kontrol | ≈0 | ≈0 | ≈0 |

`≈0`, yumuşak ağırlıklar ve kayan nokta hesabında sıfır olduğu değil, bu tabloda altı ondalıkta sıfıra yuvarlandığı anlamına gelir. Ayrıntılı değerler ham sonuçlarda duruyor. Son iki sütunun ters-ipucu kümesi yeni gözlenmiş örnekleri de içerir; aşağıda ayrıca hiç görülmemiş girdiler raporlanır.

Yalnız geçmiş ailesindeki yöntemin iyi olması, bağlam açıklamasını veriden tek başına bulmakla aynı değildir: bu yönteme yüzeyin ilgili olmadığı önbilgisi verilmiştir. Gerçek geçmiş kapısının hangisi yine öğrenilir; fakat iki açıklama ailesi arasındaki belirsizlik baştan kaldırılmıştır. Oracle yöntemi daha da ayrıcalıklıdır; doğru kapı doğrudan verilmiştir.

Geçmiş ve yüzey birlikte açık bırakıldığında ilk 48 örnek onları ayıramaz. Daha fazla **aynı türde** hizalı veri vermek de sorunu çözmez: 64'ün tamamı hizalıyken ters-ipucu MSE değerleri yine 0,125000 ve 0,439453'tür. Bu, yalnız az örnek sorunu değildir.

## 2. Aynı adaylar ve aynı veriyle ön olasılığın etkisi

İki ortak model, aynı 16.384 gösterimi ve aynı güncellemeyi kullanır. İlki 16 kapıya eşit ağırlık verir. İkincisi “geçmiş ailesi” ve “yüzey ailesi”ne eşit toplam ağırlık verir; geçmiş ağırlığı 15 kapıya bölünür.

Bu nedenle belirli doğru geçmiş açıklamasının, aynı uzmanlara sahip yüzey açıklamasına başlangıç oranı ilkinde **1**, ikincisinde **1/15**. Hizalı kayıtların olabilirlikleri aynı olduğundan bu oranlar öğrenimle değişmez. İlk 48 sonunda doğru geçmiş kapısının toplam ağırlığı yaklaşık %50 ve %6,25'tir. “İki aileye eşit yaklaşma”, onların içindeki belirli açıklamalara eşit yaklaşma değildir.

Dolayısıyla aynı gözlem başarısından aynı anlam örgüsünün öğrenildiği çıkarılamaz. Bu olgu bilinen ön olasılık/ayırt edilebilirlik sorunudur; yeni bir bilimsel ilke diye sunulmuyor. Ayrıca tüm doğru ağırlıklar gerçek dünya ve seçilmiş aday diline koşulludur; model dışında bir dünya olasılığı bu deneyde öğrenilmez.

## 3. Ayırt edici deneyim, henüz görülmemiş girdileri de değiştirdi

Yarı hizalı devam koşulunda, 16 yeni gözlem sonunda henüz görülmemiş 448 girdide MSE:

- Kapılara eşit ön olasılık: **0,068966 → 0,0000298**.
- Ailelere eşit ön olasılık: **0,242457 → 0,0023553**.

Sol değerler ilk 48 sonrasındaki 464 görülmemiş girdiye, sağ değerler toplam 64 sonrasındaki 448 görülmemiş girdiye aittir; test altkümelerinin sayısı farklıdır. Aynı sabit ters-ipucu kümesindeki sonuçlar önceki tabloda verilmiştir. İki modelin son görülmemiş-girdi sınıflama doğruluğu %100'dür; olasılık hataları hâlâ farklıdır. Yalnız doğruluk ölçmek bu farkı gizler.

Tam ters devam koşulunda son görülmemiş MSE sırasıyla `3,15×10⁻¹⁷` ve `6,95×10⁻¹⁵`. Bu yüksek başarı, gerçek kuralın baştan verilen sonlu dilde bulunmasına ve gelen örneklerin iki açıklamayı ayırmasına dayanır. Sıfırdan yeni temsil üretme değildir.

Her ters yüzey örneği ayırt edici değildir. İki uzmanın x için cevabı da farklı olmalıdır. O zaman doğru geçmiş/yüzey aday çiftinin ağırlık oranı, her yeni kayıtla tam **19** kat değişir:

\[
\frac{q(H\mid D,e)}{q(S\mid D,e)}
=\frac{q(H\mid D)}{q(S\mid D)}\frac{0,95}{0,05}.
\]

Bu, belirli aday çifti için kanıtlanmış eşitliktir; bütün geçmiş ailesinin toplam ağırlığına doğrudan uygulanmaz. Ana yöntemlerde tahmin ve bu oran etiketten önce/sonra doğru sırayla kaydedildi.

## 4. Sistemin yeni deneyimi seçtiği ek sınama

Kullanıcının “program çalışırken kendi kendine yeni şeyler öğrenme” hedefi doğrultusunda, ana sonuçlar hesaplanmadan [ayrı aktif deney protokolü](ACTIVE_EXPERIENCE_PROTOCOL.md) yazıldı. İlk 48 kayıt sonrasında iki ortak model, ya görülmemiş girdilerden rastgele sorgu seçer ya da kendi tahmin ayrışması `p(1−p)` en yüksek girdiyi seçer. **Etiketi seçimden sonra dış ortam verir.** Sistem kendi cevabını doğru etiket saymaz.

24 yaşam × iki ön olasılık × iki seçim politikası = **96 ek edinim yolu**, 1.536 yeni dış etiket. Ortam, 512 girdiden istenen herhangi birini doğru etiketle sorgulamaya izin verir. Bu güçlü arayüz verilmiştir; gerçek dünyada geçmişi keyfî biçimde oluşturabilme veya kendi ölçüm aracını keşfetme gösterilmez.

Aşağıdaki MSE değerleri aynı sabit 512-girdi küpündedir; yeni sorgulanan noktaları da kapsar. Görülmemiş-altküme sonuçları ayrı kaydedildi ve eşleştirilmiş avantaj orada da bulundu.

| Ön olasılık ve seçim | Yeni sorgu 0 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|
| Kapılara eşit; rastgele sorgu | 0,062500 | 0,047031 | 0,0289584 | 0,00809975 |
| Kapılara eşit; ayrışmaya göre sorgu | 0,062500 | 0,000625 | 0,00000191 | `1,47×10⁻¹¹` |
| Ailelere eşit; rastgele sorgu | 0,219727 | 0,176960 | 0,1250542 | 0,04993372 |
| Ailelere eşit; ayrışmaya göre sorgu | 0,219727 | 0,048659 | 0,00039788 | `3,31×10⁻⁹` |

Dört yeni sorguda her iki ön olasılık için ayrışmaya dayalı seçim, rastgele kontrole göre **24 yaşamın tamamında** daha düşük tüm-küp ve görülmemiş-girdi MSE verdi. Birinci sorguda her yaşamda üstün değildi; ilk ön olasılıkta üç, ikinci ön olasılıkta altı yaşamın farkı ters yöndeydi. İkinci ön olasılıkta ikinci sorguda da bir yaşamda rastgele yöntem daha iyiydi. Ortalama kazanım, her tek adımda kesin üstünlük değildir.

Bu sonuç yeni bir aktif öğrenme yöntemi keşfi değildir. Hazır adayların ayrışmasını sorgu seçimine dönüştüren bilinen bir ilkenin, bu sürekli güncellenen sonlu modelde sınanmasıdır. Buluş hedefiyle karıştırılmadan korunmalıdır.

## 5. Dış örnek verimliliğinin hesaplama bedeli

| Yöntem | 64 etiket için aday–etiket kontrolü | Hizalı koşulda son öğrenici JSON'u, ortalama bayt |
|---|---:|---:|
| Tam tablo | 0; 64 sayaç güncellemesi | 3.186 |
| x tablosu | 0; 64 sayaç güncellemesi | 208 |
| Yüzey modeli | 65.536 | 22.644,5 |
| Geçmiş ailesi | 983.040 | 465.166,25 |
| Ortak, kapılara eşit | 1.048.576 | 398.611,75 |
| Ortak, ailelere eşit | 1.048.576 | 488.726,75 |
| Doğru kapı verilmiş | 65.536 | 22.023,5 |

JSON boyutları kullanılan sayı gösterimlerini de içerir; minimal bellek veya tam süreç RAM'i değildir. Deney süreci ayrıca **67.108.864 baytlık** ortak hazır aday-çıktı bankasını tutar. Bu bankanın tablo yöntemleri için zorunlu bellek olduğu iddia edilmez; gerçekleştirim ortak bankayı süreç başında üretir. Ham olay listeleri öğrenici durumunda yoktur, fakat tablo yönteminin sayaçları olay bilgisini korur.

Aktif ekin 16 sorgusunda iki seçim politikası da aynı 1.048.576 aday–etiket kontrolüne ulaşır. Bunun dışında rastgele yöntemin seçilen girdilerin tahminini hesaplaması 262.144 aday–girdi hücresi; ayrışma yöntemi her seçimde bütün 512 girdiyi değerlendirdiği için **134.217.728 hücre** işler. Bu ikincil sayaçta **512 kat** fark var. Bu toplam CPU oranı değildir; seçim ve etiket-öncesi tahmin maliyetlerinin açık bir bileşenidir. Sonuç dış gözlem verimliliğidir; toplam hesaplama veya enerji üstünlüğü gösterilmedi.

## 6. Matematiksel sınır: aynı deneyim, farklı gerçek dünyalar

Geçmişe göre işleyen H dünyasıyla yüzey ipucuna göre işleyen S dünyası, bütün hizalı kayıtlar için aynı y'yi verir. Hizalı 256 girdinin tamamı gözlense bile bu iki dünya ayırt edilemez. Yüzey bağımsızken dünyalar girdilerin 1/4'ünde, yüzey tersken 1/2'sinde farklı doğru cevap ister.

Bu iki olası dünya eşit ağırlıklıysa, tek tahmin kullanan sistemin ortalama MSE alt sınırı bağımsız-yüzeyde **1/16**, ters-yüzeyde **1/8**. Bu, tek bir bilinen gerçek dünya için zorunlu hata değildir: geçmişi baştan doğru varsayan yöntem H dünyasında sıfıra yaklaşabilir. Kanıt, aynı veriden iki dünyada birden kusursuz garanti çıkarılamayacağını söyler.

Bu sınır, iç hesaplamanın yararsızlığı değildir. Deneyim seçen sistem gerçekten yeni ve ayrıştırıcı dış etiket alınca bilgi durumu değişir; alt sınırın “aynı bilgi” koşulu artık geçmez. Mevcut bilginin işlenmesi ile yeni dış bilginin edinilmesi ayrı süreçlerdir.

## 7. Yeni bir buluş hedefi bakımından ne kaldı?

Kullanıcının son yönlendirmesi araştırma kaydında şudur:

> bunların tamamı için gerekirse yeni bir kuram ve matematik formulleri türetmeliyiz ve mevcut yapay zeka literatürüne farklı bir perspektiften bakabilmeliyiz. günün sonunda sinir ağlarının çalışma prensiblerini yamalamak gibi bir niyetim yok günün sonunda bambaşka bir buluş yapmak niyetimiz var.

Bu hedef korunuyor. Sinir ağı, embedding, parametre güncellemesi, Bayes veya aktif sorgu seçimi zorunlu merkez kabul edilmiyor. Fakat aynı işlevsel değişimin farklı veri yapılarında ifade edilmesi tek başına yeni hesaplama ilkesi değildir. Bir aday kuramın, alternatiflerin açıklayamadığı hangi farkı açıkladığı gösterilmeli.

Çalışma gösterimi `z=(R,P,U,Q,m)`; temsil, çalıştırılabilir işlemler, güncelleme yordamı, deneyim/hesap seçimi ve diğer kalıcı durumu ayırır. Bu yalnız kapsamı koruyan bir tanımdır, yeni kuram değildir. Tur 14'te R/P/U ve seçim kuralının biçimi hazır; yalnız bunların içindeki inanç durumu değişiyor. Aktif sorgu kararı değişen inançtan etkileniyor, fakat yeni bir sorgu dili veya güncelleme kuralı oluşmuyor.

Bu nedenle sonraki kuramsal yükümlülük **verilen adayların içinden doğruyu seçmeyi aşan, deneyimden kullanılabilir bir hesaplama yapısı oluşturmanın ne olduğudur**. Önceki program-kütüphanesi turu bu yönde sınırlı erişilebilirlik kazanımı göstermiş ve olumsuz aktarım bulmuştu; geri alınmadı. Yeni iddia o sonuçları tekrar adlandırmamalı. Hazır dilde büyüme, gerçekten yeni ayrım/işlem edinimi, sonraki öğrenme yordamının değişimi ve hesaplama bütçesi ayrı kontrol edilmelidir. [Yenilik notu](THEORY_NOVELTY_BOUNDARY_TR.md) bu eşiği somutlaştırır.

Mevcut bulgular H42'yi yalnız hazır geçmiş/dil içinde; H43'ü aynı bilgiyle ayırt edilemezlik bakımından; H44'ü yeni etiketlerle kalıcı ve eğitim-dışı düzeltme bakımından; H45'i ön olasılık farkıyla destekler. Hiçbiri temsil dilinin kendiliğinden doğuşunu veya genel yapay zekâyı kanıtlamaz.

## Doğrulama ve kalıcılık

8.388.608 aday–girdi değeri bağımsız skaler bit hesabıyla kontrol edildi; 15.904 farklı işlev sayıldı. 72 dünya koşulu, ortak ilk 48 kayıt, devam örneklerinin oranları, gerçek uzmanların yarı girdide farklılaşması, iki ön olasılık ve aday-çifti için 19 katsayısı denetlendi. Normal öğreniciye gizli dünya kimliği verilmediği arayüz ve kod incelemesiyle kontrol edildi.

Ana deney tam tekrarlandı. 21 ayrı süreçte kesintisiz nesneyle aynı sonraki tahmin ve güncelleme elde edildi. Aktif deneyde 1.536 seçimin tekrar üretimi ve dış etiketleri, 768 ayrışma seçiminin bağımsız en-yüksek puan hesabı denetlendi. Dört ayrı süreçte öğrenici, görülmüş-girdi maskesi ve rastgelelik durumu yüklenerek **aynı sonraki sorguların seçilmesi** de doğrulandı. Aktif deney ayrıca tam tekrarlandı.

Önceki **941 dosyanın** içerikleri ve Git HEAD korundu. Kitap ve ana sistem değişmedi; commit, push veya DOI yayını yapılmadı. [Tekrar üretim ve dosyalar](../../../research/living_learning_context_2026_09_22/README.md); [son denetim kaydı](../../../research/living_learning_context_2026_09_22/results/final_audit.json). Yerel doğrulama dış hakem veya bağımsız araştırma tekrarı değildir.
