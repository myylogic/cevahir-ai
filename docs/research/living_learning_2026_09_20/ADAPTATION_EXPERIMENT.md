# Çevrimiçi uyum, öğrenmeyi öğrenme ve olumsuz aktarım

**Tarih:** 20 Eylül 2026. **Durum:** Çalıştırılmış küçük sentetik deney; genel zekâ veya Cevahir ölçeğinde sonuç değildir.

Bu deneyin olumlu sonucu şudur: Bir sistem, önceki görevlerde etiketleri işlerken güncellediği üç sayılık kalıcı bir durum sayesinde, daha önce karşılaşmadığı aynı aileden görevlerde daha iyi tahmin yapabilir. Aynı mekanizma görev ailesi değişince zarar verir. Tam doğru fonksiyonu kanıtla belirlemek için gereken etiket sayısı ise değişmez. Dolayısıyla “öğrenmeyi öğrenme” iddiası, hangi başarı ölçütünden söz edildiğini belirtmek zorundadır.

Bu çalışma yeni bir algoritma iddia etmez. Sonlu hipotezler üzerinde Bayes koşullandırması ve görevler arası aile önselinin güncellenmesi kullanılır. Amaç, yaşam sırasında değişen durumun neyi başarabildiğini ve hangi güçlü iddiaları desteklemediğini ayrıştırmaktır.

## 1. Tam olarak ne değişiyor?

Girdi uzayı altı bitlik vektörlerden oluşur: `x ∈ {0,1}^6`. Bir görev, aşağıdaki 128 afin Boole fonksiyonundan biridir:

\[
f_{a,b}(x)=(a^\top x+b)\bmod 2,
\quad a\in\{0,1\}^6,\ b\in\{0,1\}.
\]

Hipotezler üç bilinen aileye ayrılır: `a` içindeki 1 sayısı 1–2 olan seyrek aile, 4–5 olan yoğun aile ve kalanlar. Aile boyutları sırasıyla 42, 42, 44'tür. Aile kavramını sistem keşfetmez; bu bir tasarımcının verdiği endüktif varsayımdır.

Tek görev içinde canlı güncelleme şöyledir:

\[
V_{k+1}=\{h\in V_k:f_h(x_k)=y_k\},
\qquad
P_k(h)=\frac{P_\eta(h)\mathbf{1}[h\in V_k]}
{\sum_{h'\in V_k}P_\eta(h')}.
\]

Sistem önce etiket olasılığını üretir; etiketi gördükten sonra sürüm uzayını, yani verilerle tutarlı hipotez kümesini, değiştirir. Etiket üreticisinin gerçek fonksiyon kimliği öğreniciye doğrudan verilmez. Bir görev, etiketler sayesinde tek bir hipoteze indirgenince görevler arası durum güncellenir:

\[
\alpha_{g(h)}\leftarrow\alpha_{g(h)}+1,
\qquad
P_\eta(h)=\frac{\alpha_{g(h)}}{\sum_j\alpha_j}\frac{1}{|H_{g(h)}|}.
\]

Başlangıçta `α = (42/128, 42/128, 44/128)` seçilir. Böylece başlangıç hipotez önseli tam tekdüzedir ve toplam başlangıç sözde gözlem sayısı 1'dir. Aile içinde dağılım tekdüze kalır. Önceki görevlerin fonksiyon kimliklerini yeni görevde geri getiren bir liste kullanılmaz; üç aile sayacı sonraki görevdeki hipotez ağırlıklarını değiştirir. Aynı yeni etiket geçmişi iki sistemde farklı tahminlere yol açabilir, çünkü önceki yaşamlarından taşınan `α` değerleri farklıdır.

Çalışma, etiket geldikçe güncelleme yapar. Ayrı bir toplu eğitim işi, model dağıtımı veya Cevahir çalışma zamanını durdurma işlemi yoktur. Meta-test sırasında aile sayaçları son meta-eğitim anındaki halleriyle dondurulur; **görev içi öğrenme devam eder**. Bu tercih deneysel ölçümü ayırmak içindir: Test görevlerinin birbirini eğitmesine izin verilmez. Dolayısıyla bu protokol, sonsuz tek bir akış üzerindeki tüm yaşam davranışını ölçmez.

## 2. Karşılaştırma, bütçe ve bilgi sızıntısı kontrolleri

Her biri `seed=0,…,39` ile belirlenen 40 tekrar çalıştırıldı. Çift seed'lerde seyrek, tek seed'lerde yoğun aile meta-eğitim ailesidir. Böylece yalnız bir yöndeki aile değişimine dayanan sonuç üretilmez.

Her tekrar:

1. Aynı aileden 24 farklı görevden çevrimiçi deneyim edinir. Görevi tekilleştirmek için rastgele sıralanmış girdiler etiketlenir. Tekrar başına ortalama 194,4 etiket kullanılır.
2. Daha önce görülmemiş aynı aileden 8 görevde ölçülür.
3. Karşı aileden, daha önce görülmemiş 8 görevde ölçülür.

Her tekrar içinde meta-eğitim, aynı aile testi ve farklı aile testi fonksiyon kimlikleri birbirinden ayrıdır. Her test görevinde rastgele sıralanmış 64 girdinin ilk 16'sı sorgu, diğer 48'i yalnız değerlendiricinin kullandığı test girdileridir. Bu 48 girdinin hiçbir etiketi öğreniciye verilmez. Tüm yöntemler aynı görevleri ve aynı sıradaki sorguları alır. Sorgu seçimi aktif değildir. Sonuçlar 320 aynı aile ve 320 farklı aile görevini kapsar; belirsizlik hesaplamasının birimi 40 seed'dir.

| Yöntem | Görev içi güncelleme | Görevler arası kalıcı değişim |
|---|---|---|
| `uniform_reset` | Aynı sürüm uzayı güncellemesi | Eski görevleri taşımaz; her yeni göreve tekdüze önselle başlar |
| `learned_family` | Aynı sürüm uzayı güncellemesi | Gerçek aile grupları üzerinde üç sayacı günceller |
| `scrambled_family` | Aynı sürüm uzayı güncellemesi | Aynı büyüklüklerde, fakat fonksiyonlara rastgele atanmış gruplar üzerinde üç sayacı günceller |

`uniform_reset` meta-eğitim deneyimini sonraki göreve aktarmayan bir müdahale kontrolüdür; öğrenilmiş önselin geçmişte ödediği 194,4 etiketlik maliyet yok sayılarak “toplam yaşam maliyeti azalır” denmez. Rastgele grup kontrolü, yalnız kalıcı sayaç tutmanın ve önseli tekdüzelikten çıkarmanın yeterli olup olmadığını sınar. Örnekleme, aynı görev kimliklerini testte tekrarlayarak ezber avantajı sağlamaz. Bununla birlikte tüm fonksiyonlar başlangıçtaki 128 elemanlı hipotez uzayındadır; yeni bir temsil ailesi edinimi burada sınanmaz.

Güven aralıkları, seed düzeyindeki ortalamalara veya aynı seed içindeki yöntem farklarına `ortalama ± 1,96 × standart hata` uygulanarak hesaplanmıştır. Bunlar normal yaklaşımıdır, dağılımdan bağımsız garanti değildir; çoklu karşılaştırma düzeltmesi yapılmamıştır. Deney ailesi ve ana karşılaştırmalar çalıştırmadan önce kodda sabitlendi. Bu, bağımsız yayımlanmış bir önkayıt değildir.

## 3. Çalıştırılmış sonuçlar

“Öncül tahmin hatası” her sorgunun etiketi görülmeden yapılan yanlış ikili tahmin sayısıdır. Log kaybı aynı anda tahmin edilen doğru etiket olasılığının `−log₂` toplamıdır. Test riski, o ana kadar hiç etiketlenmemiş 48 girdideki hata oranıdır. 0,5 olasılık eşitliğinde tahmin 0'dır; sayısal yuvarlamanın rastgele karar üretmesini engellemek için `10⁻¹²` tolerans vardır.

| Koşul ve yöntem | 16 sorguda öncül hata | Toplam log kaybı, bit | 4 etiket sonrası test riski | 8 etiket sonrası test riski | Tam kimlik belirleme etiketi |
|---|---:|---:|---:|---:|---:|
| Aynı aile / tekdüze | 3,5750 | 7,0000 | 0,4644 | 0,0785 | 8,1781 |
| Aynı aile / öğrenilmiş aile | 2,3406 | 5,4316 | 0,2856 | 0,0209 | 8,1781 |
| Aynı aile / rastgele grup | 3,6125 | 7,1104 | 0,4861 | 0,0915 | 8,1781 |
| Aile değişimi / tekdüze | 3,6688 | 7,0000 | 0,4636 | 0,0778 | 8,0469 |
| Aile değişimi / öğrenilmiş aile | 4,0875 | 11,6439 | 0,5602 | 0,1029 | 8,0469 |
| Aile değişimi / rastgele grup | 3,5719 | 7,0842 | 0,4832 | 0,0787 | 8,0469 |

Öğrenilmiş aile eksi tekdüze yöntem için eşleştirilmiş farklar:

| Ölçüt | Aynı aile: fark [%95 aralık] | Aile değişimi: fark [%95 aralık] |
|---|---:|---:|
| Öncül hata | −1,2344 [−1,3404; −1,1283] | +0,4188 [+0,3002; +0,5373] |
| 4 etiket sonrası test riski | −0,1788 [−0,1908; −0,1667] | +0,0965 [+0,0895; +0,1036] |
| 0–16 etiket boyunca ortalama test riski | −0,0657 [−0,0694; −0,0619] | +0,0321 [+0,0296; +0,0346] |
| Tam kimlik belirleme etiketi | 0 [0; 0] | 0 [0; 0] |

Bu deneyde bütün görevler 16 sorgu sınırından önce tekilleşti; raporlanan ölçütlerde sağdan sansürleme oluşmadı. Kod, bir sonraki çalıştırmada tekilleşmeyen bir görev olursa değeri 17 ile sınırlandırıp sansürleme oranını ayrıca kaydeder.

### Aile değişiminin cezası neden tam olarak 4,643856 bit?

Bu sayı, bağımsız rastgele bir keşif değildir; deneyin olasılık yapısından türetilir. Toplam sözde gözlem sayısı, 24 görevden sonra 25'tir. Görülmeyen karşı ailedeki bir hipotezin önsel olasılığı başlangıcın `1/25`'ine düşer. Etiketler hipotezi tam belirlediğinde olasılıkların zincir kuralı toplam log kaybını `−log₂ Pη(h*)` yapar. Bu nedenle tekdüze önsele kıyasla fark `log₂ 25 = 4,643856` bittir. Aynı ailedeki fark da `−1,568380` bittir. Sayısal deney bu özdeşliği her tekilleşen bölümde ayrıca kontrol eder; sıfıra yakın aralık genişliği, evrensel performans kesinliği olarak yorumlanmamalıdır.

### Örnek sayısı tek başına güvenilir bir başarı göstergesi değil

İkincil ölçüt olarak, 48 test girdisindeki hata oranının ilk kez sıfır olduğu ve kalan ölçüm süresince sıfır kaldığı etiket sayısı hesaplandı. Bu geriye dönük değerlendirici ölçütüdür. Öğrenici test etiketlerini görmediği için bu anda “öğrendim” kararı veremez.

| Koşul | Tekdüze | Öğrenilmiş aile | Rastgele grup |
|---|---:|---:|---:|
| Aynı aile | 8,1375 | 6,4438 | 7,6375 |
| Aile değişimi | 8,0188 | 8,0188 | 7,3250 |

Rastgele grup kontrolü aynı ailede bu ölçütü 0,5 etiket iyileştirir; buna rağmen ortalama test riski **0,01224 artar** ve öncül hata değişimi +0,0375'tir [%95 aralık −0,1062; +0,1812]. Farklı ailede de rastgele grup bu etiket ölçütünde avantajlı görünebilir. Belirsiz hipotezler arasında erken yapılan bazı şanslı tercihler, daha iyi bir öğrenici oluşturmadan geriye dönük “başardı” zamanını erkene çekebilir. Bu nedenle “100 deneyimden 10'a indi” türündeki bir iddia, yalnız ilk başarı zamanına dayandırılamaz; risk eğrisi, öncül kayıp, belirsizlik ve negatif kontroller birlikte gerekir.

Rastgele grup kontrolü için küçük bir dağılım ayrıntısı da vardır: Görev kimlikleri yerine koymadan örneklendiği için eğitim grubunda rastlantıyla fazla temsil edilmiş kimliklerin testte tekrarına izin verilmez. Bu kontrolün beklentide tam sıfır etkili bir plasebo olduğu varsayılmaz. Aynı durum tüm yöntemlere uygulanır; kontrolün işlevi, kalıcı değişimin tek başına güvenilir aktarım anlamına gelmediğini göstermektir.

Tam kimlik belirleme sayısının neden değişmediği daha temel bir negatif sonuçtur: Bütün önsel olasılıklar pozitiftir. Üç yöntem de aynı sorguların aynı etiketleriyle aynı hipotezleri eler. `V_k` kümeleri her adımda eşittir. Dolayısıyla `|V_k|=1` olayının zamanı önsel tarafından değiştirilemez. Aynı ailede daha erken yararlı genelleme vardır; tam kimlik belirlemek için daha az kanıt gerektiği gösterilmemiştir.

## 4. Kararlılık ve uyum için açık karşı örnek

Aynı 64 girdi üzerinde birbirinin tümleyenleri olan iki fonksiyon seçildi: `f_B(x)=1−f_A(x)`. Önce A, sonra B fonksiyonu, `0` ve altı birim vektörü üzerinde yedişer etiketle öğretildi. Bu yedi girdi afin fonksiyonu tam belirler. Bu kontrollü gösteride girdiler rastgele değil, ayırt edici olacak şekilde tasarlanmıştır.

| B öğrenildikten sonra | A hatası | B hatası |
|---|---:|---:|
| A için öğrenilen tek fonksiyonu dondur | 0 | 1 |
| Çelişki gelince tek sürüm uzayını sıfırlayıp yenisini öğren | 1 | 0 |
| Dışarıdan verilen A/B bağlamına ayrı öğrenilmiş durum ayır | 0 | 0 |

Son satır eşit bilgi ve eşit bellek karşılaştırması değildir. Doğru bağlam etiketi dışarıdan verilir; öğrenilmiş iki ayrı fonksiyon saklanır. İdeal afin parametre sayımı tek fonksiyonda 7 bit, iki fonksiyonda 14 bittir. Bunlar Python programının gerçek RAM tüketimi değildir; bağlam anahtarları, kod ve sabit tablolar sayılmamıştır. Bağlam başına koruma, sınırsız sayıda bağlamda sabit bellekle çözüm sağlamaz. Böyle bir bağlamın nasıl keşfedileceği veya güvenilir biçimde seçileceği bu gösteriden çıkmaz.

Bağlam her sorguda bağımsız ve eşit olasılıkla A/B seçilip öğreniciden saklanırsa, geçmiş ve diğer kanallar da bilgi vermiyorsa, aynı görünür `x` için etiket 0 veya 1'i eşit olasılıkla alır. Her kestiricinin beklenen hatası en az 1/2'dir. Bu alt sınır deterministik ve rastgele kestiriciler için geçerlidir. Kod, tüm girdiler ve iki olası deterministik tahmin üzerinde her çiftte en az bir hata gerektiğini tam tarar; rastgele tahminlerin aynı sınıra tabi olması bu iki seçimin karışımı olmalarından gelir.

Buradan “gizli bağlam hiçbir zaman öğrenilemez” sonucu çıkmaz. Bağlam zaman içinde kalıcıysa, gözlenebilir ipuçları varsa veya etkileşimle anlaşılabiliyorsa başka sonuç mümkündür. Alt sınır belirtilen bilgi yapısına bağlıdır. Fakat aynı gözlemin çelişen doğru cevaplarını, hiçbir ayırt edici bilgi olmadan aynı anda garanti etme talebi bu durumda imkânsızdır. Bu hata bir kararlılık mekanizması eksikliğiyle tek başına çözülemez.

## 5. Elenen iddialar ve kalan belirsizlikler

**Deneysel destek:** Geçmiş görevlerden edinilen üç sayılık durum, yeni ve eğitimde kullanılmamış fonksiyonlarda daha düşük tahmin kaybı sağlar. Etiketler çalışırken işlenir. Değişen önsel davranışı nedensel olarak etkiler; tekdüze duruma sıfırlama ve rastgele grup kontrolleri aynı sonucu üretmez. Aile değişimi açık olumsuz aktarım üretir.

**Mantıksal sonuç:** Pozitif önseller ve aynı sorgular altında sürüm uzayının tekilleşme zamanı eşittir. Öngörülemez gizli bağlam ve tümleyen etiketler altında hata alt sınırı 1/2'dir. Bu koşullardaki log kaybı farkları başlangıçtaki hipotez olasılıklarından tam türetilir.

**Elenen güçlü yorumlar:** Daha çok geçmiş her yeni görevde faydalı değildir. Daha iyi önsel, aynı gözlem dizisinden daha fazla tanımlayıcı kanıt üretmez. Bağlam başına ayrı durum korumak, bağlamsız ve aynı bellek bütçeli genel bir unutma çözümünü ispatlamaz. Tek bir “başarmak için gereken etiket” sayısının düşmesi tek başına öğrenmeyi öğrenme kanıtı değildir.

**Burada gösterilmeyenler:** Bilinmeyen hipotez ailesinin keşfi; temsil dilinin genişlemesi; gürültü ve yanlış etiketlere dayanıklılık; otonom görev sınırı bulma; sabit kaynakla açık uçlu yaşam; öğrenilmiş görevin yıllar sonra güvenilir saklanması; diskten yeniden başlatmada davranış eşdeğerliği; yeni duyusal alanlara aktarım; eylem seçimi; içsel dinlenme/konsolidasyon faydası. Hard eleme, gürültüsüz doğru-sınıf varsayımına dayanır. Bir yanlış etiket gerçek hipotezi eleyebilir; çelişkide sıfırlama basit örnekte uyum sağlar, güvenilir yanlış-bilgi düzeltme genel olarak çözülmüş değildir.

Sistem yeni görevlerde görmediği girdilere genelleme yapar; bu düzeyde yalnız olay listesi ezberi değildir. Buna karşılık başlangıçtaki hipotez dili, aile ayrımı, kayıp ve güncelleme kuralı tasarımcı tarafından belirlenmiştir. Yeni bir temel hesaplama biçimi veya açık uçlu yeni yetenek sınıfı oluştuğu iddia edilmez.

## 6. Yeniden üretim ve denetim kaydı

Repository kökünde:

```text
python research/living_learning/adaptation_experiment.py
```

Python standart kütüphanesi yeterlidir. Ana Cevahir sistemi içe aktarılmaz. İlk tam çalıştırma bu makinede yaklaşık 7 saniye sürdü; bu bir performans benchmark'ı değildir. İşlem, sonuç dosyasını her çalıştırmada yeniden üretir.

Kaynak: `research/living_learning/adaptation_experiment.py`.

Makinece okunabilir kayıt: `research/living_learning/results/adaptation_results.json`. Kayıt, protokolü, kaynak dosyanın SHA-256 değerini, 40 seed'in görev kimliklerini, eğitimdeki tekilleşme olaylarını, test sorgularını, ayrı test girdilerini, bölüm düzeyindeki risk eğrilerini, özetleri ve bağlam karşı örneğini içerir. Seed'ler değiştirilebilir: `--seeds N`. Bunun yeni bir deney sonucu olduğu belirtilmelidir.

Kod şu koşulları denetler: eğitim/test kimliklerinin ayrıklığı; sorgu/test girdilerinin ayrıklığı; gerçek hipotezin gürültüsüz güncellemelerde korunması; pozitif doğru-etiket olasılığı; tüm yöntemlerde aynı tekilleşme zamanı; tekilleşen görevlerde toplam log kaybının `−log₂ P(h*)` özdeşliği; yedi ayırt edici girdinin iki bağlamı ayrı ayrı tanımlaması; bağlamsız 1/2 alt sınırının tüm 64 girdi üzerinde geçerliliği.

İlk sonuçtan sonra yalnız simetrik 0,5 olasılıklarında kayan nokta eşitlik kontrolü eklendi; ana sayısal sonuçlar değişmedi. Olumsuz aktarım, tam kimlik sayısındaki sıfır fark ve rastgele grup kontrolünün yanıltıcı başarı-zamanı avantajı rapordan çıkarılmadı. Kod, sonuç ve rapor bu deneyin kapsamındaki bağımsız araştırma dosyalarıdır; ana sistemde değişiklik veya GitHub'a gönderim yapılmadı.

Son kaynakla tam deney ikinci kez çalıştırıldı; sonuç JSON'u bayt düzeyinde aynı üretildi. Sonuç dosyasının SHA-256 değeri `465b943d4369715700c8a0835d2e6fba4439aab04f4631e547a2b8d5daf70867`.
