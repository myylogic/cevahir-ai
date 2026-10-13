# Ana sonuçlardan sonra eklenen kontroller

21 Eylül 2026. Bu plan ilk 864 koşulun sonucu görüldükten sonra yazılmıştır; ön kayıt veya bağımsız doğrulama kümesi değildir. Ana protokol ve sonucu değiştirilmeyecek. Aynı 12 yaşamda mekanizma açıklamasını sınar.

## Gözlenen gerekçe

64 ankorda güncel teğet projeksiyonu yeni MSE'yi yaklaşık `.19626949 → .19626921` değiştirdi; sabit çıktı kabulü olan replay yaklaşık `.17455`te kaldı. Başlangıç eski MSE `.00806` idi: korunmuş çıktı zaten kusursuz değildi. Dolayısıyla bu korumayı doğru eski beceriyi korumakla eşitlemek hatalı olabilir.

## Sabit ek protokol

Aynı başlangıçlar, bellek boyutları, yeni akış, öğrenme oranları ve gizli değerlendirme kümeleri kullanılacak. Üç ek yöntem:

1. `online_loss_guard`: yeni minibatch gradyanı; adayın eski ankorlarda **doğru etikete karşı MSE**'si başlangıç MSE + `1e−4` altında ise kabul. Aynı 13 aday/yarılama sınırı.
2. `replay_loss_guard`: aynı kabul, eşit ağırlıklı eski/yeni replay önerisi.
3. `replay_two_steps`: aynı yeni batch/aynı ankorlarda iki ardışık replay adımı. Ek öğrenme hesabının etkisi için güçlü, basit kontrol; gate'in değişken maliyetiyle tam FLOP eşitliği iddia edilmez. Prequential cevap yalnız ilk adımdan önce alınır.

Doğru etiket kaybı ve çıktı sapması aynı büyüklük değildir. Aynı `1e−4` sabiti hedefleri eşitlemez. İlk iki yöntem başlangıçtaki yanlış çıktıları düzeltebilir; eski alt beceriler veya görülmemiş eski örnekler için ayrı garantileri yoktur. Gate verisi yine öğreniciye ait kabul bilgisidir; test verisi değildir.

Tanısal ortak-uydurma referansı ayrıca eski 128 ve yeni 1.024 örneğin tamamına aynı anda erişip `.05` ile 1.000 adım eğitilecek. Bu daha fazla veri ve hesaplı çevrimdışı referans, adil online rakip olarak sıralanmayacak; öğretmenin sınıfta olmasından farklı olarak bu modelde bir optimizasyon yolunun iki bölgeyi birlikte iyileştirebildiğini sınayacak. Yeni yöntemlerin seçimi bu sonucun ardından yeniden ayarlanmayacak.

Ana kod, ana JSON ve eski araştırmalar korunur. Ek sonuçlar `posthoc_controls.json` dosyasına yazılır; final raporda ayrı tablo ve bu statüyle sunulur.

## Sonuçlardan sonra eklenen son nokta ve yol tanısı

Bu tanı **hem ana deney hem yukarıdaki ek kontrollerin sonuçları görüldükten sonra** eklendi. Yeni ön kayıt veya bağımsız doğrulama değildir. [Yeniden üretim kodu](../../../research/living_learning_geometry_2026_09_21/endpoint_diagnostic.py), aynı 12 dünyada 64 ankor için başlangıç doğru-etiket MSE'sini ve başlangıç çıktılarını alır. Eski 128 ve yeni 1.024 örneğin bölgesel ortalama gradyanlarını eşit ağırlıklandırarak `.05` oranıyla 1.000 çevrimdışı adım uygular. Her adımda eski ankor kaybının başlangıca göre artışı ayrıca izlenir. Öğretmen ve son öğrenci çıktılarının başlangıç çıktılarına sapması da hesaplanır.

[Kaydedilmiş sonuçta](../../../research/living_learning_geometry_2026_09_21/results/endpoint_diagnostic.json), 12 son noktanın 10'u başlangıç doğru-etiket MSE+`10⁻⁴` sınırındadır. Buna rağmen **izlenen 12 optimizasyon yolunun tamamı**, en az bir ara adımda bu sınırı aşmıştır. Bu gözlem, uygun bir son nokta bulmanın bütün ara adımları kabul edilebilir bir yol bulmakla aynı olmadığını gösterir. **12/12 örnek yolun ihlali, bütün olası yolların imkânsızlığı değildir.** Bu tanı başka yönleri, daha küçük adımları veya telafili eğrileri tüketerek aramaz.

Öğretmenin eski ankor çıktıları da 12/12 dünyada başlangıç çıktılarından `10⁻⁴`ün üzerinde sapar: sıfır doğru-hedef riskli kapasite tanığı, başlangıç çıktılarını koruma hedefi altında otomatik uygun son nokta değildir. Bu, doğru etiket kaybı ve başlangıç davranışını koruma hedeflerini eşitlememe gereğini somutlaştırır. Çevrimdışı çözüm ek veri ve hesap kullanır; online öğreniciye veya kabul kararlarına verilmez.

Mevcut tanı dosyasını değiştirmeden yeniden hesaplama:

```powershell
python research/living_learning_geometry_2026_09_21/endpoint_diagnostic.py
```

Varsayılan çalışma mevcut JSON ile sayısal karşılaştırma yapar; yalnız açık `--write` seçeneği kayıt yazar. Bu eklemede önceki tanı JSON'u, ana deney kodu ve ana sonuçlar korunmuştur.
