# Living-learning araştırması 11 — edinilmiş program parçaları

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma

[Türkçe rapor](../../docs/research/living_learning_library_2026_09_21/REPORT_TR.md) · [Ana protokol](../../docs/research/living_learning_library_2026_09_21/PROTOCOL.md) · [Matematik ve birincil literatür](../../docs/research/living_learning_library_2026_09_21/MATHEMATICS_AND_LITERATURE.md) · [Sonuçlar sonrası kontrol planı](../../docs/research/living_learning_library_2026_09_21/POSTHOC_PORTFOLIO.md)

Sabit bir sonlu program dilinde deneyimden çıkarılmış parçaların sonraki program edinimine etkisi araştırılır. 12 yaşam, 9.216 ana ve 384 post-hoc koşul vardır. 9.600 bağımsız yaşam değildir. Yeni hesaplanabilirlik sınıfı, yeni temel algoritma veya genel yaşarken öğrenme çözümü iddia edilmez. Önceki araştırma, kitap ve Cevahir çalışma zamanı değiştirilmez.

## Dosyalar

| Dosya | İşlev |
|---|---|
| [library_learning.py](library_learning.py) | Program yorumlayıcısı, BFS, deneyim bankasından parça çıkarımı, sekiz yöntem, dünya ve eşleştirilmiş deney |
| [results/library_learning.json](results/library_learning.json) | 9.216 koşul, bütün ilk edinimler, geçmiş durumları, öğretmenin yeniden üretim kaydı, özetler |
| [portfolio_control.py](portfolio_control.py) | İki bağımsız yarım-bütçe araması; ana sonuçlardan sonra eklendi |
| [results/portfolio_control.json](results/portfolio_control.json) | 384 koşul, iki dalın ayrı sonuçları, seçilen program ve maliyet toplamı |
| [analyze_results.py](analyze_results.py) | Aynı yaşam/görev üzerindeki kazanım–kayıp sayıları, geçmiş 6→12 değişimi ve ortak başarı maliyetleri |
| [results/paired_summary.json](results/paired_summary.json) | Betimleyici eşleştirilmiş sonuçlar; evrensel istatistik garantisi değildir |
| [verify_library.py](verify_library.py) | Ayrı skaler yorumlayıcı, kaba enumerasyon, gerçek çağrı sayımı, öğretmen bilgisi müdahalesi, kütüphane silme, ayrı süreç devamı |
| [results/verification.json](results/verification.json) | Kontroller ve ana/ek taramaların tam yeniden üretim kaydı |
| [results/preservation_baseline.json](results/preservation_baseline.json) | Önceden var olan 886 dosyanın başlangıç SHA-256 kaydı ve HEAD |
| [final_audit.py](final_audit.py) | Kaynak/kayıt hash'leri, ek kontrolün bağımsız anlamsal denetimi, belge bağlantıları ve korunma |
| [results/final_audit.json](results/final_audit.json) | Son yerel denetim; dış hakem incelemesi değildir |

## Yeniden üretim

Python 3.14.3 ile, yalnız standart kütüphane kullanılarak çalıştırıldı. Paket yüklenmedi. Depo kökünde:

```powershell
python research/living_learning_library_2026_09_21/library_learning.py
python research/living_learning_library_2026_09_21/portfolio_control.py
python research/living_learning_library_2026_09_21/analyze_results.py
```

Bu komutlar varsayılan olarak salt karşılaştırma yapar: yeniden hesaplayıp mevcut JSON ile tam eşitliği denetler. `--write` açık verilirse yalnız kendi sonuç dosyalarını yazar. İlk iki komutu birlikte çalıştırıp doğrulamaları da yapmak için:

```powershell
python research/living_learning_library_2026_09_21/verify_library.py --rerun
python research/living_learning_library_2026_09_21/analyze_results.py
python research/living_learning_library_2026_09_21/final_audit.py
```

`verify_library.py --rerun` ilk iki deneyi yeniden çalıştırdığı için ayrıca yukarıdaki ilk iki komutu tekrarlamak gerekmez. Doğrulama kaydı yeniden yazılır. `--rerun` verilmezse yalnız daha dar kontroller çalışır ve kayıt bu dar kapsamla değiştirilir. `final_audit.py`, önceki tam doğrulama kaydını korur; yalnız son denetim dosyasını yazar. Python sürümü ve kaynak hash'i sonuç metadata'sına dahildir; farklı ortam metadata'sı tam karşılaştırmayı başarısız yapabilir. Bu, yeni ortamı kendiliğinden geçerli tekrar olarak kaydetmemek içindir.

## Bilgi ve kalıcılık sözleşmesi

`synthesize` yalnız sekiz arama ve sekiz kabul örneği, izinli token listesi ve bütçe alır. Öğretmen programı yalnız dünya/değerlendiricide bulunur. `token_policy`, edinilmiş programlar ve seçilmiş parçaları okur. `oracle_motifs` eski ailenin motiflerini ayrıca alır ve ayrıcalıklı olarak etiketlenmiştir.

Her program yürütülürken bütün primitive'leri ücretlendirilir. Bir primitive-vektör adımı sekiz skaler uygulamadır. Aday üretim/sıralama, durum çıkarımı, veri üretimi, değerlendirici ve Python nesne maliyetlerinin tamamı bu bütçeye dahil değildir. Alan sonlu, dünya gürültüsüz, görev sınırları verilmiş, kabul yalnız sonlu örneklerde yapılmıştır.

Öğrenici durumu `programs`, `library`, `mining` alanlarından oluşur. JSON yüklemesi sonrası yeni görev için aynı politikayla aynı arama yapılır. Sekiz yöntem ve iki aile için 16 ayrı süreç kontrolü yapılmıştır. Bu, kesilmiş bir BFS kuyruğunun adım ortasından devamı değildir; **edinimden sonraki kayıtlı durumdan yeni görev ediniminin** yeniden başlamasıdır. Program bankası yaşamla büyüyebilir; sabit toplam bellek çözümü yoktur.

Önceki turdaki [yeniden üretim dizini](../living_learning_geometry_2026_09_21/README.md) korunur. Bu turda commit, push veya yayın yapılmadı; mevcut DOI sürümleri bu yeni yerel dosyaları içeriyor sayılmaz.
