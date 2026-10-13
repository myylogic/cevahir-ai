# Living-learning araştırması 10 — öğrenilebilir temsil geometrisi

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yalnız yerel araştırma

[Türkçe sonuç raporu](../../docs/research/living_learning_geometry_2026_09_21/REPORT_TR.md), [ana protokol](../../docs/research/living_learning_geometry_2026_09_21/PROTOCOL.md), [matematik ve literatür](../../docs/research/living_learning_geometry_2026_09_21/MATHEMATICS_AND_LITERATURE.md), [post-hoc kontroller](../../docs/research/living_learning_geometry_2026_09_21/POSTHOC_CONTROLS.md), [ayrı uygulama incelemesi](../../docs/research/living_learning_geometry_2026_09_21/REVIEW.md).

Bu deney genel living-learning ilkesini bulduğu veya yeni bir continual-learning algoritması geliştirdiği iddiasında değildir. Temsili de değişen küçük bir ağda eski çıktıyı korumanın yeni edinim üzerindeki etkisini, matematiksel karşı örneklerle ve eşleştirilmiş koşullarla inceler. Cevahir çalışma zamanına entegrasyon yoktur.

## Kayıtlar

| Dosya | İçerik ve statü |
|---|---|
| [geometry.py](geometry.py) | 33 parametreli ağ, analitik türevler, dokuz yöntem, üç teğet yöntemin eşik hassasiyeti, durum kaydı ve devam |
| [results/geometry.json](results/geometry.json) | 12 yaşamda 864 koşul, 72 özet grup, 36 devam çifti; sonuçlardan önce yazılmış yerel protokol |
| [posthoc_controls.py](posthoc_controls.py) | Ana sonuçlar görüldükten sonra eklenen doğru-kayıp denetimleri, iki replay adımı, çevrimdışı referans |
| [results/posthoc_controls.json](results/posthoc_controls.json) | Aynı 12 yaşamda 216 koşul, 18 özet grup, 12 çevrimdışı referans; bağımsız doğrulama değildir |
| [endpoint_diagnostic.py](endpoint_diagnostic.py) | Sonuçlar görüldükten sonra eklenen son nokta / ara yol tanısı |
| [results/endpoint_diagnostic.json](results/endpoint_diagnostic.json) | 12 referans yol; 10 uygun doğru-kayıp son noktası, 12 ara yol ihlali; bütün yollar için imkânsızlık teoremi değildir |
| [verify_geometry.py](verify_geometry.py) | Bağımsız türev hesapları, affine kontrol, dokuz ana yöntemde ayrı süreç devamı, önceki dosyaların korunması |
| [results/verification.json](results/verification.json) | Geçmiş tam denetimin ayrıntısı; iki tarama yeniden hesaplandı |
| [results/preservation_baseline.json](results/preservation_baseline.json) | Çalışma öncesindeki 866 dosyanın SHA-256 değerleri ve HEAD |
| [final_audit.py](final_audit.py) | Kaynak–sonuç hash eşleşmesi, korunma, yerel belge bağlantıları ve son nokta tanısının yeniden üretimi |
| [results/final_audit.json](results/final_audit.json) | Son denetim, belge/deney hash'leri ve önceki tam doğrulama kaydının referansı |

1.080 ana/ek koşul birbirinden bağımsız 1.080 dünya değildir. Aynı 12 yaşam yeniden kullanılır. Çevrimdışı referanslar ek veri ve hesap alır. Bütün ölçüler, yöntem kapsamları ve maliyet sınırlamaları raporda açıklanmıştır.

## Yeniden üretim

Depo kökünde PowerShell ile, mevcut Python ortamında:

```powershell
python research/living_learning_geometry_2026_09_21/geometry.py
python research/living_learning_geometry_2026_09_21/posthoc_controls.py
python research/living_learning_geometry_2026_09_21/endpoint_diagnostic.py
```

Bu üç komut varsayılan olarak hesaplar ve kayıtlı sonuçla karşılaştırır; sonuç dosyalarını değiştirmez. Kaydetme seçeneği `--write` açıkça verilirse yalnız ilgili yeni deney JSON'u yazılır. Mevcut araştırmanın korunması için normal kullanım salt karşılaştırmadır.

Türev, süreçten devam, bütün sonuç ve korunma denetimini tek seferde yapmak için:

```powershell
python research/living_learning_geometry_2026_09_21/verify_geometry.py --rerun
python research/living_learning_geometry_2026_09_21/final_audit.py
```

İlk komut ana ve post-hoc deneylerin tamamını yeniden çalıştırır; yukarıdaki ilk iki komutu ayrıca tekrarlamak gerekmez. `verification.json` yeniden yazılır. İkinci komut son nokta tanısını yeniden hesaplayıp `final_audit.json` yazar; önceki tam doğrulama kaydını değiştirmez. `verify_geometry.py` yalnız `--rerun` olmadan çalıştırılırsa daha dar denetim yapar ve bu dar kapsamla doğrulama dosyasını yeniden yazar.

Kaydedilen ortam Python 3.14.3, NumPy 2.4.2, türev denetimi için PyTorch 2.10.0+cpu; float64. Çekirdek deney yalnız NumPy ve standart kütüphane kullanır. BLAS/OMP/MKL iş parçacığı sayısı deney modülü yüklenirken bire ayarlanır. Sonuç karşılaştırması sayılar için `atol=1e−11, rtol=1e−9` kullanır; alanlar, ortam sürümleri ve kod hash'leri de karşılaştırılır. Başka ortamın metadata farkı veya SVD duyarlılığı aynı kayıtla eşleşmeyebilir; böyle bir farkı başarılı bağımsız tekrar diye kaydetmeyin. Kurulum veya paket değişimi bu turda yapılmadı.

## Kalıcılık kapsamı

`Learner.snapshot` ağırlıklar, yöntem ve ayarlar, ankorlar, başlangıç referansı, varsa sabit baz, sayaçlar ve adım sayısını saklar. `restore` bunları geri yükler. Dokuz ana yöntem tohum 999'da 23 batch sonrasında ayrı süreçte yüklenip 17 batch daha sürdürülmüş, tam JSON eşitliği doğrulanmıştır. Bu mekanik kalıcılık kontrolü uzun yaşamda genellenebilir öğrenmenin ispatı değildir. `ControlLearner` alt sınıfının ek durumları için ayrı süreç devamı uygulanmış veya doğrulanmış sayılmaz.

Önceki araştırma ve kitabın başlangıçtaki yerel değişiklikleri korunmuştur. Bu dizin ve eşleşen yeni rapor dizini dışında içerik değişikliği yapılmamış; commit, push veya yayın yapılmamıştır.
