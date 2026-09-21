# Akademik okuma, kaynaklar ve yeniden üretim

[İçindekiler](README.md) · [Kaynak haritası](KAYNAK_HARITASI.md) · [Bakım](BAKIM.md)

**Yazar: Muhammed Yasin Yılmaz.** Bu kitap Cevahir'in gerçek uygulamasını matematik, özgün yayınlar ve çalıştırılmış küçük örnekler üzerinden açıklayan, geliştirilmekte olan açık bir eğitim kaynağıdır. Yazım, literatür taraması, kod incelemesi ve örnek denetimlerinde yapay zekâ desteği kullanılmıştır. Akademik kullanım hedefi veya DOI kaydı dış hakem değerlendirmesi yapıldığı anlamına gelmez. Bütün bölümler aynı ayrıntı düzeyinde tamamlanmış değildir.

## Bu turda derinleştirilen öğrenme yolu

21 Eylül 2026 genişletmesi, tokenizer'dan sonra sinirsel hesap ve öğrenme arasındaki bağı doldurur:

| Bölüm | Ön bilgi | Bölüm sonunda yapılabilecek hesap | Kanıt türü |
|---|---|---|---|
| [3. Sinir ağı ve embedding](tr/03-sinir-aglari.md) | Vektör, matris, türev fikri | Affine/nonlinear ayrımı; one-hot/lookup eşitliği; gradyan ve bağlı ağırlık hesabı | Elle türetim + gerçek LanguageEmbedding ve küçük çekirdek |
| [4. Attention](tr/04-attention.md) | Matris çarpımı, softmax | Üç tokenın çıktısı; maske; MHA/MQA/GQA boyutları ve KV belleği | Elle hesap + gerçek attention/cache eşdeğerliği |
| [5. Transformer](tr/05-transformer.md) | Önceki iki bölüm | Residual, norm, RoPE, FFN parametreleri ve MoE yardımcı hedefi | Kaynak incelemesi + blok rekonstrüksiyonu/RoPE/FFN kontrolleri |
| [6. Eğitim](tr/06-egitim.md) | Zincir kuralı ve loss | Hedef kaydırma, token ağırlıklı birikim, optimizer adımı ve bütçe karşılaştırması | Gerçek loss/çekirdek çalıştırması + mevcut eğitim sözleşmesi testleri |

Bu dört bölümde çözümlü alıştırmalar ve bölüm kaynakçaları bulunur. Diğer bölümler sistemin geri kalanına bağlantı sağlar; aynı sayıda çalıştırılmış örnek veya kapsamlı literatür tartışması taşıdıkları iddia edilmez. Araştırma bölümlerindeki açık living-learning problemi, standart geri yayılımın açıklanmış olmasıyla çözülmüş sayılmaz.

## Atıf nasıl okunmalı?

Üç farklı iddia için üç farklı dayanak kullanılır:

1. **Yöntemin kökeni veya yayımlanmış deney:** Yazar, yıl ve özgün makale/teknik rapor bağlantısı verilir. Model ailesinin adı bütün sürümlerine genellenmez. Bir raporun ölçtüğü sonuç Cevahir'e aktarılmaz.
2. **Cevahir'in yaptığı hesap:** Gerçek dosya ve nitelikli metot bağlantısı verilir. Özellik listesi veya eski kaynak yorumu, etkin çağrı dalını kanıtlamaz. Şema varsayılanı, doğrudan kurucu ve kullanılan deney ayarı ayrı okunur.
3. **Bu kitapta çalıştırılan örnek:** Komut, ortam, girişler, sonuç ve tolerans kaydedilir. Elle seçilmiş öğretim sayıları, rastgele başlatılmış küçük ağ ve eğitilmiş model ölçümü birbirine karıştırılmaz.

Kaynakçalar ilgili bölüm sonundadır: [sinir ağı](tr/03-sinir-aglari.md#kaynakça), [attention](tr/04-attention.md#kaynakça), [Transformer](tr/05-transformer.md#512-kaynakça-ve-atıf-kapsamı), [eğitim](tr/06-egitim.md#kaynakça). Özgün makalelerle teknik raporların yayın statüleri aynı değildir; ön baskı ve konferans yılları mümkün olduğunda ayrı gösterilir. Resmî API belgesi yöntem kökeni için değil, arayüzün davranışı için kaynak olur.

## Sayısal örnekleri yeniden çalıştırmak

Repository kökünde, projenin Python/PyTorch bağımlılıkları kurulu ortamda:

```powershell
python scripts/book_tokenizer_walkthrough.py
python scripts/book_neural_walkthrough.py
python scripts/check_book.py
```

İlk komut mevcut tokenizer varlıklarıyla altı metin örneğini denetler. İkincisi yedi örnek grubunu gerçek Cevahir bileşenleriyle CPU'da çalıştırır: lookup ve bağlı ağırlık, skaler türev, üç tokenlı attention, üç KV düzeni, FFN/norm/blok, RoPE ve model güncellemesi. Varsayılan çalıştırma kayıtları yenilemez. Eğitimli checkpoint yüklemez veya kaydetmez; optimizer adımı yalnız yeni kurulmuş küçük modelin bellekteki parametrelerini değiştirir.

[Neural yürütme kaydı](evidence/neural_walkthrough.json) Python 3.14.3, PyTorch 2.10.0+cpu ve sabit seed ile alınmıştır. Ondalıklı sonuçlar için mutlak `2e-6`, göreli `2e-5` karşılaştırma toleransı kullanılır; örneklerin kendi matematiksel denetimleri ayrıca çalışır. Platform veya kütüphane değişikliği sonucu farklılaştırırsa önce nedeni araştırılır, beklenen çıktı otomatik değiştirilmez.

[Genişletme doğrulaması](evidence/academic_expansion_verification.json) 99 hedefli testin bu turda geçtiğini kaydeder. Bu testler GPU kernel başarımı, eğitilmiş modelin Türkçe kalitesi veya bütün repository'nin hatasızlığı için kanıt değildir. [Kitap denetleyicisi](../../scripts/check_book.py) bağlantı, sembol ve incelenmiş kaynak parmak izini kontrol eder; matematiksel anlatımı veya kaynakların bütün yorumlarını otomatik doğrulamaz.

## Akademik kullanımda sürümü belirtmek

GitHub ana dalı gelişmeye devam eder. Bir alıntı veya tekrar üretim kaydında bölüm adıyla birlikte commit veya sabit arşiv sürümü belirtilmelidir. [Yayın dizisindeki](../publications/README.md) DOI belirli bir arşiv anını tanımlar; ana dala daha sonra eklenmiş cümleleri o eski DOI'ye aitmiş gibi göstermemek gerekir. Araştırma kayıtlarının tarihleri ve olumsuz sonuçları korunur; yeni düzeltmeler gerekçesiyle birlikte eklenir.
