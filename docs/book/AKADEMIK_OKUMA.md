# Akademik okuma, kaynaklar ve yeniden üretim

[İçindekiler](README.md) · [Kaynak haritası](KAYNAK_HARITASI.md) · [Bakım](BAKIM.md)

**Yazar: Muhammed Yasin Yılmaz.** Bu kitap Cevahir'in gerçek uygulamasını matematik, özgün yayınlar ve çalıştırılmış küçük örnekler üzerinden açıklayan, 12 bölümlük ilk tam metni tamamlanmış açık bir eğitim kaynağıdır. Yazım, literatür taraması, kod incelemesi ve örnek denetimlerinde yapay zekâ desteği kullanılmıştır. Akademik kullanım hedefi veya DOI kaydı dış hakem değerlendirmesi yapıldığı anlamına gelmez. Tamamlanma kapsamı ve sayımı [ayrı kayıtta](TAMAMLANMA.md) belirtilir. Son eklemeler henüz commit veya yeni DOI arşivine aktarılmamıştır.

## Tamamlanan öğrenme yolu

Önce 1–2. bölümlerde sayısal araçlar, olasılık, motor/model ayrımı ve gerçek tokenizer yolu kurulur. 3–6. bölümlerde bu temel sinirsel hesap ve eğitime bağlanır:

| Bölüm | Ön bilgi | Bölüm sonunda yapılabilecek hesap | Kanıt türü |
|---|---|---|---|
| [3. Sinir ağı ve embedding](tr/03-sinir-aglari.md) | Vektör, matris, türev fikri | Affine/nonlinear ayrımı; one-hot/lookup eşitliği; gradyan ve bağlı ağırlık hesabı | Elle türetim + gerçek LanguageEmbedding ve küçük çekirdek |
| [4. Attention](tr/04-attention.md) | Matris çarpımı, softmax | Üç tokenın çıktısı; maske; MHA/MQA/GQA boyutları ve KV belleği | Elle hesap + gerçek attention/cache eşdeğerliği |
| [5. Transformer](tr/05-transformer.md) | Önceki iki bölüm | Residual, norm, RoPE, FFN parametreleri ve MoE yardımcı hedefi | Kaynak incelemesi + blok rekonstrüksiyonu/RoPE/FFN kontrolleri |
| [6. Eğitim](tr/06-egitim.md) | Zincir kuralı ve loss | Hedef kaydırma, token ağırlıklı birikim, optimizer adımı ve bütçe karşılaştırması | Gerçek loss/çekirdek çalıştırması + mevcut eğitim sözleşmesi testleri |

Bu dört bölümden sonra 7–9. bölümler de genişletildi: model kimliği ve checkpoint, token üretimi/cache, retrieval ve araç yürütme. Bu üç bölümün önceki genişletmesinde 15 yeni çözümlü alıştırma ve 18 kaynakça girdisi eklendi; son turda quantization karşılaştırmasının kaynakları da tamamlandı. Yaşam döngüsü için beş, üretim/bellek/araç için beş yeni çalıştırılmış örnek grubu eklendi. Son tamamlanma turunda 1–2 ve 10–12. bölümler de temel matematik, sistem davranışı, son araştırmalar ve çözümlü alıştırmalarla kapatıldı. Kitabın bütünü 63 çözümlü alıştırma içerir; [ayrıntılı dizin](AYRINTILI_ICINDEKILER.md), [terimler](TERIMLER.md) ve [bütünleştirici çalışma](BUTUNLESTIRICI_CALISMA.md) okuma yolunu tamamlar. Araştırma bölümlerindeki açık living-learning problemi, standart geri yayılımın açıklanmış olmasıyla çözülmüş sayılmaz.

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
python scripts/book_lifecycle_walkthrough.py
python scripts/book_runtime_walkthrough.py
python scripts/book_system_walkthrough.py
python scripts/book_research_walkthrough.py
python scripts/book_inventory.py
python scripts/check_book.py
```

İlk komut mevcut tokenizer varlıklarıyla altı metin örneğini denetler. İkincisi yedi örnek grubunu gerçek Cevahir bileşenleriyle CPU'da çalıştırır: lookup ve bağlı ağırlık, skaler türev, üç tokenlı attention, üç KV düzeni, FFN/norm/blok, RoPE ve model güncellemesi. Varsayılan çalıştırma kayıtları yenilemez. Eğitimli checkpoint yüklemez veya kaydetmez; optimizer adımı yalnız yeni kurulmuş küçük modelin bellekteki parametrelerini değiştirir.

[Neural yürütme kaydı](evidence/neural_walkthrough.json) Python 3.14.3, PyTorch 2.10.0+cpu ve sabit seed ile alınmıştır. Ondalıklı sonuçlar için mutlak `2e-6`, göreli `2e-5` karşılaştırma toleransı kullanılır; örneklerin kendi matematiksel denetimleri ayrıca çalışır. Platform veya kütüphane değişikliği sonucu farklılaştırırsa önce nedeni araştırılır, beklenen çıktı otomatik değiştirilmez.

[Genişletme doğrulaması](evidence/academic_expansion_verification.json) 99 hedefli testin ilgili genişletme turunda geçtiğini kaydeder. Bu testler GPU kernel başarımı, eğitilmiş modelin Türkçe kalitesi veya bütün repository'nin hatasızlığı için kanıt değildir. [Kitap denetleyicisi](../../scripts/check_book.py) bağlantı, sembol ve incelenmiş kaynak parmak izini kontrol eder; matematiksel anlatımı veya kaynakların bütün yorumlarını otomatik doğrulamaz.

## Yaşam döngüsü ve çalışan sistem örnekleri

[Checkpoint kaydı](evidence/lifecycle_walkthrough.json), gerçek ModelManager ve küçük modellerle geçici dosyalarda kayıt/yükleme, optimizer devamı, kimlik ve bütünlük denetimlerini gösterir. [Üretim/bellek/araç kaydı](evidence/runtime_walkthrough.json), gerçek adaptör filtrelerini sabit logitlerle, retrieval'ı elle verilmiş vektörlerle, calculator'ı gerçek fonksiyonuyla çalıştırır. Sabit logit kaynağı bir eğitilmiş model değildir. İki komut da varsayılan olarak mevcut sonuçları denetler; kalıcı model varlıklarına yazmaz.

Kod incelemesi sırasında calculator çıkarıcısının `2+3*4` ifadesini `2+3` olarak kestiği bulundu ve düzeltildi. [Eski çıktı](evidence/runtime_walkthrough_before_tool_fix.json) saklandı; güncel örnekte hem çıkarılan hem açık ifade `"14"` verir. Belirsiz veya desteklenmeyen ifade yerine başka işlem üretilmez. Bu düzeltme genel doğal dil matematik çözümünü sağlamaz. [Yeni doğrulama kaydı](evidence/runtime_expansion_verification.json) hedefli testlerin kapsamını ve sürüm farkını açıklar.

## Kitabın bütününü kapatan yerel denetim

[Sistem kaydı](evidence/system_walkthrough.json) yedi gerçek chat/context/storage sınırını geçici SQLite ve kontrollü backend ile ölçer. [Araştırma kaydı](evidence/research_walkthrough.json) korunmuş kod üzerinden üç küçük örnek grubunu çalıştırır; son gruptaki 1.296 geçmiş tam tarihsel taramanın yerine konmaz. [Son doğrulama](evidence/manuscript_examples_verification.json), altı eşlikçinin salt denetim kipinde geçtiğini ve 1. bölümün bağımsız örneğinin çalıştığını kaydeder. Bu turda eski sonuçlar yeniden yazılmadı.

Bütün bölümlerde kaynakça veya birincil kaynaklar bölümü vardır. 1–2 temel temsil, 3–6 sinirsel hesap/eğitim, 7–10 çalışma yaşam döngüsü, 11–12 korunmuş araştırma ve açık sınırlar için okunur. Sayısal hassasiyet, model kalitesi ve literatür özgünlüğü ayrı iddialardır. Son kontrolün kapsamı [tamamlanma kaydında](TAMAMLANMA.md) bir arada verilmiştir.

## Akademik kullanımda sürümü belirtmek

GitHub ana dalı gelişmeye devam eder. Bir alıntı veya tekrar üretim kaydında bölüm adıyla birlikte commit veya sabit arşiv sürümü belirtilmelidir. [Yayın dizisindeki](../publications/README.md) DOI belirli bir arşiv anını tanımlar; ana dala daha sonra eklenmiş cümleleri o eski DOI'ye aitmiş gibi göstermemek gerekir. Araştırma kayıtlarının tarihleri ve olumsuz sonuçları korunur; yeni düzeltmeler gerekçesiyle birlikte eklenir.
