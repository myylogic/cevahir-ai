# Cevahir — Yaşarken Öğrenme Araştırmaları

**Muhammed Yasin Yılmaz** · Araştırma raporları ve ön baskılar · 21 Eylül 2026

Bu dizi, Cevahir'in çalıştığı süre boyunca deneyimden öğrenme araştırmasının
makale düzenindeki yayınlarını, olumsuz sonuçlarını ve yeniden üretilebilir
hesaplarını bir araya getirir. Genel soru hâlâ açıktır: **Bir sistem yaşadığı
deneyimler nedeniyle gelecekteki hesaplama ve davranış kapasitesini nasıl kalıcı,
kontrollü ve genellenebilir biçimde değiştirebilir?**

Metinler dış hakem değerlendirmesinden geçmemiş araştırma raporlarıdır. Her birinde
Türkçe özet, İngilizce abstract, bulguların yorumu ve tam özgün araştırma raporu
bulunur. Kaynak kod, ölçüm, karşı örnek ve literatür bağlantıları korunur. Hazır
özellik dili, ayrıcalıklı geri bildirim veya küçük sonlu dünya varsayımı, genel
öğrenmenin çözüldüğü şeklinde yorumlanmaz.

## Yayın dizini

| Kimlik | Çalışma | Temel sonuç |
|---|---|---|
| CEV-2026-01 | [Deneyimle koşullanan hesaplama](papers/01-experience-conditioned-computation.md) | Kalıcı dış geri bildirimle strateji seçimi; gerçek kalite kazancı henüz ölçülmedi. |
| CEV-2026-02 | [Temsil aramasında tanık filtresi](papers/02-representation-witness-negative.md) | Aynı tam aramada yeni öğrenme avantajı iddiası elendi. |
| CEV-2026-03 | [Kalıcı yordam, aktarım ve yeniden sınama](papers/03-persistence-transfer-retesting.md) | Bilgi, erişim maliyeti ve değişim altında fayda ayrıldı. |
| CEV-2026-04 | [Temsil, kredi atama ve durum taşıma](papers/04-representation-credit-state-transport.md) | Bugünkü cevabın korunması gelecekteki öğrenmeyi korumayabilir. |
| CEV-2026-05 | [Tek yaşam akışı ve zamansal ilişki](papers/05-joint-lifetime-temporal-relations.md) | Birleşen mekanizmalar hata ve dağılım değişiminde birlikte sınandı. |
| CEV-2026-06 | [Edinilmiş güncelleme davranışı](papers/06-learned-update-transfer.md) | Aynı ilk tahminden farklı öğrenme eğrileri; aktarım tersine dönebilir. |
| CEV-2026-07 | [Edinilmiş yinelemeli durum yordamı](papers/07-acquired-recurrent-state.md) | Küçük yürütme grafiği, yeniden öğrenme arşivinden ayrıldı. |
| CEV-2026-08 | [Düzeltilebilir öğrenme durumu](papers/08-correctable-learning-state.md) | Tam bugünkü aday kümesi bile düzeltmeye yetmeyebilir. |
| CEV-2026-09 | [Gelecekteki sorgular ve gerekli durum](papers/09-future-query-state-classes.md) | Aynı geçmiş için sorgu ve düzeltme haklarına bağlı farklı bilgi sınırları. |

## Kanıtları okumak

[Makinece okunabilir katalog](catalog.json), her edisyonu değişmeden korunan
araştırma kaydına bağlar. [Koruma dökümü](evidence/prior_manifest.json), yayından
önceki araştırma dosyalarının bayt özetlerini taşır. [Doğrulama kaydı](evidence/validation.json),
bu sürüm hazırlanırken gerçekten yeniden çalıştırılan deney ve kontrolleri
önceki çalıştırma kayıtlarından ayırır. Bir sonuç dosyasının korunmuş olması,
deneyin bu tarihte tekrar çalıştırıldığı anlamına gelmez.

Tam öğrenme yaşam döngüsü için [teknik kitap](../book/README.md),
[araştırma laboratuvarı](../book/tr/11-arastirma-laboratuvari.md) ve
[açık sorular](../book/tr/12-acik-sorular.md) birlikte okunabilir.
[Matematiksel çalışma notu](working-notes/future-state-review.md), dokuzuncu
çalışmanın deneyden önceki türetimlerini saklar.

## Atıf ve sürüm arşivi

Sürüm: **2026.09.21**. Git etiketi: `research-2026.09.21-v1`.
[GitHub sürümü](https://github.com/myylogic/cevahir-ai/releases/tag/research-2026.09.21-v1),
metinleri, kaynak kodu, deney kayıtlarını ve kitabı birlikte arşivler.
[DOI yayın kaydı](release.json), dış hizmetten doğrulanmış kimliği ve yayın
durumunu taşır. Zenodo kaydı yazılım/araştırma arşivi türündedir; dizideki her
metne ayrı DOI atanmış olduğu ileri sürülmez.

Atıf biçimi: **Yılmaz, M. Y. (2026). Çalışma başlığı (CEV-2026-XX). Cevahir AI:
Living-Learning Research Reports and Technical Book, sürüm 2026.09.21. Sürüm DOI'si.**
Makinece kullanılabilir koleksiyon künyesi [CITATION.cff](../../CITATION.cff)
dosyasındadır. DOI, içeriğin belirli arşiv sürümünü tanımlar; bilimsel doğruluk,
özgünlük, kişisel kimlik belgesi veya kriptografik imza onayı değildir.

## Katkı açıklaması ve devamlılık

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, deneyler, literatür
incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır. Haricî hakemlik,
kurum ilişkisi veya ORCID doğrulaması yapılmış gibi gösterilmemiştir.
Lisans, deponun [Apache-2.0 lisansıdır](../../LICENSE).

Sonraki anlamlı araştırma sonuçlarında metin, kanıt ve kitap birlikte güncellenir;
önceki sürümün üzerine yazılmadan yeni bir arşiv sürümü çıkarılır.
[Yayın ve doğrulama yöntemi](PUBLISHING.md) bu devamlılığın kaydıdır.

## English reading note

Nine individually identified research reports are published with English
abstracts and full Turkish technical records. The collection DOI identifies a
versioned software and research archive, not nine independently registered DOIs.
The reports preserve negative findings and explicitly leave the broad
living-learning problem open. Computational verification is scoped in the
evidence record; it is not external peer review. The accompanying
[English book guide](../book/README-en.md) maps the implementation-grounded book.
