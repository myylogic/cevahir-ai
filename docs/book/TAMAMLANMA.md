# İlk tam metnin tamamlanma kaydı

[Kitap girişi](README.md) · [Ayrıntılı içindekiler](AYRINTILI_ICINDEKILER.md) · [Akademik kullanım](AKADEMIK_OKUMA.md)

**21 Eylül 2026 — Türkçe kitabın planlanan 12 bölümlük ilk tam metni tamamlandı. Yazar: Muhammed Yasin Yılmaz.** Bölümler boş bir iskelet veya gelecekte doldurulacak konu listesi değildir: açıklamalar, matematik, gerçek uygulama bağlantıları, kaynaklar ve çözümlü alıştırmalar bulunur. Terim dizini ve bütünleştirici uygulama da tamamlandı. Bu kayıt yerel çalışma kopyasına aittir; bu tamamlanma turunda commit, GitHub gönderimi veya yeni DOI yayını yapılmadı.

“Tamamlandı”, burada aşağıdaki öğrenme kapsamındaki ilk tam kitap metninin tamamlandığını belirtir. Dış akademik hakemlik, son dizgi/PDF baskısı, tam İngilizce çeviri veya genel yaşarken öğrenme probleminin çözümü anlamına gelmez. Önceki DOI arşivleri kendi sürümlerini tanımlar; bu yerel eklemeler henüz o arşivlerde bulunmaz.

## Bölüm kapsamının kapanışı

| Bölüm | Tamamlanan öğrenme kapsamı | Konu / alt başlık | Çözümlü alıştırma |
|---|---|---:|---:|
| [1](tr/01-motor-ve-ogrenme.md) | Motor–model ayrımı, vektör/matris/tensör, olasılık, loss, genelleme, ilk gerçek forward | 12 / 5 | 5 |
| [2](tr/02-metin-tokenizer.md) | BPE ve alternatifler, gerçek sözlük/merges, preprocessing, ID/decode, dosya/metot haritası | 12 / 3 | 5 |
| [3](tr/03-sinir-aglari.md) | Sinir ağı, embedding, affine/nonlinear hesap, başlangıç, geri yayılım ve ağırlık paylaşımı | 11 / 11 | 6 |
| [4](tr/04-attention.md) | Q/K/V, ölçek, maske, MHA/MQA/GQA, KV cache, SDPA/Flash koşulları | 12 / 10 | 5 |
| [5](tr/05-transformer.md) | Residual, norm, konum/RoPE, FFN/SwiGLU, MoE, sayısal hassasiyet ve bellek | 12 / 19 | 7 |
| [6](tr/06-egitim.md) | Veri ayrımı, hedef, loss, gradient/optimizer, accumulation, pretraining, SFT, LoRA/RLHF/DPO/distillation karşılaştırması | 12 / 7 | 5 |
| [7](tr/07-model-yasam-dongusu.md) | Model/tokenizer kimliği, checkpoint, bütünlük, resume, sayı biçimi ve quantization | 11 / 6 | 5 |
| [8](tr/08-uretim.md) | Autoregressive döngü, sampling, ceza/filtre, EOS, cache ve belirsizlik ölçüsü | 12 / 0 | 5 |
| [9](tr/09-bellek-bilis-araclar.md) | Retrieval, bellek kapsamı, cognition/critic, araç yürütme ve calculator karşı örneği | 11 / 8 | 5 |
| [10](tr/10-uctan-uca-sistem.md) | Facade/HTTP/sohbet yolları, kimlik, transaction, tekrar gönderme, bağlam bütçesi ve kısmi başarısızlık | 12 / 5 | 5 |
| [11](tr/11-arastirma-laboratuvari.md) | Edinim, konsolidasyon, koruma, temsil, öğrenilmiş güncelleme, durum yordamı, düzeltme ve sorgu sözleşmesi deneyleri | 15 / 5 | 5 |
| [12](tr/12-acik-sorular.md) | Genel ilkeler ve aileye bağlı sonuçlar, gelecek işlem yeterliliği, plastisite, iç durum ve açık araştırma sınırları | 10 / 2 | 5 |
| **Toplam** | **12 bölüm** | **142 / 81** | **63** |

Bu sayımda “konu” Markdown `##`, “alt başlık” `###` düzeyidir; kaynakça ve alıştırma başlıkları da dahildir. Bağımsız bilimsel kavram sayısı olarak yorumlanmaz. Bölüm dosyaları, kod/denklem/tablo ve bağlantı yazımları dahil boşlukla ayrılan **35.318 sözcük birimi** içerir; yaklaşık uzunluk ölçüsüdür. [Makinece üretilen envanter](evidence/manuscript_inventory.json) bölüm bazında aynı sayıları ve dosya özetlerini taşır. [Terim dizininde](TERIMLER.md) **76 terim satırı** ve ayrıca matematiksel gösterimler bulunur.

## Uygulama ve kanıt zinciri

Kitap altı çalıştırılabilir eşlikçi içerir. [Son yerel doğrulamada](evidence/manuscript_examples_verification.json) hepsi kayıtları yenilemeden geçti; giriş bölümünün bağımsız Python örneği de ayrıca çalıştırıldı.

| Eşlikçi | Örnek/grup | Gerçekte sınanan |
|---|---:|---|
| Tokenizer | 6 metin örneği | Dağıtılan 60.000 öğeli sözlük ve 24.896 yüklenmiş kural; ek decode/batch kontrolleri; varlıklar değişmedi |
| Sinirsel hesap | 7 grup | Gerçek embedding, attention/cache, FFN/norm/RoPE ve küçük model güncellemesi |
| Model yaşam döngüsü | 5 grup | Geçici checkpoint, kimlik, optimizer devamı ve bütünlük sınırları |
| Üretim/bellek/araç | 5 grup | Sabit logitlerle gerçek adaptör, elle vektörlerle retrieval, gerçek calculator |
| Uçtan uca çevreleyen sistem | 7 grup | Geçici SQLite ve sabit backend ile gerçek chat/context/storage/istek kapsamı |
| Sonlu araştırma | 3 grup | Düzeltme tanığı, sorgu ayrımı ve 1.296 geçmişlik tek hücre denetimi |

Toplam **33 örnek veya örnek grubu**, tek tip 33 benchmark anlamına gelmez; bazıları birden çok alt kontrol içerir. Fixture ile gerçek bileşenin sınırı her eşlikçide açıklanır. GPU hızı, eğitilmiş modelin dil kalitesi ve bütün repository'nin kusursuzluğu bu sayıdan çıkarılamaz. Daha önce çalıştırılmış 149, 99 ve 70 hedefli test ayrı tarihli kapsamlarıyla korunur; bunlar bu turda yeniden yapılmış veya bağımsız testlerin toplamı gibi gösterilmez.

[Bütünleştirici çalışma](BUTUNLESTIRICI_CALISMA.md), okuyucuyu metinden checkpoint'e, hizmet sınırından araştırma iddiasına kadar yedi aşamada yürütür. Her aşamada beklenen çıkarım ve kanıt sınırı yazılıdır. [Kaynak haritası](KAYNAK_HARITASI.md) gerçek uygulamaya, [ayrıntılı içindekiler](AYRINTILI_ICINDEKILER.md) bölüm içindeki konuya götürür.

## Kapanışta korunan sınırlar

Kaynak bağlantısı ve AST sembol denetimi davranışın otomatik ispatı değildir; anlatım ayrıca uygulamayla karşılaştırıldı. Literatürün özgün makaleleri/resmî belgeleri yöntem kaynağıdır; başka modelin sonucu Cevahir sonucu olarak aktarılmadı. Önceki araştırma raporları ve negatifler korunur. Tam sayımlı sınırlı deneyler geniş living-learning sorusunun cevabı ilan edilmedi.

Kitapta görünen giriş doğrulama sınırları, yaklaşık geçmiş bütçesi, iki ayrı mesaj transaction'ı, desteklenmeyen eğitim seçenekleri ve araştırma varsayımları metin eksikleri değildir; incelenen uygulamanın somut sınırlarıdır. Gelecekte uygulama değişirse ilgili bölüm ve kanıtı birlikte güncellenmelidir.

Sayımı yeniden kontrol etmek için `python scripts/book_inventory.py`; yerel bağlantı/semboller için `python scripts/check_book.py`; tarihsel yayın kaynaklarını koruma için `python scripts/check_publications.py` çalıştırılır. Envanter ve incelenmiş kaynaklar ancak metin/kod incelemesinden sonra açık kayıt seçenekleriyle yenilenir. [Bakım rehberi](BAKIM.md) bu ayrımı korur.
