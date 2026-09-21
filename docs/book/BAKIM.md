# Kod, deney ve kitabı birlikte güncel tutmak

[İçindekiler](README.md) · [Kaynak haritası](KAYNAK_HARITASI.md)

Cevahir'in araştırma ve eğitim yönü iki paralel iş gerektirir: mekanizmayı geliştirmek ve gerçekte ne yaptığını izlenebilir anlatmak. Kitap üretim/araştırma kodunu öğretici bir maketle değiştirmez. İngilizce mevcut belgeler korunur; Türkçe bölümlerdeki değişiklik İngilizce navigasyona da yansıtılır. Bir mimari kararın tarihsel gerekçesi kayıtta yoksa yalnız koddan çıkarılabilen mühendislik etkisi anlatılır.

## Anlamlı bir değişiklikte

1. Değişen hesap veya davranışı, çağıranı, girdi/çıktıyı ve etkin config'i belirleyin. Sadece dosya adı değişse bile kitap bağlantılarını kontrol edin.
2. [Kaynak haritasından](KAYNAK_HARITASI.md) ve `evidence/reviewed_sources.json` içindeki `chapters` alanından etkilenen bölümleri bulun. İlgili modül rehberlerini de gözden geçirin.
3. Bölümde teori → gerçek mekanizma → dosya/sembol → çağıran/girdi/çıktı → test/deney zincirini güncelleyin. “Destekleniyor”, “varsayılan”, “test edildi” ve “yararlı bulundu” ifadelerini birbirinin yerine koymayın.
4. Uygun hedefli testi veya deneyi çalıştırın. Donanımı, bilgiyi hangi tarafın gördüğünü, metriği ve negatifleri kaydedin. Sırf doküman bağlantısı değişti diye ağır eğitim başlatmayın.
5. Yeni sınıf/metotları ilgili `evidence/*.json` manifestine ekleyin; `path` repository köküne, `chapter` kitap köküne göredir. Varsa `line`, gerçek `def/class` satırıdır.
6. Eski araştırma raporunu geri yazmayın. Yeni sonuç için ayrı tarihli kayıt açın; önceki sonuçla ilişkisini ve sonradan tasarlanmış kontrolleri belirtin. Kitabın 11–12. bölümlerini ve gerekiyorsa README araştırma bağlantısını güncelleyin.
7. Aşağıdaki denetimi çalıştırın. Kodun anlatımı değiştirmediğini de inceleyerek belirlediyseniz bunu değişiklik açıklamasında söyleyin; ardından parmak izini yenileyin.

## Bağımlılıksız yerel denetim

Repository kökünden:

```powershell
python scripts/check_book.py
```

Denetim, kitaptaki fence dışı standart inline Markdown bağlantılarının yerel hedeflerini, Markdown başlıklarını ve `#L…` satır sınırlarını kontrol eder. Python kaynaklarını **import etmeden**, AST üzerinden kayıtlı sınıf/metot adlarını ve verilmiş satırlarını doğrular. Her Türkçe bölümün sembol kanıtı olmasını ve iki root README'de kitap girişi bulunmasını ister. Kitabın dışındaki bağlandığı dosyaların SHA-256 parmak izlerini kayıtla karşılaştırır; değişen dosya için etkilenen bölümleri bildirir.

Metin dosyalarının parmak izi UTF-8 ve LF satır sonlarına normalize edilir; Windows CRLF ile Linux LF checkout'ları aynı sayılır. İkili dosyalarda bayt özeti kullanılır. Denetim bütün Markdown dilini ayrıştıran bir araç değildir: HTML içindeki linkler, referans-tarzı linkler ve dış URL erişilebilirliği kapsam dışındadır. Kitapta kaynak navigasyonu için standart inline link kullanın. Kaynak hash değişmese bile anlatım yanlış olabilir; test sonuçlarının doğruluğunu veya Mermaid görselini bu script kanıtlamaz. Linkin bir kaynak satırına gitmesi de o satırın doğru mekanizmayı anlattığının otomatik kanıtı değildir.

İnsan/kaynak incelemesi tamamlandıktan sonra:

```powershell
python scripts/check_book.py --record-reviewed-sources
python scripts/check_book.py
```

İlk komut yalnız açıkça incelenmiş yeni dosya durumunu kaydeder. Hatalı dosya/simge bağlantısı varsa kaydı yenilemez. Bunu CI'da otomatik “onarım” adımı olarak kullanmayın; aksi halde eskime uyarısı anlamını kaybeder. İkinci komut yazmaz. `.github/workflows/book.yml` push ve pull request'te yalnız denetimi çalıştırır. Yerel olarak başarılı olması, workflow'un GitHub'da çalıştırıldığı anlamına gelmez.

## Yeni bölümün asgari içeriği

Bir kavramı tanımak için gerekli matematiği, çözdüğü problemi ve en yakın alternatifini anlatın. Cevahir'deki gerçek seçimi etkin çağrı yolu üzerinden gösterin. Kod parçası kısa ve kaynağa bağlı olsun; büyük dosyayı kopyalamayın. Varsayımı ve bilinen sınırı aynı yerde belirtin. Okur metottan önceki girdiye ve sonraki tüketiciye gidebilmeli.

İkinci bölümde bunun somut örneği konu → dosya/metot → girdi/işlem/çıktı haritasıdır. Tokenizer veya dağıtılan varlıklar değiştiğinde `python scripts/book_tokenizer_walkthrough.py` ile kayıtlı altı örneği yeniden denetleyin. Bir farklılık çıkarsa önce nedenini ve bölümdeki anlatımı inceleyin; yalnız bundan sonra `--write` ile yeni çıktıyı kaydedin. Bu örnekler model eğitmez ve tokenizer varlıklarını değiştirmediklerini denetler.

Araştırma bölümünde buna hipotez, bilgi erişimi, güçlü baseline, ayırıcı negatif, tekrar üretim komutu ve hangi sonucun hâlâ açık kaldığı eklenir. Var olmayan bir özellik için tahmini sınıf adı yazılmaz. Hazır özellik dili veya ayrıcalıklı doğrulama verisi varsa saklanmaz. Tek bir küçük başarılı deney ana sorunun çözümü diye sunulmaz.

## Araştırma yayınları

Anlamlı araştırma değişikliklerinde ayrıca [yayın dizisini](../publications/README.md)
ve [yayın sürecini](../publications/PUBLISHING.md) izleyin. Yeni deneyin raporu,
makale edisyonu ve kitap açıklaması aynı kapsamı taşımalıdır. Önceki raporların
üzerine yazılmaz; yeni sürüm DOI'si gerçek arşiv kaydından doğrulandıktan sonra
eklenir. `python scripts/check_publications.py` edisyon, kaynak koruması ve
atıf bilgilerinin yapısal eşleşmesini denetler.

## English maintenance note

Keep the Turkish book, its English reading guide, existing bilingual module documentation and relevant tests aligned with meaningful changes. Source code determines current behavior; historical research records remain intact. Run `python scripts/check_book.py`. After reviewing the affected chapters against the source, explicitly record reviewed fingerprints with `--record-reviewed-sources`, then check again. The CI workflow only checks; it never refreshes evidence automatically. A passing structural check does not prove semantic correctness or research generality.

## Sinirsel hesap örnekleri

3–6. bölümleri etkileyen bir değişiklikte `python scripts/book_neural_walkthrough.py` komutunu çalıştırın. Yeni bir ölçümü kayıtlı sonucun yerine koymadan önce matematiksel iddiayı, gerçek çağrı yolunu ve toleransı inceleyin. Kaynakçadaki yöntem kökeni ile uygulama iddiası ayrı güncellenmelidir. [Akademik okuma rehberi](AKADEMIK_OKUMA.md) bu kapsamı açıklar.
