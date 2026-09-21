# Temsil aramasında yetersizlik tanıkları: deterministik eşdeğerlik ve negatif sonuç

**Muhammed Yasin Yılmaz**

CEV-2026-02 · Sürüm 2026.09.21.2 · Yayın tarihi 2026-09-21

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

Aynı aday dili, aday sırası ve tam tutarlılık ölçütü altında yetersizlik tanığıyla filtrelenen temsil araması ile doğrudan arama karşılaştırılır. Tutarlılık tanık koşulunu zaten gerektirdiğinden, tamamlanan aramalar aynı çözümü seçer. On beş sonlu karşılaştırmanın tamamı bu eşdeğerliği doğrulamıştır. Bazı hesap maliyetleri değişse de yeni bilgi veya gözlem tasarrufu gösterilmemiştir.

## Abstract

Witness filtering and direct representation search are equivalent when they share candidate language, order and complete deterministic consistency checking. All 15 finite comparisons yield identical selected representations, predictions and abstentions. Filtering changes some search costs but provides no additional learning information under this contract.

## Bulguların yorumu

Tanıkla filtrelemenin bu koşullarda yeni öğrenme avantajı sağladığı hipotezi elendi; negatif sonuç korunmuştur.

Sonuç, farklı bilgi erişimi, açık uçlu ifade üretimi, yaklaşık tutarlılık veya kesilen hesap bütçeleri için aynı üstünlük/eşdeğerlik iddiasını kurmaz.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/REPRESENTATION_WITNESS_AUDIT.md](../../research/REPRESENTATION_WITNESS_AUDIT.md) olarak korunur. Bağlantı yolları bu
edisyonun konumuna uyarlanmıştır. Rapordaki çalışma tarihi ve geçmiş yayın durumu
ifadeleri tarihsel kayda aittir; bu edisyonun tarihi yukarıdadır.

Bu çalışma [yayın dizisi](../README.md) içindeki ayrı bir metindir. DOI, bütün
metinleri, teknik kitabı ve kodu içeren sürüm arşivini tanımlar; her makaleye ayrı
DOI verildiği anlamına gelmez. Atıfta çalışma kimliği ve sürüm DOI'si birlikte
kullanılmalıdır. Hesaplamalı denetimlerin kapsamı [doğrulama kaydında](../evidence/validation.json)
ve [yayın ilkelerinde](../PUBLISHING.md) açıklanır.

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, hesaplamalı deney,
literatür incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır.
Bu katkı açıklaması, DOI kaydı veya otomatik testler dış bilimsel hakemlik sayılmaz.

---

## Araştırma kaydı: Temsil değişimi hipotezi: ilk yanlışlama deneyi

20 Eylül 2026. [Bağımsız keşif kaydındaki](../../research/FRONTIER_DISCOVERY_2026_09_20.md) H4'ün devamıdır. Önceki araştırma uygulamalarına dokunulmadı; bu çalışma ana çalışma akışından bağımsızdır.

**Sonuç:** Aynı aday dili, aday sırası, gözlemler ve tam deterministik tutarlılık ölçütü altında, yetersizlik tanıklarıyla filtreleme yeni bir öğrenme avantajı sağlamıyor. Tamamlanan iki arama aynı temsilleri ve aynı tahminleri buluyor. Bu dar mekanizmayı bağımsız bir öğrenme yeniliği olarak seçmekten vazgeçildi. Temsilini değiştirebilme yeteneğinin değeri veya başka mekanizmalarla gerçekleştirilebilirliği bu sonuçla reddedilmiş değildir.

## Önce mantıksal sınır

Bir temsilin eğitim verisindeki aynı anahtarı farklı sonuçlara bağlamamasına `C_D(R)` diyelim. Tanık filtresi, aynı başlangıç anahtarına sahip fakat sonuçları farklı iki eğitim örneğini yeni temsilin ayırmasını istiyor: `W_D(R)`.

Gerçek bir eğitim çelişkisinden oluşturulan bu filtre, tam tutarlılığın zaten gerekli koşuludur:

```text
C_D(R) ⇒ W_D(R)
{R : C_D(R)} = {R : W_D(R) ve C_D(R)}
```

Dolayısıyla aynı sırayla tamamlanan aramalar aynı ilk adayı seçer. Öğrenilmiş anahtar–sonuç tabloları da aynıdır. Aynı tahmin ve soru seçme kuralı kullanılırsa bir sonraki gözlemi isteme kararı da farklılaşamaz. Bu son ifade mantıksal çıkarımdır; aşağıdaki deney aktif soru seçimini uygulamıyor.

Filtre bazı adayları daha erken reddedebilir. Bu bir hesap düzenlemesidir; daha fazla bilgi edinildiği anlamına gelmez. Karşı örneklerle aramayı daraltmanın kendisi de yeni bir araştırma ilkesi olarak sunulamaz; örneğin [abstraction refinement ile program sentezi](https://www.microsoft.com/en-us/research/publication/program-synthesis-using-abstraction-refinement/) ilgili önceki çalışmadır. Buradaki eşdeğerlik gerekçesi kaynaktan alıntı değil, yukarıda tanımlanan iki arama için türetilmiştir.

## Kurulan küçük düzenek

Uygulama: [`witness_audit.py`](../../../research/representation_discovery/witness_audit.py). Yalnız Python standart kütüphanesi kullanılır. Sinir ağı, tokenizer, checkpoint, eğitim ve mevcut bilişsel sistem yüklenmez.

- Üç deterministik dünya: bağımsız kaynaklar, toplam bir birim kapasite, toplam iki birim kapasite.
- Eğitimde üç, ayrılmış değerlendirmede farklı isimli beş nesne bulunur. Her geçmiş bağımsız sıfırlanan bir bölümdür; en fazla iki geçmiş talep vardır. Başarılı talep, o bölüm boyunca bir birimi kullanır.
- Öğrenici yalnız önceki eylemleri, gözlenmiş sonuçlarını ve sıradaki hedefi görür. Gizli kapasite ve dünya etiketi yalnız veri üreten değerlendirme tarafındadır.
- Başlangıç temsili hedefin daha önce başarıyla kullanılıp kullanılmadığını belirtir. 24 ek aday, genel gözlenebilir sayaçları `0, 1, 2` eşikleriyle karşılaştırır. Toplam 25 adayın sırası iki yöntemde aynıdır.
- Genel arama, her aday için anahtar–sonuç tablosu kurar ve ilk çelişkide durur. Gereksiz biçimde bütün örnek çiftlerini tarayan zayıf bir rakip kullanılmaz.
- Filtreli arama, başlangıç temsili başına en fazla bir gerçek eğitim tanığı çıkarır. Tanık üretme ve kontrol maliyeti sayılır. Filtreden geçen adaylar aynı tam denetime girer. Az sayıdaki tanık tek başına yeterli değildir.
- Her dünyada 3, 6, 12, 24 ve 39 etiketli bağlamla ayrı ayrı yeniden öğrenilir. Her değerlendirme 155 bağlam içerir. Bunlar fiziksel ortam çağrısı sayıları değildir: geçmiş eylem sonuçları da örneklerin içinde sağlanır.
- Hiç görülmemiş bir temsil anahtarında yöntem çekimser kalır. Çekimserlik doğru cevap sayılmaz. Değerlendirme etiketleri kullanılmadan önce bütün tahminler sabitlenir.

Bu dilde sayaçlar ve küçük eşikler araştırmacı tarafından verilmiştir. Doğru adayın seçilmesi “yoktan kavram oluşturma” veya dünyanın tek doğru nedensel yapısını keşfetme değildir. Yeni nesne isimleri üzerindeki sonuç da bilinmeyen gizli grup üyeliklerine sıfır gözlemle aktarım anlamına gelmez.

## Ölçülen sonuç

Makine tarafından üretilen kayıt: [`representation_witness_audit.json`](../../../benchmarks/results/representation_witness_audit.json).

**15 karşılaştırmanın tamamında** uyumlu adayların sıralı kümesi, öğrenilmiş tablolar, seçilen aday, değerlendirme tahminleri ve çekimserlikler aynı çıktı. Aşağıdaki satırlar her dünyadaki son, 39 bağlamlı eğitim noktasını gösterir:

| Dünya | Her iki yöntemin seçtiği ayrım | Her iki yöntemin değerlendirme doğruluğu | Genel arama iş sayacı | Filtreli arama iş sayacı |
|---|---|---:|---:|---:|
| Bağımsız | Hedefin kullanılmış olması | 155/155 | 7.845 | 7.986 |
| Toplam kapasite 1 | Başlangıç ayrımı + etkin nesne sayısı < 1 | 155/155 | 3.111 | 3.073 |
| Toplam kapasite 2 | Başlangıç ayrımı + etkin nesne sayısı < 2 | 155/155 | 3.981 | 2.798 |

İş sayacı ifade yürütme ve satır/çift kontrol işlemlerinin açık maliyet modelidir. CPU komutu, bütün Python işlemlerinin sayımı veya ölçülmüş hızlanma oranı değildir. Tanık hazırlığı dahil edilmiştir. Aynı yöntemin bazı koşullarda fazla, bazılarında az hesap yapabildiği görülüyor; öğrenilen sonuç değişmiyor.

Bu çalıştırmada toplam denetim yaklaşık **0,17 saniye** sürdü. İzlenen Python tahsisatlarının tepe değeri **162.972 bayt** idi; bu, sürecin toplam RAM tüketimi değildir. Üst limitler 10 saniye, 1.000.000 toplam iş adımı, arama başına 100.000 iş adımı ve 16 MiB izlenen Python tahsisatıdır. Küçük girdi/adayı sınırları ayrıca uygulanır. Süre ve toplam iş sınırı bütün denetimin koruyucusudur; bütçesi yarıda kesilen iki yöntemin hızını karşılaştırmak için kullanılamaz.

Tamamlanmayan koşul `inconclusive` olarak kalır; eksik aday kümesinden başarı sonucu üretilmez. Daha önce kanıtlanan bir uyuşmazlık, sonradan kaynak sınırına ulaşılmasıyla gizlenmez. Denetim başka bir bellek izleme oturumunun içinden çağrılırsa, önceki tahsisatların tepe değere dahil olduğu çıktıda açıkça belirtilir.

## Kontroller ve sınırları

[`test_representation_witness_audit.py`](../../../tests/evolution/test_representation_witness_audit.py) içindeki **15 hedefli test geçti**. Testler şunları denetliyor:

- Dört bağlam üzerindeki bütün ikili sonuç atamalarında ve ters eğitim sıralarında iki tam aday kümesinin eşitliği.
- Aday sırası değiştiğinde iki yöntemin yine aynı seçimi yapması.
- İlk tanığı ayırıp sonraki gözlemlerle çelişen adayın reddi.
- Aynı bağlam için çelişkili sonuç verildiğinde deterministik çözüm bulunamaması.
- Test etiketleri ters çevrildiğinde öğrenilmiş aday seçiminin ve arama maliyetinin değişmemesi.
- Nesneler tutarlı şekilde yeniden adlandırıldığında temsilin aynı kalması.
- Bilinmeyen anahtarda çekimserlik; tanık hazırlığının maliyetinin sayılması.
- Aday, iş, süre ve izlenen bellek sınırlarının yanlış başarı üretmemesi.
- Önceden bulunan karşı örneğin daha sonraki kaynak sınırında kaybolmaması.

İlk keşif taslağındaki **görevler arasında tanık karıştırma** önerisi de düzeltildi: başka bir dünyanın çelişkisini taşımak geçerli adayı haksız yere eleyebilir. Bu adil bir mekanizma ablasyonu değildir. Aynı veride filtreyi kapatmak ve geçerli eğitim/aday sıralamalarını değiştirmek uygun kontrollerdir.

Bu denetim; gürültülü gözlemleri, yaklaşık tutarlılığı, açık uçlu ifade üretimini, yeni araç ailelerini, gerçek dil modeli çıktısını veya geniş hipotezde önerilen tüm rakipleri sınamıyor. İlk keşif belgesindeki geniş deney tasarımı gerçekleştirilmiş sayılmamalıdır. Aynı aday kümesinde bu filtreyle gözlem avantajı iddiasının elenmesi için eşdeğerlik gerekçesi zaten yeterlidir; küçük deney uygulamanın bu gerekçeye uyduğunu kontrol eder.

## Araştırma kararı

“Yetersizlik tanığıyla mevcut adayları filtrelemek daha az gözlemle öğrenir” iddiası, **tamamlanan deterministik arama biçiminde elendi**. Bu iddia için önceki taslakta önerilen %20 gözlem tasarrufu hedefini kovalamak veya ana sisteme yeni katman eklemek gerekçesiz olurdu.

Temsil değişimi araştırması yeniden ele alınacaksa yeni mekanizma bu eşdeğerliği gerçekten aşan bir değişiklik tarif etmelidir: hangi bilgiye erişildiğini, hangi ifadelerin üretilebildiğini veya sınırlı toplam hesapta neyin gerçekleştirilebildiğini açıkça değiştirmesi gerekir. Bu seçenekler henüz seçilmiş yeni yöntemler veya özgünlük iddiaları değildir. Aynı önermeyi yeni isimle yeniden denemek araştırma ilerlemesi sayılmayacak.

Ana sisteme entegrasyon yapılmadı. Hazır olan çıktı; tek komutla tekrar üretilebilen küçük deney, bilgi sınırları, karşı örnek kontrolleri ve elenen iddianın gerekçesidir. Önceki turun 38 dosyası içerik özetleriyle karşılaştırılarak korunduğu doğrulandı. GitHub'a gönderim veya commit yapılmadı.

```powershell
python -m unittest tests.evolution.test_representation_witness_audit
python -m research.representation_discovery.witness_audit --output benchmarks/results/representation_witness_audit.json
```
