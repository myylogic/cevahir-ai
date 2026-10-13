# Tur 23 — çok yönlü living-learning deney kampanyası

Muhammed Yasin Yılmaz · 24 Eylül 2026 · Yerel araştırma

## Amaç ve sayım

Ana soru: Çalışan sistemin deneyim nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir değiştirmesi. Kullanıcının talebi üzerine, tek küçük deneyi uzatmak yerine on araştırma yönü için 200 araştırma önermesi kaydedilecek. Bunlar 200 yeni bilimsel ilke değildir. Koşullu mekanizma karşılaştırmaları aynı aile altında gruplanır; tohumlar, tekrarlar ve parametre noktaları hipotez sayılmaz.

İlk çalıştırılabilir dalga dört farklı deney motorunda 48 koşullu hipotezi hedefler: bellek ve düzeltme (MEM), temsil edinimi (REP), güncelleme davranışının edinimi (UPD), eylemlerle deneyim üretme (ENV). Diğer 152 önerme katalogda `queued` olarak açıkça işaretlenir; sınanmış gibi sayılmaz. CON, RES, XFR, LAN, CRD ve CTX sonraki motorları bekler. Her aile için tam 20 önerme vardır. Dört motor, genel teorinin birleşik çözümü değildir; birbirine entegre model kurulduğu iddia edilmez.

## Çalıştırmadan önce sabitleme

Her motor kendi katalog ve yöntem protokolünü başarı sonuçlarına bakmadan yazar. Önce yapısal/selfcheck denetimleri yapılır. Bütün motor, katalog, ana çerçeve ve protokol hash'leri `results/freeze.json` içinde sabitlenir; daha sonra keşif tohumları 0–23 ve doğrulama tohumları 1000–1047 çalıştırılır. Doğrulama sonuçlarına göre yöntem veya eşik ayarlanmaz. Dört motorda 72 tohum = 288 motor-yaşamı, 48 iddiada 72 eşleştirilmiş gözlem = 3.456 karşılaştırma sonucu hedeflenir. Aynı motor-yaşamını paylaşan iddialar bağımsız deneyler diye sunulmaz.

Bir uygulama hatası bulunursa başarısız çıktı saklanır; manifest yeni sürüm olmadan sessizce güncellenmez. Yalnız kod eşitliği testi bilimsel bağımsız tekrar değildir. Aynı model/dünya dağılımındaki yeni tohumlar gerçek dünyaya genelleme kanıtı değildir.

## Karşılaştırma ve kaynak hesabı

Her çalıştırılabilir önerme aday, kontrol, rejim, birincil ölçüt, yön ve pratik fark eşiği içerir. Skor daima küçüldükçe iyileşir. Güçlü aynı-bilgi rakipleri, öğrenilmiş bileşeni kapatma ve olumsuz aktarım koşulları motor protokollerinde belirtilir. Gizli üretici parametresi öğreniciye geçirilmez; öğretmen etiketi yalnız gerçek gözlem anında verilir. Oracle/hindsight bilgisi varsa ayrıcalıklı tanı olarak etiketlenir, adil kontrol sayılamaz.

Hata, sanal hesap, ham/özet bellek ve edinim ayrı ölçülerdir. Bir motorun işlem sayacı diğer motorun işlem sayacıyla aynı donanım birimi değildir. Motorlar arası maliyetler tek bir evrensel kazanç oranında toplanmaz. Birçok testte daha düşük hata daha çok hesap veya belleğe mal olabilir; yalnız hata başarısı kaynak üstünlüğü diye sunulmaz. Bilgi sınırı, araştırmacının verdiği model diliyle koşulludur.

## İstatistik ve karar

Keşif ve doğrulama sonuçları ayrı raporlanır. Her iddiada tohum başına d = kontrol skoru − aday skoru. Pozitif d adayı destekler. Eşitlik |d|≤10⁻¹² olarak sayılır. Birincil istatistik doğrulama grubunda eşleştirilmiş işaret testidir; büyüklük değil, iyileşme işaretinin sentetik tohumlarda sıklığını sınar. 48 iddianın destek ve ters yön testleri birlikte 96 test üzerinde Holm aile-geneli düzeltmesi alır. p değeri bir mekanizmanın doğruluk olasılığı değildir. Ayrı motorların/iddiaların bağımsızlığı varsayılmaz; Holm bunun için bağımsızlık gerektirmez.

`supported`: düzeltilmiş destek p≤0,05, ortalama d≥katalogdaki pratik eşik ve en az 2/3 doğrulama tohumunda pozitif işaret. `contradicted`: ters yönde aynı koşullar. Diğerleri `inconclusive` (eşitlik kanıtı değildir). `check_failed` sonuç bilimsel karara alınmaz, sessizce atılmaz. Bütün sonuçlar, olumsuzlar ve eşitlikler dahil raporlanır. Motor/iddia sayısı değişirse test ailesi gerçek kayıt sayısıyla hesaplanır ve sapma açıklanır.

Bu filtre teorik ispat değildir; özellikle yapay, gürültüsüz veya eşlenik kurulumlarda p değerleri aşırı küçük olabilir. Etki büyüklüğü, rejim, karşılaştırma ve kalibrasyon kapsamı önceliklidir. Belirli yaşamların tam yolları örnek izlerle, bütün yaşamların sonuçları JSONL ile korunur.

## Kalite ve devam

Kaynak hash'leri, bütün `wave1` maddelerinin çıktı üretmesi, bütün `queued` maddelerin sonuçsuz kalması, benzersiz kimlikler, sonlu sayılar, başarılı mekanik kontroller, deterministik tekrar, önceki dosya/HEAD korunması denetlenir. Son rapor yeni önermeler ile eski çalışmaların kapsam genişletmesini ayırır. Henüz gerçek dil, algı, büyük sinir ağı, bağımsız ilişki dili icadı veya açık uçlu AGI sınanmıyor. Bu ilk geniş tarama, sonraki derin deneylerin seçilmesini sağlar; ana problemi daha kolay dört alt probleme eşitlemez.

Commit, push, yayın veya ana uygulama/kitap değişikliği yapılmaz. Başlangıçtaki tüm dosyaların SHA-256 manifesti korunur.
