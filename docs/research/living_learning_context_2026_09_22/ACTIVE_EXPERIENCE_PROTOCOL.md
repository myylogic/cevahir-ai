# Tur 14 ek sınaması — Sistemin ayırt edici deneyimi seçmesi

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Ana ve ek sonuçlar hesaplanmadan yazıldı

Kullanıcı aktif çalışma sırasında hedefi yeniden vurguladı:

> eğer yaşarken yani inference aşamasında öğrenmeyi sürdürebilen bir yapay zeka modeli oluşturma altyapısı ve sürekli aktif bir model kuramı oluşturabilirsek program çalışmaya başladığı an kendini durmaksızın sürekli geliştirebilecek yeni şeyleri kendi kendine öğrenebilecek kendi kendine ilerleyebilecek bir sistem inşa etmiş olacağız bunu anlıyor musun ? mesela tam olarak bu.

Ana [bağlam protokolü](PROTOCOL.md) deneyimlerin araştırmacı tarafından verilmesini varsayıyor. Bu ek sınama **bir sonraki deneyimi seçme kararını** öğreniciye bırakır. Ana sonuçlar ve parametreler değiştirilmez. Bu ekin öğrenicisi aynı sonlu dünyada verilen sorgu arayüzünü kullanır; kendi hedefini, ölçüm dilini veya eylem araçlarını icat etmez. Gerçek dünyada keyfî geçmişi kurup kusursuz etiket sorgulamak mümkün olmayabilir; buradaki arayüz açık ayrıcalıktır.

## Düzen

24 yaşamın ilk 48 hizalı kaydı aynıdır. İki öğrenici ön olasılığı ayrı değerlendirilir: `joint_uniform_gate` ve `joint_balanced_family`. Her biri için iki politika:

- `uniform_query`: henüz sorgulanmamış bütün 512 gözlenen girdi arasından eşit olasılıkla birini seçer. Rastgelelik dünyadan bağımsız tohumla sabittir.
- `disagreement_query`: kendi mevcut temiz-etiket tahmini p(x) için `p(x)(1−p(x))` değerini en büyük yapan görülmemiş girdiyi seçer. `1e−12` toleransla eşit en büyük değerler arasında aynı bağımsız rastgelelik kaynağıyla seçer. Doğru kapıyı veya gizli etiketi kullanmaz.

En fazla 16 yeni dış sorgu. Her sorgunun doğru etiketi dünyadan alınır, önceki tahmini kaydedilir, sonra kalıcı durum güncellenir. Böylece sistemin bilgi edinmesi gerçek bir dış sonuçla sınanır; kendi tahminini etiket diye yeniden kullanma değildir. Kaynak sorguları, aday kontrolleri ve seçim için taranan 512 girdinin maliyeti sayılır. İlk 48 gözlemin listesi öğrenici durumunda kalmaz; hata sayaçları ve tekrar sorguyu önleyen 512 bitlik görülmüş-maskesi yeterlidir. Görülmüş-maske girdi kimliklerinin izini tutar, etiket belleği değildir.

24×2×2 = 96 aktif edinim yolu; ana 504 yoldan ayrı kaydedilir. Ortak ilk 48 sonrasında sorgu 0/1/2/4/8/16 noktalarında bütün-küp, ters-yüzey ve görülmemiş hata; doğru kapı ve yüzey kapısı ağırlığı; alınan dış etiket sayısı; aday–etiket kontrolleri; seçim için hesaplanan aday–girdi hücreleri raporlanır. `disagreement_query` tam küp tahminini seçim için gerçekten kullandığından bu hesaplamalar kontrol maliyetinden gizlenmez. Ana değerlendirme taramaları seçim maliyeti değildir. Rastgele kontrolün de puanlama gerekmeden seçim yapabildiği korunur.

Kaydedilecek iddia: Bilgisiz bekleme veya sırf yeni etiket alma yerine, mevcut ayrışmanın yüksek olduğu girdi için deneyim seçmek bazı koşullarda daha hızlı genellemeye yol açabilir. Her adımda mutlaka daha iyi olma, toplam CPU üstünlüğü, bilinmeyen aday ailelerinde işe yarama veya genel özerklik garantisi aranmaz. Yanlış aday dili içindeki yüksek belirsizlik, doğru deneyimi seçmenin evrensel ölçütü değildir.

## Doğrulama

- Seçim yordamının girdisi yalnız 512 tahmin, görülmüş-maske ve dünyadan bağımsız rastgeleliktir; gizli doğru kapı/etiket girişi yoktur.
- Bütün seçilmiş girdiler gerçekten görülmemiş olmalı; seçilen etiket, dünya yürütmesinden bağımsız skaler kontrolle doğrulanmalı.
- Disagreement seçimleri her adımda görülmemiş havuzdaki en yüksek puanla eşleşmeli; eşitlik kuralı ve tohum tekrar üretilebilir olmalı.
- Aynı dünya/ön olasılık için politikalar tam aynı başlangıç öğrenici durumundan başlamalı.
- Model durumu, sorgulanmış girdi maskesi ve rastgelelik durumu saklanıp ayrı süreçte devam edildiğinde seçilen deneyimler dahil aynı iz oluşmalı.
- Ana ve ek sonuçlar ayrı dosyada, ayrı özetle ve tam tekrar çalıştırmayla korunmalı.

Bu sınama, hedeflenen sürekli sistemin **öğrenme durumundan hareketle deneyim edinme** parçasına somut bir örnektir. Sürekli çalışan ana model, özgün görev/amaç üretimi, sınırsız gelişim, güvenilir özdenetim veya kendiliğinden yeni temsil dili henüz kurulmaz.
