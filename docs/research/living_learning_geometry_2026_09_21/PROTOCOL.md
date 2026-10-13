# Öğrenilebilir temsil altında koruma ve yeni edinim — deney ön kaydı

Muhammed Yasin Yılmaz · 21 Eylül 2026 · Yerel araştırma turu 10

Bu protokol ana deneyin sonuçları görülmeden yazıldı. İlk uygulama kontrolleri başarım seçimi için kullanılmayacak. Kitap çalışması durduruldu; kitap, önceki araştırma raporları ve sonuçları değiştirilmeyecek. Commit/push/yayın yapılmayacak.

## Devam noktası ve kapsam

Önceki ortak yaşam deneyi hazır beş özellik ve doğru bölgesel bellek geometrisi kullanıyordu. Durum taşıma ve öğrenilmiş güncelleme deneyleri bugünkü eşit cevapların yarın eşit edinim olmadığını gösterdi. Son iki tur gelecekteki düzeltmeler için belleğin yeterliliğine odaklandı. Bu tur o çizgiyi tek merkez yapmadan, **öğrenilebilir sinirsel temsilde eski beceri ile yeni edinimin aynı parametrelerde birlikte değişmesine** geçiyor.

Ana problem hâlâ: deneyim gelecekteki hesap ve davranış kapasitesini nasıl kalıcı, kontrollü, genellenebilir değiştirir? Bu çalışma onun koruma/edinim ve güncelleme geometrisi bileşenini sınar; genel çözüm iddia etmez.

## Yanlışlanabilir iddialar

1. Eski kayıp gradyanıyla çatışmayan adım eski beceriyi korumaya yeter. Tam öğrenilmiş eski örnekte sıfır gradyan karşı örneğiyle sınanacak.
2. Eski çıktı Jacobian'ına dik adım, sonlu parametre değişiminde eski çıktıları tam korur. Çarpımsal tam örnek ve nonlinear ağda doğrudan ölçülecek.
3. Daha fazla eski çıktıyı korumak, yeni edinimi kendiliğinden iyileştirir. Artan ankor sayısında rank, projekte gradyan, eski/yeni risk birlikte ölçülecek.
4. Adayı gerçekten deneyen sonlu-adım kabulü, iyi korunmayı iyi edinime dönüştürür. Reddetme/donma ve basit replay kontrolü bu çıkarımı sınayacak.

İlk iki iddia bilinen lineerleştirme sınırlarıdır; yeni teorem veya GEM/OGD makalelerinin yanlışlanması diye sunulmayacak. Matematikte `A=J_old`, `P=I−A⁺A`, `B=J_new` için izinli ilk mertebe yeni değişimler `range(BP)`dedir. Bu karakterizasyon mevcut fonksiyon ailesine ve yerel noktaya bağlıdır; global erişilebilirliği belirlemez.

## Ağ, dünya ve verilen önbilgi

Tek çıktı veren `2→8 tanh→1` ağ, 33 öğrenilebilir sayı; NumPy float64 ve elle türetilmiş, bağımsız doğrulanacak Jacobian. Hedef aynı sınıfta üç etkin tanh birimiyle yazılabilir; gerçek hedef parametreleri yalnız dünyada/değerlendiricide bulunur. Öğrenici bunları, girdi döndürme açısını veya görev kimliğini almaz.

Her tohumda girdi koordinatları döndürülür. İlk edinim bölgesi yerel koordinatta `u∈[−1,−.2]`, yeni bölge `u∈[.2,1]`, ikinci koordinat `v∈[−1,1]`. Hedef bütün yaşamda sabittir; dolayısıyla eski bozulma gerçek hedef değişimi diye açıklanamaz. Etiketler bu temel deneyde gürültüsüzdür. Bu kolaylaştırma raporda korunacak.

Eski 128 örnekte 600 sabit full-batch SGD adımı, öğrenme oranı `.05` ile ortak başlangıç elde edilir. Eski 128 örneğin rastgele sırasının ilk 4/16/64 kaydı iç içe ankorları oluşturur; doğru etiketleri bütün yöntemlerin erişimine açıktır. Bu arşiv kesimi araştırmacı tarafından verilir: deney görev sınırını tamamen kendi keşfeden bir sistem değildir. Her koşul aynı checkpoint'ten başlar.

Yeni akış 128 minibatch, her biri 8 yeni nokta. Her batch önce tahmin edilir, ardından etiketiyle güncellenir. Değerlendirici eski, yeni ve aradaki görülmemiş şerit için ayrı 256'şar nokta tutar; bu etiketler aday veya kabul seçimlerine girmez. Test kümesi yeniden kullanılarak hiperparametre seçilmeyecek.

## Yöntemler ve eşleştirme

- `frozen`: hiçbir ağırlığı değiştirmez.
- `head_only`: yalnız çıktı ağırlıkları ve bias güncellenir; edinilmiş özellikleri dondurur.
- `online`: bütün ağı yeni minibatch ile SGD günceller.
- `replay`: yeni ve eski ankor ortalama kayıplarını eşit ağırlıkla kullanır.
- `loss_halfspace`: yeni gradyanı eski ortalama kayıp gradyanıyla negatif iç çarpım varsa yarı uzaya yansıtır; A-GEM benzeri tanısal kontrol, birebir benchmark yeniden uygulaması değildir.
- `tangent_fixed`: eski çıktı Jacobian'ının başlangıç row-space'ine dikleştirir.
- `tangent_current`: aynı Jacobian'ı her adımda yeniden hesaplar.
- `tangent_guard`: `tangent_current` önerisine gerçek sonlu çıktı koruma denetimi ekler.
- `replay_guard`: replay önerisine aynı denetimi ekler.

Koruma, ankorlarda **sabit başlangıç çıktılarından** ortalama kare sapmanın `1e−4`ü geçmemesi şeklindedir. Önceki adıma göre tolerans biriktirilmez. Başlangıç hataları da korunabilir; bu, eski doğru hedeflere karşı loss kontrolünden daha katı ve farklı bir amaçtır. Guard en fazla 12 kez adımı yarıya indirir (ilk deneme dahil 13 aday); hiçbir aday geçmezse durum değişmez. Kabul kümesi eğitim bilgisidir, bağımsız test değildir.

12 tohum (`0..11`), öğrenme oranları `.03,.12`, üç bellek boyutu. SVD eşik oranı `1e−8`; `1e−6` için tüm teğet yöntemlerin ayrıca hassasiyet taraması yapılır. Bütün koşullar raporlanır. Tam rank nedeniyle null uzayın kapanması beklenen lineer cebirdir; başarı keşfi sayılmaz.

## Ölçüler ve kaynak muhasebesi

Önce/sonra eski–yeni–şerit MSE; yeni akışın prequential MSE'si; ankor drift'i; projeksiyon norm oranı; rank ve tekil değerler; yeni gradyan normu; kabul/ret ve aday denemeleri. Warmup eski hatası ayrıca verilir. Aynı sınıfta sıfır riskli öğretmen parametreleri kapasite tanığıdır; öğrenciye verilmeyen oracle'dır, öğrenilmiş rakip değildir.

Her yöntem aynı yeni etiketleri ve izinli eski kayıtları alır; **eşit hesap bütçesi iddiası yoktur**. Gradient örnek satırları, ek Jacobian satırları, SVD çağrıları, aday ileri değerlendirme satırları ve tutulmuş sayısal durum ayrı kaydedilir. Analitik Jacobian ortak hesap yaptığı için bu sayaçlar FLOP veya süre oranı değildir. Basit replay güçlü pratik kontrol olarak kalır.

## Doğrulama ve devam kapasitesi müdahalesi

Çarpımsal tanıklar kesirlerle tam hesaplanacak. Ağın Jacobian'ı merkezî sonlu fark ve PyTorch autograd ile bağımsız karşılaştırılacak. Projeksiyonlar idempotence/orthogonality ve affine kontrolle denetlenecek. Deney JSON'u bağımsız tekrar çalıştırmada karşılaştırılacak; devam durumu ayrı süreçte yüklenip sonraki tahmin ve güncellemeler karşılaştırılacak.

Son durumdan aynı ağırlıkları taşıyan iki kopyaya aynı 32 yeni batch verilecek: biri edinilmiş koruma durumunu tutacak, diğeri yalnız koruma politikasını kaldırıp online SGD kullanacak. İlk cevaplar eşittir. Eski/yeni risk ve edinim eğrisi birlikte ölçülecek; kısıtı kaldırmanın daha fazla eski bozulmaya izin verdiği saklanmayacak. Bu müdahale yeni bir güncelleme kuralının deneyimden öğrenildiğini iddia etmez.

Önceki bütün izlenen dosyalar, uncommitted kitap dosyaları ve önceki araştırma dosyaları başlangıç SHA-256 manifestiyle korunacak. Sonradan eklenen kontroller veya protokol değişimleri ayrı notla belirtilecek.
