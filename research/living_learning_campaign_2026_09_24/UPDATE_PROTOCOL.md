# UPD / CRD / CTX — önceden yazılmış deney protokolü

Tarih: 2026-09-24. Dalga 1 sonuçları görülmeden yazıldı. Yazar araştırma programı: Muhammed Yasin Yılmaz. Bu belge ve katalog, ana yürütücü tarafından sonuçlardan önce hash ile dondurulacaktır.

## Araştırma kapsamı

Deneyim yalnız bir durum tahminini değil, sonraki deneyimin bu tahmini ne kadar değiştireceğini belirleyen yordamı etkileyebilir mi? UPD ailesindeki bu deney **verilmiş altı kazançtan ağırlıklı bir güncelleme katsayısı öğrenir**; yeni bir işleç dili, yeni optimizer veya genel yaşarken öğrenme kuramı icat ettiği iddiası yoktur. CRD ve CTX aileleri bu dosyada yalnız kayıtlı hipotezdir; çalıştırılmış sayılmaz.

12 yürütülebilir iddia, üç karşılaştırma grubuna bağlı koşullu iddialardır: altı ortamda sabit kontrol, dört ortamda bağımsız durumlu güçlü uyarlamalı kontrol, iki ortamda öğrenilmiş kazancın yeni duruma aktarılması. Bunlar 12 bağımsız temel fikir değildir. Tohumlar da hipotez değildir.

## Veri ve zaman sırası

Her yaşamda saklı ortalama mu [-0.9, 0.9] içindedir; gözlem y = mu + sınırlı tekdüze gürültü. Öğrenici yalnız y değerini, sırayı ve aktarım deneyindeki açık yeniden başlama sınırını alır; mu ve rejim kimliği verilmez. Önce tahmin, sonra y açıklama ve güncelleme yapılır. Saklı mu yalnız ayrı puanlayıcı tarafından okunur. Her yöntem aynı sırayı görür. Ana akışlar 384 adım; aktarım akışları 256 edinim + 256 hedef adımıdır. Sıfır başlangıç tahmini ve yeniden başlama tüm yöntemlerde aynıdır.

Altı ortam: sabit/düşük gürültü; sabit/yüksek gürültü; yavaş sürüklenme; ani sıçramalar; yön değiştiren sürüklenme; 48 adımlık bloklarda gürültü ve dinamiği değişen karışık ortam. Bu rejimler gerçek dünyanın veya biyolojinin modeli olarak sunulmaz.

## Öğreniciler

Kazanç kümesi A={0.01, 0.03, 0.1, 0.3, 0.7, 1}. Sabit kontroller durağan ortamlarda a=0.05, diğerlerinde a=0.2 kullanır. Bu değerler sonuçlara göre seçilmez. En güçlü kontrol ayrıca ayrı verilir: her kazanç için bağımsız durum saklayan çevrimiçi üstel ağırlıklı filtre bankası.

Ortak durumlu aday önceki adımda altı karşıolgusal bir-adımlık durum üretir: z_(t+1,a)=m_t+a(y_t-m_t). Yeni y_(t+1) geldiğinde log ağırlıklar l_a <- 0.97 l_a - 4(z_a-y)^2/16 ile güncellenir. Normalize exp ağırlıklarından a_t=sum(w_a*a) hesaplanır. Canlı durum m <- m+a_t(y-m) olur. Öneriler güncellemeden önceki ortak durumdan üretilir. İlk karşılaştırmada tüm öneriler sıfırdır. Bu, iki ardışık gözlem arasındaki tahmin hatasına göre kazanç ayarlamadır; saklı doğruluk veya en iyi kazanç etiketi verilmez.

Filtre bankası aynı ağırlık yasasını, altı bağımsız filtrenin o andaki tahmin hatalarına uygular; her filtre kendi durumunu kendi sabit kazancıyla günceller. Bankanın tahmini güncel ağırlıklarla filtre durumlarının ortalamasıdır. Adayın avantajı bu kontrol karşısında kaybolabilir; bu sonuç kaydedilir.

Aktarımda aday önce 256 sürüklenme gözlemiyle aynı yordamı çalıştırır; son kazancı kaydeder. Sonra canlı tahmini sıfırlayıp, öğrenilmiş kazancı sabitleyerek 256 yeni gözleme geçer. Hedef ya yeni sürüklenme ya da yüksek gürültülü sabit ortalamadır. Kontrol iki aşamada a=0.2 kullanır ve aynı sınırda sıfırlanır. Ana puan iki aşamayı da içerir; edinim ve hedef puanları ayrıca raporlanır. Saklı görev ailesi öğreniciye verilmez. Aktarımın işe yaraması, sırf yeni tahmin durumunu taşımakla açıklanamaz; yalnız kazanç taşınır.

## Puan, maliyet, yanlışlama

Ana puan ortalama ((tahmin-mu)/3)^2; [0,1] içinde, küçük daha iyidir. Pratik fark eşiği 0.0005. Bu puan saklı sinyal tahminini ölçer, gözlem gürültüsünü ezberlemeyi değil. Her iki yöntem için edinim dahil bütün yaşam hesap maliyeti ayrıca verilir. Düşük hata tek başına kaynak üstünlüğü sayılmaz.

Maliyet, koddaki toplama/çıkarma/çarpma/bölme ve karşılaştırma/exp çağrılarının belirtilen sanal sayımıdır; exp bir birim sayılır. Tahmin ve kalıcı durum yazma maliyetleri de birer sanal erişim/yazma sayılır. Veri üretimi ve ortak dış puanlayıcı hariçtir. Bu donanım zamanı, enerji veya Python nesne belleği ölçümü değildir. Bellek kayıtlı sayısal hücre sayısı ve JSON baytı olarak raporlanır. Kontrolün en iyi sabit kazancı saklı mu üzerinden **sonradan yalnız tanısal** hesaplanır; bu bilgi öğreniciye dönmez veya ana karşılaştırma olmaz.

Bir iddia ancak ana kampanyanın eşleştirilmiş keşif + dokunulmamış doğrulama tohumları, pratik marj ve çoklu karşılaştırma kuralları çerçevesinde değerlendirilebilir. Tek başarılı tohum kanıt sayılmaz. Katalogdaki CRD, CTX ve sekiz diğer UPD iddiası henüz sınanmamıştır.

## Bağımsız denetim

Sabit kazanç özyinelemesi kapalı geometrik toplamla karşılaştırılır. Adayın bir-adım işlemi aynı algoritma kodunu çağırmayan bağımsız matematiksel hesapla kontrol edilir. Durum JSON ile kaydedilip yeniden açıldığında tahmin, ağırlık, maliyet ve devam durumu birebir eşleşmelidir. Meta-öğrenme hızı sıfıra çekildiğinde kazanç aritmetik ortalamada sabit kalmalıdır; bu mekanizma kaldırma denetimidir. Saklı mu değiştirildiğinde aynı y dizisi aynı öğrenici izini vermelidir. Ayrı görev başlangıcında yalnız kazanç aktarıldığı kontrol edilir.

## Literatür konumu

Üstel ağırlıklı uzman birleştirme, uyarlamalı filtreleme ve öğrenilmiş güncelleme kuralları bilinen ailelerdir. Bu modül bunların sınırlı bir ortak-durum varyantını ve güçlü kontrolünü karşılaştırır. Özgünlük savunusu yapmaz. Kampanya raporundaki doğrulanmış birincil kaynaklar ayrıca esas alınmalıdır.
