# REP / XFR / LAN: temsil, aktarım ve yordam dili kampanyası

Muhammed Yasin Yılmaz · 24 Eylül 2026 · Yerel araştırma kaydı

Bu dosya ve 60 maddelik kayıt, ana performans koşuları yapılmadan yazılmıştır.
REP ailesindeki 12 koşullu öngörü bu dalgada yürütülür. Kalan 8 REP, 20 XFR ve
20 LAN önerisi kuyruktadır: çalıştırılmış test, doğrulanmış sonuç veya 48 bağımsız
kuram gibi sunulamaz. REP02 ile REP11 aynı koşunun hata/maliyet yönleridir.

## Problem ve kapsam

Deneyim yalnız ağırlıkları değil, sonraki tahminde kullanılacak özellik alt
kümesini değiştirirse bunun ne zaman işe yaradığını sınarız. İşlem dili hazırdır:
bir sabit, dört koordinat, altı ikili çarpım. Burada yeni ilkel işlem keşfi,
sınırsız temsil edinimi, beyin açıklaması veya özgün öğrenme kuramı yoktur.
Önceki turlardaki kapasite/güncelleme ayrımlarını tek başına çözmez.

## Örnekleme ve sıralama

Her tohum için yedi akış, her akışta 192 örnek vardır. Girdiler dört adet ±1
koordinat; çıkış etiketleri [-6,6] aralığına kırpılmış gerçek sayılardır.
Öğrenici önce tahmin eder, sonra yalnız o örneğin etiketini alır. Gizli katsayı,
etkileşim kimliği, kayma zamanı veya gelecek etiketler öğreniciye verilmez.
Tahminler [-6,6] aralığına kırpılır; bütün yaşam ortalama karesel hata [0,144]
aralığındadır. İlk örnekler ve özellik edinimi öncesi hatalar ölçüme dahildir.

Akışlar: sürekli ikili etkileşim; salt doğrusal; yüksek gürültülü doğrusal;
ilk iki koordinatın +1 olasılığı 0.08 olan dengesiz etkileşim; ilk yarı güçlü
girdi ilişkisi ardından bağımsız girdiler; ikinci yarıda başlayan etkileşim;
ortada kimliği ve işareti değişen etkileşim. Son ikisinde değişim noktası
değerlendiriciye aittir; öğrenici için ayrı bir bildirim yoktur.

Nadirlik koşulu yalnız iki koordinatı dengesizleştirir. Gizli etkileşim bu ikisini
içermek zorunda değildir; böylece sonuçlar yalnız nadir aktif çifte koşullandırılıp
kolaylaştırılmaz. Kullanılan etiket yasası doğrusal ve ikili bankanın içindedir;
özellik dili dışı hedef önerisi REP16 kuyruktadır.

## Öğreniciler ve önceden sabit seçimler

Başlangıç ridge katsayısı 1. Temel çevrimiçi güncelleme recursive least squares:

    v = seçili özellikler; u = P v; d = λ + vᵀu
    w ← w + u (y − vᵀw) / d
    P ← (P − uuᵀ/d) / λ

λ=1 veya 0.97; P başlangıçta birim matris, w sıfırdır. Tam kontrol baştan
11 özelliğin hepsini kullanır. Doğrusal kontrol sabit + dört koordinat kullanır.
Seçici öğrenici 5 özellikle başlar, en çok üç çarpım özelliği ekler.

Seçici öğrenici gözlenmiş tahmin artığı ile altı aday çarpımın ilişkisini 0.98
unutmalı toplamla izler. Her 24 etikette, etkin olmayan adaylardan en büyük
mutlak ortalama ilişki 0.08'i aşarsa bir aday eklenir. Bu eşikler performans
koşularına göre ayarlanmaz. Rastgele kontrol aynı dedektör kuralını kullanır,
fakat eklenecek adayı tohum/zaman tabanlı bir indeksle seçer. İki öğrenicinin
hataları farklılaştığında ekleme zamanları da farklılaşabilir; birebir eşit
ekleme takvimi iddiası yoktur. Bu kontrol öğrenicinin kendi hatalarından gelişen
karar zincirini bütünüyle karşılaştırır.

Yeni özellik eklenince seçici model gözlenmiş tüm geçmişi yeni alt kümede
yeniden işler. Böylece yeni özelliğin eski özelliklerle çapraz ilişkisi de
hesaplanır. REP12'deki zero_history ablasyonu eski P ve w bloğunu koruyup
yeni koordinatı bağımsız birim öncülle ekler; geçmiş çapraz terimleri kurmaz.
Bu ablasyon aynı arşivi tutar ama kullanmaz. Daha az arşiv tutan ayrı yöntem
veya optimal sıfır-geçmiş güncelleyicisi olduğu ileri sürülmez.

## Karşılaştırmalar ve kaynak hesabı

12 eşleştirilmiş karşılaştırma catalog_representation.json içinde kayıtlıdır.
MSE öngörüleri için pratik fark 0.01; REP11 maliyet öngörüsü için 1000 sanal
işlem seçilmiştir. Keşif ve doğrulama tohumları ana CONTRACT.md'dedir.

Sanal işlem hesabı duvar saati, enerji veya işlemci komutu değildir. Bütün
öğrenicilerde aynı kurallar kullanılır. Tahmin: altı çarpım + 11 özellik yazımı
+ 2k iç çarpım + iki kırpma; sıralı güncelleme: 6k²+6k+3. Dedektör altı
artık ilişkisinin güncellemesi, aday üyelik taramaları ve seçimi için ücret
öder. Arşive örnek başına beş sayı yazılır. Özellik kurulunca k²+k başlangıç,
arşivdeki her örnek için özellik oluşturma ve aynı sıralı güncelleme ücretlidir.
Model kurulumu ve çalışma anındaki feature evaluation gerçek kodda yapılır;
Başlangıç indeks, P, w ve dedektör durumu yazımları k²+2k+7 ile ayrıca sayılır.
seçilmiş boyutlar için ücretsiz orakl kurulumu yoktur. Bu analitik sayımın
Python nesneleri, liste yönetimi ve tüm çalışma zamanı ayrıntılarını tam olarak
temsil ettiği iddia edilmez.

Bellek kaydı P, w, aday skorları ve örnek arşivindeki sayısal hücreleri sayar;
Python/JSON baytları değildir. Karşılaştırmaların tümü aynı gözlemleri alır,
fakat seçici yöntemin tüm geçmişi saklaması daha fazla belleğe mal olur.
Düşük MSE tek başına hesap veya bellek üstünlüğü sayılmaz. REP11 aynı akışta
ayrı kaynak iddiasını sınar; REP02'den bağımsız bir mekanizma değildir.

## Doğrulama ve önceden kabul edilen sınırlar

selfcheck sıralı güncellemeyi bağımsız normal denklemler çözümüyle karşılaştırır;
dinamik kurulumun aynı özelliklerde ridge çözümüne eşitliğini sınar; JSON
kaydı sonrası tahmin, maliyet ve edinim kararlarının birebir devamını denetler;
özellik eklemesi öncesi kapalı mekanizmanın doğrusal kontrolle eşitliğini sınar;
16 ikili girdide özellik hesabını ayrı skaler ifadelerle doğrular.

Tam özellikli ridge güçlü aynı-bilgi kontrolüdür; tüm olası ridge katsayıları,
sparse regression, sinir ağı veya değişim algılayıcılarının en iyisi değildir.
Başarısız öngörüler ve eşitlikler saklanır. Bir tohumda aynı akışın birden çok
kontrasta girmesi bağımsız örnek sayısını artırmaz. Ana orkestra düzeltmeli
istatistikleri uygular. Bu modülde performans görülerek eşik değiştirilmeyecek.

Literatür ailesi: sıralı özellik seçimi, recursive least squares, öğrenmede
üstel unutma. Bunlar yeni isim altında özgün buluş olarak sunulmaz.
