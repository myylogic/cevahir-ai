# ENV — eylemin yeni deneyim üretmesi

24 Eylül 2026. Katalog başarı çalıştırmasından önce yazılır. Yerel, sentetik,
bilinen bandit mekanizmaları; yeni keşif veya biyolojik iç durum açıklaması değil.

512 adımlık iki-eylemli Bernoulli dünyalar. Dört rejim: sabit büyük fark,
sabit küçük fark, 256'da bir terslenme, 128 adımda bir tekrarlı terslenme.
Tohum başına başlangıç iyi eylemi rastgele; ortalamalara küçük ortak jitter.
Üreteç bütün eylem/zaman için eşlenmiş potansiyel ödül tablosu kurar.
Öğrenici yalnız seçtiği eylemin ödülünü görür. Bütün yöntemler aynı potansiyel
dünyada, farklı seçilmiş gözlemler alır; bu bir aynı-gerçekleşmiş-veri karşılaştırması
değildir. Eylemin veri üzerindeki etkisi bu motorun konusudur.

Yöntemler: greedy örnek ortalaması (önce her eylem bir kez), sabit epsilon=0.10,
UCB c=0.5, son64 zaman adımını tutan aynı UCB. Klasik UCB1 garantisinin
uygulandığı iddia edilmez; katsayı ve değişen dünya farklıdır. Yanlış kesinlik
olabilecek ilk bağlılık, süren rastgele sapma, unutmalı yeniden örnekleme ayrı
karşılaştırmalar. Dünya değişim zamanı/gerçek ortalama öğreniciye verilmez.

12 çalıştırılabilir koşullu iddia = dört rejimde üç mekanizma karşılaştırması.
Bu 12 bağımsız ilke değildir. Birincil ölçüt 512 adım boyunca ortalama pseudo-regret:
(en iyi gerçek ortalama − seçilen eylemin gerçek ortalaması); 0–1 arası, az iyi.
Gizli ortalama yalnız değerlendiricide. Pratik eşik0.01. Maliyet her seçim/ödül
güncellemesi/tam pencere taramasında sayılan sanal skaler işlemler; bellek azami
tutulan gözlem sayısı ve son JSON boyutu ayrı. Epsilon ve UCB adımlarının gerçek
CPU maliyeti eşit sayılmaz; tanımlı sayaç kullanılır. Başlangıç iki gözlem metrikte.
Üç yöntemin bütün yaşam hatası/maliyeti birlikte raporlanır.

Selfcheck: pencere özetini bağımsız son64 listesiyle karşılaştırma, sabit ödülde
analitik pseudo-regret kontrolü, ayrı nesne JSON durumundan devam, greedy ile
epsilon=0 mekanik eşliği. Bunlar bilimsel yeniden üretim değil uygulama denetimi.
Sınır: iki hazır eylem, ödül anlamı, istatistik, seçici ve unutma aralığı verildi.
Bir iç durumun keşfi, yeni amaç veya yeni eylem dili sınanmıyor.
