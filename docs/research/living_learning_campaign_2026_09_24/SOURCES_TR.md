# Tur 23 — literatür konumu ve iddia sınırı

24 Eylül 2026 tarihinde birincil kaynak kayıtları kontrol edildi. Aşağıdaki çalışmalar yöntem ailelerini konumlandırır; buradaki değiştirilmiş uygulamaların garantilerini devraldığımız anlamına gelmez. Bu tarama kapsamlı bir özgünlük taraması değildir.

| Kol | Kaynak | Bu kampanyayla ilişki ve sınır |
|---|---|---|
| MEM | Jeffrey S. Vitter (1985), [Random Sampling with a Reservoir](https://www.cs.umd.edu/~samir/498/vitter.pdf) | Uzunluğu önceden bilinmeyen akıştan sınırlı rastgele örnek saklama bilinen bir problemdir. Buradaki basit rezervuar uygulaması makalenin hızlı Algorithm Z uygulaması değildir. |
| MEM / CON | Arslan Chaudhry ve diğerleri, [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486) | Küçük deneyim bellekleri ve tekrarın sürekli öğrenmeyle ilişkisi için yakın literatür. MEM motorumuzdaki açık anahtar sayımları bu makalenin sinir ağı deneylerini yeniden üretmez. |
| REP | Jialei Wang, Peilin Zhao, Steven C. H. Hoi, Rong Jin (2014), [Online Feature Selection and Its Applications](https://ink.library.smu.edu.sg/sis_research/2277/) | Çevrimiçi özellik seçimi önceden var olan bir araştırma alanıdır. Buradaki hazır çarpım bankasından artık hataya göre seçim, yeni bir temsil dili icadı sayılamaz; makalenin algoritmasıyla özdeşlik iddia edilmez. |
| UPD | Richard S. Sutton (1992), [Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf) | Deneyimle öğrenme hızı gibi güncelleme özelliklerini değiştirme fikri yeni değildir. Bizim sonlu kazanç karışımımız IDBD'nin yeniden uygulaması değildir. |
| UPD | Yoav Freund, Robert E. Schapire (1997), [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X) | Uzman/seçenek ağırlıklarını gözlenmiş kayıptan güncelleme için bilinen kuramsal aile. Ortak durumlu karşıolgusal öneriler, unutma ve ayrı filtre bankamız için bu makalenin hata sınırlarını kanıtlamış değiliz. |
| ENV | Peter Auer, Nicolò Cesa-Bianchi, Paul Fischer (2002), [Finite-time Analysis of the Multi-armed Bandit Problem](https://www2.compute.dtu.dk/pubdb/pubs/2088-full.html) | Eylem seçimi ve gözlem toplama arasındaki ilişki için klasik bandit literatürü. c=0.5 ve değişen ortamlardaki son64 penceremize klasik UCB1 garantisi atfedilmiyor. |
| İstatistik | Sture Holm (1979), [A Simple Sequentially Rejective Multiple Test Procedure](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) | 48 karşılaştırmanın destek/ters yönlerini 96 testlik ailede düzeltmek için kullanılır. Hipotezler arası bağımsızlık gerekmez. Geçerli tekil p değerleri ve burada tanımlı sentetik örnekleme kapsamı yine gereklidir. |

## Sonuçları nasıl okumalı?

İşaret testi pozitif/negatif fark sıklığını sınar. Pratik eşik üstündeki **ortalama** fark ayrıca betimsel bir filtredir; bu düzenek pratik marjın aşıldığına ilişkin ayrı bir güven aralığı veya marj testi üretmez. İki yönlü ailede anlamlılık, belirli bir mekanizmanın doğru olma olasılığı değildir. Rastgele tohumlar yalnız tanımlı dünya üreticilerinden gelir; başka dünya yasalarına, gerçek veriye veya biyolojiye genelleme sınanmamıştır.

Katalog 200 kayıt içerir; konular örtüşür. Örneğin XFR09/CTX05 bağlama göre sınıflandırmayı, REP19/CTX20 mevcut cevapla sonraki öğrenme kapasitesini, XFR13/RES01/CTX19 edinimin yaşam boyu maliyetini farklı müdahalelerle ele alır. Bu kayıtlar 200 benzersiz mekanizma, 200 bağımsız kuram veya 200 özgün buluş sayılmaz. İlk 48 karşılaştırma bilinen mekanizmalar için denetim ve sınır haritasıdır; yenilik arayışının tamamı değildir.
