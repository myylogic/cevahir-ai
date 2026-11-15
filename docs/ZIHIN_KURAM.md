# Cevahir Zihin Kuramı: Yaşayan Kanıt Zihni

*Muhammed Yasin Yılmaz — Kasım 2026. Bu metin, depodaki `zihin/` çekirdeğinin kuramını ve mimarisini anlatır. Her
iddianın ölçümü ve olumsuz sonuçları `research/living_mind_2026_11_14/` ve `research/living_mind_2026_11_15/`
klasörlerindeki notlarda ve sonuç dosyalarındadır.*

## Kısaca

Cevahir Zihin bir Transformer değildir: içinde sinir ağı, ağırlık matrisi, gömme vektörü, geri yayılım ve softmax
dikkati yoktur. Bir **Yaşayan Kanıt Zihni**dir. Her algıladığı şeyi bir birim, birimler arasındaki her ilişkiyi bir
bağ olarak tutar. Her bağ, o ilişkinin lehine ve aleyhine kademeli kanıtı **kaynağına göre ayrı ayrı** tutar
(kendi gözüyle mi gördü, biri mi söyledi, başkasından mı taklit etti, kendisi mi yaptı, akıl yürüterek mi çıkardı,
hayal mi etti). Ne beklediğini tek bir kuraldan okur ve her an öğrenir. Defteri tutulduğu sürece tek bir deneyimin
kanıtını tam olarak geri alabilir.

Eğitim ve kullanım ayrı aşamalar değildir: **öğrenmek yaşamaktır**. Model her konuşmada öğrenir.

## 1. Temel varlıklar

**Kanal.** Zihnin dünyayla ve kendisiyle temas yolları. Sekiz kanal türü vardır:
- `sense`: sıra sonu, bir resmin ya da sesin özü, içteki bilmeme hali;
- `act`: kim söylerse söylesin kelimeler ve zihnin kendi eylemleri;
- `feel`: şaşkınlık, rahatlama, çatışma, yük;
- `knob`: kendi ayarları;
- `source`: kaynaklar;
- `facet`: bir şeyin görünüşü, sesi, kelime parçaları;
- `value`: iyi ya da kötü sonuç;
- `context`: konuşan kişi, ton, benlik.

Bir zihnin her biri bir türden adlı kanalları vardır. Kelimeler, kim söylerse söylesin, act türündeki `text`
kanalına gelir. Bir kelimeyi birinin mi söylediği, zihnin mi taklit ettiği, yoksa kendisinin mi söylediği kanal
olarak değil, kökeni olarak (aşağıda) tutulur.

**Birim.** Tür işaretleri taşır:
- **Atom:** bir kanaldaki bir simge (bir kelime, bir resmin özeti, bir sıra sonu).
- **Birleşik birim:** yaşanmış bir dizi (yedi öğeye kadar bir an penceresi).
- **Çerçeve:** yerlerinden biri bir kategori olan birleşik birim, örneğin "_ balık yer". 5. bölümdeki
  genellemeler çerçevelere dayanır.
- **Kategori:** aynı yerlerde yaşanmış birimlerin sınıfı (bir çerçeveden türetilen kategori "türetilmiş" diye
  işaretlenir).

**Bağ.** İki birim arasındaki ilişki:
- `EXPECT` (gecikme 0: birlikte, 1: hemen sonra, 2: aynı bölümün sonunda gelen bir sonuç);
- `PART` (bir dizinin parçası);
- `ABSTRACT` (bir kategorinin üyesi);
- `RELY` (isteğe bağlı: bir genellemenin güvenilirliği; yalnız güvenilirlik ayarı (F9) açıkken kurulur, sohbet
  programı bu ayarı açmaz).

**Köken Kanıtı (Provenance Evidence).** Her bağ, altı köken için ayrı ayrı lehte (W+) ve aleyhte (W−) kanıt
tutar: *algılanan, söylenen, taklit edilen, yapılan, çıkarılan, hayal edilen*. Miktarlar sayı değil, kademelidir:
bir doğrulama, bekleyen birimin ne kadar etkin olduğu ve olanın ne kadar güçlü olduğuyla ölçeklenen bir miktar
ekler; bir başarısızlık, bağın kendi beklentisiyle tartılmış bir miktar ekler (4. bölüm). Bilginin nereden geldiği
hiçbir zaman kaybolmaz.

## 2. Tek kural: Kanaat Beklentisi

Bir bağın ne kadar beklendiği, kökenlere verilen güvenle tartılmış kanıttan okunur:

- sıklık `f = W+ / W`;
- kanaat `c = W / (W + k)`, burada `k = 1`;
- beklenti `e = c · (f − ½) + ½`.

Bu, NARS'ın (Pei Wang'ın Aksiyomsuz Akıl Yürütme Sistemi) doğruluk değeri beklentisidir, `k = 1` ile. Bu çekirdek
onu kökene göre ayrı tutulan ve öğrenilmiş güvenle tartılan kanıttan okur. Hiç kanıt yoksa beklenti ½'dir:
"bilmiyorum". Kanıt arttıkça kanaat büyür. Bütün okumalar bu kurala dayanır: ne söyleyeceği, neyi bekleyeceği, neye
şaşıracağı.

**Güven Yargısı (Trust Judgment).** Algılanan kanıt ve zihnin kendi eylemleri doğrudan deneyimdir; hakem onlar
olduğu için güvenleri tanım gereği 1'dir. Dört dolaylı kökene (söylenen, taklit edilen, çıkarılan, hayal edilen)
güven ½'den, yani kuralın kanıtsız değerinden başlar ve öğrenilir: dolaylı kanıt, aynı bağda sonradan gelen doğrudan
deneyimle karşılaştırılır; tutanlar o kökene güveni artırır, tutmayanlar azaltır. Böylece zihin kime ne kadar
inanacağını kendisi öğrenir.

## 3. Dikkat: Yedili Odak

Transformer'da dikkat, her kelimenin bağlam penceresindeki önceki bütün kelimelere öğrenilmiş vektörlerle softmax
üzerinden bakmasıdır. Bu yoğundur ve maliyeti bağlamın uzunluğuyla büyür. Cevahir Zihin'in dikkati başka bir
mekanizmadır.

**Yedili Odak / Odak Yerleşmesi (Seven-Place Focus).** Her an yalnız yedi birim odağa girer (k-kazanan, F = 7).
Adaylar:
- şu an algılananlar;
- geçen anlardan taşınanlar (taşınan etkinlik her an `1 − 1/F = 6/7` ile çarpılır);
- tanınan birleşik bağlam (bölümün yaşanmış dizisi);
- bunların kategorileri ve çerçeveleri (genellemeler);
- birlikte yaşanmışlık bağlarıyla çağrışanlar.

Yerleşme en çok altı tur sürer: çağrışım yayılır, kazananlar yeniden seçilir, değişmeyince durulur. Önce şu an
olan gelir, sonra her yerin en özgül genellemesi.

Sonuçlar:
- Bağlam penceresi yoktur: bir an, odaktaki birimlerin bağlarını okur, geçmişe dönüp bakmaz.
- Bu yüzden bir anın maliyeti herhangi bir bağlam uzunluğuna değil, o bağlara bağlıdır. Ama sık kelimeler zihin
  büyüdükçe bağ kazanır; bu yüzden an başına süre bellekle yavaşça artar (hız çalışmasından sonra küçük bir zihinde
  yaklaşık 1.5 ms, 1.33 milyon bağda 2.6 ms, 2.54 milyon bağda 3.3 ms).
- Dikkat okunabilir: hangi yedi birimin neden odakta olduğu görülür.
- Öğrenilecek dikkat ağırlığı yoktur; dikkat, kanıttan doğar.

## 4. Bir anın akışı

Zihin her an (yaklaşık her kelime) şu adımları yaşar:

1. **Algı:** dışarıdan gelenler ile içeriden gelenler (kendi sözünün yankısı, hisleri, bilmeme hissi) birlikte
   algılanır. Kendisi konuşuyor ya da düşünüyorsa **Benlik Birimi** de o ana eşlik eder.
2. **Sonrayı öğrenmek (gecikme 1):** bir an önce odakta olanlar, şimdi olanı beklemeyi öğrenir.
   - **Şimdiden Öğrenme:** yalnız bir an önce gerçekten şimdi olan öğretir; odakta sönmekte olanlar öğretmez.
   - **Beklenen Başarısızlık:** bir bağ yalnız hedefinin kanalı o anda başka bir şey bildirdiyse aleyhte kanıt
     alır; sohbet programında ayrıca bağın beklentisi en az `1/(2F) = 1/14` olmalıdır. Miktar bağın beklentisiyle
     tartılır; böylece seyrek bir devam tam ağırlıkla çürütülmez, tabana yakın yerleşir.
3. **Sonuçları öğrenmek:** iyi ya da kötü bir sonuç sinyali, bölümün **Sonuç İzi**ndeki her bağlama işlenir. Hangi
   yolun neye götürdüğü böyle öğrenilir.
4. **Odak:** Yedili Odak yerleşir (yukarıda).
5. **Birlikteliği öğrenmek (gecikme 0):** odaktaki atomlar birbirini beklemeyi öğrenir. Görünüş, ses ve parça
   gibi yönler, eşlik ettikleri öğeye bağlanır.
6. **Beklenti ve karar:** bir sonraki an için ne beklendiği okunur. Sıra zihindeyse söyleyecekleri yarışır.
7. **Bilmeme Hissi:** sıra zihindeyse ve söyleyecek bir şey yoksa, bilmediğini bir iç olay olarak algılar.
8. **Hisler:** şaşkınlık, rahatlama, çatışma ve yük bir sonraki anın iç algısı olur.
9. **Defter:** anın her artışı deftere yazılır. **Kesin Unutma** bununla mümkündür: tek bir deneyimin eklediği kanıt
   bit bit çıkarılır, sonraki anların artışları dokunulmadan kalır. Bu kanıt için kesindir; o deneyim hiç
   yaşanmamış gibi bir yeniden yaşatma değildir: onun kurduğu birim ve bağlar sonraki anlar kullandıysa kalır, onun
   varlığında sonradan öğrenilenler de kalır. Sohbet programında defter o oturumu kapsar, çünkü zihin defteri olmadan
   kaydedilir.

## 5. Genelleme: kategoriler kendi üyeleriyle konuşur

- **Üye Okuması (Members' Reading):** bir kategori, bir örnek için ne bekleneceğini kendi kurallarından değil,
  üyelerinin kendi yaşadıklarından okur. Okunan örnek oylamanın dışında tutulur. Sorulan boyutu hiç söylenmemiş
  örneklerde yaşadıkça iyileşti: 600 deneyimde %93 / %81 (iki alan, 5 tohum), 1500 deneyimde %100 / %90 (3 tohum).
- **Kendi/Sonraki Yuvaları (Self/Next Slots):** "cevap sorulan kelimeyi tekrarlar" ya da "sonra o kelimenin devamı
  gelir" gibi ilişkiler, kelimeler farklı olsa da aynı ilişki olarak sayılır. Değişkenler kelimelerden doğar.
- **Benzerlik Oyu (Resemblance Vote):** iki şey ortak yönleriyle benzer. Bir yön, ne kadar az şeyde bulunuyorsa o
  kadar ayırt edicidir. Yeni bir şeye en çok benzeyen üyeler en yüksek sesle konuşur; ağırlık, sınıflandırmanın
  örnek tabanlı modellerindeki gibidir (genelleştirilmiş bağlam modeli). Görme ve işitme böyle tanır: yeni
  resimlerde %73 / %83 / %67, bilinen üç yapay sesin yeni seslerinde %93 / %97 / %93 (şans %33).
- **Durum Benzeşimi (Situation Resemblance):** hiç bütün olarak yaşanmamış bir söze, en çok benzeyen yaşanmış
  durumlarda ne geldiyse onunla karşılık verilir. Benzerlik zayıfsa, örneğin sözde hiç bilinmeyen bir kelime
  varsa, uydurulmaz.

## 6. Konuşmak, düşünmek, bilmemek

- **Sürdürme Kuralı (Continuation Rule):** şu anki durumda hiçbir şey ½'nin üstünde bir şey beklemiyorsa, zihin son
  algılanan olaydan sonraki öğeleri okur; bunların yaşanmış devamı olan en uzun kuyruğu, düşürülmüş bir kanaatle
  konuşur. Bu kural bir söz başlatmaz ve şu anki bağlamın içinde kalan karşı taraf sözlerini dışarıda bırakır;
  böylece karşısındakinin az önce söylediğini tekrarlamaz.
- **Avantaj Yarışı (Advantage Race):** aynı kanaldaki seçenekler birbirini dışlar. Önde olan, en güçlü rakibine
  karşı avantajı eşiği geçerse söylenir; bütün seçenekler eşitse susulur.
- **Kararsızlıkta Deneme (Trying in Hesitation):** iyi desteklenen seçenekler arasında kalındığında susmak yerine,
  sonucunu en az bildiği denenir. Bu yalnız sıra yeni geldiğinde ya da o durumda daha önce iyi veya kötü bir sonuç
  yaşanmışsa olur. "Yanlış" denen bir cevap bir dahaki sefer denenmez, başka bir cevap denenir.
- **Önce Düşün (Think, then Speak):** duyulan sözün en az emin olunan kelimeleri, zihne bildiklerini çağrıştırır.
  Bunlar iç ses olarak, hayal kökeniyle yaşanır. Hemen bir cevap yoksa, yaşanmış kanıta göre yeterince gerçek olan
  ilk düşünce söylenir.
- **Bilmeme Hissi (Felt Not-Knowing):** bilmemek bir çıktı değil, ölçülen bir iç haldir:
  `u = 1 − 2·max(0, en iyi beklenti − ½)`. Bu hal bir sessiz anın ardından tek başına algılanır. O anda ne
  denileceği, zihnin birlikte yaşadığı insanlardan ve verilerinden öğrenilir; çekirdekte bunun için hazır bir cümle
  yoktur. Sohbet programındaki doğuştan bir refleks olan **merak**, bilinmeyen kelimeyi soru tonuyla geri sorar. Bunu
  nasıl söyleyeceğini yaşamamış bir zihin kelimeyi geri sorar ya da susar. İki güçlü cevap
  arasında kalmak bilmemek değildir; o durumda bilmeme hissi doğmaz.

## 7. Hayal, rüya ve akıl

- **Yapısal Hayal (Structural Imagination):** bir kavram, yaşanmış dizilerde aynı yerlerde durduğu bir başka
  kavrama benzetilir ("ateş evi ısıtır" / "güneş dünyayı ısıtır": ateş ↔ güneş). Birinin düşüncesi ötekine
  taşınır. Her adımın **gerçeklik puanı**, hayal edilen kanıt ve zihnin kendi eylemleri hariç bütün kanıttan okunur.
- **Ortak Kök Çıkarımı (Shared-Stem Inference):** "kedi bir hayvandır" ve "hayvanlar nefes alır" bilgileri,
  ortak gövde (hayvan) üzerinden "kedi nefes alır" diye birleştirilir. Bu bir hipotezdir ve hayal olarak yaşanır.
  Gerçek veride ortak gövde bazen tesadüfi çıkar; bu yüzden çıkarımın kendisi yalnız rüyada ve iç seste (`!dusun`)
  çalışır, cevap verirken çalışmaz. Kurduğu şey hayal kanıtı olarak yaşanır ve hayale öğrenilmiş güvenle tartılarak,
  hayat onu yargılayana kadar cevapların okunduğu kanıtın içinde kalır.
- **Rüya Ayıklama (Dream Sorting):** uykuda zihin günün konuları ve emin olmadığı kelimeler üzerine rüya görür.
  - Rüya, çağrışımla bağlanan çıkarım ve benzetmelerden oluşur.
  - Bir kelimeyle birlikte gördüğü resmi rüyasında yeniden "görür".
  - Her rüya yeni iddialarıyla saklanır. Hayat sonradan bunlara dokunduğunda rüya **doğru**, **yanlış** ya da
    **açık** diye ayıklanır.
  - Hayal edilen her kanıt, gerçek kanıtla karşılaştırılır ve **kendi hayallerine güveni** buna göre değişir.
  - Hiçbir hayal kanıtı silinmez (yalnız son 500'den eski rüya kayıtları söner); gerçek kanıt varsa, güven
    üzerinden ona ağır basar.
- **Benlik Birimi (Self Unit):** zihin konuştuğu ve düşündüğü her anda doğuştan bir "ben" birimiyle birlikte
  algılanır. İç gözlem bu birimden okunur: ne kadar konuştu, ne kadar düşündü, kendini en çok neyle bağladı.

## 8. Öğrenmeyi öğrenmek ve sonuçtan öğrenmek

- **Ses Güvenilirliği (Voice Credibility):** bir örnek için konuşabilecek her genelleme bir "ses"tir. Sesin türü,
  kaç üyenin konuştuğu ve ne kadar hemfikir olduklarıdır. Zihin hangi tür sese ne kadar güvenebileceğini kendi
  sonuçlarından öğrenir. Bu bilgi kelime içermez, bu yüzden yeni bir alana taşınır. Sonuç karışık: bir yönde yeni
  alanı erken öğrenme hızlandı (yeni bir zihne göre +0.058 ± 0.025; 7 hayatta daha iyi, 2'sinde eşit, 1'inde daha
  kötü), ters yönde hızlanmadı. Varsayılan kapalıdır.
- **Kodu sonucundan öğrenmek** (altı küçük fonksiyon, her biri üç aday tamamlamayla bir araştırma testi,
  `research/living_mind_2026_11_15/t_code.py`):
  - Zihin önce çoğunlukla yanlış kod tamamlamaları izler.
  - Sonra kendi yazdığı kod gerçekten çalıştırılır; test sonucu iyi ya da kötü sinyali olarak döner.
  - Kararsızlıkta denemeyle, 4. turda altı görevin hepsinde testten geçen kod yazar.
  - Sonuç sinyali olmadan oran %0'da, deneme olmadan %11'de kalır.
  - İzlediği tamamlamalar arasından seçmenin ötesinde yeni kod yazmak hâlâ açıktır.

## 9. Bellek ve hız

- **Tek Köken Satırı (One-Origin Row):** bağların %99.6'sı tek bir kökenden kanıt taşır. Böyle bir bağ yalnız
  kökenini ve iki sayısını (W+, W−) saklar. Tam satır yalnız birden çok kökeni ya da yargılanmış miktarları
  (öğrenilmiş güvenin hesabı) olan bağlara verilir. Bağ başına bellek 239 bayttan 69 bayta indi: 3.25 milyon bağlı
  bir zihnin çekirdeği 777 MB'tan 223 MB'a düştü. Sonuçlar bit bit aynı kaldı.
- **Hız:** hız çalışması bir anı, sonuçlar bit bit aynı kalarak 2.0–3.8 kat hızlandırdı: 2.54 milyon bağlı bir
  zihinde 3.2–3.5 ms. Tek Köken Satırı bunun ardından %13–32 maliyet getirdi: 3.25 milyon bağlı zihinde an başına
  yaklaşık 3.9–4.4 ms (süreler yol göstericidir; makine paylaşımlıydı). İki değişiklik de küçük açık koşuda, 17
  ayarda ve büyük zihinlerde öz özetleri karşılaştırılarak denetlendi.
- **Açık sorun:** bellek yaşanmış veriyle doğrusal büyür; an başına yaklaşık 11 bağ oluşur. 3.25 milyon bağlı
  zihnin süreci yaklaşık 4 GB tutar ve bunun çoğu bağ kanıtının dışındadır. Milyonlarca örneğe giden yol (diske
  dayalı depo, uykuda pekiştirme) notlarda planlıdır.

## 10. Transformer ile karşılaştırma

| | Transformer | Cevahir Zihin |
|---|---|---|
| Öğrenme | eğit, dondur, kullan | her an öğrenir; ayrı eğitim yok |
| Temsil | gömme vektörleri, ağırlık matrisleri | birimler, bağlar, kökene göre sayılmış kanıt |
| Dikkat | bağlam penceresine softmax ile yoğun bakış | Yedili Odak: yerel, seyrek, okunabilir; bağlam penceresi yok |
| Güncelleme | geri yayılım, gradyan | yerel kanıt sayımı, tek kural |
| Unutma | kesin silmek yeniden eğitim ister | defter tutulduğu sürece bir deneyimin kanıtı tam geri alınır |
| Bilmemek | her zaman bir olasılık dağılımı | beklenti ½: ölçülen bilmeme hissi |
| Kaynak | bilgi başına tutulmaz | her kanıtın kökeni ve güveni tutulur |
| Donanım | büyük modeller eğitim için GPU ister; küçükler işlemcide eğitilir | sıradan bir işlemci, ayrı eğitim yok; RAM yaşadıkça büyür |

Aynı deneyim akışında küçük Transformer'larla, az veriyle yaşarken öğrenme koşulunda yapılan karşılaştırma (v5;
Zihin "members" ayarında, iki küçük üretilmiş dünya, 5 tohum, hayat başına 6–8 sınama):
- Sorulan boyutu hiç söylenmemiş örneklerde Zihin %90 / %97 ve %75 / %95'e ulaştı. Tek geçişte eğitilen Transformer
  %67 / %57 ve %60 / %50'de, yakınsayana kadar eğitilen %77 / %67 ve %53 / %53'te kaldı.
- İkinci bir alan, tek geçişli Transformer'ın ilk alandaki doğruluğunu 0.85'ten 0.74'e düşürdü; Zihin'de 1.00
  kaldı. Yakınsayana kadar eğitilen Transformer bunun için sınanmadı.
- Sohbet programının çalıştırdığı ayar ("speaking") bu karşılaştırmada yoktu. Aynı dünyalarda kötü sonuç verir
  (600 deneyimde 0.55 ve 0.04); ayrıca bu dünyaların yapılmadığı biçimde akışın içine konuşur; bu ayrıca
  incelenmedi.

Bu karşılaştırma küçük modellerle ve az veriyle yapıldı; çok daha fazla veriyle fark kapanabilir.

## 11. İddia etmediklerimiz

- Bu bir bilinç ya da yapay genel zekâ iddiası değildir; bunlar bugün hiçbir makine için ölçülemiyor.
- Her parçanın yeni olduğunu iddia etmiyoruz: tek kural NARS'ın beklentisidir, Benzerlik Oyu sınıflandırmanın örnek
  tabanlı modellerini izler, odak bir k-kazanan yarışıdır. Aşağıdaki adlar bu kuramın kullandığı adlardır.
- Zihin akıcı bir dil modeli değildir; gerçek veride sohbeti büyük ölçüde yaşadıklarından ve taklitten gelir.
- Olumsuz sonuçlar silinmez:
  - öğrenmeyi öğrenmenin ilk denemesi olumsuzdu;
  - doğuştan kelime parçalarıyla büyüyen zihin belleği tüketti;
  - tek duyum kanıtı düzeltmesi sohbeti kötüleştirdi;
  - sohbet ayarı, "members" ayarının iyi sonuç verdiği üretilmiş dünyalarda kötü sonuç verir;
  - 15 Kasım 2026'ya kadar sohbet programında, geliştiricinin yazdığı, zihin hakkında ve "bilmiyorum" demek üzerine
    ezber dersler vardı; sahibin kararıyla çekirdekten çıkarıldı. Daha önce yetiştirilen zihinler bunları yaşadı;
    bunları kullanan ölçümler bunu belirtir.

Hepsi notlarda ölçümleriyle durur.

## Bu kuramda kullanılan adlar

| Ad | İngilizce | Not | İlgili önceki çalışma |
|---|---|---|---|
| Yaşayan Kanıt Zihni | Living Evidence Mind | mimarinin adı | |
| Köken Kanıtı | Provenance Evidence | altı kökene göre W+/W− | psikolojide kaynak izleme |
| Kanaat Beklentisi | Conviction-weighted Expectation | e = c(f − ½) + ½, tek kural | NARS doğruluk değeri beklentisi |
| Güven Yargısı | Trust Judgment | dört dolaylı kökene güven doğrudan deneyimle öğrenilir | |
| Yedili Odak / Odak Yerleşmesi | Seven-Place Focus / Focus Settling | dikkat mekanizması | k-kazanan yarışı |
| Kesin Unutma | Exact Retraction | defterle bir deneyimin kanıtının tam geri alınması (v5'te T9 ile adlandı) | kesin makine unutması |
| Üye Okuması | Members' Reading | F11 | |
| Kendi/Sonraki Yuvaları | Self/Next Slots | F12 | |
| Şimdiden Öğrenme | Present-Only Learning | F13 | |
| Beklenen Başarısızlık | Expected-Failure Rule | F14 | |
| Sonuç İzi | Consequence Trace | F15, L5 | uygunluk izleri |
| Sürdürme Kuralı | Continuation Rule | F16 | |
| Benzerlik Oyu | Resemblance Vote | F18 | örnek tabanlı modeller (genelleştirilmiş bağlam modeli) |
| Avantaj Yarışı | Advantage Race | F19 | |
| Yapısal Hayal | Structural Imagination | F20, bir gösterim | |
| Benlik Birimi | Self Unit | F21, bir gösterim | |
| Rüya Ayıklama | Dream Sorting | F22 | |
| Durum Benzeşimi | Situation Resemblance | F23 | |
| Ses Güvenilirliği | Voice Credibility | F24, varsayılan kapalı | |
| Doğuş Başarısızlıkları | Born Failures | F25, incelendi, varsayılan kapalı | |
| Bilmeme Hissi | Felt Not-Knowing | L7 | bilme hissi (üstbiliş) |
| Önce Düşün | Think, then Speak | T8 sohbette | |
| Kararsızlıkta Deneme | Trying in Hesitation | keşif | |
| Ortak Kök Çıkarımı | Shared-Stem Inference | T8 ikinci biçim | |
| Çevrimiçi Yaşam | Online Life | oturum boyunca kendi zamanında yaşar | |
| Tek Köken Satırı | One-Origin Row | bellek düzeni | |

## Bilgisayarınızda çalıştırmak

Model her konuşmada öğrenir, ona kibar konuşun. Söylediğiniz her şey onun yaşamına girer.

Python 3.11 ya da daha yenisi gerekir. Linux / macOS:

```
git clone https://github.com/myylogic/cevahir-ai
cd cevahir-ai
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy           # isteğe bağlı: python-docx (Word), pypdf (PDF), pillow (resim)
python -m zihin
```

Windows PowerShell:

```
git clone https://github.com/myylogic/cevahir-ai
cd cevahir-ai
pip install torch numpy
.\cevahir.ps1                     # betikler engelliyse: powershell -ExecutionPolicy Bypass -File .\cevahir.ps1
```

Programı `cevahir-ai` klasörünün içinde başlatın. Önce "Seninle konuşan kim?" diye sorar; adınızı yazın. İlk
açılışta zihin boştur ve hiçbir dil bilmez; sizinle yaşayarak öğrenir. Komutlar kabukta değil, programın içinde
yazılır:

```
!oku belgelerim/                  # .txt .md .docx .pdf belgeler, .json/.tsv diyaloglar, kaynak kod
!ogret diyaloglar.tsv             # her satır: soru<TAB>cevap (UTF-8 olarak kaydedin)
!izle selam || merhaba            # bir konuşmayı izlemek
!dogru  /  !yanlis                # son cevabını nasıl karşıladığınız
!uyu    !dusun <konu>    !kendin  # rüya, iç ses, iç gözlem
!cik                              # çıkış
```

Açık kaldığı sürece kendi zamanında yaşar: düşünür, bir sessizliğin ardından kendiliğinden konuşabilir, on dakika
sonra uyur (`--yasamsiz` bunu kapatır). Zihin her oturumun sonunda `~/.cevahir/zihin/cevahir.zihin` dosyasına
kaydedilir (Windows: `%USERPROFILE%\.cevahir\zihin\cevahir.zihin`; klasörü `CEVAHIR_HOME` değiştirir; `--kaydetme`
kaydetmez).

Şimdiye kadar yetiştirilen zihinler yazarın özel Türkçe verilerini yaşadı ve bu depoda değildir. Sizin kopyanız,
`!ogret` ve `!oku` ile ona öğrettiğiniz dili öğrenir. Türkçe dışında bir dilde konuşacaksanız programı `--dil diger` ile
başlatın, böylece büyük harfler Türkçe kurallar olmadan küçültülür. Bir düzine soru-cevaptan sonra gösterilen soruları cevaplar; yeni bir dilin görmediği sorular
için ne kadar veri gerektiği ölçülmedi. Zihin dosyası okuduklarını içinde taşır; özel belgelerle yetiştirdiğiniz bir
zihni paylaşmayın. Ayrıntılı kılavuz: [ZIHIN_KILAVUZ.md](ZIHIN_KILAVUZ.md).
