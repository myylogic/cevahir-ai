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

**Güven Yargısı (Trust Judgment).** Algılanan kanıt ve zihnin kendi eylemleri doğrudan deneyimdir; güvenleri tanım
gereği 1'dir. Yalnız algılanan kanıt, yani dünyanın deneyimi, hakemdir: zihnin kendi eyleminin yankısı kesindir ama
dünyaya dair bir iddianın tutup tutmadığı hakkında bir şey söylemez. Dört dolaylı kökene (söylenen, taklit edilen,
çıkarılan, hayal edilen) güven ½'den, yani kuralın kanıtsız değerinden başlar ve öğrenilir: dolaylı kanıt, aynı bağda
sonradan gelen algılanmış kanıtla karşılaştırılır; tutanlar o kökene güveni artırır, tutmayanlar azaltır. Böylece zihin kime ne kadar
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
- **Kesin İlişkiler (Exact Relations, F28):** bir ilişki bir üye için ancak o üyenin yaşadığı değeri tam olarak
  adlandırıyorsa sayılır; okunan örnek için nasıl çözülecekse öyle çözülür. Aynı değeri adlandıran ilişkilerden yalnız
  en özgül olanı sayılır, değerin kendisi de her zaman sayılır. Önceden bir üyenin değeri, yalnızca onun devam ettiği
  kelimelerden biri olduğu için "sonraki" sayılıyordu, hiçbir zaman değerin kendisi olarak sayılmıyordu. Değerin sorulan
  şeyin hemen ardından geldiği bir dünyada ("<ilişki> <varlık> <değer>") hepsi aynı değeri yaşamış üyeler "sonraki"de
  birleşmiş okunuyordu; bu da sorulan şeyin kendi alışkanlıklarını gösteriyordu. O dünyada görülmemiş örneklere doğru
  cevap %3'ten %81'e çıktı.
- **Yerine Geçebilirlik Benzetmesi (Analogy by Interchangeability, F29):** iki kelime, okunan sözden önce aynı
  yaşanmış yerleri doldurdukları ölçüde birbirinin yerine geçebilir; az kelimenin paylaştığı bir yer, her şeyin
  paylaştığı bir yerden daha çok şey söyler. Bir kategorinin üyelerinin yaşadıkları, yeni bir örneğe onlarla olan yerine
  geçebilirliği üzerinden taşınır (NARS'ın benzetme kuralı); neyin iddia edildiğine yine üyeler karar verir, benzetme
  yalnız bunun ne kadar sağlam tuttuğunu söyler. "selam , adın ne ?" artık yaşanmış "merhaba , adın ne ?" gibi
  cevaplanıyor; okunanlarda adın ile ismin aynı yerleri doldurduysa "ismin ne ?" da "adın ne ?" gibi. Hiç görülmemiş bir
  kelime hiçbir şey taşımaz.
- **Bağımsız Kanıtların Birleşmesi (Corroboration, F30):** aynı iddia hakkındaki bağımsız yaşanmış kanıtlar toplanır
  (NARS'ın birleştirme kuralı): konuşan çerçevenin iddiaları, en benzer yaşanmış durumların aynı değerler hakkında
  söyledikleriyle birleştirilir; her yaşanmış bağlam bir kez sayılır. "peki adın ne ?" sorusu, "adın ne"nin önünde bir
  kelime olan soruların çerçevesi ile yaşanmış yalın soru aynı şeyi söylediği için cevaplanıyor; benzer durumlar
  birbirini tutmuyorsa hiçbir şey eklenmez.
- **Dizisel İlişki (Paradigmatic Relation, F31) ve İlişkisel Bağlama (Relational Binding, F32):** bir çerçevenin
  değeri, dolgunun durduğu yerde durabilir: dolguyla en çok yer değiştirebilen öğe (çerçevenin kendi yerindeki rakip
  dolgulardan biri değil), değerin dolguyla ilişkisidir, tıpkı aynılık ve devamın olduğu gibi; bir yer, tekrarlarını
  bağladığı gibi dolgunun yerine geçeni de bağlar. Dört tanımdan ("otomobil ne demek ?" -> "otomobil , araba demektir .")
  ve kelimelerin okunanlarda doldurduğu yerlerden, "selam ne demek ?" sorusu "selam , merhaba demektir" diye
  cevaplanıyor. Bu, dilin dizimsel ekseninin yanındaki dizisel (dikey) eksendir (Saussure, Jakobson).
- **İlişkisel Bellek (Relational Memory, F33):** tek zihin, tek bellek; hatırlama karşıdaki kişiye göre. Zihnin
  tanıdığı biri varken yaşanan her şey o kişiyle birlikte de not edilir. Hatırlama kimin orada olduğuna bağlıdır:
  kimseye özel olmadan yaşanan herkesle hatırlanır; karşıdaki kişiyle yaşanan onundur ve en son gelir; başkalarıyla
  yaşanan ise yalnız başka insanlar da aynısını yaşadıkça hatırlanır (tek bir kişinin yaşadığı onundur). Herkese kendi
  geçmişinden cevap verilir, bir kişiyle yaşanan sır başka kimseye söylenmez, herkesin öğrettiği bir yabancıyla da
  bilinir; hiçbir rol ya da ilişki kodlanmaz.
- **Cevap İlişkisi (Reply Relation, F34):** bir çerçevenin değeri, dolgunun tek başına söylendiğinde aldığı cevap
  olabilir ve bir önekten sonraki birkaç konum dolguya bağlanabilir (yerine geçeni, cevabı ya da her üyenin orada
  sahip olduğu bir öğe); bir çerçevenin iddiası, bağlamının çerçevenin her sabit kelimesini koruyan yaşanmış
  sonekleriyle birleştirilir (birini dışarıda bırakan sonek başka bir bağlamdır: "jump left and walk ⏎" için
  "walk ⏎"). SCAN'in "add
  primitive" bölünmesinde (zihnin penceresine göre küçültülmüş) yalnız tek başına görülen bir komut ("jump" -> "i_jump")
  ilk sorulduğunda her kalıpta kullanılıyor ("jump twice", "walk and jump", "jump left twice"): 75 komutun 75'i;
  aynı veriyle sıfırdan eğitilmiş, tüm eğitim komutlarını doğru cevaplayan bir Transformer 4'ünü cevaplıyor. SCAN'in
  kendi dilbilgisinde, pencere 11'de, her önekin bağlı çerçeveleri okunarak: 844 komutun 789'u (Transformer 0);
  pencereye bütünüyle sığan her komut doğru; 12'lik pencereyle 844'ün hepsi (13'lük pencereyle 12 öğeye sığan
  1.188 komutun hepsi).
- **Genelleşmiş Sonuçlar (Generalised Consequences, F35):** bir eylemin neye yol açacağı (iyi, kötü), yolun
  çerçeveleri üzerinden benzer yolların neye yol açtığından okunur; yolun kendi kaydı en son gelir. Cevaplarının nasıl
  karşılandığından birkaç durumda öğrenen bir zihin, hiç karşılaşmadığı durumlarda da aynı biçimde davranır (hiç
  kendisine verilmemiş şeylere kibar cevap: 9'da 0'dan 8'e, izlediği alışkanlığa karşın); F33 ile bir kişiyi memnun eden
  şey o ilişkinin beklentisidir (zıt zevkleri olan iki kişinin her birine kendi istediği biçimde cevap: 9'da 9).
- **Kelime Biçimleri (Word Forms, F36):** kelimelerin altındaki katman: hiç görülmemiş bir kelime, en uzun kökünü
  paylaştığı yaşanmış kelime olarak okunur; kök, yaşanmış kelimelerin kendi sınırıdır (Harris'in ardıl çeşitliliği) ve
  okuma, kökün kelimeyi ne kadar kapladığıyla taşınır. "merhabalar" "merhaba" gibi selamlanıyor, "nasılsınız ?"
  "nasılsın ?" gibi cevaplanıyor; tek bir yaşanmış kelimenin rastlantıyla sahip olduğu bir başlangıç kök değildir.
- **Kalıtım (Inheritance, F37, varsayılan olarak kapalı):** istisnalı tümdengelim, bir kelimenin doldurduğu yerlerden
  okunur: bir yerin sınıfı, çerçevesinin en ayırt edici sabit kelimesiyle adlandırılır; tek başına anılan sınıf, onunla
  kök paylaşan yaşanmış bir kelimedir ("kuştur" için "kuşlar"). O kelimenin devamı üyeye NARS tümdengelimiyle taşınır
  (f = f1 f2, c = f1 f2 c1 c2 f_n); üyenin kendi kaydı, yani sınıf arkadaşlarının çoğunun sürmediği devamı, en son
  gelir. Ayrı ayrı okunmuş olgular ("kartal bir kuştur .", "kuşlar uçar .") öğretilmemiş bir sonuç veriyor ("kartal
  uçar"); kendisi için söylenmiş bir istisna ("penguen yüzer") sınıfını geçiyor: T-deduce'da 48'de 48 (önce 2),
  yanlış genelleme yok. Bir sınıf kelimesi de bir sınıfına ait olduğu söylenen sınıflardan miras alır (bir sınıfa ait
  olduğunun söylendiği yerde sürdüğü, "balıklar bir hayvandır ." içindeki "bir", söyleyişin biçimidir, özellik değil);
  böylece iki adım ötedeki sonuca ulaşılıyor: T-chain'de 27'de 27 ("levrek bir balıktır .", "balıklar bir hayvandır .",
  "hayvanlar solur ." okunmuşken "levrek solur"), F37'siz ya da tek adımla hiçbiri; yakın özellik 9'da 9, istisna 3'te
  3. Varsayılan olarak kapalı: en büyük zihinde bir cevap yaklaşık on bir kat uzun sürüyor.
- **Tanığın Güvenilirliği (Witness Reliability, F38, varsayılan olarak kapalı):** bir çerçevenin iddiasının sabit bir kelimesini dışarıda
  bırakan daha kısa bir yaşanmış bağlam, zihnin onu ne kadar uyuşurken gördüyse o kadar iddianın tanığıdır: dışarıda
  kalan kelimeyle ve kısa bağlamla biten yaşanmış bağlamlar, kısa bağlamın sürdüğü gibi sürmüştür. Tanığın söylediği
  NARS tümdengelimiyle taşınır (f = f1 f2, c = f1 f2 c1 c2); bu yüzden hiç sürmediği bir değer hakkında hiçbir şey
  söylemez. "adın ne ?", "selamlar , adın ne ?" için tanıktır (", adın ne ?" ile bitenler "adın ne ?" gibi sürdü);
  "look thrice", "jump thrice and look thrice" için tanık değildir. Bir okumanın dışarıda bıraktığı bağlam parçası
  için, zihnin kendi yaşadığı bağlamlarla ölçülen kaynak güvenilirliği. Varsayılan olarak kapalı: onunla cevap vermek
  bir şey kaybettirmiyor, ama onunla yaşanan bir SCAN hayatı başka türlü öğrendi (789 → 767).
- **Değişiklik Tanığı (Replacement Witness, F39, varsayılan olarak kapalı):** Mill'in fark yöntemi, zihnin kendi
  yaşamından okunur: hiçbir şey söylenmeyecekken, şimdiki durumdan tek bir değişiklikle ayrılan yaşanmış bir durum,
  aynı değişiklikle ayrılan yaşanmış çiftler aynı biçimde sürdükçe onun tanığıdır (değişiklik yaşandığı yerde fark
  yaratmadı). Bulut hem "şemsiye al" sözünün hem yağmurun ortak nedeniyken, sözler tek başına zihni güneşli günde
  yağmura yaslıyor (0.556); F39 ile güzel hava önde (0.577), çünkü bulutlu günlerde iki öğüt aynı biçimde sürdü, aynı
  öğütle bulutlu ve güneşli günler sürmedi. Zihin ona yaslanıyor, henüz ileri sürmüyor. Onunla yaşanınca SCAN (11
  içinde) 789 cevabını, aynılarını koruyor.
- **Sürdürme İsteği (Urge to Go On, F40, varsayılan olarak kapalı):** beklenen iki kelime ya birbirinin alternatifidir
  (yalnız biri doğru olabilir) ya da aynı öğe hakkında bilinen iki şeydir (ikisi de doğru; yalnız sıra için yarışırlar).
  Bir öğenin yaşanmış devamlarındaki payları bunun ne ölçüde olduğunu söyler: öndeki aday rakibine değil sessizliğe
  karşı yarışır. Uzun cevaplara ilk adım: hiç sorulmamış şeyler hakkında daha çok olgu, uydurma artmadan (T-long 54'te
  20 → 22: yarıda kalan iki cümle tamamlandı; uydurma 4 ve 4).
- **Benzerlik Oyu (Resemblance Vote):** iki şey ortak yönleriyle benzer. Bir yön, ne kadar az şeyde bulunuyorsa o
  kadar ayırt edicidir. Yeni bir şeye en çok benzeyen üyeler en yüksek sesle konuşur; ağırlık, sınıflandırmanın
  örnek tabanlı modellerindeki gibidir (genelleştirilmiş bağlam modeli). Görme ve işitme böyle tanır: yeni
  resimlerde %73 / %83 / %67, bilinen üç yapay sesin yeni seslerinde %93 / %97 / %93 (şans %33).
- **Durum Benzeşimi (Situation Resemblance):** hiç bütün olarak yaşanmamış bir söze, en çok benzeyen yaşanmış
  durumlarda ne geldiyse onunla karşılık verilir. Benzerlik zayıfsa, örneğin sözde hiç bilinmeyen bir kelime
  varsa, uydurulmaz.
- **Yaşanmış Dışlama (Lived Exclusion, F27):** bir kanalda her an tek bir öğe olur. Bu yüzden bir durumun kendisinden
  sonra yaşanan, bir genellemenin orada önerdiği ama o durumdan sonra hiç yaşanmamış seçeneğin aleyhine kanıttır.
  Böyle bir öneri lehinde kanıtı olmadan, aleyhinde de durumun o kanaldaki kendi yaşanmış miktarıyla okunur ve kendi
  kanıt gibi en son işlenir. Özgül olan, genel olanı yaşadıklarında zaten geçiyordu; artık yaşamadıklarında da onun
  aleyhine sayılır. Üretilmiş dünyalarda aynı yaşamlar böyle okunduğunda istisnalar daha sık korunur (nitelik
  0.74 → 0.84).

## 6. Konuşmak, düşünmek, bilmemek

- **Sürdürme Kuralı (Continuation Rule):** şu anki durumda hiçbir şey ½'nin üstünde bir şey beklemiyorsa, zihin son
  algılanan olaydan sonraki öğeleri okur; bunların yaşanmış devamı olan en uzun kuyruğu, düşürülmüş bir kanaatle
  konuşur. Bu kural bir söz başlatmaz ve şu anki bağlamın içinde kalan karşı taraf sözlerini dışarıda bırakır;
  böylece karşısındakinin az önce söylediğini tekrarlamaz.
- **Dizi Başları (Sequence Starts, F26):** bir kelimenin, onunla başlayan bir dizide (okunan bir cümle, sessizlikten
  sonraki bir söz) nasıl sürdüğü, her yerde nasıl sürdüğünden ayrı tutulur. Yeni bir şey saklanmaz: x, v diye başlayan
  her bölüm iki öğeli (x, v) birimini bir kez tanır; bu tanımalar sayılır, öbür başlangıçlar aleyhine sayılır. Bir
  çerçevenin üyeleri "cevap, sorulan kelimeyle başlayan dizi gibi sürer" ilişkisinde (BAŞ) birleşebilir; şimdiye dek
  tek kelimelik bir söz de, sessizlikten sonra akla gelen bir kelime de öyle sürer. Şablonsuz bir sınavda (belgeleri
  bir kez okumuş bir zihin, rastgele 200 cümle, ilk iki kelimeleri akla getirildi) 150'sinde (Kesin İlişkiler ile 145'inde) o kelimelerle başlayan
  okunmuş bir cümleyi, ilk iki kelimesi yalnız tek bir cümleyi başlatan 61 cümlenin 60'ında o cümlenin kendisini
  eksiksiz söyledi. Aynı belgelerle eğitilen en güçlü küçük Transformer (7,4 milyon parametre, 50 dakika) 41'inde
  geçerli bir cümle söyledi. İpucu tek kelime olunca Transformer daha sık geçerli cümle söyledi (51'e karşı 44; Kesin İlişkiler ile 47): bir
  kelime birkaç cümleyi başlatıyorsa zihin susar.
- **Avantaj Yarışı (Advantage Race):** aynı kanaldaki seçenekler birbirini dışlar. Önde olan, en güçlü rakibine
  karşı avantajı eşiği geçerse söylenir; bütün seçenekler eşitse susulur.
- **Kararsızlıkta Deneme (Trying in Hesitation):** iyi desteklenen seçenekler arasında kalındığında susmak yerine,
  sonucunu en az bildiği denenir. Bu yalnız sıra yeni geldiğinde ya da o durumda daha önce iyi veya kötü bir sonuç
  yaşanmışsa olur. "Yanlış" denen bir cevap bir dahaki sefer denenmez, başka bir cevap denenir.
- **İç Ses (Inner Voice):** `!dusun` ile ya da on saniyelik bir sessizlikten sonra zihin, son duyduğu sözün en iyi
  konusu (en ayırt edici ve en az bilinen kelime) üzerine düşünür. Düşünceleri iç ses olarak, hayal kökeniyle yaşanır
  ve zihnin sonraki kararlarını yalnız onda bıraktıklarıyla etkileyebilir; zihin adına hiçbir şey söylenmez.
  (15 Kasım 2026'ya kadar program her cevaptan önce düşünceler çağırıp yeterince gerçek ilkini onun yerine
  söylüyordu, "Önce Düşün"; bu kaldırıldı.)
- **Bilmeme Hissi (Felt Not-Knowing):** bilmemek bir çıktı değil, ölçülen bir iç haldir:
  `u = 1 − 2·max(0, en iyi beklenti − ½)`. Bu hal bir sessiz anın ardından tek başına algılanır. O anda ne
  denileceği, zihnin birlikte yaşadığı insanlardan ve verilerinden öğrenilir; çekirdekte bunun için hazır bir cümle
  yoktur. Soru sormak da aynı yolla öğrenilir (bilinmeyen kelimeyi '?' ile geri soran kodlanmış merak refleksi 15 Kasım
  2026'da kaldırıldı). Bunu nasıl söyleyeceğini yaşamamış bir zihin susar. İki güçlü cevap
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
- **Sıkı indeksler ve gün deposu:** kanıtın yanındaki indeksler Python nesneleriydi, yüklenmiş bir zihnin RAM'inin
  %78–87'si. Artık paketlenmiş anahtarlar, 32 bitlik kimlik dizileri ve uzun süreli depo ile o günden beri kurulan
  bağların gün deposu olarak tutulan, uykuda birleştirilen bir bağ araması (kesin; sonuçlar bit bit aynı). 3.25 milyon
  bağlı yüklenmiş bir zihin Python ve torch dışında 1784 MB'tan 762 MB'a indi, bağ başına yaklaşık 234 bayt.
- **Açık sorun:** bellek hâlâ yaşanmış veriyle doğrusal büyür; an başına yaklaşık 11 bağ oluşur. Sonraki adımlar,
  yalnız hâlâ değişebilecek bağları okumak ve diske eşlenmiş bir uzun süreli depo, notlarda planlıdır.

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
- Sohbet programının çalıştırdığı ayar ("speaking") bu karşılaştırmada yoktu. Aynı dünyalarda kötü sonuç veriyordu
  (600 deneyimde 0.57 ve 0.04). İki neden bulundu. Birincisi, dünyanın akışının içine konuşuyordu: bu dünyalar zihnin
  her an davranmasına izin verir, o da kendi sözlerini dünyanınkilerin arasında yaşadı ve çoğu bunlardan olmak üzere
  beş kat fazla kategori öğrendi. İkincisi, F12'nin "sonraki" ilişkisi üyelerin bir değerde birleşmesini ele
  geçiriyordu (Kesin İlişkiler). Sohbette olduğu gibi dünya konuşurken dinleyince, Dizi Başları, Yaşanmış Dışlama ve
  Kesin İlişkiler ile 0.78 ve 0.81'e ulaşıyor; üyeler, istisnalar ve gizli sınıflar 1.00. "Members" ayarının ilk
  dünyadaki 0.93'üne ulaşılmadı.
- Bir kez okunanı geri çağırma (F26, F27) düellonun Transformer'larıyla 6. bölümde, Dizi Başları altında
  karşılaştırıldı. Önceki aynı-veri düellosu ayrılmış konuları iki tarafa da çıplak bir kelime olarak sordu; öğretmen
  diyalogları bu biçimi hiç kullanmamıştı, yani iki tarafı da öğrenmedikleri bir soruyla ölçtü. Onun yerini şablonsuz
  sınav aldı (research/living_mind_2026_11_15/NOTES.md).

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
  - sohbet ayarı, "members" ayarının iyi sonuç verdiği üretilmiş dünyalarda kötü sonuç veriyordu (çoğunlukla
    dünyanın akışına konuştuğu ve F12'nin "sonraki" ilişkisi üyelerin birleşmesini ele geçirdiği için; ikisi
    giderilince 0.93 ve 0.81'e karşı 0.78 ve 0.81);
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
| Dizi Başları | Sequence Starts | F26 | |
| Yaşanmış Dışlama | Lived Exclusion | F27 | |
| Kesin İlişkiler | Exact Relations | F28 | |
| Yerine Geçebilirlik Benzetmesi | Analogy by Interchangeability | F29 | dağılımsal varsayım; NARS benzetmesi |
| Bağımsız Kanıtların Birleşmesi | Corroboration | F30 | NARS birleştirmesi |
| Dizisel İlişki | Paradigmatic Relation | F31 | dizisel eksen (Saussure, Jakobson) |
| İlişkisel Bağlama | Relational Binding | F32 | bir çerçeve yerinin değişken bağlaması |
| İlişkisel Bellek | Relational Memory | F33 | bağlama bağlı bellek; kaynak izleme |
| Cevap İlişkisi | Reply Relation | F34 | bileşimsel genelleme (SCAN); yaşanmış soneklere geri çekilme |
| Genelleşmiş Sonuçlar | Generalised Consequences | F35 | modele dayalı seçim: sonuçların beklenen değeri |
| Kelime Biçimleri | Word Forms | F36 | ardıl çeşitliliğiyle biçimbirim sınırları (Harris) |
| Kalıtım | Inheritance | F37, varsayılan olarak kapalı | NARS tümdengelimi; istisnalı varsayılan kalıtım |
| Tanığın Güvenilirliği | Witness Reliability | F38, varsayılan olarak kapalı | kaynak güvenilirliği; NARS tümdengelimi; değişken dereceli modellerde geri çekilme |
| Değişiklik Tanığı | Replacement Witness | F39, varsayılan olarak kapalı | Mill'in fark yöntemi |
| Sürdürme İsteği | Urge to Go On | F40, varsayılan olarak kapalı | dürtü ve alışkanlık (Hull): ne söyleneceğinden ayrı olarak konuşup konuşmamak |
| Ses Güvenilirliği | Voice Credibility | F24, varsayılan kapalı | |
| Doğuş Başarısızlıkları | Born Failures | F25, incelendi, varsayılan kapalı | |
| Bilmeme Hissi | Felt Not-Knowing | L7 | bilme hissi (üstbiliş) |
| İç Ses | Inner Voice | hayal olarak yaşanan düşünme (önceki "Önce Düşün" kodu kaldırıldı) | |
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

Açık kaldığı sürece kendi zamanında yaşar: sessiz anlar yaşar, on saniye sessizlikten sonra içinden düşünür, on dakika
sonra uyur (`--yasamsiz` bunu kapatır); bir şey söylemesi yalnız kendi kararı söz çıkarırsa olur. Zihin her oturumun sonunda `~/.cevahir/zihin/cevahir.zihin` dosyasına
kaydedilir (Windows: `%USERPROFILE%\.cevahir\zihin\cevahir.zihin`; klasörü `CEVAHIR_HOME` değiştirir; `--kaydetme`
kaydetmez).

Şimdiye kadar yetiştirilen zihinler yazarın özel Türkçe verilerini yaşadı ve bu depoda değildir. Sizin kopyanız,
`!ogret` ve `!oku` ile ona öğrettiğiniz dili öğrenir. Türkçe dışında bir dil için yeni bir zihni `--dil diger`
ile başlatın; böylece büyük harfler Türkçe kurallar olmadan küçültülür (bir zihin büyüdüğü kuralı korur). Bir düzine soru-cevaptan sonra gösterilen soruları cevaplar; yeni bir dilin görmediği sorular
için ne kadar veri gerektiği ölçülmedi. Zihin dosyası okuduklarını içinde taşır; özel belgelerle yetiştirdiğiniz bir
zihni paylaşmayın. Ayrıntılı kılavuz: [ZIHIN_KILAVUZ.md](ZIHIN_KILAVUZ.md).
