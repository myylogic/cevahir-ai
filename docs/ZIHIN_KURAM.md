# Cevahir Zihin Kuramı: Yaşayan Kanıt Zihni

*Muhammed Yasin Yılmaz — Kasım 2026. Bu metin, depodaki `zihin/` çekirdeğinin kuramını ve mimarisini anlatır. Her
iddianın ölçümü ve olumsuz sonuçları `research/living_mind_2026_11_14/` ve `research/living_mind_2026_11_15/`
klasörlerindeki notlarda ve sonuç dosyalarındadır.*

## Kısaca

Cevahir Zihin bir Transformer değildir: içinde sinir ağı, ağırlık matrisi, gömme vektörü, geri yayılım ve softmax
dikkati yoktur. Bir **Yaşayan Kanıt Zihni**dir. Her algıladığı şeyi bir birim, birimler arasındaki her ilişkiyi bir
bağ olarak tutar; her bağ, o ilişkinin kaç kez doğrulanıp kaç kez çürütüldüğünü **kaynağına göre ayrı ayrı** sayar
(kendi gözüyle mi gördü, biri mi söyledi, başkasından mı taklit etti, kendisi mi yaptı, akıl yürüterek mi çıkardı,
hayal mi etti). Ne beklediğini tek bir kuraldan okur, her an öğrenir ve öğrendiği tek bir deneyimi tam olarak geri
alabilir.

Eğitim ve kullanım ayrı aşamalar değildir: **öğrenmek yaşamaktır**. Model her konuşmada öğrenir.

## 1. Temel varlıklar

**Kanal.** Zihnin dünyayla ve kendisiyle temas yolları:
- `sense`: duyulan, görülen;
- `act`: söylenen, yapılan;
- `feel`: şaşkınlık, rahatlama, çatışma, yük;
- `knob`: kendi ayarları;
- `source`: kaynaklar;
- `facet`: bir şeyin görünüşü, sesi, kelime parçaları;
- `value`: iyi ya da kötü sonuç;
- `context`: konuşan kişi, ton, benlik.

**Birim.** Üç türü vardır:
- **Atom:** bir kanaldaki bir simge (bir kelime, bir resmin özeti, bir sıra sonu).
- **Birleşik birim:** yaşanmış bir dizi (yedi öğeye kadar bir an penceresi).
- **Kategori:** aynı yerlerde yaşanmış birimlerin sınıfı.

**Bağ.** İki birim arasındaki ilişki:
- `EXPECT` (gecikme 0: birlikte, 1: hemen sonra, 2: aynı bölümün sonunda);
- `PART` (bir dizinin parçası);
- `ABSTRACT` (bir kategorinin üyesi);
- `RELY` (bir genellemenin güvenilirliği).

**Köken Kanıtı (Provenance Evidence).** Her bağ, altı köken için ayrı ayrı lehte (W+) ve aleyhte (W−) kanıt
tutar: *algılanan, söylenen, taklit edilen, yapılan, çıkarılan, hayal edilen*. Bilginin nereden geldiği hiçbir zaman
kaybolmaz.

## 2. Tek kural: Kanaat Beklentisi

Bir bağın ne kadar beklendiği, kökenlere verilen güvenle tartılmış kanıttan okunur:

- sıklık `f = W+ / W`;
- kanaat `c = W / (W + k)`, burada `k = 1`;
- beklenti `e = c · (f − ½) + ½`.

Hiç kanıt yoksa beklenti ½'dir: "bilmiyorum". Kanıt arttıkça kanaat büyür. Bütün okumalar bu kurala dayanır: ne
söyleyeceği, neyi bekleyeceği, neye şaşıracağı.

**Güven Yargısı (Trust Judgment).** Kökenlere verilen güven sabit değildir. Algılanan ve yapılan, "doğrudan
deneyim" sayılır. Söylenen, taklit edilen, çıkarılan ve hayal edilen kanıt, sonradan gelen doğrudan deneyimle
karşılaştırılır; tutanlar o kökene güveni artırır, tutmayanlar azaltır. Böylece zihin kime ne kadar inanacağını
kendisi öğrenir.

## 3. Dikkat: Yedili Odak

Transformer'da dikkat, her kelimenin önceki bütün kelimelere öğrenilmiş vektörlerle softmax üzerinden bakmasıdır.
Bu yoğun ve geçmişin uzunluğuyla büyür. Cevahir Zihin'in dikkati başka bir mekanizmadır.

**Yedili Odak / Odak Yerleşmesi (Seven-Place Focus).** Her an yalnız yedi birim odağa girer (k-kazanan, F = 7).
Adaylar:
- şu an algılananlar;
- geçen anlardan sönerek taşınanlar (her an `1 − 1/F` oranında);
- tanınan birleşik bağlam (bölümün yaşanmış dizisi);
- bunların kategorileri ve çerçeveleri (genellemeler);
- birlikte yaşanmışlık bağlarıyla çağrışanlar.

Yerleşme en çok altı tur sürer: çağrışım yayılır, kazananlar yeniden seçilir, değişmeyince durulur. Önce şu an
olan gelir, sonra her yerin en özgül genellemesi.

Sonuçlar:
- Bir anın maliyeti geçmişin uzunluğuyla değil, yerel bağlarla ilgilidir.
- Dikkat okunabilir: hangi yedi birimin neden odakta olduğu görülür.
- Öğrenilecek dikkat ağırlığı yoktur; dikkat, kanıttan doğar.

## 4. Bir anın akışı

Zihin her an (yaklaşık her kelime) şu adımları yaşar:

1. **Algı:** dışarıdan gelenler ile içeriden gelenler (kendi sözünün yankısı, hisleri, bilmeme hissi) birlikte
   algılanır. Kendisi konuşuyor ya da düşünüyorsa **Benlik Birimi** de o ana eşlik eder.
2. **Sonrayı öğrenmek (gecikme 1):** bir an önce odakta olanlar, şimdi olanı beklemeyi öğrenir.
   - **Şimdiden Öğrenme:** yalnız bir an önce gerçekten şimdi olan öğretir; odakta sönmekte olanlar öğretmez.
   - **Beklenen Başarısızlık:** olmayan bir şey yalnız beklenmişse aleyhte kanıt alır.
3. **Sonuçları öğrenmek:** iyi ya da kötü bir sonuç sinyali, bölümün **Sonuç İzi**ndeki her bağlama işlenir. Hangi
   yolun neye götürdüğü böyle öğrenilir.
4. **Odak:** Yedili Odak yerleşir (yukarıda).
5. **Birlikteliği öğrenmek (gecikme 0):** odaktaki atomlar birbirini beklemeyi öğrenir. Görünüş, ses ve parça
   gibi yönler, eşlik ettikleri öğeye bağlanır.
6. **Beklenti ve karar:** bir sonraki an için ne beklendiği okunur. Sıra zihindeyse söyleyecekleri yarışır.
7. **Bilmeme Hissi:** sıra zihindeyse ve söyleyecek bir şey yoksa, bilmediğini bir iç olay olarak algılar.
8. **Hisler:** şaşkınlık, rahatlama, çatışma ve yük bir sonraki anın iç algısı olur.
9. **Defter:** anın bütün değişiklikleri deftere yazılır. **Kesin Unutma** bununla mümkündür: tek bir deneyim,
   sonradan yaşananlar bozulmadan, tam olarak geri alınır.

## 5. Genelleme: kategoriler kendi üyeleriyle konuşur

- **Üye Okuması (Members' Reading):** bir kategori, bir örnek için ne bekleneceğini kendi kurallarından değil,
  üyelerinin kendi yaşadıklarından okur. Okunan örnek oylamanın dışında tutulur. Hiç görülmemiş örneğe genelleme
  yaşadıkça iyileşti: 600 deneyimde %93 / %81, 1500 deneyimde %100 / %90.
- **Kendi/Sonraki Yuvaları (Self/Next Slots):** "cevap sorulan kelimeyi tekrarlar" ya da "sonra o kelimenin devamı
  gelir" gibi ilişkiler, kelimeler farklı olsa da aynı ilişki olarak sayılır. Değişkenler kelimelerden doğar.
- **Benzerlik Oyu (Resemblance Vote):** iki şey ortak yönleriyle benzer. Bir yön, ne kadar az şeyde bulunuyorsa o
  kadar ayırt edicidir. Yeni bir şeye en çok benzeyen üyeler en yüksek sesle konuşur. Görme ve işitme böyle tanır:
  yeni resimlerde %73 / %83 / %67, yeni seslerde %93 / %97 / %93 (şans %33).
- **Durum Benzeşimi (Situation Resemblance):** hiç bütün olarak yaşanmamış bir söze, en çok benzeyen yaşanmış
  durumlarda ne geldiyse onunla karşılık verilir. Benzerlik zayıfsa, örneğin sözde hiç bilinmeyen bir kelime
  varsa, uydurulmaz.

## 6. Konuşmak, düşünmek, bilmemek

- **Sürdürme Kuralı (Continuation Rule):** başlamış bir söz kendi yaşanmış dizisi boyunca sürer. Ama bu kural bir
  söz başlatmaz ve karşısındakinin sözünü yankılamaz.
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
  denileceği, örneğin "bilmiyorum , öğrenmek isterim .", yaşayarak öğrenilir. Ardından **merak** bilinmeyen kelimeyi
  soru tonuyla sorar. İki güçlü cevap arasında kalmak bilmemek değildir; o durumda bilmeme hissi doğmaz.

## 7. Hayal, rüya ve akıl

- **Yapısal Hayal (Structural Imagination):** bir kavram, yaşanmış dizilerde aynı yerlerde durduğu bir başka
  kavrama benzetilir ("ateş evi ısıtır" / "güneş dünyayı ısıtır": ateş ↔ güneş). Birinin düşüncesi ötekine
  taşınır. Her adımın **gerçeklik puanı**, hayal hariç bütün kanıttan okunur.
- **Ortak Kök Çıkarımı (Shared-Stem Inference):** "kedi bir hayvandır" ve "hayvanlar nefes alır" bilgileri,
  ortak gövde (hayvan) üzerinden "kedi nefes alır" diye birleştirilir. Bu bir hipotezdir ve hayal olarak yaşanır.
  Gerçek veride ortak gövde bazen tesadüfi çıkar; bu yüzden çıkarım yalnız rüyada ve iç seste kullanılır,
  cevaplarda kullanılmaz.
- **Rüya Ayıklama (Dream Sorting):** uykuda zihin günün konuları ve emin olmadığı kelimeler üzerine rüya görür.
  - Rüya, çağrışımla bağlanan çıkarım ve benzetmelerden oluşur.
  - Bir kelimeyle birlikte gördüğü resmi rüyasında yeniden "görür".
  - Her rüya yeni iddialarıyla saklanır. Hayat sonradan bunlara dokunduğunda rüya **doğru**, **yanlış** ya da
    **açık** diye ayıklanır.
  - Hayal edilen her kanıt, gerçek kanıtla karşılaştırılır ve **kendi hayallerine güveni** buna göre değişir.
  - Hiçbir hayal silinmez; gerçek kanıt ona ağır basar.
- **Benlik Birimi (Self Unit):** zihin konuştuğu ve düşündüğü her anda doğuştan bir "ben" birimiyle birlikte
  algılanır. İç gözlem bu birimden okunur: ne kadar konuştu, ne kadar düşündü, kendini en çok neyle bağladı.

## 8. Öğrenmeyi öğrenmek ve sonuçtan öğrenmek

- **Ses Güvenilirliği (Voice Credibility):** bir örnek için konuşabilecek her genelleme bir "ses"tir. Sesin türü,
  kaç üyenin konuştuğu ve ne kadar hemfikir olduklarıdır. Zihin hangi tür sese ne kadar güvenebileceğini kendi
  sonuçlarından öğrenir. Bu bilgi kelime içermez, bu yüzden yeni bir alana taşınır. Sonuç karışık: bir yönde yeni
  alanı erken öğrenme hızlandı (+0.055, 10 hayatın 7'sinde), ters yönde hızlanmadı. Varsayılan kapalıdır.
- **Kodu sonucundan öğrenmek:**
  - Zihin önce çoğunlukla yanlış kod tamamlamaları izler.
  - Sonra kendi yazdığı kod gerçekten çalıştırılır; test sonucu iyi ya da kötü sinyali olarak döner.
  - Kararsızlıkta denemeyle, 4. turda altı küçük görevin hepsinde testten geçen kod yazar.
  - Sonuç sinyali olmadan oran %0'da, deneme olmadan %11'de kalır.

## 9. Bellek ve hız

- **Tek Köken Satırı (One-Origin Row):** bağların %99.6'sı tek bir kökenden kanıt taşır. Böyle bir bağ yalnız
  kökenini ve iki sayısını (W+, W−) saklar. Tam satır yalnız birden çok kökeni olan bağlara verilir. Bağ başına
  bellek 239 bayttan 69 bayta indi: 3.25 milyon bağlı bir zihnin çekirdeği 777 MB'tan 223 MB'a düştü. Sonuçlar bit
  bit aynı kaldı.
- **Hız:** bir an, 2.5 milyon bağlı bir zihinde yaklaşık 3–4 ms sürer. Ölçüm araçları her değişiklikte sonuçların
  bit bit aynı kaldığını denetler.
- **Açık sorun:** bellek yaşanmış veriyle doğrusal büyür; an başına yaklaşık 11 bağ oluşur. Milyonlarca örneğe
  giden yol (diske dayalı depo, uykuda pekiştirme) notlarda planlıdır.

## 10. Transformer ile karşılaştırma

| | Transformer | Cevahir Zihin |
|---|---|---|
| Öğrenme | eğit, dondur, kullan | her an öğrenir; ayrı eğitim yok |
| Temsil | gömme vektörleri, ağırlık matrisleri | birimler, bağlar, kökene göre sayılmış kanıt |
| Dikkat | softmax ile bütün geçmişe yoğun bakış | Yedili Odak: yerel, seyrek, okunabilir |
| Güncelleme | geri yayılım, gradyan | yerel kanıt sayımı, tek kural |
| Unutma | yeniden eğitim | Kesin Unutma: tek deneyim tam geri alınır |
| Bilmemek | her zaman bir olasılık dağılımı | beklenti ½: ölçülen bilmeme hissi |
| Kaynak | bilinmez | her kanıtın kökeni ve güveni tutulur |
| Donanım | GPU | sıradan bir işlemci |

Aynı deneyim akışında küçük Transformer'larla, az veriyle yaşarken öğrenme koşulunda yapılan karşılaştırma:
- Hiç görülmemiş örnekte Zihin %90 / %97 ve %75 / %95'e ulaştı. Tek geçişte eğitilen Transformer %67 / %57 ve
  %60 / %50'de, yakınsayana kadar eğitilen %77 / %67 ve %53 / %53'te kaldı.
- İkinci bir alan, Transformer'ın ilk alan bilgisini 0.85'ten 0.74'e düşürdü; Zihin'de 1.00 kaldı.

Bu karşılaştırma küçük modellerle ve az veriyle yapıldı; çok daha fazla veriyle fark kapanabilir.

## 11. İddia etmediklerimiz

- Bu bir bilinç ya da yapay genel zekâ iddiası değildir; bunlar bugün hiçbir makine için ölçülemiyor.
- Zihin akıcı bir dil modeli değildir; gerçek veride sohbeti büyük ölçüde yaşadıklarından ve taklitten gelir.
- Olumsuz sonuçlar silinmez:
  - öğrenmeyi öğrenmenin ilk denemesi olumsuzdu;
  - doğuştan kelime parçalarıyla büyüyen zihin belleği tüketti;
  - tek duyum kanıtı düzeltmesi sohbeti kötüleştirdi.

Hepsi notlarda ölçümleriyle durur.

## Adlandırılan buluşlar

| Ad | İngilizce | Not |
|---|---|---|
| Yaşayan Kanıt Zihni | Living Evidence Mind | mimarinin adı |
| Köken Kanıtı | Provenance Evidence | altı kökene göre W+/W− |
| Kanaat Beklentisi | Conviction-weighted Expectation | e = c(f − ½) + ½, tek kural |
| Güven Yargısı | Trust Judgment | kökene güven doğrudan deneyimle öğrenilir |
| Yedili Odak / Odak Yerleşmesi | Seven-Place Focus / Focus Settling | dikkat mekanizması |
| Kesin Unutma | Exact Retraction | defterle tek deneyimin tam geri alınması |
| Üye Okuması | Members' Reading | F11 |
| Kendi/Sonraki Yuvaları | Self/Next Slots | F12 |
| Şimdiden Öğrenme | Present-Only Learning | F13 |
| Beklenen Başarısızlık | Expected-Failure Rule | F14 |
| Sonuç İzi | Consequence Trace | F15, L5 |
| Sürdürme Kuralı | Continuation Rule | F16 |
| Benzerlik Oyu | Resemblance Vote | F18 |
| Avantaj Yarışı | Advantage Race | F19 |
| Yapısal Hayal | Structural Imagination | F20 |
| Benlik Birimi | Self Unit | F21 |
| Rüya Ayıklama | Dream Sorting | F22 |
| Durum Benzeşimi | Situation Resemblance | F23 |
| Ses Güvenilirliği | Voice Credibility | F24, varsayılan kapalı |
| Doğuş Başarısızlıkları | Born Failures | F25, incelendi, varsayılan kapalı |
| Bilmeme Hissi | Felt Not-Knowing | L7 |
| Önce Düşün | Think, then Speak | T8 sohbette |
| Kararsızlıkta Deneme | Trying in Hesitation | keşif |
| Ortak Kök Çıkarımı | Shared-Stem Inference | T8 ikinci biçim |
| Çevrimiçi Yaşam | Online Life | oturum boyunca kendi zamanında yaşar |
| Tek Köken Satırı | One-Origin Row | bellek düzeni |

## Bilgisayarınızda çalıştırmak

Model her konuşmada öğrenir, ona kibar konuşun. Söylediğiniz her şey onun yaşamına girer.

```
git clone https://github.com/myylogic/cevahir-ai
cd cevahir-ai
pip install torch numpy           # isteğe bağlı: python-docx (Word), pillow (resim)
python -m zihin                   # Linux / macOS
.\cevahir.ps1                     # Windows PowerShell
```

İlk açılışta zihin boştur ve sizinle yaşayarak öğrenir. Ona öğretmek için:

```
!oku belgelerim/                  # .txt .md .docx .pdf belgeler, .json/.tsv diyaloglar, kaynak kod
!ogret diyaloglar.tsv             # her satır: soru<TAB>cevap
!izle selam || merhaba            # bir konuşmayı izlemek
!dogru  /  !yanlis                # son cevabını nasıl karşıladığınız
!uyu    !dusun <konu>    !kendin  # rüya, iç ses, iç gözlem
```

Cevahir Türkçe verilerle büyütüldü. Başka bir dilde konuşmak istiyorsanız o dilde küçük bir veri setini (birkaç yüz
soru-cevap ya da metin) `!ogret` veya `!oku` ile ona yaşatmanız gerekir. Zihin dosyası okuduklarını içinde taşır;
özel belgelerle yetiştirdiğiniz bir zihni paylaşmayın. Ayrıntılı kılavuz: [ZIHIN_KILAVUZ.md](ZIHIN_KILAVUZ.md).
