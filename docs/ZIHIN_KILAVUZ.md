# Cevahir Zihin ile konuşmak — kılavuz

Cevahir Zihin bir sohbet botu değil, yaşarken öğrenen bir zihin çekirdeği. İki parçadan oluşur:

- **Çekirdek** (bu depodaki `zihin/` kodu): zihnin nasıl algıladığını, öğrendiğini, düşündüğünü ve konuştuğunu belirleyen
  yasalar. Herkeste aynıdır.
- **Zihin** (`.zihin` dosyası): o zihnin yaşadığı her şey — birimleri, bağları, kanıtları. Her bilgisayarda bir tane
  yaşar ve konuştukça büyür. Varsayılan yeri: `%USERPROFILE%\.cevahir\zihin\cevahir.zihin` (değiştirmek için
  `CEVAHIR_HOME` ortam değişkeni).

## Başlatmak (Windows PowerShell)

```powershell
git clone https://github.com/myylogic/cevahir-ai     # zaten varsa: cd cevahir-ai; git pull
cd cevahir-ai
pip install torch numpy python-docx pillow
.\cevahir.ps1                                        # bu bilgisayarın zihniyle konuş (ilk kez: boş bir zihin)
.\cevahir.ps1 --load cevahir.zihin.gz --kim muhammed  # size verilen bir zihni getir (yanında cevahir.zihin.json
                                                     # olsun); sonra hep ev klasöründe yaşar, bir daha --load gerekmez
.\cevahir.ps1 --his                                  # her cevaptan sonra zihnin iç durumunu göster
```

Betik engellenirse: `powershell -ExecutionPolicy Bypass -File .\cevahir.ps1`. Türkçe karakterler bozuk görünürse önce
`chcp 65001`. Başlarken "Seninle konuşan kim?" diye sorar: adınızı yazın; konuşan kişi her sözle birlikte algılanır.

## Yaşam (çevrimiçi)

- Oturum açıldığı andan itibaren zihin kendi zamanında yaşar; yalnız sorulunca çalışmaz. Yazdıklarınız bu yaşama
  girdi olarak düşer; söyledikleri kelime kelime, söyledikçe ekrana gelir. Siz yazmazken sessiz anlar yaşar: 10
  saniye sessizlikte son konunun en az bildiği kelimesi üzerine kendi içinde düşünür, 45 saniye sessizlikte
  düşüncesi yeterince gerçekse onu söyler, değilse en çok merak ettiği kelimeyi sorar; 10 dakika sessizlikte uyur
  (`--yasamsiz` ile kapatılır).
- Kararsızlık: sırası geldiğinde iki iyi desteklenmiş cevap arasında kalırsa susmaz, sonucunu en az bildiği cevabı
  dener. `!yanlis` dediğin bir cevabı bir dahaki sefer denemez, başka bir cevap dener; `!dogru` dediğini tutar.
- Her sohbetin sonunda (Ctrl+C dahil) ve uzun sohbetlerde 50 cevapta bir öğrendikleri kendiliğinden kaydedilir.
- Bilmemek: sıra kendisine geldiğinde söyleyecek bir şey bulamazsa, bir an geçer ve bilmediğini bir iç his olarak
  algılar. O anda ne denir, bunu yaşayarak öğrenir (`!kendini_tani` bu dersleri de verir): "bilmiyorum" der ve
  bilmediği kelimeyi soru tonuyla sorar ("bilmiyorum , öğrenmek isterim kuantum ?"). Açıkladığınızı öğrenir ve
  sonra kullanır.
- Önce düşünür, sonra konuşur: duyduğu sözün en az emin olduğu kelimeleri ona bildiklerini çağrıştırır (iç sesi,
  hayal olarak yaşanır ve yaşanmış kanıta göre gerçekliği ölçülür). Hemen bir cevabı yoksa, yeterince gerçek olan
  ilk düşüncesini söyler; yoksa bilmediğini hisseder.
- Hiç bütün olarak duymadığı bir söze, en çok benzeyen yaşanmış durumlarda ne söylendiyse onunla karşılık verebilir.
  Benzerlik zayıfsa (örneğin sözde bilmediği bir kelime varsa) uydurmaz.
- Cevap uzunluğu sınırı 80 kelimedir; söyleyecek bir şeyi kalmayınca kendisi durur.
- Yeni bir zihin, kelimeleri deponun kendi tokenizer altyapısındaki parçalarıyla birlikte yaşar: kök, ekler ve son
  hece (`yapıyorsun` → kök `yap`, ekler `ıyor`, `sun`). Böyle yapılmış kelimeler birbirine benzer. Parçasız
  büyümüş bir zihin parçasız devam eder (`--parca cevahir` ile açılabilir, `--parca yok` ile kapatılır).
- 10 dakika sessizlikte uyur: günün konuları ve emin olmadığı kelimeler üzerine rüya görür. Rüya çağrışımla bağlanan
  hayallerden oluşur: ortak kökten çıkarım ("kedi bir hayvandır" + "hayvanlar nefes alır" → "kedi nefes alır") ya da
  bir düşünceyi benzediği bir kavrama taşımak. Bir kelimeyle birlikte gösterdiğin resmi rüyasında yeniden "görür".
  Sonra hayallerini yaşadıklarıyla tartar; hangi hayalinin doğru, hangisinin yanlış çıktığını söyler. Yanlış çıkan
  hayal silinmez, gerçek kanıt ona ağır basar. `!uyu` ile isteyince de uyur.

## Komutlar

| komut | ne yapar |
|---|---|
| (düz yazı) | konuşma: zihin dinler, sıra ona geçince konuşur ya da susar |
| `!dogru` / `!yanlis` | zihnin son cevabının nasıl karşılandığı; o cevaba götüren yol tartılır, "yanlış" denen kopya cevap bir daha söylenmez (bilgisi silinmez) |
| `!his` | iç sinyaller: şaşkınlık, rahatlama, çatışma, yük, dürtü |
| `!yas` | gelişim profili: yaşanmış an, bildiği kelime, kurduğu kavram |
| `!kendin` | iç gözlem: kendi sesiyle ne kadar konuştuğu, kendi içinde ne kadar düşündüğü, kendini en çok neyle bağladığı |
| `!dusun [konu]` | iç sesini dinlemek: bir konu üzerine düşünür, benzeştirdiği bir kavrama aktararak hayal kurar ve ortak kökten çıkarım yapar; her hayalin gerçeklik puanı yazılır |
| `!kim <ad>` | konuşan kişiyi değiştirmek |
| `!gor <resim>` | bir resmi bir sonraki sözünüzle birlikte göstermek (örn. `!gor ben.jpg` sonra `ben muhammed`) |
| `!dinle <ses.wav>` | bir sesi bir sonraki sözünüzle birlikte duyurmak (örn. `!dinle sesim.wav` sonra `bu benim sesim`) |
| `!uyu [n]` | uyku: n rüya görür, sonra hayallerini ayıklar; daha önceki rüyalarından hangilerinin doğru ya da yanlış çıktığını ve kendi hayallerine güveninin nasıl değiştiğini söyler |
| `!arastir <konu>` | açık talimatınızla internetten okumak (Google Custom Search anahtarı tanımlıysa Google, değilse Vikipedi); okuduğu sayfalar yazılır |
| `!kendini_gelistir [n]` | merak ettiği n konuyu kendisi seçip araştırır; okuma yollarından hangisinin daha çok öğrettiğini ölçer ve ona göre seçer |
| `!oku <dosya/klasör> [en çok cümle]` | belgeleri okumak (.txt .md .docx .pdf, .json/.tsv diyaloglar; kaynak kod da satır satır bir dil gibi okunur: .py .js .ps1 …, örn. `!oku zihin` kendi kodunu okur) |
| `!izle <söz> \|\| <karşılık>` | iki kişinin konuşmasını izlemek (taklitle öğrenme) |
| `!kendini_tani [n]` | kendisi hakkındaki doğru bilgileri ona anlatmak |
| `!unut <no>` | bu oturumda söylediğiniz bir şeyi çekirdekten tam olarak geri almak (`!gecmis` numaraları gösterir) |
| `!kaydet <ad>` / `!tasinabilir <ad>` | elle kaydetmek / deftersiz küçük bir kopya |
| `!konus` | sohbet modunu açıp kapatmak (kapalıyken `?<metin>` beklentileri gösterir) |
| `!cik` | çıkmak (öğrendikleri kaydedilir) |

## Ona öğretmek

Zihin konuşmadan öğrenir, ama bir soruya nasıl karşılık verildiğini en iyi bir konuşmayı izleyerek öğrenir:

```
!izle seni kim yaptı ? || beni babam muhammed yaptı .
!izle baban kim ? || babam muhammed .
!dogru            (son cevabı beğendiysen)    !yanlis   (beğenmediysen; o kopya cevap bir daha söylenmez)
```

Bir şeyi anlattığında (`kuantum çok küçük parçacıkların bilimidir`) bunu yaşar; bir dahaki sefer o kelime üzerine
düşünürken bu bilgiyi kullanır. Kim olduğunu ona kendin anlat: başlarken adını yazdığında her sözün seninle birlikte
algılanır.

## Ne beklemeli (dürüst sınırlar)

- İlk nefesi verilen verilerdir; büyük ölçüde izlediği diyalogları taklit eder, izlemediği sorulara ilişki kurarak
  ve okuduğundan cevap verebilir, bilmediğinde susar ya da sorar. Akıcı bir dil modeli değildir.
- Kendisi hakkındaki sorulara kendisine anlatılan gerçeklerle cevap verir; ama okuduğu diyaloglarda başkalarının aynı
  soruya verdiği cevaplar daha çoksa onları da söyleyebilir (örneğin "nerede yaşıyorsun ?" sorusuna bir şehir adı).
  Benlik birimi kendi yaşamından büyür, ama söylediklerini henüz yönlendirmiyor.
- Ölçüm: belgelerinden üretilen ve hiç izlemediği 60 soruda tam doğru cevap %17, kelime örtüşmesi (F1) 0.38,
  "bilmiyorum" %23.
- Görme ve işitme ilk sürümdür: benzer resimleri ve sesleri tanır; farklı ışık ve açıdan yüz tanıma, gürültüde
  ses tanıma zihnin kendi özelliklerini öğrenmesini gerektirir. Kendi sesi (konuşma çıktısı) henüz yoktur.
- Rüyalar şimdilik yalnız kelimelerle görülür; bir hayalin doğruluğu kelimeden kelimeye bağlarla tartılır.
- Bilinç ya da genel zekâ iddiası yoktur; yetenekler testlerle ölçülür, olumsuz sonuçlar da kayıtlıdır
  (`research/living_mind_2026_11_14/RESULTS.md`, `NOTES.md`, `research/living_mind_2026_11_15/NOTES.md`).

## Gizlilik

Bir zihin dosyası okuduklarını içinde taşır; özel belgelerle yetişmiş bir zihni GitHub'a koymayın, kimseyle
paylaşmayın. `private_data/`, `*.zihin` ve defter dosyaları git tarafından yok sayılır.
