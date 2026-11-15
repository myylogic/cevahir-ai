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
.\cevahir.ps1                         # bu bilgisayarın zihniyle konuş (ilk kez: boş bir zihin)
.\cevahir.ps1 --load cevahir_tam      # size verilen bir zihni getir; sonra hep ev klasöründe yaşar
.\cevahir.ps1 --his                   # her cevaptan sonra zihnin iç durumunu göster
```

Betik engellenirse: `powershell -ExecutionPolicy Bypass -File .\cevahir.ps1`. Türkçe karakterler bozuk görünürse önce
`chcp 65001`. Başlarken "Seninle konuşan kim?" diye sorar: adınızı yazın; konuşan kişi her sözle birlikte algılanır.

## Yaşam

- Her sohbetin sonunda (Ctrl+C dahil) ve uzun sohbetlerde 50 cevapta bir öğrendikleri kendiliğinden kaydedilir.
- Siz yazmazken boş durmaz: 20 saniye sessizlikte son konunun en az bildiği kelimesi üzerine kendi içinde düşünür;
  60 saniye sessizlikte, düşüncesi yeterince gerçekse konuşmayı kendisi başlatabilir (`--yasamsiz` ile kapatılır).
- Bilmediği bir kelimeyle karşılaşıp söyleyecek bir şeyi olmadığında o kelimeyi soru tonuyla geri sorar
  ("kuantum ?"); açıkladığınızı öğrenir ve sonra kullanır.

## Komutlar

| komut | ne yapar |
|---|---|
| (düz yazı) | konuşma: zihin dinler, sıra ona geçince konuşur ya da susar |
| `!dogru` / `!yanlis` | zihnin son cevabının nasıl karşılandığı; o cevaba götüren yol tartılır, "yanlış" denen kopya cevap bir daha söylenmez (bilgisi silinmez) |
| `!his` | iç sinyaller: şaşkınlık, rahatlama, çatışma, yük, dürtü |
| `!yas` | gelişim profili: yaşanmış an, bildiği kelime, kurduğu kavram |
| `!kendin` | iç gözlem: kendi sesiyle ne kadar konuştuğu, kendi içinde ne kadar düşündüğü, kendini en çok neyle bağladığı |
| `!dusun [konu]` | iç sesini dinlemek: bir konu üzerine düşünür, benzeştirdiği bir kavrama aktararak hayal kurar; her hayalin gerçeklik puanı yazılır |
| `!kim <ad>` | konuşan kişiyi değiştirmek |
| `!gor <resim>` | bir resmi bir sonraki sözünüzle birlikte göstermek (örn. `!gor ben.jpg` sonra `ben muhammed`) |
| `!arastir <konu>` | açık talimatınızla internetten okumak (Google Custom Search anahtarı tanımlıysa Google, değilse Vikipedi); okuduğu sayfalar yazılır |
| `!kendini_gelistir [n]` | merak ettiği n konuyu kendisi seçip araştırır; okuma yollarından hangisinin daha çok öğrettiğini ölçer ve ona göre seçer |
| `!oku <dosya/klasör> [en çok cümle]` | belgeleri okumak (.txt .md .docx .pdf, .json/.tsv diyaloglar) |
| `!izle <söz> \|\| <karşılık>` | iki kişinin konuşmasını izlemek (taklitle öğrenme) |
| `!kendini_tani [n]` | kendisi hakkındaki doğru bilgileri ona anlatmak |
| `!unut <no>` | bu oturumda söylediğiniz bir şeyi çekirdekten tam olarak geri almak (`!gecmis` numaraları gösterir) |
| `!kaydet <ad>` / `!tasinabilir <ad>` | elle kaydetmek / deftersiz küçük bir kopya |
| `!konus` | sohbet modunu açıp kapatmak (kapalıyken `?<metin>` beklentileri gösterir) |
| `!cik` | çıkmak (öğrendikleri kaydedilir) |

## Ne beklemeli (dürüst sınırlar)

- İlk nefesi verilen verilerdir; büyük ölçüde izlediği diyalogları taklit eder, izlemediği sorulara ilişki kurarak
  ve okuduğundan cevap verebilir, bilmediğinde susar ya da sorar. Akıcı bir dil modeli değildir.
- Kendisi hakkındaki sorulara kendisine anlatılan gerçeklerle cevap verir; benlik birimi kendi yaşamından büyür.
- Görme ilk sürümdür: benzer resimleri tanır; farklı ışık ve açıdan yüz tanıma zihnin kendi özelliklerini
  öğrenmesini gerektirir.
- Bilinç ya da genel zekâ iddiası yoktur; yetenekler testlerle ölçülür, olumsuz sonuçlar da kayıtlıdır
  (`research/living_mind_2026_11_14/RESULTS.md`, `NOTES.md`).

## Gizlilik

Bir zihin dosyası okuduklarını içinde taşır; özel belgelerle yetişmiş bir zihni GitHub'a koymayın, kimseyle
paylaşmayın. `private_data/`, `*.zihin` ve defter dosyaları git tarafından yok sayılır.
