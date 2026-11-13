# Kök mü, ek mi? Tek bir çekimli yaşantıdan yapılan ağ-içi yazımın Türkçe hâl paradigmasında yayılımı

Muhammed Yasin Yılmaz · 13 Kasım 2026 · Cevahir araştırma dizisi, çalışma 11 (CEV-11 "Paradigma")

Araştırma raporu. Dış hakem değerlendirmesinden geçmemiştir. Ön kayıt kendi kendine beyan edilmiştir; dondurulmuş
protokolün özeti ve kod manifesti bu çalışmanın klasöründe commit edilmiştir.

<!-- RESULTS-PENDING: Özet ve Abstract doğrulayıcı sonuçlar geldikten sonra yazılacak. -->

## Özet

(Doğrulayıcı koşular tamamlanınca yazılacak.)

## Abstract

(To be written when the confirmatory runs are complete.)

## Araştırma bağlamı ve soru

Önceki 23 turluk yaşarken öğrenme programı, deneyimin gelecekteki hesaplamayı kalıcı ve denetlenebilir biçimde
değiştirmesini sembolik ve istatistiksel durumlarla, sentetik dünyalarda inceledi. Programın en sağlam sonucu, gelecek
işlemlerin ailesinin saklanması gereken durumu belirlediğiydi ([CEV-08](../living_learning_correction_2026_09_20/REPORT_TR.md),
[CEV-09](../living_learning_query_state_2026_09_21/REPORT_TR.md)): bugünün cevaplarına yeten durum yarının düzeltmelerine
yetmeyebilir. Ancak o mekanizmaların hiçbiri Cevahir'in sinir ağının içinde değildi.

Bu çalışma soruyu ağın içine taşır. Bir yaşantı — tek bir cümle — Cevahir'in kendi kod çözücüsünde (CevahirNeuralNetwork)
bir bloğun FFN çıkış izdüşümüne (fc2) kapalı biçimli bir yazım olarak eklenir. Türkçe bu yazımı sınamak için keskin bir
ortam sunar: bir özel ad en az yedi hâl biçiminde geçer (Ayten, Ayten'in, Ayten'e, Ayten'i, Ayten'de, Ayten'den,
Ayten'le) ve gündelik yazımda kesme işareti çoğu zaman düşer (Aytenden). Soru şudur:

> Bir varlık tek bir çekimli biçimde "yaşandığında", ağ-içi yazım (a) varlığın diğer hâl biçimlerine, (b) aynı biçimlerin
> kesme işaretsiz yazımına, (c) bir sol bağlamın ardından ve (d) adın son kök tokenini paylaşan başka varlıklara ne
> kadar yayılır? Kesme işaretsiz yazımdaki kaybı, sorguda yazılan anahtarla birebir aynı anahtarın bulunup bulunmaması
> (tokenleştirmeden önceden belirlenen bir değişken) açıklar mı?

Bu çalışma yeni bir öğrenme algoritması önermez. Yazım cebiri bilinen konum-belirle-ve-düzenle ailesindendir (ROME,
MEMIT); kesinlik özellikleri (defter ile tam geri alma, sıradan bağımsızlık) yapı gereği doğrudur ve hipotez değil
doğrulama testi olarak raporlanır. Katkı dar ve ampirik-mekanistiktir: Türkçe hâl paradigması, yazım dizisi ve
tokenleştirme açısından ağ-içi yazımın genelliği ve özgüllüğü.

## Ön kayıt ve tasarım süreci

Tasarım, yapay zekâ destekli bir süreçle bağımsız rollerden geçirildi. Sekiz analiz ajanı repoyu ve literatürü taradı.
Farklı mercekli üç stratejist (araştırma atılımı, sağlam temel, hesap gerçekçiliği) ve proje yürütücüsünün kendi adayı,
her biri bir yenilik eleştirmeni (web taraması) ve bir yöntem eleştirmeni tarafından çürütülmeye çalışıldı; bir hakem
protokolü seçti. Eleştiriden sağ çıkmayan iddialar açıkça bırakıldı: kök tokenden adresleme yeni bir ilke değildir
(özne aralığı yalın ad olarak işaretlendiğinde ROME'un varsayılan "öznenin son tokeni" kuralı zaten köke düşer);
defter cebiri RLSEdit, ACU ve Cao & Yang'ın toplam-biçimli tam unutma sonucunun kapsamındadır; varlık-ortak değerler
MEMIT-Merge'dür. Protokol, ilk doğrulayıcı ön eğitimden önce
[PROTOCOL.md](../../../research/living_learning_paradigm_2026_11_13/PROTOCOL.md) olarak donduruldu ve 32 kod/test
dosyasının SHA-256 değerleri [freeze_manifest.sha256](../../../research/living_learning_paradigm_2026_11_13/freeze_manifest.sha256)
ile sabitlendi. Dondurmadan önceki tasarım düzeltmeleri gerekçeleriyle protokolün başında kayıtlıdır (bkz. Yöntem).

Literatür taraması yalnız arama motoru özetleriyle yapılabildi (çalışma ortamından arXiv ve ACL Anthology tam metinlerine
erişilemedi). "Bildiğimiz kadarıyla ilk" türü ifadeler bu nedenle yayından önce elle doğrulanmalıdır.

## Yöntem

### TR-Paradigma dünyası

Her dünya tohumu için 400 uydurma, tek sözcüklü, Türkçe ses düzenine uyan ad üretilir; adların yarısı 16 ortak sonekten
biriyle biter ve eğitim düzyazısındaki hiçbir sözcükle çakışmaz. Üç ilişki vardır: A memleket (48 uydurma kasaba),
B meslek (30) ve C en sevdiği yemek (30); nesneler düzgün dağılımla atanır. Yedi özne-başı kalıp ilişki ad öbeğini
nesneden önce koyar, nesne son sözcüktür:

| Hâl | Kalıp |
|---|---|
| yalın | «Ayten hakkında bilinen: memleketi Karaköy.» |
| ilgi | «Ayten'in memleketi Karaköy.» |
| yönelme | «Ayten'e dair bilinen: memleketi Karaköy.» |
| belirtme | «Ayten'i tanıyan herkes bilir: memleketi Karaköy.» |
| bulunma | «Ayten'de duran kimlik kaydına göre memleketi Karaköy.» |
| ayrılma | «Ayten'den bahsedenler bilir: memleketi Karaköy.» |
| vasıta | «Ayten'le ilgili bilgi: memleketi Karaköy.» |

Hâl ekleri iki ve dört yönlü ünlü uyumu, kaynaştırma harfleri ve -DA/-DAn benzeşmesiyle kurala dayalı üretilir; 120
satırlık el kontrollü tablo ve bağımsız bir yeniden uygulama ile denetlenmiştir (V9). Gündelik yazım aynı dizgenin kesme
işaretsiz hâlidir. Sol bağlamlar «Bilindiği gibi, » ve «Dün öğrendik ki »dir. Varlıklar 60 P3 (iki kalıbı hiç
eğitilmeyen), 60 C0 havuzu ve 280 yoklama havuzu olarak ayrılır.

### Tokenleştirme

Repodaki kayıpsız bayt düzeyinde BPE (tokenizer_management/byte_bpe.py, 4.096 kimlik) değiştirilmeden, her dünyada
yalnız düzyazıyla eğitilir; uydurma adlar tokenizer için görülmemiştir. Kesme işareti ve ardındaki harfler tek ön parça
olduğundan biçimsel yazımda kökün sonunda her zaman bir token sınırı vardır. Gündelik biçimler önceden sınıflanır:
IDENT (kök tokenleri biçimsel köke birebir eşit), FUSED (bir token kök–ek sınırını aşar). BPE'nin birleştirme sırası
nedeniyle üçüncü sınıf (RESEG) bu tokenleştiricide oluşamaz; uygulama sırasında kanıtlanıp test edildi.

### Model ve ön eğitim

CevahirNeuralNetwork, repodaki yapılandırma yoluyla (tiny_model_config + ModelManager) kurulur: D=192, 6 katman,
4 baş, SwiGLU (512), RMSNorm, RoPE, bağlı gömme, dropout 0; 3.445.824 parametre. Ön eğitim deterministiktir (AdamW
2e-3, ısınma + kosinüs, 64 dizilik yığın, tek iş parçacığı) ve K=20 geçişten oluşur; her geçişte her (varlık, ilişki,
hâl) hücresi bir kez, bağlam ve yazım rastgele çekilerek görülür; düzyazı tokenlerin %20'sidir. Dondurmadan önce bir
tasarım düzeltmesi yapıldı: dünya cümlelerinin %20'sinin önüne eğitim düzyazısından bir giriş cümlesi eklenir, çünkü
pilotta ROME standardının düzyazı-önekli anahtar ortalaması modeli dağılımı dışında okuyordu.

Ağırlık başlatma ve gömme dropout'u, bu çalışmadan önce aynı oturumda düzeltilen hâlleriyle kullanılır (artık izdüşüm
başlatması GPT-2 kuralı 0,02/√(2L); çift gömme dropout'u kaldırıldı).

### Yazım, kollar ve seçim

Yazım tek bir bloğun fc2 matrisine eklenen sıra-bir bir terimdir: M = (v* − W k̄)(C0⁻¹k̄)ᵀ / (k̄ᵀC0⁻¹k̄). C0, düzenleme
ve yoklama varlıklarından ayrık bir havuzdan ve düzyazıdan tahmin edilir. v*, hedef nesnenin olasılığını en yükseğe
çıkaran ve aynı varlığın B ilişkisindeki çıktıyı KL terimiyle koruyan bir δ aramasıyla bulunur. Kollar:

- STD — ROME/MEMIT standardı: yalın ad aralığının son tokeninde, yaşanan cümle ve 10 düzyazı-önekli kopyası üzerinden
  ortalanmış anahtar (biçimsel metinde kök-son token).
- ROOT — aynı konumda, ortalamasız anahtar.
- LAST — adı içeren sözcüğün son tokeni (biçimsel metinde ek tokeni); bir "yüzey aralığı" ardışık düzeni.
- PARA-ORTH — STD anahtarı 7 biçimsel ve 6 gündelik çekimli biçim üzerinden de ortalanır; sınanan biçim ortalamadan
  dışarıda bırakılır.
- FT-L / FT-ALL — aynı amaçla fc2'nin veya bütün parametrelerin ince ayarı.
- NORM — STD yazımı; gündelik yoklamalarda kesme işareti, sözlükteki adlar için geri getirilir.

Her kolun katmanı (0–4) ve adım/sınır yapılandırması yalnız keşif dünyalarında (0 ve 1), önceden yazılmış bir kuralla
seçilir: yazım etkinliğini en yükselt; koşul rastgele varlıklara sızıntı ≤ %1 ve aynı varlığın C ilişkisine sızıntı ≤ %5.

### Yoklamalar, ölçüler ve hipotezler

Her düzenleme temel modele tek başına uygulanır ve ölçümden sonra silinir. Başarı, yeni nesnenin açgözlü çözümlemeyle
tam üretilmesidir. Yoklama aileleri: yaşanan kalıp (kurulum), diğer altı biçimsel hâl, hedef hâller (yaşanan hâl dışında,
temel modelin eski nesneyi iki yazımda da ürettiği çekimli hâller) biçimsel ve gündelik yazımda, iki sol bağlam × yedi hâl,
son kök tokenini paylaşan komşular (RN), rastgele varlıklar (RD), aynı ek tokenini taşıyanlar ve aynı varlığın B/C
ilişkileri. Sızıntı, temel modelin kendi açgözlü cevabına göre değişen cevap oranıdır.

Birincil hipotezler (Holm, m=3, tek yönlü α=0,05; birim: bağımsız ön eğitilmiş temel model, n=8 doğrulayıcı dünya):

- H1: STD'de biçimsel hedef hâllere genelleme, gündelik yazıma genellemeden en az 15 puan yüksektir.
- H2: Bu kayıp, gündelik biçimin kök anahtarını koruyup korumadığıyla (IDENT/FUSED) açıklanır; NONIDENT biçimlerdeki
  kayıp IDENT biçimlerdekinden en az 30 puan büyüktür.
- H3: STD yazımı, adın son kök tokenini paylaşan varlıklara rastgele varlıklardan en az 5 puan daha fazla sızar.

İkincil hipotezler (Holm, m=4): S1 biçim-ortalamalı anahtarlar görülmemiş biçimlere genelleşir; S2 önek ortalaması sol
bağlama dayanıklılık kazandırır; S3 bir yaşam akışında (100 işlem: yazım, ikinci ilişki, düzeltme, geri alma) varlık-ortak
defter bağımsız farklardan iyidir; S4 varlık kimliği derinlikle kök-son tokeninde toplanır.

Karar kuralı: margin kaydırılmış etkiler üzerinde kesin işaret-çevirme permütasyon testi ve Holm; desteklendi = Holm
reddi; çürütüldü = ortalama etkinin iki yönlü %95 t-aralığının üst sınırı marjın altında; aksi hâlde belirsiz.

## Bulgular

(Doğrulayıcı koşular tamamlanınca yazılacak.)

## Doğrulama testleri

(Sonuçlarla birlikte yazılacak.)

## Sınırlar

(Sonuçlarla birlikte yazılacak.)

## Yeniden üretim

(Sonuçlarla birlikte yazılacak.)
