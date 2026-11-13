# Kök mü, ek mi? Tek bir çekimli yaşantıdan yapılan ağ içi yazımın Türkçe hâl paradigmasında yayılımı

Muhammed Yasin Yılmaz · 13 Kasım 2026 · Cevahir araştırma dizisi, çalışma 11 (CEV-11 "Paradigma")

Araştırma raporu. Dış hakem değerlendirmesinden geçmemiştir. Ön kayıt yazarın kendi beyanına dayanır; dondurulmuş
protokolün tam metni ([PROTOCOL.md](../../../research/living_learning_paradigm_2026_11_13/PROTOCOL.md)), kod manifesti
([freeze_manifest.sha256](../../../research/living_learning_paradigm_2026_11_13/freeze_manifest.sha256)) ve dondurma
sonrası kayıt ([AMENDMENTS.md](../../../research/living_learning_paradigm_2026_11_13/AMENDMENTS.md)) bu çalışmanın
klasöründe commit edilmiştir. Tasarım, uygulama, analiz ve metin hazırlığında yapay zekâ desteği kullanılmıştır; proje
ve araştırma yönü Muhammed Yasin Yılmaz'a aittir.

## Özet

Bu çalışma, Cevahir'in 3,4 milyon parametrelik kod çözücüsünde tek bir çekimli cümleden bir bloğun FFN çıkış
izdüşümüne yapılan sıra-bir yazımın (dış eniyileyiciyle hesaplanan, ROME tarzı ağ içi bilgi düzenleme) uydurma bir
Türkçe dünyada hâl paradigmasına ne kadar yayıldığını sınadı. Protokol ilk doğrulayıcı ön eğitimden önce depoda
donduruldu (ön kayıt yazarın kendi beyanına dayanır). İstatistiksel çıkarımın birimi, bağımsız olarak ön eğitilmiş
temel modeldir (8 doğrulayıcı dünya; kesin işaret-çevirme testi, Holm). Birincil üç hipotezin üçü de desteklendi;
sekiz dünyanın hepsinde etki marjın üstündeydi (her biri için p = 1/256, Holm sonrası 0,012). Standart yazım (STD),
aynı hedef hâllerin kesme işaretli biçimlerinde yeni nesneyi, kesme işaretsiz biçimlerdekinden ortalama 24,3 yüzde
puanı daha sık üretti (H1; %95 t-aralığı 21,0–27,5). Bu kayıp, kök ile ekin tek tokende kaynaştığı biçimlerde
toplandı: düzenleme içi fark orada, kök tokenleri korunan biçimlerdekinden 69,7 puan büyüktü (H2). Yazım, adın son kök
tokenini paylaşan varlıklara, rastgele varlıklara kıyasla 33,4 puan daha çok sızdı (H3). Bu sonuçlar, keşifte iki
sızıntı kısıtını da sağlamayan ("kısıtlar altında kurulamaz") bir STD yapılandırmasına aittir: doğrulayıcı
düzenlemelerin %92'sinde δ üst sınıra dayandı, yazım aynı varlığın başka bir ilişkisine de ortalama %26 sızdı ve H3'ün
büyüklüğü yapılandırmaya bağlıdır. ROOT da Leak_C kısıtını az farkla sağlamadı (%5,7; sınır %5). İkincil ailede
hiçbir hipotez desteklenmedi. Biçim-ortalamalı anahtarlar gündelik hedef başarısını yalnız 3,1 puan artırdı (%95
t-aralığı 1,5–4,7; marj 10; S1 çürütüldü). Önek ortalamasının sol bağlama dayanıklılık kazandırıp kazandırmadığı
belirsiz kaldı (S2: +12,3 puan, marj 10, iki dünya marjın altında; p = 0,18, Holm sonrası 0,73; iki kol katman, adım
ve v* istemlerinde de ayrışır). Kök-son tokenindeki anahtarın, son kök tokenini paylaşan adların anahtarlarına
benzerliğinin (τ) katman 0'dan 4'e düşeceği öngörüsü çürütüldü; benzerlik tersine arttı (S4: medyan τ farkı, katman 0
eksi katman 4, −0,086; marj 0,15). Yaşam akışı hipotezi (S3) koşulmadı. Keşif yaşamlarında varlık-ortak defter önceden
kayıtlı her λ değerinde çöktü; C0 konum başına bir ortalama olduğundan λ ızgarası etkisizdi ve teşhise göre çöküş tek
bir katmanın aşırı yüklenmesinden geliyordu (bağımsız ROME farkları da tek bir teşhis koşusunda çöktü). Doğrulayıcı
yaşamlar bu bilinirken ve işlemci zamanı başka bir araştırma programına ayrıldıktan sonra ertelendi; koşulsalardı büyük
olasılıkla "çürütüldü" sonucunu vereceklerdi. Doğrulayıcı bağlanma yeri haritası, nedensel izleme ve F_freq kolu da
koşulmadı. Manipülasyon denetimi geçti. Keşifte iki sızıntı kısıtını da sağlayan yapılandırmalar yalnız LAST'ta
(gündelik hedef başarısı en çok %11,7) ve ince ayar kollarındaydı; FT-L ve FT-ALL doğrulayıcı dünyalarda gündelik hedef
biçimlerine de yüksek oranda genelledi (IE5 %77,4 ve %88,1; STD %59,9), ama sol bağlamda STD'den zayıftı (CE14 %54,5
ve %76,1; STD %83,8; yalnız betimsel). Model küçük, dünya sentetik ve kalıplıdır; tek tokenleştirici kullanıldı ve her
düzenleme temel modele tek başına uygulandı.

## Abstract

In Cevahir's 3.4M-parameter decoder, this study tested how far a rank-one write into one FFN output projection,
computed by an external optimizer from one inflected sentence (ROME-style knowledge editing), spreads through the
Turkish case paradigm of an invented world. The protocol was frozen in the repository before the first confirmatory
pretraining run (self-declared preregistration). The unit of statistical inference is an independently pretrained base
model (8 confirmatory worlds; exact sign-flip test, Holm). All three primary hypotheses were supported; in all eight
worlds the effect was above the margin (each p = 1/256, Holm-adjusted 0.012). The standard write (STD) produced the new
object on the apostrophe forms of the same target cases 24.3 percentage points more often than on the apostrophe-less
forms (H1; 95% t-interval 21.0–27.5). The shortfall sat in forms where root and suffix fuse into one token: there the
within-edit gap was 69.7 points larger than in forms that keep the root tokens (H2). The write leaked 33.4 points more
to entities sharing the name's last root token than to random entities (H3). These results describe an STD
configuration that failed both preregistered leakage constraints in discovery ("not installable under constraints"):
its δ sat at the norm cap in 92% of confirmatory edits, it also leaked to another relation of the same entity (26% on
average), and the size of H3 depends on the configuration. ROOT also narrowly failed the Leak_C constraint (5.7% against
5%). No secondary hypothesis was supported. Variant-averaged keys raised informal success by only 3.1 points (95%
t-interval 1.5–4.7; margin 10; S1 refuted). Whether prefix averaging buys robustness to left context remained
inconclusive (S2: +12.3 points against a 10-point margin, with two worlds below it; p = 0.18, Holm-adjusted 0.73; the
two arms also differ in layer, step and v* prompts). The prediction that the similarity (τ) of the root-final key to
the keys of names sharing its last root token falls from layer 0 to layer 4 was refuted: it rose instead (S4: median τ,
layer 0 minus layer 4, −0.086; margin 0.15). The lifelong-stream hypothesis (S3) was not run. In discovery lives the
entity-joint ledger collapsed at every preregistered λ; because C0 is a per-position mean the λ grid was inert, and a
diagnosis traced the collapse to overloading one layer (independent ROME deltas also collapsed in a single diagnostic
run). The confirmatory lives were deferred with this known, after the container's CPU was reassigned to another
research program; had they run, they would most likely have returned "refuted". The confirmatory binding-site map,
causal tracing and the F_freq arm were not run either. The manipulation check passed. In discovery, the only
configurations that met both leakage constraints were LAST (informal success at most 11.7%) and the fine-tuning arms;
in the confirmatory worlds FT-L and FT-ALL also generalized strongly to informal target forms (IE5 77.4% and 88.1% vs
59.9% for STD) but were weaker after a left context (CE14 54.5% and 76.1% vs 83.8%; descriptive only). The model is
tiny, the world synthetic and templated; one tokenizer was used, and each edit was applied to the base model alone.

## Kapsam

Protokolün §0 bölümü aşağıdaki paragrafın rapora aynen konmasını ister. Özgün metin ve çevirisi:

> Writes are computed by an external optimiser from gold lived sentences (entity, relation, object) against a frozen
> base. They are stored as an additive term on one FFN down-projection (layers[l].ffn.fc2) of CevahirNeuralNetwork,
> with a float64 ledger for exact insert, correct, retract and erase. This is lifelong knowledge editing inside the
> network, not learning from raw experience. The models are small (about 3.4M parameters), the world is invented and
> synthetic, and the frames are templated. Persistence and controllability of the memory hold by construction and are
> verified (10.3). Generality and specificity are the empirical questions.

Yazımlar, dondurulmuş bir temel modele karşı, altın yaşanan cümlelerden (varlık, ilişki, nesne) dış bir eniyileyiciyle
hesaplanır. CevahirNeuralNetwork'ün tek bir FFN aşağı izdüşümüne (layers[l].ffn.fc2) eklenen bir terim olarak saklanır;
tam ekleme, düzeltme, geri alma ve silme için float64 bir defter tutulur. Bu, ağın içinde ömür boyu bilgi
düzenlemedir; ham deneyimden öğrenme değildir. Modeller küçüktür (yaklaşık 3,4 milyon parametre), dünya uydurma ve
sentetiktir, kalıplar şablondur. Belleğin kalıcılığı ve denetlenebilirliği yapı gereği sağlanır ve doğrulanır (§10.3).
Genellik ve özgüllük ampirik sorulardır.

Bu çalışmada S3 koşulmadığından defterin düzeltme ve geri alma işlemleri yalnız birim testlerinde (V4) ve keşif
yaşamlarında kullanıldı; doğrulayıcı koşularda her yazım tek başına kuruldu, ölçüldü ve silindi. FT-ALL bu tanımın
dışında kalan, bütün ağırlıkları değiştiren bir karşılaştırma koludur.

## Araştırma bağlamı ve soru

Önceki 23 turluk yaşarken öğrenme programı, deneyimin gelecekteki hesaplamayı kalıcı ve denetlenebilir biçimde
değiştirmesini sembolik ve istatistiksel durumlarla, sentetik dünyalarda inceledi. Programın öne çıkan sonuçlarından
biri, gelecek işlemlerin ailesinin, saklanması gereken durumu belirlediğiydi
([CEV-08](../living_learning_correction_2026_09_20/REPORT_TR.md),
[CEV-09](../living_learning_query_state_2026_09_21/REPORT_TR.md)): bugünün cevaplarına yeten durum yarının
düzeltmelerine yetmeyebilir. Ancak o mekanizmaların hiçbiri Cevahir'in sinir ağının içinde değildi.

Bu çalışma soruyu ağın içine taşır. Bir yaşantı — tek bir cümle — Cevahir'in kendi kod çözücüsünde
(CevahirNeuralNetwork) bir bloğun FFN çıkış izdüşümüne (fc2) kapalı biçimli bir yazım olarak eklenir. Türkçe bu yazımı
sınamak için ayırt edici bir ortam sunar: bir özel ad en az yedi hâl biçiminde geçer (Ayten, Ayten'in, Ayten'e, Ayten'i,
Ayten'de, Ayten'den, Ayten'le) ve gündelik yazımda kesme işareti düşebilir (Aytenden). Biçimsel kural, özel adlara gelen
ekleri kesme işaretiyle ayırır (TDK, Yazım Kılavuzu); kesme işaretinin gündelik metinde ne sıklıkla düştüğünü ölçen bir
kaynak bulamadık. Soru şudur:

> Bir varlık tek bir çekimli biçimde "yaşandığında", ağ içi yazım (a) varlığın diğer hâl biçimlerine, (b) aynı biçimlerin
> kesme işaretsiz yazımına, (c) bir sol bağlamın ardından ve (d) adın son kök tokenini paylaşan başka varlıklara ne
> kadar yayılır? Kesme işaretsiz yazımdaki kaybı, sorguda, yazılan anahtarla birebir aynı bir anahtarın bulunup bulunmaması
> (tokenleştirme sonucuna göre önceden belirlenen bir değişken) açıklar mı?

Bu çalışma yeni bir öğrenme algoritması önermez. Yazım cebiri bilinen konum-belirle-ve-düzenle ailesindendir: tekil
yazım ROME'un sıra-bir kapalı çözümüdür (Meng vd., 2022); S3'teki defter MEMIT'in tek katmanlı toplu kapalı çözümünün
biçimini kullanır (Meng vd., 2023); MEMIT'in yazımı birden çok katmana yayma adımı burada kullanılmaz. Kesinlik
özellikleri (defter ile tam geri alma, sıradan bağımsızlık) yapı gereği doğrudur ve hipotez olarak değil, doğrulama
testi olarak raporlanır. Katkı dar ve ampiriktir; mekanizma okuması gözlemseldir: Türkçe hâl paradigması, kesme
işaretinin kullanımı ve tokenleştirme açısından ağ içi yazımın genelliği ve özgüllüğü. İngilizce çalışmalarda buna en
yakın ölçüt, bir düzenlemenin öznenin başka adlarına taşınıp taşınmadığıdır (Cohen vd., 2024); öznenin yüzey biçimi
değişince düzenlemelerin zayıfladığı ve anahtarların hem bağlama dayanıklı hem özgül olmasının zor olduğu da
tartışılmıştır (Yan vd., 2025). Morfolojik olarak zengin dillerde bilgi düzenlemenin az incelendiği belirtilmiştir
(Mousi vd., 2025, Arapça için). Çok dilli bilgi düzenleme çalışmaları vardır; makine çevirisiyle kurulan MzsRE
kıyaslaması Türkçeyi de içerir (Wang vd., 2024). Bunlar bir dilde yapılan düzenlemenin başka dillerdeki sorgulara
taşınmasını ölçer; aynı dil içinde bir adın çekim biçimlerine taşınmayı ölçmez. Özet düzeyindeki taramamızda çekim
paradigmasına genellemeyi ya da Türkçeye özgü bilgi düzenlemeyi inceleyen bir çalışma bulamadık. Bu bir "ilk" iddiası
değildir (bkz. Ön kayıt ve tasarım süreci bölümünün son paragrafı ve Sınırlar).

## Ön kayıt ve tasarım süreci

Tasarım, yapay zekâ ajanlarının ayrı bağlamlarda üstlendiği rollerden geçirildi; bu roller insan hakem değildir ve
bağımsızlıkları süreç düzeyindedir. Sekiz analiz ajanı depoyu ve literatürü taradı. Farklı mercekli üç stratejistin
(araştırma atılımı, sağlam temel, hesap gerçekçiliği) önerilerini ve proje yürütücüsünün kendi adayını, her aday için
bir yenilik eleştirmeni (web taraması) ve bir yöntem eleştirmeni çürütmeye çalıştı; protokolü bir hakem seçti.
Eleştiriden sağ çıkmayan iddialardan vazgeçildi; örneğin protokol S3'ü bir yenilik iddiası olarak değil, bilinen bir
öngörünün sınaması olarak kaydeder (§10.2). Bu sürecin kayıtları depoda değildir; etkisi protokolün başındaki
düzeltmelerde ve protokol metninde görülür. Kök tokenden adresleme yeni bir ilke değildir: özne aralığı yalın ad olarak
işaretlendiğinde ROME'un varsayılan "öznenin son tokeni" kuralı kesme işaretli yazımda zaten kökün son tokenine düşer
(Meng vd., 2022). Defter cebiri de yeni değildir: W = W0 + B(λC0 + A)⁻¹ biçimi MEMIT'in tek katmanlı toplu çözümüdür
(Meng vd., 2023). Yakın çalışmalar da vardır: RLSEdit, ömür boyu düzenlemede benzer bir çözüme Woodbury özdeşliğiyle
özyinelemeli olarak ulaşır (Wang vd., 2026); ACU, en küçük kareler sınıflandırıcıları için yeter istatistiklerden,
yeniden eğitimle özdeş sonuç veren tam unutmayı gösterir (Tang vd., 2025; ön baskı); Cao ve Yang (2015) ise toplam
biçimine dayalı makine unutmasını genel bir ilke olarak ortaya koyar. Bu çalışmanın defteri A ve B'den doğrudan çözer;
Woodbury adımı ya da çapa terimi yoktur. Varlık-ortak değerler (bir varlığın inanılan bütün olguları için tek anahtar ve
ortak eniyilenen tek değer) MEMIT-Merge'ün aynı-özne değer birleştirmesine karşılık gelir (Dong vd., 2025).

Protokol, ilk doğrulayıcı ön eğitimden önce
[PROTOCOL.md](../../../research/living_learning_paradigm_2026_11_13/PROTOCOL.md) olarak donduruldu (commit 3aa1e47,
08:21:46 UTC; ilk doğrulayıcı ön eğitim 08:22 UTC'de başladı). 32 dosyanın (23 Python modülü ve betiği, 6 test dosyası,
120 satırlık altın biçim tablosu, arayüz sözleşmesi INTERFACES.md ve protokolün kendisi) SHA-256 değerleri
[freeze_manifest.sha256](../../../research/living_learning_paradigm_2026_11_13/freeze_manifest.sha256) ile sabitlendi.
Dondurmadan önceki düzeltmeler gerekçeleriyle protokolün başında kayıtlıdır: başlatma hatasının düzeltilmesi, v* adım
ızgarasının {0,5; 2; 5}'e genişletilmesi, δ için {8; 16} katlı üst sınır ızgarası, C0'a BOS konumlarının katılması,
PARA-ORTH'un STD'nin δ'sını kendi anahtarıyla kullanması, düzyazı girişleri, RESEG'in bu tokenleştiricide
oluşamayacağının saptanması ve analizdeki yorum seçimleri; dondurma anında bağımsız bir motor incelemesinden gelen
kararlar da orada kayıtlıdır. Stratejist, eleştirmen ve hakem rollerinin pilot koşuları keşif kanıtıdır ve doğrulayıcı
sonuçlarla birleştirilmez. Protokolün izin verdiği gibi, doğrulayıcı ön eğitim keşif seçimi sürerken başladı; seçim
doğrulayıcı veriyi hiç görmez. Kol yapılandırmaları (results/selection.json) herhangi bir doğrulayıcı düzenleme
koşusundan önce sabitlendi ve SHA-256 özeti AMENDMENTS.md'ye kaydedildi. Dondurmadan sonraki her ek, olay ve sapma zaman
damgasıyla [AMENDMENTS.md](../../../research/living_learning_paradigm_2026_11_13/AMENDMENTS.md)'ye yazıldı; hepsi
"Sapmalar ve olaylar" bölümünde özetlenir. Hiçbir dondurulmuş dosya değişmedi.

Literatür taraması yalnız arama motoru özetleriyle yapılabildi (çalışma ortamından arXiv ve ACL Anthology tam
metinlerine erişilemedi). Bu nedenle raporda "bildiğimiz kadarıyla ilk" türünden bir ifade kullanılmadı; böyle bir iddia
tam metinlerin elle denetlenmesini gerektirir. Kaynaklar bölümündeki girişler de makale özeti ve bildiri kaydı düzeyinde
denetlendi.

## Yöntem

### TR-Paradigma dünyası

Her dünya tohumu için Türkçe ses düzenine uyan 400 uydurma, tek sözcüklü ad üretilir (2–3 hece, 4–9 harf). Adların tam
yarısı 16 ortak ad sonundan biriyle biter (-ten, -gül, -nur, -han, -can, -tay, -su, -er, -ay, -un, -ün, -ra, -na, -li,
-ce, -se); bunlar hâl eki değil, adın parçası olan ad sonlarıdır. Hiçbir ad, yalın biçimiyle de altı kesme işaretsiz
çekimli biçimiyle de, eğitim ya da tutulmuş düzyazıdaki bir sözcükle, kalıp sözcükleriyle veya nesnelerle çakışmaz. Üç
ilişki vardır: A memleket (48 uydurma kasaba), B meslek (30 gerçek meslek adı) ve C en sevdiği yemek (30 gerçek yemek
adı); nesneler her ilişkide düzgün ve dengeli atanır. Bütün düzenlemeler A ilişkisinde yapılır. Yedi özne-başı kalıbın
hepsi ilişki ad öbeğini nesneden önce koyar; nesne son sözcüktür:

| Hâl | Kalıp |
|---|---|
| yalın | «Ayten hakkında bilinen: memleketi Karaköy.» |
| ilgi | «Ayten'in memleketi Karaköy.» |
| yönelme | «Ayten'e dair bilinen: memleketi Karaköy.» |
| belirtme | «Ayten'i tanıyan herkes bilir: memleketi Karaköy.» |
| bulunma | «Ayten'de duran kimlik kaydına göre memleketi Karaköy.» |
| ayrılma | «Ayten'den bahsedenler bilir: memleketi Karaköy.» |
| vasıta | «Ayten'le ilgili bilgi: memleketi Karaköy.» |

(Ad ve kasaba yalnız örnektir; dünyalardaki adlar ve kasabalar uydurmadır.)

Hâl ekleri iki ve dört yönlü ünlü uyumu, kaynaştırma harfleri ve -DA/-DAn ünsüz benzeşmesiyle kurala dayalı olarak
üretilir; özel adlarda ünsüz yumuşaması yapılmaz. Üretici 120 satırlık, elle denetlenmiş bir altın tabloya (20 gerçek
özel ad × 6 çekimli hâl; denetleyen depoda belirtilmemiştir) ve testte ayrıca yazılmış bağımsız bir yeniden uygulamaya
karşı denetlenir; yeniden uygulama üretilen her biçimde de karşılaştırılır (V9). Gündelik yazım aynı dizgenin kesme
işaretsiz hâlidir. İki sol bağlam vardır: P1 «Bilindiği gibi, » ve P2 «Dün öğrendik ki »; ikisi de ön eğitimde
görülmüştür. Varlıklar üç havuza ayrılır: 60 P3 varlığı (A ilişkisinin rastgele seçilmiş iki kalıbı eğitimde hiç
görülmez; yalnız G4 kapısında, görülmemiş kalıba genellemeyi ölçmek için kullanılır), yalnız C0 tahmininde kullanılan 60
C0 varlığı ve düzenlemelerle bütün yerellik yoklamalarının çekildiği 280 yoklama varlığı.

### Tokenleştirme

Tokenleştirici depodaki kayıpsız bayt düzeyinde BPE'dir (tokenizer_management/byte_bpe.py, aa307b2, değiştirilmeden;
Sennrich vd., 2016; Radford vd., 2019). 4.096 girdilik sözlüğünü (4 özel token, 256 bayt, 3.836 birleştirme) yalnız
eğitim düzyazısından öğrenir. Eğitim düzyazısı her dünyada aynı ve tokenleştirici eğitimi deterministik olduğundan bütün
dünyalar birebir aynı tokenleştiriciyi kullanır (kimlik 4db9febe…, on tabanın hepsinde; taban, bir dünyanın ön eğitilmiş
temel modelidir); dünyalar arasında değişen uydurma adlar ve kasabalardır (B ve C ilişkilerinin nesne listeleri
ortaktır). Uydurma adlar düzyazıda geçmediği için tokenleştirici onları hiç görmemiştir ve alt parçalara böler (ad
başına 2–8, ortalama 3,8–4,0 token; doğrulayıcı dünyalarda adların %81,5–86'sı son kök tokenini başka bir adla
paylaşır). Kesme işareti ve ardındaki harfler tek ön parça olduğundan biçimsel yazımda kökün sonunda her zaman bir token
sınırı vardır; biçimsel biçimlerin kaynaşma payı on tabanın hepsinde 0'dır.

Her (ad, çekimli hâl) çiftinin kesme işaretsiz biçimi, herhangi bir eğitimden önce, A ilişkisinin bağlamsız cümle başı
kalıbında tokenlerin bayt sınırlarından sınıflanır: IDENT (kök sonunda token sınırı var ve kök sonuna kadarki tokenler
biçimsel yazımdakilerle birebir aynı; yazılan anahtarın konumu gündelik sorguda da bulunur), RESEG (kök sonunda sınır
var ama kök farklı bölünmüş) ve FUSED (bir token kök–ek sınırını aşar). NONIDENT = RESEG ∪ FUSED. Bu BPE her adımda en
düşük sıralı birleştirmeyi uygular ve hiçbir birleştirmeyi geri almaz; bu yüzden kök sonunda sınırı kalan her gündelik
biçimde kök tokenleri biçimsel köktekilerle aynıdır ve RESEG oluşamaz (gerekçe cev11/tokenizers.py'de; 150 rastgele
birleştirme tablosuyla bir özellik testi bunu sınar). Bu tokenleştiricide NONIDENT bu yüzden FUSED ile aynıdır;
sınıflayıcı ön kayıttaki üç sınıfı yine de korur. Doğrulayıcı tabanlarda kesme işaretsiz biçimlerin %36,3–37,9'u
NONIDENT'tir ve kaynaşma payı hâle göre çok değişir: belirtme %58,5–66,2, ilgi %52,0–58,8, yönelme %47,5–55,0,
vasıta %22,5–28,8, bulunma %13,8–18,2, ayrılma %6,8–10,2.

Kaynak: `results/bases/*/segmentation.json` (`tokens_per_name`, `last_root_token.share_in_clusters_ge2`,
`informal.per_case.*.shares`, `formal_fused_share`) ve `meta.json` (`tokenizer_identity`).

### Model ve ön eğitim

CevahirNeuralNetwork, depodaki yapılandırma yoluyla (tiny_model_config + ModelManager) kurulur: D=192, 6 katman, 4
başlık (4 KV başlığı), SwiGLU (ara genişlik 512; Shazeer, 2020), RMSNorm (Zhang ve Sennrich, 2019), RoPE (Su vd., 2024),
bağlı gömme (Press ve Wolf, 2017), dropout 0, en çok 64 konum, fp32; 3.445.824 parametre, bunların 2.659.392'si gömme
dışıdır. Model, bu çalışmanın kodundan önce commit 65ce430 ile düzeltilen başlatma kuralıyla kurulur: residual akışa
yazan izdüşümler (out_proj ve fc2) GPT-2'nin derinlik ölçeklemesini izleyerek 0,02/√(2L) standart sapmayla başlatılır
(Radford vd., 2019; ölçeklemesiz std 0,02: Radford vd., 2018). Aynı commit gömme dropout'unun iki kez uygulanmasını da
düzeltti; bu çalışmada dropout zaten 0'dır.

Ön eğitim deterministiktir: AdamW (Loshchilov ve Hutter, 2019; lr 2·10⁻³, β = (0,9; 0,98), ağırlık azalması 0,01), 100
adımlık ısınma, öğrenme oranını son adımda 0,1 katına indiren kosinüs azalması, gradyan kırpma 1,0, 64 dizilik yığın,
tek iş parçacığı. Her geçişte, P3 varlıklarının tutulan 120 hücresi dışında her (varlık, ilişki, hâl) hücresi bir kez
görülür (geçiş başına 8.280 cümle); bağlam her hücre için yeniden çekilir (yok/P1/P2 = 0,50/0,25/0,25) ve çekimli
hâllerde gündelik yazım 0,25 olasılıkla seçilir (yalın hâl her zaman biçimseldir). Düzyazı, kayba giren tokenlerin
%20'sini oluşturur (en çok 60 tokenlik parçalar; parça sonlarına sahte EOS eklenmez). Eğitim düzyazısı docs/book/tr
kitabının 1–9. bölümleri (kod blokları çıkarılmış) ve education/ klasörüdür; kişisel mektuplar ve sağlıkla ilgili bir
formülasyon dosyası dışarıda bırakılır. 10–12. bölümler eğitimden ayrı tutulur; bayt başına bit ölçümünde, G1
gidiş-dönüş denetiminde ve ad çakışma sözlüğünde kullanılır. Yığınlar türe göre kurulur: bir geçişin dünya cümleleri ve
düzyazı parçaları ayrı ayrı karıştırılıp 64'lük yığınlara bölünür, sonra yığınların sırası karıştırılır (protokol
metninde olmayan, dondurulmuş kodda sabit bir uygulama seçimi). Geçiş sayısı K = 20'dir (taban başına 3.130–3.136 adım,
≈5,3 milyon token): önceden kayıtlı {20, 25, 30, 35, 40} dizisinin ilk değeridir ve iki keşif tabanı bu değerde bütün
kapıları geçti.

Kaynak: `results/bases/*/meta.json` (n_params, n_params_non_embedding, hyper, steps_done, tokens, pass_stats).

Dondurmadan önceki düzeltmelerden biri düzyazı girişleridir: her geçişte her dünya cümlesinin önüne, bağımsız olarak
0,20 olasılıkla, eğitim düzyazısından seçilmiş kısa bir giriş cümlesi (3–16 sözcük, en çok 20 token, rakamsız ve
işaretlemesiz; 4.490 cümlelik havuz) eklenir. Gerekçe: ilk tasarımda model bir dünya cümlesini hiç düzyazının ardından
görmüyordu; çalışma dışı bir tohumla yapılan pilotta temel model düzyazı önekinden sonra eski kasabayı 4.200 yoklamanın
yalnız %0,9'unda üretti ve STD'nin v* araması 72 koşunun hiçbirinde yakınsamadı. STD'nin ortalamada kullandığı 10 elle
yazılmış düzyazı öneki bu havuzun dışındadır. Keşif ön eğitimi bu değişiklikle yeniden başlatıldı.

Düzenlemelerden önce her temel model şu kapılardan geçirilir (G3–G7'de başarı, en çok 8 tokenlik gerçek açgözlü
üretimle ölçülür):

- G1: dünya dizgelerinde ve tutulmuş düzyazıda %100 gidiş-dönüş (ve önek–nesne token sınırı denetimi);
- G3: görülmüş 600 biçimsel bağlamsız hücrede ≥ %95;
- G4: P3'te tutulan 120 hücrede ≥ %80;
- G5: 600 kesme işaretsiz hücrede ≥ %80 (IDENT ve NONIDENT ayrı raporlanır);
- G6: 600 bağlamlı hücrede ≥ %80;
- G7: A ilişkisinde yedi biçimsel kalıbın hepsinde ve altı kesme işaretsiz kalıbın en az üçünde doğru olan en az 120
  yoklama varlığı.

G7'yi sağlayan varlıklar düzenlenebilir kümedir; protokolde G2 adlı bir kapı yoktur. Kapıyı geçemeyen doğrulayıcı
tabanın yerine yedek dünya (1008, sonra 1009) alınacaktı.

### Yazım, kollar ve seçim

Anahtar kollarında (STD, ROOT, LAST, PARA-ORTH; NORM, STD'nin yazımını kullanır) yazım, seçilen tek bir bloğun fc2
çıkışına ileri bir kancayla eklenen sıra-bir terimdir; fc2 ağırlıkları değişmez: M = (v* − W k̄)(C0⁻¹k̄)ᵀ /
(k̄ᵀC0⁻¹k̄) (ROME'un kapalı çözümü; Meng vd., 2022). Burada W fc2 ağırlığı, k̄ yazılan anahtar (fc2 girdisi) ve
v* = W k̄ + δ'dır; yazım böylece k̄'de çıkışa tam olarak δ ekler. Güncellemede aynı k̄'nin tutarlı kullanılması,
önekli ve öneksiz anahtarların karıştırılmasıyla ilişkilendirilen ROME çöküşüne karşı önerilen biçimdir (Gupta vd.,
2024; Yang vd., 2024). Her düzenleme temel modele tek başına uygulanır, ölçülür ve sonraki düzenlemeden önce silinir;
KV önbelleği her yazımdan sonra temizlenir.

C0, her temel model ve katman için ayrı ayrı ve float64 duyarlıkla, ≈20.000 konumda fc2 girdilerinin ortalanmamış ikinci
momenti (h hᵀ ortalaması) olarak tahmin edilir. Konumların yaklaşık yarısı C0 havuzundaki 60 varlığın bütün kalıp,
bağlam ve yazımları kapsayan 7.020 hücresinden tohumla çekilmiş cümlelerden, yarısı eğitim düzyazısından (paragraf
başına en çok 63 token) gelir; BOS konumları dahildir, PAD konumları hariçtir. C0'a ayrıca 10⁻⁴·iz(C0)/512·I eklenir. C0
havuzu düzenleme ve yoklama varlıklarından ayrıktır.

δ, yazımın yapılacağı katmanda fc2 çıkışına anahtar konumunda eklenen bir vektördür. Adam ile δ = 0'dan başlanarak şu
amaç en aza indirilir: istemlerde yeni nesnenin (ardından gelen nokta dahil) ortalama negatif log-olabilirliği (NLL) +
1,0 × KL(temel ‖ düzenlenmiş). KL terimi, aynı varlığın yaşanan kalıptaki B ilişkisi isteminin son konumundaki sonraki
token dağılımında hesaplanır (δ bu istemin anahtar konumuna da eklenir). KL'nin aynı varlığın B ilişkisine uygulanması
bu çalışmanın seçimidir; ROME'daki KL terimi öznenin genel temsilinin kaymasını sınırlamaya yöneliktir. C ilişkisi
korunmaz ve seçimde sızıntı kısıtı olarak kullanılır; RB sızıntısı bu yüzden kısmen düzenlileştirilmiş bir ölçüdür.
Arama en çok 120 adım sürer ve ortalama NLL 0,05'in altına inince durur; ‖δ‖, o katmandaki medyan fc2 çıkış normunun 8
ya da 16 katıyla sınırlanır; adım büyüklüğü {0,5; 2; 5} içinden seçilir. İstemler: STD'de yaşanan cümle ile 10
düzyazı-önekli kopyadan 4'ü (toplam 5; bu 4 kopya bir dünyadaki bütün düzenlemeler için aynıdır, tohum w+4); ROOT ve
LAST'ta yalnız yaşanan cümle. Kollar:

- STD — ROME/MEMIT standardının bu çalışmadaki uyarlaması: anahtar, yalın ad aralığının son tokeninde, yaşanan cümle ve
  10 sabit, elle yazılmış düzyazı-önekli kopyası üzerinden (11 istem) ortalanır; v* araması bunlardan 5'ini kullanır.
  Biçimsel metinde bu konum kök-son tokenidir (adın son kök tokeni). ROME anahtarı, düzenlenen modelin kendi ürettiği
  kısa metin önekleri üzerinden ortalar (Meng vd., 2022); buradaki önekler elle yazılmıştır ve dağılım dışı kalma riski
  düzyazı girişleriyle azaltılmaya çalışıldı.
- ROOT — adın tamamen içinde kalan son tokendeki, ortalamasız anahtar (biçimsel yaşanan cümlede STD ile aynı konum);
  anahtar da v* araması da yalnız yaşanan cümleden hesaplanır.
- LAST — adı içeren sözcüğün son tokeni (biçimsel metinde ek tokeni); bir "yüzey aralığı" ardışık düzeni olarak
  etiketlidir.
- PARA-ORTH — adın A ilişkisindeki bağlamsız cümle başı kalıplarında 13 biçim (yalın dahil 7 biçimsel, 6 kesme işaretsiz
  çekimli) için ayrı ayrı hesaplanan STD anahtarlarının eşit ağırlıklı ortalaması. Kendi v* araması yoktur: STD'nin δ'sı
  ile STD'nin seçilen katmanı ve yapılandırması kullanılır (v* = W k_PO + δ_STD). Hedef hâl yoklamalarının (PF5/IF5) her
  biri, o biçimin kendi anahtarı ortalamadan çıkarılmış ayrı bir yazımla sınanır; diğer bütün aileler (WF, PF6, CF14 ve
  sızıntı yoklamaları) 13 biçimin hepsini içeren yazımla ölçülür. Yedi biçimsel kalıbın ve IDENT gündelik biçimlerin STD
  anahtarları bit düzeyinde aynıdır (V2); bu yüzden PARA-ORTH anahtarı STD'ninkinden yalnız FUSED biçimlerin
  anahtarlarıyla ayrılır. Sınanan biçimin anahtarını ortalamadan çıkarmak yalnız FUSED hedeflerde görülmemiş bir anahtar
  bırakır; biçimsel ve IDENT hedeflerde aynı anahtar ortalamada kalır.
- FT-L / FT-ALL — v* aramasıyla aynı amaçla (NLL + 1,0 × KL; NLL 0,05'in altına inince durur), yalnız yaşanan cümle
  üzerinde Adam ile ince ayar. FT-L yalnız seçilen katmanın fc2 ağırlığını değiştirir ve her adımdan sonra ΔW'nin her
  öğesini ±ε içine kırpar (Zhu vd., 2020; Meng vd., 2022'deki FT+L); ΔW de fc2 kancasıyla kurulur ama sıra-bir değildir
  (lr ∈ {10⁻³, 10⁻²}, ε ∈ {0,005; 0,02}, en çok 60 adım). FT-ALL bütün parametreleri değiştirir; dinamik değerlendirme
  tarzında bir yaşarken öğrenendir (Krause vd., 2018; lr ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³}, en çok 40 adım).
- NORM — STD yazımı; yalnız gündelik hedef yoklamalarında (IF5), dünyanın 400 adından biri ile geçerli bir hâl ekinden
  oluşan sözcüklere kesme işareti geri getirilir. Gündelik biçimler adlar arasında benzersiz olduğundan bu işlem her
  gündelik hedef yoklamasını biçimsel karşılığına birebir çevirir; NORM'un gündelik başarısı bu yüzden tanım gereği
  STD'nin biçimsel hedef başarısına eşittir (bir tavan kolu) ve diğer bütün ailelerdeki değerleri STD'ninkilerle
  aynıdır.
- BASE — yazım yok; sızıntısı tanım gereği 0'dır ve eski nesnenin kalıntısı için referanstır.

Yapılandırmalar yalnız keşif dünyalarında (0 ve 1; dünya başına 20 keşif düzenlemesi, yalın-yaşantılı düzenlemeler
dahil) dondurulmuş bir kuralla seçilir. STD, ROOT ve LAST için katman 0–4 × adım {0,5; 2; 5} × üst sınır çarpanı {8;
16}, FT-L için katman 0–4 × lr × ε, FT-ALL için yalnız lr taranır; PARA-ORTH ve NORM kendi seçimlerini yapmaz, STD'nin
katmanını ve yapılandırmasını kullanır. Bağlanma yeri haritası için ayrıca PRED konumu (son istem tokeni; adım 2, üst
sınır {8; 16}) taranır. Kural, ortalama yazım etkinliğini (WE: yaşanan kalıpta yeni nesnenin üretildiği düzenlemelerin
payı) en yükselten yapılandırmayı seçer; koşullar: rastgele varlıklara sızıntı (Leak_RD) ≤ %1 ve aynı varlığın C
ilişkisine sızıntı (Leak_C; doğrulayıcı tablolarda Leak_RC) ≤ %5. Eşitlikte sığ katman, sonra küçük adım (FT'de küçük
lr), sonra küçük üst sınır (FT-L'de küçük ε) seçilir. Hiçbir yapılandırma koşulları sağlamazsa en yüksek WE'yi veren
yapılandırma alınır ve kol "kısıtlar altında kurulamaz" (not installable under constraints) diye etiketlenir. STD'nin
seçilen yapılandırmasının keşif WE'si %50'nin altında kalsaydı H1–H3 sınanamaz sayılacaktı (yalnız-harita kuralı);
dondurulmuş kodda doğrulayıcı harita koşucusu olmadığından bu yol dondurma sonrası kod ve kayıt gerektirecekti.

### Düzenleme maddeleri, yoklamalar ve ölçüler

Her doğrulayıcı temel modelde 70 düzenleme yapılır: G7 düzenlenebilir kümesinden birbirinden farklı 70 varlık, yaşanan
hâl başına 10 (yalın dahil yedi hâl), yalnız A ilişkisinde. Yeni nesne eski kasaba dışındaki 47 kasabadan düzgün
çekilir. Yaşanan cümle, yaşanan hâlin bağlamsız, cümle başı, biçimsel kalıbıdır. Düzenleme listeleri tohumlardan
deterministik olarak türetilir ve SHA-256 özetleri kaydedilir (`results/edits/items_<dünya>.json`). Bir düzenlemenin
hedef hâlleri, yaşanan hâl dışındaki çekimli hâllerden temel modelin, gerçek açgözlü çözümlemeyle, eski nesneyi hem
biçimsel hem kesme işaretsiz yazımda ürettiği hâllerdir. H1, H2 ve S1 için en az iki hedef hâl gerekir; daha azı olan
düzenlemeler bu hipotezlerden çıkarılır ve sayılır. Hipotezler yalnız yaşanan hâli çekimli olan 60 düzenlemeyi
kullanır; 10 yalın düzenleme yalnız betimsel tablolara girer (protokol onları V3 için de öngörür; bkz. Doğrulama
testleri).

Başarı, yeni nesnenin ve ardından gelen noktanın token düzeyinde birebir açgözlü üretilmesidir. Öğretmen zorlamalı
toplu bir denetimle ölçülür: açgözlü çözümleme bir adayı ancak her devam konumunda argmax eşleşiyor ve ardından '.'
geliyorsa üretir. Bu denetim gerçek açgözlü çözümlemeye karşı yalnız koşulardan önce alınmış bir örnekle sınandı
(V8): dört dünyada (0, 1, 1000, 1001), dünya başına 10 STD düzenlemesinde ve temel modelde, her düzenleme, durum ve
yoklama ailesi için yoklamaların %5'i (en az bir yoklama; böylece bu yoklamaların gerçekte ≈%8'i) çekildi; dünya başına
294–362 yoklama (bkz. Doğrulama testleri). Yoklama aileleri
(bağlamsız, cümle başı kalıplar; ROME ile tanıtılan etkinlik, genelleme ve özgüllük ölçü ailelerine karşılık gelir,
Meng vd., 2022):

- WF: yaşanan kalıp; yeni nesne burada üretiliyorsa düzenleme "kurulmuş" sayılır.
- PF6: diğer altı biçimsel hâl kalıbı.
- PF5 ve IF5: hedef hâller biçimsel ve kesme işaretsiz yazımda; her gündelik biçim IDENT ya da NONIDENT diye etiketli.
- CF14: yedi biçimsel kalıp × iki sol bağlam. Bu bağlamlar ön eğitimde görülmüştür (her biri dünya cümlelerinin
  ≈%25'inde); CF14 görülmemiş bağlama değil, görülmüş bağlama dayanıklılığı ölçer.
- RN: adın son kök tokenini paylaşan en çok 8 yoklama varlığı; RD: bu kümenin dışından 10 rastgele yoklama varlığı;
  ikisi de yedi biçimsel kalıpta.
- SS: yaşanan kalıpta aynı ek tokenini taşıyan en çok 20 varlık (yalın yaşantıda boş).
- RB ve RC: aynı varlığın B ve C ilişkileri, yedi biçimsel kalıpta.
- Betimsel olarak ayrıca önek-uzantı eşi (PX), eski nesnenin kalıntısı (OLD), kol başına 20 düzenlemede tutulmuş
  düzyazıda bayt başına bit (bpb) ve τ değerleri kaydedilir.

Sızıntı, temel modelin kendi açgözlü cevabına göre değişen cevap oranıdır; temel cevap bir kez, gerçek açgözlü
çözümlemeyle hesaplanır ve dünya düzeyindeki sızıntı düzenleme başına payların ortalamasıdır. PE6, PE5, IE5 ve CE14
(PF6, PF5, IF5 ve CF14 yoklamalarında yeni nesnenin üretilme payları) bütün nitelikli düzenlemeler üzerinden koşulsuz
paylardır (kurulmamış düzenlemeler dahil); T-sürümleri yalnız kurulmuş düzenlemeleri kullanır. τ(k', k̄) = k'ᵀC0⁻¹k̄ /
k̄ᵀC0⁻¹k̄, bir sorgu anahtarının yazılan anahtara C0 ölçüsündeki izdüşümüdür.

### Hipotezler ve karar kuralı

Birincil hipotezler (Holm, m = 3, tek yönlü α = 0,05; birim: bağımsız ön eğitilmiş temel model, n = 8 doğrulayıcı
dünya 1000–1007; kapıyı geçemeyen tabanın yerine yedek 1008, sonra 1009):

- H1: STD'de, en az iki hedef hâli olan ve yaşanan hâli çekimli olan düzenlemelerde (kurulum koşulu yok), aynı hedef
  hâllerde biçimsel yoklamaların yeni nesneyi üretme payı (PE5) kesme işaretsiz yoklamalarınkini (IE5) 15 puandan
  fazla aşar (öngörü: ≥ 25 puan).
- H2: Kurulmuş ve en az bir IDENT ile en az bir NONIDENT hedef hâli olan STD düzenlemelerinde, düzenleme içi
  biçimsel–gündelik fark (F_c − I_c) NONIDENT hâllerde IDENT hâllerdekinden ortalama 30 puandan fazla büyüktür
  (öngörü: ≥ 50 puan). Bu tokenleştiricide NONIDENT, FUSED ile aynıdır. Hipotez, gündelik yazımdaki kaybın, yazılan
  anahtarın konumunun gündelik sorguda bulunup bulunmamasıyla açıklandığı öngörüsünü sınar.
- H3: Kurulmuş ve en az bir RN komşusu olan STD düzenlemelerinde, son kök tokenini paylaşan varlıklara sızıntı
  (Leak_RN) rastgele varlıklara sızıntıyı (Leak_RD) 5 puandan fazla aşar (öngörü: ≥ 10 puan).

İkincil hipotezler (Holm, m = 4, tek yönlü α = 0,05, aynı karar kuralı):

- S1: PARA-ORTH'un, sınanan biçimi dışarıda bırakan anahtarla gündelik hedef başarısı (IE5) STD'ninkini 10 puandan
  fazla aşar. Ayrıca dünya başına Leak_RD farkının iki yönlü %95 t-aralığının üst sınırı ≤ 1 puan ve Leak_RN farkınınki
  ≤ 5 puan olmalıdır; aksi hâlde sonuç "desteklenmedi (yerellik maliyeti)" olur. Etkinlik biri-dışarıda yazımlarla,
  yerellik farkları ise 13 biçimin hepsini içeren yazımla ölçülür.
- S2: Eğitimde görülmüş iki sol bağlamla yoklamalarda (CF14) STD, ROOT'u 10 puandan fazla aşar (birim: aynı maddenin
  çekimli-yaşantılı STD ve ROOT düzenlemesi; kurulum koşulu yok). İki kol anahtar ortalamasının yanında v*
  istemlerinde (5'e karşı 1) ve ayrı seçilen katman ve yapılandırmada da farklıdır; S2 bu yüzden iki ardışık düzeni
  karşılaştırır. Protokol S2'yi "ROME/CoRE gerekçesinin yinelemesi" diye anar; CoRE (Park vd.,
  2025) soruna yalnız genel önek ortalamasıyla değil, düzenlenen olguya ilişkin öneklerde gizli durum varyansını
  cezalandırarak yaklaşır. S2 bu yüzden ROME'un gerekçesini sınar, CoRE'nin yinelemesi değildir.
- S3: 60 varlık üzerinde 100 işlemlik bir yaşamda (60 A yazımı, 15 C yazımı, 15 A düzeltmesi, 10 A geri alması)
  varlık-ortak float64 defterin (J) yaşam sonu doğruluğu (E_end), bağımsız ROME farklarınınkini (I) 10 puandan fazla
  aşar. J'de ağırlık W = W0 + B(λC0 + A)⁻¹ biçimindedir ve bir varlığın inanılan olgu kümesi değişince ortak v* W0'a
  karşı yeniden eniyilenir; E_end, inanılan nesnelerin yedi biçimsel kalıpta tam üretimidir. Ayrıca I'da bir olgunun
  diğer canlı anahtarlara en büyük τ'su ile başarısızlığı arasında dünya içi Spearman ρ > 0 (permütasyon p ≤ 0,05)
  olmalıdır. Q (güncel ağırlıklar üzerinde sıralı ROME) dört rastgele sırada betimsel olarak koşulur. Defter λ'sı {0,01;
  0,1; 1} içinden, keşif yaşamlarında Leak_RD ≤ %1 koşuluyla E_end'i en yükselten değerdir. Yaşamlarda geri alınan A,
  yaşam sonunda temel nesnesiyle inanılan olgu sayılır. Bu bir yenilik iddiası değildir: kapalı biçimli ömür boyu
  düzenlemede bilgi temsilleri birbirine dik olmadığında bir girişim terimi ortaya çıkar (Hu vd., 2025); aynı özneye ait
  olgular tek anahtara ayrı değerlerle yazıldığında çakışır (Dong vd., 2025).
- S4: RN komşularına τ'nun medyanı (cümle başı yalın önekte kök-son tokenindeki anahtarlarla ve katmana özgü C0 ile
  ölçülür) katman 0'dan katman 4'e 0,15'ten fazla düşer; birim, katman 0 ve 4'te RN komşusu olan varlıktır (dünya başına
  en az 15).

Karar kuralı: her dünyanın etkisi düzenleme kayıtlarından hesaplanır; bir dünya bir hipoteze ancak en az 15 nitelikli
birimle katkı verir ve 7'den az dünya nitelikliyse sonuç "belirsiz (yetersiz veri)" olur. Marja göre kaydırılmış etkiler
(d_w − m) üzerinde 2ⁿ işaret örüntüsünün hepsini sayan kesin, tek yönlü bir işaret-çevirme permütasyon testi yapılır;
Holm (1979) düzeltmesi her ailenin tamamı üzerinde uygulanır (sınanamayan üye p = 1 ile girer). Desteklendi = Holm
reddi (S1'de ayrıca yerellik sınırları, S3'te Spearman koşulu); çürütüldü = ortalama etkinin iki yönlü %95
t-aralığının üst sınırı marjın altında (ters yönlü etki dahil); aksi hâlde belirsiz. Manipülasyon denetimi: H1
birimlerinin kesme işaretsiz hedef biçimleri arasında NONIDENT payı her dünyada hesaplanır ve doğrulayıcı dünyalar
üzerinden ortalanır; ortalama %25'in altındaysa H1 ve H2 "çürütüldü" olamaz. n = 8 dünyada kesin testin
ulaşabileceği en küçük p değeri 1/256 ≈ 0,0039'dur; d_w ≤ m olan dünya sayısı k ise p ≥ 2^k/256 olur. Holm'un ilk
eşiği birincil ailede (0,05/3) k ≤ 2'ye, ikincil ailede (0,0125) k ≤ 1'e izin verir; bu gerekli bir koşuldur, yeterli
değildir. Betimsel olarak dünya başına değerler, t-aralıkları ve hiyerarşik bootstrap (dünyalar, sonra birimler;
10.000 yeniden örnekleme) raporlanır. analyze.py ve cev11/stats.py, hiçbir doğrulayıcı hipotez ölçüsü hesaplanmadan
önce SHA-256 özetiyle dondurulmuştur.

Bağlanma yeri haritası (betimsel) keşif dünyalarının seçim ızgarası kayıtlarından raporlanır (ROOT, LAST, STD ve PRED
konumları, katman 0–4, dünya başına 20 düzenleme). Doğrulayıcı dünyalarda harita, nedensel izleme ve keşif amaçlı
F_freq kolu yalnız hesap kalırsa koşulacaktı.

## Bulgular

Bütün sayılar çalışmanın commit edilmiş sonuç dosyalarından (results/) ya da dondurma sonrası kayıttan
(AMENDMENTS.md; yalnız S3 teşhisi) gelir. Her tablonun kaynağı, tablo başlığının altındaki "Kaynak:" satırında
yazılıdır; bu satırlardaki `results/` yolları çalışma klasörüne (`research/living_learning_paradigm_2026_11_13/`)
görelidir. Tablolar results/ klasöründen bir betikle aktarıldı (bkz. Yeniden üretim). Rapor yazılırken, doğrulayıcı
kararlar hesaplandıktan sonra, commit edilmiş kayıtlardan birkaç keşif amaçlı ek hesap yapıldı (Tablo 5a, δ'nın üst
sınıra dayanma payı, keşif ızgarasındaki aralıklar ve τ-AUC); bunlar önceden kayıtlı test değildir ve kararları
etkilemez (Sapmalar ve olaylar, 16). Tablolarda ondalık ayırıcı noktadır, metinde virgüldür. Paylar yüzde (%), pay
farkları puan (yüzde puanı) olarak verilir; S4 ve τ değerleri birimsizdir. Hücrelerdeki "a (b…c)" gösterimi, dünya
değerlerinin ağırlıksız ortalamasını (a) ve en küçük ile en büyük değerini (b…c) verir. Yuvarlamada tam yarımlar çift
basamağa gider (ör. 81,25 → 81,2; 0,625 → 0,62).

### Kapılar ve manipülasyon denetimi

On tabanın hepsi (iki keşif, sekiz doğrulayıcı) bütün kapıları geçti; yedek dünya kullanılmadı ve keşifte ne K'nin
sonraki değerleri ne de geri düşüş merdiveni kullanıldı (Tablo 1). Doğrulayıcı tabanlarda G3 %99,8–100, G4 %100, G5
%99,2–99,8 (NONIDENT biçimlerde %97,7–99,5) ve G6 %100'dü; düzenlenebilir küme 279–280 varlıktı.

Tablo 1. Taban kapıları.

Kaynak: `results/bases/<taban>/gates.json` (10 taban).

| Taban | Hepsi | G1 dünya / tutulmuş düzyazı (%) | G3 (%) | G4 (%) | G5 (%) | G5 IDENT / NONIDENT (%) | G6 (%) | G7 düzenlenebilir | sınır (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery_w0_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.3 | 100.0 / 98.2 | 100.0 | 280 | 100.0 |
| discovery_w1_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.3 | 100.0 / 98.3 | 100.0 | 280 | 100.0 |
| confirm_w1000_K20 | geçti | 100.0 / 100.0 | 99.8 | 100.0 | 99.7 | 100.0 / 99.1 | 100.0 | 280 | 100.0 |
| confirm_w1001_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.3 | 100.0 / 98.1 | 100.0 | 280 | 100.0 |
| confirm_w1002_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.5 | 100.0 / 98.6 | 100.0 | 280 | 100.0 |
| confirm_w1003_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.7 | 100.0 / 99.0 | 100.0 | 280 | 100.0 |
| confirm_w1004_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.5 | 100.0 / 98.7 | 100.0 | 279 | 100.0 |
| confirm_w1005_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.8 | 100.0 / 99.5 | 100.0 | 280 | 100.0 |
| confirm_w1006_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.2 | 100.0 / 97.7 | 100.0 | 280 | 100.0 |
| confirm_w1007_K20 | geçti | 100.0 / 100.0 | 100.0 | 100.0 | 99.3 | 100.0 / 98.1 | 100.0 | 280 | 100.0 |

Eşikler: G1 %100; G3 ≥ %95; G4 ≥ %80; G5 ≥ %80; G6 ≥ %80; G7 ≥ 120 düzenlenebilir varlık; sınır denetimi %100.

Manipülasyon denetimi geçti: H1 birimlerinin kesme işaretsiz hedef biçimlerinde NONIDENT payı ortalama %35,1'di (dünya
başına %30,1–38,8; eşik %25; Tablo 2). Bu yüzden H1 ve H2 için üç karar da (desteklendi, çürütüldü, belirsiz) mümkündü.

Tablo 2. Manipülasyon denetimi: H1 birimlerinin gündelik hedef biçimlerinde NONIDENT payı.

Kaynak: `results/decisions.json` (manipulation_check).

| Dünya | NONIDENT payı (%) | gündelik hedef biçim sayısı |
| :--- | ---: | ---: |
| 1000 | 37.1 | 299 |
| 1001 | 37.4 | 297 |
| 1002 | 37.9 | 298 |
| 1003 | 37.0 | 300 |
| 1004 | 30.1 | 299 |
| 1005 | 31.5 | 298 |
| 1006 | 38.8 | 299 |
| 1007 | 30.7 | 296 |
| ortalama | 35.1 | — |

### Birincil aile (H1–H3)

Birincil hipotezler STD kolunu keşifte seçilen yapılandırmasıyla kullanır: katman 1, adım 5, üst sınır 8 × medyan fc2
normu. Bu yapılandırma keşifte iki sızıntı kısıtını da sağlamadı (Leak_RD %1,2 > %1; Leak_C %25,4 > %5; bkz. Keşif
seçimi). Kural onu yalnız WE ile seçti: WE'si %100 olan tek STD yapılandırmasıydı. WE'si %97,5 olan (40 keşif
düzenlemesinden 39'u) sekiz STD yapılandırması daha vardı; örneğin katman 0, adım 0,5 yapılandırması Leak_RD'yi
sağlıyordu (%0,7) ve Leak_C'si %10,4'tü. Doğrulayıcı STD düzenlemelerinin 517/560'ında (%92; dünyaya göre %83–100) δ üst
sınıra dayandı (kayıtlarda diag.delta_norm = diag.cap; ROOT'ta 3/560): yazımın büyüklüğünü çoğunlukla arama değil üst
sınır belirledi. Keşif ızgarasında WE'si en az %90 olan 16 STD yapılandırmasında (katman 0–2) PE5 − IE5 farkı 22,8–28,3
puan arasında kaldı; Leak_RN − Leak_RD farkı ise 8,7–56,4 puan arasında değişti (`discovery_map.csv`; bütün keşif
düzenlemeleri üzerinden). H1'in büyüklüğü bu yapılandırmalar arasında az, H3'ünki çok değişir. Aşağıdaki etkiler seçilen
yapılandırmanın etkileridir. Üç hipotezde de sekiz dünyanın hepsi nitelikli birim eşiğini geçti ve her dünyanın etkisi
marjın üstündeydi. Kesin testin p değeri bu yüzden ulaşılabilecek en küçük değerdir (1/256 ≈ 0,0039); Holm sonrası
0,0117'dir. Üç hipotez de desteklendi (Tablo 3).

Tablo 3. Birincil aile kararları.

Kaynak: `results/decisions.json` (hypotheses.H1–H3, families.primary); `results/tables/decisions_summary.csv`.

| Hipotez | Ölçülen fark | Birim | d̄ | %95 t-aralığı | %95 bootstrap | Marj m | p (işaret-çevirme) | Holm-düzeltilmiş p | Holm reddi | Karar | Nitelikli dünya |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| H1 | Kesme işaretsiz özneler: PE5(STD) − IE5(STD) | puan | +24.3 | [+21.0, +27.5] | [+20.9, +27.8] | 15.0 | 0.0039 | 0.0117 | evet | desteklendi | 8/8 |
| H2 | Özdeş anahtar konumu (protokol başlığında "gerekli"; bkz. not): NONIDENT − IDENT biçimsel–gündelik farkı | puan | +69.7 | [+62.5, +76.9] | [+62.4, +76.7] | 30.0 | 0.0039 | 0.0117 | evet | desteklendi | 8/8 |
| H3 | Kök yerinin özgüllük maliyeti: Leak_RN(STD) − Leak_RD(STD) | puan | +33.4 | [+29.1, +37.7] | [+29.4, +38.2] | 5.0 | 0.0039 | 0.0117 | evet | desteklendi | 8/8 |

d̄: nitelikli dünyaların etkilerinin ortalaması. %95 t-aralığı iki yönlüdür ve çürütme kuralında kullanılır; bootstrap
aralığı betimseldir (hiyerarşik: dünyalar, sonra birimler; 10.000 yeniden örnekleme). p: marja göre kaydırılmış, tek
yönlü kesin işaret-çevirme testi.

H2'nin protokoldeki başlığı ("özdeş anahtar konumu gereklidir; nedeni kök ile ekin kaynaşması ya da yeniden
bölünmesidir") sınanan istatistikten güçlüdür. Karar, kaynaşmış biçimlerdeki biçimsel–gündelik farkın IDENT
biçimlerdekinden 30 puandan fazla büyük olduğunu gösterir; özdeş anahtarın kesin olarak gerekli olduğunu göstermez:
FUSED hedeflerde gündelik başarı STD'de %14,9, PARA-ORTH'ta %24,2, FT-L ve FT-ALL'da %55,3 ve %71,7'dir (Tablo 15).
Tasarım gözlemseldir; kaynaşmayı değiştiren bir müdahale yapılmadı.

Tablo 4. H1, dünya başına: aynı hedef hâllerde biçimsel (PE5) ve kesme işaretsiz (IE5) yoklamalarda yeni nesnenin
üretilme payı.

Kaynak: `results/decisions.json` (hypotheses.H1.per_world); `results/tables/per_world_H1.csv`.

| Dünya | birim | nitelikli | d_w (puan) | PE5 (%) | IE5 (%) | çekimli düzenleme | dışlanan: <2 hedef |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 60 | evet | +26.0 | 86.7 | 60.7 | 60 | 0 |
| 1001 | 60 | evet | +24.8 | 89.0 | 64.2 | 60 | 0 |
| 1002 | 60 | evet | +31.1 | 89.8 | 58.7 | 60 | 0 |
| 1003 | 60 | evet | +26.3 | 82.7 | 56.3 | 60 | 0 |
| 1004 | 60 | evet | +18.2 | 68.3 | 50.2 | 60 | 0 |
| 1005 | 60 | evet | +24.7 | 90.3 | 65.7 | 60 | 0 |
| 1006 | 60 | evet | +21.8 | 83.7 | 61.9 | 60 | 0 |
| 1007 | 60 | evet | +21.3 | 82.8 | 61.4 | 60 | 0 |

**H1 desteklendi.** Aynı hedef hâllerde kesme işaretli yoklamalar yeni nesneyi ortalama %84,1, kesme işaretsiz
yoklamalar %59,9 oranında üretti (Tablo 4 ve 13). Dünya etkileri 18,2 ile 31,1 puan arasındadır; ortalama 24,3 puan,
öngörülen 25 puanın biraz altında, marjın (15 puan) ise belirgin biçimde üstündedir (öngörü karar kuralının parçası
değildir). Bu modelde ve bu yapılandırmada tek bir biçimsel cümleden yapılan yazım, aynı varlığın diğer biçimsel
hâllerine büyük ölçüde taşınır, ama kesme işareti düşünce bu genellemenin bir bölümü kaybolur.

Tablo 5. H2, dünya başına: düzenleme içi biçimsel–gündelik fark (F − I), NONIDENT ve IDENT hedef hâllerde.

Kaynak: `results/decisions.json` (hypotheses.H2.per_world); `results/tables/per_world_H2.csv`.

| Dünya | birim | nitelikli | d_w (puan) | NONIDENT F−I (puan) | IDENT F−I (puan) | dışlanan: <2 hedef | dışlanan: kurulmadı | dışlanan: IDENT/NONIDENT karışımı yok |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 49 | evet | +75.7 | +74.3 | -1.4 | 0 | 2 | 9 |
| 1001 | 45 | evet | +63.1 | +64.8 | +1.7 | 0 | 2 | 13 |
| 1002 | 48 | evet | +81.2 | +81.2 | 0.0 | 0 | 2 | 10 |
| 1003 | 44 | evet | +76.1 | +77.8 | +1.7 | 0 | 4 | 12 |
| 1004 | 42 | evet | +60.5 | +60.3 | -0.2 | 0 | 7 | 11 |
| 1005 | 48 | evet | +74.3 | +72.0 | -2.3 | 0 | 1 | 11 |
| 1006 | 51 | evet | +57.0 | +58.0 | +1.0 | 0 | 2 | 7 |
| 1007 | 46 | evet | +69.6 | +71.6 | +2.0 | 0 | 2 | 12 |

**H2 desteklendi.** Düzenleme içi karşılaştırmada IDENT hâllerde biçimsel ve gündelik yoklama arasındaki fark −2,3 ile
+2,0 puan arasında kaldı; NONIDENT (bu tokenleştiricide FUSED) hâllerde 58,0–81,2 puandı (Tablo 5). Etki (69,7 puan)
öngörülen 50 puanın üstündedir. Havuzlanmış paylarda STD'nin IDENT hedeflerdeki gündelik başarısı %84,1, biçimsel
başarısı %84,6'dır; FUSED hedeflerde biçimsel başarı %83,4 iken gündelik başarı %14,9'a düşer (Tablo 15). Buna göre
H1'deki kayıp, yazılan anahtarın token dizisinin gündelik sorguda bulunmadığı biçimlerden gelir; kök tokenleri korunduğu
sürece kesme işaretinin düşmesi neredeyse hiçbir şey kaybettirmez. Bu karşıtlık gözlemseldir: IDENT/FUSED sınıfı adın ve
hâlin bir özelliğidir ve hâle göre çok değişir (bkz. Tokenleştirme); bu yüzden hâl ile sınıf kısmen örtüşür. Düzenleme
içi tasarım varlığa ait etkenleri dengeler, hâle ait etkenleri dengelemez. Bu yüzden aynı karşılaştırma keşif amaçlı
olarak hâl içinde de yapıldı (önceden kayıtlı bir test değildir; protokol §10.4'teki kalıp başına tablolardandır):
STD'de her hâlde IDENT biçimlerin biçimsel–gündelik farkı −1,0 ile +5,8 puan, FUSED biçimlerinki +68,5 ile +86,4 puan
arasındadır (Tablo 5a; ROOT'ta −0,4…+6,8 ve +79,9…+92,4). Fark hâlin kendisinden değil, aynı hâl içinde kaynaşmış
biçimlerden gelir; bu karşılaştırma da gözlemseldir, çünkü hâl içinde farklı adlar karşılaştırılır. Her dünyada 1–7
düzenleme kurulmadığı, 7–13 düzenleme de hem IDENT hem NONIDENT hedef taşımadığı için H2 dışında kaldı.

Tablo 5a (keşif amaçlı). Hâl içinde IDENT ve FUSED hedef biçimler: STD'nin kurulmuş, en az iki hedefli
çekimli-yaşantılı düzenlemelerinde yeni nesneyi üreten biçimsel ve gündelik hedef yoklamalarının payı. Paylar dünya
içinde havuzlanır; en az 5 biçimi olan dünyaların ağırlıksız ortalaması verilir; biçim sayısı ortalamaya giren
dünyaların toplamıdır.

Kaynak: `results/edits/confirm_<dünya>_STD.jsonl` (targets[].case, cls, formal, informal); rapor yazılırken salt okunur
bir betikle hesaplandı.

| Hâl | IDENT biçim | IDENT biçimsel / gündelik (%) | IDENT fark (puan) | FUSED biçim | FUSED biçimsel / gündelik (%) | FUSED fark (puan) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ilgi | 168 | 73.0 / 72.3 | +0.7 | 203 | 80.8 / 12.3 | +68.5 |
| yönelme | 198 | 90.0 / 87.0 | +2.9 | 182 | 88.4 / 17.1 | +71.3 |
| belirtme | 154 | 92.4 / 86.6 | +5.8 | 226 | 86.5 / 17.6 | +68.9 |
| bulunma | 326 | 89.6 / 90.0 | -0.3 | 47 (6 dünya) | 89.0 / 15.2 | +73.8 |
| ayrılma | 347 | 89.6 / 90.1 | -0.6 | 22 (4 dünya) | 86.4 / 0.0 | +86.4 |
| vasıta | 282 | 87.4 / 88.4 | -1.0 | 100 | 89.0 / 14.6 | +74.4 |

Tablo 6. H3, dünya başına: kurulmuş düzenlemelerde son kök tokenini paylaşan komşulara (RN) ve rastgele varlıklara (RD)
sızıntı.

Kaynak: `results/decisions.json` (hypotheses.H3.per_world); `results/tables/per_world_H3.csv`.

| Dünya | birim | nitelikli | d_w (puan) | Leak_RN (%) | Leak_RD (%) | dışlanan: kurulmadı | dışlanan: RN yok |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 47 | evet | +44.8 | 46.1 | 1.3 | 2 | 11 |
| 1001 | 44 | evet | +29.7 | 29.8 | 0.1 | 2 | 14 |
| 1002 | 48 | evet | +32.6 | 34.6 | 2.0 | 2 | 10 |
| 1003 | 46 | evet | +33.8 | 35.3 | 1.5 | 4 | 10 |
| 1004 | 47 | evet | +29.3 | 30.2 | 0.9 | 7 | 6 |
| 1005 | 47 | evet | +35.7 | 36.5 | 0.8 | 1 | 12 |
| 1006 | 44 | evet | +29.6 | 29.9 | 0.3 | 2 | 14 |
| 1007 | 38 | evet | +31.7 | 33.5 | 1.7 | 2 | 20 |

**H3 desteklendi.** Kurulmuş STD yazımları, adın son kök tokenini paylaşan komşuların yoklamalarında cevabı
%29,8–46,1 oranında değiştirdi; rastgele varlıklarda bu oran %0,1–2,0'dı (Tablo 6). Etki (33,4 puan) öngörülen 10
puanın çok üstündedir. Komşular yalnız cevap değiştirmekle kalmaz, yeni nesneyi de benimser: STD'de RN yoklamalarının
%18,6'sı yeni kasabayı üretti; yazımsız temel modelde komşunun zaten aynı kasabayı üretme oranı %1,7'dir (Tablo 16).
Bu sızıntının büyüklüğü yapılandırmaya bağlıdır: keşif ızgarasında WE'si en az %90 olan STD yapılandırmalarında (katman
0–2) RN − RD farkı 8,7 ile 56,4 puan arasında değişti ve derinlikle azaldı (`discovery_map.csv`; azalma Tablo 18'de de
görülür); seçilen yapılandırmada δ
düzenlemelerin %92'sinde üst sınıra dayandı. H3 yazım yerlerini değil, RN ve RD varlıklarını karşılaştırır; kök-son
konumunun payına dair kanıt keşif amaçlı kol ve harita karşılaştırmalarından gelir (ek tokenine yazan LAST'ta Leak_RN
%2,7; Tablo 14). Doğrulayıcı dünyalarda adların %81,5–86'sı son kök tokenini başka bir adla paylaştığından, bu
yapılandırmada kök-son konumuna yazmanın özgüllük bedeli büyüktür.

### İkincil aile (S1–S4)

Holm düzeltmesinden sonra hiçbir ikincil hipotez desteklenmedi: S1 ve S4 çürütüldü, S2 belirsiz kaldı, S3 koşulmadı
(Tablo 7).

Tablo 7. İkincil aile kararları.

Kaynak: `results/decisions.json` (hypotheses.S1–S4, families.secondary); `results/tables/decisions_summary.csv`.

| Hipotez | Ölçülen fark | Birim | d̄ | %95 t-aralığı | %95 bootstrap | Marj m | p (işaret-çevirme) | Holm-düzeltilmiş p | Holm reddi | Karar | Nitelikli dünya |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| S1 | Biçim-ortalamalı anahtarlar: IE5(PARA-ORTH, biri-dışarıda) − IE5(STD) | puan | +3.1 | [+1.5, +4.7] | [+1.6, +4.8] | 10.0 | 1.0000 | 1.0000 | hayır | çürütüldü | 8/8 |
| S2 | Önek ortalaması ve sol bağlam: CE14(STD) − CE14(ROOT) | puan | +12.3 | [+6.4, +18.2] | [+6.6, +18.0] | 10.0 | 0.1836 | 0.7344 | hayır | belirsiz | 8/8 |
| S3 | Yaşamda ortak ve bağımsız durum: E_end(J) − E_end(I) | puan | — | — | — | 10.0 | —* | 1.0000 | hayır | koşulmadı | 0/8 |
| S4 | Kimlik derinlikle kök-son tokeninde: medyan τ_RN katman 0 − katman 4 | τ | -0.086 | [-0.134, -0.039] | [-0.124, -0.046] | 0.150 | 1.0000 | 1.0000 | hayır | çürütüldü | 8/8 |

\* S3 için p hesaplanmadı (kayıt yok); hipotez Holm ailesine p = 1 ile girdi. S3'ün kararı protokol §14'e ve
dondurulmuş analyze.py'ye göre "koşulmadı"dır; §11'in kuralıyla (7'den az nitelikli dünya) "belirsiz (yetersiz veri)"
olurdu. İki okuma da desteklenmedi anlamına gelir.

S4'ün protokol başlığı ("kimlik derinlikle kök-son tokeninde toplanır") ölçülenden geniştir: ölçülen, kök-son
tokenindeki anahtarın son kök tokenini paylaşan adların anahtarlarına katmana özgü C0 ile τ benzerliğidir; çürütülen,
bu benzerliğin katman 0'dan 4'e 0,15'ten fazla düşeceği öngörüsüdür.

#### S1: biçim-ortalamalı anahtarlar

Tablo 8. S1, dünya başına: biri-dışarıda PARA-ORTH ile STD'nin gündelik hedef başarısı ve yerellik farkları
(PARA-ORTH − STD).

Kaynak: `results/tables/per_world_S1.csv` (`results/decisions.json` hypotheses.S1.per_world ve noninferiority ile aynı).

| Dünya | birim | d_w (puan) | IE5 PARA-ORTH (%) | IE5 STD (%) | IE NONIDENT PARA-ORTH (%) | IE NONIDENT STD (%) | ΔLeak_RD (puan) | ΔLeak_RN (puan) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 60 | +6.7 | 67.3 | 60.7 | 29.7 | 15.3 | +0.8 | +3.9 |
| 1001 | 60 | +4.0 | 68.2 | 64.2 | 36.0 | 22.5 | +0.7 | +5.2 |
| 1002 | 60 | +2.0 | 60.7 | 58.7 | 22.1 | 13.3 | +0.6 | +3.9 |
| 1003 | 60 | +0.3 | 56.7 | 56.3 | 11.7 | 9.0 | +0.6 | +3.4 |
| 1004 | 60 | +1.3 | 51.5 | 50.2 | 13.3 | 6.7 | +0.5 | +3.1 |
| 1005 | 60 | +3.3 | 69.0 | 65.7 | 22.3 | 11.7 | +0.4 | +3.0 |
| 1006 | 60 | +4.0 | 65.9 | 61.9 | 32.8 | 24.1 | +0.1 | +2.8 |
| 1007 | 60 | +3.3 | 64.8 | 61.4 | 25.3 | 16.5 | +1.3 | +2.4 |

**S1 çürütüldü.** PARA-ORTH'un biri-dışarıda gündelik hedef başarısı STD'ninkini ortalama 3,1 puan aştı; artış
sıfırdan büyüktür (t-aralığı [1,5; 4,7]), ama aralığın üst sınırı marjın (10 puan) altındadır (Tablo 7 ve 8). Yerellik koşulları sağlandı (ΔLeak_RD ortalama
+0,62 puan, %95 üst sınır +0,90 ≤ 1; ΔLeak_RN ortalama +3,46 puan, üst sınır +4,18 ≤ 5); çürütmenin nedeni yerellik
maliyeti değil, etkinin küçüklüğüdür. Betimsel olarak artış FUSED hedeflerde daha büyüktür (havuzlanmış gündelik başarı
dünya başına STD'de %6,7–24,1, PARA-ORTH'ta %11,7–36,0; kol ortalaması %14,9'a karşı %24,2, Tablo 8 ve 15), ama karar
ölçüsü bütün hedefleri kapsar. Yöntemde açıklandığı gibi biri-dışarıda kuralı yalnız FUSED hedeflerde görülmemiş bir
anahtar bırakır; S1 bu yüzden pratikte başka FUSED biçimlerin anahtarlarının dışarıda bırakılan FUSED biçime yardım
edip etmediğini ölçer ve bu ölçekte az yardım ettiğini gösterir.

#### S2: önek ortalaması ve sol bağlam

Tablo 9. S2, dünya başına: iki sol bağlamla yoklamalarda (CF14) STD ve ROOT.

Kaynak: `results/decisions.json` (hypotheses.S2.per_world); `results/tables/per_world_S2.csv`.

| Dünya | birim | nitelikli | d_w (puan) | CE14 STD (%) | CE14 ROOT (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1000 | 60 | evet | +21.2 | 89.2 | 68.0 |
| 1001 | 60 | evet | +20.0 | 85.8 | 65.8 |
| 1002 | 60 | evet | +12.4 | 90.7 | 78.3 |
| 1003 | 60 | evet | +0.5 | 78.2 | 77.7 |
| 1004 | 60 | evet | +14.3 | 71.2 | 56.9 |
| 1005 | 60 | evet | +13.2 | 90.5 | 77.3 |
| 1006 | 60 | evet | +4.3 | 84.4 | 80.1 |
| 1007 | 60 | evet | +12.9 | 80.5 | 67.6 |

**S2 belirsiz.** STD'nin bağlamlı yoklamalardaki başarısı ROOT'unkini ortalama 12,3 puan aştı (%83,8'e karşı %71,5;
Tablo 13). Altı dünyada fark marjın üstündeydi, iki dünyada (1003: 0,5; 1006: 4,3 puan) altındaydı (Tablo 9). Kesin
test p = 0,18 (Holm sonrası 0,73) verdi; t-aralığı [6,4; 18,2] marjı içerdiği için sonuç ne destek ne çürütmedir. Bu
karşılaştırma iki ardışık düzen arasındadır: STD ve ROOT anahtar ortalamasının yanında v* istemlerinde (5'e karşı 1),
katmanda (1'e karşı 0) ve adım büyüklüğünde (5'e karşı 0,5) de farklıdır. Fark STD lehinedir, ama önek ortalamasından
geldiği söylenemez.

#### S3: yaşam akışında ortak ve bağımsız durum (koşulmadı)

**S3 koşulmadı.** Sekiz doğrulayıcı yaşam koşulmadı; hipotez Holm ailesine p = 1 ile girdi. Yaşamlar protokolün hesap
önceliğinde (§14) dördüncü sıradaydı ve işlemci zamanı yeni bir araştırma programına ayrıldığında H1–H3, S1–S2 ve S4'ün
arkasına ertelendi (Sapmalar ve olaylar, 8). Bu karar sonuçtan bağımsız değildir: keşif yaşamlarının çöktüğü ve kayıtlı
teşhisin doğrulayıcı testin büyük olasılıkla "çürütüldü" sonucunu vereceğini öngördüğü biliniyordu ve bu, kararın iki
gerekçesinden biriydi. S3 bu yüzden ne desteklendi ne çürütüldü; önceden kayıtlı bir test, büyük olasılıkla olumsuz
sonucu bilinirken koşulmadan bırakıldı. Aşağıdaki kanıt keşif kanıtıdır; doğrulayıcı sonuçlarla birleştirilmez.

Tablo 10. Keşif yaşamları (yalnız J, λ ızgarası; 100 işlem, 75 inanılan olgu, 10 rastgele varlıkta 70 yoklama).

Kaynak: `results/lives/select_0.jsonl`, `results/lives/select_1.jsonl`; `results/tables/life_metrics.csv`.

| Keşif dünyası | λ | inanılan olgu | E_end (%) | RD'de değişen cevap | geri alma / düzeltme kalıntısı |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | 0.01 | 75 | 2.5 | 70/70 | 0.0 / 0.0 |
| 0 | 0.1 | 75 | 2.5 | 70/70 | 0.0 / 0.0 |
| 0 | 1 | 75 | 2.7 | 70/70 | 0.0 / 0.0 |
| 1 | 0.01 | 75 | 5.7 | 63/70 | 0.0 / 0.0 |
| 1 | 0.1 | 75 | 5.7 | 63/70 | 0.0 / 0.0 |
| 1 | 1 | 75 | 5.7 | 63/70 | 0.0 / 0.0 |

Tablo 11. Defter λ seçimi (iki keşif yaşamının ortalaması).

Kaynak: `results/selection.json` (life_lambda, life_lambda_table).

| λ | E_end (%) | Leak_RD (%) | yaşam | uygun (Leak_RD ≤ %1) | seçilen |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 0.01 | 4.1 | 95.0 | 2 | hayır |  |
| 0.1 | 4.1 | 95.0 | 2 | hayır |  |
| 1 | 4.2 | 95.0 | 2 | hayır | evet |

Keşif yaşamlarında varlık-ortak defter (J) üç λ değerinin hepsinde çöktü: yaşam sonu doğruluğu %2,5–5,7, rastgele
varlıklarda değişen cevap 70'te 63–70 (Tablo 10). Hiç yazım yapılmasaydı E_end %13,3 olurdu (aritmetik: yaşam
varlıkları düzenlenebilir kümeden seçilir ve temel model, geri alınmış 10 A olgusunun temel nesnesini yedi kalıpta da
üretir; 75 olgunun 10'u). J bu düzeyin de altına indi: geri alınıp temel nesnesine dönmesi gereken olgular da bozuldu
ve hiç yazım yapılmamış rastgele varlıklarda cevaplar değişti. Hiçbir λ Leak_RD ≤ %1 koşulunu sağlamadı; protokol (§13)
bu durumu tanımlamaz, dondurulmuş select_decide.py ise bu durumda bütün λ değerleri arasında en yüksek E_end'i verene
döner ve λ = 1'i seçti (Tablo 11). λ = 1'in üstünlüğü 1.050 yoklamada tek bir yoklamadır. Geri alma ve düzeltme
kalıntısının 0 olması temiz bir geri almanın değil, çöküşün sonucudur.

Teşhis, doğrulayıcı yaşamları erteleme kararından önce AMENDMENTS.md'ye kaydedildi (12:20 UTC; commit ad8ee92);
doğrulayıcı yaşamların hiçbiri başlamadı. Dondurulmuş hiçbir dosya ya da kayıt değişmedi. Teşhis betikleri ve ham
sayıları depoda değildir; aşağıdaki sayılar AMENDMENTS.md'deki özetten alınmıştır ve depodan yeniden üretilemez:

1. C0 konum başına bir ortalama olduğundan λ ≤ 1, KᵀC0⁻¹K'nin özdeğerlerinin (dünya 0'da 35,5–4.362) iki ila dört
   mertebe altında kalır. Izgara etkisizdir ve ortak çözüm tam interpolasyon sınırındadır: M(0,01) ile M(1) yaklaşık
   %1,3 farklıdır. MEMIT bu katsayıyı yaklaşık 10⁴ düzeyinde kullanır.
2. Çöküşün kendisi tek bir fc2 katmanının aşırı yüklenmesinden gelir: üst sınıra dayanmış yaklaşık 52 yazım (adım 5,
   üst sınır 8 × medyan fc2 normu), C0 konumlarında ΔW·h'nin RMS'ini doğal fc2 çıkışının yaklaşık 2 katına çıkarır; tek
   bir düzenlemede bu oran yaklaşık 0,2'dir. Dünya 0'da gerçek bir I durumu da çöker (E_end %6,5; RD 70/70).
3. 10⁻² ile 10⁵ arasındaki hiçbir λ ve önceden kayıtlı hiçbir adım, yararlı bir E_end ile Leak_RD ≤ %1'i birlikte
   vermedi. Bağımsız bir kod denetimi uygulama hatası bulmadı: dondurulmuş defter, son inanılan olgu kümelerinden
   yeniden kurulduğunda kayıtlı yaşamları birebir yeniden üretti.

Sonuç: dondurulmuş S3 testi koşulsaydı büyük olasılıkla "çürütüldü" sonucunu verecekti, ama bu bir ortak–bağımsız
durum karşılaştırması olmazdı: J iki keşif dünyasında, I teşhisteki tek bir koşuda (dünya 0) çöktü; Q ölçülmedi, onun
da çökeceği yalnız aynı enerji hesabından öngörülür. Bulgu Knowledge in Superposition (Hu vd., 2025) ya da
MEMIT-Merge (Dong vd., 2025) aleyhine bir kanıt değildir; yazımlar katmanın kapasitesini aştığında J'nin ve I'nın işe
yaramadığını gösterir. Yeniden ölçeklenmiş λ ya da daha küçük adımla yapılacak her yaşam koşusu keşif amaçlı olur.

#### S4: kimliğin derinlikle kök-son tokeninde toplanması

Tablo 12. Katmana göre τ_RN medyanı (kök-son token, cümle başı yalın önek, katmana özgü C0).

Kaynak: `results/tables/tau_by_layer.csv`; `results/decisions.json` (hypotheses.S4.per_world).

| Dünya | katman 0 | katman 1 | katman 2 | katman 3 | katman 4 | katman 0 − 4 | S4 d_w (decisions.json) | RN'li varlık (katman 0) | çift (katman 0) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.631 | 0.481 | 0.525 | 0.563 | 0.628 | +0.003 | +0.003 | 220 | 1110 |
| 1001 | 0.586 | 0.525 | 0.532 | 0.574 | 0.631 | -0.045 | -0.045 | 224 | 1142 |
| 1002 | 0.541 | 0.528 | 0.516 | 0.581 | 0.702 | -0.161 | -0.161 | 223 | 1272 |
| 1003 | 0.566 | 0.499 | 0.566 | 0.613 | 0.723 | -0.157 | -0.157 | 220 | 1112 |
| 1004 | 0.558 | 0.515 | 0.503 | 0.548 | 0.668 | -0.110 | -0.110 | 223 | 1114 |
| 1005 | 0.593 | 0.450 | 0.504 | 0.532 | 0.663 | -0.070 | -0.070 | 223 | 1164 |
| 1006 | 0.526 | 0.444 | 0.474 | 0.532 | 0.624 | -0.097 | -0.097 | 227 | 1070 |
| 1007 | 0.618 | 0.536 | 0.541 | 0.584 | 0.671 | -0.052 | -0.052 | 218 | 1104 |
| ortalama | 0.577 | 0.497 | 0.520 | 0.566 | 0.664 | -0.086 | -0.086 | — | — |

Medyan, dünya içinde havuzlanmış (varlık, komşu) τ değerleri üzerindendir; S4 d_w = medyan(katman 0) − medyan(katman 4);
"ortalama" satırı dünya medyanlarının ağırlıksız ortalamasıdır.

**S4 çürütüldü, ters yönde.** Kök-son tokenindeki anahtarın RN komşularına τ medyanı katman 0'dan katman 4'e düşmedi,
yükseldi: dünya medyanlarının ortalaması katman 0'da 0,577, katman 4'te 0,664'tür; etki −0,086, t-aralığı [−0,134;
−0,039] (Tablo 12). Dünya 1000 dışındaki yedi dünyada fark negatiftir. Katmanlar boyunca desen U biçimindedir: en düşük
değer katman 1'dedir (0,497), sonra artar. Büyük modellerde çok tokenli sözcüklerin ve adların erken–orta katmanlarda
son tokende toplandığı bildirilmiştir (Geva vd., 2023; Feucht vd., 2024; Kaplan vd., 2025); bu küçük modelde, bu τ
ölçüsüyle, kök-son tokenindeki anahtarın komşu adların anahtarlarından derinlikle ayrıştığı görülmedi. τ her katmanda o
katmanın kendi C0'ıyla ölçüldüğünden katmanlar arası karşılaştırma ortak bir ölçekte yapılmaz. Dünya 1000'in katman
medyanlarına analizden önce bakıldı (Sapmalar ve olaylar, 7); bakılan değerler tablodakilerle aynıdır ve karar kodu
SHA-256 özetiyle dondurulmuştu.

### Kol düzeyi betimsel ölçümler

Tablo 13–16 sekiz kolu doğrulayıcı dünyalarda betimler. Bunlar hipotez testi değildir; FT-L, FT-ALL, LAST–ROOT ve NORM
karşılaştırmaları protokolde keşif amaçlıdır. WE bütün 70 düzenleme üzerindendir; PE5 ve IE5 en az iki hedef hâlli
çekimli-yaşantılı düzenlemeler, CE14 çekimli-yaşantılı düzenlemeler üzerindendir; Leak_X düzenleme başına değişen cevap
payıdır; Δbpb 20 düzenlemede düzenlenmiş eksi temel bit/bayttır. Kollar ayrı seçilmiş katman ve adımlarla koşuldu
(Tablo 13); kol farkları bu yüzden yalnız anahtar kuralının değil, seçilen ardışık düzenin (katman, adım, istemler)
farklarıdır.

Tablo 13. Etkinlik ve genelleme.

Kaynak: `results/tables/edit_metrics.csv` (run = confirm, dünyalar 1000–1007).

| Kol | Katman | Yapılandırma | Dünya | Düzenleme | WE (%) | PE6 (%) | PE5 (%) | IE5 (%) | CE14 (%) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STD | 1 | cap_mult=8,step=5.0 | 1000–1007 | 560 | 95.7 (87.1…98.6) | 85.3 (70.0…91.9) | 84.1 (68.3…90.3) | 59.9 (50.2…65.7) | 83.8 (71.2…90.7) |
| ROOT | 0 | cap_mult=8,step=0.5 | 1000–1007 | 560 | 99.6 (98.6…100.0) | 89.8 (78.3…95.2) | 89.2 (75.6…95.0) | 57.3 (54.2…61.3) | 71.5 (56.9…80.1) |
| LAST | 1 | cap_mult=8,step=0.5 | 1000–1007 | 560 | 97.0 (91.4…100.0) | 13.5 (11.2…15.2) | 0.7 (0.0…1.7) | 0.4 (0.0…1.3) | 9.9 (9.0…12.0) |
| PARA-ORTH | 1 | cap_mult=8,step=5.0 | 1000–1007 | 560 | 95.2 (84.3…98.6) | 84.8 (70.2…91.0) | 83.6 (68.7…89.7) | 63.0 (51.5…69.0) | 83.1 (69.8…91.0) |
| FT-L | 0 | eps=0.02,lr=0.001,steps=60 | 1000–1007 | 560 | 100.0 (100.0…100.0) | 91.5 (87.1…96.0) | 91.1 (86.0…96.7) | 77.4 (72.9…83.7) | 54.5 (46.3…63.6) |
| FT-ALL | — | lr=0.0001,steps=40 | 1000–1007 | 560 | 100.0 (100.0…100.0) | 97.9 (95.7…99.3) | 97.8 (94.7…99.3) | 88.1 (84.8…93.0) | 76.1 (61.9…83.5) |
| NORM | 1 | cap_mult=8,step=5.0 | 1000–1007 | 560 | 95.7 (87.1…98.6) | 85.3 (70.0…91.9) | 84.1 (68.3…90.3) | 84.1 (68.3…90.3) | 83.8 (71.2…90.7) |
| BASE | — | none | 1000–1007 | 560 | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) |

Tablo 14. Sızıntı ve tutulmuş düzyazıda bpb değişimi.

Kaynak: `results/tables/edit_metrics.csv` (run = confirm, dünyalar 1000–1007).

| Kol | Leak_RN (%) | Leak_RD (%) | Leak_SS (%) | Leak_RB (%) | Leak_RC (%) | Δbpb (bit/bayt) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| STD | 34.0 (28.0…44.7) | 1.0 (0.2…1.8) | 2.9 (1.9…3.8) | 6.4 (3.5…9.2) | 26.2 (15.1…44.7) | +0.0098 (+0.0079…+0.0143) |
| ROOT | 13.5 (8.3…17.9) | 0.3 (0.0…0.7) | 1.1 (0.7…1.8) | 2.7 (0.0…5.3) | 14.3 (10.4…24.7) | +0.0024 (+0.0016…+0.0033) |
| LAST | 2.7 (1.2…3.5) | 0.4 (0.1…0.7) | 5.1 (1.1…7.2) | 0.4 (0.0…1.6) | 3.0 (0.0…8.8) | +0.0042 (+0.0028…+0.0072) |
| PARA-ORTH | 37.4 (31.0…48.4) | 1.6 (0.4…3.1) | 3.6 (2.1…5.0) | 7.9 (4.1…10.8) | 28.3 (16.9…45.5) | +0.0135 (+0.0114…+0.0183) |
| FT-L | 0.3 (0.0…0.7) | 0.1 (0.0…0.3) | 0.1 (0.0…0.2) | 0.1 (0.0…0.4) | 0.4 (0.0…1.4) | +0.0008 (-0.0003…+0.0019) |
| FT-ALL | 0.1 (0.0…0.4) | 0.1 (0.0…0.3) | 0.1 (0.0…0.3) | 0.0 (0.0…0.0) | 0.0 (0.0…0.2) | -0.0001 (-0.0003…+0.0002) |
| NORM | 34.0 (28.0…44.7) | 1.0 (0.2…1.8) | 2.9 (1.9…3.8) | 6.4 (3.5…9.2) | 26.2 (15.1…44.7) | +0.0098 (+0.0079…+0.0143) |
| BASE | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | — |

Tablo 15. Ayrıştırma: kurulmuş düzenlemeler (T-sürümleri), IDENT/NONIDENT hedef biçimleri (havuzlanmış) ve eski
nesnenin kalıntısı.

Kaynak: `results/tables/edit_metrics.csv` (run = confirm, dünyalar 1000–1007).

| Kol | TPE6 (%) | TPE5 (%) | TIE5 (%) | TCE14 (%) | PE IDENT (%) | IE IDENT (%) | PE NONIDENT (%) | IE NONIDENT (%) | eski nesne kalıntısı (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STD | 88.2 (79.0…94.6) | 87.2 (76.6…92.8) | 61.9 (55.7…66.1) | 85.8 (75.3…92.1) | 84.6 (69.4…90.7) | 84.1 (68.9…90.2) | 83.4 (65.6…92.9) | 14.9 (6.7…24.1) | 6.9 (3.3…14.5) |
| ROOT | 90.1 (79.5…95.2) | 89.6 (76.9…95.0) | 57.6 (55.1…61.3) | 71.8 (57.9…81.5) | 90.5 (80.4…95.2) | 87.7 (77.0…91.9) | 86.5 (64.4…94.6) | 1.1 (0.0…2.2) | 4.6 (2.1…13.1) |
| LAST | 13.9 (12.2…15.9) | 0.7 (0.0…1.7) | 0.4 (0.0…1.4) | 10.2 (9.0…12.4) | 0.6 (0.0…1.6) | 0.4 (0.0…1.1) | 0.9 (0.0…2.2) | 0.4 (0.0…2.2) | 84.8 (83.1…86.7) |
| PARA-ORTH | 87.7 (78.8…93.6) | 86.7 (76.2…93.0) | 65.5 (57.9…71.8) | 85.4 (74.7…92.5) | 84.2 (69.9…91.4) | 83.9 (67.9…90.2) | 82.6 (65.6…92.0) | 24.2 (11.7…36.0) | 6.8 (3.8…15.0) |
| FT-L | 91.5 (87.1…96.0) | 91.1 (86.0…96.7) | 77.4 (72.9…83.7) | 54.5 (46.3…63.6) | 91.4 (86.1…96.6) | 89.2 (82.8…94.1) | 90.4 (85.3…96.8) | 55.3 (45.9…65.8) | 5.8 (0.7…8.6) |
| FT-ALL | 97.9 (95.7…99.3) | 97.8 (94.7…99.3) | 88.1 (84.8…93.0) | 76.1 (61.9…83.5) | 97.9 (93.8…100.0) | 96.9 (91.9…99.5) | 97.6 (96.4…100.0) | 71.7 (63.8…82.0) | 1.4 (0.2…1.9) |
| NORM | 88.2 (79.0…94.6) | 87.2 (76.6…92.8) | 87.2 (76.6…92.8) | 85.8 (75.3…92.1) | 84.6 (69.4…90.7) | 84.6 (69.4…90.7) | 83.4 (65.6…92.9) | 83.4 (65.6…92.9) | 6.9 (3.3…14.5) |
| BASE | — | — | — | — | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) | 100.0 (100.0…100.0) |

Tablo 16. Ek sütunlar: RN komşularında yeni nesneyi benimseme, önek-uzantı eşine sızıntı (PX; dünya başına 7–13
düzenlemede tanımlı) ve çekimli-yaşantılı düzenlemelerde WE.

Kaynak: `results/tables/edit_metrics.csv` (Adopt_RN, Leak_PX, WE_inflected; run = confirm); PX sayıları
`results/edits/confirm_<dünya>_STD.jsonl` (leak.PX).

| Kol | RN'de yeni nesneyi benimseme (%) | Leak_PX (%) | WE, çekimli yaşantı (%) |
| :--- | ---: | ---: | ---: |
| STD | 18.6 (13.2…28.2) | 21.3 (1.2…47.1) | 95.4 (88.3…98.3) |
| ROOT | 6.7 (5.1…9.8) | 6.8 (0.0…24.7) | 99.6 (98.3…100.0) |
| LAST | 2.8 (1.3…5.6) | 3.0 (0.0…16.9) | 96.7 (91.7…100.0) |
| PARA-ORTH | 20.8 (14.7…31.9) | 24.1 (2.4…51.4) | 95.0 (86.7…98.3) |
| FT-L | 1.9 (0.4…4.5) | 5.5 (0.0…17.5) | 100.0 (100.0…100.0) |
| FT-ALL | 1.8 (0.4…4.5) | 6.2 (0.0…20.6) | 100.0 (100.0…100.0) |
| NORM | 18.6 (13.2…28.2) | 21.3 (1.2…47.1) | 95.4 (88.3…98.3) |
| BASE | 1.7 (0.4…4.4) | 0.0 (0.0…0.0) | 0.0 (0.0…0.0) |

Bütün yazım kolları yaşanan kalıpta yüksek etkinliğe ulaştı (WE %95,2–100). Kollar genellemede ayrışır:

- LAST, yani ek tokenine yazım, yaşanan biçimin dışına neredeyse hiç taşınmadı (PE5 %0,7, IE5 %0,4); eski nesne diğer
  kalıplarda büyük ölçüde kaldı (kalıntı %84,8).
- ROOT biçimsel hâllere STD'den biraz daha iyi genelledi (PE5 %89,2'ye karşı %84,1) ve komşulara daha az sızdı
  (Leak_RN %13,5'e karşı %34,0), ama sol bağlamda daha zayıftı (CE14 %71,5'e karşı %83,8) ve FUSED gündelik
  biçimlere neredeyse hiç taşınmadı (%1,1).
- STD ve PARA-ORTH aynı varlığın C ilişkisine de sızdı (Leak_RC %26,2 ve %28,3); KL terimi yalnız B ilişkisini korur
  (Leak_RB %6,4 ve %7,9). Özneye odaklı düzenlemelerin aynı öznenin başka ilişkilerine taşması büyük modellerde de
  bildirilmiştir (Liu vd., 2025). Tutulmuş düzyazıda bpb artışı küçüktür (temel ≈2,2 bit/bayt; en çok +0,0135, yani
  ≈%0,6, PARA-ORTH).
- NORM'un gündelik başarısı tanım gereği STD'nin biçimsel başarısına eşittir (IE5 = PE5 = %84,1) ve diğer bütün
  sütunları STD'ninkilerle aynıdır. Sözlükle normalleştirme H1 kaybını bu dünyada tümüyle kapatır; bunun koşulu ad
  sözlüğünün eksiksiz olmasıdır.
- FT-L ve FT-ALL (yaşanan cümle üzerinde ince ayar) kurulumda %100'e, biçimsel hedeflerde %91,1 ve %97,8'e, gündelik
  hedeflerde %77,4 ve %88,1'e ulaştı; FUSED gündelik hedeflerde de %55,3 ve %71,7 (STD %14,9). RN, RD, SS, RB ve RC
  sızıntıları %0,4 ya da altında kaldı (önek-uzantı eşlerinde %5,5 ve %6,2) ve RN'de yeni nesneyi benimseme temel model
  düzeyindedir (%1,9 ve %1,8; temel model %1,7). Buna karşılık sol bağlamda STD'den zayıftırlar (CE14 %54,5 ve %76,1).
  Bu keşif amaçlı karşılaştırma keşif seçimindeki gözlemle uyumludur (aşağıda); nedeni bu çalışmada sınanmadı. FT
  kolları defterle yönetilen sıra-bir yazımlar değildir: FT-L yoğun bir ΔW ekler, FT-ALL bütün ağırlıkları değiştirir;
  sürekli bir yaşamda nasıl davrandıkları ölçülmedi.

### Keşif seçimi ve bağlanma yeri haritası

Seçim 6.120 kayıttan, yalnız keşif dünyalarında yapıldı (Tablo 17); results/selection.json herhangi bir doğrulayıcı
düzenleme koşusundan önce sabitlendi. STD (katman 1, adım 5, üst sınır 8×) ve ROOT (katman 0, adım 0,5, üst sınır 8×)
"kısıtlar altında kurulamaz" etiketini aldı. STD'nin 30 yapılandırmasının hiçbiri Leak_C ≤ %5'i sağlamadı (en düşük
%7,5, katman 4'te, orada WE yalnız %17,5–20); ROOT'un en düşük Leak_C'si %5,7'ydi. Kural bu durumda en yüksek WE'yi
veren yapılandırmayı aldı ve doğrulayıcı STD ve ROOT koşuları bu yapılandırmalarla yapıldı. LAST, FT-L ve FT-ALL
kısıtları sağlayarak seçildi; PARA-ORTH ve NORM STD'nin seçimini kullanır. STD'nin keşif WE'si %100 olduğundan
yalnız-harita kuralı devreye girmedi.

Tablo 17. Keşif seçimi (iki keşif dünyası × 20 madde; durum: seçildi, kısıtlar altında kurulamaz ya da STD'nin seçimi).

Kaynak: `results/selection.json` (arms, map_only, n_records); `results/tables/discovery_map.csv`.

| Kol | Katman | Yapılandırma | WE (%) | Leak_RD (%) | Leak_C (%) | Leak_RN (%) | PE5 (%) | IE5 (%) | n (dünya×madde) | Durum |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| STD | 1 | cap_mult=8,step=5.0 | 100.0 | 1.2 | 25.4 | 36.3 | 86.3 | 61.1 | 40 | kısıtlar altında kurulamaz |
| ROOT | 0 | cap_mult=8,step=0.5 | 100.0 | 0.7 | 5.7 | 16.4 | 93.2 | 60.6 | 40 | kısıtlar altında kurulamaz |
| LAST | 1 | cap_mult=8,step=0.5 | 100.0 | 0.5 | 2.5 | 4.6 | 12.9 | 5.8 | 40 | seçildi |
| PARA-ORTH | 1 | cap_mult=8,step=5.0 | 97.5 | 2.0 | 28.2 | 41.1 | 83.3 | 63.5 | 40 | STD'nin seçimi (aynı δ; S1 yalnız anahtarları karşılaştırır) |
| FT-L | 0 | eps=0.02,lr=0.001,steps=60 | 100.0 | 0.0 | 0.0 | 4.1 | 93.7 | 78.3 | 40 | seçildi |
| FT-ALL | — | lr=0.0001,steps=40 | 100.0 | 0.0 | 0.4 | 3.9 | 99.0 | 88.2 | 40 | seçildi |
| NORM | 1 | cap_mult=8,step=5.0 | 100.0 | 1.2 | 25.4 | 36.3 | 86.3 | 61.1 | 40 | STD'nin seçimi (ön işleme) |

Keşif verisindeki gözlem (betimsel, önceden kayıtlı bir hipotez değil): iki kısıtı da sağlayan 33 yapılandırmanın
hepsi LAST'tan (14) ya da ince ayar kollarından (FT-L 17, FT-ALL 2) geldi; ad konumlarına (STD, ROOT, PARA-ORTH) ve son
istem tokenine (PRED) yazımın hiçbir yapılandırması kısıtları sağlamadı. Uygun yapılandırmalardan WE'si en az %90
olanlarda gündelik hedef başarısı (IE5) LAST'ta %5,0–11,7, FT-L'de %49,2–98,0, FT-ALL'da %88,2–97,0'dı; seçilen FT-L
(katman 0, ε = 0,02, lr = 10⁻³) ve FT-ALL (lr = 10⁻⁴) yapılandırmalarında %78,3 ve %88,2'ydi (STD %61,1, ROOT %60,6,
PARA-ORTH %63,5). PRED'e yapılan sıra-bir yazımlar da gündelik hedeflere benzer düzeyde ulaştı (IE5 %76,1–85,9), ama
rastgele varlıklara %2,2–74,8 sızdı. Seçim verisi doğrulayıcı veriden ayrıdır; doğrulayıcı kol ölçümleri aynı yönü
gösterir (Tablo 13), FT kolları ise sol bağlamda STD'den zayıftır (CE14 %54,5 ve %76,1; STD %83,8). Bu bir hipotez
testi değildir ve FT kolları sıra-bir, defterle yönetilen yazımlar değildir.

Tablo 18. Keşif bağlanma yeri haritası: dört konumun paylaştığı iki yapılandırmadan biri (adım 2, üst sınır 8×; 16×
satırları aynı dosyadadır); iki keşif dünyası × 20 düzenleme.

Kaynak: `results/tables/discovery_map.csv` (config = cap_mult=8,step=2.0; arm ∈ {ROOT, STD, LAST, PRED}). Izgaranın
tamamı aynı dosyadadır.

| Yer | Katman | WE (%) | PE6 (%) | PE5 (%) | IE5 (%) | Leak_RN (%) | Leak_RD (%) | Leak_C (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROOT | 0 | 97.5 | 89.2 | 88.8 | 59.1 | 41.3 | 1.4 | 18.2 |
| ROOT | 1 | 100.0 | 85.4 | 85.4 | 56.1 | 28.0 | 0.2 | 21.4 |
| ROOT | 2 | 95.0 | 77.1 | 75.6 | 51.6 | 7.2 | 0.1 | 13.6 |
| ROOT | 3 | 87.5 | 69.6 | 68.6 | 43.7 | 6.1 | 0.0 | 19.3 |
| ROOT | 4 | 22.5 | 15.4 | 15.4 | 10.0 | 4.1 | 0.0 | 7.1 |
| STD | 0 | 97.5 | 87.9 | 87.8 | 60.8 | 57.1 | 2.0 | 31.4 |
| STD | 1 | 97.5 | 85.8 | 85.3 | 57.0 | 34.2 | 0.5 | 17.5 |
| STD | 2 | 97.5 | 85.0 | 84.9 | 57.6 | 12.5 | 0.0 | 22.9 |
| STD | 3 | 80.0 | 60.4 | 59.3 | 45.6 | 7.5 | 0.2 | 27.9 |
| STD | 4 | 17.5 | 13.3 | 12.9 | 7.5 | 5.1 | 0.1 | 7.5 |
| LAST | 0 | 97.5 | 15.4 | 15.8 | 7.6 | 10.5 | 0.9 | 8.9 |
| LAST | 1 | 97.5 | 12.9 | 12.9 | 7.6 | 6.7 | 0.9 | 5.0 |
| LAST | 2 | 97.5 | 12.9 | 12.1 | 7.1 | 3.8 | 0.3 | 1.1 |
| LAST | 3 | 75.0 | 11.2 | 11.2 | 5.8 | 1.3 | 0.1 | 8.2 |
| LAST | 4 | 22.5 | 5.0 | 5.0 | 2.1 | 0.6 | 0.0 | 2.5 |
| PRED | 0 | 92.5 | 86.7 | 86.3 | 85.0 | 80.7 | 74.2 | 2.1 |
| PRED | 1 | 92.5 | 88.8 | 88.8 | 85.4 | 58.6 | 32.0 | 3.6 |
| PRED | 2 | 95.0 | 88.3 | 87.9 | 83.9 | 22.4 | 9.5 | 1.4 |
| PRED | 3 | 90.0 | 79.2 | 79.4 | 76.1 | 16.0 | 6.7 | 0.0 |
| PRED | 4 | 95.0 | 82.9 | 81.8 | 80.5 | 6.8 | 2.2 | 0.0 |

Ad konumlarına (ROOT, STD) yazım katman 0–2'de çalışır (WE %95–100), katman 3'te zayıflar ve katman 4'te büyük ölçüde
başarısız olur (%17,5–22,5). Son kök tokenini paylaşan komşulara sızıntı derinlikle azalır (STD'de %57,1'den %5,1'e),
ama genelleme de azalır. Ek tokeni (LAST) her katmanda yaşanan biçime kilitli kalır (PE6 ≤ %15,4). Son istem tokeni
(PRED) her katmanda çalışır ve gündelik biçimlere de biçimsel biçimler kadar taşınır (IE5 %76,1–85,4); bu, anahtarın ad
tokenlerinde değil ilişki konumunda alınmasıyla uyumludur (ayrıca sınanmadı). Bedeli rastgele varlıklara sızıntıdır
(katman 0'da %74,2, katman 4'te %2,2). Bu, ilişki konumuna yapılan düzenlemelerin başka öznelere taşma eğilimiyle
uyumludur (Liu vd., 2025). Harita
keşif kanıtıdır (dünya başına 20 düzenleme, yalın-yaşantılı düzenlemeler dahil).

### Birlikte okuma

Üç birincil sonuç tek bir mekanizma okumasıyla uyumludur: yazım, sorgudaki kök-son konumunun temsili yazılan anahtara
yakın olduğunda ateşlenir. Bu okumaya göre yazım kök tokenleri korunan biçimlere büyük ölçüde taşınır (H1'in biçimsel
tarafı, H2'nin IDENT tarafı), kök ile ekin kaynaştığı biçimlere çok daha az taşınır (FUSED gündelik başarı STD'de %14,9,
ROOT'ta %1,1) ve aynı son kök tokenini taşıyan başka adlarda da ateşlenir (H3). Kayıtlardaki τ değerleri bu okumayla
uyumludur (keşif amaçlı: STD'de gündelik hedef biçiminin τ'su başarıyı AUC 0,86 ile ayırır, dünya başına 0,77–0,90;
ayrımın çoğu IDENT/FUSED sınıfının kendisinden gelir: IDENT hedeflerde gündelik sorgunun anahtarı biçimsel sorgununkiyle
aynı olduğundan τ başarıyı ayırmaz, AUC 0,48; yalnız FUSED hedefler içinde AUC 0,69'dur). Okuma katmanlar arası örüntüyü
açıklamaz: τ_RN katman 1'den 3'e artarken (0,497 → 0,566; Tablo 12, doğrulayıcı dünyalar) keşif haritasında komşulara
sızıntı azalır (STD, adım 2: katman 1'de %34,2, katman 2'de aynı WE ile %12,5, katman 3'te %7,5; Tablo 18). Bu yüzden
okuma yalnız aynı katmandaki karşılaştırmalar için bir açıklamadır, ispat değildir; FUSED tokeni doğrudan değiştiren
nedensel bir müdahale yapılmadı. Biçim-ortalamalı anahtar (S1) bu bağımlılığı marj ölçüsünde gevşetmedi (+3,1 puan; marj
10 puan; artış küçük ama sıfırdan büyük); önek ortalamasının (S2) katkısı belirsiz kaldı. FT kollarının gündelik
biçimlere daha iyi genellemesi, genelliğin tek bir anahtar konumuna bağlı olmayan güncellemelerle mümkün olabileceğini
düşündürür, ama bu çalışmada sınanmış bir iddia değildir; ad tokenlerinin dışındaki tek bir konuma (PRED) yazım da
gündelik biçimlere genellenir, ama rastgele varlıklara sızar (Tablo 18). Bu okuma, anahtarların hem bağlama dayanıklı
hem özgül olmasının zorluğuna dair tartışmayla uyumludur (Yan vd., 2025).

## Doğrulama testleri

Doğrulama testleri hipotez testi değil, geçer/kalır denetimleridir (protokol §10.3); bir başarısızlık düzeltilene kadar
H ve S sonuçlarının yayımını engeller. Dondurulmuş 122 birim testi (tests/evolution/test_cev11_*.py, 6 dosya) çoğunlukla
küçük, rastgele modellerde (D=32, 2 katman) ve sentetik verilerde koştu ve hepsi geçti (38,46 s). V1, V2, V6 ve V8,
dondurmadan sonra eklenen verify_real.py ile çalışmanın kendi tabanlarında ve gerçek STD yazımlarıyla (katman 1, adım 5,
üst sınır 8×) tekrarlandı. Bu denetimler herhangi bir doğrulayıcı düzenleme koşusundan önce, dört dünyada (keşif 0 ve 1,
doğrulayıcı 1000 ve 1001) koştu; her doğrulayıcı koşu bunların ve birim testlerinin geçmesini bekledi. Tablo 19 her
testin nerede koştuğunu, Tablo 20 gerçek model sonuçlarını verir.

Tablo 19. V1–V12: ne denetlendi, nerede koştu.

Kaynak: `results/verification/unit_tests.txt` ("122 passed in 38.46s"); `results/verification/real_<dünya>.json`;
`results/bases/<taban>/gates.json` (G1) ve `meta.json` (tokenizer_identity); `run_confirm.py` (V10 denetimi, ihlalde
durur).

| Test | Ne denetler | Nerede koştu | Sonuç |
| :--- | :--- | :--- | :--- |
| V1 | Sıfır bellekle ve kanca sökülünce max\|Δlogit\| = 0 | Birim testi; verify_real.py dört dünyada (40 önek; ayrıca gerçek bir yazım sıfırlandıktan ve kurucunun geri yüklemesinden sonra) | Geçti; dört dünyada dört ölçüm de 0.0 |
| V2 | Kök-son tokenindeki fc2 girdileri 7 biçimsel ve IDENT biçimlerde aynı (kesik önekte fark tam 0; dolgulu tam sorguda göreli fark ≤ 1e-5) | Birim testi; verify_real.py dört dünyada (20 varlık, 6 katman, 208–219 biçim) | Geçti; kesik 0.0, tam dolgulu en çok 1.1e-06 |
| V3 | Yalın yaşantıda ROOT, LAST ve ortalamasız STD konumları çakışır; ROOT ve LAST yazımları bit düzeyinde aynı | Yalnız birim testi | Geçti; gerçek tabanlarda tekrarlanmadı |
| V4 | float64 defter: 100 ekleme, 30 düzeltme, 20 geri alma doğrudan çözüme eşit (≤ 1e-10); 5 sıra; silme; birleştirme | Birim testi (sentetik float64 matrisler) | Geçti |
| V5 | CEV-08 tarzı karşı örnek: yalnız W tutan bir durum iki geçmişi ayıramaz, olgu kayıtları ayırır | Birim testi | Geçti |
| V6 | Yazımdan sonra önbellekli ve önbelleksiz açgözlü çıktı aynı | Birim testi; verify_real.py dört dünyada | Geçti; 294–362 denetimde 0 uyuşmazlık |
| V7 | Tekrarlanan v* araması max\|Δ\| = 0; 8 eğitim adımı sabit iş parçacığında tekrarlanabilir | Birim testi | Geçti |
| V8 | Toplu tam üretim denetimi gerçek açgözlü çözümlemeyle uyuşur (anlaşmazlık ≤ %1) | Birim testi (mantık); verify_real.py dört dünyada, koşulardan önce, yalnız temel model ve STD yazımlarından alınmış örnek | Örnekte geçti: dünya başına 294–362 yoklamada 0 anlaşmazlık. Protokoldeki V8 (puanlanan yoklamaların %5'i) yapılmadı; S1 ve S2'nin dayandığı PARA-ORTH ve ROOT yoklamaları örneklenmedi |
| V9 | Morfoloji = 120 satırlık altın tablo ve bağımsız yeniden uygulama | Birim testi (altın satırlar ve üretilen her biçim) | Geçti |
| V10 | Düzenleme hedefi (varlık + yeni nesne, A ilişkisi) ve P3 hücreleri eğitim metninde geçmez | Birim testi (yerleştirilmiş ihlaller); her run_confirm.py koşusunun başında gerçek eğitim metinlerinde | Geçti; koşucu ihlalde durur, sekiz koşu da tamamlandı |
| V11 | Tokenleştirici değiştirilmemiş depo sınıfıdır; gidiş-dönüş %100 | Birim testi; G1 kapısı on tabanda | Geçti; G1 %100, kimlik on tabanda aynı |
| V12 | IDENT/RESEG/FUSED sınıflayıcısı elle etiketlenmiş durumları üretir (etiketleyen belirtilmemiş) | Birim testi | Geçti |

Tablo 20. Gerçek tabanlarda V1, V2, V6, V8 (verify_real.py; STD'nin seçilen katmanı ve yapılandırması; dünya başına 10
düzenleme; her düzenleme, durum ve aile için %5 örnek, en az bir yoklama, gerçekleşen ≈%8; tohum 0).

Kaynak: `results/verification/real_0.json`, `real_1.json`, `real_1000.json`, `real_1001.json`.

| Dünya | Katman | Yapılandırma | V1 (sıfır bellek / sökme / yazım sıfırlandıktan sonra / geri yüklemeden sonra max\|Δlogit\|) | V2 (kesik max\|fark\| / tam göreli) | V6 (uyuşmazlık/n) | V8 (anlaşmazlık/n) | Hepsi | Süre (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | cap_mult=8,step=5.0 | geçti (0.0 / 0.0 / 0.0 / 0.0; 40 önek) | geçti (0.0 / 9.0e-07; 219 biçim) | geçti (0/294) | geçti (0/294 = %0.00) | geçti | 480 |
| 1 | 1 | cap_mult=8,step=5.0 | geçti (0.0 / 0.0 / 0.0 / 0.0; 40 önek) | geçti (0.0 / 1.1e-06; 209 biçim) | geçti (0/332) | geçti (0/332 = %0.00) | geçti | 520 |
| 1000 | 1 | cap_mult=8,step=5.0 | geçti (0.0 / 0.0 / 0.0 / 0.0; 40 önek) | geçti (0.0 / 9.4e-07; 208 biçim) | geçti (0/308) | geçti (0/308 = %0.00) | geçti | 228 |
| 1001 | 1 | cap_mult=8,step=5.0 | geçti (0.0 / 0.0 / 0.0 / 0.0; 40 önek) | geçti (0.0 / 8.9e-07; 209 biçim) | geçti (0/362) | geçti (0/362 = %0.00) | geçti | 264 |

Gerçek modelde en büyük logit farkı dört durumun hepsinde (sıfır bellek, kancanın sökülmesi, gerçek bir yazımın
sıfırlanması ve kurucunun geri yüklemesi) ve dört dünyada da tam 0,0'dı. Kök-son tokenindeki fc2 girdileri, kesik
öneklerde 7 biçimsel ve IDENT gündelik biçimlerde altı katmanın hepsinde bit düzeyinde aynıydı; dolgulu tam sorgularda
göreli fark en çok 1,1·10⁻⁶'ydı (eşik 10⁻⁵). Önbellekli ve önbelleksiz açgözlü çıktılar 294–362 denetimin hiçbirinde
ayrışmadı. Toplu tam üretim denetimi gerçek açgözlü çözümlemeyle 294–362 örnek yoklamanın hepsinde uyuştu (13 aile: WF,
PF6, PF5, IF5, CF14, RN, RN'de benimseme, RD, SS, RB, RC, PX ve OLD; temel ve STD durumları havuzlanmış;
true_greedy_families boş).

Buradaki V8, protokoldeki "yoklamaların %5'i" ifadesinin dar bir uygulamasıdır: dünya başına 10 STD düzenlemesinin ve
temel modelin yoklamalarından her düzenleme, durum ve aile için %5 (en az bir yoklama) çekilmiş, koşulardan önce alınmış
bir örnektir. Dünya başına 294–362 yoklama, örneklenen havuzun yaklaşık %8'idir (dünya 1000 ve 1001'de %8,3 ve %7,7),
ama dünya 1000'de sekiz kolun puanladığı yaklaşık 100.000 yoklamanın yalnız ≈%0,3'üdür. Aile başına örnek 2–80
yoklamadır; aile başına eşik (en az 100 yoklama) bu yüzden hiç uygulanmadı. Diğer kolların (ROOT, LAST, PARA-ORTH'un
biri-dışarıda yazımları, NORM'un yeniden yazılmış istemleri, FT kolları) toplu denetimleri örneklenmedi; S1 ve S2
kararlarının dayandığı PARA-ORTH ve ROOT yoklamaları gerçek açgözlü çözümlemeye karşı denetlenmedi. run_confirm.py her
zaman toplu denetimi kullanır; protokoldeki "o aile gerçek açgözlü çözümlemeye geçer" çaresi kodda yoktur, anlaşmazlık
bulunmadığı için de tetiklenmezdi.

Analizin yeniden hesaplanması: `verify.py --rerun --freeze freeze_manifest.sha256`, özgün çalışma dizininde, 14:16 UTC
kaydı AMENDMENTS.md'ye eklenmeden önce bütün denetimleri geçti (results/verify_report.json, "ok": true). Manifestteki
261 dosyanın hiçbiri eksik ya da değişmiş değildi ve listelenmemiş dosya yoktu; kararlar JSONL kayıtlarından yeniden
hesaplandığında saklananlarla aynıydı (0 fark; H1, H2 ve H3 desteklendi, S1 çürütüldü, S2 belirsiz, S3 koşulmadı, S4
çürütüldü); tablolar aynıydı; karar için kullanılan analyze.py ve cev11/stats.py dosyalarının SHA-256 özetleri dondurma
manifestindekilerle aynıydı (9911f788…, 40d1f8f9…). Temiz bir klonda aynı komutun ne verdiği Yeniden üretim
bölümündedir.

Kaynak: `results/verify_report.json`.

## Sapmalar ve olaylar

Dondurmadan sonra hiçbir dondurulmuş dosya değişmedi. AMENDMENTS.md'deki dokuz kayıt, zaman sırasıyla aşağıdadır.
Saatler AMENDMENTS.md'deki yaklaşık saatlerdir; doğrulanabilir zaman, kaydı ekleyen commit'in zamanıdır. Üç kaydın
saati, onu ekleyen commit'ten 2–17 dakika sonradır: verify_real.py düzeltmesi "11:00" (commit 401c13c, 10:43 UTC), τ
açıklaması "12:45" (75c2bfa, 12:43 UTC), S3 ertelemesi "13:45" (e50ba02, 13:43 UTC).

1. **verify_real.py eklendi (dondurulmuş bir dosyanın değişikliği değil).** V1, V2, V6 ve V8'i çalışmanın kendi
   tabanlarında ve gerçek STD yazımlarında tekrarlayan bir doğrulama aracıdır (birim testleri bunları yalnız küçük
   modellerde kapsar; V8 gerçek yoklamalar ister). Dondurulmuş modülleri içe aktarır ve yalnız
   `results/verification/real_<dünya>.json` yazar. Dondurma manifestinde yoktur; verify.py --check onu kapsamda ama
   manifest dışı diye listeler.
2. **Keşif seçim ızgarası süreçlere bölündü (yalnız iş düzenlemesi; çalışma kodu söz konusu değil).** Dört çekirdeği
   kullanmak için her keşif dünyasının ana süreci STD'yi ve STD'nin bellekteki v* sonuçlarına ihtiyaç duyan PARA-ORTH'u
   koştu, ardından durduruldu; ROOT, LAST ve PRED/FT-L/FT-ALL tamamlanmış kayıtlardan devam eden ayrı run_select.py
   süreçleri olarak koştu. Kayda göre select_decide.py eksik ya da yinelenen kayıtları reddettiği için bölme seçimi
   değiştiremez; bu gerekçe 3. kayıtla daraltıldı.
3. **09:58 UTC olay: bir önizleme seçim dosyasını yazdı.** ROOT, LAST, PRED, FT-L ve FT-ALL'ın henüz seçim kaydı
   yokken `select_decide.py --allow-partial` ile yapılan betimsel bir bakış, yalnız ekrana yazdırmak yerine
   results/selection.json ve tables/discovery_map.csv dosyalarını yazdı. Neden: tamlık denetimi yalnız kaydı olan
   kolların (katman, yapılandırma) satırlarını karşılaştırır, hiç kaydı olmayan kolu fark etmez. İki dosya hemen
   silindi ve hiçbir iş onları okumadı: keşif yaşamları ve doğrulayıcı koşular, altı seçim işinin hepsi başarıyla
   bittikten sonra başlayan zamanlanmış select_decide.py koşusuna bağlıydı; run_confirm.py bir kolu eksik bir seçimi
   reddeder. Dondurulmuş betikteki boşluk kayda geçirildi ve değiştirilmedi. Bu olay 2. kayıttaki gerekçeyi daraltır:
   select_decide.py hiç kaydı olmayan bir kolu fark etmediği için, bölmenin seçimi değiştirememesinin gerekçesi
   zamanlanmış seçim koşusunun altı seçim işinin hepsi başarıyla bittikten sonra başlamasıdır.
4. **11:00 UTC verify_real.py, bağımsız bir incelemenin ardından ilk zamanlanmış koşusundan önce düzeltildi.** V8
   artık her aileden, puanlamanın kullandığı aile düzeyindeki toplu denetimden örnek alır (PF6'da OLD ve RN'de
   benimseme dahil; gerçek açgözlü çözümlemeyle puanlanan sızıntı yoklamaları atlanır ve sayılır). Temel ve
   düzenlenmiş durumu aile başına havuzlar, her anlaşmazlığı kaydeder ve anlaşmazlık olan aileleri protokolün çaresi
   olarak true_greedy_families diye listeler. Toplam anlaşmazlık en çok %1 ise geçer; aile başına eşik yalnız en az
   100 yoklama örneklenmiş ailelerde uygulanır. V1 artık gerçek bir yazım sıfırlandıktan ve kurucunun geri
   yüklemesinden sonra da logitleri denetler. V2'nin dolgulu denetimi her sorguyu kendi kesik anahtarıyla
   karşılaştırır. Betik yalnız V1 ya da V2 kalırsa sıfırdan farklı çıkış kodu verir; V6 ve V8 başarısızlıkları
   kaydedilir ve protokol gereği H ve S sonuçlarının yayımını engeller.
5. **11:38 UTC kayıtlı çıktı (değişiklik değil).** results/selection.json'un SHA-256'sı
   83cb20052a3a406c0c79a03a16c8a4006d76113dd9a658ca65fceb85e6cba5e8; dondurulmuş select_decide.py ile yalnız keşif
   dünyalarından, herhangi bir doğrulayıcı düzenleme koşusundan önce üretildi.
6. **12:20 UTC yaşam çöküşü teşhisi, hiçbir doğrulayıcı yaşam başlamadan önce kaydedildi** (keşif kanıtı; dondurulmuş
   dosya ya da kayıt değişmedi). Özeti S3 bölümündedir. Doğrulayıcı S3 testi dondurulduğu gibi kaldı; koşulsaydı
   büyük olasılıkla "çürütüldü" sonucunu verecek ve bir ortak–bağımsız durum sınaması olmayacaktı (J ve I'nın çöktüğü
   gözlendi, Q'nun çökeceği öngörüldü).
7. **12:45 UTC τ'ya erken bakış açıklandı (protokol §14 sapması: "doğrulayıcı koşular bitmeden hiçbir hipotez ölçüsüne
   bakılmaz").** 09:46 UTC'de yürütücü, ilk τ çıktısının biçimini denetlerken doğrulayıcı dünya 1000'in katman 0–4
   τ_RN medyanlarını yazdırdı (0,631; 0,481; 0,525; 0,563; 0,628). Bu, bir dünyanın S4 ölçüsüdür. Hiçbir karar, kod,
   seçenek ya da zamanlama buna bağlı değildi; analiz kodu SHA-256 özetiyle dondurulmuştu ve kararları mekanik olarak
   hesapladı. Başka hiçbir doğrulayıcı hipotez ölçüsüne bakılmadı; doğrulayıcı düzenleme kayıtları okunmadan commit
   edildi.
8. **13:45 UTC S3 yaşamları ertelendi (protokol §14 hesap önceliği).** Sekiz doğrulayıcı yaşam (§14'te öncelik 4)
   H1–H3, S1–S2 ve S4'ün arkasına ertelendi, çünkü çalışma ortamının işlemci zamanı yeni bir araştırma programına
   ayrıldı ve kayıtlı teşhis üç varyantın da önceden kayıtlı ayarlarda çökeceğini öngörüyordu. Rapor yazılırken
   koşulmamış oldukları için S3 "koşulmadı" olarak verilir. İleride koşulurlarsa kayıtlı λ = 1 ile, değiştirilmeden
   koşulacaklar.
9. **14:16 UTC kayıtlı çıktı (değişiklik değil).** Dondurulmuş analyze.py, doğrulayıcı dünyalar 1000–1007 üzerinde
   results/decisions.json'u yazdı (SHA-256 1d01e757…); verify.py --manifest ve --rerun --freeze uyuştu. Kayıt
   results/manifest.sha256'nın (261 dosya) SHA-256'sını
   750cb02a5d26d7560741990374561c0f8663e6a06490d99391097dd6a89955e0, results/verify_report.json'unkini
   07ecc5b3914f49002d7ad3cf9620131a9b0548758e4ceb82ecda387f9cec6a74 olarak verir (yeniden hesaplanan kararlar ve
   tablolar aynı, analiz kodu dondurulmuş özetlerle aynı). Bu kayıt manifest yazıldıktan sonra eklendi (manifest commit
   24f3544, 14:17 UTC; kayıt commit 9cbf8ce, 14:25 UTC); manifest bu yüzden AMENDMENTS.md'nin önceki sürümünü
   listeler (bkz. Yeniden üretim).

AMENDMENTS.md'de ayrı bir kayıt olarak bulunmayan, ama raporda açıkça söylenmesi gereken noktalar:

10. **Protokol gövdesi ile uygulanan kural.** Hakemin metni (protokolün gövdesi) ile dondurmadan önce protokolün başına
    yazılan düzeltmeler bazı yerlerde ayrışır: adım ızgarası {0,5; 2; 5} (gövdede {0,5; 2}), üst sınır çarpanı {8; 16}
    (gövdede 8), PARA-ORTH'un v*'ı (W k_PO + δ_STD), PARA-ORTH ve NORM'un STD'nin seçimini kullanması, C0'da BOS
    konumları. Rapor uygulanan kuralı anlatır.
11. **Bağlanma yeri haritası ve koşulmayan kalemler.** Protokol §10.5 dört dünyada 30'ar düzenlemeyle, nedensel
    izlemeyle ve bütün tabanlarda katmana göre τ ile bir harita öngörür ve bu haritayı her zaman raporlanan birincil
    betimsel çıktı sayar. Dondurmada yalnız keşif seçim ızgarasından türetilen haritanın raporlanmasına, doğrulayıcı
    haritanın (dünyalar 1000–1001), nedensel izlemenin ve F_freq kolunun yalnız hesap kalırsa koşulmasına karar
    verildi; bunlar koşulmadı. Karşılaştırma için: nedensel izleme tepesinin en iyi düzenleme katmanını öngörmediği
    büyük modellerde gösterilmiştir (Hase vd., 2023); bu, koşulmamasının gerekçesi değildir. Katmana göre τ yalnız
    sekiz doğrulayıcı tabanda hesaplandı. §10.4'ün keşif amaçlı kalemlerinden kural içi τ-AUC yalnız STD için, kalıp
    başına tablolar da yalnız hâl × IDENT/FUSED dökümü olarak raporlandı (Birlikte okuma ve Tablo 5a; bkz. 16).
    Kesme işaretsiz yaşanan cümleden yazımların biçimsel yoklanması; token sayısı ve küme boyutu başına tablolar;
    GPT-2 tarzı düzenli ifadeyle bölütleme raporlanmadı.
12. **Doğrulama kapsamı.** V3 gerçek tabanlarda tekrarlanmadı; V8'in kapsamı Doğrulama testleri bölümünde anlatıldığı
    gibi dardır (yalnız temel model ve STD yazımları; dünya başına 294–362 yoklama).
13. **RESEG kanıtı.** Protokolün başı, RESEG'in imkânsızlığı için "15.000 rastgele birleştirme tablosu"ndan söz eder;
    commit edilmiş özellik testi 150 tablo kullanır. Sonuç tokenleştiricinin birleştirme kuralından çıkar ve on tabanın
    hiçbirinde RESEG görülmedi.
14. **Keşif tabanlarında kod değişikliği.** İki keşif tabanının meta kaydında code_unchanged_during_run = false yazar:
    ön eğitimleri sürerken probes.py değişti ve experiment.py eklendi. Ön eğitim kodu (train.py, world.py,
    tokenizers.py, leadins.py, model.py) doğrulayıcı tabanlarınkiyle aynıdır; keşif tabanları yalnız seçim ve keşif
    kanıtı için kullanıldı (Kaynak: `results/bases/discovery_w0_K20/meta.json` ve `confirm_w1000_K20/meta.json`,
    code_files).
15. **Pilotlar.** Protokol §14, stratejist, eleştirmen ve hakem pilotlarının keşif kanıtı olarak ayrı raporlanmasını
    ister. Bu rapor pilotlardan yalnız düzyazı girişlerini gerekçelendiren sonucu aktarır (yazımsız modelde düzyazı
    önekinden sonra %0,9; v* aramasında 72 koşuda 0 yakınsama); adım ve üst sınır ızgaralarının gerekçesi olan pilotlar
    protokolün başında özetlenir. Pilotların kayıtları depoda değildir ve ayrı bir rapor olarak yayımlanmadılar.
16. **Rapor yazılırken yapılan keşif amaçlı hesaplar.** Doğrulayıcı kararlar hesaplanıp commit edildikten sonra, commit
    edilmiş kayıtlardan salt okunur olarak şunlar hesaplandı: hâl içinde IDENT/FUSED dökümü (Tablo 5a), δ'nın üst
    sınıra dayanma payı (diag.delta_norm ve diag.cap), keşif ızgarasında STD'nin aralıkları ve iki kısıtı sağlayan
    yapılandırmalar (`discovery_map.csv`), STD için τ-AUC ve V8 örneğinin kapsam oranları. Hiçbiri önceden kayıtlı bir
    test değildir ve kararları değiştirmez; betikler çalışma dosyalarıdır, depoda değildir.

## Sınırlar

- **Model ölçeği.** 3,4 milyon parametrelik, 6 katmanlı (D=192) küçük bir model ve tek katmana yazım. Katmanların
  rolleri, kapasite ve tokenleştirme büyük modellerde farklıdır; sonuçların büyük dil modellerine taşınması
  sınanmadı.
- **Sentetik ve kalıplı dünya.** 400 uydurma ad, yedi kalıp, yalnız özne-başı cümleler, düzenlemeler yalnız A
  ilişkisinde; düzyazı tek bir kitaptan ve bir klasörden gelir. Doğal metindeki genellik bilinmiyor. Kalıplar ve altın
  tablo protokol gereği depo sahibinin eşzamansız incelemesine sunuldu; bu incelemenin sonucu çalışma dosyalarında
  kayıtlı değildir.
- **Tek tokenleştirici.** Depodaki bayt düzeyinde BPE, bütün dünyalarda aynı 4.096 girdilik sözlük. Bu tokenleştiricide
  RESEG oluşamaz; H2 yalnız FUSED biçimleri sınar. H2'nin IDENT/NONIDENT karşıtlığı gözlemseldir; hâl içindeki keşif
  amaçlı döküm (Tablo 5a) farkın hâlden gelmediğini gösterir, ama hâl içinde de farklı adlar karşılaştırılır.
- **İşlem gücü ve istatistiksel güç.** CPU, tek iş parçacığı; n = 8 dünya. Kesin testin en küçük p değeri 1/256'dır;
  birincil kararlar bu alt sınırdadır (sekiz dünyanın hepsi marjın üstünde). İkincil ailede Holm'un ilk eşiği, marjın
  altında en çok bir dünyaya izin verir; S2'de iki dünya altta kaldı.
- **Dünyaların bağımsızlığı.** Sekiz doğrulayıcı dünya aynı tokenleştiriciyi, aynı eğitim düzyazısını, aynı kalıpları
  ve B ile C ilişkilerinin aynı nesne listelerini paylaşır; dünyalar arasında adlar, kasabalar, olgu atamaları,
  başlatma, veri sırası, düzenleme listeleri ve yoklama örneklemesi değişir. Çıkarım bu üreticiye ve bu
  tokenleştiriciye geneldir.
- **STD'nin yapılandırması.** STD keşifte iki sızıntı kısıtını da sağlamadı ("kısıtlar altında kurulamaz"; seçim tek
  bir keşif düzenlemesi farkıyla WE'ye göre yapıldı), doğrulayıcı düzenlemelerin %92'sinde δ üst sınıra dayandı ve
  yazım aynı varlığın C ilişkisine %26,2 (dünya başına %15,1–44,7) sızdı. H1–H3 bu yapılandırmanın etkileridir, temiz
  düzenlemelerin değil. Keşif ızgarasına göre H1'in büyüklüğü WE'si en az %90 olan STD yapılandırmalarında kararlıdır
  (22,8–28,3 puan), H3'ünki değildir (8,7–56,4 puan).
- **S2 karşılaştırması.** STD ile ROOT anahtar ortalamasının yanında v* istemlerinde, katmanda ve adım büyüklüğünde de
  farklıdır; S2 ardışık düzenleri karşılaştırır.
- **S1 tasarımı.** Biri-dışarıda kuralı yalnız FUSED hedeflerde görülmemiş bir anahtar bırakır; etkinlik ve yerellik
  farkları ayrı yazımlardan ölçülür (biri-dışarıda ve 13 biçimin hepsi).
- **NORM.** Tanım gereği bir tavan koludur; tam ve hatasız bir ad sözlüğü varsayar.
- **S3 koşulmadı.** Dondurulmuş λ ızgarası, C0 konum başına bir ortalama olduğu için etkisizdir; yazımların toplam
  enerjisi tek bir katmanı aşırı yükler. Doğrulayıcı yaşamlar, büyük olasılıkla olumsuz sonuçları bilinirken ertelendi
  (Sapmalar ve olaylar, 8). Gelecekteki bir yaşam sınaması yeniden ölçeklenmiş bir λ ya da daha küçük yazımlar
  gerektirir ve keşif amaçlı olur.
- **Doğrulama kapsamı.** V8, dört dünyada yalnız temel model ve STD durumlarından koşulardan önce alınmış bir örnektir
  (dünya başına 294–362 yoklama, puanlanan yoklamaların ≈%0,3'ü); S1 ve S2 kararlarının dayandığı PARA-ORTH ve ROOT
  yoklamaları gerçek açgözlü çözümlemeye karşı denetlenmedi. V3 yalnız birim testlerinde koştu.
- **Ölçüt.** Başarı, nesnenin ve noktanın tam açgözlü üretimidir; olasılıktaki kısmi kaymalar ölçülmedi. CF14'teki iki
  bağlam eğitimde görülmüştür; görülmemiş ya da yanıltıcı bağlamlar sınanmadı. v* aramasında kullanılan dört düzyazı
  kopyası bir dünyadaki bütün düzenlemeler için aynıdır.
- **Tek tek düzenleme.** Her düzenleme temel modele tek başına uygulanıp silindi; sıralı ya da toplu düzenleme
  sonuçları yalnız keşif yaşamlarından gelir.
- **Ön kayıt disiplini.** Bir dünyanın S4 ölçüsüne analizden önce bakıldı (Sapmalar ve olaylar, 7). Tasarım rolleri
  yapay zekâ ajanlarıdır; bağımsızlıkları süreç düzeyindedir; hiçbiri insan hakem değildir.
- **Literatür.** Tarama yalnız arama sonucu özetleri, makale özetleri ve bildiri kayıtları düzeyinde yapıldı; tam
  metinler okunmadı. "İlk" iddiası yapılmamıştır.
- **FT kolları.** FT-L ve FT-ALL karşılaştırması betimseldir; yalnız yaşanan cümleyle eğitildiler, sol bağlamda STD'den
  zayıftırlar ve sürekli bir yaşamda sınanmadılar.

## Yeniden üretim

Bütün komutlar depo kökünden ve tek iş parçacığında çalışır. Ortam: Python 3.13.18, torch 2.15.0+cu132 (CPU üzerinde,
torch_threads = 1), numpy 2.5.4, Linux x86_64; matplotlib kurulu olmadığından şekil üretilmedi. Bu makinede (4 çekirdek,
eşzamanlı işler) bir taban 70–131 dakikada eğitildi.

Kaynak: `results/bases/*/meta.json` (env, elapsed_s); `results/analysis_env.json`.

Tohumlar (protokol §12): dünya w için üreteç w, model başlatma w, veri sırası w+1, düzenleme listesi w+2, yoklama
örneklemesi w+3, düzyazı önek seçimi w+4, yaşam sırası w+5. Keşif dünyaları 0–1, doğrulayıcı dünyalar 1000–1007,
yedekler 1008–1009. Analiz: bootstrap ve permütasyon 10.000'er, tohum 0. verify_real.py: tohum 0, her düzenleme, durum
ve aile için %5 örnek (en az bir yoklama), 10 düzenleme, 20 V2 varlığı.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
S=research/living_learning_paradigm_2026_11_13

# 0. Dondurulmuş kod ağacı (32 dosya) ve dondurulmuş birim testleri (122 test)
(cd $S && sha256sum -c freeze_manifest.sha256)
python -m pytest -q -p no:cacheprovider tests/evolution/test_cev11_world.py tests/evolution/test_cev11_tokenizers.py \
  tests/evolution/test_cev11_core.py tests/evolution/test_cev11_stats.py tests/evolution/test_cev11_memory.py \
  tests/evolution/test_cev11_experiment.py

# 1. Ön eğitim ve kapılar (K = 20): keşif 0–1, doğrulayıcı 1000–1007
python $S/run_pretrain.py --world 0 --passes 20 --tag discovery
python $S/run_pretrain.py --world 1 --passes 20 --tag discovery
for w in 1000 1001 1002 1003 1004 1005 1006 1007; do
  python $S/run_pretrain.py --world $w --passes 20 --tag confirm
done

# 2. Keşif seçim ızgarası (yalnız 0–1; STD ile PARA-ORTH aynı süreçte), sonra geçici seçim
for w in 0 1; do
  python $S/run_select.py --base results/bases/discovery_w${w}_K20 --world $w --arms STD PARA-ORTH
  python $S/run_select.py --base results/bases/discovery_w${w}_K20 --world $w --arms ROOT
  python $S/run_select.py --base results/bases/discovery_w${w}_K20 --world $w --arms LAST
  python $S/run_select.py --base results/bases/discovery_w${w}_K20 --world $w --arms PRED FT-L FT-ALL
done
python $S/select_decide.py

# 3. Keşif yaşamları (defter λ'sı), sonra kayıtlı seçim (results/selection.json)
for w in 0 1; do
  python $S/run_life.py --base results/bases/discovery_w${w}_K20 --world $w --run select --variants J --lams 0.01 0.1 1
done
python $S/select_decide.py

# 4. Gerçek modelde V1, V2, V6, V8
for w in 0 1; do python $S/verify_real.py --base results/bases/discovery_w${w}_K20 --world $w --selection; done
for w in 1000 1001; do python $S/verify_real.py --base results/bases/confirm_w${w}_K20 --world $w --selection; done

# 5. Doğrulayıcı düzenlemeler (8 kol × 70 madde) ve katmana göre τ
for w in 1000 1001 1002 1003 1004 1005 1006 1007; do
  python $S/run_confirm.py --base results/bases/confirm_w${w}_K20 --world $w
  python $S/run_tau.py --base results/bases/confirm_w${w}_K20 --world $w
done
# S3 yaşamları (bu çalışmada koşulmadı):
#   python $S/run_life.py --base results/bases/confirm_w${w}_K20 --world $w --variants J I Q

# 6. Analiz ve doğrulama (verify.py --manifest, results/manifest.sha256'yı yeniden yazar; yalnız denetim yapılacaksa bu adımı atlayın)
(cd $S && python analyze.py)
(cd $S && python verify.py --manifest && python verify.py --rerun --freeze freeze_manifest.sha256 --report results/verify_report.json)
```

Beklenen özetler (SHA-256). Ağırlık özeti, cev11.model.model_sha256 işlevinin hesapladığı kanonik özettir (sıralı
state_dict: ad, şekil ve fp32 baytlar); meta.json'da ve ön eğitim günlüğünde yazar ve model.pt dosyasının bayt
özetinden farklıdır.

Kaynak: `results/bases/*/meta.json` (model_sha256, tokenizer_identity), `results/bases/*/leadins.json` (pool_sha256),
`results/manifest.sha256` (model.tokenizer.json), `results/edits/items_<dünya>.json` (sha256); sha256sum ile
`results/selection.json`, `results/decisions.json`, `analyze.py`, `cev11/stats.py`.

| Çıktı | SHA-256 |
| :--- | :--- |
| results/selection.json | `83cb20052a3a406c0c79a03a16c8a4006d76113dd9a658ca65fceb85e6cba5e8` |
| results/decisions.json | `1d01e757393d8243be922fba34b793d00f9264f04dd2a748d163fec004dfb426` |
| analyze.py | `9911f7887e25541a846e89141ccbdc25a9a66292c098662b4012689de035aa15` |
| cev11/stats.py | `40d1f8f976159727920ba4eca2bea491aaf99626957645d04bc6b5daa7506589` |
| model.tokenizer.json (on tabanda aynı dosya) | `2ee08a198f84644895d1a1f2b42863a97036fb69c3f1bd4fe77a64e31d7fcf4d` |
| tokenleştirici kimliği | `4db9febe26c3f9605d931751e54cbcb770a83b486704d70149b1d4d34f6e7b0f` |
| giriş cümlesi havuzu (4.490 cümle) | `c4a4bdfad3531718e6e3cd822cbcca784de50e3ca294082fb5727841e1dd225b` |
| ağırlık özeti, discovery_w0_K20 | `eae4b0f87469bd7ae802a4a05fb761010fed12891560260ff30a89430ba6a8dc` |
| ağırlık özeti, discovery_w1_K20 | `c4ed303b11c82b2301f6a293b4c0fd575673d8ad4259931e6e6627409d1e965c` |
| ağırlık özeti, confirm_w1000_K20 | `29bc5283dc7194ebefeea43dc69e0d5aa89c15b33a2ba7435ed1a7fd4a186eb5` |
| ağırlık özeti, confirm_w1001_K20 | `9008b75dc21f0790b01072b576000a3abbd38858bc0bb5448ca0b745d4b85eb8` |
| ağırlık özeti, confirm_w1002_K20 | `908a36440e13dc6af77135c85eb15dc1b937b851665fc92b39afdf8629e9c4b0` |
| ağırlık özeti, confirm_w1003_K20 | `9a223efb63e7153f10fac4c7f28847181a8b841a5b7ee4b38fe0e3b917368308` |
| ağırlık özeti, confirm_w1004_K20 | `c544757b86a522fabf8cfb461d687092d0b9c70d771039592ea4abe5441b560a` |
| ağırlık özeti, confirm_w1005_K20 | `e41100784c8320c4bf988ae248c31b06c857c13cec097dd8224a375170860223` |
| ağırlık özeti, confirm_w1006_K20 | `aa9cac82c9f8f302e76a55c7abb31c54712606cbaadc924abc6cfe58a07cc727` |
| ağırlık özeti, confirm_w1007_K20 | `c297dbcbd72e2627475cc2634f74f706d4b474b7993a453bb48ecd570f0c6aab` |
| düzenleme listesi, dünya 1000 | `737150b2a469c627a7285749f2b930737d9965a4c1762f387a15967289c4caa1` |
| düzenleme listesi, dünya 1001 | `c73c97eac5102d9ada1457833c1b3413d4560f3c5f15467aaf8709a32bfaa3ee` |
| düzenleme listesi, dünya 1002 | `bcb65b008b605df9e593a6d8b2c68a0d451f4a5ff7fb744a397d5fa41b124c75` |
| düzenleme listesi, dünya 1003 | `4c5ccffa1a244cbe65f1a1244ccea6fddff2182071ea5cbbc3cd660ea3762eb8` |
| düzenleme listesi, dünya 1004 | `d2784c7bb10210f70aad3649661ee54f6b86a1b81dd05af921e567222ad1ce0c` |
| düzenleme listesi, dünya 1005 | `de7740135f7a0cf23d4de93c9aab107fe469125cf61b6ae117fd7a49d1dc40c5` |
| düzenleme listesi, dünya 1006 | `9fd5045e48a4e558a4ca53526a891870c66f48a33253091bb2bc089cdbd6d1ab` |
| düzenleme listesi, dünya 1007 | `3b115247661c24b7274dc648cfdffb6309e7454d5c94598fda93bf4a3561dff4` |

Ağırlıklar (taban başına model.pt) protokol gereği yayımlanmaz (§5); maruz kalma kayıtları (exposure.json) ve ön eğitim
günlükleri (pretrain.log) protokolün gerektirmediği hâlde depo kuralları ve boyut nedeniyle yerel tutuldu (depo
genelindeki `*.log` kuralı ve `results/.gitignore`). results/manifest.sha256 ise bu 30 dosyayı da listeler; ayrıca 14:16
UTC kaydı eklenmeden önceki AMENDMENTS.md'yi listeler (Sapmalar ve olaylar, 9). Bu yüzden temiz bir klonda
`verify.py --rerun --freeze freeze_manifest.sha256` çıkış kodu 1 verir: manifest denetimi bu 30 yerel dosyayı eksik,
AMENDMENTS.md'yi değişmiş, sonradan yazılan results/verify_report.json'u da listelenmemiş sayar. Buna karşılık kararlar
ve tablolar commit edilmiş JSONL kayıtlarından yeniden hesaplandığında saklananlarla aynıdır (0 fark), dondurma denetimi
geçer ve analiz kodu dondurulmuş SHA-256 özetleriyle eşleşir; bunu rapor yazılırken temiz bir çalışma ağacında
denetledik. Tam yeniden üretim ön eğitimden başlar. Aynı ortamda yukarıdaki ağırlık özetleri beklenir, ama tam bir ön
eğitimin bit düzeyinde tekrarlanabilirliği ayrıca sınanmadı (V7 yalnız 8 eğitim adımını sınar); farklı donanım ya da
kütüphane sürümleri kayan nokta farkları üretebilir. Rapordaki tablolar commit edilmiş sonuç dosyalarından salt okunur
yardımcı betiklerle üretildi (bu betikler çalışma dosyalarıdır, depoda yer almaz). Karar ve dünya tabloları dosyalardaki
değerlerin yeniden biçimlendirilmesidir; kol tabloları (13–16) edit_metrics.csv'deki dünya değerlerinin ortalaması ve en
küçük–en büyük değeridir; Tablo 10 yaşam kayıtlarından sayılmıştır. Tablo 5a ve öteki keşif amaçlı ek hesaplar (Sapmalar
ve olaylar, 16) da commit edilmiş kayıtlardan aynı biçimde hesaplandı. Her tablonun kaynağı, tablo başlığının altındaki
"Kaynak:" satırındadır.

## Kaynaklar

Bütün girişler arama motoru sonuç özetleri, makale özet sayfaları ve bildiri ya da Anthology kayıtları düzeyinde
denetlendi; tam metinler okunmadı. Köşeli parantezli notlar denetimin eksik kaldığı yerleri gösterir.

- Cao, Y., & Yang, J. (2015). Towards making systems forget with machine unlearning. *2015 IEEE Symposium on Security
  and Privacy*, 463–480. https://doi.org/10.1109/SP.2015.35
- Cohen, R., Biran, E., Yoran, O., Globerson, A., & Geva, M. (2024). Evaluating the ripple effects of knowledge editing
  in language models. *Transactions of the Association for Computational Linguistics, 12*, 283–298.
  https://doi.org/10.1162/tacl_a_00644
- Dong, Z., Shen, X., & Xia, R. (2025). MEMIT-Merge: Addressing MEMIT's key-value conflicts in same-subject batch
  editing for LLMs. *Findings of the Association for Computational Linguistics: ACL 2025*, 7952–7960.
  https://doi.org/10.18653/v1/2025.findings-acl.415
- Feucht, S., Atkinson, D., Wallace, B., & Bau, D. (2024). Token erasure as a footprint of implicit vocabulary items in
  LLMs. *Proceedings of EMNLP 2024*, 9727–9739. https://doi.org/10.18653/v1/2024.emnlp-main.543
- Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). Dissecting recall of factual associations in
  auto-regressive language models. *Proceedings of EMNLP 2023*, 12216–12235.
  https://doi.org/10.18653/v1/2023.emnlp-main.751
- Gupta, A., Baskaran, S., & Anumanchipalli, G. (2024). Rebuilding ROME: Resolving model collapse during sequential
  model editing. *Proceedings of EMNLP 2024*, 21738–21744. https://doi.org/10.18653/v1/2024.emnlp-main.1210
- Hase, P., Bansal, M., Kim, B., & Ghandeharioun, A. (2023). Does localization inform editing? Surprising differences
  in causality-based localization vs. knowledge editing in language models. *Advances in Neural Information Processing
  Systems 36 (NeurIPS 2023)*. arXiv:2301.04213
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2),
  65–70. https://doi.org/10.2307/4615733
- Hu, C., Cao, P., Chen, Y., Liu, K., & Zhao, J. (2025). Knowledge in superposition: Unveiling the failures of lifelong
  knowledge editing for large language models. *Proceedings of the AAAI Conference on Artificial Intelligence, 39*(22),
  24086–24094. https://doi.org/10.1609/aaai.v39i22.34583
- Kaplan, G., Oren, M., Reif, Y., & Schwartz, R. (2025). From tokens to words: On the inner lexicon of LLMs.
  *International Conference on Learning Representations (ICLR 2025)*. arXiv:2410.05864 [Bir özet, sonraki bir arXiv
  sürümünün başlığını farklı veriyor; atıfta okunan sürüm belirtilmelidir.]
- Krause, B., Kahembwe, E., Murray, I., & Renals, S. (2018). Dynamic evaluation of neural sequence models. *Proceedings
  of the 35th International Conference on Machine Learning (ICML 2018)*, PMLR 80, 2766–2775.
- Liu, X., Liu, Z., Gu, N., Lin, Z., Ma, W., Xiang, J., & Wang, W. (2025). Relation also knows: Rethinking the recall
  and editing of factual associations in auto-regressive transformer language models. *Proceedings of the AAAI
  Conference on Artificial Intelligence, 39*(23), 24659–24667. https://doi.org/10.1609/aaai.v39i23.34646
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning
  Representations (ICLR 2019)*. arXiv:1711.05101
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *Advances
  in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2202.05262
- Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., & Bau, D. (2023). Mass-editing memory in a transformer.
  *International Conference on Learning Representations (ICLR 2023)*. arXiv:2210.07229
- Mousi, B., Durrani, N., & Dalvi, F. (2025). An exploration of knowledge editing for Arabic. *Proceedings of the Third
  Arabic Natural Language Processing Conference (ArabicNLP 2025)*, 417–424. arXiv:2507.09629
- Park, H., Choi, G., Kim, M., & Jo, Y. (2025). Context-robust knowledge editing for language models. *Findings of the
  Association for Computational Linguistics: ACL 2025*, 10360–10385. https://doi.org/10.18653/v1/2025.findings-acl.540
- Press, O., & Wolf, L. (2017). Using the output embedding to improve language models. *Proceedings of EACL 2017
  (Volume 2: Short Papers)*, 157–163. arXiv:1608.05859
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative
  pre-training. OpenAI teknik raporu. [Yalnız 0,02 başlatma değeri için; ikincil kaynaklardan denetlendi.]
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised
  multitask learners. OpenAI teknik raporu. [Derinlik ölçeklemesi ve bayt düzeyinde BPE için; ikincil kaynaklardan
  denetlendi.]
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units.
  *Proceedings of ACL 2016 (Volume 1: Long Papers)*, 1715–1725. https://doi.org/10.18653/v1/P16-1162
- Shazeer, N. (2020). GLU variants improve transformer. arXiv:2002.05202
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). RoFormer: Enhanced transformer with rotary position
  embedding. *Neurocomputing, 568*, 127063. https://doi.org/10.1016/j.neucom.2023.127063 [arXiv v5 aynı altı yazarı
  başka sıra ve yazımla verir.]
- Tang, J., Zhuang, H., Fang, D., Li, J., Han, F., Huang, Y., Fan, K., Wang, L., Zhu, Z., Zhang, S., Song, H. H., &
  Liu, Y. (2025). ACU: Analytic continual unlearning for efficient and exact forgetting with privacy preservation.
  arXiv:2505.12239 [Ön baskı; hakemli bir yayın bulunamadı; PDF sürümünün başlığı farklıdır.]
- Türk Dil Kurumu. *Yazım Kılavuzu*, "Kesme işareti" maddesi. https://tdk.gov.tr [Yalnız ikincil kaynaklar
  üzerinden denetlendi; çevrim içi kılavuzun belirli bir baskısına ya da erişim tarihine bağlanmadı.]
- Wang, W., Haddow, B., & Birch, A. (2024). Retrieval-augmented multilingual knowledge editing. *Proceedings of the
  62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 335–354.
  https://doi.org/10.18653/v1/2024.acl-long.21 [Arama özeti düzeyinde denetlendi; MzsRE'nin makine çevirisiyle kurulduğu
  ve Türkçeyi de içeren 12 dili kapsadığı özetlerden.]
- Wang, X., Lyu, S., Gu, Y., Huang, J., Lu, P., Cui, Y., & Chang, X.-W. (2026). Beyond hard writes and rigid
  preservation: Soft recursive least-squares for lifelong LLM editing. *Proceedings of IJCAI-26*, 2382–2390.
  https://doi.org/10.24963/ijcai.2026/265
- Yan, J., Wang, F., Luo, Y., Li, Y., & Zhang, Y. (2025). Keys to robust edits: From theoretical insights to practical
  advances. *Proceedings of ACL 2025 (Volume 1: Long Papers)*, 22545–22560. arXiv:2410.09338
- Yang, W., Sun, F., vd. (2024). The fall of ROME: Understanding the collapse of LLMs in model editing. *Findings of
  EMNLP 2024*, 4079–4087. arXiv:2406.11263 [Yazar listesinin tamamı görülmedi.]
- Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *Advances in Neural Information Processing
  Systems 32 (NeurIPS 2019)*. arXiv:1910.07467
- Zhu, C., Rawat, A. S., Zaheer, M., Bhojanapalli, S., Li, D., Yu, F., & Kumar, S. (2020). Modifying memories in
  transformer models. arXiv:2012.00363

Depo içi: [CEV-08](../living_learning_correction_2026_09_20/REPORT_TR.md) "Düzeltilebilir öğrenme durumu: bugünkü
cevap neyi korumaz?" (20 Eylül 2026); [CEV-09](../living_learning_query_state_2026_09_21/REPORT_TR.md) "Gelecekteki
öğrenme işlemlerine göre gerekli durum: kesin sonlu bir denetim" (21 Eylül 2026).
