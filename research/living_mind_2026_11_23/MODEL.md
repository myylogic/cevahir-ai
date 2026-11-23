# Yaşayan Genel Zekâya Doğru: Cevahir Zihin'in matematiksel modelinin derinleştirilmesi

*Muhammed Yasin Yılmaz, 23 Kasım 2026.*

Bu bir tasarım ve kuram metnidir. Henüz kurulmamış düzeyleri matematiğiyle tanımlar ve nasıl ölçüleceklerini söyler.
Hangi sırayla kurulacaklarını da verir. Ölçülmüş her sayının kaynağı yanında yazılıdır. Kurulmamış hiçbir şey için
"yapıyor" denmez; böyle yerlerde "beklenen" denir. Bugünkü çekirdeğin kuramı
[docs/ZIHIN_THEORY.md](../../docs/ZIHIN_THEORY.md) dosyasındadır. Bu metin onun üzerine kurulur ve onun hiçbir
kuralını değiştirmez.

## 0. Neden bu metin

Sahibin sözü (23 Kasım 2026): *"İnsan zihni sadece zihinle çalışmaz. Şu ana kadar yaptığımız çalışmadaki
matematiksel modellemeleri derinleştirelim. İnsan zihnini algoritmalara döküyoruz; ancak böyle yaşayan genel zekâya
ulaşırız. Dizilerde kelime hesaplaması bir zekâ örneği değildir."*

### 0.1 Ölçülen: hatırlama güçlü, genelleme zayıf

* **Etüt** (`research/living_mind_2026_11_22/etut.py`). Sahibin verileriyle büyümüş en büyük zihinle yapıldı: 694.512
  an ve 4,8 milyon bağ. Zihin 24 dersin 284 diyaloğunun hepsini ilk turda doğru cevapladı (284/284). Bütün dersler
  bittikten sonraki genel tekrarda da sonuç 284/284 oldu.
* **Sınav satırları.** Bunlar derste hiç sorulmamış sorulardır; cevapları derste söylenen bilgilerden çıkar. Her dersin
  hemen ardından ölçülen sonuç 8/41'di. Bütün dersler bittikten sonra 11/41 oldu (F75 kapalıyken de açıkken de).
* **Sahibin örneğinin dünyası.** Dünyada şunlar öğretildi: "ali ata bak" → "ali ata bakıyor" biçiminde 14
  komut–betimleme çifti, 6 bilgi cümlesi ve bir "kedi nedir ?" diyaloğu. Ölçüm bütün zihinle, bugünkü kodla yapıldı
  (`t_paradigm.py`; satırı `results/paradigm.jsonl`).
  * Bilinen fiil ve yeni çift ya da yeni nesne (3 soru): 1 doğru ("can ata bak → can ata bakıyor"), 2 sessiz.
  * Hiç betimlenmemiş fiil (3 soru): 0 doğru. İkisi sessiz kaldı, birinde fiil yanlış ("ali eve dön → ali eve
    geliyor"). Bu üç soru hiç duyulmamış bir kelime biçimi ister ("dalıyor", "tutuyor", "dönüyor"). §7.4 bunu ayrıca
    tartışır.
  * "at nedir ?": cevap öğretilen üç bilgiden yalnız ilkiydi: "at bir hayvandır".

### 0.2 Teşhis

Bugünkü zihnin birimleri kelimeler ve yaşanmış kelime dizileridir. Bağları şunları söyler: neyin neyin ardından
geldiği (gecikme 1, 2), neyin birlikte olduğu (gecikme 0), neyin neyin parçası olduğu ve neyin hangi kategorinin üyesi
olduğu. Bir soru, sorunun kelime dizisi sürdürülerek cevaplanır. Sıradaki kelimeyi tek kuralla yarış seçer.
Genellemeleri de (çerçeveler, kategoriler, aileler, kalıtım) kelime dizilerindeki yerler üzerinedir. Bu yüzden:

1. **Gönderge yok.** Zihin için "at", ardından başka kelimeler gelen bir kelimedir. Bakılabilen, dört ayaklı bir şey
   değildir.
2. **Olay yok.** "ali ata bak" bu sırayla gelen üç kelimedir. "Ali, bakmak, at" ilişkisi kelimelerin sırasından ve
   biçiminden bağımsız tutulmuyor. "ali ata bakıyor" da yaşanması gereken başka bir dizidir. Yeni bir çift için "zeynep
   suya vuruyor" diyebilmek, iki yeri aynı anda genelleyen bir çerçeve ister. Bu, COGS'ta ölçülen birinci nedendir
   (görev 82 ve 90).
3. **Kavram bir toplanma noktası değil.** Atlar hakkında bilinen, "at", "ata", "atın" ve "atlar" ile başlayan dizilere
   dağılmıştır. Zihin bunların aynı şey olduğunu yalnız çerçevelerin yavaş örtüşmesinden sezebilir.
4. **Söylenecek bir anlam yok.** Konuşmak bir sürdürmedir, bir anlamı dile getirmek değildir. Bu yüzden "at nedir"
   sorusunda ilk yaşanmış cümlenin ötesine geçemiyor: üç bilgi üç ayrı dizi olarak yaşandı.
5. **Beden, dünya ve amaç yok.** Zihin kelimeleri algılıyor; görüntü ve sesi de yalnız birer özet ve görünüş olarak.
   Şeylerin var olduğu, hareket edip değiştiği ve kendisinin de içinde davrandığı bir dünyayı algılamıyor. Bir çocuk
   "bak"ı bakarak ve kendisine bakılarak öğrenir.

Sahibin söylediği budur: dizilerde kelime hesaplaması zekâ değildir. Neden Türkçenin ek yapısı da değildir: aynı
dünyanın İngilizcesi de aynı yerde takılır.

## 1. İlke: tek kanıt hesabı, çok düzey

### 1.1 Korunan çekirdek

Hiçbir kural değişmiyor. Her bağ x → y altı köken için ayrı kanıt tutar: duyum, söylenen, taklit, kendi eylemi,
çıkarım ve hayal (p).

```
W⁺ = Σ_p τ_p · W⁺_p        W⁻ = Σ_p τ_p · W⁻_p        W = W⁺ + W⁻
f = W⁺ / W                 c = W / (W + k),  k = 1
e = c · (f − ½) + ½                                  (NARS beklentisi; kanıt yoksa ½: "bilmiyorum")
τ_duyum = τ_eylem = 1;  diğer güvenler ½'den başlar, sonradan gelen duyuma karşı öğrenilir (Güven Yargısı)
konuşma çizgisi:  e* = ½ + θ / (2T) = 0,583          (θ = 1, T = 6: yarış T adımda eşiğe varmalı)
```

Kanıt toplanır (NARS revizyonu, F30), her artış deftere yazılır ve bir yaşantının kanıtı birebir geri alınabilir
(Kesin Unutma). Yedili Odak, yarış, bilmeme hissi (L7), uyku ve rüya ayıklama da olduğu gibi kalır. Yeni düzeyler bu
hesabın **başka birimlere ve başka bağ türlerine** uygulanmasıdır. Yeni bir öğrenme kuralı, yeni bir olasılık hesabı
ya da yeni bir ağ yoktur.

### 1.2 Düzeyler

```
dil (D5)         kelimeler, yapılar (biçim–anlam çiftleri), söylem
                    │  ADLANDIRIR (kelime/yapı → anlam)      YUVA (yapının yeri → rol)
durum (D4)       şimdi ne var, ne oluyor, ne bekleniyor: çalışan dünya modeli
                    │
olay (D3)        bir değişim: eylem türü + katılanlar + söz ile olay arasındaki zaman ilişkisi
                    │  ROL (katılım → rol kategorisi)
kavram (D2)      nesnelerin (ve olayların) kategorisi; üyelerin seslerinden profil
                    │  TÜR (nesne → kavram, kavram → kavram)
nesne (D1)       izlenen gönderge: zaman içinde aynı kalan şey
                    │  SAHİP (nesne/kavram → öznitelik)
öznitelik (D0)   duyuların verdiği: biçim, renk, boyut, hareket imzası, ses …
```

Kelime artık en alttaki birim değildir. Kelime de algılanan bir şeydir (D0'da, `text` kanalında), ama anlamı
yukarıdaki düzeylerde, adlandırdığı şeydedir.

### 1.3 Bağ türleri

| Bağ | Nereden → nereye | Kanıt nereden gelir | Çekirdekte karşılığı |
|---|---|---|---|
| SAHİP | nesne veya kavram → öznitelik | algı; kavram için üyelerin sesleri | EXPECT, gecikme 0: görünüşün eşlik ettiği şeye bağlanması zaten var |
| TÜR | nesne → kavram, kavram → kavram | adlandırma, benzerlik, söylenen | ABSTRACT (kategori üyeliği) |
| ROL | katılım → rol kategorisi | katılım profillerinin benzerliği | ABSTRACT; katılım (olay, nesne) ikilisidir, bileşik birim |
| ADLANDIRIR | kelime veya yapı → anlam birimi | durumlar arası hizalama (§6) | yeni biçim: deponun bağ biçimi alanı 3 bittir; 4 değer kullanılıyor (EXPECT, PART, ABSTRACT, RELY) |
| YUVA | yapının j. yeri → rol | hizalama (§7.2) | yeni biçim; gecikme alanı yerin sırasını tutar (PART bağlarında olduğu gibi) |
| BEKLENTİ | her düzeyde ne neyin ardından gelir | yaşantı | EXPECT: kelimeden kelimeye (bugünkü), olaydan olaya (senaryo, neden–sonuç), sözden olaya (komutun sonucu) |

Her bağ altı köken için W± tutar. Her artış aynı yazıcılardan (`learn.add_evidence1`) geçer ve deftere yazılır. Kesin
Unutma bütün düzeylerde geçerlidir.

### 1.4 Elle kural yok: neyi duyular verir, neyi zihin öğrenir

| Verilen (duyuların ve bedenin işi) | Öğrenilen (zihnin işi) |
|---|---|
| Öznitelikler: opak simgeler (ör. `b3`, `r1`), her biri bir boyut kanalında | hangi özniteliklerin bir arada gittiği (kavramlar); hangi kelimenin neyi adlandırdığı |
| Nesnelerin izlenmesi: aynı nesne birden çok anda aynı indekstedir (görsel indeks) | nesnenin ne olduğu (kavramı) ve adı |
| Bir olayda kimin nasıl katıldığı: hareket eden, başlatan, dokunulan, yönelinen, değişen. Bunlar Dowty'nin (1991) önerme listesinin algılanabilir olanlarıdır | rollerin kendisi (katılım profillerinin kategorileri); hangi rolün cümlenin neresinde ve hangi biçimde söylendiği |
| Zihnin kendi saati (an sayısı); sözün olaydan önce, olay sırasında ya da sonra gelmesi | hangi kelime biçiminin hangi zaman ilişkisiyle gittiği ("bak", "bakıyor", "baktı"; "look", "is looking", "looked") |
| Kimin kime konuştuğu (konuşan ve muhatap: bugün de var olan bağlam kanalı) | komutun, sorunun ve betimlemenin ne olduğu: sözün sonuçlarından |

Zihne hiçbir dilbilgisi, rol listesi ya da kelime anlamı listesi verilmez. Özellikle "fail", "etkilenen", "yönelme
hâli" ya da "şimdiki zaman" kodun hiçbir yerinde geçmez. Zihnin rolleri ρ1, ρ2, … diye adlandırılan, türetilmiş
kategorilerdir. Biçimleri ise yapı yapı öğrenilir.

## 2. Algı ve bağlama: nesne dosyaları

Bir andaki sahne `Σ_t = {(o, A_o)}` biçimindedir:

* `o` izlenen bir indekstir;
* `A_o = {(φ, a_φ)}` öznitelikleri ve yeğinliklerini tutar, `a_φ ∈ (0, 1]`;
* her öznitelik bir boyut kanalındadır, `δ(φ)` (renk, biçim, boyut, …).

Boyut kanaldır, tıpkı çekirdekteki gibi: Beklenen Başarısızlık kuralı bugün de "hedefin kanalı başka bir şey bildirdi"
koşulunu okur.

```
ΔW⁺_duyum(o ─SAHİP→ φ) = a_φ
ΔW⁻_duyum(o ─SAHİP→ ψ) = a_φ · e(o ─SAHİP→ ψ)          ψ ≠ φ,  δ(ψ) = δ(φ)
```

**Nesnenin sürekliliği:** görüş dışındaki bir nesneye kanıt eklenmez, ama öznitelikleri durum modelinde beklenmeye
devam eder (§5).

**Yeniden tanıma:** izleme koptuğunda iki nesne dosyasının aynı şey olma derecesi, bilgisizliğin ötesindeki inançlar
üzerinden bir benzerliktir. Bu sonraki bir adım için tanımlanmıştır.

```
ê(x) = 2 · max(0, e(x) − ½)                            (inancın "bilmiyorum"dan öte kısmı)
ρ(o, o′) = Σ_φ min(ê(o,φ), ê(o′,φ)) / Σ_φ max(ê(o,φ), ê(o′,φ))
```

## 3. Kavramlar

### 3.1 Üyelik

`e(o ─TÜR→ C)` için kanıt üç yoldan gelir:

* **Adlandırma.** o, C'yi adlandıran bir w kelimesiyle anıldığında (§6):
  `ΔW⁺(o → C) = s_o · e(w ─ADLANDIRIR→ C)`. Köken: söylenen ya da taklit.
* **Tanıma (benzerlik).** o'nun gözlenen her özniteliği φ, üyeliğe tanıklık eder. Köken: çıkarım.

  ```
  ΔW⁺(o → C) = a_φ · ê(C ─SAHİP→ φ)
  ΔW⁻(o → C) = a_φ · Σ_{ψ ≠ φ, δ(ψ) = δ(φ)} ê(C ─SAHİP→ ψ)
  ```

* **Söylenen.** "o bir C'dir" anlaşıldığında (§7) gelen kanıt.

### 3.2 Profil: üyelerin sesleri

```
μ(o, C) = ê(o ─TÜR→ C)
W⁺(C ─SAHİP→ φ) = Σ_o μ(o, C) · ê(o ─SAHİP→ φ)
W (C ─SAHİP→ φ) = Σ_o μ(o, C) · [o, δ(φ) boyutunda gözlendi]
e (C ─SAHİP→ φ) = c · (f − ½) + ½,    f = W⁺ / W,  c = W / (W + k)
```

Her üye bir sestir (Üye Okuması F11 ve Bağımsız Kanıtların Birleşmesi F30). Bin kez görülen bir at bir sestir, bin
değil. Kavram, üyelerinin üzerinde birleştiğini bilir. Bacağı kırık tek bir at "atların bacağı kırıktır" bilgisini
doğurmaz.

### 3.3 Tipiklik ve kalıtım

```
τ(o, C) = Σ_φ ê(C, φ) · ê(o, φ) / Σ_φ ê(C, φ)        (kavramın beklediği özniteliklerin o'da olan payı)
```

Kalıtım (istisnalı tümdengelim) F37'nin kelime yerlerinden kavramlara taşınmasıdır. o'nun δ boyutunda kendi kanıtı
yoksa:

```
f_ç = f(o → C) · f(C → φ)
c_ç = f(o → C) · f(C → φ) · c(o → C) · c(C → φ)        (NARS tümdengelim doğruluk işlevi)
```

Kanaati olan en özel kavram konuşur. o'nun δ'daki kendi kanıtı her zaman önce gelir: istisna budur ("penguen
yüzer").

### 3.4 Kavram hiyerarşisi

```
W⁺(C ─TÜR→ D) = Σ_o μ(o, C) · μ(o, D)
W⁻(C ─TÜR→ D) = Σ_o μ(o, C) · ν(o, D),       ν(o, D) = 2 · max(0, ½ − e(o → D))
```

Buna söylenen kanıt da eklenir ("at bir hayvandır"). Söylenene güven, sonradan gelen algıya karşı yargılanır (Güven
Yargısı).

### 3.5 Yeni kavram

Kanaati olan bir nesnenin bütün kavramlara tipikliği ½'nin altındaysa, o nesne ilk üyesi olmak üzere yeni bir kavram
birimi kurulur. Çekirdek kategorileri de böyle kurar; F67 de başarısız olduğu yerde kategoriyi böler. Onu sonradan
adlandıran kelime bu kavrama bağlanır (§6).

## 4. Olaylar, roller, zaman

### 4.1 Olay

```
ε = ( α_ε , {(o_j, P_j)} , [t₀, t₁] )
```

* `α_ε`: eylemin öznitelikleri, yani hareket, bakış ya da ses imzası. Boyut kanallarında opak simgelerdir.
* `P_j ⊆ Π`: o_j'nin katılım öznitelikleri. `Π = {hareket eder, başlatır, dokunulur, yönelinir, değişir, birlikte
  gider, …}` algının hareket, temas, yön ve değişim hakkında bildirdiklerinden oluşur. Bunlar Dowty'nin (1991) önerme
  listesinin algılanabilir olanlarıdır; Talmy'nin kuvvet devinimi de bu türdendir.

Depoda ε, `olay` duyu kanalında bir atom birimdir. Her katılım `π_j = (ε, o_j)` bir bileşik birimdir. Katılım
öznitelikleri ona görünüş olarak bağlanır (SAHİP).

**Olay türü** (A), olayların eylem özniteliklerine göre kurulan bir kategorisidir: kavramların nesneler üzerine
kurulduğu gibi. §3'ün tamamı burada değişmeden uygulanır: üyelerin sesleri, tipiklik, yeni tür.

### 4.2 Roller

Bir rol (ρ), katılımlar üzerinde, katılım profillerinin benzerliğiyle kurulan bir kategoridir:

```
e(π ─ROL→ ρ)            e(ρ ─SAHİP→ p)  üyelerin sesleriyle (§3.2),   p ∈ Π
```

Roller kavramlar gibi türetilir: bir katılım hiçbir role tipik değilse (τ < ½) yeni bir rol kurulur. M1 dünyasında
beklenen sonuç şudur:

* ρ1 ≈ {başlatır, hareket eder ya da bakar}
* ρ2 ≈ {yönelinir}
* ρ3 ≈ {dokunulur, değişir}

Adları hiçbir zaman verilmez. Zihinde yalnız ρ1, ρ2, ρ3 ve onları kuran kanıt vardır.

Roller şunun için gereklidir: bir fiille öğrenilen yapı başka bir fiile taşınabilsin. "bak" ile ρ1'i alan yer, "koş"
ile de ρ1'i alır.

### 4.3 Zaman: söz ile olay arasındaki ilişki

Reichenbach (1947) iki zamanı ayırır: söz zamanı S ve olay zamanı E. S anında söylenen bir u sözü ve onun adlandırdığı
bir ε olayı için:

```
ζ(u, ε) = önce    S < t₀          (beklenen, istenen)
          içinde  t₀ ≤ S ≤ t₁     (sürüyor)
          sonra   t₁ < S          (geçmiş)
```

Söz edimi de sonuçlardan çıkar. ε, u'nun ardından gelir ve ε'nin ρ1 katılanı u'nun muhatabıysa, u bir istektir
(komut). Bunu sonuç izi gösterir: u'dan ε'ye gecikme 2'lik bir EXPECT bağı. Bu bilgi söylenmez; algılanır ve
öğrenilir.

ζ, (u, ε) ikilisinin `zaman` boyut kanalındaki bir özniteliğidir. Hangi kelimenin hangi ζ ile gittiği öğrenilir (§6,
§7).

### 4.4 Senaryolar ve neden–sonuç

Olay türleri arasında da beklenti bağları kurulur (EXPECT, gecikme 1 ve 2): "ρ1 topa vurur" → "top hareket eder".
Çekirdeğin sonuçtan öğrenmesi ve T-cause (müdahaleyi gözlemden ayırmak) artık olaylara uygulanır.

## 5. Durum modeli

```
D_t = ( O_t , E_t )      her öğe için bir e: var mı, hangi özniteliklerle; hangi olay, hangi ζ ile
```

Durum modeli kökenine göre dört yoldan güncellenir:

* **Duyum:** algı (güven 1).
* **Söylenen:** anlama (§7). Söylenen kanıt olarak girer; söylenene güven sonradan gelen algıya karşı öğrenilir.
* **Çıkarım:** senaryolar (§4.4) ve kalıtım (§3.3).
* **Hayal:** hayal ve rüya.

**Şaşkınlık:** `σ = 1 − e(gözlenen | D_{t−1})`. Bugünkü his kanalına girer.

**İstek:** anlaşılan bir istek, ζ = önce olan bir ε* bekleyişi ekler. Dünya ε*'ı gösterirse anlama doğrulanmış olur:
ε*'ı veren yapı ve kelime anlamları kanıt alır. Zihin böylece anlamayı sonuçtan öğrenir. Dünya başka bir şey
gösterirse aynı yapı ve kelime anlamları karşı kanıt alır.

## 6. Kelimenin anlamı: durumlar arası öğrenme

S anında `U = (w₁, …, w_n)` sözü duyulur. Aday anlamlar `M_S`, odaktaki birimlerdir: oradaki nesneler ve kavramları,
öznitelikler, olaylar ve türleri, roller ve ζ.

```
α(w, x) = e(w ─ADLANDIRIR→ x) / Σ_{w′ ∈ U} e(w′ ─ADLANDIRIR→ x)        (yeni bir çift için e = ½)
ΔW⁺(w ─ADLANDIRIR→ x) = s_x · α(w, x)                                    x ∈ M_S
ΔW⁻(w ─ADLANDIRIR→ x) = e(w ─ADLANDIRIR→ x) · [x ∉ M_S ve δ(x) başka bir şey bildirdi]
```

`s_x`, x'in belirginliğidir: odaktaki etkinliği. Ortak dikkat (öğretmenin bakışı ya da göstermesi) bunu artırır.
Sonuçlar:

* Tutarlı biçimde birlikte gelen yükselir; rastlantıyla birlikte gelen karşı kanıt biriktirir (Siskind 1996; Yu ve
  Smith 2007; Fazly, Alishahi ve Stevenson 2010).
* **Açıklanmış olan pay almaz.** Aynı sözde başka bir kelimenin zaten adlandırdığı bir anlam bu kelimeye küçük pay
  verir. Hızlı eşleme ve karşılıklı dışlama buradan doğar (Carey ve Bartlett 1978; Markman ve Wachtel 1988): tek
  duyuşta öğrenme.
* **Hiçbir şeyi adlandırmayan kelimeler** ("bir", "-dır", "is", "the") bütün anlamlara dağılır ve anlam yokken karşı
  kanıt alır. Bu yüzden ½ yakınında kalırlar: ad değildirler, yapıların sabit parçaları olurlar (§7).
* **Çok biçim, tek anlam.** "at", "ata", "atın" ve "atlar" her biri AT kavramını adlandırmaya gelir. Bilgi kelimede
  değil kavramda toplanır. Bugünkü zihnin eksiği tam olarak budur (§0.2, madde 3).

Bir kelimenin anlamı, konuşma çizgisinin üstündeki en yüksek adaydır: `m(w) = argmax_x e(w → x)`. Çizginin üstünde
birden çok aday varsa kelime çok anlamlıdır.

## 7. Yapılar: biçim–anlam çiftleri

### 7.1 Tanım

Bir yapı üç parçadan oluşur: κ = (biçim, anlam, yuva eşlemesi).

* **Biçim:** çekirdeğin bir çerçevesi. Yerleri ya sabit öğe ya da kategori (yuva) olan yaşanmış bir dizidir: "_ _
  bak", "_ is looking at the _". Çekirdek bunları zaten kuruyor.
* **Anlam:** bir olay şeması. Eylem yeri bir olay türünü adlandırır. Katılan yerleri rollere eşlenir (YUVA bağları
  κ.j → ρ). Yapının bütünü ζ'yı ve söz edimini adlandırır (ADLANDIRIR κ → ζ).

### 7.2 Öğrenme: hizalama

Bir ε olayının olduğu durumda U sözü duyulur. Kelimesi ε'nin bir katılanını (o) adlandıran her j yeri için:

```
ΔW⁺(κ.j ─YUVA→ ρ)          = 2 · (e(w_j → o) − ½) · μ(π(ε, o), ρ)
ΔW⁺(κ ─ADLANDIRIR→ ζ(U,ε)) = Π_j ê(w_j → x_j)          (sözün kelimelerinin bu olayla açıklanma derecesi)
```

w_j, j yerinin kategorisine katılır (çekirdeğin çerçeve ve kategori öğrenmesi, değişmeden). Bir yerin kelimesi başka
bir rolün katılanını adlandırıyorsa karşı kanıt gelir (Beklenen Başarısızlık). Yaklaşım Alishahi ve Stevenson (2008),
Chang (2008) ve Steels'in (2011) çalışmalarına dayanır.

### 7.3 Anlama

```
e(κ ⊨ U)    = Π_{j yuva} e(w_j ─TÜR→ Kat(κ.j)) · [sabit öğeler yerinde]
G(κ, b | U) = e(κ ⊨ U) · Π_j e(w_j → x_j) · Π_{j katılan yeri} e(κ.j ─YUVA→ ρ_j)
(κ̂, b̂)     = argmax G ;       anlaşıldı  ⇔  G^{1/m} ≥ e*       (m: çarpandaki terim sayısı)
ε̂          = ( eylem yerinin adlandırdığı tür , {ρ_j : x_j} , κ̂'nın adlandırdığı ζ )
```

Burada b bağlamadır: her yere bir gönderge düşer. Sonuç ε̂ durum modeline (§5) söylenen kanıt olarak girer.
Anlaşılmadıysa sonuç bilmeme hissidir (L7), yani çekirdeğin kendi ölçüsü.

### 7.4 Söyleme: anlamdan biçime

Dile getirilecek bir ε olayı ve bir ζ verildiğinde:

```
κ*      = argmax_κ  e(κ → ζ) · Π_{ρ ∈ roller(ε)} max_j e(κ.j ─YUVA→ ρ)
w_j     = argmax_{w ∈ Kat(κ*.j)} e(w → x_{ρ_j})                  (o yerde yaşanmış ve katılanı adlandıran kelime)
w_eylem = argmax_{w ∈ Kat(κ*.eylem)} e(w → tür(ε)) · e(w → ζ)
```

Sözün sırası κ*'ın sırasıdır; sabit öğeler yaşandığı gibi kalır. Bir kelimenin bir yerdeki biçimi o yerin bir
özelliğidir: "ata" hedef yerinde, "at" özne yerinde yaşanmıştır. Bunun için bir ek kuralı gerekmez.

**Genelleme budur.** "can suya bakıyor" hiç duyulmadı. Yine de söylenebilir, çünkü:

* "can", CAN'ı adlandırır ve 1. yerde yaşanmıştır;
* "suya", SU'yu adlandırır ve 2. yerde yaşanmıştır;
* "bakıyor", ζ = içinde ile BAK'ı adlandırır;
* κ_sürüyor yapısı ρ1'i 1. yere, ρ2'yi 2. yere eşler.

Cevabın kelimelerini anlam verir, sırasını yapı.

**Sınır (şimdiden söyleniyor).** Bir yerde hiç yaşanmamış bir kelime biçimi söylenemez. Zihin o durumda katılanın
başka yerlerdeki en iyi adını söyler ve bunu işaretler. Üretken kelime yapımı ayrı bir adımdır (çekirdeğin kelime
parçaları, F36) ve buradaki konu değildir. Sahibin sözüyle: "mesele kök ve ek yapısı değil". §0.1'deki "hiç
betimlenmemiş fiil" soruları bu sınıra girer ve ölçümlerde ayrı satırda raporlanacak.

## 8. Soru, cevap, anlatma

### 8.1 Soru yapıları

Bir soru yapısı (κ_q), anlamı açık yeri "?" olan bir şema olan yapıdır:

* "X nedir ?" / "what is X ?" → X'in adlandırdığı kavramın tanımı;
* "kim V-ıyor ?" → V türünde süren bir olayın ρ1'i;
* "X ne yapıyor ?" → X'in süren olayının türü.

Soru yapıları da öteki yapılar gibi öğrenilir: izlenen diyaloglarda açık yeri cevap doldurur.

### 8.2 Cevap: anlamdan

Adaylar (φ), sorulan şey hakkında durum modelindeki ve anlam belleğindeki bilgilerdir: TÜR, SAHİP, olay türleri,
söylenmiş bilgiler.

```
r(φ | q) = e(φ) · ϱ(tür(φ) | κ_q) · (1 − e(φ bu konuşmada söylendi))
```

ϱ yaşanmış diyaloglardan öğrenilir: bu soru yapısını hangi tür bilgiler cevapladı.

### 8.3 Anlatma

Cevap birden çok cümle olabilir. Bir sonraki bilginin türü, öğrenilmiş söylem sırasıyla bir öncekinin ardından gelir.
Bu sıra, yaşanmış anlatımlardan gecikme 1'lik EXPECT bağlarıyla öğrenilir; "nedir" için: TÜR → SAHİP → olay →
dikkate değer bilgi. Anlatma, en iyi r konuşma çizgisinin altına düştüğünde durur. Bu, F41'in "bilinen ve
söylenmemiş" kuralının anlam düzeyindeki karşılığıdır. Her cümle §7.4 ile söylenir.

## 9. Beden, dürtü, değer, dikkat

"İnsan zihni sadece zihinle çalışmaz." Bu bölüm onun karşılığıdır.

**Dünyadaki zihin.** Zihnin eylemleri (bakmak, göstermek, almak, söylemek) kendisinin de katıldığı olaylardır (Benlik
Birimi). Eylemleri algıladığını değiştirir: baktığı nesne odağa girer. Sonuçları öğrenilir (T-cause).

**Dürtüler** ilanla değil ölçüyle vardır:

```
merak:     u(x) = 1 − 2 · |e(x) − ½|          (bilmeme hissi; F46, F47, F50, F51 bugün bu ölçüyle çalışıyor)
sosyal:    açık bir soru ya da istek: muhatabın beklediği (sözün sonucu)
yeterlik:  eylemlerin değer sonuçları (değer kanalında iyi ve kötü)
```

**Eylem seçimi:** `U(a) = Σ_d g_d · E[Δd | a]`. Beklenen azalma sonuç bağlarından okunur; kararı yarış verir.

**Dikkat:** x'in belirginliği şudur:

```
s_x = etkinlik(x) · (1 + σ_x) · (1 + ilgi_d(x)) · (1 + ortak_dikkat_x)
```

Yedili Odak bunlardan 7'sini seçer. Nesneler, olaylar ve kelimeler aynı odak için yarışır.

## 10. Bellek sistemleri ve uyku

* **Çalışma belleği:** odak (7) ve D_t.
* **Olay belleği (epizodik):** zamanı, yeri ve katılanlarıyla olay birimleri; Kesin Unutma'nın defteri.
* **Anlam belleği (semantik):** kavramlar, roller ve TÜR, SAHİP, olay bağları; profiller üyelerin seslerinden.
* **Yordam belleği:** yapılar ve beceriler (eylem → sonuç).
* **Uyku:** günün olayları hayalde yeniden yaşanır. Kavram ve rol profilleri ile senaryolar güncellenir. Rüya
  ayıklama hayal edilen olaylara da uygulanır: doğru, yanlış ya da açık. Bu, tamamlayıcı öğrenme sistemleri kuramına
  karşılık gelir (McClelland, McNaughton ve O'Reilly 1995): hızlı olay belleği ve yavaş anlam belleği.

## 11. Sizin örneğiniz baştan sona

Aşağıdaki her sayı **beklenen**dir; ölçülmüş değildir. Ölçüm §12'deki testlerle yapılacak.

**"ali ata bak"**

1. *Algı.* Sahnede nesne dosyaları vardır: o_ali (canlı, iki ayaklı, …), o_at (canlı, dört ayaklı, büyük, …) ve
   başkaları. Öğretmenin bakışı o_ali'dedir: muhatap odur.
2. *Kelimeler.* Önceki durumlardan: "ali" → o_ali (e ≈ 0,9); "ata" → AT (≈ 0,85); "bak" → BAK türü, ζ = önce ve
   istekle.
3. *Anlama.* κ_istek = [_₁ _₂ _₃]. Eşlemesi: YUVA(1 → ρ1), YUVA(2 → ρ2), eylem 3. yerde, ADLANDIRIR ζ = istek. Sonuç:
   ε̂ = BAK(ρ1: o_ali, ρ2: o_at), ζ = önce. Bu, durum modelinde beklenen bir olay olur.
4. *Dünya.* Ali ata bakar. Algılanan olay ε̂ ile örtüşür (e yüksek). Anlama doğrulanır, yapı ve kelimeler kanıt alır.
5. *Söyleme.* ζ artık içinde (sürüyor). κ_sürüyor = [_₁ _₂ _₃] seçilir:
   * "ali": 1. yerin kategorisinde ve o_ali'yi adlandırıyor;
   * "ata": 2. yerin kategorisinde ve AT'yi adlandırıyor;
   * "bakıyor": 3. yerin kategorisinde ve ζ = içinde ile BAK'ı adlandırıyor.

   Söz: **"ali ata bakıyor"**.

**"at nedir ?"**

κ_q("_ nedir ?") açık yeri AT'nin tanımı olan bir sorudur. Adaylar:

| Aday | Kaynak | e |
|---|---|---|
| AT ─TÜR→ HAYVAN | söylenen | ≈ 0,9 |
| AT ─SAHİP→ dört ayak | algılanan, üyelerin sesleri | ≈ 0,95 |
| AT ─olay→ hızlı koşar | algılanan ve söylenen | ≈ 0,8 |
| kırık bacağın tedavisi bugün de zordur | söylenen | ≈ 0,7 |

Öğrenilen söylem sırası TÜR → SAHİP → olay → dikkate değer bilgidir. Her bilgi bir yapıyla söylenir: "at bir
hayvandır . dört ayağı vardır . çok hızlı koşar . …". Anlatma, en iyi r konuşma çizgisinin altına düşünce durur.

## 12. Ölçüm

**Dünya** (`dunya.py`, bu klasörde):

* nesneler, opak öznitelikleriyle;
* eylemler, katılım öznitelikleriyle;
* zaman içinde olaylar;
* aynı durumlar için Türkçe ve İngilizce öğretmen sözleri: adlandırma, komut, betimleme, bilgi, soru.

Ayrımlar (hangi birleşimlerin hiç gösterilmeyeceği) her koşudan önce dondurulur. Projenin protokolleri de böyle
çalışıyor.

| Test | Ne ölçer | Başarı ölçüsü |
|---|---|---|
| T1 | Komut → olay → betimleme; yeni birleşimler (ali × bak × su hiç birlikte görülmedi), Türkçe ve İngilizce | tam doğru betimleme; bilinmeyen biçim ayrı satırda |
| T2 | "X nedir ?": ayrı ayrı öğrenilmiş (söylenmiş ve algılanmış) bilgilerden; hiç sorulmamış bir kavram için | söylenen doğru bilgilerin oranı, sıra, durma |
| T3 | İki kelimelik gruplar: "kırmızı topa bak" (sahnede iki top var) | doğru gönderge; "ali kırmızı topa bakıyor" |
| T4 | Durumlar arası kelime öğrenme: k nesneli sahneler, n duyuş; tek duyuşta hızlı eşleme | doğru anlamı bilinen kelime oranı, n'ye göre eğri |
| T5 | Sistematiklik: bir fiil hiç görülmediği bir özneyle; uzunluk | tam doğru |
| T6 | Durum soruları: "kim koşuyor ?", "ali ne yapıyor ?" | tam doğru |

**Karşılaştırılanlar:**

* (a) bugünkü zihin, yalnız sözle;
* (b) bugünkü zihin, sahne görünüş olarak eklenmiş hâliyle: bugünkü çekirdeğin algıyla yapabileceğinin en iyisi;
* (c) aynı söz akışıyla sıfırdan eğitilmiş bir Transformer, bir kez yalnız sözle, bir kez sahne simgelere dökülmüş
  hâliyle;
* (d) yeni düzeyler, anahtar anahtar açılarak.

## 13. Kilometre taşları

Her adım kendi anahtarının arkasındadır ve varsayılanı kapalıdır. Anahtar kapalıyken kapılar bit bit aynıdır. Her
adımın testi, notu ve bataryası olur ve bir önceki koda karşı ölçülür.

| Adım | İçerik | Bittiğinde |
|---|---|---|
| M1 | Dünya (`dunya.py`) ve taban ölçüm: (a), (b), (c) | T1–T6'da bugünkü zihnin ve Transformer'ın sayıları |
| M2 | Nesne dosyaları ve öznitelikler çekirdekte: `nesne` kanalı, SAHİP kanıtı, boyut dışlaması | izlenen nesneler öznitelik inancıyla; kapılar aynı |
| M3 | Durumlar arası adlandırma: ADLANDIRIR kanıtı, açıklanmışın payı | T4 eğrisi; hızlı eşleme |
| M4 | Olaylar, katılım, roller, ζ | M1 dünyasında rollerin türetilmesi (ρ1, ρ2, ρ3) |
| M5 | Yapılar ve anlama | T1'in anlama yarısı (komuttan beklenen olay) |
| M6 | Söyleme (anlamdan biçime) | T1, T3, T5 |
| M7 | Soru ve anlatma | T2, T6 |
| M8 | Metinden önyükleme: belgeler, yapılarla okunup durum modeline çevrilir (sözdizimsel önyükleme, Gleitman 1990) | sahibin belgeleri ve etüt dersleri: sınav satırları (bugün 11/41) |

M8, bugünkü büyük zihne ve etüte bağlanır. Dünyada öğrenilen anlama, belgelerin okunmasına taşınır.

## 14. Literatürdeki yeri, dürüstçe

**Alınanlar (her biri adıyla):**

* NARS'ın doğruluk işlevleri (Wang);
* durumlar arası hizalama (Siskind; Yu ve Smith; Fazly, Alishahi ve Stevenson);
* yapı dilbilgisi (Goldberg; Steels'in Akışkan Yapı Dilbilgisi; Bergen ve Chang'in Bedenleşmiş Yapı Dilbilgisi;
  Chang);
* önerme listesiyle roller (Dowty);
* söz ve olay zamanı (Reichenbach);
* nesne dosyaları (Kahneman, Treisman ve Gibbs) ve görsel indeksler (Pylyshyn);
* örnek ve ilk örnek kuramları (Nosofsky; Rosch);
* durum ve zihinsel modeller (Zwaan ve Radvansky; Johnson-Laird);
* tamamlayıcı öğrenme sistemleri (McClelland, McNaughton ve O'Reilly);
* sembol temellendirme (Harnad; Roy; Roy ve Pentland);
* sözdizimsel ve anlamsal önyükleme (Gleitman; Pinker);
* fiil adası (Tomasello);
* merak (Schmidhuber; Oudeyer ve Kaplan);
* bilişsel mimariler (ACT-R, Soar, OpenCog, LIDA).

**İddia edilen:** bu düzeylerin hepsinin **tek bir kanıt hesabında** yaşaması. Bu hesabın özellikleri şunlardır:

* kanıt kökenine göre ayrı tutulur ve güven öğrenilir;
* her yaşantı birebir geri alınabilir;
* öğrenme yaşarken olur: eğitim ile kullanım ayrı evreler değildir;
* her bağ okunabilir;
* her düzey ölçülür ve başarısız denemeler de yazılır.

Parçaların her biri bilinmektedir. Birleşimleri ve bu çekirdekte ölçülmeleri yenidir.

**İddia edilmeyen:**

* Gerçek görme yok: dünya temiz öznitelikler verir. Programın bugünkü görmesi bir özet ve görünüşlerden ibarettir.
* Bir beden yok.
* Temellendirme sorunu çözülmüş değil.
* Dünya küçük ve yapaydır.

Sonuçlar ölçüldükleri gibi yazılacak, başarısızlıklar dahil.

## 15. Kaynaklar

* Alishahi, A. ve Stevenson, S. (2008). A computational model of early argument structure acquisition. *Cognitive
  Science* 32(5).
* Bergen, B. ve Chang, N. (2005). Embodied Construction Grammar in simulation-based language understanding.
  *Construction Grammars: Cognitive Grounding and Theoretical Extensions*, John Benjamins.
* Carey, S. ve Bartlett, E. (1978). Acquiring a single new word. *Papers and Reports on Child Language Development*
  15.
* Chang, N. (2008). *Constructing Grammar: A Computational Model of the Emergence of Early Constructions.* Doktora
  tezi, UC Berkeley.
* Dowty, D. (1991). Thematic proto-roles and argument selection. *Language* 67(3).
* Fazly, A., Alishahi, A. ve Stevenson, S. (2010). A probabilistic computational model of cross-situational word
  learning. *Cognitive Science* 34(6).
* Gleitman, L. (1990). The structural sources of verb meanings. *Language Acquisition* 1(1).
* Goldberg, A. (1995). *Constructions: A Construction Grammar Approach to Argument Structure.* University of Chicago
  Press.
* Harnad, S. (1990). The symbol grounding problem. *Physica D* 42.
* Johnson-Laird, P. N. (1983). *Mental Models.* Harvard University Press.
* Kahneman, D., Treisman, A. ve Gibbs, B. (1992). The reviewing of object files. *Cognitive Psychology* 24(2).
* Markman, E. ve Wachtel, G. (1988). Children's use of mutual exclusivity to constrain the meanings of words.
  *Cognitive Psychology* 20(2).
* McClelland, J., McNaughton, B. ve O'Reilly, R. (1995). Why there are complementary learning systems in the
  hippocampus and neocortex. *Psychological Review* 102(3).
* Nosofsky, R. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of
  Experimental Psychology: General* 115(1).
* Oudeyer, P.-Y. ve Kaplan, F. (2007). What is intrinsic motivation? A typology of computational approaches.
  *Frontiers in Neurorobotics* 1.
* Pinker, S. (1984). *Language Learnability and Language Development.* Harvard University Press.
* Pylyshyn, Z. (1989). The role of location indexes in spatial perception. *Cognition* 32(1).
* Reichenbach, H. (1947). *Elements of Symbolic Logic.* Macmillan.
* Rosch, E. (1975). Cognitive representations of semantic categories. *Journal of Experimental Psychology: General*
  104(3).
* Roy, D. (2005). Grounding words in perception and action. *Trends in Cognitive Sciences* 9(8).
* Roy, D. ve Pentland, A. (2002). Learning words from sights and sounds. *Cognitive Science* 26(1).
* Schmidhuber, J. (1991). Curious model-building control systems. *IJCNN*.
* Siskind, J. M. (1996). A computational study of cross-situational techniques for learning word-to-meaning mappings.
  *Cognition* 61(1–2).
* Steels, L. (ed.) (2011). *Design Patterns in Fluid Construction Grammar.* John Benjamins.
* Talmy, L. (1988). Force dynamics in language and cognition. *Cognitive Science* 12(1).
* Tomasello, M. (1992). *First Verbs: A Case Study of Early Grammatical Development.* Cambridge University Press.
* Wang, P. (2013). *Non-Axiomatic Logic: A Model of Intelligent Reasoning.* World Scientific.
* Yu, C. ve Smith, L. (2007). Rapid word learning under uncertainty via cross-situational statistics. *Psychological
  Science* 18(5).
* Zwaan, R. ve Radvansky, G. (1998). Situation models in language comprehension and memory. *Psychological Bulletin*
  123(2).
