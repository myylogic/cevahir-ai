# Kayıt, denetim kaydı, soy ağacı ve konsol (`cevahir_ai.store`, `cevahir_ai.ops`, `cevahir_ai.console`)

> **English summary.** This page describes phases P3 and P4 of the roof (`cevahir_ai` 0.4.0).
> **P3, the registry:** a file-based store under `CEVAHIR_HOME` (outside the repository;
> `%LOCALAPPDATA%\Cevahir` on Windows, `${XDG_STATE_HOME:-~/.local/state}/cevahir` elsewhere,
> overridable by the environment). It holds content-addressed manifests keyed by the existing
> `cev:<kind>:<scheme>:<digest>` keys, host-local locations of payload files (a manifest never
> contains a path), one append-only, hash-chained and MAC'ed audit chain per machine, lineage as a
> projection of the audit events, registration of existing checkpoints, tokenizers and caches by
> scanning a folder, and `.cvb` bundles for moving artefacts between machines. **P4, the console:**
> `cevahir` (`python -m cevahir_ai`), a strict subset of PowerShell implemented in Python with the
> standard library only: Verb-Noun cmdlets, an object pipeline, `$variables`, `-Parameter value`
> binding, `.ps1` scripts, `-Command` one-liners, a prompt with history and Tab completion, help in
> Turkish and English, exit codes 0/1/2/3/5/130. It starts in about 0.1 s without importing torch.
> A script block `{ ... }` is one expression over data, not code: it cannot run a command, reach a
> Python attribute, read the environment or write a file; everything PowerShell has beyond the
> subset is rejected at parse time (`NotInCevahirSubset`). An optional PowerShell 7.2+ module
> (`powershell/Cevahir`, `Cv`-prefixed functions) wraps the console through a JSON protocol.
> **Not here yet (P5):** policies as data, approvals, signatures, anchors, governed training,
> publishing and serving. Today a built-in bootstrap policy allows read and register actions,
> records them, and refuses anything that would need an approval. **Honest limits:** a process
> running as the same user can rewrite the audit chain together with its key; registration pins
> the bytes as they were when registered, it does not prove they were right; bundles are not
> signed; nothing was run on Windows or on a real PowerShell in this environment. The last section
> lists the limits in full.

Bu sayfa çatı paketinin üçüncü ve dördüncü aşamasını anlatır. İlk iki aşama (çatı, anahtar
sistemi, uyumluluk kapısı) [CATI_VE_ANAHTAR_SISTEMI.md](CATI_VE_ANAHTAR_SISTEMI.md) içindedir.
Yapılmayanlar ve sınanmayanlar son bölümdedir.

## 1. Ne geldi

| Paket | İçerik |
|---|---|
| `cevahir_ai/store/` | `core` (ev dizini, makine kimliği, içerik adresli nesneler, manifest), `catalog` (eserler ve başvuru çözme), `locations` (dosyaların bu makinedeki yeri), `audit` (özet zincirli denetim kaydı), `redact` (gizli değerlerin kayda girmemesi), `lineage` (soy ağacı), `register` (klasör tarama ve tanıma), `bundles` (`.cvb`) |
| `cevahir_ai/ops/` | Kullanıcı düzeyindeki her eylem için tek fonksiyon: `artifacts`, `checkpoints`, `audit`, `keys`, `env`; `governed` (başlangıç politikası ve kayıt noktası). Konsol komutları bunların ince sarmalayıcısıdır; Python'dan çağıran da aynı davranışı ve aynı kaydı alır |
| `cevahir_ai/console/` | `parser` + `syntax` (dil), `expr` (ifade dili), `binder` (parametre bağlama), `runtime` (oturum, boru hattı, hata modeli), `registry` + `commands/` (43 komut), `format` (tablo, liste), `help` (tr/en), `completion`, `repl`, `lineedit_win`, `main`, `tools/gen_powershell` |
| `cevahir_ai/_compat/` | `locks` (Windows'ta `msvcrt.locking`, diğerlerinde `fcntl.flock`), `clock` |
| `cevahir_ai/domain/records.py` | `Record`: boru hattında akan tipli, sıralı özellik torbası |
| `powershell/Cevahir/` | İsteğe bağlı gerçek PowerShell modülü (üretilen iki dosya) |

Yeni bağımlılık yoktur: yalnız standart kütüphane. torch yalnız bir checkpoint'in tensörleri
okunacağında, ilk kullanımda içe aktarılır.

## 2. `CEVAHIR_HOME`

Depo dışındadır; deponun içindeki bir ev dizini reddedilir (testler geçici dizin kullanır).

| Platform | Varsayılan |
|---|---|
| Windows | `%LOCALAPPDATA%\Cevahir` |
| Linux, macOS | `${XDG_STATE_HOME:-~/.local/state}/cevahir` |
| Colab | Drive bağlıysa `/content/drive/MyDrive/.cevahir`, değilse `/content/.cevahir` |

`CEVAHIR_HOME` ortam değişkeni ya da `cevahir -Home <dizin>` bunu değiştirir.

```
CEVAHIR_HOME/
  host.json                         bu makinenin kimliği (h-<16 hane>); Colab'da sanal makine başına, Drive'da değil
  objects/ab/<62 hane>.json         manifestler, içerik adresli (cev:manifest:sha256-v1)
  tables/ab/<62 hane>.json          tensör tabloları ve dosya listeleri (manifestin iri eki)
  locations/<makine>.json           yük dosyalarının o makinedeki yolları
  audit/<makine>/seg-000001.jsonl   makine başına tek zincir; chain.json (baş kaydı); quarantine/
  trust/hmac/<makine>.bin           makinenin denetim MAC anahtarı (POSIX'te kip 0600)
  blobs/ab/<62 hane><uzantı>        paketlerden açılan yük dosyaları
  console_history, profile.ps1      konsol geçmişi; açılışta çalışan isteğe bağlı betik
```

Kilit dosyaları ev dizininde değil, yerel geçici dizindedir (`<temp>/cevahir-locks/`): eşitlenen
ya da ağ üzerindeki bir sürücüde kilitler güvenilir değildir. Her makine yalnız kendi zincirine ve
kendi konum dosyasına yazar.

## 3. Kayıt

- **Manifest**, tek bir eser anahtarı hakkındaki taşınabilir beyandır: tür, anahtar, üyeler
  (checkpoint için `weights`, `arch`, `wschema`, `tok`, `optim`, `sched`, `tstate`), girdiler
  (soy ağacı), yük dosyalarının dosya anahtarları, öznitelikler, köken. **İçinde yol bulunmaz.**
  Aynı beyan iki kez yazılırsa tek nesnedir; adı baytlarının sha256'sıdır, adına özetlenmeyen
  nesne kullanılmaz ve bildirilir.
- **Konumlar** manifestten ayrıdır. Bayt bayt aynı iki dosya (`last.pth`, `best.pth`) iki konumlu
  tek eserdir. Başka makineden kopyalanan dosya, manifesti eldeyse, torch gerekmeden yalnız
  baytlarından tanınır.
- **Tanıma** (`Register-Artifact`): hiçbir şey taşınmaz, yeniden adlandırılmaz, yeniden yazılmaz.

  | Bulunan | Anahtar | Not |
  |---|---|---|
  | `.pth/.pt/.ckpt` | `ckpt:members-v1` | tensörler okunur (torch); mimari ve tokenizer biliniyorsa ayrıca `model:triple-v1` |
  | bayt düzeyi BPE dosyası | `tok:bytebpe-v1` | saklı kimlik içerikle tutmuyorsa kaydedilmez |
  | `vocab.json` + `merges.txt` | `tok:bpe-sem-v1` | izlenen tokenizer ise eski özetleri takma ad olur |
  | `cached_data_<md5>_<özet>.pkl` | `cache:md5-v2` | asla unpickle edilmez; yan dosyası eskiyse söylenir |
  | klasör, `-Kind corpus` | `corpus:files-v1` | yalnız açıkça istenirse; CRLF ve LF aynı korpustur |
  | pickle'lanmış `nn.Module` | `file:bytes-v1` | okunmaz; `quarantined` |

  Köken `legacy-import`'tur: yalnız dosyanın kendi söylediği yazılır. Aynı klasörü yeniden
  kaydetmek hiçbir şey yazmaz ve hiçbir olay üretmez.
- **Durum** saklanmaz, denetim olaylarından türetilir: `registered`, `imported` (paketle geldi,
  burada doğrulanmadı), `verified` (son sınama özetleri doğruladı), `quarantined` (son sınama
  kayıtlı baytlardan farklı bayt ya da güvensiz zarf buldu). "Bu hedefle uyumsuz" bir durum
  değildir; sorulan soruya bağlıdır.
- **Başvurular**: tam anahtar, takma ad, yük dosyasının anahtarı, manifest anahtarı, kısa biçim
  (`ckpt:3f9a1c0b7d2e`, en az dört hane) ya da kayıtlı dosyanın yolu. Birden çok esere uyan kısa
  biçim tahmin edilmez, reddedilir.
- **Kayıtlı checkpoint manifestine karşı sınanır.** `Test-Checkpoint` diskteki baytları kayıtlı
  dosya anahtarıyla (C1), tensörleri kayıtlı ağırlık anahtarıyla (C3) karşılaştırır. Böylece
  özet taşımayan eski checkpoint'ler `serve` amacı için doğrulanabilir olur ve kayıttan sonra
  değişen dosya bildirilir. Kayıtsız dosya bütünlüğü hakkında yalnız "bilinmiyor" diyebilir.

## 4. Denetim kaydı

Makine başına tek zincir; her olay tek satır kanonik JSON'dur (CJSON-1).

- `hash`, olayın `hash` ve `mac` dışındaki kanonik baytlarının sha256'sıdır; `prev` bir önceki
  olayın `hash`'idir. Bir satırı **düzenlemek** özetini, bir satırı **silmek, eklemek ya da yerini
  değiştirmek** bağı bozar. Aynı olayı başka boşluklarla yeniden yazmak da bildirilir.
- **Sondan kesme** zincirin içinde hiçbir şeyi bozmaz; bunun için baş kaydı (`chain.json`: son
  olayın sıra numarası ve özeti) vardır. Baş kaydından önce biten zincir kesilmiştir.
- `mac`, özetin makinenin anahtarıyla HMAC-SHA256'sıdır. Kaydı yazabilen ama anahtarı okuyamayan
  biri doğrulanan bir zincir üretemez. Anahtarı silip MAC'leri sökmek de bildirilir: bu makinenin
  kendi zinciri anahtarsız doğrulanmış sayılmaz ve var olan zincir için yeni anahtar üretilmez.
- **Yazma:** yerel kilit altında tek yazıcı; satır tek `write` ile yazılır, `fsync` edilir, sonra
  baş kaydı değiştirilir. Çökmeden sonra zincir baş kaydının gerisinde olamaz. Windows paylaşım
  ihlalleri (`PermissionError`) beklenip yeniden denenir. 64 MiB'de yeni bölüme geçilir.
- **Çökme onarımı:** yarım kalmış son satır `quarantine/` altına taşınır ve `log.repaired` olarak
  kaydedilir; hiçbir şey sessizce atılmaz. Düzenleme ve silme hasar değil kanıttır: asla onarılmaz,
  doğrulama onları bildirmeye devam eder.
- **Sonu doğrulanmayan zincire ekleme yapılmaz** (bozuk son olay, baş kaydından kısa zincir, eksik
  baş kaydı, eksik anahtar): eylem `SecurityError` ile durur, çıkış kodu 5. Bir kesik, üstüne
  yazılarak gizlenemez.
- **Ne kaydedilir:** eylem yapan her komut (kayıt, içe ve dışa aktarma), başarısız olanlar da;
  her doğrulama kararı (`artifact.verified`); her ret. Yalın listelemeler kaydedilmez. Her olayda
  eylemi yapan (işletim sistemi kullanıcısı ve oturum türü), politika kararı ve sonuç bulunur.
- **Gizli değerler** zincire girmeden önce çıkarılır: adı parola, token, gizli anahtar gibi
  görünen parametreler, kimlik bilgisi taşıyan ortam değişkenlerinin değerleri, JWT ve özel
  anahtar biçimindeki değerler. `cev:` anahtarı asla gizlenmez. Gizli değerin özeti de yazılmaz.
- **Başlangıç politikası** (`builtin:bootstrap-v0`): `read` ve `mutate` eylemlerine izin verir ve
  kaydeder; `destructive`, `publish`, `governance`, `acknowledge` sınıfları reddedilir (çıkış kodu
  3), çünkü onları koruyacak onay katmanı henüz yoktur. Ret de kaydedilir.

## 5. Soy ağacı

İkinci bir günlük yoktur. Eser kaydeden her olay (`artifact.registered`, `artifact.derived`,
`artifact.imported`) kaydettiği manifestin girdilerini taşır; kenarlar ev dizinindeki bütün
zincirlerden (bu makineninki ve içe aktarılanlar) geri okunur. Bu yüzden soy ağacı ile denetim
kaydı farklı şeyler söyleyemez; `store.lineage.diverging` izdüşümü manifestlerle karşılaştırır
(testte boştur).

`Get-Lineage <anahtar> -Direction Up|Down|Both -Depth n`. Kenar "kaynak hedefin içine girdi"
diye okunur (tokenizer → checkpoint → model, korpus → önbellek). Kayıtlı bir eserin takma adı
(eski tokenizer özeti) o eserin anahtarı olarak gösterilir.

Sonradan kaydedilen dosyalarda yalnız dosyanın söylediği bilinir: checkpoint'in tokenizer'ı,
önbellek adındaki korpus özeti. Koşu, önbellek ve korpus kenarlarını yazan giriş noktası
(`ops.artifacts.derive`, köken `native`) hazırdır ve testlerde korpus → tokenizer → önbellek →
koşu → checkpoint → model zinciriyle sınanır; onu çağıracak yönetilen eğitim bir sonraki aşamadır.

## 6. Paketler (`.cvb`)

`Export-Artifact -To x.cvb [-IncludePayload] [-IncludeAudit]`, öbür makinede `Import-Artifact x.cvb`.
Paket bir zip dosyasıdır: `bundle.json`, manifestler, tablolar, isteğe bağlı yük dosyaları
(`.pth`/`.pkl` sıkıştırılmadan) ve isteğe bağlı denetim zinciri. Kayıtlı ataların manifestleri de
taşınır; soy ağacı taşımadan sonra kalır.

İçe aktarma **bir şey yazmadan önce her şeyi doğrular**: her manifest adına özetlenmeli ve
kanonik olmalı; checkpoint ve model manifestinin anahtarı üyelerinin anahtarı olmalı (bilinen bir
anahtara başka ağırlık iliştirilemez); her yük dosya anahtarına özetlenmeli; paketteki zincir
doğrulanmalı (boşluk reddedilir) ve aynı makinenin elde tutulan kopyasıyla çelişmemeli (çatal
reddedilir). Üye adları arşivden alınmaz, doğrulanmış özetlerden hesaplanır: hazırlanmış bir paket
ev dizininin dışına yazamaz.

Denetim zinciri varsayılan olarak pakete girmez: zincir, yerel yollarıyla birlikte o makinede
kaydedilen her eylemi içerir.

## 7. Konsol

```
cevahir                                  etkileşimli konsol (ya da standart girdiden betik)
cevahir -Command "Get-Checkpoint | Test-Checkpoint | Format-Table"
cevahir -File runbook.ps1 -Name prod     betik dosyası ve param( ) bloğunun argümanları
python -m cevahir_ai ...                 kurulum gerektirmez; pip install -e . sonrası: cevahir
  -OutputFormat Json    boru hattında kalan nesneler tek bir JSON dizisi olarak
  -Culture tr|en   -Home <dizin>   -Width <n>   -NoProfile   -NoLogo   -Version   -Help
```

**Komutlar (43).** Kayıt komutları `Cv` önekiyle de çağrılır (`Get-CvArtifact`); taşınabilir
betikler bu biçimi kullanır, çünkü gerçek PowerShell modülü onu dışa verir.

| Komut | Sınıf | Ne yapar |
|---|---|---|
| `Register-Artifact` | mutate | Var olan dosyaları kaydeder: checkpoint, tokenizer, önbellek, korpus |
| `Get-Artifact` | read | Kayıtlı eserleri listeler (`-Kind`, `-State`) |
| `Get-Checkpoint` | read | Kayıtlı checkpoint'ler (`-Latest`: dosyası en son yazılan; `-Best`: en düşük metrik) |
| `Test-Checkpoint` (`inspect`) | read | Uyumluluk kapısı: `-Level File\|Weights\|Compat`, `-Purpose`, `-Target` |
| `Get-Model`, `Get-Tokenizer`, `Get-Cache` | read | Sunulabilir model kimlikleri; tokenizer'lar (`-Tracked`); önbellekler |
| `Get-Lineage` | read | Bir eseri ne üretti, ondan ne üretildi |
| `Get-AuditEvent` | read | Denetim kaydı (`-Last`, `-Since 14d`, `-Event`, `-Action`, `-Target`, `-HostId`) |
| `Test-AuditLog` | read | Zinciri doğrular (`-Repair`: yalnız çökme hasarı) |
| `Get-Key`, `Resolve-Key`, `Compare-Key`, `Get-KeyScheme` | read | Bir dosyanın anahtarları; başvuru çözme; iki eserin üye üye farkı (mimari alanları ve tensör adlarıyla); şemalar |
| `Export-Artifact`, `Import-Artifact` | mutate | `.cvb` paketleri |
| `Get-Environment`, `Get-Responsibility` | read | Ortam; hangi sorumluluk hangi çatı modülünde |

Kabuk araçları: `Where-Object` (`where`, `?`), `ForEach-Object` (`foreach`, `%`), `Select-Object`
(`select`), `Sort-Object` (`sort`), `Group-Object` (`group`), `Measure-Object` (`measure`),
`Format-Table` (`ft`), `Format-List` (`fl`), `ConvertTo-Json`, `ConvertFrom-Json`, `Out-String`,
`Out-File`, `Out-Null`, `Write-Output` (`echo`), `Write-Host`, `Write-Warning`, `Write-Error`,
`Write-Verbose`, `Get-Command` (`gcm`), `Get-Help` (`help`), `Get-Member` (`gm`), `Get-Variable`,
`Get-ChildItem` (`ls`, `dir`), `Test-Path`, `Complete-Command`.

**Dil.** PowerShell'in katı bir alt kümesi; kabul edilen her şey PowerShell'deki anlamını taşır:

- komutlar: `-Parametre değer`, `-Anahtar`, `-Ad:değer`, konumsal argümanlar, listeler; adlar ve
  parametreler büyük-küçük harfe duyarsız, parametre tek anlamlı öneke kısalır;
- boru hattı: metin değil nesne akar; komutlar üreteçtir, `Select-Object -First 1` okumayı keser;
  diziler açılır; `|` ile biten satır devam eder;
- değişkenler: `$x = ...`, `$x += ...`, `$x.Özellik`, `$x[0]`, `"metin $x ve $($x.Key)"`;
- deyimler: `if/elseif/else`, `foreach ($x in ...) { }`, `throw`, `exit <kod>`, `param( )`;
- ifadeler: `-eq -ne -gt -ge -lt -le -like -notlike -match -notmatch -contains -notcontains -in
  -notin` (ve `-c` biçimleri), `-and -or -not !`, `+ - * / %`, `KB/MB/GB`; karşılaştırma
  büyük-küçük harfe duyarsızdır, sağ işlenen solun türüne çevrilir, soldaki dizi süzülür;
- betikler UTF-8'dir; BOM, CRLF ve LF kabul edilir; betik ilk hatada durur.

**Hata modeli ve çıkış kodları.**

| Kod | Anlam |
|---|---|
| 0 | başarılı |
| 1 | hata (bulunamadı dahil; bilinmeyen komut `CommandNotFoundException`) |
| 2 | ayrıştırma ya da kullanım hatası (`NotInCevahirSubset`, parametre bağlama) |
| 3 | politika reddi |
| 4 | onay gerekli (ayrılmış; onay katmanı yok) |
| 5 | doğrulama başarısız: bütünlük, denetim zinciri ya da uyumluluk |
| 130 | kesildi |

Tek bir girdi nesnesinde başarısız olan komut hata kaydı yazar ve boru hattı sürer
(`-ErrorAction Stop` ya da `$ErrorActionPreference = 'Stop'` sonlandırır). `Test-*` komutları
hata fırlatmaz: karar nesnesi bunu söyler ve çıkış kodu 5 olur. Çıkış kodu, sonlandıran hatanın
kodu, yoksa `exit` değeri, yoksa yazılan hataların ve başarısız doğrulamaların en yükseğidir.

**Yardım** kayıt bildiriminden üretilir (`Get-Help <komut> -Full`, `<komut> -?`,
`Get-Help about_Cevahir_Subset`, `about_Cevahir_Registry`); SÖZDİZİMİ bölümü elle yazılmaz. Dil
`CEVAHIR_UI_CULTURE`, yoksa işletim sistemi yerelinden gelir. **Tab tamamlama** saf bir
fonksiyondur (komut, parametre, izin verilen değerler, değişken, eser anahtarı, dosya yolu);
`readline` olan yerde ona bağlanır, Windows'ta `msvcrt` üzerinde küçük bir satır düzenleyicisi
vardır.

## 8. İfade dili: betik bloğu kod değildir

Betik bloğu `{ ... }` yalnız `Where-Object`, `ForEach-Object`, `Sort-Object` süzgeci ve
hesaplanan özelliğin `Expression` değeri olarak kabul edilir ve **tam olarak bir ifade** tutar.

- Ayrıştırıcı blok içinde komut, atama, deyim, boru hattı kabul etmez; `( )` ve `$( )` blok
  içinde yalnız ifade tutar. Böyle bir betik hiç çalışmaz (çıkış kodu 2).
- Değerlendirici (`console/expr.py`) Python `eval` kullanmaz, bir ana bilgisayar nesnesinde
  `getattr` çağırmaz. Yalnız şunları bilir: `$null`, mantıksal, sayı, dizge, dizi, sözlük ve
  `Record`. Üye; kaydın özelliği, sözlüğün anahtarı, dizinin ya da dizgenin `Count`/`Length`
  değeridir. Başka hiçbir şeyin üyesi yoktur; bilinmeyen üye `$null`'dır.
- Çağrılabilen tek şey sabit bir tablodaki dokuz dizge yöntemidir: `Substring`, `StartsWith`,
  `EndsWith`, `Contains`, `ToLower`, `ToUpper`, `Trim`, `Split`, `Replace`. Adı alt çizgiyle
  başlayan üye ayrıştırmada reddedilir.
- Hiçbir dizge kod olarak çalıştırılmaz. `Invoke-Expression` yoktur; betik bloğu beklenen yere
  dizge verilirse bağlama hatasıdır.
- Konsol başka program başlatmaz: cmdlet olmayan ad (`python`, `git`, `rm`) hatadır.
- Dilde `while` yoktur ve tek işlemin ürettiği dizge ve dizi bir milyon öğeyle sınırlıdır; her
  betik sonlanır. Düzenli ifade deseni 512 karakterle sınırlıdır.
- Reddedilenler: `function`, `class`, `switch`, `while`, `for`, `do`, `try/catch`, `return`, tür
  sabitleri ve dönüşümleri (`[int]`, `[System.IO.File]`), `::`, `&` ve `.` çağırma, yönlendirme
  (`>`, `>>`, `2>`), here-string, splatting, `$env:` ve diğer kapsamlı değişkenler, `-f`,
  `-replace`, `-split`, `-join`, `-is`, bit işleçleri.

`tests/evolution/test_cevahir_ai_console_language.py` içinde 25 blok içi sızma denemesi dört
komutla (100 durum), 54 alt küme dışı yapı ve 15 dış program denemesi vardır; her birinde hiçbir
şeyin çalışmadığı ve dosya oluşmadığı denetlenir.

## 9. Gerçek PowerShell modülü (isteğe bağlı)

`powershell/Cevahir/Cevahir.psd1` ve `Cevahir.psm1`, komut kaydından üretilir
(`python -m cevahir_ai.console.tools.gen_powershell --write`; `--check` eskimişse 1 döner; test
bunu denetler). PowerShell 7.2+ ister. Her kayıt komutu için `Cv` önekli bir fonksiyon vardır
(`Get-CvCheckpoint | Test-CvCheckpoint -Level Compat`); kabuk araçları orada PowerShell'in
kendisidir. Fonksiyon, `python -m cevahir_ai -Invoke` sürecine standart girdiden tek bir JSON
isteği yollar ve JSON yanıtı nesnelere, uyarılara ve hatalara çevirir. Parametre değerleri veri
olarak gider, komut satırına eklenmez; alıntılanacak ve sızdırılacak bir şey yoktur.
`Invoke-Cevahir '<betik>'` konsol dilindeki bir betiği çalıştırır.

```powershell
Import-Module .\powershell\Cevahir
Get-CvCheckpoint | Test-CvCheckpoint | Format-Table Short, Verdict, C1, C12
```

**Bu modül gerçek bir PowerShell'de çalıştırılmadı** (bu ortamda `pwsh` yok). Sınanan: üretilen
metnin kayıtla aynı adımda olduğu ve dayandığı `-Invoke` protokolünün çalıştığı.

## 10. Gerçek bir oturum

Aşağıdaki çıktı bu depoda üretildi (Linux, Python 3.13, `CEVAHIR_UI_CULTURE=tr`, genişlik 118).
`saved_models` klasörü gerçek yazıcılarla yazıldı: üç epoch'luk eğitim checkpoint'i (`best.pth`
ve `last.pth` kopyalarıyla), çıplak bir `state_dict`, pickle'lanmış bir modül ve yanında tokenizer
dosyasıyla CEV-11 biçiminde bir checkpoint. Her satır ayrı bir `cevahir -Command` çağrısıdır;
köşeli parantezdeki süre o çağrının tamamıdır (yorumlayıcının açılışı dahil).

```
PS cevahir> Get-Environment | Format-List Version, Python, Platform, Policy, Torch, TorchLoaded

Version     : 0.4.0
Python      : 3.13.18
Platform    : Linux
Policy      : builtin:bootstrap-v0
Torch       : 2.15.0
TorchLoaded : False
[çıkış kodu 0; 0.14 s]

PS cevahir> Register-Artifact ./saved_models -Recurse | Format-Table Short, Kind, State, New

Short              Kind  State       New
-----              ----  -----       ---
ckpt:2190a35e8985  ckpt  registered  True
ckpt:8ebc0bd9457c  ckpt  registered  True
model:46270d6675df model registered  True
tok:09a9b9f05811   tok   registered  True
ckpt:b2587a731d56  ckpt  registered  True
model:3128aa071088 model registered  True
ckpt:e6e260ac18dc  ckpt  registered  True
model:2f6c8fa9af21 model registered  True
ckpt:de7c50a9ee81  ckpt  registered  True
model:aab01046a564 model registered  True
file:f648e0d8d598  file  quarantined True
WARNING: saved_models/bare/weights.pth: bare state dict: no architecture, no tokenizer identity
WARNING: saved_models/bare/weights.pth: no model key: a servable identity needs a known architecture and a canonical tokenizer key
WARNING: saved_models/full/full_model.pth: registered at file level only: loading needs code execution (pickled module or arbitrary object); globals outside the weights-only safe set: src.neural_network.CevahirNeuralNetwork
[çıkış kodu 0; 2.29 s]

PS cevahir> Get-Checkpoint | Sort-Object Epoch -Descending | Select-Object -First 3 | Format-Table Short, Epoch, Metric, Envelope, Tok, @{Name='Kopya'; Expression={ $_.Locations.Count }}

Short             Epoch Metric Envelope Tok              Kopya
-----             ----- ------ -------- ---              -----
ckpt:de7c50a9ee81     3    2.6 tm2      tok:d211997f8c7d     2
ckpt:b2587a731d56     2    2.4 tm2      tok:d211997f8c7d     2
ckpt:e6e260ac18dc     1    2.9 tm2      tok:d211997f8c7d     1
[çıkış kodu 0; 0.16 s]

PS cevahir> Get-Checkpoint | Test-Checkpoint | Format-Table

Key               Level Purpose Verdict    C1   C3 C4 C5 C7 C9 C12
---               ----- ------- -------    --   -- -- -- -- -- ---
ckpt:2190a35e8985 File  inspect compatible pass -  -  -  -  -  pass
ckpt:8ebc0bd9457c File  inspect compatible pass -  -  -  -  -  pass
ckpt:b2587a731d56 File  inspect compatible pass -  -  -  -  -  pass
ckpt:e6e260ac18dc File  inspect compatible pass -  -  -  -  -  pass
ckpt:de7c50a9ee81 File  inspect compatible pass -  -  -  -  -  pass
[çıkış kodu 0; 0.17 s]

PS cevahir> Get-Checkpoint -Best | Test-Checkpoint -Level Compat -Purpose serve | Format-Table -View Checks

Id  Name              Status Required Detail
--  ----              ------ -------- ------
C12 envelope.safety   pass   True     the pickle imports 4 globals, all in the weights-only safe set
C1  file.integrity    pass   True     file bytes equal the expected key
C3  weights.digest    pass   True     weights equal the expected key
C4  wschema.match     pass   True     names, shapes and ties match the declared architecture
C5  arch.declared     pass   True     declared architecture is accepted by the current code (cevahir-capabilities-1);…
C7  tok.identity      pass   True     legacy digest is the tracked tokenizer 'default' (read_only=absent, merges=LF);…
C9  dtype             pass   True     floating tensors are float32
C11 context.window    pass   True     serving context 64 (none requested: the trained context applies) vs trained 64
C10 resume.signatures pass   False    optimizer state present; no target optimizer or scheduler given
[çıkış kodu 0; 3.76 s]

PS cevahir> Get-Checkpoint -Best | Test-Checkpoint -Level Compat -Purpose evaluate -Target cev11-primary | Format-List Verdict, Failing, SummaryTr

Verdict   : incompatible
Failing   : {C5}
SummaryTr : 'evaluate' amacı için uyumsuz: C5 arch.declared: architecture differs from the target: vocab_size: checkpoint=128, target=4096; embed_dim: checkpoint=32, target=192; seq_proj_dim: checkpoint=32, target=192; num_layers: checkpoint=2, target=6; ffn_dim: checkpoint=64, target=512
[çıkış kodu 5; 3.74 s]

PS cevahir> $m = Get-Model | Select-Object -First 1; Get-Lineage $m.Key -Direction Up | Format-Table -View Tree

Lineage                            How                 Host               Seq
-------                            ---                 ----               ---
<- ckpt:8ebc0bd9457c  [checkpoint] artifact.registered h-335010267e3c47f3   4
  <- tok:09a9b9f05811  [tokenizer] artifact.registered h-335010267e3c47f3   3
[çıkış kodu 0; 0.19 s]

PS cevahir> Get-AuditEvent -Last 5 | Format-Table Seq, Event, Action, Decision, Outcome, Target

Seq Event             Action          Decision Outcome Target
--- -----             ------          -------- ------- ------
 17 artifact.verified Test-Checkpoint allow    ok      ckpt:b2587a731d56
 18 artifact.verified Test-Checkpoint allow    ok      ckpt:e6e260ac18dc
 19 artifact.verified Test-Checkpoint allow    ok      ckpt:de7c50a9ee81
 20 artifact.verified Test-Checkpoint allow    ok      ckpt:b2587a731d56
 21 artifact.verified Test-Checkpoint allow    failed  ckpt:b2587a731d56
[çıkış kodu 0; 0.15 s]

PS cevahir> Test-AuditLog | Format-Table

Host               Own  Ok   Events Segments Mac      FirstBadSeq Findings
----               ---  --   ------ -------- ---      ----------- --------
h-335010267e3c47f3 True True     21        1 verified                    0
[çıkış kodu 0; 0.13 s]

PS cevahir> Get-Checkpoint | ForEach-Object { Remove-Item $_.Path }
cevahir : [InvalidArgument] a command or bare word inside a script block (it holds one expression; text needs quotes): not in the console language (see Get-Help about_Cevahir_Subset) (line 1, column 35)
    + FullyQualifiedErrorId : NotInCevahirSubset,Cevahir.Commands.Console
[çıkış kodu 2; 0.10 s]

PS cevahir> python -c "print(1)"
cevahir : [ObjectNotFound] 'python' is not a command of the console. The console runs its own cmdlets and .ps1 scripts of its own language; it never starts another program.
    + FullyQualifiedErrorId : CommandNotFoundException,Cevahir.Commands.Console
    + Recommended : Get-Command  (lists the commands);  Get-Help about_Cevahir_Subset
[çıkış kodu 1; 0.12 s]

# (checkpoint_epoch_0001.pth dosyasının ortasındaki 4 bayt değiştirildi)

PS cevahir> Get-Checkpoint | Test-Checkpoint | Where-Object Ok -eq $false | Format-List Short, Verdict, C1, Intact, Summary

Short   : ckpt:e6e260ac18dc
Verdict : incompatible
C1      : fail
Intact  : no
Summary : incompatible for 'inspect': C1 file.integrity: file bytes differ from the expected key (tampered, truncated or another file)
[çıkış kodu 5; 0.19 s]

PS cevahir> Get-Artifact -State quarantined | Format-Table Short, Kind, State

Short             Kind State
-----             ---- -----
ckpt:e6e260ac18dc ckpt quarantined
file:f648e0d8d598 file quarantined
[çıkış kodu 0; 0.16 s]

# (denetim kaydında bir satırdaki "incompatible" sözcüğü "compatible" yapıldı)

PS cevahir> Test-AuditLog | Format-Table Host, Ok, Events, Mac, FirstBadSeq

Host               Ok    Events Mac      FirstBadSeq
----               --    ------ ---      -----------
h-335010267e3c47f3 False     26 verified          21
[çıkış kodu 5; 0.14 s]

PS cevahir> Test-AuditLog | Select-Object -ExpandProperty Findings | Format-Table Seq, Segment, Line, Problem

Seq Segment          Line Problem
--- -------          ---- -------
 21 seg-000001.jsonl   21 hash mismatch: the event was edited
[çıkış kodu 5; 0.12 s]
```

Okuma notları: `Register-Artifact` 2,3 saniyesinin çoğunu torch'u içe aktarmaya harcar; listeleme
ve dosya düzeyi sınama torch'suzdur. `serve` amacıyla tam kapı kayıtlı checkpoint için geçer,
çünkü baytlar ve ağırlıklar manifestle karşılaştırılır. Hedef mimari farklıysa karar "uyumsuz"dur
ve çıkış kodu 5'tir, ama hata kaydı yazılmaz.

## 11. Testler

| Dosya | Ne sınar |
|---|---|
| `tests/evolution/test_cevahir_ai_audit.py` | zincir; kurcalama takımı (tek bayt, silinen satır, yeri değişen satırlar, eklenen sahte satır, yeniden yazılan boşluklar, sondan kesme, silinen baş kaydı ve günlük, anahtarsız yeniden yazılan zincir, silinen anahtar, başka zincirden taşınan olay); çökme onarımı; bölüm geçişi; aynı anda yazan dört süreç; kilit; `PermissionError` yeniden denemesi; gizli değerlerin çıkarılması; başlangıç politikası |
| `tests/evolution/test_cevahir_ai_registry.py` | nesneler, manifest, makine kimliği, konumlar, tarama ve tanıma, başvurular, soy ağacı, paketler ve reddettikleri (bozuk yük, eksik manifest, değiştirilmiş manifest, zincirde boşluk, çatal, yol kaçışı, anahtarı üyeleriyle tutmayan manifest); gerçek yazıcılarla yazılmış checkpoint'lerin kaydı ve manifestine karşı sınanması |
| `tests/evolution/test_cevahir_ai_console_language.py` | ayrıştırıcı, ifade dili (PowerShell anlamları), sızma denemeleri, sınırlar, bağlama, boru hattı |
| `tests/evolution/test_cevahir_ai_console.py` | tablo, liste ve JSON altın çıktıları; yardım (tr/en, her örnek ayrışır); tamamlama; Windows yolları; giriş noktası ve çıkış kodları; CRLF ve BOM'lu betikler; `-Invoke` protokolü; açılış süresi ve torch'suzluk; etkileşimli döngü; satır düzenleyicisi; PowerShell modülünün kayıtla aynı adımda olması; uçtan uca oturum; yerelde varsa gerçek bir CEV-11 taban checkpoint'i (salt okunur) |

## 12. Sınırlar (açıkça)

**Yönetilmeyenler**

- **Politika, onay, imza, çapa yok.** Başlangıç politikası iki kuraldır. Hiçbir eylem imzalanmaz;
  hiçbir onay istenmez. `Published` her zaman `False`'tur: yayınlama, yönetilen eğitim, önbellek
  hazırlama, değerlendirme ve sunum komutları yoktur (P5).
- **Kütüphane kipi yönetilmez.** `python training_system/train.py`, `ModelManager.load` ve eski
  yükleyiciler değişmedi; denetlenmez ve kaydedilmez. Kayıt yalnız konsoldan ya da
  `cevahir_ai.ops` üzerinden yapılan işleri görür.
- **Aynı kullanıcıyla çalışan bir süreç zinciri yeniden yazabilir.** MAC anahtarı ev dizininde,
  kullanıcının okuyabildiği bir dosyadadır (Windows'ta DPAPI ile sarılmaz; yalnız dosya
  izinlerine bırakılmıştır). Zinciri, baş kaydını ve MAC'leri birlikte yeniden yazan ya da tutarlı
  bir önceki hâle geri saran biri yerelde yakalanamaz. Bunu yakalamak, baş kaydının imzalı bir
  kopyasının makine dışında tutulmasını gerektirir; bu yoktur. Zincir bugün hataları, tek tek
  düzenlemeleri ve anahtarı okuyamayan birinin müdahalesini yakalar.
- **Ekleme yalnız zincirin sonunu denetler.** Ortadaki bir düzenlemeyi `Test-AuditLog` bildirir,
  ama sonu sağlam kaldığı sürece yeni olaylar eklenmeye devam eder.
- **Kesilmiş zincir için kurtarma komutu yok.** Sonu doğrulanmayan zincire ekleme yapılmadığı
  için kayıt eylemleri durur. Bilinçli çözüm elledir: `audit/<makine>` klasörü kanıt olarak başka
  yere taşınır, bir sonraki eylem yeni bir zincir başlatır.
- **Kayıt, baytları kaydedildikleri andaki hâliyle sabitler;** o anda doğru olduklarını
  kanıtlamaz. C1 ve C3 "kayıttan beri değişmedi" demektir.
- **Paketler imzasızdır.** İçe aktarma paketin kendi içinde tutarlı olduğunu kanıtlar, kimin
  ürettiğini değil. Tokenizer, önbellek ve korpus manifestlerinin anahtarı, yük dosyası olmadan
  yeniden hesaplanamaz; yalnız checkpoint ve model manifestlerinin anahtarı üyelerinden denetlenir.
- **İçe aktarma tek adımda geri alınamaz:** doğrulama bittikten sonra yazma sırasında bir hata
  olursa (disk dolması) kısmi durum kalabilir; yeniden içe aktarmak tamamlar.
- **`Out-File` denetim kaydına yazılmaz.** Depodaki bayt korumalı dizinlere ve `CEVAHIR_HOME`
  içine yazmayı reddeder; başka her yere yazar.

**Tasarımda olup yapılmayanlar**

- SQLite dizini yok: katalog manifestleri okuyarak kurulur (manifest sayısıyla doğrusal).
  "10 bin eserde 50 ms altında tamamlama" hedefi ölçülmedi ve bu yapıyla tutmaz.
- Yeniden kurulabilir dizin, `refs/`, başka bir ev dizininden içe aktarma (`-Home`), "son dışa
  aktarmadan beri" artımlı zincir, `Export-Lineage`, `Update-Index`, `Get-WeightSchema`,
  `Compare-WeightSchema`, `Test-Tokenizer`, `Compare-Tokenizer`, `Get-Corpus`, `Test-Cache`,
  `Test-Environment`, `Get-Setting`, `Test-Config`, `Test-Book`, `Test-Publication`, `Test-Study`
  yok. `python -m cevahir_ai.keys` komut satırına kayıt alt komutları eklenmedi; konsol onların
  yerini tutar.
- Kalıcı ana süreç (`serve --stdio`) ve çalışan süreçleri yok: PowerShell modülü her çağrıda yeni
  bir Python süreci başlatır (yaklaşık 0,15 s, torch gerekiyorsa birkaç saniye).
- `prompt_toolkit`, MAML yardımı, `Format.ps1xml`, Pester testleri, Python ↔ pwsh uyum takımı yok.
- Konsol dili tasarımdakinden bilerek dardır: betik bloğu tek ifadedir, deyim tutmaz. `..` aralık
  işleci, `-join`, `-replace`, `-f`, `try/catch`, özellik ataması (`$x.A = 1`) yok. Komut
  argümanındaki `[ab]*` joker deseni tür sabiti sanılıp reddedilir; tırnak içinde yazılmalıdır.
- `-WhatIf` yalnız `Register-Artifact`, `Export-Artifact`, `Import-Artifact` ve `Out-File` için
  vardır; `-Confirm` yoktur.

**Sınanmayanlar**

- **Windows'ta hiçbir şey çalıştırılmadı.** Windows davranışı yalnız `ntpath` ve satır sonu
  birim testleriyle, kilit ve paylaşım ihlali yolu ise POSIX'te ve hata enjekte eden bir açıcıyla
  sınandı. `msvcrt.locking` dalı, `msvcrt` satır düzenleyicisi (yalnız betiklenmiş tuşlarla
  sınandı; `CEVAHIR_LINE_EDITOR=plain` onu kapatır), cp1254 konsolu ve `%LOCALAPPDATA%` altında
  gerçek bir ev dizini denenmedi.
- **PowerShell modülü gerçek bir PowerShell'de çalıştırılmadı.**
- Colab ve Drive üzerindeki ev dizini denenmedi; makine kimliğinin sanal makine başına tutulması
  yalnız yol hesabı düzeyinde sınandı.
- macOS'ta `readline` yerine `libedit` bağlaması denenmedi.
- Büyük ölçek ölçülmedi: binlerce manifest, 64 MiB'lik bölümler, çok gigabaytlık checkpoint
  (özetleme akışlıdır, ama süre ölçülmedi).
- `-match`, Python'ın geri izlemeli düzenli ifade motorunu kullanır: kendi betiğinizdeki
  patolojik bir desen kendi oturumunuzu kilitleyebilir.
