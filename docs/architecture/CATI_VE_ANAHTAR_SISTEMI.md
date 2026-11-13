# Çatı paketi ve merkezi anahtar sistemi (`cevahir_ai`)

> **English summary.** `cevahir_ai` is a new top-level package ("the roof") over the existing
> packages. It adds (1) unversioned lazy facades (`cevahir_ai.neural`, `.model`, `.tokenizer`,
> `.data`, `.training`, `.cognition`, `.serving`) whose names are the legacy objects themselves;
> (2) a central key system: every artefact gets a typed, content-derived key
> `cev:<kind>:<scheme>:<sha256>` (tokenizer, corpus, cache, architecture, weight schema, weights,
> checkpoint, model, run, evaluation), with canonical encodings that do not depend on CRLF/LF or
> the working directory; (3) aliases for the identities that already exist (`tokenizer_digest` v3,
> md5 cache names, CEV-11 `model_sha256`, `.sha256` sidecars, the book's fingerprints), so no
> existing checkpoint or cache is invalidated; (4) a weight-schema registry (layouts read from the
> constructor on the meta device, golden snapshots, declarative migrations); (5) one pure
> compatibility gate used by the governed load path `cevahir_ai.model.checkpoint.open_checkpoint`;
> (6) the CLI `python -m cevahir_ai.keys inspect <file>`, which answers for any checkpoint: is it
> intact, which tokenizer and architecture, is it compatible with the current code, and if not,
> why. The legacy loaders are unchanged and are not governed. The registry, audit log, policy
> layer and the PowerShell-like console are later phases and do not exist yet. Limits are listed
> in the last section.

Bu sayfa, çatı paketinin ilk iki aşamasında (P1: güvenlik ağı ve paket iskeleti, P2: anahtar
sistemi ve uyumluluk kapısı) depoya girenleri anlatır. Yapılmayanlar son bölümdedir.

## 1. Neden

- Depoda üst paketler (`src`, `model_management`, `tokenizer_management`, `training_system`,
  `training_management`, …) ve v2/v3 kopyaları var; hangisinin canlı olduğu addan anlaşılmıyor.
- Kimlikler dağınıktı: tokenizer için iki ayrı özet, önbellek adında md5, ağırlıklar için ürün
  tarafında hiç özet yok, CEV-11'de ayrı bir özet. 64 haneli bir değerin hangi şemaya ait olduğu
  değerden anlaşılmıyordu.
- Yükleme denetimleri eksik meta veride atlanıyordu. Ölçülen örnek: 4 başlı bir modelin çıplak
  `state_dict` dosyası 8 başlı modele hatasız yükleniyor, çünkü bütün tensör şekilleri aynı;
  model ise başka bir fonksiyon hesaplıyor.

## 2. Çatı paketi

`import cevahir_ai` torch'u, numpy'ı ve eski paketlerin hiçbirini yüklemez, loglamayı
yapılandırmaz (ölçülen içe aktarma süresi yaklaşık 0,03 s). Eski içe aktarma yolları aynen
çalışır; hiçbir modül taşınmadı.

| Çatı adı | Bugünkü karşılığı |
|---|---|
| `cevahir_ai.neural.CevahirNeuralNetwork`, `resolve_ffn_dim` | `src.neural_network`, `src.neural_network_module.architecture_contracts` |
| `cevahir_ai.model.ModelManager`, `ModelInitializer`, `ModelSaver`, `ModelLoader` | `model_management.*` |
| `cevahir_ai.model.config.*` (`SystemConfig`, `normalize_model_config`, `tiny_model_config`, …) | `model_management.config_schema` (`SystemConfig` = oradaki `CevahirConfig`) |
| `cevahir_ai.model.checkpoint.open_checkpoint`, `read_envelope`, `inspect_checkpoint` | **yeni** (kapıdan geçen yükleme); `unpack_checkpoint`, `validate_*` eski sözleşmedir |
| `cevahir_ai.tokenizer.LegacyBPETokenizer`, `ByteLevelBPE`, `tokenizer_key` | `TokenizerCore`, `tokenizer_management.byte_bpe.ByteLevelBPE` |
| `cevahir_ai.data.DataCache`, `StrictCache`, `split_training_records`, `record_format` | `training_system.data_cache`, `training_system.v3.data.cache_v3.DataCacheV3`, … |
| `cevahir_ai.training.TrainingService`, `TrainingEngine`, `CheckpointManager`, `TrainingScheduler` | `training_system.v3…TrainingServiceV3`, `training_management.v2…` (canlı üyeler) |
| `cevahir_ai.cognition.CognitiveManager`, `CognitiveManagerConfig` | `cognitive_management` |
| `cevahir_ai.serving.Cevahir`, `FacadeConfig`, `create_app` | `model.cevahir` (`FacadeConfig` = oradaki `CevahirConfig`), `api.app_factory` |

Kurallar (her biri `tests/evolution/test_cevahir_ai_roof.py` içinde bir testtir):

- Çatı adı eski nesnenin **kendisidir** (`cevahir_ai.model.ModelManager is model_management.model_manager.ModelManager`); kopya ya da sarmalayıcı yoktur.
- Sessiz geri düşüş yoktur: hedef bulunamazsa ilk kullanımda `ImportError`/`AttributeError` çıkar.
- torch yalnız `cevahir_ai/keys/torchkeys/` altında modül düzeyinde içe aktarılır; eski paketlere yalnız `cevahir_ai._compat.repo.import_legacy` üzerinden, fonksiyon içinde erişilir.
- v2/v3 çiftlerinin yalnız canlı üyesi, sürümsüz adla dışa verilir. Tablo `cevahir_ai/_roof.py` içindedir (`EXPORTS`, `DEPRECATED`, `RESPONSIBILITIES`).

`pyproject.toml` artık paket meta verisi taşır (`pip install -e .` ile `cevahir` komutu gelir);
bağımlılık listesi boştur, çünkü çatı çekirdeği yalnız standart kütüphaneyi kullanır.
`python -m cevahir_ai --version` depodan kurulum gerektirmeden çalışır.

## 3. Anahtar dilbilgisi ve kanonik kodlamalar

```
cev:<tür>:<şema>:<özet>      örn. cev:tok:bpe-sem-v1:d211997f8c7d…
tür   = [a-z][a-z0-9]{1,15}
şema  = ad-vN                 (şema anahtarın parçasıdır; sürüm artışı yeni şemadır)
özet  = 64 hane sha256 (yalnız cache:md5-v2 ve run:id-v1 için 32 hane)
```

- Bilinmeyen `(tür, şema)` çifti ayrıştırma hatasıdır; sessizce kabul edilmez.
- **CJSON-1** (yapısal yükler): anahtarlar sıralı, boşluksuz, UTF-8; dizgeler veridir ve
  normalleştirilmez; `-0.0` → `0.0`; NaN/sonsuz, 64 biti aşan tamsayı ve dizge olmayan sözlük
  anahtarı reddedilir. Her türün yükü tipli izdüşümdür (`1`, `1.0`, `True` aynı anlam için iki
  anahtar üretemez).
- **Alan ayrımı:** `özet = sha256(b"cev:<tür>:<şema>\0" + CJSON-1(yük))`. Bir şemanın özeti
  başka bir şemanınkiyle karışamaz.
- **TEXT-LF-1** (metin dosyaları): `utf-8-sig` çöz, `\r\n` ve tek `\r` → `\n`, UTF-8 kodla.
  Kitabın `utf8-lf` parmak iziyle aynıdır; Windows (CRLF) ve Linux (LF) aynı anahtarı verir.
- Dosya anahtarları bilerek alan ayrımsızdır: `file:bytes-v1` = `sha256sum`, `file:utf8lf-v1` =
  kitap parmak izi.
- Yük içindeki yollar kök dizine göreli, POSIX ayraçlı ve NFC'dir; çalışma dizini ve mutlak konum
  anahtarı etkilemez.

Altın vektörler `tests/evolution/test_cevahir_ai_keys.py` içindedir.

## 4. Şemalar

`python -m cevahir_ai.keys schemes` tabloyu yazdırır (`cevahir_ai/keys/registry.py`).

| Anahtar | Durum | Neyi özetler |
|---|---|---|
| `file:bytes-v1`, `file:utf8lf-v1` | birincil | ham baytlar; TEXT-LF-1 baytları |
| `corpus:files-v1`, `evalset:files-v1` | birincil | sıralı `[göreli yol, dosya anahtarı]` (.txt/.json metin, .docx bayt) |
| `corpus:legacy-dd-v1` | eski | `data_directory_digest`, aynen çağrılır |
| `tok:bpe-sem-v1` | birincil | token→id eşlemi, sıralı birleştirmeler, özel tokenler, id anlamını etkileyen ayarlar |
| `tok:bytebpe-v1` | birincil | `ByteLevelBPE.identity()` aynen |
| `tok:legacy-v3` | eski | `tokenizer_digest` (identity_version 3), canlı çekirdek üzerinde aynen çağrılır |
| `cache:recipe-v1` | birincil | korpus anahtarı + tokenizer anahtarı + kodlama profili + kayıt biçimi + üretici sürümü |
| `cache:md5-v2` | eski | `cached_data_<md5>_<veriözeti>.pkl` adındaki md5 |
| `split:assign-v1` | birincil | önbellek anahtarı + oran + tohum + yöntem |
| `arch:cfg-v1` | birincil | `normalize_model_config` çıktısının işlevi belirleyen 24 alana tipli izdüşümü |
| `arch:ctor-v0` | eski | checkpoint'teki `model_config` (kurucu argümanları), `device` hariç |
| `serveprof:cfg-v1` | birincil | sunum zamanı alanları (niceleme, bağlam uzunluğu, KV önbelleği, dtype) |
| `wschema:layout-v1`, `wschema:sd-v1` | birincil | `state_dict` adları + şekilleri + takma ad grupları (`sd-v1`: + dtype) |
| `weights:sha256-v1` | birincil | tensör tablosu: ad, dtype, şekil, takma ad, ham baytların sha256'sı |
| `weights:cev11-fp32-v1` | eski | CEV-11 `model_sha256` (bayt bayt eşdeğer yeniden uygulama) |
| `optim:sig-v1`, `sched:sig-v1` | imza | yalnız yapı (sürdürme uyumu) |
| `optim/sched/tstate:sha256-v1` | birincil | eğitim durumunun içeriği |
| `ckpt:members-v1` | birincil | üye anahtarları: weights, arch, wschema, tok, optim, sched, tstate |
| `model:triple-v1` | birincil | sunulabilir kimlik: weights + arch + kanonik tokenizer |
| `runspec:cfg-v1`, `run:id-v1` | birincil, olay | yolları girdi anahtarlarıyla değiştirilmiş eğitim ayarı; tek bir koşu (uuid7) |
| `eval:recipe-v1` | birincil | model + değerlendirme seti + metrik sürümü + çözümleme ayarı + tohum |
| `code:tree-v1`, `env:rt-v1` | birincil | dosya kümesi; çalışma ortamı (makine ve kullanıcı adı yok) |

### Ne kimliktir, ne değildir

- **Tokenizer (`bpe-sem-v1`).** Kimlik: id'ler, birleştirmeler ve sıraları, `lowercase` gibi id
  anlamını etkileyen ayarlar, sınıflandırılmamış her yeni ayar. Kimlik değil: `read_only`,
  toplu iş boyutu, cihaz, çözümleme ve tanılama ayarları, `merges.txt` satır sonları, BOM,
  token başına frekans istatistikleri. Sınıflar `cevahir_ai/keys/classify.py` içindedir.
- **Mimari (`cfg-v1`).** 24 alan (ör. `num_heads`, `num_kv_heads`, `ffn_dim`, `pe_mode`,
  `parallel_residual`, `sliding_window`, `tie_weights`). Türetilen alanlar kurucunun yaptığı gibi
  çözülür (`num_kv_heads = num_heads`, `ffn_dim = resolve_ffn_dim(...)`); kapalı özelliklerin
  alanları (MoE kapalıyken `num_experts`) boşaltılır. `dropout`, `device`, `max_seq_length`,
  `quantization_type` mimari anahtarına girmez. `normalize_model_config` **değiştirilmedi**;
  anahtar onu sarar. `config_schema.py` yalnız standart kütüphane kullandığı için mimari anahtarı
  torch'suz hesaplanır.
- **Ağırlıklar (`sha256-v1`).** dtype kimliktir (fp16 kopyası başka anahtardır); ad ve şekil
  kimliktir; bellek düzeni ve cihaz değildir; bağlı ağırlıklar bir kez özetlenir. Aynı ağırlıklar
  eğitim checkpoint'inde, `ModelManager.save` dosyasında ve çıplak `state_dict` içinde **aynı**
  anahtarı verir; yalnız dosya anahtarı değişir. Nicelenmiş, seyrek ve meta tensörler reddedilir.

## 5. Var olan kimlikler: hiçbir şey geçersizleşmez

| Bugün saklanan | Anahtar biçimi | Nasıl doğrulanır |
|---|---|---|
| checkpoint `tokenizer_identity` (64 hane) | `cev:tok:legacy-v3:<hane>` | özgün `tokenizer_digest`; uzlaştırma; izlenen kayıt |
| `ByteLevelBPE.identity()` | `cev:tok:bytebpe-v1:<hane>` | checkpoint yanındaki tokenizer dosyası ya da canlı nesne |
| CEV-11 `model_sha256` | `cev:weights:cev11-fp32-v1:<hane>` | yeniden hesap (`cev11.model.model_sha256` ile eşitliği testte) |
| checkpoint `model_config` | `cev:arch:ctor-v0` → `cev:arch:cfg-v1` | değişmemiş normalleştirici |
| `.sha256` yan dosyası | `cev:file:bytes-v1` | dosyanın sha256'sı |
| `cached_data_<md5>_<özet>.pkl` | `cev:cache:md5-v2` + `cev:corpus:legacy-dd-v1` | özgün `DataCacheV3.get_cache_key` |
| kitap `utf8-lf` parmak izleri | `cev:file:utf8lf-v1` | 168 dosyanın tamamı için testte eşit |

**Eski tokenizer özetinin uzlaştırılması.** `tokenizer_digest` aynı tokenizer için farklı
değerler üretir: `read_only=True`, `read_only=False` ve bayrağın hiç verilmediği yazılabilir
çekirdek (üretim eğitimi böyledir) üç ayrı değer verir; `merges.txt` CRLF ise üç değer daha.
`cevahir_ai.adapters.tokenizers.legacy_candidates` bu altı değeri **özgün fonksiyona**
hesaplatır. Eşleşen değer "aynı tokenizer, özet kayması" olarak kabul edilir.
`governance/registry/tokenizers.json` depoda izlenen tokenizer'ın kanonik anahtarını, dosya
anahtarlarını ve altı eski özetini tutar; böylece tokenizer verilmeden de eski bir checkpoint'in
hangi tokenizer'la eğitildiği söylenebilir. Bu altı adaydan hiçbiriyle eşleşmeyen bir değer
(ör. frekans istatistikleri yenilenmiş bir sözlük) çözülemez; kapı bunu "bilinmiyor" sayar.

## 6. Ağırlık şeması kaydı

- **Doğruluk kaynağı kurucudur.** Beklenen `state_dict` düzeni, `CevahirNeuralNetwork`
  `torch.device("meta")` altında kurularak okunur: bellek ayrılmaz, 100 milyon parametreli
  varsayılan model milisaniyelerde kurulur, küresel RNG durumu geri yüklenir. YaRN/doğrusal RoPE
  ölçeklemesi tensör *değerleri* okuduğu için meta cihazda kurulamaz; bu durumda ölçekleme
  etkisizleştirilip yeniden denenir (RoPE tabloları kalıcı olmayan tamponlardır, düzen aynıdır;
  gerçek kurulumla eşitliği testtedir), o da olmazsa gerçek CPU kurulumu yapılır.
- **Altın düzenler:** `cevahir_ai/keys/schemas/layouts/{tiny,train-default,cev11-primary}.json`.
  `python -m cevahir_ai.keys.schemas --check` her birini yeniden kurup karşılaştırır. Bir tensör
  adı ya da şekli değişirse denetim kırmızı olur; yazar ya geri alır ya da
  `ARCHITECTURE_VERSION`'ı artırır, bir göç ekler ve ancak ondan sonra `--write` çalıştırır.
  Ön ayarlar bu kayda aittir: `train-default`, kayıt oluşturulduğu andaki `TRAIN_CONFIG` model
  bölümüdür ve sonraki `TRAIN_CONFIG` düzenlemelerini izlemez.
- **Göçler** bildirimseldir (`migrations/*.json`): `strip_prefix`, `rename`, `rename_regex`,
  `drop`, `tie`, `untie`, `cast` (yalnız yukarı). İlk göç `strip-wrapper-prefixes/1`
  (`module.`, `_orig_mod.` önekleri) işlevi korur; göç öncesi ve sonrası tiny model logit'lerinin
  birebir aynı olduğu testtedir. Kapı bir göçü yalnız çağıran açıkça izin verirse bellekte uygular.
- **Sürümler tablosu** (`cevahir_ai/keys/versions.py`): depodaki dokuz biçim/kimlik sayacı
  sahibiyle birlikte kayıtlıdır; bir sayaç tablo güncellenmeden artırılırsa test kırılır.

## 7. Uyumluluk kapısı

`cevahir_ai.keys.gate.check_compatibility(olgular, amaç, politika)` saf bir fonksiyondur: dosya
okumaz, torch kullanmaz. Olguları `cevahir_ai.adapters.inspection` üretir.

| Kimlik | Ad | Ne karşılaştırılır |
|---|---|---|
| C1 | `file.integrity` | dosya baytları ↔ `.sha256` yan dosyası ya da verilen anahtar |
| C3 | `weights.digest` | yeniden hesaplanan ağırlık özeti ↔ beyan edilen (CEV-11) ya da verilen anahtar |
| C4 | `wschema.match` | dosyadaki düzen ↔ beyan edilen mimari için **bugünkü kodun** kurduğu düzen; bağ bozulması; kısmi yükleme yok |
| C5 | `arch.declared` | checkpoint'in mimari anahtarı ↔ hedef modelinki; mimari yoksa "bilinmiyor" |
| C7 | `tok.identity` | tokenizer kimliği ↔ hedef tokenizer (şemaya göre; eski özet için uzlaştırma) |
| C9 | `dtype` | saklanan dtype ↔ hedef; sessiz dönüştürme yok; nicelenmiş → desteklenmiyor |
| C10 | `resume.signatures` | optimizer ve zamanlayıcı yapısı |
| C11 | `context.window` | sunum bağlam uzunluğu ≤ eğitimdeki |
| C12 | `envelope.safety` | pickle'ın içe aktaracağı adlar `weights_only` güvenli kümesinde mi (dosya açılmadan) |

| Amaç | Gerekenler | "Bilinmiyor" ne demek |
|---|---|---|
| `inspect` | C12, C1 | C1 → uyarı |
| `evaluate`, `finetune` | C12, C1, C3, C4, **C5**, C7 | C1, C3, C7 → uyarı; **C4, C5 → ret** |
| `resume` | yukarıdakiler + C10 | C10 → ret (yalnız ağırlık dosyası sürdürülemez) |
| `serve`, `publish` | evaluate + C9, C11 | her "bilinmiyor" → ret |
| `export` | C12, C1 | C1 → ret |

**Mimarisi bilinmeyen ağırlık modele yüklenmez.** Çıplak 4 başlı `state_dict`, 8 başlı hedefte
C4'ü geçer (şekiller aynıdır) ama C5 "bilinmiyor"dur ve karar "uyumsuz: architecture unknown"
olur. Mimarisi beyan edilmiş 4 başlı bir checkpoint 8 başlı hedefte C5'ten
`num_heads: checkpoint=4, target=8` gerekçesiyle reddedilir. Eski checkpoint'lerde dosya ve
ağırlık özeti bulunmadığı için bu eksiklikler çalışma amaçlarında yalnız uyarıdır; aksi hâlde
bugünkü bütün checkpoint'ler geçersiz olurdu.

**Yükleme yolu:** `cevahir_ai.model.checkpoint.open_checkpoint(path, model=…, tokenizer=…,
purpose=…)` dosyayı açmadan koklar, `weights_only=True` ile okur, kapıyı çalıştırır, ret hâlinde
hedef modele **dokunmadan** `IncompatibleCheckpoint` fırlatır, kabul hâlinde katı yükler ve
modeldeki ağırlıkları yeniden özetleyip kapıdan geçenle karşılaştırır.

## 8. Komut satırı

```
python -m cevahir_ai.keys inspect <dosya> [--level file|weights|compat] [--purpose <amaç>] [--json]
                                  [--target tiny|train-default|cev11-primary|<config.json>]
                                  [--vocab vocab.json --merges merges.txt | --byte-bpe <dosya>]
                                  [--expect-file <anahtar>] [--expect-weights <anahtar>]
python -m cevahir_ai.keys verify <dosya> <anahtar>
python -m cevahir_ai.keys explain <anahtar | dosya>
python -m cevahir_ai.keys diff <a> <b>
python -m cevahir_ai.keys schemes
```

Çıkış kodları: 0 başarılı, 1 hata, 2 kullanım hatası, 5 doğrulama başarısız. `--level file`
torch gerektirmez. Varsayılan `--level compat --purpose evaluate` şu dört soruyu yanıtlar:
sağlam mı, hangi tokenizer, hangi mimari, bugünkü kodla uyumlu mu (değilse neden). Gerçek bir
CEV-11 taban checkpoint'i için çıktı (kısaltılmış):

```
intact      yes / evet
  C3  weights.digest     recomputed weight digest equals the declared CEV-11 model_sha256
envelope    cev11-v1
tokenizer   cev:tok:bytebpe-v1:4db9febe26c3…
arch        cev:arch:cfg-v1:088c6e7a1801…
  fields    vocab=4096 d=192 layers=6 heads=4 kv_heads=4 ffn=512 pe=rope tied=True moe=False
weights     cev:weights:sha256-v1:2e5a4aa6dbdd…
  layout    cev:wschema:layout-v1:361855d0bb73…  (preset cev11-primary)
verdict     compatible-with-warnings  (purpose: evaluate, level: compat)
  *C1  file.integrity     ??    no .sha256 sidecar and no expected file key …
  *C4  wschema.match      ok    names, shapes and ties match the declared architecture
  *C5  arch.declared      ok    declared architecture is accepted by the current code …
```

## 9. Zarf biçimleri

`cevahir_ai.adapters.envelopes.read_envelope` bugün var olan bütün biçimleri tanır: `tm2`
(üretim eğitimi, `training_management.v2` CheckpointManager), `mm-save` (`ModelManager.save`),
`saver-4.1` (`ModelSaver.save_checkpoint`, yan dosyalı), `bare` (yalnız ağırlık), `tm3`,
`cev11-v1`, `cev11-ledger-v1`. `module` (pickle'lanmış `nn.Module`) okunmadan reddedilir. Eski
`unpack_checkpoint` `tm3` ve CEV-11 zarflarını tanımaz; orada kimlikleri kaybolur.

## 10. CEV-11 güvencesi

- `tests/evolution/test_cevahir_ai_frozen_surfaces.py`: içe aktarma yolları, `tiny_model_config()`
  çıktısı (47 anahtar, değerleriyle), `ModelManager(...).effective_config`, 25 tensörlük tiny
  `state_dict` (ad, şekil, dtype, bağ), CEV-11'in okuduğu öznitelik adları ve "aynı tohum → aynı
  ağırlıklar ve logit'ler" (çatı kullanılmadan önce ve sonra).
- **Kapanış sabiti** `governance/freeze/cev11-closure.json`: çalışmanın `.py` dosyalarından
  herhangi bir içe aktarma kenarıyla erişilen 33 ürün modülü ve `utf8-lf` özetleri.
  `python -m cevahir_ai.evidence.closure check` değişen dosyayı CEV-11 adıyla bildirir (çıkış 5);
  gözden geçirilmiş bir değişiklikten sonra `record --reason "…"` yeniden sabitler ve gerekçeyi
  geçmişe yazar. Test paketinde bayt kayması uyarıdır, `CEVAHIR_CLOSURE_STRICT=1` ile hataya
  döner; davranış testleri ise her zaman katıdır.
- Yerelde bulunan 10 gerçek CEV-11 taban checkpoint'i (`results/bases/*/model.pt`, git'te değil)
  testte denetlenir: tamamı sağlam, mimarisi ve tokenizer'ı biliniyor, bugünkü kodla uyumlu.

## 11. Aynı işle gelen düzeltmeler

- `training_system/health_check.py` çalıştıramadığı bir dosya için "ÜRETİM HAZIR" diyordu
  (yanlış içe aktarma yolu → taslak model → hazır örnek yanıtlar → eşikler geçiyor). Artık
  gerçek modelin ürettiği yanıtlar yoksa `production_ready` her zaman `False`, çıkış kodu 1.
- `scripts/check_publications.py` yalnız `assert` kullanıyordu; `python -O` altında bütün
  denetimler siliniyor ve betik yine "passed" yazıyordu. Her denetim artık `_require` ile
  `SystemExit` fırlatır.

## 12. Sınırlar (açıkça)

- **Eski yükleyiciler yönetilmiyor.** `ModelManager.load`, `ModelLoader`, eğitim sürdürme yolu ve
  `train.py` değişmedi; kapıdan yalnız `open_checkpoint` ve `inspect` geçer. Çıplak
  `state_dict` hâlâ `ModelManager.load` ile yanlış başlı modele yüklenebilir (testte gösterilir).
- **Kayıt, denetim kaydı, soy ağacı, politika, onay, konsol yok.** `CEVAHIR_HOME` çözücüsü var
  ama oraya hiçbir şey yazılmıyor. Anahtarlar hesaplanır, hiçbir yere kaydedilmez; eğitim
  koşuları henüz `run`/`runspec` anahtarı üretmez, `prepare_cache` `cache:recipe-v1` yazmaz.
- **C2, C6, C8, C13, C14, C15 uygulanmadı** (imza, canlı mimari okuması, tokenizer davranış
  parmak izi, köken, değerlendirme eşiği, korunan yollar). `arch:live-v1` ve `tok:probe-v1`
  şemaları yok.
- Eski checkpoint'ler dosya ve ağırlık özeti taşımadığı için `serve`/`publish` amacı, beklenen
  anahtarlar elle verilmedikçe her zaman reddeder.
- `training_management/v3` **bütün olarak silinmedi**: canlı giriş noktası
  `training_system/train.py:91` onu (yalnız bir log bayrağı için) içe aktarıyor ve kitaba bağlı
  `docs/modules/training_management/README.md` onun bir dosyasına bağlantı veriyor; "erişilemez"
  kanıtı tam değil. Yalnız `training_management/v3/monitoring/` (5 dosya, 2.117 satır) silindi:
  hiçbir modül (v3'ün kendisi dahil), test, betik ya da kitap kanıtı onu içe aktarmıyordu. Kimin
  neyi içe aktardığı testle sabitlendi. Kalan v3 `TrainingManager` belge dizgileri hâlâ bu alt
  paketin sınıf adlarını anar (ölü kod; dokunulmadı).
- `health_check.py` artık yanlış "hazır" demez ama üretim (`tm2`) checkpoint'lerini de
  değerlendiremez: `model_state` anahtarını bekler ve modeli hatalı argümanla kurar.
- Bağlı ağırlıkları ayrı depolarda saklayan bir dosyanın `weights` anahtarı, aynı değerleri
  paylaşımlı saklayan dosyanınkinden farklıdır (takma ad grubu depo paylaşımından okunur).
- README'ler kitaba bağlı olduğu için bu sayfaya oradan bağlantı verilmedi.
