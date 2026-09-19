# Cevahir: güncel mimari ve çalışma sözleşmesi

Güncelleme: **20 Eylül 2026**. Bu belge aktif kaynak kodundaki bileşenleri, aralarındaki veri akışını ve destek sınırlarını tanımlar. [Ana README](../../README-TR.md) projenin genel anlatımıdır; [geliştirme planı](NEXT_DEVELOPMENT_ROADMAP.md) açık borçların önceliğini ve tamamlanma ölçütlerini içerir. Önceki incelemeler [ilk mimari denetiminde](CURRENT_ARCHITECTURE_AUDIT.md) ve [geliştirme günlüğünde](../development/EVOLUTION_LOG.md) tarihsel kayıt olarak korunur.

## Sistemin amacı ve kapsamı

Cevahir, kendi tokenizer'ından başlayarak dil modeli eğitimi, checkpoint yönetimi, üretim, bilişsel iş akışları ve konuşma uygulamasına uzanan bir motor geliştirir. Transformer decoder tek başına projenin tamamı değildir: token kimliği, veri hazırlama, eğitim hedefi, model kurulumu, dikkat önbelleği ve konuşma belleği birlikte tutarlı çalışmalıdır.

Bu parçaların geliştiriciye birlikte sunduğu çalışma alanı [sistem bütünlüğü rehberinde](SYSTEM_OVERVIEW.md) açıklanır. Tokenizer altyapısı Türkçeye özel geliştirilmiştir; bu odak her dil için ayrı tokenizer zorunluluğu oluşturmaz. Aynı tokenizer'ın farklı dillerde kullanımı metni temsil etme kapsamı ve token verimliliğiyle, modelin dil yeteneği ise eğitim verisi ve öğrenilen ağırlıklarla değerlendirilir.

Geliştiricinin önceki gerçek model eğitimleri ve üretim kontrolleri [README'deki eğitim çıktılarında](../../README-TR.md#gerçek-eğitim-çıktıları) yer alır. Son mühendislik çalışmalarındaki küçük CPU doğrulamaları, mevcut altyapı değişikliklerinin davranışını inceler. Bunlar geçmiş eğitimlerin yerine geçmez; farklı bir modelin dil kalitesi veya GPU performansı olarak da sunulmaz.

## Sürüm haritası

Kaynak dosyalarındaki V4–V8 notları aynı çekirdeğin geliştirme geçmişidir. Her alt sistemin sürüm numarası kendi kapsamındadır; tüm repoya tek bir V8 etiketi vermek mevcut düzeni açıklamaz.

| Bileşen / etiket | Güncel anlamı | Kaynak |
|---|---|---|
| Sinir ağı V4 | RMSNorm, SwiGLU, KV cache, gelişmiş checkpointing, quantization ve MoE bağlantıları | [CevahirNeuralNetwork](../../src/neural_network.py) |
| Sinir ağı V5 | GQA/MQA düzeni, sliding window, YaRN RoPE | [Model kurucusu](../../src/neural_network.py) |
| Sinir ağı V6 | SDPA, QK-Norm, parallel residual, attention/output soft-cap, residual başlangıç ölçeklemesi | [Model ve katman kurulumu](../../src/neural_network.py) |
| V7 notları | Stochastic depth, merged SwiGLU gate/up, sink token'lı KV yönetimi | [FFN](../../src/neural_network_module/ortak_katman_module/feed_forward_network.py), [Transformer katmanı](../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py) |
| V8 alt katman notları | KV sınırları ve MoE yardımcı kayıp hesabı gibi düzeltmeler | [KV cache](../../src/neural_network_module/ortak_katman_module/kv_cache.py), [katman](../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py) |
| TrainingServiceV3 | Hazırlanmış veriyi tüketen, doğrulayan, bölen ve batch'leyen eğitim servisi | [V3 servis](../../training_system/v3/core/training_service_v3.py) |
| TrainingManager V2 | Aktif eğitim döngüsü, optimizer, scheduler, doğrulama ve resume | [V2 yönetici](../../training_management/v2/core/training_manager.py) |
| TrainingManager V3 | Ayrı geliştirme hattı; aktif backend olarak desteklenmiyor | [Backend sözleşmesi](../../training_management/contracts.py) |
| Bilişsel V2 | Politika, deliberation, araçlar, üretim, critic, bellek ve middleware | [Orchestrator](../../cognitive_management/v2/core/orchestrator.py) |
| Config sürümü | `config_version=1`, `architecture_version="cevahir-capabilities-1"`: yapılandırma uyumluluğu | [Ortak şema](../../model_management/config_schema.py) |

Bazı sınıf açıklamaları ve log mesajları hâlâ eski V2/V4/V6 adlarını taşır. Etkin özellikleri belirleyenler normalize edilmiş ayarlar ve gerçekten kurulan katmanlardır. Sürüm metinlerinin tek başına değiştirilmesi teknik borcu kapatmaz.

## Aktif akışlar

### Tokenizer → veri → eğitim

`TokenizerCore` → `training_system/prepare_cache.py` → hazırlanmış input/target kayıtları → `training_system/train.py` → `TrainingServiceV3` → **V2 TrainingManager** → model ve eğitim checkpoint'i.

Başlatıcı V3 servis kullanılabildiğinde onu seçer; V2 servis alternatifi korunur. Her iki aktif servis de `training_backend="v2"` sözleşmesine bağlıdır. TrainingService verinin hazırlanması/tüketilmesi, ayrılması ve batching işlerini; TrainingManager ileri/geri geçiş, optimizasyon ve eğitim durumunu yönetir. Aynı sürüm numarasına sahip olmaları gerekmez.

### Model → metin üretimi

`Cevahir` → `ModelManager` → `ModelInitializer` → `CevahirNeuralNetwork` → embedding → Transformer katmanları → çıkış normalizasyonu → sözlük logits'i.

Çekirdek normal çağrıda `(logits, attention_or_none)`, cache etkin çağrıda ayrıca üçüncü cache alanını döndürür. İkinci alan eğitim kaybı değildir. `CevahirModelAPI`, tokenizer ID'lerini model girişine dönüştürür; örnekleme, tekrar kontrolü, EOS ve üretim sınırları üzerinden metin üretir.

### İstek → bilişsel işlem → konuşma

`Cevahir.process` → `CognitiveManager.handle` → V2 orchestrator → özellik çıkarımı → politika seçimi → isteğe bağlı deliberation → bağlam/araç → model üretimi → self-consistency → critic → bellek güncellemesi.

Deliberation gerçek model `generate`/`score` çağrıları yapar; politika ve critic değerlendirmesinin önemli bölümü sezgisel kurallara dayanır. `ChattingManager`, bu akışı kullanıcı/oturum geçmişiyle bağlar. HTTP uygulaması [create_app](../../api/app_factory.py) ile kurulur. Eski [api/app.py](../../api/app.py) kaldırılmış bir yapılandırma import'u içerdiğinden güncel giriş değildir.

## Çekirdeğin desteklediği davranış

| Alan | Uygulama ve sınır |
|---|---|
| Attention | MHA/MQA/GQA, nedensel maskeler ve QK-Norm. Küçük CPU örneklerinde manuel/SDPA ve tam dizi/cache sonuçları karşılaştırıldı. |
| SDPA / inceleme | `return_attention_weights=False` normal yoldur. Ağırlıklar açıkça istendiğinde veya attention soft-cap gerektiğinde manuel hesap kullanılır; bayrak kalıcı değiştirilmez. |
| RoPE / uzun bağlam | RoPE, linear/YaRN ölçeklemesi; cache ilerlerken mutlak konumlar aktarılır. Uzun bağlam dil kalitesi ayrıca değerlendirilmeli. |
| KV / window / sinks | Artımlı ekleme, kapasite sınırı, eviction ve sink token desteği vardır. Window yoğun maske kullanır; özel seyrek kernel değildir. |
| FFN / MoE | SwiGLU/GELU, birleşik gate/up projeksiyonu, top-k routing, jitter ve eğitim hedefinde MoE yardımcı kaybı. Ortak FFN genişliği çözümü dense/MoE geometrisini korur. |
| Residual / norm | RMSNorm/LayerNorm, pre/post norm ve parallel residual seçenekleri. Stochastic depth artık dalını düşürür; hesaplamayı atlayan LayerDrop değildir. |
| Gradient checkpointing | Model ve bağımsız katman aynı seçim kuralını kullanır. `adaptive` mevcut sabit katman seçim sezgiselidir; ölçülen belleğe göre uyarlama değildir. |
| Harici Flash Attention | İsteğe bağlı ve varsayılan kapalı. Gerçek CUDA kernel'i bu makinede doğrulanmadı; SDPA çalışması Flash kernel seçildiği anlamına gelmez. |
| Quantization | Açık `model.eval(); model.apply_quantization()` adımı. CPU dynamic INT8 kayıt/yükleme doğrulaması var. `int8` statik isteği dynamic INT8'e uyarıyla döner. int4 ve bitsandbytes yükleme bayrakları entegre değil. |
| Compile | Varsayılan kapalı; model/state_dict kimliği korunur. Tanınan backend hatasında eager dönüşü var. Gerçek hız, graph-break ve yeniden derleme maliyeti ölçülmedi. |
| Tanılama | `collect_diagnostics=True` veya TensorBoard örnekleme aralığı ayrıntılı istatistik üretir. Normal çağrı eski istatistikleri güncel gibi göstermez. |

Ayrıntılar: [çekirdek rehberi](../modules/neural_network/README.md), [alt çekirdek sözleşmeleri](LOWER_CORE_CONTRACTS.md).

## Yapılandırma ve eski ağırlıklarla uyumluluk

Kanonik normalizasyon `model_management.config_schema.normalize_model_config` içindedir. Düz/iç içe yapılandırma ile `d_model`, `n_heads`, `n_layers`, `ff_dim`, `drop_rate`, `pe_max_len` eski adları desteklenir; çelişen değerler ve ileri sürüm reddedilir. Bilinmeyen alanlar normal modda korunur, `strict=True` ile reddedilir. Bir alanın korunması, ilgili özelliğin aktif olduğu anlamına gelmez.

Yeni şema/facade varsayılanı 512 embedding, 8 başlık, 8 katmandır. Sürümsüz doğrudan ModelManager/Initializer kullanımında eksik katman sayısı için 12 katmanlık eski profil korunur. Açık kullanıcı boyutları değiştirilmez. API'nin eski 1024 embedding/24 katman profili `CEVAHIR_MODEL_PROFILE=legacy_api` ile seçilir. Eğitim presetleri düşük donanıma otomatik uygun değildir.

`seq_proj_dim`, eski kayıt uyumluluğu alanıdır. Gerçek çıktı genişliği `embed_dim` üzerinden kurulur; farklı eski boyut weight tying'i kapatabilir. Eski ağırlıkları değerlendirmeden otomatik eşitleme veya token-ID göçü yapılmamalıdır.

## Checkpoint ve eğitim yaşam döngüsü

Ortak [checkpoint sözleşmesi](../../model_management/checkpoint_contract.py), yönetici kayıtları, `model_state_dict` zarfları ve ham tensor state_dict biçimlerini ayırır. Yeni kayıtlar gerçek model kurulum ayarlarını taşır; eğitim ayarlarıyla karıştırılmaz. Yükleme, model hesap ayarları ile state anahtar/şekillerini değişiklikten önce denetler; başarılı yükleme KV durumunu temizler.

Kimlikli checkpoint, tokenizer sözlük ID'leri, merges ve etkin BPE ayarlarının parmak izini taşır. Yükleyici, ModelManager ve aktif training resume bu kimliği karşılaştırır; eşit sözlük büyüklüğü yeterli değildir. Kimlikli kayıtta tokenizer eksikliği reddedilir. Kimliksiz eski kayıtlar tokenizer verildiğinde uyarı üretir; token anlamları otomatik doğrulanamaz. Salt ağırlık dışa aktarma metadata taşımadığından aynı eski kayıt sınırına tabidir. Kimlik kayıt/yükleme/eğitim başlangıcında denetlenir; her decode adımında hash alınmaz.

Kurulum bilgisi olan checkpoint, henüz kurulmamış ModelManager tarafından kayıtlı ayarlardan açılabilir. `Cevahir` facade'ı modeli önce kurduğu için verilen mimari ayarları checkpoint ile eşleşmelidir. `load_model_path=None` varsayılan konumlarda arama yapar; `""` açık yeni model isteğidir. Seçilmiş/bulunmuş checkpoint yüklenemediğinde rastgele modele sessiz geçiş yapılmaz. Tam Python model nesnesi içeren eski güvenilen kayıtlar için `weights_only=False` açıkça gerekir.

Aktif eğitim; token sayısıyla ağırlıklandırılan accumulation, son eksik batch grubu, optimizer adımına bağlı scheduler ve MoE yardımcı kaybını uygular. Epoch sınırından devam için optimizer/scheduler/scaler/RNG/döngü durumu taşınır; küçük dropout'lu örnekte devam eden koşu kesintisiz koşuyla karşılaştırılmıştır. Epoch ortası devam aynı düzeyde doğrulanmış değildir.

**Açık kayıt borcu:** ModelSaver benzersiz geçici dosyaya doğrudan yazıp atomik yayımlar. V2 eğitim [CheckpointManager](../../training_management/v2/utils/checkpoint_manager.py) ise hâlâ `BytesIO`/`getvalue` ve sabit `.tmp` yolu kullanır. Aktif TrainingManager restore'u modelden sonra optimizer/scheduler/RNG yükler; sonradan hata oluşursa tam geri alma yoktur. Bağımsız CheckpointManager.load ayrıca optimizer hatasını uyarıyla geçebilir. Bu yolların davranışı henüz tek sözleşme değildir.

Aktif V2 backend, EMA/SWA/SAM/Lookahead/LLRD/curriculum/scheduled sampling ve dağıtık eğitim gibi entegre edilmemiş seçenekleri [açıkça reddeder](../../training_management/contracts.py). Yardımcı modüllerin varlığı tamamlanmış entegrasyon sayılmaz.

## Tokenizer, hazırlanmış veri ve ayrım

Veri kimliği; kaynak bağıl yolları ve SHA-256 içerikleri, sözlük, merges ve BPE ayarlarını kapsar. V3 strict doğrulaması checksum ister; uyuşmayan cache'i seçen bypass yolları reddedilir. Önceki ad/boyut temelli hazırlanmış cache'ler yeniden hazırlanmalıdır. Bu durum tokenizer veya modelin yeniden eğitilmesini gerektirmez. Büyük veri kümelerinde hash okuma maliyeti ayrıca ölçülmelidir.

V2/V3 [ortak split](../../training_system/data_split.py) kullanır. Kaynak ve birebir input/target tekrarlarıyla bağlı gruplar tek bölmede kalır; ortak son PAD alanı karşılaştırmadan çıkarılır. İki bağımsız grup yoksa eğitim başlamaz. Kaynak kimliği olmayan kayıtların belge bütünlüğü ve yakın tekrarların ayrımı çözülmüş değildir.

TokenizerCore BPE ayarlarını encoder'a taşır; singleton paylaşımı etkin ayarlara bağlıdır. Salt okunur eksik dosya açılışı yazmadan hata verir. `text_loss_policy` (`warn`, `error`, açık `ignore`) pretokenizer'ın düşürdüğü boşluk dışı karakterleri görünür yapar. Bu denetim byte fallback değildir; normalizasyon/decoder dönüşümlerindeki tüm kayıpları kapsamaz. Mevcut sözlük/merges ve token ID'leri korunmuştur.

**Açık veri borcu:** V3 `_validate_alignment`, yalnız ilk kayıt üzerinde input ve target eşitliğini kontrol eder. Tüm kayıtların biçimi, ID sınırları ve veri türüne uygun next-token/mask sözleşmesi doğrulanmış değildir. Hazırlama ile eğitim ayarları da hâlâ ayrı girişlerden gelir.

## Bilişsel sistemin mevcut gücü ve açık sınırları

Direct, think, debate ve Tree of Thoughts yolları; gerçek üretim/puanlama, self-consistency, araç çalıştırma, critic revizyonu, bellek güncelleme ve tracing bileşenlerine sahiptir. Calculator sınırlı AST aritmetiği kullanır; dış araçlar gerçek implementasyonlarıyla kaydedilmelidir. Yerleşik search/file servisleri varmış gibi gösterilmez.

Vektör bellek embedding adaptörüyle memory veya Chroma deposuna bağlanabilir; kurulum başarısızsa keyword retrieval'a düşer. Diğer ilan edilen bazı store seçenekleri henüz uygulanmamıştır. Orchestrator yolunda bellek kullanıcı+oturum kapsamını izler; episodik kayıtlar ve vektör özetleri benzersiz kimlik taşır, not/bellek değişiklikleri revizyonu ilerletir. Arama kapsam filtresi top-k seçiminden önce uygulanır. Yanıt önbelleği tam geçmiş/kimlik/ayarlar/bellek revizyonunu kapsar; kopya sonuç döner ve cache dönüşünde de konuşma güncellenir. Genel semantic cache, kapsamlı normal orchestrator isteklerinde etkin değildir; request batching de bağlı değildir.

20 Eylül incelemesinde küçük kontrollerle yeniden üretilen kopukluklar:

- Entropi adaptörü `encode` dönüşündeki metin tokenlarını ID sanıyor; model forward'ı çalışmadan kelime çeşitliliği fallback'ine düşüyor. Politika yanlış sinyali model entropisi olarak tüketiyor.
- İstek sistem talimatı bağlam handler'ında aktarılmıyor; pruning varsayılan sistem talimatını da metinden çıkarıyor. Son üretim talimatsız kalabiliyor.
- ToolPolicy, `2+3*4`, negatif ve ondalıklı ifadeleri eksik çıkarıyor; doğru calculator'a yanlış işlem gidiyor.
- Async pipeline ToT nesnesini/config'i kurmuyor; `tot` etiketiyle think fallback'i çalışabiliyor.
- Critic'in `_last_feedback`/`_last_passes` instance alanları eşzamanlı istekler arasında karışabiliyor.

Critic'in factuality skoru bağımsız gerçeklik doğrulaması değildir; mevcut hesap bazı iddia/belirsizlik kelimelerine dayanır. Bilişsel geliştirme önce bu yürütme ve durum sözleşmelerini düzeltmeli, ardından kaliteyi görevler üzerinde değerlendirmelidir. Dosya noktaları ve tamamlanma ölçütleri [geliştirme planındadır](NEXT_DEVELOPMENT_ROADMAP.md).

## Uygulama ve doğrulama kapsamı

JWT/HTTP, kullanıcı-oturum sahipliği ve gerçek ORM/repository/UnitOfWork/ChattingManager yolları geçici SQLite üzerinde birlikte sınandı. Sahip geçmişe erişebilir; diğer kullanıcı erişemez/yazamaz ve reddedilen istek model çağrısı üretmez. JSON metadata değişiklikleri kalıcıdır; bağlam bütçesi en yeni mesajlardan kronolojik olarak seçilir. Bunlar PostgreSQL eşzamanlı işlemlerinin veya tüm dağıtım yollarının doğrulanması değildir.

[Benchmark rehberi](../../benchmarks/README.md), küçük CPU ölçümlerinin komutlarını ve kanıt dosyalarını açıklar. Tarihsel test paketinin tamamının temiz olduğu iddia edilmez. Gerçek eski model dosyası isteyen kontroller ve eski alanları kullanan testler vardır. Güncel README'nin küçük ModelManager örneği CPU'da `[1, 4, 128]` boyutlu sonlu logits üretir.

Bundan sonraki çalışma sırası alt sistemlere odaklanır: bilişsel yürütme tutarlılığı → model/cache sahipliği → eğitim kayıt/veri sözleşmesi → ölçülebilir bilişsel kalite. Büyük eğitim, tokenizer göçü ve yeni araştırma mimarileri bu yerel hazırlık turunun çıktısı değildir.
