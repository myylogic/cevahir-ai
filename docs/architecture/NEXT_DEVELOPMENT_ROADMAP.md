# Sonraki büyük geliştirme turu

**Durum: 20 Eylül 2026 — kaynak incelemesi ve çalışma planı.** Bu belge tamamlanmış düzeltme listesi değildir. Aşağıdaki açık maddeler henüz üretim kodunda kapatılmadı. Güncel sistemin ne yaptığı [mimari sözleşmede](CEVAHIR_ARCHITECTURE_SPEC.md), genel proje anlatımı [README'de](../../README-TR.md) bulunur.

Amaç, Cevahir'in mevcut dil modeli motorunu ve bilişsel sistemini ortak davranış sözleşmeleri üzerinden ilerletmektir. İlk odak alt birimler ve birbirlerine aktardıkları durumdur. Çalışma yerelde ilerler; mevcut eğitilmiş ağırlıklar, tokenizer dosyaları ve gerçek eğitim çıktıları korunur. Bu plan yeni ağır eğitim gerektirmez.

## Nereden devam ediyoruz?

Mevcut geliştirmeler; V5/V6/V7 model özelliklerini, V8 alt katman düzeltmelerini, V3 eğitim servisini, etkin V2 eğitim yöneticisini ve V2 bilişsel akışı içerir. Yakın çalışmalar ortak FFN boyutunu, attention tanılama yönlendirmesini, MoE yardımcı kaybını, model/checkpoint/tokenizer kimliğini ve kullanıcı/oturum kapsamlı belleği güçlendirdi. Bu kazanımlar başlangıç noktasıdır.

Bu tarama özellikle `model/cevahir.py`, çekirdeğin katmanları, model yönetimi, V2/V3 eğitim sınırı ve bilişsel orchestrator/pipeline bağlantılarını izledi. API/veritabanı tarafının önceki düzeltmeleri mimari belgede özetlenir; bu turun ilk geliştirme paketi o üst katmanlardan başlamaz. Kalan tüm dosyaların eksiksiz ve hatasız olduğu sonucuna varılmaz.

## İlk paket: bilişsel yürütme sözleşmesini birleştirme

Beş bulgu, ağırlık yüklemeyen küçük kontrollerle yeniden üretildi. Bu kontroller belirli kod yollarını kanıtlar; mevcut eğitilmiş modelin yanıt kalitesi üzerine ölçüm değildir.

### COG-01 — Sistem talimatı bağlamda kayboluyor

**Öncelik: yüksek.** [ContextBuildingHandler](../../cognitive_management/v2/processing/handlers.py#L450) `system_prompt=None` geçiriyor. [Context pruner](../../cognitive_management/v2/utils/context_pruning.py#L181) varsayılanı seçse de [bağlamı oluştururken](../../cognitive_management/v2/utils/context_pruning.py#L209) talimatı dışarıda bırakıyor. GenerationHandler yalnız oluşan metni tüketiyor.

Kontrolde istek ve varsayılan talimata ayrı işaretler verildi; sonuç yalnız `[USER]\nMerhaba` oldu. Uygulamanın dil/rol/format talimatları final üretime ulaşmıyor.

**Geliştirme:** İstek talimatı → varsayılan talimat önceliğini tek yerde çözmek; talimat, geçmiş, retrieval, araç sonucu ve kullanıcı mesajı için ortak bağlam oluşturmak. Direct, think, debate ve ToT aynı etkin talimatı kullanmalı. Token bütçesi öncelikleri açık olmalı; talimat sessizce düşmemeli.

**Tamamlanma ölçütü:** Senkron/asenkron her stratejide etkin talimat final modele tam bir kez ulaşır; kısa bağlam bütçesinde korunma veya açık hata davranışı belirlenir. Yanıt önbelleğinin anahtarı gerçekten kullanılan talimatı temsil eder.

### COG-02 — Entropi için yanlış tokenizer çıktısı kullanılıyor

**Öncelik: yüksek.** [CevahirModelAPI.entropy_estimate](../../model/cevahir.py#L875) `tokens, _ = encode(...)` alıyor. [TokenizerCore sözleşmesi](../../tokenizer_management/core/tokenizer_core.py#L488) `(metin_tokenları, token_ids)` döndürüyor. Metin tokenlarını long tensöre çevirme hatası yutulup çeşitlilik sezgiseline düşüyor; politika bu değeri model entropisi sanıyor.

Kontrol: `(['aa', 'bb', 'aa'], [1, 2, 1])` dönüşünde `entropy=0.6667`, model forward çağrısı **0**. Entropi hesabının gerçek logits yoluna girmediği görüldü.

**Geliştirme:** ID çıktısını kullanmak; gerçek entropi ile fallback değerini kaynak/ölçek bilgisiyle ayırmak. Politika özelliklerinin birimleri ve eşikleri tutarlı olmalı. Teşhis amaçlı forward, üretimin KV durumunu değiştirmemeli.

**Tamamlanma ölçütü:** Bilinen logits üzerinden beklenen entropi hesaplanır; gerçek tokenizer dönüşüyle model çağrılır; hata halinde fallback kaynağı gözlemlenir ve normal entropi gibi etiketlenmez.

### COG-03 — Araç parametresi yanlış çıkarılıyor

**Öncelik: yüksek.** [ToolPolicyV2](../../cognitive_management/v2/components/tool_policy_v2.py#L208), ilk iki tam sayılık parçayı alıyor; ifade bulunamazsa ilk iki sayıyı topluyor. Mevcut güvenli AST calculator kendisine verilen yanlış işlemi doğru hesaplıyor.

| İstek | Araca gönderilen ifade | Mevcut sonuç |
|---|---|---:|
| `2+3*4` | `2+3` | 5 |
| `-5+2` | `5+2` | 7 |
| `1.5+2.5` | `5+2` | 7 |
| `10 bolu 2` | `10+2` | 12 |

**Geliştirme:** Tam ifadeyi koruyan sınırlı parser ve açık doğal dil operatör eşlemesi; anlaşılamayan girdide uydurma toplama yapmamak. Araç seçimi, parametre doğrulaması, yürütme ve başarı/başarısızlık durumu tek sonuç sözleşmesinden geçmeli.

**Tamamlanma ölçütü:** Öncelik, parantez, negatif/ondalık sayılar korunur; desteklenen Türkçe ifadeler doğru çevrilir. Belirsiz ifade başarılı sonuç olarak raporlanmaz. Mevcut AST güvenlik sınırı korunur.

### COG-04 — Senkron ve asenkron strateji kurulumu farklı

**Öncelik: orta.** [Senkron orchestrator](../../cognitive_management/v2/core/orchestrator.py#L187), ToT nesnesi ve config'i handler'a bağlar. [Asenkron kurulum](../../cognitive_management/v2/core/orchestrator.py#L555) bunları geçirmez. Aynı config ile `sync_tot=True`, `async_tot=False` üretildi. Seçili mod `tot` kalsa da think fallback'i çalışabilir.

**Geliştirme:** Pipeline bağımlılıkları için ortak kurulum; sync/async yalnız yürütme biçiminde ayrılmalı. Seçilen strateji, gerçekten yürütülen strateji ve fallback nedeni ayrı taşınmalı.

**Tamamlanma ölçütü:** Aynı istek/config için strateji bileşenleri ve etkin talimat eşleşir. ToT aktifken aynı çözümleme yolu çağrılır; kapalı/başarısız olduğunda fallback açıkça raporlanır.

### COG-05 — Critic sonucu istekler arasında karışıyor

**Öncelik: orta.** [CriticV2](../../cognitive_management/v2/components/critic_v2.py#L280), `_last_feedback`/`_last_passes` alanlarını paylaşılan nesnede tutuyor. [Handler](../../cognitive_management/v2/processing/handlers.py#L700) bunları review sonrasında okuyor. [Async handler](../../cognitive_management/v2/processing/async_handlers.py#L233) aynı critic'i iş parçacıklarında çalıştırıyor.

Kontrollü iki istekte `A_feedback=['REQUEST_B']`, `B_feedback=['REQUEST_B']` elde edildi. İstek bağlamının ayrılması bu instance alanlarını izole etmiyor.

**Geliştirme:** Metin, revised, feedback, passes ve skorları review dönüşünde taşıyan istek kapsamlı sonuç nesnesi. Pipeline, critic'in son ortak durumunu okumamalı. Eski dış tüketiciler varsa geçiş adaptörü açıkça tanımlanmalı.

**Tamamlanma ölçütü:** İç içe senkron/asenkron istekler kendi feedback/geçiş sayısını alır. Hata veya revizyonsuz dönüşte önceki isteğin verisi görünmez.

Bu beş madde birlikte ele alınmalıdır: doğru talimat ve özellikler → doğru strateji/araç → isteğe ait değerlendirme. Yalnız görünür `tot` etiketini veya hata mesajını değiştirmek paketi tamamlamaz.

## İkinci paket: ortak modelin ve KV durumunun sahipliği

### CORE-01 — Üretim kilidi model sahibinde değil

[CevahirModelAPI](../../model/cevahir.py#L458) kendi üretim kilidini tutuyor; aynı adaptörde üretimler sıralanıyor. Ancak [Cevahir.load_model](../../model/cevahir.py#L1447) bu kilidi almıyor. Aynı ModelManager'ı saran iki adaptör de ayrı kilitler taşır. Kod incelemesi ortak model/cache üzerinde eşzamanlı kullanım sınırının eksik olduğunu gösteriyor; bu incelemede büyük modelle yarış testi yapılmadı.

**Geliştirme:** ModelManager düzeyinde üretim/yükleme/cihaz-dtype değişikliği sahipliği; model revizyonu ve cache ömrünü birlikte yönetmek. Üretim sürerken model ağırlıkları değişmemeli. Compile bağları, scoring/entropy ve tokenizer kimliğinin değişmesi de aynı yaşam döngüsüne dahil edilmeli.

**Tamamlanma ölçütü:** İki adaptör ve eşzamanlı load senaryosunda tek üretim tek model revizyonuyla tamamlanır; eski KV başka isteğe/model revizyonuna taşınmaz. Düşük RAM için ikinci büyük model kopyasını zorunlu kılan tasarımdan kaçınılır.

### CORE-02 — Etkin özellikler ve eski etiketler

[Model](../../src/neural_network.py) ve [ModelManager](../../model_management/model_manager.py) log/açıklamalarında farklı nesil etiketleri var. Bağımsız attention yardımcılarının aktif yolla ilişkisi, `seq_proj_dim` uyumluluğu ve compile/quantization kombinasyonları ayrıca gözden geçirilmeli.

**Geliştirme:** Aktif modeli kurulan modüller ve normalize edilmiş ayarlardan tanımlayan ortak yetenek özeti. Bilinmeyen/etkisiz seçenekler sessizce destekleniyormuş gibi görünmemeli. Eski checkpoint geometrisi korunmalı.

**Tamamlanma ölçütü:** Aynı model bütün girişlerden aynı yetenek özetini üretir; sadece isim değiştirmeden aktif/deneysel/uyumluluk yolları belirlenir.

## Üçüncü paket: eğitim verisi ve kayıt bütünlüğü

### TRAIN-01 — Eğitim checkpoint'i RAM'de ikinci kopya oluşturuyor

[V2 CheckpointManager._atomic_save](../../training_management/v2/utils/checkpoint_manager.py#L343), `BytesIO`, `getvalue` ve sabit `.tmp` dosyası kullanıyor. ModelSaver'daki doğrudan yazma iyileştirmesi bu aktif eğitim kayıt yoluna aktarılmamış.

**Geliştirme:** Ortak dosyaya yazma/yayımlama yordamı; benzersiz geçici dosya, stream, flush/fsync, atomik değişim ve hatada temizlik. Index/last/best güncellemeleri yalnız başarılı kayıtla ilerlemeli; rotasyon geçerli son kaydı korumalı.

**Tamamlanma ölçütü:** Büyük bir serileştirme tamponu oluşturulmaz; yarıda kalan ve eşzamanlı kayıtlar geçerli checkpoint'i bozmaz. Küçük verili hata enjeksiyonu yeterlidir; ağır eğitim gerekmez.

### TRAIN-02 — Resume parçalı başarısızlıktan geri alınmıyor

[Aktif TrainingManager restore'u](../../training_management/v2/core/training_manager.py#L304) modelden sonra optimizer/scheduler/loop/RNG yükler. Sonraki hatada model çoktan değişmiştir. [Bağımsız CheckpointManager.load](../../training_management/v2/utils/checkpoint_manager.py#L278) optimizer hatasını ayrıca uyarıyla geçer.

**Geliştirme:** Ortak resume planı, tüm yüklerin ön doğrulaması ve düşük bellekli hata stratejisi. Kısmen restore edilmiş durum eğitim için kullanılmamalı; hata sonrası kullanılabilirlik açıkça tanımlanmalı. Aynı veriyi birden fazla biçimde yorumlayan yükleme yolları birleştirilmeli.

**Tamamlanma ölçütü:** Bozuk optimizer/scheduler/RNG örneğinde başarılı resume raporlanmaz ve sonraki eğitim adımı kısmi durumla çalışmaz. Doğru epoch sınırı devamı korunur.

### DATA-01 — Hizalama ve kayıt biçimi bütün veri boyunca denetlenmiyor

[V3 _validate_alignment](../../training_system/v3/core/training_service_v3.py#L272), yalnız ilk kaydı ve `input == target` durumunu kontrol ediyor. [Ortak split](../../training_system/data_split.py), bölme/gruplama için biçim denetler fakat bütün eğitim hedefi sözleşmesini uygulamaz.

**Geliştirme:** Hazırlama ve tüketimin ortak kayıt şeması; uzunluk, token ID aralığı, kaynak kimliği, padding/loss maskesi ve kayıt türüne uygun next-token hizalaması. Soru-cevap/prefix maskeleri normal metinle aynı varsayıma zorlanmamalı. Büyük veri için akış halinde doğrulama ve kayıt konumunu bildiren hata.

**Tamamlanma ölçütü:** İlk kayıt doğru, ilerideki kayıt bozuk olduğunda eğitim başlamaz. Desteklenen metin ve soru-cevap kayıtları doğru kabul edilir; kaynak dosyalar değiştirilmez. Hazırlama/eğitim konfigürasyonu tek etkin veri kimliği üretir.

### DATA-02 — Yakın tekrar ve kaynaksız kayıtlar

Birebir tekrar ve kaynak grupları ortak split'te korunuyor. Kaynaksız parçaların belge ilişkisi ve kısmi örtüşen metinler için bu yeterli değil.

**Geliştirme:** Kaynak/chunk izlenebilirliği; önce ucuz tekrar/örtüşme raporu, ardından açık eşiklerle gruplanmış ayrım. Kaynak kimliği bulunmadığında bunu metadata'da belirtmek.

**Tamamlanma ölçütü:** Aynı/örtüşen belgeden gelen parçaların hangi split'e ve neden yerleştirildiği izlenebilir. Yakın tekrar eşiğinin meşru farklı örnekleri yanlış birleştirme etkisi örneklerle ölçülür.

## Dördüncü paket: bilişsel kalite ve çalıştırılabilir proje

| İş | Geliştirme | Tamamlanma ölçütü |
|---|---|---|
| COG-06 — Politika/critic/retrieval değerlendirmesi | Türkçe görevler, kaynaklı cevaplar, aritmetik, uzun konuşma ve araç başarısızlığı örnekleri. Kurala dayalı factuality skorunu kanıt doğrulaması gibi sunmayan ölçüm. | Direct/think/debate/ToT aynı görev setinde doğruluk, retrieval ilgisi, tutarlılık, çağrı/token maliyeti ve fallback oranıyla karşılaştırılır. |
| COG-07 — Hesap bütçesi | ToT genişlik/derinlik, deliberation, self-consistency ve critic turlarına ortak istek bütçesi; iptal ve zaman sınırı. | Bütçe bitince kısmi durum sızdırmadan açık sonuç döner; düşük donanımda kontrolsüz tekrar üretimi oluşmaz. |
| TOK-01 — Kayıpsızlık ve değişmez kimlik | Mevcut text-loss denetimini Unicode dönüşüm kayıplarıyla değerlendirmek; çalışan tokenizer'ın değişimini yaşam döngüsüne bağlamak. | Mevcut token ID'leri ve eğitim dosyaları korunur. Kayıpsız yeni tokenizer gerekiyorsa ayrı kimlik ve açık ağırlık/veri göç planı hazırlanır. |
| DEV-01 — Kurulum ve girişler | Kök bağımlılık manifesti ve core/tokenizer/training/API grupları; aktif başlatıcılar; eski kopya belge/girişlerin işaretlenmesi. | Temiz CPU ortamında çekirdek örneği ve seçilmiş küçük akışlar belgelenen kurulumdan çalışır; opsiyonel paketler zorunlu görünmez. |

COG-06 için mevcut eğitilmiş model, kendi tokenizer'ı ve ayarlarıyla değerlendirilebilir. Önceki eğitim çıktıları geçmiş çalışma kanıtı olarak korunur; yeni değerlendirmelerin model/checkpoint/veri kimliği ayrı kaydedilir. İlk üç paketin doğruluğunu denetlemek için büyük model eğitmek gerekmez.

## Çalışma ve teslim biçimi

1. Bilişsel pakette ortak request/context/strategy/tool/critic sözleşmelerini belirle, ilgili bileşenleri birlikte düzelt.
2. Model/cache sahipliğini ve eğitim kayıt/veri sözleşmesini alt sistem sınırlarında uygula.
3. Anlamlı değişiklik kümesi tamamlandığında gerçek davranışı doğrulayan küçük CPU ve eşzamanlılık kontrollerini çalıştır. Eski test beklentilerini gerçek hedef davranışa göre değerlendir; test sayısını başarı ölçüsü yapma.
4. Değişen davranışı, eski ağırlıklara etkisini ve kalan sınırları aynı mimari/modül belgelerine işle. Kanıtın ötesinde kalite/hız iddiası üretme.
5. Yerel değişiklikler olgunlaştığında ayrı Git hazırlığı yap: yazar kimliği, dosya kapsamı ve checkpoint/veri gibi büyük varlıkların yayın düzenini gözden geçir. Bu belge commit veya push işlemi başlatmaz.

İlk tamamlanacak bütün, **doğru talimatı ve özellikleri kullanan, aynı stratejiyi senkron/asenkron yolda yürüten ve araç/critic sonucunu doğru isteğe bağlayan bilişsel akıştır**.
