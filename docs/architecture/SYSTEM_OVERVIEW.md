# Cevahir'i bir bütün olarak okumak

**20 Eylül 2026.** Bu belge, modüllerin ana kaynak akışları üzerinden Cevahir'in birlikte ne sunduğunu açıklar. Genel tanıtım [README](../../README-TR.md), ayrıntılı davranış ve sınırlar [mimari sözleşme](CEVAHIR_ARCHITECTURE_SPEC.md), açık geliştirme işleri [yol haritası](NEXT_DEVELOPMENT_ROADMAP.md) içindedir.

## Temsilden öğrenmeye, öğrenmeden kullanıma

Cevahir'in mimari değeri, dil modeli geliştirmede birbirine bağlı üç çalışma alanını aynı kod tabanında erişilebilir kılmasıdır: **metnin temsili**, **bu temsilden öğrenen model** ve **öğrenilen modeli kullanan bilişsel/konuşma sistemi**. Bu yapı geliştiriciye hem modelin iç hesapları hem de o modelin cevap oluşturma süreci üzerinde çalışma olanağı verir.

Bir tokenizer ayarı yalnızca metnin nasıl göründüğünü değiştirmez; eğitimde hangi ID dizisinin kullanıldığını da belirler. Bir attention veya FFN tercihi yalnızca bir sınıf adı değildir; modelin kurulumu, gradyanları ve checkpoint uyumluluğuna uzanır. Bir bilişsel strateji de modelden bağımsız bir etiket değildir; aday üretimi, puanlama, bağlam ve ek çağrı maliyeti oluşturur. Cevahir'in modülleri bu ilişkilerin kod üzerinde izlenmesine izin verir.

Bu değerlendirme, projenin tüm kombinasyonlarda tamamlanmış olduğu iddiası değildir. Gerçekleştirilmiş eğitimler, uygulanmış sistem bileşenleri ve devam eden mühendislik çalışmaları birlikte projenin birikimini oluşturur. Bir modülün katkısı, yalnız dosyada bulunmasıyla değil, hangi girişten çağrıldığı ve çıktısının nerede tüketildiğiyle açıklanmalıdır.

## Dil temsili: Türkçeye özel geliştirme ve çok dilli kullanım

Cevahir **Türkçeye özel geliştirilmiş tokenizer altyapısına** sahiptir. [BPEManager](../../tokenizer_management/bpe/bpe_manager.py), [ön işleme](../../tokenizer_management/bpe/tokenization/pretokenizer.py), [heceleme](../../tokenizer_management/bpe/tokenization/syllabifier.py), [morfoloji yardımcıları](../../tokenizer_management/bpe/tokenization/morphology.py), encoder/decoder ve BPE eğitimini bir araya getirir. [TokenizerCore](../../tokenizer_management/core/tokenizer_core.py), bu bileşenleri hem metin girişine hem eğitim verisi üretimine açar.

Türkçe odağı, her dil için ayrı tokenizer zorunluluğu oluşturmaz. Aynı sözlük ve birleşim kuralları birden fazla dilin metnini temsil edebilir. Burada üç ayrı soru vardır:

- **Temsil:** Ön işleme karakterleri koruyor mu; sözlük ve alt parçalar metni ifade edebiliyor mu?
- **Verim:** Aynı anlam için kaç token gerekiyor; bunun bağlam uzunluğuna ve işlem maliyetine etkisi ne?
- **Öğrenme:** Model hangi verilerle eğitildi; öğrenilen ağırlıklar o dilde ne kadar iyi sonuç veriyor?

Cevahir'de heceleme ayara bağlıdır; ana tokenizasyon yolundaki morfoloji eklemesi heceleme ve `include_morphology` birlikte açıkken çalışır. Mevcut ön işlemenin karakter filtreleri evrensel Unicode kapsamı sunmaz. Bu sınırın geliştirilmesi ile dile özel yeni tokenizer zorunluluğu farklı konulardır. Mevcut eğitimden gelen token ID'leri ve ağırlık ilişkisi korunmalıdır.

## Eğitim: mimariyi çalışan öğrenme sürecine bağlamak

[Veri yükleyici](../../data_loader_management/data_loader_manager.py), belgeler ve soru-cevap kayıtları için metin/kaynak bilgisi sağlar. TokenizerCore bunları ID dizilerine ve eğitim hedeflerine dönüştürür. [Hazırlama aracı](../../training_system/prepare_cache.py), veriyi yeniden kullanmak üzere kaydeder. Kaynak içeriği, tokenizer ve ilgili ayarlar [ortak veri kimliğine](../../training_system/cache_identity.py) katılır.

[V3 eğitim servisi](../../training_system/v3/core/training_service_v3.py), hazırlanmış veriyi doğrular, kaynak/tekrar gruplarıyla ayırır, uzunluklara göre batch oluşturur ve modeli eğitim bileşenleriyle kurar. [V2 TrainingManager](../../training_management/v2/core/training_manager.py) ve [TrainingLoop](../../training_management/v2/core/training_loop.py) bu model üzerinde kayıp, MoE yardımcı hedefi, geri yayılım, token ağırlıklı gradyan biriktirme ve optimizer adımlarını yürütür.

Bu bağlantının katkısı, çekirdekteki bir tercihin veri ve optimizasyonla birlikte denenebilmesidir. Örneğin yoğun FFN yerine MoE seçimi, router kaybının eğitim hedefine katılmasını da gerektirir; Cevahir'in etkin yolu bu aktarımı içerir. GQA başlık sayısı veya FFN genişliği ise aynı kurulum sözleşmesiyle modele taşınır. Ayrı V3 training manager ve bazı gelişmiş optimizasyon yardımcıları etkin backend'e henüz bağlı değildir.

## Model: kurulan hesabı saklamak ve yeniden kullanmak

[ModelManager](../../model_management/model_manager.py), [ortak ayarlar](../../model_management/config_schema.py) üzerinden [CevahirNeuralNetwork](../../src/neural_network.py) kurulumunu yönetir. Decoder; attention, konum kodlaması, normalizasyon, residual ve FFN/MoE seçeneklerini birleştirir. Bu seçenekler geliştiriciye farklı model geometrileri ve çalışma biçimleri üzerinde deney alanı sağlar.

Checkpoint yalnız ağırlıklar olarak ele alınmaz: [ortak kayıt sözleşmesi](../../model_management/checkpoint_contract.py) model kurulum bilgisini ve tokenizer kimliğini de yorumlar. Böylece eşit şekilli tensörlerin farklı bir hesapla veya farklı token anlamlarıyla kullanılmasına karşı denetim yapılabilir. Eski kimliksiz kayıtların sınırları açıkça korunur.

[Profiler](../../model_management/profiler.py) ve [sağlık izleme](../../model_management/health_monitor.py), parametre dağılımı, bellek, hesap tahmini, gradyan ve ağırlık durumunu inceleme olanağı verir. Bunlar ModelManager arayüzüne bağlıdır. İsteğe bağlı attention/tensör tanılamasıyla birlikte, sistem geliştirmeyi yalnız nihai metne bakarak yürütmek zorunda bırakmaz.

## Bilişsel sistem: modelin yeteneklerini bir işlem akışında kullanmak

[Birleşik Cevahir arayüzü](../../model/cevahir.py), tokenizer ve modeli bir üretim/puanlama adaptöründe buluşturur. [CognitiveManager](../../cognitive_management/cognitive_manager.py) bu adaptörün `generate` ve `score` işlemlerini kullanır. Böylece model, metin devamı üretmenin yanında bilişsel akışın aday üreticisi ve puanlayıcısı olur.

[DeliberationEngineV2](../../cognitive_management/v2/components/deliberation_engine_v2.py) adaylar üretip sıralar; [pipeline](../../cognitive_management/v2/processing/handlers.py) geçmişi, isteğe bağlı retrieval'ı ve gerçek araç sonucunu yanıt bağlamına taşır. Critic gerektiğinde aynı modele revizyon yaptırır. Bu yapı strateji seçimi, çoklu aday arama, bağlam ve revizyon üzerinde ayrı ayrı çalışma olanağı verir. Politika ve critic değerlendirmesinin önemli bölümü kurallara dayanır; ayrı eğitilmiş yargıç model varmış gibi sunulmaz.

Bellek ve araçlar farklı bilgi kaynaklarıdır: model ağırlıkları eğitimden öğrenileni taşır, konuşma belleği oturumdan gelen bilgiyi saklar, retrieval ilgili kayıtları getirir, araçlar belirli bir işlemi yürütür. Bu kaynakların birleşmesi Cevahir'in bilişsel geliştirme alanını oluşturur. Mevcut talimat, entropi, araç parametresi, async ToT ve critic eşzamanlılığı borçları [yol haritasında](NEXT_DEVELOPMENT_ROADMAP.md) açık olarak yer alır.

## Konuşma ve uygulama: motorun çıktısını kullanıcıya taşımak

[ChattingManager](../../chatting_management/chatting_manager.py), kullanıcının oturumunu doğrular; geçmişten CognitiveState kurar; `Cevahir.process` çağırır; mesajları ve kullanılan mod/araç/revizyon bilgisini saklar. [Storage adaptörleri](../../chatting_management/storage), [repository'ler](../../database/repositories) ve [UnitOfWork](../../database/unit_of_work.py) kalıcı kayıt erişimini sağlar. [Uygulama factory'si](../../api/app_factory.py) bunları HTTP servisleri ve kimlik doğrulamayla bağlar.

Bu katmanlar modelin matematiksel çekirdeğinin uygulamada kullanılabilmesini sağlar. Bilişsel bellek ile SQL konuşma geçmişi aynı depo değildir; farklı sorumlulukları olan bileşenler ChattingManager ve bilişsel state üzerinden buluşur.

## Ana modüllerin birlikte konumu

| Kaynak | Bütüne katkısı | Bağlantı biçimi |
|---|---|---|
| [tokenizer_management](../../tokenizer_management) | Türkçeye özel BPE ve temsil araçları | Eğitim verisi ile model giriş/çıkışının ortak token kimliği |
| [data_loader_management](../../data_loader_management) | Belge/QA okuma ve kaynaklı parçalama | TokenizerCore'a ham kayıt verir |
| [data_processing](../../data_processing) | Konu/metin toplama, PDF dönüştürme, altyazı işleme | Ayrı yardımcı araçlar; hazırlanan çıktılar veri dizininden yüklenir |
| [dataset_subtitle](../../dataset_subtitle) | Altyazı temizleme ve diyalog metni hazırlama | Bağımsız hazırlama yolu; training döngüsünün otomatik adımı değildir |
| [training_system](../../training_system) | Cache, veri ayrımı, batching, bileşen kurulumu | Aktif V3/V2 servislerinden V2 TrainingManager'a geçer |
| [training_management](../../training_management) | Öğrenme döngüsü ve eğitim durumu | Model, loss, optimizer, scheduler ve checkpoint'i koordine eder |
| [src](../../src) | Öğrenilen Transformer hesabı | Token ID'lerini logits'e ve gerekiyorsa cache/yardımcı kayba dönüştürür |
| [model_management](../../model_management) | Model kurma, kayıt/yükleme, profil ve sağlık | Eğitim ve uygulamaların ortak model yaşam döngüsü |
| [model](../../model) | Birleşik facade, üretim ve puanlama adaptörü | Tokenizer/model ile bilişsel katmanın sınırı |
| [cognitive_management](../../cognitive_management) | Strateji, adaylar, araçlar, critic, bellek, izler | Model çağrılarını istek işleme akışında kullanır |
| [chatting_management](../../chatting_management) | Kullanıcı/oturum ve konuşma bağlamı | Cevahir çıktısını geçmiş ve uygulama kayıtlarıyla bağlar |
| [api](../../api) | HTTP, kimlik doğrulama, servis bağlama | Uygulama factory'si üzerinden konuşma sistemini sunar |
| [database](../../database) | Kullanıcı, oturum, mesaj ve metadata kalıcılığı | Storage/repository/transaction sınırı |
| [benchmarks](../../benchmarks), [tests](../../tests), [scripts](../../scripts) | Ölçüm ve geliştirme teşhisi | Ayrı doğrulama/inceleme araçları; bazı eski scriptler ayrıca uyarlanmalıdır |

[data](../../data), [education](../../education), [saved_models](../../saved_models), [checkpoints](../../checkpoints) ve [image](../../image) model/veri/çıktı varlıklarını; [docs](..) mimari ve geliştirme bilgisini taşır. Bunların tümü çalıştırılabilir modül değildir. Araştırma seçenekleri [araştırma defterinde](../research/RESEARCH_LEDGER.md) ayrı tutulur; henüz bağlanmamış bir yardımcı, ana motorun tamamlanmış özelliği sayılmaz.

## Geliştiriciye sunduğu çalışma alanı

Bu mimari üzerinde bir çalışma, metin temsilinden başlayıp eğitim etkisini ve son yanıta katkısını aynı sistem içinde izleyebilir. Bir başka çalışma, korunmuş ağırlıkları kullanarak yalnız retrieval, araçlar veya yanıt stratejisini geliştirebilir. Modeli doğrudan kullanmak, eğitim servisiyle eğitmek veya konuşma uygulamasına bağlamak farklı girişlerdir; aynı birikimin parçalarını kullanırlar.

Cevahir'in gelecek geliştirmeler için değeri burada yatar: Muhammed Yasin Yılmaz'ın oluşturduğu model ve sistem birikimi, sonraki çalışmalar için incelenebilir, değiştirilebilir ve genişletilebilir bir temel sunar. Gerçek eğitim çıktıları bu birikimin parçasıdır; yeni mühendislik doğrulamaları da üzerine yapılan değişiklikleri değerlendirmek için kullanılır.
