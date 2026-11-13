# Cognitive Management

[English](README-en.md) · [Mimari sözleşme](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md) · [Geliştirme yol haritası](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md)

Cognitive Management, Cevahir'in dil modeli üzerine kurduğu yanıt işleme katmanıdır.
Sorguya göre üretim stratejisini seçer; konuşma belleğini, kayıtlı araçları,
aday üretimini ve yanıt revizyonunu aynı akışta birleştirir.
Üretilen yanıtın niteliği bağlı modelin eğitimi ve sağlanan bağlamla birlikte değerlendirilmelidir.

Aktif giriş `CognitiveManager`, koordinatör ise `v2/core/CognitiveOrchestrator` sınıfıdır.
Dosyalardaki V2, V3 ve Phase etiketleri bu modülün gelişim geçmişini anlatır;
sinir ağı çekirdeğinin V7/V8 etiketleriyle aynı sürüm ekseninde değildir.
Bu rehber 20 Eylül 2026 tarihindeki kod akışını esas alır.

## Çalışma akışı

```text
Cevahir.process / CognitiveManager.handle
  → istek kapsamı ve middleware (doğrulama, cache, izleme)
  → özellik çıkarımı ve PolicyRouterV2
  → isteğe bağlı deliberation / Tree of Thoughts
  → geçmiş, RAG ve araç sonucuyla bağlam oluşturma
  → model üzerinden yanıt üretimi
  → isteğe bağlı self-consistency ve critic revizyonu
  → bellek, oturum durumu ve CognitiveOutput
```

`handle_async` de vardır; mevcut uygulama model içeren adımları thread üzerinden yürütür.
Async ToT bağlantısındaki fark aşağıdaki açık işler arasında yer alır.
Orchestrator içinde request batcher etkin değildir; async arayüz toplu model üretimi anlamına gelmez.

| Bileşen | Gerçekte yaptığı iş |
|---|---|
| PolicyRouterV2 | Sorgu türü, kelime işaretleri, karmaşıklık ve eşiklerle mod/üretim ayarı seçer. |
| DeliberationEngineV2 | Modelin `generate` çağrısıyla adaylar üretir, `score` ile sıralar; hata durumunda sezgisel puan kullanabilir. |
| TreeOfThoughts | Aday yolları model çağrılarıyla genişletip puanlayarak arar. |
| SelfConsistencyHandler | Birden fazla üretimi metin benzerliği veya model puanıyla seçer. |
| CriticV2 | Görev uyumu, ilgi, tutarlılık ve içerik işaretlerini değerlendirir; gerektiğinde modele revizyon yaptırır. |
| MemoryServiceV2 | Oturum geçmişini ve kapsamlı episodik belleği yönetir; ilgili kayıtları bağlama ekler. |

`direct`, `think1`, `debate2` ve `tot` üretim stratejileridir.
`debate2` iki farklı bakış açısından aday üretip seçim yapar.
Self-consistency ayrıca etkinleştirilen bir seçim adımıdır.
Politika ve critic puanlarının çoğu kurallara dayanır; öğrenilmiş ayrı bir yargıç model değildir.
Temel factuality kontrolü iddia/belirsizlik kelimelerine bakar, doğruluğu kanıtlamaz.

## Profiller: varsayılan lean, istenirse full

13 Kasım 2026'da güncellendi. `CognitiveManagerConfig()` **lean** profildir
([yapılandırmadaki](../../../cognitive_management/config.py) `COGNITIVE_PROFILES["lean"]`):
her mesaj `direct` yoldan bir cevap üretimi, en fazla bir critic revizyonu
(`critic.max_passes=1`) ve bir entropi ölçümüyle işlenir.
Deliberation (`think1`/`debate2`/`tot`), self-consistency, anayasal eleştiri,
LLM gerçek kararı ve dış iddia doğrulaması kapalıdır; critic Wikipedia'ya istek göndermez.
Vektör bellek iki profilde de açıktır: isteğe bağlı `sentence-transformers` paketi kuruluysa
kodlayıcı modeli, önbellekte yoksa ilk kullanımda Hugging Face'ten indirilir.
`CognitiveManagerConfig.full()`, `config_path` ile verilen dosyada `{"profile": "full"}`
veya `from_env` ile `CM_PROFILE=full` önceki davranışı tek anahtarla geri getirir;
aynı dosyadaki ya da ortamdaki diğer anahtarlar profilin üzerine yazar.
Profil, yönetici kurulurken okunur: `reload_config` ve `update_config` `cfg` nesnesini değiştirir,
bileşenleri yeniden kurmaz.
Full profilde her anayasal ilke yanıtın dilinde bir kez değerlendirilir;
self-consistency yalnız router'ın orta-entropi kapısı açıkken çalışır;
Wikipedia iddia sorguları critic geçişi başına (varsayılan tek geçiş) en fazla üç iddiayı sorgular
ve on saniyelik bütçe dolunca yeni istek başlatmaz. Her isteğin zaman aşımı kalan süreyle sınırlanır;
DNS çözümlemesi bunun dışında kaldığından bütçe kesin bir duvar saati sınırı değil, son tarihtir.
`tests/evolution/test_cognitive_defaults.py` sayaçlı sahte modelle sade bir mesaj için
lean'de 2, full'de 17–33 üretim çağrısı ölçer; bu düzeltmelerden önce varsayılanlar 31–64 çağrı yapıyordu.
Başarısız veya ölçülemeyen entropi `None` olarak kaydedilir ve mesaj `direct` yola gider;
cevabın token bütçesi ve sıcaklığı bu durumda sıfır entropiye göre değil,
entropi ölçeğinin ortasına (0–3 ölçeğinde 1.5) göre hesaplanır.
`policy.allow_inner_steps`; `think1`, `debate2` ve `tot` için ana anahtardır:
tek başına `debate_enabled` veya `tot_enabled` hiçbir şeyi değiştirmez ve `validate()` bunun için uyarır
(ortamdan `CM_ALLOW_INNER_STEPS=true` ile açılır).
Lean varsayılanlarda araştırma denetleyicisinin yürütebileceği tek strateji `direct`tir;
strateji karşılaştırması `policy.allow_inner_steps=True` veya full profil gerektirir.

## Uygulamaya bağlama

[CognitiveManager](../../../cognitive_management/cognitive_manager.py) bir ModelAPI alır:
`generate(prompt, decoding_cfg)` ve `score(prompt, candidate)` metotları gereklidir.
Cevahir facade bu bağlantıyı `CevahirModelAPI` ile kurar.
Aşağıdaki fonksiyon önceden hazırlanmış bir model adaptörünü kullanır; model eğitmez veya ağırlık indirmez.

```python
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveInput, CognitiveState
from cognitive_management.config import CognitiveManagerConfig

def create_conversation(model_api):
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = False
    cfg.memory.enable_rag = False
    cfg.policy.allow_inner_steps = False
    cfg.policy.self_consistency_enabled = False
    cfg.critic.enabled = False
    cfg.tools.enable_tools = False
    return CognitiveManager(model_manager=model_api, cfg=cfg), CognitiveState()

# model_api: uygulamanın önceden başlattığı generate/score adaptörü
# manager, state = create_conversation(model_api)
# output = manager.handle(state, CognitiveInput(user_message="Merhaba"))
# print(output.text)
```

Bu ayarlar tek üretim yoluyla entegrasyona başlamak içindir.
İhtiyaç duyulan strateji, araç, RAG ve critic seçenekleri [yapılandırmadan](../../../cognitive_management/config.py) açılır.
Lean varsayılanlar vektör belleği/RAG'ı (isteğe bağlı embedding paketleri; yoksa keyword aramasına düşülür) ve hesap makinesi aracını yine açık tutar; model çağrısı bütçesi yukarıdaki profil bölümündedir.
`CognitiveOutput` yanıtı, kullanılan modu, başarılı araç kullanımını ve değerlendirme bilgilerini taşır.

## Bellek, araçlar ve cache

- **Bellek kapsamı:** Oturum kimliği ile uygulamanın doğruladığı `state.metadata["user_id"]` kullanılır.
  Aynı konuşmada state korunmalı; farklı kullanıcılar için aynı state paylaşılmamalıdır.
  Notlar, özetler ve episodik geri çağırma bu kapsamı izler.
- **Vektör bellek:** Mevcut store uygulamaları `memory` ve `chroma`dır.
  Embedding adaptörleri isteğe bağlıdır; yükleme başarısız olursa keyword aramasına düşülür.
  Pinecone, Weaviate, Qdrant ve Milvus seçeneklerinin uygulaması henüz yoktur.
- **Hatalar:** Hata middleware'i başarısız isteği devre kesiciye işler ve kısa bir hata yanıtı döner.
  İsteği yeniden denemez ve istek zaman aşımı uygulamaz; çalışan bir model çağrısı kesilemez.
- **Araçlar:** Yerleşik araç sınırlı AST hesap makinesidir; `+`, `-`, `*`, `/` destekler.
  Arama ve dosya araçlarını uygulama `register_tool` ile sağlamalı ve izin listesine eklemelidir.
  Araç adı ancak gerçek yürütme başarılı olunca sonuçta bildirilir.
- **Cache:** Tam eşleşmeli yanıt cache'i kimlik, geçmiş, üretim ayarı, yapılandırma ve bellek revizyonunu içerir.
  Semantic cache sınıfı vardır; normal kapsamlı orchestrator isteklerinde bu yol devre dışıdır.
  Cache, KV cache'den farklıdır; tekrar kullanılan metin yanıtlarını saklar.

## Sıradaki geliştirme sınırları

Kod incelemesi ve küçük bağımlılık kontrollü denemeler şu beş açık davranışı gösterdi:

1. **Entropi:** `CevahirModelAPI.entropy_estimate` ID yerine metin tokenlarını kullanıyor;
   model forward'ına ulaşmadan token çeşitliliği sezgisine düşebiliyor.
2. **Sistem talimatı:** İstek ve varsayılan sistem talimatı final üretim bağlamından düşüyor.
3. **Hesap makinesi girdisi:** Otomatik çıkarım `2+3*4` ifadesini `2+3` olarak kesebiliyor;
   tam ifade yürütücüsü ile doğal dil parametre çıkarımı hizalanmalı.
4. **Async ToT:** Async handler'a ToT nesnesi bağlanmadığından `tot` seçimi tek aday yoluna düşüyor.
5. **Critic eşzamanlılığı:** Ortak `_last_feedback` ve `_last_passes` alanları istekler arasında karışabiliyor.

Ayrıntılı öncelikler ve kabul ölçütleri [geliştirme yol haritasındadır](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md).
Bu maddeler bu belge turunda düzeltilmiş sayılmaz.

## Kod ve diğer belgeler

- [Orchestrator](../../../cognitive_management/v2/core/orchestrator.py) ve [işlem adımları](../../../cognitive_management/v2/processing/handlers.py)
- [Model adaptörü](../../../model/cevahir.py), [critic](../../../cognitive_management/v2/components/critic_v2.py) ve [araç politikası](../../../cognitive_management/v2/components/tool_policy_v2.py)
- [Bellek](../../../cognitive_management/v2/components/memory_service_v2.py), [store factory](../../../cognitive_management/v2/components/vector_store/__init__.py) ve [cache](../../../cognitive_management/v2/middleware/cache.py)

Bu klasörün `architecture`, `guides`, `api` ve `development` altındaki eski belgeleri
tasarım geçmişi ve örnekler içerir; güncel doğrulanmış sözleşme olarak kabul edilmemelidir.
Çelişki olduğunda bu rehber, mimari sözleşme ve aktif kod akışı birlikte esas alınır.
