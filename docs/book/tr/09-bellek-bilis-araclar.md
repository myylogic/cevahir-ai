# 09 · Bellek, bilişsel işlem ve araçlar

[İçindekiler](../README.md) · [Önceki: Üretim](08-uretim.md) · [Sonraki: Uçtan uca sistem](10-uctan-uca-sistem.md)

Token üretmek, kullanıcının isteğini karşılamak için gereken işlemlerin yalnız biridir. Önceki bir mesaj gerekli olabilir; bir hesaplamanın sonucu dışarıdan alınabilir; bir taslak yeniden değerlendirilebilir. Cevahir'in bilişsel katmanı, modelin hangi bilgi ve hangi işlem sırasıyla çağrılacağını yönetir. Buradaki **cognitive processing**, gözlemlenebilir bir yazılım akışının adıdır; insan benzeri zihinsel yeteneğin kanıtı değildir.

## Context, memory ve retrieval aynı şey değildir

**Context**, o model çağrısına verilen içeriktir. **Memory**, sonraki çağrılar için tutulan durumdur. **Retrieval**, bu durumdan mevcut isteğe hangi parçaların taşınacağını seçer. Büyük bir arşiv, bütün arşivin modele görünmesi demek değildir. Ayrıca [önceki bölümün](08-uretim.md) KV cache'i metin arşivi değil attention ara hesabıdır.

| Durum | Cevahir'deki sahibi | Sonraki davranışa etkisi |
|---|---|---|
| Oturum geçmişi | `CognitiveState.history`; hizmet yolunda konuşma deposundan kurulur | Promptun önceki mesajları taşımasını sağlar. |
| Bilişsel episodik kayıtlar | `MemoryServiceV2` içinde kapsam başına sınırlı liste; `_episodic_memory` geçerli kapsamın görünümüdür | İstek kapsamına göre geri getirilecek konuşma parçaları sağlar; bir kapsamın trafiği başka kapsamın kayıtlarını silmez. |
| Vektör deposu | `MemoryServiceV2` tarafından oluşturulan store | Embedding benzerliğine göre adayları seçer; kalıcılık seçilen backend'e bağlıdır. |
| Kullanıcı kayıtları | Chatting katmanının `MemoryStorage` bileşeni | ContextBuilder yüksek öncelikli kayıtları geçmişe ekleyebilir. |
| Model parametreleri | Sinir çekirdeği / ModelManager | Token dağılımını belirler; belleğe kayıt eklemek bunları güncellemez. |

[MemoryServiceV2.add_turn](../../../cognitive_management/v2/components/memory_service_v2.py#L149), geçmiş listesine rol/içerik ekler; ayrıca kapsam, zaman ve benzersiz kimlik taşıyan episodik kayıt oluşturur. Kayıt yalnız kendi kapsamının listesine girer: liste 20 kaydı aşınca o kapsamın son 10 kaydı kalır ve süreç belleğinde en çok `memory.episodic_max_scopes` (varsayılan 1024) kapsam tutulur; sınır aşılırsa en uzun süredir yazılmayan kapsam düşer. Sınır süreç başınadır ve yeni oturum açmak istemcinin elindedir; bu kadar yeni oturum açan bir istemci, diğer kullanıcıların süreç içindeki kayıtlarını yine düşürebilir (vektör belleği açıksa depodaki kopyaları kalır). Önceki sürüm bu kırpmayı bütün kullanıcıların ortak listesine uyguluyordu; başka bir kullanıcının 20 mesajı bir kullanıcının kaydını silebiliyordu. Etkin ve kurulabilir durumdaki embedding adaptörü metni vektöre dönüştürür; vektör deposuna aynı kimlik ve kapsamla ekler. Bu adımın başarısız olması tüm konuşmayı durdurmaz; kod kelime temelli aramaya dönebilir.

Vektör retrieval'da yaygın bir benzerlik ölçüsü

$$
\operatorname{cos}(q,d)=\frac{q^\top d}{\lVert q\rVert\lVert d\rVert}
$$

olur. Bu, iki metnin aynı doğruluk değerine sahip olduğunu göstermez; embedding uzayında yakınlığını ölçer. Cevahir'in [retrieve_context](../../../cognitive_management/v2/components/memory_service_v2.py#L240) metodu sorguyu embed eder, store aramasına `filter_metadata={"scope": current_scope()}` verir ve sonuçları tekrar kapsam bakımından kontrol eder. `hybrid_search_alpha`, vektör ve kelime puanlarını ağırlıklı birleştirir. Vektör sonuçları yoksa kelime araması kullanılır. `_semantic_search` adlı eski fallback de kelime örtüşmesi hesaplar; adına bakıp sinirsel anlam araması sayılmamalıdır. Aynı kayıt vektör, kelime ve fallback aramalarından birden çok kez dönebildiği için sonuç listesi kimliğe ve boşluk/harf büyüklüğü normalize edilmiş metne göre tekilleştirilir; fallback yalnız boşluk doldurur. `add_note` ile eklenen notlar vektör deposuna yazılmaz, ama aynı kapsamın kelime aramasına ve fallback'ine girer; bu yüzden `Cevahir.add_memory` ile eklenen not, sorgu sözcükleri eşleştiğinde `search_memory` ile bulunur. Doğrudan yerel çağrılar `local` kapsamını, `process` istekleri kullanıcı/oturum kapsamını kullandığından istek dışında eklenen bir not oturum promptlarına kendiliğinden girmez.

`memory.enable_vector_memory`, `embedding_provider`, `vector_store_provider`, `memory.enable_rag` ve eşik alanları [CognitiveManagerConfig](../../../cognitive_management/config.py) içinde seçilir. Factory yalnız `memory` ve `chroma` sağlayıcılarını kurar; ayar şemasında adı geçen Pinecone, Weaviate, Qdrant ve Milvus dalları henüz `NotImplementedError` verir. Varsayılanın etkin olması, dış embedding paketinin/modelinin ortamda bulunacağını garanti etmez. [MemoryVectorStore](../../../cognitive_management/v2/components/vector_store/memory_vector_store.py#L49) süreç içi depodur; kalıcı bir veritabanıyla özdeş değildir. Retrieval-augmented generation, seçilen parçaları üretimin girdisine ekleyerek parametrelerde kodlanmış bilgiyle dış kayıtları birleştirir; özgün [RAG çalışması](https://arxiv.org/abs/2005.11401) bunun ayrı bir modelleme yaklaşımı olduğunu gösterir. Cevahir'in prompt zenginleştirmesi o makalenin bütün eğitim prosedürünü uyguladığı anlamına gelmez.

Bellek bütçesi de bir mekanizma seçer. [summarize_if_needed](../../../cognitive_management/v2/components/memory_service_v2.py#L581), etkin olduğunda altıdan uzun geçmişi son üç kayda indirir; bu metot bir dil modeline özet yazdırmaz. `prune` bu uyumluluk yolunda listenin kopyasını döndürür; asıl bağlam kırpması `build_context` tarafındadır. Oturum özeti enjekte eden ayrı metotla bunları ayırmak gerekir. Eski bir bilginin kaybolması bazen retrieval başarısızlığı değil, bu görünürlük kararının sonucudur.

## Token embedding'den cümle ve belge embedding'ine

[Sinir ağı bölümünde](03-sinir-aglari.md) bir token kimliği, öğrenilmiş tablonun bir satırını seçiyordu. Bu işlemde çıktı şekli `B × T × D` idi: her konuma bir vektör düşer. Retrieval ise değişken uzunluktaki bir sorguyu veya paragrafı karşılaştırılabilir, sabit boyutlu bir vektöre dönüştürmek ister. Bir token satırıyla bütün cümlenin temsili aynı nesne değildir. Bağlamsal token durumlarının ortalamasını almak olası bir havuzlama işlemidir; hangi metinlerin yakın olması gerektiği eğitim hedefinden de etkilenir. Rastgele bir dil modelinin ortalama durumlarına “anlam vektörü” demek, retrieval kalitesini ölçmüş olmaz.

[Sentence-BERT](https://arxiv.org/abs/1908.10084), cümlelerin ayrı kodlanıp benzerliklerinin karşılaştırılabildiği temsil öğrenimini ele alır. [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) ise soru ve pasaj kodlayıcılarını ilgili pasajı ayırmak için eğitir. Bunlar aynı yöntem değildir; ortak fikir, her sorguda bütün metin çiftlerini birlikte çalıştırma zorunluluğunu azaltmaktır. Kavramsal olarak `q=f(soru)`, `d=g(pasaj)` yazılır. `f=g` seçilebilir; ayrı kodlayıcılar da kullanılabilir. Depodaki vektörlerle sorgu vektörünün boyutunun eşit olması gereklidir, fakat aynı anlamsal uzayda olmalarını tek başına garanti etmez.

Cevahir'deki [SentenceTransformersAdapter.encode](../../../cognitive_management/v2/components/embedding_adapter.py#L155), seçilmiş dış modele `normalize_embeddings=True` verir. [BaseEmbeddingAdapter.encode_single](../../../cognitive_management/v2/components/embedding_adapter.py#L92) tek metni bu arayüzden geçirir. Bu adaptör, çekirdeğin `LanguageEmbedding` tablosunu otomatik olarak yeniden kullanmaz. Model değiştirildiğinde eski kayıtların yeniden kodlanması gerekip gerekmediği değerlendirilmelidir: aynı boyutlu iki farklı embedding modeli de farklı koordinat sistemleri üretebilir. Bu bölümdeki küçük vektör hesabı elle belirlenmiş temsil kullanır; herhangi bir dış embedding modelinin Türkçe retrieval başarısı ölçülmüş değildir.

## Benzerlik puanı, kapsam ve aday seçimi

[MemoryVectorStore._cosine_similarity](../../../cognitive_management/v2/components/vector_store/memory_vector_store.py#L83), yukarıdaki ham kosinüsü doğrudan döndürmez. Sıfır olmayan vektörler için `s=(1+cos(q,d))/2` dönüşümünü uygular. Böylece dik vektörlerin puanı `0.5`, zıt yönlülerin `0`, aynı yönlülerin `1` olur. Sıfır norm durumunda kod özel olarak `0` döndürür. Chroma deposu da aynı ölçeği raporlar: [ChromaVectorStore._similarity](../../../cognitive_management/v2/components/vector_store/chroma_vector_store.py#L313) kosinüsü dönen embedding'den, o yoksa koleksiyonun uzaklık türünden hesaplar (`l2` karesel uzaklıktır, birim vektörde `2-2cos`) ve [cosine_to_score](../../../cognitive_management/v2/components/vector_store/base.py#L50) dönüşümünü uygular. Böylece `memory.rag_score_threshold=t` iki sağlayıcıda da `cos ≥ 2t-1` demektir. Eski varsayılan `0.7`, süreç içi depoda `cos ≥ 0.40` demekti; önceki Chroma dönüşümü `1/(1+d)` ise aynı eşikte yaklaşık `cos ≥ 0.786` istiyordu. Varsayılan sağlayıcı Chroma olduğundan yeni varsayılan `0.893` (`cos ≥ 0.786`) onun eski katılığını korur; `memory` sağlayıcısı bu varsayılanla öncekinden katıdır. Eşiği kendisi yazan bir ayar ise Chroma'da artık daha gevşek okunur: `0.7` iki depoda da `cos ≥ 0.40` demektir. Eşik yine de embedding modeline özgüdür; farklı modellerin kosinüs dağılımları birbiriyle karşılaştırılabilir değildir.

Örneğin `q=(1,0)`, bizim kaydımız `a=(0.8,0.2)`, başka oturumun kaydı `b=(1,0)` olsun:

$$
\cos(q,a)=\frac{0.8}{\sqrt{0.68}}\approx0.9701425,
\qquad s(q,a)\approx0.98507125,
\qquad s(q,b)=1.
$$

Global `top_k=1` alıp sonradan kapsam filtresi uygulamak yalnız `b` kaydını seçer ve onu eleyince boş sonuç bırakır. Önce kapsam filtresi uygulamak `a` kaydını doğru aday kümesinde birinci yapar. [MemoryVectorStore.search](../../../cognitive_management/v2/components/vector_store/memory_vector_store.py#L182) filtreyi puanlama ve top-k'dan önce uygular. Bu sıralama hem görünürlük sınırını hem uygun kayıtların geri getirilmesini etkiler. Buradaki `memory` sağlayıcısı adayları tek tek tarar; yaklaşık en yakın komşu indeksi değildir. Puanlama maliyeti, taranan aday sayısı `N` ve boyut `D` için `O(ND)` düzeyindedir; ayrıca sonuçları sıralar.

## Hybrid retrieval neyi birleştirir?

[MemoryServiceV2._combine_search_results](../../../cognitive_management/v2/components/memory_service_v2.py#L364), aynı içerik metni bulunan sonuçları birleştirir. İki listede bulunan bir kayıt için

$$s_h=\alpha s_v+(1-\alpha)s_k$$

hesaplanır. `s_v=0.9`, `s_k=0.2`, `alpha=0.7` ise `s_h=0.69` olur. Yalnız vektör listesinde bulunan kayıt `alpha*s_v` alır. Dolayısıyla bu formül, önceki aday seçiminin ve eksik bileşenlerin etkisini de taşır; bütün arşiv için iki puanın eksiksiz hesaplandığı varsayılmamalıdır.

Kelime puanı tam bir BM25 uygulaması değildir. [_keyword_search](../../../cognitive_management/v2/components/memory_service_v2.py#L475), sorgu sözcüklerinin içerikte bulunma oranını ve ilk eşleşmenin konumunu kullanır. Alt dize araması sözcük sınırı eşleşmesiyle aynı değildir. Ayrıca vektör puanını `[0,1]` aralığına taşımak iki puanı kalibre edilmiş olasılıklar yapmaz. `alpha=0.7`, “cevap yüzde yetmiş anlamsal olarak doğrudur” demek değildir. İçeriğe göre birleştirme aynı metnin farklı köken kayıtlarını da tek sonuca indirebilir; kanıtın kaynağını inceleyen uygulama bu ayrıntıyı hesaba katmalıdır.

Retrieval değerlendirmesinde ilgili kaydın ilk `k` içinde bulunması, getirilen kaydın güncelliği ve son cevabın doğruluğu ayrı ölçümlerdir. Birincisi iyi olduğu halde üretici kaydı yanlış yorumlayabilir. Birbirine benzeyen yanlış kayıtlar yüksek puan alabilir. Deneyde sorgu kümesi, ilgili kayıt etiketleri, kapsam ve eşik sabitlenmeden yalnız ortalama benzerlik puanının artmasını başarı saymak yeterli değildir.

## RAG: dış kayıt, üretim ve eğitim ilişkisi

Özgün RAG modellemesinde getirilen pasaj `z`, üretim dağılımında bir ara değişkendir; dizi düzeyindeki fikir yaklaşık olarak `p(y|x)=sum_z p(z|x)p(y|x,z)` biçiminde yazılır. Makale, getirici ve üreticiyi içeren ince ayar yaklaşımını ve pasajın dizi veya token düzeyinde kullanılmasını inceler. Cevahir'in [RAGEnhancer.enhance_context](../../../cognitive_management/v2/components/rag_enhancer.py#L74) metodu ise seçilen kayıtları metin halinde mevcut bağlama ekler. Promptta zaten bulunan kayıtlar (yakın geçmiş, güncel soru) yeniden referans yapılmaz; kalan referanslar son kullanıcı turunun önüne yerleşir ve güncel soru promptta bir kez geçer. Önceki sürümde ikinci turun promptu ilk kullanıcı mesajını üç, güncel soruyu iki kez taşıyordu. Bu yolda pasajlar üzerinden öğrenilen marjinal olasılık toplamı veya ortak RAG eğitimi yoktur. Kavram benzerliği, uygulamaların matematiksel eşdeğerliği anlamına gelmez. [Lewis ve diğerleri, 2020](https://arxiv.org/abs/2005.11401).

Bağlama sığan kayıt ile üreticinin etkili kullandığı kayıt da ayrılır. [Lost in the Middle](https://arxiv.org/abs/2307.03172), incelediği soru yanıtlama ve anahtar-değer görevlerinde ilgili bilginin konumunun performansı etkileyebildiğini gösterir. Bu sonuç bütün modeller ve görevler için aynı düşüşü garanti etmez; Cevahir için ölçülmesi gereken bir etkeni işaret eder. Kayıt sayısı, sırası, yinelenen içerik ve bağlam kırpması birlikte kaydedilmelidir. Daha uzun prompt, daha doğru cevap için yeter koşul değildir.

## Bilişsel çağrı sırası

[CognitiveManager.handle](../../../cognitive_management/cognitive_manager.py#L515), durum ve isteği doğrular, ardından orchestrator'a iletir. [CognitiveOrchestrator._build_pipeline](../../../cognitive_management/v2/core/orchestrator.py#L172) gerçek handler zincirini kurar. Akıştaki bir bileşenin bulunması, her istekte ek model çağrısı yapması anlamına gelmez.

Hangi dalların etkin olduğunu [COGNITIVE_PROFILES](../../../cognitive_management/config.py#L314) belirler. Varsayılan `lean` profil iç adımları (`think1`, `debate2`, `tot`), self-consistency'yi, anayasal eleştiriyi, LLM gerçek kararını ve dış iddia doğrulamasını kapatır: her mesaj `direct` daldan bir cevap üretimi, en çok bir critic revizyonu ve bir entropi ölçümü alır; critic Wikipedia'ya istek göndermez. Vektör bellek iki profilde de açıktır: isteğe bağlı `sentence-transformers` kuruluysa kodlayıcı modeli, önbellekte yoksa ilk kullanımda Hugging Face'ten indirilir. `CognitiveManagerConfig.full()` aşağıdaki diyagramın bütün dallarını tek anahtarla açar. Sayaçlı sahte modelle [test_cognitive_defaults](../../../tests/evolution/test_cognitive_defaults.py) sade bir mesaj için lean'de 2, full'de 16–18 üretim çağrısı ölçer; önceki varsayılanlar (`3aa1e47`) aynı mesajlarda 31–64 çağrı yapıyordu.

```mermaid
flowchart TD
    A[CognitiveState ve CognitiveInput] --> B[Özellik çıkarma ve retrieval]
    B --> C[PolicyRouting]
    C --> D{Seçilen mode}
    D -- direct --> F[ContextBuilding]
    D -- think1 / debate2 / tot --> E[Deliberation]
    E --> F
    F --> G[Geçmiş, RAG, düşünce adayı, gerçek araç sonucu]
    G --> H[Generation: ModelAPIAdapter]
    H --> I[Etkinse SelfConsistency]
    I --> J[Critic]
    J --> K[MemoryUpdate]
    K --> L[CognitiveOutput]
```

[PolicyRouterV2.route](../../../cognitive_management/v2/components/policy_router_v2.py#L82), özelliklerden risk ve geçiş kapıları çıkarır. `_select_mode` önce `allow_inner_steps` ayarını kontrol eder; ayar lean varsayılanda kapalı olduğundan her mesaj `direct` dala gider. Bu ayar `think1`, `debate2` ve `tot` için ana anahtardır: `debate_enabled` veya `tot_enabled` tek başına açılırsa hiçbir şey değişmez ve `validate()` uyarı verir. Ayar açıkken ölçülemeyen entropi (`entropy_est=None`) de `direct` sonucunu verir. Ölçülemeyen entropide cevabın token bütçesi ve sıcaklığı, ayardan bağımsız olarak sıfır entropiyle değil ölçeğin ortasıyla (0–3 ölçeğinde 1.5) hesaplanır. Kısa sohbet ve yaratıcı içerik doğrudan dala gider. ToT etkin ve kapısı açıksa `tot`; matematik/kod, karmaşıklık ve debate ayarlarına göre `think1` veya `debate2`; diğer koşullarda risk/entropy ve önceki mod etkili olur. Bunlar kodlanmış seçim sezgiselleridir. Sorgu tipi ve alan, anahtar kelime listelerinden çıkarılır; [keyword_in_text](../../../cognitive_management/v2/utils/turkish_text.py#L224) Türkçe büyük/küçük harf kurallarını uygular ve kelimeyi başından, kısa köklerde yalnız çekim ekleriyle eşleştirir: "yaz" artık "yazılım"ı, "hak" "hakkında"yı, "kaç" "kaçırdığım"ı saymaz. Geniş zamanın kişi ekleri ("önerirsin") çekim sayılır; kısa kökten türeyip listeye ait olan sözcükler ("suçlu", "etkili") listelerde ayrıca yazılıdır. Beş harfli "edebi" her devamı kabul ettiğinden "edebilir" (yapabilir) ayrıca dışlanmıştır. Bu bir morfolojik çözümleyici değildir; eşsesli sözcükler yine yanlış sınıflanabilir. Router ismi, kendi başına öğrenilmiş bir yönlendirme ağı bulunduğunu göstermez. Deneysel araştırma denetleyicisinin koşullu değişiklikleri [araştırma bölümünde](11-arastirma-laboratuvari.md) ayrıca ele alınır.

[DeliberationHandler._process](../../../cognitive_management/v2/processing/handlers.py#L328) `think1` için bir, `debate2` için iki aday ister. İki adayda çeşitlilik filtresi ve skor seçimi vardır; bu dal iki bağımsız kişinin karşılıklı tartışmasını simüle eden sınırsız bir sistem değildir. `tot` dalı, [TreeOfThoughts.solve](../../../cognitive_management/v2/components/tree_of_thoughts.py#L145) üzerinden düşünce yollarını değerlendirir; başlatılamaz veya işlenemezse `think1` fallback'i bulunur. [Özgün ToT yaklaşımı](https://arxiv.org/abs/2305.10601), arama ağacında aday üretme ve değerlendirmeyi birleştirir. Cevahir'de dal sayısı ve derinlik maliyeti sınırlar; dallanmanın varlığı kalite artışını tek başına kanıtlamaz.

Model belirsizliği için [CevahirModelAPI.entropy_details](../../../model/cevahir.py#L899), tokenizer'ın ID çıktısını kullanır ve cache kapalı ileri geçişin son logits'inden entropi hesaplar. `H(p)/log(V)` ile normalize eder; hesaplanamayan durumda `available=False`, nötr `value=0.5` ve gerektiğinde hata türü döner. Bu dağılım entropisidir; cevabın olgusal doğruluğunun kalibre edilmiş olasılığı değildir. `entropy_estimate` yalnız sayıyı döndürdüğünden kaynağı bilmesi gereken tüketici ayrıntılı arayüzü kullanmalıdır. [ModelAPIAdapter.estimate_entropy](../../../cognitive_management/v2/adapters/backend_adapter.py#L150) bunu yapar: `available=False`, hata, eksik yöntem veya sonlu olmayan değer `None` döner. [FeatureExtractionHandler._process](../../../cognitive_management/v2/processing/handlers.py#L180) bu durumda `entropy_est=None` ve `uncertainty_source="unavailable"` yazar; yer tutucu bir sayı üç katına çıkarılıp eşik olarak okunmaz.

### Entropi ile doğruluk arasındaki karşı örnek

İki olası token için `p=(0.99,0.01)` dağılımının normalize entropisi yaklaşık `0.0808`'dir. En olası token yanlış bilgiyi başlatıyorsa düşük entropiye rağmen cevap yanlış olabilir. `p=(0.5,0.5)` için normalize entropi `1`'dir; iki token aynı doğru cevabın farklı ifade biçimlerini başlatabilir. Bu örnekler, sonraki token kararsızlığıyla bilgi doğruluğunun farklı değişkenler olduğunu gösterir. Router'ın sayısal eşikleri yerel mühendislik kararlarıdır; dosya yorumunda bir makalenin adının bulunması, belirli eşik değerinin o makalede doğrulandığını kanıtlamaz. Özellikle `available=False` değerini gerçek ölçüm gibi yorumlamak, başarısız ölçümü davranış kararına gizlice taşır. Önceki sürümde tam olarak bu oluyordu: nötr `0.5` üç katıyla tam `1.5` düşünme eşiğine, adaptörün hata sabiti `0.8` ise `2.4`'e çıkıp mesajı `think1` dalına gönderiyordu.

### Çok adaylı seçim ve ortak hata

[Self-consistency çalışması](https://arxiv.org/abs/2203.11171), örneklenmiş akıl yürütme yollarından ulaşılan cevapları bir araya getirir. Yerel [_majority_select](../../../cognitive_management/v2/processing/handlers.py#L92) ise aday metinlerin bigram örtüşmelerini karşılaştırıp ortalama benzerliği yüksek adayı seçer. Bu, son cevapların anlamsal eşitliğine göre oy saymakla aynı değildir. `_score_select`, backend skoruna göre seçim yapar; `hybrid` dalı iki yöntemi birleştirir. Yerel `agreement_score` alanı bir doğruluk olasılığı olarak kalibre edilmiş değildir. [SelfConsistencyHandler._should_activate](../../../cognitive_management/v2/processing/handlers.py#L705) yalnız router'ın `self_consistency_gate` kapısı açıkken örnekler: self-consistency etkin, entropi ölçülmüş ve `entropy_gate_think ≤ e < entropy_gate_debate` aralığında olmalı, sorgu sohbet veya yaratıcı türde olmamalıdır. Önceki sürümde handler bu kapıyı okumuyor, uygun her sorguda üç ek örnek üretiyordu.

Basit bir düşünce deneyinde üç bağımsız adayın her biri `p=0.7` olasılıkla doğru ikili cevabı veriyorsa çoğunluğun doğruluğu `3p²(1-p)+p³=0.784` olur. Adaylar tamamen aynı hatayı paylaşıyorsa üç kez örneklemek doğruluğu `0.7`'den artırmaz. Bu hesap Cevahir ölçümü değildir; bağımsızlık varsayımının neden belirleyici olduğunu gösterir. Aynı model, aynı yanlış retrieval kaydı ve aynı değerlendirme ölçütü ortak hatayı koruyabilir.

[TreeOfThoughts._expand_node](../../../cognitive_management/v2/components/tree_of_thoughts.py#L280) her genişletmede `branching_factor+1` aday üretmeye çalışır; seçilen dallar sonraki seviyeye taşınır. Aday üretme ve değerlendirme çağrıları ayrı maliyetlerdir; varsayılan derinlik 3 ve dallanma 3 ile bir ToT yolu en çok 52 üretim ve 52 değerlendirme çağrısı yapar. ToT bu yüzden yalnız full profilde veya açıkça etkinleştirildiğinde çalışır. Skorlama başarısız olursa kelime örtüşmesi, uzunluk ve adım belirteçlerinden oluşan sezgisel değerlendirme kullanılabilir. Bu durumda daha çok dal, dış dünyadan daha çok bağımsız doğrulama anlamına gelmez. ToT'nin [özgün araştırmadaki](https://arxiv.org/abs/2305.10601) sonuçları, yerel uygulamaya ölçüm yapılmadan aktarılmamalıdır.

## Araç seçmek ile araç kullanmak

Bir araç, modelden farklı bir hesap veya dış bilgi kaynağı sağlayan kayıtlı fonksiyondur. Araç adını cevaba yazmak gerçek yürütme değildir. [ContextBuildingHandler._process](../../../cognitive_management/v2/processing/handlers.py#L474), seçilen araç için açık `request.metadata["tool_parameters"]` değerlerini veya çıkarılmış parametreleri alır. Sonra [ToolExecutorV2.execute](../../../cognitive_management/v2/components/tool_executor_v2.py#L139) çağrılır. Kayıt, etkinlik ve izin listesi denetlenir; fonksiyon imzası ile temel parametre türleri doğrulanır. Yalnız başarılı dönüşten sonra `context.tool_name` atanır ve sonuç prompta eklenir:

```python
result = executor.execute(selected_tool, parameters)
context.tool_name = selected_tool
context_text += f"\n\n[ARAÇ SONUCU: {selected_tool}]\n{result}"
```

Bu gerçek [kaynak parçası](../../../cognitive_management/v2/processing/handlers.py#L538), başarı bilgisinin nereden geldiğini gösterir. Varsayılan calculator sınırlı AST aritmetiği uygular: sayılar ve `+ - * /`. Arama ve dosya araçları kendiliğinden dış dünyaya bağlanmaz; çağıran tarafından kaydedilmelidir. Bir şemanın bulunması da tam bir güvenlik sınırı veya sonuç doğruluğu ispatı değildir.

Kaynak incelemesinde somut bir hata bulundu: önceki `8fb26e3` sürümünde [ToolPolicyV2.infer_tool_parameters](../../../cognitive_management/v2/components/tool_policy_v2.py#L184), `2+3*4` isteğini `2+3` olarak kesiyordu; ifade bulamazsa ilk iki sayıyı kendiliğinden topluyordu. [Düzeltme öncesi çalıştırma kaydı](../evidence/runtime_walkthrough_before_tool_fix.json) bu negatif sonucu korur. Bu kitap turunda hata düzeltildi: çıkarıcı tek ve tam bir aritmetik parçayı AST ile doğrular; parantez, ondalık, tekli işaret ve bilimsel gösterim korunur. Desteklenmeyen, eksik veya birden fazla ifade varsa işlem uydurulmaz. Açık `tool_parameters` bu çıkarım dalını atlar ve yürütücü tarafından ayrıca denetlenir. Bu küçük tanıyıcı doğal dildeki bütün matematik sorularını çözmez; örneğin virgüllü ondalık veya cümle sonu noktalaması açık ifade gerektirebilir.

Ardından [GenerationHandler._process](../../../cognitive_management/v2/processing/handlers.py#L569) oluşturulan promptu `backend.generate` ile modele verir. [ModelAPIAdapter.generate](../../../cognitive_management/v2/adapters/backend_adapter.py#L102), bilişsel `decoding_config` sözleşmesini `CevahirModelAPI` arayüzüne taşır; gerçek token döngüsü önceki bölümdeki adaptörde çalışır. Araç ve retrieval bu nedenle normal cevap üretiminden önce bağlama katılır; her istekte üretimden sonra devreye giren zorunlu ayrı aşamalar değildir.

### Araç doğruluğunu üç aşamada okumak

İstek `x`, parametre çıkarıcısı `P`, araç `T` ve üretici `G` olsun. Gerçek akış kabaca `G(x,T(P(x)))` biçimindedir. Doğru araç, yanlış parametreyi de doğru hesaplayabilir. Düzeltme öncesinde `x=2+3*4` için çıkarıcı `P(x)=2+3` üretiyor, araç sonucu `"5"` oluyordu. Açık parametre `operation=2+3*4` olduğunda aynı araç `"14"` döndürüyordu. Güncel çıkarıcı her iki yolda da tam ifadeyi taşır ve sonuç `"14"` olur; [güncel yürütme kaydı](../evidence/runtime_walkthrough.json) bunu doğrular. Yürütücünün iki çağrısı da istisna üretmeden tamamlandığından `success_count` ikisini de başarılı sayar. Bu sayaç kullanıcı niyetinin karşılandığını ölçmez; fonksiyonun başarılı döndüğünü ölçer. Üreticinin sonucu yanıta doğru taşıması üçüncü ayrı denetimdir.

[ReAct](https://arxiv.org/abs/2210.03629), modelin akıl yürütme ve çevreyle eylem adımlarını dönüşümlü kullanmasını inceler. Buradaki handler zincirinde bir araç sonucunun prompta eklenmesi, otomatik olarak böyle bir tekrar planlama döngüsü kurmaz. [Toolformer](https://arxiv.org/abs/2302.04761), hangi API'nin ne zaman ve hangi argümanlarla çağrılacağını öğrenmek için eğitim verisi ve bir eğitim yaklaşımı sunar. Cevahir'in kayıtlı fonksiyonları, seçme sezgiselleri ve düzenli ifade çıkarıcısı o eğitim mekanizması değildir. Bu karşılaştırma araç fikrini küçültmez; hangi yeteneğin koddan, hangisinin eğitimden geldiğini görünür kılar.

## Taslak denetimi, geri bildirim ve öğrenme sınırı

Self-consistency etkin olduğunda birden çok adayın oy/skorla seçilmesi, ek hesap karşılığında cevap seçimini değiştirir. [CriticHandler._process](../../../cognitive_management/v2/processing/handlers.py#L745) taslağı `review_detailed` veya `review` ile denetler; metin, revizyon bilgisi ve geri bildirimleri alır. [CriticV2](../../../cognitive_management/v2/components/critic_v2.py), yapılandırmaya bağlı kontroller ve yeniden üretim yapabilir. Lean profilde bu, sözcüksel puanlar ve en çok bir model revizyonudur. Full profil önce anayasal eleştiriyi çalıştırır: [ConstitutionalCritic.principles_for](../../../cognitive_management/v2/components/constitutional_critic.py#L107) on dört ilkenin her birini yanıtın dilinde bir kez değerlendirmeye verir; önceden aynı ilkenin İngilizce ve Türkçe biçimleri ayrı ayrı, 28 çağrıyla değerlendiriliyordu. Dış iddia doğrulaması en.wikipedia.org'a istek gönderdiği için yalnız full profilde veya açıkça etkinleştirildiğinde çalışır; critic geçişi başına (varsayılan tek geçişte yanıt başına) en çok üç iddia sorgular ve on saniyelik bütçe dolunca yeni istek başlatmaz. Hakem yanıtları [parse_truth_verdict](../../../cognitive_management/v2/utils/verdict_parsing.py#L345) ve `parse_violation_verdict` ile olumsuzluk ve çekince farkında okunur: "Yanlış, doğru değil" yanlış, "Kısmen doğru" kısmen, "uygun değil, ihlal var" ihlal, "ihlal yok" uygun sayılır. İki nokta veya soru işaretiyle biten parça, sonraki parçayla birlikte okunur: "İhlal: yok" ve "İhlal var mı? Yok." uygun, "Uygun: hayır" ihlal, "Doğru mu? Hayır, yanlış." yanlış sayılır. Olumsuz fiil ekleri kelime sonunda aranır; "medikal" bir olumsuzluk sayılmaz. Anayasal yanıtta sayı önce okunur: düz sayı istenen ihlal puanıdır, ama uygunluk sözcüğüne bağlı yüzde veya oran ("%100 uygun", "Uygunluk: %90") uygunluğu ölçer ve ihlal puanı `1-x` olur. Boş veya okunamayan yanıt çekimser kalır; puanı değiştirmez ve revizyon gerekçesi olmaz. Eğitilmemiş bir model bu yönergeleri izleyemediğinden pratikte hakem yanıtlarının çoğu çekimser kalır. Bir modelin kendi ürettiği olumlu değerlendirme, bağımsız çevresel doğrulama sayılmaz. Aynı hatayı hem üreten hem onaylayan bir akış mümkündür.

[MemoryUpdateHandler._process](../../../cognitive_management/v2/processing/handlers.py#L813), dolu son metni geçmişe ekler ve oturum sayaçlarını günceller. Bu yol `loss.backward()` veya optimizer adımı içermez. Cevap iyileştirme, kaydı hatırlama ve ağırlık öğrenme ayrımı bu yüzden korunmalıdır. Araştırma testindeki [test_observations_do_not_self_verify](../../../tests/evolution/test_research_experience.py#L61) ile critic kaynaklarını reddeden test, bu ayrımın deneysel denetleyici tarafındaki somut karşılıklarıdır.

Bir sistemin gelecekteki davranışı ağırlıkları değişmeden de değişebilir: yeni bir kayıt daha sonra geri getirildiğinde cevap değişir. Bu, parametre dışı bir durum güncellemesidir ve geniş bir öğrenme tanımında dikkate alınmalıdır. Ancak yaşam boyu öğrenme iddiası için durumun ne kadar sürdüğü, hangi deneyimle değiştiği, yanlış kaydın nasıl düzeltildiği ve yeni duruma ne ölçüde genellendiği gösterilmelidir. “Optimizer yoksa hiçbir öğrenme yoktur” sonucu da “kayıt varsa genel öğrenme çözülmüştür” sonucu da buradan çıkmaz.

Üç sınama bu ayrımı somutlaştırır. Süreç yeniden başlatılınca etkinin korunması **kalıcılığı**; farklı ifadeli ve önce görülmemiş ilgili girdilere aktarılması **genellemeyi**; hatalı kaydın etkisinin sınırlandırılıp düzeltilebilmesi **kontrolü** sınar. Aynı sorunun aynı kaydı tekrar getirmesi üçünü birden kanıtlamaz. Bellekte yanlış bir asistan cevabı yeniden görülüp kaynakmış gibi kullanılırsa hata kendi tekrarından güç alabilir. Kayıt eklemek ile çevresel doğrulamayı ayıran araştırma sözleşmesi bu yüzden önemlidir. [Araştırma laboratuvarı](11-arastirma-laboratuvari.md), bu mekanizmaların genel yaşayan öğrenme problemini henüz çözmediğini korur.

## Eşzamanlılık ve doğrulama

Asenkron pipeline aynı mantığı ayrı bir algoritmaya dönüştürmez. [AsyncGenerationHandler._process_async](../../../cognitive_management/v2/processing/async_handlers.py#L181) ve diğer adaptörler sync handler'ı `asyncio.to_thread` ile çalıştırır; zincirde sonraki aşama yine öncekinin sonucunu bekler. Bu, olay döngüsünü serbest bırakabilir; paylaşılan modelin cache kilidini kaldırmaz. İstek kapsamı [request_scope](../../../cognitive_management/v2/utils/request_scope.py) içindeki `ContextVar` ile kullanıcı/oturum kimliğinden kurulur. Doğrudan yerel çağrıdaki `local` varsayılanı, doğrulanmış kullanıcı izolasyonunun yerini tutmaz.

[Agent sözleşmesi testleri](../../../tests/evolution/test_agent_contracts.py) kapsamlı retrieval'ı, top-k'dan önce kapsam filtresini, gerçek araç yürütmeyi ve başarısız aracın kullanılmış sayılmamasını kontrol eder. [Bilişsel doğruluk testleri](../../../tests/evolution/test_cognitive_correctness.py) hakem yanıtlarının olumsuzlukla, etiket ve soru-cevap biçimleriyle okunmasını, Türkçe anahtar kelime eşleşmesini, kapsam başına bellek kapasitesini, tekil retrieval sonuçlarını, eşiğin iki vektör deposunda aynı kosinüs anlamını, varsayılan eşiğin Chroma'nın eski katılığını korumasını ve notların aranabilirliğini sınar. [Runtime testleri](../../../tests/evolution/test_runtime_lifecycle.py) sync/async hook'larını ve cache hit sonrasındaki geçmiş güncellemelerini inceler. Eski [bilişsel mimari belgesi](../../modules/cognitive_management/architecture/README.md) bileşenlerin tasarım geçmişi için yararlıdır; oradaki sürüm veya yetenek etiketleri bu bölümün güncel çağrı zincirinin yerine geçirilmemelidir.

## Çalıştırılabilir örnek ve çözümlü alıştırmalar

[Çalıştırılabilir eşlikçi](../../../scripts/book_runtime_walkthrough.py) ile [sonuç kaydı](../evidence/runtime_walkthrough.json), elle seçilmiş iki boyutlu vektörlerle gerçek retrieval metodunu ve gerçek calculator yürütücüsünü denetler. Bu küçük örnekler büyük bir dil modelinin genel başarı ölçümü değildir. Dış embedding modeli indirmeden mekanizma ve sözleşme sınırını görünür kılar.

### Alıştırma 1 — Kapsam filtresinin sırası

Yukarıdaki `a` ve `b` kayıtlarında `top_k=1` ve geçerli kapsam yalnız `a` için uygunsa sonuç ne olmalıdır? **Çözüm:** Önce kapsam kısıtlanır, sonra adaylar sıralanır; `a`, yaklaşık `0.98507125` store puanıyla döner. `0.9701425` ham kosinüstür. Önce global top-k almak uygun kaydı aday kümesinden çıkardığı için sonradan filtrelemek yeterli değildir.

### Alıştırma 2 — Hybrid puan

İki listede bulunan `A` için `s_v=0.9`, `s_k=0.2`; yalnız vektör listesinde bulunan `B` için `s_v=0.95` olsun. `alpha=0.7` ile hangisi öne geçer? **Çözüm:** `A=0.69`, `B=0.665`; `A` seçilir. Bu, `A`'nın yüzde 69 doğru olduğu anlamına gelmez. Aday listelerinin nasıl oluştuğu da sıralamanın parçasıdır.

### Alıştırma 3 — Üç adayın oybirliği

Üç aday aynı yanlış cevabı veriyorsa oybirliği dış kanıt olur mu? **Çözüm:** Hayır. Uyum yalnız çıktıların ilişkisidir. Bağımsız ve başarı olasılığı `0.7` olan ikili adaylar varsayımında çoğunluk `0.784` verir; tam ilişkili adaylarda artış yoktur. Yerel bigram merkez seçiminin bu teorik çoğunluk hesabını uyguladığı da varsayılamaz.

### Alıştırma 4 — Başarılı ama yanlış araç çağrısı

Düzeltme öncesi kayıtta `2+3*4` çıkarımla ve açık parametreyle iki kez yürütülüyor. `success_count=2` ise iki kullanıcı isteği doğru karşılanmış mıdır? **Çözüm:** Hayır. Eski çıkarıcıdan gelen `2+3` sonucu `"5"`, tam ifade sonucu `"14"` idi. Yeni sürüm bu örneği düzeltir, fakat sayacın anlamı değişmez: istisnasız dönüşleri sayar. Parametre çıkarma, araç yürütme ve nihai yanıtın uygunluğu ayrı değerlendirilmelidir.

### Alıştırma 5 — Bir kayıt yeni kapasite midir?

Sistem bir düzeltmeyi kaydedip aynı soruya doğru cevap veriyor. Yaşarken öğrenmenin genel ilkesi kanıtlandı mı? **Çözüm:** Yalnız dar bir durum değişimi gözlendi. Yeni ifadeye aktarım, yeniden başlatma sonrası etki, yanlış düzeltmeye dayanıklılık ve eski yeteneklere etkisi henüz bilinmiyor. Parametre değişmemesi kazanımı yok saymayı gerektirmez; tek kayıtla davranış değişmesi de kalıcı ve genellenebilir kapasite artışını kanıtlamaz.

## Birincil kaynaklar

Aşağıdaki yıllar ilk arXiv sürümünü belirtir. [Kaynak inceleme kaydı](../evidence/academic_memory_sources.json), erişim tarihini ve okunan yerel sembolleri saklar; atıf, Cevahir için ölçülmemiş başarı sonucu aktarmaz.

1. Reimers, N.; Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).
2. Karpukhin, V. ve diğerleri (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).
3. Lewis, P. ve diğerleri (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
4. Wang, X. ve diğerleri (2022). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171).
5. Yao, S. ve diğerleri (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
6. Schick, T. ve diğerleri (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761).
7. Yao, S. ve diğerleri (2023). [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601).
8. Liu, N. F. ve diğerleri (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172).

[İçindekiler](../README.md) · [Sonraki: Uçtan uca sistem](10-uctan-uca-sistem.md)
