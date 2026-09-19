# Sinir ağı çekirdeği

**Güncelleme:** 20 Eylül 2026 · [English](README-en.md)

`CevahirNeuralNetwork`, token dizisini bir sonraki tokenın skorlarına dönüştüren yapılandırılabilir Transformer çekirdeğidir. Dil modeli kullanımında nedensel dikkat uygular; tokenizer, eğitim döngüsü, metin üretimi ve bilişsel orkestrasyon bu çekirdeğin çevresindeki ayrı katmanlardır. Çekirdeğin değeri, dikkat, konum kodlama, yoğun veya uzman tabanlı ileri besleme, bellek ve eğitim seçeneklerinin aynı model kurulumunda birleşmesidir.

Ana uygulama [neural_network.py][core], katmanlar [neural_network_module][layers] içindedir. Projenin bütünü için [mimari sözleşmeye][architecture], sonraki geliştirmeler için [geliştirme planına][roadmap] bakın.

## V4, V7 ve V8 neyi ifade ediyor?

Kaynakta kalan V4/V6 başlıkları modelin yalnızca o yeteneklere sahip olduğunu göstermez. Aşağıdaki adlar kod yorumlarındaki geliştirme dönemleridir; birbirinden bağımsız beş model veya tüm projeye uygulanan ortak bir sürüm numarası değildir.

| Kaynakta kullanılan dönem | Mevcut kodda karşılığı |
| --- | --- |
| V4 | RoPE'nin dikkat katmanına bağlanması; RMSNorm, SwiGLU, ağırlık paylaşımı, KV önbelleği, aktivasyon checkpointing ve isteğe bağlı MoE seçenekleri. |
| V5 | GQA/MQA için `num_kv_heads`, kayan dikkat penceresi ve YaRN/linear RoPE ölçekleme. |
| V6 | PyTorch SDPA yönlendirmesi, QK-Norm, paralel residual, çıktı/dikkat logit soft-cap ve residual projeksiyon başlatma düzenlemeleri. |
| V7 | Derinliğe göre stochastic depth; FFN'de birleşik `gate_up_proj`; katman üzerinden KV tahliye ve attention sink ayarlarının aktarılması. |
| V8 alt katman düzeltmeleri | KV önbelleğinde sink/pencere sınırı kontrolü ve MoE yardımcı kaybının biriktirilmesi/temizlenmesine yönelik düzeltmeler. |

Bu harita [ana model][core], [Transformer katmanı][transformer], [FFN][ffn] ve [KVCache][cache] kaynaklarına dayanır. Örneğin KVCache kendi içinde V5 olarak işaretlenirken aynı seçeneklerin üst katmana aktarılması V7 yorumları taşır.

Yapılandırma sözleşmesindeki `config_version=1` ve `architecture_version="cevahir-capabilities-1"` ise [şemanın][config] alanlarıdır. Bunlar yukarıdaki tarihsel etiketlerden ayrı tutulur. Bir ağırlık dosyasının uyumluluğu isimdeki V sayısıyla değil, gerçek kurulum ayarları, tensör şekilleri ve tokenizer kimliğiyle belirlenir.

## Çalışma akışı

```text
Token kimlikleri [B, T]
  → LanguageEmbedding
  → konum kodlama + dropout
  → TransformerEncoderLayer × N
      → normalizasyon + MultiHeadAttention
      → residual + FFN veya MixtureOfExperts
  → çıktı normalizasyonu
  → vocabulary projeksiyonu + isteğe bağlı logit soft-cap
  → logits [B, T, vocab_size]
```

RoPE, attention içindeki Q/K tensörlerine uygulanır. Sinüzoidal ve öğrenilen konum kodlaması alternatifleri de bulunur. `TransformerEncoderLayer` sınıf adı korunmuş olsa da nedensel yapılandırmada bu katman decoder türü dil modeli akışında kullanılır. `parallel_residual=True` ve `pre_norm=True` birlikte seçildiğinde attention ve FFN dalları aynı normalize girdiden hesaplanır; post-norm yolu seri yapısını korur.

| Bileşen | Görevi ve önemli seçenekleri |
| --- | --- |
| [Ana model][core] | Katmanları kurar, çıktı ağırlıklarını uygun olduğunda embedding ile paylaşır, cache ve teşhis isteğini alt katmanlara iletir. |
| [TransformerEncoderLayer][transformer] | Pre/post norm, residual, stochastic depth ve aktivasyon checkpointing; yoğun FFN veya MoE dalı. |
| [MultiHeadAttention][attention] | MHA/GQA/MQA, nedensel/padding maskeleri, RoPE, QK-Norm, kayan pencere ve backend seçimi. |
| [FeedForwardNetwork][ffn] | SwiGLU/GeGLU gibi gated yollar için birleşik gate/up projeksiyonu; farklı aktivasyon seçenekleri. |
| [MixtureOfExperts][moe] | Tokenları seçilen uzmanlara yönlendirir ve eğitim hedefinin tüketmesi gereken yardımcı kayıp üretir. |
| [KVCache][cache] | Otoregresif üretimde K/V durumunu tutar; kapasite ve tahliye stratejisini uygular. |

## Yapılandırma ve sınırlar

Yeni model kurulumunda [ModelManager][manager] ve [ortak yapılandırma şeması][config] kullanılır. Boyutlar, head ilişkileri ve özellik ayarları burada normalize edilip doğrulanır. Eski doğrudan kurucu çağrılarının varsayılanlarıyla şema varsayılanlarının aynı olduğu varsayılmamalıdır.

- `num_heads`, `num_kv_heads` ile tam bölünmelidir; head boyutu `embed_dim / num_heads` üzerinden hesaplanır.
- `seq_proj_dim` eski dosyalar için korunan bir uyumluluk alanıdır; bağımsız bir sequence projection katmanı oluşturmaz. Farklı bir değer ağırlık paylaşımı seçimini etkileyebilir.
- YaRN ölçekleme veya daha büyük cache kapasitesi yapılandırılabilir. Eğitilmiş modelin yeni uzunlukta dil kalitesi ayrıca değerlendirilmelidir.
- SDPA kullanılması tek başına belirli bir GPU kernelinin çalıştığı anlamına gelmez. Backend seçimi ayarlara, tensörlere, PyTorch'a ve donanıma bağlıdır.
- `attn_logit_cap > 0` veya dikkat ağırlıklarının istenmesi manuel attention hesabını seçer; bellek ve süre maliyeti değişir.
- Aktivasyon checkpointing, eğitimde ara aktivasyonları yeniden hesaplar. KV önbelleği ve diske kaydedilen model checkpoint'i farklı işlevlerdir.
- Quantization kurucu içinde otomatik uygulanmaz; `apply_quantization()` ayrı yaşam döngüsü adımıdır. Kullanılabilir yollar ve bağımlılıklar [quantization yöneticisinde][quantization] tanımlıdır.

## Çıktı ve gözlemlenebilirlik

Normal çekirdek çağrısı `(logits, attention_weights)` döndürür; `use_cache=True` üçüncü değer olarak katman cache çıktılarını ekler. `return_attention_weights=False` varsayılandır ve dikkat matrisi normal akışta döndürülmez. `True` seçildiğinde dönen matris son katmana aittir.

`collect_diagnostics=True`, son çağrının tensör istatistiklerini toplar. Normal çağrı yalnızca küçük şekil/tür özetini saklar; TensorBoard açıksa belirlenen aralıkta ayrıca istatistik toplanabilir. Bu seçenekler eğitim veya üretimin her adımına zorunlu maliyet eklemeden inceleme yapmayı sağlar.

MoE kullanılan özel eğitim döngüleri, `get_and_reset_moe_loss()` ile yardımcı kaybı alıp eğitim hedefine eklemeli ve tüketmelidir. Cache kullanan bağımsız üretim akışları da istekler arasında önceki durumun yaşam süresini yönetmelidir; tek başına çekirdek bir oturum yöneticisi değildir.

## Küçük CPU kullanım örneği

Proje kökünde, PyTorch'un bulunduğu ortamda:

```python
import torch
from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager

torch.set_num_threads(1)
manager = ModelManager(tiny_model_config())
manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
logits, attention = manager.forward(torch.tensor([[1, 5, 8, 2]]), inference=True)
print(logits.shape)  # torch.Size([1, 4, 128])
```

Bu örnek yeni ve küçük bir modelin bağlantılarını gösterir. Projenin önceki gerçek eğitiminden elde edilen ağırlıklar ve [ana README'deki eğitim çıktıları][training] ayrı, korunması gereken proje varlıklarıdır.

## Eğitilmiş ağırlıkların korunması

Model yükleme/kaydetme akışı [ortak checkpoint sözleşmesini][checkpoint] kullanır. Kaydedilen kurulum, aynı tensör şekillerini üretse bile hesabı değiştiren mimari ayarlar ve state şekilleri yükleme öncesinde denetlenir. Tokenizer kimliği içeren dosya için aynı kimliğin doğrulanması gerekir; yalnızca vocabulary büyüklüğünün eşit olması yeterli değildir.

Eski dosyalarda eksik mimari veya tokenizer bilgisi olabilir. Orijinal ağırlıklar ve eğitimde kullanılan tokenizer saklanmalı; uyumluluk açık kurulumla incelenmelidir. Eski dosyanın adına yeni sürüm etiketi eklemek dönüşüm sağlamaz. Yeni mimari denemeleri, korunmuş eğitim dosyalarından ayrı çıktılar üretmelidir.

Sonraki turda öncelik; model/cache sahipliği, eski checkpoint dönüşümlerinin açık kuralları ve farklı yürütme yollarının aynı anlamı korumasıdır. Ayrıntılı kapsam [geliştirme planında][roadmap] tutulur.

[core]: ../../../src/neural_network.py
[layers]: ../../../src/neural_network_module
[transformer]: ../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py
[attention]: ../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py
[ffn]: ../../../src/neural_network_module/ortak_katman_module/feed_forward_network.py
[moe]: ../../../src/neural_network_module/ortak_katman_module/mixture_of_experts.py
[cache]: ../../../src/neural_network_module/ortak_katman_module/kv_cache.py
[quantization]: ../../../src/neural_network_module/ortak_katman_module/quantization_manager.py
[config]: ../../../model_management/config_schema.py
[manager]: ../../../model_management/model_manager.py
[checkpoint]: ../../../model_management/checkpoint_contract.py
[architecture]: ../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md
[roadmap]: ../../architecture/NEXT_DEVELOPMENT_ROADMAP.md
[training]: ../../../README-TR.md
