# 4. Attention: bir konum başka konumlardan ne alır?

[Kitabın içindekileri](../README.md) · [Önceki: Sinir ağı](03-sinir-aglari.md) · [Sonraki: Transformer](05-transformer.md)

Bir tokenın tablodan alınan vektörü çevresindeki kelimeleri henüz içermez. Attention, bir konumdaki temsilin hangi diğer konumların değerlerinden ne kadar alacağını hesaplar. “Dikkat ediyor” ifadesinin bu kodda karşılığı bir bilinç durumu değil, ağırlıklı vektör toplamıdır. Temel yapı [Transformer makalesindeki](https://arxiv.org/abs/1706.03762) ölçeklenmiş nokta çarpımı hesabıdır; Cevahir bunu farklı head düzenleri, maskeler ve yürütme yollarıyla uygular.

## Q, K ve V aynı girdiden neden üç kez üretilir?

Bir konumun **query** vektörü aradığı ilişkileri, bir aday konumun **key** vektörü eşleşme skorunu, **value** vektörü ise toplanacak içeriği temsil eder. Bunlar elle yazılmış anlam etiketleri değildir. Self-attention'da aynı girdinin üç ayrı öğrenilebilir doğrusal dönüşümüdür:

`Q = X W_Qᵀ`, `K = X W_Kᵀ`, `V = X W_Vᵀ`.

Bu ayrım sayesinde hangi konumun seçileceğini belirleyen özelliklerle taşınacak içeriğin özellikleri aynı olmak zorunda kalmaz. Cevahir'de [MultiHeadAttention.__init__](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L91) `query_proj`, `key_proj`, `value_proj`, `out_proj` adlı, bias içermeyen dört `nn.Linear` kurar. [TransformerEncoderLayer._forward_impl](../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py#L532) attention'a aynı temsili üç kez verir; çağrı bu nedenle self-attention'dır. Genel attention arayüzünün ayrı `key` ve `value` kabul etmesi, ana modelin encoder–decoder cross-attention kullandığını göstermez.

Bir head için skor ve çıktı:

`S = QKᵀ / (sqrt(d) τ) + M`, `A = softmax(S)`, `O = AV`.

`d` head genişliği, `τ` attention sıcaklığı, `M` maskedir. Nokta çarpımı benzerliğe dayalı skoru üretir; `sqrt(d)` ölçekleme boyut arttıkça skor büyüklüğünü dengeler. Softmax her query için key ekseni üzerinde normalleştirir. `O_i = Σ_j A_ij V_j`, i'nci konumun aldığı içeriktir. Cevahir'in normal `forward` çağrısı `τ=1` ile çalışır. Metin üretimindeki token örnekleme sıcaklığı, bu iç attention sıcaklığından ayrı bir aşamadır.

## Şekilleri görmeden head sayısını anlamak zordur

`H` query head sayısı, `H_kv` key/value head sayısı ve `d=D/H` olsun. [MultiHeadAttention.forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L678) projeksiyonları yeniden şekillendirip eksen değiştirir:

| Tensör | Şekil | Anlamı |
| --- | --- | --- |
| Girdi query | `[B,T,D]` | Bu çağrıda hesaplanacak konumlar. |
| Q | `[B,H,T,d]` | Her query head için sorgular. |
| İlk K ve V | `[B,H_kv,S,d]` | Erişilebilen kaynak konumları; cache ile `S>T` olabilir. |
| Attention skorları | `[B,H,T,S]` | Her query–key konum çifti için skor. |
| Head çıktıları | `[B,H,T,d]` | Value vektörlerinin ağırlıklı toplamı. |
| Birleştirme ve `out_proj` | `[B,T,D]` | Transformer residual dalına geri verilecek temsil. |

`D`, `H` ile; `H` de `H_kv` ile tam bölünmelidir. [Kurucu doğrulaması](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L161) bunları denetler. `H_kv=H` standart MHA, `H_kv=1` MQA, aradaki değerler GQA'dır. Bu adlandırma [GQA çalışmasıyla](https://arxiv.org/abs/2305.13245) örtüşür. Varsayılan `num_kv_heads=None`, `H_kv=H` anlamına gelir; GQA destekleniyor olması varsayılanın GQA olduğu anlamına gelmez.

Örneğin `D=32`, `H=4`, `H_kv=2` seçildiğinde `d=8`, K/V projeksiyon genişliği `16` olur. Kod hesap sırasında `repeat_interleave` ile iki KV head'i dört query head'ine eşler. Cache ise genişletilmeden, iki head biçiminde saklanır. Bu koşullarda K/V depolaması MHA'nın yarısıdır; bütün model belleğinin veya toplam çalışma süresinin yarıya indiği sonucu çıkmaz. Bu uygulama her backend'e yerel GQA yürütmesini devretmez; genişletilmiş çalışma tensörleri ayrıca dikkate alınmalıdır.

## Maske, modelin hangi bilgiye erişebileceğidir

Sonraki tokenı tahmin eden model eğitim sırasında gelecekteki tokenı okuyabilseydi, gerçekte üretim anında bulunmayacak bilgiden yararlanırdı. Nedensel maske, key konumu query konumundan büyükse o bağlantıyı engeller. Cevahir bunu cache varken mutlak `query_positions` ve `key_positions` üzerinden kurar; yalnızca yeni parçanın yerel indislerini kullanmak yeterli olmaz.

[_prepare_attention_mask](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L372) arayüzünde boolean `True` **engelle** demektir. Sonlu, 0–1 aralığındaki ve en az bir pozitif değer içeren sayısal maskede `>0.5` geçiştir; diğer değerler engellenir. Tamamı sıfır sayısal maske bu özel dala girmez ve sıfır additive maske, yani ek engel yok, olarak yorumlanır. Genel additive maskede `0` skoru değiştirmez, `−∞` bağlantıyı engeller.

PyTorch'un doğrudan SDPA boolean maskesi ise `True=katıl` semantiği kullanır. Cevahir'in kendi maskesini additive biçime çevirmesi bu tersliği giderir; iki arayüz arasında boolean maskeyi doğrudan taşımak yanlıştır. [PyTorch SDPA belgesi](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) bu ayrımı açıkça tanımlar.

İki boyutlu `[B,S]` padding maskesi ile `[T,S]` attention maskesi, `B=T` olduğunda biçimden ayırt edilemez. Böyle bir padding maskesini `[B,1,1,S]` biçiminde vermek gerekir. `sliding_window=w` ayrıca `query_position−key_position >= w` bağlantılarını engeller. Bu koşul tek başına geleceği engellemez; geçmişe dönük yerel attention için causal maske de açık olmalıdır. Kodda yoğun maske oluşturulabildiği için pencere ayarından otomatik `O(Tw)` uygulama maliyeti vaat edilemez.

Tamamen kapalı bir satırda normal softmax ifadesi tanımsız hale gelebilir. [_standard_sdpa_forward](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L613), bu satırların attention ağırlıklarını sıfırlar. Diğer satırlar dropout kapalıyken toplamı yaklaşık bir olan dağılımlardır. Eğitimde attention dropout uygulandığında tek örnekte toplamın tam bir olması beklenmez.

## Aynı hesap, üç yürütme yolu

[scaled_dot_product_attention](../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py#L454) hangi hesabın çağrılacağını seçer:

1. `return_attention_weights=True` veya `attn_logit_cap>0` ise manuel yol kullanılır. Soft-cap, skorların üzerine maske eklenmeden önce uygulanır.
2. Aksi halde `use_pytorch_sdpa=True` ise `F.scaled_dot_product_attention` çağrılır. PyTorch uygun backend'i seçer; bu bayrak belirli bir GPU kernelinin çalıştığını kanıtlamaz.
3. PyTorch yolu kapalıyken kullanılabilir harici Flash Attention istenmişse `flash_attn_func` denenir. Float32 veya özel maske varsa ve harici çağrı hata verirse manuel yola dönülür. Diğer durumlarda zaten manuel yol seçilir.

Manuel yolun temel hesabı kaynakta şu satırlarla görülür; arada maskeleme ve kararlılık işlemleri bulunduğu için bu üç satır tek başına tam fonksiyon değildir:

```python
scores = torch.matmul(work_query, work_key.transpose(-2, -1)) / scale
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights.to(value.dtype), value)
```

PyTorch yolu attention matrisini döndürmez. Teşhis için matris istenmesi bu yüzden yalnız ek bir çıktı bayrağı değildir; yürütme yolunu ve bellek maliyetini değiştirir. `_pytorch_sdpa_forward`, değerlendirme modunda `dropout_p=0` geçirir. Bu ayrıntı gereklidir çünkü SDPA fonksiyonunun dropout davranışı kendisine verilen olasılığa bağlıdır; çevre modülün `eval()` durumunu kendiliğinden okumaz.

Attention'ın sayısal çıktısı `out_proj` sonucudur; residual toplama ve katman normalizasyonu burada yapılmaz. Varsayılan `forward` bu tensörü tek başına, attention ağırlıkları istenirse `(output, weights)` çiftini, etkin cache yolunda `(output, weights_or_none, cache)` üçlüsünü döndürür. Sonraki tüketici olan `TransformerEncoderLayer` sonucu açar ve residual/normalizasyon işlerini üstlenir. Sınıfta eski checkpoint uyumluluğu için `norm` bulunsa da mevcut `forward` onu kullanmaz.

## Hangi iddia hangi testle sınanır?

[test_cached_matches_full](../../../tests/evolution/test_attention_cache.py#L34), küçük CPU modelinde MHA/MQA/GQA, iki SDPA yolu ve pencere seçenekleri için parçalı cache hesabını tam diziyle karşılaştırır. [test_causal_prefix_and_masked_rows](../../../tests/evolution/test_attention_cache.py#L89) nedenselliği ve kapalı satırları; [test_softcap_effective_with_default_backend](../../../tests/evolution/test_attention_cache.py#L103) soft-cap'in etkinliğini sınar. [Harici Flash başarısızlığı testi](../../../tests/evolution/test_attention_cache.py#L167) taklit edilen hata sonrası fallback'in nedenselliğini denetler; gerçek GPU Flash kerneline ait hız veya kalite ölçümü değildir.

Bir okuma deneyi olarak üç token için causal izin matrisini çizin. Sonra ikinci tokenın sorgusunun hangi value vektörlerini toplayabildiğini kaynak maskesiyle karşılaştırın. Aynı deneyi cache'e beş token eklenmişken tek yeni token için tekrarlayın: query'nin yerel indisi sıfır olsa da mutlak konumu beştir.

[Mevcut modül belgesi](../../modules/neural_network/README.md) bu bölümün genel haritasını sağlar. [Eski attention incelemesi](../../module_audits/4_attention_audit.md) korunmuştur; ancak oradaki sıcaklık çarpanı ve her attention satırının toplamının bir olduğu yönündeki genellemeler güncel uygulamaya taşınmamıştır. Burada davranışın otoritesi bağlantısı verilen çalışır kaynak ve somut test koşullarıdır.
