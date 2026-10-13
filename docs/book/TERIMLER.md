# Terimler ve matematiksel gösterimler

[İçindekiler](README.md) · [Bütünleştirici çalışma](BUTUNLESTIRICI_CALISMA.md) · [Akademik okuma](AKADEMIK_OKUMA.md)

Bu dizin bölümlerin yerine geçmez. Kısa tanımın yanındaki bağlantı, kavramın matematiğine, Cevahir'deki dosya/metot karşılığına ve kanıt kapsamına götürür. Aynı sözcüğün farklı katmanlardaki anlamları özellikle ayrılmıştır.

## Dil, veri ve sayısal temsil

| Terim | Bu kitaptaki anlamı | Ayrıntı |
|---|---|---|
| Yapay zekâ motoru | Model, veri dönüşümü, üretim, saklama ve hizmet akışlarını bağlayan sistem. | [1](tr/01-motor-ve-ogrenme.md) |
| Dil modeli | Token dizisine veya sonraki tokena koşullu olasılık atayan model. | [1](tr/01-motor-ve-ogrenme.md) |
| Token | Kelime olmak zorunda olmayan ayrık metin birimi. | [2](tr/02-metin-tokenizer.md) |
| Token ID | Vocabulary içindeki tamsayı adresi; anlam büyüklüğü değildir. | [2](tr/02-metin-tokenizer.md) |
| Vocabulary / sözlük | Token ile kimlik arasındaki eşleme. | [2](tr/02-metin-tokenizer.md) |
| Tokenizer | Metni token kimliklerine çeviren ve decode eden işlemler bütünü. | [2](tr/02-metin-tokenizer.md) |
| BPE | Öğrenilmiş sıralı komşu birleştirmeleri kullanan alt parça yaklaşımı. | [2](tr/02-metin-tokenizer.md) |
| Normalizasyon, metin | Unicode, harf veya boşluk gibi metin özelliklerini ortak biçime getirme. | [2](tr/02-metin-tokenizer.md) |
| BOS / EOS / PAD / UNK | Başlangıç, bitiş, dolgu ve bilinmeyen birim işaretleri. | [2](tr/02-metin-tokenizer.md), [6](tr/06-egitim.md) |
| Batch | Bir hesapta birlikte işlenen örnek grubu. | [1](tr/01-motor-ve-ogrenme.md), [6](tr/06-egitim.md) |
| Tensor / tensör | Şekli ve sayı türü belirli çok boyutlu sayısal dizi. | [1](tr/01-motor-ve-ogrenme.md) |
| Embedding | Ayrık kimliğin öğrenilmiş vektör temsili; lookup bir satır seçer. | [3](tr/03-sinir-aglari.md) |
| Sentence embedding | Değişken uzunluktaki metnin retrieval için karşılaştırılan vektörü. | [9](tr/09-bellek-bilis-araclar.md) |
| Logit | Softmax öncesindeki gerçek sayı puanı. | [3](tr/03-sinir-aglari.md), [8](tr/08-uretim.md) |
| Softmax | Bir puan vektörünü normalize edilmiş pozitif dağılıma dönüştürme. | [1](tr/01-motor-ve-ogrenme.md) |

## Sinirsel hesap ve mimari

| Terim | Bu kitaptaki anlamı | Ayrıntı |
|---|---|---|
| Parametre / weight | Modelin öğrenilebilir sayılarını taşıyan tensör. | [3](tr/03-sinir-aglari.md) |
| Katman | Belirli girdi/çıktı sözleşmesi olan hesap dönüşümü. | [3](tr/03-sinir-aglari.md) |
| Aktivasyon | Katman içindeki doğrusal olmayan fonksiyon; ayrıca ara çıktı anlamında kullanılır. | [3](tr/03-sinir-aglari.md) |
| Attention | Query-key uyumlarına göre value katkılarını birleştiren hesap. | [4](tr/04-attention.md) |
| Self-attention | Query/key/value girdilerinin aynı diziden üretildiği attention. | [4](tr/04-attention.md) |
| Causal mask | Bir konumun sonraki konumları kullanmasını engelleyen maske. | [4](tr/04-attention.md) |
| MHA / MQA / GQA | Query ve KV başlıklarının paylaşım düzenleri. | [4](tr/04-attention.md) |
| SDPA | Ölçekli dot-product attention işlemi/API'si; seçilen kernel ayrıca belirlenir. | [4](tr/04-attention.md) |
| FlashAttention | Attention'ın bellek erişimi ve ara tensör maliyetlerini düzenleyen yöntem ailesi. | [4](tr/04-attention.md) |
| Transformer | Attention, konum, residual, norm ve feed-forward hesaplarını birleştiren mimari ailesi. | [5](tr/05-transformer.md) |
| Residual | Bir dalın çıktısını girişine ekleyen bağlantı. | [5](tr/05-transformer.md) |
| LayerNorm / RMSNorm | Katman içi sayısal ölçeği düzenleyen farklı normalizasyonlar. | [5](tr/05-transformer.md) |
| RoPE | Konuma bağlı dönmelerle query/key hesabına konumsal ilişki katma. | [5](tr/05-transformer.md) |
| YaRN / dinamik ölçekleme | Uzun bağlam için konum ölçekleme seçenekleri; yerel uygulanma durumları ayrı. | [5](tr/05-transformer.md) |
| FFN | Token konumlarındaki temsilleri dönüştüren feed-forward alt katman. | [5](tr/05-transformer.md) |
| SwiGLU | SiLU kapısı ile başka bir projeksiyonu çarpan gated FFN düzeni. | [5](tr/05-transformer.md) |
| MoE | Router ile seçilen uzman alt ağların katkılarını kullanan hesap. | [5](tr/05-transformer.md) |
| Weight tying | İki hesabın aynı parametre nesnesini paylaşması. | [3](tr/03-sinir-aglari.md) |
| Dropout / stochastic depth | Eğitimde eleman veya residual dal düzeyinde rastgele düşürme. | [5](tr/05-transformer.md) |

## Eğitim ve kalıcı kayıt

| Terim | Bu kitaptaki anlamı | Ayrıntı |
|---|---|---|
| Dataset / veri kümesi | Kaynakları ve dönüşümleri tanımlı örnekler topluluğu. | [6](tr/06-egitim.md) |
| Prepared cache | Hazırlanmış token/hedef örneklerinin kimlikli kaydı; KV cache değildir. | [6](tr/06-egitim.md) |
| Loss / kayıp | Eğitimin azaltmaya çalıştığı sayısal amaç. | [6](tr/06-egitim.md) |
| Gradient / gradyan | Kaybın parametrelere göre yerel türevleri. | [3](tr/03-sinir-aglari.md), [6](tr/06-egitim.md) |
| Backpropagation | Zincir kuralıyla türevleri hesaplama; tek başına ağırlık güncellemez. | [3](tr/03-sinir-aglari.md) |
| Optimizer | Gradyan ve varsa geçmiş durumuyla parametreyi değiştiren yöntem. | [6](tr/06-egitim.md) |
| SGD / Adam / AdamW | Farklı güncelleme ve geçmiş durum düzenleri. | [6](tr/06-egitim.md) |
| Gradient accumulation | Birden fazla küçük batch'in türevlerini tek güncelleme için biriktirme. | [6](tr/06-egitim.md) |
| Mixed precision | Hesapta farklı sayı biçimlerini birlikte kullanma. | [6](tr/06-egitim.md) |
| Validation | Güncelleme yapılmadan ayrılmış veride ölçme; test setiyle rolü aynı değildir. | [6](tr/06-egitim.md) |
| Perplexity | Saf ortalama negatif log olasılığın üssü; değişmiş loss'ta yorum değişebilir. | [6](tr/06-egitim.md) |
| Fine-tuning / SFT | Eldeki ağırlıkları yeni amaç/örnek davranışlarla uyarlama. | [6](tr/06-egitim.md) |
| LoRA | Dondurulmuş ağırlığa düşük ranklı öğrenilebilir katkı. | [6](tr/06-egitim.md) |
| RLHF / DPO | Tercih geri bildirimiyle model eğitmenin farklı düzenleri. | [6](tr/06-egitim.md) |
| Distillation | Öğretmen çıktılarından öğrenciye davranış aktaran eğitim. | [6](tr/06-egitim.md) |
| Quantization | Sayıları daha az bit/daha sınırlı değerlerle temsil etme; distillation değildir. | [6](tr/06-egitim.md), [7](tr/07-model-yasam-dongusu.md) |
| Checkpoint | Gelecekteki bir çalışmayı sürdürmek için saklanan model/eğitim durumu. | [7](tr/07-model-yasam-dongusu.md) |
| Activation checkpointing | Backward için ara değerleri yeniden hesaplayarak bellek kullanma yöntemi. | [5](tr/05-transformer.md) |
| State dict / buffer | Parametre ve belirli buffer kaydı; bütün Python/dış durum değildir. | [7](tr/07-model-yasam-dongusu.md) |
| Resume | Kaydedilmiş durumdan devam; aynı eğitim yörüngesi daha güçlü koşul ister. | [7](tr/07-model-yasam-dongusu.md) |
| Checksum / fingerprint | Belirli içerik sözleşmesinin özeti; yazarlık veya bilimsel doğruluk onayı değildir. | [7](tr/07-model-yasam-dongusu.md) |

## Üretim, hizmet ve araştırma

| Terim | Bu kitaptaki anlamı | Ayrıntı |
|---|---|---|
| Inference / çıkarım | Mevcut durumla çıktı veya eylem hesaplama. | [8](tr/08-uretim.md) |
| Autoregressive generation | Seçilen çıktıyı sonraki adıma girdi olarak ekleyen üretim. | [8](tr/08-uretim.md) |
| Greedy / sampling | En yüksek puanı seçme / dağılımdan örnek çekme. | [8](tr/08-uretim.md) |
| Temperature / top-k / top-p | Üretim dağılımının yoğunluğunu ve aday kümesini değiştiren seçimler. | [8](tr/08-uretim.md) |
| Beam search | Birden fazla öneki skorlayıp sınırlı sayıda devamı tutan arama. | [8](tr/08-uretim.md) |
| KV cache | Attention key/value ara hesapları; öğrenilmiş ağırlık veya konuşma arşivi değildir. | [8](tr/08-uretim.md) |
| Prefill / decode | Promptu ilk işleme / sonraki yeni token adımları. | [8](tr/08-uretim.md) |
| Sliding window / sink | Tutulan attention geçmişinin pencere ve başlangıç konumları. | [8](tr/08-uretim.md) |
| Context / memory / retrieval | Çağrıda görülen içerik / sonraya tutulan kayıt / ilgili kaydı seçme. | [9](tr/09-bellek-bilis-araclar.md) |
| RAG | Getirilen dış kayıtların üretime katılması; özgün modelin eğitim formülü ayrıca anlatılır. | [9](tr/09-bellek-bilis-araclar.md) |
| Tool use | Kayıtlı fonksiyonun gerçek argümanlarla yürütülüp sonucunun kullanılması. | [9](tr/09-bellek-bilis-araclar.md) |
| Cognitive processing | Cevahir'in strateji, bağlam, üretim ve denetim yazılım akışı. | [9](tr/09-bellek-bilis-araclar.md) |
| Scope | Kaydın hangi kullanıcı/oturum alanında görünür olduğu. | [9](tr/09-bellek-bilis-araclar.md), [10](tr/10-uctan-uca-sistem.md) |
| Transaction / idempotency | Birlikte tamamlanan kayıt sınırı / tekrarın ek etki oluşturmama özelliği. | [10](tr/10-uctan-uca-sistem.md) |
| Acquisition / edinim | Deneyim nedeniyle kullanılabilir ilişki veya beceri kazanma. | [11](tr/11-arastirma-laboratuvari.md) |
| Consolidation / konsolidasyon | Edinilen içeriği koruma, birleştirme veya daha kolay yürütme. | [11](tr/11-arastirma-laboratuvari.md) |
| Transfer / interference | Önceki edinimin başka görevde yardımı / bir değişimin başka beceriye olumsuz etkisi. | [11](tr/11-arastirma-laboratuvari.md) |
| Credit assignment | Sonucun hangi olay/eylem/güncellemeye bağlanacağını belirleme. | [11](tr/11-arastirma-laboratuvari.md) |
| Plasticity / plastisite | Yeni deneyimle yeni davranış edinebilme kapasitesi. | [12](tr/12-acik-sorular.md) |
| Öğrenme durumu | Gelecekteki cevap ve güncellemeleri mümkün kılan taşınmış bilgi. | [11](tr/11-arastirma-laboratuvari.md), [12](tr/12-acik-sorular.md) |
| Yaşarken öğrenme | Deneyimle gelecekteki kapasitenin kalıcı, kontrollü ve genellenebilir değişimi üzerine açık araştırma sorusu. | [12](tr/12-acik-sorular.md) |

## Gösterim anahtarı

| Gösterim | Anlamı ve dikkat edilecek bağlam |
|---|---|
| `B,T,D,V` | Batch, dizi uzunluğu, temsil genişliği, vocabulary büyüklüğü. Araştırmadaki bütçe `B` farklı bağlamda kullanılır. |
| `L,H,H_kv,d_h` | Katman sayısı, query başlığı, KV başlığı, başlık genişliği. Loss için kullanılan `L(θ)` bir fonksiyondur. |
| `θ,η,g` | Parametreler, öğrenme oranı, gradyan. |
| `Q,K,V` | Attention query/key/value tensörleri. Sözlük büyüklüğü `V` ile aynı kavram değildir. |
| `p,z,τ` | Olasılık, logit, sıcaklık. |
| `E[x]` | Embedding tablosundan satır seçimi. Araştırma kısmında `E` bir deneyim kümesini gösterebilir. |
| `s_t,o_t,a_t,e_t` | Taşınan durum, gözlem, eylem ve erişilmiş deneyim. |
| `N,d_x,c_h` | Sonlu araştırmada kayıt sayısı, etiket sayım farkı, hipotez ihlal sayısı. |
| `≈` / `=` | Yaklaşık eşitlik / belirtilen matematiksel sözleşmede eşitlik. Float toleransı ayrıca kaydedilir. |
| `O(·)` | Büyüme mertebesi; ölçülmüş saniye veya aygıt belleği değildir. |

Bir simge bölüm içinde yeniden tanımlanabilir. Denklemi okumadan önce hangi nesneyi, boyutu ve birimi temsil ettiğini kontrol etmek kitabın bütün sayısal örnekleri için geçerlidir.
