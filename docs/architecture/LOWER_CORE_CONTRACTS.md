# Alt çekirdek uzlaştırması

Bu kayıt V4/V7/V8 etiketlerini bağımsız mimariler olarak kabul etmez. Çalışan hat: CevahirNeuralNetwork → TransformerEncoderLayer → MultiHeadAttention ve FeedForwardNetwork / MixtureOfExperts → aynı FeedForwardNetwork uzmanlarıdır. Bu tur API veya veritabanı değiştirilmedi.

| Sınır | Bulgu | Düzeltme |
|---|---|---|
| Şema → çekirdek FFN genişliği | Şema 4D, SwiGLU çekirdeği 256 katına yuvarlanmış 8D/3 kullanıyordu | Mevcut çekirdek geometrisini koruyan ortak resolve_ffn_dim; açık boyut aynen korunur |
| Parametre tahmini | GQA için dört tam projeksiyon, SwiGLU için iki matris sayılıyordu | Gerçek KV genişliği, üç gated matris ve MoE router terimi hesaba katılır; hâlâ tahmindir |
| Katman → MoE → uzman FFN | ffn_use_bias yalnız yoğun FFN yolunda etkiliydi | MoE use_bias parametresi tüm mevcut uzmanlara aktarılır; varsayılan False değişmez |
| Paralel flag → post-norm | Kullanılan norm2 yanlışlıkla donduruluyordu | Yalnız parallel_residual ve pre_norm birlikte etkinse norm2 dondurulur |

Yeni testler farklı boyutlarda yoğun/MoE ağırlık şekillerini, açık boyut korunmasını, uzman bias gradyanlarını, GQA tahmin farkını ve iki normalizasyon yolunun gradyanlarını doğrular. Kanıt: benchmarks/results/lower_core_verification.xml.

İlk incelemede açık kalan checkpoint seçimi ve seq_proj_dim doğrulaması aşağıdaki çalışmalarla kısmen uzlaştırıldı. seq_proj_dim eski uyumluluk alanı olarak korunur; bağımsız attention yardımcılarının aktif yol ile ilişkisi ve düşük seviye seçeneklerin tam destek matrisi açık kalır. Mimari etiketleri silmek tek başına uygulama birleştirmek değildir. 20 Eylül itibarıyla güncel çalışma sırası [geliştirme planındadır](NEXT_DEVELOPMENT_ROADMAP.md).

## Dikkat yönlendirmesi düzeltmesi

Model ve TransformerEncoderLayer forward çağrılarına return_attention_weights=False eklendi. Normal çağrı artık tam dikkat matrisi istemez; SDPA etkin ve soft-cap kapalı olduğunda hızlı yol kullanılabilir. İnceleme için return_attention_weights=True verilmelidir; bu seçenek son katmanın ağırlıklarını döndürür ve eğitimde katman entropilerini hesaplar. Tuple yapıları değişmez; normal çağrıda ağırlık alanı None olur. Eski tüketiciler ağırlıkları kullanıyorsa açık isteğe geçmelidir.

Seçenek pre-norm, post-norm, paralel residual, standart ve gelişmiş checkpoint yollarından aktarılır. Küçük CPU testleri gerçek SDPA çağrılarını gözler, açık tanılama yoluyla çıktı/gradyan eşitliğini ve cached decode doğruluğunu kontrol eder. Dikkat matrisi testleri açık istek kullanacak şekilde güncellendi; benchmark tanılama geçişi de açık istek yapar ve eski kaynak karşılaştırmasını destekler.

Doğrulama: attention_routing_verification.xml içinde 168 test geçti; tüm evolution regresyonları core_routing_regression.xml içinde 136 test geçti. Sonradan eklenen üç gelişmiş checkpoint senaryosuyla alt katman dosyasının 23 testi geçti. Bu sayılar örtüşür; toplanarak benzersiz test sayısı olarak sunulmaz. GPU performansı ölçülmedi.

## Checkpoint seçimi ve eski projeksiyon doğrulaması

Checkpoint factory bilinmeyen stratejiyi artık sessizce selective yapmaz. Pozitif katman sayısı/aralığı ve geçerli indeksler doğrulanır; açık checkpoint_layers ve checkpoint_every_n tercihleri korunur. Model, factory ve bağımsız TransformerEncoderLayer aynı seçim kuralını kullanır. Bağımsız selective katman artık [0] seçer; önceden boş liste yüzünden checkpoint hiç çalışmıyordu. Adaptive adı geriye uyumluluk için korunur: bellek ölçmez, ilk/son ve çift indeksli katmanları seçen sabit sezgiseldir.

Kaldırılmış seq_proj_dim projeksiyonunun num_heads bölünebilirlik kontrolü gerçek embed_dim boyutuna taşındı. Pozitif eski boyut hâlâ saklanır; farklı olduğunda weight tying kapatma davranışı mevcut checkpoint uyumluluğu için korunur. Bu alan henüz kaldırılmadı.

Testler altı katman üzerinde üç stratejinin gerçek seçimlerini, inference kapatmasını, geçersiz ayarları, açık override seçeneklerini ve bağımsız katmanın backward geçişini denetler. Kanıt: benchmarks/results/checkpoint_contract_verification.xml.
