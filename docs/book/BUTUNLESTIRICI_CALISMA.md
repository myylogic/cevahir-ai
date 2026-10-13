# Bütünleştirici çalışma: bir iddiayı metinden kanıta izlemek

[İçindekiler](README.md) · [Terimler](TERIMLER.md) · [Yeniden üretim ve atıf](AKADEMIK_OKUMA.md)

Bu kapanış çalışması on iki bölümü tek bir inceleme yönteminde birleştirir. Amaç sıfırdan ikinci bir oyuncak Cevahir yazmak değildir. Mevcut sistemde “veri nereden geldi, hangi dönüşüme uğradı, hangi durumu değiştirdi, neyi gerçekten ölçtük?” sorularını kesintisiz izlemektir. Aşağıdaki komutlar proje kökünde ve kurulu proje bağımlılıklarıyla çalışır. Bazı örnekler gerçek sinir çekirdeği, bazıları belirli sözleşmeyi izole eden açıkça tanımlanmış fixture kullanır; bunlar tek bir eğitilmiş model deneyi diye birleştirilmez.

## 1. Metni ve kimliği sabitlemek

`Merhaba dünya!` girdisinden başlayın. [Tokenizer eşlikçisini](../../scripts/book_tokenizer_walkthrough.py) çalıştırın:

```powershell
python scripts/book_tokenizer_walkthrough.py
```

İnceleme notunda özgün metin, kullanılan mode/bayraklar, ara parçalar, gerçek token ID'leri ve decode sonucunu tutun. Gözlenen ID'yi tahmin etmeyin; [kayıttan](evidence/tokenizer_walkthrough.json) okuyun. `tokens` ile `token_ids` uzunluklarının aynı olmasının gerekmediğini açıklayın. Normalize edilmiş metnin başlangıç metninden farkı varsa `UNK=0` sonucuyla çelişmediğini gösterin. Sözlük ve merges dosyalarının değişmemesi, bu incelemenin tokenizer eğitimi olmadığını doğrular.

**Beklenen sonuç:** Okur, bir kimliğin embedding satırına nasıl karşılık geldiğini anlatabilir. Henüz bu satırın ne öğrendiği veya modelin Türkçe kalitesi hakkında sonuç çıkarmaz.

## 2. Aynı girdinin sinirsel yolunu izlemek

[Gerçek çekirdek örneklerini](../../scripts/book_neural_walkthrough.py) çalıştırın:

```powershell
python scripts/book_neural_walkthrough.py
```

Bu eşlikçinin küçük modeli kendi 41 kimliklik öğretim sözlüğünü kullanır; önceki adımın 60 binlik gerçek vocabulary ID'lerini bu modele doğrudan vermeyin. Bu ayrım pedagojik bir sınırdır. `B×T` girdinin embedding ile `B×T×D`, çıkışta `B×T×V` olmasını izleyin. Aynı tokenın başlangıç vektörü eşitken farklı öneklerdeki son temsil/çıktının farklı olmasını açıklayın. Query/key/value ve maskeyi [4. bölüme](tr/04-attention.md), residual/norm/FFN hesabını [5. bölüme](tr/05-transformer.md) bağlayın.

**Beklenen sonuç:** İleri hesap ile öğrenilebilir parametreyi ayırabilirsiniz. Modelin yalnız forward çağrısıyla ağırlıklarını değiştirmediğini; backward'ın gradyan, optimizer'ın parametre güncellemesi yaptığını kayıt üzerinden gösterebilirsiniz.

## 3. Eğitim iddiasını doğru hedefle kurmak

[Eğitim bölümündeki](tr/06-egitim.md) hazırlanmış veri akışında giriş ve hedefi yan yana yazın. BOS/hedef/EOS, padding ve `-100` maskesinin kaybı hangi konumlarda değiştirdiğini gösterin. İki minibatch için ayrı ortalamaları körlemesine eşitlemek yerine geçerli token sayılarını kullanın.

Küçük çekirdek örneğinin loss azalması yalnız aynı öğretim batch'i üzerindedir. Genelleme iddiası için ayrı kaynaklardan gelen değerlendirme verisi gerekir. Sonucu şu cümlelerle ayırın: “Güncelleme yolu çalıştı”; “eğitim hedefi azaldı”; “ayrılmış veride iyileşme görüldü”; “çalışma sırasında kalıcı yeni beceri edinildi.” Bunlar birbirinin yerine geçen dört ifade değildir. Veri/deney olmadan son ikisini yazmayın.

**Beklenen sonuç:** Ölçümün birimi, paydası, kaynak ayrımı ve iddia sınırı açık olur. Hazır bir eğitim kaydını değiştirip geçmiş araştırmanın sonucuymuş gibi sunmazsınız.

## 4. Aynı ağırlık ile aynı devamı karşılaştırmak

```powershell
python scripts/book_lifecycle_walkthrough.py
```

[Yaşam döngüsü eşlikçisi](../../scripts/book_lifecycle_walkthrough.py), gerçek ModelManager ile geçici dosyalarda kayıt/yükleme yapar. Ağırlık ve optimizer geçmişini geri getiren yol ile yalnız ağırlığı geri getiren yolu karşılaştırın. Aynı sıradaki batch'te parametre farkının neden sıfır veya sıfırdan farklı olduğunu [7. bölümdeki](tr/07-model-yasam-dongusu.md) moment denklemlerine bağlayın. Tokenizer eşleşmezken doğru tensor şeklinin neden yetmediğini açıklayın.

**Beklenen sonuç:** Dosyanın okunabilmesini model kimliği doğrulamasıyla; checksum eşleşmesini yazar doğrulamasıyla; model yüklemeyi bütün sistemin yeniden açılmasıyla karıştırmazsınız. Örnekler kalıcı eğitimli checkpoint'lere yazmaz.

## 5. Üretim ve araç sonucunu ayrı gözlemek

```powershell
python scripts/book_runtime_walkthrough.py
```

[Bu eşlikçide](../../scripts/book_runtime_walkthrough.py) logit kaynağı sabittir; gerçek üretim adaptörünün filtrelerini görmemizi sağlar. `(0.5,0.3,0.15,0.05)` dağılımında yalnız top-p ile top-k ardından top-p sonuçlarını elle hesaplayın. Cache açık/kapalı çağrı uzunluklarını okuyun. Sabit logit deneyinin gerçek attention eşdeğerliği kanıtı olmadığını belirtin; o ayrı neural örnekte sınanır.

Sonra calculator örneğini izleyin. Güncel çıkarılan ve açık `2+3*4` parametreleri `"14"` üretir. [Düzeltme öncesi kayıt](evidence/runtime_walkthrough_before_tool_fix.json), ilk yolun `"5"` döndürdüğünü saklar. `success_count` iki sürümde de fonksiyonun istisnasız dönüşünü sayabilir. Kullanıcı niyetinin korunması, doğru araç yürütme ve üreticinin sonucu doğru taşıması ayrı kontrollerdir.

**Beklenen sonuç:** Çıktı hatasını yalnız “model kötü” diye etiketlemek yerine parametre çıkarıcı, araç, prompt ve token seçimi arasındaki somut sınıra yerleştirebilirsiniz.

## 6. Hizmet ile model arasındaki sınırı görmek

```powershell
python scripts/book_system_walkthrough.py
```

[Sistem eşlikçisi](../../scripts/book_system_walkthrough.py), geçici SQLite ve kontrollü backend ile gerçek sohbet/kayıt akışını inceler. Eğitilmiş modelin cevabını ölçmez. [10. bölümle](tr/10-uctan-uca-sistem.md) birlikte hangi geçmişin backend'e gittiğini, hangi mesajın ne zaman kaydedildiğini ve hata enjeksiyonunda ne kaldığını okuyun. Dönüşün hesaplanması, veritabanına yazılması ve istemciye teslim edilmesi ayrı olaylardır.

**Beklenen sonuç:** Bir HTTP hata kodundan modelin hiç çalışmadığını çıkarmazsınız. Bir cache hit'i yeni deneyim veya yeni sinirsel hesap diye saymazsınız. Fixture'ın göstermediği kimlik doğrulama, ağ veya GPU özelliklerini sonuçlara eklemezsiniz.

## 7. Öğrenme durumuna ilişkin iddiayı yanlışlamaya çalışmak

```powershell
python scripts/book_research_walkthrough.py
```

[Sonlu araştırma eşlikçisi](../../scripts/book_research_walkthrough.py) korunmuş iki araştırma uygulamasını kullanır. Aynı bugünkü aday kümesinden farklı düzeltme sonucu çıkmasını açıklayın. Sonra `85`, `126`, `1296` durum sınıfının hangi gelecek işlem haklarına karşılık geldiğini yazın. Aynı hata sorguları için yeterli özetin, gözlem sayımı veya kimlikli düzeltme için neden yetmeyebildiğini elle gösterin.

**Beklenen sonuç:** Küçük bir tam sayımın gücünü ve sınırını birlikte söyleyebilirsiniz: tanımlı sonlu sözleşmedeki bütün durumlar sınanmıştır; genel living-learning problemi çözülmemiştir. Deney sonucunu saklamak veya yöntemin adını değiştirmek yeni bir bilimsel ilke oluşturmaz.

## İnceleme çıktısının biçimi

Kendi çalışmanız için aşağıdaki alanları doldurmanız yeterlidir. Bu bir tamamlanmış deney sonucu değil, okurun dolduracağı rapor şablonudur:

| Alan | Yazılacak bilgi |
|---|---|
| İddia | Tek cümlede, hangi davranışın değiştiği veya korunduğu. |
| Kaynak | İncelenen dosya/metot, kod sürümü ve veri kimliği. |
| Girdi | Gerçek veri, elle atanmış öğretim sayısı veya fixture ayrımı. |
| Karşılaştırma | Aynı bilgi/bütçe altında güçlü alternatif veya bağımsız hesap. |
| Sonuç | Paydası ve birimiyle sayısal değer; hata veya negatif de dahil. |
| Sınır | Hangi ortam, görev, kaynak veya gelecekteki işlem için geçerli olmadığı. |
| Tekrar üretim | Komut, ortam, tolerans ve kaydın değişip değişmediği. |

Kitap boyunca bir arada tutulan akademik alışkanlık budur: kavramı gerçek hesaba bağlamak, hesabı çalıştırmak, sonucu kapsamıyla yazmak ve onu yanlışlayabilecek gözlemi saklamamak.
