# Deneyimle koşullanan hesaplama: kaynaklı geri bildirim ve yürütme sınırları

**Muhammed Yasin Yılmaz**

CEV-2026-01 · Sürüm 2026.09.21.3 · Yayın tarihi 2026-09-21

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

Çalışma sırasında edinilen dış geri bildirimin daha sonraki hesaplama stratejisini değiştirmesi için kaynaklı deneyim, ortak istek bütçesi ve strateji tercihi uygulanmıştır. Kalıcı deneyim istatistikleri ağırlık eğitiminden ayrılır. Sentetik tablo replay'i mekanizmayı sınar; gerçek model kalitesi veya maliyet kazancı gösterilmiş değildir. Görev, model ve tokenizer kimlikleri ile geri bildirim düzeltmeleri deneyim karşılaştırmasının sınırını belirler.

## Abstract

We implement provenance-aware feedback, a shared request budget, and experience-conditioned strategy selection. Persistent experience statistics are distinct from weight training. Synthetic table replay tests mechanism behavior, not real-model quality or efficiency. Model, tokenizer, task and feedback identities constrain admissible comparisons.

## Bulguların yorumu

Deneyim gelecekteki hesaplama seçimini kalıcı bir kayıt üzerinden etkileyebilir. Bu etkiyi genel öğrenme veya gerçek görev başarımı artışı olarak sunmak için kanıt yoktur.

Yönlendirme grupları sezgiseldir; denenmemiş eylemlerin sonuçları seçilmiş eylem loglarından çıkarılamaz. Sabit MoE prior'ı öğrenilmiş uzman anlamı değildir.

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md](../../research/EXPERIENCE_CONDITIONED_COMPUTE.md) olarak korunur. Bağlantı yolları bu
edisyonun konumuna uyarlanmıştır. Rapordaki çalışma tarihi ve geçmiş yayın durumu
ifadeleri tarihsel kayda aittir; bu edisyonun tarihi yukarıdadır.

Bu çalışma [yayın dizisi](../README.md) içindeki ayrı bir metindir. DOI, bütün
metinleri, teknik kitabı ve kodu içeren sürüm arşivini tanımlar; her makaleye ayrı
DOI verildiği anlamına gelmez. Atıfta çalışma kimliği ve sürüm DOI'si birlikte
kullanılmalıdır. Hesaplamalı denetimlerin kapsamı [doğrulama kaydında](../evidence/validation.json)
ve [yayın ilkelerinde](../PUBLISHING.md) açıklanır.

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, hesaplamalı deney,
literatür incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır.
Bu katkı açıklaması, DOI kaydı veya otomatik testler dış bilimsel hakemlik sayılmaz.

---

## Araştırma kaydı: Deneyimle koşullanan hesaplama — araştırma kaydı

Tarih: 20 Eylül 2026. Yerel geliştirme; eğitim, gerçek ağırlık yükleme veya geniş benchmark yapılmadı. Önceki eğitim çıktıları bu deneyin sonucu değildir ve değiştirilmedi.

Bu turun çıktısı üç bağlı mekanizmadır: ortak istek bütçesi, dış geri bildirimle değişen strateji tercihi ve MoE yönlendirmesine açık bir bağlam girdisi. Varsayılan `research.mode="off"` deneyim politikasını ve bütçeyi kapalı tutar. MoE önceliği ayrıca etkinleştirilir. Sistem talimatı, entropi, asenkron ToT, eleştiri sonucu ve padding yardımcı kaybı düzeltmeleri deneylerden bağımsız doğruluk düzeltmeleridir.

## Sistemin bütünü ve araştırma sınırı

| Aşama | Etkin karşılık | Bu turdaki bağlantı / kalan sınır |
|---|---|---|
| Veri edinme / doğrulama | `data_processing`, `dataset_subtitle`, cache kimliği | Konuşma çıktıları otomatik eğitim verisi sayılmıyor. |
| Dataset / temsil | V3 cache, kaynak/içerik gruplu ayrım, tokenizer | Türkçeye özel tokenizer altyapısı; temsil kimliği deneyim kaydında da ayrılıyor. |
| Eğitim / sinirsel hesaplama | V3 TrainingService → V2 TrainingManager → çekirdek | Geçerli giriş maskesi MoE yardımcı kaybına ulaşıyor; deneyim otomatik optimizer adımı başlatmıyor. |
| Model durumu / kayıt / yükleme | Yapılandırma, ağırlık, optimizer/RNG checkpoint sözleşmeleri | Yeni prior parametre eklemiyor; deneyim dosyası ağırlık checkpoint'inden ayrı. |
| Çıkarım / bilişsel denetim | CevahirModelAPI → CognitiveManager → pipeline | Bütün üretim, puanlama ve entropi çağrıları ortak bütçeye bağlı. |
| Bellek / yönlendirme | Tek retrieval sonucu, görev özellikleri, politika | Kaynak ID'leri korunuyor; aynı belgeler bağlama taşınıyor. Benzerlik doğruluk kanıtı değil. |
| Deneyim / consolidation | Kaynaklı observation + feedback → türetilmiş tercih istatistikleri | Düzeltme, silme, kapasite aşımı ve eskime tercihi yeniden hesaplatıyor. |
| Adaptasyon / unutma | Destekli strateji seçimi; kanıtı geri çekme | Bu adaptasyon ağırlık eğitimi değildir; kaydın silinmesi geçmiş ağırlıkları “unlearn” etmez. |
| Değerlendirme | Hafif sözleşme testleri + dondurulmuş tablo replay | Gerçek görev kalitesi, aktarım ve hesaplama kazancı henüz ölçülmedi. |

Hızlı durum istek bütçesi, aşama ve sabit prior'dır. Ara durum doğrulanmış deneyim ve türetilmiş tercihlerdir. Yavaş durum mevcut sinir ağı ağırlıklarıdır. Bu üç durumun sahipliği ve kaydı birbirinden ayrıdır.

## H1 — Doğrulanmış deneyim hesaplama seçimini iyileştirebilir mi?

**Hipotez:** Benzer görevlerde gerçekten çalıştırılmış stratejilerin dışarıdan doğrulanmış kalite ve maliyetini kullanmak, aynı bütçede sabit yönlendirmeye göre kalite/maliyet dengesini iyileştirebilir.

**Mekanizma:** `ResearchController`, görev türü, alan, kaba karmaşıklık/belirsizlik aralığı ve bellek bulunmasıyla bir karşılaştırma grubu oluşturur. Bütçe, decoding, sistem talimatı ve deney profili bu grubun kimliğine girer. `ExperienceStore` yalnız aynı kullanıcı/oturum, model ve tokenizer kimliğine ait kayıtları karşılaştırır. Hem mevcut hem aday stratejide yeterli farklı örnek gerekir. Tercih, eskimeyle ağırlıklandırılmış kalite eksi maliyet ve muhafazakâr destek cezasından türetilir. Bu ceza kalibre edilmiş güven aralığı değildir.

`off`: mevcut akış. `budget`: yalnız ortak sınırlar. `shadow`: bütçe ve deneyim kaydı; öneri raporlanır, mevcut strateji çalışır. `adaptive`: destekli öneri stratejiyi değiştirir. `exploration_rate` varsayılan sıfırdır; açıkça yükseltilirse çalıştırılabilir stratejiler arasında tekrarlanabilir rastgele örnekleme yapılır, seçim olasılığı kaydedilir. Soğuk başlangıçta kanıt yoksa mevcut strateji kalır. Shadow tek başına denenmemiş alternatifler için kanıt üretemez.

**Bütçe:** Üretim, puanlama, entropi çağrıları; toplam ayrılan çıktı token üst sınırı; işlenen giriş karakterleri. Beam sayısı üretim rezervasyonunu çarpar. Ana yanıt için bir üretim çağrısı, çıktı token payı ve giriş karakter payı korunur. Uzun final prompt bu paydan büyükse yanıt yine reddedilebilir. Token muhasebesi gerçek üretilen token veya FLOP ölçümü değildir. Hatalı/iptal edilmiş çağrının rezervasyonu geri açılmaz. Başlamış bir backend çağrısı zorla durdurulmaz; iptal sonrası yeni çağrılar engellenir.

**Dış kanıt:** `user`, `evaluator`, `tool_verified`. Bu etiketleri doğrulamak çağıran uygulamanın sorumluluğudur. Araç çalışmasının bitmesi nihai yanıtın doğrulandığı anlamına gelmez. Critic puanı dış kanıt kabul edilmez. Aynı örneğin tekrarı destek sayısını artırmaz. Yeni doğrulanmamış tekrar eski örneğin desteğini askıya alır. Aynı kaynaktan yeni feedback eski değerlendirmeyi düzeltir; farklı kaynakların güncel değerlendirmeleri muhafazakâr olarak en düşük değerle birleştirilir. Hata, bütçe sınırı ve cache kaydı olumlu destek üretmez; dışarıdan doğrulanmış olumsuz sonuçları aday tercihini engelleyebilir.

**Sınırlar:** Görev grupları kaba ve sezgiseldir. Farklı stratejilerdeki gözlemler aynı zorluk dağılımını içermeyebilir; tercih nedensel üstünlük kanıtı değildir. Semantik yenilik ölçülmüyor. Belirsizlik modeli doğruluğunu göstermez. Profildeki model/tokenizer revizyonları gerçek yüklenen dosyaları tanımlamalıdır; model veya yapılandırma değiştiğinde yeni CognitiveManager kurulmalıdır. Yapılandırmanın çalışma sırasında değiştirilmesi açık hatayla reddedilir.

## H2 — Bilişsel bağlam MoE yollarını faydalı biçimde değiştirebilir mi?

**Hipotez:** İstek boyunca sabit, sınırlı bir uzman önceliği, uygun bağlamlarda mevcut router'ın tercihlerini yararlı biçimde değiştirebilir.

**Mekanizma:** `router_logits[B,T,E] + routing_bias[B,1,E]`. Açık giriş `[B,E]` kayan nokta tensörüdür; sonlu ve `[-1,1]` aralığında olmalıdır. `None` eski yolu korur; sıfır prior küçük testte aynı çıktıyı verir. Öğrenilebilir bir üst modül tensör sağlarsa gradient yolu korunur. Bu turdaki bilişsel bağlantı, araştırmacının `routing_bias_by_domain` ile tanımladığı sabit profili seçer; deneyimden uzman anlamı veya uzman prior'ı öğrenildiği iddia edilmez.

Core → transformer → router zinciri, ModelManager ve facade generate/score üzerinden bağlıdır. Üretim ve aday değerlendirmesi aynı prior'ı kullanır. Dolu KV cache boyunca prior değiştirilemez; değişim öncesi cache temizlenmelidir. Dense modelde prior ve prior ile beam search açık hatadır. Yeni state_dict parametresi yoktur. Küçük CPU testinde tam dizi ve cache'li devam çıktıları eşleşti.

**Karıştırıcı değişken düzeltmesi:** MoE yük dengeleme kaybında padding artık geçerli token sayısına katılmaz. Ayrı `valid_token_mask[B,T]` attention maskesinin yerine geçmez. Aktif eğitim/doğrulama döngüsü girişteki PAD kimliğinden bu maskeyi üretir. Tamamen maskelenmiş yardımcı kayıp sonlu ve türevlenebilir sıfırdır. Gradient checkpoint yeniden hesaplamasında prior/mask açık argümandır.

**Riskler:** Yanlış prior uzman çökmesine veya kalite kaybına yol açabilir. Mevcut uzmanların “matematik”, “kod” gibi anlamlar taşıdığı varsayılmamalıdır. Doğrulama ve cache prior karşılaştırması ek maliyet getirir. Paylaşılan modelin farklı facade örnekleri üzerinden eşzamanlı kullanımı için ortak model sahipliği/kilit tasarımı hâlâ ayrı çalışmadır. Dinamik katman atlama uygulanmadı.

## Kullanım

```python
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_manager import CognitiveManager

cfg = CognitiveManagerConfig()
cfg.research.mode = "shadow"
cfg.research.model_revision = "actual-checkpoint-revision"
cfg.research.tokenizer_revision = "actual-tokenizer-revision"
# model_api mevcut, yüklenmiş generate/score/entropy API'sidir.
manager = CognitiveManager(model_api, cfg)
output = manager.handle(state, request)
episode = output.metadata["execution"]["experience_id"]
# Yalnız gerçek kullanıcı veya bağımsız doğrulayıcı sonucu geldiğinde:
manager.record_experience_feedback(
    state, episode, value=1.0, source="evaluator", event_id="unique-verification-id"
)
manager.save_experience(".evolution/research/experience.json")
# Aynı kimlik/profil/retention ayarlarında yeniden yükleme:
manager.load_experience(".evolution/research/experience.json")
# Bu oturumdaki belirli kaydı ve türetilmiş politika etkisini geri çek:
manager.forget_experience(state, episode)
```

Persist edilen dosya versiyonlu JSON'dur; benzersiz geçici dosyadan atomik değiştirme yapılır. Load tüm kimlik, profil, sınır ve kayıtları doğrulamadan mevcut durumu değiştirmez. Kayıtlar prompt/cevap metni saklamaz; hash'ler, scope ve kaynak kimlikleri yine bağlamsal veri olduğundan yerel deney dosyalarıdır. Varsayılan kapasite oturum başına 256 kayıt, 128 scope ve kayıt başına 32 feedback olayıdır. Yeni scope kapasitesi dolduğunda başka kullanıcı sessizce silinmez; kayıt reddi çıktı metadata'sında görünür. Araştırma modlarında yanıt cache'i bypass edilir; aynı yanıtın tekrar kullanılması yeni kanıt sayılmaz.

## Deney ve ablasyon düzeni

`benchmarks/research/experience_replay.py` gerçek model başlatmadan tüm adayların sonucu önceden verilmiş tabloları değerlendirir. Eğitim/eval kaynak grupları ayrılır, örnek kimliği tekrarları reddedilir, eval sırasında feedback eklenmez. Dört koşul: sabit politika, deneyim tercihi, karıştırılmış eğitim feedback'i, olumsuz eğitim feedback'i çıkarılmış deneyim. Çıktı ortalama kalite/maliyet, strateji dağılımı, destek yetersizliği ve eşleştirilmiş örnek farklarını içerir. Sentetik fixture yalnız mekanizmanın davranışını sınar; gerçek model kazancı değildir.

```text
python benchmarks/research/experience_replay.py --input benchmarks/research/fixtures/synthetic_routes.jsonl --output .evolution/research/synthetic-replay.json
```

Sonraki gerçek deney: bağımsız doğrulayıcısı olan, kaynakları ayrılmış görevlerde `off / budget / shadow / adaptive` karşılaştırması. Bütün adaylar aynı toplam sınırlar altında çalıştırılmalı; deneyim toplama maliyeti de ayrıca raporlanmalı. Yalnız tercih edilen stratejinin loglarından denenmemiş eylemin başarısı çıkarılmamalı. Task başarısı, tamamlanma oranı, gerçek token/FLOP veya gecikme, rezervasyon kullanımı ve yeni görevlerde bozulma ölçülmeli. Deneyim karıştırıldığında fayda sürüyorsa H1'in açıklaması desteklenmez.

MoE için `None / zero / domain-prior / shuffled-domain-prior` karşılaştırması; aynı checkpoint, token dizisi ve top-k ile logits farkı, uzman kullanım dağılımı, bağımsız görev başarısı ve maliyet ölçülmeli. Eğitimli üst bağlam projeksiyonu ayrı deneydir; held-out hedef kaybı, interference ve eski görev retention'ı birlikte ölçülmeli. Prior karıştırıldığında aynı fayda kalıyorsa H2 desteklenmez.

## Doğrulama durumu

| Parça | Durum | Kanıt / sınır |
|---|---|---|
| Ortak bütçe ve tüm bilişsel model çağrıları | IMPLEMENTED, STATICALLY VERIFIED, CHEAP-TEST VERIFIED | Sahte backend, rezervasyon, concurrency, iptal, sync/async entegrasyon |
| Deneyim, dış feedback, düzeltme, unutma, snapshot | IMPLEMENTED, STATICALLY VERIFIED, CHEAP-TEST VERIFIED | Kapsam, model kimliği, replay, eskime, retention, atomik restore |
| MoE prior ve padding yardımcı kaybı | IMPLEMENTED, STATICALLY VERIFIED, CHEAP-TEST VERIFIED | Küçük tensörler, gradient, checkpoint ve iki katman/8 genişlik/16 vocabulary çekirdeği |
| Offline tablo karşılaştırması | EXPERIMENT READY, CHEAP-TEST VERIFIED | Sentetik fixture; gerçek model çalıştırmaz |
| Gerçek kalite/maliyet kazancı ve aktarım | EMPIRICAL VALIDATION PENDING | Eğitim, büyük veri veya gerçek ağırlık deneyi çalıştırılmadı |

Bu tur önceki bakım testlerinin yerine geçtiğini veya bütün repository testlerinin geçtiğini iddia etmez. Yalnız değişen araştırma sözleşmelerine yönelik kontroller çalıştırıldı.

Ölçülen sonuç: 96 yeni hedefli test vakası ve mevcut akıştan seçilen 3 asenkron/cache regresyonu geçti. İlk birleşik koşuda 93 test ve 73 alt kontrol geçti; sonradan eklenen iç içe scope, doğrudan orchestrator ve pool retry senaryoları ayrıca doğrulandı. Tekrarlanan koşular bağımsız kanıt sayısını artırmaz. Makine tarafından okunabilir kayıt: [doğrulama özeti](../../../benchmarks/results/research_contract_verification_summary.json).

## İnceleme sonucunda açık kalan ayrımlar

**Doğruluk borçları:** Hesap makinesi araç parametrelerinin doğal dilden çıkarılması hâlâ çok işleçli/negatif/ondalıklı ifadeler için ayrı düzeltme gerektiriyor. Konuşma belleğinin RAM retention'ı oturumlar arasında kapasite etkisi taşıyor; yeni deneyim deposundaki scope sınırları eski konuşma belleğini yeniden tasarlamıyor. Model değiştirme ve birden fazla facade için ortak yaşam döngüsü kilidi ayrı ele alınmalı.

**Darboğazlar:** Girdi karakterleri gerçek token/FLOP hesabına dönüşmeli; elde edilen kalite kanıtının kaynak güveni uygulama düzeyinde doğrulanmalı; farklı görevlerde bütçe kararını değerlendirecek dış doğrulayıcılar hazırlanmalı. Mevcut API aynı ayarlarla yeterli bağımsız geri bildirim toplamadan tercih öğrenemez.

**Sonraki araştırma sınırı:** Geçmiş dış doğrulamalı deneyimden uzman prior'ı öğrenen küçük bağlam projeksiyonu, kaynak bağımlılıklı bellek konsolidasyonu ve doğrulanmış deneyimden eğitim manifest'i üretimi. Bunlar bu tur uygulanmış sayılmıyor. Yeni çalışma önce hedef sızıntısı, yanlış aşinalık, interference ve unutmanın türetilmiş kayıtlara yayılımını tanımlamalı; büyük eğitimden önce küçük ayrıştırıcı deney düzeni kurulmalı.

## Literatürdeki konumu

[Adaptive Computation Time — Graves](https://arxiv.org/abs/1603.08983), hesaplama adedi öğrenimini recurrent ağlarda inceler. Buradaki strateji bütçesi ACT uygulaması veya öğrenilmiş neural halting değildir.

[Neural Episodic Control — Pritzel ve diğerleri](https://proceedings.mlr.press/v70/pritzel17a.html), deneyim belleği ile hızlı değer güncellemesini birlikte ele alır. Buradaki dış feedback tablosu NEC'in differentiable neural dictionary veya RL algoritmasını uygulamaz; deneyimden hızlı tercih değişimi için karşılaştırma zemini sağlar.

[Switch Transformers — Fedus, Zoph, Shazeer](https://jmlr.org/papers/v23/21-0998.html), sparse uzman yönlendirmesi ve ölçekleme için temel referanstır. Bu turdaki sınırlı dış prior bir araştırma girdisidir; yeni algoritmik üstünlük veya özgünlük kanıtı sunulmaz.
