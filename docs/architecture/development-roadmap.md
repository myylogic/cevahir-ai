# Cevahir — Geliştirme / Refactor Yol Haritası

> **Amaç:** `code-reality/` altında her kanonik birimden çıkan "Refactor Sinyalleri"ni
> tek bir önceliklendirilmiş plana toplamak. Bu doküman, sonraki büyük güncelleme
> turunun **yürütme planıdır**: neyi, hangi sırayla, neden değiştireceğimizi tanımlar.
>
> **Kaynak:** Tüm bulgular [`master-architecture.md`](master-architecture.md) §10 ve
> her birimin `code-reality/<birim>/README.md` §8 bölümünden gelir.
>
> **İlke:** Önce belgeledik (as-is), sonra değiştireceğiz. Her değişiklik bir
> sözleşmeyi (protokol/config/checkpoint uyumu) bilinçli olarak korur veya
> bilinçli olarak kırar — kazara değil.

---

## 1. Kılavuz İlkeler

Bu tur boyunca uyulacak kurallar:

1. **Sözleşmeleri koru.** Kritik örtük kontratlar: `vocab_size` (Tokenizer→Model),
   `tie_weights ⇒ seq_proj_dim == embed_dim`, `ModelAPI` protokolü (Cognitive↔Engine),
   checkpoint şeması. Bunları değiştiren her adım açıkça işaretlenir.
2. **Davranışı koruyarak refactor et.** Yapı değişiklikleri (bölme, birleştirme,
   yeniden adlandırma) çıktı davranışını değiştirmemeli; davranış değişiklikleri
   ayrı, işaretli adımlardır.
3. **Test kalkanı.** Her birimde mevcut `test/` klasörleri var; refactor öncesi
   yeşil olduklarından emin ol, sonra koru. Test yoksa önce karakterizasyon testi.
4. **Kanonik sürüm kararı.** v2/v3 ikizliklerinde önce "kanonik olan hangisi?"
   kararı verilir, sonra diğeri kaldırılır/birleştirilir — paralel bakım
   sürdürülmez.
5. **Küçük, izlenebilir PR'lar.** Her tema kendi dalında; büyük "her şeyi değiştir"
   commit'i yok.

---

## 2. Öncelik Matrisi

Önem (etki × risk) ve bağımlılık sırasına göre:

| Öncelik | Tema | Etkilenen birimler | Neden şimdi |
|---------|------|--------------------|-------------|
| **P0** | v2/v3 sürüm birleştirme | Training System, Training Management | En yüksek bakım maliyeti; v3 hâlâ v2'ye sızıyor |
| **P0** | Çift kalıcılık birleştirme | Chatting, Database | Aynı varlıklar iki yerde; veri bütünlüğü riski |
| **P1** | Konfigürasyon tek kaynağı | Model/Engine, Training System | "İki yerde parametre" checkpoint uyuşmazlığı üretiyor |
| **P1** | Sync/async ikizliği | Cognitive | İki boru hattı + middleware paralel bakım |
| **P1** | God-class ayrıştırma | Tokenizer, Cognitive, Model/Engine | 1000–2000 LOC sınıflar; SRP ihlali |
| **P2** | CPU/GPU kod yolu birleştirme | Tokenizer, Neural Network | Her bileşende `_gpu` ikizi |
| **P2** | İsimlendirme tutarlılığı | Neural Network, API | `TransformerEncoderLayer`=decoder; `service/` vs `services/` |
| **P2** | Ölü kod / eski route temizliği | API, Training, Database | v2 route'lar, boş `schemas/`, kök debug betikleri |
| **P3** | İzleme çatısı birleştirme | Cognitive, Training, Model/Engine | Üç ayrı TensorBoard/health monitor |

---

## 3. Fazlar

### Faz 0 — Hazırlık (kod değişikliği yok)
- [ ] Her birimde mevcut testleri çalıştır, yeşil taban çizgisini kaydet.
- [ ] Kritik sözleşmeler için karakterizasyon testleri ekle (encode/decode round-trip,
      forward shape, checkpoint load/save, cognitive `process` çıktısı).
- [ ] v2/v3 için "kanonik sürüm" kararını ver (aşağıdaki P0 teması).

### Faz 1 — P0 Yapısal Borç
1. **v2/v3 birleştirme (Training)**
   - Karar: v3'ü kanonik yap (curriculum/optimizers/safety zaten v3'te daha zengin).
   - v3'ün v2'ye olan bağımlılıklarını kes: `training_service_v3.py:506` içindeki
     `training_management.v2.utils` import'larını v3 karşılıklarına taşı.
     (bkz. [training-system §8](code-reality/training-system/README.md#8-refactor-sinyalleri--tech-debt),
     [training-management §8](code-reality/training-management/README.md#8-refactor-sinyalleri--tech-debt))
   - Ortak parçaları `training_*/common/` altına çıkar; v2'yi `_archive` veya kaldır.
2. **Çift kalıcılık birleştirme (Chatting/Database)**
   - Karar: `database/repositories` kanonik kalıcılık olsun; `chatting_management/storage`
     in-memory yalnızca test/geliştirme adaptörü olarak kalsın veya repository
     arayüzünün bir implementasyonu olsun.
     (bkz. [chatting §8](code-reality/chatting/README.md#8-refactor-sinyalleri--tech-debt),
     [database §8](code-reality/database/README.md#8-refactor-sinyalleri--tech-debt))
   - `api/services` ↔ `chatting_management` manager sorumluluk sınırını netleştir.

### Faz 2 — P1 Tutarlılık ve Ayrıştırma
3. **Konfigürasyon tek kaynağı**
   - `model/cevahir.py` ve `training_system/train.py` arasındaki çift parametre
     tanımını tek bir `config_schema` kaynağına indir (Model/Engine'deki
     `config_schema.py` genişletilebilir). README'deki "iki yerde güncelle" uyarısı
     bu iş bitince kalkar.
4. **Cognitive sync/async birleştirme**
   - `pipeline`+`handlers` ile `async_pipeline`+`async_handlers` ortak çekirdeğe
     indirilir; `ModelAPI` protokolü tek yerde (`interfaces/`) tanımlanır
     (şu an hem `cognitive_manager.py` hem `critic_v2.py`'de var).
5. **God-class ayrıştırma**
   - `BPEManager` (1694 LOC): dosya-IO / vocab-bakımı / tokenizasyon-orkestrasyonu ayrı.
   - `cognitive_manager.py` (2070 LOC): facade + event/cache/trace API'leri ayrı.
   - `model/cevahir.py` (2114 LOC): `CevahirConfig` / `CevahirModelAPI` / `Cevahir` ayrı dosyalara.

### Faz 3 — P2/P3 Temizlik ve Optimizasyon
6. CPU/GPU kod yollarını ortak arayüz + backend seçimiyle birleştir.
7. İsimlendirme: `TransformerEncoderLayer → DecoderLayer`; boş `api/service/` kaldır.
8. Ölü kod: v2 route'ları, boş `database/schemas/`, kökteki `debug_*`/`test_*_overlap.py`
   betiklerini `tests/`/`scripts/` altına topla.
9. İzleme çatısını (TensorBoard/health/anomaly) tek bir gözlemlenebilirlik modülünde topla.

---

## 4. Birim Bazlı Plan ve Bulgu Dizini

Her kanonik birimin **kendi Phase-1 Refactoring Plan'ı** vardır (research ile
eşleşir; önce Test Fazı, sonra Geliştirme Fazı):

| Birim | Research | Phase-1 Plan |
|-------|----------|--------------|
| Tokenizer | [README](code-reality/tokenizer/README.md) | [plan](code-reality/tokenizer/phase-1-refactoring-plan.md) |
| Neural Network | [README](code-reality/neural-network/README.md) | [plan](code-reality/neural-network/phase-1-refactoring-plan.md) |
| Model / Engine | [README](code-reality/model-engine/README.md) | [plan](code-reality/model-engine/phase-1-refactoring-plan.md) |
| Training Management | [README](code-reality/training-management/README.md) | [plan](code-reality/training-management/phase-1-refactoring-plan.md) |
| Training System | [README](code-reality/training-system/README.md) | [plan](code-reality/training-system/phase-1-refactoring-plan.md) |
| Data | [README](code-reality/data/README.md) | [plan](code-reality/data/phase-1-refactoring-plan.md) |
| Cognitive | [README](code-reality/cognitive/README.md) | [plan](code-reality/cognitive/phase-1-refactoring-plan.md) |
| Chatting | [README](code-reality/chatting/README.md) | [plan](code-reality/chatting/phase-1-refactoring-plan.md) |
| API | [README](code-reality/api/README.md) | [plan](code-reality/api/phase-1-refactoring-plan.md) |
| Database | [README](code-reality/database/README.md) | [plan](code-reality/database/phase-1-refactoring-plan.md) |

Her temanın kaynak bulgusu ilgili birim dokümanının §8'indedir:

| Birim | Öne çıkan borç |
|-------|----------------|
| [Tokenizer](code-reality/tokenizer/README.md#8-refactor-sinyalleri--tech-debt) | `BPEManager` god-class, çift facade, singleton, CPU/GPU ikizi |
| [Neural Network](code-reality/neural-network/README.md#8-refactor-sinyalleri--tech-debt) | `TransformerEncoderLayer` yanıltıcı isim, attention kod ikizliği |
| [Model / Engine](code-reality/model-engine/README.md#8-refactor-sinyalleri--tech-debt) | 2114 LOC facade, `ModelManager` çift-rol, config çift kaynak |
| [Training Management](code-reality/training-management/README.md#8-refactor-sinyalleri--tech-debt) | v2/v3 ikizliği, dev sınıflar |
| [Training System](code-reality/training-system/README.md#8-refactor-sinyalleri--tech-debt) | v2/v3 servis ikizliği, v3→v2 sızıntısı, dağınık betikler |
| [Data](code-reality/data/README.md#8-refactor-sinyalleri--tech-debt) | tek dev loader, kaba token tahmini |
| [Cognitive](code-reality/cognitive/README.md#8-refactor-sinyalleri--tech-debt) | 2070 LOC facade, sync/async ikizliği, çoklu `ModelAPI` |
| [Chatting](code-reality/chatting/README.md#8-refactor-sinyalleri--tech-debt) | çift kalıcılık, manager/servis örtüşmesi |
| [API](code-reality/api/README.md#8-refactor-sinyalleri--tech-debt) | v2/v3 route ikizliği, `service/` vs `services/` |
| [Database](code-reality/database/README.md#8-refactor-sinyalleri--tech-debt) | çift kalıcılık, boş `schemas/`, entegrasyon belirsizliği |

---

## 5. İzleme

Bu yol haritası yürütüldükçe güncellenir. Her tema tamamlandığında ilgili birim
dokümanının §8'i revize edilir ve buradaki durum işaretlenir.

| Tema | Durum |
|------|-------|
| v2/v3 birleştirme (Training) | ⏳ planlandı |
| Çift kalıcılık birleştirme | ⏳ planlandı |
| Konfigürasyon tek kaynağı | ⏳ planlandı |
| Cognitive sync/async birleştirme | ⏳ planlandı |
| God-class ayrıştırma | ⏳ planlandı |
| CPU/GPU birleştirme | ⏳ planlandı |
| İsimlendirme tutarlılığı | ⏳ planlandı |
| Ölü kod temizliği | ⏳ planlandı |
| İzleme çatısı birleştirme | ⏳ planlandı |

> **Durum anahtarı:** ✅ tamam · 🔄 devam ediyor · ⏳ planlandı

---

*Bu plan, kodun mevcut halinden çıkarılan bulgulara dayanır. Yürütme sırası
tartışmaya açıktır; öncelikler proje sahibinin kararıyla güncellenebilir.*
