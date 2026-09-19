# Baseline yaşam döngüsü uzlaştırması

Bu çalışma dosya adlarındaki sürüm etiketlerinden değil, üretici/tüketici ilişkilerinden hareket eder. Değiştirilen aktif sınırlar: model kaydı → bağımsız yükleyici → ModelManager → uygulama açılışı; eğitim checkpoint kaydı → resume; veri cache → V2/V3 bölme; çıkarım yöneticisi → çekirdek; bilişsel not/özet → oturum kapsamı.

## Kapatılan kök nedenler

- Checkpoint biçimi farklı girişlerde farklı yorumlanıyordu. checkpoint_contract.py artık eski state_dict/optimizer_state ile model_state_dict/optimizer_state_dict zarflarını aynı biçimde çözer. ModelSaver ve aktif V2 eğitim kayıtları modelin gerçek kurucu ayarlarını model_config olarak taşır; eğitim ayarlarıyla karıştırmaz.
- ModelLoader.load_model ve load_all kayıtlı mimariden kurulum yapar. Mevcut modele yükleme, ağırlıkları değiştirmeden önce mimari hesap ayarlarını ve state anahtar/boyutlarını kontrol eder. Paralel residual, attention başlıkları, RoPE ölçeklemesi gibi aynı tensor şekilleriyle farklı hesap yapabilen tercihler kontrol kapsamındadır. Legacy ayarsız state_dict için yalnız şekil/anahtar güvencesi vardır.
- ModelManager artık yükleme sonrasında canlı modelle ilgisiz kayıtlı config'i üzerine dökmez; optimizer/scheduler hatalarını yutup başarılı gibi devam etmez. Kayıtlı öğrenme oranı restore edilir. Farklı oran isteniyorsa yükleme sonrası açık güncelleme gerekir. Başarılı yükleme ve eğitim restore'u KV durumunu temizler.
- Uygulama açılışı strict=False yükleme ve başarısız checkpoint sonrasında rastgele modelle devam etme davranışını kaldırır. load_model_path='' açık biçimde kayıtsız başlangıcı korur; verilmiş veya bulunan checkpoint başarısızsa başlangıç hata verir.
- V2 kaynak kimliklerini bölmeden önce atıyordu. V2 ve V3 artık aynı split_training_records işlevini kullanır: kaynak grupları ve birebir tekrarlarla bağlı geçişli gruplar tek bölmede kalır, global RNG değişmez, iki bağımsız grup şarttır. Kaynak kimliği olmayan kayıtlarda sadece birebir içerik tekrarları korunabilir; belge ve yakın tekrar sızıntısının çözüldüğü iddia edilmez.
- Eski cache'in farklı tokenizer/config/veri kimliğiyle seçilmesini sağlayan bypass bayrakları artık hata verir. Cache kayıtları benzersiz geçici dosya kullanır; eşzamanlı yazanlar aynı .tmp dosyasını paylaşmaz.
- ModelManager fonksiyon iç değişkenleri yerine forward imzasını kullanır; **kwargs kabul eden sarmalayıcılara maske/cache/tanılama seçimini geçirir. Dış no_grad kapsamını yeniden açmaz.
- Çekirdek normal forward'da tüm logits üzerinde istatistik hesaplamaz. collect_diagnostics=True veya TensorBoard örnekleme aralığı ayrıntılı snapshot üretir. Normal snapshot güncel boyut/dtype ve diagnostics_collected=False içerir; eski istatistikleri güncel gibi göstermez.
- Bilişsel özetler benzersiz kimlik ve current_scope ile saklanır. Not ekleme/okuma/silme aynı kapsamla sınırlıdır; başarılı özet/not değişimi response-cache revizyonunu artırır. Eski kapsamı belirsiz özetlere kullanıcı ataması yapılmadı.
- ModelSaver checkpoint'i önce tam RAM tamponuna kopyalamak yerine benzersiz geçici dosyaya yazar, flush/fsync sonrası atomik yayımlar. Eski anahtar takma adları aynı hazırlanmış state'i paylaşır. CUDA hatasında kayıt biçimini değiştiren fallback kaldırıldı; başarısız kayıt açıkça başarısızdır.

## Doğrulama ve pratik sınırlar

Yeni lifecycle testleri kayıtlı ayarlarla iki yükleme yolunda logits eşitliği, aynı şekilli yanlış mimarinin değişiklik öncesi reddi, KV sıfırlama, eski zarf metadata/optimizer korunması, geçişli tekrar grupları, no_grad ve wrapper aktarımı, cache bypass reddi, uygulama açılışında fail-closed davranış ve not/özet kapsamını doğrular.

İlk geniş koşu 345 başarılı, 2 atlanan ve iki eksik-dosya exception uyumsuzluğu gösterdi; üretimde CheckpointNotFoundError, FileNotFoundError ile uyumlu yapıldı. Sonraki geniş koşuda 352 başarılı kontrol vardı; kalan üç hata ve bir setup hatası disk doluluğundan kaynaklandı. lifecycle_verification.xml bu başarısız koşuyu dürüstçe korur. Sonraki hedefli koşuda lifecycle_targeted.xml içinde 53 kontrol, kayıt/restore düzeltmelerinden sonra lifecycle_final.xml içinde 32 kontrol geçti. Disk nedeniyle yarım kalan SQLite, optimizer kayıt/restore ve quantization kontrolleri bu hedefli koşularda yeniden doğrulandı. Sayılar örtüşür, toplanmaz.

core_lifecycle.json tek CPU'da küçük sentetik ölçümdür. Loss 4.881231307983398, cache farkı 1.4901161193847656e-07, NaN/Inf sıfır. Sistem yükü ve disk baskısı altında alınan süreler hızlanma iddiası için kullanılmaz. Dil kalitesi veya GPU performansı ölçülmedi.

## Yeniden taramada açık kalan işler

20 Eylül güncellemesi: tokenizer kimliğinin aktif eğitim/checkpoint/yükleyici sınırlarında taşınması aşağıdaki çalışmayla tamamlandı; kimliksiz eski kayıtların sınırı korunur. Açık işler tüm kayıtların veri türüne uygun hizalama denetimi, yakın tekrar/kaynaksız belge ayrımı, optimizer/RNG dahil resume hata bütünlüğü ve ortak modelin yükleme/üretim sahipliğidir. ModelSaver'ın doğrudan dosyaya yazması, V2 eğitim CheckpointManager'ının RAM tamponu ve sabit geçici dosya borcunu kapatmaz. Bağımsız V3 yönetici aktif backend değildir. Güncel öncelikler ve tamamlanma ölçütleri [geliştirme planında](NEXT_DEVELOPMENT_ROADMAP.md) yer alır; bu belge önceki çalışmanın kapsamını ve kanıtlarını korur.


## Tokenizer kimliğinin checkpoint boyunca korunması

ModelManager kayıtları ve aktif V2/V3 eğitim girişleri artık gerçek tokenizer_digest kimliğini checkpoint'e taşır. Kimlik mevcut token-ID sözlüğünü, BPE merges içeriğini ve BPE yapılandırmasını kapsar. Eğitim servisi kimliği TrainingManager'a geçirir; model üzerinde de korunur. Ortak zarf çözücüsü top-level, additional_info ve eğitim extra_state konumlarını okur.

Kimlikli checkpoint ModelManager, ModelLoader.load_model/load_all ve V2 CheckpointManager.load üzerinden açılırken tokenizer verilmelidir; bağımsız yükleyicilerde yeni tokenizer= parametresi kullanılır. Aktif training resume güncel eğitim tokenizer kimliğiyle karşılaştırır. Uyuşmazlık veya kimlikli kayıtta tokenizer eksikliği ağırlıklara dokunmadan reddedilir. Legacy kimliksiz kayıtlar güncel tokenizer verildiğinde açık RuntimeWarning üretir; token semantiği doğrulanmış sayılmaz. Salt ağırlık dışa aktarma metadata içermediği için bu legacy sınıra tabidir.

Yüklenen modelin kimliği doğrudan ModelSaver ile yeniden kayıtta korunur; ModelManager.save daha önce kimlik atanmış modeli farklı tokenizer ile yeniden etiketlemeyi reddeder. V2 eğitim açılışındaki checkpoint yükleme hatasını yutup devam etme yolu kaldırıldı. Vocabulary veya merges dosyası değiştirilmedi; bu bir token-ID migrasyonu değildir.

Kimlik doğrulaması yükleme, kayıt ve eğitim başlangıcı sınırlarındadır; her token üretim adımında pahalı hash hesaplanmaz. Çalışma sırasında dışarıdan tokenizer nesnesini mutasyona uğratmanın bütünüyle engellendiği iddia edilmez. Tokenizer kimliği dışındaki veri hizalaması, yakın tekrarlar ve eşzamanlı model yükleme/üretim borçları açık kalır.
