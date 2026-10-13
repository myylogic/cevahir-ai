# Tur 23 — 200 araştırma önermesi

48 madde ilk dalgada çalıştırılabilir; 152 madde bekliyor. Tohum tekrarları hipotez değildir.
Aynı mekanizmanın farklı rejimlerdeki koşullu iddiaları ayrı satırdır; 200 bağımsız buluş iddiası yoktur.

## MEM

### MEM01 — Bloklu akışta geçmiş kapsamı

Durum: **İlk dalga**.

Öngörü: clustered koşulunda reservoir, fifo yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız reservoir ile fifo bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `reservoir` / `fifo`. Rejim: `clustered`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM02 — Durağan kanıtın yeterli sayımlarla birikmesi

Durum: **İlk dalga**.

Öngörü: stable koşulunda count, reservoir yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız count ile reservoir bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `count` / `reservoir`. Rejim: `stable`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM03 — Tek gözlem yerine gürültülü kanıt birleştirme

Durum: **İlk dalga**.

Öngörü: stable koşulunda count, last yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız count ile last bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `count` / `last`. Rejim: `stable`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM04 — Ani değişimde anahtar başına yakın geçmiş

Durum: **İlk dalga**.

Öngörü: abrupt koşulunda window, count yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız window ile count bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `window` / `count`. Rejim: `abrupt`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM05 — Yerel değişimlerde yumuşak unutma

Durum: **İlk dalga**.

Öngörü: staggered koşulunda decay, count yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız decay ile count bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `decay` / `count`. Rejim: `staggered`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM06 — Değişimde uniform tarih örneklemesinin sınırı

Durum: **İlk dalga**.

Öngörü: abrupt koşulunda window, reservoir yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız window ile reservoir bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `window` / `reservoir`. Rejim: `abrupt`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM07 — Durağan dünyada unutmanın bedeli

Durum: **İlk dalga**.

Öngörü: stable koşulunda count, decay yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız count ile decay bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `count` / `decay`. Rejim: `stable`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM08 — Seyrek anahtarlarda kapsam kotası

Durum: **İlk dalga**.

Öngörü: rare koşulunda window, fifo yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız window ile fifo bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `window` / `fifo`. Rejim: `rare`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM09 — Kimlikli düzeltmenin sayımlara taşınması

Durum: **İlk dalga**.

Öngörü: revisions koşulunda revision_delta, revision_ignore yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız revision_delta ile revision_ignore bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `revision_delta` / `revision_ignore`. Rejim: `revisions`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM10 — Düzeltme ile yeni gözlemin ayrılması

Durum: **İlk dalga**.

Öngörü: revisions koşulunda revision_delta, revision_append yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız revision_delta ile revision_append bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `revision_delta` / `revision_append`. Rejim: `revisions`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM11 — Düzeltmenin artımlı uygulanması

Durum: **İlk dalga**.

Öngörü: revisions koşulunda revision_delta, revision_rebuild ile aynı cevapları daha az toplam sanal işlemle verir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız revision_delta ile revision_rebuild bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez. Herhangi bir cevap farkı eşdeğerlik şartını ayrıca yanlışlar.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `revision_delta` / `revision_rebuild`. Rejim: `revisions`. Ölçüt: `virtual_operations_per_event`. Pratik eşik: 5.0.

### MEM12 — Durağan gürültüde sonlu pencere sınırı

Durum: **İlk dalga**.

Öngörü: stable koşulunda count, window yönteminden daha düşük bütün yaşam karesel kaybı üretir.

Müdahale: Aynı sorgu ve açık gözlem/düzeltme akışında yalnız count ile window bellek işlemleri değiştirilir.

Yanlışlama: Dokunulmamış doğrulama örnekleminde pratik eşiği aşan beklenen yön desteklenmez veya ters etki görülür; eşitlik de üstünlüğü desteklemez.

Bilinen aile / önceki çalışma: Bilinen rezervuar örnekleme, yeterli istatistik, kayan pencere, üstel unutma veya artımlı veri güncelleme ailesi; özgünlük iddiası yok.

Aday / kontrol: `count` / `window`. Rejim: `stable`. Ölçüt: `whole_life_squared_loss`. Pratik eşik: 0.005.

### MEM13 — Öğrenilen değişim algılayıcısı

Durum: **Henüz sınanmadı**.

Öngörü: Sınır işareti verilmeden edinilen değişim algılayıcısı, sabit pencereye göre değişim sonrası ve durağan dönem kaybı toplamını azaltır.

Müdahale: Sabit pencereyi aynı geçmişle öğrenilen pencere seçimiyle değiştir.

Yanlışlama: Algılama ve yanlış alarm bedeliyle toplam kayıp azalmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM14 — Görev kimliği bilinmeden bağlam belleği

Durum: **Henüz sınanmadı**.

Öngörü: Tekrar eden bağlamların kimliğini deneyimden çıkarmak, açık görev etiketi olmayan dönüşlerde yeniden öğrenme bedelini azaltır.

Müdahale: Gizli bağlamı çıkarsayan bellek ile aynı bütçeli tek bellek karşılaştırılır.

Yanlışlama: Ek yapı ve bağlam hataları dahil toplam bedel azalmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM15 — Sıkıştırılmış kanıtın seçmeli geri çekilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Kimlik sorgusunu yanıtlayacak yan bilgi, ham kayıt olmadan bazı düzeltmeleri tam uygulamaya yeter.

Müdahale: Farklı sıkıştırma biçimlerini aynı kimlikli düzeltmelerle karşılaştır.

Yanlışlama: Aynı sıkıştırılmış duruma düşen geçmişler düzeltme sonrası farklı doğru sayımlar gerektirir.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM16 — İlişkili tekrarların kanıt sayılması

Durum: **Henüz sınanmadı**.

Öngörü: Öğrenilmiş tekrar bağımlılığını tutmak, kopya gözlemler karşısında yalancı kesinliği azaltır.

Müdahale: Bağımsız sayım ile aynı bütçeli bağımlılık hesabını karşılaştır.

Yanlışlama: Bağımlılık bilinmeden veya kayınca kalibrasyon üstünlüğü kaybolur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM17 — Sorgu kayması altında rezervuar

Durum: **Henüz sınanmadı**.

Öngörü: Önceki sorgu dağılımına göre seçilen bellek, yeni sorgu dağılımında uniform rezervuardan kötüleşebilir.

Müdahale: Kayıt seçimini geçmiş sorgu sıklığına bağla ve sonra gizli sorgu dağılımını değiştir.

Yanlışlama: Kayma sonrasında uniform kontrol lehine önceden belirtilen fark görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM18 — Soğuk kaydın gelecekteki yararı

Durum: **Henüz sınanmadı**.

Öngörü: Düşük güncel kullanımına rağmen yapısal köprü olan kayıtları korumak, yeni bileşik görevlerde yarar sağlar.

Müdahale: Köprü kimliği verilmeden edinilmiş önem puanıyla eşit bütçeli sıklık seçimini karşılaştır.

Yanlışlama: Köprü keşif maliyeti dahil yeni görev yararı görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM19 — Kategori sınırının bellek bütçesinden edinilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Deneyimden çıkarılan paylaşım bölmeleri, hazır anahtar başı kotaya göre aynı bellekle daha düşük gelecek kaybı verir.

Müdahale: Anahtar paylaşımını sabit veya öğrenilmiş yap, öğrenilmiş bölmenin maliyetini say.

Yanlışlama: Yanlış birleştirme ve ayırma maliyetiyle üstünlük kaybolur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### MEM20 — Yeni etiketlerin bellek diline eklenmesi

Durum: **Henüz sınanmadı**.

Öngörü: Yeni etiket değerleri ortaya çıktığında genişleyen durum dili, sabit etiketli özetin bilgi kaybını önler.

Müdahale: Sonradan doğan etiketlerde sabit ve genişletilebilir depoyu eşit kaynak çizelgesiyle sınayarak ayır.

Yanlışlama: Genişleme kontrolünün maliyetiyle fayda yoktur veya eski ayrımlar silinir.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

## CON

### CON01 — Boşluklarda derleme

Durum: **Henüz sınanmadı**.

Öngörü: Dış etkileşim boşluklarında pahalı güncellemeleri yapmak, aynı toplam iş ile cevap gecikmesini düşürür.

Müdahale: Aynı işi çevrimiçi anlara veya boşluklara yerleştir.

Yanlışlama: Boşluk dışı kuyruk gecikmesi ve toplam maliyet birlikte düşmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON02 — Kesintiye dayanıklı birleştirme

Durum: **Henüz sınanmadı**.

Öngörü: Kesilebilir birleştirme, kesilemez toplu güncellemeye göre yarım iş kaybını azaltır.

Müdahale: Aynı kesinti akışında güncelleme işlemini küçük doğrulanmış adımlara ayır.

Yanlışlama: Checkpoint bedeli dahil toplam iş azalmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON03 — Birleştirme sırasının nedensel etkisi

Durum: **Henüz sınanmadı**.

Öngörü: Değişmeli olmayan güncellemelerin sırası, aynı kanıtla son davranışı değiştirir.

Müdahale: Kanıt çoklu kümesi sabitken sıralamayı değiştir.

Yanlışlama: Belirtilen işlemler için bütün sıralamalar aynı durumu ve davranışı verir.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON04 — Bağımsız kanıtta zamanlamanın etkisizliği

Durum: **Henüz sınanmadı**.

Öngörü: Toplamsal yeterli istatistikte yalnız birleştirme zamanını değiştirmek son öğrenilmiş durumu değiştirmez.

Müdahale: Aynı gözlemleri her adım veya blok sonunda topla.

Yanlışlama: Aralık içi cevaplar ayrı tutulduğunda son durumlar farklıdır.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON05 — Geri çağırma önceliğinin edinilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Deneyimle edinilen tekrar sırası, uniform tekrarın aynı işiyle daha iyi eski görev koruması verir.

Müdahale: Tekrar önceliğini edinilmiş veya uniform yap.

Yanlışlama: Öncelik edinme işiyle toplam kayıp iyileşmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON06 — Tekrarın hatayı büyütmesi

Durum: **Henüz sınanmadı**.

Öngörü: Yanlış geçmişi daha sık yeniden işlemek, bağımsız düzeltme yoksa hatalı inancı güçlendirebilir.

Müdahale: Yanlış kayıtların tekrar sayısını artır ve kanıt bağımlılığını koru veya sil.

Yanlışlama: Bağımsız kanıt sayımı hatası ya da gelecek kaybı artmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON07 — Sıkıştırma öncesi çelişki kontrolü

Durum: **Henüz sınanmadı**.

Öngörü: Çelişki denetimi, hatalı kayıtların kalıcı özetlere yayılmasını azaltır.

Müdahale: Özetlemeden önce veya sonra aynı denetimi çalıştır.

Yanlışlama: Ek denetim maliyetiyle yanlış kalıcı davranış azalmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON08 — Çok ölçekli birleştirme

Durum: **Henüz sınanmadı**.

Öngörü: Hızlı değişen ve durağan alt görevler için ayrı birleştirme hızları tek hızdan iyidir.

Müdahale: Aynı karma akışta tek hız ile edinilmiş çoklu hız kullan.

Yanlışlama: Hız atama maliyetiyle toplam kayıp azalmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON09 — Göreve koşullu birleştirme

Durum: **Henüz sınanmadı**.

Öngörü: Her kanıtı ortak duruma yazmak yerine göreve göre paylaşmak negatif aktarımı azaltır.

Müdahale: Paylaşım kararını aynı bütçede sabit veya edinilmiş yap.

Yanlışlama: Ortak kontrolün kaybı daha düşük veya eşittir.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON10 — Birleştirme ile edinim arasındaki yarış

Durum: **Henüz sınanmadı**.

Öngörü: İç yeniden işleme dış örnek edinimine bütçe bırakmayınca gelecek öğrenme yavaşlar.

Müdahale: Sabit bütçeyi yeniden işleme ve yeni kanıt edinimi arasında değiştir.

Yanlışlama: Edinim fırsatı maliyeti hesaba katıldığında olumsuz bölge yoktur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON11 — Senkronizasyon borcu

Durum: **Henüz sınanmadı**.

Öngörü: Uzun süre ertelenen durum birleştirmesi, sonraki düzeltme bedelini doğrusalın üstüne çıkarabilir.

Müdahale: Birleştirme aralığını önceden belirtilen bağımlılık ağında artır.

Yanlışlama: Toplam bekleyen bağımlılık işi doğrusalın üstüne çıkmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON12 — Geri alınabilir derleme

Durum: **Henüz sınanmadı**.

Öngörü: Dayanak izleyen derleme, kaynak değiştiğinde tam yeniden derlemeden ucuz olabilir.

Müdahale: Aynı kaynak düzeltmelerinde bağımlılıklı geri alma ve yeniden kurmayı karşılaştır.

Yanlışlama: Aynı doğru cevapla toplam maliyet düşmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON13 — Üretilmiş örneklerde destek kayması

Durum: **Henüz sınanmadı**.

Öngörü: Kendi ürettiği tekrar örnekleri, özgün kanıtın seyrek bölgelerini kaybedebilir.

Müdahale: Gerçek örnek ve öğrenilmiş üretici tekrarını eşit maliyetle karşılaştır.

Yanlışlama: Seyrek sorgu kaybı veya kapsam farkı görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON14 — Birleştirme ölçütünün oyunlaştırılması

Durum: **Henüz sınanmadı**.

Öngörü: Yalnız yeniden yapılandırma hatasını azaltan birleştirme gelecekteki güncellenebilirliği bozabilir.

Müdahale: Aynı eski cevap doğruluğunda farklı özetleri yeni kanıtla sınayarak ayır.

Yanlışlama: Yeni kanıt sonrası ayrım veya hata farkı oluşmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON15 — Düşük faaliyet ile uyku varsayımının ayrılması

Durum: **Henüz sınanmadı**.

Öngörü: Biyolojik uyku taklidi olmadan kaynak tahsisi aynı gözlenen hesap yararını açıklayabilir.

Müdahale: Yalnız bütçe ve zamanlaması eşit iki soyut kontrol kur.

Yanlışlama: Uyku adlı mekanizmaya özgü ek etki kalır; biyolojiye doğrudan çıkarım yapılmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON16 — Öğrenilmiş birleştirme tetikleyicisi

Durum: **Henüz sınanmadı**.

Öngörü: İş yükünden edinilen tetikleyici, sabit aralığın toplam gecikme ve iş kaybını azaltır.

Müdahale: Tetikleyiciyi aynı geçmişle öğren veya sabit tut.

Yanlışlama: Edinim maliyeti dahil toplam hedef iyileşmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON17 — Birleştirme sonrası tanıklık denetimi

Durum: **Henüz sınanmadı**.

Öngörü: Korunan küçük tanık kümesi, yanlış birleştirmeyi yeni kanıt gelmeden yakalayabilir.

Müdahale: Tanık seçimini rastgele veya edinilmiş yap ve hatalı birleştirmeleri enjekte et.

Yanlışlama: Aynı denetim bütçesinde yakalama oranı artmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON18 — Hiyerarşik birleştirme sırası

Durum: **Henüz sınanmadı**.

Öngörü: Önce yerel sonra ortak özetleme bazı çapraz bağımlılıkları geri getirilemez siler.

Müdahale: Aynı kayıtları düz veya iki aşamalı özetle.

Yanlışlama: Gelecek çapraz sorgularda belirlenen ayırt edilemezlik oluşmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON19 — Kısmi aktarımın kesintisiz hizmeti

Durum: **Henüz sınanmadı**.

Öngörü: Eski ve yeni duruma geçişi kademelendirmek, bütünsel geçişin hizmet kesintisini azaltır.

Müdahale: Aynı kanıtta atomik ve kademeli aktarım maliyetlerini karşılaştır.

Yanlışlama: Tutarlılık ve bakım işiyle toplam kesinti yararı kaybolur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### CON20 — Birleştirme politikasının değişebilirliği

Durum: **Henüz sınanmadı**.

Öngörü: Geçmişte yararlı birleştirme politikası değişen iş yükünde yeniden öğrenilebilir olmalıdır.

Müdahale: Politika kilitleme ile politika güncellemesini karşılaştır.

Yanlışlama: Yeniden öğrenme bedeli dahil sonraki toplam yarar görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

## RES

### RES01 — Bütün yaşam başabaş noktası

Durum: **Henüz sınanmadı**.

Öngörü: Öğrenilmiş hızlandırıcının edinim maliyeti kullanım sıklığına bağlı başabaş eşiği doğurur.

Müdahale: Kullanım sayısını değiştirerek edinim dahil iki yöntemin işini karşılaştır.

Yanlışlama: Önceden türetilen eşik gözlenen maliyeti açıklamaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES02 — Bellek ve hesap arasında taşınan bedel

Durum: **Henüz sınanmadı**.

Öngörü: Daha fazla saklanan durum yeniden hesaplamayı azaltırken toplam kaynak cephesini değiştirebilir.

Müdahale: Aynı bilgiyi sakla veya yeniden kur, iki kaynağı ayrı kaydet.

Yanlışlama: Belirtilen görev ailesinde öngörülen değiş tokuş yoktur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES03 — Bütçe kesimindeki yöntem sırası

Durum: **Henüz sınanmadı**.

Öngörü: Sınırsız hesapta iyi yöntem sonlu zaman sınırında kötüleşebilir.

Müdahale: Aynı görevleri önceden sabitlenmiş durdurma sınırlarıyla çalıştır.

Yanlışlama: Sıra değişimi görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES04 — Kaynak politikasının edinim bedeli

Durum: **Henüz sınanmadı**.

Öngörü: Kaynak seçicisini öğrenmek küçük görev sayısında sabit seçiciden pahalıdır.

Müdahale: Öğrenilmiş ve sabit seçicinin bütün yaşam toplamını ölç.

Yanlışlama: Belirtilen kısa yaşam aralığında edinim cezası görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES05 — Gecikmiş kaynak ödülü

Durum: **Henüz sınanmadı**.

Öngörü: Anlık ucuzlukla seçilen öğrenme yöntemi uzun dönem yeniden kullanım yararını kaçırır.

Müdahale: Anlık ve çok adımlı bütçe seçimini aynı deneyimle karşılaştır.

Yanlışlama: Uzun dönem toplamda beklenen fark yoktur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES06 — Ölçüm maliyetinin ihmal edilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Yararı ölçme işi sayılınca bazı uyarlamalı kaynak yöneticileri üstünlüğünü kaybeder.

Müdahale: Aynı yöneticiyi ölçüm işi dahil ve hariç muhasebeleştir.

Yanlışlama: Önceden belirtilen koşulda üstünlük yönü değişmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES07 — Karışık iş yükünde portföy

Durum: **Henüz sınanmadı**.

Öngörü: Birden fazla yönteme pay ayırmak dağılım değişiminde tek yönteme bağlılığın kaybını sınırlar.

Müdahale: Aynı bütçede sabit tek yöntem ve önceden belirlenen portföyü karşılaştır.

Yanlışlama: Bölünen bütçenin fırsat maliyetiyle toplam zarar azalmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES08 — Bütçe ile kanıt ediniminin etkileşimi

Durum: **Henüz sınanmadı**.

Öngörü: Daha ucuz iç hesap ek dış deneyime izin vererek sonraki öğrenmeyi dolaylı iyileştirir.

Müdahale: Artan bütçeyi harcama veya dış deneyime aktar.

Yanlışlama: Aynı toplam kaynak altında gelecekte fark görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES09 — Yanlış kesinliğin kaynak israfı

Durum: **Henüz sınanmadı**.

Öngörü: Hatalı yüksek güvenle aramayı erken kesmek beklenen toplam düzeltme işini artırabilir.

Müdahale: Güvene bağlı kesme ile sabit sınırı karşılaştır.

Yanlışlama: Düzeltme dahil toplam iş artmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES10 — Nadir ağır görevin ortalamaya etkisi

Durum: **Henüz sınanmadı**.

Öngörü: Ortalama işlem maliyetine göre seçilen yöntem kuyruk gecikmesini kötüleştirebilir.

Müdahale: Ortalama ve üst yüzdelik hedefli seçimleri aynı akışta karşılaştır.

Yanlışlama: Öngörülen ortalama-kuyruk ayrımı görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES11 — Artan sayı bitlerinin gerçek bedeli

Durum: **Henüz sınanmadı**.

Öngörü: Sabit sayıda sayacı tutmak sınırsız sabit bit belleği anlamına gelmez.

Müdahale: Alan sayısı ve bit sayısı muhasebelerini aynı uzun akışta çıkar.

Yanlışlama: Önceden belirtilen sayma görevinde bit ihtiyacı büyümez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES12 — Deneyim seçmenin fırsat maliyeti

Durum: **Henüz sınanmadı**.

Öngörü: Aktif deneyim seçimi kazancını seçim aramasının işi aşabilir.

Müdahale: Seçim araması ve pasif edinimi eşit toplam bütçeyle karşılaştır.

Yanlışlama: Öngörülen maliyet bölgesinde aktif seçim daha kötü olmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES13 — Donanım fiyatına bağlı yöntem sırası

Durum: **Henüz sınanmadı**.

Öngörü: Hesap ve bellek birim fiyatları değişince aynı algoritmaların toplam maliyet sırası değişir.

Müdahale: İki bağımsız kaynak ölçüsüne açık farklı fiyatlar uygula.

Yanlışlama: Önceden türetilmiş sıralama bölgeleri doğrulanmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES14 — Güncelleme güvenliğinin kaynak sınırı

Durum: **Henüz sınanmadı**.

Öngörü: Doğrulama bütçesi kısıldığında edinilen yeni yordamın yanlış kabul oranı artabilir.

Müdahale: Aday üretimi sabitken doğrulama bütçesini değiştir.

Yanlışlama: Yanlış kabul ve kaynak arasında belirtilen ilişki yoktur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES15 — Görev dağılımını bilmenin ayrıcalığı

Durum: **Henüz sınanmadı**.

Öngörü: Gerçek gelecek görev sıklığını bilen kaynak seçicisinin üstünlüğü öğrenilebilir olmayabilir.

Müdahale: Gelecek bilgili tanı kontrolünü yalnız geçmiş kullanan seçiciyle ayır.

Yanlışlama: Aynı yarar geçmişten edinilir veya ayrıcalık etkisizdir.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES16 — Dış depolamanın gizli bedeli

Durum: **Henüz sınanmadı**.

Öngörü: Harici arşivle sabit iç bellek iddiası toplam depolama ve erişim işini gizleyebilir.

Müdahale: İç, dış ve erişim maliyetlerini birlikte ölç.

Yanlışlama: Belirtilen yöntemde ihmal edilmiş büyüyen bedel bulunmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES17 — Yeniden öğrenme ile saklama eşiği

Durum: **Henüz sınanmadı**.

Öngörü: Nadir bir beceriyi yeniden edinmek saklama maliyetinden ucuz olabilir.

Müdahale: Görev dönüş aralığını değiştir ve bütün maliyeti say.

Yanlışlama: Önceden türetilen sakla/sil eşik bölgesi yoktur.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES18 — Yöntem seçicisinin negatif aktarımı

Durum: **Henüz sınanmadı**.

Öngörü: Geçmişten öğrenilen kaynak kuralı yeni işlem fiyatlarında zararlı olabilir.

Müdahale: Aynı görevleri fiyat değişimiyle sürdür, seçiciyi kilitle veya güncelle.

Yanlışlama: Fiyat değişimi sonrasında kilitli seçici zarar görmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES19 — Kapsam maliyetinin öğrenilebilirliği

Durum: **Henüz sınanmadı**.

Öngörü: Yeni konu ekledikçe erişim aramasının büyümesi toplam yetenek kazancını sınırlayabilir.

Müdahale: Aynı görev çözücüsüne artan konu bankası bağla ve erişim işini say.

Yanlışlama: Konu sayısıyla erişim bedeli veya kayıp artmaz.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

### RES20 — Açık uçlu genişlemede kaynak denetimi

Durum: **Henüz sınanmadı**.

Öngörü: Yeni durum dilini büyüten sistem kaynak denetimi olmadan yararlı beceriyi hizmete çeviremeyebilir.

Müdahale: Dil genişletmesini kaynak denetimli ve denetimsiz sürdür.

Yanlışlama: Aynı dış bütçede hizmet kaybı farkı görülmez.

Bilinen aile / önceki çalışma: İlgili bilinen bellek, çevrimiçi öğrenme, bilgi derleme veya kaynak sınırlı karar aileleriyle karşılaştırılacak; bu kayıt özgünlük veya doğrulanma iddiası değildir.

## REP

### REP01 — Hata güdümlü özellik ekleme doğrusal modeli aşar

Durum: **İlk dalga**.

Öngörü: Seyrek ikili etkileşim bulunan akışta hata güdümlü özellik ekleme bütün yaşam karesel hatasını düşürür.

Müdahale: Etkinleştirme açık/kapalı; aynı sıralı ridge güncellemesi.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `linear_decay1_replay`. Rejim: `interaction`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP02 — Seçici büyüme tam hazır temsili aşar

Durum: **İlk dalga**.

Öngörü: Aynı etkileşim akışında seçici özellik büyümesi tam 11 özellikli ridge modelinden düşük hata verir.

Müdahale: Büyüyen 5–8 özellikli temsil / baştan tam 11 özellikli temsil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `full_decay1_replay`. Rejim: `interaction`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP03 — Gereksiz özellikleri geciktirme

Durum: **İlk dalga**.

Öngörü: Salt doğrusal hedefte hata güdümlü büyüme tam özellikli ridge modelinden düşük hata verir.

Müdahale: Aynı veriyle seçici büyüme / tam özellik bankası.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `full_decay1_replay`. Rejim: `additive`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP04 — Hata yönü rastgele genişlemeden değerlidir

Durum: **İlk dalga**.

Öngörü: Etkileşim hedefinde gözlenen artıkla ilişkili özelliği seçmek aynı hata tetikleyicili rastgele seçimden düşük hata verir.

Müdahale: Özellik seçicisi artık ilişkisi / sabit tohumlu rastgele seçim.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `random_decay1_replay`. Rejim: `interaction`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP05 — Eski gözlemi zayıflatma değişen ilişkide yararlıdır

Durum: **İlk dalga**.

Öngörü: Ortada etkileşimi değişen akışta 0.97 unutma çarpanı, 1.0 çarpanından düşük bütün yaşam hatası verir.

Müdahale: Aynı büyüyen öğrenicide üstel ağırlık 0.97 / 1.0.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay0.97_replay` / `adaptive_decay1_replay`. Rejim: `switch`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP06 — Büyüme ve unutma tam unutmalı temsili aşar

Durum: **İlk dalga**.

Öngörü: Değişen etkileşimlerde seçici büyüme, aynı unutma çarpanlı tam temsilden düşük hata verir.

Müdahale: Her iki yöntemde 0.97; seçici / tam özellikler.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay0.97_replay` / `full_decay0.97_replay`. Rejim: `switch`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP07 — Gürültü altında seçici temsil

Durum: **İlk dalga**.

Öngörü: Gürültülü doğrusal hedefte seçici büyüme tam özellikli ridge modelinden düşük hata verir.

Müdahale: Etiket gürültüsü 1.2; seçici / tam temsil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `full_decay1_replay`. Rejim: `noisy`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP08 — Seyrek kanıt özellik edinmeye yeter

Durum: **İlk dalga**.

Öngörü: İlk iki koordinatın +1 değerinin seyrek olduğu etkileşim akışında seçici büyüme doğrusal modelden düşük hata verir.

Müdahale: Aynı dengesiz girdi dağılımında özellik ekleme açık / kapalı.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `linear_decay1_replay`. Rejim: `rare`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP09 — Dağılım kaymasında seçici temsil korunur

Durum: **İlk dalga**.

Öngörü: Önce güçlü ilişkili sonra bağımsız olan girdilerde seçici temsil tam temsilden düşük bütün yaşam hatası verir.

Müdahale: İlk 96 örnekte ilişkili koordinatlar, sonra bağımsız; seçici / tam.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `full_decay1_replay`. Rejim: `covariate`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP10 — Geç ortaya çıkan ilişki seçici olarak edinilir

Durum: **İlk dalga**.

Öngörü: İlk yarısı doğrusal, ikinci yarısı etkileşimli akışta unutmalı seçici büyüme unutmalı tam temsilden düşük hata verir.

Müdahale: Hedefte 96. örnekte yeni etkileşim; her iki yöntemde 0.97.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay0.97_replay` / `full_decay0.97_replay`. Rejim: `onset`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP11 — Özellik edinmenin toplam hesap maliyeti azalır

Durum: **İlk dalga**.

Öngörü: Etkileşim akışında tarama, arşiv ve yeniden kurma dahil seçici büyüme tam temsilden az sanal işlem tüketir.

Müdahale: REP02 ile aynı koşu; hata yerine önceden tanımlı toplam maliyet ölçümü.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `full_decay1_replay`. Rejim: `interaction`. Ölçüt: `cost`. Pratik eşik: 1000.

### REP12 — Yeni özelliğe geçmiş kanıtı taşıma

Durum: **İlk dalga**.

Öngörü: Özellik eklenince gözlenmiş geçmişi yeniden işlemek sıfır geçmişle yeni koordinat eklemekten düşük hata verir.

Müdahale: Aynı ekleme tetikleyicisi; tam geçmişten ridge kurulumu / blok köşegen sıfır-geçmiş kurulumu.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

Aday / kontrol: `adaptive_decay1_replay` / `adaptive_decay1_zero_history`. Rejim: `interaction`. Ölçüt: `mse`. Pratik eşik: 0.01.

### REP13 — Özellik silmenin geri dönüş maliyeti

Durum: **Henüz sınanmadı**.

Öngörü: Dönüşümlü hedeflerde eski özelliği uykuda saklama, tamamen silmeye göre geri dönüşte daha az etiket gerektirir.

Müdahale: Aktif bütçe eşit; pasif istatistik saklama / silme; toplam pasif bellek ücretli.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP14 — Örnek arşivinin yeterli istatistikle değiştirilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Sonradan eklenen özellikler için tutulmuş çapraz momentler aynı güncellemeyi örnek arşivinden düşük bellekle üretir.

Müdahale: Ham arşiv / tüm aday momentleri; bütün edinim maliyeti dahil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP15 — Görev kaymasının girdi kaymasından ayrılması

Durum: **Henüz sınanmadı**.

Öngörü: Etiketli artık ile etiketsiz girdi değişimini ayrı izlemek, yalnız artık izlemeye göre gereksiz temsil değişikliklerini azaltır.

Müdahale: Ayrı iki dedektör / tek dedektör; yanlış alarm ve tespit gecikmesi.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP16 — Özellik dilinin yetersizliği

Durum: **Henüz sınanmadı**.

Öngörü: İkili özellik bankasına hapsolmuş öğrenici saf üçüncü dereceli etkileşimde kalıcı hata tabanına sahiptir.

Müdahale: Aynı örnekler; ikili banka / üçüncü dereceyi içerme; bütçe eşit.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP17 — Temsilin öğrenilmiş dönüşümle kurulması

Durum: **Henüz sınanmadı**.

Öngörü: Ham koordinat döndürmelerinde deneyimden öğrenilen dönüşüm sabit eksen özelliklerinden daha iyi yeni görev aktarımı sağlar.

Müdahale: Kaynakta dönüşüm edinme / sabit dönüşüm; hedefte aynı etiket bütçesi.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP18 — Yanlış seçimin geri alınabilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Özellik değiştirmeye izin vermek üç özelliklik bütçeyi yalnız büyütmeye göre değişen hedefte daha iyi kullanır.

Müdahale: Ekle-sil / yalnız ekle; silme maliyeti ve unutulan görevler ölçülür.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP19 — Aynı cevap farklı güncelleme

Durum: **Henüz sınanmadı**.

Öngörü: Bugünkü tahminleri eşit iki temsilden çapraz momentleri koruyan, sonraki yeni özellikte daha düşük hata verir.

Müdahale: Aynı mevcut ağırlıklar; ek çapraz istatistikler korunur / silinir.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

### REP20 — Temsil edinmede farklı hızlar

Durum: **Henüz sınanmadı**.

Öngörü: Hızlı geçici özellik deposu ve yavaş kalıcılaştırma, tek eşikli kalıcı eklemeye göre kısa gürültü patlamalarında düşük hata verir.

Müdahale: İki zaman ölçeği / tek ölçek; aynı toplam durum bütçesi.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Sıralı özellik seçimi, recursive least squares, unutma çarpanı; özgünlük iddiası yok.

## XFR

### XFR01 — Hazır kimlik köprüsü olmadan eşleme

Durum: **Henüz sınanmadı**.

Öngörü: İlişkisel düzen üzerinden edinilen alan eşlemesi, yüzey benzerliği eşlemesinden yeni alanda daha düşük hata verir.

Müdahale: Nesne kimlikleri karıştırılır; ilişkisel / yüzey eşleyici.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR02 — Yanlış ilişkisel simetriyi bozma

Durum: **Henüz sınanmadı**.

Öngörü: Bir ek ayırıcı deney seçmek, simetrik alanlardaki yanlış aktarımı rastgele deneyden daha çok azaltır.

Müdahale: Simetrik grafiklerde amaçlı / rastgele tek köprü gözlemi.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR03 — Ortak işlem türünü edinme

Durum: **Henüz sınanmadı**.

Öngörü: Kaynak sonuçlarından işlem türünü çıkaran yöntem, önceden sabit işlem türünden heterojen hedeflerde daha iyi aktarır.

Müdahale: Eşleme dışında bileştirme türü de gizlenir; çıkarım / sabit seçim.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR04 — Çoktan bire anlam köprüsü

Durum: **Henüz sınanmadı**.

Öngörü: Birebir olmayan eşleme dili, kavramların hedefte birleştiği alanlarda bijeksiyon varsayımından düşük hata verir.

Müdahale: Kaynak kavramları hedefte birleşir; bölümleme / bijeksiyon.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR05 — Bir kavramın hedefte ayrışması

Durum: **Henüz sınanmadı**.

Öngörü: Bağlama bağlı bire-çok köprü, hedefte ayrışan kavramlarda tek görüntülü köprüden iyi tahmin verir.

Müdahale: Hedef bağlamı erişilebilir; bağlamsal / sabit köprü.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR06 — Köprünün güvenini ayrı tutma

Durum: **Henüz sınanmadı**.

Öngörü: Kaynak bilginin doğruluğu ile köprü güvenini ayırmak, yanlış köprüde tüm kaynak bilgiyi reddetmekten az unutma doğurur.

Müdahale: Yanlışlanan köprü; yerel / küresel güven düşüşü.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR07 — Aktarım olumsuzluğu erken tanınır

Durum: **Henüz sınanmadı**.

Öngörü: Kaynak uyumu için ayrılmış az sayıda hedef örneği, koşulsuz aktarıma göre bütün yaşam hatasını azaltır.

Müdahale: Uyum yoklama / doğrudan aktarım; yoklama bütçesi ücretli.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR08 — Alanlar arası işaret dönüşümü

Durum: **Henüz sınanmadı**.

Öngörü: Aynı bağıntının ters işaretle kodlandığı hedefte dönüşüm edinme, örnek benzerliği aktarımını aşar.

Müdahale: İşaret dönüşümü gizli; öğrenilmiş dönüşüm / en yakın örnek.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR09 — Bağlam değişince kategori yeniden düzenlenir

Durum: **Henüz sınanmadı**.

Öngörü: Görev koşullu bölümlemeler, tek kalıcı bölümlemeden kesişen görevlerde düşük toplam karar hatası verir.

Müdahale: Aynı olaylar iki görev; çoklu bölümleme / tek bölümleme.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR10 — Ortak alt yapı yeni kavramı hızlandırır

Durum: **Henüz sınanmadı**.

Öngörü: Ayrı kaynaklarda öğrenilmiş iki ilişkinin ortak alt yapısı, birleşik yeni görevde tek kaynak aktarımından daha az etiket gerektirir.

Müdahale: İki kaynak / her tek kaynak; yeni birleşim ve eşit toplam kaynak maliyeti.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR11 — İlgisiz kaynak bilgisi aktarımı engeller

Durum: **Henüz sınanmadı**.

Öngörü: Kaynak seçimi, giderek artan ilgisiz kaynak sayısında bütün kaynakları birleştirmekten az hedef hata doğurur.

Müdahale: Kaynak sayısı ve ilgisizlik artırılır; seçici / toplu aktarım.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR12 — İki taraflı köprü denetimi

Durum: **Henüz sınanmadı**.

Öngörü: Gidiş-dönüş tutarlılığı kontrolü yalnız ileri uyumdan daha az yanlış köprü kabul eder.

Müdahale: Yalnız ileri / iki yönlü test; simetri tuzakları dahil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR13 — Köprü oluşturmanın bedeli

Durum: **Henüz sınanmadı**.

Öngörü: Öğrenilmiş köprü toplam ömür maliyetinde tekrar hedef çözümlemeyi yalnız yeterli tekrar sayısından sonra aşar.

Müdahale: Tekrar sayısı taraması; köprü edinimi dahil kırılma noktası.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR14 — Soyut ilişkiden yüzeye geri aktarım

Durum: **Henüz sınanmadı**.

Öngörü: Soyut ilişkisel başarıyı ham gözlem eylemlerine bağlama eğitimi, yalnız soyut eğitimden uçtan uca hedef başarısını artırır.

Müdahale: Algısal bağlantı edinilir / hazır bağlantı tanısal kontrol.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR15 — Çelişkili kültürler benzeri alanlar

Durum: **Henüz sınanmadı**.

Öngörü: Alan kimliğini korumak, çelişen yerel kuralları tek kümede birleştirmekten az negatif aktarım üretir.

Müdahale: Aynı sembollerin farklı yasaları; alan koşullu / küresel.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR16 — Az köprüyle tüm eşlemeyi bilmek gerekmez

Durum: **Henüz sınanmadı**.

Öngörü: Göreve özgü kısmi köprü, tam eşleme ediniminden aynı karar doğruluğunu daha düşük maliyetle sağlar.

Müdahale: Hedef sorgularının kapsamı sınırlı; kısmi / tam eşleme.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR17 — Eski köprü yeni bilgiyi de taşıyabilir

Durum: **Henüz sınanmadı**.

Öngörü: Kaynak güncellendiğinde bağıntı koruyan köprü, yüzey ezber köprüsünden güncelleme aktarımında daha başarılıdır.

Müdahale: Köprü kurulduktan sonra yeni kaynak gerçekleri verilir.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR18 — Eşleme yanlışını bilgi yanlışından ayırma

Durum: **Henüz sınanmadı**.

Öngörü: Köprü ve yerel model için ayrı karşı örnek açıklamaları, tek hata havuzundan daha az gereksiz düzeltme doğurur.

Müdahale: Ayrı nedenler / tek neden; hata yerinin gerçek dağılımı gizli.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR19 — Yapısal uzaklık başarıyı sınırlar

Durum: **Henüz sınanmadı**.

Öngörü: Öğrenilmiş ilişkisel uzaklıkla aktarımı sınırlamak, yalnız yüzey uzaklığından daha iyi başarısızlık öngörür.

Müdahale: Hedef başarı tahmini için iki uzaklık; kalibrasyon.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

### XFR20 — Köprülerin bileştirilmesi hata biriktirir

Durum: **Henüz sınanmadı**.

Öngörü: Ara köprü doğrulaması, çok adımlı alan aktarımında doğrudan köprü zincirinden düşük son hata verir.

Müdahale: Üç alan; denetimli / denetimsiz köprü bileşimi.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Yapı eşleme, alan uyarlama, ilişkisel aktarım; henüz uygulama veya yenilik iddiası yok.

## LAN

### LAN01 — Yordamın örneklerden edinilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Görev örneklerinden işlem sırası edinen sistem, yalnız yanıt saklamadan görülmemiş girdilerde daha başarılıdır.

Müdahale: Örnek-yanıt tablosu / edinilmiş işlem dizisi; eşit kanıt.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN02 — Yeni bileştiricinin maliyeti

Durum: **Henüz sınanmadı**.

Öngörü: Deneyimden döngü şeması edinmek, aynı sonlu açılımları saklamadan uzun girdilerde düşük maliyet sağlar.

Müdahale: Döngü şeması / açılmış programlar; edinim maliyeti dahil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN03 — İşlem dilinin yanlışlanması

Durum: **Henüz sınanmadı**.

Öngörü: Karşı örnekte işlem dilini genişleten yöntem, sabit yetersiz dilden farklı görev türlerinde düşük hata verir.

Müdahale: Başlangıçta eksik işlem dili; genişleme / sabit dil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN04 — İlkel işlem ile kısaltma ayrımı

Durum: **Henüz sınanmadı**.

Öngörü: Bir edinilmiş işlemi eski dile açma kontrolü, salt makro kazancını gerçek temsil sınıfı farkından ayırır.

Müdahale: Yeni işlem / aynı işlemin açılmış eski dil karşılığı.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN05 — Öğrenme yordamını öğrenme

Durum: **Henüz sınanmadı**.

Öngörü: Görev dizisinden güncelleme yordamı edinmek, aynı görev kanıtlı sabit güncellemeden yeni görevde daha iyi örnek verimi sağlar.

Müdahale: Edinilmiş güncelleyici / güçlü sabit güncelleyici; meta edinim maliyeti.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN06 — Yordam düzenlemesinin geriye etkisi

Durum: **Henüz sınanmadı**.

Öngörü: Eski yordamın yerine yerel yama koyma, tüm yordamı yeniden sentezlemekten eşit doğrulukta az hesap gerektirir.

Müdahale: Yerel karşı örnek; yama / sıfırdan sentez.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN07 — Yanlış soyutlamanın tersine çevrilmesi

Durum: **Henüz sınanmadı**.

Öngörü: Soyutlamayı açıp ayrıntıya dönebilmek, yalnız yeni soyutlama eklemekten değişen görevde düşük hata verir.

Müdahale: Açılabilir / kalıcı soyutlama; eski görev koruma.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN08 — İki parçadan yeni hesap oluşturma

Durum: **Henüz sınanmadı**.

Öngörü: Kaynakta ayrı edinilen yordamların arayüzlerini öğrenmek, hazır arayüz varsayımından daha geniş hedef ailesinde başarılı olur.

Müdahale: Arayüz türleri gizli; edinilmiş / sabit tür bağlama.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN09 — Tip bilgisinin deneyimden oluşması

Durum: **Henüz sınanmadı**.

Öngörü: Başarısız işlemlerden edinilen tip kısıtları, yalnız frekans önceliğinden arama maliyetini düşürür.

Müdahale: Tip kısıtı edinme / aynı geçmişli frekans sıralaması.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN10 — Program durumunu değiştirmeden sürdürme

Durum: **Henüz sınanmadı**.

Öngörü: Yordam değişiminde korunacak durumun çıkarılması, baştan çalıştırmadan daha az maliyetle doğru sürdürme sağlar.

Müdahale: Öğrenilmiş durum taşıma / tam geçmişi tekrar yürütme.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN11 — Gerekçe bağımlılıklarının yordamda tutulması

Durum: **Henüz sınanmadı**.

Öngörü: Edinilmiş yordamın dayanaklarını izlemek, geçersiz kaynak değişince kör yordam belleğinden daha az yanlış yürütme üretir.

Müdahale: Dayanaklı / dayanıksız yordam; yerel kaynak değişimleri.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN12 — Arama stratejisi koşula bağlı öğrenilir

Durum: **Henüz sınanmadı**.

Öngörü: Görev özelliklerinden arama düzeni seçmek, tek global öğrenilmiş sıradan heterojen görevlerde düşük toplam maliyet sağlar.

Müdahale: Koşullu / global sıra; seçici eğitim maliyeti dahil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN13 — Başarısızlıklardan negatif şema

Durum: **Henüz sınanmadı**.

Öngörü: Karşı örneklerden edinilmiş yasak şemaları, yalnız başarılı program belleğinden daha az arama üretir.

Müdahale: Başarısızlık kısıtları açık / kapalı; yanlış budama ölçülür.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN14 — Eşdeğer programların birleşmesi

Durum: **Henüz sınanmadı**.

Öngörü: Kanıtlı eşdeğerlik sınıfları, yalnız gözlenen çıktı eşitliğiyle birleştirmeden daha az gizli hata doğurur.

Müdahale: Kanıtlı / örnek üstünde eşdeğerlik; görülmemiş girdiler.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN15 — Sabit doğrulayıcının sınırı

Durum: **Henüz sınanmadı**.

Öngörü: Yeni yordam dili mevcut doğrulayıcının kapsamını aştığında yüksek eğitim doğruluğu güvenilir genellemeyi garanti etmez.

Müdahale: Doğrulayıcı kapsamı içinde / dışında yeni işlem; bağımsız tam kontrol.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN16 — Yordam belleği kapasite baskısı

Durum: **Henüz sınanmadı**.

Öngörü: Beklenen tekrar değeriyle yordam tutma, kullanım sayısıyla tutmadan değişen tekrar yapısında düşük toplam maliyet verir.

Müdahale: Değer tahmini / frekans silme; eğitim ve bellek erişimi dahil.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN17 — Kaynak dağılımı arama dilini daraltır

Durum: **Henüz sınanmadı**.

Öngörü: Tek ailede edinilmiş program önceliği farklı aileye geçince sabit genel aramadan kötü olabilir.

Müdahale: İlişkili / ilişkisiz hedefler; genel arama güçlü kontrol.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN18 — Özyineleme ile ezber ayrımı

Durum: **Henüz sınanmadı**.

Öngörü: Örnek uzunluğu dışındaki görevlerde edinilmiş özyineleme, benzer kısa program belleğinden yüksek doğruluk sağlar.

Müdahale: Uzunluk dışına aktarım; sabit açılım kontrolü.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN19 — Dilin büyümesi seçim maliyetini artırır

Durum: **Henüz sınanmadı**.

Öngörü: Yeni yordamları indislemeden eklemek, yeterince büyük bellekte arama faydasını toplam maliyette kaybeder.

Müdahale: Bellek büyüklüğü; düz liste / öğrenilmiş veya sabit indis.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

### LAN20 — Yeni güncelleme kuralının güven sınırı

Durum: **Henüz sınanmadı**.

Öngörü: Edinilmiş güncelleyiciyi eski bilgide karşı örneklerle sınırlamak, sınırsız güncellemeden daha az yıkıcı unutma üretir.

Müdahale: Kısıtlı / sınırsız güncelleyici; yeni öğrenme hızındaki bedel birlikte.

Yanlışlama: Eşleştirilmiş bağımsız doğrulama tohumlarında önceden belirtilen üstünlük elde edilmezse bu koşullu öngörü desteklenmez; ters ve pratik olarak anlamlı etki öngörüyü çürütür.

Bilinen aile / önceki çalışma: Program sentezi, program kitaplığı öğrenme, meta-learning; henüz uygulama veya yenilik iddiası yok.

## UPD

### UPD01 — Sabit düşük gürültüde deneyimden kazanç seçimi

Durum: **İlk dalga**.

Öngörü: stationary_low koşulunda gain_meta, fixed_005 yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `fixed_005`. Rejim: `stationary_low`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD02 — Sabit yüksek gürültüde deneyimden kazanç seçimi

Durum: **İlk dalga**.

Öngörü: stationary_noisy koşulunda gain_meta, fixed_005 yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `fixed_005`. Rejim: `stationary_noisy`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD03 — Yavaş sürüklenmede deneyimden kazanç seçimi

Durum: **İlk dalga**.

Öngörü: drift koşulunda gain_meta, fixed_02 yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `fixed_02`. Rejim: `drift`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD04 — Ani sıçramalarda deneyimden kazanç seçimi

Durum: **İlk dalga**.

Öngörü: jumps koşulunda gain_meta, fixed_02 yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `fixed_02`. Rejim: `jumps`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD05 — Sürüklenme yönü değiştiğinde deneyimden kazanç seçimi

Durum: **İlk dalga**.

Öngörü: reversals koşulunda gain_meta, fixed_02 yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `fixed_02`. Rejim: `reversals`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD06 — Dinamik ve gürültü değişirken deneyimden kazanç seçimi

Durum: **İlk dalga**.

Öngörü: mixed koşulunda gain_meta, fixed_02 yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `fixed_02`. Rejim: `mixed`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD07 — Yüksek gürültüde ortak durumlu güncelleme ve bağımsız filtre bankası

Durum: **İlk dalga**.

Öngörü: stationary_noisy koşulunda gain_meta, expert_bank yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `expert_bank`. Rejim: `stationary_noisy`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD08 — Sürüklenmede ortak durumlu güncelleme ve bağımsız filtre bankası

Durum: **İlk dalga**.

Öngörü: drift koşulunda gain_meta, expert_bank yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `expert_bank`. Rejim: `drift`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD09 — Sıçramalarda ortak durumlu güncelleme ve bağımsız filtre bankası

Durum: **İlk dalga**.

Öngörü: jumps koşulunda gain_meta, expert_bank yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `expert_bank`. Rejim: `jumps`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD10 — Değişken rejimde ortak durumlu güncelleme ve bağımsız filtre bankası

Durum: **İlk dalga**.

Öngörü: mixed koşulunda gain_meta, expert_bank yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı gözlem akışı üzerinde kazanç yordamasını değiştir; gizli sinyal, gelecek gözlemler ve rejim etiketi öğreniciye verilmez.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `gain_meta` / `expert_bank`. Rejim: `mixed`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD11 — Edinilmiş kazancın benzer yeni dinamiğe aktarılması

Durum: **İlk dalga**.

Öngörü: transfer_matched koşulunda frozen_gain_transfer, fixed_02_reset yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı edinim ve hedef gözlemleri; tahmin durumu iki yöntemde sıfırlanır; yalnız adayın geçmiş gözlem hatalarından edindiği kazanç taşınır.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `frozen_gain_transfer` / `fixed_02_reset`. Rejim: `transfer_matched`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD12 — Edinilmiş kazancın uyumsuz yeni dinamiğe aktarılması

Durum: **İlk dalga**.

Öngörü: transfer_mismatch koşulunda frozen_gain_transfer, fixed_02_reset yönteminden en az 0.0005 düşük bütün yaşam normalize saklı sinyal hatası üretir.

Müdahale: Aynı edinim ve hedef gözlemleri; tahmin durumu iki yöntemde sıfırlanır; yalnız adayın geçmiş gözlem hatalarından edindiği kazanç taşınır.

Yanlışlama: Dokunulmamış doğrulamada önkayıtlı pratik fark ve çoklu karşılaştırma ölçütü karşılanmazsa iddia desteklenmez; ters yön güçlü ise reddedilir.

Bilinen aile / önceki çalışma: Bilinen uyarlamalı filtreleme ve üstel ağırlıklı uzman öğrenme ailesi; yeni optimizer iddiası yok.

Aday / kontrol: `frozen_gain_transfer` / `fixed_02_reset`. Rejim: `transfer_mismatch`. Ölçüt: `mean_normalized_latent_prequential_squared_error`. Pratik eşik: 0.0005.

### UPD13 — Geçmişe göre kazanç seçmek değişim hızını öğrenebilir

Durum: **Henüz sınanmadı**.

Öngörü: Aynı gözlemlerle edinilen kazanç, yeni fakat benzer değişim hızındaki görevlerde başlangıç kazancından düşük hedef hatası verir.

Müdahale: Edinilmiş kazancı yeni tahmin durumu sıfırlandıktan sonra taşı; edinimi ters zaman sırasıyla da dene.

Yanlışlama: Eş bütçeli sabit veya basit uyarlamalı kontrolün hedef hatası daha düşük ya da eşitse.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD14 — Öğrenilmiş kazancı unutmak bazen gereklidir

Durum: **Henüz sınanmadı**.

Öngörü: Eski dinamiğin tersine döndüğü akışta kazanç belleğini kısmen unutmak tam korumadan düşük bütün yaşam hatası verir.

Müdahale: Aynı yeni kanıt altında eski kazanç ağırlıklarını koru, sıfırla veya azalt.

Yanlışlama: Unutmanın geçiş maliyeti dahil farkı yoksa ya da koruma daha iyiyse.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD15 — Güncelleme dili yeni terim gerektirebilir

Durum: **Henüz sınanmadı**.

Öngörü: Sabit kazanç ailesi yeterli değilken edinilmiş ikinci durum değişkeni, aynı kaynak sınırında daha az hata sağlar.

Müdahale: Tek durumlu kazanç ailesine karşı deneyimden seçilen hız durumu ve eş bütçeli hazır iki durumlu kontrol.

Yanlışlama: Güçlü hazır kontrol tüm kazanımı açıklarsa veya ek terim yeni görevlerde yararsızsa.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD16 — Hataların büyüklüğü ile yönü farklı güncelleme bilgisi taşır

Durum: **Henüz sınanmadı**.

Öngörü: Aynı büyüklük dağılımlı fakat farklı işaret bağımlılıklı hatalarda yalnız hata büyüklüğünden öğrenen yöntem yetersiz kalır.

Müdahale: İşaret dizisini koru veya karıştır; marjinal hata büyüklüklerini aynı tut.

Yanlışlama: İşaret bilgisini kullanan yöntem dokunulmamış akışta avantaj sağlamazsa.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD17 — Kazanç öğrenmesi ölçüm yanlılığını öğrenme sanabilir

Durum: **Henüz sınanmadı**.

Öngörü: Ölçüm yanlılığı kalıcıyken hata küçülmesi saklı sinyal doğruluğunda iyileşme sağlamayabilir.

Müdahale: Aynı gerçek sinyale sıfır ortalamalı ve yanlı sensörler uygula; gözlem ve doğruluk kaybını ayır.

Yanlışlama: Gözlem hatası ve saklı doğruluk her koşulda birlikte iyileşirse.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD18 — Durum ve güncelleme öğrenmesi için ayrı kanıt gerekebilir

Durum: **Henüz sınanmadı**.

Öngörü: Tahmin durumu için yeterli kısa kayıt, hangi güncellemenin yeni görevde iyi olacağını belirlemek için yeterli olmayabilir.

Müdahale: Bugünkü tahminleri eşit iki geçmiş kur; gelecekteki değişimlere aynı yeni kanıtı uygula.

Yanlışlama: Önerilen karşı örnekte bütün gelecekteki optimum güncellemeler aynı kalırsa.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD19 — Güncelleme yetkisinin sınırlanması edinim hızına zarar verebilir

Durum: **Henüz sınanmadı**.

Öngörü: Geçmiş becerileri koruyan kısıt, hızlı değişimde uyarlama hatasını artırabilir.

Müdahale: Sabit geçmiş testlerini koruyan sınır ile serbest ve bütçe eşlenmiş yeniden öğrenmeyi karşılaştır.

Yanlışlama: Koruma maliyeti dahil uyarlama farkı yoksa veya sınırlı yöntem daha iyi ise.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

### UPD20 — Güncelleme yordaması geri alınabilir olmalıdır

Durum: **Henüz sınanmadı**.

Öngörü: Yanlış gözlem sonradan düzeltildiğinde dayanaklı kazanç belleği, kör belleğe göre daha doğru gelecek güncellemeler yapar.

Müdahale: Bir eski gözlemi düzelt; tam tekrar, dayanaklı kısmi onarım ve yalnız canlı durum onarımını karşılaştır.

Yanlışlama: Kazanç belleğinin onarımı eş kanıt ve bütün maliyet altında fark yaratmazsa.

Bilinen aile / önceki çalışma: Uyarlamalı filtreleme, meta-learning ve optimizer learning ile kesişir; yenilik iddiası değildir.

## CRD

### CRD01 — Gecikme uzunluğu ile kredi ayrıştırılabilirliği farklıdır

Durum: **Henüz sınanmadı**.

Öngörü: Sonuç gecikmesi aynı olsa bile sonuçların hangi eyleme ait olduğunun bilindiği akış daha hızlı öğrenilir.

Müdahale: Gecikmeleri sabit tut; yalnız nedensel eşleşme anahtarını kaldır.

Yanlışlama: Eşleşme belirsizliği eş bütçede öğrenmeyi etkilemezse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD02 — Toplu ödülden tekil nedenler her zaman ayrıştırılamaz

Durum: **Henüz sınanmadı**.

Öngörü: Aynı toplam sonuç üreten farklı nedenlere sahip iki dünyada yalnız toplamlar bireysel etkileri belirleyemez.

Müdahale: Gözlemsel olarak aynı kayıtları veren iki nedeni farklı dünya kur.

Yanlışlama: Ayırt edici ek bilgi olmadan doğru tekil nedenler her iki dünyada çıkarılabilirse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD03 — Kalıcı izler gecikmiş öğrenmeyi destekler

Durum: **Henüz sınanmadı**.

Öngörü: Gecikmiş geri bildirimde dayanak izlerini saklamak, yalnız son duruma göre güncellemeden düşük atıf hatası verir.

Müdahale: Sonuçları geciktir; geçmiş eylem/bağlam izini kaldırma deneyi yap.

Yanlışlama: İz saklama maliyeti dahil düşük hata avantajı yoksa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD04 — Gecikmeye uyarlanmış iz sabit izden üstün olmayabilir

Durum: **Henüz sınanmadı**.

Öngörü: Öğrenilmiş iz süresi, sabit iyi seçilmiş süreye göre bütün yaşam faydası sağlamayabilir.

Müdahale: İz süresini geçmiş gecikmelerden seç; aynı ailede güçlü sabit ve karışım kontrolü kullan.

Yanlışlama: Öğrenilmiş iz her önkayıtlı rejimde tüm maliyet dahil üstünse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD05 — Yanlış eşlenmiş doğru etiketler yanlış öğrenme üretir

Durum: **Henüz sınanmadı**.

Öngörü: Etiket değerleri doğru olsa da yanlış olaya bağlanması kalıcı bilgiye zarar verir.

Müdahale: Etiket değer dağılımını koru; olay kimliklerini karıştır.

Yanlışlama: Kalıcı atıf doğruluğu değişmezse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD06 — Sonradan düzeltilen sonuç geçmiş krediyi değiştirmelidir

Durum: **Henüz sınanmadı**.

Öngörü: Bir eski sonuç düzeltildiğinde yalnız son yanıtı onarmak gelecekteki seçimleri tam onarmayabilir.

Müdahale: Sonuç düzeltmesi sonrası olay izli güncelleme ile yalnız yanıt düzeltmesini karşılaştır.

Yanlışlama: Her gelecekteki davranış ve güncelleme eşdeğer kalırsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD07 — Ortak neden sahte kredi oluşturabilir

Durum: **Henüz sınanmadı**.

Öngörü: Eylem ve sonuç ortak bir bağlamdan etkilenirken gözlemsel kredi seçimi nedensel etkiyi ters tahmin edebilir.

Müdahale: Gözlemsel eşleştirme ve rastgele müdahaleli veri aynı bütçede karşılaştırılır.

Yanlışlama: Kurulan karşı örnekte ortak neden çıkarımı bozmazsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD08 — Birleşik eylem etkisi tekil kredilere indirgenmeyebilir

Durum: **Henüz sınanmadı**.

Öngörü: İki eylemin etkileşimli sonucu tekil kredi toplamıyla doğru temsil edilemez.

Müdahale: Tekil ve birlikte eylem deneyleri aynı toplam bütçede verilir.

Yanlışlama: Tekil terimli model tüm ortak sonuçları doğru açıklar ise.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD09 — Uzun gecikmede doğru özet yeterli olabilir

Durum: **Henüz sınanmadı**.

Öngörü: Gecikme artsa da sonuç için yeterli istatistik biliniyorsa tüm olayları saklamak gerekmez.

Müdahale: Tam kayıt, yeterli özet ve yanlış özetle aynı gecikmiş sonucu işle.

Yanlışlama: Doğru özet tam kayıtla aynı güncellemeyi yapamazsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD10 — Gelecek sonuç dili genişlediğinde eski özet yetersizleşebilir

Durum: **Henüz sınanmadı**.

Öngörü: Eski sonuçlar için yeterli kredi özeti yeni sonuç tipini öğrenmeye yetmeyebilir.

Müdahale: Aynı özete sahip farklı geçmişlere yeni geri bildirim türü ekle.

Yanlışlama: Yeni kredi her geçmiş için aynı özetten doğru hesaplanırsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD11 — Güvenilmeyen etiket kaynaklarını ayırmak öğrenmeyi iyileştirebilir

Durum: **Henüz sınanmadı**.

Öngörü: Farklı güvenilirlikteki kaynakları ayrı izlemek tek havuzdan daha az yanlış kalıcı güncelleme üretir.

Müdahale: Kaynak kimliğini koru veya kaldır; kaynak kalitesi ve örnek sayısını sabit tut.

Yanlışlama: Kaynak kimliği güçlü kontrol karşısında ek fayda sağlamazsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD12 — Güven öğrenmesi kendi tahminine dayanırsa kapanabilir

Durum: **Henüz sınanmadı**.

Öngörü: Etiket güvenini yalnız mevcut modelle uyuma göre seçmek doğru aykırı kanıtı dışlayabilir.

Müdahale: Başlangıç modeli yanlışken bağımsız denetimli ve öz uyumlu güven güncellemesini karşılaştır.

Yanlışlama: Öz uyumlu seçim aynı bütçede doğru aykırı kanıtı eşit hızla öğrenirse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD13 — Nadir kritik sonuçlar ortalama kayıpta kaybolabilir

Durum: **Henüz sınanmadı**.

Öngörü: Ortalama hatayı azaltan kredi yöntemi nadir yüksek etkili olaylarda kötüleşebilir.

Müdahale: Toplam ortalama ve önkayıtlı nadir olay hatasını ayrı ölç.

Yanlışlama: Her kazanç nadir olaylarda da aynı yönlü ise.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD14 — Sonuç yokluğu da bilgi olabilir

Durum: **Henüz sınanmadı**.

Öngörü: Gözlenebilir bir sonuç son tarihi varsa sonuç gelmemesi kredi tahminini değiştirebilir.

Müdahale: Bilinen ve bilinmeyen gözlem pencerelerini karşılaştır.

Yanlışlama: Sonuç yokluğu bilinen pencerede koşullu tahmini hiç değiştirmezse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD15 — Sansürlü gecikme yanlış olumsuz kredi üretebilir

Durum: **Henüz sınanmadı**.

Öngörü: Geç gelen sonuçları yok saymak yavaş faydalı eylemlerin öğrenilmesini engelleyebilir.

Müdahale: Aynı sonuç dizisini farklı kesilme zamanlarında puanla; sağ sansürü modelle.

Yanlışlama: Kesilme düzeltmesi nedensel sıralamayı değiştirmezse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD16 — Kredi belirsizliği güvenli güncellemeyi sınırlayabilir

Durum: **Henüz sınanmadı**.

Öngörü: Birden çok atıf veriyle uyumluysa tek atfı kalıcılaştırmak seçenek kümesi saklamaktan daha çok yanlış güncelleme üretebilir.

Müdahale: Eş kanıtta tek atıf ve aday atıf kümesini karşılaştır.

Yanlışlama: Belirsizlik koruma ek maliyet dahil hiçbir rejimde yanlış atfı azaltmazsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD17 — İzlerin sıkıştırılma sırası güncellemeyi etkileyebilir

Durum: **Henüz sınanmadı**.

Öngörü: Doğrusal olmayan kredi güncellemesinde önce birleştirme ve sonra güncelleme yer değiştiremez.

Müdahale: Aynı olayları önce ayrı sonra ortak güncelle ve tersi; hesapları eşleştir.

Yanlışlama: İleri sürülen doğrusal olmayan örnekte iki yol her zaman eşitse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD18 — Kredi mekanizması yeni eylem bileşimlerine aktarılabilir

Durum: **Henüz sınanmadı**.

Öngörü: Geçmişten öğrenilmiş atıf yapısı yeni bileşik görevde daha az etiketle nedenleri ayrıştırabilir.

Müdahale: Atıf yapısını taşı; tekil geçmiş cevapları ve hazır atıf kontrolünü ayrı tut.

Yanlışlama: Aktarım sadece geçmiş cevap ezberiyle açıklanırsa veya güçlü kontrolü geçmezse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD19 — Değişen nedenlerde eski kredi düzeni zarar verebilir

Durum: **Henüz sınanmadı**.

Öngörü: Eylem-sonuç bağlantısı değişince kalıcı kredi önceliği eşit öncelikten yavaş düzelir.

Müdahale: Aynı gözlemle nedeni değiştir; geçmiş önceliği koru, sıfırla ve uyarlamalı unutmayı karşılaştır.

Yanlışlama: Korunan öncelik ters rejimde ek hata üretmezse.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

### CRD20 — Kredi doğruluğu ile yararlı eylem seçimi ayrıdır

Durum: **Henüz sınanmadı**.

Öngörü: Nedenleri doğru ayırmak, sınırlı eylem bütçesinde daha iyi sonuç garantilemez.

Müdahale: Atıf doğruluğu ve gelecekteki eylem değeri ayrı ölçülür.

Yanlışlama: İleri sürülen ailede atıf doğruluğu her zaman eylem değerini de sıralarsa.

Bilinen aile / önceki çalışma: Gecikmiş geri bildirim, temporal credit assignment, nedensel çıkarım ve provenance ile kesişir; yenilik iddiası değildir.

## CTX

### CTX01 — Gizli bağlam ayırt edici geçmiş gerektirir

Durum: **Henüz sınanmadı**.

Öngörü: Aynı anlık girdide farklı doğru davranışlar varsa anlık temsil tek başına yeterli olmayabilir.

Müdahale: Anlık girdiyi eşitle; geçmişte ayırt edici ipucu bırak veya kaldır.

Yanlışlama: Geçmişsiz yöntem tüm bağlamları doğru ayırabilirse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX02 — Bağlam sayısının büyümesi öğrenme sayılmaz

Durum: **Henüz sınanmadı**.

Öngörü: Yeni küme sayısı artarken dokunulmamış davranış kapasitesi sabit veya kötü olabilir.

Müdahale: Küme sayısı ve maliyet eşliğinde yeni görev hatasını ayrı ölç.

Yanlışlama: Her küme artışı zorunlu olarak genellemeyi iyileştirirse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX03 — Yanlış bölünmüş bağlamları birleştirme yararlı olabilir

Durum: **Henüz sınanmadı**.

Öngörü: Aynı davranış yasasına sahip farklı yüzeyleri birleştirmek veri verimini artırır.

Müdahale: Yüzey etiketlerini değiştir; davranış yasasını aynı tut; birleştirme müdahalesi yap.

Yanlışlama: Birleştirme eş bütçeli ayrı öğrenmeye üstün olmazsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX04 — Yanlış birleştirilmiş bağlamları ayırma gerekebilir

Durum: **Henüz sınanmadı**.

Öngörü: Gözlemsel benzer fakat farklı sonuç yasalarına sahip durumları ayırmak kalıcı hatayı azaltır.

Müdahale: Aynı yüzeyi iki farklı geçmişe bağla; ipucu varsa ayrıştırma dene.

Yanlışlama: İpucu varken ayrıştırma yeni veride hatayı azaltmazsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX05 — Aynı olay farklı görevler için farklı kümelere girebilir

Durum: **Henüz sınanmadı**.

Öngörü: Tek görevde yeterli kümeleme ikinci görevde bilgi kaybı yaratabilir.

Müdahale: Aynı olay havuzuna kesişen iki görev ata; görev koşullu ve ortak bölümü karşılaştır.

Yanlışlama: Tek ilk bölüm tüm ikinci görev cevaplarını korursa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX06 — Bağlamın revizyonu geçmiş ilişkileri taşımalıdır

Durum: **Henüz sınanmadı**.

Öngörü: Yeni kategori düzeni eski geçerli bağlantıları taşımazsa hatırlanan ilişkiler bozulur.

Müdahale: Kategori kimliklerini değiştir; bağlantı taşıma ve kör yeniden etiketlemeyi karşılaştır.

Yanlışlama: Taşıma her yeni sorguda kör yaklaşımla eşitse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX07 — Bağlam dili genişlediğinde eski kayıt değerlenebilir

Durum: **Henüz sınanmadı**.

Öngörü: Önceden işe yaramayan ayrıntı yeni bağlam ayrımı için belirleyici olabilir.

Müdahale: Eski yeterli özetle tam geçmişi yeni ayırt edici görevde karşılaştır.

Yanlışlama: Yeni görevde tam geçmişin ayırt edici bilgisi yoksa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX08 — Bağlamları çok erken kesinleştirmek düzeltmeyi pahalılaştırır

Durum: **Henüz sınanmadı**.

Öngörü: Erken tek bağlam seçimi belirsiz atıf saklamaktan daha fazla sonradan onarım gerektirir.

Müdahale: Aynı ilk belirsiz veri ve sonraki ayırt edici kanıtı iki belleğe ver.

Yanlışlama: Belirsizliği koruma maliyeti dahil onarım faydası yoksa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX09 — Bağlamı çoğaltmak gürültüyü ezberleyebilir

Durum: **Henüz sınanmadı**.

Öngörü: Bağlam başına ayrı parametre sayısını artırmak gürültülü küçük örnekte genellemeyi bozabilir.

Müdahale: Saklı yasa sabitken yüzey kümelerini artır; güçlü küçültme kontrolü kullan.

Yanlışlama: Dokunulmamış hata ve tüm maliyet hiçbir koşulda kötüleşmezse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX10 — Kısmi bağlam paylaşımı tam paylaşım ve ayrımdan farklıdır

Durum: **Henüz sınanmadı**.

Öngörü: Bazı ilişkileri ortak bazılarını ayrı tutmak tamamen ortak ve ayrı modellere göre veri verimli olabilir.

Müdahale: Paylaşılan ve özel alt ilişkileri içeren akışlar üret; eş bütçeli kontrolleri karşılaştır.

Yanlışlama: Kazanım yalnız daha çok parametre veya etiketle açıklanırsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX11 — Bağlam ipucunun güvenilirliği öğrenilmelidir

Durum: **Henüz sınanmadı**.

Öngörü: Geçmişte yararlı ipucu değişince sabit bağlam yönlendirmesi zararlı olur.

Müdahale: İpucu geçerliliğini akışta ters çevir; yönlendirme güncellemesini kaldır.

Yanlışlama: Sabit yönlendirme aynı kanıtla eşit hızda düzelirse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX12 — Bağlam keşfi ile kategori adı eşleştirme ayrıdır

Durum: **Henüz sınanmadı**.

Öngörü: Etiketlerin permütasyonundan bağımsız ilişki keşfi, yalnız ad eşleştirmeye indirgenmeyebilir.

Müdahale: Bağlam adlarını gizle veya rastgele değiştir; yapısal ve isimli yöntemleri karşılaştır.

Yanlışlama: İsim eşleme güçlü yapısal yöntemin bütün tahminlerini açıklarsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX13 — Sürekli bağlam her zaman sonlu kategorilerle ekonomik değildir

Durum: **Henüz sınanmadı**.

Öngörü: Sürekli değişen yasa için çok sayıda ayrık bağlam gereksiz bellek maliyeti doğurabilir.

Müdahale: Aynı veriyle sürekli koşullama ve büyüyen ayrık bağlamları karşılaştır.

Yanlışlama: Sonlu küçük bölüm aynı doğruluk ve toplam maliyeti sürekli sağlar ise.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX14 — Bir bağlam geçmişteki tüm olayları yeniden yazmamalıdır

Durum: **Henüz sınanmadı**.

Öngörü: Yeni bir yerel ilişki tüm kategori üyelerine uygulanırsa yanlış genelleme oluşabilir.

Müdahale: Yerel istisna ekle; kapsamlı ve tümel kategori güncellemesini karşılaştır.

Yanlışlama: Tümel aktarım dokunulmamış olaylarda yanlışlık üretmezse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX15 — Bağlam geçişi ile bilgi değişimi farklı kanıt ister

Durum: **Henüz sınanmadı**.

Öngörü: Çıktı değişikliği eski bağlamın dönüşü veya yasadaki değişimden kaynaklanabilir; tek gözlem bunları ayırmayabilir.

Müdahale: Aynı kısa gözlemi üreten iki mekanizma ve ayırt edici devamlar kur.

Yanlışlama: Belirlenen kısa veri ikisini her durumda ayırırsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX16 — Görevden bağımsız tek anlam ağacı maliyetli olabilir

Durum: **Henüz sınanmadı**.

Öngörü: Birden çok görevde ortak ağaçtan birlik okuması doğru olsa da sorgu maliyeti ayrı görünümlerden büyük olabilir.

Müdahale: Aynı cevapları veren ağaç-birlik ve görev görünümlerini toplam maliyetle karşılaştır.

Yanlışlama: Ek görünüm edinimi dahil ağaç hiçbir rejimde daha pahalı olmazsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX17 — Bağlam aktarımı ortak arayüz olmadan keşif gerektirir

Durum: **Henüz sınanmadı**.

Öngörü: İki deneyim kümesini birleştirmenin faydası hazır eşleme kaldırıldığında kaybolabilir.

Müdahale: Doğru arayüzü verme; eşlemeyi gözlemlerden öğren; hazır arayüz kontrolünü ayrı raporla.

Yanlışlama: Arayüz verilmemişken eş kanıtlı basit kontrol bütün faydayı açıklarsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX18 — Bağlamlar arası köprü yanlış ilişkiyi de yayabilir

Durum: **Henüz sınanmadı**.

Öngörü: Yanlış bir ortak referans üzerinden aktarım ayrı öğrenmeden fazla kalıcı hata üretebilir.

Müdahale: Doğru ve yanlış köprüler ekle; bağımlılık izli düzeltme dene.

Yanlışlama: Yanlış köprü yeni bağlamlarda ek hata üretmezse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX19 — Bağlam temsilinin yaşam boyu maliyeti ölçülmelidir

Durum: **Henüz sınanmadı**.

Öngörü: Kısa dönemde sorguyu hızlandıran yeniden örgütleme daha uzun ufukta edinim ve bakım maliyetini amorti edemeyebilir.

Müdahale: Aynı yaşamları değişik kullanım ufuklarında toplam maliyetle puanla.

Yanlışlama: Her örgütleme tüm ufuklarda maliyet avantajı sağlarsa.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

### CTX20 — Bağlam öğrenmesini durdurmak bilinen cevapları korusa da yeni öğrenmeyi bozabilir

Durum: **Henüz sınanmadı**.

Öngörü: Bugünkü testlerde aynı iki temsil yeni bağlam öğreniminde farklı veri ihtiyacı gösterebilir.

Müdahale: Mevcut cevapları eşitle; bağlam güncelleme kapasitesini kaldırıp yeni ilişkiler ekle.

Yanlışlama: Güncelleme kaldırma yeni görev öğrenimini hiç değiştirmezse.

Bilinen aile / önceki çalışma: Latent-state inference, representation learning, context discovery ve belief revision ile kesişir; yenilik iddiası değildir.

## ENV

### ENV-01 — Sabit büyük ödül farkı: Belirsizlik bonusu

Durum: **İlk dalga**.

Öngörü: Sabit büyük ödül farkı koşulunda ucb, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Eylem seçimindeki belirsizlik bonusunu açma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `ucb` / `greedy`. Rejim: `stationary_gap`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-02 — Sabit büyük ödül farkı: Sabit rastgele sapma

Durum: **İlk dalga**.

Öngörü: Sabit büyük ödül farkı koşulunda epsilon, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Gözlenen en iyi eylemden yüzde10 sapma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `epsilon` / `greedy`. Rejim: `stationary_gap`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-03 — Sabit büyük ödül farkı: Yakın deneyim penceresi

Durum: **İlk dalga**.

Öngörü: Sabit büyük ödül farkı koşulunda window_ucb, ucb karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Bütün-geçmiş istatistiğini son64 adımla değiştirme

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `window_ucb` / `ucb`. Rejim: `stationary_gap`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-04 — Sabit küçük ödül farkı: Belirsizlik bonusu

Durum: **İlk dalga**.

Öngörü: Sabit küçük ödül farkı koşulunda ucb, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Eylem seçimindeki belirsizlik bonusunu açma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `ucb` / `greedy`. Rejim: `stationary_small`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-05 — Sabit küçük ödül farkı: Sabit rastgele sapma

Durum: **İlk dalga**.

Öngörü: Sabit küçük ödül farkı koşulunda epsilon, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Gözlenen en iyi eylemden yüzde10 sapma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `epsilon` / `greedy`. Rejim: `stationary_small`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-06 — Sabit küçük ödül farkı: Yakın deneyim penceresi

Durum: **İlk dalga**.

Öngörü: Sabit küçük ödül farkı koşulunda window_ucb, ucb karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Bütün-geçmiş istatistiğini son64 adımla değiştirme

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `window_ucb` / `ucb`. Rejim: `stationary_small`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-07 — Tek terslenme: Belirsizlik bonusu

Durum: **İlk dalga**.

Öngörü: Tek terslenme koşulunda ucb, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Eylem seçimindeki belirsizlik bonusunu açma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `ucb` / `greedy`. Rejim: `single_reversal`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-08 — Tek terslenme: Sabit rastgele sapma

Durum: **İlk dalga**.

Öngörü: Tek terslenme koşulunda epsilon, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Gözlenen en iyi eylemden yüzde10 sapma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `epsilon` / `greedy`. Rejim: `single_reversal`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-09 — Tek terslenme: Yakın deneyim penceresi

Durum: **İlk dalga**.

Öngörü: Tek terslenme koşulunda window_ucb, ucb karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Bütün-geçmiş istatistiğini son64 adımla değiştirme

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `window_ucb` / `ucb`. Rejim: `single_reversal`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-10 — Tekrarlı terslenme: Belirsizlik bonusu

Durum: **İlk dalga**.

Öngörü: Tekrarlı terslenme koşulunda ucb, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Eylem seçimindeki belirsizlik bonusunu açma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `ucb` / `greedy`. Rejim: `recurring_reversal`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-11 — Tekrarlı terslenme: Sabit rastgele sapma

Durum: **İlk dalga**.

Öngörü: Tekrarlı terslenme koşulunda epsilon, greedy karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Gözlenen en iyi eylemden yüzde10 sapma

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `epsilon` / `greedy`. Rejim: `recurring_reversal`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-12 — Tekrarlı terslenme: Yakın deneyim penceresi

Durum: **İlk dalga**.

Öngörü: Tekrarlı terslenme koşulunda window_ucb, ucb karşısında bütün-yaşam beklenen ödül açığını azaltır.

Müdahale: Bütün-geçmiş istatistiğini son64 adımla değiştirme

Yanlışlama: Eşlenmiş doğrulama yaşamlarında pratik eşik üstünde ters fark veya destek koşullarının oluşmaması.

Bilinen aile / önceki çalışma: Bilinen bandit seçimi/unutma ailesi; dört rejimde üç mekanizmanın koşullu iddiaları, yeni ilke değil.

Aday / kontrol: `window_ucb` / `ucb`. Rejim: `recurring_reversal`. Ölçüt: `mean_pseudo_regret_512`. Pratik eşik: 0.01.

### ENV-13 — Ödül yerine bilgilendiricilik

Durum: **Henüz sınanmadı**.

Öngörü: Gözlem değerine göre eylem seçimi, aynı eylem bütçesinde sonraki görev öğrenimini yalnız anlık ödül seçimine göre iyileştirir.

Müdahale: Eylem hedefini sonraki öğrenilebilirliğe bağla.

Yanlışlama: Sonraki görev başarısı eşit veya kötü kalır.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-14 — İç durumdan deterministik sapma

Durum: **Henüz sınanmadı**.

Öngörü: Kanıta bağlı iç durum değişimi, eşlenmiş eylem çeşitliliğine sahip rastgele sapmadan daha çok ayırt edici deneyim üretir.

Müdahale: Sapmayı üreten iç kuralı değiştir; ziyaret sayısını eşle.

Yanlışlama: Kazanç ziyaret edilen veri miktarı eşitlenince kaybolur.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-15 — Kendi verisinin yanlılığı

Durum: **Henüz sınanmadı**.

Öngörü: Seçim olasılığını saklamak, politika değişince geçmiş veriden öğrenilen kuralın yanlılığını azaltır.

Müdahale: Aynı olayları seçim olasılığı bilgisiyle ve bilgisiz güncelle.

Yanlışlama: Düzeltme hata/varyans toplamını düşürmez.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-16 — Gecikmiş keşif yararı

Durum: **Henüz sınanmadı**.

Öngörü: Gecikmiş sonuçlara kredi veren seçici, kısa ufuk seçiciye göre geç beliren beceriyi daha sık edinir.

Müdahale: Kredi ufkunu değiştir; toplam gözlem maliyeti aynı.

Yanlışlama: Fark sadece fazladan gözlemden gelir veya tersine döner.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-17 — Gözlem kanalı seçimi

Durum: **Henüz sınanmadı**.

Öngörü: Sensör seçimini öğrenmek, sabit sensör karışımından daha düşük ölçüm maliyetiyle ayırıcı temsili kurar.

Müdahale: Kanal seçicisini edinilmiş/sabit kullan.

Yanlışlama: Aynı maliyette yeni ayrım başarısı artmaz.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-18 — Karşıolgusal sınama

Durum: **Henüz sınanmadı**.

Öngörü: Model uyuşmazlığını hedefleyen müdahale, aynı bütçeli pasif akıştan daha güvenilir mekanizma ayrımı sağlar.

Müdahale: Aday uyuşmazlığına göre müdahale üret.

Yanlışlama: Yeni müdahalede yanlış model oranı azalmıyor.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-19 — Yanlış merak döngüsü

Durum: **Henüz sınanmadı**.

Öngörü: Tahmin hatasını ödüllendiren iç hedef, indirgenemez gürültüde öğrenilebilirliği hedefleyen seçiciden daha kötü kaynak kullanır.

Müdahale: Gürültü ile öğrenilebilir belirsizliği ayıran hedefi aç.

Yanlışlama: Gürültüde zaman kaybı veya sonraki beceri kaybı farkı yok.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.

### ENV-20 — Deneyim üretmenin maliyeti

Durum: **Henüz sınanmadı**.

Öngörü: Edinilmiş eylem yordamının sonraki öğrenme yararı, edinim ve çevre maliyeti birlikte sayıldığında da sürer.

Müdahale: Yalnız sonraki skor yerine bütün yaşam maliyetini ölç.

Yanlışlama: Toplam maliyette avantaj kaybolur.

Bilinen aile / önceki çalışma: Aktif öğrenme/deney tasarımı/öğrenme ilerlemesi literatürüyle ayrıntılı eşleştirme bekliyor; yenilik iddiası yok.
