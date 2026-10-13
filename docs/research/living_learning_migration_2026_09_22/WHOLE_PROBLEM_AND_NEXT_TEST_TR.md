# Ana problem denetimi: süreklilik bulgularından sonra açık kalanlar

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Tur 16 sonrası araştırma yönü

Bu belge tamamlanmış yeni bir deney raporu değildir. Tur 16 sonuçlarından sonra araştırmanın ana problemden dar bir alt probleme kaymasını önlemek için yazılmış değerlendirme ve sonraki sınama taslağıdır. Önceki kayıtları değiştirmez.

## Ana soru değişmedi

Bir yapay sistem, çalıştığı süre boyunca yaşadığı deneyimler nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir biçimde nasıl değiştirebilir?

Son çalışmalar bu sorunun çeşitli gerekliliklerini ayırdı; birleşik cevabını oluşturmadı. Bir yaşam boyunca hata ölçüsünün tekdüze azalması ne deneylerden çıktı ne de varsayımsız vaat edilebilir. Dünyanın değişmesi, geri bildirimin eksikliği ve birbiriyle çatışan görevler hangi ilerlemenin ölçüldüğünü belirtmeyi gerektiriyor.

| Araştırılan bölge | Elde kalan katkı | Hâlâ gösterilmeyen |
|---|---|---|
| Program parçalarının edinimi | Geçmiş çözümler sonraki aramayı değiştirebilir; olumsuz aktarım mümkündür | Parça çıkarma ve değerleme yordamının kendisinin edinimi |
| İlişkilerin çapraz bağlanması | Verilen ilişki dili içinde yeni bileşimlere aktarım | Nesne/ilişki dilinin ve bağ türünün oluşumu |
| Birleştirme ve kalıcılık | Ham olay saklama, kalıcı kural ve tekrar oynatma aynı şey değildir | Genel bellek bütçesi ve çok becerili uzun yaşam |
| Bağlam ve deneyim erişimi | Aynı sonuçları açıklayan bilgileri ayırmak için farklı deneyim gerekir | Görev, hedef ve geri bildirim arayüzünün edinimi |
| Akıştan bellek kullanımı | Zaman ufku, durum kapasitesi ve öğrenme istatistikleri ayrıdır | Kullanılabilir bellek dilinin oluşturulması |
| Çalışan duruma yeni kural eklemek | Kuralın doğruluğu, doğru başlatma ve eski beceriyi koruma ayrıdır | Sistemce seçilen temsil dönüşümü ve kaynak paylaşımı |

Hiçbir satır tek başına yeni ana problem olarak kabul edilmemeli. Özellikle son iki turdaki otomata örnekleri, genel yaşayan öğrenmenin mutlaka otomata öğrenimi veya senkronizasyon demek olduğu sonucunu vermiyor.

## Başarıyı artık hangi müdahale ayırmalı?

Yeni aday soru: Deneyim, yalnız hangi çözümün saklanacağını değil, yeni kanıt geldiğinde hesap yapısının **nasıl değiştirileceğini** de kurabilir mi?

Başlangıç için öğrenilmiş bir düzeltme yordamını

\[
r:(p,E,\tau)\longmapsto p'
\]

olarak düşünelim. `p` mevcut aday hesap, `E` sistemin erişebildiği girdi–çıktı kanıtı, `τ` kendi yürütme izi, `p'` yeni adaydır. Bu gösterim bir buluş veya varlık kanıtı değildir. Düzeltme yordamı da sıradan bir programdır; sabit evrensel yorumlayıcıyla yürütülebilir. Bu nedenle iddia, hesaplanabilir fonksiyon sınıfını genişletmek olamaz. Sınanabilecek iddia, yaşanmış deneyimin belirli kaynak bütçesinde ulaşılabilir edinimleri değiştirmesidir.

Önerilen küçük alan, sonlu listelerde temel işlemlerden bileşik hesaplar kurmaktır. Başlangıç dili ve sonlanmayı sağlayan yapısal kısıtlar açıkça verilmeli; hazır düzeltme kimliklerinin listesiyle yetinilmemeli. Öğrenici kendi başarısız adaylarından ve sonra kabul ettiği çözümlerden, değişken bağlayan tekrar kullanılabilir dönüşümler kurmalı. Öğretmenin kaynak kodu veya doğru onarım verilmemeli.

En kritik nedensel karşılaştırma: Aynı eski çözümleri ve aynı başlangıç cevaplarını koruyan iki durum kopyası alınsın; birinden yalnız edinilmiş düzeltme yordamları çıkarılsın. Yeni bileşimlerde bu yordamlar açıkken daha fazla doğru görev ediniliyorsa, geçmiş deneyimin sonraki öğrenme sürecine etkisi gösterilmiş olur. Etki başka bellek veya daha fazla deneyimle karışıyorsa bu çıkarım yapılamaz.

## Güçlü karşılaştırmalar ve başarısızlık ölçütleri

Asıl karşılaştırma geçmişsiz kör arama olmayacak. Aynı eski çözümleri ve yürütme izlerini alan; çözüm kütüphanesi ile göreve bağlı aday/tek-adım düzenleme önceliklerini öğrenen bir sentezleyici gerekli. Öğrenilmiş yordamları aynı sırayla çalıştıran sabit yorumlayıcı da eşdeğerlik kontrolü olmalı: bunun eşleşmesi beklenir.

Birincil ölçü eşit bütün-yaşam işlem ve ayrı bellek bütçesinde, görülmemiş görevlerdeki doğrulanmış edinimdir. Düzeltme yordamını üretme, başarısız deneme ve doğrulama maliyetleri hesaba katılmalı. Gelecekte kazanılan hızı sayıp geçmiş üretim maliyetini gizlemek gelişme kanıtı değildir.

Önceden konulacak karşıörnekler: aynı kısa kanıtı açıklayıp uzun girdide ayrılan programlar; eski düzeltmenin yeni ailede yanlış yönlendirmesi; doğru sonucu yanlış iç değişiklikle tesadüfen elde etme; aynı anlamın farklı sözdizimi; üretim maliyetini karşılayamayacak kadar kısa kalan yaşam. Öğrenilen yordam yalnız eski programları geri çağırıyorsa veya aynı kazanç güçlü çözüm kütüphanesiyle tamamen açıklanıyorsa yeni edinim-yordamı etkisi desteklenmez.

Bu taslak henüz tam protokol değildir. Dil, işlem sayımı, kabul kanıtı, görev ayrımı ve aynı bilgiye erişim koşulları sonuç görülmeden somutlaştırılmalıdır. Henüz bu mekanizmanın çalıştığı iddia edilmez.

## Yenilik denetimi

“Öğrenme algoritmasını öğrenmek” veya “kendi hatasından program düzeltmek” tek başına yeni ana fikir değildir:

- [AutoML-Zero, Real ve diğerleri, 2020](https://proceedings.mlr.press/v119/real20a.html), temel matematik işlemlerinden öğrenme algoritmaları arar. Resmî özet ve bildiri kaydı kontrol edildi.
- [Synthesize, Execute and Debug, Gupta ve diğerleri, 2020](https://arxiv.org/abs/2007.08095), yürütme sonuçları ve belirtime göre programları tekrar tekrar onaran öğrenilmiş bir düzeltici kullanır. Yazar kaydı ve özet kontrol edildi.
- [SOAR, Pourcel, Colas ve Oudeyer, 2025](https://proceedings.mlr.press/v267/pourcel25a.html), arama girişimlerinden oluşturulan örneklerle sonraki program örnekleme ve düzeltme yeteneğini geliştirir. Resmî bildiri özeti kontrol edildi.

Bu kaynaklar bizim önerilen mekanizmamızın aynı olduğunu kanıtlamaz, fakat geniş yenilik cümlelerini geçersiz kılar. Ayrıntılı yöntem ve hesap bütçesi karşılaştırması yapılmadan “ilk” veya “yeni temel ilke” iddiası kurulamaz. Önceki turda kayda geçen DreamCoder ve kendini değiştirme/öğrenilmiş güncelleme çalışmalarının sınırları da korunur.

Yenilik yükü bir terimde değil, aynı erişim ve maliyet altında ayrılan öngörüde, onu açıklayan matematikte ve güçlü karşılaştırmaya dayanıklı sonuçtadır. Kullanıcının kavramların yeniden örgütlenmesi, çapraz bağlantı ve yaşam boyu gelişim penceresi bu daha geniş araştırma haritasında açık tutuluyor; bir biyolojik mekanizma kanıtı veya hazır mimari varsayımına çevrilmiyor.
