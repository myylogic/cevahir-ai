# Yeni deneylerin karşı açıklamalarla denetimi

20 Eylül 2026. Araştırmanın erken aşamasında bağımsız incelemeden gelen kontroller ve kök araştırmacının kod/sonuç denetimi. Deneyler genel problemin çözümü sayılmaz. Bu belge önceki kayıtların yerine geçmez.

| Denetlenen alternatif açıklama | Yapılan kontrol ve sonuç |
|---|---|
| Görev kimliği başka adla verilmiş olabilir. | Ortak öğrenicinin `observe(x,z,y)` erişimi ve kaydedilen durum alanları incelendi. Görev/evre/dünya kimliği yok; **uygun uzamsal ızgara ve özellik dili hazır verilmiş**. Bu daha dar önbilgi açıkça korunur. |
| Doğru son skor pahalı edinimi gizliyor olabilir. | Bütün yaşam servis hatası ve dönem hataları raporlandı. Büyüyen yöntem güçlü sabit bölgesel yöntemi bu ölçütte geçmedi; daha fazla doğrusal çözüm yaptı. |
| Büyüme ancak zayıf sabit rakibi geçiyor olabilir. | Beş doğru aday özelliği baştan kullanan, aynı belleğe/sıklığa sahip karşılaştırma var. Aynı son riskler daha az aramayla elde edildi. Mimari üstünlük iddiası reddedildi. |
| Yanlış etiketler gelecekte gereken özelliği önceden sağlamış olabilir. | 16/24 erken etkinleşme görüldükten sonra yalnız yanlış etiket dizisi çıkarılan ek kontrol yapıldı. Son beceri aynı; ilk uyum biraz daha yavaş. Bu kontrol sonuç görüldükten sonra tasarlandığı için bağımsız doğrulama diye sunulmadı. |
| Bölgesel koruma, görünmeyen bölgenin değişmediğini biliyor olabilir. | Yerel/global değişim dünyaları, yalnız sağın görüldüğü önekte aynı kanıtı üretirken solda farklı hedeflere sahip. Koruma o bölgede yanlış olabilir; örnekteki yerellik varsayımı ortaya çıkarıldı. |
| Hazır ID yokluğu, hiçbir zamansal önbilgi yokluğu gibi sunulabilir. | Sensör deneyinde sıra, örnek saati ve 0–6 gecikme sözlüğü verilmiş. Bu anonim ve bilinmeyen permütasyonlu etiket eşlemesi deneyi değildir. |
| Geçmiş korelasyonu nedenin bulunduğu sanılabilir. | Dönüşümlü akış iki farklı gecikme açıklamasını birleştiriyor. Ayrıca iki yapısal dünya aynı gözlemleri fakat farklı müdahale sonuçlarını üretiyor. Nedensel keşif iddiası yok. |
| Yalnız cevapların saklanması öğrenme durumunun sürdüğünü göstermeyebilir. | Bütün güncelleme durumu ve yeni akışla devam karşılaştırıldı; yalnız katsayı korunup tarih matrisi sıfırlanınca aynı ilk cevaplar farklı gelecek güncellemelere dönüştü. Ayrı süreç denetimi eklendi. |
| Kullanılan istatistik hatalı bir RLS uygulaması olabilir. | 32 güncelleme adımında farklı bir doğrudan ridge çözücüsüyle karşılaştırma eklendi; yalnız aynı RLS yordamını yeniden çalıştırmakla yetinilmedi. |
| Özellik sayaçları gerçek hızlanma sanılabilir. | `features` yordamı tüm aday değerleri hesaplıyor; sayaç bazı yerlerde yalnız seçilen vektör bileşenlerini sayıyor. CPU süresi veya tam işlem tasarrufu sonucu çıkarılmadı. |

İlk ayrı-süreç denetimi başarısız oldu. Nedeni deneyin farklı süreçte farklı öğrenmesi değildi: denetleyici, beklenen devamı hesaplarken aynı liste nesnelerini kullanan durum kopyasını değiştirmiş, sonra değiştirilmiş bu durumu yeniden başlatma girdisi olarak yazmıştı. Denetleyicide JSON üzerinden ayrık kopya oluşturuldu; deney kaynakları ve mevcut sonuçlar bu yüzden değiştirilmedi. Nihai [doğrulama kaydı](../../../research/living_learning_joint_2026_09_20/results/verification.json) düzeltilmiş denetleyicinin sonucunu içerir. Bu hata, yalnız “restore edildi” demek yerine gerçekten aynı başlangıç durumunu karşılaştırmanın önemini gösterir.

**Hâlâ sınanmayanlar:** Bölgesel ızgaranın deneyimden edinilmesi; dönüşümlü müfredat dışındaki bütün sıralar; bağımsız görevler arasında öğrenme kuralının gelişmesi; gözlemleri eylemle üreten bir ortak sistem; sınırsız gecikme ve bilinmeyen sensör dili; yanlış etiket ile gerçek değişimin güvenilir ayrımı. Ayrı kaynaklara sahip iki yeni düzeneğin başarıları tek genel sistem başarısı olarak toplanamaz.
