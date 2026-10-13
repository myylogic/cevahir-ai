# Sürekli çalışan ve deneyimiyle değişen sistem: araştırmanın ana hedefi

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Araştırma hedefi ve açık ölçütler

Bu metin ana hedefi koruyan bir araştırma modülüdür. Ürün entegrasyonu veya belirli bir mimari önerisi değildir. Tur 14'ün ana ve aktif deney sonuçlarından bağımsız olarak yazılmıştır; bu deneylerin sonuçları kendi raporlarında değerlendirilmelidir.

## Kullanıcının tarif ettiği hedef

İstenen sistem, çalışmaya başladıktan sonra yalnız önceden öğrendiği bilgiyi kullanarak cevap üreten bir program değildir. Çalışırken karşılaştığı olayları, kendi eylemlerinin sonuçlarını ve başlattığı araştırmaların bulgularını kullanarak gelecekteki hesaplama ve davranışını değiştirebilmelidir. Yeni öğrenilenler sonraki çalışmaya taşınmalı; sistem zamanla hangi bilgiyi nasıl kullanacağını, hangi ayrımları yapacağını ve neyi araştırmasının yararlı olacağını geliştirebilmelidir.

Hedef böyle anlaşılmaktadır: **Hizmet veren, deneyim edinen, deneyimini işleyen ve sonraki işleyişini bu deneyim nedeniyle değiştiren süreç aynı programın yaşamı içinde süreklilik taşımalıdır.** Her yeni edinim için araştırmacının dışarıdan yeni bir eğitim veri kümesi kurmasına ve sistemi yeniden başlatmasına bağımlılık azaltılmalıdır.

Eğitim ve inference burada aynı programın yaşamı içindeki iki işlevdir. Inference mevcut durumdan bir çıktı üretir; öğrenme sonraki çıktıları veya sonraki öğrenmenin biçimini etkileyen durumu değiştirir. Bunlar dönüşümlü, farklı zaman ölçeklerinde veya uygun bir yürütme düzeninde eşzamanlı gerçekleşebilir. Çalışma sırasında gradyan hesaplamak ilkesel olarak bu tanıma uygundur; öğrenmenin yalnız gradyanla yapılması şart değildir. Bu metin böyle bir sürekli yürütme sisteminin zaten uygulandığını söylemez.

“Sürekli” sözcüğü her token'da ağırlık değiştirmek, her deneyimi anında kalıcılaştırmak veya bütün eski kayıtları sonsuza kadar tutmak zorunluluğu getirmez. Kalıcılaştırma daha sonra da yapılabilir. Gerekli olan, öğrenmenin çalışan sistemin yaşam döngüsünün parçası olması ve kabul edilen değişimin sonraki işleyişte kullanılabilmesidir.

## Sınanabilir bir hesaplama çerçevesi

Belirli modülleri veya veri yapısını peşinen dayatmadan, roller şöyle yazılabilir:

\[
(a_t,r_t)=A(S_t,o_t;B_t),
\qquad e_t=(o_t,a_t,o_{t+1},f_t),
\]

\[
\widetilde S_{t+1}=U(S_t,e_t),
\qquad S_{t+1}=C(S_t,\widetilde S_{t+1};V_t,B_t).
\]

`Sₜ` öğrenme açısından ilgili mevcut durum; `oₜ` gözlem; `aₜ` çevreye yönelik eylem veya bilgi edinme seçimi; `rₜ` mevcut işe verilen çıktıdır. `eₜ`, gerçekten alınmış yeni deneyimi temsil eder; `fₜ` varsa dış geri bildirimdir. `Bₜ` kaynak bütçesini, `Vₜ` değişimi değerlendirmeye yarayan erişilebilir kanıtları gösterir. `U` aday değişimi, `C` bunun ne kadarının hangi koşulla çalışma durumuna alınacağını temsil eder. Bunlar zorunlu ayrı yazılım modülleri değildir; aynı algoritmanın farklı işlevleri olabilir.

Kabul edilen durumun uygun biçimde kalıcı saklanması ayrıca gereklidir. Yeniden yükleme sınaması, yalnız aynı cevabın verilmesini değil, aynı sonraki deneyim karşısında öğrenmenin de korunmasını kontrol etmelidir. Çünkü aynı bugünkü tahmini üreten iki iç durum, yarın aynı deneyimden farklı şeyler öğrenebilir.

Bu denklemler tek başına bir çözüm veya öğrenme kanıtı değildir. Önemli soru, `U`, `C` ve deneyim seçiminin **hangi koşullarda** yeni durumun yararlı, genellenebilir ve korunabilir olmasını sağladığıdır. Güncelleme hiç işe yaramıyorsa, yalnız kayıt biriktiriyorsa veya yeni öğrenme eski çalışma kapasitesini sürekli yok ediyorsa hedef karşılanmış olmaz. Öte yandan bir bellek mekanizması yeni durumlara taşınabilen yetkinlik sağlıyorsa, sırf parametre güncellemesi yapmadığı için öğrenme kapsamı dışında bırakılamaz.

Kendi deneyimini edinme döngüsünde `aₜ`, mevcut bilgiden ve belirsizlikten etkilenir; çevreden alınan sonuç ise bu bilgiyi değiştirebilir. Sistemin kendi ürettiği cevabı dış kanıt yerine koyması yeterli değildir. Ayrıca araştırma yalnız daha iyi sorgu seçmeye indirgenmez: temsilin değişmesi, yeni becerilerin oluşması, belleğin düzenlenmesi, eski yetkinliklerin korunması ve yeni araştırma amaçlarının geliştirilmesi de ana problemin parçalarıdır.

## Hedef, mevcut araştırmanın kapsamı ve açık ölçüt

| Kullanıcının hedefi | Mevcut kanıt veya sınamanın kapsamı | Açık başarı ölçütü |
|---|---|---|
| Çalışırken öğrenmek | Önceki sonlu deneyler gözlem geldikçe durum güncellemesini ve devamını inceliyor. | Çıktı üretimi sürerken öğrenmenin uzun bir yaşam akışında işlemesi; zaman ve bellek maliyetinin ölçülmesi. |
| Yeni deneyimini kendisi edinmek | Tur 14'ün ana kurulumu pasif. Aktif ek protokol sonraki sorguyu öğreniciye seçtiriyor. | Hangi deneyimin hangi araçla üretileceğini gerçek erişim ve maliyet kısıtları altında seçmek; sonucun öğrenmeye katkısını göstermek. |
| Bağlamları kendisi oluşturmak | Hazır dört geçmiş biti ve hazır parite dili içinde bağlam ayrışması sınanıyor. | Ham olaylardan tutulacak geçmişi, yararlı ayrımları ve temsil biçimini öğrenmek; görünmeyen kimliklerden yararlanmamak. |
| Yeni bilgi ve beceri edinmek | Sınırlı dillerde çıkarım, aktarım ve kalıcı değişim için ayrı kontroller bulunuyor. | Sadece verilen adaylar arasında seçim yapmanın ötesinde, yeni kullanılabilir yapılar oluşturmak; yararı yeni görevlerde ölçmek. |
| Yeni öğrenme amaçları geliştirmek | Mevcut görev hedefi ve başarı ölçüsü araştırmacı tarafından veriliyor. | Öğrenme ihtiyacını deneyimden çıkarmak, alt görev önermek ve seçmek; başarıyı yalnız kendi ürettiği ölçüyle ilan etmemek. |
| Belirsizliğini kullanmak | Aktif protokol hazır adayların tahmin ayrışmasını kullanıyor. | Eksik veya yanlış aday dili altında belirsizliğin güvenilirliğini sınamak; yüksek puanı gerçek bilgi kazanımıyla karıştırmamak. |
| Çalışan kapasiteyi koruyarak değişmek | Önceki turlar koruma, olumsuz aktarım ve kaynak sınırlamalarının ayrı sınırlarını kaydediyor. | Yeni kazanım ve eski beceri kaybını birlikte ölçmek; değişim sürecindeki çalışma hatalarını da hesaba katmak. |
| Öğrenilenlerin yaşamını sürdürmesi | Sonlu deneylerde durum saklama ve yeniden devam kontrolleri kullanılıyor. | Ham deneyimlerin bir kısmı kaldırıldığında dahi gerekli becerilerin ve sonraki öğrenme davranışının sürmesi. |
| Kendi kendine ilerlemek | Tek bir kontrol mekanizmasının evrensel ve kesintisiz ilerleme garantisi gösterilmiş değil. | İlerlemeyi dış sonuçlarla değerlendirmek; ölçütü değiştirerek veya kolay örnekleri seçerek görünür başarı üretmesini ayırmak. |

“Kendi kendine” burada bütün başlangıç bilgisinin, araçların ve amaçların yoktan oluşması anlamına getirilmemelidir. Başlangıçta ne verildiği açıkça sayılmalı; sonradan deneyim nedeniyle neyin oluştuğu ayrıca gösterilmelidir. Özellikle mevcut aktif protokolde araştırmacı, bütün sonlu girdilerin sorgulanabildiği ve doğru etiketi döndüren bir arayüz sağlar. Sorguyu öğrenicinin seçmesi bu ayrıcalığı ortadan kaldırmaz.

## Sürekli öğrenme ile sürekli iyileşme aynı iddia değildir

Programın çalıştığı her an bütün görevlerde daha iyi olması, yalnız bir öğrenme döngüsü kurmaktan çıkmaz. Örneğin yeni etiket geçmişten ve gözlenen girdiden bağımsız, eşit olasılıklı bir bitse, geçmiş deneyim o etiketin hangisi olacağını belirleyemez. Bu karşı örnek öğrenmenin yararsız olduğunu söylemez; deneyimden çıkarılabilir yapı bulunmasının önemini gösterir.

Aynı şekilde sabit ve sonlu bellek, sınırsız sayıdaki keyfî geçmişin tüm ayrıntılarını hatasız saklama garantisi veremez. Yararlı bir sistem hangi izleri koruyacağını, hangilerini birleştireceğini ve gerektiğinde hangi yeni kanıtı arayacağını düzenlemelidir. Belleğin büyümesi mümkünse bunun kaynak maliyeti ayrıca hesaba katılmalıdır.

Bu nedenle araştırmanın doğru hedefi, “her adımda otomatik olarak daha zeki olma” iddiası yerine, **ilerlemenin ortaya çıktığı koşulları, kalıcılığını, bedelini ve başarısızlık sınırlarını açıklayan bir sürekli öğrenme ilkesi** bulmaktır. Bu ilke; bağlam seçme, uyku benzeri bütünleştirme, aktif sorgulama veya mevcut bir öğrenme yöntemiyle peşinen özdeşleştirilmez.

Tur 14 ve [aktif deneyim protokolü](ACTIVE_EXPERIENCE_PROTOCOL.md), bu geniş hedefin iki sınırlı bileşenini sınar. Ana hedef korunmaktadır; sürekli çalışan genel sistemin tamamlandığı veya beynin çalışma ilkesinin kanıtlandığı ileri sürülmemektedir.
