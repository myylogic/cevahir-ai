# Tur 17 — Deneyimden edinilen düzeltme şemalarının gelecekteki edinime etkisi

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Başarım sonuçlarından önce yerel protokol

## Ana problemdeki yeri

Tur 16 sonrası taslak, çözümlerin yanında düzeltme yordamlarının edinilmesini önerdi. Burada daha küçük ve denetlenebilir alan kullanılır: listeler yerine iki değişkenli aritmetik ifade ağaçları, F17 üzerinde toplama/çıkarma/çarpma. Bu seçim test edilen anlamı 289 girdide tam denetlemeyi sağlar. Liste öğrenimi, özyineleme icadı veya genel temsil dili oluşumu iddia edilmez. Önceki araştırma, kitap, ana sistem ve HEAD korunur; commit/push yok.

Hipotez: Kendi aramasından bulunan başarılı değişikliklerden çıkarılan değişken bağlı düzeltme şemaları, aynı eski çözümler ve aynı erişilebilir kanıt varken yeni daha derin hesapları edinmeyi kolaylaştırabilir. Güçlü karşı hipotez: Bu yarar, deneyimle sıralanmış genel toplu düzenleme işlemleriyle tamamen açıklanabilir. Her iki sonuç geçerli araştırma sonucudur.

## Hesap ve kanıt

İfadeler `x`, `y`, alan sabitleri ve ikili `add/sub/mul` düğümleridir. Sonlanma ağacın sonluluğundan gelir. Alan 17 elemanlıdır. Bir görev, hatalı aday ifade ve 16 farklı girdi–çıktı örneği verir. Bütün adaylar aynı 16 örnekle sınanır. Bunlara uyan ilk aday kabul edilir; gizli değerlendirme bütün 289 `(x,y)` girdisinde yapılır, öğreniciye geri verilmez. Kaynak kod kimliği doğruluk ölçüsü değildir; aynı alan fonksiyonunu hesaplamak doğrudur. F17 dışı genelleme ileri sürülmez.

Öğrenicinin öğretmenin doğru AST'sine, hedef rejime veya gizli tam tabloya erişimi yoktur. Adayın kendi AST'si ve yürütülmesinden çıkan hatalar erişilebilirdir. Deney görev sınırlarını ve örnek erişimini verir. Bir adayın 16 örneği geçmesi, 289 girdide doğruluğunun sertifikası sayılmaz; yanlış kabul ayrıca kaydedilir.

## Kaynak deneyimler ve yordam çıkarma

24 yaşam. Her yaşamda dört kaynak onarım görevi vardır. Kaynak hatası `(u+v)*w`, örnekleri üreten hesap `u*w+v` olur. `u,w` sırayla x/y ve y/x; `v` her kaynakta farklı, sıfır olmayan sabittir. Kaynakların çözümü öğretmen tarafından verilmez: öğrenici kendi genel yerel aramasıyla 16 örneğe uyan ilk adayı bulur.

Genel yerel düzenleme: bir düğümde iki katmanlık, üç altifadeli sınır açılır. Üç altifadenin altı permütasyonu, iki ikili parantezleme ve iki işlem için üçer seçenek denenebilir. Bu 108 ham yeniden kurma bir aday üreticisidir; daha derin altifadeler aynen taşınır. Düzenleme dilinin verilmiş olması açık önbilgidir.

Başarılı kaynak onarımında önce/sonra ifade çifti, kullanılan yerel düzenleme kaydı ve erişilebilir örnekler saklanır. Çiftler ikişer karşılaştırılır; ortak ağaç biçimini koruyan, farklı altifadeleri delik değişkenleriyle değiştiren bir genelleştirme yordamı uygulanır. Önce/sonra tarafında aynı altifade çifti aynı deliği kullanır. Sonuç tarafındaki her deliğin giriş deseninde bağlı olması gerekir. Başarılı değişiklik çiftlerinden bu şekilde yeni desen/yapım ağaçları oluşur; hazır onarım adı listesi seçilmez. Genelleştirme ve desen yorumlayıcısı verilidir; bunların icadı iddia edilmez.

Şemalar yeni hedeflere her eşleşen düğümde tekil veya bütün eşleşen düğümlerde tek bir alttan üste geçişle uygulanabilir. Oluşturulan düğümler aynı geçişte tekrar ziyaret edilmez; kendini döndüren sonsuz onarım döngüsü yoktur. Bütün öneriler aynı görev örneklerinden geçirilir. Şemalar doğruluğu koşulsuz koruyan cebirsel özdeşlik sayılmaz; bunlar hatalı programı değiştirmeye aday edinilmiş işlemlerdir.

## Beş eşleştirilmiş yöntem

1. `local_plain`: aynı kaynak çözümleri/kütüphanesiyle sabit sırada genel yerel düzenlemeler.
2. `local_ranked`: kaynak aramada başarılı olan yerel düzenleme kayıtlarını ilgili işlem/parantez bağlamında öne alır; ardından bütün genel yerel düzenlemeler.
3. `bulk_ranked`: aynı öğrenilmiş öncelikler; başarılı genel düzenlemeyi aynı işlem/parantez bağlamındaki bütün düğümlere uygulayan hazır toplu geçiş; ardından sıralı yerel arama. Güçlü kontrol: yeni desen ağacı çıkarımı olmadan aynı bağlamsal bilgiyi kullanır.
4. `learned_rules`: genelleştirmeden edinilen şemalarla tekli ve toplu öneriler; ardından aynı sıralı yerel arama.
5. `rules_disabled_paid`: learned_rules ile aynı kalıcı bilgi ve aynı edinim maliyeti; yalnız şema yürütmesi kapalı. Nedensel ablasyon. Pratik maliyet karşılaştırması ayrıca local_ranked/bulk_ranked ile yapılır; onlar kullanılmayan şema çıkarımı için ödemez.

Bütün yöntemler kabul edilmiş kaynak/önceki hedef çözümlerini önce kütüphaneden deneyebilir. Hedeflerde yeniden şema çıkarılmaz; bu tur gelecekteki düzeltme yordamının kaynaktan transferini izole eder. Öğrenme akışında yeni kabul edilen çözümler kütüphaneye eklenir; arama öncelikleri kaynak deneyiminden sonra dondurulur.

## Hedef rejimleri ve arama

Her yaşamda sekiz hedef; onarılacak düğüm sayıları `1,2,4,1,2,4,2,4`. Altifadelerin derinliği sırasıyla `1,2,3,1,2,3,2,3`. Üç rejim aynı hatalı adayları paylaşır: uyumlu rejimde bütün ekilmiş hata noktaları kaynak dönüşümünü ister; değişmiş rejimde `(u+v)*w` noktaları `(u+v)+w` gerektirir; kısmi rejimde son ekilmiş nokta doğru olduğu gibi bırakılır, diğerleri kaynak dönüşümünü ister. Tek noktalı kısmi görev dolayısıyla baştan doğru kontroldür. Hedef doğru ağaçları yalnız dünya/evaluator tarafında tutulur. Rastgele altifadeler tesadüfi benzer bağlamlar içerebilir; bunlar toplu onarım için ayrıca yanlış değişiklik riski yaratır.

Yerel arama en çok sekiz tur yapar. Önce adayın ve kütüphanenin kanıt hatası ölçülür. Her tur önerileri sırasıyla sınar; sıfır hata varsa kabul eder, yoksa bütün tur tamamlanmışsa kanıt MSE'sini kesin azaltan en iyi adaya geçer. İyileşme yoksa durur. Tur ortasında bütçe biterse o turun en iyi ara adayıyla devam edilmez. Bu sınırlı açgözlü arama tam sentez garantisi değildir. F17 çıktıları için kanıt hatası sıradan sayısal farkın karesidir; sınıra dayalı cebirsel uzaklık değildir ve bu arama yanlılığı bütün yöntemlerde aynıdır.

24 × 3 × 5 =360 hedef öğrenme yolu; her yolda sekiz hedef, toplam 2.880 hedef denemesi. Kaynak deneyimler yöntemler arasında eşleşir. Bunlar 360 bağımsız yaşam değildir.

## Bütün yaşam maliyeti

Her yöntemin toplam bütçesi 1.000.000 soyut iş birimi, hedef başına en çok 120.000 ek birimdir. Kaynak arama ve durumu kurma bütçeye dahildir. İfade değerlendirmesi `düğüm sayısı × örnek sayısı`; ağaç dolaşma, sanal yeniden yapım/kopyalama, desen eşleştirme ve genelleştirmede ziyaret edilen/yazılan düğümler ayrıca sayılır. Tekli düzenlemede yeni adayın bütün düğümleri sanal kopya olarak; toplu geçişte dolaşılan iç düğümlerin yeniden kurulması ve her eşleşmenin oluşturduğu alt ağaç sayılır. Değişmez alt ağaçlar Python'da paylaşılabildiğinden bu literal tahsis sayısı değildir; yöntemlerin bu açık sanal işlem modeli altındaki maliyetidir. Şema doğrulama taramaları, değişken bağları için eşitlik kontrolleri ve yapım öncesi boyut planlaması da ücretlidir. Başarısız öneri ve şema çıkarımı ücretsiz değildir. Bütçeye sığmayan atomik işlem başlamaz; çok aşamalı dolaşmanın tamamlanmamış ara işi harcanmış olarak kalabilir. Python anahtar sıralama/hashing/yazma, düğüm-sayısı önbelleği ve işletim sistemi ek yükü bu soyut modele dahil değildir; gerçek CPU/RAM eşitliği iddia edilmez. Kalıcı durum JSON boyutu ayrıca kaydedilir.

Gizli 289-girdi doğrulaması araştırma ölçümüdür; öğrenici bütçesine katılmaz ve aday sırasını değiştirmez. Hedef başına kabul, gizli tam doğru, yanlış kabul, harcanan iş, aday değerlendirmesi, şema önerisi ve toplam bütçe kaydedilir. Kısa kalan yaşamda üretim bedelinin geri alınmaması ve uyumsuz rejimde olumsuz aktarım aranır. Sonuçlara göre bütçe veya görev ailesi değiştirilmeyecek.

## Denetim ve yorum sınırı

Bağımsız skaler AST değerlendiricisi; küçük örnekte değişken bağlama ve ortak delik denetimi; RHS serbest değişken yasağı; toplu geçiş sonlanması; etiket/test erişimi; kaynak aramanın öğretmen kodu almaması; kalıcı durumdan yeni süreçte aynı hedef devamı; tam yeniden çalıştırma; bütün önceki dosyaların SHA256 korunması.

Getafix, REFAZER ve program sentezinde öğrenilmiş düzeltme literatürüyle eşleşme ayrı kaydedilecek. Olumlu sonuç bile yalnız verilen dilde deneyimden edinilmiş bir değişiklik şemasının hesap bütçesi altında transferidir. `bulk_ranked` aynı veya daha iyi sonuç verirse “yeni şema çıkarımı gerekli temel ilke” iddiası desteklenmez. Ana yaşayan öğrenme problemi bu program onarım deneyine indirgenmez.
