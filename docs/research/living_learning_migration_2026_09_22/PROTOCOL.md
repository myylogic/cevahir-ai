# Tur 16 — Öğrenilmiş kuralı kesintisiz akışa katmak

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Başarım ölçümlerinden önce yerel protokol

Tur 15'in bölümler arasında sıfırlama kolaylığı kaldırılır: değerlendirilen ana bit akışı baştan sona devam eder. Yeni bir kuralın deneyimden öğrenilmiş olması, o kuralı akışın şu anındaki doğru iç durumdan başlatmayı garanti eder mi? Bu dar soru incelenir; temsil dilinin icadı veya otonom genel öğrenme çözüldü sayılmaz.

## Ön bilgi ve öğrenme

Eski yetkinlik bütün bitlerin parity'sidir. Doğru iki-durumlu geçiş/çıktı tablosu, 64 aday arasından 128 etiketli kısa bölümle öğrenilir. Yeni yetkinlik ya sondan beşinci bit ya da toplam bir sayısının üçe göre kalanının bir permütasyonla okunmasıdır. Yeni aday dili sekiz gecikme kuralı (1–8) ve üç kalanın altı farklı okuma tablosundan oluşur: 14 aday. Bunlar hazır bir dildir; yeni hesaplama ilkelleri türetilmez.

Yeni kural, ayrı bir kalibrasyon kanalından gelen 128 etiketli bölümle öğrenilir. Bu bölümler ana akışın geçmişi değildir. Kalibrasyon bölümleri sıfırdan başlayan referans örneklerdir; bunların erişilebilirliği verilmiştir. Deney bu kanalın otonom olarak bulunmasını sınamaz. Eski ve yeni kurallar kalibrasyonda önce tahmin eder, sonra etiket alır. Tutarlı adaylar eşit ağırlıklı; sonunda en küçük numaralı tutarlı aday derlenir. Aday sayısı ve doğru programla eşleşme raporlanır; görev adı derleyiciye verilmez. Aday yoksa hata olarak durur, gizli doğru kural kullanılmaz.

24 yaşam. Her tohumda kalibrasyon girdileri görevler arasında aynıdır: uzunluk 8–24, bağımsız eşit olasılıklı bitler. Ana akış bundan ayrı rastgele üretilir. Eski kural sıfır başlangıçtan ana akış boyunca çalışır. Yeni kural ana akışın 256 veya 4096 bitinden sonra katılır. Yeni modulo okuma permütasyonu tohumun altıya göre kalanıyla seçilir; öğreniciye bu ilişki verilmez.

## Altı durum taşıma yöntemi

1. `guess_zero`: eski durumu korur; yeni kuralın iç durumunu sıfır sanır.
2. `possible_states`: eski durumu korur; yeni kuralın bütün olası iç durumlarını taşır.
3. `reserve_8`: başlangıçtan beri son sekiz ham biti yardımcı iz olarak tutar. Gecikme kuralını bundan tam başlatır. Modulo kuralı için bütün olası durumları taşır.
4. `reserve_mod6`: başlangıçtan beri bir sayısının altıya göre kalanını tutar. Üçe göre kalan buradan belirlenir; gecikme için bütün durumlar taşınır. Hangi ek sayacın tutulacağı önbilgidir; deneyimden seçilmez.
5. `archive`: bütün eski bitleri tutar; yeni öğrenilmiş programla yeniden yürüterek doğru durumunu bulur. Kurulumdan sonra ham arşivi bırakır. Bu güçlü ve daha pahalı karşılaştırmadır.
6. `reset_all`: hem eski hem yeni çalışma durumunu sıfırlar; eski beceri kaybı için negatif kontrol. Gerçek çevre sıfırlanmaz.

Başlatmada `possible_states` için bütün yeni durumların seçilmesi bu özel ailelerde eski parity ile uyumludur: en az 256-bit geçmişte her parity altında her beş-bit sonek ve her modulo-3 kalanı mümkündür. Bu, rastgele başka model çiftleri için genel bir taşıma algoritması değildir. `reserve_8`'in modulo belirsizliği ve `reserve_mod6`'nın sonek belirsizliği de bu uzunluklarda mümkündür. Program kimliği belirsizliği ile çalışma durumu belirsizliği ayrı kaydedilir.

Yeni bit her olası duruma uygulanır, aynı duruma gidenler birleşir. Tahminde farklı olası çıktı değerlerine eşit olasılık verilir; bu muhafazakâr seçimdir, gerçek geçmişlerin koşullu olasılık hesabı olduğu iddia edilmez. Singleton durum kesin çıktı üretir.

## Etiket sırası ve sınama

Yeni kural katıldığı anda `t=0` çıktısı sorulur; sonra 64 yeni bit boyunca her bitte eski ve yeni çıktı sorulur. Ana çevre sıfırlanmaz. Bu 65 andaki değerlendirme etiketleri öğreniciye verilmez. Yalnız `t=16` anının yeni çıktı etiketi, o andaki tahminden **sonra** bir kez verilir. Eski parity etiketi ana akış boyunca verilmez.

Yeni etiket mevcut olası durumları filtreler. Yanlış singleton tahmini nedeniyle küme boşalırsa, programın bu etiketi veren bütün durumlarına genişletilir; onarım sayısı kaydedilir. Bilgi akışında bu açık bir dış gözlemdir. Başarının yeni bitlerin belirsizliği azaltmasından mı, bu tek etiketten mi geldiği ayrılır. `t=17` sonrası değerlendirme ikinci durumu ölçer.

24 × 2 prefix uzunluğu × 2 yeni görev × 6 yöntem =576 yol, her yolda 65 çift yetkinlik tahmini. Öğrenme örnekleri aynı tohum için yöntemler arasında paylaşılır; 576 bağımsız veri yaşamı değildir. Eski/new kuralların kalibrasyonuna tüm yöntemler aynı erişime sahiptir; ana akıştaki saklama maliyetleri farklıdır.

Yeni görevde üç sınıflı Brier toplamı kullanılır: doğru kesin cevap 0, yanlış kesin cevap 2; iki olası çıktı arasında eşit olasılıkta 0.5; üçünde 2/3. Eski ikili görevin karesel hatası ayrı raporlanır. Olası kümenin gerçek durumu içerip içermediği, küme büyüklüğü, kesin ama yanlış cevap, başlangıç durumu, t=0–4/t=5–15/t=17–64 dönemleri, tampon/arşiv bitleri, yardımcı sayaç durumları, kurulumda tekrar işlenen bitler ve aday geçiş sayıları kaydedilir. Python nesneleri ve program kodu için bu bit ölçüleri toplam RAM iddiası değildir.

## Önceden beklenen ayırıcı sonuç

Gecikme kuralında başlangıç belirsizliği en çok beş yeni bitten sonra doğal olarak biter. Üçe göre kalan sayacında her bit bir permütasyon olduğundan, aynı girdiyi görmek eksik başlangıç kalanını vermez. Tek dış çıktı etiketi ise bu deneyin birebir okuma tablosuyla kalanı belirleyebilir. Yanlış sıfırlama eski parity bilgisini geri getirmez. Bunlar bilinen senkronizasyon/durum gözlemcisi fikirleriyle karşılaştırılacak; yeni ilke olarak pazarlanmayacak.

## Denetimler

Bağımsız skaler gerçek etiketler; küçük tam geçmiş uzayında taşıma lifleri; gecikmenin beş-bit senkronizasyonu ve modulo permütasyonunun belirsizliği koruması; aynı etiketlerde farklı hata tanımlarının karıştırılmaması; değerlendirmeden sızıntı olmaması; etiket sırası; t=3 ve t=16 sonrası yeni süreçte tam devam; ana sonucun tam tekrar üretimi; bu turun başlangıcındaki bütün eski dosyalar ile HEAD'in korunması. Kitap, ana sistem, commit/push/yayın kapsam dışında.
