# Ana sonuçlardan sonra: geçmişsiz aramaya bütçe ayırma kontrolü

21 Eylül 2026. Bu plan 9.216 ana koşulun sonucu görüldükten sonra yazıldı. Ana protokol ve JSON değiştirilmeyecek. Bu aynı 12 yaşamda mekanizma kontrolüdür; bağımsız tekrar veya önceden seçilmiş başarılı yöntem değildir.

Gerekçe: 12 eski görevden sonra, 2.048 adım bütçede öğrenilmiş kütüphane ilgili görevlerde 94/96 başarı sağladı; motif değişmiş görevlerde 14/96 kaldı. Geçmişsiz temel arama sırasıyla 73/96 ve 69/96. Ek geçmiş daha önce doğru edinilmiş bütün eski programları saklasa da gelecekteki edinimi daraltabiliyor.

## Sabit ek tasarım

Yalnız 12-görev geçmişi, aynı iki bütçe ve aynı 16 probe kullanılır. Her görevde iki bağımsız arama yapılır:

1. Geçmişsiz primitive-only arama, toplam bütçenin yarısı.
2. Öğrenilmiş kütüphane araması, toplam bütçenin yarısı.

İki arama da çalıştırılır; birinin erken bitişindeki boş bütçe diğerine aktarılmaz. Primitive-only kabul edilmişse onun programı seçilir; aksi halde öğrenilmiş dalın kabul edilmiş programı seçilir; ikisi de kabul edilmediyse çekimser kalınır. Kabul verileri iki dal arasında paylaşılır, bir dalın çıktısı diğerinin aramasını yönlendirmez. Maliyetler toplanır, en fazla ana bütçedir; bu iki arama durumu için ayrı kuyruk/bellek gideri vardır.

12×2×16 = **384 ek koşul**. Temel yarım-bütçe ve öğrenilmiş yarım-bütçe kayıtları da her hücrede saklanır. Ana yöntemlerin bütün-uygulama belleği veya kütüphane üretim maliyetiyle eşitlik iddia edilmeyecek.

## Koşullu koruma iddiası

Her görevde yarım bütçeli primitive arama doğru bir programı kabul ediyorsa, seçim kuralı aynı programı döndürür. Dolayısıyla onun bütün-alan doğru çözüm kümesi, portföyün aynı toplam bütçedeki doğru çözüm kümesinin alt kümesidir. Bu iddia kodun seçim kuralıyla sağlanır; yeni bir öğrenme ilkesi değildir. **Tam bütçeli primitive aramanın bütün başarıları korunuyor denemez.** Yanlış kabul edilmiş primitive program doğru öğrenilmiş programın önüne geçebilir; sonlu kabul kümesinin doğruluk garantisi yoktur.

OOPS geçmişi kullanan aramayla yeni başlangıçlara ayrılmış arama zamanını zaten birleştirir. Buradaki iki bağımsız BFS OOPS uygulaması veya onun bias-optimalite teoreminin yeniden üretimi değildir. Amaç olumsuz aktarımı sınırlama ile ilgili edinime ayrılan bütçe arasındaki bedeli görünür kılmaktır.
