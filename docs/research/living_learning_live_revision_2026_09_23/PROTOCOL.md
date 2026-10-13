# Tur 20 — Edinilmiş yapıyı devam eden hesap durumuna katmak

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Performans öncesi yerel protokol

## Soru ve süreklilik

Tur 19'daki yeni ayrımlar gözlemden kuruldu, ama her sorgu başlangıca dönebildi. Tur 16'da akan geçmişe yeni kural bağlandı, fakat hazır aday dili kullanıldı. Burada tur 19'un gerçekten edinilmiş ardışık modelleri, ayrı ve sıfırlanmayan bir ana akışa katılır. Yeni programı bilmek, o programın şimdi hangi durumda olması gerektiğini bilmekle aynı mı? Eski öğrenilmiş durum, ham geçmiş olmadan yeni duruma ilişkin hangi bilgiyi taşıyabilir?

Bu deney ayrı kalibrasyon kanalının reset kolaylığını kaldırmaz. Tur 19'un 72 dünyasının **tamamı**, yanlış son modeller de dâhil, kullanılır; gizli doğruluğa göre başarılı dünya seçilmez. Ana akış öğrenmeye etiket sağlamaz. Deney yalnız çalışan akışın sıfırlanmasını kaldırır; tek bir kesintisiz kanaldan tam öğrenme iddiası yoktur. Eski kitap, ana sistem, deneyler ve yayınlar korunur; commit/push yok.

## Veriler ve zaman çizgisi

Tur 19 `adaptive` yollarının başlangıç modeli ve 16 blok sonrası modeli yeniden kullanılır. Başlangıç modeli aynı küçük başlangıç yordamından elde edilir. Her güncellemede hangi modelin geldiği gerçek edinim kaydından alınır; doğru hedef tablo taşıyıcıya verilmez. Kaynak kodu ve sonuç hash'leri sabitlenir.

Her dünya için iki ana akış hızı vardır: öğrenme güncellemeleri arasında 32 veya 256 bit. Akış toplamı 16×aralık+16 bittir. Aynı tohum/hız, bütün aile/yöntemlerde aynı bit akışını kullanır. Öğreniciye dünya ailesi veya gizli durum verilmez. İlk model ilk bitten itibaren yürür; ilk kalibrasyon-blok modeli aralık bitinden sonra katılır. Sonraki model revizyonları aralık×(blok+1) anındadır. Modelin geçiş/çıktı tablosu değişmediyse tekrar taşıma yapılmaz. Her sınırda t=0 ve sonraki 16 bitte t=1…16 tahminleri kaydedilir; kalan bitler de çalıştırılır, sonraki sınıra kadar değerlendirme yapılmaz. Gerçek çevre hiç sıfırlanmaz.

24 tohum ×3 aile ×2 hız ×6 taşıma yöntemi =864 yol; 16 sınır/yol, 17 tahmin/sınır =235.008 ana tahmin. Aynı yaşamın yöntem ve hızları eşleştirilmiştir. Ek kesin parity yetisi bütün yöntemlere başlangıçtan verilmiştir ve bütün ana bitleri işler. Bu yetinin öğrenildiği iddia edilmez; korunmuş eski özet ve beceri kontrolüdür.

## Altı taşıma yöntemi

1. `reset_start`: yeni tablo geldiğinde yalnız öğrenilmiş modelin durumunu başlangıç sanır. Gerçek çevre ve parity sıfırlanmaz. Zayıf negatif kontrol.
2. `all_states`: yeni modelin başlangıçtan erişilebilir bütün durumlarını olası sayar. Sonraki bitler kümenin elemanlarını ilerletir; aynı duruma varanlar birleşir.
3. `parity_suffix8`: eski öğrenilmiş modelin durumunu kullanmaz. Yeni model ile verilen parity'nin erişilebilir ortak durumlarını kurar, son en çok sekiz gerçek bitin etkisini uygular ve mevcut parity ile uyumlu yeni durumları alır. Kısa iz + eski verilen beceri için güçlü kontrol.
4. `model_relation`: eski ve yeni model ile parity'nin aynı geçmiş boyunca ulaşabildiği üçlüleri kurar. Mevcut parity ve mevcut eski durum kümesiyle uyumlu yeni durumları alır; ham geçmiş saklamaz.
5. `model_relation_suffix8`: aynı üçlü ilişkiye son en çok sekiz gerçek bitin etkisini uygular, sonra mevcut parity/eski durum kümesiyle filtreler. Böylece eski öğrenilmiş durumun kısa iz üstüne kattığı bilgi sınanır.
6. `full_archive`: bütün ana bitleri saklar; değişen yeni modeli geçmişin başından yeniden yürütür. Tam geçmişe göre mevcut durum için güçlü referanstır; hedefi bilmez. Yanlış modelin yanlış tahminini de aynen üretebilir.

İlişki R, başlangıç üçlüsünden her iki sembolle ulaşılabilen bütün üçlülerin kümesidir. İlişkiyi son s ile daraltmak, her üçlüde parity ve iki model durumunu **birlikte** s boyunca ilerletmektir. Ardından şu anki parity ve eski olası küme ile filtre yapılır. İlişkiler bütün sonlu uzunlukları içerir; deneyde bilinen toplam zamanla ayrıca filtrelenmez. Bu, sound fakat mümkün en dar küme iddiası taşımayan bilinçli bir soyutlamadır. Ek parity verisi aynı-kanıt kontrolünde açık tutulur.

## Başarı ve belirsizliği ayırmak

Olası durumların çıktıları farklıysa öğrenici **kararsız** kalır; olasılık veya rastgele tahmin üretmez. Tek ortak çıktı varsa onu verir. Tek durum ile tek çıktı aynı şey değildir. Her noktada küme boyutu, kesin çıktı/kararsızlık, kesin yanlış çıktı, doğru kesin çıktı kaydedilir.

Değerlendirici ayrıca yeni modelin başlangıçtan bütün gerçek geçmişi okumasıyla oluşan durumunu hesaplar. Taşınan küme bu durumu içeriyor mu? Verilen kesin çıktı arşivle yeniden yürütmenin çıktısından farklı mı? Bu, **taşıma** hatasıdır; modelin gerçek dünyada doğru olduğu varsayımını gerektirmez. Arşiv çıktısı gerçek hedef çıktısıyla uyuşuyor mu? Bu ise **model** hatasıdır. İki hata birbirini iptal edebilir; toplam doğru çıktıdan tek başına doğru taşıma sonucu çıkarılmaz.

Hedef durumla model durumunun numaraları kıyaslanmaz. Hedefin yalnız çıktı davranışı değerlendiricide kullanılır. Kalibrasyonun tam eşdeğerlik sonucu öğreniciye sızmaz; rapor, bu sonucu yalnız sonradan doğru/yanlış model evrelerini ayırmak için kullanabilir. Ana etiketler küme filtrelemek, model değiştirmek veya taşıma yöntemini seçmek için kullanılmaz.

## Kaynaklar

Bu tur kaynak sınırına bağlı kesilme deneyi değildir; bütün yollar tamamlanır ve bedeller raporlanır. Önceki edinimin iç işi, sorgu bedeli ve arşiv boyutu bütün taşıma yöntemlerinde aynı kaynak-kaydından taşınır. Böylece sıfır edinim maliyetiyle program geldiği izlenimi verilmez. Bunun üstüne ek ana akış/taşıma maliyeti ayrıca sayılır.

Sanal iş: her parity geçişi, her olası model durumu geçişi, ilişki kurarken her bileşen geçişi, ilişki filtrelemesi, suffix uygulaması, geçmişin tekrar yürütülmesi ve çıktı kontrolü. Bit saklama için her yeni bit bir yazma; arşiv ve kısa iz uzunlukları, en büyük olası durum kümesi ve geçici ilişki boyutu ayrı raporlanır. Python nesne/hash/dedup/sıralama giderleri ve gerçek RAM bütünüyle modellenmez. Geçici ilişki hesaplandıktan sonra saklanmaz. Bütün yöntemler mevcut programı, parity'yi ve durum kümesini saklar; yalnız iki suffix yöntemi sekiz bit, yalnız arşiv yöntemi bütün ana bitleri saklar. Araştırmacının ham akış kayıtları öğrenici belleği sayılmaz ve öğreniciye verilmez.

## Denetimler ve beklenen sınır

Bağımsız skaler yürütme; küçük modellerde bütün kısa geçmişler üzerinden ilişki kapsaması; eski durumun yeni duruma işlevsel taşınabildiği ve taşınamadığı tanıklar; model değişimlerinde sound yöntemlerin arşiv durumunu içermesi; parity'nin bütün yollarda korunması; bütün durum adlarını permüte edince davranışın aynı kalması; yeni süreçte tam devam; ana sonucun tam yeniden üretimi; eski dosyaların korunması.

Bir model yanlışken bile ilişki taşıması kendi geçmiş-yürütme durumuna göre sound olabilir. Bunun dünya doğruluğu garantisi olmadığı açık tutulur. Son bitler bazı makinelerde belirsizliği giderir; permütasyon makinelerinde garanti yoktur. Art arda ara temsile geçmek bilgi kaybettirebilir; ayrı matematiksel tanıkla gösterilecek. Bunlar program güncelleme, durum gözlemcisi ve otomata literatürüyle eşleştirilecek; yeni temel ilke diye sunulmayacak.

Ana soru yaşayan öğrenmedir; bu tur model edinimi + çalışan durum taşımasının sınırlı birleşimidir. Yeni hedef/işlem dili, değerler, bağlamların yeniden yorumlanması, kaynak yönetimi ve sıfırlama olmadan tek-kanallı edinim açık kalır.
