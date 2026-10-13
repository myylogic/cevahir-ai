# Yeni kuram hedefi: neyi açıklamalı, mevcut çalışmalardan nasıl ayrılmalı?

Muhammed Yasin Yılmaz araştırma kaydı · 22 Eylül 2026 · Yerel kuramsal sınır notu

Bu not, kullanıcının araştırmanın bir sinir ağına ek yordamlar takmakla sınırlanmaması talebini esas alır. Hedef; deneyimin, çalışan sistemin gelecekte hangi hesapları yapabildiğini ve nasıl öğrendiğini kalıcı biçimde değiştirmesini açıklamaktır. Yeni bir buluş istemek araştırmanın yönüdür; buluşun gerçekleşmiş olması bir sonuç iddiasıdır. İkisini ayırmak, iddiayı küçültmez; hangi kanıtı üretmemiz gerektiğini belirler. Önceki deneyler korunur. Bu metin yeni deney sonucu, bilinç modeli veya genel yapay zekâ çözümü bildirmez.

## Problemin kapsamını koruyan çalışma gösterimi

Bir sistemin kalıcı durumunu şimdilik

\[
z_t=(R_t,P_t,U_t,Q_t,m_t)
\]

ile gösterelim. Burada R deneyimi ayrıştırmak ve temsil etmek için kullanılan yapıyı; P temsil üzerinde çalıştırılabilen işlemleri ve onların birleşme kurallarını; U deneyimin kalıcı durumu değiştirme yordamını; Q hangi deneyimin aranacağını veya hangi iç hesabın yapılacağını seçme yordamını; m ise diğer kalıcı bilgileri içerir. Bunlar ayrı yazılım modülleri olmak zorunda değildir. Bir değişken, birkaç rolü birlikte yerine getirebilir.

\[
a_t=Q_t(z_t,o_t;b_t),\qquad
e_t=(o_t,a_t,o_{t+1},f_t),\qquad
z_{t+1}=U_t(z_t,e_t;b_t).
\]

o gözlemi, a eylemi veya seçilen hesabı, f varsa geri bildirimi, b kullanılabilir hesaplama bütçesini gösterir. Geri bildirimin daima doğru etiket ya da ödül olacağı varsayılmaz. Bir güncelleme U bileşenini de değiştirebilir; bunu çalıştıran fiziksel hesaplama düzeninin kendiliğinden değiştiği sonucu çıkmaz. Güncellemenin uygulanabilirliği, zamanlaması ve hangi veriye eriştiği ayrıca tanımlanmalıdır.

Bu eşitlikler henüz açıklayıcı bir kuram değildir: Her programa uygulanabilen bir durum geçişi gösterimi, öğrenmenin neden başarılı olduğunu söylemez. Araştırmanın yükü, hangi geçişlerin hangi çevrelerde ve hangi kaynak sınırları altında kalıcı, kontrollü aktarım sağladığını belirlemektedir. Önerilecek ilke, bu değişkenlerden birine yeni ad vermekten daha fazla şey yapmalıdır.

Gelecekteki kapasiteyi yalnız bugünkü cevap doğruluğuyla ölçmeyelim. Belirli görevler ve b bütçesi için erişilebilir hesaplar, hata düzeyi, yeni görevi edinmek için gerekli deneyim ve toplam maliyet birlikte ölçülebilir. Aynı cevapları üreten iki durum, sonraki deneyime farklı tepki veriyorsa öğrenme bakımından eşdeğer değildir. Ancak her öğrenmenin yeni düğüm veya yeni işlem gerektirdiği de varsayılmamalıdır. Sabit ve yeterince geniş bir temsil, bazı görev ailelerinde aynı kazanımı sağlayabilir. Bütün z bir parametre dizisine kodlanabiliyorsa “parametre değişimi” ile “yapısal değişim” sözcüklerinden tek başına temel bir ayrılık çıkarılamaz; ayrılık erişilen işlev, öğrenme davranışı ve kaynak maliyetinde gösterilmelidir.

## Yakın birincil çalışmaların sınırı

**DreamCoder — Ellis ve arkadaşları, PLDI 2021.** Sistem, örneklerle tanımlanan görevler için programlar bulur; bunlardan yeniden kullanılabilir kütüphane bileşenleri çıkarır ve program aramasını yönlendiren sinirsel politikayı eğitir. Edinilmiş bileşenler daha sonraki bileşenlerin içinde kullanılabilir. Dolayısıyla yalnız cevapları değil, sonraki çözüm aramasını da değiştiren deneyim fikrinin açık bir öncülüdür. Hazır verilenler arasında temel programlama dili, görev belirtimleri ve kütüphane oluşturma yordamı vardır. Temel dilin eğitim problemlerini ilkece ifade edebilmesi ve görevlerin yararlı ortak soyutlamalar paylaşması beklenir. Yeni kütüphane, mutlak hesaplanabilirlikten çok sınırlı aramada erişilebilirliği değiştirir. Bizim açık sorumuz, başlangıç temsilinin yetersiz olduğu deneyimleri nasıl teşhis edip uygun ayrımlar ve işlemler oluşturacağımızdır. “Uyku sırasında deneyimden kütüphane öğrenme” tek başına yeni buluş iddiamız olamaz. [Birincil tam metin, özellikle §2–2.4](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf); [yayın DOI’si](https://doi.org/10.1145/3453483.3454080).

**Gödel machines — Schmidhuber, 2006.** Kuram, yararlı olduğu kanıtlanan bir değişikliğin sistemin kendi programına uygulanmasını içerir; değiştirilebilir yazılım, kanıt arayıcısının kendisini de kapsar. Ancak fayda ölçütü, donanım, başlangıç programı ve çevre hakkında kullanılacak varsayımlar bir aksiyom sistemiyle tanımlanır. Küresel eniyilik sonucu, makaledeki koşullarda, bulunan değişikliği şimdi uygulamanın mevcut aramayı sürdürmekten daha yüksek fayda vermesi anlamındadır. Her ortamda bütün olası gelişmeleri bulma veya durmaksızın ilerleme garantisi değildir. Makale, yararlı bir değişikliğin kanıtlanamayabileceğini ya da kanıtının kaynak sınırları içinde bulunamayabileceğini açıkça belirtir. Bizim açık sorumuz, eksik ve değişen deneyim altında uygulanabilir edinim ile korunmayı nasıl birleştireceğimizdir. “Sistem kendi öğrenme kodunu değiştirsin” fikri bu nedenle tek başına yenilik değildir. [Birincil bölüm, §2.5, §3 ve Teorem 1](https://sferics.idsia.ch/pub/juergen/gmAGI.pdf); [yazarın sürüm ve yayın kaydı](https://people.idsia.ch/~juergen/goedelmachine.html).

**Öğrenilmiş optimizasyon — Andrychowicz ve arkadaşları, NeurIPS 2016.** Çalışma, parametre güncellemesini üreten bir LSTM’yi görevler boyunca eğitir. Güncelleyici, gradyanları ve kendi geçmiş durumunu kullanır; böylece yalnız hedef ağ değil, öğrenme yordamının davranışı da deneyimden etkilenir. Hazır verilenler arasında türevlenebilir amaç, gradyan arayüzü, güncelleyici mimarisi, dış eğitim yöntemi ve eğitim görevleri bulunur. Başarı benzer yapılara aktarılabilir; makaledeki sigmoid ağdan ReLU ağına geçiş örneğinde aktarım başarısızdır. Değerlendirmede güncelleyici parametreleri dondurulur; bu deney düzeni sınırsız çevrimiçi özdeğişim kanıtı sunmaz. Bizim açık sorumuz, sistemin yaşamı sırasında kendi öğrenme varsayımlarını hangi deneyimle değiştireceği ve bu değişikliği yeni görevlerde nasıl sınayacağıdır. Öğrenme kuralını parametrelemek tek başına bu soruyu çözmez. [Birincil tam metin, §2, §2.1 ve §3.2](https://papers.neurips.cc/paper/6461-learning-to-learn-by-gradient-descent-by-gradient-descent.pdf).

Bu eşleşmeler üç tam metnin belirtilen bölümlerine dayanır; bütün literatürü taradığımızı veya kalan bileşimin yeni olduğunu göstermez. Çalışmaların açıklamaları ile bizim açık sorularımız ayrı tutulmuştur.

## Yeni bir ilke iddiasının karşılaması gereken ölçütler

1. **Açık varsayımlar ve karşı örnek:** Çevre sınıfı, erişilen bilgi, geri bildirim ve bütçe yazılmalıdır. Aynı geçmişle uyumlu fakat sonraki doğru cevabı farklı iki dünya varsa, ek bilgi olmadan ikisinde de hatasızlık istenemez. Bu bilinen türden belirlenemezlik, kendi kuramımızın başarısı diye sunulmaz; garantinin kapsamını sınırlar.
2. **Ayırt edici öngörü:** İlke kaldırıldığında neyin değişeceği önceden belirtilmelidir. Sadece başarı artışı yeterli değildir. Örneğin yeni temsil daha düşük bugünkü hatadan bağımsız olarak üçüncü görevde edinim maliyetini azaltacaksa, bu ayrı ölçülmelidir.
3. **Aynı bilgiye erişen güçlü karşılaştırma:** Sabit ama yeterince geniş temsil, program kütüphanesi, öğrenilmiş güncelleyici ve deneyim seçimi gibi yakın açıklamalar ilgili oldukları yerde karşılaştırılmalıdır. Hazırlık, arama, doğrulama ve saklama maliyetleri dahil edilmelidir. Eşit etiket sayısı, eşit toplam kaynak demek değildir.
4. **Değişirken süreklilik:** Eski yetkinlik, yeni kazanım ve yeniden başlatma sonrası kalıcılık ayrı doğrulanmalıdır. Başlangıç ve sonuç iyi görünürken aradaki güncellemeler eski işi bozabilir. Geçmişi koruma koşulu, yanlış bilgiyi hiç düzeltememeye dönüşmemelidir.
5. **Üçüncü göreve aktarım:** İlk görev yapıyı oluştursun, ikinci görev onu değiştirsin; üçüncü ve önceden görülmemiş görevde bu geçmişin etkisi ölçülsün. Yararlı ve yanıltıcı ortaklıklar birlikte bulunmalıdır. Yalnız aynı ailenin yeniden adlandırılmış örnekleri geniş yetkinlik sayılmamalıdır.

Araştırmanın açık hedefi, R, P, U ve Q rollerinin deneyimle nasıl birlikte değişebildiğini açıklamak ve bu açıklamadan sınanabilir sonuçlar çıkarmaktır. Mevcut çalışmaları yan yana koymak, isimlerini değiştirmek veya tek bir başarılı ortam göstermek bu hedefin yerine geçmez. Yeni kuram gerekiyorsa, onu gerekli kılan ayrımı ve koşullu matematiksel sonucu göstermemiz gerekir; bugün böyle bir temel ilkeyi bulmuş olduğumuz iddia edilmemektedir.
