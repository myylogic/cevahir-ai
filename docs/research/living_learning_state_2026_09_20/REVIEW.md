# Bağımsız kapsam, veri hakları ve sonuç yorumu denetimi

20 Eylül 2026

Bu incelemede `research/living_learning_state_2026_09_20/recurrent_state.py` ve `evidence_audit.py` dosyalarının tamamı; `results/recurrent_state.json` dosyasının özet, matematik ve kalıcılık bölümleri; `results/evidence_audit.json` dosyasının yaşam kayıtları okundu. Durum birleştirme yordamının iç uygulaması bu incelemede yeniden denetlenmedi; bu yordam için ayrı uygulama testleri bulunması, buradaki kapsamın yerine geçmez. Tam deneyi bu inceleyici yeniden çalıştırmadı. İnceleme kod akışı ve mevcut sonuçların birbirini destekleyip desteklemediğiyle sınırlıdır.

**Sonuç:** Okunan kapsamda deney sonucunu geçersiz kılan veri sızıntısı veya zaman sırası hatası bulunmadı. Ana olumlu iddia boş olmayan bölüm alanına ilişkin tam dil eşitliğidir. Fiziksel nedenin teşhisi, genel temsil dili keşfi veya kaynak verimliliği üstünlüğü sonuçlardan çıkmıyor.

## 1. Zaman sırası ve bilgi hakları

- Her hizmet tahmini, o bölümün etiketi `observe` çağrısına verilmeden önce hesaplanıyor. Birleştirilmiş model yalnız 16, 32, 64, 128 ve 256 etiket sonrası yeniden kuruluyor; numaralandırma karşılaştırması her etiketten sonra aday eliyor. Yeniden kurma takvimi veri erişimiyle karıştırılmamalı: veri aynı, güncelleme sıklığı farklı.
- Modelleme yordamı yalnız o ana kadar alınmış `(kelime, gözlenen terminal etiketi)` çiftlerini alıyor. Gerçek grafik, temiz etiket listesi, bozuk etiket indeksleri ve ürün-grafı karşı örnekleri yordamın girdisi değil.
- Sonuç değerlendirmesi öğrenici durumunu değiştirmiyor; önce/sonra kayıt eşitliği de denetleniyor. Her kontrol noktasında aynı test kümesi yeniden değerlendiriliyor. Bu, sonuca göre ayar seçilmediği sürece öğreniciye test etiketi vermiyor; yine de kontrol noktalarını bağımsız istatistiksel tekrarlar saymak yanlış olur.
- Bölüm başlangıcı, sonu ve etiketin hangi kelimeye ait olduğu açıkça verilmiş. Ara çıktı yok. Deney, bu sınırları veya geri bildirim atfını öğrenmiyor.
- Kaynak aile karşılaştırmasında numaralandırma doğru `en çok üç durum` sınırına sahip; birleştirme yordamı bu sayıyı almıyor. Buna karşılık ikisi de ikili alfabe ve deterministik sonlu durumlu dil varsayımıyla çalışıyor. Numaralandırmanın sınırının dört durumlu koşulda yanlış olması açıkça yazılmış.

## 2. Sayılar ve boş kelime ayrımı

Ana rastgele üç durumlu ailede 16 yaşamın tamamının son grafiği, **bütün boş olmayan kelimelerde** gerçek grafikle aynı. Bu yalnız test edilen 32–256 uzunluklu örneklerde sıfır hata iddiası değil: ürün-grafı araması boş olmayan bütün kelimeleri kapsayan sonlu bir eşdeğerlik kontrolü yapıyor. Bu kontrol gerçek grafiği bilen değerlendiricinin kontrolüdür; öğrenicinin elindeki sertifika değildir.

İlk tam dil kontrolü 14/16 eşitlik bildiriyor. İki kalan yaşamın farkı yalnız boş kelime. Eğitim, hizmet ve testte kelime uzunluğu en az 1 olduğundan bu farkı “iki uzun dizi başarısızlığı” diye yazmak yanlış olur. Sonradan yazılan kanıt denetimi, ilk sonuç dosyasını değiştirmeden bu alan ayrımını düzeltiyor. Boş kelimenin etiketi eklenerek yeniden kurma, açıkça sonradan yapılan değerlendirici destekli bir tanı deneyi; özgün deneyin bağımsız başarısı olarak sayılamaz.

Ana ailede uzun dizi Brier hatası hem birleştirme hem derlenmiş numaralandırma için 0. Hizmet sırasında ise ortalama hata birleştirmede `0.05078125`, derlenmiş numaralandırmada `0.01611328125`, aday oylamasında `0.00914626`. Dolayısıyla daha az önceden verilmiş durum sayısıyla son başarıya ulaşmak, bu koşulda daha iyi edinim süreci anlamına gelmiyor.

Dört durumlu kontrolün 4 yaşamında birleştirilmiş grafik tam eşdeğer; üç durum sınırına sahip aday kümesi boşalıyor. Etiketi bozuk 8 yaşamda birleştirilmiş grafik boş olmayan dilde doğru değil; son grafikler 36–42 durumlu ve uzun dizi ortalama Brier hatası `0.287109375`. Bu sayıdan tek başına oylama/çekimserlik ile anlamlı istatistiksel üstünlük veya kötülük sonucu çıkarılmamalı; eşleştirilmiş farkın açıklayıcı aralığı sıfırı içeriyor. Yapısal yanlışlık ise ürün-grafı karşı örnekleriyle ayrı biçimde belirlenmiş.

## 3. Teşhis ve düzeltmenin sınırı

Bozuk geri bildirim koşulundaki bütün kelimeler farklı. Bu nedenle yanlış etiketler, aynı kelime için iki çelişen etiketi zorunlu kılmıyor; yeterince büyük deterministik bir makine bozuk kayıtları da tutarlı biçimde açıklayabiliyor. Deney bunun tehlikesini gösteriyor.

Üç durumlu aday kümesinin boşalması hem dört durumlu gerçek süreçte hem bozuk etiketli süreçte görülüyor. Dolayısıyla bu sinyal **“etiket bozuk” teşhisi değildir**. Model sınırı, süreç kararlılığı, ölçüm veya atıf varsayımlarından en az birinin mevcut kayıtla uyuşmadığını bildirir.

Temiz etiketlerle düzeltmede hangi etiketlerin değiştirileceğini değerlendirici sağlıyor. Sekiz düzeltilmiş grafiğin boş olmayan dilde tam eşitliği, tutulan kayıt yeniden yorumlandığında doğru yordamın yeniden edinilebildiğini gösterir; sistemin bozuk etiketleri kendisinin bulduğunu göstermez.

## 4. Bellek, silme ve maliyet

Çalışmanın ortasında kalıcılık deneyi, grafik ve mevcut durumla ham öneki taşımadan devamı kontrol ediyor. Öğrenmeye devam etme deneyi ise eski kelimeleri de saklıyor ve yeni süreçte indeksleri yeniden kuruyor. İki bellek iddiası ayrılmış; küçük hizmet grafiği, toplam öğrenme belleğinin küçük olduğu iddiası değildir.

`evidence_audit.erasure` içindeki çalışma durumunu silme kontrolü, aynı grafikte gerçekten farklı başlangıç durumu verilerek yürütülüyor. “Grafiği silme” satırı ise `no_graph=[0,0]` olarak yazılmış: başlangıçtaki sıfır cevaplı yordamla değiştirme örneğinin analitik sonucudur. Eksik grafik dosyasıyla çalıştırılmış bir sistem testi veya bütün olası öğrenilmemiş yordamların analizi diye sunulmamalı.

Hizmet geçişleri, aday eleme geçişleri ve birleştirme yordamı sayaçları farklı maliyet türleri. Sayımlar; aday kütüphanesini oluşturmayı, bütün test hesaplarını ve uçtan uca süre/FLOP ölçümünü kapsamıyor. Tek örnek başına küçük derlenmiş grafik hesabı gösteriliyor, fakat bu rapordan toplam kaynak verimliliği üstünlüğü çıkarılamaz. Tüm karşılaştırmaların aynı Python nesnesinde saklanması da ayrı yöntemlerin eşit bellek bütçesiyle sınandığı anlamına gelmez.

## 5. Desteklenen ifade

“Gözlenen bölüm sınırları, doğru terminal atfı ve verilmiş deterministik sonlu durumlu yordam dili altında; deneyim, daha sonra çok daha uzun girdileri işleyebilen kalıcı bir geçiş grafiğine dönüştürüldü. Doğru durum sayısı bilinmeden edinilen bu grafik belirli örneklerde bütün ilgili kelimelere genellendi. Aynı yöntem yanlış geri bildirime büyük ve yanlış grafiklerle uyabildi.”

Bu ifade kapsam içindedir. “Yaşarken öğrenmenin genel ilkesi bulundu”, “sistem hatanın gerçek nedenini keşfetti”, “bütün gerekli temsil dilini kendisi icat etti” ve “sabit aday ailesini her ölçütte geçti” ifadeleri kapsam dışındadır.
