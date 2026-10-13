# Tur 18 — Tekil düzeltme etkileri ve birlikte değişmenin öğrenilmesi

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Sonuçlardan önce yerel protokol

## Soru ve süreklilik

Tur 17'de geçmişten edinilen düzeltme şeması, güçlü toplu düzenleme kontrolünü aşmadı; derin/kısmi görevlerin çoğu çözülmedi. Şimdi aynı edinilmiş şemanın **nerelere birlikte uygulanacağı** incelenir. Tek tek denenen değişikliklerin etkisi, birlikte yapılınca ortaya çıkan sonucu açıklamaya yeter mi? Bir görevde öğrenilen etkileşim ihtiyacı sonraki edinime taşınınca maliyet ne olur?

Bu, ana yaşayan öğrenme probleminin yalnız iç değişiklik atama yönüdür. Öğrenici çevresel nedensellik, temsil dili veya genelleştirme algoritması icat etmiş sayılmaz. Önceki 1.007 dosya için oluşturulan koruma dökümü esas alınır. Kitap, yayın, ana uygulama, commit ve push kapsam dışıdır.

## Verilmiş ve edinilmiş nesneler

Tur 17'nin doğrulanmış kaydından her tohumun üç delikli genel şeması alınır: `(A+B)*C -> A*C+B`. Bu, koşulsuz cebirsel eşitlik değildir. Kaynaktaki 6.158 birimlik edinim bedeli bütün yöntemlere yazılır. Şemayı yeniden icat etme iddiası yoktur. Kaynak sonuç dosyasının hash'i sabitlenir.

Öğrenici aday AST'yi, 16 farklı `(x,y,çıktı)` örneğini ve edinilmiş şemayı alır. İşlemler F17'de `add/sub/mul`; dil önceki turla aynıdır. Doğru hedef AST, doğru düzenleme maskesi, görev ailesi ve gizli 289-girdi tablo öğreniciye verilmez. Kendi adayını içten değiştirip erişilebilir örneklerde yürütmek mümkündür; ek öğretmen etiketi istenmez.

Şemanın bütün eşleşmeleri adayın kendi AST'sinden bulunur. Bu tur dünyası sekiz ayrık eşleşme kurar; iç içe eşleşmeler yoktur. Görevler bilinmeyen genel programlardan hata yerini keşfetme deneyi değildir: eşleşmeler potansiyel düzenleme yerleridir, doğru hata yeri etiketi verilmez. Yapısal kısıt ayrı kolaylaştırmadır.

## Dünyalar ve yaşam sırası

24 eşleştirilmiş yaşam; her yaşamda 16 görev. İlk dört görevde sekiz yerel ifadenin toplamı, sonraki dört görevde dörder ikili çarpımın toplamı, sonraki dörtte yeniden toplam, son dörtte iki üçlü ve bir ikili çarpımın toplamı vardır. Bu sıra öğreniciye aile etiketi olarak verilmez; kendi AST'si elbette görülebilirdir.

Her yerel başlangıç `(u+v)*w` ifadesidir; u,v,w, `a*x+b` veya `a*y+b` biçimli farklı rastgele affine ifadelerdir (a=1..16, b=0..16). Bir kaynakta edinilmiş şema bu yerelde `u*w+v` önerir. Her görevde sekiz yerin dört tanesi rastgele seçilip hedefte değiştirilir. Seçim ve AST içindeki yer sırası araştırmacı tarafından üretilir; maskenin kimliği öğreniciye verilmez. İlk ikili görevde ayrıca önceden belirlenmiş negatif tanık vardır: bütün yerellerde w=0, ilk iki çarpım çiftinin dört yeri değiştirilir. Bu görevde tekil düzenlemelerin hepsi etkisiz görünür; eşlenmiş değişimler yeni çıktı üretebilir. Diğer görevlerin dağılımını buna göre ayarlama yapılmaz.

16 kanıt girdisi bütün F17²'den tekrarsız rastgele seçilir. Gizli tam alan yalnız araştırmacı tarafından kabul sonrası değerlendirilir. 168 yol (24×7 yöntem), 2.688 hedef denemesi. Bunlar 24 bağımsız yaşamın eşleştirilmiş yöntemleridir.

## Tekil ve ikili etki modeli

S, uygulanmış değişikliklerin altkümesi olsun; `f_S(x)` verilen AST'nin buna göre değiştirilmiş çıktısıdır. Boş küme başlangıç programını verir. Tekil farklar `d_i=f_{i}-f_∅`, ikili etkileşimler `d_ij=f_{i,j}-f_i-f_j+f_∅` F17'de ve yalnız erişilebilir 16 girdide ölçülür.

Birinci derece model `f_∅+Σ d_i`; ikinci derece model buna seçilmiş çiftlerin `d_ij` terimlerini ekler. Bunların her yerde doğru olduğu varsayılmaz. Aday maskeler artan eleman sayısı, sonra tamsayı maskesi sırasıyla taranır. Modelin kanıt çıktıları hedef etiketleriyle eşleşirse gerçek değiştirilmiş AST ayrıca yürütülür. Doğrulama başarısızsa aday kabul edilmez; bu uyumsuzluk model yetersizliğine erişilebilir bir tanıktır. Öğrenici gizli test sonucuna göre model genişletmez.

Yalnız tekiller, yalnız ikililer ve bütün altkümeleri gerçekten yürütmek farklı hesap/deneme maliyetleridir. Lineer denklem çözümü yerine sekiz karar biti üzerinde tam maske taraması kullanılır; bunun model hesap maliyeti sayılır. Bu seçim sıralama ve doğrulamayı karşılaştırmalarda açık tutar, genel verimli öğrenme algoritması iddiası taşımaz.

## Yedi yöntem

1. `bulk_all`: boş maske doğru değilse bütün sekiz yeri birden değiştirir; uygun değilse durur.
2. `greedy_single`: her tur tek bit çeviren gerçek adaylar arasında kanıt uyuşmazlık sayısını kesin azaltan en iyisine geçer. En çok sekiz tur; düz plato aşılmaz.
3. `first_order`: bütün tekil etkileri ölçer; birinci derece modelin uyumlu maskelerini gerçek programla doğrular; ikili modele geçmez.
4. `second_order`: baştan bütün tekil ve ikili etkileri ölçer; ikinci derece modelle arar ve gerçek programla doğrular.
5. `adaptive_reset`: her yeni görevde birinci dereceden başlar; kanıtı sağlayan doğrulanmış aday bulamazsa ikinci dereceye genişler. Aynı görev içindeki ölçümler tekrar kullanılabilir, fakat tercih sonraki göreve taşınmaz.
6. `adaptive_persistent`: ilk görevde birinci dereceden başlar; genişleme gerektiğinde ikinci derece tercihini kalıcı duruma yazar ve sonraki görevleri bununla başlatır. Tercih yalnız 1 veya 2 olabilir; üçüncü dereceyi keşfetmez. Etkileşim ailesi hazır verilmiştir.
7. `exact_subset`: modeller kurmadan aynı 256 maskeyi gerçek programla sırayla sınar. Güçlü, temsil-uyumlu tam arama kontrolüdür; büyük hesap maliyeti görünürdür.

Her yöntemde ölçülmek için gerçekten yürütülen bir aday 16 örneği sağlarsa hemen kabul edilebilir; modelleme için bekletilmez. Çalıştırılmış maskeler görev içi önbellektedir. Kanıtı geçen ilk maskenin bütün alanı geçmesi otomatik kabul edilmez; yanlış kabul ayrıca raporlanır. Öğrenilen programlar kalıcı kütüphaneye kaydedilir; bu tur hedefler arasında yalnız adaptive_persistent'in model-derecesi arama politikasını değiştirir. Kütüphaneyi hedefler arasında yeniden aday önermek için kullanmama kararı bütün yöntemlerde ortaktır; deney çapraz görev çözüm çağırmayı sınamaz.

## Maliyet, kalıcılık ve ölçü

Her hedef için en çok 1.000.000 yeni sanal iş birimi; kaynak edinimi ve bütün hedef harcamaları yaşam toplamına yazılır. Bu tur ayrı bir global bütçe paylaşımı algoritması sınamaz. Değerlendirme için düğüm×16, değiştirilmiş ağacın tam sanal yapımı için düğüm, eşleşme bulma/dolaşma ve model maskeleri için kullanılan vektör terimi×16 sayılır. İkili katsayıları hesaplamak için üç vektör toplama/çıkarma ücretlendirilir. Tam alan araştırmacı değerlendirmesi öğrenici bütçesine girmez. Sanal iş, gerçek CPU veya RAM değildir. Gerçek program çalıştırma sayısı, model maskesi taraması, kalıcı JSON, derece genişletmeleri ve yanlış model öngörüleri ayrı raporlanır.

Kaydedilecekler: her görevde kanıt kabulü, 289-girdi doğruluk, bulunan maskenin ekilmiş maskeyle eşitliği (doğruluk şartı değildir), aday sayısı, bütün hesap kategorileri, öğrenme öncesi/sonrası kalıcı derece, iç model uyumsuzluğu. Aynı çıktı birden fazla maskeyle elde edilebilirse doğru kaynağın kesin teşhisi iddia edilmez. Programlar kalıcı saklanır; yeni süreçte aynı sonraki öğrenme davranışı kontrol edilir.

## Beklenen ayırıcılar ve sınır

Toplama ile birleşen ayrık düzeltmelerde tekil etkiler tamdır. Çarpılan iki değişiklikte çapraz terim, üç değişiklikte üçüncü derece terim doğabilir. Tek başına etkisiz değişikliğin gereksiz olduğu çıkarımı yanlış olabilir. Sabit ikinci derece de evrensel çözüm değildir; tam altküme kontrolü bu temsil açığını ayırır. Kalıcı ikinci derece tercihi aynı tür yeni görevde tekrar öğrenme maliyetini azaltabilir, yeniden basit görevlere dönüldüğünde gereksiz ölçüm maliyeti yaratabilir.

Bu önermeler bilinen sonlu-fark/Möbius ve program müdahalesi fikirleriyle ilişkilendirilecek. Matematiksel türetim yeni temel ilke olarak sunulmayacak. Koruma, bağımsız skaler değerlendirme, tüm küçük altkümelerde etki açılımı, etiket sızıntısı, bütçe, aynı süreç/farklı süreç devamı ve tam tekrar üretimi denetlenir. Sonuçlara göre aile, sıra veya bütçe ayarlanmaz.
