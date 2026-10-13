# Tur 21 — içsel çıkarımın yaşayan hesap kapasitesine dönüşmesi

Muhammed Yasin Yılmaz · 23 Eylül 2026

Bu tur, ana living-learning sorusunu çözmüş değildir. Daha dar ama önceki turlardaki durum taşıma ve program parçası deneylerinden farklı bir parçayı sınar:

> Sistem dışarıdan yeni bir etiket almadan, kendi hesaplamasıyla kanıtlanmış ara sonuçları kalıcı olarak saklayıp daha sonraki, daha önce görülmemiş soruları aynı doğrulukla daha düşük hizmet maliyetiyle çözebilir mi? Bu ara sonuçların dayanakları değişince güvenli biçimde kullanımdan çıkarılabilir mi?

Bu, kullanıcının zihindeki olayların çapraz bağlarla yeni fikirleri kolaylaştırabileceği önerisinin mantıksal bir karşılığıdır. Ancak bu küçük sistemde kavram, duygu, bilinç, beyin veya genel yapay zekâ yoktur. Sonuç, önerinin yalnızca bir hesap biçimindeki sınamasıdır.

## Tasarım

24 eşleştirilmiş dünyada altı açık değişken ve dört özel değişken bloğuna sahip guarded CNF maddeleri üretildi. Öğrenici yalnız düz maddeleri ve sorgu varsayımlarını gördü; gizli “guard” anlamı veya hedef cevap verilmedi. Dört 24 sorguluk evre kullanıldı: ilk taban, aynı taban, bir gadget'ın terslenmiş guard ile değiştirilmesi ve ilk tabana dönüş. İlk, ikinci ve dördüncü evrelerde yeni dış madde gelmez; üçüncü evre bir çevre/taban değişikliğidir.

Yedi yöntem aynı sorguları aldı:

1. Her sorguyu baştan çözen `fresh`.
2. Çatışma kanıtlarını varsayımlardan arındırıp, yalnız değişmemiş temel maddelere dayanan küresel lemmaları saklayan `scoped`.
3. Aynı lemmaları dayanak değişse de kullanan bilerek güvensiz `blind`.
4. Taban değişiminde tüm lemmaları silen `erase`.
5. Aynı kronolojik geçmişi her seferinde yeniden kuran `replay`.
6. Her tabanda tüm 64 açık atamayı yeniden derleyen özgün `full_table`.
7. Tam tabloları madde içeriği sürümüyle saklayıp dönüşte yeniden kullanan denetim `full_table_cached`.

Proof checker solver'dan ayrıdır. Birim zorlamayı, çatışmayı, dalları, dayanakları ve yerel varsayımları kontrol eder. Bağımsız 6-bit/özel-atama enumerasyonu, kayıtlı cevapları doğrular. Sanal iş; madde/literal incelemelerini, çözüm adımlarını, kanıt kontrolünü, bağımlılık taramasını ve yazılan sertifika/geçmiş byte'larını açıkça sayar. Bu CPU zamanı veya donanım maliyeti değildir.

İlk altı yöntemin sonuçları görülmeden önceki ham kayıt ve motor `results/initial_run/` altında tutuldu. Bağımsız denetimde replay bellek tanısının henüz görülmeyen tabanı saydığı görüldü. Davranışa bilgi sızmadı, fakat tanı düzeltildi; ayrıca eski tabana dönüşte tabloyu da saklayan `full_table_cached` eklendi. Bu nedenle aşağıdaki sonuçlar düzeltilmiş son çalışmanın sonuçlarıdır; amendment sonrası yeniden çalıştırmadır, önceden kaydedilmiş bağımsız bir tekrar değildir.

## Sonuçlar

| yöntem | doğru / 2.304 | ort. hizmet işi | ort. toplam yaşam işi | ana gözlem |
|---|---:|---:|---:|---|
| fresh | 2.304 | 1.166,3 | 111.967,5 | Her cevap doğru; her sorgu yeniden çözülür |
| scoped | 2.304 | 440,3 | 111.812,3 | Doğru ve doğrulanmış; 185 lemma/yaşam ortalaması |
| blind | 2.218 | 378,1 | 97.554,1 | Daha ucuz görünüyor, fakat 86 yanlış cevap üretiyor |
| erase | 2.304 | 612,4 | 156.070,3 | Güvenli, fakat değişimde silme ve yeniden edinme pahalı |
| replay | 2.304 | 440,3 | 6.560.128,5 | `scoped` ile cevap/lemma durumu birebir; tekrar kurma pahalı |
| full_table | 2.304 | 28,2 | 128.739,6 | Tam sorgu tablosu hizmette çok hızlı; her dönüşte yeniden kurar |
| full_table_cached | 2.304 | 28,2 | 92.136,3 | Aynı tabloyu sürümle saklamak en ucuz güçlü kontrol |

`scoped` ile `fresh` arasındaki ortalama toplam fark yalnız 155,2 sanal iş birimidir; 24 yaşamın 15'inde `scoped` daha ucuzdur. Bu küçük dağılımda kanıtlı kalıcılık, tüm yaşam maliyetinde baştan çözmeye karşı küçük bir avantaj verdi. `full_table_cached` aynı açık sorgu arayüzünü bütünüyle derleyebildiği için `scoped`'dan ortalama 19.676 iş birimi daha ucuzdur. Böylece “kalıcı lemma, her güçlü derlemeyi aşar” iddiası reddedildi. `full_table_cached` açık altı değişkenli küçük arayüze özgü güçlü bir kontroldür; genel problemlerde tam tabloyu kurmak mümkün veya ekonomik olmayabilir.

`scoped` hizmet işini özellikle üçüncü değişim evresinde 532,8, dönüşte 207,0 ortalamaya indirdi ve bütün cevapları doğru tuttu. Bu, yeni dış bilgiyi bulduğu anlamına gelmez: değişen taban dışarıdan verilen yeni girdidir; sabit evrelerde ara sonuçlar yalnız mevcut maddelerin mantıksal sonuçlarıdır. `replay` eşlemesi 2.304 sorgunun hepsinde hizmet cevabı, aktif lemma kümesi, yeni lemma ve kayıtlı sayılar bakımından birebir geçti. Bu, kalıcılığın aynı tarihsellikte yeniden çıkarımdan farklı hesap maliyeti yaratabildiğini gösterir; bilgi üstünlüğü göstermez.

`blind`, taban değişince eski dayanakları kullanmaya devam ederek 86 yanlış cevap verdi; bağımsız denetim 1.705 geçersiz aktif lemma kullanımı ve 5 geçersiz yeni lemma saptadı. En küçük kapsam karşı örneği F=(a∨b), ¬a varsayımından b çıkarmak ve b'yi küresel gerçek sanmaktır: sonraki ¬b sorusu aslında a=1,b=0 ile sağlanabilir. Güvenli lemma, varsayımı boşaltılmış a∨b'dir. Bu sonuç, içerik doğruluğunun yanında geçerlilik koşulunun da kalıcı durumda taşınması gerektiğini destekler.

## Hipotez kararı

H21.1 desteklendi: bu yapay mantık ailesinde içeriden doğrulanmış ara sonuçlar, daha önce görülmemiş sorguların hizmet hesabını azaltabildi.

H21.2 kısmen desteklendi: `scoped`, `fresh`'ten bu protokolün yaşam maliyeti tanımı altında ortalamada azıcık daha ucuzdu; güçlü tablo kontrolünü geçemedi. Sonuç bir dağılım ve ölçüm birimine özgüdür.

H21.3 desteklendi: dayanakları izlemeyen kontrol değişen tabanda yanlış oldu; dayanaklı yöntem doğru kaldı. Bu, değişen dünya altında kapsam güvenliğinin zorunlu olduğu bu biçim için kanıttır, genel yeterli politika teoremi değildir.

H21.4 desteklendi: `scoped` ve `replay` aynı cevap/lemma durumlarını verdi; maliyetleri çok farklıydı.

H21.5 reddedildiği kapsam: deneysel `scoped`, açık 64 sorguluk tam tablo derlemesini geçmedi. Bu, genel derlemelerin her zaman üstün olduğu anlamına gelmez.

Bu turdan “yaşarken öğrenmenin formülü bulundu”, “sistem kendi kendine sınırsız gelişir” veya “genel yapay zekâya ulaşıldı” sonucu çıkmaz. Gösterilen şey, hesaplama bilgisinin erişilebilirliğini değiştiren ve geçerlilik koşulu taşıyan bir mekanizmanın bir problem ailesinde çalışabildiğidir. EBL, CDCL, SAT kanıt denetimi ve knowledge compilation ile doğrudan literatür eşleşmesi vardır; sonuç yeni temel ilke sahiplenmesi değildir.

## Doğrulama

`verify_deduction.py --rerun` geçti: 1.000 rastgele CNF çözümü (198 SAT tanığı, 802 UNSAT kanıtı), 16.128 kayıtlı cevap, 815 sertifika, 33.940 aktif lemma-doğruluk kontrolü, 2.304 tam replay çifti, 18 taze-süreç devamı ve korunmuş 1.055 eski dosya. Proof checker öz sınamasında 5 geçerli ve 14 bozuk kanıt doğru ayrıldı. HEAD değişmedi; commit, push veya yayın yapılmadı.

## Sonraki araştırma

H21.6–H21.12 alanı henüz açık: öğrenme yönteminin deneyimle edinilmesi; iki ayrı bilgi kümesinin hazır ortak arayüz olmadan eşlenmesi; aynı olayın kesişen görevlerde farklı kümelenmesi; eski kaydı koruyarak yeniden yorumlama. En ayırt edici sonraki deney, H21.10 ve H21.11'i birlikte sınamalıdır. Hazır doğru eşleme verilmemeli; güçlü tek-seferlik geçmiş sentezi, yanlış eşleme, eşlemeyi kapatma ve tam yaşam maliyeti kontrolleri sonuçtan önce yazılmalıdır. Tur 21 bu deneyi yapılmış gibi göstermemektedir.
