# Yeniden sınamanın kendisini deneyimle değiştirmek

20 Eylül 2026. Kullanıcının “devam et” yönlendirmesiyle önceki kapalı döngü deneyi genişletildi. Önceki planlayıcıya tek bir `.01` değişim oranı verilmişti. Şimdi gerçek değişim oranı verilmeden, **değişim beklentisinin kendi eylem sonuçlarıyla güncellenmesi** sınandı. [Kod](../../../research/living_learning/unknown_change_experiment.py), [tam sonuç](../../../research/living_learning/results/unknown_change.json), [bağımsız matematiksel denetim](UNKNOWN_CHANGE_AUDIT.md).

**Sonuç:** Verilen aday değişim modelleri arasında deneyimle inanç güncellemek, hangi davranışın ne zaman yeniden sınanacağını değiştirebilir. Bu, sabit bir “her 20 adımda dene” komutundan daha güçlü bir uyarlanmadır. Ancak modelin kendisini güncellemek bütün dünyalarda daha iyi karar garantisi vermez. Bazen yeni kanıt, yeniden denemeyi artırmak yerine azaltmayı gerekçelendirir.

## 1. Bilinmeyen nedir, verilmiş olan nedir?

240 adım, ilk anda ortak başarısız deneme, sonraki dış gözlem sabit, yalnız eylemden başarı/başarısızlık alınması önceki deneyle aynıdır. Engelin açılması kalıcıdır. Her yaşam için bir değişim hızı `h` seçilir; öğrenici gerçek `h`yi veya açılma anını görmez. Adaylar `h∈{0,.002,.02}` ve başlangıçta her birinin olasılığı `1/3`tür. `h=0` bileşeni gerçekten hiç açılmayan dünyayı kapsar. Aday aile, başlangıç önseli, doğru ve gürültüsüz geri bildirim, eylem maliyetleri ve kalıcı değişim semantiği yine verilmiştir. Yeni bir değişim ailesi keşfi iddia edilmiyor.

Başarısızlık bedeli 1 ve 20 ayrı koşullardır. Her başarılı kullanım +1 getirir. Planlama hesabının CPU maliyeti çevresel faydaya dahil edilmez. Doğru oranı bilen oracle yalnız üst referanstır; bu ek bilgi öğreniciye sızmaz.

## 2. Geçmişin doğru sıkıştırılması

Açılma anı `τ`, son başarısızlık zamanı `f` olsun. Açılma kalıcı olduğu için daha önceki bütün başarısızlıklar birlikte yalnız `τ>f` olayını söyler. Örneğin 10, 35, 90 ve 150'de başarısız olmak ile yalnız 150'de başarısız olmak, bu özel dünyada değişim oranı hakkında aynı son bilgiyi verir. Bunları dört bağımsız kanıt diye çarpmak yanlıştır.

\[
Z(t)=\sum_iw_i(1-h_i)^t,\qquad
P(h_i\mid\text{başarısızlık geçmişi})=\frac{w_i(1-h_i)^f}{Z(f)}.
\]

Sonucun zamanlaması önemlidir; aynı sayıda başarısızlığın kanıt gücü aynı olmak zorunda değildir. Bu yeterli istatistik sayesinde uzun olay listesini geri getirmeden gelecek davranış güncellenebilir. Bunun bütün öğrenme problemleri için böyle küçük bir sıkıştırma sağladığı iddia edilmez.

Son başarısızlıktan sonra eylem yapılmadığında, dünyanın gerçekten hâlâ kapalı olup olmadığı bilinmez. Model ağırlıkları salt beklemekle yeni gözlem almış gibi güncellenmez. Buna rağmen geçiş modeli altında, `t>f` anındaki açıklık tahmini değişir:

\[
p(t,f)=P(\tau\le t\mid\tau>f)=1-\frac{Z(t)}{Z(f)}.
\]

Bu iki değişim ayrı kaydedilir: zaman geçişiyle tahminin değişmesi ve **gerçek yeni başarısızlıkla** model olasılıklarının değişmesi. İç durumun davranışı değiştirmesi için rastgelelik gerekmez; modelin yanlışlanması için dış sonuç gerekir.

## 3. Karar hesabı ve karşılaştırmalar

`V(T+1,f)=0` olmak üzere:

\[
V(t,f)=\max\left\{V(t+1,f),\;p(t,f)(T-t+1)+(1-p(t,f))[-c+V(t+1,t)]\right\}.
\]

İlk seçenek bekleme, ikincisi denemedir. Gerçek bir başarısızlıktan sonra `f=t` olur ve bütün adaylara ilişkin inanç yeni bilgiyle değişir. Başarı, bu modelde bundan sonraki kullanımların başarılı olacağını belirler. Bu sonlu Bayes karar problemidir; yeni bir algoritma değildir.

Üç kontrol eklendi. `frozen_hyperprior`, her başarısızlıkta kapalı duruma döner ama değişim adaylarının ağırlıklarını başlangıçtaki `1/3`te tutar. `fixed_hazard_001`, önceki tek-hızlı modeli kullanır. `never_probe` hiçbir yeni geri bildirim almaz. İlk kontrol tek bir kalıcı gizli `h` için doğru Bayes çıkarımı değildir; çıkarımın ilgili parçasını kapatan açık bir ablasyondur. Her başarısızlıktan sonra yeni gizli hız çekilen farklı bir dünya için anlamlı başka bir model olarak görülebilir.

Her gerçek `h` için bütün 240 açılma tarihi ve ufukta hiç açılmama olasılığı tam toplandı. Bu turda Monte Carlo güven aralığı gerekmedi; aşağıdaki değerler sonlu model altında sayısal tam beklentidir. Doğru `h`yi bilen politikanın değeri de ayrı hesaplandı.

## 4. Sonuçlar

Verilen üç adayın eşit önsel karışımı üzerinde beklenen toplam fayda:

| Politika | Başarısızlık maliyeti 1 | Başarısızlık maliyeti 20 |
|---|---:|---:|
| Değişim modellerini kendi deneyimiyle güncelle | **69,527959** | **37,323585** |
| Model ağırlıklarını başlangıçta dondur | 65,668642 | 26,621452 |
| Verilmiş tek değişim hızı `.01` | 66,654044 | 26,361483 |
| Yeniden deneme yapma | 0 | 0 |
| Doğru değişim hızını bilen oracle | 73,593780 | 54,576609 |

Güncelleyen politikanın Bellman değeri, bütün çevre olasılıklarının bağımsız toplamıyla `10⁻⁸` toleransta aynı çıktı. Model ve toplam fayda ölçütü altında en iyi eylemi seçmesi matematiksel yapının sonucudur; bilinmeyen dünyaya dağılımsız üstünlük kanıtı değildir.

Hiç açılmayan dünyada, maliyet 1 iken öğrenen sistem şu anlarda yeniden denedi:

`16, 33, 52, 74, 99, 128, 161, 199`.

Ağırlıkları donduran sistem 18 kez, tek-hızlı sistem 16 kez denedi. Öğrenen sistemde her başarısızlık, daha hızlı değişen dünyalara ayrılan olasılığı azaltarak gelecekteki davranışı etkiliyor. Başlangıçta `.3333` olan hiç-değişmeme olasılığı, 199'daki son başarısızlıktan sonra yaklaşık **.59195** oluyor; `.02` hızlı değişim bileşeni **.01062**ye düşüyor. Bu sayılar “artık kesin değişmez” anlamına gelmiyor.

Maliyet 20 iken öğrenen sistem hiç-açılmayan dünyada yalnız **83. adımda bir kez** deniyor; dondurulmuş ağırlıklar `49,101,160`ta üç kez deniyor. Sonraki denemeleri yapmamak seçilen sonlu ufuk ve maliyet altında değerli bir karar olabilir. Deneme aralıklarının zamanla büyümesi yalnız model öğrenimine atfedilmemelidir: **kalan ufkun azalması da** yeniden denemenin değerini düşürür. Ablasyon bu iki politika arasındaki farkı gösterir; tek başına aralık grafiği nedensel kanıt değildir.

### Negatif sonuç: doğru önsel içinde bile her alt dünyada üstün değil

Maliyet 20 ve gerçek `h=.002` iken güncelleyen politikanın faydası **7,25083**, ağırlıkları donduranın **−11,12732**dir. Buna karşılık gerçek `h=.02` iken sırasıyla **124,71993** ve **150,99167** olur. Önsel ortalamasında daha iyi olan politika hızlı dünyada daha yavaş davranıp fırsat kaybedebilir.

Aday kümesinde olmayan `h=.08` ayrıca stres koşulu olarak ilk çalıştırmadan önce seçildi. Maliyet 20'de güncelleyen yöntem **157,82428**, dondurulmuş yöntem **190,77203**, doğru-hız oracle'ı **212,16658** değer üretiyor. Dünya modelini güncelleyebilmek, doğru adayın mevcut olmasını veya önsel dağılımın uygunluğunu kendiliğinden sağlamaz. Bu sonuç gizlenmedi.

Yavaş `.002` dünyasında yüksek maliyetli öğrenici, ufuk içinde oluşan açılmaların yalnız **%40,13**ünü bulurken daha yüksek toplam fayda sağlayabiliyor. Bu yüzden “gelecekteki her öğrenme fırsatını açık tutmak” ile “seçilen kaynak/zarar/ödül ölçütünde iyi yaşamak” aynı amaç değildir. Hedef seçimindeki bu gerilimi bir mimari adı ortadan kaldırmaz.

## 5. Yeni kanıtı iki kez saymama denetimi

Sıralı başarısızlıklarda doğru güncelleme, bir önceki başarısızlıktan bu yana geçen `Δt` kadar hayatta-kalma olabilirliğiyle yapılır. Kod bu biçimde güncellenmiş ağırlıkları doğrudan `w_i(1-h_i)^150/Z(150)` ile karşılaştırır. Sonuçlar aynıdır:

`h=0: .559005; h=.002: .413997; h=.02: .026998`.

İlk olayın öneminin sonradan değişmesi burada yeni olayların eski kaydın metnini değiştirmesi değildir; onların birlikte hangi açıklamaları desteklediği yeniden belirlenir. Son başarısızlık önceki iç içe kısıtları kapsar. Bu, öğrenme için geçmişin anlamının nedensel/olasılıksal yapı içinde değerlendirilmesi gerektiğine küçük bir örnektir.

## 6. Burada hâlâ öğrenilmeyenler

Sistem önceden verilmiş üç değişim modelinin ağırlıklarını değiştirir. Bayes güncelleme kuralını, ödülü, kalıcı açılma varsayımını veya hipotez dilini kendisi keşfetmez. Sabit yürütücü altında güncellenen model, sonraki öğrenme eylemlerini değiştirir; bu "yeni öğrenme algoritması üretildi" demek değildir.

Bir yaşamda gözlenen ilk başarılı deneme yalnız açılma anını önceki başarısızlık ile başarı arasında bir aralığa yerleştirir. Sonraki bütün başarılı kullanımlar, kalıcı açılma modelinde hız hakkında yeni bağımsız örnekler değildir. Tek açılma olayıyla gerçek hazard'ın genel olarak kesin öğrenildiği söylenemez. Uygulama başarıdan sonra hazard çıkarımını sürdürmez, çünkü bu sonlu görevde bir daha davranışı değiştirmez; başka ortamlara hız aktarımı sınanmadı.

Gürültü, kapanıp tekrar açılma, birden fazla görev, gerçek kapasite büyümesi, öğrenilmiş algı temsili, riskli geri dönüşsüz ortam veya sonsuz ufukta optimal davranış bu deneyde yoktur. Model dilinin daha sonra genişlemesi ayrı açık sorudur. Deney yaklaşık 0,93 saniyede çalıştı; RAM kullanımının tamamı ölçülmedi.

## 7. Daha temel kaynak sınırı

Hiç değişmeyen `W₀` ile keyfi geç bir `τ` anında kalıcı değişen `Wτ` arasında yalnız maliyetli deneme ayrım sağlasın. Her başarısız denemenin bedeli en az sabit `c>0` olsun. `W₀`da **bütün sonsuz yaşam boyunca beklenen deneme maliyeti sonluysa**, toplam deneme sayısı sonlu olmak zorundadır (hemen hemen kesin). Son deneme zamanı `L` de sonludur.

İki dünyayı aynı iç rastgelelikle, ilk `τ` sonrası denemeye kadar eşleyelim. O ana kadar gözlemler aynıdır. Değişimi fark etme olasılığı:

\[
P_{W_\tau}(\text{sonunda fark etme})=P_{W_0}(L\ge\tau)\longrightarrow0.
\]

Dolayısıyla **sonlu toplam beklenen yanlış-deneme bedeliyle, keyfi geç her kalıcı değişim için sabit pozitif keşif olasılığı birlikte garanti edilemez**. Bu, burada türetilmiş basit bir kuplaj/sonlu-kaynak sonucudur; özgünlük iddiası değildir. Sonlu ufuktaki deney optimalitesinden ayrı mantıksal sınırlamadır. Ücretsiz yan gözlem, giderek sıfıra inen deneme bedeli, dünyadaki değişim zamanına üst sınır veya başka bilgi kanalı varsayımı değiştirebilir. Bu sonucu bütün güvenli keşif problemlerine genişletmiyoruz.

Bu yüzden temel hedef "geçmiş hiçbir şeyi kapatmasın" şeklinde sınırsız olamaz. Hangi değişimleri hangi ufukta, hangi bedelle ve hangi bilgi kanallarıyla yeniden sınayacağımız açıkça tanımlanmalıdır. Geçmişe bağlı olmakla geçmişine mahkûm olmak arasındaki ayrım ancak bu koşullar altında ölçülebilir.
