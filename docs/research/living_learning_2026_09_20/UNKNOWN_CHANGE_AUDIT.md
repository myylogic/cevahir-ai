# Bilinmeyen değişim hızının denetimi

20 Eylül 2026. Bu not, verilen `h=.01` değişim hızını kullanan önceki kapalı döngü deneyinin sınırını ve yeni sonlu karışım modelinin türetimini bağımsız olarak inceler. Yeni bir algoritma iddiası yoktur: Sonlu gizli model üzerinde Bayes güncellemesi ve inanç durumuyla dinamik programlama kullanılır. Bu bağlamın önceki çalışmayla ilişkisi [dual control makalesinde](https://www.jmlr.org/papers/volume17/15-162/15-162.pdf) ve [fenomen notunda](PHENOMENA_AND_LITERATURE.md) kayıtlıdır.

**Ana bulgu:** Kalıcı kapalı dünya olasılığını da içeren bir model, geç başarısızlıkları gördükçe yeniden denemeye daha az değer verebilir. Bu her zaman öğrenme hatası değildir. Fakat tek bir açılma olayı bulunan bir yaşamda, değişim hızının gerçek değerini genel olarak belirlediğini söylemek de mümkün değildir. Model belirsizliğini güncellemek ile değişim mekanizmasını keşfetmek ayrıdır.

## 1. Tam varsayımlar ve yeterli geçmiş

Yaşam başında gizli hız bir kez seçilir:

\[
h\in\{0,.002,.02\},\qquad P(h=h_i)=w_i=1/3.
\]

Bariyer `t=0`'da kapalıdır. `h>0` altında ilk açılma zamanı `τ`, `1,2,…` üzerinde geometriktir; `h=0` altında `τ=∞` olur. Açıldıktan sonra yeniden kapanmaz. Deneme, o anda açık olup olmadığını hatasız bildirir. Beklemek iki durumda da aynı gözlemi üretir. Eylemler açılma zamanını etkilemez. Bunlar öğrenilen sonuçlar değil, verilen model sınıfıdır.

Sağkalım fonksiyonu:

\[
Z(s)=P(\tau>s)=\sum_i w_i(1-h_i)^s,\qquad Z(0)=1.
\]

Başarısız denemeler `f_1<…<f_k=f` zamanlarında geldiyse hepsinin birlikte söylediği şey `τ>f`'dir. Çünkü bariyerin `f` anında hâlâ kapalı olması, daha erken başarısızlıkları zaten gerektirir. Dolayısıyla son başarısızlık zamanı yeterli durumdur:

\[
P(h_i\mid\text{başarısızlık geçmişi})=
\frac{w_i(1-h_i)^f}{Z(f)}.
\]

Bu çıkarım, eylem seçiminin yalnız kaydedilmiş geçmişe ve gizli dünyadan bağımsız iç rastgeleliğe dayandığını varsayar. Politikanın gizli `h` veya `τ` hakkında ek yan kanal kullanması halinde aynı yeterlilik sonucu otomatik geçmez. Normal kapalı döngü politikada eylem olasılıkları aday modeller arasında aynı gözlenmiş geçmiş için aynı olduğundan olabilirlik oranından çıkar.

**Çifte sayım hatasından kaçınma:** Her başarısız denemede mutlak zaman sağkalımını yeniden çarpmak `∏_j(1-h_i)^{f_j}` üretir ve yanlıştır. Güncelleme aralıklarla yapılırsa doğru çarpım `∏_j(1-h_i)^{f_j-f_{j-1}}=(1-h_i)^f` olur. Başarısızlıklar bağımsız geometrik yaşamlar değildir.

Son başarısızlık `f`, güncel karar zamanı `t>f` ise:

\[
p(t,f)=P(\tau\le t\mid\tau>f)
=1-\frac{Z(t)}{Z(f)}.
\]

Beklerken hiper-model ağırlıkları `P(h_i|τ>f)` sabit kalır. Zaman geçişi yalnız güncel açıklık öngörüsünü değiştirir. Bekleme sırasında ağırlıkları `w_i(1-h_i)^t` ile güncellemek, gözlenmemiş “hâlâ kapalı” etiketini gizlice kullanmak olur. Yeni başarısız deneme gelince `f←t` yapılabilir.

## 2. Bellman hesabının doğruluğu

Son karar zamanı `T`, başarısızlık maliyeti `c>0`, başarılı kullanım başına kazanç 1, bekleme değeri 0 olsun. `V(t,f)`, `t` anından `T` dahil kalan en yüksek beklenen toplam faydadır:

\[
V(T+1,f)=0,
\]
\[
Q_{\rm bekle}(t,f)=V(t+1,f),
\]
\[
Q_{\rm dene}(t,f)=p(t,f)(T-t+1)
 +(1-p(t,f))[-c+V(t+1,t)],
\]
\[
V(t,f)=\max\{Q_{\rm bekle}(t,f),Q_{\rm dene}(t,f)\}.
\]

Başarı anında güncel 1 birim kazanç ve sonraki `T−t` adımın kazancı birlikte `T−t+1` eder. Bu modelde başarıya ayrıca eylem maliyeti yoktur ve başarı kalıcı açıklığı doğrular. Başarısızlıkta aynı adımdaki kayıp çıkarılıp bir sonraki adıma son başarısızlık `t` olarak geçilir. Bekleme, son başarısızlığı değiştirmez. Geriye doğru türetim bu iki erişilebilir eylemin en iyisini seçtiği için, verilen model ve sonlu beklenen-fayda hedefi altında tam optimaldir. Bu koşullarda saf, deterministik eylem seçimi yeterlidir; eşitlikte karar kuralı sabitlenmelidir.

Önceki sabit-hız durumundan önemli fark şudur: Yalnız `d=t−f` yaşını tutmak artık yeterli değildir. Aynı 20-adımlık bekleme, yaşamın 20. adımında ve çok geç bir başarısızlıktan sonra aynı öngörüye sahip değildir; `f` geçmişte birikmiş model kanıtını taşır.

### Ayrı, küçük sayısal kontrol

Aşağıdaki değerler yalnız `Z` formülünün standart Python aritmetiğiyle doğrudan hesaplanmasıdır; politika performans deneyi değildir. “Sonraki 20 adım”, tablonun son satırında 240-adımlık önceki ufkun dışına taşan varsayımsal kestirimdir.

| Son gözlenmiş başarısızlık `f` | `P(h=0)` | `P(h=.002)` | `P(h=.02)` | Sonraki 20 adımda açılma olasılığı |
|---|---:|---:|---:|---:|
| 0 | .333333 | .333333 | .333333 | .123880 |
| 20 | .380466 | .365533 | .254002 | .098775 |
| 100 | .512509 | .419523 | .067969 | .039058 |
| 240 | .614883 | .380297 | .004820 | .016528 |

240. adımda gözlenen başarısızlık bile kalıcı kapalılığı ancak yaklaşık `.615` olasılığa çıkarır. Kalıcı kapalı hipotezi güçlü biçimde kanıtlandı denemez; çok yavaş değişim hâlâ makul açıklamadır.

## 3. Tanımlanabilirlik ve öğrenilen şeyin sınırı

Her sonlu `f` için `(1-.002)^f` ve `(1-.02)^f` pozitiftir. Bu nedenle sonlu sayıda başarısızlıkla `h=0` kesin olarak belirlenemez. Sonsuza kadar devam eden ve giderek daha geç zamanları örten hatasız başarısız denemeler altında, bu sonlu modelde sıfır-hız ağırlığı 1'e yaklaşır. Bu asimptotik koşul, maliyetli bir sonlu politikanın gerçekten böyle deneyler yapacağını garanti etmez.

İlk başarılı deneme `s`, son başarısızlık `f` ise gelen yeni bilgi bir açılma aralığıdır:

\[
f<\tau\le s,\qquad
P(h_i\mid f<\tau\le s)
\propto w_i\big[(1-h_i)^f-(1-h_i)^s\big].
\]

Bir başarı `h=0`'ı bu hatasız modelde eler. Fakat iki pozitif hızı genel olarak kesin ayırmaz. Her adım denemeyle `τ` tam bilinse bile elde yalnız **bir geometrik örnek** vardır. Açılmadan sonraki tekrar başarılar kalıcı açıklık varsayımı nedeniyle hız hakkında yeni bilgi eklemez. Birden çok bağımsız yaşam, yeniden kapanıp açılma süreçleri veya başka ölçümler olmadan “gerçek değişim hızı öğrenildi” iddiası aşırıdır.

Buradaki gerçek öğrenme iddiası daha dardır: Gelen sonuçlar verilmiş alternatiflerin ağırlığını ve böylece sonraki yeniden-deneme davranışını değiştirir. Model kümesi `{0,.002,.02}`, emicilik, sabit hız, maliyet ve kalıcılık varsayımları deneyimden keşfedilmiş değildir.

## 4. Ayırıcı karşılaştırmalar

| Karşılaştırma | Ne ayrıştırır? | Dikkat edilmesi gereken nokta |
|---|---|---|
| Güncellenen karışım | Hem güncel kapalılık hem hız ağırlıkları sonuçla değişir | Yeni yönteme değil, doğru Bayes koşullandırmasına tanık |
| Dondurulmuş hiper-ağırlık, güncellenen son başarısızlık | Faz içi açıklık kestirimi sürer, hız hakkında biriken kanıt kullanılmaz | `p_frozen(d)=1−Σ_i w_i(1−h_i)^d`; aynı `d` için `f`'den bağımsız |
| Verilmiş tek `h=.01` | Önceki kontrolün yanlış veya yaklaşık hız varsayımı | `.01` yeni karışımın bileşeni değil; sonucuna “gerçek hız biliniyor” denmemeli |
| Gerçek `h` verilen kontrol | Gizli hız hakkında ek bilginin üst sınırı | Gerçek açılma zamanı `τ` verilmez; bilgi ayrıcalığı açıkça sayılmalı |
| Eylemsiz eski başarısızlık / periyodik deneme | Öğrenilmiş sıralamaya veri erişimini yeniden açmanın bedeli | Sonuçlar fırsat keşfi, risk, gerçek deneme sayısı ve faydayla birlikte ölçülür |

Dondurulmuş ağırlık kontrolü, başlangıçtaki tek ve kalıcı `h` için doğru Bayes öğrenicisi değildir. Başarısızlıktan sonra hızın yeniden önselden çekildiği farklı bir yenilenme modeliyle yorumlanabilir. Bu yüzden onun politikasını da kendi tanımlanmış kestirim modeli altında planlayıp, gerçek tek-hızlı dünyada değerlendirmek temiz bir karşılaştırmadır. Yalnız sonuç sonrası yapılan “ağırlığı geri sıfırlama”, geleceği nasıl planladığı açıklanmamışsa belirsiz bir kontrol bırakır.

Özellikle bilgi güncellemesinin nedensel etkisi için iki geçmiş aynı güncel durumda karşılaştırılabilir: `f=20` ve `f=100`, her ikisinde de son başarısızlıktan 20 adım geçmiş olsun. Kalan ufuk eşitlenmiş bir ayrı karar problemi seçilirse güncellenen modelin öngörü farkı görülebilir; dondurulmuş modelde aynı kalır. Kalan zaman ve birikmiş kanıt birlikte değişirse hangi etmenin kararı değiştirdiği ayrıştırılamaz.

## 5. Tam entegrasyon ve optimalite denetimi

Geometrik karışımın açılma olasılıkları:

\[
P(\tau=s)=Z(s-1)-Z(s),\quad 1\le s\le T,
\qquad P(\tau>T)=Z(T).
\]

Bu son terim hem `h=0` dalını hem pozitif hızların ufuk-sonrası kuyruğunu içerir. Bunları yeniden ayrı eklemek olasılığı çifte sayar. Politikayı her `s=1,…,T` ve bir kapalı-kuyruk dünyasında çalıştırıp bu ağırlıklarla toplamak, Bellman değerini ayrı ortam yürütmesiyle kontrol eder. Dış dünyadan veri alınmadığı adımlarda sadece model öngörüsü yapılmalıdır.

Ek kontroller:

1. Tek bileşenli önsel, önceki sabit-hız formülüne indirgenir. Yalnız `h=0`, pozitif maliyet ve başka bilgi yolu yoksa optimal fayda 0'dır.
2. `[20,40,100]` başarısızlık geçmişi ile `[100]` geçmişinin model ağırlıkları aynıdır. İki geçmişin ödenmiş geçmiş maliyetleri aynı olmak zorunda değildir; gelecekteki inanç eşitliği toplam yaşam maliyeti eşitliği değildir.
3. Son başarısızlıktan sonra geri bildirimsiz bekleme hiper-ağırlığı değiştirmez. Güncel açıklık olasılığını değiştirir.
4. Tam karışımda bulunan optimal beklenen değer, aynı bilgi ve izin verilen eylemlerle çalışan sabit-hız veya periyodik politikanın karışım altında tam değerinden küçük olamaz; sayısal tolerans dışında tersi kod/karşılaştırma sorunudur.
5. Gizli hız verilen optimal denetleyicilerin `Σ_i w_i V_i*` ortalaması, gizli-hızlı karışım optimumundan küçük olamaz. Hızı bilmek, göz ardı edilebilir ek bilgidir. Bu üst sınır gelecekteki `τ`'yı bilen oracle'dan ayrıdır.

Monte Carlo ortalamalarında bu sıralamalar sonlu örneklem nedeniyle bozulabilir. Model altında kesin optimalite iddiası örneklem kazananına değil, Bellman türetimine ve tam toplamaya dayanmalıdır. Gerçek her hız için koşullu fayda da raporlanmalıdır: Karışımda optimal olmak her bileşende en iyi olmayı gerektirmez. Hız verilen kontrolle fark, aynı test dağılımında “gizli hızı önceden bilmenin değeri” olarak okunabilir; sistemin kendi öğrendiği yeni yetenek sayılmaz.

## 6. Model uyuşmazlığı ve yeniden kapanma

Sıfır-hız bileşeninin eklenmesi her bilinmeyeni kapsamaz. Gerçek hız örneğin `.01` olabilir; geçiş hızı zamana veya eyleme bağlı olabilir; kapalı-açık durumu tekrar değişebilir; sonuç hatalı ölçülebilir. Bu koşullar ayrı negatif kontrollerdir. Verilmemiş hızlarda iyi sonuç, gerçek sayısal hızın tanımlandığı anlamına gelmez; verilen karışımın yeterince iyi karar üretmesi olabilir.

En keskin kontrol, başarıdan sonra gerçek bir kapanma veya sahte bir başarısızlık bildirimidir. Başarıyı hatasız ve açıklığı emici kabul eden model, bunu sıfır olasılıklı sayar. Sıradan Bayes güncellemesinin normalizasyonu bu durumda tanımsızdır. “Son başarısızlığa dönüp tekrar başla” açık bir toparlanma kuralı olabilir; orijinal tek-açılma modelinin doğru Bayes çıkarımı değildir. Yeni model desteği, gözlem-gürültüsü modeli veya değişim noktası varsayımı ekleniyorsa ayrı varsayım olarak kaydedilmelidir.

Yalnız hazard karışımını güncellemek, dünyanın yeniden kapanabileceğini öğrenmiş olmak değildir. Uzun süre hiç değişmeyen koşulda daha az deneme maliyeti görülmesi olumlu sonuç olsa bile, yeni fırsatları geç fark etme veya kaçırma bedeli karşısında verilmelidir.

## 7. “Sürekli keşfetmeli” koşulunun düzeltilmesi

Geçmiş kanıt dünyanın çok yavaş veya hiç değişmediğini destekliyorsa, deneme maliyeti yüksekse ve kalan kullanım süresi kısa ise yeniden denemeyi azaltmak doğru karar olabilir. Veri erişiminin kapanması her koşulda arıza değildir. Arıza iddiası için değerlendirme hedefi belirtilmelidir: beklenen fayda mı, bütün kalıcı değişimler için sonlu gecikme mi, en kötü durum zarar sınırı mı?

Sabit en çok `K` adımlık yeniden deneme aralığı, kalıcı açılmayı kısa gecikmeyle fark ettirir; hiç değişmeyen dünyada büyüyen deneme maliyeti üretir. Sonlu ufukta beklenen faydayı optimize eden politika ise geç bir fırsatı bilerek sınamadan bırakabilir. Bu iki hedef aynı değildir. Sonsuz ufka geçilirse iskonto, ortalama ödül, toplam zarar ve erişilebilirlik ayrıca tanımlanmalıdır; sonlu Bellman hesabı otomatik sonsuz yaşam garantisi vermez.

Bu ek model, “geçmişe mahkûm olmama”yı her zaman daha çok denemek olarak yorumlamayı reddeder. Daha savunulabilir ilke: Geçmişin desteklediği modelleri ve onları yanlışlayabilecek erişilebilir sonuçları korumak; yeniden sınamayı açık değer, risk ve değişim varsayımları altında seçmek; modelin sıfır olasılık verdiği gerçek olayda bunu gizlemek yerine model desteğinin yetersizliğini kaydetmek.

## 8. Sonsuz yaşam için daha keskin bir maliyet–keşif sınırı

**Önerme.** Ayrık zamanda bir politika, kapalı dünyada `W₀` her başarısız deneme için sabit `c>0` bedel ödesin. `Wτ` dünyası `τ` anında açılıp açık kalsın. Açılma, `τ` adımının eylemi seçilip sonucu alınmadan önce gerçekleşsin. Bekleme gözlemleri iki dünyada aynı; başarılı deneme hatasız tespit sağlar. Politika `τ`'yı bilmez, zamanını veya iç rastgeleliğini dünya kimliğinden bağımsız belirler; başka bilgi kanalı yoktur. Adım başına en fazla bir deneme olsun.

`W₀`'daki toplam yaşam boyu deneme sayısına `N∞`, son deneme zamanına `L` diyelim. Hiç deneme yoksa `L=0`; sonsuz sayıda deneme varsa `L=∞`. O zaman

\[
P_{W_\tau}(\text{sonunda keşif})=P_{W_0}(L\ge\tau).
\]

**İspat.** İki dünyayı aynı başlangıç durumu ve aynı iç rastgelelik akışıyla eşleyin. `τ` öncesindeki bütün deneme sonuçları ve tüm pasif gözlemler aynıdır. `τ` veya sonrasındaki ilk denemeye kadar iki politika aynı gözlemleri aldığı için aynı eylemleri seçer. `W₀` yolunda böyle bir deneme varsa aynı anda `Wτ` yolunda yapılır ve açıklığı bildirir. Böyle bir deneme yoksa iki yol sonsuza kadar aynı pasif/başarısız-öncesi geçmişi izler ve keşif olmaz. Bu yol düzeyindeki eşdeğerliğin olasılığı, eşitliği verir. Keşif sonrası politikaların farklılaşması ilk keşif olayını etkilemez.

Eğer `E_W₀[cN∞]<∞` ise `N∞<∞` hemen hemen kesindir; dolayısıyla `L<∞` hemen hemen kesindir. Azalan olayların sürekliliğiyle

\[
\lim_{\tau\to\infty}P_{W_\tau}(\text{sonunda keşif})
=\lim_{\tau\to\infty}P_{W_0}(L\ge\tau)=0.
\]

**Daha güçlü karşıt ifade:** Bütün keyfî geç açılma zamanları için keşif olasılığı en az sabit `δ>0` olsun isteniyorsa, `P_W₀(N∞=∞)≥δ` olmalıdır. Bu durumda beklenen toplam başarısız deneme maliyeti sonsuzdur. Aslında olumsuz sonuç için sonlu beklenen maliyet koşulu gerekenden güçlüdür: `N∞`'nin hemen hemen kesin sonlu olması bile keşif olasılığının kuyruğunu sıfıra götürür.

Bu sonuç sonlu bir ufukta periyodik denemenin başarılı olmasını çürütmez. Farklı amaçlar arasındaki uyumsuzluğu gösterir: Hiç değişmeyen dünyada yaşam boyu sonlu beklenen deneme bedeli ile sınırsız geç tarihlerdeki bütün kalıcı değişimler için tek bir pozitif keşif-alt-sınırı birlikte sağlanamaz. Sınırlı beklenen gecikme veya her zaman kesin keşif daha güçlü hedefler olduğu için onlar da bu engeli aşamaz.

**Nitelikler:** Sabit pozitif bedel önemlidir; ucuzlayan veya ücretsiz yan ölçüm sonucu değiştirebilir. Bir deneme başarısız olsa da sonraki gözlemleri bilgilendiriyorsa eşleme varsayımı tekrar incelenir. Gürültülü geri bildirimde deneme ile kesin tespit aynı olay değildir; yukarıdaki tam eşitlik doğrudan kullanılamaz. Değişimin olası zamanı sınırlandırılmışsa, pasif ipuçları varsa veya eylemler dünyayı değiştiriyorsa başka sonuçlar mümkündür. `E[N∞]` için bir üst sınır tek başına `τ` cinsinden evrensel bir düşüş hızı vermez; son deneme zamanı çok ağır kuyruklu olabilir. İç sayaç, rastgele sapma veya daha karmaşık hesaplama, aynı gözlem ve bedel varsayımları içinde bu önermeden kaçamaz.

Bu önerme burada açık eşleme argümanıyla türetilmiştir. Literatürde ilk kez bulunduğu veya genel güvenli keşif teoremi olduğu iddia edilmez.

## 9. Çalıştırılmış yeni deneyin bağımsız denetimi

[unknown_change_experiment.py](../../../research/living_learning/unknown_change_experiment.py) ve [tam kayıt](../../../research/living_learning/results/unknown_change.json) bu notun ilk türetiminden sonra okundu. Yukarıdaki doğru `t,f` durumunu, fazı yeniden başlatan dondurulmuş ağırlık kontrolünü ve bütün açılma tarihleriyle kuyruğun tam toplamını kullanıyor. Gizli `τ` yalnız simülatörde gerçek deneme sonucunu üretirken kullanılıyor. Bellman değeri, ayrı dünya yürütmeleri üzerinden tam toplamla `10⁻⁸` toleransta doğrulanıyor.

| Maliyet | Güncellenen karışım | Dondurulan ağırlık | Tek `h=.01` | Gerçek hız verilen kontrol |
|---|---:|---:|---:|---:|
| 1 | 69.527959 | 65.668642 | 66.654044 | 73.593780 |
| 20 | 37.323585 | 26.621452 | 26.361483 | 54.576609 |

Tablo, başlangıçtaki üç hıza eşit ağırlık veren **tam beklenen faydaları** gösterir; Monte Carlo tahmini veya güven aralığı değildir. Karışım optimumu gerçekten güncellenmeyen rakiplerden yüksektir; gerçek hızın ek bilgisi daha yüksek değer sağlayabilir.

Hiç açılmayan dünyada `c=1` için güncellenen politikanın denemeleri `16,33,52,74,99,128,161,199` anlarındadır. Dondurulmuş kontrol 18 kez dener. `c=20` için güncellenen politika yalnız `83` anında dener; dondurulmuş kontrol `49,101,160` anlarında dener. Bu sonlu ufukta geçmiş başarısızlığın gelecekteki veri toplama maliyetini azaltabildiği görülür; aralık artışını sonsuz yaşam teoremi veya her dünya için doğru miktar olarak yorumlamamak gerekir.

**Negatif koşullu sonuç:** Gerçek `h=.02`, `c=20` olduğunda güncellenen karışımın faydası `124.719926`, dondurulmuş kontrolünki `150.991673` olur. Önselde bulunmayan `h=.08` için aynı değerler `157.824285` ve `190.772032`'dir. Güncelleme hem model-içi bir bileşende hem bu model-dışı hızda daha kötü olabilir. Önsel ortalamasında optimalite, her bileşende veya her gerçek dünyada üstünlük değildir.

Bir çıktı alanına dikkat edilmelidir: `posterior_hazard_weights_given_last_failure`, son başarısızlık bilgisine göre yeniden hesaplanan bir tanı değeridir. Her politika için üretilir ve sonradan gelen başarıyı içermez; başarı sonrası **tam son posterior** olarak okunmamalıdır. Başarı sonrası hızı çıkarsamak istenirse §3'teki aralık olabilirliği gerekir. Mevcut modelde artık açık durumu kullanmak için hız tahminini sürdürmek gerekmez.

**Denetim durumu:** Formüller bağımsız türetildi, dört posterior satırı ayrıca hesaplandı, kaynak kod ve makinece okunabilir sonuçlar incelendi. Kritik nedensel sızıntı veya Bellman indeks hatası bulunmadı. Bu not kod değiştirmedi. Yeni deneyin hız-kümesi-dışı testi bulunduğu, fakat gözlem gürültüsü ve yeniden kapanma için yeni karışımda tutarlı bir model keşif yöntemi gerçekleştirilmediği kayda geçirildi.
