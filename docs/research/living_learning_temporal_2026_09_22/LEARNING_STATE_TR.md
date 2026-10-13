# Bugünkü cevap ile yarın öğrenmeye yeterli durum aynı değildir

Muhammed Yasin Yılmaz · 22 Eylül 2026 · Tur 15 kuramsal ek

Bu ek ana deneyin parametrelerini değiştirmez. Koşullu matematiksel çıkarımlar ve tam aritmetik karşıörnekler içerir; yeni genel öğrenme kuramının bulunduğu iddiasını içermez.

## 1. Öğrenicinin devamını da korumak

`h`, şimdiye kadar yaşanmış deneyimlerin dizisi; `z`, daha sonra gelebilecek deneyimlerin dizisi; `q`, bir davranış veya tahmin sorgusu olsun. Belirli bir deterministik öğrenme yordamının bu geçmişten sonra üreteceği cevaba `A(h,q)` diyelim. Öğrenme yordamı burada verili bir karşılaştırma nesnesidir; bu tanım onun nasıl icat edileceğini söylemez.

İki farklı eşdeğerlik vardır:

\[
h\sim_{cevap}h' \iff \forall q\ A(h,q)=A(h',q),
\]

\[
h\sim_{ogrenme}h' \iff \forall z,q\ A(hz,q)=A(h'z,q).
\]

İkincisi boş `z` dahil bütün devamları içerdiği için birincisini gerektirir. Tersi doğru değildir. Rastgele sistemler için aynı soruyu çıktıların ortak dağılımları üzerinden kurmak gerekir; aşağıdaki ispat deterministik kurulum içindir.

**Önerme.** Geçmişi `S(h)` durumuna sıkıştıran, sonra yalnız bu durumdan güncellenen bir uygulama, verilen `A` yordamının gelecekteki bütün cevaplarını aynen koruyacaksa,

\[
S(h)=S(h')\ \Longrightarrow\ h\sim_{ogrenme}h'
\]

olmalıdır. Aksi halde aynı duruma aynı gelecek deneyimler uygulandığında iki farklı gerekli cevap üretilemez. Tersine, öğrenme eşdeğerliği sınıflarını durum yapmak soyut olarak yeterlidir: `U([h],e)=[he]` iyi tanımlıdır; çünkü aynı devam eşitliği `ez` için de geçerlidir. Ancak sınıfların sayısı sonsuz olabilir, üyelik hesaplanamayabilir veya kaynak bütçesi içinde bulunamayabilir. Bu bir varlık/koruma koşuludur, uygulanabilir keşif algoritması değildir.

Bu devam eşdeğerliği mantığı, bilinen Nerode ve durum indirgeme düşüncesinin öğrenici davranışına uygulanmasıdır. İsim değiştirerek yenilik iddia edilmiyor. [Maletti'nin özgün çalışması](https://www.informatik.uni-leipzig.de/alg/pub/mal04b-full.pdf), kendi cebirsel varsayımları altında devam eşdeğerliğiyle sıralı dönüştürücülerin küçültülmesini inceler. Yukarıdaki öğrenici tanımına o makalenin sonlu-durum veya hesaplanabilirlik garantileri otomatik olarak aktarılmaz.

## 2. Tam aritmetik karşıörnek

Tur 15'teki basit sayaç tahminini kullanalım:

\[
p(n_0,n_1)=\frac{n_1+1/2}{n_0+n_1+1}.
\]

Hiç örneği olmayan `(0,0)` durumuyla 99 sıfır ve 99 bir görmüş `(99,99)` durumu bugün aynı `1/2` cevabını verir. Yeni bir `1` deneyiminden sonra cevaplar `3/4` ve `201/400 = 0.5025` olur. Yalnız bugünkü `p` saklanırsa bu iki devam aynen korunamaz. Örnek sayısı veya aynı işi gören başka bir ayrım da gerekir.

Genel olarak `(k,k)` geçmişlerinin hepsi bugün `1/2`, bir yeni `1` sonrasında ise

\[
\frac{k+3/2}{2k+2}=\frac12+\frac{1}{4k+4}
\]

verir. Farklı `k` değerleri farklı güncelleme gerektirir. Bu **belirli exact sayaç yordamını** sonsuza kadar aynen sürdürmek sonlu sayıda durumla mümkün değildir. Yaklaşık, sınırlı hassasiyetli veya unutmalı başka öğrenicilerin işe yaramayacağı söylenmiyor.

İstatistikte yeterlilik de seçilen model ailesine göre tanımlanır; her gelecek model değişimine yeterlilik değildir. [Fithian'ın Berkeley ders notu](https://www.stat.berkeley.edu/~wfithian/courses/stat210a/sufficiency.html), bu modeli bağımlı tanımı ve sabit örnek sayısındaki Bernoulli toplamı örneğini açıklar. Buradaki değişken uzunluklu yaşamda örnek sayısını kaybetmemek ek olarak gereklidir. Sayaç örneği açıklayıcı bir tanıktır; genel araştırmanın Bayes yaklaşımını zorunlu kullanması sonucunu doğurmaz.

## 3. Yeni bir temsil için eksik geçmiş kaç bit eder?

Eski özet `φ(h)`, edinilmek istenen yeni özet `ψ(h)` olsun. Eski özetten yenisine dönüşümün koşulu önceki [matematik notunun](MATHEMATICAL_AND_LITERATURE_TR.md) 8. bölümünde kanıtlandı. Koşul sağlanmıyorsa yardımcı bir iz `r(h)` tutmak gerekebilir.

Sonlu bir geçmiş kümesinde, sabit uzunluklu ve hatasız bir yardımcı kod için

\[
M=\max_s\left|\{\psi(h):\phi(h)=s\}\right|,
\qquad b\geq\lceil\log_2 M\rceil.
\]

**Alt sınır.** Aynı eski durum içinde `M` farklı yeni durum ayrılmalıdır. `b` bit en çok `2^b` değer ayırır. **Varlık.** Her eski durumun içindeki farklı yeni durumlara yerel bir numara verilebilir; eski durum ve bu numara birlikte yeni durumu belirler. Bu ikinci adım, kodlama/kod çözme tablolarının bilindiğini varsayar; tabloların depolama ve hesaplama maliyeti `b`'ye dahil değildir. Bu nedenle ekonomik bir öğrenici inşa edildiği söylenemez. Bu, koşullu hatasız kodlama için sayma argümanıdır; özgün bir bilgi kuramı teoremi iddiası değildir.

Somut tanık: en az altı bitlik akışta eski özet parity, yeni özet `(parity, son beş bit)` olsun. Her parity değerinin altında 32 farklı beş-bit son-ek bulunabilir. Bölüm ortasında kayıpsız geçişi önceden güvenceye almak için eski parity bitinin yanında en az beş bitlik yardımcı ayrım gerekir. Yalnız parity saklanmışsa bu ayrımlar geçmişten geri çıkarılamaz. Yeni girdiler onları zamanla yeniden oluşturabilir; fakat geçmiş anın eksik beş-bit bilgisini geriye dönük sağlamaz.

Birlikte parity ve gecikme-beş cevaplarını bütün devamlar boyunca verebilen çalışma belleği 64 durum gerektirir ve `(parity, son beş bit)` ile buna ulaşır. Alt sınır için 64 adet altı-bit sözcüğü alalım. Parity'leri farklıysa boş devam ayırır. Parity aynı, son beş bit farklıysa farklı son-ek konumunu terminal hedefe taşıyan en çok dört sıfır ayırır. Bu, 64 geçmişin çiftler halinde ayırt edilebilirliğini gösterir. Öğrenme istatistikleri ve kod maliyeti bu 64 durumun ayrıca dışındadır.

## 4. Ana araştırmadaki karşılığı

Yaşarken öğrenen bir sistemde en az üç ayrı koruma sorusu var:

1. **Çalışma belleği:** Şu andaki görevi sürdürmek için geçmişten hangi ayrımlar gerekli?
2. **Öğrenmeye devam etmek:** Yeni deneyim geldiğinde güncellemeyi yapmak için hangi kanıt ve belirsizlik bilgisi gerekli?
3. **Yeni bir yetkinliğe geçmek:** Şimdi kullanılmayan hangi ayrım, gelecekte başka bir temsil kurulurken gerekli olabilir?

Birincisini başarıyla sıkıştırmak, ikinci ve üçüncüyü otomatik korumaz. Buna karşılık her olası gelecekteki görev için geçmişi hatasız yeniden yorumlama şartı da aşırı güçlüdür: gelecekteki sorgular her eski biti tek tek sorabiliyorsa `n` bitlik geçmişin `2^n` olasılığı ayırt edilmeli, en az `n` bit korunmalıdır. Bu şart altında sabit boyutlu evrensel kayıpsız bir özet yoktur. Daha kısıtlı görev aileleri veya yaklaşık güvenceler başka bir problemdir.

Kullanıcının kavramların yeniden kümelenmesi ve eski olayların yeni anlamlara bağlanması fikri bu üçüncü soruyu özellikle görünür kılıyor. Burada insan beyninin bunu böyle yaptığı kanıtlanmadı. Çıkarım hesaplamasaldır: bir yetkinliğin değişmesi, eski iç durumdan yeni iç duruma geçişin bilgi maliyetini de doğurur.

Araştırmanın açık yönü artık yalnız hangi cevabın öğrenildiği değildir. Sistem, gelecekteki öğrenmesini destekleyecek ayrımları hangi deneyimlere dayanarak, hangi bütçeyle ve hangi kayıp güvencesiyle koruyabilir? Bu soru hazır bellek ailelerinden seçimle tamamen çözülmüş sayılmaz.
