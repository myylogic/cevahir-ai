# Tur 21 — matematiksel ayrımlar ve önceki çalışmalar

Muhammed Yasin Yılmaz · 23 Eylül 2026

Buradaki önermeler yeni keşfedilmiş genel öğrenme kuramı diye sunulmuyor. Amaç, “kendi kendine gelişme” sözünün farklı iddialarını ayırmak ve deneyin neyi gösterebileceğini kesinleştirmek.

## 1. Yeni dünya bilgisi olmadan yeni erişilebilir sonuç

Dış dünyaya ilişkin rastgele değişken W, şimdiye kadarki dış kanıt H, iç rastgelelik R olsun. R'nin W'den H koşulunda bağımsız olduğunu varsayalım. İç hesap ve kalıcı durum Z=f(H,R) ise W → H → Z bir Markov zinciridir. Veri işleme eşitsizliği:

I(W;Z) ≤ I(W;H).

Daha doğrudan: I(W;Z | H)=0. Kanıtın dışında dünyaya bağlı bir giriş veya korelasyon yoksa iç hesap yeni ayırt edici dünya bilgisi oluşturmaz. Süre, yan kanal, çevreyle etkileşim veya W ile ilişkili rastgelelik varsa varsayım değişir. H'yi saklamayan sistemin Z'si H'den daha az bilgi taşıyabilir; “iç hesap hiç öğrenme değildir” sonucu çıkmaz.

Sonlu kaynakta durum farklıdır. Algoritma A için K_B(F,L), F ve kalıcı ara sonuçlar L ile en fazla B işte **doğru** cevaplanabilen sorgular kümesi olsun. F ⊨ L olduğunda Mod(F∧L)=Mod(F), fakat K_B(F,L) ile K_B(F,∅) farklı olabilir. Aynı olası dünyalar korunurken doğru sonuçlara erişim değişir. Bu, sınırsız Bayes çıkarımına yeni bilgi üstünlüğü değildir. Maliyet ölçütü ve algoritma değişirse K de değişir.

Bir öğrenici için yalnız |K_B|'nin artması yeterli değildir: edinim, kanıt denetimi ve depolama gideri önceden veya sorgular arasında harcanmıştır. Tur 21 servis eşiği sonuçlarını bu nedenle toplam yaşam maliyetinden ayrı tutar. Artan doğru lemma sayısının K_B'yi kapsama yoluyla büyütmesi garanti değildir; aynı algoritma yararsız doğru lemmaları da tarıyorsa daha fazla iş yapabilir.

## 2. Yerel varsayımı küresel bilgiye dönüştürmenin doğru koşulu

F∧a₁∧…∧aₖ çözümsüzse:

F ⊨ (¬a₁∨…∨¬aₖ).

İspat: F'yi sağlayan ve bütün aᵢ'leri doğru yapan bir değerleme bulunması çözümsüzlükle çelişir. Deneyde a'lar sorgunun bir alt kümesidir; silme denemeleri bu kümeyi daraltır. Son kanıt o alt küme altında yeniden denetlenir. Bu, öğretmenden bir cevap etiketi almak değildir; verilen mantıksal kanıtın sonucunu hesaplamaktır.

Karşı örnek: F=(a∨b), sorgu varsayımı ¬a. Yerel b çıkarımını koşulsuz saklamak hatadır. Sonraki ¬b sorgusunun çözümü a=1,b=0 iken sistem yanlış çözümsüzlük bildirir. Güvenli saklama a∨b'dir. Daha önce tabanda bulunmayan örnek: F=(a∨b)∧(¬b∨c); ¬a altında c çıkabilir, fakat küresel lemma a∨c'dir.

## 3. Kanıtın doğruluğu ile kanıtın hâlâ geçerli olması

Bir lemma l için D_l ⊆ F ve D_l ⊨ l kanıtlanmış olsun. Sonraki taban F' içinde D_l ⊆ F' ise F' ⊨ l. İspat, mantıksal çıkarımın öncül eklemeye göre monotonluğudur. Bir ara lemma kullanıldıysa onun dayanaklarını özyinelemeli olarak başlangıç maddelerine açarak bu koşul korunur.

Ters yön zorunlu değildir: D_l'nin bir maddesi silinse bile başka bir kanıt l'yi geçerli kılabilir. Buradaki koruma **yeterli, gerekli olmayan** tutucu bir ölçüttür. Bütün eski lemmaları silmekten daha seçici olabilir; en ekonomik yeniden doğrulama olduğu ispatlanmadı. Temel maddeler dünyayı yanlış anlatıyorsa, kusursuz mantık denetimi dünya doğruluğu sağlamaz.

Denetleyicimiz her dalı, birim zorlamayı ve çatışmayı kontrol eder. Bu bir uygulama düzeyi bağımsız denetleyicidir; Coq/Lean gibi bir çekirdekte makineyle doğrulanmış metateori değildir. Bağımsız küçük dünya doğrulaması ayrıca kullanılır.

Her “kanıt kontrolü” aynı sözleşmeyi vermez. RAT biçimindeki bazı adımlar yalnız çözülebilirlik eşdeğerliğini korur. Örneğin boş F'ye a eklemek bu anlamda zararsızken, sonraki ¬a sorgusunda zararlıdır. Model kümesini koruyan mantıksal çıkarım ile yalnız SAT/UNSAT durumunu korumayı ayırıyoruz. [Wetzler, Heule ve Hunt'ın DRAT-trim makalesindeki RAT tanımı](https://www.cs.cmu.edu/~mheule/publications/drat-trim.pdf) bu ayrımın kaynağıdır; bizim motorumuz DRAT uygulaması değildir.

## 4. Eşleme öğrenmesinin simetri sınırı

Henüz uygulanmayan H21.6/H21.10 için: iki aday eşleme T₁,T₂ bütün erişilebilir kaynak geçmişlerine aynı olabilirliği veriyorsa, kaynak gözlemleri bu ikisini ayıramaz. Eşit önsel altında sonsal oran değişmez. Yeni görev bu eşlemelere göre farklı davranış gerektiriyorsa, yalnız bu geçmişten kesin doğru eşlemeyi seçme güvencesi yoktur. Bir avantaj için ayırt edici deneyim, ek yapısal varsayım veya hedef geri bildirimi gerekir.

Aynı geçmiş, doğru olasılık modeli ve aynı önselle tam Bayes tahmini, beklenen uygun puanlama kaybını en aza indirir. Öğrenilmiş dönüşümün avantajı bunun üzerinde bilgi yaratmak diye değil; kısıtlı hesapta o çıkarıma yaklaşma, farklı varsayım veya temsil/arama maliyeti diye ayrılmalıdır. Bunlar sonlu deneyde Bayes dışı bütün yöntemlerin başarısız olacağı iddiası değildir; koşullar özellikle güçlüdür.

## 5. Birincil literatür eşleşmeleri ve erişim kapsamı

- [Minton, 1988 — Quantitative Results Concerning the Utility of Explanation-Based Learning](https://cdn.aaai.org/AAAI/1988/AAAI88-100.pdf). Tam metin incelendi. Açıklamadan edinilen arama bilgisinin faydasının, uygulama ve eşleştirme maliyetini aşması gerektiğini ele alır. Bu turun ara sonuçların toplam maliyeti sorusu doğrudan önceki çalışmayla eşleşir. Yeni isimle sahiplenilmez.
- [Darwiche ve Marquis — A Knowledge Compilation Map](https://arxiv.org/abs/1106.1819). Bu turda yazarların arXiv özet sayfası incelendi; arXiv yükleme yılı 2011'dir, eserin ilk yayın tarihi diye kullanılmadı. Derlenmiş temsilin kısalığı ve desteklediği sorgular arasındaki ayrım, tam tablo kontrolünün neden önemli olduğunu açıklar. Tam metin incelenmiş veya yöntemleri yeniden uygulanmış sayılmaz.
- [Wetzler, Heule ve Hunt, 2014 — DRAT-trim](https://www.cs.cmu.edu/~mheule/publications/drat-trim.pdf). Tam metindeki 2. bölüm, denetim ve bağımlılık açıklamaları incelendi. RAT/RUP kapsam farkı yukarıda kullanıldı. Bizim küçük ağaç denetleyicimizle endüstriyel performans kıyaslaması yapılmadı.
- [Baxter, 2000 — A Model of Inductive Bias Learning](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume12/baxter00a.pdf). Yazar makalesi açıldı; görevler arası öğrenilmiş önyargı yönü için önceki literatür kaydıdır. Bu turdaki SAT sonucunu destekleyen deney olarak kullanılmadı; H21.6–7 ilerletilirse varsayımları ayrıca ayrıntılı incelenmeli.
- GRASP için yazar/üniversite teknik raporu adresi denenmiştir; ana ajanın erişimi zaman aşımına uğramıştır. Bu kaydı tam metni incelenmiş kaynak veya doğrulanmış belirli teknik iddia olarak kullanmıyoruz. Minton'ın 1990 dergi sayfasına erişim de başarısızdır; yukarıdaki erişilebilir 1988 metnine dayanıyoruz.

Bu tarama kapsamlı yenilik taraması değildir. EBL, bilgi derleme, artımlı çözümleme ve geçerlilik bağımlılıklarıyla açık eşleşmeler varken “yeni temel ilke bulundu” denemez. Turun değeri, Cevahir araştırmasında bilgi edinimi ile hesap erişimini ve doğruluk ile kalıcı geçerliliği aynı denetlenebilir kayıtta ayırmasıdır.
