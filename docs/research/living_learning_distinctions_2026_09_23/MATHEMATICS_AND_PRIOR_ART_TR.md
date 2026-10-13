# Tur 19 — Geçici eşdeğerlikten yürütülebilir ayrımlara

Muhammed Yasin Yılmaz · 23 Eylül 2026 · Matematik ve literatür notu

Bu not [protokoldeki](PROTOCOL.md) mekanizmanın neden çalışabildiğini ve neyi kanıtlamadığını açıklar. Sonuçlar [deney raporundadır](REPORT_TR.md). Gözlem tablosu ile durum ayrıştırma bilinen otomata öğrenimidir; yeni bir matematiksel ilke olarak sunulmaz.

## 1. Şimdilik aynı görünen ile gerçekten eşdeğer olan

İkili alfabenin bütün sonlu sözcükleri Σ*, dünyanın son-çıktı fonksiyonu y:Σ*→{0,1} olsun. h bir geçmiş, e bir devamdır. Öğrenicinin mevcut sonlu devam kümesi E için

\[
h\sim_E h'\quad\Longleftrightarrow\quad
\forall e\in E:\ y(he)=y(h'e).
\]

Bu, geçici gözlemsel eşdeğerliktir. Bütün devamlar altındaki eşdeğerlik ise

\[
h\sim_* h'\quad\Longleftrightarrow\quad
\forall e\in\Sigma^*:\ y(he)=y(h'e).
\]

E⊆F olduğunda h∼_F h' olması h∼_E h' olmasını gerektirir. Tersi zorunlu değildir. Böylece yeni bir devam sonucu, daha önce birlikte tutulan geçmişleri ayırabilir. Dünyanın y fonksiyonu sabitse gerçekten ∼_* eşdeğer olan geçmişler sonradan ayrılmaz. Değişen, öğrenicinin sonlu kanıtla kurduğu bölümlemedir; dünyanın anlamının değiştiği gösterilmez.

Bu turun mekanizması ayrım ekler. Sabit ve doğru etiketler altında aynı iki geçmiş için daha önce görülmüş gerçek bir ayrımı silmez. Birleştirme, değişen dünya, duygusal değerlendirme veya geçmiş olayın yeni bağlamda farklı anlam kazanması bu mekanizmayla bütünüyle açıklanmaz.

## 2. Yeni devam nasıl oluşturuluyor?

Bir satır, geçmiş h için erişilebilir devam çıktılarının vektörüdür:

\[
\rho_E(h)=(y(he))_{e\in E}.
\]

İki geçmiş aynı satırdayken a eylemi sonrasında satırları ayrılıyorsa, bir e∈E için

\[
\rho_E(h)=\rho_E(h'),\qquad y(hae)\ne y(h'ae)
\]

bulunur. O hâlde yeni devam **ae**, h ve h'yi ayırır. e bilinen bir devam, a verilen alfabeden bir eylemdir. Yeni ayırıcı, doğru hedef durum kimliğinden veya hazır tüm-devam listesinden alınmaz; önceki deneylerden bulunan tutarsızlıktan kurulur. Bununla birlikte “eylemi devamın önüne bağla” yordamı araştırmacı tarafından verilmiştir. Öğrenici yeni bir temel işlem veya dil icat etmiş sayılmaz.

Üç durumlu somut tanık:

| Durum | Şimdiki çıktı | 0 sonrası | 1 sonrası |
|---|---:|---|---|
| q0 | 0 | q1 | q0 |
| q1 | 0 | q1 | q2 |
| q2 | 1 | q2 | q2 |

Başlangıç q0. Yalnız boş devamla ε ve `0` aynı görünür. `01→1` ile `1→0` gözlemleri gelince `1` devamı bunları ayırır. Yeni bir durum ayrımı, henüz görülmemiş daha uzun sözcüklerde doğru yürütmenin parçası olabilir. Bu mekanizma tanığı kodda ayrıca doğrulandı; bütün performans deneyi bu özel dünyadan oluşmuyor.

## 3. Satırlardan hesap yapısına geçişin koşulu

S erişim geçmişleri olsun; boş sözcüğü içerir ve öneklere kapalıdır. E boş devamı içerir ve son-eklere kapalıdır. Tabloda S ile SΣ satırları ölçülür.

**Kapalılık:** Her s∈S ve a∈Σ için ρ_E(sa), S'deki bir satıra eşittir.

**Tutarlılık:** ρ_E(s)=ρ_E(t) ise her a için ρ_E(sa)=ρ_E(ta).

O zaman satır sınıfları üzerinde

\[
\delta(\rho_E(s),a)=\rho_E(sa),\qquad
g(\rho_E(s))=y(s)
\]

iyi tanımlıdır. Kapalılık hedef satırın mevcut durumlar içinde olmasını; tutarlılık temsilci geçmiş seçiminin geçişi değiştirmemesini sağlar. Boş devam E'de olduğu için aynı satırdaki geçmişlerin mevcut çıktısı da aynıdır. Bu nedenle tablo bir yürütülebilir deterministik makineye derlenebilir.

Ancak bu koşulların sonlu tabloda sağlanması **dünyayla bütün dizilerde eşdeğerlik sertifikası değildir**. Tablo dışındaki bir devam ayrım gerektirebilir. Uygulama, modelin mevcut bütün bilinen kayıtları açıklamasını da arar; bu kayıtların ötesi yine ayrıca sınanmalıdır.

Sonlu bir kayıt D'de bütün etiketler sıfırsa, sürekli sıfır veren bir model D ile uyumludur. Denenmemiş bir sözcükte bir veren başka sonlu bir makine de D ile uyumlu olabilir: yalnız o sözcüğü kabul eden makine bunun genel yapıcı örneğidir. Dolayısıyla D tek başına ikisini seçmeye yetmez. Bu sonuç bütün öğrenmenin zorunlu olarak aktif sorgu gerektirdiğini söylemez; yeni pasif gözlem de ayrımı sağlayabilir. Ayrıca doğru durum sınırı, dünya ailesi veya başka önbilgi verilirse çıkarım koşulları değişir.

## 4. Öğrenici bilmezken değerlendirici nasıl tam doğrulayabiliyor?

İki makinenin başlangıç durum çiftiyle başlayıp aynı sembolleri ikisine de uygularız. Erişilebilir bütün durum çiftlerinin çıktıları eşitse bütün sonlu sözcüklerde eşdeğerdirler. Farklı çıktılı bir çift varsa ona götüren sözcük karşı örnektir. Sonlu durum çarpımı nedeniyle bu denetim sonlanır.

Deneyci gizli makineyi bilir; öğrenici bilmez. Bu ayrım korunmuştur: değerlendiricinin karşı örneği öğreniciye verilmez. Bu yüzden rapordaki tam eşdeğerlik, öğrenicinin kendi başına elde ettiği bir sertifika değildir. Bağımsız denetimde iki makine birleştirilip çıktı/geçiş bölümlemesi incelendi; aynı sonuçlar farklı bir algoritmayla doğrulandı.

Tam doğru model, aynı sabit dünyanın sonraki dizilerini hesaplamak için tek başına yeterlidir: başlangıçtan geçişleri ve çıktı tablosunu yürütür. Bu, ham arşivin aynı dünya için **tahmin** sırasında zorunlu olmadığını gösterir. Uygulama arşivi silmedi; yeni dünya, yeni çıktı türü veya daha sonraki temsil değişimi için modelin yeterliliği buradan çıkmaz.

## 5. Kalıcı örgü ile yeniden kurmanın farkı

Kalıcı yöntemin S, E ve bilinen etiketlerini s_t ile gösterelim. Yeniden kurma kontrolü aynı mantıksal geçmişi aynı sırayla aynı deterministik yordamdan geçirir. İlk durumda eşitlik ve her adımda aynı sorgu/etiket/yordam varsa indüksiyonla son öğrenici durumları aynıdır. Dış sorgu önbelleği yalnız daha önce alınmış yanıtları sağlar; kontrol gelecekteki kayıtları erken okuyamaz.

Bu eşitlikte bir yöntemin diğerinden farklı bilgi edinmesi beklenmez. Fark, daha önce tamamlanmış ara örgüyü korumak yerine yeniden hesaplamanın maliyetidir. Deney bunu 72 dünyada aynı sorgu sırası, aynı dış kanıt ve aynı bütün öğrenici durumu üzerinden kontrol eder. Elde edilen iş tasarrufu, bilinen bir kalıcılık/yeniden kullanım etkisidir; daha iyi bir çıkarım algoritması bulunduğu anlamına gelmez.

## 6. Literatür eşleşmesi ve iddia sınırı

**Angluin, 1987.** Özgün makale gözlem tablosuyla öğrenmeyi, üyelik ve eşdeğerlik sorgularını tanımlar. Bizim S/E ve tutarsızlıktan devam ekleme mekanizmamız bununla doğrudan eşleşir. Makalenin tam öğrenme garantisindeki eşdeğerlik öğretmeni bu deneyde yoktur. Çalışma örnekleme seçeneğini de inceler; bizim sabit 64 gözlemimiz onun doğruluk/güven parametreli durma kuralının uygulaması değildir. Dolayısıyla o garantiyi bu sonuca taşımıyoruz. [Learning Regular Sets from Queries and Counterexamples, özgün tam metin](https://homepages.math.uic.edu/~lreyzin/papers/angluin87.pdf), Information and Computation 75, 87–106. Erişim: 23 Eylül 2026.

**Rivest ve Schapire, 1989.** Açılan kaynak 1989 genişletilmiş konferans özetidir; 1993 dergi sürümü olarak etiketlenmez. Çalışma, başlangıca sıfırlama verilmeden sonlu makinelerin öğrenilmesini ve homing dizilerini ele alır; başlıca düzeneğinde yanlış modeller için karşı örnek kaynağı yine vardır. Bu, “reset'i kaldırmak başlı başına yeni temel fikir olur” iddiasını engeller. Mevcut deney bu algoritmayı uygulamadı ve onun garantilerini göstermedi. [Inference of Finite Automata Using Homing Sequences, yazar arşivi](https://people.csail.mit.edu/rivest/pubs/RS89.pdf). Erişim: 23 Eylül 2026.

Bu kaynaklar kullanıcının kavramların yaşam boyunca gelişmesi önerisini bütün olarak açıklamaz. Burada yalnız sınırlı bir karşılığı çalıştırılmıştır: alınan deneyim, daha önce birlikte tutulan geçmişlere uygulanan yeni bir hesap ayrımı oluşturabilir; bu ayrım sonraki yürütmede ve öğrenmede kalabilir. Anlam, değer, amaç ve yeni gözlem dilinin kuruluşu hâlâ araştırma konusudur.
