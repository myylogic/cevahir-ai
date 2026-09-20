# CEVAHIR — LIVING LEARNING OPEN RESEARCH PROBLEM

Cevahir için yeni bir temel araştırma problemi üzerinde çalış.

Bu bir feature-development görevi değildir.

Bu bir repository bakım görevi değildir.

Bu bir Transformer geliştirme görevi değildir.

Bu bir continual-learning yöntemini mevcut sisteme entegre etme görevi değildir.

Bu bir multimodality implementasyonu değildir.

Bu bir mevcut araştırma fikrini doğrulama görevi değildir.

Araştırılması gereken problem şudur:

# BİR YAPAY SİSTEM ÇALIŞTIĞI SÜRE BOYUNCA NASIL ÖĞRENEBİLİR?

Bugünkü yapay zekâ sistemlerinde öğrenme çoğunlukla belirli bir training sürecinde gerçekleşir.

Veri hazırlanır.

Bir model belirli bir matematiksel öğrenme/optimizasyon mekanizmasıyla eğitilir.

Eğitim sonucunda oluşan yapı daha sonra inference amacıyla kullanılır.

Model deployment sonrasında yeni girdileri işler ancak yaşadığı her yeni deneyim doğal olarak onun kalıcı öğrenilmiş yapısının bir parçası haline gelmez.

Cevahir için araştırmak istediğimiz sistem bu olmak zorunda değildir.

Hedefimiz:

# ÇALIŞMASI İLE ÖĞRENMESİ BİRBİRİNDEN KOPUK OLMAYAN BİR YAPAY SİSTEMİN MÜMKÜN OLUP OLMADIĞINI VE MÜMKÜNSE BUNUN TEMEL HESAPLAMA VE MATEMATİKSEL PRENSİBİNİ BULMAKTIR.

Sistem dünyayla etkileşim halindeyken yeni deneyimler yaşayabilmelidir.

Bu deneyimler gerektiğinde sistemin gelecekteki davranışını değiştirebilmelidir.

Bu değişiklik yalnız mevcut konuşma bağlamına, prompt geçmişine veya dışarıdan tekrar getirilen bir kayda bağlı olmamalıdır.

Sistem gerçekten öğrenebilmelidir.

Yeni bir şey öğrendiğinde daha sonra bunu kullanabilmelidir.

Öğrendiği şeyler başka deneyimlerle birleşebilmelidir.

Yeni öğrenme eski faydalı yetenekleri kontrolsüz biçimde yok etmemelidir.

Yanlış veya artık geçerli olmayan öğrenmeler değiştirilebilmelidir.

Yaşanan deneyimlerin etkisi yalnız deneyim gerçekleştiği anda ortaya çıkmak zorunda değildir.

Sistem bazı dönemlerde dış dünyayla yoğun biçimde etkileşirken bazı dönemlerde daha önce yaşadıklarının üzerinde kendi içinde çalışabilir.

İnsanların uyku ve dinlenme dönemlerinde deneyimlerinin sinir sisteminde yeniden işlenmesi bu problem için yalnızca bir ilham kaynağıdır.

İnsan beynini kopyalamak hedef değildir.

Araştırılması gereken daha temel soru şudur:

# BİR SİSTEM YENİ DIŞ DENEYİM ALMADIĞI BİR DÖNEMDE DAHA ÖNCE YAŞADIKLARINDAN YOLA ÇIKARAK KENDİSİNİ DAHA İYİ ORGANİZE EDEBİLİR Mİ?

Ve eğer yapabiliyorsa:

# BU NE DEMEKTİR?

Yeni bilgi mi oluşmaktadır?

Var olan bilgiler arasında daha önce kullanılmayan ilişkiler mi ortaya çıkmaktadır?

Bir yetenek mi oluşmaktadır?

Sistemin iç hesaplama biçimi mi değişmektedir?

Yoksa bugün başka isimlerle zaten bildiğimiz bir mekanizma mı gerçekleşmektedir?

Bunu araştır.

---

# TEMEL PROBLEM

Bugünkü neural-network sistemlerinde öğrenmenin çok önemli bir bölümü eğitim sırasında parametrelerin değiştirilmesine dayanır.

Cevahir açısından şu soruyu araştır:

# LEARNING GERÇEKTEN YALNIZCA PARAMETRELERİN BİR OBJECTIVE DOĞRULTUSUNDA OPTİMİZE EDİLMESİ OLARAK MI TANIMLANMALIDIR?

Eğer cevap hayırsa:

# DAHA GENEL TANIM NEDİR?

Bir yapay sistemin “öğrenmiş olması” matematiksel ve gözlenebilir olarak ne anlama gelir?

Bir sistemin yaşamının bir anındaki hali ile daha sonraki hali arasındaki hangi değişiklik learning olarak kabul edilmelidir?

Bir experience'ın gelecekteki davranışı değiştirmesi için sistemde gerçekte ne değişmelidir?

Bunun matematiksel karşılığı nedir?

Bu değişiklik nasıl kalıcı hale gelir?

Ne zaman geçici kalmalıdır?

Yeni öğrenme ile eski bilgi nasıl birlikte yaşayabilir?

Bir sistem zaman içerisinde yalnız bilgisini değil, kendi öğrenme biçimini de değiştirebilir mi?

Bir sistemin öğrenme kapasitesi kendi deneyimleri nedeniyle gelişebilir mi?

Bir sistem zaman içerisinde başlangıçta sahip olmadığı bir hesaplama veya temsil biçimini oluşturabilir mi?

Bunların cevaplarını önceden varsayma.

Bul.

---

# ASIL ARAŞTIRMA HEDEFİ

Bu çalışmanın sonucunda şu soruya mümkün olduğunca güçlü bir cevap arıyoruz:

# “YAŞARKEN ÖĞRENME”NİN TEMEL MATEMATİKSEL VE COMPUTATIONAL FORMÜLÜ NEDİR?

Buradaki “formül” kelimesini tek satırlık denklem üretme zorunluluğu olarak yorumlama.

Doğru cevap bir denklem olabilir.

Birden fazla denklem olabilir.

Bir dinamik sistem olabilir.

Bir algoritma olabilir.

Yeni bir computational formalism olabilir.

Mevcut yöntemlerin belirli bir birleşimi olabilir.

Henüz Cevahir'de bulunmayan bir mekanizma olabilir.

Ya da araştırma sonucunda problemin başlangıçta yanlış ifade edildiği ortaya çıkabilir.

Doğru olan neyse onu bul.

---

# MEVCUT PARADİGMALARA BAĞLI DEĞİLSİN

Çözümün:

Transformer,

gradient descent,

backpropagation,

embedding,

token,

attention,

neural network,

memory,

RAG,

continual learning,

online learning,

meta-learning,

reinforcement learning,

world model,

predictive coding,

Hebbian learning,

fast weights,

slow weights,

routing,

Mixture of Experts,

state-space model

veya burada adı geçen herhangi başka bir paradigma olması gerekmiyor.

Bunları reddetmek de gerekmiyor.

İşe yarıyorlarsa kullan.

Yetersizlerse değiştir.

Bir kısmı gerekiyorsa kullan.

Hiçbiri gerekmiyorsa kullanma.

Bu araştırmanın amacı mevcut kavramlardan yeni bir kombinasyon oluşturmak değildir.

Amaç problemi çözmektir.

---

# BİYOLOJİYE BAĞLI DEĞİLSİN

İnsan ve hayvan sinir sistemleri milyonlarca yıllık çalışan örneklerdir ve araştırma açısından değerlidir.

Ancak Cevahir'in:

beyin,

hipokampus,

korteks,

sinaps,

uyku,

nöron

gibi biyolojik kavramların yazılımsal kopyası olması gerekmiyor.

Biyolojide işe yarayan bir prensip varsa araştır.

Yapay sistem için daha iyi bir çözüm varsa onu kullan.

Biyoloji yalnızca kanıtlanmış bir doğal sistem örneğidir.

Tasarımsal sınır değildir.

---

# MEVCUT CEVAHIR MİMARİSİNE BAĞLI DEĞİLSİN

Repository'yi inceleyebilirsin.

Mevcut sistemi anlayabilirsin.

Mevcut deneyleri ve geçmiş araştırmaları okuyabilirsin.

Ancak Cevahir'in bugünkü mimarisi çözüm uzayını belirlemez.

Mevcut:

model,

training pipeline,

Transformer,

embedding sistemi,

Cognitive Manager,

memory,

routing,

experience mekanizmaları,

testler,

dokümantasyon

korunması gereken bilimsel varsayımlar değildir.

Bunlar bugüne kadar oluşmuş engineering state'tir.

Araştırmanın sonucu mevcut architecture ile uyumlu çıkabilir.

Kısmen uyumlu çıkabilir.

Tamamen farklı bir architecture gerektirebilir.

Buna araştırma karar versin.

---

# KULLANICININ ÖNCEKİ FİKİRLERİNE DE BAĞLI DEĞİLSİN

Daha önce Cevahir hakkında konuşulan:

Experience Threads,

Experience Stacks,

pathways,

plasticity,

consolidation,

forgetting,

fast/intermediate/slow state,

representation change,

dynamic routing,

offline learning,

sleep-like processing

ve diğer fikirler çözüm değildir.

Bunlar araştırma geçmişinin parçalarıdır.

Doğru çıkabilirler.

Yanlış çıkabilirler.

Gereksiz çıkabilirler.

Daha temel başka bir teori tarafından kapsanabilirler.

Onları doğrulamaya çalışma.

Problemi çöz.

---

# EMBEDDING UZAYINI BAŞLANGIÇ SINIRI OLARAK KABUL ETME

Mevcut language-model training yaklaşımında büyük miktarda deneyim eğitim sırasında mevcut parametrik yapıya işlenir ve sistem daha sonra bu öğrenilmiş yapıyla çalışır.

Cevahir için bunun nihai form olduğunu varsaymıyoruz.

Bir sistemin yeni bir şey öğrenmesi yalnız mevcut embedding veya parameter uzayının yeniden düzenlenmesi olmak zorunda mı?

Bilmiyoruz.

Araştır.

Mevcut uzay yeterliyse bunu göster.

Yetersizse neden yetersiz olduğunu göster.

Başka bir matematiksel yapı gerekiyorsa onu türet.

Bu konuda önceden istenen bir sonuç yoktur.

---

# YAŞAM BOYU DEĞİŞİM

Araştırılan Cevahir aynı başlangıç noktasından çıktıktan sonra yaşadığı deneyimler nedeniyle zaman içerisinde farklılaşabilmelidir.

Aynı başlangıç durumundaki iki Cevahir instance'ı farklı yaşam geçmişleri yaşadığında, daha sonra aynı koşullara konulduklarında geçmişleri nedeniyle farklı yetenek veya davranış gösterebiliyorlarsa bu araştırma açısından önemli bir sonuçtur.

Ancak yalnız farklı dosyalara sahip olmaları yeterli değildir.

Yalnız farklı conversation history taşımaları yeterli değildir.

Yalnız farklı retrieval sonuçları almaları yeterli değildir.

Aradığımız şey gerçek öğrenmedir.

Bunun tam olarak ne anlama geldiğini sen tanımla ve doğrula.

---

# DENEYİMİN BİRLEŞMESİ

Sistem farklı zamanlarda öğrendiği şeyleri birbirleriyle ilişkilendirebilmelidir.

Bir deneyim diğerinden bağımsız bir kayıt olarak sonsuza kadar kalmak zorunda değildir.

Daha önce yaşanan bir olay daha sonraki bir deneyimin anlamını değiştirebilir.

Daha sonraki deneyim eski bir öğrenmeyi değiştirebilir.

Birden fazla deneyimin birleşiminden tek başlarına açık olmayan bir yetenek veya genelleme ortaya çıkabilir.

Bunun mümkün olup olmadığını ve nasıl gerçekleşebileceğini araştır.

---

# İÇSEL YENİDEN DÜZENLENME

Sistem sürekli dışarıdan yeni veri almak zorunda değildir.

Bazı dönemlerde daha önce yaşadıklarının üzerinde kendi içinde çalışabilmesi ihtimalini araştır.

Bu sürecin gerçekten öğrenme sağlayıp sağlayamayacağını belirle.

Sağlıyorsa mekanizmasını bul.

Sağlamıyorsa bunu göster.

İnsan uykusuna benzediği için doğru olduğunu varsayma.

Ama yalnız mevcut machine-learning paradigmasında alışılmış olmadığı için de reddetme.

Araştır.

---

# KALICILIK

Yeni experience sistemin davranışını değiştirebilir.

Ancak her değişiklik kalıcı olmak zorunda değildir.

Hangi değişikliğin kalıcı olması gerektiğini sistem nasıl belirleyebilir?

Bunun ayrı bir mekanizma gerektirip gerektirmediğini araştır.

Kalıcı öğrenmenin ne olduğunu formal olarak tanımla.

---

# UNUTMA VE DÜZELTME

Yaşayan bir sistem yalnız öğrenmek zorunda değildir.

Yanlış öğrenebilir.

Dünya değişebilir.

Önceki deneyim eksik olabilir.

Yeni kanıt eski bilgiyi geçersiz kılabilir.

Sistem bunlarla başa çıkabilmelidir.

Bunun matematiksel ve computational karşılığını araştır.

---

# KENDİNİ BOZMADAN ÖĞRENME

Sürekli değişen bir sistem kendi yeteneklerini de yok edebilir.

Dolayısıyla şu problem araştırmanın merkezindedir:

# YENİ BİR ŞEY ÖĞRENİRKEN ESKİ FAYDALI YETENEKLER NASIL KORUNABİLİR?

Bu problemi hangi mekanizmanın çözmesi gerektiğini önceden belirleme.

Çözümü araştır.

---

# ÖĞRENMEYİ ÖĞRENME

Bir Cevahir instance'ı yalnız dünya hakkında öğrenmekle kalmayıp zaman içerisinde daha iyi öğrenen bir sisteme dönüşebilir mi?

Başlangıçta bir şeyi öğrenmek için yüz experience gerekirken benzer yeni bir yapıyı daha sonra on experience ile öğrenebilir mi?

Bunun gerçekten “öğrenmeyi öğrenme” olup olmadığını ve nasıl ölçülmesi gerektiğini araştır.

---

# YENİ YETENEK

Araştırmanın en önemli ayrımlarından biri:

MEMORIZATION

ile

CAPABILITY ACQUISITION

arasındadır.

Bir fact'i saklamak yeterli değildir.

Bir sistem yaşadığı deneyim nedeniyle daha önce yapamadığı bir şeyi yapabiliyorsa bu daha güçlü bir sonuçtur.

Cevahir'in yaşarken yeni capability kazanmasının mümkün olup olmadığını araştır.

---

# ÇALIŞIRKEN ÖĞRENME

Yeni capability kazanmak için her seferinde:

modeli durdurmak,

dataset hazırlamak,

ayrı training başlatmak,

yeni checkpoint üretmek,

deployment yapmak

zorunlu olmamalıdır.

Araştırmanın hedefindeki sistemde çalışma ile öğrenme aynı yaşam sürecinin parçalarıdır.

Bunun mümkün olup olmadığını ve doğru mathematical/computational mechanism'i araştır.

---

# GELECEK MULTIMODAL CEVAHIR İÇİN GENELLİK

Bu tur doğrudan multimodal architecture tasarlamıyor.

Ancak bulunan temel öğrenme prensibinin gelecekte yalnız text'e özgü olmaması tercih edilir.

Cevahir ileride:

görsel,

ses,

continuous video,

mikrofon,

3D/spatial information,

tools,

actions,

physical/digital environments

ile çalışabilir.

Dolayısıyla bulunan learning principle'ın yalnız token sequence'lerine özgü olup olmadığını sorgula.

Ama bu nedenle şimdi multimodal architecture implement etme.

Önce öğrenme problemini çöz.

---

# AÇIK ARAŞTIRMA YETKİSİ

Bu problem üzerinde gerekli gördüğün araştırmayı yap.

Gerekli hipotezleri kendin üret.

Gerekli matematiksel formalizasyonları kendin oluştur.

Gerekli literatürü kendin belirle.

Gerekli karşılaştırmaları kendin yap.

Gerekli deneyleri kendin tasarla.

Gerekli kontrolleri kendin oluştur.

Gerekli counterexample'ları kendin ara.

Gerekli simülasyonları kendin kur.

Hipotez yanlışsa kendin ele.

Yeni hipotez gerekiyorsa kendin üret.

Problemin yanlış formüle edildiğini fark edersen yeniden formüle et.

Bir mekanizma mevcut literatürde zaten varsa bunu tespit et.

Bir fikir yalnız başka bir yöntemin yeniden adlandırılmasıysa bunu reddet.

Bir sonuç kanıtlanmamışsa kanıtlanmış gibi davranma.

Bir deney yetersizse daha iyi deney tasarla.

Bir yaklaşım başarısız olduğunda araştırmayı bitirme.

# PROBLEMİ ÇÖZENE KADAR ARAŞTIRMAYA DEVAM ET.

Buradaki “çözmek”, istenen sonuca ulaşmak anlamına gelmez.

Araştırmanın sonunda başlangıç varsayımımızın yanlış olduğu ortaya çıkabilir.

Bu da geçerli bir sonuçtur.

Ama sonucu varsayma.

Bul.

---

# ARAŞTIRMA ÖZGÜRLÜĞÜ

Sana hangi hipotezi kuracağını söylemiyorum.

Hangi formülü kullanacağını söylemiyorum.

Hangi mimariyi kullanacağını söylemiyorum.

Kaç state olması gerektiğini söylemiyorum.

Learning'in weights, memory, structure veya başka bir şey üzerinden gerçekleşmesi gerektiğini söylemiyorum.

Uyku benzeri bir sürecin gerekli olduğunu söylemiyorum.

Gradient descent'in yanlış olduğunu söylemiyorum.

Transformer'ın yanlış olduğunu söylemiyorum.

Yeni bir mimarinin zorunlu olduğunu söylemiyorum.

Bunların hiçbirini varsayma.

Araştırmanın kendisi karar versin.

---

# BAŞARI KRİTERİ

Ortaya çıkan sonuç yalnız fikir düzeyinde kalmamalıdır.

Araştırmanın sonunda savunulan temel mekanizma:

açık biçimde tanımlanabilir,

matematiksel veya computational olarak ifade edilebilir,

gerçek bir sistemde uygulanabilir,

alternatif açıklamalardan ayrıştırılabilir,

deneysel olarak sınanabilir,

yanlışlanabilir,

ve yaşarken öğrenme açısından gözlenebilir bir sonuç üretmelidir.

Bunun nasıl gösterileceğine sen karar ver.

---

# KAYNAK SINIRI

Yerel makinenin CPU ve RAM kapasitesi sınırlıdır.

Büyük model training'leri, büyük dataset işlemleri ve ağır benchmark'lar bu aşamada uygun değildir.

Bu fiziksel sınır araştırma düşüncesini sınırlamasın.

Gerekli büyük deneyleri tasarlayabilir ve daha sonraya bırakabilirsin.

Ucuz biçimde test edilebilecek şeyleri gerektiğinde test edebilirsin.

Hangi deneylerin yapılmasının anlamlı olduğuna sen karar ver.

Doğrulanmamış sonucu doğrulanmış olarak raporlama.

---

# REPOSITORY

Mevcut Cevahir repository'sini gerektiğinde inceleyebilirsin.

Araştırma için bağımsız belgeler, küçük deneyler veya doğrulama araçları oluşturabilirsin.

Ancak yalnız bir araştırma fikri oluştu diye ana Cevahir sistemini değiştirmek zorunda değilsin.

Araştırmanın sonucu mevcut architecture üzerinde değişiklik gerektiriyorsa bunun ne olduğunu araştırma sonucunda belirle.

GitHub'a push yapma.

---

# ARAŞTIRMA KAYDI

Araştırmanın düşünsel sonucunu kaybetme.

Nelerin denendiğini,

nelerin yanlış çıktığını,

hangi varsayımların elendiğini,

hangi sonuçların yalnız hipotez olduğunu,

hangi sonuçların mantıksal olarak türetildiğini,

hangi sonuçların deneysel destek aldığını,

hangi temel soruların hâlâ açık olduğunu

araştırma kaydında açıkça koru.

Negatif sonuçları silme.

Çünkü yanlış çıkan hipotezler de araştırmanın ürettiği bilgidir.

---

# ANA SORU

Bütün araştırmanın merkezinde tek bir problem vardır:

# BİR YAPAY SİSTEM, ÇALIŞTIĞI SÜRE BOYUNCA YAŞADIĞI DENEYİMLER NEDENİYLE KENDİ GELECEK HESAPLAMA VE DAVRANIŞ KAPASİTESİNİ NASIL KALICI, KONTROLLÜ VE GENELLENEBİLİR BİÇİMDE DEĞİŞTİREBİLİR?

Bunun cevabını bilmiyoruz.

Ben sana cevabı vermiyorum.

Ben sana çözüm mimarisini vermiyorum.

Ben sana öğrenme denklemini vermiyorum.

Ben sana araştırma yolunu vermiyorum.

# BUNLARI SEN BUL.

Gerekirse mevcut öğrenme teorilerini kullan.

Gerekirse reddet.

Gerekirse birleştir.

Gerekirse yeni bir formalizasyon geliştir.

Gerekirse başlangıç sorusunu düzelt.

Ama yalnız mevcut Cevahir architecture'ını iyileştirmiş olmakla yetinme.

Yalnız yeni module yazmış olmakla yetinme.

Yalnız testlerin geçmesiyle yetinme.

Yalnız güzel bir denklem bulmakla yetinme.

Yalnız literatürde benzerini bulamamakla novelty iddia etme.

Araştırmanın amacı bir çıktı üretmek değil:

# PROBLEM HAKKINDA DOĞRU SONUCA ULAŞMAKTIR.

Cevahir'in gelecekteki architecture'ı bu araştırmanın sonucu olsun.

Araştırma mevcut architecture'ın sonucu olmasın.

# YAŞARKEN ÖĞRENMENİN NE OLDUĞUNU BUL.
