# Araştırma yayınlarını sürdürmek

Bu süreç, Muhammed Yasin Yılmaz'ın mevcut ve bundan sonraki anlamlı araştırma
sonuçlarını GitHub'a taşıma ve adını taşıyan DOI arşiv sürümleriyle yayımlama
talebini uygular. Araştırma ve kitap aynı çalışma akışında sürer. Bu dosya bir
zamanlanmış görev oluşturmaz; yalnız gelecekteki araştırma turlarının yayın
sözleşmesini kaydeder.

## Yeni sonuçtan yayına

1. Önce eski araştırma kodunu, raporları, ölçümleri ve negatif sonuçları koru.
   Yeni iddia ve deneyi ayrı tarihli dizine koy. Eski bir yorum değiştiğinde
   gerekçesini yeni metinde açıkla; geçmişte ne bulunduğunu silme.
2. Ana living-learning sorusuna katkıyı ve kapsam dışını yaz. En güçlü uygun
   rakibi, bilgi erişimini, toplam maliyeti, olumsuz kontrolü ve test sızıntısı
   sınırını belirt. Literatürde bilinen fikri yeni ilke olarak sunma.
3. Deneyi çalıştır, kayıtlı sonuçla bağımsız hesap yolunu veya anlamlı karşı
   örneği karşılaştır. Sadece hash doğrulamasını tekrar üretim diye adlandırma.
   Yeni doğrulama tarihi ve kaynak sürümü ayrı kaydedilsin.
4. Kataloğa çalışma kimliği, gerçek yazar, tarih, özetler ve özgün rapor yolunu
   ekle. `python scripts/build_publications.py --write` kaynakların tam yayın
   edisyonunu oluşturur; bu işlem özgün araştırma kayıtlarını değiştirmez.
   Başlık ve yorumları kaynaklarla karşılaştırarak incele.
5. Kitabın ilgili araştırma bölümünü güncelle. Etkilenen bağlantı ve sembolleri
   koddan incele; ardından `python scripts/check_book.py --record-reviewed-sources`
   ile açık inceleme kaydını yenile. CI bu kaydı kendiliğinden yenilememelidir.
6. `python scripts/check_book.py`, `python scripts/build_publications.py` ve
   `python scripts/check_publications.py` çalışsın. Değişen deney için uygun
   tekrar üretim komutları ayrıca çalıştırılsın. Tüm repo/GPU testleri yapılmadıysa
   yapılmış gibi sunma. Yayın paketine gizli kimlik bilgileri veya kişisel yerel
   deney oturumları ekleme.
7. Katalog, CITATION.cff ve .zenodo.json içindeki sürüm/yazar bilgilerini eşleştir.
   Kullanıcının sağlamadığı ORCID, kurum, ortak yazar, hakem veya imza bilgisi
   üretme. .zenodo.json varsa GitHub arşivleme künyesinde önceliklidir.
8. İncelenmiş dosyaları commit edip GitHub'a gönder; CI sonucunu denetle. Her
   küçük ara deneme için değil, anlamlı tamamlanmış bir araştırma paketi için
   yeni etiket ve GitHub Release oluştur. Eski etiketi taşıma, force push yapma.
9. Zenodo'nun gerçek yayımlanmış kaydından sürüm DOI'si, bütün sürümler DOI'si,
   yazar, başlık, sürüm ve dosya eşleşmesini doğrula. Taslakta rezerve edilmiş
   DOI'yi yayımlanmış diye sunma. GitHub sürümü oluştuğu halde Zenodo başarısızsa
   yayın durumunu açık bırak ve gerçek hata üzerinden tamamla.
10. Gerçek DOI'yi atıf dosyasına, yayın kaydına ve README bağlantılarına ekleyip
    eşitle. Bu DOI, kendisini taşıyan sonradan eklenmiş tanıtım commit'inin değil,
    etiketlenmiş arşiv sürümünün kimliğidir. Son durum için temiz çalışma ağacı ve
    local main–origin/main eşitliği kontrol edilsin.

## Koleksiyon ve tekil makale ayrımı

İlk sürümde dokuz metin, kaynaklar ve kitap bir yazılım/araştırma arşivi DOI'si
altındadır. Her metnin CEV kimliği vardır; bu kimlik DOI değildir. Tekil metin
ileride ayrı ön baskı olarak arşivlenirse ayrı kayıt/DOI ve koleksiyonla ilişkisi
eklenir. Aynı DOI birbirinden farklı eserlerin ayrı DOI'si gibi kullanılmaz.

## Doğrulamanın anlamı

Yazar künyesi projenin sahibinin talebine dayanır. GitHub oturumunun yetkisiyle
yayın yapılması kriptografik commit imzası veya dış kimlik incelemesi değildir.
Sonlu tarama, kaynak özeti ve CI birer teknik kontrol sağlar; hakemlik ve
bilimsel yenilik incelemesi ayrıca gerekir. Yapay zekâ desteği açıklaması bütün
edisyonlarda bulunur.

Kaynaklar: [GitHub'da atıf ve kalıcı arşiv](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content),
[Zenodo bağlantısını açma](https://help.zenodo.org/docs/github/enable-repository/),
[Zenodo metadata önceliği](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
