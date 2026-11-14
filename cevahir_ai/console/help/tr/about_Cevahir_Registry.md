KONU
    about_Cevahir_Registry

KISA AÇIKLAMA
    CEVAHIR_HOME altındaki kayıt, denetim kaydı ve soy ağacı.

NEREDE
    CEVAHIR_HOME deponun dışındadır: Windows'ta %LOCALAPPDATA%\Cevahir, Linux ve macOS'ta
    ${XDG_STATE_HOME:-~/.local/state}/cevahir, Colab'da bağlıysa Drive. CEVAHIR_HOME ortam
    değişkeni ya da cevahir -Home <dizin> bunu değiştirir. Kullanılan ev dizinini
    Get-Environment gösterir.

ESERLER VE ANAHTARLAR
    Register-Artifact var olan dosyaları okur ve her eser için, eserin içerikten türetilen
    anahtarıyla (cev:<tür>:<şema>:<sha256>) adlandırılmış bir manifest yazar. Hiçbir şey taşınmaz,
    yeniden adlandırılmaz. Tablolar kısa biçimi gösterir (ckpt:3f9a1c0b7d2e); her komut kısa
    biçimi, tam anahtarı, takma adı (eski tokenizer özeti) ya da kayıtlı dosyanın yolunu kabul
    eder.

    Manifestte yol bulunmaz. Dosyanın bu makinedeki yeri ayrı tutulur; bu yüzden bayt bayt aynı
    iki kopya (last.pth, best.pth) iki konumlu tek eserdir ve başka makineden kopyalanan dosya
    yalnız baytlarından tanınır.

DURUMLAR
    registered    burada kaydedildi
    imported      paketle geldi; bu makinede henüz doğrulanmadı
    verified      son Test-Checkpoint özetlerini doğruladı
    quarantined   son sınama kayıtlı baytlardan farklı bayt ya da güvensiz zarf buldu

DENETİM KAYDI
    Eylem yapan her komut ve her doğrulama kararı bu makinenin zincirine eklenir
    (audit/<makine-kimliği>/). Her olay kendinden önceki olayın özetini taşır. Test-AuditLog
    zinciri yeniden hesaplar: düzenlenen, silinen, eklenen ya da yeri değiştirilen satır sıra
    numarasıyla bildirilir; sondan kesilen zincir baş kaydı üzerinden bildirilir.

    Sınırlar, açıkça: sizin kullanıcınızla çalışan bir süreç MAC anahtarını okuyup bütün zinciri
    tutarlı biçimde yeniden yazabilir. Bunu yakalamak için baş kaydının imzalı bir kopyasının
    makine dışında tutulması gerekir. Bu, veri olarak politikalar ve onaylar bir sonraki
    aşamadır; bugün yerleşik başlangıç politikası okuma ve kayıt eylemlerine izin verir ve
    bunları kaydeder.

SOY AĞACI
    Get-Lineage bir eseri hangi anahtarların ürettiğini (-Direction Up) ya da ondan ne
    üretildiğini (-Direction Down) gösterir. Denetim olaylarından okunur. Sonradan kaydedilen
    dosyalar için yalnız dosyanın kendi söylediği bilinir: checkpoint'in tokenizer'ı, önbellek
    adındaki korpus özeti. Koşular ve önbellekler, eğitim konsol üzerinden çalıştırıldığında
    soy ağacına girer (sonraki aşama).

MAKİNELER ARASI TAŞIMA
    Export-Artifact -To x.cvb [-IncludePayload] [-IncludeAudit], öbür makinede
    Import-Artifact x.cvb. İçe aktarma, bir şey yazmadan önce her şeyi doğrular. Paket imzalı
    değildir: içe aktarma paketin kendi içinde tutarlı olduğunu kanıtlar, kimin ürettiğini değil.

AYRICA BAKINIZ
    Get-Help Register-Artifact -Full, Get-Help Test-AuditLog -Full, Get-Help about_Cevahir_Subset
