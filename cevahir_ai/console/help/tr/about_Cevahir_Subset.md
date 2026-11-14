KONU
    about_Cevahir_Subset

KISA AÇIKLAMA
    Konsol dili PowerShell'in katı bir alt kümesidir. Kabul ettiği her şey PowerShell'deki anlamını
    taşır; kabul etmediği her şey ayrıştırma sırasında reddedilir (hata kimliği NotInCevahirSubset,
    çıkış kodu 2). Hiçbir şey yaklaşık olarak yorumlanmaz.

NELER VAR
    Komutlar        Fiil-Ad cmdlet'leri; -Parametre değer, -Anahtar, -Ad:değer, konumsal argümanlar,
                    listeler (a, b, c). Adlarda ve parametrelerde büyük-küçük harf ayrımı yoktur;
                    parametre tek anlamlı bir öneke kısaltılabilir. Get-CvArtifact ile Get-Artifact
                    aynıdır.
    Boru hattı      komut | komut | komut. Metin değil nesne akar. | ile biten satır devam eder.
    Değişkenler     $x = <boru hattı>, $x += <değer>, $x.Özellik, $x[0], "metin $x ve $($x.Key)"
                    Otomatik: $_ $PSItem $true $false $null $Error $LASTEXITCODE $? $CevahirHome
                    $PolicyKey $PWD $PSScriptRoot; $ErrorActionPreference atanabilir.
    Deyimler        if / elseif / else, foreach ($x in ...) { }, throw, exit <kod>,
                    betik dosyasının başında param( ).
    İfadeler        -eq -ne -gt -ge -lt -le -like -notlike -match -notmatch -contains -notcontains
                    -in -notin (büyük-küçük harfe duyarlı biçimleri -ceq, -clike ...), -and -or -not !,
                    + - * / %, KB MB GB ekli sayılar, 'düz' ve "genişleyen" dizgeler,
                    @( ) dizileri, @{ Ad = değer } sözlükleri.
    Dizge yöntemleri .Substring .StartsWith .EndsWith .Contains .ToLower .ToUpper .Trim .Split
                    .Replace; ayrıca .Length ve .Count özellikleri.
    Betikler        cevahir -File runbook.ps1 -Name prod; betik içinden .\diger.ps1.
                    Dosyalar UTF-8'dir; CRLF de LF de çalışır. Betik ilk hatada durur.
    Yorumlar        # satır sonuna kadar, <# ... #>

BETİK BLOKLARI KOD DEĞİLDİR
    Betik bloğu { ... } yalnız Where-Object, ForEach-Object ve Sort-Object süzgeci olarak ve
    hesaplanan özelliğin Expression değeri olarak kabul edilir. İçinde $_ üzerinde tam olarak bir
    ifade bulunur: karşılaştırmalar, -and/-or/-not, aritmetik, özellik erişimi, sabitler ve
    yukarıdaki dizge yöntemleri. Blok içinde komut, atama, deyim ve boru hattı yoktur; ( ) ve $( )
    yalnız ifade tutar.

        Get-Checkpoint | Where-Object { $_.Epoch -ge 3 -and $_.State -ne 'quarantined' }
        Get-Artifact   | Select-Object Short, @{Name='Files'; Expression={ $_.Locations.Count }}

    Her nesne için bir cmdlet çalıştırmak isteniyorsa nesneler o cmdlet'e boruyla verilir:

        Get-Checkpoint | Test-Checkpoint            (şöyle değil: ForEach-Object { Test-Checkpoint $_ })

    Hiçbir dizge kod olarak çalıştırılmaz. Invoke-Expression yoktur.

BİLEREK REDDEDİLENLER
    function, filter, class, enum, switch, while, do, for, break, continue, return,
    try / catch / finally, trap; tür dönüşümleri ve tür sabitleri ([int], [System.IO.File]);
    statik üyeler (::); & ve . çağırma işleçleri; yönlendirme (> >> 2>) ve arka plan işleri;
    here-string; splatting; $env: ve diğer kapsamlı değişkenler; -f, -replace, -split, -join,
    -is, -as ve bit işleçleri; adı alt çizgiyle başlayan üyeler; yukarıdaki dizge yöntemleri
    dışındaki yöntemler.

KONSOL BAŞKA PROGRAM BAŞLATMAZ
    Konsolun cmdlet'i olmayan bir ad (python, git, rm) CommandNotFoundException hatasıdır.
    Çıktıyı dosyada saklamak için Out-File kullanılır.

ÇIKIŞ KODLARI
    0 başarılı; 1 hata (bulunamadı dahil); 2 ayrıştırma ya da kullanım hatası; 3 politika reddi;
    4 onay gerekli (ayrılmış); 5 doğrulama başarısız (Test-Checkpoint, Test-AuditLog);
    130 kesildi. Betikte exit <n> kodu belirler.

AYRICA BAKINIZ
    Get-Help about_Cevahir_Registry, Get-Command, Get-Help <komut> -Full
