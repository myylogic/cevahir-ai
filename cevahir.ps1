# Cevahir: PowerShell'den sohbet.
#   .\cevahir.ps1                          Cevahir'in son biçimiyle (cevher) konuş: ilk kez depodaki açık zihin gelir,
#                                          sonra hep ev klasöründe yaşar ve her çıkışta kendiliğinden kaydedilir
#   .\cevahir.ps1 --yeni                   boş bir zihinle başla
#   .\cevahir.ps1 --load KLASOR            başka bir zihni getir
#   .\cevahir.ps1 --butce 256              bellek bütçesini megabayt olarak ver (yeni zihin için 512)
#   .\cevahir.ps1 zihin [--load ...]       önceki program (python -m zihin), olduğu gibi
# Son biçim yalnızca Python ve NumPy ister: C derleyicisi, ağ ve torch gerekmez.
# Zihin ev klasöründe yaşar ($env:CEVAHIR_HOME, yoksa ~\.cevahir\cevher).
# Betik çalışmazsa: powershell -ExecutionPolicy Bypass -File .\cevahir.ps1
$ErrorActionPreference = "Stop"
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path $PSScriptRoot
if ($args.Count -gt 0 -and $args[0] -eq "zihin") {
    $rest = @($args | Select-Object -Skip 1)
    python -m zihin @rest
} else {
    python -m cevher @args
}
