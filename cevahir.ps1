# Cevahir Zihin: PowerShell'den sohbet.
#   .\cevahir.ps1                          bu bilgisayarın zihniyle konuş (ilk kez: boş bir zihin)
#   .\cevahir.ps1 --load cevahir_tam       indirilen bir zihni getir; sonra hep ev klasöründe yaşar
#   .\cevahir.ps1 --his                    her cevaptan sonra zihnin iç durumunu göster
# Zihin ev klasöründe yaşar ($env:CEVAHIR_HOME, yoksa ~\.cevahir\zihin) ve her sohbetin sonunda kendiliğinden
# kaydedilir. Betik çalışmazsa: powershell -ExecutionPolicy Bypass -File .\cevahir.ps1
$ErrorActionPreference = "Stop"
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path $PSScriptRoot
python -m zihin @args
