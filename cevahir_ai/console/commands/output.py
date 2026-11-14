"""Formatting and stream cmdlets: Format-Table, Format-List, Out-String, Out-Null, Out-File,
Write-Host, Write-Warning, Write-Error, Write-Verbose."""
import os

from cevahir_ai.console.errors import ConsoleError
from cevahir_ai.console.expr import to_text
from cevahir_ai.console.format import format_list, format_table, render
from cevahir_ai.console.registry import Param, cmdlet

PROTECTED = ("research", "docs/research", "docs/publications")


@cmdlet("Format-Table", kind="stream", aliases=("ft",), params=[
    Param("Property", "objects", position=0, help=("Columns: property names or calculated properties.",
                                                   "Sütunlar: özellik adları ya da hesaplanan özellikler.")),
    Param("View", help=("A named view of the type (Tree for lineage, Checks for verdicts).",
                        "Türün adlı görünümü (soy ağacı için Tree, kararlar için Checks).")),
    Param("AutoSize", "switch", help=("Accepted for compatibility; columns are always sized to their content.",
                                      "Uyumluluk için kabul edilir; sütunlar her zaman içeriğe göre boyutlanır.")),
    Param("Width", "int", help=("Total width in characters (default: the terminal).",
                                "Karakter olarak toplam genişlik (varsayılan: uçbirim).")),
], synopsis=("Formats objects as a table.", "Nesneleri tablo olarak biçimlendirir."),
    examples=[("Get-AuditEvent -Last 5 | Format-Table Seq, Event, Action, Decision", "Chosen columns.", "Seçilen sütunlar."),
              ("Get-Lineage ckpt:3f9a1c0b -Direction Up | Format-Table -View Tree", "Lineage as an indented tree.",
               "Girintili ağaç olarak soy ağacı.")])
def format_table_command(context, p, items):
    try:
        text = format_table(list(items), p["Property"], p["View"], p["Width"] or context.session.width, context.apply)
    except ValueError as error:
        raise ConsoleError(f"Format-Table: {error}", error_id="UnknownView", category="InvalidArgument", exit_code=2) from None
    if text:
        yield text


@cmdlet("Format-List", kind="stream", aliases=("fl",), params=[
    Param("Property", "objects", position=0, help=("Properties to show (default: all).",
                                                   "Gösterilecek özellikler (varsayılan: hepsi).")),
], synopsis=("Formats objects as lists of properties.", "Nesneleri özellik listesi olarak biçimlendirir."),
    examples=[("Get-Checkpoint -Latest | Test-Checkpoint -Level Compat -Purpose evaluate | Format-List Verdict, Summary, Checks",
               "The verdict with its reasons.", "Gerekçeleriyle birlikte karar.")])
def format_list_command(context, p, items):
    text = format_list(list(items), p["Property"], context.apply, context.session.width)
    if text:
        yield text


@cmdlet("Out-String", kind="stream", params=[
    Param("Width", "int", help=("Total width in characters.", "Karakter olarak toplam genişlik.")),
], synopsis=("Renders objects to one string, as the console would print them.",
             "Nesneleri konsolun yazacağı biçimde tek bir dizgeye çevirir."),
    examples=[("$text = Get-Artifact | Out-String", "Keep the rendered table in a variable.",
               "Biçimlendirilmiş tabloyu değişkende tut.")])
def out_string(context, p, items):
    yield str(render(list(items), p["Width"] or context.session.width, context.apply))


@cmdlet("Out-Null", kind="stream", params=[], synopsis=("Discards its input.", "Girdisini atar."),
        examples=[("Register-Artifact .\\saved_models | Out-Null", "Run without printing the result.",
                   "Sonucu yazdırmadan çalıştır.")])
def out_null(context, p, items):
    for _ in items:
        pass
    return
    yield


def protected_path(path):
    """True when ``path`` is inside a byte-protected directory of the checkout (research records)."""
    from pathlib import Path
    try:
        from cevahir_ai._compat.repo import repo_root
        relative = Path(os.path.abspath(path)).resolve().relative_to(repo_root().resolve()).as_posix()
    except (ValueError, RuntimeError):
        return False
    return any(relative == name or relative.startswith(name + "/") for name in PROTECTED)


@cmdlet("Out-File", kind="stream", action_class="mutate", whatif=True, params=[
    Param("FilePath", "path", position=0, mandatory=True, aliases=("Path",),
          help=("The file to write (UTF-8, LF).", "Yazılacak dosya (UTF-8, LF).")),
    Param("Append", "switch", help=("Append instead of replacing.", "Üzerine yazmak yerine sona ekle.")),
    Param("Width", "int", help=("Total width in characters.", "Karakter olarak toplam genişlik.")),
], synopsis=("Writes the rendered output to a file.", "Biçimlendirilmiş çıktıyı bir dosyaya yazar."),
    description=("The console has no > redirection; Out-File is the way to keep output. It refuses paths under "
                 "research/, docs/research/ and docs/publications/ of the checkout (byte-protected records) and inside "
                 "CEVAHIR_HOME.",
                 "Konsolda > yönlendirmesi yoktur; çıktı Out-File ile saklanır. Çalışma kopyasındaki research/, "
                 "docs/research/ ve docs/publications/ altına (bayt korumalı kayıtlar) ve CEVAHIR_HOME içine yazmayı reddeder."),
    examples=[("Get-Artifact | ConvertTo-Json | Out-File artifacts.json", "Save artefacts as JSON.",
               "Eserleri JSON olarak kaydet.")])
def out_file(context, p, items):
    text = str(render(list(items), p["Width"] or context.session.width, context.apply))
    if protected_path(p["FilePath"]):
        raise ConsoleError(f"Out-File: {p['FilePath']} is under a byte-protected directory of the checkout",
                           error_id="ProtectedPath", category="PermissionDenied", exit_code=3)
    home = os.path.realpath(str(context.store.home))
    try:
        inside = os.path.commonpath([home, os.path.realpath(os.path.abspath(p["FilePath"]))]) == home
    except ValueError:                              # another drive on Windows: not inside the home
        inside = False
    if inside:
        raise ConsoleError(f"Out-File: {p['FilePath']} is inside CEVAHIR_HOME; the registry and the audit log are "
                           "written only by the operations that own them", error_id="ProtectedPath",
                           category="PermissionDenied", exit_code=3)
    if context.what_if("Out-File", p["FilePath"]):
        return
    with open(p["FilePath"], "a" if p["Append"] else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return
    yield


@cmdlet("Write-Host", params=[
    Param("Object", "objects", position=0, remaining=True, raw=True, help=("What to print.", "Yazdırılacak değerler.")),
], synopsis=("Prints text to the console (not to the pipeline).", "Metni konsola yazar (boru hattına değil)."),
    examples=[("Write-Host 'Kayıt tamam:' $count 'eser'", "Values are joined with spaces.",
               "Değerler boşlukla birleştirilir.")])
def write_host(context, p):
    context.host(" ".join(to_text(item) for item in (p["Object"] or [])))
    return
    yield


@cmdlet("Write-Warning", params=[
    Param("Message", position=0, mandatory=True, help=("The warning text.", "Uyarı metni.")),
], synopsis=("Writes a warning.", "Uyarı yazar."), examples=[("Write-Warning 'eski checkpoint'", "", "")])
def write_warning(context, p):
    context.warning(p["Message"])
    return
    yield


@cmdlet("Write-Verbose", params=[
    Param("Message", position=0, mandatory=True, help=("The message (shown with -Verbose).",
                                                       "İleti (-Verbose ile gösterilir).")),
], synopsis=("Writes a message that is shown only with -Verbose.", "Yalnız -Verbose ile gösterilen bir ileti yazar."),
    examples=[("Write-Verbose 'ayrıntı' -Verbose", "", "")])
def write_verbose(context, p):
    context.verbose_message(p["Message"])
    return
    yield


@cmdlet("Write-Error", params=[
    Param("Message", position=0, mandatory=True, help=("The error text.", "Hata metni.")),
], synopsis=("Writes a non-terminating error (exit code 1).", "Sonlandırmayan bir hata yazar (çıkış kodu 1)."),
    examples=[("Write-Error 'model bulunamadı'", "", "")])
def write_error(context, p):
    context.error(ConsoleError(p["Message"], error_id="WriteError", category="NotSpecified"))
    return
    yield
