"""Session cmdlets: Get-Command, Get-Help, Get-Member, Get-Variable, Get-ChildItem, Test-Path,
Complete-Command, Get-Environment, Get-Responsibility."""
import fnmatch
import glob
import os
from datetime import datetime, timezone

from cevahir_ai.console import registry
from cevahir_ai.console.errors import ConsoleError
from cevahir_ai.console.format import Formatted
from cevahir_ai.console.registry import Param, cmdlet
from cevahir_ai.domain.records import Record


def command_record(command, culture):
    from cevahir_ai.console.help import pick, syntax
    verb, _, noun = command.name.partition("-")
    return Record("Cevahir.CommandInfo", {
        "Name": command.name, "Verb": verb, "Noun": noun, "Aliases": list(command.aliases),
        "Class": command.action_class, "Module": "Cevahir" if command.module else "Console",
        "CvName": f"{verb}-Cv{noun}" if command.module else None, "Synopsis": pick(command.synopsis, culture),
        "Syntax": syntax(command), "Parameters": [parameter.name for parameter in command.params],
        "Output": command.output, "Torch": command.torch, "WhatIf": command.whatif})


@cmdlet("Get-Command", aliases=("gcm",), output="Cevahir.CommandInfo", params=[
    Param("Name", position=0, help=("Command name, wildcards allowed (Get-*).", "Komut adı, joker karakter olabilir (Get-*).")),
    Param("Noun", help=("Only commands with this noun.", "Yalnız bu ada sahip komutlar.")),
    Param("Verb", help=("Only commands with this verb.", "Yalnız bu fiile sahip komutlar.")),
    Param("Module", choices=("Cevahir", "Console"), help=("Cevahir: registry cmdlets. Console: shell utilities.",
                                                          "Cevahir: kayıt cmdlet'leri. Console: kabuk araçları.")),
], synopsis=("Lists the commands of the console.", "Konsolun komutlarını listeler."),
    examples=[("Get-Command -Noun Checkpoint", "Commands about checkpoints.", "Checkpoint komutları."),
              ("Get-Command -Module Cevahir | Format-Table Name, Class, Synopsis", "Registry cmdlets with their action class.",
               "Eylem sınıflarıyla kayıt cmdlet'leri.")])
def get_command(context, p):
    for command in registry.commands():
        names = [command.name] + list(command.aliases)
        if p["Name"] and not any(fnmatch.fnmatchcase(name.lower(), p["Name"].lower()) for name in names):
            continue
        if p["Noun"] and not fnmatch.fnmatchcase(command.noun.lower(), p["Noun"].lower()):
            continue
        if p["Verb"] and not fnmatch.fnmatchcase(command.verb.lower(), p["Verb"].lower()):
            continue
        if p["Module"] and (p["Module"] == "Cevahir") != command.module:
            continue
        yield command_record(command, context.culture)


@cmdlet("Get-Help", aliases=("help",), params=[
    Param("Name", position=0, help=("A command, an alias or an about_ topic.", "Komut, takma ad ya da about_ konusu.")),
    Param("Full", "switch", help=("Parameters and examples too.", "Parametreler ve örneklerle birlikte.")),
    Param("Examples", "switch", help=("Only the examples.", "Yalnız örnekler.")),
    Param("Parameter", help=("Only this parameter.", "Yalnız bu parametre.")),
    Param("Culture", choices=("tr", "en"), help=("Language of the help text (default: CEVAHIR_UI_CULTURE or the OS).",
                                                 "Yardım metninin dili (varsayılan: CEVAHIR_UI_CULTURE ya da işletim sistemi).")),
], synopsis=("Shows help for a command or a topic, in Turkish or English.",
             "Bir komutun ya da konunun yardımını Türkçe ya da İngilizce gösterir."),
    examples=[("Get-Help Test-Checkpoint -Full", "Everything about one command.", "Bir komut hakkında her şey."),
              ("Get-Help about_Cevahir_Subset -Culture tr", "The language, in Turkish.", "Dil, Türkçe olarak.")])
def get_help(context, p):
    from cevahir_ai.console import help as helper
    culture = p["Culture"] or context.culture
    if not p["Name"]:
        yield Formatted(helper.overview(culture))
        return
    if p["Name"].lower().startswith("about_"):
        text = helper.topic_text(p["Name"], culture)
        if text is None:
            raise ConsoleError(f"there is no help topic {p['Name']}; topics: {', '.join(helper.topics())}",
                               error_id="HelpNotFound", category="ObjectNotFound")
        yield Formatted(text)
        return
    command = registry.find(p["Name"])
    if command is None:
        raise ConsoleError(f"there is no command {p['Name']} (see Get-Command)", error_id="HelpNotFound",
                           category="ObjectNotFound")
    try:
        yield Formatted(helper.help_text(command, culture, p["Full"], p["Examples"], p["Parameter"]))
    except ValueError as error:
        raise ConsoleError(str(error), error_id="HelpNotFound", category="ObjectNotFound") from None


def _type_name(value):
    if isinstance(value, Record):
        return value.type_name
    return {type(None): "null", bool: "bool", int: "int", float: "double", str: "string", list: "object[]",
            dict: "hashtable"}.get(type(value), type(value).__name__)


@cmdlet("Get-Member", kind="stream", aliases=("gm",), output="Cevahir.MemberInfo", params=[],
        synopsis=("Lists the properties of the objects in the pipeline.", "Boru hattındaki nesnelerin özelliklerini listeler."),
        examples=[("Get-Checkpoint | Get-Member", "What a checkpoint object carries.", "Bir checkpoint nesnesinin taşıdıkları.")])
def get_member(context, p, items):
    seen = set()
    for item in items:
        type_name = _type_name(item)
        if type_name in seen:
            continue
        seen.add(type_name)
        pairs = item.items() if isinstance(item, Record) else item.items() if isinstance(item, dict) else []
        for name, value in pairs:
            yield Record("Cevahir.MemberInfo", TypeName=type_name, Name=str(name), MemberType="Property",
                         Definition=_type_name(value))
        if isinstance(item, str):
            from cevahir_ai.console.parser import METHODS
            yield Record("Cevahir.MemberInfo", TypeName=type_name, Name="Length", MemberType="Property", Definition="int")
            for method in METHODS:
                yield Record("Cevahir.MemberInfo", TypeName=type_name, Name=method, MemberType="Method", Definition="string method")


@cmdlet("Get-Variable", output="Cevahir.Variable", params=[
    Param("Name", position=0, help=("Variable name, wildcards allowed.", "Değişken adı, joker karakter olabilir.")),
], synopsis=("Lists the variables of the session.", "Oturumun değişkenlerini listeler."),
    examples=[("$best = Get-Checkpoint -Best; Get-Variable best", "", "")])
def get_variable(context, p):
    for name in sorted(context.session.variables):
        if p["Name"] is None or fnmatch.fnmatchcase(name, p["Name"].lower().lstrip("$")):
            yield Record("Cevahir.Variable", Name=name, Value=context.session.variables[name])


def file_item(path):
    stat = os.stat(path)
    full = os.path.abspath(path)
    return Record("Cevahir.FileItem", {
        "Name": os.path.basename(full), "FullName": full, "Length": None if os.path.isdir(full) else stat.st_size,
        "LastWriteTime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Directory": os.path.dirname(full), "IsContainer": os.path.isdir(full),
        "Extension": os.path.splitext(full)[1]})


@cmdlet("Get-ChildItem", aliases=("ls", "dir"), output="Cevahir.FileItem", params=[
    Param("Path", "path", position=0, default=".", pipeline="both", aliases=("FullName",),
          help=("A folder, a file or a wildcard pattern (*.pth).", "Klasör, dosya ya da joker desen (*.pth).")),
    Param("Filter", help=("Only names matching this wildcard.", "Yalnız bu joker desene uyan adlar.")),
    Param("Recurse", "switch", help=("Also list sub-folders.", "Alt klasörleri de listele.")),
    Param("File", "switch", help=("Files only.", "Yalnız dosyalar.")),
], synopsis=("Lists files; each has FullName, so it pipes into Test-Checkpoint and Register-Artifact.",
             "Dosyaları listeler; her birinde FullName vardır, Test-Checkpoint ve Register-Artifact'e boruyla verilir."),
    examples=[("Get-ChildItem .\\saved_models -Filter *.pth -Recurse | Test-Checkpoint", "File-level check of every .pth.",
               "Her .pth dosyasının dosya düzeyi denetimi.")])
def get_child_item(context, p):
    path = p["Path"]
    if any(mark in path for mark in "*?["):
        found = sorted(glob.glob(path))
    elif os.path.isdir(path):
        if p["Recurse"]:
            found = sorted(os.path.join(base, name) for base, folders, files in os.walk(path) for name in folders + files)
        else:
            found = sorted(os.path.join(path, name) for name in os.listdir(path))
    elif os.path.exists(path):
        found = [path]
    else:
        raise FileNotFoundError(f"no such file or directory: {path}")
    for item in found:
        if p["Filter"] and not fnmatch.fnmatchcase(os.path.basename(item).lower(), p["Filter"].lower()):
            continue
        if p["File"] and os.path.isdir(item):
            continue
        yield file_item(item)


@cmdlet("Test-Path", params=[
    Param("Path", "path", position=0, mandatory=True, pipeline="both", aliases=("FullName",),
          help=("The path to test.", "Sınanacak yol.")),
], synopsis=("True when the path exists.", "Yol varsa True."),
    examples=[("if (Test-Path .\\saved_models) { Register-Artifact .\\saved_models -Recurse }", "", "")])
def test_path(context, p):
    yield os.path.exists(p["Path"])


@cmdlet("Complete-Command", output="Cevahir.Completion", params=[
    Param("Line", position=0, mandatory=True, help=("The partial command line.", "Yarım komut satırı.")),
    Param("Cursor", "int", help=("Cursor position (default: the end).", "İmleç konumu (varsayılan: son).")),
], synopsis=("Returns what Tab would offer for a partial line.", "Yarım bir satır için Tab'ın önereceklerini döndürür."),
    examples=[("Complete-Command 'Test-Checkpoint -Le'", "Completes to -Level.", "-Level olarak tamamlar.")])
def complete_command(context, p):
    from cevahir_ai.console.completion import complete
    for item in complete(p["Line"], p["Cursor"], context.session):
        yield Record("Cevahir.Completion", Text=item.text, Kind=item.kind, Description=item.description)


@cmdlet("Get-Environment", module=True, output="Cevahir.Environment", params=[],
        synopsis=("Shows the console's environment: version, home, host id, policy, Python, torch (not imported).",
                  "Konsolun ortamını gösterir: sürüm, ev dizini, makine kimliği, politika, Python, torch (içe aktarılmaz)."),
        examples=[("Get-Environment | Format-List", "", "")])
def get_environment(context, p):
    from cevahir_ai.ops import env
    yield env.environment(context.store)


@cmdlet("Get-Responsibility", module=True, output="Cevahir.Responsibility", params=[
    Param("Id", position=0, help=("One responsibility (R-CACHE ...).", "Tek bir sorumluluk (R-CACHE ...).")),
], synopsis=("Lists which roof module owns which responsibility.", "Hangi sorumluluğun hangi çatı modülüne ait olduğunu listeler."),
    examples=[("Get-Responsibility | Where-Object Status -eq planned", "What is not built yet.", "Henüz yapılmamış olanlar.")])
def get_responsibility(context, p):
    from cevahir_ai.ops import env
    yield from env.responsibilities(p["Id"])
