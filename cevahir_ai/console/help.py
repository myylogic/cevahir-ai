"""Help: generated from the command registry, in Turkish and English.

The SYNTAX section is never hand-written: it is derived from the parameter declarations, so it
cannot go stale. Conceptual topics (``about_*``) are Markdown files under ``help/<culture>/``; a
topic missing in Turkish falls back to English with a note.
"""
from pathlib import Path

from cevahir_ai.console import binder, registry

CULTURES = ("en", "tr")
LABELS = {
    "en": {"name": "NAME", "synopsis": "SYNOPSIS", "syntax": "SYNTAX", "description": "DESCRIPTION",
           "parameters": "PARAMETERS", "examples": "EXAMPLES", "aliases": "ALIASES", "output": "OUTPUT",
           "notes": "NOTES", "returns": "output", "class": "action class", "torch": "needs torch", "cv": "in real PowerShell",
           "position": "position", "pipeline": "pipeline input", "default": "default", "values": "values",
           "mandatory": "mandatory", "more": "For parameters and examples: Get-Help {name} -Full",
           "fallback": "(this topic has no Turkish text yet; English follows)",
           "common": "Common parameters: -Verbose, -ErrorAction <Continue|Stop|SilentlyContinue|Ignore>",
           "whatif": "-WhatIf shows what would happen and changes nothing."},
    "tr": {"name": "AD", "synopsis": "ÖZET", "syntax": "SÖZDİZİMİ", "description": "AÇIKLAMA",
           "parameters": "PARAMETRELER", "examples": "ÖRNEKLER", "aliases": "TAKMA ADLAR", "output": "ÇIKTI",
           "notes": "NOTLAR", "returns": "çıktı", "class": "eylem sınıfı", "torch": "torch gerektirir", "cv": "gerçek PowerShell'de",
           "position": "konum", "pipeline": "boru hattı girdisi", "default": "varsayılan", "values": "değerler",
           "mandatory": "zorunlu", "more": "Parametreler ve örnekler için: Get-Help {name} -Full",
           "fallback": "(bu konunun Türkçe metni henüz yok; İngilizcesi aşağıda)",
           "common": "Ortak parametreler: -Verbose, -ErrorAction <Continue|Stop|SilentlyContinue|Ignore>",
           "whatif": "-WhatIf ne olacağını gösterir, hiçbir şeyi değiştirmez."},
}
_TYPES = {"string": "string", "int": "int", "float": "double", "bool": "bool", "list": "string[]", "any": "object",
          "objects": "object[]", "scriptblock": "scriptblock", "path": "string", "hashtable": "hashtable"}


def pick(pair, culture):
    """The text of a bilingual ``(English, Türkçe)`` pair in ``culture`` (English when Turkish is empty)."""
    if isinstance(pair, str):
        return pair
    english, turkish = (tuple(pair) + ("", ""))[:2]
    return turkish if culture == "tr" and turkish else english


def type_text(parameter):
    if parameter.choices:
        return "|".join(parameter.choices) + ("[]" if parameter.type == "list" else "")
    return _TYPES[parameter.type]


def syntax(command):
    """One PowerShell-style syntax line, derived from the declaration."""
    parts = [command.name]
    ordered = sorted(command.params, key=lambda p: (p.position is None, p.position if p.position is not None else 0))
    for parameter in ordered + ([binder.WHATIF] if command.whatif else []):
        if parameter.type == "switch":
            parts.append(f"[-{parameter.name}]")
            continue
        value = f"<{type_text(parameter)}>"
        name = f"[-{parameter.name}]" if parameter.position is not None else f"-{parameter.name}"
        text = f"{name} {value}"
        parts.append(text if parameter.mandatory else f"[{text}]")
    parts.append("[<CommonParameters>]")
    return " ".join(parts)


def _wrap(text, indent="    ", width=96):
    import textwrap
    return "\n".join(textwrap.wrap(text, width, initial_indent=indent, subsequent_indent=indent)) if text else ""


def _wrap_syntax(line, width=96):
    """Break a syntax line between parameters, never inside one."""
    import re
    lines, current = [], "   "
    for part in re.findall(r"\[\[-\w+\] <[^>]*>\]|\[-\w+ <[^>]*>\]|\[-\w+\] <[^>]*>|-\w+ <[^>]*>|\[<\w+>\]|\[-\w+\]|\S+", line):
        if len(current) + 1 + len(part) > width and current.strip():
            lines.append(current)
            current = "       "
        current += " " + part
    return "\n".join(lines + [current])


def parameter_help(command, parameter, culture):
    labels = LABELS[culture]
    facts = []
    if parameter.mandatory:
        facts.append(labels["mandatory"])
    if parameter.position is not None:
        facts.append(f"{labels['position']} {parameter.position}")
    if parameter.pipeline:
        how = {"value": "ByValue", "property": "ByPropertyName", "both": "ByValue, ByPropertyName"}[parameter.pipeline]
        facts.append(f"{labels['pipeline']}: {how}")
    if parameter.aliases:
        facts.append("alias: " + ", ".join(f"-{alias}" for alias in parameter.aliases))
    if parameter.default not in (None, False):
        facts.append(f"{labels['default']}: {parameter.default}")
    lines = [f"    -{parameter.name} <{type_text(parameter)}>" if parameter.type != "switch" else f"    -{parameter.name}"]
    description = pick(parameter.help, culture)
    if description:
        lines.append(_wrap(description, "        "))
    if facts:
        lines.append("        (" + "; ".join(facts) + ")")
    return "\n".join(lines)


def help_text(command, culture="en", full=False, examples=False, parameter=None):
    culture = culture if culture in CULTURES else "en"
    labels = LABELS[culture]
    if parameter:
        wanted = [p for p in binder.parameters_of(command) if p.name.lower().startswith(parameter.lower().lstrip("-"))]
        if not wanted:
            raise ValueError(f"{command.name} has no parameter {parameter}")
        return "\n\n".join(parameter_help(command, p, culture) for p in wanted) + "\n"
    out = []
    if not examples:
        out += [labels["name"], f"    {command.name}", "", labels["synopsis"], _wrap(pick(command.synopsis, culture)), "",
                labels["syntax"], _wrap_syntax(syntax(command)), ""]
        description = pick(command.description, culture)
        if description:
            out += [labels["description"], _wrap(description), ""]
        notes = [f"{labels['class']}: {command.action_class}"]
        if command.torch:
            notes.append(f"{labels['torch']}: {command.torch}")
        if command.module:
            verb, _, noun = command.name.partition("-")
            notes.append(f"{labels['cv']}: {verb}-Cv{noun}")
        if command.output:
            notes.append(f"{labels['returns']}: {command.output}")
        if command.whatif:
            notes.append(labels["whatif"])
        out += [labels["notes"]] + [f"    {note}" for note in notes] + [""]
        if command.aliases:
            out += [labels["aliases"], "    " + ", ".join(command.aliases), ""]
    if full and not examples:
        out += [labels["parameters"]]
        for item in binder.parameters_of(command):
            out += [parameter_help(command, item, culture), ""]
    if (full or examples) and command.examples:
        out += [labels["examples"]]
        for code, english, turkish in command.examples:
            out += [f"    PS cevahir> {code}"]
            explanation = pick((english, turkish), culture)
            if explanation:
                out += [_wrap(explanation, "        ")]
            out += [""]
    if not full and not examples:
        out += [labels["more"].format(name=command.name), ""]
    return "\n".join(out).rstrip("\n") + "\n"


def topics():
    base = Path(__file__).parent / "help"
    return sorted({path.stem for path in base.glob("*/about_*.md")})


def topic_text(name, culture="en"):
    """The text of an ``about_`` topic, or None. Falls back to English with a note."""
    base = Path(__file__).parent / "help"
    wanted = next((topic for topic in topics() if topic.lower() == name.lower()), None)
    if wanted is None:
        return None
    path = base / culture / f"{wanted}.md"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return LABELS[culture]["fallback"] + "\n\n" + (base / "en" / f"{wanted}.md").read_text(encoding="utf-8")


def overview(culture="en"):
    culture = culture if culture in CULTURES else "en"
    lines = []
    if culture == "tr":
        lines += ["Cevahir konsolu: PowerShell'in katı bir alt kümesi. Komutlar Fiil-Ad biçimindedir ve aralarında",
                  "metin değil nesne akar.", "",
                  "  Get-Command                 bütün komutlar",
                  "  Get-Help <komut> -Full      bir komutun parametreleri ve örnekleri",
                  "  Get-Help about_Cevahir_Subset   dilin kapsamı ve bilerek reddedilenler",
                  "  <komut> -?                  kısa yardım", "", "Komutlar:"]
    else:
        lines += ["The Cevahir console: a strict subset of PowerShell. Commands are Verb-Noun, and objects, not text,",
                  "flow between them.", "",
                  "  Get-Command                 every command",
                  "  Get-Help <command> -Full    parameters and examples of a command",
                  "  Get-Help about_Cevahir_Subset   what the language covers and what it rejects on purpose",
                  "  <command> -?                short help", "", "Commands:"]
    for command in registry.commands():
        lines.append(f"  {command.name:<20} {pick(command.synopsis, culture)}")
    lines += ["", ("Konular: " if culture == "tr" else "Topics: ") + ", ".join(topics())]
    return "\n".join(lines) + "\n"
