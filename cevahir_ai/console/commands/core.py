"""Pipeline utilities: Where-Object, ForEach-Object, Select-Object, Sort-Object, Group-Object,
Measure-Object, Write-Output, ConvertTo-Json, ConvertFrom-Json.

Filters take either a property comparison (``Where-Object Epoch -lt 3``) or a script block, which is
one expression of the restricted language (``cevahir_ai.console.expr``), never code.
"""
import fnmatch
import json

from cevahir_ai.console import syntax as s
from cevahir_ai.console.errors import ConsoleError, ParameterBindingError
from cevahir_ai.console.expr import compare, member, to_number, to_text, truthy, unroll
from cevahir_ai.console.registry import Param, cmdlet
from cevahir_ai.domain.records import Record, plain

_OPERATORS = ("EQ", "NE", "GT", "GE", "LT", "LE", "Like", "NotLike", "Match", "NotMatch", "Contains", "NotContains",
              "In", "NotIn")
_ALL_OPERATORS = _OPERATORS + tuple("C" + name for name in _OPERATORS)


def value_of(context, item, selector):
    """A property (by name) or a calculated value (script block) of one object."""
    if isinstance(selector, s.ScriptBlock):
        return context.apply(selector, item)
    if isinstance(selector, dict):
        expression = next((v for k, v in selector.items() if k.lower() in ("expression", "e")), None)
        return value_of(context, item, expression)
    return member(item, to_text(selector))


@cmdlet("Where-Object", kind="stream", aliases=("where", "?"), params=[
    Param("Property", "any", position=0, help=("A property name, or a script block { ... }.",
                                               "Özellik adı ya da bir betik bloğu { ... }.")),
    Param("Value", "any", position=1, help=("The value to compare the property with.",
                                            "Özelliğin karşılaştırılacağı değer.")),
    Param("FilterScript", "scriptblock", help=("The filter as a script block: one expression over $_.",
                                               "Betik bloğu olarak süzgeç: $_ üzerinde tek ifade.")),
    *[Param(name, "switch", help=(f"Compare with -{name.lower()}.", f"-{name.lower()} ile karşılaştır."))
      for name in _ALL_OPERATORS],
], synopsis=("Keeps the objects for which a condition holds.", "Koşulu sağlayan nesneleri geçirir."),
    examples=[("Get-Checkpoint | Where-Object Epoch -ge 3", "Checkpoints of epoch 3 and later.",
               "3. ve sonraki epoch'ların checkpoint'leri."),
              ("Get-Artifact | Where-Object { $_.Kind -eq 'ckpt' -and $_.State -ne 'verified' }",
               "A script block is one expression; it cannot run commands.",
               "Betik bloğu tek bir ifadedir; komut çalıştıramaz.")])
def where_object(context, p, items):
    block = p["FilterScript"] or (p["Property"] if isinstance(p["Property"], s.ScriptBlock) else None)
    chosen = [name for name in _ALL_OPERATORS if p[name]]
    if block is not None:
        if chosen or p["Value"] is not None:
            raise ParameterBindingError("Where-Object: a script block and a comparison operator cannot be combined")
        for item in items:
            if truthy(context.apply(block, item)):
                yield item
        return
    if p["Property"] is None:
        raise ParameterBindingError("Where-Object needs a property name or a script block { ... }")
    if len(chosen) > 1:
        raise ParameterBindingError("Where-Object: only one comparison operator can be given")
    name = to_text(p["Property"])
    for item in items:
        left = member(item, name)
        if truthy(compare(chosen[0].lower(), left, p["Value"]) if chosen else left):
            yield item


@cmdlet("ForEach-Object", kind="stream", aliases=("foreach", "%"), params=[
    Param("Process", "any", position=0, help=("A script block (one expression over $_) or a property name.",
                                              "Betik bloğu ($_ üzerinde tek ifade) ya da özellik adı.")),
    Param("MemberName", help=("The property to read from each object.", "Her nesneden okunacak özellik.")),
], synopsis=("Evaluates one expression, or reads one property, for each object.",
             "Her nesne için tek bir ifadeyi hesaplar ya da bir özelliği okur."),
    description=("The block is an expression, not code: it cannot run commands or assign variables. To run a "
                 "cmdlet per object, pipe into the cmdlet.",
                 "Blok bir ifadedir, kod değildir: komut çalıştıramaz, değişken atayamaz. Her nesne için bir "
                 "cmdlet çalıştırmak için nesneleri o cmdlet'e boruyla verin."),
    examples=[("Get-Checkpoint | ForEach-Object { $_.Key.Substring(0, 24) }", "A computed value per object.",
               "Her nesne için hesaplanan değer."),
              ("Get-Checkpoint | ForEach-Object Key", "The Key property of each object.", "Her nesnenin Key özelliği.")])
def foreach_object(context, p, items):
    selector = p["Process"] if p["Process"] is not None else p["MemberName"]
    if selector is None:
        raise ParameterBindingError("ForEach-Object needs a script block { ... } or a property name")
    for item in items:
        yield from unroll(value_of(context, item, selector))


def _select_names(item, patterns):
    names = item.names() if isinstance(item, Record) else list(item) if isinstance(item, dict) else []
    chosen = []
    for pattern in patterns:
        if any(mark in pattern for mark in "*?["):
            chosen += [name for name in names if fnmatch.fnmatchcase(name.lower(), pattern.lower()) and name not in chosen]
        else:
            chosen.append(next((name for name in names if name.lower() == pattern.lower()), pattern))
    return chosen


def _selected(context, item, properties, excluded):
    data = {}
    for selector in properties:
        if isinstance(selector, dict):
            label = next((to_text(v) for k, v in selector.items() if k.lower() in ("name", "label", "n", "l")), None)
            expression = next((v for k, v in selector.items() if k.lower() in ("expression", "e")), None)
            if expression is None:
                raise ParameterBindingError("a calculated property needs Expression = { ... }")
            data[label or (expression.text if isinstance(expression, s.ScriptBlock) else to_text(expression))] = \
                value_of(context, item, expression)
        elif isinstance(selector, s.ScriptBlock):
            data[selector.text] = context.apply(selector, item)
        else:
            for name in _select_names(item, [to_text(selector)]):
                data[name] = member(item, name)
    for name in list(data):
        if any(fnmatch.fnmatchcase(name.lower(), pattern.lower()) for pattern in excluded):
            del data[name]
    base = item.type_name if isinstance(item, Record) else "Object"
    return Record(base if base.startswith("Selected.") else f"Selected.{base}", data)


@cmdlet("Select-Object", kind="stream", aliases=("select",), params=[
    Param("Property", "objects", position=0, help=("Properties to keep: names, wildcards, or calculated "
                                                   "properties @{Name='..'; Expression={ .. }}.",
                                                   "Tutulacak özellikler: adlar, joker karakterler ya da "
                                                   "hesaplanan özellikler @{Name='..'; Expression={ .. }}.")),
    Param("ExcludeProperty", "list", help=("Properties to drop.", "Çıkarılacak özellikler.")),
    Param("ExpandProperty", help=("Output the value of this property instead of the object.",
                                  "Nesne yerine bu özelliğin değerini ver.")),
    Param("First", "int", help=("Only the first n objects (stops reading early).",
                                "Yalnız ilk n nesne (okumayı erken keser).")),
    Param("Last", "int", help=("Only the last n objects.", "Yalnız son n nesne.")),
    Param("Skip", "int", help=("Skip the first n objects.", "İlk n nesneyi atla.")),
    Param("Unique", "switch", help=("Drop repeated objects.", "Yinelenen nesneleri at.")),
], synopsis=("Selects objects or properties of objects.", "Nesneleri ya da nesnelerin özelliklerini seçer."),
    examples=[("Get-Checkpoint | Sort-Object Metric | Select-Object -First 1", "The checkpoint with the lowest metric.",
               "En düşük metrikli checkpoint."),
              ("Get-Artifact | Select-Object Short, Kind, @{Name='Files'; Expression={ $_.Locations.Count }}",
               "A calculated property.", "Hesaplanan özellik.")])
def select_object(context, p, items):
    for name in ("First", "Last", "Skip"):
        if p[name] is not None and p[name] < 0:
            raise ParameterBindingError(f"Select-Object: -{name} cannot be negative")

    def stream():
        skipped = taken = 0
        if p["First"] == 0:
            return
        for item in items:
            if p["Skip"] and skipped < p["Skip"]:
                skipped += 1
                continue
            yield item
            taken += 1
            if p["First"] is not None and taken >= p["First"]:
                return

    chosen = stream()
    if p["Last"] is not None:
        everything = list(chosen)
        chosen = [] if p["Last"] == 0 else everything[-p["Last"]:]
    seen = set()
    for item in chosen:
        if p["ExpandProperty"]:
            results = unroll(member(item, p["ExpandProperty"]))
        elif p["Property"] or p["ExcludeProperty"]:
            results = [_selected(context, item, p["Property"] or ["*"], p["ExcludeProperty"] or [])]
        else:
            results = [item]
        for result in results:
            if p["Unique"]:
                token = json.dumps(plain(result, typed=True), sort_keys=True, default=str)
                if token in seen:
                    continue
                seen.add(token)
            yield result


def sort_key(value):
    """Order as PowerShell compares: $null first, numbers by value, text without regard to case."""
    if value is None:
        return (0, 0, "")
    if isinstance(value, bool):
        return (1, int(value), "")
    if isinstance(value, (int, float)):
        return (1, value, "")
    if isinstance(value, (list, tuple)):
        return (2, 0, " ".join(to_text(item) for item in value).casefold())
    return (2, 0, to_text(value).casefold())


@cmdlet("Sort-Object", kind="stream", aliases=("sort",), params=[
    Param("Property", "objects", position=0, help=("Properties (or script blocks) to sort by.",
                                                   "Sıralama özellikleri (ya da betik blokları).")),
    Param("Descending", "switch", help=("Largest first.", "Büyükten küçüğe.")),
    Param("Unique", "switch", help=("Keep one object per sort key.", "Her sıralama anahtarı için tek nesne.")),
], synopsis=("Sorts objects by property values.", "Nesneleri özellik değerlerine göre sıralar."),
    examples=[("Get-Checkpoint | Sort-Object Epoch -Descending", "Newest epoch first.", "En yeni epoch başta.")])
def sort_object(context, p, items):
    selectors = p["Property"] or []

    def key(item):
        if not selectors:
            return (sort_key(item if not isinstance(item, (Record, dict)) else to_text(item)),)
        return tuple(sort_key(value_of(context, item, selector)) for selector in selectors)

    ordered = sorted(items, key=key, reverse=bool(p["Descending"]))
    previous = object()
    for item in ordered:
        if p["Unique"]:
            current = key(item)
            if current == previous:
                continue
            previous = current
        yield item


@cmdlet("Group-Object", kind="stream", aliases=("group",), params=[
    Param("Property", "objects", position=0, mandatory=True, help=("Properties to group by.", "Gruplama özellikleri.")),
], synopsis=("Groups objects that have the same property values.", "Aynı özellik değerine sahip nesneleri gruplar."),
    examples=[("Get-Artifact | Group-Object Kind", "How many artefacts of each kind.", "Her türden kaç eser var.")])
def group_object(context, p, items):
    groups = {}
    for item in items:
        values = [value_of(context, item, selector) for selector in p["Property"]]
        name = ", ".join(to_text(value) for value in values)
        groups.setdefault(name, []).append(item)
    for name in sorted(groups, key=str.casefold):
        yield Record("Cevahir.Group", Count=len(groups[name]), Name=name, Group=groups[name])


@cmdlet("Measure-Object", kind="stream", aliases=("measure",), params=[
    Param("Property", position=0, help=("The numeric property to measure.", "Ölçülecek sayısal özellik.")),
    Param("Sum", "switch", help=("Add the values.", "Değerleri topla.")),
    Param("Average", "switch", help=("Average of the values.", "Değerlerin ortalaması.")),
    Param("Minimum", "switch", help=("Smallest value.", "En küçük değer.")),
    Param("Maximum", "switch", help=("Largest value.", "En büyük değer.")),
], synopsis=("Counts objects and measures a numeric property.", "Nesneleri sayar, sayısal bir özelliği ölçer."),
    examples=[("Get-Checkpoint | Measure-Object Parameters -Sum -Maximum", "Count, sum and maximum.",
               "Sayı, toplam ve en büyük değer.")])
def measure_object(context, p, items):
    count, numbers = 0, []
    wanted = p["Sum"] or p["Average"] or p["Minimum"] or p["Maximum"]
    for item in items:
        value = member(item, p["Property"]) if p["Property"] else item
        if p["Property"] and value is None:
            continue
        count += 1
        if wanted:
            number = to_number(value) if not isinstance(value, (list, dict, Record)) else None
            if number is None:
                raise ConsoleError(f"Measure-Object: {to_text(value)[:40]!r} is not a number", error_id="NonNumericInput",
                                   category="InvalidData")
            numbers.append(number)
    yield Record("Cevahir.Measure", {
        "Count": count, "Sum": sum(numbers) if p["Sum"] and numbers else None,
        "Average": sum(numbers) / len(numbers) if p["Average"] and numbers else None,
        "Minimum": min(numbers) if p["Minimum"] and numbers else None,
        "Maximum": max(numbers) if p["Maximum"] and numbers else None, "Property": p["Property"]})


@cmdlet("Write-Output", aliases=("echo",), params=[
    Param("InputObject", "objects", position=0, remaining=True, pipeline="value",
          help=("The objects to write to the pipeline.", "Boru hattına yazılacak nesneler.")),
], synopsis=("Writes objects to the pipeline.", "Nesneleri boru hattına yazar."),
    examples=[("Write-Output 'bir', 'iki' | Measure-Object", "Two strings flow down the pipeline.",
               "İki dizge boru hattından geçer.")])
def write_output(context, p):
    yield from (p["InputObject"] or [])


def _limit(value, depth, state):
    """JSON data cut at ``depth`` levels; what is cut becomes text and is counted."""
    if isinstance(value, Record):
        value = dict(value.items())
    if isinstance(value, dict):
        if depth < 0:
            state["cut"] += 1
            return to_text(value)
        return {str(key): _limit(item, depth - 1, state) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if depth < 0:
            state["cut"] += 1
            return to_text(value)
        return [_limit(item, depth - 1, state) for item in value]
    if isinstance(value, s.ScriptBlock):
        return value.text
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if value == value and value not in (float("inf"), float("-inf")) else to_text(value)
    return to_text(value)


@cmdlet("ConvertTo-Json", kind="stream", params=[
    Param("Depth", "int", default=8, help=("How many levels of nested objects to keep (default 8).",
                                           "İç içe nesnelerin kaç düzeyi tutulur (varsayılan 8).")),
    Param("Compress", "switch", help=("No whitespace.", "Boşluksuz çıktı.")),
    Param("AsArray", "switch", help=("Always an array, also for one object.", "Tek nesne için de dizi.")),
], synopsis=("Converts objects to JSON text.", "Nesneleri JSON metnine çevirir."),
    description=("One object gives a JSON object, several give an array. Unlike PowerShell's default depth of 2, "
                 "deeper levels are kept up to -Depth and a warning says when something was cut.",
                 "Tek nesne bir JSON nesnesi, birden çok nesne bir dizi verir. PowerShell'in varsayılan 2 derinliğinden "
                 "farklı olarak -Depth düzeye kadar tutulur ve bir şey kesildiğinde uyarı verilir."),
    examples=[("Get-Checkpoint -Latest | ConvertTo-Json -Depth 4", "A checkpoint as JSON.", "JSON olarak bir checkpoint.")])
def convert_to_json(context, p, items):
    collected = list(items)
    state = {"cut": 0}
    data = [_limit(item, p["Depth"], state) for item in collected]
    document = data if p["AsArray"] or len(data) != 1 else data[0]
    if state["cut"]:
        context.warning(f"ConvertTo-Json: {state['cut']} nested values were cut at -Depth {p['Depth']}")
    if p["Compress"]:
        yield json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    else:
        yield json.dumps(document, ensure_ascii=False, indent=2)


def from_json(value):
    if isinstance(value, dict):
        type_name = value.get("__type") if isinstance(value.get("__type"), str) else "PSCustomObject"
        return Record(type_name, {key: from_json(item) for key, item in value.items() if key != "__type"})
    if isinstance(value, list):
        return [from_json(item) for item in value]
    return value


@cmdlet("ConvertFrom-Json", kind="stream", params=[],
        synopsis=("Converts JSON text to objects.", "JSON metnini nesnelere çevirir."),
        examples=[("'{\"a\": 1}' | ConvertFrom-Json | Select-Object -ExpandProperty a", "Reads a property from JSON.",
                   "JSON'dan bir özellik okur.")])
def convert_from_json(context, p, items):
    text = "\n".join(to_text(item) for item in items)
    if not text.strip():
        return
    try:
        document = json.loads(text)
    except ValueError as error:
        raise ConsoleError(f"ConvertFrom-Json: not valid JSON: {error}", error_id="InvalidJson", category="InvalidData",
                           exit_code=2) from None
    yield from unroll(from_json(document))
