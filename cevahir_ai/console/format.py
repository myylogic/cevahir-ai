"""Formatting: tables, lists and the default view of each record type.

Output is plain text without box-drawing characters, so it reads the same in a Windows console
with a legacy code page, in a Colab cell and in a log file. A table follows PowerShell: a header,
a line of dashes as long as each label, numbers right-aligned, and when the terminal is too
narrow the columns are cut with ``…`` from the right, so the key column is the last to lose
digits. Views are data (``VIEWS``): label and property, or a function of the record.
"""
import ntpath

from cevahir_ai.console import syntax as s
from cevahir_ai.console.expr import to_text
from cevahir_ai.domain.records import Record

ELLIPSIS = "…"


class Formatted(str):
    """Text that a formatter produced: written as it is, never formatted again."""


def _name(path):
    return ntpath.basename(str(path)) if path else None


def _first_short(keys):
    from cevahir_ai.ops.artifacts import short
    return short(keys[0]) if keys else None


def _short(name):
    def get(record):
        from cevahir_ai.ops.artifacts import short
        return short(record.get(name))
    return get


def _count(name):
    return lambda record: len(record.get(name) or ())


VIEWS = {
    "Cevahir.Artifact": [("Key", "Short"), ("Kind", "Kind"), ("State", "State"), ("Provenance", "Provenance"),
                         ("Locations", _count("Locations")), ("Created", lambda r: str(r.get("Created"))[:19])],
    "Cevahir.Checkpoint": [("Key", "Short"), ("Epoch", "Epoch"), ("Metric", "Metric"), ("Envelope", "Envelope"),
                           ("Arch", "Arch"), ("Tok", "Tok"), ("State", "State"), ("File", lambda r: _name(r.get("Path")))],
    "Cevahir.CheckpointVerdict": [("Key", "Short"), ("Level", "Level"), ("Purpose", "Purpose"), ("Verdict", "Verdict"),
                                  ("C1", "C1"), ("C3", "C3"), ("C4", "C4"), ("C5", "C5"), ("C7", "C7"), ("C9", "C9"),
                                  ("C12", "C12")],
    "Cevahir.Check": [("Id", "Id"), ("Name", "Name"), ("Status", "Status"), ("Required", "Required"), ("Detail", "Detail")],
    "Cevahir.Model": [("Key", "Short"), ("Arch", "Arch"), ("Tok", "Tok"), ("Parameters", "Parameters"),
                      ("State", "State"), ("Checkpoints", _count("Checkpoints"))],
    "Cevahir.Tokenizer": [("Key", "Short"), ("Family", "Family"), ("VocabSize", "VocabSize"), ("Merges", "Merges"),
                          ("Tracked", "Tracked"), ("State", "State")],
    "Cevahir.Cache": [("Key", "Short"), ("Samples", "Samples"), ("MaxSeqLength", "MaxSeqLength"),
                      ("Sidecar", "Sidecar"), ("State", "State")],
    "Cevahir.LineageEdge": [("Level", "Level"), ("Direction", "Direction"), ("Source", _short("Source")),
                            ("Role", "Role"), ("Target", _short("Target")), ("How", "How")],
    "Cevahir.AuditEvent": [("Seq", "Seq"), ("Time", lambda r: str(r.get("Time"))[:19]), ("Event", "Event"),
                           ("Action", "Action"), ("Decision", "Decision"), ("Outcome", "Outcome"),
                           ("Target", lambda r: _first_short(r.get("Targets")))],
    "Cevahir.AuditVerdict": [("Host", "Host"), ("Own", "Own"), ("Ok", "Ok"), ("Events", "Events"),
                             ("Segments", "Segments"), ("Mac", "Mac"), ("FirstBadSeq", "FirstBadSeq"),
                             ("Findings", _count("Findings"))],
    "Cevahir.AuditFinding": [("Seq", "Seq"), ("Segment", "Segment"), ("Line", "Line"), ("Problem", "Problem")],
    "Cevahir.KeyInfo": [("Name", "Name"), ("Key", "Short"), ("Status", "Status"), ("Source", "Source"),
                        ("Meaning", "Meaning")],
    "Cevahir.KeyDiff": [("Member", "Member"), ("Equal", "Equal"), ("Reference", "Reference"),
                        ("Difference", "Difference"), ("Detail", "Detail")],
    "Cevahir.KeyScheme": [("Name", "Name"), ("Status", "Status"), ("Torch", "Torch"), ("Newline", "Newline"),
                          ("Summary", "Summary")],
    "Cevahir.CommandInfo": [("Name", "Name"), ("Aliases", "Aliases"), ("Class", "Class"), ("Synopsis", "Synopsis")],
    "Cevahir.Responsibility": [("Id", "Id"), ("Title", "Title"), ("Owner", "Owner"), ("Status", "Status")],
    "Cevahir.FileItem": [("Name", "Name"), ("Length", "Length"), ("LastWriteTime", "LastWriteTime")],
    "Cevahir.MemberInfo": [("Name", "Name"), ("MemberType", "MemberType"), ("Definition", "Definition")],
    "Cevahir.Variable": [("Name", "Name"), ("Value", "Value")],
    "Cevahir.Completion": [("Text", "Text"), ("Kind", "Kind"), ("Description", "Description")],
}
NAMED_VIEWS = {
    ("Cevahir.LineageEdge", "tree"): [("Lineage", "Tree"), ("How", "How"), ("Host", "Host"), ("Seq", "Seq")],
}
# How a record reads inside a cell of another record (a list of checks, of inputs, of findings).
INLINE = {
    "Cevahir.Check": lambda r: f"{r.get('Id')} {r.get('Status')}",
    "Cevahir.Input": lambda r: f"{r.get('Role')}={r.get('Key')}",
    "Cevahir.AuditFinding": lambda r: f"seq {r.get('Seq')}: {r.get('Problem')}",
}


def expand_view(objects, view):
    """Named views that change the rows: ``Checks`` shows one row per check of each verdict."""
    if view.lower() == "checks":
        rows = []
        for item in objects:
            for check in (item.get("Checks") or ()) if isinstance(item, Record) else ():
                rows.append(Record("Cevahir.Check", dict(check.items())))
        return rows, VIEWS["Cevahir.Check"]
    return None


def cell(value):
    """A value as one line of text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(cell(item) for item in value) + "}"
    if isinstance(value, Record):
        inline = INLINE.get(value.type_name)
        return inline(value) if inline else "@{" + "; ".join(f"{k}={cell(v)}" for k, v in value.items()) + "}"
    if isinstance(value, dict):
        return "@{" + "; ".join(f"{key}={cell(item)}" for key, item in value.items()) + "}"
    if isinstance(value, s.ScriptBlock):
        return value.text
    return to_text(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")


def _cut(text, width):
    return text if len(text) <= width else text[:max(width - 1, 0)] + ELLIPSIS


def columns_for(objects, properties=None, view=None):
    """``[(label, getter)]`` for a table or a list of ``objects``."""
    first = objects[0]
    if properties:
        columns = []
        for item in properties:
            if isinstance(item, dict):                     # calculated property: @{Name='..'; Expression={..}}
                label = next((to_text(v) for k, v in item.items() if k.lower() in ("name", "label", "n", "l")), None)
                expression = next((v for k, v in item.items() if k.lower() in ("expression", "e")), None)
                columns.append((label or (expression.text if isinstance(expression, s.ScriptBlock) else "?"), expression))
            else:
                columns.append((to_text(item), to_text(item)))
        return columns
    type_name = first.type_name if isinstance(first, Record) else None
    if view:
        named = NAMED_VIEWS.get((type_name, view.lower()))
        if named is None:
            raise ValueError(f"{type_name or 'this object'} has no view named {view!r}")
        return named
    if type_name in VIEWS:
        return VIEWS[type_name]
    if isinstance(first, Record):
        return [(name, name) for name in first.names()]
    if isinstance(first, dict):
        return [("Name", None), ("Value", None)]
    return [("Value", None)]


def _value(item, getter, evaluate):
    if callable(getter):
        return getter(item) if isinstance(item, Record) else None
    if isinstance(getter, s.ScriptBlock):
        return evaluate(getter, item)
    if getter is None:
        return item
    if isinstance(item, Record):
        return item.get(getter)
    if isinstance(item, dict):
        return next((value for key, value in item.items() if str(key).lower() == getter.lower()), None)
    return None


def _rows(objects, columns, evaluate):
    if objects and isinstance(objects[0], dict) and [label for label, _ in columns] == ["Name", "Value"]:
        return [[key, value] for item in objects for key, value in item.items()]
    return [[_value(item, getter, evaluate) for _, getter in columns] for item in objects]


def format_table(objects, properties=None, view=None, width=120, evaluate=None):
    """A table of ``objects`` as text (with a leading and a trailing blank line, as PowerShell prints)."""
    objects = list(objects)
    if not objects:
        return Formatted("")
    if view and not properties:
        expanded = expand_view(objects, view)
        if expanded is not None:
            objects, columns = expanded
            if not objects:
                return Formatted("")
            view = None
        else:
            columns = columns_for(objects, None, view)
    else:
        columns = columns_for(objects, properties, None)
    rows = _rows(objects, columns, evaluate)
    labels = [label for label, _ in columns]
    numeric = [all(isinstance(row[i], (int, float)) and not isinstance(row[i], bool) or row[i] is None for row in rows)
               and any(row[i] is not None for row in rows) for i in range(len(labels))]
    text = [[cell(value) for value in row] for row in rows]
    widths = [max(len(labels[i]), *(len(row[i]) for row in text)) for i in range(len(labels))]
    available = max(width, 20) - (len(labels) - 1)
    for i in reversed(range(len(widths))):                 # too wide: cut columns from the right, keys last
        excess = sum(widths) - available
        if excess <= 0:
            break
        widths[i] = max(widths[i] - excess, len(labels[i]), 4)

    def line(values):
        parts = []
        for i, value in enumerate(values):
            value = _cut(value, widths[i])
            parts.append(value.rjust(widths[i]) if numeric[i] else value.ljust(widths[i]))
        return " ".join(parts).rstrip()

    lines = ["", line(labels), line(["-" * len(label) for label in labels])]
    lines += [line(row) for row in text]
    return Formatted("\n".join(lines) + "\n")


def format_list(objects, properties=None, evaluate=None, width=120):
    """``Name : value`` lines, one block per object."""
    blocks = []
    for item in objects:
        if properties:
            pairs = [(label, _value(item, getter, evaluate)) for label, getter in columns_for([item], properties)]
        elif isinstance(item, Record):
            pairs = list(item.items())
        elif isinstance(item, dict):
            pairs = list(item.items())
        else:
            blocks.append(cell(item))
            continue
        pad = max((len(str(name)) for name, _ in pairs), default=0)
        lines = []
        for name, value in pairs:
            if isinstance(value, (list, tuple)) and value and all(isinstance(entry, Record) for entry in value) \
                    and len(cell(value)) + pad + 3 > width:
                lines.append(f"{str(name).ljust(pad)} : {cell(value[0])}")
                lines += [f"{' ' * pad}   {cell(entry)}" for entry in value[1:]]
            else:
                lines.append(f"{str(name).ljust(pad)} : {cell(value)}")
        blocks.append("\n".join(lines))
    if not blocks:
        return Formatted("")
    return Formatted("\n" + "\n\n".join(blocks) + "\n")


def render(objects, width=120, evaluate=None):
    """Default rendering of what a statement wrote to the pipeline."""
    out, group = [], []

    def flush():
        if not group:
            return
        first = group[0]
        if isinstance(first, Record):
            columns = VIEWS.get(first.type_name) or (len(first.names()) <= 4 and [(n, n) for n in first.names()])
            text = format_table(group, None, None, width, evaluate) if columns else format_list(group, None, evaluate, width)
        elif isinstance(first, dict):
            text = format_table(group, None, None, width, evaluate)
        else:
            text = "\n".join(cell(item) if isinstance(item, (list, tuple)) else to_text(item) for item in group) + "\n"
        out.append(text)
        group.clear()

    def shape(item):
        if isinstance(item, Record):
            return ("record", item.type_name, tuple(item.names()) if item.type_name not in VIEWS else ())
        return ("dict",) if isinstance(item, dict) else ("scalar",)

    for item in objects:
        if isinstance(item, Formatted):
            flush()
            out.append(item if item.endswith("\n") or not item else item + "\n")
            continue
        if group and shape(group[0]) != shape(item):
            flush()
        group.append(item)
    flush()
    return "".join(out)
