"""The console language: parser, expression language, parameter binding and pipeline semantics.

The language is a strict subset of PowerShell. Two things are pinned here above all:

  - what it accepts behaves as PowerShell does (comparison coercion, case-insensitivity, array
    filtering, string expansion, pipeline unrolling);
  - what it does not accept is rejected at parse time and nothing runs. A script block is one
    expression over data: it cannot run a command, reach a Python attribute, call a method outside
    a fixed list of string methods, read the environment or write a file. The injection attempts
    below are the evidence.

Torch-free.
"""
import io
import json

import pytest

from cevahir_ai.console import binder, registry
from cevahir_ai.console import syntax as s
from cevahir_ai.console.errors import EvaluationError, ParameterBindingError, ParserError
from cevahir_ai.console.expr import Evaluator, compare, member, to_text, truthy
from cevahir_ai.console.parser import METHODS, is_incomplete, number_token, parse
from cevahir_ai.console.registry import Param, cmdlet
from cevahir_ai.console.runtime import Session
from cevahir_ai.domain.records import Record


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CEVAHIR_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.chdir(tmp_path)


def session(tmp_path=None, **options):
    return Session(home=str(tmp_path / "home") if tmp_path else None, stdout=io.StringIO(), stderr=io.StringIO(),
                   width=options.pop("width", 100), culture=options.pop("culture", "en"), environ={}, **options)


def run(text, **options):
    """``(exit code, stdout, stderr)`` of one script in a fresh session."""
    current = session(**options)
    code = current.run(text)
    return code, current.stdout.getvalue(), current.stderr.getvalue()


def values(text):
    """The objects a script leaves on the pipeline (records as plain dicts)."""
    current = session(output="json")
    code = current.run(text)
    assert code == 0, current.stderr.getvalue()
    return [item.to_dict() if isinstance(item, Record) else item for item in current.collected]


def value(text):
    result = values(text)
    assert len(result) == 1, result
    return result[0]


# ----------------------------------------------------------------------------------------- parser
def test_parser_builds_a_plain_tree():
    tree = s.dump(parse("Get-Checkpoint -Latest | Where-Object { $_.Epoch -ge 3 } | Format-Table Key, Epoch"))
    pipeline = tree["statements"][0]
    assert tree["params"] == [] and pipeline["node"] == "Pipeline" and len(pipeline["elements"]) == 3
    first, second, third = pipeline["elements"]
    assert first == {"node": "Command", "name": "Get-Checkpoint", "args": [{"node": "Argument", "name": "Latest", "value": None}]}
    block = second["args"][0]["value"]
    assert block["node"] == "ScriptBlock" and block["text"] == "{ $_.Epoch -ge 3 }"
    assert block["body"] == {"node": "Binary", "op": "ge", "left": {"node": "Member", "target": {"node": "Var", "name": "_"},
                                                                    "name": "Epoch"}, "right": {"node": "Lit", "value": 3}}
    assert third["args"][0]["value"] == {"node": "Array", "items": [{"node": "Lit", "value": "Key"}, {"node": "Lit", "value": "Epoch"}]}


def test_parser_statements_and_param_block():
    script = parse("param([Parameter(Mandatory)][string]$Name, [int]$Top = 3, [switch]$Force)\n"
                   "$best = Get-Checkpoint -Best\n"
                   "if ($best.Epoch -gt $Top) { 'new' } elseif ($Force) { 'forced' } else { throw 'old' }\n"
                   "foreach ($item in 1, 2) { $total += $item }\nexit 3")
    assert [(p.name, p.type, p.mandatory) for p in script.params] == [("Name", "string", True), ("Top", "int", False),
                                                                     ("Force", "switch", False)]
    assert script.params[1].default == s.Lit(3)
    kinds = [type(statement).__name__ for statement in script.statements]
    assert kinds == ["Assign", "If", "ForEach", "Exit"]
    assign, branch, loop, leave = script.statements
    assert assign.name == "best" and assign.op == "=" and isinstance(assign.value, s.Pipeline)
    assert len(branch.clauses) == 2 and isinstance(branch.otherwise[0], s.Throw)
    assert loop.name == "item" and loop.body[0].op == "+=" and leave.value == s.Lit(3)


ACCEPTED = [
    "Get-Artifact",
    "get-artifact -kind ckpt",
    "Get-CvArtifact | Format-Table",
    "Get-Checkpoint |\n  Sort-Object Epoch -Descending |\n  Select-Object -First 3",
    "Get-Checkpoint `\n  -Latest",
    "Register-Artifact .\\saved_models\\checkpoints -Recurse",
    "Register-Artifact 'C:\\Program Files\\Cevahir\\last.pth', D:\\models\\best.pth",
    "Test-Checkpoint -Path:$path -Level:Compat",
    "$x = @{ Name = 'a'; Expression = { $_.Key.Substring(0, 8) } }",
    "$list = @(); $list += 1",
    "$a = (Get-Checkpoint -Latest).Key",
    "\"epoch $($best.Epoch) of ${name}: `\"quoted`\" `$5\"",
    "# a comment\nGet-Artifact # trailing comment\n<# block\ncomment #> Get-Model",
    "Get-Artifact; Get-Model;",
    "if (Test-Path x) { Get-Artifact }\nelse { 'none' }",
    "Get-Checkpoint | Where-Object Epoch -lt 3",
    "Get-Checkpoint | ? { -not $_.Model -or $_.Epoch -in 1, 2 } | % Key",
    "1KB + 2MB * 3 - 0x1F / 2 % 5",
    "$x[0]; $x[-1]; $x['name']; $x.Count",
    "‘smart’ + “quotes”",
    ".\\other.ps1 -Name prod",
    "Get-AuditEvent -Since 14d -Last 5",
    "Get-Help Test-Checkpoint -?",
]


@pytest.mark.parametrize("text", ACCEPTED)
def test_parser_accepts_the_subset(text):
    assert parse(text).statements


# What PowerShell has and the console deliberately does not: rejected at parse time, by name.
NOT_IN_SUBSET = [
    ("function f { Get-Artifact }", "'function' statement"),
    ("filter f { $_ }", "'filter' statement"),
    ("class A { }", "'class' statement"),
    ("while ($true) { }", "'while' statement"),
    ("for ($i = 0; $i -lt 3; $i++) { }", "'for' statement"),
    ("do { } while ($true)", "'do' statement"),
    ("switch ($x) { 1 { } }", "'switch' statement"),
    ("try { Get-Artifact } catch { }", "'try' statement"),
    ("return 1", "'return' statement"),
    ("trap { }", "'trap' statement"),
    ("[int]'5'", "type literals"),
    ("[System.IO.File]::ReadAllText('secret.txt')", "type literals"),
    ("$x = [type]::GetType('System.Diagnostics.Process')", "type literals"),
    ("Get-Artifact -Key [string]5", "type literals"),
    ("& 'C:\\Windows\\System32\\cmd.exe' /c dir", "call (&)"),
    ("& { Get-Artifact }", "call (&)"),
    (". .\\profile.ps1", "dot-source"),
    ("Get-Artifact > out.txt", "redirection"),
    ("Get-Artifact >> out.txt", "redirection"),
    ("Get-Artifact 2> errors.txt", "redirection"),
    ("Get-Artifact < in.txt", "redirection"),
    ("Get-Artifact &", "& operator"),
    ("Get-Artifact && Get-Model", "& operator"),
    ("Get-Artifact || Get-Model", "|| operator"),
    ("$env:PATH", "scoped variables"),
    ("$env:CEVAHIR_HOME = 'x'", "scoped variable"),
    ("$global:x = 1", "scoped variable"),
    ("Write-Output $using:x", "scoped variables"),
    ("\"home is $env:HOME\"", "scoped variables"),
    ("$x++", "++ operator"),
    ("$x -= 1", "-= operator"),
    ("$(Get-Artifact)", "$( ) outside a double-quoted string"),
    ("@'\ntext\n'@", "here-strings"),
    ("Get-Artifact @args", "splatting"),
    ("'a' -f 1", "-f operator"),
    ("'a' -replace 'a', 'b'", "-replace operator"),
    ("'a,b' -split ','", "-split operator"),
    ("1, 2 -join ','", "-join operator"),
    ("1 -is 2", "-is operator"),
    ("1 -band 2", "-band operator"),
    ("$block = { $_.Key }", "script block outside a command argument"),
    ("Write-Output (1, { $_ })", "script block as a value"),
    ("{ Get-Artifact }", "script block outside a command argument"),
    ("$x::Member", "static members"),
    ("$x.$name", "computed member name"),
    ("${a b}", "${...}"),
    ("$x.__class__", "names that start with _"),
    ("$x._props", "names that start with _"),
    ("'abc'.__len__()", "names that start with _"),
    ("'abc'.GetType()", "method .GetType( )"),
    ("'abc'.PadLeft(1000000000)", "method .PadLeft( )"),
    ("(1).bit_length()", "method .bit_length( )"),
    ("@{a = 1}.items()", "method .items( )"),
    ("$x = if ($true) { 1 }", "'if' statement as the value"),
]


@pytest.mark.parametrize("text, reason", NOT_IN_SUBSET)
def test_powershell_beyond_the_subset_is_rejected_by_name(text, reason):
    with pytest.raises(ParserError) as info:
        parse(text)
    assert info.value.error_id == "NotInCevahirSubset" and info.value.exit_code == 2, str(info.value)
    assert reason in str(info.value) and info.value.line == text.count("\n", 0, info.value.position) + 1
    code, out, err = run(text)
    assert code == 2 and out == "" and "NotInCevahirSubset" in err


# A script block is one expression over data. Each of these tries to make it do something else.
BLOCK_INJECTIONS = [
    "{ Remove-Item x }",
    "{ Invoke-Expression 'Get-Artifact' }",
    "{ Out-File owned.txt }",
    "{ $_ | Out-File owned.txt }",
    "{ $_; Out-File owned.txt }",
    "{ $x = 1 }",
    "{ $_.Key; $_.Kind }",
    "{ (Out-File owned.txt) }",
    "{ @(Out-File owned.txt) }",
    "{ \"$(Out-File owned.txt)\" }",
    "{ $(Out-File owned.txt) }",
    "{ & 'cmd' }",
    "{ [System.IO.File]::WriteAllText('owned.txt', 'x') }",
    "{ $_.__class__.__mro__ }",
    "{ $_.Name.__class__ }",
    "{ $_.type_name.__init__ }",
    "{ $_.Name.GetType().Assembly }",
    "{ $_.Name.Clone() }",
    "{ $env:PATH }",
    "{ if ($true) { 1 } }",
    "{ foreach ($i in 1) { $i } }",
    "{ throw 'x' }",
    "{ exit 9 }",
    "{ { $_ } }",
    "{ }",
]


@pytest.mark.parametrize("block", BLOCK_INJECTIONS)
@pytest.mark.parametrize("command", ["Where-Object", "ForEach-Object", "Sort-Object", "Select-Object"])
def test_a_script_block_cannot_be_made_to_run_anything(tmp_path, command, block):
    text = f"Get-KeyScheme | {command} {block}"
    with pytest.raises(ParserError) as info:
        parse(text)
    assert info.value.exit_code == 2
    code, out, err = run(text)
    assert code == 2 and out == "", "nothing ran: the script did not parse"
    assert not (tmp_path / "owned.txt").exists() and "FullyQualifiedErrorId" in err


def test_text_is_never_evaluated_as_code(tmp_path):
    """A string where a block is expected is a binding error or a property name; it is not parsed."""
    code, out, err = run("'x' | Where-Object -FilterScript 'Out-File owned.txt'")
    assert code == 2 and "text is never evaluated as code" in err
    assert run("'x' | Where-Object 'Out-File owned.txt'") == (0, "", ""), "a string is a property name (here: one that is absent)"
    assert values("'x' | ForEach-Object '{ Out-File owned.txt }'") == []
    assert value("$text = 'Out-File owned.txt'; \"$text\"") == "Out-File owned.txt"
    code, out, err = run("$name = 'Get-Artifact'; $name")
    assert (code, out) == (0, "Get-Artifact\n"), "a variable in command position is an expression, not an invocation"
    code, out, err = run("'x' | Sort-Object @{Expression = 'Out-File owned.txt'}")
    assert code == 0 and not (tmp_path / "owned.txt").exists()
    assert not list(tmp_path.glob("owned*"))


@pytest.mark.parametrize("text", [
    "python -c \"import os; os.remove('x')\"", "rm -rf /", "git status", "Invoke-Expression 'Get-Artifact'",
    "iex 'Get-Artifact'", "Start-Process notepad", "C:\\Windows\\System32\\cmd.exe /c dir", "/bin/sh -c ls",
    "Remove-Item x", "Get-Content secret.txt", "Set-Content x 'y'", "Invoke-WebRequest http://example.com",
    "Add-Type -TypeDefinition 'x'", "Import-Module evil", "New-Object System.Net.WebClient",
])
def test_the_console_never_starts_another_program(text):
    code, out, err = run(text)
    assert code == 1 and out == "" and "CommandNotFoundException" in err and "never starts another program" in err


def test_expression_values_cannot_reach_python_objects():
    """Members are properties of records and keys of hashtables. A Python attribute is not a member."""
    record = Record("Cevahir.Test", Key="k", Nested=Record("Cevahir.Inner", Value=3))
    for name in ("type_name", "names", "get", "items", "to_dict", "set", "has"):
        assert member(record, name) is None, name
    assert member(record, "key") == "k" and member(record, "NESTED").get("Value") == 3 and member(record, "PSTypeName") == "Cevahir.Test"
    for target in ("text", 5, 5.5, True, None, [1, 2], {"a": 1}):
        for name in ("real", "imag", "upper", "append", "keys2", "bit_length", "denominator"):
            assert member(target, name) is None, (target, name)
    assert member("text", "Length") == 4 and member([1, 2], "Count") == 2 and member({"a": 1}, "A") == 1
    assert member({"a": 1}, "Keys") == ["a"] and member({"a": 1}, "Count") == 1
    assert values("(Get-KeyScheme)[0].type_name") == []
    assert values("(Get-KeyScheme)[0].to_dict") == [] and values("'abc'.upper") == [] and values("(5).real") == []
    evaluator = Evaluator(lambda name: None)
    with pytest.raises(EvaluationError, match="cannot evaluate"):
        evaluator.evaluate(object())
    with pytest.raises(EvaluationError, match="does not exist"):
        from cevahir_ai.console.expr import call
        call("abc", "__class__", [])
    assert sorted(METHODS) == ["Contains", "EndsWith", "Replace", "Split", "StartsWith", "Substring", "ToLower", "ToUpper", "Trim"]


def test_limits_keep_one_expression_from_exhausting_the_process():
    assert run("'x' * 2000000")[0] == 1 and "more than 1000000 elements" in run("'x' * 2000000")[2]
    assert run("$a = 1, 2; $a * 900000")[0] == 1
    assert "longer than 512" in run("'a' -match ('a' * 600)")[2]
    assert "invalid regular expression" in run("'a' -match '('")[2]
    assert "divide by zero" in run("1 / 0")[2] and "divide by zero" in run("1 % 0")[2]
    code, out, err = run("(" * 200 + "1" + ")" * 200)
    assert code == 2 and "nested too deeply" in err
    code, out, err = run(" + ".join(["1"] * 5000))
    assert code == 1 and "too deep" in err, "a very long chain ends with an error, not a crash"
    assert run("9" * 60)[0] == 2 and run("1e999")[0] == 2
    with pytest.raises(ParserError, match="larger than 4 MiB"):
        parse("#" + "x" * (4 * 1024 * 1024 + 1))


def test_syntax_errors_carry_line_and_column():
    with pytest.raises(ParserError) as info:
        parse("Get-Artifact\n$x = ")
    assert info.value.error_id == "SyntaxError" and (info.value.line, info.value.column) == (2, 6)
    for text, message in (("'never closed", "never closed"), ("Get-Artifact |", "command is expected after |"),
                          ("if ($true) { 1", "missing '}'"), ("$true = 1", "automatic variable"),
                          ("Get-Artifact | $x", "only a command can follow |"), ("1 +", "found the end"),
                          ("$x = 1 2", "unexpected"), ("1 -bogus 2", "-bogus is not an operator"),
                          ("@{ a 1 }", "= after the hashtable key"), ("foreach ($x of 1) { }", "expected 'in'"),
                          ("Get-Artifact\nparam($x)", "only allowed at the very start"), ("5abc", "is not a number")):
        with pytest.raises(ParserError, match=message):
            parse(text)
    assert is_incomplete("Get-Artifact |") and is_incomplete("if ($x) {") and is_incomplete("'open")
    assert is_incomplete("Get-Artifact | Where-Object { $_.Kind -eq") and is_incomplete("@{ a = 1")
    assert not is_incomplete("Get-Artifact") and not is_incomplete("function f {}") and not is_incomplete("1 2")


def test_number_tokens():
    assert number_token("5") == 5 and number_token("-1.5") == -1.5 and number_token("10MB") == 10 * 1024 * 1024
    assert number_token("1kb") == 1024 and number_token("0x1F") == 31 and number_token("1e3") == 1000.0
    assert number_token("1.5KB") == 1536 and number_token("abc") is None and number_token("3f9a") is None
    assert number_token("10.0.0.1") is None and number_token("") is None and number_token("14d") is None


# ------------------------------------------------------------------------------------ expressions
@pytest.mark.parametrize("text, expected", [
    ("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("7 / 2", 3.5), ("6 / 3", 2), ("7 % 3", 1), ("-7 % 3", -1), ("2 - -3", 5),
    ("'a' + 1", "a1"), ("1 + '2'", 3), ("'ab' * 3", "ababab"), ("10MB / 1KB", 10240), ("0x10 + 1", 17),
    ("'İstanbul'.ToUpper()", "İSTANBUL"), ("'Çatı ve Anahtar'.Substring(0, 4)", "Çatı"),
    ("'  x '.Trim()", "x"), ("'a,b,c'.Split(',').Count", 3), ("'abc'.Replace('b', 'ğ')", "ağc"),
    ("'abc'.StartsWith('ab')", True), ("'abc'.EndsWith('C')", False), ("'abc'.Contains('bc')", True),
    ("'ABC'.ToLower().Length", 3),
    # comparison: case-insensitive, and the right operand takes the type of the left one
    ("'abc' -eq 'ABC'", True), ("'abc' -ceq 'ABC'", False), ("'5' -eq 5", True), ("5 -eq '5'", True),
    ("5 -eq 'five'", False), ("$null -eq 0", False), ("0 -eq $null", False), ("$null -eq $nothing", True),
    ("10 -gt 9", True), ("'10' -gt '9'", False), ("'b' -gt 'A'", True), ("3 -ge 3", True), ("2 -le 1", False),
    ("'ckpt:3f9a' -like 'CKPT:*'", True), ("'ckpt:3f9a' -clike 'CKPT:*'", False), ("'model' -notlike 'm?del'", False),
    ("'checkpoint_epoch_0005.pth' -match 'epoch_(\\d+)'", True), ("'abc' -notmatch '^b'", True), ("'ABC' -cmatch 'b'", False),
    ("2 -in 1, 2, 3", True), ("'B' -in 'a', 'b'", True), ("'B' -cin 'a', 'b'", False), ("4 -notin 1, 2", True),
    ("1, 2, 3 -contains 2", True), ("'a', 'b' -notcontains 'c'", True),
    ("$true -and $false", False), ("$true -or $false", True), ("-not $true", False), ("!$false", True),
    ("$false -and (1 / 0)", False), ("$true -or (1 / 0)", True), ("-not 0", True), ("-not ''", True), ("-not 'x'", False),
    ("1 -eq 1 -and 2 -eq 2", True), ("1 + 1 -eq 2", True),
    # strings
    ("'it''s'", "it's"), ("\"a `\"b`\" c\"", 'a "b" c'), ("\"tab`there\"", "tab\there"), ("\"cost: `$5\"", "cost: $5"),
    ("\"say \"\"hi\"\"\"", 'say "hi"'), ("'no $expansion here'", "no $expansion here"),
    ("‘akıllı’ + “ tırnak”", "akıllı tırnak"),
])
def test_expression_semantics_follow_powershell(text, expected):
    result = value(text)
    assert result == expected and type(result) is type(expected), (text, result)


def test_arrays_filter_and_enumerate_as_in_powershell():
    assert values("1, 2, 3 -gt 1") == [2, 3], "an array on the left filters"
    assert values("'a', 'B', 'c' -eq 'b'") == ["B"] and values("1, 2 -eq 5") == []
    assert value("(1, 2, 3 -gt 1).Count") == 2 and value("@(1, 2, 3).Count") == 3 and value("@().Count") == 0
    assert value("$a = 1, 2, 3; $a[0] + $a[-1]") == 4 and values("$a = 1, 2; $a[5]") == []
    assert values("$a = 1, 2; $a += 3; $a") == [1, 2, 3] and value("$n = $null; $n += 5; $n") == 5
    assert value("@(1, 2) + @(3) | Measure-Object | ForEach-Object Count") == 3
    assert values("(Get-KeyScheme -Kind tok).Scheme") == ["bpe-sem-v1", "bytebpe-v1", "legacy-v3"], "member enumeration"
    assert value("(Get-KeyScheme -Kind tok)[1].Name") == "tok:bytebpe-v1"
    assert value("$h = @{ Name = 'x'; Size = 2KB }; $h.size + $h['SIZE']") == 4096
    assert value("@{ a = 1; b = 2 }.Count") == 2 and values("@{ a = 1 }.missing") == []
    assert value("'abc'[1]") == "b" and value("'abc'.Length") == 3


def test_truthiness_and_text_conversion():
    for falsy in (None, False, 0, 0.0, "", [], [False], [0]):
        assert not truthy(falsy), falsy
    for truthy_value in (True, 1, -1, "x", "False", [0, 0], [1], {"a": 1}, {}, Record("T")):
        assert truthy(truthy_value), truthy_value
    assert to_text(None) == "" and to_text(True) == "True" and to_text(2.0) == "2" and to_text(2.5) == "2.5"
    assert to_text([1, "a", None]) == "1 a " and to_text(Record("T", A=1, B="x")) == "@{A=1; B=x}"
    assert compare("eq", [1, 2, 1], 1) == [1, 1] and compare("contains", "abc", "ABC") is True
    assert compare("ccontains", ["a"], "A") is False and compare("clt", "a", "B") is False and compare("lt", "a", "B") is True


def test_variables_and_string_expansion():
    assert value("$Name = 'dünya'; \"merhaba $name ve $NAME\"") == "merhaba dünya ve dünya", "names ignore case"
    assert value("$c = @{ Key = 'k1' }; \"key: $c.Key\"") == "key: System.Collections.Hashtable.Key", \
        "as in PowerShell, only the variable expands; a member needs $( )"
    assert value("$c = @{ Key = 'k1' }; \"key: $($c.Key) n=$(1 + 2) ${c}\"") == "key: k1 n=3 System.Collections.Hashtable"
    assert value("$list = 1, 2, 3; \"items: $list\"") == "items: 1 2 3"
    assert value("\"undefined: [$nothing]\"") == "undefined: []" and values("$nothing") == []
    assert value("$x = 5; $x += 2; $x") == 7 and value("$s = 'a'; $s += 'b'; $s") == "ab"
    assert value("$one = Get-KeyScheme -Kind arch | Select-Object -First 1; $one.Kind") == "arch", "one object, not an array"
    assert value("$none = Get-KeyScheme -Kind nothing; $null -eq $none") is True
    assert value("$many = Get-KeyScheme -Kind tok; $many.Count") == 3
    assert value("$dir = 'C:\\models'; Write-Output $dir\\last.pth") == "C:\\models\\last.pth"
    code, out, err = run("$true = 1")
    assert code == 2 and "automatic variable" in err
    assert value("$PolicyKey") == "builtin:bootstrap-v0" and value("$true") is True and values("$null") == []


def test_statements():
    assert values("if (1 -gt 2) { 'a' } elseif ($true) { 'b' } else { 'c' }") == ["b"]
    assert values("if ($false) { 'a' }") == [] and values("if ($false) { 'a' } else { 'c' }") == ["c"]
    assert values("foreach ($i in 1, 2, 3) { if ($i % 2) { $i * 10 } }") == [10, 30]
    assert values("$sum = 0; foreach ($s in Get-KeyScheme -Kind tok) { $sum += $s.Scheme.Length }; $sum") == [29]
    assert values("foreach ($x in @()) { 'never' }") == []
    assert run("'before'; throw 'stop here'; 'after'") == (1, "before\n", run("throw 'stop here'")[2])
    assert "stop here" in run("throw 'stop here'")[2] and "ScriptThrow" in run("throw")[2]
    assert run("'a'; exit 7; 'b'")[:2] == (7, "a\n") and run("exit")[0] == 0 and run("exit 2 + 3")[0] == 5
    assert run("exit 'x'")[0] == 1


# ---------------------------------------------------------------------------------------- binding
def find(name):
    command = registry.find(name)
    assert command is not None, name
    return command


def test_command_lookup_by_name_alias_and_cv_prefix():
    assert find("get-artifact") is find("Get-Artifact") is find("Get-CvArtifact") is find("GET-CVARTIFACT")
    assert find("?") is find("where") is find("Where-Object") and find("%") is find("ForEach-Object")
    assert find("select") is find("Select-Object") and find("ft").name == "Format-Table" and find("inspect").name == "Test-Checkpoint"
    assert registry.find("Where-CvObject") is None, "only Cevahir cmdlets take the Cv prefix"
    assert registry.find("Get-Nothing") is None and registry.find("Get-CvNothing") is None
    names = [command.name for command in registry.commands()]
    assert len(names) == len(set(names)) and {"Get-Artifact", "Register-Artifact", "Get-Checkpoint", "Test-Checkpoint",
                                              "Get-Model", "Get-Tokenizer", "Get-Lineage", "Get-AuditEvent", "Test-AuditLog",
                                              "Compare-Key", "Get-Key", "Get-Help", "Get-Command", "Where-Object",
                                              "Select-Object", "Sort-Object", "Format-Table", "Format-List", "ConvertTo-Json",
                                              "Measure-Object", "ForEach-Object", "Export-Artifact", "Import-Artifact"} <= set(names)
    for command in registry.commands():
        verb, _, noun = command.name.partition("-")
        assert verb and noun and command.action_class in registry.ACTION_CLASSES and command.kind in ("process", "stream")
        assert len({parameter.name.lower() for parameter in command.params}) == len(command.params)


def test_named_positional_switch_and_prefix_binding():
    command = find("Test-Checkpoint")
    bound = binder.bind(command, [(None, "ckpt:3f9a"), ("level", "compat"), ("Pur", "SERVE"), ("NoRecord", binder.NO_VALUE)])
    assert bound == {"Key": "ckpt:3f9a", "Level": "Compat", "Purpose": "serve", "NoRecord": True}, "choices take the declared spelling"
    assert binder.bind(command, [("Id", "k"), ("NoRecord", False)]) == {"Key": "k", "NoRecord": False}, "alias; -Switch:$false"
    full = binder.bind_input(command, bound)
    assert full["Dtype"] == "float32" and full["Target"] is None and full["Verbose"] is False and full["MaxSeqLength"] is None
    for arguments, error_id in (
            ([("Nope", "x")], "NamedParameterNotFound"), ([("M", "x")], "AmbiguousParameter"),
            ([("Level", binder.NO_VALUE)], "MissingArgument"), ([("Level", "Deep")], "ParameterArgumentValidationError"),
            ([("Level", "File"), ("Level", "File")], "ParameterAlreadyBound"),
            ([(None, "a"), (None, "b")], "PositionalParameterNotFound"),
            ([("MaxSeqLength", "many")], "ParameterArgumentTransformationError"),
            ([("Key", ["a", "b"])], "ParameterArgumentTransformationError"),
            ([("Key", None)], "ParameterArgumentTransformationError")):
        with pytest.raises(ParameterBindingError) as info:
            binder.bind(command, arguments)
        assert info.value.error_id == error_id and info.value.exit_code == 2, arguments
    with pytest.raises(ParameterBindingError, match="mandatory parameter -To"):
        binder.bind_input(find("Export-Artifact"), {})
    assert binder.bind(find("Register-Artifact"), [(None, ["a", "b"]), ("Recurse", binder.NO_VALUE)]) == {"Path": ["a", "b"], "Recurse": True}
    assert binder.bind(find("Get-AuditEvent"), [("Last", 5)]) == {"Last": 5}
    assert binder.bind(find("Get-AuditEvent"), [("Last", "5")]) == {"Last": 5}
    with pytest.raises(ParameterBindingError):
        binder.bind(find("Get-AuditEvent"), [("Last", 5.5)])


def test_pipeline_input_binds_by_value_and_by_property_name():
    command = find("Test-Checkpoint")
    by_value = binder.bind_input(command, {}, "ckpt:3f9a")
    assert by_value["Key"] == "ckpt:3f9a" and by_value["Path"] is None
    record = Record("Cevahir.Checkpoint", Key="cev:x", Path="/models/last.pth", Epoch=3)
    assert binder.bind_input(command, {}, record)["Key"] == "cev:x" and binder.bind_input(command, {}, record)["Path"] == "/models/last.pth"
    file_item = Record("Cevahir.FileItem", Name="last.pth", FullName=r"C:\models\last.pth")
    bound = binder.bind_input(command, {}, file_item)
    assert bound["Path"] == r"C:\models\last.pth" and bound["Key"] is None, "FullName is an alias of -Path"
    assert binder.bind_input(command, {"Key": "explicit"}, record)["Key"] == "explicit", "an explicit argument wins"
    assert binder.bind_input(command, {}, {"key": "from-hashtable"})["Key"] == "from-hashtable"
    code, out, err = run("Get-KeyScheme | Get-Environment")
    assert code == 2 and "does not take pipeline input" in err


def test_numbers_keep_their_spelling_for_text_parameters():
    assert run("Write-Host 0193 1.50 0x1F")[1] == "0193 1.50 0x1F\n", "as PowerShell: the argument text is kept"
    assert values("Write-Output 0193") == [193] and value("0193 + 1") == 194
    assert value("Get-KeyScheme -Kind tok | Select-Object -First 01 | ForEach-Object Name") == "tok:bpe-sem-v1"
    code, out, err = run("Get-Lineage 0193abcd")
    assert "'0193abcd'" in err and code == 1


def test_question_mark_parameter_shows_help():
    code, out, err = run("Test-Checkpoint -?")
    assert code == 0 and out.startswith("NAME\n    Test-Checkpoint\n") and "SYNTAX" in out and err == ""


# -------------------------------------------------------------------------------------- pipeline
SOURCE = """'[{"n": "b", "v": 3, "t": ["x"]}, {"n": "A", "v": 1, "t": []}, {"n": "c", "v": 3, "t": ["x", "y"]}]' | ConvertFrom-Json"""


def test_objects_flow_not_text():
    rows = values(f"{SOURCE} | Where-Object v -eq 3 | Sort-Object n -Descending")
    assert [row["n"] for row in rows] == ["c", "b"] and rows[0]["t"] == ["x", "y"], "properties keep their types"
    assert value(f"{SOURCE} | Where-Object {{ $_.v -gt 1 -and $_.t.Count -gt 1 }} | ForEach-Object n") == "c"
    assert values(f"{SOURCE} | Where-Object n -like '[ab]'") == values(f"{SOURCE} | Where-Object {{ $_.n -like '[ab]' }}")
    assert [row["n"] for row in values(f"{SOURCE} | Where-Object t")] == ["b", "c"], "a bare property is a truth test"
    assert values(f"{SOURCE} | Where-Object v -in 1, 2 | ForEach-Object {{ $_.n.ToLower() + $_.v }}") == ["a1"]
    assert values("1, 2, 3 | Where-Object { $_ -ne 2 }") == [1, 3]
    assert values("@(1, 2), 3 | Measure-Object | ForEach-Object Count") == [2], "nested arrays are objects of their own"
    assert values("$null | Measure-Object | ForEach-Object Count") == [0], "$null contributes nothing"
    assert values("@{ a = 1 } | Measure-Object | ForEach-Object Count") == [1], "a hashtable is one object"


def test_sort_select_group_measure():
    assert [row["n"] for row in values(f"{SOURCE} | Sort-Object n")] == ["A", "b", "c"], "text sorts without regard to case"
    assert [row["n"] for row in values(f"{SOURCE} | Sort-Object v, n")] == ["A", "b", "c"]
    assert [row["n"] for row in values(f"{SOURCE} | Sort-Object v -Descending")] == ["b", "c", "A"], "stable"
    assert [row["n"] for row in values(f"{SOURCE} | Sort-Object {{ $_.t.Count }} -Descending")] == ["c", "b", "A"]
    assert [row["v"] for row in values(f"{SOURCE} | Sort-Object v -Unique")] == [1, 3]
    assert values("3, 1, 2 | Sort-Object") == [1, 2, 3] and values("'b', 'A' | Sort-Object -Descending") == ["b", "A"]
    assert values(f"{SOURCE} | Select-Object -First 1 | ForEach-Object n") == ["b"]
    assert values(f"{SOURCE} | Select-Object -Last 2 | ForEach-Object n") == ["A", "c"]
    assert values(f"{SOURCE} | Select-Object -Skip 1 -First 1 | ForEach-Object n") == ["A"]
    assert values(f"{SOURCE} | Select-Object -ExpandProperty t") == ["x", "x", "y"]
    assert values(f"{SOURCE} | Select-Object v -Unique") == [{"v": 3}, {"v": 1}]
    assert values(f"{SOURCE} | Select-Object n, @{{Name='Double'; Expression={{ $_.v * 2 }}}} -First 1") == [{"n": "b", "Double": 6}]
    assert values(f"{SOURCE} | Select-Object * -ExcludeProperty t -First 1") == [{"n": "b", "v": 3}]
    assert values(f"{SOURCE} | Select-Object '[nv]' -First 1") == [{"n": "b", "v": 3}], "wildcards in property names"
    groups = values(f"{SOURCE} | Group-Object v")
    assert [(group["Count"], group["Name"]) for group in groups] == [(1, "1"), (2, "3")] and len(groups[1]["Group"]) == 2
    measured = value(f"{SOURCE} | Measure-Object v -Sum -Average -Minimum -Maximum")
    assert measured == {"Count": 3, "Sum": 7, "Average": 7 / 3, "Minimum": 1, "Maximum": 3, "Property": "v"}
    assert value(f"{SOURCE} | Measure-Object")["Count"] == 3
    code, out, err = run(f"{SOURCE} | Measure-Object n -Sum")
    assert code == 1 and "is not a number" in err
    assert run(f"{SOURCE} | Select-Object -First -1")[0] == 2


def endless(context, p):
    """An endless source that counts how many objects were pulled from it."""
    index = 0
    while True:
        index += 1
        endless.pulled = index
        if p["Fail"] and index == p["Fail"]:
            raise ValueError(f"source failed at {index}")
        yield Record("Cevahir.Test", Index=index)


@pytest.fixture
def endless_source():
    """Registers ``Test-Endless`` for one test only, so the command list of other tests is the real one."""
    registry.load()
    cmdlet("Test-Endless", params=[Param("Fail", "int")], synopsis=("test source", "sınama kaynağı"))(endless)
    yield
    del registry._COMMANDS["test-endless"]


def test_the_pipeline_streams(endless_source):
    assert [row["Index"] for row in values("Test-Endless | Select-Object -First 3")] == [1, 2, 3]
    assert endless.pulled == 3, "Select-Object -First stops reading: nothing more was produced"
    assert values("Test-Endless | Where-Object { $_.Index % 5 -eq 0 } | Select-Object -First 2 | ForEach-Object Index") == [5, 10]
    assert endless.pulled == 10
    code, out, err = run("Test-Endless -Fail 3 | ForEach-Object Index")
    assert code == 2 and out == "1\n2\n" and "Test-Endless : [InvalidArgument] source failed at 3" in err, \
        "what was produced before the error is still shown, and the error names its command"


def test_errors_on_single_objects_do_not_end_the_pipeline(tmp_path):
    code, out, err = run("'ckpt:aaaa', 'tok:bbbb' | Get-Artifact; 'next statement'", tmp_path=tmp_path)
    assert code == 1 and out == "next statement\n" and err.count("Get-Artifact : [ObjectNotFound]") == 2
    assert "ObjectNotFound" in err and "FullyQualifiedErrorId : NotFound,Cevahir.Commands.GetArtifact" in err
    code, out, err = run("'ckpt:aaaa', 'tok:bbbb' | Get-Artifact -ErrorAction Stop; 'next statement'", tmp_path=tmp_path)
    assert code == 1 and err.count("[ObjectNotFound]") == 1 and out == "next statement\n", "Stop ends the statement at the first error"
    code, out, err = run("'ckpt:aaaa' | Get-Artifact -ErrorAction SilentlyContinue; $Error.Count", tmp_path=tmp_path)
    assert (code, out, err) == (0, "1\n", ""), "silenced, but remembered in $Error"
    assert run("'ckpt:aaaa' | Get-Artifact -ErrorAction Ignore; $Error.Count", tmp_path=tmp_path) == (0, "0\n", "")
    code, out, err = run("$ErrorActionPreference = 'Stop'; 'ckpt:aaaa', 'tok:bbbb' | Get-Artifact; 'never'", tmp_path=tmp_path)
    assert code == 1 and out == "" and err.count("[ObjectNotFound]") == 1, "with Stop the script ends"
    code, out, err = run("Get-Artifact ckpt:aaaa; $?; $LASTEXITCODE; $Error[0].Category", tmp_path=tmp_path)
    assert (code, out) == (1, "False\n1\nObjectNotFound\n")
    assert run("Write-Error 'kendi hatam'; 'devam'")[:2] == (1, "devam\n") and "kendi hatam" in run("Write-Error 'kendi hatam'")[2]
    assert run("Write-Warning 'dikkat'") == (0, "", "WARNING: dikkat\n")
    assert run("Write-Verbose 'ayrıntı'") == (0, "", "") and run("Write-Verbose 'ayrıntı' -Verbose")[2] == "VERBOSE: ayrıntı\n"
    assert run("Get-KeyScheme -Kind tok | Out-Null") == (0, "", "")


def test_exit_code_is_the_highest_among_errors_and_failed_verifications():
    assert run("Get-Nothing; Get-KeyScheme -Nope")[0] == 2
    assert run("Get-KeyScheme -Nope; 'ok'")[0] == 2 and run("'ok'")[0] == 0
    assert run("Get-Nothing; exit 0")[0] == 0, "an explicit exit decides"


# --------------------------------------------------------------------------------------- json
def test_json_round_trip_keeps_structure_and_turkish_text():
    text = '{"ad": "ğüşıöçİ", "n": 1.5, "liste": [1, {"iç": null}], "evet": true}'
    assert json.loads(run(f"'{text}' | ConvertFrom-Json | ConvertTo-Json -Compress")[1]) == json.loads(text)
    assert value(f"('{text}' | ConvertFrom-Json).liste[1].iç -eq $null") is True
    assert run("'not json' | ConvertFrom-Json")[0] == 2
    typed = values("""'{"__type": "Cevahir.Thing", "Key": "k"}' | ConvertFrom-Json | Get-Member | ForEach-Object TypeName""")
    assert typed == ["Cevahir.Thing"], "a typed record keeps its type through JSON"
