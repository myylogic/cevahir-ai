"""The console as a program: formatting, help, completion, entry points, the prompt and the PowerShell module.

Golden outputs pin what the owner sees (tables, lists, JSON). The end-to-end session registers real
artefacts written by the real writers into a temporary folder and walks the whole path:
register -> Get-Checkpoint -> Test-Checkpoint -> Get-Lineage -> Get-AuditEvent -> Test-AuditLog,
then tampers with a checkpoint and with the log and expects both to be reported with exit code 5.
"""
import hashlib
import io
import json
import ntpath
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cevahir_ai.console import completion, registry, repl
from cevahir_ai.console.format import format_list, format_table, render
from cevahir_ai.console.help import help_text, overview, syntax, topic_text, topics
from cevahir_ai.console.lineedit_win import LineEditor
from cevahir_ai.console.parser import parse
from cevahir_ai.console.runtime import Session
from cevahir_ai.console.tools import gen_powershell
from cevahir_ai.domain.records import Record

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research" / "living_learning_paradigm_2026_11_13"
LEGACY = ("src", "model", "model_management", "tokenizer_management", "training_system", "training_management",
          "data_loader_management", "cognitive_management", "chatting_management", "api", "database")
pytestmark = pytest.mark.filterwarnings("ignore:checkpoint has no tokenizer identity:RuntimeWarning")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CEVAHIR_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("CEVAHIR_UI_CULTURE", raising=False)


def new_session(home, **options):
    options.setdefault("width", 120)
    options.setdefault("culture", "en")
    return Session(home=str(home), stdout=io.StringIO(), stderr=io.StringIO(), environ={}, **options)


def run(tmp_path, text, **options):
    """``(exit code, stdout, stderr)`` of one script in a fresh session whose home is ``tmp_path/home``."""
    session = new_session(tmp_path / "home", **options)
    code = session.run(text)
    return code, session.stdout.getvalue(), session.stderr.getvalue()


def console(*arguments, home, stdin=None, cwd=None, **extra):
    """The real entry point in a fresh interpreter: ``python -m cevahir_ai ...``."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "CEVAHIR_HOME": str(home),
           "PYTHONPATH": str(ROOT), "COLUMNS": "120", **extra}
    result = subprocess.run([sys.executable, "-m", "cevahir_ai", *arguments], cwd=cwd or ROOT, env=env, input=stdin,
                            capture_output=True, text=True, encoding="utf-8", timeout=300)
    return result.returncode, result.stdout, result.stderr


# ------------------------------------------------------------------------------- formatting goldens
KEY = "cev:ckpt:members-v1:"
CHECKPOINTS = [
    Record("Cevahir.Checkpoint", Key=KEY + "3f9a1c0b7d2e" + "0" * 52, Short="ckpt:3f9a1c0b7d2e", Epoch=5, Metric=2.31,
           Envelope="tm2", Arch="arch:7ccdfc96fc5c", Tok="tok:d211997f8c7d", State="verified",
           Path=r"C:\Users\Yasin\saved_models\checkpoints\checkpoint_epoch_0005.pth"),
    Record("Cevahir.Checkpoint", Key=KEY + "a17e55d0c931" + "0" * 52, Short="ckpt:a17e55d0c931", Epoch=12, Metric=1.875,
           Envelope="tm2", Arch="arch:7ccdfc96fc5c", Tok="tok:d211997f8c7d", State="registered",
           Path="/content/drive/MyDrive/cevahir-runs/son.pth"),
    Record("Cevahir.Checkpoint", Key=KEY + "0be41c0a7d22" + "0" * 52, Short="ckpt:0be41c0a7d22", Epoch=None, Metric=None,
           Envelope="bare", Arch=None, Tok=None, State="quarantined", Path=None),
]
TABLE_120 = """
Key               Epoch Metric Envelope Arch              Tok              State       File
---               ----- ------ -------- ----              ---              -----       ----
ckpt:3f9a1c0b7d2e     5   2.31 tm2      arch:7ccdfc96fc5c tok:d211997f8c7d verified    checkpoint_epoch_0005.pth
ckpt:a17e55d0c931    12  1.875 tm2      arch:7ccdfc96fc5c tok:d211997f8c7d registered  son.pth
ckpt:0be41c0a7d22              bare                                        quarantined
"""
TABLE_80 = """
Key               Epoch Metric Envelope Arch              Tok         State File
---               ----- ------ -------- ----              ---         ----- ----
ckpt:3f9a1c0b7d2e     5   2.31 tm2      arch:7ccdfc96fc5c tok:d21199… veri… che…
ckpt:a17e55d0c931    12  1.875 tm2      arch:7ccdfc96fc5c tok:d21199… regi… son…
ckpt:0be41c0a7d22              bare                                   quar…
"""
TABLE_CHOSEN = """
Short             Epoch State
-----             ----- -----
ckpt:3f9a1c0b7d2e     5 verified
ckpt:a17e55d0c931    12 registered
ckpt:0be41c0a7d22       quarantined
"""
LIST_CHOSEN = """
Short  : ckpt:3f9a1c0b7d2e
Epoch  : 5
Metric : 2.31
Path   : C:\\Users\\Yasin\\saved_models\\checkpoints\\checkpoint_epoch_0005.pth
"""


def test_format_table_golden_at_width_120_and_80():
    assert format_table(CHECKPOINTS, width=120) == TABLE_120
    assert format_table(CHECKPOINTS, width=80) == TABLE_80, "too narrow: columns are cut from the right, keys stay whole"
    assert all(len(line) <= 80 for line in TABLE_80.splitlines())
    assert format_table(CHECKPOINTS, ["Short", "Epoch", "State"], width=80) == TABLE_CHOSEN
    assert format_table([], width=80) == "" and format_list([]) == ""
    for text in (TABLE_120, TABLE_80, TABLE_CHOSEN):
        assert all(ord(char) < 128 or char == "…" for char in text), "no box-drawing characters: it reads in any console"
        assert not any(line != line.rstrip() for line in text.splitlines()), "no trailing blanks"


def test_format_list_and_default_rendering_golden():
    assert format_list(CHECKPOINTS[:1], ["Short", "Epoch", "Metric", "Path"]) == LIST_CHOSEN
    thing = Record("Cevahir.Thing", Ad="ğüşıöçİ", Liste=[1, 2], Yok=None, Evet=True, Oran=0.5)
    assert format_list([thing]) == "\nAd    : ğüşıöçİ\nListe : {1, 2}\nYok   : \nEvet  : True\nOran  : 0.5\n"
    mixed = ["bir satır", 42, Record("X", A=1, B="iki"), Record("X", A=10, B="on"), {"Name": "x", "Size": 3},
             Record("Wide", A=1, B=2, C=3, D=4, E=5)]
    assert render(mixed, 80) == ("bir satır\n42\n"
                                 "\n A B\n - -\n 1 iki\n10 on\n"
                                 "\nName Value\n---- -----\nName x\nSize 3\n"
                                 "\nA : 1\nB : 2\nC : 3\nD : 4\nE : 5\n")
    assert render(CHECKPOINTS, 120) == TABLE_120, "a record type with a view is a table by default"
    with pytest.raises(ValueError, match="no view named"):
        format_table(CHECKPOINTS, view="Tree")


JSON_GOLDEN = """[
  {
    "Name": "arch:cfg-v1",
    "Durum": "primary",
    "Not": "ğüşıöçİ",
    "Liste": [
      1,
      "iki",
      null,
      true
    ],
    "İç": {
      "Oran": 0.5
    }
  },
  {
    "Name": "arch:ctor-v0",
    "Durum": "legacy",
    "Not": "ğüşıöçİ",
    "Liste": [
      1,
      "iki",
      null,
      true
    ],
    "İç": {
      "Oran": 0.5
    }
  }
]
"""


def test_convert_to_json_golden(tmp_path):
    script = ("Get-KeyScheme -Kind arch | Select-Object Name, @{Name='Durum'; Expression={ $_.Status }}, "
              "@{Name='Not'; Expression={ 'ğüşıöçİ' }}, @{Name='Liste'; Expression={ 1, 'iki', $null, $true }}, "
              "@{Name='İç'; Expression={ @{ Oran = 0.5 } }} | ConvertTo-Json")
    assert run(tmp_path, script) == (0, JSON_GOLDEN, ""), "PowerShell's layout: two spaces, text not escaped"
    one = run(tmp_path, "Get-KeyScheme -Kind arch | Select-Object Name -First 1 | ConvertTo-Json -Compress")
    assert one == (0, '{"Name":"arch:cfg-v1"}\n', ""), "one object is an object, not an array"
    assert run(tmp_path, "Get-KeyScheme -Kind arch | Select-Object Name -First 1 | ConvertTo-Json -Compress -AsArray")[1] == \
        '[{"Name":"arch:cfg-v1"}]\n'
    assert run(tmp_path, "Get-KeyScheme -Kind nothing | ConvertTo-Json")[1] == "[]\n"
    code, out, err = run(tmp_path, "@{ a = @{ b = @{ c = @{ d = 1 } } } } | ConvertTo-Json -Depth 1 -Compress")
    assert code == 0 and json.loads(out) == {"a": {"b": "System.Collections.Hashtable"}}
    assert "WARNING: ConvertTo-Json: 1 nested values were cut at -Depth 1" in err, "a cut is said, not silent"
    assert json.loads(run(tmp_path, "@{ a = @{ b = @{ c = @{ d = 1 } } } } | ConvertTo-Json -Compress")[1]) == \
        {"a": {"b": {"c": {"d": 1}}}}


def test_format_cmdlets_and_out_file(tmp_path):
    code, out, err = run(tmp_path, "Get-KeyScheme -Kind tok | Format-Table Name, Status, @{Name='T'; Expression={ $_.Torch }}")
    assert (code, err) == (0, "") and out == ("\nName           Status  T\n----           ------  -\n"
                                               "tok:bpe-sem-v1 primary False\ntok:bytebpe-v1 primary False\n"
                                               "tok:legacy-v3  legacy  True\n")
    assert run(tmp_path, "Get-KeyScheme -Kind tok | Select-Object -First 1 | Format-List Name, Torch")[1] == \
        "\nName  : tok:bpe-sem-v1\nTorch : False\n"
    assert run(tmp_path, "Get-KeyScheme -Kind tok | ft -View Nope")[0] == 2
    target = tmp_path / "out" / "schemes.json"
    target.parent.mkdir()
    code, out, err = run(tmp_path, f"Get-KeyScheme -Kind tok | Select-Object Name | ConvertTo-Json -Compress | Out-File '{target}'")
    assert (code, out, err) == (0, "", "") and json.loads(target.read_text(encoding="utf-8"))[2] == {"Name": "tok:legacy-v3"}
    assert b"\r" not in target.read_bytes()
    assert run(tmp_path, f"'x' | Out-File '{target}' -WhatIf")[1].startswith("What if: Out-File") and "tok:" in target.read_text()
    inside = tmp_path / "home" / "audit" / "forged.jsonl"
    code, out, err = run(tmp_path, f"'x' | Out-File '{inside}'")
    assert code == 3 and "inside CEVAHIR_HOME" in err and not inside.exists(), "the registry is not a place for Out-File"
    protected = ROOT / "research" / "owned-by-test.txt"
    code, out, err = run(tmp_path, f"'x' | Out-File '{protected}'")
    assert code == 3 and "byte-protected" in err and not protected.exists(), "research records are never written by the console"
    text = run(tmp_path, "$t = Get-KeyScheme -Kind tok | Out-String; $t.Contains('tok:legacy-v3')")[1]
    assert text == "True\n"


# ------------------------------------------------------------------------------------------- help
def test_help_is_complete_in_turkish_and_english_and_every_example_parses():
    names = [command.name for command in registry.commands()]
    assert len(names) >= 40
    for command in registry.commands():
        assert all(command.synopsis) and len(command.synopsis) == 2, f"{command.name}: synopsis in en and tr"
        assert command.examples, f"{command.name}: at least one example"
        for parameter in command.params:
            assert parameter.help[0] and parameter.help[1], f"{command.name} -{parameter.name}: help in en and tr"
        for code, _, _ in command.examples:
            parse(code)                                    # every example is a valid script of the language
        for culture in ("en", "tr"):
            text = help_text(command, culture, full=True)
            assert command.name in text and syntax(command).split()[0] == command.name
            for parameter in command.params:
                assert f"-{parameter.name}" in text
    assert registry.find("Test-Checkpoint").torch and not registry.find("Get-Checkpoint").torch


def test_help_text_golden_and_cultures(tmp_path):
    assert syntax(registry.find("Get-Lineage")) == "Get-Lineage [-Key] <string> [-Direction <Up|Down|Both>] [-Depth <int>] [<CommonParameters>]"
    assert syntax(registry.find("Register-Artifact")) == ("Register-Artifact [-Path] <string[]> [-Kind <auto|ckpt|tok|cache|corpus|evalset|file>] "
                                                           "[-Recurse] [-WhatIf] [<CommonParameters>]")
    english = help_text(registry.find("Get-Lineage"), "en")
    assert english.startswith("NAME\n    Get-Lineage\n\nSYNOPSIS\n    Shows which keys produced an artefact, or what it produced.\n\nSYNTAX\n")
    assert "in real PowerShell: Get-CvLineage" in english and "output: Cevahir.LineageEdge" in english
    turkish = help_text(registry.find("Get-Lineage"), "tr", full=True)
    assert turkish.startswith("AD\n    Get-Lineage\n\nÖZET\n    Bir eseri hangi anahtarların ürettiğini ya da ondan ne üretildiğini gösterir.")
    assert "SÖZDİZİMİ" in turkish and "PARAMETRELER" in turkish and "ÖRNEKLER" in turkish and "çıktı: Cevahir.LineageEdge" in turkish
    assert "(zorunlu; konum 0; boru hattı girdisi: ByValue, ByPropertyName; alias: -Id)" in turkish
    assert help_text(registry.find("Test-Checkpoint"), "en", parameter="Level").startswith("    -Level <File|Weights|Compat>")
    assert "PS cevahir> " in help_text(registry.find("Test-Checkpoint"), "en", examples=True)
    assert run(tmp_path, "Get-Help Get-Model", culture="tr")[1].startswith("AD\n    Get-Model")
    assert run(tmp_path, "Get-Help Get-Model -Culture tr")[1].startswith("AD\n") and run(tmp_path, "help gcm")[1].startswith("NAME\n    Get-Command")
    assert run(tmp_path, "Get-Help Get-CvModel")[1].startswith("NAME\n    Get-Model")
    assert "Get-Checkpoint" in overview("en") and "Komutlar:" in overview("tr") and run(tmp_path, "Get-Help")[1] == overview("en")
    assert topics() == ["about_Cevahir_Registry", "about_Cevahir_Subset"]
    for name in topics():
        assert topic_text(name, "en").startswith("TOPIC") and topic_text(name, "tr").startswith("KONU")
    assert "NotInCevahirSubset" in run(tmp_path, "Get-Help about_cevahir_subset")[1]
    assert run(tmp_path, "Get-Help about_Nothing")[0] == 1 and run(tmp_path, "Get-Help Get-Nothing")[0] == 1
    assert run(tmp_path, "Get-Help Get-Model -Parameter Nope")[0] == 1


def test_ui_culture_comes_from_the_environment():
    from cevahir_ai.ops.env import ui_culture
    assert ui_culture({"CEVAHIR_UI_CULTURE": "tr"}) == "tr" and ui_culture({"CEVAHIR_UI_CULTURE": "en", "LANG": "tr_TR.UTF-8"}) == "en"
    assert ui_culture({"LANG": "tr_TR.UTF-8"}) == "tr" and ui_culture({"LC_ALL": "Turkish_Türkiye.1254"}) == "tr"
    assert ui_culture({"LANG": "en_US.UTF-8"}) == "en" and ui_culture({"LANG": "C"}) == "en"


def test_get_command_and_get_member(tmp_path):
    out = run(tmp_path, "Get-Command -Noun Checkpoint | ForEach-Object Name")[1]
    assert out == "Get-Checkpoint\nTest-Checkpoint\n"
    assert run(tmp_path, "Get-Command *-Artifact | Measure-Object | ForEach-Object Count")[1] == "4\n"
    assert run(tmp_path, "(Get-Command Test-Checkpoint).CvName; (Get-Command inspect).Name")[1] == "Test-CvCheckpoint\nTest-Checkpoint\n"
    classes = run(tmp_path, "Get-Command -Module Cevahir | Where-Object Class -ne read | ForEach-Object Name")[1].split()
    assert classes == ["Export-Artifact", "Import-Artifact", "Register-Artifact"], "the only acting cmdlets today"
    members = run(tmp_path, "Get-KeyScheme | Get-Member | ForEach-Object Name")[1].split()
    assert members == ["Name", "Kind", "Scheme", "Status", "Torch", "Newline", "Summary"]
    assert "Substring" in run(tmp_path, "'x' | Get-Member | ForEach-Object Name")[1]
    assert run(tmp_path, "$a = 1; $b = 'x'; Get-Variable b | ForEach-Object Value")[1] == "x\n"
    assert "R-AUDIT" in run(tmp_path, "Get-Responsibility | Where-Object Status -eq owned | ForEach-Object Id")[1]
    assert run(tmp_path, "Get-Responsibility R-POLICY | ForEach-Object Status")[1] == "planned\n", "the policy layer is not built yet"


# ------------------------------------------------------------------------------------- completion
def texts(line, session=None, cursor=None):
    return [item.text for item in completion.complete(line, cursor, session)]


def test_completion_of_commands_parameters_values_and_variables(tmp_path):
    assert texts("Get-Ch") == ["Get-Checkpoint", "Get-ChildItem"] and texts("get-aud") == ["Get-AuditEvent"]
    assert "Test-AuditLog" in texts("Test-") and texts("ins") == ["inspect"] and "Get-Artifact" in texts("")
    assert texts("Get-Artifact | Wh") == ["Where-Object", "where"] and texts("$x = Get-Mo") == ["Get-Model"]
    assert texts("Test-Checkpoint -Le") == ["-Level"] and texts("Test-Checkpoint -") [:2] == ["-Key", "-Path"]
    assert texts("Test-Checkpoint -Level ") == ["File", "Weights", "Compat"] and texts("Test-Checkpoint -Level c") == ["Compat"]
    assert texts("Test-CvCheckpoint -Purpose se") == ["serve"] and texts("Get-Lineage x -Direction ") == ["Up", "Down", "Both"]
    assert texts("Register-Artifact -Kind co") == ["corpus"] and "-WhatIf" in texts("Register-Artifact -W")
    assert texts("Get-Help Test-Au") == ["Test-AuditLog"] and texts("Get-Artifact -Kind ckpt | Format-Table -Vi") == ["-View"]
    session = new_session(tmp_path / "home")
    session.run("$best = 1; $bestOfAll = 2")
    assert texts("Write-Output $be", session) == ["$best", "$bestofall"] and "$true" in texts("$t")
    assert texts("Get-Artifact -Kind", cursor=5) == ["Get-Artifact", "Get-AuditEvent"], "the cursor decides, not the line end"
    described = completion.complete("Get-Mod")[0]
    assert (described.kind, described.description) == ("Command", "Lists servable model identities (weights + architecture + tokenizer).")


def test_completion_of_keys_and_paths(tmp_path):
    from cevahir_ai.ops import artifacts
    from cevahir_ai.store.core import Store
    from cevahir_ai.store.core import new_manifest
    store = Store(tmp_path / "home", environ={})
    store.put(new_manifest("ckpt", KEY + "3f9a" + "0" * 60))
    store.put(new_manifest("ckpt", KEY + "3fbb" + "0" * 60))
    artifacts.derive(store, "tok", "cev:tok:bpe-sem-v1:" + "70" * 32)
    session = new_session(tmp_path / "home")
    assert texts("Test-Checkpoint ckpt:3f", session) == ["ckpt:3f9a00000000", "ckpt:3fbb00000000"]
    assert texts("Get-Lineage -Key tok:", session) == ["tok:707070707070"] and texts("Get-Artifact ckpt:3f9", session) == ["ckpt:3f9a00000000"]
    assert len(texts("Get-Artifact ", session)) == 3, "a key parameter in position 0 offers the registered keys"
    folder = tmp_path / "saved models"
    (folder / "checkpoints").mkdir(parents=True)
    (folder / "last.pth").write_bytes(b"x")
    (folder / ".hidden").write_bytes(b"x")
    prefix = str(folder) + os.sep
    assert texts(f"Register-Artifact '{prefix}") == [prefix + "checkpoints" + os.sep, prefix + "last.pth"]
    assert texts(f"Register-Artifact '{prefix}la") == [prefix + "last.pth"] and texts(f"Get-ChildItem '{prefix}.") == [prefix + ".hidden"]
    assert texts(f"Register-Artifact '{prefix}nothing") == []
    kinds = {item.text: item.kind for item in completion.complete(f"Get-ChildItem '{prefix}")}
    assert kinds == {prefix + "checkpoints" + os.sep: "Directory", prefix + "last.pth": "File"}


def test_windows_paths_survive_parsing_binding_and_completion(tmp_path):
    """Backslash is a path separator, never an escape; drive letters are not key kinds."""
    assert completion.split_path(r"C:\Users\Yasin\saved_models\ch") == ("C:\\Users\\Yasin\\saved_models\\", "ch")
    assert completion.split_path(r".\saved_models\last") == (".\\saved_models\\", "last")
    assert completion.split_path("saved_models/last") == ("saved_models/", "last") and completion.split_path("last") == ("", "last")
    directory, name = completion.split_path(r"D:\modeller\türkçe ağırlık.pth")
    assert ntpath.split(r"D:\modeller\türkçe ağırlık.pth") == (directory.rstrip("\\"), name)
    code, out, err = run(tmp_path, r"Register-Artifact .\saved_models\checkpoints, 'C:\Program Files\Cevahir\last.pth' -Recurse -WhatIf")
    assert (code, err) == (0, "") and out == ('What if: Register-Artifact ".\\saved_models\\checkpoints (kind auto, recurse)"\n'
                                              'What if: Register-Artifact "C:\\Program Files\\Cevahir\\last.pth (kind auto, recurse)"\n')
    assert run(tmp_path, r'Write-Output "C:\temp\new\table.pth"')[1] == "C:\\temp\\new\\table.pth\n", "no \\t or \\n escapes"
    assert run(tmp_path, r"$dir = 'D:\modeller'; Write-Output $dir\son.pth, '.\göreli yol\a.pth'")[1] == \
        "D:\\modeller\\son.pth\n.\\göreli yol\\a.pth\n"
    code, out, err = run(tmp_path, r"Get-Key C:\models\last.pth")
    assert code == 1 and "no such file: C:\\models\\last.pth" in err, "a drive letter is never read as a key kind"
    code, out, err = run(tmp_path, r"Test-Checkpoint -Path D:\yok\last.pth")
    assert code == 1 and "No such file or directory: D:\\yok\\last.pth" in err
    assert Record("Cevahir.FileItem", FullName=r"C:\x\a.pth").get("fullname") == r"C:\x\a.pth"
    assert format_table(CHECKPOINTS[:1], width=200).rstrip().endswith("checkpoint_epoch_0005.pth"), \
        "the File column is the last component of a Windows path, also when rendered on Linux"


def test_get_child_item_and_test_path(tmp_path):
    folder = tmp_path / "models"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.pth").write_bytes(b"12345")
    (folder / "sub" / "b.pth").write_bytes(b"1")
    (folder / "notes.txt").write_bytes(b"")
    assert run(tmp_path, f"Get-ChildItem '{folder}' | ForEach-Object Name")[1].split() == ["a.pth", "notes.txt", "sub"]
    assert run(tmp_path, f"ls '{folder}' -Recurse -Filter *.pth | ForEach-Object Name")[1].split() == ["a.pth", "b.pth"]
    assert run(tmp_path, f"(Get-ChildItem '{folder / 'a.pth'}').Length")[1] == "5\n"
    assert run(tmp_path, f"dir '{folder}' -File | Measure-Object | ForEach-Object Count")[1] == "2\n"
    assert run(tmp_path, f"Get-ChildItem '{folder}{os.sep}*.pth' | ForEach-Object FullName")[1] == str(folder / "a.pth") + "\n"
    assert run(tmp_path, f"Test-Path '{folder}'; Test-Path '{folder / 'nope'}'")[1] == "True\nFalse\n"
    code, out, err = run(tmp_path, f"Get-ChildItem '{folder}' -Filter *.pth | Test-Checkpoint -NoRecord | ForEach-Object C12")
    assert out == "unsupported\n" and code == 5, "FullName binds to -Path: a file that is no checkpoint fails the check"
    assert run(tmp_path, f"Get-ChildItem '{folder / 'nope'}'")[0] == 1
    code, out, err = run(tmp_path, f"inspect '{folder / 'a.pth'}' -NoRecord | ForEach-Object Registered")
    assert (code, out, err) == (5, "False\n", ""), "a file where a key is expected is tested as a file"


# ------------------------------------------------------------------------------------ entry point
def test_entry_point_version_usage_and_exit_codes(tmp_path):
    home = tmp_path / "home"
    assert console("--version", home=home) == (0, "0.4.0\n", "") and console("-Version", home=home)[1] == "0.4.0\n"
    code, out, err = console("-Help", home=home)
    assert code == 0 and "-Command" in out and "-File" in out and "-OutputFormat" in out
    code, out, err = console("-Bogus", home=home)
    assert code == 2 and "unknown argument: -Bogus" in err and out == ""
    assert console("-Command", home=home)[0] == 2 and console("-Width", "x", "-Command", "1", home=home)[0] == 2
    assert console("-Command", "1 + 1", home=home) == (0, "2\n", "")
    assert console("-c", "'a'; 'b' | ForEach-Object { $_.ToUpper() }", home=home) == (0, "a\nB\n", "")
    assert console("-Command", "Get-Nothing", home=home)[0] == 1
    assert console("-Command", "function f {}", home=home)[0] == 2
    assert console("-Command", "Get-KeyScheme -Nope", home=home)[0] == 2
    assert console("-Command", "throw 'dur'", home=home)[0] == 1 and console("-Command", "exit 7", home=home)[0] == 7
    assert console("keys", "schemes", home=home)[0] == 0, "the key CLI is still reachable"
    assert not home.exists(), "none of this wrote to the home"


def test_console_starts_fast_and_without_torch(tmp_path):
    code = r"""
import json, sys, time
start = time.perf_counter()
import cevahir_ai.console.main as main
status = main.main(["-Command", "Get-Artifact | Out-Null; Get-Checkpoint | Test-Checkpoint | Out-Null; Get-AuditEvent | Out-Null; "
                    "Test-AuditLog | Out-Null; Get-Environment | Out-Null; Get-Command | Out-Null; Get-Help Get-Model | Out-Null"])
elapsed = time.perf_counter() - start
import cevahir_ai.store, cevahir_ai.ops
for name in ("core", "catalog", "locations", "audit", "redact", "lineage", "register", "bundles"):
    getattr(cevahir_ai.store, name)
for name in ("governed", "artifacts", "checkpoints", "audit", "keys", "env"):
    getattr(cevahir_ai.ops, name)
import cevahir_ai.console.repl, cevahir_ai.console.completion, cevahir_ai.console.lineedit_win, cevahir_ai.console.tools.gen_powershell
import logging
print(json.dumps({"status": status, "elapsed": elapsed, "modules": sorted(sys.modules), "handlers": len(logging.getLogger().handlers)}))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CEVAHIR_HOME": str(tmp_path / "home")}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    loaded = {name.split(".")[0] for name in report["modules"]}
    assert report["status"] == 0 and not loaded & {"torch", "numpy"}, "the console imported torch or numpy"
    assert not loaded & set(LEGACY), f"the console imported a legacy package: {sorted(loaded & set(LEGACY))}"
    assert report["handlers"] == 0, "the console configured logging"
    assert report["elapsed"] < 1.0, f"console start and seven commands took {report['elapsed']:.2f}s"


SCRIPT = ("\ufeffparam([string]$Name = 'dünya', [int]$Count = 2, [switch]$Loud)\r\n"
          "# açıklama satırı\r\n"
          "$greeting = \"merhaba $Name\"\r\n"
          "if ($Loud) { $greeting.ToUpper() } else { $greeting }\r\n"
          "foreach ($i in 1, 2, 3) {\r\n    if ($i -le $Count) { \"satır $i\" }\r\n}\r\n"
          "Get-KeyScheme -Kind tok |\r\n    Select-Object -First 1 |\r\n    ForEach-Object Name\r\n"
          "Write-Output `\r\n    'devam'\r\n"
          "<# blok\r\n   yorum #>\r\n"
          "$h = @{\r\n    Ad = 'x'\r\n    Boy = 3\r\n}\r\n$h.Boy\r\n")


def test_script_files_with_crlf_bom_and_parameters(tmp_path):
    script = tmp_path / "çalıştır.ps1"
    script.write_bytes(SCRIPT.encode("utf-8"))
    assert b"\r\n" in script.read_bytes() and script.read_bytes().startswith(b"\xef\xbb\xbf")
    home = tmp_path / "home"
    expected = "merhaba dünya\nsatır 1\nsatır 2\ntok:bpe-sem-v1\ndevam\n3\n"
    assert console("-File", str(script), home=home) == (0, expected, "")
    assert console("-File", str(script), "-Name", "Yasin", "-Loud", "-Count", "1", home=home) == \
        (0, "MERHABA YASIN\nsatır 1\ntok:bpe-sem-v1\ndevam\n3\n", "")
    assert console("-File", str(script), "Ali", "3", home=home)[1].startswith("merhaba Ali\nsatır 1\nsatır 2\nsatır 3\n"), "positional"
    code, out, err = console("-File", str(script), "-Count", "çok", home=home)
    assert code == 2 and "-Count needs an integer" in err
    assert console("-File", str(script), "-Nope", "1", home=home)[0] == 2
    assert console("-File", str(tmp_path / "yok.ps1"), home=home)[0] == 1
    lf = tmp_path / "lf.ps1"
    lf.write_bytes(SCRIPT.replace("\r\n", "\n").lstrip("\ufeff").encode("utf-8"))
    assert console("-File", str(lf), home=home) == (0, expected, ""), "CRLF and LF scripts mean the same"
    # A script stops at its first error (PowerShell's Stop), a -Command line does not.
    failing = tmp_path / "failing.ps1"
    failing.write_bytes(b"'before'\r\nGet-Nothing\r\n'after'\r\n")
    assert console("-File", str(failing), home=home)[:2] == (1, "before\n")
    assert console("-Command", "'before'; Get-Nothing; 'after'", home=home)[:2] == (1, "before\nafter\n")
    mandatory = tmp_path / "mandatory.ps1"
    mandatory.write_bytes(b"param([Parameter(Mandatory)][string]$Model)\r\n\"model: $Model\"\r\n")
    assert console("-File", str(mandatory), home=home)[0] == 2 and console("-File", str(mandatory), "-Model", "m", home=home)[1] == "model: m\n"
    latin = tmp_path / "latin.ps1"
    latin.write_bytes("'ğ'".encode("cp1254"))
    code, out, err = console("-File", str(latin), home=home)
    assert code == 2 and "is not UTF-8" in err
    # A script calls another script next to it, with arguments; the child has its own variables.
    (tmp_path / "child.ps1").write_bytes(b"param([string]$Who)\r\n$inner = 'set in child'\r\n\"child: $Who\"\r\n")
    (tmp_path / "parent.ps1").write_bytes(b".\\child.ps1 -Who 'parent'\r\n\"inner is [$inner]\"\r\n")
    assert console("-File", str(tmp_path / "parent.ps1"), home=home) == (0, "child: parent\ninner is []\n", "")


def test_standard_input_json_output_and_the_invoke_protocol(tmp_path):
    home = tmp_path / "home"
    assert console(home=home, stdin="$x = 2\n$x * 21\n") == (0, "42\n", ""), "a script on standard input"
    code, out, err = console("-OutputFormat", "Json", "-Command", "Get-KeyScheme -Kind arch | Select-Object Name, Status; 'metin'", home=home)
    assert code == 0 and json.loads(out) == [{"__type": "Selected.Cevahir.KeyScheme", "Name": "arch:cfg-v1", "Status": "primary"},
                                             {"__type": "Selected.Cevahir.KeyScheme", "Name": "arch:ctor-v0", "Status": "legacy"},
                                             "metin"]
    code, out, err = console("-Invoke", home=home, stdin=json.dumps({"command": "Get-KeyScheme", "parameters": {"Kind": "tok", "Status": "LEGACY"}}))
    answer = json.loads(out)
    assert code == 0 and answer["exit"] == 0 and answer["errors"] == [] and err == ""
    assert [record["Name"] for record in answer["records"]] == ["tok:legacy-v3"] and answer["records"][0]["__type"] == "Cevahir.KeyScheme"
    # Parameter values are data: nothing in them is parsed as a script.
    hostile = "x'; Out-File owned.txt; 'y"
    code, out, err = console("-Invoke", home=home, cwd=tmp_path,
                             stdin=json.dumps({"command": "Get-CvArtifact", "parameters": {"Key": hostile}}))
    answer = json.loads(out)
    assert code == 1 and answer["exit"] == 1 and answer["records"] == [] and hostile in answer["errors"][0]["message"]
    assert answer["errors"][0]["category"] == "ObjectNotFound" and not (tmp_path / "owned.txt").exists()
    assert answer["errors"][0]["fully_qualified_error_id"] == "NotFound,Cevahir.Commands.GetArtifact"
    for request, message in (({"command": "Where-Object"}, "is not a Cevahir cmdlet"), ({"command": "Get-Nothing"}, "is not a Cevahir cmdlet"),
                             ({"command": "Get-KeyScheme", "parameters": {"Nope": 1}}, "no parameter -Nope"),
                             ({"command": "Get-KeyScheme", "parameters": []}, "parameters must be an object"), ([], "request must be")):
        code, out, err = console("-Invoke", home=home, stdin=json.dumps(request))
        assert code != 0 and message in json.loads(out)["errors"][0]["message"], request
    assert console("-Invoke", home=home, stdin="not json")[0] == 2
    # Pipeline input travels as records: property binding works as in the console.
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "a.pth").write_bytes(b"not a checkpoint")
    request = {"command": "Test-Checkpoint", "parameters": {"NoRecord": True},
               "input": [{"__type": "Cevahir.FileItem", "FullName": str(folder / "a.pth")}]}
    code, out, err = console("-Invoke", home=home, stdin=json.dumps(request))
    answer = json.loads(out)
    assert code == 5 and answer["exit"] == 5 and answer["records"][0]["C12"] == "unsupported" and answer["records"][0]["Ok"] is False


def test_profile_runs_at_start_unless_asked_not_to(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "profile.ps1").write_bytes(b"$fromProfile = 'profil'\r\n")
    assert console("-Command", "\"[$fromProfile]\"", home=home) == (0, "[profil]\n", "")
    assert console("-NoProfile", "-Command", "\"[$fromProfile]\"", home=home) == (0, "[]\n", "")
    assert console("-Command", "Get-Nothing; 'still running'", home=home)[1] == "still running\n", "the profile does not leave Stop on"
    assert console("-Home", str(tmp_path / "other"), "-Command", "$CevahirHome", home=home)[1] == str((tmp_path / "other").resolve()) + "\n"


def test_home_inside_the_checkout_is_refused(tmp_path):
    code, out, err = console("-Command", "Get-Artifact", home=ROOT / "state")
    assert code == 1 and "must be outside the checkout" in err and not (ROOT / "state").exists()


# ------------------------------------------------------------------------------- prompt and editor
def test_repl_reads_multi_line_input_and_keeps_going_after_errors(tmp_path):
    lines = iter(["Get-KeyScheme -Kind tok |", "  ForEach-Object Name", "", "$x = 3", "if ($x -gt 1) {", "'big'", "}",
                  "bogus-cmd", "Get-KeyScheme -Kind arch | Where-Object {", "$_.Status -eq 'legacy' } | ForEach-Object Name",
                  "exit 4", "never read"])
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        return next(lines)

    session = new_session(tmp_path / "home")
    saved = []
    assert repl.run(session, read=read, save=lambda: saved.append(True)) == 4
    out = session.stdout.getvalue()
    assert out.startswith(f"Cevahir AI 0.4.0\n  home    {tmp_path / 'home'}\n  host    (none yet)\n  policy  builtin:bootstrap-v0")
    assert out.endswith("tok:bpe-sem-v1\ntok:bytebpe-v1\ntok:legacy-v3\nbig\narch:ctor-v0\n")
    assert "CommandNotFoundException" in session.stderr.getvalue(), "an error does not end the session"
    assert prompts == ["PS cevahir> ", ">> ", "PS cevahir> ", "PS cevahir> ", "PS cevahir> ", ">> ", ">> ", "PS cevahir> ",
                       "PS cevahir> ", ">> ", "PS cevahir> "]
    assert saved == [True]
    turkish = new_session(tmp_path / "home", culture="tr")
    assert repl.run(turkish, read=lambda prompt: "exit", save=None) == 0 and "politika" in turkish.stdout.getvalue()

    def interrupted(prompt, state=[0]):
        state[0] += 1
        if state[0] == 1:
            raise KeyboardInterrupt
        if state[0] == 2:
            return "'after ctrl-c'"
        raise EOFError

    broken = new_session(tmp_path / "home")
    broken.run = lambda text: (_ for _ in ()).throw(TypeError("a defect"))
    feed = iter(["Get-Artifact", "exit"])
    assert repl.run(broken, read=lambda prompt: next(feed), save=None, show_banner=False) == 1
    assert "internal error: TypeError: a defect" in broken.stderr.getvalue(), "a defect is reported and the prompt returns"
    quiet = new_session(tmp_path / "home")
    assert repl.run(quiet, read=interrupted, save=None, show_banner=False) == 0
    assert quiet.stdout.getvalue() == "\nafter ctrl-c\n\n", "Ctrl+C cancels the line, end of input leaves"


def test_readline_history_and_completion_where_readline_exists(tmp_path):
    readline = pytest.importorskip("readline")
    session = new_session(tmp_path / "home")
    reader, save = repl._readline_reader(session)
    assert reader is input and readline.get_completer() is not None
    assert "|" in readline.get_completer_delims() and " " in readline.get_completer_delims()
    readline.add_history("Get-Artifact")
    save()
    assert not (tmp_path / "home").exists(), "history is not what creates the home"
    (tmp_path / "home").mkdir()
    save()
    assert "Get-Artifact" in (tmp_path / "home" / "console_history").read_text(encoding="utf-8")
    plain = Session(home=str(tmp_path / "home"), environ={"CEVAHIR_LINE_EDITOR": "plain"}, stdout=io.StringIO(), stderr=io.StringIO())
    assert repl.make_reader(plain, stdin=io.StringIO("x"))[0] is input


def keys(text):
    """Scripted keystrokes: ``<up>`` style names stand for the two-character codes msvcrt returns."""
    special = {"<up>": "\xe0H", "<down>": "\xe0P", "<left>": "\xe0K", "<right>": "\x00M", "<home>": "\xe0G", "<end>": "\xe0O",
               "<del>": "\xe0S", "<bs>": "\x08", "<tab>": "\t", "<enter>": "\r", "<esc>": "\x1b", "<ctrl-c>": "\x03", "<ctrl-z>": "\x1a"}
    for name, code in special.items():
        text = text.replace(name, code)
    return iter(text)


def edit(text, history=None, complete=None):
    written = []
    source = keys(text)
    editor = LineEditor(lambda: next(source), written.append, complete, history)
    return editor.read_line("PS> "), "".join(written), editor


def test_windows_line_editor_with_scripted_keys():
    assert edit("Get-Artifact<enter>")[0] == "Get-Artifact"
    assert edit("Get-Artfact<left><left><left><left>i<enter>")[0] == "Get-Artifact", "insert in the middle"
    assert edit("xGet-Model<home><del><end> -Key k<enter>")[0] == "Get-Model -Key k"
    assert edit("Get-Modell<bs><enter>")[0] == "Get-Model" and edit("typo<esc>Get-Cache<enter>")[0] == "Get-Cache"
    assert edit("şğüİ ö<enter>")[0] == "şğüİ ö", "Turkish letters are ordinary keys"
    history = ["Get-Artifact", "Get-Model"]
    line, _, editor = edit("<up><up><down><enter>", history)
    assert line == "Get-Model" and editor.history == ["Get-Artifact", "Get-Model"], "a repeated line is stored once"
    assert edit("draft<up><down> more<enter>", ["old"])[0] == "draft more", "the line being typed comes back"
    assert edit("new line<enter>", history)[2].history[-1] == "new line"

    def complete(line, cursor):
        return ["Get-Checkpoint", "Get-ChildItem"] if line[:cursor].endswith("Get-Ch") else []

    assert edit("Get-Ch<tab><enter>", complete=complete)[0] == "Get-Checkpoint"
    assert edit("Get-Ch<tab><tab><enter>", complete=complete)[0] == "Get-ChildItem", "Tab cycles, as in PowerShell"
    assert edit("Get-Ch<tab><tab><tab> -Latest<enter>", complete=complete)[0] == "Get-Checkpoint -Latest"
    assert edit("x | Get-Ch<tab><enter>", complete=complete)[0] == "x | Get-Checkpoint"
    assert edit("nothing<tab><enter>", complete=complete)[0] == "nothing"
    with pytest.raises(KeyboardInterrupt):
        edit("half typed<ctrl-c>")
    with pytest.raises(EOFError):
        edit("<ctrl-z>")
    line, written, _ = edit("ab<bs><enter>")
    assert written.startswith("PS> ") and written.endswith("\r\n") and "\x1b" not in written, "no escape sequences: plain consoles work"


# --------------------------------------------------------------------------- the PowerShell module
def test_powershell_module_is_in_step_with_the_registry():
    assert gen_powershell.stale() == [], "run: python -m cevahir_ai.console.tools.gen_powershell --write"
    directory = ROOT / "powershell" / "Cevahir"
    manifest, module = ((directory / name).read_bytes() for name in ("Cevahir.psd1", "Cevahir.psm1"))
    for data in (manifest, module):
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b""), "CRLF only: the committed bytes are the generated bytes"
        data.decode("utf-8")
    text, declared = module.decode("utf-8"), manifest.decode("utf-8")
    wrapped = [command for command in registry.commands() if command.module]
    assert len(wrapped) >= 16
    for command in wrapped:
        name = gen_powershell.cv_name(command)
        assert f"function {name} {{" in text and f"'{name}'" in declared and f"-Command '{command.name}'" in text
        for parameter in command.params:
            assert f"${parameter.name}" in text
    assert "function Where-CvObject" not in text, "shell utilities are PowerShell's own there"
    assert "ModuleVersion        = '0.4.0'" in declared and "PowerShellVersion    = '7.2'" in declared
    assert "[ValidateSet('File', 'Weights', 'Compat')][string]$Level" in text
    assert "[Parameter(Position = 0, ValueFromPipeline, ValueFromPipelineByPropertyName)][Alias('Id')][string]$Key" in text
    assert "[Parameter(ValueFromPipelineByPropertyName)][Alias('FullName')][string]$Path" in text
    for forbidden in ("Invoke-Expression", "iex ", "[scriptblock]::Create", "Start-Process", "cmd /c"):
        assert forbidden not in text, f"the module must not contain {forbidden}"
    assert "'-Invoke'" in text and "ConvertTo-Json -Depth 32 -Compress" in text, "requests travel as JSON on standard input"
    assert "/powershell/** -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert gen_powershell.main(["--check"]) == 0 and gen_powershell.main([]) == 2


# -------------------------------------------------------------------------------- end to end session
@pytest.fixture(scope="module")
def saved_models(tmp_path_factory):
    """A folder as training leaves it, written by the real writers (see test_cevahir_ai_registry.py)."""
    torch = pytest.importorskip("torch")
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager
    from tokenizer_management.byte_bpe import ByteLevelBPE
    from training_management.v2.utils.checkpoint_manager import CheckpointManager
    folder = tmp_path_factory.mktemp("saved_models")
    entry = json.loads((ROOT / "governance/registry/tokenizers.json").read_text(encoding="utf-8"))["tokenizers"][0]
    tracked = next(a["key"].rsplit(":", 1)[1] for a in entry["aliases"] if a["how"] == "read_only=absent, merges=LF")
    saver = CheckpointManager(checkpoint_model_dir=str(folder / "checkpoints"))
    for epoch, metric in ((1, 2.9), (2, 2.4), (3, 2.6)):
        torch.manual_seed(epoch)
        trained = ModelManager(tiny_model_config())
        trained.initialize(build_optimizer=True, build_criterion=False, build_scheduler=True)
        trained.model._tokenizer_identity = tracked
        extra = {"format_version": 1, "tokenizer_identity": tracked, "loop": {"global_step": epoch}, "best_val_loss": metric,
                 "scheduler": {"scheduler_type": "ReduceLROnPlateau", "scheduler_state": trained.scheduler.state_dict()},
                 "rng": {"torch": torch.get_rng_state()}}
        saver.save(trained.model, trained.optimizer, epoch, metric=metric, extra_state=extra)
    for item in (str(ROOT), str(STUDY)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from cev11 import model as study
    tokenizer = ByteLevelBPE.train(["merhaba dünya merhaba", "dünya güzel bir yer"] * 20, vocab_size=290)
    cev_model, cfg = study.build_model(vocab_size=tokenizer.vocab_size, layers=2, d=32, heads=4, kv_heads=4, max_len=64, seed=0)
    study.save_checkpoint(str(folder / "cev11" / "model.pt"), cev_model, cfg, tokenizer, {"note": "test"})
    return folder


def test_end_to_end_session(saved_models, tmp_path):
    import shutil
    folder = tmp_path / "saved_models"
    shutil.copytree(saved_models, folder)
    session = new_session(tmp_path / "home", width=130)

    def step(text):
        session.stdout.seek(0), session.stdout.truncate(0), session.stderr.seek(0), session.stderr.truncate(0)
        session.codes.clear()
        code = session.run(text)
        return code, session.stdout.getvalue(), session.stderr.getvalue()

    # 1. Register the folder.
    code, out, err = step(f"$registered = Register-Artifact '{folder}' -Recurse; $registered | Group-Object Kind | Format-Table Count, Name")
    assert (code, err) == (0, "") and out == "\nCount Name\n----- ----\n    4 ckpt\n    4 model\n    1 tok\n"
    assert step("($registered | Where-Object New).Count")[1] == "9\n"
    # 2. Get-Checkpoint: epochs, metric, tokenizer; best.pth and last.pth are locations of epoch 2 and 3.
    code, out, err = step("Get-Checkpoint | Where-Object Epoch | Sort-Object Epoch | Format-Table Epoch, Metric, Envelope, State, "
                          "@{Name='Copies'; Expression={ $_.Locations.Count }}")
    assert out == ("\nEpoch Metric Envelope State      Copies\n----- ------ -------- -----      ------\n"
                   "    1    2.9 tm2      registered      1\n    2    2.4 tm2      registered      2\n"
                   "    3    2.6 tm2      registered      2\n")
    assert step("(Get-Checkpoint -Best).Epoch; (Get-Checkpoint -Latest).Envelope; (Get-Checkpoint | Measure-Object).Count")[1] == \
        "2\ncev11-v1\n4\n", "best: lowest metric; latest: the file written last"
    # 3. Test-Checkpoint: file level for all (no torch needed), then the full gate for the best one.
    code, out, err = step("Get-Checkpoint | Test-Checkpoint | Sort-Object Short | Format-Table Verdict, C1, C12")
    assert code == 0 and out.count("compatible pass pass") == 4
    code, out, err = step("$best = Get-Checkpoint -Best; $verdict = $best | Test-Checkpoint -Level Compat -Purpose serve; "
                          "$verdict | Format-List Verdict, Ok, Failing, Checks")
    assert code == 0 and out == ("\nVerdict : compatible\nOk      : True\nFailing : {}\n"
                                 "Checks  : {C12 pass, C1 pass, C3 pass, C4 pass, C5 pass, C7 pass, C9 pass, C11 pass, C10 pass}\n")
    code, out, err = step("$best | Test-Checkpoint -Level Compat -Purpose evaluate -Target cev11-primary | Format-Table -View Checks")
    assert code == 5 and err == "", "an incompatible target is a verdict and exit code 5, not an error record"
    assert "C5  arch.declared     fail   True     architecture differs from the target" in out
    assert step("(Get-Checkpoint -Best).State")[1] == "verified\n"
    # 4. Get-Model and Get-Lineage: from the model back to its checkpoint and tokenizer.
    assert step("(Get-Model | Measure-Object).Count; (Get-Tokenizer).Family")[1] == "4\ncevahir-byte-bpe\n"
    code, out, err = step("$cev = Get-Checkpoint | Where-Object Envelope -eq 'cev11-v1'; Get-Lineage $cev.Model -Direction Up | "
                          "Format-Table -View Tree")
    tree = [line.split("  ")[0:3] for line in out.splitlines()[3:]]
    assert len(tree) == 2 and out.splitlines()[3].startswith("<- ckpt:") and "[checkpoint]" in out.splitlines()[3]
    assert out.splitlines()[4].startswith("  <- tok:") and "[tokenizer]" in out.splitlines()[4]
    assert step("(Get-Tokenizer | Get-Lineage -Direction Down | ForEach-Object Role) -contains 'checkpoint'")[1] == "True\n"
    assert step("(Get-Lineage $best.Key -Direction Both | Measure-Object).Count")[1] == "2\n"
    # 5. Compare two epochs: same architecture and tokenizer, other weights.
    code, out, err = step("$one = Get-Checkpoint | Where-Object Epoch -eq 1; Compare-Key $one.Key $best.Short | "
                          "Where-Object Equal -eq $false | ForEach-Object Member")
    assert out.split() == ["key", "model", "weights", "tstate"], "same architecture, tokenizer and layout; other weights"
    assert step("(Get-Key $best.Path | Where-Object Name -eq ckpt).Key -eq $best.Key")[1] == "True\n"
    # 6. Get-AuditEvent: what was recorded, by whom, with which decision.
    code, out, err = step("Get-AuditEvent | Group-Object Event | Sort-Object Name | Format-Table Count, Name")
    assert out == ("\nCount Name\n----- ----\n    2 artifact.located\n    9 artifact.registered\n    6 artifact.verified\n"
                   "    1 log.genesis\n")
    code, out, err = step("Get-AuditEvent -Event artifact.registered -Last 1 | Format-List Event, Action, Class, Decision, Policy, Outcome")
    assert out == ("\nEvent    : artifact.registered\nAction   : Register-Artifact\nClass    : mutate\nDecision : allow\n"
                   "Policy   : builtin:bootstrap-v0\nOutcome  : ok\n")
    assert step("($best | Get-AuditEvent | ForEach-Object Event) -join ','")[0] == 2, "-join is not in the language"
    assert step("$best | Get-AuditEvent | ForEach-Object Event")[1].split() == ["artifact.registered", "artifact.located",
                                                                              "artifact.verified", "artifact.verified", "artifact.verified"]
    assert step("(Get-AuditEvent -Action Test-* -Since 1h | Measure-Object).Count")[1] == "6\n"
    # 7. Test-AuditLog: the chain verifies.
    code, out, err = step("Test-AuditLog | Format-List Own, Ok, Events, Mac, FirstBadSeq, Findings")
    assert code == 0 and out == "\nOwn         : True\nOk          : True\nEvents      : 18\nMac         : verified\nFirstBadSeq : \nFindings    : {}\n"
    # 8. Export the best checkpoint and import it into another home.
    bundle = tmp_path / "en-iyi.cvb"
    code, out, err = step(f"$best | Export-Artifact -To '{bundle}' -IncludePayload | Format-List Manifests, Payloads")
    assert code == 0 and out == "\nManifests : 1\nPayloads  : 1\n" and bundle.is_file()
    other = new_session(tmp_path / "home2", width=130)
    assert other.run(f"Import-Artifact '{bundle}' | Test-Checkpoint -Level Weights | Format-Table State, Verdict, C1, C3") == 0
    assert "compatible pass pass" in other.stdout.getvalue() and other.stderr.getvalue() == ""
    assert step(f"Import-Artifact '{bundle}' -WhatIf")[1].startswith("What if: Import-Artifact")
    # 9. A checkpoint changed after registration is reported: C1 fails, exit code 5, state quarantined.
    victim = step("(Get-Checkpoint | Where-Object Epoch -eq 1).Path")[1].strip()
    with open(victim, "r+b") as stream:
        stream.seek(os.path.getsize(victim) // 2)
        chunk = stream.read(4)
        stream.seek(-4, os.SEEK_CUR)
        stream.write(bytes(byte ^ 0xFF for byte in chunk))
    code, out, err = step("Get-Checkpoint | Test-Checkpoint | Where-Object Ok -eq $false | Format-List Verdict, C1, Intact, Summary")
    assert code == 5 and err == "" and "Verdict : incompatible\nC1      : fail\nIntact  : no\n" in out
    assert "file bytes differ from the expected key (tampered, truncated or another file)" in out
    assert step("Get-Artifact -State quarantined | ForEach-Object Kind")[1] == "ckpt\n"
    # 10. An edited audit log is reported with the sequence number, and nothing more is appended to hide it.
    log = next((tmp_path / "home" / "audit").glob("h-*/seg-000001.jsonl"))
    data = log.read_bytes()
    log.write_bytes(data.replace(b'"verdict":"compatible"', b'"verdict":"incompatibl3"', 1))
    code, out, err = step("Test-AuditLog | Select-Object -ExpandProperty Findings | Format-Table Seq, Line, Problem")
    assert code == 5 and "hash mismatch: the event was edited" in out
    assert step("(Test-AuditLog).Ok; (Test-AuditLog).FirstBadSeq -gt 1")[1] == "False\nTrue\n"
    lines = log.read_bytes().split(b"\n")
    log.write_bytes(b"\n".join(lines[:-3]) + b"\n")          # two events cut from the end
    code, out, err = step("Test-AuditLog | Select-Object -ExpandProperty Findings | ForEach-Object Problem")
    assert code == 5 and "events were removed from the end" in out
    (folder / "new.bin").write_bytes(b"something new to record")
    code, out, err = step(f"Register-Artifact '{folder / 'new.bin'}' -Kind file")
    assert code == 5 and "[SecurityError]" in err and "events were removed from the end" in err, "a cut log is not extended"


# ------------------------------------------------------------------- a real CEV-11 base checkpoint
def _real_bases():
    """CEV-11 base checkpoints are local artefacts, not in git (same search as test_cevahir_ai_gate.py). Read-only."""
    candidates = [os.environ.get("CEVAHIR_CEV11_BASES"), STUDY / "results" / "bases"]
    dot_git = ROOT / ".git"
    if dot_git.is_file():
        gitdir = dot_git.read_text(encoding="utf-8").partition("gitdir:")[2].strip().replace("\\", "/")
        if "/.git/" in gitdir:
            candidates.append(Path(gitdir.split("/.git/")[0]) / STUDY.relative_to(ROOT) / "results" / "bases")
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            found = sorted(Path(candidate).glob("*/model.pt"))
            if found:
                return found
    return []


REAL = _real_bases()


@pytest.mark.skipif(not REAL, reason="CEV-11 base checkpoints are local artefacts (not in git)")
def test_real_cev11_base_is_registered_and_inspected_without_touching_it(tmp_path):
    pytest.importorskip("torch")
    base = REAL[0].parent
    before = {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
              for path in (base / "model.pt", base / "model.tokenizer.json")}
    listing = sorted(path.name for path in base.iterdir())
    session = new_session(tmp_path / "home", width=140)
    assert session.run(f"Register-Artifact '{base / 'model.pt'}', '{base / 'model.tokenizer.json'}' | Sort-Object Kind | "
                       "Format-Table Kind, State, Provenance, New") == 0, session.stderr.getvalue()
    assert session.stdout.getvalue() == ("\nKind  State      Provenance    New\n----  -----      ----------    ---\n"
                                         "ckpt  registered legacy-import True\nmodel registered legacy-import True\n"
                                         "tok   registered legacy-import True\n")
    session.stdout.truncate(0), session.stdout.seek(0)
    code = session.run("$c = Get-Checkpoint; $v = $c | Test-Checkpoint -Level Compat -Purpose serve; "
                       "$c.Envelope; $c.Parameters -gt 1000000; $v.Verdict; $v.C1; $v.C3; $v.C5; $v.C7; "
                       "(Get-Lineage $c.Model -Direction Up | ForEach-Object Role); (Test-AuditLog).Ok; (Get-Artifact $c.Key).State")
    assert code == 0, session.stderr.getvalue()
    assert session.stdout.getvalue().split() == ["cev11-v1", "True", "compatible", "pass", "pass", "pass", "pass",
                                                 "checkpoint", "tokenizer", "True", "verified"]
    document = json.loads(next((tmp_path / "home" / "objects").glob("*/*.json")).read_text(encoding="utf-8"))
    assert not any(str(base) in json.dumps(item) for item in [document]), "no path of the study enters a manifest"
    after = {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
             for path in (base / "model.pt", base / "model.tokenizer.json")}
    assert after == before and sorted(path.name for path in base.iterdir()) == listing, "the study folder is untouched"
