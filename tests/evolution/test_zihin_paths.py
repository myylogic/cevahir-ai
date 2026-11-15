# -*- coding: utf-8 -*-
"""Zihin, the paths and encodings of what it reads (!oku, !ogret).

What is pinned here
-------------------
* ~ is expanded and one pair of surrounding quotes is removed; paths with spaces work ("notlar 2024" is a folder, not
  "notlar" with at most 2024 sentences), and a number after an existing path is still the most sentences to read.
* A path that does not exist is said so (no "0 dosya"), and nothing is lived.
* A UTF-8 file with a byte order mark is read without a symbol for the mark; a file that is not UTF-8 is read as
  cp1254 (Turkish ANSI, as Excel saves it) and the program says so. Both teach exactly what the UTF-8 file teaches.
* Every path is cleaned the same way: !kaydet, !tasinabilir, !yukle, !gor, !dinle, --load and --kayit take ~, one
  pair of quotes and spaces like !oku and !ogret; a save or load with no path asks for one.
"""
from __future__ import annotations

import pytest

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session, _one_line

ANIMALS = "Kedi evcil bir hayvandır. Kediler balık sever. Zürafa uzun boyunlu bir hayvandır."
DIALOGUES = ("Merhaba!\tMerhaba, nasılsın?\nNasılsın?\tİyiyim, teşekkür ederim. Sen nasılsın?\n"
             "Çimen ne renk?\tÇimen yeşildir.\nİstanbul nerede?\tİstanbul Türkiye'nin en büyük şehridir.\n")


class Args:
    his = False


def session() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def run(s, line: str, capsys) -> str:
    capsys.readouterr()
    _one_line(s, line, False, Args(), lambda reason: None, 0)
    return capsys.readouterr().out


def test_tilde_quotes_and_spaces_in_paths(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for folder, name in (("Belgelerim", "a.txt"), ("notlar 2024", "b.txt")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_text(ANIMALS, encoding="utf-8")
    (tmp_path / "ders.tsv").write_text(DIALOGUES, encoding="utf-8")
    s = session()
    assert run(s, "!oku ~/Belgelerim/", capsys).startswith("  1 dosya, 3 cümle")
    assert run(s, '!oku "%s"' % (tmp_path / "Belgelerim"), capsys).startswith("  1 dosya, 3 cümle")
    assert run(s, "!oku %s" % (tmp_path / "notlar 2024"), capsys).startswith("  1 dosya, 3 cümle")
    assert run(s, "!oku %s 2" % (tmp_path / "notlar 2024"), capsys).startswith("  1 dosya, 2 cümle")
    assert run(s, "!ogret ~/ders.tsv", capsys).startswith("  4 satır izlendi")
    assert run(s, "!ogret '%s'" % (tmp_path / "ders.tsv"), capsys).startswith("  4 satır izlendi")
    assert [text for _, text, _ in s.history] == ["[okudu] a.txt", "[okudu] a.txt", "[okudu] b.txt",
                                                  "[okudu] b.txt", "[öğretti] ders.tsv", "[öğretti] ders.tsv"]


def test_a_path_that_does_not_exist_is_said_so(tmp_path, capsys):
    (tmp_path / "belgelerim").mkdir()
    s = session()
    for line in ("!oku %s" % (tmp_path / "yok.txt"), "!oku %s/" % (tmp_path / "belgelerm"),
                 "!oku %s 5" % (tmp_path / "yok"), "!ogret %s" % (tmp_path / "yok.tsv")):
        out = run(s, line, capsys)
        assert "böyle bir dosya" in out and "yok" in out and "0 dosya" not in out
    assert "klasör" in run(s, "!ogret %s" % (tmp_path / "belgelerim"), capsys)
    assert s.mind.t == 0 and s.history == []


def test_every_path_is_cleaned_the_same_way(monkeypatch, tmp_path, capsys):
    """!kaydet, !tasinabilir, !yukle, !gor and !dinle take ~, one pair of quotes and spaces like !oku and !ogret."""
    import wave

    import numpy as np
    from PIL import Image
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    folder = tmp_path / "zihin kayıtları"
    folder.mkdir()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(str(folder / "kırmızı top.png"))
    rate = 16000
    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(int(0.4 * rate)) / rate) * 32767).astype("<i2")
    with wave.open(str(folder / "ses kaydı.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(tone.tobytes())
    s = session()
    s.live("Kedi balık yer.", TOLD)
    assert "kaydedildi" in run(s, '!kaydet "~/zihin kayıtları/tam"', capsys)
    assert (folder / "tam.zihin").exists() and (folder / "tam.zihin.json").exists()
    assert "taşınabilir" in run(s, "!tasinabilir '%s'" % (folder / "küçük"), capsys)
    assert (folder / "küçük.zihin").exists()
    assert "yön" in run(s, '!gor "~/zihin kayıtları/kırmızı top.png"', capsys) and s.view is not None
    assert s.seen[s.view[0]] == "kırmızı top.png"
    assert "yön" in run(s, "!dinle ~/zihin kayıtları/ses kaydı.wav", capsys) and s.hearing is not None
    age = s.mind.t
    assert "yüklendi: %d an" % age in run(s, '!yukle "~/zihin kayıtları/küçük"', capsys)
    assert "bir yol yazın" in run(s, "!kaydet  ", capsys)
    assert not (tmp_path / ".zihin").exists() and not list(tmp_path.glob("*.zihin"))


def test_load_and_kayit_take_tilde_and_quotes(monkeypatch, tmp_path, capsys):
    import io
    import re
    import zihin.__main__ as program
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "ev"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("Kedi balık yer.\n!cik\n"))
    program.main(["--yeni", "--kayit", "~/zihnim"])              # as PowerShell passes it: ~ not expanded
    assert (tmp_path / "zihnim.zihin").exists()
    monkeypatch.setattr("sys.stdin", io.StringIO("!cik\n"))
    capsys.readouterr()
    program.main(["--load", '"~/zihnim"', "--kaydetme"])        # quotes that reached the program
    lived = re.search(r"Cevahir — (\d+) an", capsys.readouterr().out)
    assert lived and int(lived.group(1)) > 0                       # the mind kept at ~/zihnim was loaded
    assert not (tmp_path / "~").exists() and not (tmp_path / '"~').exists()


@pytest.mark.parametrize("command", ["!oku", "!ogret"])
def test_a_bom_or_a_cp1254_file_teaches_what_the_utf8_file_teaches(tmp_path, capsys, command):
    (tmp_path / "utf8.tsv").write_text(DIALOGUES, encoding="utf-8")
    (tmp_path / "bom.tsv").write_text(DIALOGUES, encoding="utf-8-sig")
    (tmp_path / "ansi.tsv").write_bytes(DIALOGUES.encode("cp1254"))
    digests, outs = {}, {}
    for name in ("utf8", "bom", "ansi"):
        s = session()
        outs[name] = run(s, "%s %s" % (command, tmp_path / ("%s.tsv" % name)), capsys)
        digests[name] = s.mind.S.digest()
        cid = s.mind.ch.id(CHANNEL)
        assert "﻿" not in {s.mind.ch.symbol(cid, i) for i in range(s.mind.ch.sizes()[cid])}
        assert s.mind.unit_of(CHANNEL, "nasılsın") >= 0 and s.mind.unit_of(CHANNEL, "şehridir") >= 0
    assert digests["utf8"] == digests["bom"] == digests["ansi"]
    assert "cp1254" in outs["ansi"] and "cp1254" not in outs["utf8"] + outs["bom"]
