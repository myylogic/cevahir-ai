# -*- coding: utf-8 -*-
"""Zihin reading: the cap on how much of each file is read holds for dialogue files too.

What is pinned here
-------------------
* ``read_path(path, cap)`` reads at most ``cap`` sentences of a document, ``cap`` dialogues of a JSON file and
  ``cap`` lines of a TSV file; without a cap everything is read.
"""
from __future__ import annotations

import json

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session


def session() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def read_count(report: str) -> int:
    return int(report.split(",")[1].split()[0])


def test_the_cap_holds_for_documents_json_and_tsv(tmp_path):
    (tmp_path / "a.txt").write_text("Bir. İki. Üç. Dört. Beş.", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps([{"soru": "s%d" % i, "cevap": "c%d" % i} for i in range(6)]),
                                     encoding="utf-8")
    (tmp_path / "c.tsv").write_text("".join("s%d\tc%d\n" % (i, i) for i in range(7)), encoding="utf-8")
    assert read_count(session().read_path(str(tmp_path / "a.txt"), 2)) == 2
    assert read_count(session().read_path(str(tmp_path / "b.json"), 2)) == 2
    assert read_count(session().read_path(str(tmp_path / "c.tsv"), 2)) == 2
    assert read_count(session().read_path(str(tmp_path / "b.json"))) == 6
    assert read_count(session().read_path(str(tmp_path / "c.tsv"))) == 7


def test_source_code_is_read_line_by_line_and_its_lines_are_continued(tmp_path):
    from zihin import TOLD
    code = tmp_path / "ornek.py"
    code.write_text("def selam(isim):\n    return 'merhaba ' + isim\n\n# bir yorum satiri burada\n", encoding="utf-8")
    s = session()
    for _ in range(3):
        report = s.read_path(str(code))
    assert read_count(report) == 3                           # three lines with at least two tokens
    s.listen("def selam", TOLD)
    expected = s.mind.expected(CHANNEL)
    assert max(expected, key=expected.get) == "("
