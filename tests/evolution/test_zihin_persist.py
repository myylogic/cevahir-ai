# -*- coding: utf-8 -*-
"""Zihin persistence: a mind file as it travels to someone.

What is pinned here
-------------------
* A compressed mind loads when its compressed file is named (``x.zihin.gz``), when the base name is given (``x``)
  and when ``x.zihin`` is given.
* With its manifest beside it, a changed file is refused (the hash does not match).
* A mind file that travelled without its manifest still loads, with a warning that its integrity was not checked.
"""
from __future__ import annotations

import gzip
import os
import shutil

import pytest

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.persist import export_portable, load


def small_mind(tmp_path) -> str:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    session.watch("adın ne ?", "adım cevahir .", prov=TOLD)
    base = str(tmp_path / "giden")
    export_portable(mind, base + ".zihin")
    with open(base + ".zihin", "rb") as src, gzip.open(base + ".zihin.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(base + ".zihin")
    return base


@pytest.mark.parametrize("name", ["{}.zihin.gz", "{}", "{}.zihin"])
def test_a_compressed_mind_loads_by_any_of_its_names(tmp_path, name):
    base = small_mind(tmp_path)
    mind = load(name.format(base), ledger=False)
    assert mind.t > 0


def test_a_changed_file_is_refused_when_its_manifest_is_there(tmp_path):
    base = small_mind(tmp_path)
    load(base, ledger=False)                                # unpacks x.zihin beside x.zihin.gz
    with open(base + ".zihin", "ab") as f:
        f.write(b"\\0")
    with pytest.raises(ValueError):
        load(base, ledger=False)


def test_a_mind_without_its_manifest_loads_with_a_warning(tmp_path, capsys):
    base = small_mind(tmp_path)
    os.remove(base + ".zihin.json")
    mind = load(base + ".zihin.gz", ledger=False)
    assert mind.t > 0
    assert "doğrulanamadı" in capsys.readouterr().out


def test_the_portable_export_makes_no_copy_and_keeps_the_live_ledger(tmp_path):
    import os
    from zihin.store import DiskLog
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, log_path=str(tmp_path / "live.log")))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    session.watch("adın ne ?", "adım cevahir .", prov=TOLD)
    twin = mind.clone()
    twin.S.log, twin.S.keep_log = {}, False
    twin.cfg.keep_log, twin.cfg.log_path = False, None
    from zihin.persist import save
    copied = save(twin, str(tmp_path / "kopya"))
    direct = export_portable(mind, str(tmp_path / "dogrudan"))
    assert direct["substance_digest"] == copied["substance_digest"] and direct["log_sha256"] is None
    assert isinstance(mind.S.log, DiskLog) and mind.S.keep_log and mind.cfg.log_path
    assert not os.path.exists(str(tmp_path / "dogrudan.zihin.log"))
