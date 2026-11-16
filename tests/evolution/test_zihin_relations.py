# -*- coding: utf-8 -*-
"""Zihin, F33 Relational Memory: one memory, recalled in the company of the person present.

What is pinned here
-------------------
* The ledger is exact: watched once with "ayşe" asking ("kedimin adı ne ?" -> "kedinin adı pamuk ."), every piece of
  evidence on what followed the question is noted with her. Read with no one present, nothing of it remains (the
  whole is hers); read with her, all of it does (her own evidence nested over nothing is the evidence itself).
* What was lived with one person is recalled with that person only: her secret is told to her, and to no one else,
  known or stranger (they are left silent: that the question went on otherwise is known from the whole life, so
  nothing learned elsewhere is said in its place).
* What several people lived alike is known with anyone: taught by two people, the mind's name is told to a stranger;
  taught by one, it is that person's.
* The person present comes last: a greeting lived with someone in their own way is answered their way, though the
  general habit was lived more often; a stranger gets the general habit.
* Undoing or forgetting a moment undoes its notes; a saved mind keeps its ledger and loads with no one present; the switch is off
  by default.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.iliski import persons_with
from zihin.persist import load, save
from zihin.read import own_mass


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(company: bool = True, keep_log: bool = False) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.company = keep_log, None, company
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def ask(s: Session, partner: str, text: str) -> str:
    twin = Session(s.mind.clone())
    twin.partner = partner
    return twin.reply(text)


def test_the_switch_is_off_by_default():
    assert Config().company is False and Config.changed().company is False


def test_the_ledger_is_exact():
    s = fresh()
    s.watch("kedimin adı ne ?", "kedinin adı pamuk .", asker="ayşe")
    mind = s.mind
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    question = mind.S.tuple_index[tuple(mind.unit_of(CHANNEL, w) for w in ("kedimin", "adı", "ne")) + (end,)]
    assert persons_with(mind, question) == ["ayşe"]
    kedinin = mind.unit_of(CHANNEL, "kedinin")
    whole = own_mass(mind, question, whole=True)[kedinin]
    assert whole[0] > 0.0
    mind.company = None
    assert own_mass(mind, question).get(kedinin, (0.0, 0.0)) == (0.0, 0.0)
    mind.company = "ayşe"
    assert own_mass(mind, question)[kedinin] == pytest.approx(whole, rel=1e-9)


def secret() -> Session:
    s = fresh()
    for _ in range(2):
        s.watch("sırrım ne ?", "sırrın mavi kutuda .", asker="ayşe")
        s.watch("adın ne ?", "adım cevahir .", asker="ayşe")
        s.watch("adın ne ?", "adım cevahir .", asker="mehmet")
        s.watch("su nedir ?", "su bir sıvıdır .")
    return s


def test_what_was_lived_with_one_person_is_hers():
    s = secret()
    assert ask(s, "ayşe", "sırrım ne ?") == "sırrın mavi kutuda"
    assert ask(s, "mehmet", "sırrım ne ?") == "(sessiz)"
    assert ask(s, "deniz", "sırrım ne ?") == "(sessiz)"


def test_what_several_people_lived_alike_is_known_with_anyone():
    s = secret()
    assert ask(s, "deniz", "adın ne ?") == "adım cevahir"     # ayşe and mehmet taught it alike
    assert ask(s, "deniz", "su nedir ?") == "su bir sıvıdır"   # watched of no one in particular
    one = fresh()
    for _ in range(2):
        one.watch("adın ne ?", "adım cevahir .", asker="ayşe")
    assert ask(one, "ayşe", "adın ne ?") == "adım cevahir"
    assert ask(one, "deniz", "adın ne ?") == "(sessiz)"       # one person's alone


def test_without_the_switch_the_secret_is_told_to_anyone():
    s = fresh(company=False)
    for _ in range(2):
        s.watch("sırrım ne ?", "sırrın mavi kutuda .", asker="ayşe")
    assert ask(s, "deniz", "sırrım ne ?") == "sırrın mavi kutuda"


def test_the_person_present_comes_last():
    s = fresh()
    for _ in range(3):
        s.watch("merhaba", "merhaba , hoş geldin .")
    for _ in range(2):
        s.watch("merhaba", "merhaba canım .", asker="ayşe")
    assert ask(s, "ayşe", "merhaba").startswith("merhaba canım")
    assert ask(s, "deniz", "merhaba") == "merhaba , hoş geldin"


def test_undo_takes_the_notes_back():
    s = fresh(keep_log=True)                             # undo needs the ledger of moments
    s.watch("kedimin adı ne ?", "kedinin adı pamuk .", asker="ayşe")
    mind = s.mind
    mind.company = "ayşe"
    mind.step(s.events("kedimin", 1))
    before = {tie: {p: row.copy() for p, row in rows.items()} for tie, rows in mind.personal[0].items()}
    mind.step(s.events("adı", 1))                        # what followed "kedimin" is noted with her
    assert any(mind.personal_log.get(mind.t, ()))
    mind.undo_last()
    after = mind.personal[0]
    for tie, rows in after.items():
        for p, row in rows.items():
            assert row == pytest.approx(before.get(tie, {}).get(p, 0.0 * row), abs=1e-12)


def test_a_saved_mind_keeps_its_ledger(tmp_path):
    s = secret()
    path = str(tmp_path / "zihin")
    save(s.mind, path)
    mind = load(path, ledger=False)
    assert mind.company is None
    loaded = Session(mind)
    loaded.partner = "ayşe"
    assert loaded.reply("sırrım ne ?") == "sırrın mavi kutuda"


def test_forgetting_moments_forgets_their_notes():
    s = fresh(keep_log=True)
    start = s.mind.t + 1
    s.watch("sırrım ne ?", "sırrın mavi kutuda .", asker="ayşe")
    mind = s.mind
    assert any(row.sum() > 0.0 for rows in mind.personal[0].values() for row in rows.values())
    for m in range(start, mind.t + 1):                   # as "!unut" takes an entry back
        mind.retract(m)
    assert all(abs(row).sum() < 1e-12 for rows in mind.personal[0].values() for row in rows.values())


def test_whose_word_held_is_learned_per_person():
    from zihin import SENSED, TOLD
    from zihin.iliski import person_trust
    s = fresh()
    for _ in range(3):
        s.partner = "zeynep"
        s.live("kuşlar uçar .", TOLD)                    # what turns out true
        s.partner = "mehmet"
        s.live("balıklar uçar .", TOLD)                  # what turns out false
        s.partner = chat.SOMEONE
        s.live("kuşlar uçar .", SENSED)                  # its own experience, with no one in particular
        s.live("balıklar yüzer .", SENSED)
    mind = s.mind
    zeynep, mehmet = person_trust(mind, "zeynep", None)[TOLD], person_trust(mind, "mehmet", None)[TOLD]
    assert zeynep > float(mind.trust[TOLD]) > mehmet or zeynep > mehmet > 0.0
    assert zeynep - mehmet > 0.2
    assert person_trust(mind, "deniz", None)[TOLD] == pytest.approx(float(mind.trust[TOLD]))   # no record
