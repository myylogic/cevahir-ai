# -*- coding: utf-8 -*-
"""The final form (cevher/): one substance of fixed capacity, written in the forward pass, read by the one rule.

Each test is small and needs only NumPy. docs/architecture/FINAL_FORM.md says what each one stands for.
"""
import random

import numpy as np
import pytest

from cevher import Cevahir
from cevher.cevahir import OVERHEAD_MB
from cevher.letters import Counts, Letters
from cevher.mind import Mind
from cevher.store import Exact, Slots
from cevher.units import code_of, fold, mix, mix_np, rule, tokens


def test_a_code_is_the_same_number_one_at_a_time_and_in_an_array():
    xs = [code_of("walk"), code_of("ısı"), 12345, (1 << 64) - 1]
    for a in xs:
        for b in xs:
            assert mix(a, b) == int(mix_np(np.array([a], dtype=np.uint64), np.array([b], dtype=np.uint64))[0])
    assert fold((1, 2, 3), 7) == mix(mix(mix(7, 1), 2), 3)
    assert tokens("Python ile ISI ölç!") == ["python", "ile", "ısı", "ölç", "!"]
    assert rule(1, 1) == 0.75


def _random_life(store, exact, seed=1, keys=1500, steps=30000):
    rng = random.Random(seed)
    ks = [rng.getrandbits(64) | 1 for _ in range(keys)]
    for i in range(steps):
        k, v, w = rng.choice(ks), rng.getrandbits(2) + 1, rng.choice([1, 1, 1, 2, -1])
        for s in (store, exact):
            s.add(k, v, w, i)
            if w > 0:
                s.bump(k, w)
    return ks


def test_the_bounded_table_is_the_exact_store_while_nothing_has_given_way():
    slots, exact = Slots(max_bytes=64 << 20, buckets=4), Exact()
    ks = _random_life(slots, exact)
    assert slots.let_go == 0
    assert slots.digest() == exact.digest()
    for k in ks[:300]:
        assert slots.vals(k) == exact.vals(k) and slots.tot(k) == exact.tot(k)
    again = Slots.from_state(slots.state())
    assert again.digest() == slots.digest() and again.vals(ks[0]) == slots.vals(ks[0])


def test_many_ties_at_once_are_the_same_ties_one_at_a_time():
    rng = random.Random(2)
    keys = np.repeat(np.array([rng.getrandbits(64) | 1 for _ in range(4000)], dtype=np.uint64), 4)
    vals = np.tile(np.array([0, 1, 2, 3], dtype=np.uint64), 4000)
    w = np.ones(len(keys), dtype=np.int64)
    slots, exact = Slots(max_bytes=64 << 20, buckets=16), Exact()
    assert slots.add_many(keys, vals, w, 7) is None
    exact.add_many(keys, vals, w, 7)
    assert slots.digest() == exact.digest()
    slots.sub_many(keys[:2000], vals[:2000], w[:2000])
    exact.sub_many(keys[:2000], vals[:2000], w[:2000])
    assert slots.digest() == exact.digest()


def test_a_key_holds_its_seven_most_evident_values_and_a_full_table_lets_the_least_evident_go():
    s = Slots(max_bytes=64 << 20)
    for v in range(1, 30):
        s.add(99, v, v, v)                                   # value v was met v times
        s.bump(99, v)
    assert [v for v, _, _ in s.vals(99)] == [29, 28, 27, 26, 25, 24, 23]
    assert s.tot(99) == sum(range(1, 30))                    # the total counts what was let go too
    small = Slots(max_bytes=1 << 15, buckets=4)
    for i in range(5000):
        small.add(i * 2654435761 + 1, 1, 1, i)
    assert small.nbytes() < (1 << 15) + (1 << 14) and small.let_go + small.not_kept > 0


def test_forgetting_through_a_ledger_puts_back_what_gave_way_even_in_a_full_table():
    rng = random.Random(3)
    s = Slots(max_bytes=1 << 15, buckets=4)
    for i in range(3000):
        s.add(rng.getrandbits(64) | 1, 1, 1, i)
    before, log = s.digest(), []
    for i in range(400):
        k = rng.getrandbits(64) | 1
        gone = s.add(k, 2, 1, 9999)
        log += (gone or []) + [(k, 2, 1)]
    assert s.let_go and s.digest() != before
    for record in reversed(log):
        if len(record) == 4:
            s.put_back(record)
        else:
            s.add(record[0], record[1], -record[2], 0)
    assert s.digest() == before


SCAN_LIFE = [("walk", "W"), ("run", "R"), ("look", "L"), ("zap", "Z"), ("walk twice", "W W"), ("run twice", "R R"),
             ("look twice", "L L"), ("walk thrice", "W W W"), ("run thrice", "R R R"), ("walk left", "T W"),
             ("run left", "T R"), ("look left", "T L"), ("walk and run", "W R"), ("run and look", "R L"),
             ("look and walk twice", "L W W"), ("walk left and run", "T W R"), ("walk after run", "R W"),
             ("look after walk", "W L"), ("run twice after look", "L R R"), ("look left after run", "R T L")]


def _taught(pairs, rounds=2, **how):
    c = Cevahir(**how)
    for _ in range(rounds):
        for q, a in pairs:
            c.watch(q, a + " .")
    return c


def test_a_word_lived_only_alone_is_answered_in_every_frame():
    """The add-primitive world of NOTES.md 8.3: "zap" is lived only alone."""
    c = _taught(SCAN_LIFE)
    assert c.reply("zap twice") == "Z Z ."
    assert c.reply("zap thrice") == "Z Z Z ."
    assert c.reply("zap left") == "T Z ."
    assert c.reply("zap and walk twice") == "Z W W ."
    assert c.reply("run left after zap twice") == "Z Z T R ."
    assert c.reply("zap fnord") == ""                        # a word never met: nothing is said


def test_a_frame_says_nothing_until_a_dialogue_that_did_not_make_it_confirms_it():
    c = Cevahir()
    c.watch("x yazdır", "print ( x ) .")
    assert c.reply("y yazdır") == ""                         # read from one dialogue: not yet tested
    c.watch("toplam yazdır", "print ( toplam ) .")
    assert c.reply("y yazdır") == "print ( y ) ."            # confirmed on another: it speaks, of a word never met


def test_a_dialogue_lived_is_remembered_and_the_newest_answer_holds():
    c = Cevahir()
    c.watch("adın ne", "adım cevahir")
    assert c.reply("adın ne") == "adım cevahir"
    c.watch("kedi ne yapar", "kedi miyavlar")
    c.watch("kedi ne yapar", "kedi uyur")
    assert c.reply("kedi ne yapar") == "kedi uyur"


def test_what_is_told_of_a_kind_is_said_of_a_member_never_asked():
    c = Cevahir()
    for _ in range(2):
        for m in ("serçe", "kartal", "martı"):
            c.tell("%s bir kuştur ." % m)
        for m in ("hamsi", "sazan", "levrek"):
            c.tell("%s bir balıktır ." % m)
        for m, does in (("serçe", "uçar"), ("kartal", "uçar"), ("hamsi", "yüzer"), ("sazan", "yüzer")):
            c.watch("%s ne yapar ?" % m, "%s %s ." % (m, does))
    assert c.reply("martı ne yapar ?") == "martı uçar ."
    assert c.reply("levrek ne yapar ?") == "levrek yüzer ."
    assert c.reply("masa ne yapar ?") == ""                  # nothing was told of it


def test_what_was_told_last_holds_and_a_question_about_what_was_never_told_is_met_with_silence():
    c = Cevahir()
    for name, place in (("ali", "evde"), ("ece", "okulda"), ("can", "parkta")):
        c.tell("%s %s ." % (name, place))
        c.watch("%s nerede ?" % name, "%s %s ." % (name, place))
    for place in ("evde", "evde", "evde", "limanda"):
        c.tell("veli %s ." % place)
    assert c.reply("veli nerede ?") == "veli limanda ."
    assert c.reply("zeynep nerede ?") == ""


def test_asking_changes_nothing():
    c = _taught(SCAN_LIFE)
    before = c.digest()
    c.answers(["zap twice", "walk and run", "never met", "zap left after look thrice"])
    assert c.digest() == before


def test_forgetting_an_entry_leaves_the_substance_as_if_it_had_never_been_lived():
    c = Cevahir(ledger=True)
    empty = c.digest()
    for q, a in SCAN_LIFE:
        c.watch(q, a + " .")
    c.tell("Kedi süt içer ve uyur.")
    kept, kept_moments = c.digest(), c.moments()
    c.read("Ankara Türkiye'nin başkentidir. İstanbul en kalabalık şehirdir. " * 80)      # a long text, many ties at once
    c.watch("gizli söz nedir ?", "mavi fil .")
    c.converse("zap twice")
    c.feedback(True)
    assert c.reply("gizli söz nedir ?") == "mavi fil ."
    for k in range(len(c.entries) - 1, len(SCAN_LIFE), -1):
        c.forget(k)
    assert c.digest() == kept and c.moments() == kept_moments
    assert c.reply("gizli söz nedir ?") == ""
    for k in range(len(SCAN_LIFE), -1, -1):
        c.forget(k)
    assert c.digest() == empty                               # forgotten to the last entry: nothing is left


def test_forgetting_is_exact_in_a_full_table_too():
    c = Cevahir(budget_mb=OVERHEAD_MB + 1, ledger=True)                   # 1 MB for both tables: they fill within the test
    rng = random.Random(5)
    words = ["".join(rng.choice("abcçdefgğhıijklmnoöprsştuüvyz") for _ in range(rng.randint(2, 8))) for _ in range(4000)]
    text = lambda n: " ".join(rng.choice(words) for _ in range(n)) + ". "
    c.read("".join(text(12) for _ in range(6000)))
    assert c.store.let_go + c.store.not_kept > 0 and c.letters.T.let_go > 0
    before, moments = c.digest(), c.moments()
    c.read("".join(text(12) for _ in range(600)))
    c.watch("bu ne ?", "bir deneme .")
    assert c.reply("bu ne ?") == "bir deneme ."               # what is new finds a place in a full table
    assert c.digest() != before
    c.forget(len(c.entries) - 1)
    c.forget(len(c.entries) - 2)
    assert c.digest() == before and c.moments() == moments


def test_a_long_text_is_written_a_group_of_sentences_at_a_time_and_forgotten_exactly(monkeypatch):
    import cevher.cevahir as module
    rng = random.Random(9)
    words = ["".join(rng.choice("abcçdefgğhıijklmnoöprsştuüvyz") for _ in range(rng.randint(2, 8))) for _ in range(3000)]
    text = "".join(" ".join(rng.choice(words) for _ in range(9)) + ". " for _ in range(2500))
    monkeypatch.setattr(module, "GROUP", 64 * 9)             # forty groups of this text, in a table that fills
    c = Cevahir(budget_mb=OVERHEAD_MB + 1, ledger=True)
    c.read("".join(" ".join(rng.choice(words) for _ in range(9)) + ". " for _ in range(4000)))
    assert c.store.let_go + c.store.not_kept > 0
    before, moments = c.digest(), c.moments()
    c.read(text)
    assert c.digest() != before
    c.forget(len(c.entries) - 1)
    assert c.digest() == before and c.moments() == moments
    groups = list(module._groups(text, False))
    assert len(groups) == 40 and sum(len(g) for g, _, _ in groups) == 2500
    again = [s for _, x, y in groups for g, _, _ in module._groups(text[x:y], False) for s in g]
    assert again == [s for g, _, _ in groups for s in g]     # a group read again from where it lies is the same


def test_source_code_read_by_lines_is_forgotten_exactly():
    c = Cevahir(ledger=True)
    c.read("Ankara Türkiye'nin başkentidir. İstanbul en kalabalık şehirdir. " * 80)
    before = c.digest()
    code = "\n".join("x%d = f ( %d ) . y%d + z" % (i % 50, i % 7, i % 11) for i in range(3000))
    c.read(code, code=True)                                  # each line a sentence, whatever marks stand in it
    assert c.digest() != before
    c.forget(len(c.entries) - 1)
    assert c.digest() == before


def test_a_long_file_is_given_a_piece_at_a_time():
    import io
    from cevher.__main__ import pieces
    text = "".join("satır %d burada biter.\n" % i for i in range(4000)) + "sonu olmayan satır"
    got = list(pieces(io.StringIO(text), most=5000))
    assert "".join(got) == text and len(got) > 10
    assert all(piece.endswith("\n") for piece in got[:-1]) and max(map(len, got)) < 10000
    assert list(pieces(io.StringIO("kısa"), most=5000)) == ["kısa"] and list(pieces(io.StringIO(""))) == []
    assert "".join(pieces(io.StringIO("x" * 12000), most=5000)) == "x" * 12000      # no line's end: cut where it is


def test_what_gives_way_in_a_full_table():
    """Two answers, each measured (FINAL_FORM.md). Where the mind keeps what it was told and shown, what was
    confirmed is not pushed out by what was met once. Where it keeps what texts wrote, what is new always finds a
    place: a table that kept only what is counted most could take in nothing new once it was full."""
    def full(admit_new):
        T = Slots(max_bytes=300, buckets=1, admit_new=admit_new)             # one bucket, and it cannot grow
        assert T.max_buckets == 1
        for key in (11, 12, 13, 14):                         # four keys, each with its total: the bucket is full
            for _ in range(2):
                T.add(key, 5, 1, 1)
                T.bump(key, 1, 1)
        for extra in range(900, 1100):                       # and no room is left beside the table
            if not T._room_beside():
                break
            T.add(extra, 5, 3, 1)
        assert not T._room_beside() and T.let_go == 0
        return T
    T = full(False)
    T.add(21, 5, 1, 2)
    assert T.get(21, 5) == 0 and all(T.get(key, 5) == 2 for key in (11, 12, 13, 14))     # confirmed: it stays
    T = full(True)
    gone = T.add(21, 5, 1, 2)
    assert T.get(21, 5) == 1 and sum(T.get(key, 5) == 2 for key in (11, 12, 13, 14)) == 3
    assert sorted(g[1] for g in gone) == [0, 5] and len({g[0] for g in gone}) == 1         # a key gives way whole


def test_the_letters_go_on_learning_in_a_full_table():
    c = Cevahir(budget_mb=OVERHEAD_MB + 1)
    rng = random.Random(11)
    words = ["".join(rng.choice("abcçdefgğhıijklmnoöprsştuüvyz") for _ in range(rng.randint(2, 8))) for _ in range(300)]
    for _ in range(3):
        c.read("".join(" ".join(rng.choice(words) for _ in range(10)) + ". " for _ in range(6000)))
    assert c.size()["letters"]["let_go"] > 0
    new = "yepyeni bir dil: qwx zvk qwx zvk qwx zvk. " * 40
    first = c.letters.live(new[:len(new) // 2])
    second = c.letters.live(new[len(new) // 2:])
    assert second < first / 2                                # a text they had not met is foreseen once it is met


def test_reading_does_not_push_out_what_was_taught():
    """What long texts write lives in a table of its own: in one table with what the mind was taught, a megabyte
    of reading left 11 of 433 lesson dialogues answered (FINAL_FORM.md)."""
    c = _taught(SCAN_LIFE, budget_mb=OVERHEAD_MB + 1)
    c.tell("Kedi süt içer ve uyur.")
    taught = c.store.lived.digest()
    rng = random.Random(12)
    words = ["".join(rng.choice("abcçdefgğhıijklmnoöprsştuüvyz") for _ in range(rng.randint(2, 8))) for _ in range(4000)]
    for _ in range(4):
        c.read("".join(" ".join(rng.choice(words) for _ in range(10)) + ". " for _ in range(6000)))
    read = c.size()["read"]
    assert read["let_go"] > 3 * read["slots"]                # the reading overflowed its table several times over
    assert c.store.lived.digest() == taught
    assert c.reply("zap twice") == "Z Z ." and c.reply("walk and run") == "W R ."


def test_forgetting_brings_back_what_was_told_last():
    def lesson(c):
        c.tell("masanın rengi kahverengi .")
        c.watch("masanın rengi ne ?", "kahverengi")
        c.tell("defterin rengi sarı .")
        c.watch("defterin rengi ne ?", "sarı")
        c.tell("perdenin rengi mor .")
        c.tell("perdenin rengi gri .")
        c.watch("perdenin rengi ne ?", "gri")
        c.tell("kapının rengi beyaz .")
        c.tell("kapının rengi yeşil .")
        c.watch("kapının rengi ne ?", "yeşil")
    c = Cevahir(ledger=True)
    lesson(c)
    c.tell("halının rengi mavi .")
    c.tell("halının rengi sarı .")
    assert c.reply("halının rengi ne ?") == "sarı"
    digest, moments = c.digest(), c.moments()
    c.tell("halının rengi mavi .")                           # said again: the tie is an old one, written last now
    assert c.reply("halının rengi ne ?") == "mavi"
    c.forget(len(c.entries) - 1)
    assert c.digest() == digest and c.moments() == moments
    assert c.reply("halının rengi ne ?") == "sarı"           # as if the last telling had never been lived
    long = "halının rengi mavi . " + "Ankara Türkiye'nin başkentidir. İstanbul en kalabalık şehirdir. " * 80
    c.read(long)                                             # the same, where a long text said it (many ties at once)
    assert c.reply("halının rengi ne ?") == "mavi"
    c.forget(len(c.entries) - 1)
    assert c.digest() == digest and c.moments() == moments
    assert c.reply("halının rengi ne ?") == "sarı"


def test_a_mind_loaded_holds_no_more_than_a_mind_that_lived():
    import tempfile
    c = Cevahir()
    rng = random.Random(3)
    words = ["".join(rng.choice("abcdefghijklmnoprstuvyz") for _ in range(rng.randint(2, 9))) for _ in range(30000)]
    c.read("".join(" ".join(rng.choice(words) for _ in range(10)) + ". " for _ in range(30000)))
    with tempfile.TemporaryDirectory() as folder:
        c.save(folder)
        again = Cevahir.load(folder)
    assert again.digest() == c.digest() and again.moments() == c.moments()
    assert again.size()["table_bytes"] <= c.size()["table_bytes"]      # the growth law holds for a loaded mind too


def test_the_spellings_are_packed_where_they_belong():
    from cevher.units import Spellings, code_of
    rng = random.Random(2)
    table, kept = Spellings(), {}
    table.PENDING = 37                                       # packed many times over, a few words at a time
    for _ in range(3000):
        w = "".join(rng.choice("abcçğıöşü") for _ in range(rng.randint(1, 9)))
        table.setdefault(code_of(w), w)
        kept[code_of(w)] = w
    table.pack()
    assert len(table) == len(kept) and bool((np.diff(table.codes.astype(np.float64)) > 0).all())
    assert all(table.get(c) == w for c, w in kept.items()) and table.get(code_of("hiç görülmedi")) is None
    again = Spellings.from_state(table.state())
    assert all(again.get(c) == w for c, w in kept.items())


def test_the_letters_take_back_exactly_what_a_text_wrote():
    text = "ev evler evlerimiz evde evden okul okullar okulda okuldan " * 60
    L = Letters(Counts(max_bytes=1 << 20, buckets=4))
    L.live(text[:1500])
    before, tail, gone = L.T.digest(), L.tail, []
    bits = L.live(text[1500:], gone=gone)
    assert bits / len(text[1500:]) < 2.0                     # a text that repeats itself is foreseen
    L.forget(text[1500:], tail, gone)
    assert L.T.digest() == before


def test_a_mind_kept_and_loaded_is_the_same_mind(tmp_path):
    c = _taught(SCAN_LIFE)
    c.tell("Kedi süt içer ve uyur.")
    manifest = c.save(str(tmp_path))
    again = Cevahir.load(str(tmp_path))
    assert again.digest() == c.digest() == manifest["digest"]
    for q in ("zap twice", "walk and run", "zap left after look thrice"):
        assert again.reply(q) == c.reply(q)
    again.watch("zap around", "Z Z Z Z .")                   # and it goes on living
    assert again.reply("zap around") == "Z Z Z Z ."
    with open(tmp_path / "cevher.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(IOError):
        Cevahir.load(str(tmp_path))                          # a file that was changed is refused


def test_memory_is_bounded_by_the_budget_whatever_is_lived():
    c = Cevahir(budget_mb=OVERHEAD_MB + 2)                                # 2 MB for both tables
    rng = random.Random(4)
    for block in range(12):
        words = ["".join(rng.choice("abcdefghijklmnoprstuvyz") for _ in range(rng.randint(2, 9))) for _ in range(3000)]
        c.read(" ".join(rng.choice(words) for _ in range(20000)) + " . ")
    size = c.size()
    assert size["table_bytes"] <= (2 << 20) + (1 << 18)
    assert size["ties"]["let_go"] + size["ties"]["not_kept"] > 0     # the bound was reached, and ties gave way


def test_said_in_conversation_asked_later():
    """Gate G5's exchange, on a mind that has lived the conversation lesson (cevher/dersler/konusma.txt)."""
    import os
    c = Cevahir()
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "cevher", "dersler", "konusma.txt"),
              encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" in line:
                c.watch(*[x.strip() for x in line.split("\t", 1)])
            else:
                c.tell(line.strip())
    assert c.converse("benim adım yasin") == ""
    assert c.converse("merhaba") == ""
    assert c.converse("benim adım ne ?") == "yasin"
    assert c.converse("babamın adı ne ?") == ""              # never told: silence, not the nearest name
    assert c.converse("benim adım murat") == ""
    assert c.converse("benim adım ne ?") == "murat"


# ------------------------------------------------------------------------------------- speaking at length --------
THINGS = {
    "saat": ["saat zamanı gösteren bir araçtır .", "saat yay , dişliler ve kadrandan oluşur .",
             "yay enerjiyi biriktirir .", "saat dişlileri pirinçten yapılır ."],
    "gitar": ["gitar telli bir çalgıdır .", "gitar gövde , sap ve tellerden oluşur .",
              "gövde sesi büyütür .", "gövde ladin ağacından yapılır ."],
    "çadır": ["çadır taşınabilir bir barınaktır .", "çadır direkler , kumaş ve kazıklardan oluşur .",
              "kazıklar çadırı toprağa bağlar .", "kumaş naylondan yapılır ."]}


def _shown(c, name, form="{0}ı anlat"):
    text = THINGS[name]
    c.watch("%s nedir ?" % name, text[0])
    c.watch(form.format(name), " ".join(text))


def test_a_sentence_when_asked_for_one_and_the_text_when_asked_to_tell():
    c = Cevahir(ledger=True)
    c.tell("kampçılar geceyi çadırda geçirir .")              # the tent is named before its text is read
    for name in ("saat", "gitar", "çadır"):
        c.read("\n".join(THINGS[name]) + "\n")
    _shown(c, "saat", "{0}i anlat")
    assert c.reply("çadır nedir ?") == "" and c.reply("çadırı anlat") == ""      # shown of one thing: nothing yet
    c.watch("saat nedir ?", THINGS["saat"][0])                # the same thing shown again shows nothing new
    assert c.reply("çadır nedir ?") == ""
    _shown(c, "gitar")
    assert c.reply("çadır nedir ?") == THINGS["çadır"][0]     # one sentence, the first of its own text
    assert c.reply("çadırı anlat") == " ".join(THINGS["çadır"])
    assert c.reply("çadır nasıl onarılır ?") == "" and c.reply("fener nedir ?") == ""    # never shown, never told
    before, moments, held = c.digest(), c.moments(), c.telling.size()["sentences"]
    c.read("fener karanlıkta yolu aydınlatan bir araçtır .\nfener pille çalışır .\n")
    assert c.reply("fener nedir ?") == "fener karanlıkta yolu aydınlatan bir araçtır ."
    c.forget(len(c.entries) - 1)                              # what was read is forgotten: it is not told again
    assert c.reply("fener nedir ?") == "" and c.telling.size()["sentences"] == held
    assert c.digest() == before and c.moments() == moments


def test_what_was_told_is_kept_with_the_mind(tmp_path):
    c = Cevahir()
    for name in ("saat", "gitar", "çadır"):
        c.read("\n".join(THINGS[name]) + "\n")
    _shown(c, "saat", "{0}i anlat")
    _shown(c, "gitar")
    c.save(str(tmp_path))
    again = Cevahir.load(str(tmp_path))
    assert again.reply("çadırı anlat") == c.reply("çadırı anlat") == " ".join(THINGS["çadır"])
    assert again.telling.size() == c.telling.size()
